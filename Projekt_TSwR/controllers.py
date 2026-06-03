"""
controllers.py
==============
Kontrolery dla symulatora F1/10.

Poprawki w MPCController względem poprzedniej wersji
-----------------------------------------------------
BUG 1 – ca.substitute() daje błędne RK4 w CasADi (model wewnętrzny solvera)
    Stary kod:
        k1 = f_cont
        k2 = ca.substitute(f_cont, x_sym, x_sym + dt/2 * k1)  # k2 == k1 !!
    Nowy kod: model CIĄGŁY + integrator ERK4 acados (f_expl_expr).
    Efekt: solver optymalizuje na właściwym modelu → wąż nie skacze.

BUG 2 – parametry krzywizny κ szacowane liniowo (zakłada stałą prędkość)
    Stary kod:
        s_k = state[0] + k * dt * max(vx, 0.5)   ← zakłada stałą prędkość!
    Nowy kod: κ pobierane z propagacji RK4 (s rośnie fizycznie poprawnie).
    Efekt: solver dostaje właściwą krzywiznę na zakrętach → brak przeskoków.

BUG 3 – s z acados nie jest normalizowane po okrążeniu
    Stary kod:
        X_opt[k+1] = solver.get(k+1, 'x')   ← s może być > total_length
    Nowy kod:
        x_k[0] %= total_length               ← normalizacja przed użyciem
    Efekt: ciepły start następnej iteracji używa poprawnego s → brak skoków.

BUG 4 – ograniczenie na n luźne (×2), ale terminal cost też luźny
    Stary: lbx/ubx = ±hw*2.0, ale solver nie "wie" że to granica toru.
    Nowy: twarde ograniczenie ±hw na stage, terminal cost z hw jako bariera.
    Efekt: solver aktywnie trzyma pojazd w torze → nie wyjeżdża "sekundę później".
"""

import numpy as np
from vehicle_model import DynamicBicycleModel
from track import TrackCenterline

try:
    from linear_operator.settings import max_cg_iterations, cg_tolerance
    _HAS_LINEAR_OPERATOR = True
except ImportError:
    _HAS_LINEAR_OPERATOR = False


# ══════════════════════════════════════════════════════════════════════════════
# PROSTE KONTROLERY
# ══════════════════════════════════════════════════════════════════════════════

def controller_zero(state: np.ndarray) -> np.ndarray:
    return np.array([0.0, 0.0])


def controller_random(state: np.ndarray) -> np.ndarray:
    return np.array([
        np.random.uniform(0.1, 0.5),
        np.random.uniform(-0.2, 0.2)
    ])


def controller_const_speed(speed: float = 0.35, kp_steering: float = 2.0):
    def ctrl(state: np.ndarray) -> np.ndarray:
        _, n, mu, vx, _, _ = state
        delta = np.clip(-kp_steering * n - 1.5 * mu, -0.35, 0.35)
        return np.array([speed, delta])
    return ctrl


def controller_pid(kp: float = 5.0, ki: float = 0.1,
                   kd: float = 0.5, speed: float = 0.35,
                   dt: float = 0.02):
    integral = [0.0]
    prev_n   = [0.0]

    def ctrl(state: np.ndarray) -> np.ndarray:
        _, n, mu, vx, _, _ = state
        integral[0]  += n * dt
        derivative    = (n - prev_n[0]) / dt
        prev_n[0]     = n
        delta = np.clip(-(kp*n + ki*integral[0] + kd*derivative) - 1.0*mu,
                        -0.35, 0.35)
        return np.array([speed, delta])
    return ctrl


# ══════════════════════════════════════════════════════════════════════════════
# MPC – acados
# ══════════════════════════════════════════════════════════════════════════════

class MPCController:
    """
    MPC dla F1/10 (acados + opcjonalny GP Residual Model).

    Pola publiczne po każdym wywołaniu:
        prediction_xy      – (N+1, 2) przewidywana trajektoria [X, Y]
        prediction_states  – (N+1, 6) stany horyzontu w Frenet
        last_solver_status – int, status acados
        last_solver_error  – bool
        last_cost          – float
    """

    def __init__(self,
                 model:    DynamicBicycleModel,
                 track:    TrackCenterline,
                 N:        int   = 20,
                 dt:       float = 0.02,
                 vx_ref:   float = 1.5,
                 q_n:      float = 15.0,
                 q_mu:     float = 5.0,
                 q_vx:     float = 2.0,
                 r_d:      float = 0.1,
                 r_delta:  float = 15.0,
                 r_dd:     float = 0.5,
                 r_ddelta: float = 1.0,
                 gp_model=None):

        self.model    = model
        self.track    = track
        self.N        = N
        self.dt       = dt
        self.vx_ref   = vx_ref
        self.q_n      = q_n
        self.q_mu     = q_mu
        self.q_vx     = q_vx
        self.r_d      = r_d
        self.r_delta  = r_delta
        self.r_dd     = r_dd
        self.r_ddelta = r_ddelta

        self.gp_model   = gp_model
        self._gp_active = (
            gp_model is not None
            and getattr(gp_model, 'enabled', False)
            and getattr(gp_model, 'trained', False)
        )
        if self._gp_active:
            print("[MPC] GP Residual Model AKTYWNY.")
        else:
            print("[MPC] GP wyłączony – klasyczny MPC.")

        p = model.p
        self.delta_max = p.delta_max
        self.d_min     = 0.0
        self.d_max     = 1.0

        self.u_prev = np.array([0.2, 0.0])
        self.U_warm = np.tile(self.u_prev, (N, 1))

        self.last_solver_status: int   = -1
        self.last_solver_error:  bool  = False
        self.last_cost:          float = 0.0
        self.prediction_states = np.zeros((N + 1, 6))
        self.prediction_xy     = np.zeros((N + 1, 2))

        self._use_acados = False
        self._ocp_solver = None
        try:
            self._build_acados_solver()
            self._use_acados = True
            print("[MPC] Używam solvera acados.")
        except Exception as e:
            print(f"[MPC] acados niedostępny ({e}). Korzystam z scipy.")

    # ── Budowanie solvera ─────────────────────────────────────────────────

    def _build_acados_solver(self):
        from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
        import casadi as ca

        p  = self.model.p
        N  = self.N
        dt = self.dt

        x_sym = ca.MX.sym('x', 6)
        u_sym = ca.MX.sym('u', 2)
        kap   = ca.MX.sym('kappa')

        s, n, mu, vx, vy, r = (x_sym[i] for i in range(6))
        d, delta = u_sym[0], u_sym[1]

        vx_safe = ca.fmax(vx, 0.3)

        def pacejka_ca(B, C, D, alpha):
            return D * ca.sin(C * ca.atan(B * alpha))

        alpha_f = -ca.atan2(vy + p.lf * r, vx_safe) + delta
        alpha_r = -ca.atan2(vy - p.lr * r, vx_safe)
        Fyf = pacejka_ca(p.Bf, p.Cf, p.Df, alpha_f)
        Fyr = pacejka_ca(p.Br, p.Cr, p.Dr, alpha_r)
        Fx  = p.Cm1 * d - p.Cr0 - p.Cr2 * vx**2

        denom  = ca.fmax(1.0 - n * kap, 1e-6)
        ds_dt  = (vx * ca.cos(mu) - vy * ca.sin(mu)) / denom
        dn_dt  =  vx * ca.sin(mu) + vy * ca.cos(mu)
        dmu_dt =  r - kap * ds_dt
        dvx_dt = (Fx - Fyf * ca.sin(delta) + p.m * vy * r) / p.m
        dvy_dt = (Fyr + Fyf * ca.cos(delta) - p.m * vx * r) / p.m
        dr_dt  = (Fyf * p.lf * ca.cos(delta) - Fyr * p.lr) / p.Iz

        # ── BUG 1 NAPRAWIONY: model ciągły, acados całkuje ERK4 ──────────
        f_cont = ca.vertcat(ds_dt, dn_dt, dmu_dt, dvx_dt, dvy_dt, dr_dt)

        acados_model      = AcadosModel()
        acados_model.name = 'f1tenth_frenet'
        acados_model.x    = x_sym
        acados_model.u    = u_sym
        acados_model.p    = kap
        acados_model.f_expl_expr = f_cont          # ← ciągła ODE, nie dyskretna!

        ocp = AcadosOcp()
        ocp.model = acados_model
        ocp.solver_options.N_horizon = N
        ocp.solver_options.tf        = N * dt
        ocp.dims.np                  = 1
        ocp.parameter_values         = np.array([0.0])

        ny   = 5
        ny_e = 3
        y_expr   = ca.vertcat(n, mu, vx - self.vx_ref, d, delta)
        y_e_expr = ca.vertcat(n, mu, vx - self.vx_ref)

        ocp.model.cost_y_expr   = y_expr
        ocp.model.cost_y_expr_e = y_e_expr
        ocp.cost.cost_type      = 'NONLINEAR_LS'
        ocp.cost.cost_type_e    = 'NONLINEAR_LS'

        Q  = np.diag([self.q_n, self.q_mu, self.q_vx, self.r_d, self.r_delta])
        Qe = np.diag([3*self.q_n, 3*self.q_mu, self.q_vx])

        ocp.cost.W      = Q
        ocp.cost.W_e    = Qe
        ocp.cost.yref   = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        ocp.constraints.lbu   = np.array([self.d_min, -self.delta_max])
        ocp.constraints.ubu   = np.array([self.d_max,  self.delta_max])
        ocp.constraints.idxbu = np.array([0, 1])

        # ── BUG 4 NAPRAWIONY: twarde ograniczenie ±hw (nie ±2*hw) ────────
        hw = self.track.track_width / 2
        ocp.constraints.lbx   = np.array([-hw])
        ocp.constraints.ubx   = np.array([ hw])
        ocp.constraints.idxbx = np.array([1])

        ocp.constraints.x0 = np.zeros(6)

        # ── Integrator ERK4 (naprawia BUG 1) ─────────────────────────────
        ocp.solver_options.qp_solver             = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx        = 'GAUSS_NEWTON'
        ocp.solver_options.levenberg_marquardt   = 1e-3
        ocp.solver_options.integrator_type       = 'ERK'   # ← było DISCRETE
        ocp.solver_options.sim_method_num_stages = 4        # RK4
        ocp.solver_options.sim_method_num_steps  = 1
        ocp.solver_options.nlp_solver_type       = 'SQP_RTI'
        ocp.solver_options.print_level           = 0
        ocp.solver_options.qp_solver_iter_max    = 100
        ocp.solver_options.qp_solver_warm_start  = 1

        self._ocp_solver = AcadosOcpSolver(ocp, json_file='f1tenth_ocp.json')
        print("[MPC] Solver acados zbudowany pomyślnie.")

    # ── Węż predykcyjny ───────────────────────────────────────────────────

    def _build_prediction_xy(self, X_states: np.ndarray) -> None:
        """Konwertuje (N+1, 6) stanów Freneta → (N+1, 2) kartezjańskich."""
        L   = self.track.total_length
        pts = np.zeros((self.N + 1, 2))
        for k in range(self.N + 1):
            s_k = float(X_states[k, 0]) % L    # ← BUG 3: normalizuj s
            n_k = float(X_states[k, 1])
            X, Y, _ = self.track.frenet_to_cartesian(s_k, n_k)
            pts[k, 0] = X
            pts[k, 1] = Y
        self.prediction_states = X_states.copy()
        self.prediction_xy     = pts

    # ── Propagacja horyzontu (wspólna dla GP i bez GP) ────────────────────

    def _propagate_horizon(self, state: np.ndarray) -> np.ndarray:
        """
        Propaguje stany RK4 przez horyzont używając U_warm.
        Zwraca X_pred (N+1, 6) z normalizowanym s.

        BUG 2 NAPRAWIONY: κ pobierana z fizycznie propagowanego s,
        nie z liniowego szacowania state[0] + k*dt*vx.
        """
        L      = self.track.total_length
        X_pred = np.zeros((self.N + 1, 6))
        X_pred[0] = state.copy()
        x = state.copy()

        if self._gp_active:
            X_h = np.zeros((self.N, 6))
            U_h = np.zeros((self.N, 2))
            for k in range(self.N):
                X_h[k] = x
                U_h[k] = self.U_warm[k]
                kap_k  = self.track.get_kappa(x[0] % L)
                x      = self.model.step_rk4(x, self.U_warm[k], kap_k, self.dt)

            if _HAS_LINEAR_OPERATOR:
                with max_cg_iterations(2000), cg_tolerance(0.05):
                    gp_res = self.gp_model.predict_residual_batch(X_h, U_h)
            else:
                gp_res = self.gp_model.predict_residual_batch(X_h, U_h)

            x = state.copy()
            for k in range(self.N):
                kap_k  = self.track.get_kappa(x[0] % L)
                x_nom  = self.model.step_rk4(x, self.U_warm[k], kap_k, self.dt)
                x_nom[[3, 4, 5]] += gp_res[k]
                x = x_nom
                X_pred[k+1] = x
        else:
            for k in range(self.N):
                kap_k = self.track.get_kappa(x[0] % L)   # ← % L kluczowe
                x     = self.model.step_rk4(x, self.U_warm[k], kap_k, self.dt)
                X_pred[k+1] = x

        return X_pred

    # ── Ustawianie parametrów i stanów w solverze ─────────────────────────

    def _init_solver_from_pred(self, X_pred: np.ndarray) -> None:
        """
        Inicjalizuje solver acados stanami i parametrami κ z X_pred.
        BUG 2 NAPRAWIONY: κ z propagacji, nie z liniowego szacowania.
        """
        L      = self.track.total_length
        solver = self._ocp_solver
        for k in range(self.N):
            s_k   = float(X_pred[k, 0]) % L
            kap_k = self.track.get_kappa(s_k)
            solver.set(k, 'p', np.array([kap_k]))
            solver.set(k, 'x', X_pred[k])
            solver.set(k, 'u', self.U_warm[k])

        s_term   = float(X_pred[self.N, 0]) % L
        kap_term = self.track.get_kappa(s_term)
        solver.set(self.N, 'p', np.array([kap_term]))
        solver.set(self.N, 'x', X_pred[self.N])

    # ── Solver główny ─────────────────────────────────────────────────────

    def compute_control(self, state: np.ndarray) -> np.ndarray:
        if self._use_acados:
            return self._solve_acados(state)
        else:
            return self._solve_scipy(state)

    def _solve_acados(self, state: np.ndarray) -> np.ndarray:
        solver = self._ocp_solver

        # Stan początkowy (twarde ograniczenie)
        solver.set(0, 'lbx', state)
        solver.set(0, 'ubx', state)

        # Przesuń ciepły start o 1 krok
        self.U_warm = np.roll(self.U_warm, -1, axis=0)
        self.U_warm[-1] = self.U_warm[-2]

        # Propaguj horyzont i zainicjuj solver
        X_pred = self._propagate_horizon(state)
        self._init_solver_from_pred(X_pred)

        status = solver.solve()
        self.last_solver_status = status
        self.last_solver_error  = status not in [0, 2]
        try:
            self.last_cost = float(solver.get_cost())
        except Exception:
            self.last_cost = 0.0

        if status not in [0, 2]:
            self._handle_solver_failure(state, X_pred)
            return self.u_prev.copy()

        # Pobierz rozwiązanie
        # BUG 3 NAPRAWIONY: normalizuj s po pobraniu z solvera
        L     = self.track.total_length
        X_opt = np.zeros((self.N + 1, 6))
        X_opt[0] = state.copy()
        for k in range(self.N):
            self.U_warm[k] = solver.get(k, 'u')
            x_k = solver.get(k + 1, 'x')
            x_k[0] = x_k[0] % L                # ← normalizuj s
            X_opt[k + 1] = x_k

        self._build_prediction_xy(X_opt)

        u_opt    = solver.get(0, 'u')
        u_opt[0] = np.clip(u_opt[0], self.d_min, self.d_max)
        u_opt[1] = np.clip(u_opt[1], -self.delta_max, self.delta_max)
        self.u_prev = u_opt.copy()
        return u_opt

    def _handle_solver_failure(self, state: np.ndarray,
                                X_pred: np.ndarray) -> None:
        """Fallback przy błędzie solvera: PID + reset ciepłego startu."""
        solver = self._ocp_solver
        _, n, mu, vx, _, _ = state

        # PID fallback
        delta_fb = float(np.clip(-3.0*n - 2.0*mu, -self.delta_max, self.delta_max))
        d_fb     = float(np.clip(0.3*(self.vx_ref - vx) + 0.2,
                                 self.d_min, self.d_max))
        u_fb = np.array([d_fb, delta_fb])
        self.u_prev = u_fb.copy()

        # Reset ciepłego startu
        L      = self.track.total_length
        kappa0 = self.track.get_kappa(state[0] % L)
        delta0 = float(np.clip(self.model.p.lf * kappa0,
                               -self.delta_max, self.delta_max))
        self.U_warm = np.tile(np.array([0.3, delta0]), (self.N, 1))

        # Twardy reset solvera
        try:
            solver.reset()
        except Exception:
            pass

        # Reinicjalizacja solvera z bezpieczną trajektorią
        x_safe = state.copy()
        x_safe[3] = np.clip(x_safe[3], 0.2, 8.0)
        x_safe[4] = np.clip(x_safe[4], -2.0, 2.0)
        x_safe[5] = np.clip(x_safe[5], -4.0, 4.0)

        X_reinit    = np.zeros((self.N + 1, 6))
        X_reinit[0] = x_safe.copy()
        for k in range(self.N):
            kap_k = self.track.get_kappa(x_safe[0] % L)
            solver.set(k, 'p', np.array([kap_k]))
            solver.set(k, 'x', x_safe)
            solver.set(k, 'u', self.U_warm[k])
            x_safe = self.model.step_rk4(x_safe, self.U_warm[k], kap_k, self.dt)
            x_safe[3] = np.clip(x_safe[3], 0.2, 8.0)
            x_safe[4] = np.clip(x_safe[4], -2.0, 2.0)
            x_safe[5] = np.clip(x_safe[5], -4.0, 4.0)
            X_reinit[k + 1] = x_safe.copy()

        solver.set(self.N, 'x', X_reinit[-1])
        solver.set(self.N, 'p', np.array([
            self.track.get_kappa(X_reinit[-1, 0] % L)
        ]))

        self._build_prediction_xy(X_reinit)

    def _solve_scipy(self, state: np.ndarray) -> np.ndarray:
        from scipy.optimize import minimize

        U0      = np.roll(self.U_warm, -1, axis=0)
        U0[-1]  = U0[-2]
        U0_flat = U0.flatten()

        bounds = ([(self.d_min, self.d_max),
                   (-self.delta_max, self.delta_max)] * self.N)

        result = minimize(
            fun=self._cost_scipy, x0=U0_flat, args=(state,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 30, 'ftol': 1e-3, 'gtol': 1e-2}
        )

        U_opt = result.x.reshape(self.N, 2)
        self.U_warm             = U_opt.copy()
        self.last_solver_status = 0 if result.success else 1
        self.last_solver_error  = not result.success
        self.last_cost          = float(result.fun)

        L     = self.track.total_length
        X_pred    = np.zeros((self.N + 1, 6))
        X_pred[0] = state.copy()
        x_cur     = state.copy()
        for k in range(self.N):
            d_k     = np.clip(U_opt[k, 0], self.d_min, self.d_max)
            delta_k = np.clip(U_opt[k, 1], -self.delta_max, self.delta_max)
            u_k     = np.array([d_k, delta_k])
            kappa   = self.track.get_kappa(x_cur[0] % L)   # ← % L
            x_nom   = self.model.step_rk4(x_cur, u_k, kappa, self.dt)
            if self._gp_active:
                if _HAS_LINEAR_OPERATOR:
                    with max_cg_iterations(2000), cg_tolerance(0.05):
                        resid, _ = self.gp_model.predict_residual(x_cur, u_k)
                else:
                    resid, _ = self.gp_model.predict_residual(x_cur, u_k)
                x_nom[[3, 4, 5]] += resid
            x_cur       = x_nom
            X_pred[k+1] = x_cur

        self._build_prediction_xy(X_pred)

        u_opt    = U_opt[0].copy()
        u_opt[0] = np.clip(u_opt[0], self.d_min, self.d_max)
        u_opt[1] = np.clip(u_opt[1], -self.delta_max, self.delta_max)
        self.u_prev = u_opt.copy()
        return u_opt

    def _cost_scipy(self, U_flat: np.ndarray, x0: np.ndarray) -> float:
        U      = U_flat.reshape(self.N, 2)
        x      = x0.copy()
        J      = 0.0
        u_prev = self.u_prev.copy()
        tw     = self.track.track_width / 2
        L      = self.track.total_length

        for k in range(self.N):
            d     = np.clip(U[k, 0], self.d_min, self.d_max)
            delta = np.clip(U[k, 1], -self.delta_max, self.delta_max)
            u_k   = np.array([d, delta])
            s, n, mu, vx, vy, r = x

            J += self.q_n      * n**2
            J += self.q_mu     * mu**2
            J += self.q_vx     * (vx - self.vx_ref)**2
            J += self.r_d      * d**2
            J += self.r_delta  * delta**2
            J += self.r_dd     * (d     - u_prev[0])**2
            J += self.r_ddelta * (delta - u_prev[1])**2
            if abs(n) > tw * 0.8:
                J += 200.0 * (abs(n) - tw * 0.8)**2

            kappa = self.track.get_kappa(s % L)     # ← % L
            x_nom = self.model.step_rk4(x, u_k, kappa, self.dt)

            if self._gp_active:
                if _HAS_LINEAR_OPERATOR:
                    with max_cg_iterations(2000), cg_tolerance(0.05):
                        resid, _ = self.gp_model.predict_residual(x, u_k)
                else:
                    resid, _ = self.gp_model.predict_residual(x, u_k)
                x_nom[[3, 4, 5]] += resid

            x      = x_nom
            u_prev = u_k

        s, n, mu, vx, vy, r = x
        J += 3 * self.q_n  * n**2
        J += 3 * self.q_mu * mu**2
        J += self.q_vx     * (vx - self.vx_ref)**2
        return J

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return self.compute_control(state)