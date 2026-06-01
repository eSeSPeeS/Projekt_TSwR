"""
controllers.py
==============
Kontrolery dla symulatora F1/10.

Dostępne:
    controller_zero()         – brak sterowania
    controller_const_speed()  – prosty regulator P
    controller_pid()          – regulator PID
    controller_random()       – losowe sterowanie (testy)
    MPCController             – MPC zaimplementowany w acados

Interfejs każdego kontrolera:
    u = ctrl(state) → np.ndarray([d, delta])
    state = [s, n, mu, vx, vy, r]
"""

import numpy as np
from vehicle_params import VehicleParams
from vehicle_model import DynamicBicycleModel
from track import TrackCenterline
from linear_operator.settings import max_cg_iterations, cg_tolerance

# ══════════════════════════════════════════════════════════════════════════════
# PROSTE KONTROLERY (do testów i porównań)
# ══════════════════════════════════════════════════════════════════════════════

def controller_zero(state: np.ndarray) -> np.ndarray:
    """Brak sterowania – pojazd zwalnia."""
    return np.array([0.0, 0.0])


def controller_random(state: np.ndarray) -> np.ndarray:
    """Losowe sterowanie – tylko do testów modelu."""
    return np.array([
        np.random.uniform(0.1, 0.5),
        np.random.uniform(-0.2, 0.2)
    ])


def controller_const_speed(speed: float = 0.35,
                           kp_steering: float = 2.0):
    """
    Prosty regulator P:
        delta = -kp·n - 1.5·mu
        d = stała

    Args:
        speed:       stały sygnał napędowy [0, 1]
        kp_steering: wzmocnienie proporcjonalne dla n
    """
    def ctrl(state: np.ndarray) -> np.ndarray:
        _, n, mu, vx, _, _ = state
        delta = -kp_steering * n - 1.5 * mu
        delta = np.clip(delta, -0.35, 0.35)
        return np.array([speed, delta])
    return ctrl


def controller_pid(kp: float = 5.0, ki: float = 0.1,
                   kd: float = 0.5, speed: float = 0.35,
                   dt: float = 0.02):
    """
    Regulator PID na błąd boczny n.

    Args:
        kp, ki, kd: wzmocnienia PID
        speed:      stały sygnał napędowy
        dt:         krok czasowy (musi zgadzać się z symulatorem)
    """
    integral = [0.0]
    prev_n   = [0.0]

    def ctrl(state: np.ndarray) -> np.ndarray:
        _, n, mu, vx, _, _ = state

        integral[0]   += n * dt
        derivative     = (n - prev_n[0]) / dt
        prev_n[0]      = n

        delta = -(kp * n + ki * integral[0] + kd * derivative) - 1.0 * mu
        delta = np.clip(delta, -0.35, 0.35)
        return np.array([speed, delta])

    return ctrl


# ══════════════════════════════════════════════════════════════════════════════
# MPC – acados
# ══════════════════════════════════════════════════════════════════════════════

class MPCController:
    """
    Model Predictive Control dla F1/10 zaimplementowany w acados.

    Sformułowanie:
        min Σ_{k=0}^{N-1} [ q_n·n² + q_mu·mu² + q_vx·(vx-vx_ref)²
                           + r_d·d² + r_delta·delta²
                           + r_dd·Δd² + r_ddelta·Δdelta² ]
          + terminal cost (×3)

        s.t. x_{k+1} = f(x_k, u_k, κ) + B_d·g(x_k, u_k) ← RK4 + GP
             |delta| ≤ delta_max
             0 ≤ d ≤ 1
             |n| ≤ track_width/2   ← miękkie

    Opcjonalna integracja z Gaussowskim modelem resztkowym (GP):
      • Jeśli gp_model jest podany i wytrenowany, resztka GP jest dodawana
        do predykcji stanów w trakcie optymalizacji scipy (warm-start)
        oraz w fazie propagacji stanu dla acados.
      • gp_model=None lub gp_model.enabled=False → klasyczny MPC.

    Jeśli acados nie jest zainstalowane, automatycznie odpada do
    solvera scipy L-BFGS-B (wolniejszy, ale działa wszędzie).

    Wąż predykcyjny:
      Po każdym wywołaniu compute_control() dostępne jest pole:
          self.prediction_xy  →  np.ndarray kształtu (N+1, 2)
      zawierające przewidywaną trajektorię pojazdu w układzie kartezjańskim
      (X, Y) dla bieżącego horyzontu MPC.  Symulator używa tego pola do
      rysowania węża w trybie realtime_plot.

    Args:
        model:    DynamicBicycleModel
        track:    TrackCenterline
        N:        horyzont predykcji (liczba kroków)
        dt:       krok czasowy [s]
        vx_ref:   docelowa prędkość [m/s]
        q_n:      kara za odchylenie boczne
        q_mu:     kara za kąt mu
        q_vx:     kara za błąd prędkości
        r_d:      regularyzacja sygnału napędowego
        r_delta:  regularyzacja kąta skrętu
        r_dd:     kara za zmianę d (płynność)
        r_ddelta: kara za zmianę delta (płynność)
        gp_model: GPResidualModel lub None – model GP do korekcji dynamiki
    """

    def __init__(self,
                 model: DynamicBicycleModel,
                 track: TrackCenterline,
                 N: int = 20,
                 dt: float = 0.02,
                 vx_ref: float = 1.5,
                 q_n: float = 15.0,
                 q_mu: float = 5.0,
                 q_vx: float = 2.0,
                 r_d: float = 0.1,
                 r_delta: float = 15.0,
                 r_dd: float = 0.5,
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

        # ── GP Residual Model ─────────────────────────────────────────────
        self.gp_model  = gp_model
        self._gp_active = (
            gp_model is not None
            and getattr(gp_model, 'enabled', False)
            and getattr(gp_model, 'trained', False)
        )

        if self._gp_active:
            print("[MPC] GP Residual Model AKTYWNY – korekcja dynamiki włączona.")
        elif gp_model is not None:
            print("[MPC] GP podany, ale nieaktywny (enabled=False lub nietrend.).")
        else:
            print("[MPC] GP wyłączony – klasyczny MPC.")

        p = model.p
        self.delta_max = p.delta_max
        self.d_min     = 0.0
        self.d_max     = 1.0

        self.u_prev  = np.array([0.2, 0.0])
        self.U_warm  = np.tile(self.u_prev, (N, 1))

        # ── Wąż predykcyjny – pole publiczne ────────────────────────────
        # Kształt (N+1, 2): punkt bieżący + N punktów horyzontu w [X, Y]
        self.prediction_xy: np.ndarray = np.zeros((N + 1, 2))

        # Próba użycia acados
        self._use_acados  = False
        self._ocp_solver  = None
        try:
            self._build_acados_solver()
            self._use_acados = True
            print("[MPC] Używam solvera acados.")
        except Exception as e:
            print(f"[MPC] acados niedostępny ({e}). "
                  f"Korzystam z scipy L-BFGS-B.")

    # ── Budowanie solvera acados ──────────────────────────────────────────

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

        vx_safe = ca.fmax(vx, 1e-3)

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

        f_cont = ca.vertcat(ds_dt, dn_dt, dmu_dt, dvx_dt, dvy_dt, dr_dt)

        k1 = f_cont
        k2 = ca.substitute(f_cont, x_sym, x_sym + dt/2 * k1)
        k3 = ca.substitute(f_cont, x_sym, x_sym + dt/2 * k2)
        k4 = ca.substitute(f_cont, x_sym, x_sym + dt * k3)
        x_next = x_sym + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

        model       = AcadosModel()
        model.name  = 'f1tenth_frenet'
        model.x     = x_sym
        model.u     = u_sym
        model.p     = kap
        model.disc_dyn_expr = x_next

        ocp = AcadosOcp()
        ocp.model = model
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

        ocp.cost.cost_type   = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        Q  = np.diag([self.q_n, self.q_mu, self.q_vx,
                      self.r_d, self.r_delta])
        Qe = np.diag([3*self.q_n, 3*self.q_mu, self.q_vx])

        ocp.cost.W     = Q
        ocp.cost.W_e   = Qe
        ocp.cost.yref  = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        ocp.constraints.lbu    = np.array([self.d_min, -self.delta_max])
        ocp.constraints.ubu    = np.array([self.d_max,  self.delta_max])
        ocp.constraints.idxbu  = np.array([0, 1])

        hw = self.track.track_width / 2
        ocp.constraints.lbx   = np.array([-hw * 2.0])
        ocp.constraints.ubx   = np.array([ hw * 2.0])
        ocp.constraints.idxbx = np.array([1])

        ocp.constraints.x0 = np.zeros(6)

        #ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.qp_solver      = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.levenberg_marquardt   = 1e-3   # ← DODAĆ
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.print_level     = 0
        ocp.solver_options.qp_solver_iter_max   = 100    # domyślnie 50 
        ocp.solver_options.qp_solver_warm_start = 1      # ciepły start QP między iteracjami SQP

        self._ocp_solver = AcadosOcpSolver(ocp, json_file='f1tenth_ocp.json')
        print("[MPC] Solver acados zbudowany pomyślnie.")

    # ── Obliczanie sterowania ─────────────────────────────────────────────

    def compute_control(self, state: np.ndarray) -> np.ndarray:
        """Oblicza optymalne sterowanie dla bieżącego stanu."""
        if self._use_acados:
            return self._solve_acados(state)
        else:
            return self._solve_scipy(state)

    # ── Pomocnicza: budowanie węża predykcyjnego ──────────────────────────

    def _build_prediction_xy(self, state: np.ndarray,
                              X_states: np.ndarray) -> None:
        """
        Konwertuje N+1 stanów Freneta (s, n) na kartezjańskie (X, Y)
        i zapisuje wynik w self.prediction_xy.

        Args:
            state:    bieżący stan pojazdu [s, n, mu, vx, vy, r]
            X_states: macierz kształtu (N+1, 6) – stany horyzontu
                      (wiersz 0 = stan bieżący, wiersze 1..N = horyzont)
        """
        pts = np.zeros((self.N + 1, 2))
        for k in range(self.N + 1):
            s_k = float(X_states[k, 0])
            n_k = float(X_states[k, 1])
            X, Y, _ = self.track.frenet_to_cartesian(s_k, n_k)
            pts[k, 0] = X
            pts[k, 1] = Y
        self.prediction_xy = pts

    def _solve_acados(self, state: np.ndarray) -> np.ndarray:
        """Rozwiązuje OCP za pomocą acados (SQP-RTI)."""
        solver = self._ocp_solver

        solver.set(0, 'lbx', state)
        solver.set(0, 'ubx', state)

        for k in range(self.N):
            s_k   = state[0] + k * self.dt * max(state[3], 0.5)
            kap_k = self.track.get_kappa(s_k)
            solver.set(k, 'p', np.array([kap_k]))
        solver.set(self.N, 'p', np.array([kap_k]))

        self.U_warm = np.roll(self.U_warm, -1, axis=0)
        self.U_warm[-1] = self.U_warm[-2]

        x_current = state.copy()

        if self._gp_active:
            X_horizon = np.zeros((self.N, 6))
            U_horizon = np.zeros((self.N, 2))
            x_prop = state.copy()
            for k in range(self.N):
                X_horizon[k] = x_prop
                U_horizon[k] = self.U_warm[k]
                kap_k  = self.track.get_kappa(x_prop[0])
                x_prop = self.model.step_rk4(x_prop, self.U_warm[k], kap_k, self.dt)
            
            with max_cg_iterations(2000), cg_tolerance(0.05):
                gp_residuals = self.gp_model.predict_residual_batch(X_horizon, U_horizon)

            x_current = state.copy()
            X_pred = np.zeros((self.N + 1, 6))
            X_pred[0] = state.copy()
            for k in range(self.N):
                solver.set(k, "u", self.U_warm[k])
                solver.set(k, "x", x_current)
                kap_k  = self.track.get_kappa(x_current[0])
                x_nom  = self.model.step_rk4(x_current, self.U_warm[k], kap_k, self.dt)
                x_nom[[3, 4, 5]] += gp_residuals[k]
                x_current  = x_nom
                X_pred[k+1] = x_current
        else:
            X_pred = np.zeros((self.N + 1, 6))
            X_pred[0] = state.copy()
            for k in range(self.N):
                solver.set(k, "u", self.U_warm[k])
                solver.set(k, "x", x_current)
                kap_k     = self.track.get_kappa(x_current[0])
                x_current = self.model.step_rk4(x_current, self.U_warm[k], kap_k, self.dt)
                X_pred[k+1] = x_current

        solver.set(self.N, "x", x_current)

        status = solver.solve()

        if status not in [0, 2]:
            print(f"[acados] status={status}, przechodzę na bezpieczny fallback i reset solvera")

            s, n, mu, vx, vy, r = state
            X, Y, psi_track = self.track.frenet_to_cartesian(s, n)
            psi     = psi_track + mu
            X_front = X + self.model.p.lf * np.cos(psi)
            Y_front = Y + self.model.p.lf * np.sin(psi)

            print(
                f"s={s:.2f}m | n={n:.3f}m | mu={mu:.3f}rad | "
                f"vx={vx:.2f}m/s | vy={vy:.2f}m/s | r={r:.2f}rad/s | "
                f"Xf={X_front:.3f}m | Yf={Y_front:.3f}m | psi={psi:.3f}rad | "
                f"u_prev=[d={self.u_prev[0]:.3f}, delta={self.u_prev[1]*180/np.pi:.1f}°]"
            )

            # Bezpieczne sterowanie awaryjne
            _, n, mu, vx, _, _ = state
            delta_fb   = float(np.clip(-3.0 * n - 2.0 * mu,
                                    -self.delta_max, self.delta_max))
            d_fb       = float(np.clip(0.3 * (self.vx_ref - vx) + 0.2,
                                    self.d_min, self.d_max))
            u_fallback = np.array([d_fb, delta_fb])

            self.u_prev = u_fallback.copy()

            # ── 1. Zresetuj warm start wejść ─────────────────────────────
            kappa0 = self.track.get_kappa(state[0])
            delta0 = float(np.clip(self.model.p.lf * kappa0,
                                -self.delta_max, self.delta_max))
            self.U_warm = np.tile(np.array([0.3, delta0]), (self.N, 1))

            # ── 2. Twardy reset wewnętrznego solvera acados ─────────────
            try:
                solver.reset()
            except Exception as e:
                print(f"[acados] reset() nieudany: {e}")

            # ── 3. Odbuduj spójną trajektorię stanów do inicjalizacji ────
            x_safe = state.copy()
            x_safe[3] = np.clip(x_safe[3], 0.2, 8.0)   # vx
            x_safe[4] = np.clip(x_safe[4], -2.0, 2.0)  # vy
            x_safe[5] = np.clip(x_safe[5], -4.0, 4.0)  # r

            X_reinit = np.zeros((self.N + 1, 6))
            X_reinit[0] = x_safe.copy()

            for k in range(self.N):
                kap_k = self.track.get_kappa(x_safe[0])

                solver.set(k, "p", np.array([kap_k]))
                solver.set(k, "x", x_safe)
                solver.set(k, "u", self.U_warm[k])

                x_safe = self.model.step_rk4(x_safe, self.U_warm[k], kap_k, self.dt)
                x_safe[3] = np.clip(x_safe[3], 0.2, 8.0)
                x_safe[4] = np.clip(x_safe[4], -2.0, 2.0)
                x_safe[5] = np.clip(x_safe[5], -4.0, 4.0)

                X_reinit[k + 1] = x_safe.copy()

            solver.set(self.N, "x", X_reinit[-1])
            solver.set(self.N, "p", np.array([self.track.get_kappa(X_reinit[-1, 0])]))

            # ── 4. Wąż predykcyjny z bezpiecznej trajektorii ─────────────
            self._build_prediction_xy(state, X_reinit)

            return u_fallback

        # Pobierz rozwiązanie z solvera i zbuduj węża z optymalnych stanów
        X_opt = np.zeros((self.N + 1, 6))
        X_opt[0] = state.copy()
        for k in range(self.N):
            self.U_warm[k] = solver.get(k, 'u')
            X_opt[k+1]     = solver.get(k+1, 'x')

        self._build_prediction_xy(state, X_opt)

        u_opt    = solver.get(0, 'u')
        u_opt[0] = np.clip(u_opt[0], self.d_min, self.d_max)
        u_opt[1] = np.clip(u_opt[1], -self.delta_max, self.delta_max)
        self.u_prev = u_opt.copy()
        return u_opt

    def _solve_scipy(self, state: np.ndarray) -> np.ndarray:
        """Fallback: scipy L-BFGS-B (gdy acados niedostępny)."""
        from scipy.optimize import minimize

        U0      = np.roll(self.U_warm, -1, axis=0)
        U0[-1]  = U0[-2]
        U0_flat = U0.flatten()

        bounds = []
        for _ in range(self.N):
            bounds.append((self.d_min, self.d_max))
            bounds.append((-self.delta_max, self.delta_max))

        result = minimize(
            fun=self._cost_scipy,
            x0=U0_flat,
            args=(state,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 30, 'ftol': 1e-3, 'gtol': 1e-2}
        )

        U_opt       = result.x.reshape(self.N, 2)
        self.U_warm = U_opt.copy()

        # Propagacja trajektorii do węża predykcyjnego
        X_pred    = np.zeros((self.N + 1, 6))
        X_pred[0] = state.copy()
        x_cur     = state.copy()
        for k in range(self.N):
            d_k     = np.clip(U_opt[k, 0], self.d_min, self.d_max)
            delta_k = np.clip(U_opt[k, 1], -self.delta_max, self.delta_max)
            u_k     = np.array([d_k, delta_k])
            kappa   = self.track.get_kappa(x_cur[0])
            x_nom   = self.model.step_rk4(x_cur, u_k, kappa, self.dt)
            if self._gp_active:
                resid, _ = self.gp_model.predict_residual(x_cur, u_k)
                x_nom[[3, 4, 5]] += resid
            x_cur       = x_nom
            X_pred[k+1] = x_cur

        self._build_prediction_xy(state, X_pred)

        u_opt    = U_opt[0].copy()
        u_opt[0] = np.clip(u_opt[0], self.d_min, self.d_max)
        u_opt[1] = np.clip(u_opt[1], -self.delta_max, self.delta_max)
        self.u_prev = u_opt.copy()
        return u_opt

    def _cost_scipy(self, U_flat: np.ndarray,
                    x0: np.ndarray) -> float:
        """Funkcja kosztu dla scipy (z opcjonalną korekcją GP)."""
        U      = U_flat.reshape(self.N, 2)
        x      = x0.copy()
        J      = 0.0
        u_prev = self.u_prev.copy()
        tw     = self.track.track_width / 2

        for k in range(self.N):
            d     = np.clip(U[k, 0], self.d_min, self.d_max)
            delta = np.clip(U[k, 1], -self.delta_max, self.delta_max)
            u_k   = np.array([d, delta])
            s, n, mu, vx, vy, r = x

            J += self.q_n    * n**2
            J += self.q_mu   * mu**2
            J += self.q_vx   * (vx - self.vx_ref)**2
            J += self.r_d    * d**2
            J += self.r_delta * delta**2
            J += self.r_dd   * (d     - u_prev[0])**2
            J += self.r_ddelta * (delta - u_prev[1])**2
            if abs(n) > tw * 0.8:
                J += 100.0 * (abs(n) - tw * 0.8)**2

            kappa = self.track.get_kappa(s)
            x_nom = self.model.step_rk4(x, u_k, kappa, self.dt)

            if self._gp_active:
                with max_cg_iterations(2000), cg_tolerance(0.05):
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