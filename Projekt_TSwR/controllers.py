"""
controllers.py
==============
Kontrolery dla symulatora F1/10.

Dostępne:
  controller_zero()          – brak sterowania
  controller_const_speed()   – prosty regulator P
  controller_pid()           – regulator PID
  controller_random()        – losowe sterowanie (testy)
  MPCController              – MPC zaimplementowany w acados

Interfejs każdego kontrolera:
    u = ctrl(state)   →  np.ndarray([d, delta])
    state = [s, n, mu, vx, vy, r]
"""

import numpy as np
from vehicle_params import VehicleParams
from vehicle_model import DynamicBicycleModel
from track import TrackCenterline


# ══════════════════════════════════════════════════════════════════════════════
#  PROSTE KONTROLERY (do testów i porównań)
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
      d     = stała

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
                   kd: float = 0.5,  speed: float = 0.35,
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

        integral[0] += n * dt
        derivative    = (n - prev_n[0]) / dt
        prev_n[0]     = n

        delta = -(kp * n + ki * integral[0] + kd * derivative) - 1.0 * mu
        delta = np.clip(delta, -0.35, 0.35)
        return np.array([speed, delta])

    return ctrl


# ══════════════════════════════════════════════════════════════════════════════
#  MPC – acados
# ══════════════════════════════════════════════════════════════════════════════

class MPCController:
    """
    Model Predictive Control dla F1/10 zaimplementowany w acados.

    Sformułowanie:
      min  Σ_{k=0}^{N-1} [ q_n·n² + q_mu·mu² + q_vx·(vx-vx_ref)²
                           + r_d·d² + r_delta·delta²
                           + r_dd·Δd² + r_ddelta·Δdelta² ]
           + terminal cost (×3)

      s.t. x_{k+1} = f(x_k, u_k, κ)   ← dynamika RK4 (dyskretna)
           |delta| ≤ delta_max
           0 ≤ d ≤ 1
           |n| ≤ track_width/2          ← miękkie

    Jeśli acados nie jest zainstalowane, automatycznie odpada do
    solvera scipy L-BFGS-B (wolniejszy, ale działa wszędzie).

    Instalacja acados (Linux):
        https://docs.acados.org/installation/index.html

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
                 r_delta:  float = 0.5,
                 r_dd:     float = 0.5,
                 r_ddelta: float = 1.0):

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

        p = model.p
        self.delta_max = p.delta_max
        self.d_min     = 0.0
        self.d_max     = 1.0

        self.u_prev  = np.array([0.2, 0.0])
        self.U_warm  = np.tile(self.u_prev, (N, 1))

        # Próba użycia acados
        self._use_acados = False
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
        """
        Buduje solver acados OCP dla modelu F1/10.

        Model w acados musi być opisany symboliczne (CasADi).
        Tutaj tworzymy uproszczony model liniowy wokół punktu roboczego
        jako punkt startowy – docelowo zastąp pełnym nieliniowym modelem.
        """
        from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
        import casadi as ca

        p   = self.model.p
        N   = self.N
        dt  = self.dt

        # ── Model symboliczny (CasADi) ────────────────────────────────────
        # Stany: [s, n, mu, vx, vy, r]
        # Wejścia: [d, delta]

        x_sym  = ca.MX.sym('x', 6)
        u_sym  = ca.MX.sym('u', 2)
        kap    = ca.MX.sym('kappa')   # parametr (krzywizna)

        s, n, mu, vx, vy, r = (x_sym[i] for i in range(6))
        d, delta = u_sym[0], u_sym[1]

        vx_safe = ca.fmax(vx, 1e-3)

        # Pacejka (CasADi)
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

        # RK4 (dyskretny)
        k1 = f_cont
        k2 = ca.substitute(f_cont, x_sym, x_sym + dt/2 * k1)
        k3 = ca.substitute(f_cont, x_sym, x_sym + dt/2 * k2)
        k4 = ca.substitute(f_cont, x_sym, x_sym + dt   * k3)
        x_next = x_sym + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

        # ── AcadosModel ───────────────────────────────────────────────────
        model     = AcadosModel()
        model.name = 'f1tenth_frenet'
        model.x    = x_sym
        model.u    = u_sym
        model.p    = kap
        model.disc_dyn_expr = x_next

        # ── OCP ───────────────────────────────────────────────────────────
        ocp                         = AcadosOcp()
        ocp.model                   = model
        ocp.dims.N                  = N
        ocp.solver_options.tf       = N * dt
        ocp.dims.np                 = 1   # jeden parametr: kappa

        # Funkcja kosztu (EXTERNAL → własna)
        # Używamy NONLINEAR_LS: y = [n, mu, vx-vx_ref, d, delta]
        ny   = 5   # rozmiar residuum
        ny_e = 3   # terminal: [n, mu, vx-vx_ref]

        y_expr   = ca.vertcat(n, mu, vx - self.vx_ref, d, delta)
        y_e_expr = ca.vertcat(n, mu, vx - self.vx_ref)

        ocp.model.cost_y_expr   = y_expr
        ocp.model.cost_y_expr_e = y_e_expr

        ocp.cost.cost_type   = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        Q  = np.diag([self.q_n, self.q_mu, self.q_vx,
                      self.r_d, self.r_delta])
        Qe = np.diag([3*self.q_n, 3*self.q_mu, self.q_vx])

        ocp.cost.W   = Q
        ocp.cost.W_e = Qe
        ocp.cost.yref   = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        # Ograniczenia wejść
        ocp.constraints.lbu   = np.array([self.d_min, -self.delta_max])
        ocp.constraints.ubu   = np.array([self.d_max,  self.delta_max])
        ocp.constraints.idxbu = np.array([0, 1])

        # Miękkie ograniczenie na n (granica toru)
        hw = self.track.track_width / 2
        ocp.constraints.lbx   = np.array([-hw])
        ocp.constraints.ubx   = np.array([ hw])
        ocp.constraints.idxbx = np.array([1])   # indeks n w wektorze x

        # Stan początkowy
        ocp.constraints.x0 = np.zeros(6)

        # Opcje solvera
        ocp.solver_options.qp_solver        = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx    = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type   = 'DISCRETE'
        ocp.solver_options.nlp_solver_type   = 'SQP_RTI'   # real-time iteration
        ocp.solver_options.print_level       = 0

        self._ocp_solver = AcadosOcpSolver(ocp, json_file='f1tenth_ocp.json')
        print("[MPC] Solver acados zbudowany pomyślnie.")

    # ── Obliczanie sterowania ─────────────────────────────────────────────

    def compute_control(self, state: np.ndarray) -> np.ndarray:
        """Oblicza optymalne sterowanie dla bieżącego stanu."""
        if self._use_acados:
            return self._solve_acados(state)
        else:
            return self._solve_scipy(state)

    def _solve_acados(self, state: np.ndarray) -> np.ndarray:
        """Rozwiązuje OCP za pomocą acados (SQP-RTI)."""
        solver = self._ocp_solver
        kappa  = self.track.get_kappa(state[0])

        # Ustaw stan początkowy
        solver.set(0, 'lbx', state)
        solver.set(0, 'ubx', state)

        # Ustaw parametry (krzywizna) dla każdego kroku
        for k in range(self.N):
            s_k   = state[0] + k * self.dt * max(state[3], 0.5)
            kap_k = self.track.get_kappa(s_k)
            solver.set(k, 'p', np.array([kap_k]))

        # Ciepły start
        for k in range(self.N):
            solver.set(k, 'u', self.U_warm[k])

        status = solver.solve()
        if status not in [0, 2]:   # 0=OK, 2=max_iter (akceptowalne w RTI)
            print(f"[acados] status={status}, używam poprzedniego u")
            return self.u_prev.copy()

        # Pobierz pierwsze sterowanie
        u_opt = solver.get(0, 'u')

        # Zapisz ciepły start
        for k in range(self.N):
            self.U_warm[k] = solver.get(k, 'u')

        u_opt[0] = np.clip(u_opt[0], self.d_min,      self.d_max)
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
            bounds.append((self.d_min,       self.d_max))
            bounds.append((-self.delta_max,  self.delta_max))

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

        u_opt    = U_opt[0].copy()
        u_opt[0] = np.clip(u_opt[0], self.d_min,      self.d_max)
        u_opt[1] = np.clip(u_opt[1], -self.delta_max, self.delta_max)
        self.u_prev = u_opt.copy()
        return u_opt

    def _cost_scipy(self, U_flat: np.ndarray,
                    x0: np.ndarray) -> float:
        """Funkcja kosztu dla scipy."""
        U      = U_flat.reshape(self.N, 2)
        x      = x0.copy()
        J      = 0.0
        u_prev = self.u_prev.copy()
        tw     = self.track.track_width / 2

        for k in range(self.N):
            d     = np.clip(U[k, 0], self.d_min, self.d_max)
            delta = np.clip(U[k, 1], -self.delta_max, self.delta_max)
            s, n, mu, vx, vy, r = x

            J += self.q_n    * n**2
            J += self.q_mu   * mu**2
            J += self.q_vx   * (vx - self.vx_ref)**2
            J += self.r_d    * d**2
            J += self.r_delta * delta**2
            J += self.r_dd    * (d     - u_prev[0])**2
            J += self.r_ddelta * (delta - u_prev[1])**2
            if abs(n) > tw * 0.8:
                J += 100.0 * (abs(n) - tw * 0.8)**2

            kappa  = self.track.get_kappa(s)
            x      = self.model.step_rk4(x, np.array([d, delta]), kappa, self.dt)
            u_prev = np.array([d, delta])

        s, n, mu, vx, vy, r = x
        J += 3 * self.q_n  * n**2
        J += 3 * self.q_mu * mu**2
        J +=     self.q_vx * (vx - self.vx_ref)**2
        return J

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return self.compute_control(state)
