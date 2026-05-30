"""
vehicle_model.py
================
Dynamiczny model jednośladu (single-track bicycle model)
z oponami Pacejki w układzie współrzędnych Freneta.

Stany: x = [s, n, mu, vx, vy, r]
  s   – postęp wzdłuż toru [m]
  n   – odchylenie boczne od centerline [m]
  mu  – kąt pojazdu względem toru [rad]
  vx  – prędkość podłużna (body frame) [m/s]
  vy  – prędkość boczna (body frame) [m/s]
  r   – prędkość kątowa yaw [rad/s]

Wejścia: u = [d, delta]
  d     – sygnał napędowy [0, 1]
  delta – kąt skrętu przednich kół [rad]

Źródło: Kabzan et al., arXiv:2003.04882, równania (1)–(7)
"""

import numpy as np
from vehicle_params import VehicleParams


class DynamicBicycleModel:
    """
    Model dynamiczny F1/10 w układzie Freneta.

    Równania ciągłe:
      ṡ   = (vx·cos(mu) - vy·sin(mu)) / (1 - n·κ)
      ṅ   =  vx·sin(mu) + vy·cos(mu)
      μ̇   =  r - κ·ṡ
      v̇x  = (Fx - Fyf·sin(δ) + m·vy·r) / m
      v̇y  = (Fyr + Fyf·cos(δ) - m·vx·r) / m
      ṙ   = (Fyf·lf·cos(δ) - Fyr·lr) / Iz
    """

    def __init__(self, params: VehicleParams):
        self.p = params

    # ── Siły opon (Pacejka) ───────────────────────────────────────────────

    def pacejka(self, B: float, C: float, D: float, alpha: float) -> float:
        """Uproszczony model Pacejki: Fy = D·sin(C·arctan(B·α))"""
        return D * np.sin(C * np.arctan(B * alpha))

    def tire_forces(self, vx: float, vy: float, r: float,
                    delta: float):
        """Oblicza siły boczne opon [N]: (Fyf, Fyr)."""
        p = self.p
        vx_safe = max(abs(vx), 1e-3)

        alpha_f = -np.arctan2(vy + p.lf * r, vx_safe) + delta
        alpha_r = -np.arctan2(vy - p.lr * r, vx_safe)

        Fyf = self.pacejka(p.Bf, p.Cf, p.Df, alpha_f)
        Fyr = self.pacejka(p.Br, p.Cr, p.Dr, alpha_r)
        return Fyf, Fyr

    def drive_force(self, d: float, vx: float) -> float:
        """Siła napędowa [N]."""
        p = self.p
        return p.Cm1 * d - p.Cr0 - p.Cr2 * vx ** 2

    # ── Równania różniczkowe ──────────────────────────────────────────────

    def f(self, x: np.ndarray, u: np.ndarray, kappa: float) -> np.ndarray:
        """
        Ciągła prawa strona: ẋ = f(x, u, κ)

        Args:
            x:     stan [s, n, mu, vx, vy, r]
            u:     wejście [d, delta]
            kappa: krzywizna toru w s

        Returns:
            dx/dt  shape (6,)
        """
        p = self.p
        s, n, mu, vx, vy, r = x
        d, delta = u

        delta = np.clip(delta, -p.delta_max, p.delta_max)
        d     = np.clip(d, 0.0, 1.0)
        vx    = max(vx, 0.0)

        Fyf, Fyr = self.tire_forces(vx, vy, r, delta)
        Fx       = self.drive_force(d, vx)

        denom = 1.0 - n * kappa
        denom = np.sign(denom) * max(abs(denom), 1e-6)

        ds_dt  = (vx * np.cos(mu) - vy * np.sin(mu)) / denom
        dn_dt  =  vx * np.sin(mu) + vy * np.cos(mu)
        dmu_dt =  r - kappa * ds_dt

        dvx_dt = (Fx - Fyf * np.sin(delta) + p.m * vy * r) / p.m
        dvy_dt = (Fyr + Fyf * np.cos(delta) - p.m * vx * r) / p.m
        dr_dt  = (Fyf * p.lf * np.cos(delta) - Fyr * p.lr) / p.Iz

        return np.array([ds_dt, dn_dt, dmu_dt, dvx_dt, dvy_dt, dr_dt])

    # ── Całkowanie ────────────────────────────────────────────────────────

    def step_rk4(self, x: np.ndarray, u: np.ndarray,
                 kappa: float, dt: float) -> np.ndarray:
        """Jeden krok RK4."""
        k1 = self.f(x,              u, kappa)
        k2 = self.f(x + dt/2 * k1, u, kappa)
        k3 = self.f(x + dt/2 * k2, u, kappa)
        k4 = self.f(x + dt   * k3, u, kappa)
        x_next = x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        x_next[3] = np.clip(x_next[3], self.p.v_min, self.p.v_max)
        return x_next