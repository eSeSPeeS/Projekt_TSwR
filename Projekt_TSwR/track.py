"""
track.py
========
Reprezentacja toru wyścigowego jako centerline z krzywizną κ(s).
Obsługuje układ Freneta: konwersja (s, n) ↔ (X, Y).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrackCenterline:
    """
    Tor opisany punktami centerline + krzywizna κ w każdym punkcie.

    Atrybuty:
        x, y       – współrzędne kartezjańskie punktów centerline [m]
        kappa      – krzywizna w każdym punkcie [1/m]
        s_breaks   – narastający łuk od startu [m]
        track_width– szerokość toru [m]
    """
    x:           np.ndarray
    y:           np.ndarray
    kappa:       np.ndarray
    s_breaks:    np.ndarray
    track_width: float = 0.35

    # ── Fabryki ───────────────────────────────────────────────────────────

    @classmethod
    def make_oval(cls, length: float = 8.0, width: float = 4.0,
                  n_points: int = 500,
                  track_width: float = 0.35) -> "TrackCenterline":
        """Tor owalny (elipsa)."""
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        a, b = length / 2, width / 2

        x = a * np.cos(t)
        y = b * np.sin(t)

        # Krzywizna elipsy
        kappa = (a * b) / (a**2 * np.sin(t)**2 + b**2 * np.cos(t)**2) ** 1.5

        dx = np.diff(x, append=x[0])
        dy = np.diff(y, append=y[0])
        ds = np.sqrt(dx**2 + dy**2)
        s_breaks = np.concatenate([[0], np.cumsum(ds[:-1])])

        return cls(x=x, y=y, kappa=kappa,
                   s_breaks=s_breaks, track_width=track_width)

    @classmethod
    def make_figure8(cls, r: float = 2.5, n_points: int = 800,
                     track_width: float = 0.35) -> "TrackCenterline":
        """Tor w kształcie ósemki."""
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        x = r * np.sin(t)
        y = r * np.sin(t) * np.cos(t)

        dx  = np.gradient(x, t)
        dy  = np.gradient(y, t)
        ddx = np.gradient(dx, t)
        ddy = np.gradient(dy, t)

        denom = (dx**2 + dy**2) ** 1.5
        denom = np.where(denom < 1e-6, 1e-6, denom)
        kappa = (dx * ddy - dy * ddx) / denom

        ds_dt  = np.sqrt(dx**2 + dy**2)
        dt_arr = np.diff(t, append=t[-1] - t[-2])
        ds     = ds_dt * np.abs(dt_arr)
        s_breaks = np.concatenate([[0], np.cumsum(ds[:-1])])

        return cls(x=x, y=y, kappa=kappa,
                   s_breaks=s_breaks, track_width=track_width)

    # ── Właściwości ───────────────────────────────────────────────────────

    @property
    def total_length(self) -> float:
        return float(self.s_breaks[-1])

    # ── Zapytania ─────────────────────────────────────────────────────────

    def get_kappa(self, s: float) -> float:
        """Krzywizna w punkcie s (z interpolacją liniową)."""
        s_mod = s % self.total_length
        return float(np.interp(s_mod, self.s_breaks, self.kappa))

    def frenet_to_cartesian(self, s: float,
                            n: float) -> Tuple[float, float, float]:
        """
        Konwersja Frenet (s, n) → kartezjańskie (X, Y, psi_track).

        Returns:
            X, Y        – pozycja pojazdu [m]
            psi_track   – orientacja toru w tym punkcie [rad]
        """
        s_mod = s % self.total_length
        idx   = int(np.searchsorted(self.s_breaks, s_mod)) - 1
        idx   = np.clip(idx, 0, len(self.x) - 1)

        idx_next  = (idx + 1) % len(self.x)
        dx        = self.x[idx_next] - self.x[idx]
        dy        = self.y[idx_next] - self.y[idx]
        psi_track = np.arctan2(dy, dx)

        X = self.x[idx] + n * np.cos(psi_track + np.pi / 2)
        Y = self.y[idx] + n * np.sin(psi_track + np.pi / 2)
        return X, Y, psi_track

    def get_track_boundaries(self):
        """
        Zwraca punkty lewej i prawej krawędzi toru.
        Przydatne do rysowania.
        """
        hw = self.track_width / 2
        n_pts = len(self.x)
        left_x, left_y, right_x, right_y = [], [], [], []

        for i in range(n_pts):
            j = (i + 1) % n_pts
            dx = self.x[j] - self.x[i]
            dy = self.y[j] - self.y[i]
            norm = np.sqrt(dx**2 + dy**2) + 1e-9
            nx, ny = -dy / norm, dx / norm

            left_x.append(self.x[i] + hw * nx)
            left_y.append(self.y[i] + hw * ny)
            right_x.append(self.x[i] - hw * nx)
            right_y.append(self.y[i] - hw * ny)

        # Zamknij pętle
        for lst in [left_x, left_y, right_x, right_y]:
            lst.append(lst[0])

        return (np.array(left_x),  np.array(left_y),
                np.array(right_x), np.array(right_y))
