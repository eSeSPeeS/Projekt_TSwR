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
    @classmethod
    def make_pocket(cls, n_points: int = 1200,
                    track_width: float = 1.5) -> "TrackCenterline":
        """Tor typu pocket/U-turn, szeroki i bez przecięć."""
        pts = np.array([
            [-6.0, -4.5],
            [-6.0,  4.5],
            [-2.2,  4.5],
            [-2.2,  1.2],
            [ 0.0,  1.2],
            [ 0.0,  4.5],
            [ 6.0,  4.5],
            [ 6.0, -4.5],
            [-6.0, -4.5],
        ], dtype=float)

        seg_lens = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        total_len = np.sum(seg_lens)

        samples = []
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            m = max(20, int(n_points * seg_lens[i] / total_len))
            t = np.linspace(0, 1, m, endpoint=False)
            seg = (1 - t)[:, None] * p0 + t[:, None] * p1
            samples.append(seg)

        xy = np.vstack(samples)

        # lekkie wygładzenie
        win = 21
        pad = win // 2
        x_pad = np.pad(xy[:, 0], (pad, pad), mode='wrap')
        y_pad = np.pad(xy[:, 1], (pad, pad), mode='wrap')
        kernel = np.ones(win) / win
        x = np.convolve(x_pad, kernel, mode='valid')
        y = np.convolve(y_pad, kernel, mode='valid')

        t = np.linspace(0, 2 * np.pi, len(x), endpoint=False)
        dx = np.gradient(x, t)
        dy = np.gradient(y, t)
        ddx = np.gradient(dx, t)
        ddy = np.gradient(dy, t)

        denom = (dx**2 + dy**2) ** 1.5
        denom = np.where(denom < 1e-6, 1e-6, denom)
        kappa = (dx * ddy - dy * ddx) / denom

        ds = np.sqrt(np.diff(x, append=x[0])**2 + np.diff(y, append=y[0])**2)
        s_breaks = np.concatenate([[0], np.cumsum(ds[:-1])])

        return cls(
            x=x,
            y=y,
            kappa=kappa,
            s_breaks=s_breaks,
            track_width=track_width,
        )
    @classmethod
    def make_technical(cls, base_r: float = 6.0,
                    n_points: int = 1200,
                    track_width: float = 1.5) -> "TrackCenterline":
        """Techniczny tor bez przecięć, z wieloma zakrętami."""
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

        r = (
            base_r
            + 1.2 * np.sin(2 * t)
            + 0.9 * np.sin(3 * t + 0.4)
            + 0.5 * np.sin(5 * t - 0.7)
        )

        x = r * np.cos(t)
        y = 0.85 * r * np.sin(t)

        dx = np.gradient(x, t)
        dy = np.gradient(y, t)
        ddx = np.gradient(dx, t)
        ddy = np.gradient(dy, t)

        denom = (dx**2 + dy**2) ** 1.5
        denom = np.where(denom < 1e-6, 1e-6, denom)
        kappa = (dx * ddy - dy * ddx) / denom

        ds_dt = np.sqrt(dx**2 + dy**2)
        dt_arr = np.diff(t, append=t[-1] - t[-2])
        ds = ds_dt * np.abs(dt_arr)
        s_breaks = np.concatenate([[0], np.cumsum(ds[:-1])])

        return cls(
            x=x,
            y=y,
            kappa=kappa,
            s_breaks=s_breaks,
            track_width=track_width,
        )
    @classmethod
    def make_technical_sharp(cls, base_r: float = 6.5,
                            n_points: int = 1400,
                            track_width: float = 1.5) -> "TrackCenterline":
        """Techniczny tor bez przecięć, z ostrzejszymi zakrętami."""
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

        r = (
            base_r
            + 1.4 * np.sin(2 * t)
            + 1.0 * np.sin(3 * t + 0.3)
            + 0.8 * np.sin(5 * t - 0.6)
            + 0.35 * np.sin(7 * t + 1.0)
        )

        x = r * np.cos(t)
        y = 0.82 * r * np.sin(t)

        dx = np.gradient(x, t)
        dy = np.gradient(y, t)
        ddx = np.gradient(dx, t)
        ddy = np.gradient(dy, t)

        denom = (dx**2 + dy**2) ** 1.5
        denom = np.where(denom < 1e-6, 1e-6, denom)
        kappa = (dx * ddy - dy * ddx) / denom

        ds_dt = np.sqrt(dx**2 + dy**2)
        dt_arr = np.diff(t, append=t[-1] - t[-2])
        ds = ds_dt * np.abs(dt_arr)
        s_breaks = np.concatenate([[0], np.cumsum(ds[:-1])])

        return cls(
            x=x,
            y=y,
            kappa=kappa,
            s_breaks=s_breaks,
            track_width=track_width,
        )
    @classmethod
    def make_chicane(cls, length: float = 10.0, height: float = 3.0,
                    n_points: int = 900,
                    track_width: float = 0.60) -> "TrackCenterline":
        """Trudny tor z szykanami, ale szeroki."""
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

        x = (length / 2.0) * np.cos(t)
        y = (
            0.9 * np.sin(t)
            + 0.55 * np.sin(2 * t)
            - 0.35 * np.sin(3 * t)
        ) * height

        dx = np.gradient(x, t)
        dy = np.gradient(y, t)
        ddx = np.gradient(dx, t)
        ddy = np.gradient(dy, t)

        denom = (dx**2 + dy**2) ** 1.5
        denom = np.where(denom < 1e-6, 1e-6, denom)
        kappa = (dx * ddy - dy * ddx) / denom

        ds_dt = np.sqrt(dx**2 + dy**2)
        dt_arr = np.diff(t, append=t[-1] - t[-2])
        ds = ds_dt * np.abs(dt_arr)
        s_breaks = np.concatenate([[0], np.cumsum(ds[:-1])])

        return cls(
            x=x,
            y=y,
            kappa=kappa,
            s_breaks=s_breaks,
            track_width=track_width,
        )
    
    @classmethod
    def make_road_loop(cls, length: float = 10.0, width: float = 6.0,
                       n_points: int = 700,
                       track_width: float = 0.35) -> "TrackCenterline":
        """
        Prosty zamknięty tor drogowy:
        dwie dłuższe proste i dwa zakręty o różnej ciasności,
        bez samoprzecięć.
        """
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

        # Gładka zamknięta pętla bardziej "drogowa" niż elipsa
        x = (length / 2) * np.cos(t) + 0.8 * np.cos(2 * t)
        y = (width / 2) * np.sin(t) - 0.6 * np.sin(2 * t)

        dx = np.gradient(x, t)
        dy = np.gradient(y, t)
        ddx = np.gradient(dx, t)
        ddy = np.gradient(dy, t)

        denom = (dx**2 + dy**2) ** 1.5
        denom = np.where(denom < 1e-6, 1e-6, denom)
        kappa = (dx * ddy - dy * ddx) / denom

        ds_dt = np.sqrt(dx**2 + dy**2)
        dt_arr = np.diff(t, append=t[-1] - t[-2])
        ds = ds_dt * np.abs(dt_arr)
        s_breaks = np.concatenate([[0], np.cumsum(ds[:-1])])

        return cls(
            x=x,
            y=y,
            kappa=kappa,
            s_breaks=s_breaks,
            track_width=track_width
        )

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