"""
F1tenth Vehicle Simulator
=========================
Model dynamiczny pojazdu oparty na publikacji:
  Liniger et al. "Learning-based Model Predictive Control for Autonomous Racing"
  arXiv:2003.04882

Model: Single-track (bicycle model) z oponami Pacejki w układzie Freneta.

Stany: x = [s, n, mu, vx, vy, r]
  s   - postęp wzdłuż toru [m]
  n   - boczne odchylenie od środka toru [m]
  mu  - kąt między pojazdem a osią toru [rad]
  vx  - prędkość wzdłużna w układzie pojazdu [m/s]
  vy  - prędkość boczna w układzie pojazdu [m/s]
  r   - prędkość kątowa (yaw rate) [rad/s]

Wejścia: u = [wheel_speed (lub akceleracja), delta (kąt skrętu kół)]
  wheel_speed  - prędkość kół / siła napędowa
  delta        - kąt skrętu przednich kół [rad]

Tor: reprezentowany jako ciąg punktów centerline z krzywizną kappa(s).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple
import time

# ── Sprawdź czy PyBullet jest dostępny ──────────────────────────────────────
try:
    import pybullet as pb
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("[INFO] PyBullet niedostępny – wizualizacja tylko przez matplotlib.")


# ═══════════════════════════════════════════════════════════════════════════
#  PARAMETRY POJAZDU  (F1/10 ~1:10 skala, wzorowane na arXiv:2003.04882)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VehicleParams:
    """Parametry fizyczne pojazdu F1/10."""

    # Geometria
    lf: float = 0.15875   # odległość środka masy od osi przedniej [m]
    lr: float = 0.17145   # odległość środka masy od osi tylnej   [m]
    
    # Masa i bezwładność
    m:  float = 3.47      # masa [kg]
    Iz: float = 0.04712   # moment bezwładności wokół osi Z [kg·m²]

    # Opony – model Pacejki (uproszczony, dla F1/10)
    # Fy = D * sin(C * arctan(B * alpha))
    Bf: float = 2.579     # sztywność przednich opon (B)
    Cf: float = 1.2       # kształt przednich opon (C)
    Df: float = 0.192     # szczytowa siła boczna przednich opon (D) [×mg per tire]

    Br: float = 3.3852    # sztywność tylnych opon (B)
    Cr: float = 1.2       # kształt tylnych opon (C)
    Dr: float = 0.1737    # szczytowa siła boczna tylnych opon (D)

    # Napęd
    Cm1: float = 0.287    # współczynnik siły napędowej [N per unit input]
    Cm2: float = 0.0545   # współczynnik tłumienia [N·s/m]
    Cr0: float = 0.0518   # opór toczenia [N]
    Cr2: float = 0.00035  # opór aerodynamiczny [N·s²/m²]

    # Ograniczenia
    delta_max: float  = 0.35  # max kąt skrętu [rad] (~20 deg)
    v_min:     float  = 0.0   # min prędkość
    v_max:     float  = 8.0   # max prędkość [m/s]
    
    # Właściwości fizyczne
    g: float  = 9.81

    @property
    def L(self) -> float:
        return self.lf + self.lr


# ═══════════════════════════════════════════════════════════════════════════
#  CENTERLINE TORU
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrackCenterline:
    """
    Tor opisany jako ciąg punktów z krzywizną.
    
    Układ Freneta wymaga znajomości krzywizny kappa(s) w każdym punkcie toru.
    'n' to odchylenie od linii środkowej (n=0 → środek toru).
    """
    x:        np.ndarray   # współrzędne X punktów centerline [m]
    y:        np.ndarray   # współrzędne Y punktów centerline [m]
    kappa:    np.ndarray   # krzywizna w każdym punkcie [1/m]
    s_breaks: np.ndarray   # narastający łuk od początku [m]
    track_width: float = 0.35  # szerokość toru [m] (tw z datasetu)

    @classmethod
    def make_oval(cls, length: float = 6.0, width: float = 3.0,
                  n_points: int = 500, track_width: float = 0.35) -> "TrackCenterline":
        """Generuje owalny tor (elipsa) jako centerline."""
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        a, b = length / 2, width / 2  # półosie elipsy

        x = a * np.cos(t)
        y = b * np.sin(t)

        # Krzywizna elipsy: kappa = (a*b) / (a²sin²t + b²cos²t)^(3/2)
        kappa = (a * b) / (a**2 * np.sin(t)**2 + b**2 * np.cos(t)**2) ** 1.5

        # Długość łuku (numerycznie)
        dx = np.diff(x, append=x[0])
        dy = np.diff(y, append=y[0])
        ds = np.sqrt(dx**2 + dy**2)
        s_breaks = np.concatenate([[0], np.cumsum(ds[:-1])])

        return cls(x=x, y=y, kappa=kappa, s_breaks=s_breaks, track_width=track_width)

    @classmethod
    def make_figure8(cls, r: float = 2.5, n_points: int = 800,
                     track_width: float = 0.35) -> "TrackCenterline":
        """Generuje tor w kształcie ósemki."""
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        x = r * np.sin(t)
        y = r * np.sin(t) * np.cos(t)

        dx = np.gradient(x, t)
        dy = np.gradient(y, t)
        ddx = np.gradient(dx, t)
        ddy = np.gradient(dy, t)

        # Krzywizna: kappa = (dx*ddy - dy*ddx) / (dx²+dy²)^(3/2)
        denom = (dx**2 + dy**2) ** 1.5
        denom = np.where(denom < 1e-6, 1e-6, denom)
        kappa = (dx * ddy - dy * ddx) / denom

        ds_dt = np.sqrt(dx**2 + dy**2)
        dt_arr = np.diff(t, append=t[-1] - t[-2])
        ds = ds_dt * np.abs(dt_arr)
        s_breaks = np.concatenate([[0], np.cumsum(ds[:-1])])

        return cls(x=x, y=y, kappa=kappa, s_breaks=s_breaks, track_width=track_width)

    @property
    def total_length(self) -> float:
        return float(self.s_breaks[-1])

    def get_kappa(self, s: float) -> float:
        """Zwraca krzywizną dla danego s (z interpolacją)."""
        s_mod = s % self.total_length
        return float(np.interp(s_mod, self.s_breaks, self.kappa))

    def frenet_to_cartesian(self, s: float, n: float) -> Tuple[float, float, float]:
        """Konwertuje współrzędne Freneta (s, n) → kartezjańskie (X, Y, psi)."""
        s_mod = s % self.total_length

        # Znajdź najbliższy punkt centerline
        idx = np.searchsorted(self.s_breaks, s_mod) - 1
        idx = np.clip(idx, 0, len(self.x) - 1)

        # Tangent toru w tym punkcie
        idx_next = (idx + 1) % len(self.x)
        dx = self.x[idx_next] - self.x[idx]
        dy = self.y[idx_next] - self.y[idx]
        psi_track = np.arctan2(dy, dx)

        # Punkt na torze + odsunięcie prostopadle o n
        X = self.x[idx] + n * np.cos(psi_track + np.pi / 2)
        Y = self.y[idx] + n * np.sin(psi_track + np.pi / 2)
        return X, Y, psi_track


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL DYNAMICZNY  (arXiv:2003.04882, równanie 1-7)
# ═══════════════════════════════════════════════════════════════════════════

class DynamicBicycleModel:
    """
    Pełny dynamiczny model jednośladu (single-track / bicycle model)
    z oponami Pacejki, w układzie Freneta.

    Równania ruchu (continuous time):

      Układ Freneta:
        ṡ   = (vx·cos(mu) - vy·sin(mu)) / (1 - n·kappa(s))
        ṅ   =  vx·sin(mu) + vy·cos(mu)
        μ̇   =  r - kappa(s)·ṡ

      Dynamika pojazdu (ciało sztywne):
        v̇x  =  (Fx - Fyf·sin(delta) + m·vy·r) / m
        v̇y  =  (Fyr + Fyf·cos(delta) - m·vx·r) / m
        ṙ   =  (Fyf·lf·cos(delta) - Fyr·lr) / Iz

    Siły opon – model Pacejki (Magic Formula uproszczony):
        alpha_f = -arctan((vy + lf·r)/vx) + delta   # kąt poślizgu przód
        alpha_r = -arctan((vy - lr·r)/vx)            # kąt poślizgu tył
        Fy_i = D_i · sin(C_i · arctan(B_i · alpha_i))

    Siła napędowa (tylne koła):
        Fx = Cm1·d - Cm2·d·vx - Cr0 - Cr2·vx²
        gdzie d = wheel_speed (wejście, 0..1)
    """

    def __init__(self, params: VehicleParams):
        self.p = params

    def pacejka(self, B: float, C: float, D: float, alpha: float) -> float:
        """Model Pacejki: Fy = D·sin(C·arctan(B·alpha))"""
        return D * np.sin(C * np.arctan(B * alpha))

    def tire_forces(self, vx: float, vy: float, r: float,
                    delta: float) -> Tuple[float, float]:
        """Oblicza siły boczne opon przednich i tylnych [N]."""
        p = self.p
        eps = 1e-3  # unika dzielenia przez zero przy małej prędkości

        vx_safe = max(abs(vx), eps)

        # Kąty poślizgu
        alpha_f = -np.arctan2(vy + p.lf * r, vx_safe) + delta
        alpha_r = -np.arctan2(vy - p.lr * r, vx_safe)

        Fyf = self.pacejka(p.Bf, p.Cf, p.Df, alpha_f)
        Fyr = self.pacejka(p.Br, p.Cr, p.Dr, alpha_r)

        return Fyf, Fyr

    def drive_force(self, d: float, vx: float) -> float:
        """Oblicza siłę napędową na tylnych kołach."""
        p = self.p
        Fx = p.Cm1 * d - p.Cm2 * d * vx - p.Cr0 - p.Cr2 * vx**2
        return Fx

    def f(self, x: np.ndarray, u: np.ndarray, kappa: float) -> np.ndarray:
        """
        Prawa strona równań różniczkowych: ẋ = f(x, u, kappa)
        
        Args:
            x:     stan [s, n, mu, vx, vy, r]
            u:     wejście [wheel_speed/d, delta]
            kappa: krzywizna toru w punkcie s

        Returns:
            dx/dt: pochodna stanu
        """
        p = self.p
        s, n, mu, vx, vy, r = x
        d, delta = u

        # Ogranicz wejścia
        delta = np.clip(delta, -p.delta_max, p.delta_max)
        d     = np.clip(d, 0.0, 1.0)

        # Zabezpieczenie przed zerową prędkością
        vx = max(vx, 0.0)

        # Siły
        Fyf, Fyr = self.tire_forces(vx, vy, r, delta)
        Fx       = self.drive_force(d, vx)

        # Mianownik Freneta (1 - n·κ)
        denom = 1.0 - n * kappa
        if abs(denom) < 1e-6:
            denom = np.sign(denom) * 1e-6

        # Równania Freneta
        ds_dt  = (vx * np.cos(mu) - vy * np.sin(mu)) / denom
        dn_dt  =  vx * np.sin(mu) + vy * np.cos(mu)
        dmu_dt =  r - kappa * ds_dt

        # Dynamika pojazdu
        dvx_dt = (Fx - Fyf * np.sin(delta) + p.m * vy * r) / p.m
        dvy_dt = (Fyr + Fyf * np.cos(delta) - p.m * vx * r) / p.m
        dr_dt  = (Fyf * p.lf * np.cos(delta) - Fyr * p.lr) / p.Iz

        return np.array([ds_dt, dn_dt, dmu_dt, dvx_dt, dvy_dt, dr_dt])

    def step_rk4(self, x: np.ndarray, u: np.ndarray,
                 kappa: float, dt: float) -> np.ndarray:
        """Całkowanie RK4 o jeden krok dt."""
        k1 = self.f(x,                u, kappa)
        k2 = self.f(x + dt/2 * k1,   u, kappa)
        k3 = self.f(x + dt/2 * k2,   u, kappa)
        k4 = self.f(x + dt   * k3,   u, kappa)
        x_next = x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

        # Ogranicz prędkości
        x_next[3] = np.clip(x_next[3], self.p.v_min, self.p.v_max)
        return x_next


# ═══════════════════════════════════════════════════════════════════════════
#  LIVE 2D WIZUALIZACJA
# ═══════════════════════════════════════════════════════════════════════════

class LiveVisualizer2D:
    """
    Wizualizacja samochodu w czasie rzeczywistym w matplotlib (widok z góry).

    Rysuje:
      - tor z krawędziami i centerline
      - samochód jako prostokąt z 4 kółkami (przednie skręcają z delta)
      - ślad trajektorii kolorowany prędkością vx
      - panele HUD: prędkość, n, delta, postęp okrążenia
      - wykresy live: vx, n, sterowanie (ostatnie N sekund)

    Użycie:
        viz = LiveVisualizer2D(track, params)
        viz.start()
        for step in ...:
            viz.update(state, u, t)
        viz.close()
    """

    # wymiary rysunkowe samochodu [m] – dla czytelności trochę powiększone
    CAR_LENGTH = 0.55
    CAR_WIDTH  = 0.22
    WHEEL_L    = 0.14
    WHEEL_W    = 0.06

    def __init__(self, track: "TrackCenterline",
                 params: "VehicleParams",
                 history_secs: float = 5.0,
                 dt: float = 0.02,
                 follow_cam: bool = True):
        self.track        = track
        self.params       = params
        self.history_len  = int(history_secs / dt)
        self.follow_cam   = follow_cam

        # Bufory historii (ring-buffer)
        self._xs:  list = []
        self._ys:  list = []
        self._vxs: list = []
        self._ns:  list = []
        self._ds:  list = []
        self._deltas: list = []
        self._ts:  list = []

        self._fig = None
        self._ax_track = None
        self._track_drawn = False

        # Wstępnie oblicz kartezjańskie centerline i krawędzie
        self._precompute_track()

    def _precompute_track(self):
        track = self.track
        n_pts = len(track.x)
        hw    = track.track_width / 2

        lx, ly, rx, ry = [], [], [], []
        for i in range(n_pts):
            j = (i + 1) % n_pts
            dx = track.x[j] - track.x[i]
            dy = track.y[j] - track.y[i]
            norm = np.hypot(dx, dy) + 1e-9
            nx_v, ny_v = -dy / norm, dx / norm
            lx.append(track.x[i] + hw * nx_v)
            ly.append(track.y[i] + hw * ny_v)
            rx.append(track.x[i] - hw * nx_v)
            ry.append(track.y[i] - hw * ny_v)

        # Zamknij pętlę
        lx.append(lx[0]); ly.append(ly[0])
        rx.append(rx[0]); ry.append(ry[0])

        self._lx = np.array(lx); self._ly = np.array(ly)
        self._rx = np.array(rx); self._ry = np.array(ry)

    # ── Budowanie figury ───────────────────────────────────────────────────

    def start(self):
        """Tworzy okno matplotlib i rysuje statyczne elementy toru."""
        plt.ion()
        self._fig = plt.figure(figsize=(14, 9), facecolor='#0d0d0d')
        self._fig.canvas.manager.set_window_title('F1/10 Symulator – widok live 2D')

        # Układ: lewo-tor duży | prawo-panele
        gs = self._fig.add_gridspec(3, 2,
                                     width_ratios=[2.2, 1],
                                     hspace=0.45, wspace=0.3,
                                     left=0.05, right=0.97,
                                     top=0.93, bottom=0.08)

        self._ax_track = self._fig.add_subplot(gs[:, 0])
        self._ax_vx    = self._fig.add_subplot(gs[0, 1])
        self._ax_n     = self._fig.add_subplot(gs[1, 1])
        self._ax_ctrl  = self._fig.add_subplot(gs[2, 1])

        self._setup_track_ax()
        self._setup_plot_axes()

        # Inicjalizuj obiekty rysunkowe
        self._trail_sc = self._ax_track.scatter(
            [], [], c=[], cmap='plasma', s=6,
            vmin=0, vmax=self.params.v_max, zorder=4, alpha=0.85)

        # Samochód – prostokąt (patch)
        self._car_patch = patches.FancyBboxPatch(
            (0, 0), self.CAR_LENGTH, self.CAR_WIDTH,
            boxstyle="round,pad=0.01",
            linewidth=1.5, edgecolor='#00e5ff',
            facecolor='#0d47a1', zorder=8)
        self._ax_track.add_patch(self._car_patch)

        # Kierunek – strzałka
        self._arrow = self._ax_track.annotate(
            '', xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#00e5ff', lw=2),
            zorder=9)

        # 4 koła
        self._wheel_patches = []
        for _ in range(4):
            wp = patches.Rectangle(
                (0, 0), self.WHEEL_L, self.WHEEL_W,
                linewidth=1, edgecolor='#ffeb3b',
                facecolor='#212121', zorder=9)
            self._ax_track.add_patch(wp)
            self._wheel_patches.append(wp)

        # HUD – tekst
        hud_kw = dict(transform=self._ax_track.transAxes,
                      fontsize=10, color='#e0e0e0',
                      fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='#1a1a2e', alpha=0.85, edgecolor='#444'))
        self._hud_speed = self._ax_track.text(
            0.02, 0.97, '', va='top', ha='left', **hud_kw)
        self._hud_state = self._ax_track.text(
            0.02, 0.82, '', va='top', ha='left', **hud_kw)
        self._hud_lap   = self._ax_track.text(
            0.75, 0.97, '', va='top', ha='left', **hud_kw)

        # Wykresy live – linie
        lkw = dict(linewidth=1.6)
        self._line_vx, = self._ax_vx.plot([], [], color='#ff6f00', **lkw)
        self._line_n,  = self._ax_n.plot([], [],  color='#e53935', **lkw)
        self._line_d,  = self._ax_ctrl.plot([], [], color='#43a047', **lkw, label='throttle')
        self._line_dl, = self._ax_ctrl.plot([], [], color='#1e88e5', **lkw, label='delta [rad]')
        self._ax_ctrl.legend(loc='upper right', fontsize=7,
                              facecolor='#1a1a1a', labelcolor='white', framealpha=0.7)

        # Colorbar dla śladu
        sm = plt.cm.ScalarMappable(cmap='plasma',
                                    norm=plt.Normalize(0, self.params.v_max))
        sm.set_array([])
        cb = self._fig.colorbar(sm, ax=self._ax_track, fraction=0.025, pad=0.01)
        cb.set_label('vx [m/s]', color='#e0e0e0', fontsize=9)
        cb.ax.yaxis.set_tick_params(color='#e0e0e0')
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='#e0e0e0')

        self._fig.canvas.draw()
        plt.pause(0.05)

    def _setup_track_ax(self):
        ax = self._ax_track
        ax.set_facecolor('#111111')
        ax.tick_params(colors='#888')
        ax.spines[:].set_color('#333')

        # Powierzchnia toru (szara strefa)
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection
        surf_x = list(self._lx) + list(self._rx[::-1])
        surf_y = list(self._ly) + list(self._ry[::-1])
        poly = MplPolygon(list(zip(surf_x, surf_y)), closed=True)
        pc   = PatchCollection([poly], facecolor='#2a2a2a',
                               edgecolor='none', zorder=0)
        ax.add_collection(pc)

        # Krawędzie – białe kreski
        ax.plot(self._lx, self._ly, color='#ffffff', linewidth=1.5,
                linestyle='--', alpha=0.6, zorder=1)
        ax.plot(self._rx, self._ry, color='#ffffff', linewidth=1.5,
                linestyle='--', alpha=0.6, zorder=1)

        # Centerline – żółta przerywana
        cx = list(self.track.x) + [self.track.x[0]]
        cy = list(self.track.y) + [self.track.y[0]]
        ax.plot(cx, cy, color='#ffd600', linewidth=0.8,
                linestyle=':', alpha=0.5, zorder=2)

        # Linia startu – zielono-biała szachownica
        X0, Y0, psi0 = self.track.frenet_to_cartesian(0, 0)
        hw = self.track.track_width / 2
        nx_v = -np.sin(psi0); ny_v = np.cos(psi0)
        ax.plot([X0 - hw*nx_v, X0 + hw*nx_v],
                [Y0 - hw*ny_v, Y0 + hw*ny_v],
                color='#76ff03', linewidth=2.5, zorder=3, label='start')

        ax.set_aspect('equal')
        ax.set_title('Widok toru – live 2D', color='#e0e0e0',
                     fontsize=12, pad=8)
        ax.set_xlabel('X [m]', color='#888', fontsize=9)
        ax.set_ylabel('Y [m]', color='#888', fontsize=9)

        # Granice osi – trochę większe niż tor
        pad = 1.0
        all_x = np.concatenate([self._lx, self._rx])
        all_y = np.concatenate([self._ly, self._ry])
        self._xlim = (all_x.min() - pad, all_x.max() + pad)
        self._ylim = (all_y.min() - pad, all_y.max() + pad)
        ax.set_xlim(*self._xlim)
        ax.set_ylim(*self._ylim)

    def _setup_plot_axes(self):
        dark_kw = dict(facecolor='#111111')
        for ax, title, ylabel, color in [
            (self._ax_vx,   'Prędkość vx',      'vx [m/s]',  '#ff6f00'),
            (self._ax_n,    'Odchylenie n',      'n [m]',     '#e53935'),
            (self._ax_ctrl, 'Sterowanie',        'wartość',   '#888888'),
        ]:
            ax.set_facecolor(dark_kw['facecolor'])
            ax.tick_params(colors='#888', labelsize=8)
            ax.spines[:].set_color('#333')
            ax.set_title(title, color='#e0e0e0', fontsize=9, pad=4)
            ax.set_ylabel(ylabel, color=color, fontsize=8)
            ax.set_xlabel('t [s]', color='#888', fontsize=8)
            ax.grid(True, alpha=0.15, color='#555')

        # Linie odniesienia
        self._ax_n.axhline(y= self.track.track_width/2,
                            color='#ff1744', lw=0.8, ls='--', alpha=0.6)
        self._ax_n.axhline(y=-self.track.track_width/2,
                            color='#ff1744', lw=0.8, ls='--', alpha=0.6)
        self._ax_vx.axhline(y=self.params.v_max,
                             color='#ff6f00', lw=0.8, ls='--', alpha=0.4)

    # ── Rysowanie samochodu ────────────────────────────────────────────────

    def _draw_car(self, X: float, Y: float, psi: float, delta: float):
        """Rysuje samochód jako prostokąt z 4 kółkami."""
        L  = self.CAR_LENGTH
        W  = self.CAR_WIDTH
        WL = self.WHEEL_L
        WW = self.WHEEL_W
        lf = self.params.lf
        lr = self.params.lr

        def rot(pts, angle):
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]])
            return (R @ pts.T).T

        # ── nadwozie ──
        # narożniki w układzie lokalnym pojazdu (środek = środek masy)
        corners_local = np.array([
            [ lf,      W/2],
            [ lf,     -W/2],
            [-lr,     -W/2],
            [-lr,      W/2],
        ])
        corners_world = rot(corners_local, psi) + np.array([X, Y])
        self._car_patch.set_visible(False)  # zastąpiony przez Polygon poniżej

        # Rysujemy nadwozie jako wypełniony wielokąt (zbuforowany)
        if not hasattr(self, '_car_poly'):
            from matplotlib.patches import Polygon as CarPoly
            self._car_poly = CarPoly(corners_world, closed=True,
                                     facecolor='#0d47a1', edgecolor='#00e5ff',
                                     linewidth=1.8, zorder=8)
            self._ax_track.add_patch(self._car_poly)
            # Pasek kabiny – ciemniejszy prostokąt w środku
            self._cabin_poly = CarPoly(corners_world * 0, closed=True,
                                       facecolor='#1565c0', edgecolor='none',
                                       alpha=0.6, zorder=9)
            self._ax_track.add_patch(self._cabin_poly)
        else:
            self._car_poly.set_xy(corners_world)
            # kabina (środkowa 1/3)
            cabin_local = np.array([
                [ lf*0.4,   W*0.35],
                [ lf*0.4,  -W*0.35],
                [-lr*0.5,  -W*0.35],
                [-lr*0.5,   W*0.35],
            ])
            cabin_world = rot(cabin_local, psi) + np.array([X, Y])
            self._cabin_poly.set_xy(cabin_world)

        # ── strzałka kierunku ──
        arr_end = np.array([X, Y]) + rot(np.array([[L*0.55, 0]]), psi)[0]
        self._arrow.set_position((X, Y))
        self._arrow.xy = arr_end

        # ── koła ──
        # Pozycje w układzie lokalnym: [przód-lewy, przód-prawy, tył-lewy, tył-prawy]
        wheel_pos_local = np.array([
            [ lf,  W/2 + WW*0.3],
            [ lf, -W/2 - WW*0.3],
            [-lr,  W/2 + WW*0.3],
            [-lr, -W/2 - WW*0.3],
        ])
        wheel_angles = [psi + delta, psi + delta, psi, psi]  # przednie skręcają

        for i, (wp, (lx_l, ly_l), wa) in enumerate(
                zip(self._wheel_patches, wheel_pos_local, wheel_angles)):

            # Środek koła w układzie świata
            cx_w, cy_w = rot(np.array([[lx_l, ly_l]]), psi)[0] + np.array([X, Y])

            # Narożniki koła w układzie lokalnym koła
            wc_local = np.array([
                [ WL/2,  WW/2],
                [ WL/2, -WW/2],
                [-WL/2, -WW/2],
                [-WL/2,  WW/2],
            ])
            wc_world = rot(wc_local, wa) + np.array([cx_w, cy_w])

            if not hasattr(self, '_wheel_polys'):
                self._wheel_polys = []
            if len(self._wheel_polys) <= i:
                from matplotlib.patches import Polygon as WPoly
                col = '#ffeb3b' if i < 2 else '#bdbdbd'  # przód żółty, tył szary
                p = WPoly(wc_world, closed=True,
                           facecolor='#212121', edgecolor=col,
                           linewidth=1.2, zorder=10)
                self._ax_track.add_patch(p)
                self._wheel_polys.append(p)
            else:
                self._wheel_polys[i].set_xy(wc_world)

    # ── Aktualizacja ───────────────────────────────────────────────────────

    def update(self, state: np.ndarray, u: np.ndarray, t: float):
        """
        Wywołuj co krok symulacji.

        Args:
            state: [s, n, mu, vx, vy, r]
            u:     [wheel_speed, delta]
            t:     bieżący czas [s]
        """
        if self._fig is None:
            return

        s, n, mu, vx, vy, r = state
        d, delta = u

        # Frenet → kartezjańskie
        X, Y, psi_track = self.track.frenet_to_cartesian(s, n)
        psi = psi_track + mu

        # Bufor historii
        self._xs.append(X);   self._ys.append(Y)
        self._vxs.append(vx); self._ns.append(n)
        self._ds.append(d);   self._deltas.append(delta)
        self._ts.append(t)

        # Przytnij do history_len
        if len(self._xs) > self.history_len:
            self._xs    = self._xs[-self.history_len:]
            self._ys    = self._ys[-self.history_len:]
            self._vxs   = self._vxs[-self.history_len:]
            self._ns    = self._ns[-self.history_len:]
            self._ds    = self._ds[-self.history_len:]
            self._deltas = self._deltas[-self.history_len:]
            self._ts    = self._ts[-self.history_len:]

        # ── ślad trajektorii ──
        self._trail_sc.set_offsets(np.c_[self._xs, self._ys])
        self._trail_sc.set_array(np.array(self._vxs))

        # ── samochód ──
        self._draw_car(X, Y, psi, delta)

        # ── kamera (follow cam) ──
        if self.follow_cam:
            view_r = max(self.track.track_width * 4, 2.5)
            self._ax_track.set_xlim(X - view_r, X + view_r)
            self._ax_track.set_ylim(Y - view_r, Y + view_r)

        # ── HUD ──
        kmh = vx * 3.6
        bar_len = int(min(kmh / (self.params.v_max * 3.6) * 12, 12))
        speed_bar = '█' * bar_len + '░' * (12 - bar_len)
        self._hud_speed.set_text(
            f' vx  {kmh:5.1f} km/h\n'
            f' [{speed_bar}]')

        self._hud_state.set_text(
            f' n   {n:+.3f} m\n'
            f' mu  {mu:+.3f} rad\n'
            f' vy  {vy:+.3f} m/s\n'
            f' r   {r:+.4f} rad/s\n'
            f' δ   {np.degrees(delta):+.1f}°\n'
            f' d   {d:.2f}')

        laps = s / self.track.total_length
        lap_int = int(laps)
        lap_frac = laps - lap_int
        prog_bar = int(lap_frac * 10)
        self._hud_lap.set_text(
            f' Okr: {lap_int}\n'
            f' [{("█"*prog_bar + "░"*(10-prog_bar))}]')

        # ── wykresy live ──
        ts = np.array(self._ts)
        self._line_vx.set_data(ts, self._vxs)
        self._line_n.set_data(ts, self._ns)
        self._line_d.set_data(ts, self._ds)
        self._line_dl.set_data(ts, self._deltas)

        for ax, data in [(self._ax_vx, self._vxs),
                         (self._ax_n, self._ns),
                         (self._ax_ctrl, self._ds + self._deltas)]:
            ax.set_xlim(ts[0] if len(ts) > 1 else 0, ts[-1] + 0.1)
            mn, mx = min(data), max(data)
            pad = max((mx - mn) * 0.15, 0.05)
            ax.set_ylim(mn - pad, mx + pad)

        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self):
        if self._fig is not None:
            plt.ioff()
            plt.show()   # zatrzymaj i pokaż ostatni stan

class F1tenthSimulator:
    """
    Główna klasa symulatora F1/10.

    Łączy:
      - model dynamiczny pojazdu (DynamicBicycleModel)
      - tor (TrackCenterline)
      - opcjonalną wizualizację PyBullet
      - wizualizację matplotlib (zawsze dostępna)

    Użycie:
        sim = F1tenthSimulator(track, params)
        sim.reset()
        for step in range(N):
            u = controller(sim.state)   # ← tu podpinamy MPC
            sim.step(u)
        sim.render()
    """

    def __init__(self,
                 track:          TrackCenterline,
                 vehicle_params: Optional[VehicleParams] = None,
                 dt:             float = 0.02,
                 use_pybullet:   bool  = True,
                 live_viz:       bool  = True,
                 follow_cam:     bool  = True):

        self.track   = track
        self.params  = vehicle_params or VehicleParams()
        self.model   = DynamicBicycleModel(self.params)
        self.dt      = dt
        self.time    = 0.0

        # Historia trajektorii
        self.state_history:  list = []
        self.input_history:  list = []
        self.time_history:   list = []

        # Stan bieżący [s, n, mu, vx, vy, r]
        self.state = np.zeros(6)

        # Live 2D visualizer
        self.viz: Optional[LiveVisualizer2D] = None
        if live_viz:
            self.viz = LiveVisualizer2D(track, self.params,
                                        history_secs=6.0,
                                        dt=dt,
                                        follow_cam=follow_cam)
            self.viz.start()

        # PyBullet
        self.use_pybullet = use_pybullet and PYBULLET_AVAILABLE
        self.pb_client = None
        self.car_id    = None
        self.track_visual_ids = []

        if self.use_pybullet:
            self._init_pybullet()

    # ──────────────────────────────────────────────────────────────────────
    #  INICJALIZACJA
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, s0: float = 0.0, n0: float = 0.0,
              mu0: float = 0.0, vx0: float = 1.0) -> np.ndarray:
        """Resetuje symulator do stanu początkowego."""
        self.state = np.array([s0, n0, mu0, vx0, 0.0, 0.0])
        self.time  = 0.0
        self.state_history  = [self.state.copy()]
        self.input_history  = []
        self.time_history   = [0.0]

        if self.use_pybullet and self.car_id is not None:
            X, Y, psi = self.track.frenet_to_cartesian(s0, n0)
            pb.resetBasePositionAndOrientation(
                self.car_id,
                [X, Y, 0.05],
                pb.getQuaternionFromEuler([0, 0, psi + mu0]),
                physicsClientId=self.pb_client
            )

        return self.state.copy()

    # ──────────────────────────────────────────────────────────────────────
    #  KROK SYMULACJI
    # ──────────────────────────────────────────────────────────────────────

    def step(self, u: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Wykonuje jeden krok symulacji.

        Args:
            u: wejście [wheel_speed, delta]
        
        Returns:
            state: nowy stan [s, n, mu, vx, vy, r]
            info:  słownik z dodatkowymi informacjami
        """
        u = np.asarray(u, dtype=float)

        # Krzywizna toru w bieżącej pozycji
        kappa = self.track.get_kappa(self.state[0])

        # Krok modelu dynamicznego (RK4)
        self.state = self.model.step_rk4(self.state, u, kappa, self.dt)
        self.time += self.dt

        # Historia
        self.state_history.append(self.state.copy())
        self.input_history.append(u.copy())
        self.time_history.append(self.time)

        # Informacje diagnostyczne
        info = {
            "kappa":      kappa,
            "lap_progress": self.state[0] / self.track.total_length,
            "out_of_bounds": abs(self.state[1]) > self.track.track_width / 2,
        }

        # Aktualizuj PyBullet
        if self.use_pybullet and self.car_id is not None:
            self._update_pybullet()

        # Aktualizuj live 2D wizualizację
        if self.viz is not None:
            self.viz.update(self.state, u, self.time)

        return self.state.copy(), info

    def run(self, controller: Callable, n_steps: int,
            verbose: bool = True,
            viz_every: int = 1) -> dict:
        """
        Uruchamia symulację przez n_steps kroków.

        Args:
            controller: funkcja u = controller(state) → np.ndarray shape (2,)
            n_steps:    liczba kroków
            verbose:    drukuj postęp co 100 kroków
            viz_every:  aktualizuj wizualizację co N kroków (1 = każdy krok)

        Returns:
            słownik z historią stanu, wejść i czasu
        """
        out_of_bounds_count = 0

        for i in range(n_steps):
            u = controller(self.state)

            # Tymczasowo wyłącz auto-update viz w step() dla kontroli częstości
            _viz = self.viz
            if i % viz_every != 0:
                self.viz = None
            _, info = self.step(u)
            self.viz = _viz

            # Ręczna aktualizacja wizualizacji co viz_every kroków
            if i % viz_every == 0 and self.viz is not None:
                self.viz.update(self.state, u, self.time)

            if info["out_of_bounds"]:
                out_of_bounds_count += 1

            if verbose and i % 100 == 0:
                s, n, mu, vx, vy, r = self.state
                print(f"  t={self.time:.2f}s | s={s:.2f}m | n={n:.3f}m "
                      f"| vx={vx:.2f}m/s | delta={u[1]*180/np.pi:.1f}°")

        if verbose:
            laps = self.state[0] / self.track.total_length
            print(f"\nZakończono: {n_steps} kroków ({self.time:.2f}s), "
                  f"{laps:.2f} okrążeń, "
                  f"wyjść poza tor: {out_of_bounds_count}")

        if self.viz is not None:
            self.viz.close()

        return {
            "states": np.array(self.state_history),
            "inputs": np.array(self.input_history),
            "times":  np.array(self.time_history),
        }

    # ──────────────────────────────────────────────────────────────────────
    #  PYBULLET
    # ──────────────────────────────────────────────────────────────────────

    def _init_pybullet(self):
        """Inicjalizuje środowisko PyBullet."""
        try:
            self.pb_client = pb.connect(pb.GUI)
            pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            pb.setGravity(0, 0, -9.81, physicsClientId=self.pb_client)
            pb.setTimeStep(self.dt, physicsClientId=self.pb_client)

            # Podłoga
            pb.loadURDF("plane.urdf", physicsClientId=self.pb_client)

            # Samochód – prosty prostopadłościan jako placeholder
            col_id = pb.createCollisionShape(
                pb.GEOM_BOX,
                halfExtents=[self.params.lf + self.params.lr,
                             0.08, 0.04],
                physicsClientId=self.pb_client
            )
            vis_id = pb.createVisualShape(
                pb.GEOM_BOX,
                halfExtents=[self.params.lf + self.params.lr,
                             0.08, 0.04],
                rgbaColor=[0.1, 0.5, 1.0, 1.0],
                physicsClientId=self.pb_client
            )
            self.car_id = pb.createMultiBody(
                baseMass=self.params.m,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[0, 0, 0.05],
                physicsClientId=self.pb_client
            )

            # Narysuj tor
            self._draw_track_pybullet()

            # Kamera
            pb.resetDebugVisualizerCamera(
                cameraDistance=8,
                cameraYaw=0,
                cameraPitch=-60,
                cameraTargetPosition=[0, 0, 0],
                physicsClientId=self.pb_client
            )
            print("[PyBullet] Środowisko zainicjalizowane.")

        except Exception as e:
            print(f"[PyBullet] Błąd inicjalizacji: {e}")
            self.use_pybullet = False

    def _draw_track_pybullet(self):
        """Rysuje centerline toru w PyBullet jako linie."""
        if not self.use_pybullet:
            return
        n_pts = len(self.track.x)
        for i in range(n_pts):
            j = (i + 1) % n_pts
            x1, y1 = self.track.x[i], self.track.y[i]
            x2, y2 = self.track.x[j], self.track.y[j]

            # Centerline – żółty
            pb.addUserDebugLine(
                [x1, y1, 0.01], [x2, y2, 0.01],
                lineColorRGB=[1, 1, 0], lineWidth=1,
                physicsClientId=self.pb_client
            )

            # Krawędzie toru (lewy/prawy) – białe, co kilka punktów
            if i % 3 == 0:
                hw = self.track.track_width / 2
                # Tangent → normalny
                if i < n_pts - 1:
                    dx = self.track.x[i+1] - self.track.x[i]
                    dy = self.track.y[i+1] - self.track.y[i]
                else:
                    dx = self.track.x[0] - self.track.x[-1]
                    dy = self.track.y[0] - self.track.y[-1]
                norm = np.sqrt(dx**2 + dy**2) + 1e-9
                nx_v, ny_v = -dy/norm, dx/norm  # wektor normalny

                for sign, col in [(1, [1,1,1]), (-1, [1,1,1])]:
                    xl = x1 + sign * hw * nx_v
                    yl = y1 + sign * hw * ny_v
                    xr = x2 + sign * hw * nx_v
                    yr = y2 + sign * hw * ny_v
                    if i % 9 == 0:  # co 9. punkt – krótki odcinek
                        pb.addUserDebugLine(
                            [xl, yl, 0.01], [xr, yr, 0.01],
                            lineColorRGB=col, lineWidth=1,
                            physicsClientId=self.pb_client
                        )

    def _update_pybullet(self):
        """Aktualizuje pozycję samochodu w PyBullet."""
        s, n, mu, vx, vy, r = self.state
        X, Y, psi_track = self.track.frenet_to_cartesian(s, n)
        psi_vehicle = psi_track + mu

        pb.resetBasePositionAndOrientation(
            self.car_id,
            [X, Y, 0.05],
            pb.getQuaternionFromEuler([0, 0, psi_vehicle]),
            physicsClientId=self.pb_client
        )
        pb.stepSimulation(physicsClientId=self.pb_client)

    def close_pybullet(self):
        if self.pb_client is not None:
            pb.disconnect(self.pb_client)
            self.pb_client = None

    # ──────────────────────────────────────────────────────────────────────
    #  WIZUALIZACJA MATPLOTLIB
    # ──────────────────────────────────────────────────────────────────────

    def plot_results(self):
        """Rysuje wyniki symulacji."""
        if len(self.state_history) < 2:
            print("Brak danych do narysowania.")
            return

        states = np.array(self.state_history)   # (T, 6)
        inputs = np.array(self.input_history)    # (T-1, 2)
        times  = np.array(self.time_history)     # (T,)

        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle("F1/10 Symulator – historia symulacji", fontsize=14, fontweight='bold')

        labels_states = ['s [m]', 'n [m]', 'mu [rad]', 'vx [m/s]', 'vy [m/s]', 'r [rad/s]']
        colors_states = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']

        for i, ax in enumerate(axes.flat):
            ax.plot(times, states[:, i], color=colors_states[i], linewidth=1.8)
            ax.set_ylabel(labels_states[i], fontsize=11)
            ax.set_xlabel('czas [s]')
            ax.grid(True, alpha=0.3)
            if i == 1:
                ax.axhline(y=self.track.track_width/2,  color='red',
                           linestyle='--', alpha=0.6, label='granica toru')
                ax.axhline(y=-self.track.track_width/2, color='red',
                           linestyle='--', alpha=0.6)
                ax.legend(fontsize=9)

        plt.tight_layout()
        plt.show()

    def plot_trajectory(self):
        """Rysuje trajektorię pojazdu na torze."""
        if len(self.state_history) < 2:
            print("Brak danych.")
            return

        states = np.array(self.state_history)
        track  = self.track

        # Konwertuj Frenet → kartezjańskie
        xs, ys = [], []
        for row in states[::5]:  # co 5. punkt dla szybkości
            X, Y, _ = track.frenet_to_cartesian(row[0], row[1])
            xs.append(X)
            ys.append(Y)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Tor
        hw = track.track_width / 2
        n_pts = len(track.x)

        left_x, left_y, right_x, right_y = [], [], [], []
        for i in range(n_pts):
            j = (i + 1) % n_pts
            dx = track.x[j] - track.x[i]
            dy = track.y[j] - track.y[i]
            norm = np.sqrt(dx**2 + dy**2) + 1e-9
            nx_v, ny_v = -dy/norm, dx/norm
            left_x.append(track.x[i] + hw * nx_v)
            left_y.append(track.y[i] + hw * ny_v)
            right_x.append(track.x[i] - hw * nx_v)
            right_y.append(track.y[i] - hw * ny_v)

        left_x.append(left_x[0]); left_y.append(left_y[0])
        right_x.append(right_x[0]); right_y.append(right_y[0])

        ax.fill(left_x + right_x[::-1],
                left_y + right_y[::-1],
                alpha=0.15, color='gray', label='tor')
        ax.plot(left_x,  left_y,  'k-', linewidth=1.5, alpha=0.6)
        ax.plot(right_x, right_y, 'k-', linewidth=1.5, alpha=0.6)
        ax.plot(track.x, track.y, 'y--', linewidth=1, alpha=0.8, label='centerline')

        # Trajektoria (kolorowana prędkością)
        vx_vals = states[::5, 3]
        sc = ax.scatter(xs, ys, c=vx_vals, cmap='plasma',
                        s=8, zorder=5, label='trajektoria')
        plt.colorbar(sc, ax=ax, label='vx [m/s]')

        ax.plot(xs[0], ys[0], 'go', markersize=10, label='start', zorder=6)
        ax.plot(xs[-1], ys[-1], 'r^', markersize=10, label='koniec', zorder=6)

        ax.set_aspect('equal')
        ax.set_title('Trajektoria pojazdu na torze')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def animate(self, interval_ms: int = 30) -> FuncAnimation:
        """Animacja trajektorii w matplotlib."""
        states = np.array(self.state_history)
        track  = self.track

        xs, ys = [], []
        for row in states:
            X, Y, _ = track.frenet_to_cartesian(row[0], row[1])
            xs.append(X)
            ys.append(Y)
        xs = np.array(xs)
        ys = np.array(ys)

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_aspect('equal')
        ax.plot(track.x, track.y, 'y--', linewidth=1, alpha=0.6)

        hw = track.track_width / 2
        n_pts = len(track.x)
        for i in range(n_pts):
            j = (i + 1) % n_pts
            dx = track.x[j] - track.x[i]
            dy = track.y[j] - track.y[i]
            norm = np.sqrt(dx**2 + dy**2) + 1e-9
            nx_v, ny_v = -dy/norm, dx/norm
            for sign in [1, -1]:
                ax.plot(
                    [track.x[i] + sign*hw*nx_v, track.x[j] + sign*hw*nx_v],
                    [track.y[i] + sign*hw*ny_v, track.y[j] + sign*hw*ny_v],
                    'k-', linewidth=1, alpha=0.4
                )

        trail, = ax.plot([], [], 'b-', linewidth=1.5, alpha=0.6)
        car_dot, = ax.plot([], [], 'ro', markersize=10, zorder=10)
        title = ax.set_title('')
        ax.set_xlim(xs.min()-1, xs.max()+1)
        ax.set_ylim(ys.min()-1, ys.max()+1)

        trail_len = 50  # punkty historii do pokazania

        def update(frame):
            lo = max(0, frame - trail_len)
            trail.set_data(xs[lo:frame+1], ys[lo:frame+1])
            car_dot.set_data([xs[frame]], [ys[frame]])
            s, n, mu, vx = states[frame, :4]
            title.set_text(
                f"t={self.time_history[frame]:.2f}s | "
                f"s={s:.2f}m | n={n:.3f}m | vx={vx:.2f}m/s"
            )
            return trail, car_dot, title

        ani = FuncAnimation(fig, update,
                            frames=len(xs),
                            interval=interval_ms,
                            blit=True)
        plt.tight_layout()
        return ani


# ═══════════════════════════════════════════════════════════════════════════
#  PRZYKŁADOWE KONTROLERY (do testowania symulatora)
# ═══════════════════════════════════════════════════════════════════════════

def controller_zero(state: np.ndarray) -> np.ndarray:
    """Brak sterowania – samochód zatrzymuje się."""
    return np.array([0.0, 0.0])


def controller_const_speed(speed: float = 0.3, kp_steering: float = 2.0):
    """
    Prosty kontroler proporcjonalny:
    - stała prędkość kół
    - steering proporcjonalny do n (utrzymywanie środka toru)
    (NIE jest to MPC – tylko placeholder do testów symulatora)
    """
    def controller(state: np.ndarray) -> np.ndarray:
        _, n, mu, vx, _, _ = state
        # P-kontroler: skręcaj żeby zmniejszyć n i mu
        delta = -kp_steering * n - 1.5 * mu
        delta = np.clip(delta, -0.35, 0.35)
        return np.array([speed, delta])
    return controller


def controller_random(state: np.ndarray) -> np.ndarray:
    """Losowe sterowanie – do testów jak model reaguje na chaos."""
    d     = np.random.uniform(0.1, 0.5)
    delta = np.random.uniform(-0.2, 0.2)
    return np.array([d, delta])


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN – demonstracja
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  F1/10 Symulator – demonstracja")
    print("  Model: single-track dynamics + Pacejka tires")
    print("  Układ: Frenet [s, n, mu, vx, vy, r]")
    print("=" * 60)

    # ── Wybierz tor ──────────────────────────────────────────────────────
    # Opcje: 'oval', 'figure8'
    TRACK_TYPE = 'oval'

    # ── Opcje wizualizacji ───────────────────────────────────────────────
    LIVE_VIZ   = True   # live 2D animacja w matplotlib
    FOLLOW_CAM = True   # kamera śledzi samochód (False = cały tor widoczny)
    PYBULLET   = False  # PyBullet 3D (wymaga: pip install pybullet)

    # ── Czas symulacji ───────────────────────────────────────────────────
    SIM_SECONDS = 20.0
    DT = 0.02
    N_STEPS = int(SIM_SECONDS / DT)

    # 1. Stwórz tor
    print(f"\n[1/3] Generowanie toru '{TRACK_TYPE}'...")
    if TRACK_TYPE == 'oval':
        track = TrackCenterline.make_oval(length=8.0, width=4.0, track_width=0.35)
    else:
        track = TrackCenterline.make_figure8(r=2.5, track_width=0.35)
    print(f"      Długość toru: {track.total_length:.2f} m")

    # 2. Symulator + wizualizacja live
    print("[2/3] Inicjalizacja symulatora i wizualizacji...")
    params = VehicleParams()
    sim = F1tenthSimulator(
        track=track,
        vehicle_params=params,
        dt=DT,
        use_pybullet=PYBULLET,
        live_viz=LIVE_VIZ,
        follow_cam=FOLLOW_CAM,
    )
    sim.reset(s0=0.0, n0=0.0, mu0=0.0, vx0=1.0)

    # 3. Symulacja z prostym kontrolerem P (placeholder przed MPC)
    print(f"[3/3] Symulacja {SIM_SECONDS:.0f}s z kontrolerem proporcjonalnym...")
    print("      (okno matplotlib otworzy się automatycznie)\n")

    ctrl = controller_const_speed(speed=0.38, kp_steering=2.0)

    # viz_every=1 → aktualizuj co krok; zwiększ jeśli za wolno
    result = sim.run(ctrl, n_steps=N_STEPS, verbose=True, viz_every=1)

    if sim.use_pybullet:
        print("\nPyBullet: naciśnij Enter żeby zamknąć...")
        input()
        sim.close_pybullet()

    print("\nGotowe! Symulator działa.")
    print("Następny krok: podpięcie regulatora MPC (kamień milowy 2).")
