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
    lf: float = 0.15875
    lr: float = 0.17145
    m:  float = 3.47
    Iz: float = 0.04712

    # Pacejka - poprawione wartości
    Bf: float = 9.242
    Cf: float = 1.2
    Df: float = 134.585   # ← było 0.192, siła w [N] nie w [×mg] !

    Br: float = 17.716
    Cr: float = 1.2
    Dr: float = 159.919   # ← było 0.1737

    # Napęd - też poprawione
    Cm1: float = 20.0
    Cm2: float = 0.0
    Cr0: float = 0.05
    Cr2: float = 0.0

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
#  SYMULATOR
# ═══════════════════════════════════════════════════════════════════════════

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
                 use_pybullet:   bool  = True):

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

        return self.state.copy(), info

    def run(self, controller: Callable, n_steps: int,
            verbose: bool = True) -> dict:
        """
        Uruchamia symulację przez n_steps kroków.

        Args:
            controller: funkcja u = controller(state) → np.ndarray shape (2,)
            n_steps:    liczba kroków
            verbose:    drukuj postęp

        Returns:
            słownik z historią stanu, wejść i czasu
        """
        out_of_bounds_count = 0

        for i in range(n_steps):
            u = controller(self.state)
            _, info = self.step(u)

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

def controller_pid(kp=2.0, ki=0.1, kd=0.5, speed=0.35):
    """
    Kontroler PID utrzymujący pojazd na środku toru.
    P - proporcjonalny do n (aktualne odchylenie)
    I - całka z n (eliminuje błąd stały)
    D - pochodna n (tłumi oscylacje)
    """
    integral = [0.0]
    prev_n   = [0.0]
    dt       = 0.02  # musi zgadzać się z dt symulatora

    def controller(state):
        _, n, mu, vx, _, _ = state

        # Człony PID
        integral[0] += n * dt
        derivative    = (n - prev_n[0]) / dt
        prev_n[0]     = n

        delta = -(kp * n + ki * integral[0] + kd * derivative) - 1.0 * mu
        delta = np.clip(delta, -0.35, 0.35)
        return np.array([speed, delta])

    return controller


# ═══════════════════════════════════════════════════════════════════════════
#  MPC
# ═══════════════════════════════════════════════════════════════════════════

from scipy.optimize import minimize

class MPCController:

    def __init__(self, model, track, N=20, dt=0.02, vx_ref=1.5,
                 q_n=15.0, q_mu=5.0, q_vx=2.0,
                 r_d=0.1, r_delta=0.5, r_dd=0.5, r_ddelta=1.0):

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

        self.delta_max = model.p.delta_max
        self.d_max     = 1.0
        self.d_min     = 0.0
        self.u_prev    = np.array([0.2, 0.0])
        self.U_warm    = np.tile(self.u_prev, (N, 1))

    def _cost(self, U_flat, x0):
        U      = U_flat.reshape(self.N, 2)
        x      = x0.copy()
        J      = 0.0
        u_prev = self.u_prev.copy()

        for k in range(self.N):
            u     = U[k]
            d     = np.clip(u[0], self.d_min, self.d_max)
            delta = np.clip(u[1], -self.delta_max, self.delta_max)

            s, n, mu, vx, vy, r = x

            J += self.q_n   * n**2
            J += self.q_mu  * mu**2
            J += self.q_vx  * (vx - self.vx_ref)**2
            J += self.r_d   * d**2
            J += self.r_delta * delta**2
            J += self.r_dd    * (d     - u_prev[0])**2
            J += self.r_ddelta * (delta - u_prev[1])**2

            # Miękka kara za wyjście poza tor
            tw_half = self.track.track_width / 2.0
            if abs(n) > tw_half * 0.8:
                J += 100.0 * (abs(n) - tw_half * 0.8)**2

            kappa = self.track.get_kappa(s)
            x     = self.model.step_rk4(x, np.array([d, delta]), kappa, self.dt)
            u_prev = np.array([d, delta])

        # Koszt końcowy
        s, n, mu, vx, vy, r = x
        J += 3.0 * self.q_n  * n**2
        J += 3.0 * self.q_mu * mu**2
        J += 1.0 * self.q_vx * (vx - self.vx_ref)**2

        return J

    def compute_control(self, state):
        U0      = np.roll(self.U_warm, -1, axis=0)
        U0[-1]  = U0[-2]
        U0_flat = U0.flatten()

        bounds = []
        for _ in range(self.N):
            bounds.append((self.d_min,       self.d_max))
            bounds.append((-self.delta_max,  self.delta_max))

        result = minimize(
            fun=self._cost,
            x0=U0_flat,
            args=(state,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 20, 'ftol': 1e-3, 'gtol': 1e-2}
        )

        U_opt      = result.x.reshape(self.N, 2)
        self.U_warm = U_opt.copy()

        u_opt    = U_opt[0].copy()
        u_opt[0] = np.clip(u_opt[0], self.d_min,      self.d_max)
        u_opt[1] = np.clip(u_opt[1], -self.delta_max, self.delta_max)

        self.u_prev = u_opt.copy()
        return u_opt

    def __call__(self, state):
        return self.compute_control(state)

# ═══════════════════════════════════════════════════════════════════════════
#  MAIN – demonstracja
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  F1/10 Symulator – demonstracja")
    print("  Model: single-track dynamics + Pacejka tires")
    print("  Układ: Frenet [s, n, mu, vx, vy, r]")
    print("=" * 60)

    # 1. Stwórz tor
    print("\n[1/4] Generowanie toru owalnego...")
    track = TrackCenterline.make_oval(length=8.0, width=4.0,
                                      track_width=0.35)
    print(f"      Długość toru: {track.total_length:.2f} m")

    # 2. Stwórz symulator
    print("[2/4] Inicjalizacja symulatora...")
    params = VehicleParams()
    sim = F1tenthSimulator(
        track=track,
        vehicle_params=params,
        dt=0.02,
        #use_pybullet=True   # zmień na False jeśli PyBullet niedostępny
        use_pybullet = False
    )

    # 3. Reset i uruchom
    print("[3/4] Symulacja")
    sim.reset(s0=0.0, n0=0.0, mu0=0.0, vx0=1.0)

    ctrl = controller_const_speed(speed=0.35, kp_steering=1.8)
    # ctrl = controller_zero
    # ctrl = controller_random
    # ctrl = controller_pid(kp=2.0, ki=0.1, kd=0.5, speed=0.35)

    # Użyj MPC:
    # ctrl = MPCController(
    #     model=sim.model,
    #     track=track,
    #     N=10,  # horyzont 20 kroków × 0.02s = 0.4s w przód
    #     dt=0.02,
    #     vx_ref=4.5,  # docelowa prędkość [m/s]
    #     q_n=15.0,  # kara za zjazd z toru
    #     q_mu=5.0,  # kara za kąt
    #     q_vx=2.0,  # kara za prędkość
    # )


    # Symulacja 10 sekund (500 kroków po 20ms)
    n_steps = 200
    result  = sim.run(ctrl, n_steps=n_steps, verbose=True)

    # 4. Wyniki
    print("\n[4/4] Rysowanie wyników...")
    sim.plot_trajectory()
    sim.plot_results()

    # Animacja (odkomentuj jeśli chcesz)
    ani = sim.animate(interval_ms=20)
    plt.show()

    if sim.use_pybullet:
        print("\nPyBullet: naciśnij Enter żeby zamknąć...")
        input()
        sim.close_pybullet()

    print("\nGotowe! Symulator działa.")
    print("Następny krok: podpięcie regulatora MPC (kamień milowy 2).")


