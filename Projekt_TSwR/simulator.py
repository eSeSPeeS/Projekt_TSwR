"""
simulator.py
============
Klasa F1tenthSimulator — łączy model dynamiczny, tor i wizualizację.

Nowości:
- wizualizacja matplotlib w czasie rzeczywistym podczas symulacji,
- parametr log_every określający co który krok wypisywać log,
- końcowe wykresy i animacja pokazują się po symulacji,
- WĄŻ PREDYKCYJNY: jeśli kontroler ma pole prediction_xy (MPCController),
  na wykresie realtime rysowany jest zielony wąż przewidywanej trajektorii.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Callable, Tuple

from vehicle_params import VehicleParams
from vehicle_model import DynamicBicycleModel
from track import TrackCenterline

try:
    import pybullet as pb
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("[INFO] PyBullet niedostępny – tylko matplotlib.")



# ══════════════════════════════════════════════════════════════════════════════
# LOGGER SYMULACJI
# ══════════════════════════════════════════════════════════════════════════════

import csv, json, datetime

class SimulationLogger:
    """
    Zapisuje dane symulacji do pliku CSV i opcjonalnie JSON.

    Dla każdej iteracji zapisuje:
      - czas, stany Freneta [s, n, mu, vx, vy, r]
      - pozycja kartezjańska pojazdu i osi przedniej [X, Y, psi, Xf, Yf]
      - sterowanie [d, delta_deg]
      - status solvera (kod, błąd T/F, koszt)
      - pełny wąż predykcyjny (N+1 punktów Freneta i kartezjańskich)

    Użycie:
        logger = SimulationLogger("wyniki.csv")
        # ... w pętli po każdym kroku:
        logger.log(t, state, u, track, ctrl)
        # na końcu:
        logger.close()
        logger.save_json("wyniki.json")   # opcjonalnie
    """

    # Kolumny stałe (bez predykcji)
    FIXED_COLS = [
        "iter", "t_s",
        "s_m", "n_m", "mu_rad", "vx_ms", "vy_ms", "r_rads",
        "X_m", "Y_m", "psi_rad",
        "Xf_m", "Yf_m",
        "d", "delta_deg", "delta_rad",
        "solver_status", "solver_error", "solver_cost",
    ]

    def __init__(self, csv_path: str, lf: float = 0.15875):
        """
        Args:
            csv_path: ścieżka do pliku .csv
            lf:       odległość CoG → oś przednia [m]
        """
        self.csv_path  = csv_path
        self.lf        = lf
        self._rows: list = []
        self._N: int    = 0          # horyzont predykcji (ustalany przy pierwszym logu)
        self._header_written = False
        self._csvfile  = open(csv_path, 'w', newline='', encoding='utf-8')
        self._writer   = None        # tworzony po ustaleniu N

        print(f"[Logger] Otwarto: {csv_path}")

    def log(self,
            iteration:  int,
            t:          float,
            state:      np.ndarray,
            u:          np.ndarray,
            track,
            controller=None) -> None:
        """
        Loguje jeden krok symulacji.

        Args:
            iteration:  numer iteracji
            t:          czas symulacji [s]
            state:      stan pojazdu [s, n, mu, vx, vy, r]
            u:          sterowanie [d, delta]
            track:      TrackCenterline (do konwersji Frenet→kart.)
            controller: MPCController lub None (do pobrania statusu solvera
                        i danych predykcji)
        """
        s, n, mu, vx, vy, r = state
        d, delta = float(u[0]), float(u[1])

        X, Y, psi_track = track.frenet_to_cartesian(s, n)
        psi    = psi_track + mu
        Xf     = X + self.lf * np.cos(psi)
        Yf     = Y + self.lf * np.sin(psi)

        # Status solvera
        solver_status = -1
        solver_error  = False
        solver_cost   = 0.0
        if controller is not None and hasattr(controller, 'last_solver_status'):
            solver_status = int(controller.last_solver_status)
            solver_error  = bool(controller.last_solver_error)
            solver_cost   = float(controller.last_cost)

        # Predykcja MPC
        pred_states = None   # (N+1, 6) Frenet
        pred_xy     = None   # (N+1, 2) kartezjańskie
        N = 0
        if controller is not None:
            if hasattr(controller, 'prediction_states') and controller.prediction_states is not None:
                pred_states = controller.prediction_states
                N = len(pred_states) - 1
            if hasattr(controller, 'prediction_xy') and controller.prediction_xy is not None:
                pred_xy = controller.prediction_xy

        # Ustal nagłówek przy pierwszym logu (N może być != 0)
        if not self._header_written:
            self._N = N
            pred_cols = []
            for k in range(N + 1):
                for col in ['s', 'n', 'mu', 'vx', 'vy', 'r']:
                    pred_cols.append(f"pred_{k}_{col}")
                pred_cols.append(f"pred_{k}_X")
                pred_cols.append(f"pred_{k}_Y")
            header = self.FIXED_COLS + pred_cols
            self._writer = csv.DictWriter(self._csvfile, fieldnames=header)
            self._writer.writeheader()
            self._header_written = True

        # Buduj wiersz
        row = {
            "iter":          iteration,
            "t_s":           round(t, 4),
            "s_m":           round(float(s), 6),
            "n_m":           round(float(n), 6),
            "mu_rad":        round(float(mu), 6),
            "vx_ms":         round(float(vx), 6),
            "vy_ms":         round(float(vy), 6),
            "r_rads":        round(float(r), 6),
            "X_m":           round(float(X), 6),
            "Y_m":           round(float(Y), 6),
            "psi_rad":       round(float(psi), 6),
            "Xf_m":          round(float(Xf), 6),
            "Yf_m":          round(float(Yf), 6),
            "d":             round(d, 6),
            "delta_deg":     round(float(delta * 180 / np.pi), 4),
            "delta_rad":     round(float(delta), 6),
            "solver_status": solver_status,
            "solver_error":  int(solver_error),
            "solver_cost":   round(solver_cost, 6),
        }

        # Dane predykcji
        for k in range(self._N + 1):
            if pred_states is not None and k < len(pred_states):
                ps, pn, pmu, pvx, pvy, pr = pred_states[k]
            else:
                ps = pn = pmu = pvx = pvy = pr = 0.0
            if pred_xy is not None and k < len(pred_xy):
                pX, pY = pred_xy[k]
            else:
                pX = pY = 0.0

            row[f"pred_{k}_s"]   = round(float(ps),  6)
            row[f"pred_{k}_n"]   = round(float(pn),  6)
            row[f"pred_{k}_mu"]  = round(float(pmu), 6)
            row[f"pred_{k}_vx"]  = round(float(pvx), 6)
            row[f"pred_{k}_vy"]  = round(float(pvy), 6)
            row[f"pred_{k}_r"]   = round(float(pr),  6)
            row[f"pred_{k}_X"]   = round(float(pX),  6)
            row[f"pred_{k}_Y"]   = round(float(pY),  6)

        self._writer.writerow(row)
        self._rows.append(row)

    def flush(self) -> None:
        """Wymusza zapis bufora na dysk."""
        self._csvfile.flush()

    def close(self) -> None:
        """Zamyka plik CSV."""
        if not self._csvfile.closed:
            self._csvfile.flush()
            self._csvfile.close()
        print(f"[Logger] Zamknięto: {self.csv_path} ({len(self._rows)} wierszy)")

    def save_json(self, json_path: str) -> None:
        """
        Zapisuje te same dane do pliku JSON (czytelny, ale większy).

        Args:
            json_path: ścieżka do pliku .json
        """
        meta = {
            "created":      datetime.datetime.now().isoformat(),
            "csv_path":     self.csv_path,
            "n_steps":      len(self._rows),
            "horizon_N":    self._N,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({"meta": meta, "data": self._rows}, f,
                      indent=2, ensure_ascii=False)
        print(f"[Logger] JSON zapisany: {json_path}")

    def summary(self) -> dict:
        """Zwraca podstawowe statystyki po symulacji."""
        if not self._rows:
            return {}
        n_vals  = [abs(r['n_m'])         for r in self._rows]
        vx_vals = [r['vx_ms']            for r in self._rows]
        err_cnt = sum(r['solver_error']  for r in self._rows)
        oob     = sum(1 for r in self._rows
                      if abs(r['n_m']) > 0.5)   # >0.5m od centerline – heurystyczne
        return {
            'n_steps':       len(self._rows),
            'solver_errors': err_cnt,
            'out_of_bounds': oob,
            'n_max_m':       max(n_vals),
            'n_mean_m':      sum(n_vals) / len(n_vals),
            'vx_mean_ms':    sum(vx_vals) / len(vx_vals),
            'vx_max_ms':     max(vx_vals),
        }


class F1tenthSimulator:
    def __init__(self,
                 track: TrackCenterline,
                 vehicle_params: Optional[VehicleParams] = None,
                 dt: float = 0.02,
                 use_pybullet: bool = True):
        self.track  = track
        self.params = vehicle_params or VehicleParams()
        self.model  = DynamicBicycleModel(self.params)
        self.dt     = dt
        self.time   = 0.0

        self.state_history: list = []
        self.input_history:  list = []
        self.time_history:   list = []
        self.state = np.zeros(6)

        self.use_pybullet = use_pybullet and PYBULLET_AVAILABLE
        self.pb_client    = None
        self.car_id       = None

        # ── Realtime plot state ───────────────────────────────────────────
        self._rt_fig      = None
        self._rt_ax       = None
        self._rt_title    = None
        self._rt_car_dot  = None
        self._rt_trail    = None
        self._rt_text     = None
        self._rt_enabled  = False
        self._rt_trail_len = 100

        # Wąż predykcyjny – Line2D tworzony dynamicznie w _setup_realtime_plot
        self._rt_snake    = None   # linia węża (zielona)
        self._rt_snake_dots = None  # punkty węża (kółka)
        self._controller_ref = None  # referencja do kontrolera (ustawiana w run())

        if self.use_pybullet:
            self._init_pybullet()

    # ══════════════════════════════════════════════════════════════════════
    # RESET / STEP / RUN
    # ══════════════════════════════════════════════════════════════════════

    def reset(self, s0: float = 1.0, n0: float = 0.0,
              mu0: float = 0.0, vx0: float = 1.0) -> np.ndarray:
        self.state = np.array([s0, n0, mu0, vx0, 0.0, 0.0], dtype=float)
        self.time  = 0.0
        self.state_history = [self.state.copy()]
        self.input_history  = []
        self.time_history   = [0.0]

        if self.use_pybullet and self.car_id is not None:
            X, Y, psi = self.track.frenet_to_cartesian(s0, n0)
            pb.resetBasePositionAndOrientation(
                self.car_id,
                [X, Y, 0.05],
                pb.getQuaternionFromEuler([0, 0, psi + mu0]),
                physicsClientId=self.pb_client,
            )

        if self._rt_enabled:
            self._update_realtime_plot(force=True)

        return self.state.copy()

    def step(self, u: np.ndarray) -> Tuple[np.ndarray, dict]:
        u     = np.asarray(u, dtype=float)
        kappa = self.track.get_kappa(self.state[0])

        self.state  = self.model.step_rk4(self.state, u, kappa, self.dt)
        self.time  += self.dt

        self.state_history.append(self.state.copy())
        self.input_history.append(u.copy())
        self.time_history.append(self.time)

        info = {
            "kappa":        kappa,
            "lap_progress": self.state[0] / self.track.total_length,
            "out_of_bounds": abs(self.state[1]) > self.track.track_width / 2,
        }

        if self.use_pybullet and self.car_id is not None:
            self._update_pybullet()

        return self.state.copy(), info

    def run(self,
            controller: Callable,
            n_steps: int,
            verbose: bool = True,
            log_every: int = 100,
            realtime_plot: bool = False,
            realtime_interval_steps: int = 1,
            realtime_pause: float = 0.001,
            logger: "SimulationLogger" = None,
            log_flush_every: int = 50) -> dict:
        """
        Args:
            logger:          SimulationLogger lub None – jeśli podany, każdy krok
                             jest zapisywany do pliku CSV.
            log_flush_every: co ile kroków wymuszać zapis bufora na dysk.
        """
        oob = 0
        log_every               = max(1, int(log_every))
        realtime_interval_steps = max(1, int(realtime_interval_steps))

        # Zachowaj referencję do kontrolera – potrzebna dla węża predykcyjnego
        self._controller_ref = controller

        if realtime_plot:
            self._setup_realtime_plot()
            self._update_realtime_plot(force=True)

        for i in range(n_steps):
            u = controller(self.state)
            _, info = self.step(u)

            if info["out_of_bounds"]:
                oob += 1

            # ── Logger ────────────────────────────────────────────────────
            if logger is not None:
                logger.log(
                    iteration=i,
                    t=self.time,
                    state=self.state,
                    u=u,
                    track=self.track,
                    controller=controller,
                )
                if i % log_flush_every == 0:
                    logger.flush()

            if verbose and i % log_every == 0:
                s, n, mu, vx, vy, r = self.state
                X, Y, psi_track = self.track.frenet_to_cartesian(s, n)
                psi     = psi_track + mu
                X_front = X + self.params.lf * np.cos(psi)
                Y_front = Y + self.params.lf * np.sin(psi)
                # Pokaż status solvera w logu jeśli MPC
                solver_info = ""
                if hasattr(controller, 'last_solver_status'):
                    st  = controller.last_solver_status
                    err = controller.last_solver_error
                    solver_info = f" | solver={st}{' ERR' if err else ''}"
                print(
                    f"iter={i:04d} | "
                    f"t={self.time:.2f}s | s={s:.2f}m | n={n:.3f}m | "
                    f"vx={vx:.2f}m/s | delta={u[1] * 180 / np.pi:.1f}° | "
                    f"Xf={X_front:.3f}m | Yf={Y_front:.3f}m | "
                    f"psi={psi:.3f}rad{solver_info}"
                )

            if realtime_plot and (i % realtime_interval_steps == 0):
                self._update_realtime_plot()
                plt.pause(realtime_pause)

        if realtime_plot:
            self._update_realtime_plot(force=True)
            plt.pause(0.001)

        if verbose:
            laps = self.state[0] / self.track.total_length
            print(
                f"\nZakończono: {n_steps} kroków ({self.time:.2f}s), "
                f"{laps:.2f} okrążeń, wyjść poza tor: {oob}"
            )

        return {
            "states": np.array(self.state_history),
            "inputs": np.array(self.input_history),
            "times":  np.array(self.time_history),
        }

    # ══════════════════════════════════════════════════════════════════════
    # PYBULLET
    # ══════════════════════════════════════════════════════════════════════

    def _init_pybullet(self):
        try:
            self.pb_client = pb.connect(pb.GUI)
            pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            pb.setGravity(0, 0, -9.81, physicsClientId=self.pb_client)
            pb.setTimeStep(self.dt, physicsClientId=self.pb_client)
            pb.loadURDF("plane.urdf", physicsClientId=self.pb_client)

            col_id = pb.createCollisionShape(
                pb.GEOM_BOX,
                halfExtents=[self.params.L, 0.08, 0.04],
                physicsClientId=self.pb_client,
            )
            vis_id = pb.createVisualShape(
                pb.GEOM_BOX,
                halfExtents=[self.params.L, 0.08, 0.04],
                rgbaColor=[0.1, 0.5, 1.0, 1.0],
                physicsClientId=self.pb_client,
            )
            self.car_id = pb.createMultiBody(
                baseMass=self.params.m,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[0, 0, 0.05],
                physicsClientId=self.pb_client,
            )
            self._draw_track_pybullet()
            pb.resetDebugVisualizerCamera(
                cameraDistance=8,
                cameraYaw=0,
                cameraPitch=-60,
                cameraTargetPosition=[0, 0, 0],
                physicsClientId=self.pb_client,
            )
            print("[PyBullet] Środowisko zainicjalizowane.")
        except Exception as e:
            print(f"[PyBullet] Błąd: {e}")
            self.use_pybullet = False

    def _draw_track_pybullet(self):
        if not self.use_pybullet:
            return
        lx, ly, rx, ry = self.track.get_track_boundaries()
        n = len(self.track.x)

        for i in range(n):
            j = (i + 1) % n
            pb.addUserDebugLine(
                [self.track.x[i], self.track.y[i], 0.01],
                [self.track.x[j], self.track.y[j], 0.01],
                lineColorRGB=[1, 1, 0], lineWidth=1,
                physicsClientId=self.pb_client,
            )
            if i % 3 == 0:
                pb.addUserDebugLine(
                    [lx[i], ly[i], 0.01], [lx[j], ly[j], 0.01],
                    lineColorRGB=[1, 1, 1], lineWidth=1,
                    physicsClientId=self.pb_client,
                )
                pb.addUserDebugLine(
                    [rx[i], ry[i], 0.01], [rx[j], ry[j], 0.01],
                    lineColorRGB=[1, 1, 1], lineWidth=1,
                    physicsClientId=self.pb_client,
                )

    def _update_pybullet(self):
        try:
            s, n, mu, vx, vy, r = self.state
            X, Y, psi_track = self.track.frenet_to_cartesian(s, n)
            pb.resetBasePositionAndOrientation(
                self.car_id,
                [X, Y, 0.05],
                pb.getQuaternionFromEuler([0, 0, psi_track + mu]),
                physicsClientId=self.pb_client,
            )
            pb.stepSimulation(physicsClientId=self.pb_client)
        except Exception:
            print("[PyBullet] Okno zamknięte – kontynuuję bez wizualizacji.")
            self.use_pybullet = False

    def close_pybullet(self):
        if self.pb_client is not None:
            try:
                pb.disconnect(self.pb_client)
            except Exception:
                pass
            self.pb_client = None

    # ══════════════════════════════════════════════════════════════════════
    # REALTIME PLOT (z wężem predykcyjnym)
    # ══════════════════════════════════════════════════════════════════════

    def _setup_realtime_plot(self):
        if self._rt_fig is not None:
            self._rt_enabled = True
            return

        plt.ion()
        self._rt_fig, self._rt_ax = plt.subplots(figsize=(9, 7))
        ax = self._rt_ax
        ax.set_aspect('equal')

        lx, ly, rx, ry = self.track.get_track_boundaries()
        ax.fill(
            np.concatenate([lx, rx[::-1]]),
            np.concatenate([ly, ry[::-1]]),
            alpha=0.10, color='gray',
        )
        ax.plot(self.track.x, self.track.y, 'y--', lw=1, alpha=0.7, label='centerline')
        ax.plot(lx, ly, 'k-', lw=1.5, alpha=0.6)
        ax.plot(rx, ry, 'k-', lw=1.5, alpha=0.6)

        self._rt_trail,     = ax.plot([], [], 'b-',  lw=1.8, alpha=0.8,  label='trajektoria')
        self._rt_car_dot,   = ax.plot([], [], 'ro',  ms=10,  zorder=10,  label='pojazd')

        # ── Wąż predykcyjny ──────────────────────────────────────────────
        # Linia łącząca punkty horyzontu
        self._rt_snake,     = ax.plot([], [], color='#00cc44', lw=2.0,
                                      alpha=0.85, zorder=8, label='predykcja MPC')
        # Kółka na każdym kroku horyzontu
        self._rt_snake_dots, = ax.plot([], [], 'o', color='#00cc44',
                                       ms=4, alpha=0.7, zorder=9)

        self._rt_title = ax.set_title('Symulacja w czasie rzeczywistym')
        self._rt_text  = ax.text(
            0.02, 0.98, '', transform=ax.transAxes,
            va='top', ha='left',
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        pad = 1.0
        ax.set_xlim(float(np.min(lx)) - pad, float(np.max(lx)) + pad)
        ax.set_ylim(float(np.min(ly)) - pad, float(np.max(ly)) + pad)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        self._rt_enabled = True
        self._rt_fig.canvas.draw_idle()

    def _update_realtime_plot(self, force: bool = False):
        if not self._rt_enabled or self._rt_fig is None or len(self.state_history) == 0:
            return

        states = np.array(self.state_history)
        sample = states[-self._rt_trail_len:]
        xs, ys = [], []
        for row in sample:
            X, Y, _ = self.track.frenet_to_cartesian(row[0], row[1])
            xs.append(X)
            ys.append(Y)

        Xc, Yc, _ = self.track.frenet_to_cartesian(states[-1, 0], states[-1, 1])
        self._rt_trail.set_data(xs, ys)
        self._rt_car_dot.set_data([Xc], [Yc])

        # ── Wąż predykcyjny ──────────────────────────────────────────────
        ctrl = self._controller_ref
        if (ctrl is not None
                and hasattr(ctrl, 'prediction_xy')
                and ctrl.prediction_xy is not None
                and len(ctrl.prediction_xy) > 1):
            px = ctrl.prediction_xy[:, 0]
            py = ctrl.prediction_xy[:, 1]
            self._rt_snake.set_data(px, py)
            self._rt_snake_dots.set_data(px, py)
        else:
            # Kontroler bez predykcji – ukryj węża
            self._rt_snake.set_data([], [])
            self._rt_snake_dots.set_data([], [])

        s, n, mu, vx, vy, r = states[-1]
        self._rt_title.set_text('Symulacja w czasie rzeczywistym')
        self._rt_text.set_text(
            f"t  = {self.time:.2f} s\n"
            f"s  = {s:.2f} m\n"
            f"n  = {n:.3f} m\n"
            f"mu = {mu:.3f} rad\n"
            f"vx = {vx:.2f} m/s"
        )

        if force:
            self._rt_fig.canvas.draw_idle()
            self._rt_fig.canvas.flush_events()

    # ══════════════════════════════════════════════════════════════════════
    # MATPLOTLIB – wykresy końcowe i animacja
    # ══════════════════════════════════════════════════════════════════════

    def plot_trajectory(self):
        if len(self.state_history) < 2:
            print("Brak danych.")
            return

        states = np.array(self.state_history)
        xs, ys = [], []
        for row in states[::5]:
            X, Y, _ = self.track.frenet_to_cartesian(row[0], row[1])
            xs.append(X)
            ys.append(Y)

        lx, ly, rx, ry = self.track.get_track_boundaries()

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.fill(np.concatenate([lx, rx[::-1]]),
                np.concatenate([ly, ry[::-1]]),
                alpha=0.12, color='gray')
        ax.plot(lx, ly, 'k-', lw=1.5, alpha=0.6)
        ax.plot(rx, ry, 'k-', lw=1.5, alpha=0.6)
        ax.plot(self.track.x, self.track.y, 'y--', lw=1, alpha=0.7, label='centerline')

        vx_vals = states[::5, 3]
        sc = ax.scatter(xs, ys, c=vx_vals, cmap='plasma', s=8, zorder=5)
        plt.colorbar(sc, ax=ax, label='vx [m/s]')

        ax.plot(xs[0],  ys[0],  'go', ms=10, zorder=6, label='start')
        ax.plot(xs[-1], ys[-1], 'r^', ms=10, zorder=6, label='koniec')

        ax.set_aspect('equal')
        ax.set_title('Trajektoria pojazdu')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_results(self):
        if len(self.state_history) < 2:
            print("Brak danych.")
            return

        states = np.array(self.state_history)
        inputs = np.array(self.input_history)
        times  = np.array(self.time_history)

        fig, axes = plt.subplots(4, 2, figsize=(14, 12))
        fig.suptitle("Historia symulacji F1/10", fontsize=14, fontweight='bold')

        state_labels = ['s [m]', 'n [m]', 'mu [rad]', 'vx [m/s]', 'vy [m/s]', 'r [rad/s]']
        state_colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']
        input_labels = ['d (napęd)', 'delta [rad]']
        input_colors = ['#FF5722', '#607D8B']

        for i in range(6):
            ax = axes.flat[i]
            ax.plot(times, states[:, i], color=state_colors[i], lw=1.8)
            ax.set_ylabel(state_labels[i])
            ax.set_xlabel('czas [s]')
            ax.grid(True, alpha=0.3)
            if i == 1:
                hw = self.track.track_width / 2
                ax.axhline( hw, color='red', ls='--', alpha=0.6, label='granica')
                ax.axhline(-hw, color='red', ls='--', alpha=0.6)
                ax.legend(fontsize=8)

        t_in = times[1:]
        for i in range(2):
            ax = axes.flat[6 + i]
            ax.plot(t_in, inputs[:, i], color=input_colors[i], lw=1.8)
            ax.set_ylabel(input_labels[i])
            ax.set_xlabel('czas [s]')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def animate(self, interval_ms: int = 30) -> FuncAnimation:
        states = np.array(self.state_history)
        xs, ys = [], []
        for row in states:
            X, Y, _ = self.track.frenet_to_cartesian(row[0], row[1])
            xs.append(X)
            ys.append(Y)
        xs = np.array(xs)
        ys = np.array(ys)

        lx, ly, rx, ry = self.track.get_track_boundaries()

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_aspect('equal')
        ax.plot(self.track.x, self.track.y, 'y--', lw=1, alpha=0.6)
        ax.plot(lx, ly, 'k-', lw=1.5, alpha=0.5)
        ax.plot(rx, ry, 'k-', lw=1.5, alpha=0.5)

        trail,    = ax.plot([], [], 'b-', lw=1.5, alpha=0.7)
        car_dot,  = ax.plot([], [], 'ro', ms=10,  zorder=10)
        title      = ax.set_title('')

        ax.set_xlim(xs.min() - 1, xs.max() + 1)
        ax.set_ylim(ys.min() - 1, ys.max() + 1)

        trail_len = 60

        def update(frame):
            lo = max(0, frame - trail_len)
            trail.set_data(xs[lo:frame + 1], ys[lo:frame + 1])
            car_dot.set_data([xs[frame]], [ys[frame]])
            s, n, mu, vx = states[frame, :4]
            title.set_text(
                f"t={self.time_history[frame]:.2f}s | "
                f"s={s:.2f}m | n={n:.3f}m | vx={vx:.2f}m/s"
            )
            return trail, car_dot, title

        ani = FuncAnimation(fig, update, frames=len(xs),
                            interval=interval_ms, blit=True)
        plt.tight_layout()
        return ani