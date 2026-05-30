"""
simulator.py
============
Klasa F1tenthSimulator — łączy model dynamiczny, tor i wizualizację.

Nowości:
- wizualizacja matplotlib w czasie rzeczywistym podczas symulacji,
- parametr log_every określający co który krok wypisywać log,
- końcowe wykresy i animacja pokazują się po symulacji.
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


class F1tenthSimulator:
    def __init__(self,
                 track: TrackCenterline,
                 vehicle_params: Optional[VehicleParams] = None,
                 dt: float = 0.02,
                 use_pybullet: bool = True):
        self.track = track
        self.params = vehicle_params or VehicleParams()
        self.model = DynamicBicycleModel(self.params)
        self.dt = dt
        self.time = 0.0

        self.state_history: list = []
        self.input_history: list = []
        self.time_history: list = []
        self.state = np.zeros(6)

        self.use_pybullet = use_pybullet and PYBULLET_AVAILABLE
        self.pb_client = None
        self.car_id = None

        self._rt_fig = None
        self._rt_ax = None
        self._rt_title = None
        self._rt_car_dot = None
        self._rt_trail = None
        self._rt_text = None
        self._rt_enabled = False
        self._rt_trail_len = 100

        if self.use_pybullet:
            self._init_pybullet()

    def reset(self, s0: float = 0.0, n0: float = 0.0,
              mu0: float = 0.0, vx0: float = 1.0) -> np.ndarray:
        self.state = np.array([s0, n0, mu0, vx0, 0.0, 0.0], dtype=float)
        self.time = 0.0
        self.state_history = [self.state.copy()]
        self.input_history = []
        self.time_history = [0.0]

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
        u = np.asarray(u, dtype=float)
        kappa = self.track.get_kappa(self.state[0])

        self.state = self.model.step_rk4(self.state, u, kappa, self.dt)
        self.time += self.dt

        self.state_history.append(self.state.copy())
        self.input_history.append(u.copy())
        self.time_history.append(self.time)

        info = {
            "kappa": kappa,
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
            realtime_pause: float = 0.001) -> dict:
        oob = 0
        log_every = max(1, int(log_every))
        realtime_interval_steps = max(1, int(realtime_interval_steps))

        if realtime_plot:
            self._setup_realtime_plot()
            self._update_realtime_plot(force=True)

        for i in range(n_steps):
            u = controller(self.state)
            _, info = self.step(u)

            if info["out_of_bounds"]:
                oob += 1

            if verbose and i % log_every == 0:
                s, n, mu, vx, vy, r = self.state
                X, Y, psi_track = self.track.frenet_to_cartesian(s, n)
                psi = psi_track + mu
                X_front = X + self.params.lf * np.cos(psi)
                Y_front = Y + self.params.lf * np.sin(psi)
                print(
                    f"iter={i:04d} | "
                    f"t={self.time:.2f}s | s={s:.2f}m | n={n:.3f}m | "
                    f"vx={vx:.2f}m/s | delta={u[1] * 180 / np.pi:.1f}° | "
                    f"Xf={X_front:.3f}m | Yf={Y_front:.3f}m | "
                    f"psi={psi:.3f}rad"
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
            "times": np.array(self.time_history),
        }

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
            alpha=0.10,
            color='gray',
        )
        ax.plot(self.track.x, self.track.y, 'y--', lw=1, alpha=0.7, label='centerline')
        ax.plot(lx, ly, 'k-', lw=1.5, alpha=0.6)
        ax.plot(rx, ry, 'k-', lw=1.5, alpha=0.6)

        self._rt_trail, = ax.plot([], [], 'b-', lw=1.8, alpha=0.8, label='trajektoria')
        self._rt_car_dot, = ax.plot([], [], 'ro', ms=8, zorder=10, label='pojazd')
        self._rt_title = ax.set_title('Symulacja w czasie rzeczywistym')
        self._rt_text = ax.text(
            0.02, 0.98, '', transform=ax.transAxes,
            va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        pad = 1.0
        ax.set_xlim(float(np.min(lx)) - pad, float(np.max(lx)) + pad)
        ax.set_ylim(float(np.min(ly)) - pad, float(np.max(ly)) + pad)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
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

        s, n, mu, vx, vy, r = states[-1]
        self._rt_title.set_text('Symulacja w czasie rzeczywistym')
        self._rt_text.set_text(
            f"t = {self.time:.2f} s\n"
            f"s = {s:.2f} m\n"
            f"n = {n:.3f} m\n"
            f"mu = {mu:.3f} rad\n"
            f"vx = {vx:.2f} m/s"
        )

        if force:
            self._rt_fig.canvas.draw_idle()
        self._rt_fig.canvas.flush_events()

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

        ax.plot(xs[0], ys[0], 'go', ms=10, zorder=6, label='start')
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
        times = np.array(self.time_history)

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
                ax.axhline(hw, color='red', ls='--', alpha=0.6, label='granica')
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

        trail, = ax.plot([], [], 'b-', lw=1.5, alpha=0.7)
        car_dot, = ax.plot([], [], 'ro', ms=10, zorder=10)
        title = ax.set_title('')

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

        ani = FuncAnimation(fig, update, frames=len(xs), interval=interval_ms, blit=True)
        plt.tight_layout()
        return ani