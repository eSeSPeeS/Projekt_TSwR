"""
main.py
=======
Punkt wejścia symulatora F1/10.

Zmień tylko dwie zmienne na górze:
    KONTROLER    = 'p' | 'pid' | 'random' | 'mpc'
    WIZUALIZACJA = 'wykresy' | 'pybullet' | 'oba'
"""

import matplotlib.pyplot as plt

from vehicle_params import VehicleParams
from vehicle_model  import DynamicBicycleModel
from track          import TrackCenterline
from simulator      import F1tenthSimulator
from controllers    import (controller_zero, controller_const_speed,
                            controller_pid, controller_random,
                            MPCController)

# ══════════════════════════════════════════════════════════════════════════════
#  KONFIGURACJA  ← zmień tutaj
# ══════════════════════════════════════════════════════════════════════════════

KONTROLER    = 'mpc'       # 'p' | 'pid' | 'random' | 'mpc'
WIZUALIZACJA = 'pybullet'   # 'wykresy' | 'pybullet' | 'oba'

# Parametry symulacji
N_STEPS  = 500             # liczba kroków (500 × 0.02s = 10s)
DT       = 0.02            # krok czasowy [s]
VX0      = 1.0             # prędkość początkowa [m/s]

# Parametry toru
TOR = 'oval'               # 'oval' | 'figure8'

# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  F1/10 Symulator")
    print(f"  Kontroler: {KONTROLER}   Wizualizacja: {WIZUALIZACJA}")
    print("=" * 60)

    # ── 1. Tor ────────────────────────────────────────────────────────────
    print("\n[1/4] Generowanie toru...")
    if TOR == 'oval':
        track = TrackCenterline.make_oval(
            length=8.0, width=4.0, track_width=0.35)
    elif TOR == 'figure8':
        track = TrackCenterline.make_figure8(r=2.5, track_width=0.35)
    else:
        raise ValueError(f"Nieznany tor: {TOR}")
    print(f"      Długość: {track.total_length:.2f} m")

    # ── 2. Symulator ──────────────────────────────────────────────────────
    print("[2/4] Inicjalizacja symulatora...")
    params     = VehicleParams()
    use_pb     = WIZUALIZACJA in ('pybullet', 'oba')
    sim        = F1tenthSimulator(track=track, vehicle_params=params,
                                  dt=DT, use_pybullet=use_pb)
    sim.reset(vx0=VX0)

    # ── 3. Kontroler ──────────────────────────────────────────────────────
    print(f"[3/4] Symulacja – kontroler '{KONTROLER}'...")

    if KONTROLER == 'p':
        ctrl = controller_const_speed(speed=0.35, kp_steering=2.0)

    elif KONTROLER == 'pid':
        ctrl = controller_pid(kp=5.0, ki=0.1, kd=0.8,
                              speed=0.35, dt=DT)

    elif KONTROLER == 'random':
        ctrl = controller_random

    elif KONTROLER == 'mpc':
        ctrl = MPCController(
            model    = sim.model,
            track    = track,
            N        = 15,        # horyzont: 15 × 0.02s = 0.3s
            dt       = DT,
            vx_ref   = 1.5,       # docelowa prędkość [m/s]
            q_n      = 15.0,      # kara za odchylenie boczne
            q_mu     = 5.0,       # kara za kąt mu
            q_vx     = 2.0,       # kara za błąd prędkości
            r_d      = 0.1,       # regularyzacja napędu
            r_delta  = 0.5,       # regularyzacja skrętu
            r_dd     = 0.5,       # kara za zmianę napędu
            r_ddelta = 1.0,       # kara za zmianę skrętu
        )
    else:
        raise ValueError(f"Nieznany kontroler: {KONTROLER}")

    result = sim.run(ctrl, n_steps=N_STEPS, verbose=True)

    # ── 4. Wizualizacja ───────────────────────────────────────────────────
    print("\n[4/4] Wizualizacja...")

    if WIZUALIZACJA in ('wykresy', 'oba'):
        sim.plot_trajectory()
        sim.plot_results()
        ani = sim.animate(interval_ms=20)
        plt.show()

    if WIZUALIZACJA in ('pybullet', 'oba') and sim.use_pybullet:
        print("\nPyBullet: naciśnij Enter żeby zamknąć...")
        input()
        sim.close_pybullet()

    print("\nGotowe!")


if __name__ == "__main__":
    main()
