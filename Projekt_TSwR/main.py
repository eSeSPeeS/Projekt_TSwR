"""
main.py
=======
Punkt wejścia symulatora F1/10.
"""

import matplotlib.pyplot as plt

from vehicle_params import VehicleParams
from track import TrackCenterline
from simulator import F1tenthSimulator
from controllers import (
    controller_const_speed,
    controller_pid,
    controller_random,
    MPCController,
)

KONTROLER = 'mpc'          # 'p' | 'pid' | 'random' | 'mpc'
WIZUALIZACJA = 'wykresy'   # 'wykresy' | 'pybullet' | 'oba'

N_STEPS = 500
DT = 0.02
VX0 = 0.5
TOR = 'figure8'               # 'oval' | 'figure8' | 'road' | 'technical' | 'technical_sharp'

REALTIME_MATPLOTLIB = True
REALTIME_INTERVAL_STEPS = 3
REALTIME_PAUSE = 0.001

LOG_EVERY = 10            # co który krok wypisywać log


def main():
    print('=' * 60)
    print(' F1/10 Symulator')
    print(f' Kontroler: {KONTROLER}   Wizualizacja: {WIZUALIZACJA}')
    print('=' * 60)

    print('\n[1/4] Generowanie toru...')
    if TOR == 'oval':
        track = TrackCenterline.make_oval(length=8.0, width=4.0, track_width=0.35)
    elif TOR == 'figure8':
        track = TrackCenterline.make_figure8(r=2.5, track_width=0.35)
    elif TOR == 'road':
        track = TrackCenterline.make_road_loop(length=10.0, width=6.0, track_width=0.7)
    elif TOR == 'chicane':
        track = TrackCenterline.make_chicane(length=10.0, height=3.0, track_width=1.0)
    elif TOR == 'technical':
        track = TrackCenterline.make_technical(base_r=6.0, track_width=1.5)
    elif TOR == 'technical_sharp':
        track = TrackCenterline.make_technical_sharp(base_r=6.5, track_width=1.5)
    else:
        raise ValueError(f"Nieznany tor: {TOR}")
    print(f' Długość: {track.total_length:.2f} m')

    print('[2/4] Inicjalizacja symulatora...')
    params = VehicleParams()
    use_pb = WIZUALIZACJA in ('pybullet', 'oba')
    sim = F1tenthSimulator(track=track, vehicle_params=params, dt=DT, use_pybullet=use_pb)
    sim.reset(vx0=VX0)

    print(f"[3/4] Symulacja – kontroler '{KONTROLER}'...")
    if KONTROLER == 'p':
        ctrl = controller_const_speed(speed=0.35, kp_steering=2.0)
    elif KONTROLER == 'pid':
        ctrl = controller_pid(kp=5.0, ki=0.1, kd=0.8, speed=0.35, dt=DT)
    elif KONTROLER == 'random':
        ctrl = controller_random
    elif KONTROLER == 'mpc':
        ctrl = MPCController(
        model=sim.model,
        track=track,
        N=90,          # Horyzont predykcji — MPC "patrzy" 18 kroków do przodu
                    # (18 × 0.02s = 0.36s). Więcej = lepsze planowanie zakrętów,
                    # ale wolniejsze obliczenia.

        dt=DT,         # Krok czasowy [s] — musi być taki sam jak w symulatorze.

        vx_ref=2.5,    # Prędkość docelowa [m/s]. Przy q_vx=0.0 jest ignorowana
                    # (patrz niżej) — możesz tu wpisać cokolwiek.

        # ── Wagi funkcji kosztu ───────────────────────────────────────────────
        # Im większa waga, tym mocniej MPC "karze" odchylenie od zera.

        q_n=1.0,       # Kara za odchylenie boczne n [m] od centerline.
                    # Główna waga "trzymania toru". Zwiększ jeśli pojazd
                    # za bardzo dryfuje od środka.

        q_mu=1.0,      # Kara za kąt pojazdu względem toru μ [rad].
                    # Tłumi oscylacje kątowe — pojazd jedzie "prosto"
                    # względem osi toru, nie bokiem.

        q_vx=2.0,      # Kara za błąd prędkości (vx - vx_ref)². Aktualnie
                    # wyłączona (=0). Pojazd nie stara się utrzymać vx_ref,
                    # prędkość zależy tylko od r_d.

        r_d=0.1,       # Kara za wielkość sygnału napędowego d ∈ [0,1].
                    # Mała wartość = MPC chętnie przyspiesza. Zwiększ
                    # jeśli pojazd przyspiesza zbyt agresywnie.

        r_delta=0.5,   # Kara za kąt skrętu δ [rad]. Tłumi ostre skręty.
                    # Zwiększ jeśli pojazd "szarpie" kierownicą.

        r_dd=0.5,      # Kara za ZMIANĘ napędu (d_k - d_{k-1})².
                    # Wygładza przyspieszanie/hamowanie między krokami.

        r_ddelta=0.2,  # Kara za ZMIANĘ skrętu (δ_k - δ_{k-1})².
                    # Wygładza ruchy kierownicy. Zwiększ przy drganiach
                    # układu kierowniczego.
    )
    else:
        raise ValueError(f'Nieznany kontroler: {KONTROLER}')

    sim.run(
        ctrl,
        n_steps=N_STEPS,
        verbose=True,
        log_every=LOG_EVERY,
        realtime_plot=REALTIME_MATPLOTLIB and WIZUALIZACJA in ('wykresy', 'oba'),
        realtime_interval_steps=REALTIME_INTERVAL_STEPS,
        realtime_pause=REALTIME_PAUSE,
    )

    print('\n[4/4] Wizualizacja końcowa...')
    if WIZUALIZACJA in ('wykresy', 'oba'):
        plt.ioff()
        sim.plot_trajectory()
        sim.plot_results()
        ani = sim.animate(interval_ms=20)
        plt.show(block=True)

    if WIZUALIZACJA in ('pybullet', 'oba') and sim.use_pybullet:
        print('\nPyBullet: naciśnij Enter żeby zamknąć...')
        input()
        sim.close_pybullet()

    print('\nGotowe!')


if __name__ == '__main__':
    main()
