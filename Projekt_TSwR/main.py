"""
main.py
=======
Punkt wejścia symulatora F1/10.

Nowość: opcjonalna integracja z Gaussowskim modelem resztkowym (GP).
  • GP_ENABLED = True   → trening GP na danych .npz + korekcja dynamiki MPC
  • GP_ENABLED = False  → klasyczny MPC bez GP
"""

import matplotlib.pyplot as plt
from pathlib import Path

from vehicle_params import VehicleParams
from track import TrackCenterline
from simulator import F1tenthSimulator
from controllers import (
    controller_const_speed,
    controller_pid,
    controller_random,
    MPCController,
)

# ══════════════════════════════════════════════════════════════════════════════
#  Konfiguracja
# ══════════════════════════════════════════════════════════════════════════════

KONTROLER   = 'mpc'          # 'p' | 'pid' | 'random' | 'mpc'
WIZUALIZACJA = 'wykresy'     # 'wykresy' | 'pybullet' | 'oba'

N_STEPS = 5000
DT      = 0.02
VX0     = 2.5
TOR     = 'road'        # 'oval' | 'figure8' | 'road' | 'technical' | 'technical_sharp'

REALTIME_MATPLOTLIB      = True
REALTIME_INTERVAL_STEPS  = 3
REALTIME_PAUSE           = 0.001

LOG_EVERY = 10               # co który krok wypisywać log

# ── Konfiguracja GP ───────────────────────────────────────────────────────────
# GP_ENABLED = True  → Gaussowski model resztkowy aktywny
# GP_ENABLED = False → klasyczny MPC (bez GP, szybciej)
GP_ENABLED       = True

# Folder z plikami .npz do treningu GP.
# Ustaw na właściwy katalog na swoim komputerze.
# Możesz też podać listę konkretnych plików w GP_NPZ_FILES poniżej.
GP_DATA_FOLDER   = "f1enth_long_track_sens"

# Alternatywnie: lista konkretnych plików .npz (nadpisuje GP_DATA_FOLDER).
# Pozostaw None żeby używać GP_DATA_FOLDER.
GP_NPZ_FILES     = None   # np. ["plik1.npz", "plik2.npz"]

# Ścieżka do zapisanego/wczytanego modelu GP (None = nie zapisuj / ładuj).
# Jeśli plik istnieje, GP zostanie wczytany zamiast trenować od nowa.
GP_SAVE_PATH     = "gp_model.pt"

GP_MAX_DATA      = 5500    # maks. punktów treningowych (subsample)
GP_N_TRAIN_ITER  = 380     # iteracje optymalizacji hiperparametrów
GP_DEVICE        = 'cpu'  # 'cpu' lub 'cuda'


def _collect_npz_paths():
    """Zbiera ścieżki do plików .npz wg konfiguracji."""
    if GP_NPZ_FILES is not None:
        paths = [Path(p) for p in GP_NPZ_FILES]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"[GP] UWAGA: brak plików: {missing}")
        return [p for p in paths if p.exists()]

    folder = Path(GP_DATA_FOLDER)
    if not folder.exists():
        print(f"[GP] UWAGA: folder '{GP_DATA_FOLDER}' nie istnieje.")
        return []
    paths = sorted(folder.glob("*.npz"))
    if not paths:
        print(f"[GP] UWAGA: brak plików .npz w '{GP_DATA_FOLDER}'.")
    return paths


def _build_gp(track, vehicle_model):
    """
    Buduje i trenuje (lub wczytuje) model GP.
    Zwraca GPResidualModel lub None jeśli GP jest wyłączony/niedostępny.
    """
    if not GP_ENABLED:
        return None

    from gaussian import GPResidualModel

    gp = GPResidualModel(
        enabled=True,
        max_data=GP_MAX_DATA,
        n_train_iter=GP_N_TRAIN_ITER,
        device=GP_DEVICE,
    )

    # Spróbuj wczytać zapisany model
    save_path = Path(GP_SAVE_PATH) if GP_SAVE_PATH else None
    if save_path and save_path.exists():
        print(f"[GP] Wczytywanie modelu z '{GP_SAVE_PATH}'...")
        gp.load(str(save_path))
        gp.print_info()
        return gp

    # Trening na nowych danych
    npz_paths = _collect_npz_paths()
    if not npz_paths:
        print("[GP] Brak danych do treningu – GP wyłączony.")
        gp.enabled = False
        return gp

    print(f"[GP] Trening na {len(npz_paths)} pliku/plikach...")
    gp.train_from_npz(npz_paths, vehicle_model, dt=DT, verbose=True)
    gp.print_info()

    # Zapisz wytrenowany model
    if save_path:
        gp.save(str(save_path))

    return gp


# ══════════════════════════════════════════════════════════════════════════════
#  Główna funkcja
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print(' F1/10 Symulator')
    print(f' Kontroler: {KONTROLER}   Wizualizacja: {WIZUALIZACJA}')
    print(f' GP Residual Model: {"AKTYWNY" if GP_ENABLED else "WYŁĄCZONY"}')
    print('=' * 60)

    print('\n[1/4] Generowanie toru...')
    if TOR == 'oval':
        track = TrackCenterline.make_oval(length=8.0, width=4.0, track_width=0.35)
    elif TOR == 'figure8':
        track = TrackCenterline.make_figure8(r=2.5, track_width=0.5)
    elif TOR == 'road':
        track = TrackCenterline.make_road_loop(length=10.0, width=6.0, track_width=0.7)
    elif TOR == 'chicane':
        track = TrackCenterline.make_chicane(length=10.0, height=3.0, track_width=1.0)
    elif TOR == 'technical':
        track = TrackCenterline.make_technical(base_r=6.0, track_width=1.5)
    elif TOR == 'technical_sharp':
        track = TrackCenterline.make_technical_sharp(base_r=6.5, track_width=1.5)
    elif TOR == 'pocket':
        track = TrackCenterline.make_pocket(track_width=0.6)
    else:
        raise ValueError(f"Nieznany tor: {TOR}")
    print(f' Długość: {track.total_length:.2f} m')

    print('[2/4] Inicjalizacja symulatora...')
    params = VehicleParams()
    use_pb = WIZUALIZACJA in ('pybullet', 'oba')
    sim = F1tenthSimulator(track=track, vehicle_params=params, dt=DT, use_pybullet=use_pb)
    sim.reset(vx0=VX0)

    # ── Opcjonalny GP ─────────────────────────────────────────────────────────
    gp_model = None
    if KONTROLER == 'mpc' and GP_ENABLED:
        print('\n[GP] Przygotowanie modelu GP...')
        gp_model = _build_gp(track, sim.model)

    print(f"\n[3/4] Symulacja – kontroler '{KONTROLER}'...")
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
            N=30,          # Horyzont predykcji — MPC "patrzy" 18 kroków do przodu
                    # (18 × 0.02s = 0.36s). Więcej = lepsze planowanie zakrętów,
                    # ale wolniejsze obliczenia.

            dt=DT,         # Krok czasowy [s] — musi być taki sam jak w symulatorze.

            vx_ref=3.5,    # Prędkość docelowa [m/s]. Przy q_vx=0.0 jest ignorowana
                        # (patrz niżej) — możesz tu wpisać cokolwiek.

            # ── Wagi funkcji kosztu ───────────────────────────────────────────────
            # Im większa waga, tym mocniej MPC "karze" odchylenie od zera.

            q_n=1.0,       # Kara za odchylenie boczne n [m] od centerline.
                        # Główna waga "trzymania toru". Zwiększ jeśli pojazd
                        # za bardzo dryfuje od środka.

            q_mu=1.0,      # Kara za kąt pojazdu względem toru μ [rad].
                        # Tłumi oscylacje kątowe — pojazd jedzie "prosto"
                        # względem osi toru, nie bokiem.

            q_vx=0.0,      # Kara za błąd prędkości (vx - vx_ref)². Aktualnie
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
                        # układu kierowniczego.`

            # ── GP Residual Model ─────────────────────────────────────────
            # Podaj wytrenowany GPResidualModel lub None żeby wyłączyć.
            gp_model=gp_model,
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
