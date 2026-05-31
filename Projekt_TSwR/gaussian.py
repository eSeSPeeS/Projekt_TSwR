"""
gaussian.py
===========
Gaussowski model resztkowy (GP Residual Model) dla symulatora F1/10.

Inspiracja: Lahr et al., "L4acados: Learning-based models for acados,
applied to Gaussian process-based predictive control", arXiv:2411.19258

Architektura:
─────────────
  • GP uczy się resztek modelu dynamicznego na prędkościach bocznych
    i kątowych: docelowe stany to [vx, vy, r] (3 niezależne GP).

  • Wejścia (features) GP: [vx, vy, r, d, delta] — te same co w papierze
    (sekcja IV-D), bez składowych pozycyjnych s/n/mu, bo te zależą
    od toru a nie od dynamiki pojazdu.

  • Bd: macierz 6×3 — tylko składowe [vx, vy, r] są korygowane przez GP.
    Analogicznie do Bd^T = [0_{3×3}, I_{3×3}] z papieru.

  • Trening odbywa się na plikach .npz z folderu f1enth_long_track_sens.
    Każdy krok czasowy daje jedną próbkę: (x_t, u_t) → resid_{t+1}.

Użycie:
───────
    from gaussian import GPResidualModel

    # Trening
    gp = GPResidualModel()
    gp.train_from_npz(list_of_npz_paths, track, dt=0.02)
    gp.save("gp_model.pt")

    # Predykcja (korekta MPC)
    x_corrected = gp.correct_state(x_nominal, x_state, u_input)

    # Integracja z MPCController
    mpc = MPCController(..., gp_model=gp)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood


# ══════════════════════════════════════════════════════════════════════════════
#  Stałe
# ══════════════════════════════════════════════════════════════════════════════

#: Wymiary wektora stanu [s, n, mu, vx, vy, r]
NX = 6
#: Wymiary wektora sterowania [d, delta]
NU = 2
#: Indeksy stanów korygowanych przez GP (vx=3, vy=4, r=5)
GP_OUTPUT_IDX = [3, 4, 5]
#: Rozmiar wyjścia GP
N_GP_OUTPUTS = len(GP_OUTPUT_IDX)  # 3
#: Rozmiar wejścia GP (features: vx, vy, r, d, delta)
N_GP_FEATURES = 5


# ══════════════════════════════════════════════════════════════════════════════
#  Ekstrakcja danych z pliku .npz
# ══════════════════════════════════════════════════════════════════════════════

def load_npz_residuals(
    npz_path: Path,
    vehicle_model,
    track,
    dt: float = 0.02,
    min_vx: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wczytuje jeden plik .npz i oblicza resztki modelu nominalnego.

    Dane wejściowe (features) GP:
        z = [vx, vy, r, d, delta]   (shape: T×5)

    Dane wyjściowe (targets) GP:
        y = x_{t+1}[vx,vy,r] - f(x_t, u_t)[vx,vy,r]   (shape: T×3)

    Args:
        npz_path:      ścieżka do pliku .npz
        vehicle_model: instancja DynamicBicycleModel
        track:         instancja TrackCenterline (do pobierania kappa)
        dt:            krok czasowy [s]
        min_vx:        minimalny vx (odfiltrowanie startu/stopu)

    Returns:
        (features, targets) – gotowe numpy arrays
    """
    data = np.load(str(npz_path), allow_pickle=True)

    x0  = data['x0'][:, :6]   # (T, 6): [s, n, mu, vx, vy, r]
    u0s = data['u0s']          # (T, 2): [d, delta]
    valid = data['sens_valid'] # (T,): bool

    features_list = []
    targets_list  = []

    for t in range(len(x0) - 1):
        if not valid[t]:
            continue

        x_t = x0[t]
        u_t = u0s[t]
        x_next = x0[t + 1]

        # Odfiltruj punkty przy zbyt małej prędkości
        if x_t[3] < min_vx or x_next[3] < min_vx:
            continue

        # Krzywizna w bieżącej pozycji
        kappa = track.get_kappa(x_t[0])

        # Jeden krok modelu nominalnego
        x_nom_next = vehicle_model.step_rk4(x_t, u_t, kappa, dt)

        # Resztka: tylko składowe [vx, vy, r]
        residual = x_next[GP_OUTPUT_IDX] - x_nom_next[GP_OUTPUT_IDX]

        # Filtruj outlierów (błędy > 5× odch. stand. heurystycznie)
        if np.any(np.abs(residual) > 2.0):
            continue

        # Feature: [vx, vy, r, d, delta]
        z = np.array([x_t[3], x_t[4], x_t[5], u_t[0], u_t[1]], dtype=np.float32)

        features_list.append(z)
        targets_list.append(residual.astype(np.float32))

    if len(features_list) == 0:
        return np.zeros((0, N_GP_FEATURES), np.float32), np.zeros((0, N_GP_OUTPUTS), np.float32)

    return np.vstack(features_list), np.vstack(targets_list)


# ══════════════════════════════════════════════════════════════════════════════
#  Model GP (jeden wyjściowy)
# ══════════════════════════════════════════════════════════════════════════════

class _SingleOutputGP(ExactGP):
    """
    Dokładny GP z kernelem Matérn-5/2 i Automatic Relevance Determination.

    Matérn-5/2 jest preferowany w robotyce bo modeluje funkcje ciągłe
    ze skokowymi gradientami (np. efekty poślizgu opon).
    """

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: GaussianLikelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=N_GP_FEATURES)
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean  = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


# ══════════════════════════════════════════════════════════════════════════════
#  Główna klasa GPResidualModel
# ══════════════════════════════════════════════════════════════════════════════

class GPResidualModel:
    """
    Model resztkowy oparty na procesach Gaussa dla F1/10.

    Trenuje 3 niezależne GP (jeden na każdą składową [vx, vy, r]).
    Podczas predykcji poprawia następny stan wyjściowy modelu nominalnego.

    Atrybuty:
        enabled:   bool – czy GP jest aktywny (False = czysty MPC)
        trained:   bool – czy GP ma wytrenowane dane
        max_data:  int  – maksymalna liczba punktów treningowych
        device:    str  – 'cpu' lub 'cuda'

    Przykład:
        gp = GPResidualModel(max_data=500, device='cpu')
        gp.train_from_npz(paths, track, vehicle_model, dt=0.02)
        x_corrected = gp.correct_state(x_nominal, x_current, u)
    """

    def __init__(self,
                 enabled:   bool = True,
                 max_data:  int  = 2500,
                 device:    str  = 'cpu',
                 n_train_iter: int = 1000):
        """
        Args:
            enabled:      True = GP aktywny, False = brak korekcji
            max_data:     maks. liczba punktów treningowych (subsample)
            device:       'cpu' lub 'cuda' (dla GPU)
            n_train_iter: liczba iteracji optymalizacji hiperparametrów
        """
        self.enabled       = enabled
        self.max_data      = max_data
        self.device        = torch.device(device)
        self.n_train_iter  = n_train_iter
        self.trained       = False

        self._models: List[_SingleOutputGP]     = []
        self._likelihoods: List[GaussianLikelihood] = []

        # Statystyki normalizacji
        self._feat_mean: Optional[np.ndarray] = None
        self._feat_std:  Optional[np.ndarray] = None

        print(f"[GP] Inicjalizacja: enabled={enabled}, max_data={max_data}, device={device}")

    # ── Normalizacja cech ─────────────────────────────────────────────────

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Standaryzuje features do N(0,1)."""
        if self._feat_mean is None:
            return X
        return (X - self._feat_mean) / (self._feat_std + 1e-8)

    def _fit_normalizer(self, X: np.ndarray):
        self._feat_mean = X.mean(axis=0)
        self._feat_std  = X.std(axis=0)
        self._feat_std  = np.where(self._feat_std < 1e-6, 1.0, self._feat_std)

    # ── Trening ───────────────────────────────────────────────────────────

    def train_from_npz(
        self,
        npz_paths: List[Path],
        track,
        vehicle_model,
        dt: float = 0.02,
        verbose: bool = True,
    ) -> None:
        """
        Trenuje GP na zebranych danych z plików .npz.

        Args:
            npz_paths:     lista ścieżek do plików .npz
            track:         TrackCenterline (do kappa)
            vehicle_model: DynamicBicycleModel (model nominalny)
            dt:            krok czasowy symulacji [s]
            verbose:       wypisuj postęp
        """
        if not self.enabled:
            print("[GP] GP wyłączony – pomijam trening.")
            return

        print(f"\n[GP] Ładowanie danych z {len(npz_paths)} pliku/plików...")
        all_features = []
        all_targets  = []

        for path in npz_paths:
            path = Path(path)
            if not path.exists():
                print(f"[GP]   UWAGA: plik nie istnieje: {path}")
                continue
            feat, targ = load_npz_residuals(path, vehicle_model, track, dt)
            if feat.shape[0] == 0:
                print(f"[GP]   {path.name}: brak danych po filtrowaniu")
                continue
            all_features.append(feat)
            all_targets.append(targ)
            if verbose:
                print(f"[GP]   {path.name}: {feat.shape[0]} próbek  "
                      f"(resid vx={targ[:,0].std():.4f}, "
                      f"vy={targ[:,1].std():.4f}, "
                      f"r={targ[:,2].std():.4f})")

        if len(all_features) == 0:
            print("[GP] BŁĄD: brak danych do treningu!")
            return

        X = np.vstack(all_features)  # (N_total, 5)
        Y = np.vstack(all_targets)   # (N_total, 3)

        # Subsample jeśli za dużo danych
        if X.shape[0] > self.max_data:
            idx = np.random.choice(X.shape[0], self.max_data, replace=False)
            X, Y = X[idx], Y[idx]
            if verbose:
                print(f"[GP] Subsample → {self.max_data} punktów treningowych")
        else:
            if verbose:
                print(f"[GP] Łączna liczba próbek: {X.shape[0]}")

        # Normalizacja features
        self._fit_normalizer(X)
        X_norm = self._normalize(X)

        # Buduj i trenuj 3 niezależne GP
        self._models      = []
        self._likelihoods = []

        output_names = ['vx', 'vy', 'r']

        for i, name in enumerate(output_names):
            y_i = Y[:, i]

            train_x = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
            train_y = torch.tensor(y_i,    dtype=torch.float32).to(self.device)

            likelihood = GaussianLikelihood().to(self.device)
            gp_model   = _SingleOutputGP(train_x, train_y, likelihood).to(self.device)

            gp_model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(
                list(gp_model.parameters()) + list(likelihood.parameters()),
                lr=0.05
            )
            mll = ExactMarginalLogLikelihood(likelihood, gp_model)

            if verbose:
                print(f"[GP]   Trenuję GP_{name} ({self.n_train_iter} iter)...", end="", flush=True)

            prev_loss = float('inf')
            for it in range(self.n_train_iter):
                optimizer.zero_grad()
                output = gp_model(train_x)
                loss   = -mll(output, train_y)
                loss.backward()
                optimizer.step()

                # Wczesne zatrzymanie jeśli strata zbieżna
                if it > 20 and abs(prev_loss - loss.item()) < 1e-5:
                    if verbose:
                        print(f" (zbieżność po {it+1} iter)", end="")
                    break
                prev_loss = loss.item()

            if verbose:
                lengthscales = gp_model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
                noise = likelihood.noise.item()
                print(f" loss={loss.item():.4f}, noise={noise:.4f}")
                feat_labels = ['vx', 'vy', 'r', 'd', 'delta']
                ls_str = ", ".join(f"{feat_labels[j]}:{lengthscales[j]:.3f}" for j in range(len(lengthscales)))
                print(f"[GP]     lengthscales=({ls_str})")

            gp_model.eval()
            likelihood.eval()

            self._models.append(gp_model)
            self._likelihoods.append(likelihood)

        self.trained = True
        print("[GP] Trening zakończony pomyślnie.\n")

    # ── Predykcja ─────────────────────────────────────────────────────────

    def predict_residual(
        self,
        x_state: np.ndarray,
        u_input: np.ndarray,
        return_variance: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Przewiduje resztkę [vx, vy, r] dla podanego stanu i sterowania.

        Args:
            x_state:         stan pojazdu [s, n, mu, vx, vy, r]
            u_input:         sterowanie [d, delta]
            return_variance: czy zwracać wariancję predykcji

        Returns:
            (mean_residual, variance) – shape (3,) każdy
            Jeśli return_variance=False, variance=None
        """
        if not self.enabled or not self.trained:
            zeros = np.zeros(N_GP_OUTPUTS)
            return (zeros, zeros if return_variance else None)

        # Feature: [vx, vy, r, d, delta]
        z = np.array([x_state[3], x_state[4], x_state[5],
                      u_input[0], u_input[1]], dtype=np.float32)
        z_norm = self._normalize(z[np.newaxis, :])  # (1, 5)

        test_x = torch.tensor(z_norm, dtype=torch.float32).to(self.device)

        means = []
        variances = []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for gp_model, likelihood in zip(self._models, self._likelihoods):
                pred = likelihood(gp_model(test_x))
                means.append(pred.mean.cpu().numpy()[0])
                variances.append(pred.variance.cpu().numpy()[0])

        mean_arr = np.array(means)      # (3,)
        var_arr  = np.array(variances)  # (3,)

        return (mean_arr, var_arr if return_variance else None)

    def predict_residual_batch(
        self,
        X_states: np.ndarray,
        U_inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Batchowa predykcja resztek dla całego horyzontu MPC.

        Args:
            X_states: (N, 6) – stany horyzontu
            U_inputs: (N, 2) – sterowania horyzontu

        Returns:
            residuals: (N, 3) – resztki [vx, vy, r]
        """
        if not self.enabled or not self.trained:
            return np.zeros((X_states.shape[0], N_GP_OUTPUTS))

        N = X_states.shape[0]
        # Features: [vx, vy, r, d, delta]
        Z = np.column_stack([
            X_states[:, 3], X_states[:, 4], X_states[:, 5],
            U_inputs[:, 0], U_inputs[:, 1]
        ]).astype(np.float32)

        Z_norm = self._normalize(Z)  # (N, 5)
        test_x = torch.tensor(Z_norm, dtype=torch.float32).to(self.device)

        residuals = np.zeros((N, N_GP_OUTPUTS))

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i, (gp_model, likelihood) in enumerate(
                    zip(self._models, self._likelihoods)):
                pred = likelihood(gp_model(test_x))
                residuals[:, i] = pred.mean.cpu().numpy()

        return residuals

    # ── Korekta stanu ─────────────────────────────────────────────────────

    def correct_state(
        self,
        x_nominal_next: np.ndarray,
        x_current:      np.ndarray,
        u_input:        np.ndarray,
    ) -> np.ndarray:
        """
        Koryguje przewidziany następny stan o resztkę GP.

        Implementuje: x_{t+1} = f(x_t, u_t) + B_d · g(x_t, u_t)
        gdzie B_d wybiera składowe [vx, vy, r].

        Args:
            x_nominal_next: następny stan z modelu nominalnego (6,)
            x_current:      bieżący stan pojazdu (6,)
            u_input:        sterowanie (2,)

        Returns:
            x_corrected: (6,) – stan z korekcją GP
        """
        if not self.enabled or not self.trained:
            return x_nominal_next

        residual, _ = self.predict_residual(x_current, u_input,
                                             return_variance=False)

        x_corrected = x_nominal_next.copy()
        x_corrected[GP_OUTPUT_IDX] += residual
        return x_corrected

    # ── Zapis / wczytanie ─────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Zapisuje wytrenowany model GP do pliku .pt.

        Args:
            path: ścieżka docelowa (np. 'gp_model.pt')
        """
        if not self.trained:
            print("[GP] UWAGA: model nie jest wytrenowany – pominięto zapis.")
            return

        state_dicts = [m.state_dict() for m in self._models]
        lik_dicts   = [l.state_dict() for l in self._likelihoods]

        # Potrzebujemy danych treningowych do odtworzenia ExactGP
        train_xs = [m.train_inputs[0].cpu()  for m in self._models]
        train_ys = [m.train_targets.cpu()    for m in self._models]

        torch.save({
            'state_dicts':    state_dicts,
            'lik_dicts':      lik_dicts,
            'train_xs':       train_xs,
            'train_ys':       train_ys,
            'feat_mean':      self._feat_mean,
            'feat_std':       self._feat_std,
            'enabled':        self.enabled,
            'max_data':       self.max_data,
            'n_train_iter':   self.n_train_iter,
        }, path)
        print(f"[GP] Model zapisany: {path}")

    def load(self, path: str) -> None:
        """
        Wczytuje wcześniej zapisany model GP.

        Args:
            path: ścieżka do pliku .pt
        """
        if not os.path.exists(path):
            print(f"[GP] UWAGA: plik modelu nie istnieje: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self._feat_mean  = checkpoint['feat_mean']
        self._feat_std   = checkpoint['feat_std']
        self.enabled     = checkpoint.get('enabled', True)
        self.max_data    = checkpoint.get('max_data', self.max_data)

        self._models      = []
        self._likelihoods = []

        for train_x, train_y, sd, ld in zip(
                checkpoint['train_xs'],
                checkpoint['train_ys'],
                checkpoint['state_dicts'],
                checkpoint['lik_dicts']):

            train_x = train_x.to(self.device)
            train_y = train_y.to(self.device)

            likelihood = GaussianLikelihood().to(self.device)
            gp_model   = _SingleOutputGP(train_x, train_y, likelihood).to(self.device)

            gp_model.load_state_dict(sd)
            likelihood.load_state_dict(ld)

            gp_model.eval()
            likelihood.eval()

            self._models.append(gp_model)
            self._likelihoods.append(likelihood)

        self.trained = True
        print(f"[GP] Model wczytany z: {path}")
        print(f"[GP]   Punktów treningowych: {self._models[0].train_inputs[0].shape[0]}")

    # ── Statystyki i diagnostyka ──────────────────────────────────────────

    def get_info(self) -> dict:
        """Zwraca słownik z informacjami o modelu GP."""
        info = {
            'enabled': self.enabled,
            'trained': self.trained,
            'max_data': self.max_data,
            'device': str(self.device),
            'n_outputs': len(self._models),
        }
        if self.trained:
            info['n_train_points'] = self._models[0].train_inputs[0].shape[0]
            info['output_states']  = ['vx', 'vy', 'r']
            info['feature_names']  = ['vx', 'vy', 'r', 'd', 'delta']
            ls_all = []
            for m in self._models:
                ls = m.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
                ls_all.append(ls.tolist())
            info['lengthscales'] = ls_all  # (3, 5)
        return info

    def print_info(self) -> None:
        """Wypisuje informacje diagnostyczne."""
        info = self.get_info()
        print("\n[GP] ══ Informacje o modelu GP ══")
        print(f"  Aktywny:   {info['enabled']}")
        print(f"  Wytrenowany: {info['trained']}")
        if info['trained']:
            print(f"  Punkty treningowe: {info['n_train_points']}")
            print(f"  Wyjście GP:  {info['output_states']}")
            print(f"  Cechy (in):  {info['feature_names']}")
            feat_l = info['feature_names']
            out_l  = info['output_states']
            for i, (out_name, ls) in enumerate(zip(out_l, info['lengthscales'])):
                ls_str = ", ".join(f"{feat_l[j]}:{ls[j]:.3f}" for j in range(len(ls)))
                print(f"  Lengthscales GP_{out_name}: ({ls_str})")
        print()

    # ── Skrótowe __call__ ─────────────────────────────────────────────────

    def __call__(self, x_state: np.ndarray,
                 u_input: np.ndarray) -> np.ndarray:
        """
        Skrót: zwraca przewidywaną resztkę [vx, vy, r] dla podanego (x, u).

        Returns:
            residual (3,) lub zeros jeśli GP wyłączony/nietrędowany
        """
        residual, _ = self.predict_residual(x_state, u_input)
        return residual


# ══════════════════════════════════════════════════════════════════════════════
#  Pomocnicza funkcja treningu (wygodne API)
# ══════════════════════════════════════════════════════════════════════════════

def train_gp_from_folder(
    folder: str,
    track,
    vehicle_model,
    dt:            float = 0.02,
    max_data:      int   = 500,
    n_train_iter:  int   = 100,
    save_path:     Optional[str] = None,
    device:        str   = 'cpu',
    verbose:       bool  = True,
) -> GPResidualModel:
    """
    Trenuje GPResidualModel na wszystkich plikach .npz z podanego folderu.

    Args:
        folder:       ścieżka do folderu z plikami .npz
        track:        TrackCenterline (do obliczania kappa)
        vehicle_model: DynamicBicycleModel (model nominalny)
        dt:           krok czasowy [s]
        max_data:     maks. punktów treningowych (subsample)
        n_train_iter: liczba iteracji optymalizacji
        save_path:    jeśli podana, zapisuje model do pliku
        device:       'cpu' lub 'cuda'
        verbose:      wypisuj postęp

    Returns:
        Wytrenowany GPResidualModel
    """
    folder_path = Path(folder)
    npz_files   = sorted(folder_path.glob("*.npz"))

    if len(npz_files) == 0:
        raise FileNotFoundError(f"Brak plików .npz w folderze: {folder}")

    print(f"[GP] Znaleziono {len(npz_files)} plików .npz w '{folder}'")

    gp = GPResidualModel(
        enabled=True,
        max_data=max_data,
        n_train_iter=n_train_iter,
        device=device,
    )
    gp.train_from_npz(npz_files, track, vehicle_model, dt=dt, verbose=verbose)

    if save_path is not None:
        gp.save(save_path)

    return gp


# ══════════════════════════════════════════════════════════════════════════════
#  Walidacja (opcjonalna)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_gp(
    gp: GPResidualModel,
    npz_paths: List[Path],
    track,
    vehicle_model,
    dt: float = 0.02,
    verbose: bool = True,
) -> dict:
    """
    Ocenia jakość modelu GP na zbiorze walidacyjnym.

    Oblicza RMSE resztki dla każdego wyjścia GP.

    Returns:
        dict z RMSE dla ['vx', 'vy', 'r']
    """
    all_feat = []
    all_targ = []

    for path in npz_paths:
        feat, targ = load_npz_residuals(path, vehicle_model, track, dt)
        if feat.shape[0] > 0:
            all_feat.append(feat)
            all_targ.append(targ)

    if len(all_feat) == 0:
        return {}

    X = np.vstack(all_feat)
    Y = np.vstack(all_targ)

    # Predykcja batchowa
    U_dummy = X[:, 3:5]  # d, delta are last two features
    X_state_dummy = np.column_stack([
        np.zeros((len(X), 3)),   # s, n, mu (nie używane przez GP)
        X[:, 0], X[:, 1], X[:, 2]  # vx, vy, r
    ])

    pred = gp.predict_residual_batch(X_state_dummy, U_dummy)

    rmse = {}
    output_names = ['vx', 'vy', 'r']
    for i, name in enumerate(output_names):
        err  = Y[:, i] - pred[:, i]
        rmse[name] = float(np.sqrt(np.mean(err**2)))

    if verbose:
        print("[GP] Walidacja RMSE resztek:")
        for name, val in rmse.items():
            print(f"  residual_{name}: RMSE = {val:.6f} m/s")

    return rmse


# ══════════════════════════════════════════════════════════════════════════════
#  Demonstracja (uruchom: python gaussian.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from vehicle_params import VehicleParams
    from vehicle_model import DynamicBicycleModel
    from track import TrackCenterline

    print("=" * 60)
    print(" Demonstracja GP Residual Model dla F1/10")
    print("=" * 60)

    # Parametry
    DATA_FOLDER = "f1enth_long_track_sens"   # folder z plikami .npz
    SAVE_PATH   = "gp_model.pt"
    MAX_DATA    = 500
    N_ITER      = 80
    DT          = 0.02

    # Przygotuj tor i model (do obliczania resztek)
    track = TrackCenterline.make_technical(base_r=6.0, track_width=1.5)
    params = VehicleParams()
    vehicle_model = DynamicBicycleModel(params)

    folder_path = Path(DATA_FOLDER)
    if not folder_path.exists():
        # Fallback: użyj lokalnych plików .npz jeśli folder nie istnieje
        local_npz = sorted(Path(".").glob("*.npz"))
        if len(local_npz) == 0:
            print(f"[DEMO] Folder '{DATA_FOLDER}' nie istnieje i brak lokalnych .npz")
            print("[DEMO] Uruchom z właściwą ścieżką do danych.")
            sys.exit(0)
        npz_files = local_npz
        print(f"[DEMO] Używam lokalnych plików: {[f.name for f in npz_files]}")
    else:
        npz_files = sorted(folder_path.glob("*.npz"))
        print(f"[DEMO] Pliki w '{DATA_FOLDER}': {[f.name for f in npz_files]}")

    # Trening
    gp = GPResidualModel(enabled=True, max_data=MAX_DATA, n_train_iter=N_ITER)
    gp.train_from_npz(npz_files, track, vehicle_model, dt=DT, verbose=True)

    gp.print_info()

    # Test predykcji
    x_test = np.array([10.0, 0.05, 0.1, 3.0, 0.2, 0.5])  # stan testowy
    u_test = np.array([0.3, 0.15])                          # sterowanie testowe

    resid, var = gp.predict_residual(x_test, u_test, return_variance=True)
    print(f"[DEMO] Test predykcji:")
    print(f"  x = {x_test}, u = {u_test}")
    print(f"  residual [vx, vy, r] = {resid}")
    print(f"  variance [vx, vy, r] = {var}")

    # Zapis
    gp.save(SAVE_PATH)

    # Test wczytania
    gp2 = GPResidualModel()
    gp2.load(SAVE_PATH)
    resid2, _ = gp2.predict_residual(x_test, u_test)
    print(f"\n[DEMO] Po wczytaniu: residual = {resid2}")
    print(f"[DEMO] Różnica: {np.abs(resid - resid2).max():.2e}")
    print("\n[DEMO] Zakończono pomyślnie.")
