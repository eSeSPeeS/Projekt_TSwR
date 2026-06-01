"""
gaussian.py
===========
Gaussowski model resztkowy (GP Residual Model) dla symulatora F1/10.

Inspiracja: Lahr et al., "L4acados: Learning-based models for acados,
applied to Gaussian process-based predictive control", arXiv:2411.19258

Architektura po analizie danych:
─────────────────────────────────
Format pliku .npz (ustalony przez analizę dataset_reader_sens.py):
  x0[:, :6]  = stan pojazdu [s, n, mu, vx, vy, r]
  x0[:, 6]   = vx_target  – cel prędkości z MPC (koreluje z napędem)
  x0[:, 7]   = delta      – kąt skrętu zastosowany do pojazdu
  u0s        = [vx_opt, delta_opt] – pierwsze sterowanie z solvera MPC

Wejścia GP (features): [vx, vy, r, delta, vx_target]
  vx_target jest kluczową cechą dla predykcji dvx (R²=0.69).
  delta jest główną cechą dla dvy i dr (R²~0.30).

Wyjście GP (targets): resztka modelu nominalnego dla [vx, vy, r]
  resid = x_next[vx,vy,r] - f_nom(x, d_proxy, delta)[vx,vy,r]
  gdzie d_proxy = clip(vx_target / vx_scale, 0, 1), vx_scale dobierane
  tak by residua były możliwie małe (GP uczy się tylko korekty, nie całości).

Integracja z MPC:
  x_{k+1}[vx,vy,r] = f_nom(x_k, u_k)[vx,vy,r] + GP(x_k, u_k)
  Korekta GP jest małą poprawką; gdy GP jest wyłączony działa czysty MPC.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood


# ══════════════════════════════════════════════════════════════════════════════
#  Stałe
# ══════════════════════════════════════════════════════════════════════════════

NX            = 6                # [s, n, mu, vx, vy, r]
NU            = 2                # [d, delta]
GP_OUTPUT_IDX = [3, 4, 5]       # indeksy [vx, vy, r] w wektorze stanu
N_GP_OUTPUTS  = 3               # vx, vy, r
N_GP_FEATURES = 5               # [vx, vy, r, delta, vx_target]

# Skala zamiany vx_target → d_proxy (d_proxy = clip(vx_target/VX_SCALE, 0, 1))
# Dobrana jako typowa prędkość maksymalna z danych (~5 m/s)
VX_SCALE      = 5.0


# ══════════════════════════════════════════════════════════════════════════════
#  Ekstrakcja danych z pliku .npz
# ══════════════════════════════════════════════════════════════════════════════

def load_npz_residuals(
    npz_path: Path,
    vehicle_model,
    dt: float = 0.02,
    min_vx: float = 0.5,
    max_residual: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wczytuje plik .npz i oblicza resztki modelu nominalnego dla GP.

    Format danych (ustalony przez dataset_reader_sens.py):
      x0[:, :6] = stan [s, n, mu, vx, vy, r]
      x0[:,  6] = vx_target (cel prędkości z MPC – proxy dla napędu)
      x0[:,  7] = delta_applied (kąt skrętu zastosowany do pojazdu)

    Features GP:  z = [vx, vy, r, delta, vx_target]   (N_GP_FEATURES=5)
    Targets  GP:  y = x_{t+1}[vx,vy,r] - f_nom(x_t, u_proxy_t)[vx,vy,r]

    Proxy sterowania dla modelu nominalnego:
      d_proxy = clip(vx_target / VX_SCALE, 0, 1)
      delta   = delta_applied

    Args:
        npz_path:     ścieżka do pliku .npz
        vehicle_model: DynamicBicycleModel (model nominalny)
        dt:           krok czasowy [s]
        min_vx:       minimalne vx do filtrowania (start/stop)
        max_residual: próg odfiltrowania outlierów

    Returns:
        (features, targets) – numpy arrays kształtu (N, 5) i (N, 3)
    """
    data  = np.load(str(npz_path), allow_pickle=True)
    x0    = data['x0']           # (T, 8)
    valid = data['sens_valid']   # (T,) bool

    state      = x0[:, :6]      # [s, n, mu, vx, vy, r]
    vx_target  = x0[:, 6]      # vx_target z MPC
    delta_app  = x0[:, 7]      # delta zastosowane

    features_list: List[np.ndarray] = []
    targets_list:  List[np.ndarray] = []

    for t in range(len(state) - 1):
        if not valid[t]:
            continue

        x_t    = state[t]
        x_next = state[t + 1]
        vxt    = vx_target[t]
        dlt    = delta_app[t]

        # Filtruj za małe prędkości
        if x_t[3] < min_vx or x_next[3] < min_vx:
            continue

        # Proxy sterowania dla modelu nominalnego
        d_proxy = float(np.clip(vxt / VX_SCALE, 0.0, 1.0))
        u_proxy = np.array([d_proxy, dlt])

        # Jeden krok modelu nominalnego (kappa=0: vx/vy/r nie zależą od kappa)
        x_nom_next = vehicle_model.step_rk4(x_t, u_proxy, 0.0, dt)

        # Resztka: rzeczywiste - nominalne dla [vx, vy, r]
        residual = x_next[GP_OUTPUT_IDX] - x_nom_next[GP_OUTPUT_IDX]

        # Odfiltruj outlierów
        if np.any(np.abs(residual) > max_residual):
            continue

        # Feature: [vx, vy, r, delta, vx_target]
        z = np.array([x_t[3], x_t[4], x_t[5], dlt, vxt], dtype=np.float32)

        features_list.append(z)
        targets_list.append(residual.astype(np.float32))

    if not features_list:
        empty_f = np.zeros((0, N_GP_FEATURES), np.float32)
        empty_t = np.zeros((0, N_GP_OUTPUTS),  np.float32)
        return empty_f, empty_t

    return np.vstack(features_list), np.vstack(targets_list)


def extract_features_from_state(
    x_state: np.ndarray,
    u_input: np.ndarray,
) -> np.ndarray:
    """
    Buduje wektor cech GP z bieżącego stanu i sterowania MPC.

    Mapowanie: (x, u) -> z = [vx, vy, r, delta, vx_target]
      vx_target = d_mpc * VX_SCALE  (odwrotność proxy z treningu)

    Args:
        x_state: stan pojazdu [s, n, mu, vx, vy, r]
        u_input: sterowanie MPC [d, delta]

    Returns:
        z: wektor cech (5,)
    """
    vx, vy, r = x_state[3], x_state[4], x_state[5]
    d, delta  = u_input[0], u_input[1]

    # Odwróć proxy: d_proxy = vx_target / VX_SCALE => vx_target = d * VX_SCALE
    # Klipujemy d do [0,1] jak w modelu nominalnym
    vx_target = float(np.clip(d, 0.0, 1.0)) * VX_SCALE

    return np.array([vx, vy, r, delta, vx_target], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Model GP (jeden wyjściowy)
# ══════════════════════════════════════════════════════════════════════════════

class _SingleOutputGP(ExactGP):
    """
    Dokładny GP z kernelem Matérn-5/2 i ARD.
    Matérn-5/2 modeluje funkcje z ograniczonymi gradientami –
    odpowiednie dla dynamiki opon.
    """

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: GaussianLikelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = ZeroMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=N_GP_FEATURES)
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x),
                                  self.covar_module(x))


# ══════════════════════════════════════════════════════════════════════════════
#  Główna klasa GPResidualModel
# ══════════════════════════════════════════════════════════════════════════════

class GPResidualModel:
    """
    Model resztkowy GP dla F1/10 – poprawia predykcję [vx, vy, r].

    Trzy niezależne GP (po jednym na vx, vy, r).
    Cechy wejściowe: [vx, vy, r, delta, vx_target].
    Cel: resztka modelu nominalnego.

    Użycie:
        gp = GPResidualModel(enabled=True, max_data=1000)
        gp.train_from_npz(paths, vehicle_model, dt=0.02)
        gp.save("gp_model.pt")

        # w pętli sterowania:
        x_next_corrected = gp.correct_state(x_nom_next, x_current, u)
    """

    def __init__(self,
                 enabled:      bool = True,
                 max_data:     int  = 1000,
                 device:       str  = 'cpu',
                 n_train_iter: int  = 150):
        self.enabled      = enabled
        self.max_data     = max_data
        self.device       = torch.device(device)
        self.n_train_iter = n_train_iter
        self.trained      = False

        self._models:      List[_SingleOutputGP]     = []
        self._likelihoods: List[GaussianLikelihood]  = []

        self._feat_mean: Optional[np.ndarray] = None
        self._feat_std:  Optional[np.ndarray] = None

        print(f"[GP] Inicjalizacja: enabled={enabled}, "
              f"max_data={max_data}, device={device}")

    # ── Normalizacja ──────────────────────────────────────────────────────

    def _fit_normalizer(self, X: np.ndarray) -> None:
        self._feat_mean = X.mean(axis=0).astype(np.float32)
        self._feat_std  = X.std(axis=0).astype(np.float32)
        self._feat_std  = np.where(self._feat_std < 1e-6, 1.0, self._feat_std)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self._feat_mean is None:
            return X
        return ((X - self._feat_mean) / self._feat_std).astype(np.float32)

    # ── Trening ───────────────────────────────────────────────────────────

    def train_from_npz(
        self,
        npz_paths: List[Path],
        vehicle_model,
        dt:      float = 0.02,
        verbose: bool  = True,
    ) -> None:
        """
        Trenuje GP na plikach .npz.

        Args:
            npz_paths:     lista ścieżek .npz
            vehicle_model: DynamicBicycleModel (model nominalny)
            dt:            krok czasowy [s]
            verbose:       wypisuj postęp
        """
        if not self.enabled:
            print("[GP] GP wyłączony – pomijam trening.")
            return

        print(f"\n[GP] Wczytywanie danych z {len(npz_paths)} pliku/plików...")
        all_X, all_Y = [], []

        for path in npz_paths:
            path = Path(path)
            if not path.exists():
                print(f"[GP]   BRAK: {path}")
                continue
            X_i, Y_i = load_npz_residuals(path, vehicle_model, dt)
            if X_i.shape[0] == 0:
                print(f"[GP]   {path.name}: 0 próbek po filtrowaniu")
                continue
            all_X.append(X_i)
            all_Y.append(Y_i)
            if verbose:
                print(f"[GP]   {path.name}: {X_i.shape[0]} próbek  "
                      f"resid std=[vx:{Y_i[:,0].std():.4f}, "
                      f"vy:{Y_i[:,1].std():.4f}, "
                      f"r:{Y_i[:,2].std():.4f}]")

        if not all_X:
            print("[GP] Brak danych!")
            return

        X = np.vstack(all_X)
        Y = np.vstack(all_Y)

        # Wypisz rozkład resztek przed subsamplowaniem
        if verbose:
            print(f"[GP] Łącznie {X.shape[0]} próbek")
            for i, n in enumerate(['vx', 'vy', 'r']):
                print(f"[GP]   resid_{n}: mean={Y[:,i].mean():.4f}  "
                      f"std={Y[:,i].std():.4f}  "
                      f"max|r|={np.abs(Y[:,i]).max():.4f}")

        # Subsampling
        if X.shape[0] > self.max_data:
            idx = np.random.choice(X.shape[0], self.max_data, replace=False)
            X, Y = X[idx], Y[idx]
            if verbose:
                print(f"[GP] Subsampling → {self.max_data} punktów")

        # Normalizacja cech
        self._fit_normalizer(X)
        X_norm = self._normalize(X)

        # Trening 3 niezależnych GP
        self._models, self._likelihoods = [], []
        output_names = ['vx', 'vy', 'r']

        for i, name in enumerate(output_names):
            y_i    = Y[:, i]
            train_x = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
            train_y = torch.tensor(y_i,    dtype=torch.float32).to(self.device)

            likelihood = GaussianLikelihood().to(self.device)
            gp         = _SingleOutputGP(train_x, train_y, likelihood).to(self.device)

            gp.train(); likelihood.train()
            optimizer = torch.optim.Adam(
                list(gp.parameters()) + list(likelihood.parameters()), lr=0.05)
            mll = ExactMarginalLogLikelihood(likelihood, gp)

            if verbose:
                print(f"[GP]   Trenuję GP_{name}...", end="", flush=True)

            prev_loss = float('inf')
            for it in range(self.n_train_iter):
                optimizer.zero_grad()
                out  = gp(train_x)
                loss = -mll(out, train_y)
                loss.backward()
                optimizer.step()
                if it > 30 and abs(prev_loss - loss.item()) < 1e-5:
                    if verbose:
                        print(f" (zbieżność @ iter {it+1})", end="")
                    break
                prev_loss = loss.item()

            if verbose:
                ls = gp.covar_module.base_kernel.lengthscale \
                       .detach().cpu().numpy().flatten()
                noise = likelihood.noise.item()
                feat_l = ['vx','vy','r','delta','vx_tgt']
                ls_str = ", ".join(f"{feat_l[j]}:{ls[j]:.3f}"
                                   for j in range(len(ls)))
                print(f" loss={loss.item():.4f}, noise={noise:.5f}")
                print(f"[GP]     ls=({ls_str})")

            gp.eval(); likelihood.eval()
            self._models.append(gp)
            self._likelihoods.append(likelihood)

        self.trained = True
        print("[GP] Trening zakończony.\n")

    # ── Predykcja ─────────────────────────────────────────────────────────

    def predict_residual(
        self,
        x_state: np.ndarray,
        u_input: np.ndarray,
        return_variance: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Przewiduje resztkę [vx, vy, r] dla jednego punktu.

        Args:
            x_state: stan [s, n, mu, vx, vy, r]
            u_input: sterowanie MPC [d, delta]
            return_variance: czy zwracać wariancję

        Returns:
            (mean, variance|None) – shape (3,)
        """
        if not self.enabled or not self.trained:
            zeros = np.zeros(N_GP_OUTPUTS)
            return zeros, (zeros if return_variance else None)

        z      = extract_features_from_state(x_state, u_input)
        z_norm = self._normalize(z[np.newaxis, :])
        test_x = torch.tensor(z_norm, dtype=torch.float32).to(self.device)

        means, variances = [], []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for gp, lik in zip(self._models, self._likelihoods):
                pred = lik(gp(test_x))
                means.append(pred.mean.cpu().item())
                variances.append(pred.variance.cpu().item())

        m = np.array(means,     dtype=np.float64)
        v = np.array(variances, dtype=np.float64)
        return m, (v if return_variance else None)

    def predict_residual_batch(
        self,
        X_states: np.ndarray,
        U_inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Batchowa predykcja resztek dla całego horyzontu MPC.

        Args:
            X_states: (N, 6) stany horyzontu
            U_inputs: (N, 2) sterowania horyzontu

        Returns:
            residuals: (N, 3) resztki [vx, vy, r]
        """
        if not self.enabled or not self.trained:
            return np.zeros((X_states.shape[0], N_GP_OUTPUTS))

        N = X_states.shape[0]
        Z = np.vstack([
            extract_features_from_state(X_states[k], U_inputs[k])
            for k in range(N)
        ])                               # (N, 5)
        Z_norm = self._normalize(Z)
        test_x = torch.tensor(Z_norm, dtype=torch.float32).to(self.device)

        out = np.zeros((N, N_GP_OUTPUTS))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i, (gp, lik) in enumerate(zip(self._models, self._likelihoods)):
                pred       = lik(gp(test_x))
                out[:, i]  = pred.mean.cpu().numpy()
        return out

    # ── Korekta stanu ─────────────────────────────────────────────────────

    def correct_state(
        self,
        x_nominal_next: np.ndarray,
        x_current:      np.ndarray,
        u_input:        np.ndarray,
    ) -> np.ndarray:
        """
        Koryguje przewidywany następny stan o resztkę GP.

        x_{t+1} = f(x_t, u_t) + B_d · g(x_t, u_t)
        Bd wybiera składowe [vx, vy, r].
        """
        if not self.enabled or not self.trained:
            return x_nominal_next
        residual, _ = self.predict_residual(x_current, u_input)
        x_corr = x_nominal_next.copy()
        x_corr[GP_OUTPUT_IDX] += residual
        return x_corr

    # ── Zapis / wczytanie ─────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Zapisuje model GP do pliku .pt."""
        if not self.trained:
            print("[GP] Model niewytrend. – pomijam zapis.")
            return

        torch.save({
            'state_dicts':  [m.state_dict()  for m in self._models],
            'lik_dicts':    [l.state_dict()  for l in self._likelihoods],
            'train_xs':     [m.train_inputs[0].cpu() for m in self._models],
            'train_ys':     [m.train_targets.cpu()   for m in self._models],
            'feat_mean':    self._feat_mean,
            'feat_std':     self._feat_std,
            'enabled':      self.enabled,
            'max_data':     self.max_data,
            'n_train_iter': self.n_train_iter,
        }, path)
        print(f"[GP] Zapisano: {path}")

    def load(self, path: str) -> None:
        """Wczytuje model GP z pliku .pt."""
        if not os.path.exists(path):
            print(f"[GP] Brak pliku: {path}")
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self._feat_mean   = ckpt['feat_mean']
        self._feat_std    = ckpt['feat_std']
        self.enabled      = ckpt.get('enabled', True)
        self.max_data     = ckpt.get('max_data', self.max_data)

        self._models, self._likelihoods = [], []
        for tx, ty, sd, ld in zip(ckpt['train_xs'], ckpt['train_ys'],
                                   ckpt['state_dicts'], ckpt['lik_dicts']):
            tx = tx.to(self.device)
            ty = ty.to(self.device)
            lik = GaussianLikelihood().to(self.device)
            gp  = _SingleOutputGP(tx, ty, lik).to(self.device)
            gp.load_state_dict(sd); lik.load_state_dict(ld)
            gp.eval(); lik.eval()
            self._models.append(gp)
            self._likelihoods.append(lik)

        self.trained = True
        n = self._models[0].train_inputs[0].shape[0]
        print(f"[GP] Wczytano z: {path}  ({n} punktów treningowych)")

    # ── Diagnostyka ───────────────────────────────────────────────────────

    def get_info(self) -> dict:
        info = {'enabled': self.enabled, 'trained': self.trained,
                'max_data': self.max_data, 'device': str(self.device)}
        if self.trained:
            info['n_train_points'] = self._models[0].train_inputs[0].shape[0]
            info['output_states']  = ['vx', 'vy', 'r']
            info['feature_names']  = ['vx', 'vy', 'r', 'delta', 'vx_target']
            info['lengthscales']   = [
                m.covar_module.base_kernel.lengthscale
                 .detach().cpu().numpy().flatten().tolist()
                for m in self._models
            ]
            info['noise'] = [l.noise.item() for l in self._likelihoods]
        return info

    def print_info(self) -> None:
        info = self.get_info()
        print("\n[GP] ══ Informacje o modelu GP ══")
        print(f"  Aktywny:     {info['enabled']}")
        print(f"  Wytrenowany: {info['trained']}")
        if info['trained']:
            print(f"  Próbki treningowe: {info['n_train_points']}")
            print(f"  Wyjście: {info['output_states']}")
            print(f"  Cechy:   {info['feature_names']}")
            fl = info['feature_names']
            for name, ls, noise in zip(info['output_states'],
                                        info['lengthscales'],
                                        info['noise']):
                ls_str = ", ".join(f"{fl[j]}:{ls[j]:.3f}"
                                   for j in range(len(ls)))
                print(f"  GP_{name}: ls=({ls_str}), noise={noise:.5f}")
        print()

    def __call__(self, x_state: np.ndarray,
                 u_input: np.ndarray) -> np.ndarray:
        residual, _ = self.predict_residual(x_state, u_input)
        return residual


# ══════════════════════════════════════════════════════════════════════════════
#  Walidacja i diagnostyka
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_gp(
    gp:           GPResidualModel,
    npz_paths:    List[Path],
    vehicle_model,
    dt:           float = 0.02,
    verbose:      bool  = True,
) -> dict:
    """
    Ocenia model GP na zbiorze walidacyjnym.

    Returns:
        dict z RMSE i bias dla ['vx', 'vy', 'r']
    """
    all_X, all_Y = [], []
    for path in npz_paths:
        X_i, Y_i = load_npz_residuals(path, vehicle_model, dt)
        if X_i.shape[0] > 0:
            all_X.append(X_i); all_Y.append(Y_i)
    if not all_X:
        return {}

    X = np.vstack(all_X)
    Y = np.vstack(all_Y)

    # Zbuduj macierz stanów i sterowań dla batch predykcji
    # X = [vx, vy, r, delta, vx_target] → odtwórz x_state i u_input
    # x_state: vx=X[:,0], vy=X[:,1], r=X[:,2], pozostałe=0
    # u_input: d = vx_target/VX_SCALE = X[:,4]/VX_SCALE, delta=X[:,3]
    N = len(X)
    X_state = np.zeros((N, 6))
    X_state[:, 3] = X[:, 0]  # vx
    X_state[:, 4] = X[:, 1]  # vy
    X_state[:, 5] = X[:, 2]  # r

    U_input = np.zeros((N, 2))
    U_input[:, 0] = np.clip(X[:, 4] / VX_SCALE, 0, 1)  # d = vx_tgt/VX_SCALE
    U_input[:, 1] = X[:, 3]                              # delta

    pred = gp.predict_residual_batch(X_state, U_input)

    result = {}
    names  = ['vx', 'vy', 'r']
    if verbose:
        print("[GP] Walidacja (RMSE resztek):")
    for i, n in enumerate(names):
        err  = Y[:, i] - pred[:, i]
        rmse = float(np.sqrt(np.mean(err**2)))
        bias = float(err.mean())
        result[n] = {'rmse': rmse, 'bias': bias}
        if verbose:
            baseline_rmse = float(np.sqrt(np.mean(Y[:, i]**2)))
            print(f"  resid_{n}: RMSE={rmse:.5f}  bias={bias:.5f}  "
                  f"baseline(0-pred)={baseline_rmse:.5f}  "
                  f"R²={1-rmse**2/baseline_rmse**2:.3f}")
    return result