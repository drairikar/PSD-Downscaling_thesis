# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nvtx
import torch
import tqdm
import os
# in src/utils/utils.py
from physicsnemo.utils.generative.utils import StackedRandomGenerator
from tueplots import bundles, figsizes
from scipy.stats import ks_2samp  

from typing import Callable, Optional

import torch
from torch import Tensor
from .. import constants
from physicsnemo.utils.patching import GridPatching2D

import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def init_wandb_metrics(wandb_logger):
    """
    Set up wandb metrics to track
    """
    experiment = wandb_logger.experiment
    experiment.define_metric("val_mean_loss", summary="min")
    # for step in constants.VAL_STEP_LOG_ERRORS:
    #     experiment.define_metric(f"val_loss_unroll{step}", summary="min")




def fractional_plot_bundle(fraction):
    """
    Get the tueplots bundle, but with figure width as a fraction of
    the page width.
    """
    bundle = bundles.neurips2023(usetex=False, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (
        original_figsize[0] / fraction,
        original_figsize[1],
    )
    return bundle


def load_dataset_stats(dataset_name, device="cpu"):
    """
    Load arrays with stored dataset statistics from pre-processing
    """
    static_dir_path = os.path.join(dataset_name, "static")

    def loads_file(fn):
        return torch.load(
            os.path.join(static_dir_path, fn), map_location=device
        )

    data_mean = loads_file("parameter_mean.pt")  # (d_features,)
    data_std = loads_file("parameter_std.pt")  # (d_features,)

    return {
        "data_mean": data_mean,
        "data_std": data_std,
    }

def compute_psd_2d(data: torch.Tensor, dx: float = 1.0):
    """
    Compute radially averaged 2D Power Spectral Density.

    Parameters
    ----------
    data : torch.Tensor
        Shape (N, H, W) — N samples.
    dx : float
        Grid spacing (same in both directions).

    Returns
    -------
    k_centers : np.ndarray
        Wavenumber bin centers.
    psd_radial : np.ndarray
        Radially averaged PSD.
    """
    N, H, W = data.shape

    # 2D FFT, normalize by grid size
    fft_2d = torch.fft.fft2(data) / (H * W)
    psd_2d = torch.abs(fft_2d) ** 2
    psd_2d = psd_2d.mean(dim=0)  # average over samples → (H, W)

    # Wavenumber grids
    kx = torch.fft.fftfreq(W, d=dx)
    ky = torch.fft.fftfreq(H, d=dx)
    KY, KX = torch.meshgrid(ky, kx, indexing='ij')
    K = torch.sqrt(KX ** 2 + KY ** 2)

    # Radial binning
    k_max = min(kx.abs().max(), ky.abs().max())
    n_bins = min(H, W) // 2
    k_bin_edges = torch.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])

    psd_radial = torch.zeros(n_bins)
    for i in range(n_bins):
        mask = (K >= k_bin_edges[i]) & (K < k_bin_edges[i + 1])
        if mask.any():
            psd_radial[i] = psd_2d[mask].mean()

    return k_centers.numpy(), psd_radial.numpy()


def compute_derived_fields(data: torch.Tensor, var_names: list) -> dict:
    """
    Compute derived meteorological fields from raw variables.

    Parameters
    ----------
    data : torch.Tensor
        Shape (N, C, H, W).
    var_names : list
        List of variable names matching channel order.

    Returns
    -------
    dict
        Keys: 'vorticity', 'divergence', 'kinetic_energy'.
        Values: torch.Tensor of shape (N, H, W).
    """
    u_idx = var_names.index("u10")
    v_idx = var_names.index("v10")
    u = data[:, u_idx, :, :]  # (N, H, W)
    v = data[:, v_idx, :, :]

    dvdx = torch.gradient(v, dim=-1)[0]
    dudy = torch.gradient(u, dim=-2)[0]
    dudx = torch.gradient(u, dim=-1)[0]
    dvdy = torch.gradient(v, dim=-2)[0]

    derived = {
        "vorticity": dvdx - dudy,
        "divergence": dudx + dvdy,
        "kinetic_energy": 0.5 * (u ** 2 + v ** 2),
    }

    # Cn2 (atmospheric turbulence) — dry-air approximation
    # t2m = data[:, var_names.index("t2m"), :, :]    # K
    # # # sshf = -data[:, var_names.index("sshf"), :, :] / 1.216e3  # K m/s (convert from W/m² to K m/s using specific heat capacity of air)
    # # # ustar = data[:, var_names.index("zust"), :, :]  # m/s
    # sp = data[:, var_names.index("sp"), :, :]  / 100     # HPa

    # A = 7.9e-5         # K/Pa
    # Z = 2.0            # reference height m
    # KAPPA = 0.4        # von Kármán constant
    # G = 9.81           # gravitational acceleration m/s²

    # # Obukhov length (preserve sign, guard against division by zero)
    # theta_v = t2m
    # ustar_safe = ustar.clamp(min=1e-6)
    # sshf_safe = torch.where(sshf.abs() < 1e-6, torch.full_like(sshf, 1e-6), sshf)
    # L = -(ustar_safe ** 3 * theta_v) / (KAPPA * G * sshf_safe)

    # stability parameter, clamp to avoid extreme values
    # zeta = (Z / L).clamp(-10.0, 10.0)

    # Wyngaard (1971) stability function
    # g_zeta = torch.where(
    #     zeta < 0,
    #     4.9 * (1 - 6.1 * zeta) ** (-2.0 / 3.0),  # unstable
    #     4.9 * (1 + 2.2 * zeta),                  # stable
    # )

    # CT2 = (-sshf / ustar_safe) ** 2 * Z ** (-2.0 / 3.0) * g_zeta
    # Cn2 = (A * sp / t2m ** 2) ** 2 * CT2

    # # Replace any remaining non-finite values with zero
    # Cn2 = torch.where(torch.isfinite(Cn2), Cn2, torch.zeros_like(Cn2))

    # derived["Cn2"] = Cn2

    return derived




def compute_metrics(
    path_gt: str,
    path_pred: str,
    save_dir: str,
    *,
    var_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compare every .npy file that exists in BOTH `path_gt` and `path_pred`,
    assuming each file has shape (H, W, C) with the same C variables.

    Returns a nested dict: metrics[var_name][metric] = value
    and writes the same information to <save_dir>/metrics.txt.
    """

    path_gt, path_pred = Path(path_gt), Path(path_pred)
    gt_files   = {f.name: f for f in path_gt.glob("*.npy")}
    pred_files = {f.name: f for f in path_pred.glob("*.npy")}
    common     = sorted(gt_files.keys() & pred_files.keys())

    if not common:
        raise FileNotFoundError("No overlapping .npy filenames in the two folders.")

    # ----- discover channel count from the first file
    first = np.load(gt_files[common[0]])
    if first.ndim != 3:
        raise ValueError(
            f"Expected shape (H, W, C). Found {first.shape} in {common[0]!r}"
        )
    C = first.shape[-1]
    if var_names is None:
        var_names = [f"var{c}" for c in range(C)]
    if len(var_names) != C:
        raise ValueError("Length of var_names must equal number of channels (C).")

    # accumulators: metric_sums[var][metric] = running total
    metric_sums = {v: {"MAE": 0.0, "RMSE": 0.0, "SSIM": 0.0, "PSNR": 0.0}
                   for v in var_names}
    n_files = 0

    for fname in common:
        gt  = np.load(gt_files[fname]).astype(np.float32)
        prd = np.load(pred_files[fname]).astype(np.float32)

        gt  = ensure_channels_last(gt,  C)
        prd = ensure_channels_last(prd, C)

        if gt.shape != prd.shape:
            raise ValueError(f"Shape mismatch for {fname}: {gt.shape} vs {prd.shape}")
        if gt.shape[-1] != C:
            raise ValueError(f"Channel count changed in {fname}: got {gt.shape[-1]}, expected {C}")

        for c, vname in enumerate(var_names):
            g = gt[..., c]
            p = prd[..., c]

            metric_sums[vname]["MAE"]    += compute_mae(p, g)
            metric_sums[vname]["RMSE"]   += compute_rmse(p, g)
            metric_sums[vname]["SSIM"]   += compute_ssim_metric(p, g)
            metric_sums[vname]["PSNR"]   += compute_psnr_metric(p, g)
            metric_sums[vname]["Cramer"] += compute_cramer(p, g)

        n_files += 1

    # --- final averages
    metrics = {
        v: {m: total / n_files for m, total in metric_sums[v].items()}
        for v in var_names
    }

    # --- save
    os.makedirs(save_dir, exist_ok=True)
    with open(Path(save_dir) / "metrics_new.txt", "w") as fh:
        for v in var_names:
            fh.write(f"[{v}]\n")
            for m, val in metrics[v].items():
                fh.write(f"  {m}: {val:.6f}\n")
            fh.write("\n")

    return metrics



def compute_mae(p, g):
    return np.abs(p - g).mean()

def compute_rmse(p, g):
    return np.sqrt(np.mean((p - g) ** 2))

def compute_ssim_metric(p, g):
    dr = g.max() - g.min()
    return ssim(g, p, data_range=dr)

def compute_psnr_metric(p, g):
    dr = g.max() - g.min()
    return psnr(g, p, data_range=dr)

def compute_cramer(p, g):
    gx = torch.from_numpy(g.flatten()).float().unsqueeze(1)
    px = torch.from_numpy(p.flatten()).float().unsqueeze(1)
    dxy = torch.cdist(gx, px).mean()
    dgg = torch.cdist(gx, gx).mean()
    dpp = torch.cdist(px, px).mean()
    return (2 * dxy - dgg - dpp).item()

def compute_wind_rmse(u_p, v_p, u_g, v_g):
    """RMSE of wind‑speed magnitude."""
    speed_p = np.hypot(u_p, v_p)
    speed_g = np.hypot(u_g, v_g)
    return np.sqrt(np.mean((speed_p - speed_g) ** 2))

def compute_vorticity_rms(u_p, v_p, u_g, v_g, dx=5500, dy=5500):
    """RMS of vorticity error  ζ = dv/dx − du/dy  (finite‑difference, same grid)."""
    ζ_p = np.gradient(v_p, dx, axis=1) - np.gradient(u_p, dy, axis=0)
    ζ_g = np.gradient(v_g, dx, axis=1) - np.gradient(u_g, dy, axis=0)
    return np.sqrt(np.mean((ζ_p - ζ_g) ** 2))

def compute_ks_metric(p, g):
    """Two‑sample Kolmogorov–Smirnov statistic (two‑sided)."""
    return ks_2samp(p.ravel(), g.ravel(), alternative="two-sided").statistic

def compute_hill_metric(p, g, k=100):
    """Absolute difference of Hill tail indices (heavy‑tail focus)."""
    def hill(x, k):
        x = np.abs(x.ravel()) + 1e-6  # ensure positive
        x_sorted = np.sort(x)[::-1]
        k = min(k, len(x_sorted) - 1)
        x_k = x_sorted[k]
        return (1.0 / k) * np.log(x_sorted[:k] / x_k).sum()
    return abs(hill(p, k) - hill(g, k))


def ensure_channels_last(arr: np.ndarray, C_expected: int) -> np.ndarray:
    """Convert (H,W,C) or (C,H,W) → (H,W,C). Error if neither axis matches C."""
    if arr.shape[-1] == C_expected:      # already (H,W,C)
        return arr
    if arr.shape[0] == C_expected:       # assume (C,H,W)
        return np.moveaxis(arr, 0, -1)   # → (H,W,C)
    raise ValueError(f"Cannot locate channel axis in shape {arr.shape}")



if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from pprint import pprint

    parser = argparse.ArgumentParser(
        description="Compute MAE, RMSE, SSIM, PSNR for two folders of (H,W,C) .npy files."
    )
    parser.add_argument("--path_gt",   required=True, help="Directory with ground-truth .npy files")
    parser.add_argument("--path_pred", required=True, help="Directory with prediction  .npy files")
    parser.add_argument("--save_dir",  required=True, help="Where metrics_new.txt will be written")
    parser.add_argument(
        "--var_names",
        nargs="*",
        default=None,
        help="Optional list of variable names (length must equal channel count, e.g. "
             "--var_names u10 v10 t2m sshf zust).",
    )
    args = parser.parse_args()

    # Resolve paths so the metrics file ends up exactly where you expect
    metrics = compute_metrics(
        Path(args.path_gt).expanduser(),
        Path(args.path_pred).expanduser(),
        Path(args.save_dir).expanduser(),
        var_names=['u10', 'v10', 't2m', 'sshf', 'zust'],
    )

    pprint(metrics)
    
    