#!/usr/bin/env python3
"""Compare Vanilla DPS, Sequential DPS, and deterministic UNet PSDs.

The DPS ``.npz`` files already contain PSD curves for one evaluation window.
The UNet inference directory can contain a much longer test period, so this
script selects UNet files by timestamp before applying the same radially
averaged 2-D PSD calculation used by the DPS evaluation code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr


VARIABLES = ("t2m", "u10", "v10")
CHANNELS = {"u10": 0, "v10": 1, "t2m": 2}
TITLES = {"t2m": r"$T_{2m}$", "u10": r"$u_{10}$", "v10": r"$v_{10}$"}

ROOT = Path(__file__).resolve().parent
DEFAULT_UNET_DIR = (
    ROOT
    / "inference_outputs"
    / "UNet-test2-PSDLoss-UNet-CNN-09_08_10-5355"
    / "files"
)
DEFAULT_VANILLA_DIR = (
    ROOT
    / "psd_plots_Geodiff_CERRA"
    / "GeoDiffPrior_test_cerraeval-test--geodiff_era5_cerra-08_03_18-9839"
    / "psd"
    / "CentralEurope"
)
DEFAULT_SEQUENTIAL_DIR = (
    ROOT
    / "psd_plots_Geodiff_CERRA_seq"
    / "GeoDiffPriorCERRASeqeval-test--geodiff_seq-08_03_18-1242"
    / "psd"
    / "CentralEurope"
)
DEFAULT_CERRA = Path(
    "/projects/0/prjs0951/Carlo/zz_processed_data_new/"
    "CERRA/test/CentralEurope.nc"
)


def radial_psd_2d(data: np.ndarray, dx: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Match ``src.utils.utils.compute_psd_2d`` exactly.

    ``data`` has shape ``(sample, y, x)``. The spatial FFT is calculated for
    each timestamp first and power is then averaged over timestamps.
    """
    if data.ndim != 3:
        raise ValueError(f"PSD input must have shape (sample, y, x), got {data.shape}.")

    tensor = torch.as_tensor(data, dtype=torch.float32)
    _, height, width = tensor.shape
    spectrum = torch.fft.fft2(tensor) / (height * width)
    power = torch.mean(torch.abs(spectrum) ** 2, dim=0)

    kx = torch.fft.fftfreq(width, d=dx)
    ky = torch.fft.fftfreq(height, d=dx)
    grid_y, grid_x = torch.meshgrid(ky, kx, indexing="ij")
    radius = torch.sqrt(grid_x**2 + grid_y**2)

    n_bins = min(height, width) // 2
    edges = torch.linspace(0.0, min(kx.abs().max(), ky.abs().max()), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    radial = torch.zeros(n_bins)
    for index in range(n_bins):
        mask = (radius >= edges[index]) & (radius < edges[index + 1])
        if mask.any():
            radial[index] = power[mask].mean()
    return centers.numpy().astype(np.float64), radial.numpy().astype(np.float64)


def evaluation_times(
    start: str, end: str, cadence_hours: int
) -> pd.DatetimeIndex:
    if cadence_hours < 1:
        raise ValueError("--cadence-hours must be positive.")
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    if end_time <= start_time:
        raise ValueError("--end must be later than --start.")
    return pd.date_range(
        start=start_time,
        end=end_time,
        freq=pd.Timedelta(hours=cadence_hours),
        inclusive="left",
    )


def load_unet(unet_dir: Path, times: pd.DatetimeIndex) -> np.ndarray:
    files = [
        unet_dir / f"nwp_{timestamp.strftime('%Y%m%dT%H%M%S')}.npy"
        for timestamp in times
    ]
    missing = [path for path in files if not path.is_file()]
    if missing:
        preview = ", ".join(path.name for path in missing[:5])
        raise FileNotFoundError(
            f"Run 5355 is missing {len(missing)} requested timestamp(s): {preview}"
        )

    stack = np.stack([np.load(path) for path in files])
    if stack.ndim != 4 or stack.shape[1] != len(CHANNELS):
        raise ValueError(
            "Expected run 5355 files in (channel, y, x) order with channels "
            f"[u10, v10, t2m], got stacked shape {stack.shape}."
        )
    return stack


def load_cerra(cerra_path: Path, times: pd.DatetimeIndex) -> np.ndarray:
    with xr.open_dataset(cerra_path, engine="h5netcdf") as dataset:
        selected = dataset[["u10", "v10", "t2m"]].sel(time=times)
        selected_times = pd.DatetimeIndex(selected.time.values)
        if not selected_times.equals(times):
            raise ValueError("CERRA did not return the exact requested timestamps.")
        return np.stack(
            [selected[name].values for name in ("u10", "v10", "t2m")], axis=1
        )


def _first_present(archive: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> np.ndarray:
    for key in keys:
        if key in archive:
            return np.asarray(archive[key], dtype=np.float64)
    raise KeyError(f"None of {keys} is present; found keys {archive.files}.")


def load_dps_curve(directory: Path, variable: str) -> dict[str, np.ndarray]:
    path = directory / f"{variable}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing DPS PSD archive: {path}")
    with np.load(path) as archive:
        return {
            "k_target": _first_present(archive, ("k_cerra", "k_target")),
            "psd_target": _first_present(archive, ("psd_cerra", "psd_target")),
            "k_prediction": _first_present(archive, ("k_prediction", "k_pred")),
            "psd_prediction": _first_present(
                archive, ("psd_prediction", "psd_pred")
            ),
        }


def check_curve(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape:
        raise ValueError(f"{name} has shape {actual.shape}; expected {expected.shape}.")
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=2e-4,
        atol=1e-10,
        err_msg=(
            f"{name} does not use the requested CERRA time window. Check the DPS "
            "evaluation start/end times and regenerate its .npz files if necessary."
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA_DIR)
    parser.add_argument("--sequential-dir", type=Path, default=DEFAULT_SEQUENTIAL_DIR)
    parser.add_argument("--unet-dir", type=Path, default=DEFAULT_UNET_DIR)
    parser.add_argument("--cerra-path", type=Path, default=DEFAULT_CERRA)
    parser.add_argument("--start", default="2021-01-01T01:00:00")
    parser.add_argument("--end", default="2021-01-02T01:00:00")
    parser.add_argument("--cadence-hours", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "dps_unet_psd.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    times = evaluation_times(args.start, args.end, args.cadence_hours)
    print(f"Using {len(times)} timestamps from {times[0]} through {times[-1]}.")

    unet = load_unet(args.unet_dir, times)
    cerra = load_cerra(args.cerra_path, times)
    if unet.shape != cerra.shape:
        raise ValueError(f"UNet shape {unet.shape} does not match CERRA {cerra.shape}.")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    styles = {
        "CERRA": {"color": "black", "linewidth": 2.4},
        "Vanilla DPS": {"color": "tab:blue", "linewidth": 2.0},
        "Sequential DPS": {"color": "tab:orange", "linewidth": 2.0},
        "UNet": {"color": "tab:green", "linewidth": 2.0},
    }

    for axis, variable in zip(axes, VARIABLES):
        channel = CHANNELS[variable]
        k_cerra, psd_cerra = radial_psd_2d(cerra[:, channel])
        k_unet, psd_unet = radial_psd_2d(unet[:, channel])
        vanilla = load_dps_curve(args.vanilla_dir, variable)
        sequential = load_dps_curve(args.sequential_dir, variable)

        check_curve(f"Vanilla DPS {variable} wavenumbers", vanilla["k_target"], k_cerra)
        check_curve(f"Sequential DPS {variable} wavenumbers", sequential["k_target"], k_cerra)
        check_curve(f"Vanilla DPS {variable} CERRA", vanilla["psd_target"], psd_cerra)
        check_curve(
            f"Sequential DPS {variable} CERRA", sequential["psd_target"], psd_cerra
        )

        series = {
            "CERRA": (k_cerra, psd_cerra),
            "Vanilla DPS": (vanilla["k_prediction"], vanilla["psd_prediction"]),
            "Sequential DPS": (
                sequential["k_prediction"],
                sequential["psd_prediction"],
            ),
            "UNet": (k_unet, psd_unet),
        }
        for label, (wavenumber, psd) in series.items():
            valid = (wavenumber > 0) & (psd > 0) & np.isfinite(psd)
            axis.loglog(wavenumber[valid], psd[valid], label=label, **styles[label])

        axis.set_title(TITLES[variable], fontsize=14)
        axis.set_xlabel("Wavenumber")
        axis.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("Power spectral density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
