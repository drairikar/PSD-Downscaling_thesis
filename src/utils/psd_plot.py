#!/usr/bin/env python3
"""Create thesis-ready PSD panels and per-variable PSD tables.

Two 1 x 3 figures are written:

* ``primary_variables_psd``: u10, v10, and t2m
* ``derived_variables_psd``: vorticity, divergence, and kinetic energy

CERRA and ERA5 references may be NetCDF files. Model predictions may be a
directory of channel-first ``.npy`` samples, a single ``.npy`` sample, or a
stacked ``.npy`` file. Internally, all data use ``(sample, y, x, channel)``.
"""

from __future__ import annotations

import pathlib
from typing import Dict, Iterable, Iterator, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PRIMARY_VARS = ("u10", "v10", "t2m")
DERIVED_VARS = ("vorticity", "divergence", "kinetic_energy")
CHANNEL_MAP = {name: index for index, name in enumerate(PRIMARY_VARS)}

VARIABLE_TITLES = {
    "u10": "$u_{10}$",
    "v10": "$v_{10}$",
    "t2m": "$T_{2m}$)",
    "vorticity": "Vorticity",
    "divergence": "Divergence",
    "kinetic_energy": "Kinetic energy",
}

CERRA_PATH = pathlib.Path(
    "/projects/0/prjs0951/Carlo/zz_processed_data_new/CERRA/test/CentralEurope.nc"
)
ERA5_PATH = pathlib.Path(
    "/projects/0/prjs0951/Carlo/zz_processed_data_new/ERA5/test/CentralEurope.nc"
)

MODEL_PATHS: Dict[str, pathlib.Path] = {
    "U-Net": pathlib.Path(
        "/home/csaccardi1/Devashish_thesis/PSD-Downscaling_thesis/"
        "inference_outputs/UNet-test2-PSDLoss-UNet-CNN-09_08_10-5355/files"
    ),
    "FNO": pathlib.Path(
        "/home/csaccardi1/Devashish_thesis/PSD-Downscaling_thesis/"
        "inference_outputs/FNO-testv2-FNO-09_10_10-6674/files"
    ),
    "U-NO": pathlib.Path(
        "/home/csaccardi1/Devashish_thesis/PSD-Downscaling_thesis/"
        "inference_outputs/UNO_final-UNO-09_12_23-2417/files"
    ),
    "FNO_175": pathlib.Path(
        "/home/csaccardi1/Devashish_thesis/PSD-Downscaling_thesis/inference_outputs_resolution/test175/FNO-testv2-FNO-09_10_10-6674/files"
    ),
}

# ERA5's nominal longitudinal grid spacing. The effective CERRA spacing below
# is scaled by the ratio of the ERA5 and CERRA longitudinal grid sizes.
ERA5_DX_DEG = 0.25
OUT_DIR = pathlib.Path("plot_tests_thesis_res")
BATCH_SIZE = 32

SERIES_STYLE = {
    "CERRA": {"color": "#1f1f1f", "linestyle": "-", "linewidth": 2.4},
    "U-Net": {"color": "#0072B2", "linestyle": "-", "linewidth": 1.8},
    "FNO": {"color": "#D55E00", "linestyle": "--", "linewidth": 1.8},
    "U-NO": {"color": "#CC79A7", "linestyle": "-.", "linewidth": 1.8},
}


# -----------------------------------------------------------------------------
# I/O and calculations
# -----------------------------------------------------------------------------
def _to_channel_last(array: np.ndarray) -> np.ndarray:
    """Return an array with shape (sample, y, x, channel)."""
    if array.ndim == 3:  # (channel, y, x)
        array = array[None, ...]
    if array.ndim != 4:
        raise ValueError(f"Expected a 3-D or 4-D array, got shape {array.shape}.")

    if array.shape[1] == len(PRIMARY_VARS):  # (sample, channel, y, x)
        array = array.transpose(0, 2, 3, 1)
    elif array.shape[-1] != len(PRIMARY_VARS):
        raise ValueError(
            "Could not identify the channel axis in array with shape "
            f"{array.shape}; expected exactly {len(PRIMARY_VARS)} channels."
        )
    return np.asarray(array, dtype=np.float32)


def load_stack(path: pathlib.Path) -> np.ndarray:
    """Load NetCDF or NumPy data as (sample, y, x, channel)."""
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".nc":
        with xr.open_dataset(path, engine="h5netcdf") as dataset:
            missing = [name for name in PRIMARY_VARS if name not in dataset]
            if missing:
                raise KeyError(f"{path} is missing variables: {missing}")
            return np.stack(
                [dataset[name].values for name in PRIMARY_VARS], axis=-1
            ).astype(np.float32, copy=False)

    if path.is_file():
        return _to_channel_last(np.load(path))

    files = sorted(path.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy prediction files found under {path}")
    return _to_channel_last(np.stack([np.load(file) for file in files]))


def iter_stack_batches(
    path: pathlib.Path, batch_size: int = BATCH_SIZE
) -> Iterator[np.ndarray]:
    """Yield channel-last batches without materializing the full dataset."""
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".nc":
        with xr.open_dataset(path, engine="h5netcdf") as dataset:
            missing = [name for name in PRIMARY_VARS if name not in dataset]
            if missing:
                raise KeyError(f"{path} is missing variables: {missing}")
            n_samples = dataset.sizes["time"]
            for start in range(0, n_samples, batch_size):
                selection = dataset.isel(time=slice(start, start + batch_size))
                yield np.stack(
                    [selection[name].values for name in PRIMARY_VARS], axis=-1
                ).astype(np.float32, copy=False)
        return

    if path.is_file():
        stack = load_stack(path)
        for start in range(0, len(stack), batch_size):
            yield stack[start : start + batch_size]
        return

    files = sorted(path.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy prediction files found under {path}")
    for start in range(0, len(files), batch_size):
        yield _to_channel_last(
            np.stack([np.load(file) for file in files[start : start + batch_size]])
        )


def stack_shape(path: pathlib.Path) -> Tuple[int, int, int]:
    """Return (samples, y, x) using metadata and at most one sample read."""
    if path.suffix == ".nc":
        with xr.open_dataset(path, engine="h5netcdf") as dataset:
            return dataset.sizes["time"], dataset.sizes["y"], dataset.sizes["x"]

    if path.is_file():
        stack = load_stack(path)
        return stack.shape[:3]

    files = sorted(path.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy prediction files found under {path}")
    sample = _to_channel_last(np.load(files[0]))
    return len(files), sample.shape[1], sample.shape[2]


def extract_var(stack: np.ndarray, variable: str) -> np.ndarray:
    """Extract a primary field or calculate a grid-space derived field."""
    if variable in CHANNEL_MAP:
        return stack[..., CHANNEL_MAP[variable]]

    u10 = stack[..., CHANNEL_MAP["u10"]]
    v10 = stack[..., CHANNEL_MAP["v10"]]
    du_dx = np.gradient(u10, axis=2)
    du_dy = np.gradient(u10, axis=1)
    dv_dx = np.gradient(v10, axis=2)
    dv_dy = np.gradient(v10, axis=1)

    if variable == "vorticity":
        return dv_dx - du_dy
    if variable == "divergence":
        return du_dx + dv_dy
    if variable == "kinetic_energy":
        return 0.5 * (u10**2 + v10**2)
    raise KeyError(f"Unknown variable: {variable}")


def get_psd(
    data: np.ndarray, dx: float, axis: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the one-sided longitudinal PSD and wavenumber grid."""
    data = np.moveaxis(data, axis, -1)
    n_points = data.shape[-1]
    spectrum = np.fft.rfft(data, axis=-1) / n_points
    psd = 2.0 * np.abs(spectrum) ** 2
    wavenumber = np.fft.rfftfreq(n_points, d=dx)
    return wavenumber, psd


def psd_for_var(
    stack: np.ndarray, variable: str, dx: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean longitudinal PSD over samples and latitude rows."""
    wavenumber, psd = get_psd(extract_var(stack, variable), dx, axis=2)
    return wavenumber, psd.mean(axis=(0, 1))


def calculate_dataset_psds(
    path: pathlib.Path, dx: float
) -> Tuple[
    Dict[str, Tuple[np.ndarray, np.ndarray]], Tuple[int, int, int]
]:
    """Calculate all six mean PSDs incrementally for one dataset."""
    sums: Dict[str, np.ndarray] = {}
    wavenumbers: Dict[str, np.ndarray] = {}
    row_counts: Dict[str, int] = {}
    n_samples = 0
    spatial_shape = None

    for batch in iter_stack_batches(path):
        if spatial_shape is None:
            spatial_shape = batch.shape[1:3]
        elif batch.shape[1:3] != spatial_shape:
            raise ValueError(
                f"Inconsistent spatial shapes under {path}: "
                f"{batch.shape[1:3]} and {spatial_shape}."
            )

        n_samples += batch.shape[0]
        for variable in (*PRIMARY_VARS, *DERIVED_VARS):
            wavenumber, psd = get_psd(extract_var(batch, variable), dx, axis=2)
            batch_sum = psd.sum(axis=(0, 1))
            if variable not in sums:
                sums[variable] = batch_sum
                wavenumbers[variable] = wavenumber
                row_counts[variable] = batch.shape[0] * batch.shape[1]
            else:
                sums[variable] += batch_sum
                row_counts[variable] += batch.shape[0] * batch.shape[1]

    if spatial_shape is None:
        raise ValueError(f"No samples loaded from {path}")

    curves = {
        variable: (wavenumbers[variable], sums[variable] / row_counts[variable])
        for variable in (*PRIMARY_VARS, *DERIVED_VARS)
    }
    return curves, (n_samples, *spatial_shape)


def save_psd_csv(
    variable: str,
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> pathlib.Path:
    """Save the plotted CERRA and model curves in a single CSV file."""
    reference_k = curves["CERRA"][0]
    columns = [reference_k]
    names = ["wavenumber_cycles_per_degree"]

    for label, (wavenumber, psd) in curves.items():
        if not np.allclose(wavenumber, reference_k):
            raise ValueError(f"{label} has a different grid for {variable}.")
        names.append(label)
        columns.append(psd)

    output_path = OUT_DIR / f"{variable}_psd.csv"
    np.savetxt(
        output_path,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(names),
        comments="",
    )
    return output_path


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_psd_group(
    variables: Iterable[str],
    all_curves: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    output_stem: str,
) -> Tuple[pathlib.Path, pathlib.Path]:
    """Plot three variables as aligned small multiples with one legend."""
    variables = tuple(variables)
    if len(variables) != 3:
        raise ValueError("A 1 x 3 PSD figure requires exactly three variables.")

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), sharex=True)
    legend_handles = []
    legend_labels = []

    for panel_index, (axis, variable) in enumerate(zip(axes, variables)):
        for label, (wavenumber, psd) in all_curves[variable].items():
            # The zero-frequency/DC component cannot be displayed on a log axis.
            positive = wavenumber > 0
            style = SERIES_STYLE.get(label, {"linewidth": 1.8})
            (line,) = axis.loglog(
                wavenumber[positive], psd[positive], label=label, **style
            )
            if panel_index == 0:
                legend_handles.append(line)
                legend_labels.append(label)

        axis.set_title(VARIABLE_TITLES[variable], fontsize=11, pad=8)
        axis.set_xlabel(r"Wavenumber $k$ (cycles degree$^{-1}$)")
        axis.grid(True, which="major", color="#d9d9d9", linewidth=0.7)
        axis.grid(True, which="minor", color="#eeeeee", linewidth=0.45)
        axis.tick_params(labelsize=9)

    axes[0].set_ylabel("Power spectral density")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_labels),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.0)

    png_path = OUT_DIR / f"{output_stem}.png"
    pdf_path = OUT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
        }
    )

    cerra_shape = stack_shape(CERRA_PATH)
    era5_shape = stack_shape(ERA5_PATH)
    if cerra_shape[0] != era5_shape[0]:
        raise ValueError(
            f"CERRA has {cerra_shape[0]} samples but ERA5 has {era5_shape[0]}."
        )
    dx_cerra = ERA5_DX_DEG * era5_shape[2] / cerra_shape[2]
    print(
        f"Calculating CERRA PSDs in batches (shape={cerra_shape}, "
        f"dx={dx_cerra:.5f} degrees) ..."
    )
    cerra_curves, calculated_cerra_shape = calculate_dataset_psds(
        CERRA_PATH, dx_cerra
    )
    if calculated_cerra_shape != cerra_shape:
        raise ValueError("CERRA shape changed while it was being read.")

    dataset_curves = {"CERRA": cerra_curves}
    for label, path in MODEL_PATHS.items():
        print(f"Calculating {label} PSDs in batches ...")
        curves, prediction_shape = calculate_dataset_psds(path, dx_cerra)
        if prediction_shape != cerra_shape:
            raise ValueError(
                f"{label} shape {prediction_shape} does not match CERRA "
                f"shape {cerra_shape}."
            )
        dataset_curves[label] = curves

    all_curves: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for variable in (*PRIMARY_VARS, *DERIVED_VARS):
        curves = {
            label: variable_curves[variable]
            for label, variable_curves in dataset_curves.items()
        }
        all_curves[variable] = curves
        csv_path = save_psd_csv(variable, curves)
        print(f"Saved {csv_path}")

    primary_paths = plot_psd_group(
        PRIMARY_VARS, all_curves, "primary_variables_psd"
    )
    derived_paths = plot_psd_group(
        DERIVED_VARS, all_curves, "derived_variables_psd"
    )
    print(f"Saved {primary_paths[0]} and {primary_paths[1]}")
    print(f"Saved {derived_paths[0]} and {derived_paths[1]}")
    save_psd_csv(variable, curves)

    primary_paths = plot_psd_group(
        PRIMARY_VARS, all_curves, "primary_variables_psd"
    )
    derived_paths = plot_psd_group(
        DERIVED_VARS, all_curves, "derived_variables_psd"
    )

    print("Saved:")
    for path in (*primary_paths, *derived_paths):
        print(f"  {path.resolve()}")
    print(f"Per-variable CSV tables are in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
