"""Build a same-timestamp model-comparison figure without model inference.

The defaults reproduce the u10 example stored as ``u10_sample_1.png``:
2021-01-01 04:00 UTC, where u10 is channel zero.  FNO, U-NO, and the
deterministic U-Net are read from saved physical-unit outputs.  Vanilla and
Sequential DPS are read from their saved eight-member ensemble-mean bundles
and converted from normalized values to physical units.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

# Keep plotting caches on local writable storage on the cluster.
_CACHE_ROOT = Path(tempfile.gettempdir()) / "thesis-prediction-figure-cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parent
GEODIFF_ROOT = PROJECT_ROOT.parent / "GeoDiff_Devashish"
DATA_ROOT = Path("/projects/0/prjs0951/Carlo/zz_processed_data_new")

CHANNELS = {"u10": 0, "v10": 1, "t2m": 2}
UNITS = {"u10": "m/s", "v10": "m/s", "t2m": "K"}

DEFAULT_FNO_DIR = (
    PROJECT_ROOT
    / "inference_outputs/FNO-testv2-FNO-09_10_10-6674/files"
)
DEFAULT_UNO_DIR = (
    PROJECT_ROOT
    / "inference_outputs/UNO_final-UNO-09_12_23-2417/files"
)
DEFAULT_UNET_BUNDLE = PROJECT_ROOT / "outputs_unet_seq/january_2021.pt"
DEFAULT_VANILLA_BUNDLE = (
    GEODIFF_ROOT / "outputs_vanilla/january_2021_e8.pt"
)
DEFAULT_SEQUENTIAL_BUNDLE = (
    GEODIFF_ROOT / "outputs_sequence/january_2021_e8_v2.pt"
)
DEFAULT_STATS_DIR = DATA_ROOT / "ERA5/statistics"
DEFAULT_CERRA = DATA_ROOT / "CERRA/test/CentralEurope.nc"
DEFAULT_ERA5 = DATA_ROOT / "ERA5/test/CentralEurope.nc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot saved same-timestamp predictions without inference."
    )
    parser.add_argument("--timestamp", default="2021-01-01T04:00:00")
    parser.add_argument("--variable", choices=CHANNELS, default="u10")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "thesis_prediction_figures",
    )
    parser.add_argument("--fno-dir", type=Path, default=DEFAULT_FNO_DIR)
    parser.add_argument("--uno-dir", type=Path, default=DEFAULT_UNO_DIR)
    parser.add_argument(
        "--unet-bundle", type=Path, default=DEFAULT_UNET_BUNDLE
    )
    parser.add_argument(
        "--vanilla-bundle", type=Path, default=DEFAULT_VANILLA_BUNDLE
    )
    parser.add_argument(
        "--sequential-bundle", type=Path, default=DEFAULT_SEQUENTIAL_BUNDLE
    )
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--cerra", type=Path, default=DEFAULT_CERRA)
    parser.add_argument("--era5", type=Path, default=DEFAULT_ERA5)
    return parser.parse_args()


def load_bundle(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    # These are locally generated, trusted bundles. mmap avoids loading the
    # multi-gigabyte DPS ensemble tensors into RAM.
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )


def time_index(bundle: dict, timestamp: np.datetime64, label: str) -> int:
    times = np.asarray(bundle["times"]).astype("datetime64[ns]")
    matches = np.flatnonzero(times == timestamp.astype("datetime64[ns]"))
    if matches.size != 1:
        raise ValueError(
            f"Expected one {label} frame at {timestamp}, found {matches.size}."
        )
    return int(matches[0])


def load_npy_prediction(directory: Path, timestamp: np.datetime64) -> np.ndarray:
    stamp = np.datetime_as_string(timestamp, unit="s").replace("-", "").replace(
        ":", ""
    )
    path = directory / f"nwp_{stamp}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    prediction = np.load(path, mmap_mode="r")
    if prediction.ndim != 3 or prediction.shape[0] != len(CHANNELS):
        raise ValueError(f"Unexpected prediction shape in {path}: {prediction.shape}")
    return prediction


def load_data_field(path: Path, variable: str, timestamp: np.datetime64) -> np.ndarray:
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        field = dataset[variable].sel(time=timestamp).values.astype(np.float32)
    if field.ndim != 2:
        raise ValueError(f"Expected a 2D {variable} field in {path}, got {field.shape}.")
    return field


def physical_dps_frame(
    bundle: dict,
    timestamp: np.datetime64,
    channel: int,
    mean: np.ndarray,
    std: np.ndarray,
    label: str,
) -> np.ndarray:
    index = time_index(bundle, timestamp, label)
    frame = bundle["pred_seq"][0, index, channel].float().numpy()
    if bool(bundle.get("normalized", False)):
        frame = frame * std[channel] + mean[channel]
    return frame


def physical_unet_frame(
    bundle: dict,
    timestamp: np.datetime64,
    channel: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    index = time_index(bundle, timestamp, "U-Net")
    frame = bundle["pred_seq"][0, index, channel].float().numpy()
    if bool(bundle.get("normalized", False)):
        frame = frame * std[channel] + mean[channel]
    return frame


def save_individual(field: np.ndarray, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(3.2, 3.2))
    axis.imshow(field, cmap="plasma", origin="lower")
    axis.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_comparison(
    fields: list[tuple[str, np.ndarray]],
    path_stem: Path,
    variable: str,
    timestamp: np.datetime64,
    shared_scale: bool,
) -> None:
    figure, axes = plt.subplots(1, len(fields), figsize=(15.4, 2.8))

    plot_kwargs: dict[str, object] = {"cmap": "plasma", "origin": "lower"}
    if shared_scale:
        finite_values = np.concatenate(
            [field[np.isfinite(field)].reshape(-1) for _, field in fields]
        )
        plot_kwargs.update(
            vmin=float(finite_values.min()),
            vmax=float(finite_values.max()),
        )

    image = None
    for panel, (axis, (title, field)) in enumerate(zip(axes, fields)):
        image = axis.imshow(field, **plot_kwargs)
        axis.set_title(f"({chr(97 + panel)}) {title}", fontsize=10)
        axis.axis("off")

    timestamp_text = np.datetime_as_string(timestamp, unit="m").replace("T", " ")
    figure.suptitle(
        f"{variable} ({UNITS[variable]}), {timestamp_text} UTC",
        fontsize=11,
        y=0.98,
    )
    if shared_scale and image is not None:
        colorbar = figure.colorbar(
            image,
            ax=axes,
            orientation="horizontal",
            fraction=0.07,
            pad=0.06,
            aspect=55,
        )
        colorbar.set_label(UNITS[variable])

    figure.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.85,
        bottom=0.20 if shared_scale else 0.04,
        wspace=0.04,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(
            path_stem.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    timestamp = np.datetime64(args.timestamp, "ns")
    variable = args.variable
    channel = CHANNELS[variable]

    dynamic_mean = np.load(args.stats_dir / "dynamic_mean.npy")[: len(CHANNELS)]
    dynamic_std = np.load(args.stats_dir / "dynamic_std.npy")[: len(CHANNELS)]

    unet_bundle = load_bundle(args.unet_bundle)
    vanilla_bundle = load_bundle(args.vanilla_bundle)
    sequential_bundle = load_bundle(args.sequential_bundle)

    fields = [
        ("ERA5 input", load_data_field(args.era5, variable, timestamp)),
        ("CERRA target", load_data_field(args.cerra, variable, timestamp)),
        (
            "U-Net",
            physical_unet_frame(
                unet_bundle, timestamp, channel, dynamic_mean, dynamic_std
            ),
        ),
        ("FNO", np.asarray(load_npy_prediction(args.fno_dir, timestamp)[channel])),
        ("U-NO", np.asarray(load_npy_prediction(args.uno_dir, timestamp)[channel])),
        (
            "Vanilla DPS",
            physical_dps_frame(
                vanilla_bundle,
                timestamp,
                channel,
                dynamic_mean,
                dynamic_std,
                "Vanilla DPS",
            ),
        ),
        (
            "Sequential DPS",
            physical_dps_frame(
                sequential_bundle,
                timestamp,
                channel,
                dynamic_mean,
                dynamic_std,
                "Sequential DPS",
            ),
        ),
    ]

    high_resolution_shape = fields[1][1].shape
    for label, field in fields[2:]:
        if field.shape != high_resolution_shape:
            raise ValueError(
                f"{label} shape {field.shape} does not match CERRA "
                f"shape {high_resolution_shape}."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = np.datetime_as_string(timestamp, unit="s").replace("-", "").replace(
        ":", ""
    )
    prefix = f"{variable}_{stamp}"

    individual_names = (
        "era5",
        "cerra",
        "unet",
        "fno",
        "uno",
        "vanilla_dps",
        "sequential_dps",
    )
    for name, (_, field) in zip(individual_names, fields):
        save_individual(field, args.output_dir / f"{prefix}_{name}.png")
    save_comparison(
        fields,
        args.output_dir / f"{prefix}_comparison_autoscale",
        variable,
        timestamp,
        shared_scale=False,
    )
    save_comparison(
        fields,
        args.output_dir / f"{prefix}_comparison_shared_scale",
        variable,
        timestamp,
        shared_scale=True,
    )

    print(f"Timestamp index: {time_index(sequential_bundle, timestamp, 'sequence')}")
    print(f"Variable channel: {channel}")
    print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
