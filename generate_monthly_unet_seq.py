from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset_seq import CerraPriorDatasetSequence
from src.models.unet_seq import UNetSequenceWrapper

VARIABLE_NAMES = ["u10", "v10", "t2m"]

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--region", default="CentralEurope")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of local temporal windows evaluated together.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='For example "cuda", "cuda:0", or "cpu".',
    )

    return parser.parse_args()


def load_config(path):

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    defaults = {
        "local_window": 3,
        "sequence_length": 8,
        "anneal_epochs": 200,
        "init_lambda": 0.0,
        "max_lambda": 0.0,
        "loss_type": None,
        "N_grid_channels": None,
        "savepreds_path": None,
        "wandb_project": None,
        "checkpoint_level": 0,
    }

    for key, value in defaults.items():
        config.setdefault(key, value)

    # Monthly generation must not restrict the dataset to one window.
    config["sequence_start_time"] = None

    return argparse.Namespace(**config)


def select_monthly_windows(dataset, start, end):

    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)

    if end_time < start_time:
        raise ValueError(f"End time {end_time} is before start time {start_time}.")

    cadence = dataset.cadence
    sequence_length = dataset.sequence_length

    window_duration = cadence*sequence_length
    requested_duration = end_time - start_time

    if requested_duration % window_duration != pd.Timedelta(0):
        raise ValueError(
            f"Requested duration {requested_duration} is not a multiple of "
            f"the window duration {window_duration}."
        )

    num_windows = int(requested_duration / window_duration)

    requested_start = [
        start_time + i * window_duration for i in range(num_windows)
    ]

    available_start = {
        pd.Timestamp(dataset.time[int(raw_index)]) : int(raw_index)
        for raw_index in dataset.indices
    }

    missing = [
        timestamp 
        for timestamp in requested_start
        if timestamp not in available_start
    ]

    if missing:
        raise ValueError(
            f"Some requested start times are not available in the dataset: "
            f"{missing[:10]}"
        )

    raw_start_idx = np.asarray(
        [
            available_start[timestamp] for timestamp in requested_start
        ],
        dtype=np.int64,
    )

    selected_time = []

    for raw_idx in raw_start_idx:
        window_times = dataset.time[
            raw_idx : raw_idx + sequence_length
        ]
        selected_time.append(
            np.asarray(window_times, dtype="datetime64[ns]")
            )

    selected_time = np.concatenate(selected_time)

    expected_time = pd.date_range(
        start=start_time,
        end=end_time,
        freq=cadence,
        inclusive="left",
    ).to_numpy(dtype="datetime64[ns]")

    if not np.array_equal(selected_time, expected_time):
        raise ValueError(
            f"Selected time {selected_time} does not match expected time "
            f"{expected_time}."
        )

    return raw_start_idx, selected_time


def restore_dataset_sequence(
    tensor,
    sequence_length,
    frame_channels,
):
    """
    Dataset sample:
        [T*C,H,W] -> [T,C,H,W]
    """

    if tensor.ndim == 4:
        return tensor

    if tensor.ndim != 3:
        raise ValueError(
            f"Unexpected dataset tensor shape {tuple(tensor.shape)}."
        )

    expected_channels = sequence_length * frame_channels

    if tensor.shape[0] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} flattened channels, "
            f"got {tensor.shape[0]}."
        )

    return tensor.unflatten(
        dim=0,
        sizes=(sequence_length, frame_channels),
    )


def assemble_month(
    dataset,
    raw_start_indices,
):
    original_indices = dataset.indices
    dataset.indices = raw_start_indices

    cerra_windows = []
    era5_windows = []
    orography = None

    try:
        for index in range(len(dataset)):
            cerra, current_orography, era5 = dataset[index]

            cerra = restore_dataset_sequence(
                cerra,
                dataset.sequence_length,
                len(dataset.cerra_vars),
            ).float()

            era5 = restore_dataset_sequence(
                era5,
                dataset.sequence_length,
                len(dataset.era5_vars),
            ).float()

            cerra_windows.append(cerra)
            era5_windows.append(era5)

            if orography is None:
                orography = current_orography.float()
            elif not torch.equal(orography, current_orography.float()):
                raise ValueError(
                    "Orography differs between temporal windows."
                )

    finally:
        dataset.indices = original_indices

    # Add one batch dimension.
    cerra_month = torch.cat(
        cerra_windows,
        dim=0,
    ).unsqueeze(0)

    era5_month = torch.cat(
        era5_windows,
        dim=0,
    ).unsqueeze(0)

    orography = orography.unsqueeze(0)

    return cerra_month, orography, era5_month


def generate_continuous_prediction(
        model,
        era5_month,
        orography,
        device,
        inference_batch_size,
):

    if era5_month.ndim != 5 or era5_month.shape[0] != 1:
        raise ValueError(
        "Monthly generator currently expects ERA5 shape "
        "[1,T,C,h,w]."
    )

    model.eval()
    model.to(device)

    total_time = era5_month.shape[1]
    frame_channels = era5_month.shape[2]
    local_window = int(model.local_window)

    if local_window % 2 == 0:
        raise ValueError("local_window must be odd.")

    if total_time < local_window:
        raise ValueError(
            f"T={total_time} is shorter than "
            f"local_window={local_window}."
        )

    half_window = local_window // 2

    frame_requests = {}

    for temporal_index in range(total_time):
        start = min(
            max(temporal_index - half_window, 0),
            total_time - local_window,
        )

        local_index = temporal_index - start

        frame_requests.setdefault(start, []).append(
            (temporal_index, local_index)
        )

    unique_starts = sorted(frame_requests)
    output_frames = [None] * total_time

    for offset in range(
        0,
        len(unique_starts),
        inference_batch_size,
    ):
        batch_starts = unique_starts[
            offset : offset + inference_batch_size
        ]

        era5_batch = torch.stack(
            [
                era5_month[
                    0,
                    start : start + local_window,
                ]
                for start in batch_starts
            ],
            dim=0,
        ).to(device)

        orography_batch = orography.expand(
            len(batch_starts),
            -1,
            -1,
            -1,
        ).to(device)

        prediction_batch = model(
            guidance=era5_batch,
            orography=orography_batch,
        )

        # Compatibility in case forward() returns flattened local channels.
        if prediction_batch.ndim == 4:
            prediction_batch = prediction_batch.unflatten(
                dim=1,
                sizes=(local_window, frame_channels),
            )

        if prediction_batch.ndim != 5:
            raise RuntimeError(
                "UNet prediction must have shape [B,W,C,H,W]."
            )

        for batch_index, start in enumerate(batch_starts):
            for temporal_index, local_index in frame_requests[start]:
                output_frames[temporal_index] = (
                    prediction_batch[
                        batch_index,
                        local_index,
                    ]
                    .detach()
                    .float()
                    .cpu()
                )

        print(
            f"Predicted local windows "
            f"{offset + 1}-"
            f"{min(offset + inference_batch_size, len(unique_starts))}"
            f"/{len(unique_starts)}",
            flush=True,
        )

    missing = [
        index
        for index, value in enumerate(output_frames)
        if value is None
    ]

    if missing:
        raise RuntimeError(
            f"Missing output frames: {missing}"
        )

    return torch.stack(
        output_frames,
        dim=0,
    ).unsqueeze(0)


def unnormalize(sequence, mean, std):

    mean = torch.as_tensor(mean, dtype=sequence.dtype).reshape(1, 1, -1, 1, 1)

    std = torch.as_tensor(std, dtype=sequence.dtype).reshape(1, 1, -1, 1, 1)

    return sequence * std + mean

def temporal_correlation_map(
    prediction,
    target,
    eps=1e-8,
):
    """
    prediction and target: [T,C,H,W]
    returns: [C,H,W]
    """

    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: {prediction.shape} != {target.shape}"
        )

    channel_maps = []

    # Work one variable at a time to reduce peak CPU memory.
    for channel in range(prediction.shape[1]):
        x = prediction[:, channel].double()
        y = target[:, channel].double()

        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

        numerator = (x * y).sum(dim=0)

        denominator = torch.sqrt(
            x.square().sum(dim=0)
            * y.square().sum(dim=0)
        )

        correlation = (
            numerator / denominator.clamp_min(eps)
        )

        correlation[denominator <= eps] = torch.nan
        channel_maps.append(correlation.float())

    return torch.stack(channel_maps, dim=0)


def summarize_correlation(correlation):
    summary = {}

    for channel, name in enumerate(VARIABLE_NAMES):
        values = correlation[channel]
        values = values[torch.isfinite(values)]

        summary[name] = {
            "mean": (
                float(values.mean())
                if values.numel()
                else None
            ),
            "median": (
                float(values.median())
                if values.numel()
                else None
            ),
            "q05": (
                float(torch.quantile(values, 0.05))
                if values.numel()
                else None
            ),
            "q95": (
                float(torch.quantile(values, 0.95))
                if values.numel()
                else None
            ),
            "valid_grid_points": int(values.numel()),
        }

    return summary


def compute_temporal_metrics(
    prediction,
    target,
):
    """
    prediction and target: [1,T,C,H,W]
    """

    prediction = prediction[0].float()
    target = target[0].float()

    raw_map = temporal_correlation_map(
        prediction,
        target,
    )

    prediction_increment = (
        prediction[1:] - prediction[:-1]
    )
    target_increment = (
        target[1:] - target[:-1]
    )

    increment_map = temporal_correlation_map(
        prediction_increment,
        target_increment,
    )

    rmse = torch.sqrt(
        (prediction.double() - target.double())
        .square()
        .mean(dim=(0, 2, 3))
    ).float()

    increment_rmse = torch.sqrt(
        (
            prediction_increment.double()
            - target_increment.double()
        )
        .square()
        .mean(dim=(0, 2, 3))
    ).float()

    return {
        "raw_correlation_map": raw_map,
        "increment_correlation_map": increment_map,
        "raw_summary": summarize_correlation(raw_map),
        "increment_summary": summarize_correlation(
            increment_map
        ),
        "rmse_per_variable": {
            name: float(rmse[index])
            for index, name in enumerate(VARIABLE_NAMES)
        },
        "increment_rmse_per_variable": {
            name: float(increment_rmse[index])
            for index, name in enumerate(VARIABLE_NAMES)
        },
        "number_of_frames": int(prediction.shape[0]),
        "number_of_increments": int(
            prediction_increment.shape[0]
        ),
    }


def main():
    
    cli = parse_args()
    config = load_config(cli.config)

    config.load = cli.checkpoint

    if cli.batch_size < 1:
        raise ValueError(
            f"Batch size must be positive, got {cli.batch_size}."
        )

    device = torch.device(
        cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Loading checkpoint: {cli.checkpoint}")

    model = UNetSequenceWrapper.load_from_checkpoint(
        cli.checkpoint,
        args=config,
        map_location=device,
    )

    dataset = CerraPriorDatasetSequence(
        root_dir_cerra=config.dataset_cerra,
        root_dir_era5=config.dataset_era5,
        split=cli.split,
        region=cli.region,
        cerra_vars=VARIABLE_NAMES,
        era5_vars=VARIABLE_NAMES,
        sequence_length=int(config.sequence_length),
        sequence_stride=1,
        cadence=int(config.cadence),
        flatten=True,
        n_samples=None,
        sequence_start_time=None,
    )

    try:
        raw_start_idx, all_times = select_monthly_windows(
            dataset,
            cli.start,
            cli.end,
        )

        cerra_normalized, orography, era5_normalized = assemble_month(
            dataset,
            raw_start_idx,
        )

        if cerra_normalized.shape[1] != len(all_times):
            raise RuntimeError(
                f"Expected {len(all_times)} time steps, "
                f"got {cerra_normalized.shape[1]}."
            )

        prediction_normalized = generate_continuous_prediction(
            model=model,
            era5_month=era5_normalized,
            orography=orography,
            device=device,
            inference_batch_size=cli.batch_size,
        )

        prediction_phy = unnormalize(
            prediction_normalized,
            mean=dataset.cerra_mean,
            std=dataset.cerra_std,
        )

        target_phy = unnormalize(
            cerra_normalized,
            mean=dataset.cerra_mean,
            std=dataset.cerra_std,
        )

        era5_phy = unnormalize(
            era5_normalized,
            mean=dataset.era5_mean,
            std=dataset.era5_std,
        )


    finally:
        dataset.close()

    bundle = {
        "pred_seq": prediction_phy.cpu(),
        "hr_seq": target_phy.cpu(),
        "lr_seq": era5_phy.cpu(),
        "times": all_times,
        "timestep_hours": float(config.cadence),
        "sequence_length": int(config.sequence_length),
        "local_window": int(model.local_window),
        "model": "UNet-Local-Seq",
        "deterministic": True,
        "normalized": False,
        "continuous_sliding_inference": True,
    }

    output_path = Path(cli.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(bundle, output_path)

    print(f"Saved sequence bundle: {output_path}")
    print(f"pred_seq: {tuple(bundle['pred_seq'].shape)}")
    print(f"hr_seq:   {tuple(bundle['hr_seq'].shape)}")
    print(f"lr_seq:   {tuple(bundle['lr_seq'].shape)}")

    temporal_metrics = compute_temporal_metrics(
        prediction_phy,
        target_phy,
    )


    metrics_path = (
        Path(cli.metrics_output)
        if cli.metrics_output
        else output_path.with_name(
            f"{output_path.stem}_metrics.pt"
        )
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(temporal_metrics, metrics_path)

    print(f"Saved metrics: {metrics_path}")
    print("Raw temporal correlation:")
    print(
        json.dumps(
            temporal_metrics["raw_summary"],
            indent=2,
        )
    )
    print("Increment correlation:")
    print(
        json.dumps(
            temporal_metrics["increment_summary"],
            indent=2,
        )
    )
    print("RMSE:")
    print(
        json.dumps(
            temporal_metrics["rmse_per_variable"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()