from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import ERA5toCERRA2
from src.models.UNO import UNOWrapper
from src.models.fno_v1 import FNOWrapper
from src.models.unet import UNetWrapper

VARIABLE_NAMES = ["u10", "v10", "t2m"]
MODEL_CLASSES = {
    "UNet-CNN": UNetWrapper,
    "FNO": FNOWrapper,
    "UNO": UNOWrapper,
}

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate a continuous monthly bundle with independently evaluated "
            "UNet, FNO, or U-NO frames."
        )
    )
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
        help="Number of independent timestamps evaluated together.",
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
        "cadence": 3,
        "standardize": True,
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

    return argparse.Namespace(**config)


def select_monthly_frames(dataset, start, end, cadence_hours):

    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)

    if end_time <= start_time:
        raise ValueError(
            f"End time {end_time} must be after start time {start_time}."
        )

    cadence = pd.Timedelta(hours=float(cadence_hours))
    if cadence <= pd.Timedelta(0):
        raise ValueError(f"Cadence must be positive, got {cadence}.")
    requested_duration = end_time - start_time
    if requested_duration % cadence != pd.Timedelta(0):
        raise ValueError(
            f"Requested duration {requested_duration} is not a multiple of "
            f"the {cadence} cadence."
        )
    expected_time = pd.date_range(
        start=start_time,
        end=end_time,
        freq=cadence,
        inclusive="left",
    )
    available_indices = {
        pd.Timestamp(dataset.time[int(raw_index)]): int(raw_index)
        for raw_index in dataset.indices
    }
    missing = [
        timestamp
        for timestamp in expected_time
        if timestamp not in available_indices
    ]
    if missing:
        raise ValueError(
            "Some requested timestamps are not available in the dataset: "
            f"{missing[:10]}"
        )
    raw_indices = np.asarray(
        [available_indices[timestamp] for timestamp in expected_time],
        dtype=np.int64,
    )
    return raw_indices, expected_time.to_numpy(dtype="datetime64[ns]")


def assemble_month(
    dataset,
    raw_indices,
):
    original_indices = dataset.indices
    dataset.indices = raw_indices

    target_frames = []
    conditioning_frames = []
    era5_frames = []

    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            conditioning, target = sample[:2]
            target_frames.append(target.float())
            conditioning_frames.append(conditioning.float())
            era5_frames.append(
                conditioning[: len(dataset.era5_vars)].float()
            )

    finally:
        dataset.indices = original_indices

    return (
        torch.stack(target_frames).unsqueeze(0),
        torch.stack(conditioning_frames).unsqueeze(0),
        torch.stack(era5_frames).unsqueeze(0),
    )


def generate_monthly_prediction(
        model,
        conditioning_month,
        device,
        inference_batch_size,
):

    if conditioning_month.ndim != 5 or conditioning_month.shape[0] != 1:
        raise ValueError(
            "Monthly generator expects conditioning shape [1,T,C,H,W]."
        )

    model.eval()
    model.to(device)

    total_time = conditioning_month.shape[1]
    output_frames = []

    with torch.inference_mode():
        for offset in range(0, total_time, inference_batch_size):
            stop = min(offset + inference_batch_size, total_time)
            conditioning_batch = conditioning_month[0, offset:stop].to(device)
            target_shape = (
                conditioning_batch.shape[0],
                int(model.img_out_channels),
                *conditioning_batch.shape[-2:],
            )
            model_input = torch.zeros(
                target_shape,
                dtype=conditioning_batch.dtype,
                device=device,
            )
            prediction_batch = model(
                x=model_input,
                img_lr=conditioning_batch,
            )
            if prediction_batch.shape != target_shape:
                raise RuntimeError(
                    f"Model returned {tuple(prediction_batch.shape)}, "
                    f"expected {target_shape}."
                )
            output_frames.append(prediction_batch.detach().float().cpu())
            print(
                f"Predicted frames {offset + 1}-{stop}/{total_time}",
                flush=True,
            )

    return torch.cat(output_frames, dim=0).unsqueeze(0)


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

    model_name = getattr(config, "model", None)
    if model_name not in MODEL_CLASSES:
        raise ValueError(
            f"Unsupported model {model_name!r}. This generator supports only "
            f"{', '.join(MODEL_CLASSES)}."
        )

    print(f"Loading checkpoint: {cli.checkpoint}")
    print(f"Model: {model_name}")

    model = MODEL_CLASSES[model_name].load_from_checkpoint(
        cli.checkpoint,
        args=config,
        map_location=device,
    )

    dataset = ERA5toCERRA2(
        dataset_name_CERRA=config.dataset_cerra,
        dataset_name_ERA5=config.dataset_era5,
        split=cli.split,
        standardize=bool(config.standardize),
        region=cli.region,
        cerra_vars=VARIABLE_NAMES,
        era5_vars=VARIABLE_NAMES,
        n_samples=None,
    )

    try:
        raw_indices, all_times = select_monthly_frames(
            dataset,
            cli.start,
            cli.end,
            config.cadence,
        )

        cerra_normalized, conditioning_normalized, era5_normalized = assemble_month(
            dataset,
            raw_indices,
        )

        if cerra_normalized.shape[1] != len(all_times):
            raise RuntimeError(
                f"Expected {len(all_times)} time steps, "
                f"got {cerra_normalized.shape[1]}."
            )

        prediction_normalized = generate_monthly_prediction(
            model=model,
            conditioning_month=conditioning_normalized,
            device=device,
            inference_batch_size=cli.batch_size,
        )

        prediction_phy = unnormalize(
            prediction_normalized,
            mean=dataset.data_mean_CERRA,
            std=dataset.data_std_CERRA,
        )

        target_phy = unnormalize(
            cerra_normalized,
            mean=dataset.data_mean_CERRA,
            std=dataset.data_std_CERRA,
        )

        era5_phy = unnormalize(
            era5_normalized,
            mean=dataset.data_mean_era5,
            std=dataset.data_std_era5,
        )


    finally:
        dataset.close()

    bundle = {
        "pred_seq": prediction_phy.cpu(),
        "hr_seq": target_phy.cpu(),
        "lr_seq": era5_phy.cpu(),
        "times": all_times,
        "timestep_hours": float(config.cadence),
        "sequence_length": int(len(all_times)),
        "model": model_name,
        "deterministic": True,
        "normalized": False,
        "independent_frame_inference": True,
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
