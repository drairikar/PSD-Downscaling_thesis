"""Checkpoint-free ERA5 bicubic baseline for the single-frame test dataset."""

import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torchmetrics.functional import structural_similarity_index_measure


class BicubicWrapper(pl.LightningModule):
    """Evaluate the weather channels already interpolated by ERA5toCERRA2.

    The dataset resizes native ERA5 to native CERRA with bicubic interpolation
    (align_corners=False), optionally block-averages both fields, and then
    standardizes them. No second interpolation or terrain correction is used.
    """

    variable_names = ("u10", "v10", "t2m")

    def __init__(self, args):
        super().__init__()
        if (args.img_in_channels, args.img_out_channels) != (4, 3):
            raise ValueError("Bicubic expects ERA5 u10/v10/t2m plus orography, and 3 outputs.")
        self.savepreds_path = args.savepreds_path
        # Float64 sums give global pixel-weighted errors, including when the
        # last batch is smaller than the others.
        self.register_buffer("squared_error_sum", torch.zeros(3, dtype=torch.float64))
        self.register_buffer("absolute_error_sum", torch.zeros(3, dtype=torch.float64))
        self.register_buffer("ssim_sum", torch.zeros(3, dtype=torch.float64))
        self.register_buffer("pixel_count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("sample_count", torch.zeros((), dtype=torch.float64))

    def forward(self, x, img_lr=None, **kwargs):
        """Accept conditioning directly or the deterministic wrappers' signature."""
        conditioning = x if img_lr is None else img_lr
        if conditioning.ndim != 4 or conditioning.shape[1] != 4:
            raise ValueError("Expected conditioning shape [batch, 4, height, width].")
        return conditioning[:, :3].float()

    def on_test_epoch_start(self):
        for value in (
            self.squared_error_sum, self.absolute_error_sum, self.ssim_sum,
            self.pixel_count, self.sample_count,
        ):
            value.zero_()

    def test_step(self, batch, batch_idx):
        conditioning, target, statistics, timestamps = batch
        # Use guidance statistics for ERA5 and target statistics for CERRA.
        # The dataset supplies identity statistics when standardize=False.
        prediction = (
            self(conditioning) * statistics["std_era5"][:, :3]
            + statistics["mean_era5"][:, :3]
        )
        target = target.float() * statistics["std_CERRA"] + statistics["mean_CERRA"]
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction {prediction.shape} and target {target.shape} differ.")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("Bicubic evaluation requires finite ERA5 and CERRA fields.")

        error = prediction.double() - target.double()
        self.squared_error_sum.add_(error.square().sum(dim=(0, 2, 3)))
        self.absolute_error_sum.add_(error.abs().sum(dim=(0, 2, 3)))
        self.pixel_count.add_(target.shape[0] * target.shape[2] * target.shape[3])
        self.sample_count.add_(target.shape[0])

        # SSIM per image and variable is independent of batching. A range
        # floor also handles spatially constant targets.
        for index in range(3):
            for sample in range(target.shape[0]):
                truth = target[sample:sample + 1, index:index + 1]
                pred = prediction[sample:sample + 1, index:index + 1]
                data_range = (truth.max() - truth.min()).clamp_min(1e-6).item()
                score = structural_similarity_index_measure(pred, truth, data_range=data_range)
                self.ssim_sum[index].add_(score.double())

        if self.savepreds_path:
            output = Path(self.savepreds_path) / "Bicubic" / "files"
            output.mkdir(parents=True, exist_ok=True)
            for timestamp, frame in zip(timestamps, prediction.detach().cpu().numpy()):
                np.save(output / f"nwp_{timestamp}.npy", frame)

    def on_test_epoch_end(self):
        if self.sample_count.item() == 0:
            raise ValueError("Cannot evaluate an empty dataset.")
        mse = self.squared_error_sum / self.pixel_count
        mae = self.absolute_error_sum / self.pixel_count
        ssim = self.ssim_sum / self.sample_count
        metrics = {}
        for index, name in enumerate(self.variable_names):
            metrics[f"test_mse_{name}"] = mse[index]
            metrics[f"test_mae_{name}"] = mae[index]
            metrics[f"test_rmse_{name}"] = mse[index].sqrt()
            metrics[f"test_ssim_{name}"] = ssim[index]
        self.log_dict(metrics)

        if self.savepreds_path:
            output = Path(self.savepreds_path) / "Bicubic"
            output.mkdir(parents=True, exist_ok=True)
            report = {
                "model": "Bicubic",
                "region": getattr(self, "current_region", "unknown"),
                "num_samples": int(self.sample_count.item()),
                "variables": list(self.variable_names),
                "metrics": {key: value.item() for key, value in metrics.items()},
            }
            with (output / "metrics.json").open("w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
