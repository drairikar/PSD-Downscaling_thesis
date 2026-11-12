"""Verify the baseline against known physical-unit fields and uneven batches."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ERA5toCERRA2 = load_module("bicubic_dataset", "src/data/dataset.py").ERA5toCERRA2
BicubicWrapper = load_module("bicubic_model", "src/models/bicubic.py").BicubicWrapper
ExperimentConfig = load_module("bicubic_config", "src/config/base_config.py").ExperimentConfig


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def paired_fields(tmp_path):
    era5_root, cerra_root = tmp_path / "ERA5", tmp_path / "CERRA"
    stats = era5_root / "statistics"
    stats.mkdir(parents=True)
    np.save(stats / "dynamic_mean.npy", np.array([1, 2, 275, 0, 0, 0], dtype=np.float32))
    np.save(stats / "dynamic_std.npy", np.array([3, 4, 5, 1, 1, 1], dtype=np.float32))
    np.save(stats / "forcing_mean.npy", np.array([10], dtype=np.float32))
    np.save(stats / "forcing_std.npy", np.array([2], dtype=np.float32))
    yy, xx = np.mgrid[:8, :6].astype(np.float32)
    native = np.stack([xx + yy, xx - yy, 275 + yy + xx * 0.5])
    native = np.stack([native + t for t in range(3)])
    # Known bicubic prediction and spatially uniform, time-varying errors.
    prediction = F.interpolate(torch.from_numpy(native), size=(32, 24), mode="bicubic", align_corners=False).numpy()
    offsets = np.array([1, 2, 3], dtype=np.float32)[:, None, None, None] * np.array([1, 2, 4], dtype=np.float32)[None, :, None, None]
    target = prediction + offsets
    times = np.array(["2021-01-01T00", "2021-01-01T03", "2021-01-01T06"], dtype="datetime64[ns]")
    for root, data in ((era5_root, native), (cerra_root, target)):
        (root / "test").mkdir(parents=True)
        xr.Dataset(
            {name: (("time", "y", "x"), data[:, index]) for index, name in enumerate(("u10", "v10", "t2m"))},
            coords={"time": times},
        ).to_netcdf(root / "test/CentralEurope.nc", engine="h5netcdf")
    xr.Dataset({"orog": (("y", "x"), np.full((32, 24), 1000, dtype=np.float32))}).to_netcdf(
        cerra_root / "test/static_CentralEurope.nc", engine="h5netcdf"
    )
    return cerra_root, era5_root, prediction, offsets


def evaluate(dataset, output, batch_size):
    model = BicubicWrapper(SimpleNamespace(img_in_channels=4, img_out_channels=3, savepreds_path=str(output)))
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    metrics = trainer.test(model, DataLoader(dataset, batch_size=batch_size), verbose=False)[0]
    return model, metrics


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("resolution", [None, [16, 12]])
def test_physical_predictions_and_batch_independent_metrics(paired_fields, tmp_path, standardize, resolution):
    cerra, era5, expected, offsets = paired_fields
    dataset = ERA5toCERRA2(str(cerra), str(era5), split="test", standardize=standardize, img_resolution=resolution)
    try:
        model, metrics = evaluate(dataset, tmp_path / "batched", batch_size=2)
        _, single_metrics = evaluate(dataset, tmp_path / "single", batch_size=1)
        assert not list(model.parameters())
        assert metrics == pytest.approx(single_metrics, abs=1e-7)
        if resolution:
            expected = expected.reshape(3, 3, 16, 2, 12, 2).mean((3, 5))
        files = sorted((tmp_path / "batched/Bicubic/files").glob("nwp_*.npy"))
        assert len(files) == 3
        np.testing.assert_allclose(np.stack([np.load(path) for path in files]), expected, atol=4e-5)
        for index, name in enumerate(model.variable_names):
            errors = offsets[:, index].astype(np.float64)
            assert metrics[f"test_mse_{name}"] == pytest.approx(np.mean(errors ** 2), abs=3e-4)
            assert metrics[f"test_mae_{name}"] == pytest.approx(np.mean(errors), abs=3e-5)
            assert metrics[f"test_rmse_{name}"] == pytest.approx(np.sqrt(np.mean(errors ** 2)), abs=3e-5)
            assert np.isfinite(metrics[f"test_ssim_{name}"])
        report = json.loads((tmp_path / "batched/Bicubic/metrics.json").read_text())
        assert report["num_samples"] == 3
        assert report["metrics"] == pytest.approx(metrics)
    finally:
        dataset.close()


def test_distinct_guidance_and_target_stats_and_constant_ssim(tmp_path):
    model = BicubicWrapper(SimpleNamespace(img_in_channels=4, img_out_channels=3, savepreds_path=str(tmp_path)))
    # Both standardized tensors are zero, but physical predictions are 6 and
    # targets are 4. This detects accidentally using target stats for ERA5.
    conditioning = torch.zeros(1, 4, 16, 16)
    target = torch.zeros(1, 3, 16, 16)
    stats = {
        "mean_era5": torch.full((1, 4, 1, 1), 6.0),
        "std_era5": torch.ones(1, 4, 1, 1),
        "mean_CERRA": torch.full((1, 3, 1, 1), 4.0),
        "std_CERRA": torch.ones(1, 3, 1, 1),
    }
    model.test_step((conditioning, target, stats, ["example"]), 0)
    torch.testing.assert_close(model.squared_error_sum / model.pixel_count, torch.full((3,), 4.0, dtype=torch.float64))
    assert torch.isfinite(model.ssim_sum).all()
    np.testing.assert_array_equal(np.load(tmp_path / "Bicubic/files/nwp_example.npy"), np.full((3, 16, 16), 6.0))


def test_forward_excludes_orography_and_supports_wrapper_signature():
    model = BicubicWrapper(SimpleNamespace(img_in_channels=4, img_out_channels=3, savepreds_path=None))
    conditioning = torch.randn(2, 4, 16, 12)
    direct = model(conditioning)
    conditioning[:, 3] = 1e8
    torch.testing.assert_close(model(torch.zeros_like(direct), img_lr=conditioning), direct)
    with pytest.raises(ValueError, match="conditioning shape"):
        model(torch.zeros(2, 3, 16, 12))


@pytest.mark.parametrize("extra", [{"eval": None}, {"eval": "val"}, {"load": "unused.ckpt"}, {"restore_opt": 1}, {"img_out_channels": 4}])
def test_invalid_baseline_config_is_rejected(tmp_path, extra):
    config = dict(model="Bicubic", eval="test", dataset_cerra=str(tmp_path), dataset_era5=str(tmp_path), img_in_channels=4, img_out_channels=3)
    with pytest.raises(ValueError, match="Bicubic"):
        ExperimentConfig.from_dict({**config, **extra})


def test_valid_baseline_config(tmp_path):
    config = ExperimentConfig.from_dict(dict(model="Bicubic", eval="test", dataset_cerra=str(tmp_path), dataset_era5=str(tmp_path), img_in_channels=4, img_out_channels=3))
    assert config.model.model_type == "Bicubic"
    assert config.load is None
