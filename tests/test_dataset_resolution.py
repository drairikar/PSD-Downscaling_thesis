"""Check resolution changes without importing optional model dependencies."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr


spec = importlib.util.spec_from_file_location(
    "resolution_dataset", Path(__file__).resolve().parents[1] / "src/data/dataset.py"
)
dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_module)
ERA5toCERRA2 = dataset_module.ERA5toCERRA2


@pytest.fixture
def make_dataset(tmp_path):
    cerra_root = tmp_path / "CERRA"
    era5_root = tmp_path / "ERA5"
    stats = era5_root / "statistics"
    stats.mkdir(parents=True)
    np.save(stats / "dynamic_mean.npy", np.arange(6, dtype=np.float32))
    np.save(stats / "dynamic_std.npy", np.arange(1, 7, dtype=np.float32))
    np.save(stats / "forcing_mean.npy", np.array([10], dtype=np.float32))
    np.save(stats / "forcing_std.npy", np.array([2], dtype=np.float32))

    for split in ("train", "val", "test"):
        for root, shape in ((cerra_root, (8, 6)), (era5_root, (4, 3))):
            (root / split).mkdir(parents=True, exist_ok=True)
            frame = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            xr.Dataset(
                {
                    name: (("time", "y", "x"), (frame + index)[None])
                    for index, name in enumerate(("u10", "v10", "t2m"))
                },
                coords={"time": np.array(["2021-01-01"], dtype="datetime64[ns]")},
            ).to_netcdf(root / split / "CentralEurope.nc", engine="h5netcdf")
        xr.Dataset(
            {"orog": (("y", "x"), np.arange(48, dtype=np.float32).reshape(8, 6))}
        ).to_netcdf(cerra_root / split / "static_CentralEurope.nc", engine="h5netcdf")

    datasets = []

    def create(**kwargs):
        dataset = ERA5toCERRA2(str(cerra_root), str(era5_root), **kwargs)
        datasets.append(dataset)
        return dataset

    yield create
    for dataset in datasets:
        dataset.close()


def block_mean(frame):
    """Independent reference using NumPy rather than the pooling operation."""
    channels, height, width = frame.shape
    return frame.numpy().reshape(channels, height // 2, 2, width // 2, 2).mean((2, 4))


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("standardize", [False, True])
def test_full_region_averages_and_return_contract(make_dataset, split, standardize):
    native = make_dataset(split=split, standardize=standardize)
    coarse = make_dataset(split=split, standardize=standardize, img_resolution=[4, 3])
    native_sample, coarse_sample = native[0], coarse[0]

    assert coarse_sample[0].shape == (4, 4, 3)
    assert coarse_sample[1].shape == (3, 4, 3)
    for native_frame, coarse_frame in zip(native_sample[:2], coarse_sample[:2]):
        np.testing.assert_allclose(coarse_frame.numpy(), block_mean(native_frame), atol=1e-5)
        torch.testing.assert_close(coarse_frame.mean((-2, -1)), native_frame.mean((-2, -1)))

    assert len(coarse_sample) == (4 if split == "test" else 2)
    if split == "test":
        assert coarse_sample[3] == native_sample[3] == "20210101T000000"
        for key in native_sample[2]:
            torch.testing.assert_close(coarse_sample[2][key], native_sample[2][key])
        stats = coarse_sample[2]
        restored = coarse_sample[1] * stats["std_CERRA"] + stats["mean_CERRA"]
        raw = make_dataset(split="test", standardize=False)[0][1]
        np.testing.assert_allclose(restored.numpy(), block_mean(raw), atol=1e-5)


def test_native_resolution_is_unchanged(make_dataset):
    default = make_dataset(split="train")
    explicit = make_dataset(split="train", img_resolution=[8, 6])
    for default_frame, explicit_frame in zip(default[0], explicit[0]):
        torch.testing.assert_close(default_frame, explicit_frame, rtol=0, atol=0)


@pytest.mark.parametrize(
    "resolution",
    [[0, 3], [-4, 3], [4.0, 3], [True, 3], [4], [4, 3, 2], [3, 3], [16, 12]],
)
def test_invalid_resolution_is_rejected(make_dataset, resolution):
    with pytest.raises(ValueError, match="resolution"):
        make_dataset(split="train", img_resolution=resolution)
