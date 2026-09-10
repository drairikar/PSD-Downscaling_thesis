"""Single-frame ERA5-to-CERRA datasets."""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr


DYNAMIC_VARS = ["u10", "v10", "t2m", "sshf", "zust", "sp"]


class ERA5toCERRA2(torch.utils.data.Dataset):
    """Load independent ERA5/CERRA timestamp pairs from regional NetCDF files.

    Every timestamp is one sample. ERA5 dynamic fields are interpolated onto
    the CERRA grid and concatenated with normalized CERRA orography. Training
    and validation return ``(conditioning, target)``; testing additionally
    returns normalization statistics and a timestamp name, as expected by the
    deterministic model wrappers.
    """

    def __init__(
        self,
        dataset_name_CERRA,
        dataset_name_ERA5,
        split,
        standardize=True,
        subset=False,
        region="CentralEurope",
        cerra_vars=("u10", "v10", "t2m"),
        era5_vars=("u10", "v10", "t2m"),
        n_samples=None,
    ):
        super().__init__()

        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown dataset split: {split}")
        if dataset_name_CERRA is None or dataset_name_ERA5 is None:
            raise ValueError(
                "Deterministic ERA5-to-CERRA training requires both datasets."
            )

        self.root_dir_cerra = dataset_name_CERRA
        self.root_dir_era5 = dataset_name_ERA5
        self.split = split
        self.standardize = bool(standardize)
        self.region = region
        self.cerra_vars = list(cerra_vars)
        self.era5_vars = list(era5_vars)

        unknown_vars = (
            set(self.cerra_vars) | set(self.era5_vars)
        ) - set(DYNAMIC_VARS)
        if unknown_vars:
            raise ValueError(
                f"No normalization statistics are defined for {sorted(unknown_vars)}."
            )

        cerra_path = os.path.join(
            dataset_name_CERRA, split, f"{region}.nc"
        )
        era5_path = os.path.join(
            dataset_name_ERA5, split, f"{region}.nc"
        )
        static_path = os.path.join(
            dataset_name_CERRA, split, f"static_{region}.nc"
        )

        print("CERRA path:", cerra_path)
        print("ERA5 path:", era5_path)

        self.cerra_ds = xr.open_dataset(cerra_path, engine="h5netcdf")
        self.era5_ds = xr.open_dataset(era5_path, engine="h5netcdf")

        cerra_time = pd.DatetimeIndex(self.cerra_ds.time.values)
        era5_time = pd.DatetimeIndex(self.era5_ds.time.values)
        self._validate_time(cerra_time, "CERRA")
        self._validate_time(era5_time, "ERA5")
        if not cerra_time.equals(era5_time):
            raise ValueError(
                "CERRA and ERA5 timestamps do not match exactly for "
                f"split={split!r}, region={region!r}."
            )

        self.time = cerra_time
        self.indices = np.arange(len(self.time), dtype=np.int64)

        if subset and n_samples is None:
            n_samples = 5000
        if n_samples is not None:
            n_samples = int(n_samples)
            if n_samples < 1:
                raise ValueError(f"n_samples must be positive, got {n_samples}.")
            self.indices = self.indices[:n_samples]

        stats_dir = os.path.join(dataset_name_ERA5, "statistics")
        dynamic_mean = np.load(
            os.path.join(stats_dir, "dynamic_mean.npy")
        ).astype(np.float32)
        dynamic_std = np.load(
            os.path.join(stats_dir, "dynamic_std.npy")
        ).astype(np.float32)
        expected_stats_shape = (len(DYNAMIC_VARS),)
        if dynamic_mean.shape != expected_stats_shape:
            raise ValueError(
                "dynamic_mean.npy must contain one value per dynamic variable; "
                f"expected {expected_stats_shape}, got {dynamic_mean.shape}."
            )
        if dynamic_std.shape != expected_stats_shape:
            raise ValueError(
                "dynamic_std.npy must contain one value per dynamic variable; "
                f"expected {expected_stats_shape}, got {dynamic_std.shape}."
            )

        stats_index = {
            variable: index for index, variable in enumerate(DYNAMIC_VARS)
        }
        cerra_stats_idx = [stats_index[var] for var in self.cerra_vars]
        era5_stats_idx = [stats_index[var] for var in self.era5_vars]

        # Match dataset_seq.py: shared dynamic statistics stored below the
        # ERA5 root normalize both guidance and target variables.
        self.data_mean_CERRA = torch.from_numpy(dynamic_mean[cerra_stats_idx])
        self.data_std_CERRA = torch.from_numpy(dynamic_std[cerra_stats_idx])
        self.data_mean_era5 = torch.from_numpy(dynamic_mean[era5_stats_idx])
        self.data_std_era5 = torch.from_numpy(dynamic_std[era5_stats_idx])

        if torch.any(self.data_std_CERRA <= 0):
            raise ValueError("CERRA standard deviations must be positive.")
        if torch.any(self.data_std_era5 <= 0):
            raise ValueError("ERA5 standard deviations must be positive.")

        forcing_mean = np.asarray(
            np.load(os.path.join(stats_dir, "forcing_mean.npy")),
            dtype=np.float32,
        ).reshape(-1)
        forcing_std = np.asarray(
            np.load(os.path.join(stats_dir, "forcing_std.npy")),
            dtype=np.float32,
        ).reshape(-1)
        if forcing_mean.size != 1 or forcing_std.size != 1:
            raise ValueError(
                "Expected scalar forcing statistics for the orography channel, "
                f"got mean={forcing_mean.shape}, std={forcing_std.shape}."
            )

        self.orography_mean = torch.tensor(
            float(forcing_mean[0]), dtype=torch.float32
        )
        self.orography_std = torch.tensor(
            float(forcing_std[0]), dtype=torch.float32
        )
        if self.orography_std <= 0:
            raise ValueError("Orography standard deviation must be positive.")

        with xr.open_dataset(static_path, engine="h5netcdf") as static_ds:
            orography = (
                static_ds["orog"]
                .squeeze(drop=True)
                .values.astype(np.float32)
            )
        if orography.ndim != 2:
            raise ValueError(
                "Expected two-dimensional orography after squeezing singleton "
                f"dimensions, got {orography.shape}."
            )

        self.orography = torch.from_numpy(orography).unsqueeze(0)
        if self.standardize:
            self.orography = (
                self.orography - self.orography_mean
            ) / self.orography_std
        self.output_size = tuple(self.orography.shape[-2:])

    @staticmethod
    def _validate_time(time, dataset_name):
        if time.has_duplicates:
            raise ValueError(f"{dataset_name} contains duplicate timestamps.")
        if not time.is_monotonic_increasing:
            raise ValueError(f"{dataset_name} timestamps are not sorted.")

    @staticmethod
    def _load_frame(dataset, variables, time_index):
        frame = (
            dataset[list(variables)]
            .isel(time=int(time_index))
            .to_array(dim="variable")
            .sel(variable=list(variables))
            .transpose("variable", ...)
            .values.astype(np.float32)
        )
        if frame.ndim != 3:
            raise ValueError(
                "Expected [variable, height, width], "
                f"received {frame.shape}."
            )
        return torch.from_numpy(frame)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        time_index = int(self.indices[idx])
        sample_CERRA = self._load_frame(
            self.cerra_ds, self.cerra_vars, time_index
        )
        sample_era5 = self._load_frame(
            self.era5_ds, self.era5_vars, time_index
        )

        sample_era5 = F.interpolate(
            sample_era5.unsqueeze(0),
            size=self.output_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        if sample_CERRA.shape[-2:] != self.output_size:
            raise ValueError(
                f"CERRA frame has shape {sample_CERRA.shape[-2:]}, "
                f"but static data have shape {self.output_size}."
            )

        if self.standardize:
            sample_CERRA = (
                sample_CERRA - self.data_mean_CERRA[:, None, None]
            ) / self.data_std_CERRA[:, None, None]
            sample_era5 = (
                sample_era5 - self.data_mean_era5[:, None, None]
            ) / self.data_std_era5[:, None, None]

        conditioning = torch.cat([sample_era5, self.orography], dim=0)

        if self.split != "test":
            return conditioning, sample_CERRA

        if self.standardize:
            target_mean = self.data_mean_CERRA
            target_std = self.data_std_CERRA
            conditioning_mean = torch.cat(
                [self.data_mean_era5, self.orography_mean.reshape(1)]
            )
            conditioning_std = torch.cat(
                [self.data_std_era5, self.orography_std.reshape(1)]
            )
        else:
            target_mean = torch.zeros_like(self.data_mean_CERRA)
            target_std = torch.ones_like(self.data_std_CERRA)
            conditioning_mean = torch.zeros(
                conditioning.shape[0], dtype=torch.float32
            )
            conditioning_std = torch.ones(
                conditioning.shape[0], dtype=torch.float32
            )

        statistics = {
            "mean_CERRA": target_mean[:, None, None],
            "std_CERRA": target_std[:, None, None],
            "mean_era5": conditioning_mean[:, None, None],
            "std_era5": conditioning_std[:, None, None],
        }
        timestamp_name = self.time[time_index].strftime("%Y%m%dT%H%M%S")
        return conditioning, sample_CERRA, statistics, timestamp_name

    def close(self):
        cerra_ds = getattr(self, "cerra_ds", None)
        era5_ds = getattr(self, "era5_ds", None)
        if cerra_ds is not None:
            cerra_ds.close()
        if era5_ds is not None:
            era5_ds.close()


class CerraEra5SuperResDataset(ERA5toCERRA2):
    """Backward-compatible alias for the single-frame dataset."""
