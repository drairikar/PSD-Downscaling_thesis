import os
import warnings

# Third-party
import numpy as np
import torch
import xarray as xr
import pandas as pd     

DYNAMIC_VARS = ["u10", "v10", "t2m", "sshf", "zust", "sp"]

class CerraPriorDatasetSequence(torch.utils.data.Dataset):
    """
    Super-Resolution Dataset (Upsampled ERA5 + CERRA Forcing).
    
    Logic:
    1. Loads Low-Res ERA5 (85x85) and Time Embeddings.
    2. Upsamples ERA5+Time to High-Res (384x384) using Bicubic interpolation.
    3. Concatenates with High-Res CERRA Forcing (Orography).
    4. Returns (Full_Input, CERRA_Target).
    """
    def __init__(self, 
                 root_dir_cerra,
                 root_dir_era5,
                 split,
                 region,
                #  cerra_vars=['u10', 'v10', 't2m', 'sshf', 'zust', 'sp'],
                #  era5_vars=['u10', 'v10', 't2m', 'sshf', 'zust', 'sp'],
                cerra_vars = ['u10', 'v10', 't2m'],
                era5_vars = ['u10', 'v10', 't2m'],
                 sequence_length = 8,           ##3h cadence *8 = 24h
                 sequence_stride = None,
                 cadence = 3,                     ##3h cadence
                 flatten=True,
                 n_samples=None,
                 crop=False,
                 pangu_sequence_dir = None,
                 sequence_start_time = None,
                 ):
        super().__init__()

        
        self.cerra_vars = cerra_vars
        self.era5_vars = era5_vars
        self.sequence_length = sequence_length
        self.sequence_stride = (
            sequence_length if sequence_stride is None else sequence_stride
        )
        self.cadence = pd.Timedelta(hours=cadence)
        self.flatten = flatten
        
        if self.sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {self.sequence_length}.")
        
        if self.sequence_stride < 1:
            raise ValueError(f"sequence_stride must be >= 1, got {self.sequence_stride}.")
    
        # Paths
        cerra_path = os.path.join(root_dir_cerra, split, f"{region}.nc")
        era5_path = os.path.join(root_dir_era5, split, f"{region}.nc")
        cerra_orography_path = os.path.join(root_dir_cerra, split, f"static_{region}.nc")

        # 1. Load Statistics

        # self.era5_mean = np.load(os.path.join(root_dir_era5, "statistics", "dynamic_mean.npy"))
        # self.era5_std = np.load(os.path.join(root_dir_era5, "statistics", "dynamic_std.npy"))

        all_dynamic_mean = np.load(os.path.join(root_dir_era5, "statistics", "dynamic_mean.npy"))
        all_dynamic_std = np.load(os.path.join(root_dir_era5, "statistics", "dynamic_std.npy"))

        stats_idx = {
            variable : index
            for index, variable in enumerate(DYNAMIC_VARS)
        }

        cerra_stats_idx = [stats_idx[var] for var in self.cerra_vars]
        era5_stats_idx = [stats_idx[var] for var in self.era5_vars]

        self.cerra_mean = all_dynamic_mean[cerra_stats_idx]
        self.cerra_std = all_dynamic_std[cerra_stats_idx]
        self.era5_mean = all_dynamic_mean[era5_stats_idx]
        self.era5_std = all_dynamic_std[era5_stats_idx]

        self.era5_orography_mean = np.load(os.path.join(root_dir_era5, "statistics", "forcing_mean.npy"))
        self.era5_orography_std = np.load(os.path.join(root_dir_era5, "statistics", "forcing_std.npy"))

        # 2. Open Datasets (Lazy Xarray)
        self.cerra_ds = xr.open_dataset(cerra_path, engine="h5netcdf")
        self.era5_ds = xr.open_dataset(era5_path, engine="h5netcdf")

        self.cerra_time = pd.DatetimeIndex(
            self.cerra_ds.time.values
        )

        if self.cerra_time.has_duplicates:
            raise ValueError("CERRA contains duplicate timestamps")
        
        if not self.cerra_time.is_monotonic_increasing:
            raise ValueError("CERRA timestamps are not sorted in increasing order")

        self.time = self.cerra_time

        cerra_lat = np.asarray(self.cerra_ds.latitude.values)
        era5_lat = np.asarray(self.era5_ds.latitude.values)

        if cerra_lat.ndim != 2 or not np.all(np.diff(cerra_lat, axis=0) > 0) :
            raise ValueError("CERRA latitude must increase along y")
        
        if era5_lat.ndim != 2 or not np.all(np.diff(era5_lat, axis=0) > 0):
            raise ValueError("ERA5 latitude must increase along y")
        
                
        # 3. Load Static Data into RAM (Optimization)
        # We perform the static normalization ONCE here to save CPU cycles in __getitem__
        ds_static_cerra = xr.open_dataset(cerra_orography_path, engine="h5netcdf")
        raw_static_cerra = (
            ds_static_cerra["orog"]
            .squeeze(drop=True)
            .values.astype(np.float32)
        )
        if raw_static_cerra.ndim != 2:
            raise ValueError(
                "Expected CERRA orography to be 2D after squeezing singleton "
                f"dimensions, got shape {raw_static_cerra.shape}."
            )
        raw_static_cerra = raw_static_cerra[None, ...]
        self.cerra_orography = torch.from_numpy(
            (raw_static_cerra - self.era5_orography_mean) / self.era5_orography_std
        )
        ds_static_cerra.close()

        # 4. Temporal subsampling
        total = len(self.time)
        candidate_start = range(
            0,
            total - self.sequence_length + 1,
            self.sequence_stride
        )
        valid_start = []
        for start in candidate_start:
            stop = start + self.sequence_length
            sequence_time = self.time[start:stop]
            time_differences = np.diff(sequence_time.view("int64"))
            expected_diff = self.cadence.value

            if np.all(time_differences == expected_diff):
                valid_start.append(start)
        
        self.indices = np.array(valid_start, dtype=np.int64)

        if sequence_start_time is not None:
            requested_start = pd.Timestamp(sequence_start_time)
            matching_indices = [
                start_idx
                for start_idx in self.indices
                if self.time[start_idx] == requested_start
            ]

            if len(matching_indices)!= 1:
                available =[
                    str(self.time[idx])
                    for idx in self.indices[:10]
                ]
                raise ValueError(
                    f"Requested start time {requested_start} does not match any valid sequence, found {len(matching_indices)}. "
                    f"Available times: {available}"
                )
            
            self.indices = np.array(matching_indices, dtype=np.int64)

            start_idx = int(self.indices[0])
            cerra_sequence_times = self.time[
                start_idx : start_idx + self.sequence_length
            ]
            
        if len(self.indices) == 0:
            raise ValueError(
                "No valid sequences found. Check that the time coordinates in the "
                "CERRA and ERA5 datasets are consistent with the specified "
                f"sequence_length ({self.sequence_length}), sequence_stride "
                f"({self.sequence_stride}), and cadence ({self.cadence})."
            )
        
        if n_samples is not None and n_samples < len(self.indices):
            rng = np.random.default_rng(seed=42)
            selected = rng.choice(len(self.indices), size=n_samples, replace=False)
            self.indices = np.sort(self.indices[selected])
            
        
    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx):
        # real_idx = int(self.indices[idx])
        start_idx = int(self.indices[idx])
        stop_idx = start_idx + self.sequence_length
        
        # 1. Load Dynamic Data
        cerra = torch.from_numpy(self._load_dynamic_sequence(self.cerra_ds, self.cerra_vars, start_idx, stop_idx))
        era5 = torch.from_numpy(self._load_dynamic_sequence(self.era5_ds, self.era5_vars, start_idx, stop_idx))
        
        # 2. Normalize Dynamic Data
        cerra = (cerra - self.cerra_mean[:, None, None]) / self.cerra_std[:, None, None]
        era5 = (era5 - self.era5_mean[:, None, None]) / self.era5_std[:, None, None]

        if self.flatten:
            cerra = cerra.flatten(0,1)
            era5 = era5.flatten(0,1)

       
              
        return cerra, self.cerra_orography, era5
    
    
    def _load_dynamic_sequence(self, dataset, variables,start,stop):
        
        """Lazy load specific time step."""

        data = (
        dataset[list(variables)]
        .isel(time=slice(start, stop))
        .to_array(dim="variable")
        .sel(variable=list(variables))
        .transpose("time", "variable", ...)
        .values.astype(np.float32)
    )

        if data.ndim != 4:
            raise ValueError(
                "Expected [time, variable, height, width], "
                f"received {data.shape}."
            )

        if data.shape[0] != self.sequence_length:
            raise ValueError(
                f"Expected {self.sequence_length} timestamps, "
                f"received {data.shape[0]}."
            )

        return data
        
        
    def close(self):
        if self.cerra_ds:
            self.cerra_ds.close()
        if self.era5_ds:
            self.era5_ds.close()
