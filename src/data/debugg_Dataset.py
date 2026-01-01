import numpy as np
import os

cerra_path = "/aspire/CarloData/NeurIPS-workshop/CentralEurope_2014_2020/CERRA/samples/train"
era5_path = "/aspire/CarloData/NeurIPS-workshop/CentralEurope_2014_2020/ERA5/samples/train"

cerra_files = [f for f in os.listdir(cerra_path) if f.endswith('.npy')]
era5_files = [f for f in os.listdir(era5_path) if f.endswith('.npy')]

if cerra_files:
    sample_cerra = np.load(os.path.join(cerra_path, cerra_files[0]))
    print(f"CERRA sample shape: {sample_cerra.shape}")
    
if era5_files:
    sample_era5 = np.load(os.path.join(era5_path, era5_files[0]))
    print(f"ERA5 sample shape: {sample_era5.shape}")
    
    