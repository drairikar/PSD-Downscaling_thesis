import numpy as np
import os

# cerra_path = "/aspire/CarloData/NeurIPS-workshop/CentralEurope_2014_2020/CERRA/samples/train"
cerra_test_path = "/aspire/CarloData/NeurIPS-workshop/Scandinavia/CERRA/samples/test"
# era5_path = "/aspire/CarloData/NeurIPS-workshop/Scandinavia/ERA5/samples/train"
era5_test_path = "/aspire/CarloData/NeurIPS-workshop/Scandinavia/ERA5/samples/test"

cerra_files = [f for f in os.listdir(cerra_test_path) if f.endswith('.npy')]
era5_files = [f for f in os.listdir(era5_test_path) if f.endswith('.npy')]

print(f"CERRA test filenames ({len(cerra_files)}):")
for f in sorted(cerra_files)[::-1][:5]:  ##only 5 files
    print(f"  {f}") ##only 5 files
    

# print(f"ERA5 train filenames ({len(era5_files)}):")
# for f in sorted(era5_files)[:5]:
#     print(f"  {f}")
#     ##print columns of the first file
#     first_era5_file = np.load(os.path.join(era5_test_path, era5_files[0]))
#     print(f"Columns in the first ERA5 file: {first_era5_file.shape}")

if cerra_files:
    sample_cerra = np.load(os.path.join(cerra_test_path, cerra_files[0]))
    print(f"CERRA sample shape: {sample_cerra.shape}")
    num_files_CERRA = len(cerra_files)
    print(f"Number of CERRA files : {num_files_CERRA}")
    
    # for c in range(first_cerra_file.shape[-1]):
    #     print(f" CERRA channel {c} stats: min={first_cerra_file[...,c].min()}, max={first_cerra_file[...,c].max()}, mean={first_cerra_file[...,c].mean()}")
 

if era5_files:
    sample_era5 = np.load(os.path.join(era5_test_path, era5_files[0]))
    print(f"ERA5 sample shape: {sample_era5.shape}")
    num_files_ERA5 = len(era5_files)
    print(f"Number of ERA5 files: {num_files_ERA5}")
    # first_era5_file = np.load(os.path.join(era5_test_path, era5_files[0]))
    # # print(f"Columns in the first ERA5 file: {first_era5_file.dtype}")
    # print(f"Columns in the first ERA5 file: {first_era5_file.shape[-1]}")
 
