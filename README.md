# PSD-Downscaling thesis

ERA5-to-CERRA weather downscaling with U-Net, FNO, U-NO, a local sequence
U-Net, and bicubic interpolation. The target variables are `u10`, `v10`, and
`t2m`. Spectral analyses also examine vorticity, divergence, and kinetic energy.

## Repository structure

```text
.
├── main.py                           # Training and evaluation
├── generate_monthly_deterministic.py  # Independent U-Net/FNO/U-NO trajectories
├── generate_monthly_unet_seq.py       # Local sequence U-Net trajectories
├── metrics.py                        # Legacy saved-array metrics
├── src/
│   ├── config/                       # Configuration validation and adapters
│   ├── data/                         # Regional NetCDF datasets
│   ├── models/                       # Model wrappers and Fourier losses
│   └── utils/                        # Metrics, plotting, and shared functions
├── yaml_configs/                     # Model and experiment configurations
├── sbatch/                           # Cluster launch scripts
├── tests/                            # Dataset and bicubic regression tests
├── physicsnemo/                      # Bundled model implementation
└── saved_models/                     # Local model checkpoints
```

## Requirements

The reference environment uses Python **3.13.0**, Torch **2.10.0**, PyTorch
Lightning **2.6.1**, and NeuralOperator **2.0.0**. Training and full-resolution
model evaluation should use a compatible NVIDIA GPU; the tests and bicubic
baseline support CPU execution.

[`requirements.txt`](requirements.txt) contains older dependency pins and needs
reconciliation before a fresh installation. Additional direct dependencies
include `cartopy`, `h5netcdf`, `nvtx`, and `tueplots`; tests require `pytest`.
A fresh installation has not been validated against that manifest.

Run commands from the repository root to use the bundled `physicsnemo/` code.
For the existing project environment:

```bash
conda activate devashish
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Models and configurations

| Model | Key | Configurations |
| --- | --- | --- |
| U-Net | `UNet-CNN` | [`UNet_train.yaml`](yaml_configs/UNet/UNet_train.yaml) |
| FNO | `FNO` | [`FNO_train.yaml`](yaml_configs/UNet/FNO_train.yaml), [`FNO_train_Iberia.yaml`](yaml_configs/UNet/FNO_train_Iberia.yaml) |
| U-NO | `UNO` | [`UNO_train.yaml`](yaml_configs/UNet/UNO_train.yaml), [`UNO_train_iberia.yaml`](yaml_configs/UNet/UNO_train_iberia.yaml) |
| Local sequence U-Net | `UNet-Seq` | [`UNet_seq_train.yaml`](yaml_configs/UNet/UNet_seq_train.yaml), [`UNet_seq_test.yaml`](yaml_configs/UNet/UNet_seq_test.yaml) |
| Bicubic | `Bicubic` | [`Bicubic_test.yaml`](yaml_configs/Baselines/Bicubic_test.yaml) |
| FNO at 175 × 175 | `FNO` | [`FNO_train_175.yaml`](yaml_configs/UNet/FNO_train_175.yaml), [`FNO_test_175.yaml`](yaml_configs/UNet/FNO_test_175.yaml) |

Several files named `_train.yaml` currently set `eval: test`. Check `model`,
`eval`, `region`, `load`, and `img_resolution` before running a configuration.
The current `UNet_train.yaml` selects Iberia. Additional diffusion, AFNO, and
DSFNO wrappers remain in the codebase; the AFNO/DSFNO example checkpoints are
not present locally.

## Data

Set `dataset_cerra` and `dataset_era5` in the YAML to processed-data roots with:

```text
CERRA/<split>/<Region>.nc
CERRA/<split>/static_<Region>.nc
ERA5/<split>/<Region>.nc
ERA5/statistics/dynamic_mean.npy
ERA5/statistics/dynamic_std.npy
ERA5/statistics/forcing_mean.npy
ERA5/statistics/forcing_std.npy
```

`<split>` is `train`, `val`, or `test`. ERA5/CERRA timestamps must match and
cover aligned regional domains. Dynamic statistics follow
`[u10, v10, t2m, sshf, zust, sp]`; both datasets use the selected ERA5 statistics.

Single-frame inputs contain bicubically resized ERA5 `u10`, `v10`, and `t2m`,
plus CERRA orography. Current native regional grids are 350 × 350. Setting
`img_resolution: [175, 175]` in `main.py` applies full-region 2 × 2 block
averaging. Sequence datasets require consecutive three-hour samples.

## Training and evaluation

Edit the chosen YAML for your data paths, model, region, resolution, and run
settings. YAML values override matching CLI arguments; `main.py` has no
`--override` option.

For training, set `eval: null`, `load: null`, and `resume: null`:

```bash
WANDB_MODE=offline python main.py --config yaml_configs/UNet/UNet_seq_train.yaml
```

For evaluation, set `eval: test` and `load` to a matching checkpoint:

```bash
WANDB_MODE=offline python main.py --config yaml_configs/UNet/UNet_seq_test.yaml
```

`WANDB_MODE=offline` stores W&B logs locally. Checkpoints are saved under
`saved_models/<run>/`: `min_val_loss.ckpt` is the best validation checkpoint,
and `last.ckpt` stores the latest training state. Use `resume` to continue
training with trainer state. Setting `wandb_project: null` selects a fallback
logger; learned models require a TensorBoard backend for that option.

### Bicubic baseline

```bash
python main.py --config yaml_configs/Baselines/Bicubic_test.yaml
```

Bicubic evaluation requires no checkpoint or training. It uses the three ERA5
fields without orography and reports physical-unit MSE, MAE, RMSE, and SSIM.
Set the region and resolution in the YAML; only `eval: test` is supported.

Bicubic RMSE is calculated from full-dataset MSE and SSIM is averaged per image.
Learned wrappers currently use batch-averaged RMSE and batch SSIM ranges, so
account for these aggregation differences when comparing logged metrics.

## Monthly trajectories

- [`generate_monthly_deterministic.py`](generate_monthly_deterministic.py)
  evaluates independent U-Net, FNO, or U-NO frames.
- [`generate_monthly_unet_seq.py`](generate_monthly_unet_seq.py) produces a
  continuous trajectory with the local sequence U-Net.

```bash
python generate_monthly_deterministic.py --help
python generate_monthly_unet_seq.py --help
```

Both scripts require a config, matching checkpoint, region, start/end times,
and destination path. `--start` is inclusive and `--end` is exclusive. They
save physical-unit `pred_seq`, `hr_seq`, and `lr_seq` tensors with
`[1, T, 3, H, W]` layout, timestamps, and temporal metrics. The monthly
scripts currently use native-grid datasets; use `main.py` for resolution
experiments. These trajectories supply baselines for the sibling GeoDiff
repository's temporal comparisons.

## Analysis

[`src/utils/psd_plot.py`](src/utils/psd_plot.py) calculates longitudinal spatial
PSDs for the three primary and three derived variables. Set `CERRA_PATH`,
`ERA5_PATH`, `MODEL_PATHS`, and `OUT_DIR` in the script, then run:

```bash
python src/utils/psd_plot.py
```

The calculation uses a one-dimensional FFT along longitude, averaging over
samples and latitude. It exports primary/derived three-panel figures and
per-variable CSV spectra. Derived gradients use grid-index spacing.

All selected predictions must match the reference sample count and spatial
shape. The current `FNO_175` entry has a different resolution from the native
reference: exclude it for a 350 × 350 comparison, or prepare matched references
for a separate resolution comparison.

Sequence-model test hooks export spatial PSD arrays and plots.
[`metrics.py`](metrics.py) contains older saved-array metrics; review its
channel/layout assumptions and CLI before reuse.

## Tests and reproducibility

```bash
CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg python -m pytest -q tests
```

The 27 CPU test cases cover full-region resolution changes, normalization,
bicubic predictions, metric aggregation, and baseline configuration validation:

- [`test_dataset_resolution.py`](tests/test_dataset_resolution.py)
- [`test_bicubic.py`](tests/test_bicubic.py)

For each thesis or paper result, record the exact command, YAML, code revision,
checkpoint, data/statistics version, region, split, resolution, and environment.
The tests use synthetic data. Seeded runs use `deterministic=False`, so seeds
alone do not guarantee identical training results.
