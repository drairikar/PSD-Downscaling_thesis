#!/bin/sh
#SBATCH -J Devashish-FNO
#SBATCH --nodes=1
#SBATCH -p gpu_h100
#SBATCH -t 2-23:59:59
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=100G
#SBATCH --mail-type=END
#SBATCH --mail-user=d.y.rairikar@student.tudelft.nl
#SBATCH --output=job-dev-uno-%j.log   # Save stdout to job-<jobid>.log
#SBATCH --error=job-dev-uno-%j.err    # Save stderr to job-<jobid>.err

# Load any required modules (if needed)
module load 2023
# Source the conda initialization script from the correct path.
# source /sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0/etc/profile.d/conda.sh
source /home/csaccardi1/miniconda3/etc/profile.d/conda.sh
# Activate the "devashish" environment.
conda activate devashish

# Prepend the activated environment’s bin directory to PATH.
export PATH="$CONDA_PREFIX/bin:$PATH"

# export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_DEBUG=INFO
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=1800
# unset NCCL_P2P_DISABLE=1

# Diagnostic prints to verify correct environment activation.
echo "Active environment: $CONDA_DEFAULT_ENV"
echo "Using python: $(which python)"
python --version

# Run your training script.
export WANDB_API_KEY=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

# srun python main.py --config="yaml_configs/UNet/UNO_train.yaml"
# srun python main.py --config="yaml_configs/UNet/AFNO_train.yaml"
# srun python main.py --config="yaml_configs/UNet/DSFNO_train.yaml"
# srun python main.py --config="yaml_configs/UNet/UNet_train.yaml"
python src/utils/psd_plot.py
