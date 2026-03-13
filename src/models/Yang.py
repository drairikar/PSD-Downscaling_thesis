"""
Network modules for downscaling FNO model (DSFNO).
Original Author: Qidong Yang
Date: 2022-08-26
Refactored for integration with ERA5-CERRA training pipeline.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Dict, Any, Callable, Tuple
import os
import random
import matplotlib.pyplot as plt
import wandb
import numpy as np
from torchmetrics.functional import structural_similarity_index_measure as ssim
from .. import constants
from ..utils import vis
from scipy.fft import fft

from .unet import RegressionLoss
from .losses.fourier_losses import FourierLossETH, FourierLossDelft, FourierLossHK, FourierLossCarlo
from torch.optim import AdamW

##### Constraint Layer #####

class SoftmaxConstraint(nn.Module):
    """
    Applies a softmax-based constraint to ensure conservation of mass/energy
    between low-resolution and high-resolution predictions.
    """
    def __init__(self, exp_factor: float = 1.0):
        super(SoftmaxConstraint, self).__init__()
        self.exp_factor = exp_factor

    def forward(self, x: torch.Tensor, y: torch.Tensor, upsample_factor: int) -> torch.Tensor:
        """
        Args:
            x: Low-resolution input (n_batch, in_channel, size_x, size_y)
            y: High-resolution prediction (n_batch, in_channel, up_size_x, up_size_y)
            upsample_factor: Upsampling factor
        
        Returns:
            Constrained high-resolution output
        """
        if upsample_factor <= 1:
            # No super-resolution patching is needed when LR/HR resolutions match.
            return y

        # Clamp logits before exp to avoid inf/NaN in mixed precision.
        y = torch.exp(torch.clamp(y * self.exp_factor, min=-20.0, max=20.0))
        avg_y = F.avg_pool2d(y, kernel_size=upsample_factor)
        avg_y = avg_y.clamp_min(torch.finfo(y.dtype).eps)
        
        # Create upsampling kernel on the same device as x
        ones_kernel = torch.ones(
            (upsample_factor, upsample_factor), 
            device=x.device, 
            dtype=x.dtype
        )

        out = y * torch.kron(x / avg_y, ones_kernel)
        
        return out


##### Model Modules #####

class SpectralConv2d(nn.Module):
    """
    2D Fourier Neural Operator layer.
    Performs FFT, linear transform in spectral space, and inverse FFT.
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super(SpectralConv2d, self).__init__()
        
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = modes1
        self.modes2 = modes2

        # Initialization scale for stability
        self.scale = (1 / (2 * in_channels)) ** 0.5
        
        # Learnable spectral weights
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(
                self.in_channels, self.out_channels, 
                self.modes1, self.modes2, 
                dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(
                self.in_channels, self.out_channels, 
                self.modes1, self.modes2, 
                dtype=torch.cfloat
            )
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication in spectral space."""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor, upsample_factor: int) -> torch.Tensor:
        """
        Args:
            x: Input (n_batch, in_channels, n_dim1, n_dim2)
            upsample_factor: Spatial upsampling factor
        
        Returns:
            Output (n_batch, out_channels, dim1, dim2)
        """
        batch_size = x.shape[0]
        dim1 = int(upsample_factor * x.shape[2])
        dim2 = int(upsample_factor * x.shape[3])

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)

        # Determine number of modes to use (handle edge cases)
        modes1_use = int(min(self.modes1, dim1 // 2, x.shape[2] // 2))
        modes2_use = int(min(self.modes2, dim2 // 2 + 1, x.shape[3] // 2 + 1))

        # Initialize output in spectral space
        out_ft = torch.zeros(
            batch_size, self.out_channels, dim1, dim2 // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Multiply relevant Fourier modes
        out_ft[:, :, :modes1_use, :modes2_use] = self.compl_mul2d(
            x_ft[:, :, :modes1_use, :modes2_use], 
            self.weights1[:, :, :modes1_use, :modes2_use]
        )
        out_ft[:, :, -modes1_use:, :modes2_use] = self.compl_mul2d(
            x_ft[:, :, -modes1_use:, :modes2_use], 
            self.weights2[:, :, -modes1_use:, :modes2_use]
        )

        # Return to physical space with proper scaling
        x = torch.fft.irfft2(out_ft, s=(dim1, dim2))
        x = x * (upsample_factor ** 2)
        
        return x


class OperatorBlock(nn.Module):
    """
    FNO operator block combining spectral convolution and spatial convolution.
    """
    def __init__(self, n_channels: int, modes1: int, modes2: int, activation: bool = True):
        super(OperatorBlock, self).__init__()
        
        self.n_channels = n_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.activation = activation

        self.conv = SpectralConv2d(n_channels, n_channels, modes1, modes2)
        self.w = nn.Conv2d(n_channels, n_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (n_batch, n_channels, height, width)
        
        Returns:
            Processed tensor with same shape
        """
        x1 = self.conv(x, upsample_factor=1)
        x2 = self.w(x)
        x = x1 + x2

        if self.activation:
            x = F.gelu(x)

        return x


class ResidualBlock(nn.Module):
    """
    Standard residual block with two convolutional layers.
    """
    def __init__(self, n_channels: int):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            n_channels, n_channels, 
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            n_channels, n_channels, 
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual
        
        return out


##### Downscaling Model #####

class DSFNO(nn.Module):
    """
    Downscaling Fourier Neural Operator (DSFNO).
    
    Architecture:
    1. Initial conv + residual blocks (feature extraction at LR)
    2. Bicubic upsampling
    3. FNO blocks (refinement at HR)
    4. MLP decoder
    5. Optional constraint layer
    """
    def __init__(
        self, 
        in_channels: int = 1,
        out_channels: int = 1,
        n_channels: int = 64,
        n_residual_blocks: int = 4,
        n_operator_blocks: int = 2,
        modes: int = 18,
        apply_constraint: bool = True
    ):
        super(DSFNO, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = n_channels
        self.n_residual_blocks = n_residual_blocks
        self.n_operator_blocks = n_operator_blocks
        self.modes = modes
        self.apply_constraint = apply_constraint

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Residual blocks for feature extraction
        self.res_blocks = nn.ModuleList([
            ResidualBlock(n_channels) for _ in range(n_residual_blocks)
        ])

        # Second convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # FNO blocks for refinement
        self.fno_blocks = nn.ModuleList()
        for i in range(n_operator_blocks - 1):
            self.fno_blocks.append(
                OperatorBlock(n_channels, modes, modes, activation=True)
            )
        self.fno_blocks.append(
            OperatorBlock(n_channels, modes, modes, activation=False)
        )

        # Channel reduction MLP
        self.fc1 = nn.Linear(n_channels, 128)
        self.fc2 = nn.Linear(128, self.out_channels)

        # Optional constraint layer
        if apply_constraint:
            self.constraint = SoftmaxConstraint()

    def forward(self, x: torch.Tensor, upsample_factor: int) -> torch.Tensor:
        """
        Args:
            x: Input tensor (n_batch, size_x, size_y, in_channels) or
               (n_batch, in_channels, size_x, size_y)
            upsample_factor: Spatial upsampling factor
        
        Returns:
            High-resolution output (n_batch, up_size_x, up_size_y, out_channels)
        """
        # Handle different input formats
        if x.dim() == 4 and x.shape[-1] == self.in_channels:
            # Input is (batch, height, width, channels)
            x = x.permute(0, 3, 1, 2)
        
        # Feature extraction at low resolution
        out = self.conv1(x)
        
        for layer in self.res_blocks:
            out = layer(out)
        
        out = self.conv2(out)
        
        # Upsampling
        out = F.interpolate(
            out, 
            scale_factor=upsample_factor, 
            mode='bicubic', 
            align_corners=False
        )
        
        # Refinement at high resolution
        for layer in self.fno_blocks:
            out = layer(out)
        
        # Channel dimension last for MLP
        out = out.permute(0, 2, 3, 1)
        
        # Decode to output channels
        out = self.fc1(out)
        out = F.gelu(out)
        out = self.fc2(out)
        
        # Apply constraint if enabled
        ##Slice x to out_channels for constraint layer to match dimensions of CERRA channels (15-> 5)
        if self.apply_constraint and upsample_factor > 1:
            out = out.permute(0, 3, 1, 2)
            out = self.constraint(x[:, :self.out_channels, :, :], out, upsample_factor)
            out = out.permute(0, 2, 3, 1)

        
        return out


##### PyTorch Lightning Wrapper #####

class DSFNOWrapper(pl.LightningModule):
    """
    PyTorch Lightning wrapper for DSFNO model compatible with ERA5-CERRA pipeline.
    """
    def __init__(self, args):
        super(DSFNOWrapper, self).__init__()
        
        self.save_hyperparameters()
        self.args = args
        if isinstance(args.img_resolution, int):
            self.img_shape_x = self.img_shape_y = args.img_resolution
        else:
            self.img_shape_y = args.img_resolution[0]
            self.img_shape_x = args.img_resolution[1]

        self.img_in_channels = args.img_in_channels
        self.img_out_channels = args.img_out_channels
        self.lr = args.lr
        self.wandb_project = args.wandb_project
        self.savepreds_path = args.savepreds_path
        self.load = args.load
        self.checkpoint_level = args.checkpoint_level
        
        # Extract model parameters
        # in_channels = getattr(args, 'img_in_channels', 5)
        # n_channels = getattr(args, 'dsfno_hidden_channels', 64)
        # n_residual_blocks = getattr(args, 'dsfno_n_residual_blocks', 4)
        # n_operator_blocks = getattr(args, 'dsfno_n_operator_blocks', 2)
        # modes = getattr(args, 'dsfno_modes', 18)
        # apply_constraint = getattr(args, 'dsfno_apply_constraint', True)
        
        # Initialize model
        self.model = DSFNO(
            in_channels=args.img_in_channels,
            out_channels=args.img_out_channels,
            n_channels=args.n_channels,
            n_residual_blocks=args.n_residual_blocks,
            n_operator_blocks=args.n_operator_blocks,
            modes=args.modes,
            apply_constraint=args.apply_constraint
        )
        
        # Loss configuration
        loss_dict = {
            'Fourier_ETH': FourierLossETH,
            'Fourier_Delft': FourierLossDelft,
            'Fourier_HK': FourierLossHK,
            'Fourier_Carlo': FourierLossCarlo,

        }

        loss_func = loss_dict[args.loss_type]() if args.loss_type in loss_dict else None
        self.loss_fn = RegressionLoss(args.init_lambda, args.max_lambda, args.anneal_epochs, loss_func)
        
        # Input ERA5 is already resized to CERRA resolution in the dataset loader.
        self.upsample_factor = 1

        # The softmax conservation constraint is only meaningful for true upsampling (>1).
        if self.model.apply_constraint and self.upsample_factor <= 1:
            self.model.apply_constraint = False
            print("DSFNO: disabling SoftmaxConstraint because upsample_factor <= 1.")

    def forward(self, x: torch.Tensor,
    img_lr: torch.Tensor,
    **model_kwargs: dict, 
    ) -> torch.Tensor:
        """Forward pass through the model."""
        del x
        if img_lr is None:
            raise ValueError("img_lr must be provided for DSFNO forward pass.")
        
        img_lr = img_lr.float()
        device_type = "cuda" if img_lr.is_cuda else "cpu"
        with torch.autocast(device_type = device_type, enabled=False):
            F_x = self.model(img_lr, upsample_factor=self.upsample_factor)

        if F_x.shape[1] != self.img_out_channels:
            F_x = F_x.permute(0, 3, 1, 2)  # (batch, channels, height, width)

        return F_x.float()

    def training_step(self, batch, *args):
        img_lr, img_clean, *rest = batch
        img_clean = img_clean.float()
        img_lr = img_lr.float()
        loss, _, _, loss_space, loss_amp = self.loss_fn(
            net=self,
            img_clean=img_clean,
            img_lr=img_lr,
            current_epoch=self.current_epoch,
        )
        
        log_dict = { 
            # "train_loss": loss,
            "train_loss_epoch": loss}
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, *args):
        img_lr, img_clean = batch
        img_clean = img_clean.float()
        img_lr = img_lr.float()
        # print("lr image", img_lr.shape)
        # print("clean image", img_clean.shape)
        val_loss, ground_truth, predictions, loss_space, loss_amp = self.loss_fn(
            net=self,
            img_clean=img_clean,
            img_lr=img_lr,
            current_epoch=self.current_epoch
        )
        
        val_log_dict = {
            
            "val_loss": val_loss
            # "val_loss_epoch": val_loss,
            }
        self.log_dict(val_log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        batch_idx = args[0]
        
        if (
            self.trainer.is_global_zero
            and (not self.trainer.sanity_checking)
            and batch_idx == 0
            and self.current_epoch % 10 == 0
            and self.wandb_project is not None
        ):
            self.load_metrics_and_plots(predictions, ground_truth, batch_idx, mask=None)


    def test_step(self, batch, batch_idx: int) -> dict:
        """Same as UNetWrapper test_step"""
        img_lr, img_clean, diz_stats, img_lr_name = batch

        batch_size = img_clean.shape[0]
        img_clean = img_clean.float()
        img_lr = img_lr.float()

        _, ground_truth, predictions, _, _ = self.loss_fn(
            net=self,
            img_clean=img_clean,
            img_lr=img_lr,
            current_epoch=self.current_epoch
        )

        # Un-normalize
        high_res_mean = diz_stats["mean_CERRA"]
        high_res_std = diz_stats["std_CERRA"]
        predictions = predictions * high_res_std + high_res_mean
        ground_truth = ground_truth * high_res_std + high_res_mean

        if self.savepreds_path:
            savepath = self.savepreds_path + "/" + self.load.split("/")[-2] + "/files"
            os.makedirs(savepath, exist_ok=True)
            preds_cpu = predictions.detach().cpu().numpy()
            for i in range(batch_size):
                base_name = img_lr_name[i]
                out_path = os.path.join(savepath, f"nwp_{base_name}")
                np.save(out_path, preds_cpu[i])

        # Compute metrics
        mse_all = torch.mean((predictions - ground_truth) ** 2).detach()
        mae_all = torch.mean(torch.abs(predictions - ground_truth)).detach()
        rmse_all = torch.sqrt(mse_all)
        data_range = (ground_truth.max() - ground_truth.min()).item()
        ssim_all = ssim(predictions, ground_truth, data_range=data_range).detach()

        var_names = ['u10', 'v10', 't2m', 'sshf', 'zust']
        mse_vars, mae_vars, rmse_vars, ssim_vars = {}, {}, {}, {}

        for i, var_name in enumerate(var_names):
            pred_i = predictions[:, i, :, :]
            gt_i = ground_truth[:, i, :, :]
            mse_val = torch.mean((pred_i - gt_i) ** 2).detach()
            mse_vars[f"test_mse_{var_name}"] = mse_val
            mae_vars[f"test_mae_{var_name}"] = torch.mean(torch.abs(pred_i - gt_i)).detach()
            rmse_vars[f"test_rmse_{var_name}"] = torch.sqrt(mse_val)
            data_range_i = (gt_i.max() - gt_i.min()).item()
            ssim_vars[f"test_ssim_{var_name}"] = ssim(
                pred_i.unsqueeze(1), gt_i.unsqueeze(1), data_range=data_range_i
            ).detach()

        log_metrics = {
            "test_mse": mse_all, "test_mae": mae_all,
            "test_rmse": rmse_all, "test_ssim": ssim_all,
        }
        log_metrics.update(mse_vars)
        log_metrics.update(mae_vars)
        log_metrics.update(rmse_vars)
        log_metrics.update(ssim_vars)

        self.log_dict(log_metrics, prog_bar=False, on_epoch=True, sync_dist=True)
        self.plot_preds(predictions, ground_truth, img_lr, diz_stats)
        return log_metrics

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.epochs,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


    def load_metrics_and_plots(self, prediction, high_res, batch_idx, mask=None):
        """Same as UNetWrapper"""
        prediction = prediction.permute(0, 2, 3, 1).flatten(1, 2)
        high_res = high_res.permute(0, 2, 3, 1).flatten(1, 2)
        
        if mask is None:
            mask = torch.ones_like(high_res[:, :, 0])
        
        log_plot_dict = {}
        n_channels = prediction.shape[-1]
        if n_channels < 1:
            return
        var_i = random.randint(0, n_channels - 1)
        
        var_name = (
            constants.PARAM_NAMES_SHORT_CERRA[var_i]
            if var_i < len(constants.PARAM_NAMES_SHORT_CERRA)
            else f"var_{var_i}"
        )
        var_unit = (
            constants.PARAM_UNITS_CERRA[var_i]
            if var_i < len(constants.PARAM_UNITS_CERRA)
            else ""
        )
        
        sample = random.randint(0, prediction.shape[0] - 1)
        pred_states = prediction[sample, :, var_i]
        target_state = high_res[sample, :, var_i]
        plot_title = f"{var_name} ({var_unit})"

        log_plot_dict[f"pred_{var_name}"] = vis.plot_ensemble_prediction(
            pred_states, target_state, obs_mask=mask[sample], title=f"{plot_title} (prior)",
        )

        if not self.trainer.sanity_checking:
            wandb.log(log_plot_dict)
        plt.close("all")

    def plot_preds(self, prediction, high_res, img_lr, diz_stats):
        """Same as UNetWrapper"""
        
        low_res_mean = diz_stats["mean_era5"]
        low_res_std = diz_stats["std_era5"]
        img_lr = img_lr * low_res_std + low_res_mean

        sample_idx = random.randint(0, prediction.shape[0] - 1)
        var_i = random.randint(0, prediction.shape[1] - 1)
        var_name = constants.PARAM_NAMES_SHORT_CERRA[var_i]
        var_unit = constants.PARAM_UNITS_CERRA[var_i]

        input_img = img_lr[sample_idx, var_i, :, :].detach().cpu().numpy()
        target_img = high_res[sample_idx, var_i, :, :].detach().cpu().numpy()
        pred_img = prediction[sample_idx, var_i, :, :].detach().cpu().numpy()
        residual_img = target_img - pred_img

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(input_img, cmap='plasma', origin='lower')
        axes[0].set_title("Input"); axes[0].axis("off")
        axes[1].imshow(target_img, cmap='plasma', origin='lower')
        axes[1].set_title("Target"); axes[1].axis("off")
        axes[2].imshow(pred_img, cmap='plasma', origin='lower')
        axes[2].set_title("Prediction"); axes[2].axis("off")
        axes[3].imshow(residual_img, cmap='plasma', origin='lower')
        axes[3].set_title("Residual"); axes[3].axis("off")
        fig.suptitle(f"{var_name} ({var_unit})", fontsize=16)

        if not self.savepreds_path:
            plt.close(fig)
            return

        run_name = self.load.split("/")[-2]
        save_dir = os.path.join(self.savepreds_path, run_name, "pred_plots")
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"{var_name}_sample_{sample_idx}.png")
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)
