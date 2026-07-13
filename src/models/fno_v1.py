import torch
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt

# from physicsnemo.models.fno import FNO
from .. import constants
from ..utils import vis

from typing import Callable, Optional, Tuple
import random
from torchmetrics.functional import structural_similarity_index_measure as ssim_func
import os
import numpy as np
from scipy.fft import fft
import math
import torch.nn as nn
from .unet import RegressionLoss
from .losses.fourier_losses import FourierLossETH, FourierLossDelft, FourierLossHK, FourierLossCarlo
from physicsnemo.models.fno import FNO
# from .rrdb import RRDBNet

from torch.nn import functional as F


class FNOWrapper(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
             
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
        self.fno_width = args.fno_layer_size
        # self.num_rrdb = getattr(args, 'num_rrdb', 4)
        
        
        # feature_channels = args.fno_layer_size
        # self.feature_extractor = RRDBNet(
        #     in_channels=args.img_in_channels,
        #     out_channels=feature_channels,
        #     channels=64,
        #     growth_channels=32,
        #     num_rrdb=getattr(args, 'num_rrdb', 4)
        # )
                
        
        self.model = FNO(
                       
            in_channels=args.img_in_channels,
            # in_channels=feature_channels,
            out_channels=args.img_out_channels,
            decoder_layers = getattr(args, 'decoder_layers', 1),
            decoder_layer_size=getattr(args, 'decoder_layer_size', 64),
            dimension=2,
            # latent_channels=args.fno_layer_size,
            latent_channels=self.fno_width,
            num_fno_layers=args.num_fno_layers,
            num_fno_modes= args.num_fno_modes,
            padding=args.fno_padding,
            coord_features=args.coord_features,
                    
        )
        
        loss_dict = {
           
            'Fourier_ETH': FourierLossETH,
            'Fourier_Delft': FourierLossDelft,
            'Fourier_HK': FourierLossHK,
            'Fourier_Carlo': FourierLossCarlo,
        }
        
        loss_func = loss_dict[args.loss_type]() if args.loss_type in loss_dict else None
        self.loss_fn = RegressionLoss(args.init_lambda, args.max_lambda, args.anneal_epochs, loss_func)
        

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs: dict,
    ) -> torch.Tensor:
        """
        Forward pass of the FNO wrapper model.
        """
        expected_hw = (self.img_shape_y, self.img_shape_x)
        
        if img_lr is None:
            raise ValueError("Low-resolution image 'img_lr' must be provided.")
            
        if img_lr.shape[-2:] != expected_hw:
            raise ValueError(
                f"Input tensor has shape {img_lr.shape[1]} channels, "
                f"but model expects {self.img_in_channels} channels"
                f"Note coord_features adds 2 extra channels if enabled."
                f" Expected spatial dimensions {expected_hw}, "
            )

        img_lr = img_lr.float()
        # lr_features = self.feature_extractor(img_lr)
        # if img_lr is not None:
        #     x = torch.cat((x, img_lr), dim=1) 
        # # F_x = self.model(x)
        
        device_type = "cuda" if img_lr.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            F_x = self.model(img_lr)
            # F_x = self.model(lr_features)
                            
      
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
        
        log_dict = {"train_loss_epoch": loss}
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
        
        val_log_dict = {"val_loss": val_loss}
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
        mse_all = torch.mean((predictions - ground_truth) ** 2)
        mae_all = torch.mean(torch.abs(predictions - ground_truth))
        rmse_all = torch.sqrt(mse_all)
        data_range = (ground_truth.max() - ground_truth.min()).item()
        ssim_all = ssim_func(predictions, ground_truth, data_range=data_range)

        var_names = ['u10', 'v10', 't2m', 'sshf', 'zust']
        # mse_vars, mae_vars, rmse_vars, ssim_vars = {}, {}, {}, {}
        log_metrics = {}

        for i, var_name in enumerate(var_names):
            pred_i = predictions[:, i, :, :]
            gt_i = ground_truth[:, i, :, :]
            mse_val = torch.mean((pred_i - gt_i) ** 2)
            log_metrics[f"test_mse_{var_name}"] = mse_val
            log_metrics[f"test_mae_{var_name}"] = torch.mean(torch.abs(pred_i - gt_i))
            log_metrics[f"test_rmse_{var_name}"] = torch.sqrt(mse_val)
            data_range_i = (gt_i.max() - gt_i.min()).item()
            log_metrics[f"test_ssim_{var_name}"] = ssim_func(
                pred_i.unsqueeze(1), gt_i.unsqueeze(1), data_range=data_range_i
            )

        # log_metrics = {
        #     "test_mse": mse_all, "test_mae": mae_all,
        #     "test_rmse": rmse_all, "test_ssim": ssim_all,
        # }
        # log_metrics.update(mse_vars)
        # log_metrics.update(mae_vars)
        # log_metrics.update(rmse_vars)
        # log_metrics.update(ssim_vars)

        self.log_dict(log_metrics, prog_bar=False, on_epoch=True, sync_dist=True)
        self.plot_preds(predictions, ground_truth, img_lr, diz_stats)
        return log_metrics

    # def test_step(self, batch, batch_idx):
    #     # compute forward pass, compute MSE, MAE, plot PSD, and plot some examples
    #     target, x = batch 
    #     D_yn = self(x, force_fp32=False)
    #     mse = F.mse_loss(D_yn, target)
    #     mae = F.l1_loss(D_yn, target)
    #     self.log("test_mse", mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    #     self.log("test_mae", mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    #     return {"test_mse": mse.detach(), "test_mae": mae.detach()}

    # def on_test_epoch_end(self):
    #     """Print aggregated test metrics (already reduced by Lightning)."""
    #     if self.trainer.is_global_zero:
    #         mse = self.trainer.callback_metrics.get("test_mse", None)
    #         mae = self.trainer.callback_metrics.get("test_mae", None)
    #         if mse is not None and mae is not None:
    #             print(f"Test MSE (epoch): {mse:.4f}, Test MAE (epoch): {mae:.4f}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

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