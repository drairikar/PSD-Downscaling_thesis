import torch
import pytorch_lightning as pl
import importlib
import wandb
import matplotlib.pyplot as plt
from .. import constants
from ..utils import vis
from typing import Callable, Optional, Tuple
import random
from torchmetrics.functional import structural_similarity_index_measure as ssim_func
import os
import numpy as np
from scipy.fft import fft
import math
import torch.nn.functional as F
from pathlib import Path
from .losses.fourier_losses import FourierLossETH, FourierLossDelft, FourierLossHK, FourierLossCarlo
from ..utils.utils import compute_psd_2d, compute_derived_fields

network_module = importlib.import_module("physicsnemo.models.diffusion")


class UNetSequenceWrapper(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        
              
        # for compatibility with older versions that took only 1 dimension
        if isinstance(args.img_resolution, int):
            self.img_shape_x = self.img_shape_y = args.img_resolution
        else:
            self.img_shape_y = args.img_resolution[0]
            self.img_shape_x = args.img_resolution[1]

        self.img_in_channels = args.img_in_channels
        self.frame_channels = int(args.img_out_channels)
        self.sequence_length = int(args.sequence_length)
        self.local_window = int(getattr(args, "local_window", 3))  # default to 3 if not provided
        if self.local_window < 1 or self.local_window % 2 == 0:
            raise ValueError(
                f"local_window must be a positive odd integer, got {self.local_window}."
            )
        if self.local_window > self.sequence_length:
            raise ValueError(
                f"local_window ({self.local_window}) cannot be greater than sequence_length ({self.sequence_length})."
            )
        
        self.trajectory_channels = self.sequence_length * self.frame_channels
        self.local_channels = self.local_window * self.frame_channels
        self.orography_channels = 1
        self.network_in_channels = (
            self.local_channels + self.orography_channels
        )

        self.lr = args.lr
        self.wandb_project = args.wandb_project
        self.savepreds_path = args.savepreds_path
        self.load = args.load
        self.output_size = (self.img_shape_y, self.img_shape_x)
        
        self.model_kwargs = {
            'checkpoint_level': args.checkpoint_level,
            'N_grid_channels': args.N_grid_channels,
            'embedding_type': args.embedding_type,
            'model_channels': args.model_channels,
            'channel_mult': args.channel_mult,
            'attn_resolutions': args.attn_resolutions,
        }

        model_class = getattr(network_module, args.model_type)
        self.model = model_class(
            img_resolution=args.img_resolution,
            in_channels=self.network_in_channels,
            out_channels=self.local_channels,
            checkpoint_level=args.checkpoint_level,
            embedding_type=args.embedding_type,
            model_channels=args.model_channels,
            channel_mult=args.channel_mult,
            attn_resolutions=args.attn_resolutions,
        )
        
        loss_ditc = {
            "FourierLossETH": FourierLossETH,
            "FourierLossDelft": FourierLossDelft,
            "FourierLossHK": FourierLossHK,
            "FourierLossCarlo": FourierLossCarlo
        }
        
        loss_func = loss_ditc[args.loss_type]() if args.loss_type in loss_ditc else None

        self.loss_fn = RegressionLoss(args.init_lambda, args.max_lambda, args.anneal_epochs, loss_func)

        # Accumulate squared errors in physical space and take the square root
        # only once at the end of testing. This makes RMSE independent of test
        # batch boundaries and supports correct distributed aggregation.
        self.register_buffer(
            "_rmse_sse",
            torch.zeros(self.frame_channels, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_rmse_count",
            torch.zeros(self.frame_channels, dtype=torch.float64),
            persistent=False,
        )

    def set_normalization_stats(
        self,
        target_mean,
        target_std,
        guidance_mean,
        guidance_std,
    ):
        """Attach dataset normalization statistics used during evaluation."""

        def _as_sequence_stat(value, name):
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if tensor.ndim != 1 or tensor.numel() != self.frame_channels:
                raise ValueError(
                    f"{name} must contain one value per frame channel "
                    f"({self.frame_channels}), got shape {tuple(tensor.shape)}."
                )
            return tensor.reshape(1, 1, self.frame_channels, 1, 1)

        sequence_std = _as_sequence_stat(target_std, "target_std")
        guidance_sequence_std = _as_sequence_stat(
            guidance_std, "guidance_std"
        )
        if torch.any(sequence_std <= 0) or torch.any(guidance_sequence_std <= 0):
            raise ValueError("Normalization standard deviations must be positive.")

        # These values belong to the runtime dataset rather than the learned
        # model, so do not persist them in checkpoints.
        self.register_buffer(
            "sequence_mean",
            _as_sequence_stat(target_mean, "target_mean"),
            persistent=False,
        )
        self.register_buffer("sequence_std", sequence_std, persistent=False)
        self.register_buffer(
            "guidance_sequence_mean",
            _as_sequence_stat(guidance_mean, "guidance_mean"),
            persistent=False,
        )
        self.register_buffer(
            "guidance_sequence_std",
            guidance_sequence_std,
            persistent=False,
        )

    def _prepare_sequence(self, tensor):
        """
        [B, T, C, H, W] -> [B, T*C, H, W]
        """
        if tensor.ndim == 5:
            batch, time, channels, height, width = tensor.shape

            if time != self.sequence_length:
                raise ValueError(
                    f"Expected sequence length {self.sequence_length}, "
                    f"but got {time}."
                )
            if channels != self.frame_channels:
                raise ValueError(
                    f"Expected frame channels {self.frame_channels}, "
                    f"but got {channels}."
                )

            return tensor.flatten(1,2)

        if tensor.ndim == 4:
            if tensor.shape[1] != self.sequence_channels:
                raise ValueError(
                    f"Expected sequence channels {self.sequence_channels}, "
                    f"but got {tensor.shape[1]}."
                )
            
            return tensor

        raise ValueError(
            f"Expected tensor of shape [B, T, C, H, W] or [B, T*C, H, W], "
            f"but got {tensor.shape}."
        )

    
    def _restore_sequence(self, tensor):
        """
        [B, T*C, H, W] -> [B, T, C, H, W]
        """

        if tensor.ndim!= 4:
            raise ValueError(
                f"Expected tensor of shape [B, T*C, H, W], "
                f"but got {tensor.shape}."
            )

        return tensor.unflatten(
            dim=1,
            sizes = (self.sequence_length, self.frame_channels)
        )

    

    def forward(self, guidance, orography):

        guidance = self._prepare_sequence(guidance).float()
        orography = orography.float()

        if guidance.shape[-2:]!= self.output_size:
            guidance = F.interpolate(guidance, size= self.output_size, mode="bicubic", align_corners=False)

        if orography.shape[-2:]!= self.output_size:
            raise ValueError(
                f"Orography has shape {orography.shape[-2:]} expected {self.output_size}." )

        if orography.shape[1] != 1:
            raise ValueError(
                f"Orography has {orography.shape[1]} channels, expected 1."
            )

        model_input = torch.cat([guidance, orography.to(guidance.dtype)], dim=1)

        if model_input.shape[1] != self.network_in_channels:
            raise ValueError(
                f"Model input has {model_input.shape[1]} channels, "
                f"expected {self.network_in_channels}."
            )

        noise_labels = torch.zeros(
            model_input.shape[0],
            device = model_input.device,
        )

        H, W = model_input.shape[-2:]
        pad_h = (-H) % 32
        pad_w = (-W) % 32
        model_input = F.pad(
            model_input,
            (0, pad_w, 0, pad_h),
            mode="reflect",
        )

        prediction = self.model(model_input, noise_labels, class_labels=None)
        prediction = prediction[..., :H, :W]

        return prediction.float()

    def training_step(self, batch, *args):

        cerra_target, orography, era5 = batch
        cerra_target = self._prepare_sequence(cerra_target).float()

        prediction = self(
            guidance=era5,
            orography=orography
        )

        # squared_error = (prediction - cerra_target) ** 2
        loss = F.mse_loss(prediction, cerra_target)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        return loss

    def validation_step(self, batch, *args):

        cerra_target, orography, era5 = batch
        cerra_target = self._prepare_sequence(cerra_target).float()

        prediction = self(
            guidance=era5,
            orography=orography
        )

        loss = F.mse_loss(prediction, cerra_target)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        return loss

    def on_test_epoch_start(self):
        self._rmse_sse.zero_()
        self._rmse_count.zero_()
        self._test_preds = []
        self._test_targets = []
              
    
    def test_step(self, batch, batch_idx: int):

        """
        Evaluate model on a test batch, save un-normalized predictions if requested,
        and compute metrics on the un-normalized data.
        """

        cerra_target, orography, era5 = batch
        cerra_target = self._prepare_sequence(cerra_target).float()
        prediction = self(
            guidance=era5,
            orography=orography
        )

        pred_sequence = self._restore_sequence(prediction)
        target_sequence = self._restore_sequence(cerra_target)

        era5_sequence = self._restore_sequence(self._prepare_sequence(era5))

        if not hasattr(self, "sequence_mean"):
            raise RuntimeError(
                "Evaluation normalization statistics are not configured. "
                "Call set_normalization_stats() with the evaluation dataset "
                "statistics before trainer.test()."
            )

        mean = self.sequence_mean.to(
            device=pred_sequence.device,
            dtype=pred_sequence.dtype,
        )
        std = self.sequence_std.to(
            device=pred_sequence.device,
            dtype=pred_sequence.dtype,
        )

        pred_physical = pred_sequence * std + mean
        target_physical = target_sequence * std + mean

        guidance_mean = self.guidance_sequence_mean.to(
            device=pred_sequence.device,
            dtype=pred_sequence.dtype,
        )

        guidance_std = self.guidance_sequence_std.to(
            device=pred_sequence.device,
            dtype=pred_sequence.dtype,
        )

        era5_physical = era5_sequence * guidance_std + guidance_mean

        # Sum errors over batch, time, and space, leaving one value per
        # physical variable. Accumulate in float64 for numerical stability.
        with torch.no_grad():
            squared_error = (
                pred_physical.double() - target_physical.double()
            ).square()
            batch_sse = squared_error.sum(dim=(0, 1, 3, 4))
            batch_count = (
                pred_physical.shape[0]
                * pred_physical.shape[1]
                * pred_physical.shape[3]
                * pred_physical.shape[4]
            )
            self._rmse_sse += batch_sse
            self._rmse_count += torch.full_like(
                self._rmse_count,
                float(batch_count),
            )

        if self.savepreds_path and self.trainer.is_global_zero:
            plot_predictions = pred_physical.flatten(0,1)
            plot_targets = target_physical.flatten(0,1)
            plot_era5 = era5_sequence.flatten(0,1)

            diz_stats = {
                "mean_era5": torch.squeeze(
                    torch.as_tensor(
                        self.guidance_sequence_mean,
                        device=era5_sequence.device,
                        dtype=era5_sequence.dtype,
                    ),
                    dim=1,
                ),
                "std_era5": torch.squeeze(
                    torch.as_tensor(
                        self.guidance_sequence_std,
                        device=era5_sequence.device,
                        dtype=era5_sequence.dtype,
                    ),
                    dim=1,
                ),
            }

            self.plot_preds(
                plot_predictions,
                plot_targets,
                plot_era5,
                diz_stats,
            )

        if self.trainer.is_global_zero:
            self._test_preds.append(
                pred_physical.detach().float().cpu().flatten(0,1).unsqueeze(0)

            )
            self._test_targets.append(
                target_physical.detach().float().cpu().flatten(0,1)
            )

    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt
            
            
    def load_metrics_and_plots(self, prediction, high_res, batch_idx, mask=None):
        
        #reshap from (B, C, H, W) to (B, num_grid_nodes, C)
        prediction = prediction.permute(0, 2, 3, 1).flatten(1, 2)
        high_res = high_res.permute(0, 2, 3, 1).flatten(1, 2)
        
        if mask is None:
            mask = torch.ones_like(high_res[:, :, 0])
        
        # Plot samples
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
                
        # var_i = random.randint(0, len(constants.PARAM_NAMES_SHORT_CERRA) - 1)
        # var_name = constants.PARAM_NAMES_SHORT_CERRA[var_i]
        # var_unit = constants.PARAM_UNITS_CERRA[var_i]
        
        sample = random.randint(0, prediction.shape[0] - 1)

        pred_states = prediction[
            sample, :, var_i
        ]  # (S, num_grid_nodes)
        
        target_state = high_res[
            sample, :, var_i
        ]  # (num_grid_nodes,)

        plot_title = (
            f"{var_name} ({var_unit})"
        )

        # Make plots
        log_plot_dict[
            f"pred_{var_name}"
        ] = vis.plot_ensemble_prediction(
            pred_states,
            target_state,
            obs_mask = mask[sample],
            title=f"{plot_title} (prior)",
        )

        if not self.trainer.sanity_checking:
            # Log all plots to wandb
            wandb.log(log_plot_dict)

        plt.close("all")   
        
        
    def plot_preds(self, prediction, high_res, img_lr, diz_stats):
        """
        Plot one random sample from the batch (single variable):
        [low-res input, high-res target, prediction, residual].
        Here, `prediction` and `high_res` are already un-normalized. We only
        need to un-normalize `img_lr` for plotting.
        """
        # If you need statistics to un‐normalize img_lr for plotting:
        low_res_mean = diz_stats["mean_era5"]
        low_res_std  = diz_stats["std_era5"]

        # Un‐normalize img_lr before plotting
        img_lr = img_lr * low_res_std + low_res_mean

        # Select one sequence frame/sample, then plot every physical variable
        # for that same sample so evaluation always produces a complete set.
        sample_idx = random.randint(0, prediction.shape[0] - 1)
        var_names = ["u10", "v10", "t2m"]
        var_units = ["m/s", "m/s", "K"]
        if prediction.shape[1] != len(var_names):
            raise ValueError(
                f"Expected {len(var_names)} output variables, "
                f"but got {prediction.shape[1]}."
            )

        run_name = self.load.split("/")[-2]
        save_dir = os.path.join(
            self.savepreds_path,
            run_name,
            "pred_plots_sequence",
        )
        os.makedirs(save_dir, exist_ok=True)

        for var_i, (var_name, var_unit) in enumerate(zip(var_names, var_units)):
            # Extract 2D images for plotting (B, C, H, W → H, W).
            input_img = img_lr[sample_idx, var_i].detach().cpu().numpy()
            target_img = high_res[sample_idx, var_i].detach().cpu().numpy()
            pred_img = prediction[sample_idx, var_i].detach().cpu().numpy()
            residual_img = target_img - pred_img

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            for axis, image, title in zip(
                axes,
                [input_img, target_img, pred_img, residual_img],
                ["Input", "Target", "Prediction", "Residual"],
            ):
                axis.imshow(image, cmap="plasma", origin="lower")
                axis.set_title(title)
                axis.axis("off")

            fig.suptitle(f"{var_name} ({var_unit})", fontsize=16)
            fname = os.path.join(
                save_dir,
                f"{var_name}_sample_{sample_idx}.png",
            )
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)


    def _log_psd_scalars(self, k, psd_values, region, var_name, source_label):
        """Log PSD as scalars in log10 space with wavenumber as custom x-axis."""
        k_np = np.asarray(k[1:], dtype=np.float64)
        psd_np = np.asarray(psd_values[1:], dtype=np.float64)

        metric_key = f"psd/{region}/{var_name}/{source_label}"
        x_key = f"psd/{region}/{var_name}/log10_wavenumber"
        wandb.define_metric(x_key, hidden=True)
        wandb.define_metric(metric_key, step_metric=x_key)

        log_k = np.log10(k_np)
        log_psd = np.log10(psd_np)

        for ki, pi in zip(log_k, log_psd):
            wandb.log({x_key: float(ki), metric_key: float(pi)})

    
    @staticmethod
    def _save_psd_plot(
        k_tgt,
        psd_tgt,
        k_pred,
        psd_pred,
        region,
        var_name,
        run_name,
    ):
        """Save a log-log comparison of target and predicted PSDs."""
        k_tgt = np.asarray(k_tgt, dtype=np.float64)
        psd_tgt = np.asarray(psd_tgt, dtype=np.float64)
        k_pred = np.asarray(k_pred, dtype=np.float64)
        psd_pred = np.asarray(psd_pred, dtype=np.float64)

        if k_tgt.shape != psd_tgt.shape:
            raise ValueError(
                f"Target wavenumber and PSD shapes differ: "
                f"{k_tgt.shape} != {psd_tgt.shape}."
            )
        if k_pred.shape != psd_pred.shape:
            raise ValueError(
                f"Prediction wavenumber and PSD shapes differ: "
                f"{k_pred.shape} != {psd_pred.shape}."
            )

        valid_tgt = (
            np.isfinite(k_tgt)
            & np.isfinite(psd_tgt)
            & (k_tgt > 0)
            & (psd_tgt > 0)
        )
        valid_pred = (
            np.isfinite(k_pred)
            & np.isfinite(psd_pred)
            & (k_pred > 0)
            & (psd_pred > 0)
        )
        if not valid_tgt.any() or not valid_pred.any():
            raise ValueError(
                f"Cannot plot PSD for {var_name}: no positive finite values."
            )

        save_dir = Path("psd_plots_Unet_seq") / run_name / "psd" / region
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{var_name}.png"

        fig, ax = plt.subplots()
        ax.plot(np.log10(k_tgt[valid_tgt]), np.log10(psd_tgt[valid_tgt]), label="CERRA", linewidth=2)
        ax.plot(np.log10(k_pred[valid_pred]), np.log10(psd_pred[valid_pred]), label=run_name, linewidth=2)
        ax.set_xlabel("log10(Wavenumber)")
        ax.set_ylabel("log10(PSD)")
        ax.set_title(f"PSD Comparison for {var_name} ({region})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    

    def on_test_epoch_end(self):
        region = getattr(self, "current_region", "unknown")

        sse = self._rmse_sse.clone()
        count = self._rmse_count.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sse, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)

        if torch.any(count <= 0):
            raise RuntimeError("Cannot compute test RMSE without evaluated samples.")

        rmse_per_var = torch.sqrt(sse / count)
        # These names match CerraPriorDatasetSequence.cerra_vars and the
        # GeoDiff sequence metric namespace.
        var_names = ["u10", "v10", "t2m"][:rmse_per_var.numel()]
        for var_idx, var_name in enumerate(var_names):
            self.log(
                f"test/{region}/rmse/{var_name}",
                rmse_per_var[var_idx].float(),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

        overall_rmse = torch.sqrt(sse.sum() / count.sum())
        self.log(
            f"test/{region}/rmse",
            overall_rmse.float(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        
        if self.trainer.is_global_zero and len(self._test_preds) > 0:
            all_preds = torch.cat(self._test_preds, dim=1)    # (n_ens, N_total, C, H, W)
            all_targets = torch.cat(self._test_targets, dim=0) # (N_total, C, H, W)
            n_ens = all_preds.shape[0]

            # var_names = constants.PARAM_NAMES_SHORT_CERRA
            var_names = ["u10", "v10", "t2m"]
            run_name = wandb.run.name if wandb.run else "prediction"

            if self.wandb_project is not None and not self.trainer.sanity_checking:
                # PSD for raw variables — mean of individual member PSDs
                for var_i, var_name in enumerate(var_names):
                    psd_members = []
                    for m in range(n_ens):
                        k_pred, psd_m = compute_psd_2d(all_preds[m, :, var_i, :, :])
                        psd_members.append(psd_m)
                    psd_pred = np.mean(psd_members, axis=0)
                    k_tgt, psd_tgt = compute_psd_2d(all_targets[:, var_i, :, :])

                    if self.wandb_project is not None:
                        self._log_psd_scalars(k_tgt, psd_tgt, region, var_name, "CERRA")
                        self._log_psd_scalars(k_pred, psd_pred, region, var_name, run_name)
                    self._save_psd_plot(
                        k_tgt, psd_tgt, k_pred, psd_pred,
                        region, var_name, run_name,
                    )

                # PSD for derived quantities — mean of individual member PSDs
                for derived_name in constants.DERIVED_VAR_NAMES:
                    psd_members = []
                    for m in range(n_ens):
                        pred_derived_m = compute_derived_fields(all_preds[m], var_names)
                        k_pred, psd_m = compute_psd_2d(pred_derived_m[derived_name])
                        psd_members.append(psd_m)
                    psd_pred = np.mean(psd_members, axis=0)

                    tgt_derived = compute_derived_fields(all_targets, var_names)
                    k_tgt, psd_tgt = compute_psd_2d(tgt_derived[derived_name])

                    if self.wandb_project is not None:
                        self._log_psd_scalars(k_tgt, psd_tgt, region, derived_name, "CERRA")
                        self._log_psd_scalars(k_pred, psd_pred, region, derived_name, run_name)
                    self._save_psd_plot(
                        k_tgt, psd_tgt, k_pred, psd_pred,
                        region, derived_name, run_name,
                    )

        # Clear accumulators for next region
        self._test_preds.clear()
        self._test_targets.clear()

    
class RegressionLoss:
    """
    Regression loss function for the deterministic predictions.
    Note: this loss does not apply any reduction.

    Attributes
    ----------
    sigma_data: float
        Standard deviation for data. Deprecated and ignored.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(self, 
                 init_lambda: float, 
                 max_lambda: float, 
                 anneal_epochs: int, 
                 loss_func: Optional[Callable] = None,
                 eps: float = 1e-12):
        
        self.init_lambda = init_lambda
        self.max_lambda = max_lambda
        self.anneal_epochs = anneal_epochs
        self.loss_func = loss_func
        self.eps = eps                         # to avoid log(0)
        
    
    # @staticmethod
    # def psd2d(a: torch.Tensor, *, dx: float, dy: float,
    #         eps: float = 1e-12) -> torch.Tensor:
    #     H, W = a.shape[-2:]
    #     fft   = torch.fft.rfftn(a, dim=(-2, -1)) / (H * W)
    #     psd   = 2.0 * (fft.real**2 + fft.imag**2)
    #     return psd.clamp_min(eps)     
    
    @staticmethod
    def get_psd_torch(a: torch.Tensor, *, dx: float, dim: int = -1
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        a   : (..., N)   real-valued signal
        dx  : grid spacing (same units as the data’s physical dimension)
        dim : dimension over which to take the FFT
        """
        N  = a.shape[dim]
        dk = 2 * math.pi / dx                      # angular‐wavenumber step
        k  = torch.arange(0, N // 2, device=a.device, dtype=a.dtype) * dk / N

        v_ft  = torch.fft.rfft(a, dim=dim)         # shape (..., N//2+1)
        v_ft  = v_ft.narrow(dim, 0, N // 2)        # drop Nyquist to match k
        psd   = (v_ft.real.square() + v_ft.imag.square()) / (N * dx)

        return k, psd  

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: torch.Tensor,
        img_lr: torch.Tensor,
        current_epoch: int,
        augment_pipe: Optional[
            Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]
        ] = None,
    ) -> torch.Tensor:
        """
        Calculate and return the regression loss for
        deterministic predictions.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network model that will make predictions.
            Expected signature: `net(x, img_lr,
            augment_labels=augment_labels, force_fp32=False)`, where:
                x (torch.Tensor): Tensor of shape (B, C_hr, H, W). Is zero-filled.
                img_lr (torch.Tensor): Low-resolution input of shape (B, C_lr, H, W)
                augment_labels (torch.Tensor, optional): Optional augmentation
                labels, returned by `augment_pipe`.
                force_fp32 (bool, optional): Whether to force the model to use
                fp32, by default False.
            Returns:
                torch.Tensor: Predictions of shape (B, C_hr, H, W)

        img_clean : torch.Tensor
            High-resolution input images of shape (B, C_hr, H, W).
            Used as ground truth and for data augmentation if 'augment_pipe' is provided.

        img_lr : torch.Tensor
            Low-resolution input images of shape (B, C_lr, H, W).
            Used as input to the neural network.

        augment_pipe : callable, optional
            An optional data augmentation function.
            Expected signature:
                img_tot (torch.Tensor): Concatenated high and low resolution
                    images of shape (B, C_hr+C_lr, H, W)
            Returns:
                Tuple[torch.Tensor, Optional[torch.Tensor]]:
                    - Augmented images of shape (B, C_hr+C_lr, H, W)
                    - Optional augmentation labels

        Returns
        -------
        torch.Tensor
            A tensor representing the per-sample element-wise squared
            difference between the network's predictions and the high
            resolution images `img_clean` (possibly data-augmented by
            `augment_pipe`).
            Shape: (B, C_hr, H, W), same as `img_clean`.
        """

        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]

        zero_input = torch.zeros_like(y, device=img_clean.device)
        D_yn = net(zero_input, y_lr, force_fp32=False, augment_labels=augment_labels)
        # loss_mse = torch.mean((D_yn - y) ** 2)
        
        # ─── PSD loss ────────────────────────────────────────────────
        # psd_pred = self.psd2d(
        #     D_yn, dx=5.5, dy=5.5)
        # psd_true = self.psd2d(
        #     y,   dx=5.5, dy=5.5)
        
        # k, psd_pred = self.get_psd_torch(
        #     D_yn.permute(0, 2, 3, 1), dx=5.5, dim=2)
        # _, psd_true = self.get_psd_torch(
        #     y.permute(0, 2, 3, 1),   dx=5.5, dim=2)
        
        # freq_weights = (k / k.max()).pow(2)  # shape: (F,)
        # w = freq_weights.view(1, 1, -1, 1)  

        # RMSE between log-PSDs (scalar)
        # diff_log_psd = torch.log(psd_pred + self.eps) - torch.log(psd_true + self.eps)
        # psd_loss = torch.sqrt(torch.mean(w * diff_log_psd ** 2))
        # psd_loss = torch.sqrt(torch.mean(diff_log_psd ** 2))
        
        if self.loss_func is not None:
            term_space, term_amp = self.loss_func(D_yn, y)
            lambda_psd = min(self.init_lambda + (self.max_lambda - self.init_lambda) * (current_epoch / self.anneal_epochs)**2, self.max_lambda)
            loss_amp = lambda_psd * term_amp
            loss_space = term_space
            loss = loss_space + loss_amp
        else:
            loss = torch.mean((D_yn - y) ** 2)
            loss_space = loss #MSE
            loss_amp = torch.tensor(0.0, device=D_yn.device) #just zero
            

        return loss, y, D_yn, loss_space, loss_amp #, lambda_psd, lambda_psd * psd_loss
    
