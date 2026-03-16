"""
Backward compatibility layer for configuration management.
"""

from typing import Dict, Any
from .base_config import (
    ExperimentConfig, 
    FNOModelConfig, 
    UNetModelConfig, 
    DiffusionModelConfig,
    UNOModelConfig,
    YangModelConfig,
    AFNOModelConfig,
)


def convert_legacy_config_to_new(legacy_args) -> ExperimentConfig:
    """Convert legacy argument parser namespace to new configuration format."""
    config_dict = {}
    exclude_args = {'config', 'use_new_config', 'validate_config'}
    
    for key, value in vars(legacy_args).items():
        if value is not None and key not in exclude_args:
            config_dict[key] = value
    
    return ExperimentConfig.from_dict(config_dict)


def create_legacy_args_from_config(config: ExperimentConfig):
    """Create legacy argument namespace from new configuration."""
    
    class LegacyArgs:
        def __init__(self, config: ExperimentConfig):
            # Dataset args
            self.dataset_cerra = config.dataset.cerra_path
            self.dataset_era5 = config.dataset.era5_path
            self.img_in_channels = config.dataset.img_in_channels
            self.img_out_channels = config.dataset.img_out_channels
            self.img_resolution = config.dataset.img_resolution
            self.subset_ds = config.dataset.subset_size or 0
            
            # Model args - common
            self.model = config.model.model_type
            self.model_type = getattr(config, 'model_type', None)
            self.checkpoint_level = config.model.checkpoint_level
            
            # Model-specific args - set based on model type
            self._set_model_specific_args(config.model)
            
            # Training args
            self.lr = config.training.lr
            self.epochs = config.training.epochs
            self.batch_size = config.training.batch_size
            self.val_interval = config.training.val_interval
            self.precision = config.training.precision
            self.n_workers = config.training.n_workers
            self.seed = config.training.seed
            self.lr_decay = config.training.lr_decay
            self.lr_rampup = config.training.lr_rampup
            self.grad_clip_threshold = config.training.grad_clip_threshold
            self.anneal_epochs = config.training.anneal_epochs
            self.init_lambda = config.training.init_lambda
            self.max_lambda = config.training.max_lambda
            self.lambda_psd = config.training.lambda_psd
            self.loss_type = config.training.loss_type
            
            # Other args
            self.wandb_project = config.wandb_project
            self.run_name = config.run_name
            self.savepreds_path = config.save_preds_path
            self.save_preds_path = config.save_preds_path
            self.load = config.load
            self.restore_opt = config.restore_opt
            self.eval = config.eval
            self.resume = config.resume
            self.regression_net = config.regression_net
            self.gridtype = config.gridtype
            self.hr_mean_conditioning = config.hr_mean_conditioning
            self.num_ensembles = config.num_ensembles
            self.output_variables = config.output_variables
        
        def _set_model_specific_args(self, model_config):
            """Set model-specific arguments based on model type."""
            
            if isinstance(model_config, FNOModelConfig):
                self.num_fno_layers = model_config.num_fno_layers
                self.fno_layer_size = model_config.fno_layer_size
                self.num_fno_modes = model_config.num_fno_modes
                self.fno_padding = model_config.fno_padding
                self.coord_features = model_config.coord_features
                self.decoder_layers = model_config.decoder_layers
                self.decoder_layer_size = model_config.decoder_layer_size
                # Set UNet params to None for FNO
                self.model_channels = None
                self.channel_mult = None
                self.attn_resolutions = None
                self.embedding_type = None
                self.N_grid_channels = None
                
            elif isinstance(model_config, (UNetModelConfig, DiffusionModelConfig)):
                self.model_channels = model_config.model_channels
                self.channel_mult = model_config.channel_mult
                self.attn_resolutions = model_config.attn_resolutions
                self.embedding_type = model_config.embedding_type
                self.N_grid_channels = model_config.n_grid_channels
                self.model_type = getattr(model_config, 'network_type', 'SongUNetPosEmbd')
                # Set FNO params to None for UNet/Diffusion
                self.num_fno_layers = None
                self.fno_layer_size = None
                self.num_fno_modes = None
                self.fno_padding = None
                self.coord_features = None
                self.decoder_layers = None
                self.decoder_layer_size = None
                
                # Diffusion-specific
                if isinstance(model_config, DiffusionModelConfig):
                    self.num_diffusion_steps = model_config.num_diffusion_steps
                    self.noise_schedule = model_config.noise_schedule
                    self.beta_start = model_config.beta_start
                    self.beta_end = model_config.beta_end

            elif isinstance(model_config, UNOModelConfig):
                self.hidden_channels = model_config.hidden_channels
                self.projection_channels = model_config.projection_channels
                self.lifting_channels = model_config.lifting_channels
                self.uno_out_channels = model_config.uno_out_channels
                self.uno_n_modes = model_config.uno_n_modes
                self.uno_scalings = model_config.uno_scalings
                self.positional_embedding = model_config.positional_embedding
                self.horizontal_skips_map = model_config.horizontal_skips_map
                self.channel_mlp_skip = model_config.channel_mlp_skip
                self.n_layers = model_config.n_layers
                # Set FNO and Diffusion params to None for UNO
                self.num_fno_layers = None
                self.fno_layer_size = None
                self.num_fno_modes = None
                self.fno_padding = None
                self.coord_features = None
                self.decoder_layers = None
                self.decoder_layer_size = None
                self.model_channels = None
                self.channel_mult = None
                self.attn_resolutions = None
                self.embedding_type = None
                self.N_grid_channels = None

            elif isinstance(model_config, YangModelConfig):
                self.n_channels = model_config.n_channels
                self.n_residual_blocks = model_config.n_residual_blocks
                self.n_operator_blocks = model_config.n_operator_blocks
                self.modes = model_config.modes
                self.apply_constraint = model_config.apply_constraint
                # Set other model params to None for DSFNO
                self.num_fno_layers = None
                self.fno_layer_size = None
                self.num_fno_modes = None
                self.fno_padding = None
                self.coord_features = None
                self.decoder_layers = None
                self.decoder_layer_size = None
                self.model_channels = None
                self.channel_mult = None
                self.attn_resolutions = None
                self.embedding_type = None
                self.N_grid_channels = None
                self.hidden_channels = None
                self.projection_channels = None
                self.lifting_channels = None
                self.uno_out_channels = None
                self.uno_n_modes = None
                self.uno_scalings = None
                self.positional_embedding = None
                self.horizontal_skips_map = None
                self.channel_mlp_skip = None
                self.n_layers = None

            elif isinstance(model_config, AFNOModelConfig):
                self.afno_patch_size = model_config.afno_patch_size
                self.afno_embed_dim = model_config.afno_embed_dim
                self.afno_depth = model_config.afno_depth
                self.afno_mlp_ratio = model_config.afno_mlp_ratio
                self.afno_drop_rate = model_config.afno_drop_rate
                self.afno_num_blocks = model_config.afno_num_blocks
                self.afno_sparsity_threshold = model_config.afno_sparsity_threshold
                self.afno_hard_thresholding_fraction = model_config.afno_hard_thresholding_fraction
                # Set other model params to None for AFNO
                self.num_fno_layers = None
                self.fno_layer_size = None
                self.num_fno_modes = None
                self.fno_padding = None
                self.coord_features = None
                self.decoder_layers = None
                self.decoder_layer_size = None
                self.model_channels = None
                self.channel_mult = None
                self.attn_resolutions = None
                self.embedding_type = None
                self.N_grid_channels = None
                self.hidden_channels = None
                self.projection_channels = None
                self.lifting_channels = None
                self.uno_out_channels = None
                self.uno_n_modes = None
                self.uno_scalings = None
                self.positional_embedding = None
                self.horizontal_skips_map = None
                self.channel_mlp_skip = None
                self.n_layers = None
    
    return LegacyArgs(config)
