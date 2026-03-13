"""
Base configuration classes with validation and type safety.

This module provides structured configuration management that maintains
backward compatibility with existing YAML configurations while adding
validation, type safety, and better organization.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import yaml
import torch
from enum import Enum


class ModelType(Enum):
    """Supported model types."""
    UNET_CNN = "UNet-CNN"
    DIFFUSION = "Diffusion"
    FNO = "FNO"
    UNO = "UNO"
    DSFNO = "DSFNO"

class PrecisionType(Enum):
    """Supported precision types."""
    FP32 = "32"
    FP16 = "16"
    FP16_MIXED = "16-mixed"
    BF16 = "bf16"

@dataclass
class DatasetConfig:
    """Dataset configuration with validation."""
    cerra_path: str
    era5_path: str
    img_in_channels: int = 5
    img_out_channels: int = 5
    img_resolution: List[int] = field(default_factory=lambda: [300, 300])
    subset_size: Optional[int] = None
    standardize: bool = True
        
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate dataset configuration."""
        if not Path(self.cerra_path).exists():
            raise ValueError(f"CERRA path does not exist: {self.cerra_path}")
        if not Path(self.era5_path).exists():
            raise ValueError(f"ERA5 path does not exist: {self.era5_path}")
        
        if len(self.img_resolution) != 2:
            raise ValueError("img_resolution must be a list of 2 integers [height, width]")
        if any(dim <= 0 for dim in self.img_resolution):
            raise ValueError("img_resolution dimensions must be positive")
        
        if self.img_in_channels <= 0 or self.img_out_channels <= 0:
            raise ValueError("Channel counts must be positive")


# ============================================================================
# MODEL-SPECIFIC CONFIGURATIONS
# ============================================================================

@dataclass
class BaseModelConfig:
    """Base model configuration - common to all models."""
    model_type: str = "FNO"
    checkpoint_level: int = 0
    
    def _validate_base(self):
        """Validate base model configuration."""

        if self.checkpoint_level < 0:  
            raise ValueError("checkpoint_level must be non-negative")

        # try:
        #     ModelType(self.model_type)
        # except ValueError:
        #     raise ValueError(f"Unsupported model type: {self.model_type}")


@dataclass
class FNOModelConfig(BaseModelConfig):
    """FNO-specific model configuration."""
    model_type: str = "FNO"
    num_fno_layers: int = 4
    fno_layer_size: int = 64
    num_fno_modes: int = 16
    fno_padding: int = 8
    coord_features: bool = True
    decoder_layers: int = 1
    decoder_layer_size: int = 32
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        self._validate_base()
        if self.num_fno_layers <= 0:
            raise ValueError("num_fno_layers must be positive")
        if self.fno_layer_size <= 0:
            raise ValueError("fno_layer_size must be positive")
        if self.num_fno_modes <= 0:
            raise ValueError("num_fno_modes must be positive")


@dataclass
class UNetModelConfig(BaseModelConfig):
    """UNet-specific model configuration."""
    model_type: str = "UNet-CNN"
    network_type: str = "SongUNetPosEmbd"  # Specific UNet architecture
    model_channels: int = 64
    channel_mult: List[int] = field(default_factory=lambda: [1, 2, 2])
    attn_resolutions: List[int] = field(default_factory=lambda: [16])
    embedding_type: str = "zero"
    n_grid_channels: int = 4
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        self._validate_base()

        valid_architectures = ["SongUNetPosEmbd", 
                               "UNet-CNN", 
                               'DhariwalUNet',
                               ]

        valid_embeddings = ["zero", "sinusoidal", "learnable", "linear", "positional"]
        if self.network_type not in valid_architectures:
            raise ValueError(f"Invalid network type: {self.network_type}")
        if self.embedding_type not in valid_embeddings:
            raise ValueError(f"Invalid embedding type: {self.embedding_type}")
        if any(mult <= 0 for mult in self.channel_mult):
            raise ValueError("Channel multipliers must be positive")
        if any(res <= 0 for res in self.attn_resolutions):
            raise ValueError("Attention resolutions must be positive")


@dataclass
class DiffusionModelConfig(BaseModelConfig):
    """Diffusion-specific model configuration."""
    model_type: str = "Diffusion"
    # UNet backbone params (diffusion uses UNet internally)
    model_channels: int = 64
    channel_mult: List[int] = field(default_factory=lambda: [1, 2, 2])
    attn_resolutions: List[int] = field(default_factory=lambda: [16])
    embedding_type: str = "zero"
    n_grid_channels: int = 4
    # Diffusion-specific params
    num_diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        self._validate_base()
        valid_schedules = ["linear", "cosine", "quadratic"]
        if self.noise_schedule not in valid_schedules:
            raise ValueError(f"Invalid noise schedule: {self.noise_schedule}")
        if self.beta_start >= self.beta_end:
            raise ValueError("beta_start must be less than beta_end")

@dataclass
class UNOModelConfig(BaseModelConfig):

    model_type: str = "UNO"
    hidden_channels: int = 64
    projection_channels: int = 64
    lifting_channels: int = 64
    positional_embedding: str = "grid"  ##gridEmbedding2D,gridEmbeddingNd
    uno_out_channels: List[int] = field(default_factory=lambda: [32, 64, 64, 64, 32])
    uno_n_modes: List[List[int]] = field(default_factory=lambda: [[16, 16], [12, 12], [12, 12], [16, 16], [16, 16]])
    uno_scalings: List[List[float]] = field(default_factory=lambda: [[1.0, 1.0], [0.5, 0.5], [1, 1], [2, 2], [1, 1]])
    horizontal_skips_map: Optional[Dict[str, List[int]]] = None
    channel_mlp_skip: str = 'linear'
    n_layers: int = 5

    def __post_init__(self):
        self._validate()

    def _validate(self):
        self._validate_base()
        if self.hidden_channels <= 0:
            raise ValueError("hidden channels must be positive")
        if self.projection_channels <= 0:
            raise ValueError("projection channels must be positive")
        if self.n_layers <= 0:
            raise ValueError("number of layers must be positive")

        if len(self.uno_out_channels) != self.n_layers:
            raise ValueError("Length of uno_out_channels must match n_layers")
        if len(self.uno_n_modes) != self.n_layers:
            raise ValueError("Length of uno_n_modes must match n_layers")
        if len(self.uno_scalings) != self.n_layers:
            raise ValueError("Length of uno_scalings must match n_layers")

        for m in self.uno_n_modes:
            if not (isinstance(m, list) and len(m) == 2 and all(int(x) > 0 for x in m)):
                raise ValueError("Each entry of uno_n_modes must be [modes_y, modes_x] with positive ints")

        for s in self.uno_scalings:
            if not (isinstance(s, list) and len(s) == 2 and all(float(x) > 0 for x in s)):
                raise ValueError("Each entry of uno_scalings must be [scale_y, scale_x] with positive floats")

@dataclass
class YangModelConfig(BaseModelConfig):

    model_type: str = "DSFNO"
    n_channels: int = 64
    n_residual_blocks: int = 4
    n_operator_blocks: int = 2
    modes: int = 16
    apply_constraint: bool = True

    def __post_init__(self):
        self._validate()

    def _validate(self):
        self._validate_base()
        if self.n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if self.n_residual_blocks <= 0:
            raise ValueError("n_residual_blocks must be positive")
        if self.n_operator_blocks <= 0:
            raise ValueError("n_operator_blocks must be positive")
        if self.modes <= 0:
            raise ValueError("modes must be positive")

# Type alias for model configs
ModelConfig = Union[FNOModelConfig, UNetModelConfig, DiffusionModelConfig, UNOModelConfig, YangModelConfig]


def create_model_config(config_dict: Dict[str, Any]) -> ModelConfig:
    """Factory function to create the appropriate model config based on model type."""
    model_type = config_dict.get('model', config_dict.get('model_type', 'FNO'))
    
    # Map model type to config class
    config_classes = {
        'FNO': FNOModelConfig,
        'UNet-CNN': UNetModelConfig,
        'Diffusion': DiffusionModelConfig,
        'UNO': UNOModelConfig,
        'DSFNO': YangModelConfig,
    }
    
    if model_type not in config_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    config_class = config_classes[model_type]
    
    # Filter parameters relevant to this model type
    model_params = _extract_model_params(config_dict, model_type)
    
    return config_class(**model_params)


def _extract_model_params(config_dict: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """Extract model-specific parameters from config dictionary."""
    
    # Common params for all models
    common_keys = ['model_type', 'checkpoint_level']
    
    # Model-specific parameter mappings
    model_param_keys = {
        'FNO': [
            'num_fno_layers', 'fno_layer_size', 'num_fno_modes', 
            'fno_padding', 'coord_features', 'decoder_layers', 'decoder_layer_size'
        ],
        'UNet-CNN': [
            'model_channels', 'channel_mult', 'attn_resolutions',
            'embedding_type', 'n_grid_channels'
        ],
        'Diffusion': [
            'model_channels', 'channel_mult', 'attn_resolutions',
            'embedding_type', 'n_grid_channels',
            'num_diffusion_steps', 'noise_schedule', 'beta_start', 'beta_end'
        ],
        'UNO': [
            'hidden_channels', 'projection_channels', 'lifting_channels', 'positional_embedding',
            'uno_out_channels', 'uno_n_modes', 'uno_scalings', 'horizontal_skips_map',
            'channel_mlp_skip', 'n_layers'
        ],

        'DSFNO': [
            'n_channels', 'n_residual_blocks', 'n_operator_blocks', 'modes', 'apply_constraint'
        ],
    }
    
    allowed_keys = set(common_keys + model_param_keys.get(model_type, []))
    
    # Handle legacy key mappings
    legacy_mappings = {
        'model': 'model_type',
        'N_grid_channels': 'n_grid_channels',
    }
    
    params = {}
    for key, value in config_dict.items():
        # Map legacy keys
        mapped_key = legacy_mappings.get(key, key)
        
        if mapped_key in allowed_keys and value is not None:
            params[mapped_key] = value
    
    if 'model' in config_dict:
        params['model_type'] = config_dict['model']
    
    # 'model_type' in YAML for UNet is the network architecture (SongUNetPosEmbd)
    if model_type == 'UNet-CNN' and 'model_type' in config_dict:
        params['network_type'] = config_dict['model_type']
        params['model_type'] = 'UNet-CNN'   # restore correct family name
    
    return params
    
    return params


# ============================================================================
# TRAINING AND EXPERIMENT CONFIGS
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration with validation."""
    lr: float = 2e-4
    epochs: int = 200
    batch_size: int = 8
    val_interval: int = 1
    precision: str = "32"
    n_workers: int = 8
    seed: int = 42
    lr_decay: int = 1
    lr_rampup: int = 0
    grad_clip_threshold: Optional[float] = None
    anneal_epochs: int = 200
    init_lambda: float = 0.0
    max_lambda: float = 0.1
    lambda_psd: float = 0.1
    loss_type: Optional[str] = None
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if self.lr <= 0:
            raise ValueError("Learning rate must be positive")
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        try:
            PrecisionType(self.precision)
        except ValueError:
            raise ValueError(f"Unsupported precision: {self.precision}")
        if not 0 <= self.init_lambda <= self.max_lambda:
            raise ValueError("init_lambda must be <= max_lambda and >= 0")


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    dataset: DatasetConfig
    model: ModelConfig  # Now accepts any model config type
    training: TrainingConfig
    
    # Optional settings
    wandb_project: Optional[str] = None
    run_name: Optional[str] = None
    save_preds_path: Optional[str] = None
    load: Optional[str] = None
    restore_opt: int = 0
    eval: Optional[str] = None
    resume: Optional[str] = None
    regression_net: Optional[str] = None
    gridtype: Optional[str] = None
    hr_mean_conditioning: bool = True
    num_ensembles: int = 32
    output_variables: Optional[List[str]] = None
    model_type: Optional[str] = None
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if self.eval is not None and self.eval not in ["val", "test"]:
            raise ValueError("eval must be 'val', 'test', or None")
        if self.load and not Path(self.load).exists():
            raise ValueError(f"Checkpoint path does not exist: {self.load}")
        if self.save_preds_path:
            Path(self.save_preds_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        
        # Create dataset config
        dataset_params = {}
        dataset_mappings = {
            'dataset_cerra': 'cerra_path',
            'dataset_era5': 'era5_path',
            'subset_ds': 'subset_size',
        }
        dataset_keys = [
            'img_in_channels', 'img_out_channels', 'img_resolution', 'standardize'
        ]
        
        for old_key, new_key in dataset_mappings.items():
            if old_key in config_dict:
                dataset_params[new_key] = config_dict[old_key]
        
        for key in dataset_keys:
            if key in config_dict:
                dataset_params[key] = config_dict[key]
        
        dataset_config = DatasetConfig(**dataset_params)
        
        # Create model config using factory
        model_config = create_model_config(config_dict)
        
        # Create training config
        training_keys = [
            'lr', 'epochs', 'batch_size', 'val_interval', 'precision', 'n_workers',
            'seed', 'lr_decay', 'lr_rampup', 'grad_clip_threshold', 'anneal_epochs',
            'init_lambda', 'max_lambda', 'lambda_psd', 'loss_type'
        ]
        training_params = {k: v for k, v in config_dict.items() if k in training_keys and v is not None}
        training_config = TrainingConfig(**training_params)
        
        # Extract remaining settings
        all_used_keys = set(
            list(dataset_mappings.keys()) + dataset_keys +
            training_keys + _get_all_model_keys()
        )
        
        remaining_config = {k: v for k, v in config_dict.items() if k not in all_used_keys}
        
        # Map legacy parameter names
        if 'savepreds_path' in remaining_config:
            remaining_config['save_preds_path'] = remaining_config.pop('savepreds_path')
        
        return cls(
            dataset=dataset_config,
            model=model_config,
            training=training_config,
            **remaining_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_yaml(self, output_path: str) -> None:
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)


def _get_all_model_keys() -> List[str]:
    """Get all possible model parameter keys across all model types."""
    return [
        'model', 'model_type', 'checkpoint_level',
        # FNO
        'num_fno_layers', 'fno_layer_size', 'num_fno_modes',
        'fno_padding', 'coord_features', 'decoder_layers', 'decoder_layer_size',
        # UNet/Diffusion
        'model_channels', 'channel_mult', 'attn_resolutions',
        'embedding_type', 'n_grid_channels', 'N_grid_channels',
        # Diffusion-specific
        'num_diffusion_steps', 'noise_schedule', 'beta_start', 'beta_end',
        # UNO-specific
        'hidden_channels', 'projection_channels', 'lifting_channels', 'positional_embedding',
        'uno_out_channels', 'uno_n_modes', 'uno_scalings', 'horizontal_skips_map',
        'channel_mlp_skip', 'n_layers',
        # Yang-specific (DSFNO)
        'n_channels', 'n_residual_blocks', 'n_operator_blocks', 'modes', 'apply_constraint',
    ]
    
