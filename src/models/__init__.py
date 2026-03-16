"""
Model definitions for PSD-Downscaling.

This module contains the neural network models and their PyTorch Lightning wrappers.
"""

from .unet import UNetWrapper
from .diffusion import DiffusionWrapper
from .fno_v1 import FNOWrapper
from .UNO import UNOWrapper
from .Yang import DSFNOWrapper
from .afno import AFNOWrapper

__all__ = ['UNetWrapper', 'DiffusionWrapper', 'FNOWrapper', 'UNOWrapper', 'DSFNOWrapper', 'AFNOWrapper']
