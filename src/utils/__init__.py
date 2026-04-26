"""Utility functions for video captioning system."""

import os
import random
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification. Options: 'auto', 'cuda', 'mps', 'cpu'.
        
    Returns:
        PyTorch device object.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level.
        
    Returns:
        Configured logger.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: Union[int, float]) -> str:
    """Format large numbers with appropriate suffixes.
    
    Args:
        num: Number to format.
        
    Returns:
        Formatted string.
    """
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(num)


def create_directory(path: str) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration object.
    """
    from omegaconf import OmegaConf
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object.
        save_path: Path to save configuration.
    """
    from omegaconf import OmegaConf
    
    OmegaConf.save(config, save_path)


def filter_caption(caption: str, max_length: int = 200) -> str:
    """Filter and clean caption text.
    
    Args:
        caption: Input caption text.
        max_length: Maximum caption length.
        
    Returns:
        Filtered caption.
    """
    # Remove extra whitespace
    caption = " ".join(caption.split())
    
    # Truncate if too long
    if len(caption) > max_length:
        caption = caption[:max_length].rsplit(" ", 1)[0] + "..."
    
    return caption


def validate_video_file(video_path: str) -> bool:
    """Validate video file format and accessibility.
    
    Args:
        video_path: Path to video file.
        
    Returns:
        True if video is valid, False otherwise.
    """
    if not os.path.exists(video_path):
        return False
    
    # Check file extension
    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if not any(video_path.lower().endswith(ext) for ext in valid_extensions):
        return False
    
    return True


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Model size in MB.
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            restore_best_weights: Whether to restore best weights.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            val_score: Current validation score.
            model: Model to potentially save weights.
            
        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
            
        return False
    
    def save_checkpoint(self, model: nn.Module) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save.
        """
        self.best_weights = model.state_dict().copy()
