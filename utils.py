"""
Utility functions for Uyghur ASR.
"""

import os
import warnings
import torch


def setup_environment():
    """
    Configure environment for training.
    
    Sets up CUDA, suppresses warnings, and optimizes cuDNN.
    """
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    # Environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    # Enable cuDNN benchmarking for faster training
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def print_dataset_stats(train_dataset, val_dataset, test_dataset):
    """
    Print dataset statistics.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
    """
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")


def clear_gpu_cache():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_gpu_info() -> dict:
    """
    Get GPU information.
    
    Returns:
        Dictionary with GPU name and memory
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "name": torch.cuda.get_device_name(),
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "current_memory_gb": torch.cuda.memory_allocated() / 1024**3,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3
    }


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "2h 30m 15s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
