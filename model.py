"""
Whisper model initialization and configuration.
"""

import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)

from config import ModelConfig


def check_gpu() -> torch.device:
    """
    Check GPU availability and return device.
    
    Returns:
        torch.device for CUDA or CPU
        
    Raises:
        RuntimeError if CUDA is not available
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Please check your GPU setup.")
    
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return device


def load_processor(model_name: str) -> WhisperProcessor:
    """
    Load Whisper processor (feature extractor + tokenizer).
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        WhisperProcessor instance
    """
    return WhisperProcessor.from_pretrained(model_name)


def load_model(
    model_name: str,
    device: torch.device,
    enable_gradient_checkpointing: bool = True
) -> WhisperForConditionalGeneration:
    """
    Load and configure Whisper model for training.
    
    Args:
        model_name: HuggingFace model identifier
        device: Target device (GPU/CPU)
        enable_gradient_checkpointing: Enable memory-efficient training
        
    Returns:
        Configured WhisperForConditionalGeneration model
    """
    print(f"Loading {model_name}...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Configure for training (disable forced tokens)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    
    # Enable gradient checkpointing for memory efficiency
    if enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Move to device
    model = model.to(device)
    
    return model


def setup_model(config: ModelConfig, device: torch.device = None):
    """
    Complete model setup with processor and model.
    
    Args:
        config: Model configuration
        device: Target device (auto-detected if None)
        
    Returns:
        Tuple of (processor, model, device)
    """
    if device is None:
        device = check_gpu()
    
    processor = load_processor(config.name)
    model = load_model(config.name, device)
    
    return processor, model, device
