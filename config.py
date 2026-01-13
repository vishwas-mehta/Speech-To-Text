"""
Configuration settings for Uyghur ASR training.
"""

__version__ = "1.0.0"
__author__ = "Vishwas Mehta"

import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Whisper model configuration."""
    name: str = "openai/whisper-medium"
    max_length: int = 448
    use_cache: bool = False


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    warmup_steps: int = 100
    max_steps: int = 1000
    eval_steps: int = 125
    save_steps: int = 250
    logging_steps: int = 25
    weight_decay: float = 0.01
    fp16: bool = True
    dataloader_workers: int = 2


@dataclass
class DataConfig:
    """Dataset configuration."""
    sample_rate: int = 16000
    test_split: float = 0.05
    random_seed: int = 42


@dataclass
class Config:
    """Main configuration container."""
    
    def __init__(self, root_dir: str = None):
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).resolve().parent
        self.data_dir = self.root_dir / "data"
        self.output_dir = self.root_dir / "whisper_results"
        self.audio_dir = self.data_dir / "wavs"
        
        # Sub-configs
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
    
    @property
    def train_csv(self) -> Path:
        return self.data_dir / "train.csv"
    
    @property
    def test_csv(self) -> Path:
        return self.data_dir / "test.csv"
    
    @property
    def model_save_path(self) -> Path:
        return self.output_dir / "final_model"


# Environment setup
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
