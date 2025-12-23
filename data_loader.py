"""
Data loading utilities for Uyghur ASR.
"""

import pandas as pd
from pathlib import Path
from datasets import Dataset, Audio
from sklearn.model_selection import train_test_split
from typing import Tuple

from config import Config


def load_dataframes(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSV files."""
    train_df = pd.read_csv(config.train_csv)
    test_df = pd.read_csv(config.test_csv)
    return train_df, test_df


def split_train_val(
    train_df: pd.DataFrame, 
    config: Config
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split training data into train and validation sets."""
    train_data, val_data = train_test_split(
        train_df,
        test_size=config.data.test_split,
        random_state=config.data.random_seed
    )
    return train_data.copy(), val_data.copy()


def add_audio_paths(df: pd.DataFrame, audio_dir: Path) -> pd.DataFrame:
    """Add full audio file paths to dataframe."""
    df = df.copy()
    df["audio"] = df["filepath"].apply(
        lambda x: str(audio_dir / Path(x).name)
    )
    return df


def create_hf_dataset(df: pd.DataFrame, sample_rate: int = 16000) -> Dataset:
    """Convert pandas DataFrame to HuggingFace Dataset with audio."""
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
    return dataset


def load_datasets(config: Config) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load and prepare all datasets.
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Load CSVs
    train_df, test_df = load_dataframes(config)
    
    # Split train/val
    train_data, val_data = split_train_val(train_df, config)
    
    # Add audio paths
    train_data = add_audio_paths(train_data, config.audio_dir)
    val_data = add_audio_paths(val_data, config.audio_dir)
    test_df = add_audio_paths(test_df, config.audio_dir)
    
    # Create HF datasets
    train_dataset = create_hf_dataset(train_data, config.data.sample_rate)
    val_dataset = create_hf_dataset(val_data, config.data.sample_rate)
    test_dataset = create_hf_dataset(test_df, config.data.sample_rate)
    
    return train_dataset, val_dataset, test_dataset
