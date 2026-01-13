"""
Main entry point for Uyghur ASR training.

This script orchestrates the complete training pipeline:
1. Load and preprocess datasets
2. Initialize Whisper model
3. Train the model
4. Generate predictions and save submission
"""

import pandas as pd

from config import Config
from utils import setup_environment, print_dataset_stats, clear_gpu_cache
from data_loader import load_datasets, load_dataframes
from preprocessor import preprocess_dataset
from model import setup_model
from trainer import create_trainer, train
from inference import generate_predictions, create_submission, save_model, print_gpu_stats


def main():
    """Main training pipeline."""
    
    # Setup environment
    setup_environment()
    
    # Initialize configuration
    config = Config()
    
    # Setup model and processor
    processor, model, device = setup_model(config.model)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = load_datasets(config)
    
    # Load original test dataframe for submission
    _, test_df = load_dataframes(config)
    
    # Print statistics
    print_dataset_stats(train_dataset, val_dataset, test_dataset)
    
    # Preprocess datasets
    print("Preprocessing datasets...")
    train_dataset = preprocess_dataset(train_dataset, processor, config.model.max_length)
    val_dataset = preprocess_dataset(val_dataset, processor, config.model.max_length)
    test_dataset = preprocess_dataset(test_dataset, processor, config.model.max_length)
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    # Train
    print(f"Starting GPU-accelerated training with {config.model.name}...")
    clear_gpu_cache()
    train(trainer)
    
    # Generate predictions
    transcriptions = generate_predictions(trainer, test_dataset, processor)
    
    # Create submission
    submission_path = config.output_dir / "submission.csv"
    create_submission(test_df, transcriptions, submission_path)
    
    # Save model
    save_model(model, processor, config.model_save_path)
    
    # Print GPU stats
    print_gpu_stats()
    
    print("=" * 50)
    print("Training completed successfully!")
    print(f"Results saved to: {config.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
