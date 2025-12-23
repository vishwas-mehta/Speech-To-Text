"""
Inference and prediction generation.
"""

import torch
import pandas as pd
from pathlib import Path
from transformers import WhisperProcessor, Seq2SeqTrainer

from config import Config


def generate_predictions(
    trainer: Seq2SeqTrainer,
    test_dataset,
    processor: WhisperProcessor
) -> list:
    """
    Generate predictions on test dataset.
    
    Args:
        trainer: Trained Seq2SeqTrainer
        test_dataset: Preprocessed test dataset
        processor: WhisperProcessor for decoding
        
    Returns:
        List of transcription strings
    """
    print("Generating predictions...")
    
    predictions = trainer.predict(test_dataset)
    pred_ids = predictions.predictions
    
    # Decode predictions to text
    transcriptions = processor.batch_decode(pred_ids, skip_special_tokens=True)
    
    return transcriptions


def create_submission(
    test_df: pd.DataFrame,
    transcriptions: list,
    output_path: Path
) -> pd.DataFrame:
    """
    Create submission CSV file.
    
    Args:
        test_df: Original test dataframe with IDs
        transcriptions: List of predicted transcriptions
        output_path: Path to save submission CSV
        
    Returns:
        Submission dataframe
    """
    submission_df = test_df.copy()
    submission_df["transcription"] = transcriptions
    
    # Save submission with only required columns
    submission_df[["ID", "transcription"]].to_csv(output_path, index=False)
    
    print(f"Saved submission to: {output_path}")
    
    return submission_df


def save_model(model, processor: WhisperProcessor, save_path: Path):
    """
    Save fine-tuned model and processor.
    
    Args:
        model: Trained Whisper model
        processor: WhisperProcessor
        save_path: Directory to save model
    """
    save_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    
    print(f"Model saved to: {save_path}")


def print_gpu_stats():
    """Print GPU memory usage statistics."""
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory usage: {peak_memory:.2f} GB")
