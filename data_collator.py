"""
Data collator for Whisper training batches.
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Union

from transformers import WhisperProcessor


@dataclass
class WhisperDataCollator:
    """
    Custom data collator for Whisper model training.
    
    Handles padding of input features and labels, and masks
    padding tokens in labels with -100 for loss calculation.
    """
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(
        self, 
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate batch of features.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Batched and padded tensors
        """
        # Process input features (mel spectrograms)
        input_features = [
            {"input_features": feature["input_features"]} 
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, 
            return_tensors="pt"
        )
        
        # Process labels if present (training mode)
        if "labels" in features[0]:
            label_features = [
                {"input_ids": feature["labels"]} 
                for feature in features
            ]
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            )
            
            # Replace padding with -100 for loss calculation
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch["attention_mask"].ne(1), -100
            )
            
            # Remove decoder start token if present at beginning
            if (labels[:, 0] == self.decoder_start_token_id).all():
                labels = labels[:, 1:]
            
            batch["labels"] = labels
        
        return batch


def create_data_collator(processor: WhisperProcessor, model) -> WhisperDataCollator:
    """
    Create a data collator for the given processor and model.
    
    Args:
        processor: WhisperProcessor instance
        model: WhisperForConditionalGeneration model
        
    Returns:
        Configured WhisperDataCollator
    """
    return WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )
