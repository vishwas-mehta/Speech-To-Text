"""
Audio preprocessing for Whisper ASR.
"""

from transformers import WhisperProcessor


def normalize_audio(audio_array):
    """
    Normalize audio to prevent clipping.
    
    Args:
        audio_array: Raw audio waveform
        
    Returns:
        Normalized audio array
    """
    if len(audio_array) > 0:
        max_val = max(abs(audio_array))
        if max_val > 0:
            return audio_array / max_val
    return audio_array


def create_preprocess_function(processor: WhisperProcessor, max_length: int = 448):
    """
    Create a preprocessing function for dataset mapping.
    
    Args:
        processor: WhisperProcessor instance
        max_length: Maximum token length for transcriptions
        
    Returns:
        Preprocessing function for dataset.map()
    """
    
    def preprocess_function(batch):
        """Process a single batch of audio data."""
        audio = batch["audio"]
        
        # Normalize audio
        audio_array = normalize_audio(audio["array"])
        
        # Extract mel spectrogram features
        batch["input_features"] = processor.feature_extractor(
            audio_array,
            sampling_rate=audio["sampling_rate"],
            return_tensors="np"
        ).input_features[0]
        
        # Tokenize transcription if available (training data)
        if "transcription" in batch:
            batch["labels"] = processor.tokenizer(
                batch["transcription"],
                max_length=max_length,
                truncation=True,
                padding=False
            ).input_ids
        
        return batch
    
    return preprocess_function


def preprocess_dataset(dataset, processor: WhisperProcessor, max_length: int = 448):
    """
    Apply preprocessing to entire dataset.
    
    Args:
        dataset: HuggingFace Dataset
        processor: WhisperProcessor instance
        max_length: Maximum token length
        
    Returns:
        Preprocessed dataset
    """
    preprocess_fn = create_preprocess_function(processor, max_length)
    
    processed = dataset.map(
        preprocess_fn,
        remove_columns=dataset.column_names,
        desc="Preprocessing audio"
    )
    
    return processed
