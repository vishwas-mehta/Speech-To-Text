"""
Training pipeline for Whisper ASR.
"""

import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperProcessor,
    WhisperForConditionalGeneration
)

from config import Config, TrainingConfig
from data_collator import create_data_collator
from metrics import create_compute_metrics_fn


def create_training_args(config: Config) -> Seq2SeqTrainingArguments:
    """
    Create training arguments from config.
    
    Args:
        config: Main configuration object
        
    Returns:
        Configured Seq2SeqTrainingArguments
    """
    tc = config.training
    
    return Seq2SeqTrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=tc.batch_size,
        per_device_eval_batch_size=tc.eval_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=tc.learning_rate,
        warmup_steps=tc.warmup_steps,
        max_steps=tc.max_steps,
        eval_strategy="steps",
        eval_steps=tc.eval_steps,
        save_steps=tc.save_steps,
        logging_steps=tc.logging_steps,
        generation_max_length=config.model.max_length,
        generation_num_beams=1,  # Greedy for speed
        predict_with_generate=True,
        fp16=tc.fp16,
        dataloader_num_workers=tc.dataloader_workers,
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        save_total_limit=3,
        report_to="none",
        dataloader_pin_memory=True,
        group_by_length=True,
        lr_scheduler_type="cosine",
        weight_decay=tc.weight_decay,
    )


def create_trainer(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    train_dataset,
    val_dataset,
    config: Config
) -> Seq2SeqTrainer:
    """
    Create and configure the trainer.
    
    Args:
        model: Whisper model
        processor: Whisper processor
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Configuration object
        
    Returns:
        Configured Seq2SeqTrainer
    """
    training_args = create_training_args(config)
    data_collator = create_data_collator(processor, model)
    compute_metrics = create_compute_metrics_fn(processor)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )
    
    return trainer


def train(trainer: Seq2SeqTrainer):
    """
    Execute training.
    
    Args:
        trainer: Configured trainer
        
    Returns:
        Training result
    """
    print("Starting training...")
    print("Estimated training time: 2-3 hours")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Train
    result = trainer.train()
    
    print(f"Training completed!")
    print(f"Final training loss: {result.training_loss:.4f}")
    
    return result
