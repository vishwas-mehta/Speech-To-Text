"""
Evaluation metrics for ASR.
"""

import evaluate
from transformers import WhisperProcessor


# Global metric instance
_cer_metric = None


def get_cer_metric():
    """
    Get or initialize CER (Character Error Rate) metric.
    
    Returns:
        evaluate.Metric instance for CER
    """
    global _cer_metric
    if _cer_metric is None:
        _cer_metric = evaluate.load("cer")
    return _cer_metric


def create_compute_metrics_fn(processor: WhisperProcessor):
    """
    Create a compute_metrics function for the trainer.
    
    Args:
        processor: WhisperProcessor for decoding predictions
        
    Returns:
        Function compatible with Trainer.compute_metrics
    """
    cer_metric = get_cer_metric()
    
    def compute_metrics(eval_pred):
        """
        Compute CER metric for evaluation.
        
        Args:
            eval_pred: EvalPrediction object with predictions and labels
            
        Returns:
            Dictionary with CER score (as percentage)
        """
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids
        
        # Replace -100 (ignored tokens) with pad token for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        # Decode to text
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        
        # Calculate CER
        cer_score = cer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"cer": 100 * cer_score}
    
    return compute_metrics


def calculate_cer(predictions: list, references: list) -> float:
    """
    Calculate CER between predictions and references.
    
    Args:
        predictions: List of predicted transcriptions
        references: List of ground truth transcriptions
        
    Returns:
        CER score as percentage
    """
    cer_metric = get_cer_metric()
    cer_score = cer_metric.compute(predictions=predictions, references=references)
    return 100 * cer_score
