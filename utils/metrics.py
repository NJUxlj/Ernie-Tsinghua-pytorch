
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ErnieMetrics:
    """
    Metrics calculator for ERNIE model evaluation.
    Handles both MLM and entity prediction tasks.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.mlm_correct = 0
        self.mlm_total = 0
        self.entity_correct = 0
        self.entity_total = 0
        self.mlm_predictions = []
        self.mlm_labels = []
        self.entity_predictions = []
        self.entity_labels = []
    
    def update(
        self,
        mlm_logits: torch.Tensor,
        mlm_labels: torch.Tensor,
        entity_logits: torch.Tensor,
        entity_labels: torch.Tensor
    ) -> None:
        """
        Update metrics with new batch of predictions.
        
        Args:
            mlm_logits: Predicted logits for MLM task
            mlm_labels: True labels for MLM task
            entity_logits: Predicted logits for entity prediction task
            entity_labels: True labels for entity prediction task
        """
        # MLM metrics
        mlm_predictions = torch.argmax(mlm_logits, dim=-1)
        mlm_mask = mlm_labels != -100
        self.mlm_correct += (mlm_predictions[mlm_mask] == mlm_labels[mlm_mask]).sum().item()
        self.mlm_total += mlm_mask.sum().item()
        
        # Entity prediction metrics
        entity_predictions = torch.argmax(entity_logits, dim=-1)
        entity_mask = entity_labels != -100
        self.entity_correct += (entity_predictions[entity_mask] == entity_labels[entity_mask]).sum().item()
        self.entity_total += entity_mask.sum().item()
        
        # Store predictions and labels for detailed metrics
        self.mlm_predictions.extend(mlm_predictions[mlm_mask].cpu().numpy())
        self.mlm_labels.extend(mlm_labels[mlm_mask].cpu().numpy())
        self.entity_predictions.extend(entity_predictions[entity_mask].cpu().numpy())
        self.entity_labels.extend(entity_labels[entity_mask].cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}
        
        # MLM metrics
        if self.mlm_total > 0:
            metrics["mlm_accuracy"] = self.mlm_correct / self.mlm_total
            mlm_precision, mlm_recall, mlm_f1, _ = precision_recall_fscore_support(
                self.mlm_labels,
                self.mlm_predictions,
                average='macro'
            )
            metrics.update({
                "mlm_precision": mlm_precision,
                "mlm_recall": mlm_recall,
                "mlm_f1": mlm_f1
            })
        
        # Entity prediction metrics
        if self.entity_total > 0:
            metrics["entity_accuracy"] = self.entity_correct / self.entity_total
            entity_precision, entity_recall, entity_f1, _ = precision_recall_fscore_support(
                self.entity_labels,
                self.entity_predictions,
                average='macro'
            )
            metrics.update({
                "entity_precision": entity_precision,
                "entity_recall": entity_recall,
                "entity_f1": entity_f1
            })
        
        return metrics
    
    def get_progress_metrics(self) -> Dict[str, float]:
        """
        Get current metrics for progress monitoring.
        
        Returns:
            Dictionary containing current accuracy metrics
        """
        metrics = {}
        if self.mlm_total > 0:
            metrics["mlm_accuracy"] = self.mlm_correct / self.mlm_total
        if self.entity_total > 0:
            metrics["entity_accuracy"] = self.entity_correct / self.entity_total
        return metrics

def compute_perplexity(loss: torch.Tensor) -> float:
    """
    Compute perplexity from loss.
    
    Args:
        loss: Loss tensor
        
    Returns:
        Perplexity value
    """
    return float(torch.exp(loss).item())

def compute_entity_stats(
    entity_labels: torch.Tensor,
    entity_predictions: torch.Tensor,
    ignore_index: int = -100
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score for entity prediction task.
    
    Args:
        entity_labels: True entity labels
        entity_predictions: Predicted entity labels
        ignore_index: Index to ignore in computation
        
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    mask = entity_labels != ignore_index
    labels = entity_labels[mask].cpu().numpy()
    preds = entity_predictions[mask].cpu().numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average='macro'
    )
    
    return precision, recall, f1

