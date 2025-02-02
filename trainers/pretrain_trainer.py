
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from typing import Dict, List, Optional, Union
from transformers.trainer_utils import EvalPrediction
import numpy as np
from utils.metrics import compute_metrics

class ErniePreTrainer(Trainer):
    """
    Custom trainer for ERNIE pretraining
    Handles multiple training objectives: MLM, NSP, and dEA
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the combined loss for ERNIE pretraining
        Combines MLM, NSP, and dEA losses with appropriate weights
        """
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids'],
            entity_ids=inputs['entity_ids'],
            labels=inputs['mlm_labels'],
            next_sentence_label=inputs['next_sentence_label'],
        )
        
        # Combine losses with weights
        # Default weights: MLM (1.0), NSP (1.0), dEA (0.5)
        total_loss = outputs.mlm_loss + outputs.nsp_loss + 0.5 * outputs.dea_loss
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Union[Dict[str, float], Dict[str, float]]:
        """
        Custom evaluation loop that handles multiple metrics
        """
        self.model.eval()
        total_eval_loss = 0
        total_eval_metrics = {
            'mlm_accuracy': 0.,
            'nsp_accuracy': 0.,
            'dea_accuracy': 0.,
        }
        total_samples = 0
        
        for step, inputs in enumerate(dataloader):
            with torch.no_grad():
                inputs = self._prepare_inputs(inputs)
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    token_type_ids=inputs['token_type_ids'],
                    entity_ids=inputs['entity_ids'],
                    labels=inputs['mlm_labels'],
                    next_sentence_label=inputs['next_sentence_label'],
                )
                
                # Compute metrics
                metrics = compute_metrics(EvalPrediction(
                    predictions={
                        'mlm_logits': outputs.mlm_logits.cpu().numpy(),
                        'nsp_logits': outputs.nsp_logits.cpu().numpy(),
                        'dea_logits': outputs.dea_logits.cpu().numpy(),
                    },
                    label_ids={
                        'mlm_labels': inputs['mlm_labels'].cpu().numpy(),
                        'nsp_labels': inputs['next_sentence_label'].cpu().numpy(),
                        'entity_ids': inputs['entity_ids'].cpu().numpy(),
                    }
                ))
                
                batch_size = inputs['input_ids'].size(0)
                total_samples += batch_size
                total_eval_loss += outputs.loss.item() * batch_size
                
                for metric_name, metric_value in metrics.items():
                    total_eval_metrics[metric_name] += metric_value * batch_size
        
        # Compute average metrics
        eval_loss = total_eval_loss / total_samples
        eval_metrics = {
            f"{metric_key_prefix}_{k}": v / total_samples 
            for k, v in total_eval_metrics.items()
        }
        eval_metrics[f"{metric_key_prefix}_loss"] = eval_loss
        
        return eval_metrics
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save checkpoint with custom metrics
        """
        checkpoint_folder = f"{self.args.output_dir}/checkpoint-{self.state.global_step}"
        
        # Save custom metrics
        if metrics is not None:
            metric_dict = {
                'mlm_loss': metrics.get('eval_mlm_loss', 0),
                'nsp_loss': metrics.get('eval_nsp_loss', 0),
                'dea_loss': metrics.get('eval_dea_loss', 0),
                'mlm_accuracy': metrics.get('eval_mlm_accuracy', 0),
                'nsp_accuracy': metrics.get('eval_nsp_accuracy', 0),
                'dea_accuracy': metrics.get('eval_dea_accuracy', 0),
            }
            with open(f"{checkpoint_folder}/metrics.json", 'w') as f:
                json.dump(metric_dict, f, indent=2)
        
        # Call parent class checkpoint saving
        super()._save_checkpoint(model, trial, metrics)

def create_ernie_trainer(
    model,
    args: TrainingArguments,
    train_dataset,
    eval_dataset=None,
    data_collator=None,
    tokenizer=None,
):
    """
    Factory function to create an ERNIE trainer instance
    """
    trainer = ErniePreTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    return trainer

