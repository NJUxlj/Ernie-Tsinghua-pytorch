
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin

@dataclass
class ErnieDataCollator(DataCollatorMixin):
    """
    Data collator for ERNIE model. Handles both MLM and entity linking tasks.
    
    Args:
        tokenizer: The tokenizer used to process the text
        mlm_probability: Probability of masking tokens for MLM task
        entity_probability: Probability of masking entities for entity prediction task
        max_length: Maximum sequence length
    """
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    entity_probability: float = 0.15
    max_length: int = 512
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Handle special tokens and padding
        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Get device
        device = batch["input_ids"].device
        
        # Create MLM labels
        labels = batch["input_ids"].clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = self._get_special_tokens_mask(batch["input_ids"])
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        # Create entity prediction labels
        entity_labels = batch["entity_ids"].clone()
        entity_probability_matrix = torch.full(entity_labels.shape, self.entity_probability)
        entity_masked_indices = torch.bernoulli(entity_probability_matrix).bool()
        entity_labels[~entity_masked_indices] = -100
        
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        batch["input_ids"][indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=device)
        batch["input_ids"][indices_random] = random_words[indices_random]
        
        batch["mlm_labels"] = labels
        batch["entity_labels"] = entity_labels
        
        return batch
    
    def _get_special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Creates a mask for special tokens to avoid masking them."""
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in input_ids.tolist()
        ]
        return torch.tensor(special_tokens_mask, dtype=torch.bool)

