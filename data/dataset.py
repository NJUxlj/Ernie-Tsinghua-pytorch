
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import json
import random
from transformers import PreTrainedTokenizer
from dataclasses import dataclass

class ErniePretrainingDataset(Dataset):
    """
    Dataset for ERNIE pretraining, supporting MLM, NSP and dEA tasks
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        kg_path: str,
        max_seq_length: int = 512,
        max_entity_length: int = 64,
        mlm_probability: float = 0.15
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_entity_length = max_entity_length
        self.mlm_probability = mlm_probability
        
        # Load text corpus
        print(f"Loading text corpus from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.examples = [json.loads(line) for line in f]
            
        # Load knowledge graph
        print(f"Loading knowledge graph from {kg_path}")
        with open(kg_path, 'r', encoding='utf-8') as f:
            self.kg = json.load(f)
            
        # Create entity to id mapping
        self.entity2id = {entity: idx for idx, entity in enumerate(self.kg['entities'])}
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Get text segments and entities
        text_a = example['text_a']
        text_b = example['text_b'] if random.random() > 0.5 else self._get_random_text()
        is_next = 1 if example['text_b'] == text_b else 0
        
        entities_a = example['entities_a']
        entities_b = example['entities_b'] if is_next else []
        
        # Tokenize text
        tokenized = self.tokenizer(
            text_a,
            text_b,
            max_length=self.max_seq_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Convert entities to ids and pad
        entity_ids = [self.entity2id[e] for e in entities_a + entities_b]
        entity_ids = entity_ids[:self.max_entity_length]
        entity_ids += [0] * (self.max_entity_length - len(entity_ids))
        
        # Create masked LM labels
        input_ids = tokenized['input_ids'].squeeze()
        mlm_labels = self._create_mlm_labels(input_ids.clone())
        
        return {
            'input_ids': input_ids,
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'token_type_ids': tokenized['token_type_ids'].squeeze(),
            'entity_ids': torch.tensor(entity_ids),
            'next_sentence_label': torch.tensor(is_next),
            'mlm_labels': mlm_labels
        }
    
    def _create_mlm_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create masked language model labels for BERT-style pretraining"""
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        labels[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% of the time, replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        labels[indices_random] = random_words[indices_random]
        
        # The rest of the time (10% of the time) keep the masked input tokens unchanged
        return labels
    
    def _get_random_text(self) -> str:
        """Get a random text segment for negative NSP examples"""
        random_idx = random.randint(0, len(self.examples) - 1)
        return self.examples[random_idx]['text_b']

@dataclass
class ErnieDataCollator:
    """
    Data collator for ERNIE pretraining
    Combines a batch of dataset items into a training batch
    """
    tokenizer: PreTrainedTokenizer
    max_length: Optional[int] = 512
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'token_type_ids': torch.stack([f['token_type_ids'] for f in features]),
            'entity_ids': torch.stack([f['entity_ids'] for f in features]),
            'next_sentence_label': torch.stack([f['next_sentence_label'] for f in features]),
            'mlm_labels': torch.stack([f['mlm_labels'] for f in features])
        }
        
        return batch

