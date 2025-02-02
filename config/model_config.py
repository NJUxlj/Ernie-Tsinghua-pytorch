from dataclasses import dataclass
from typing import Optional

@dataclass
class ErnieConfig:
    """Configuration class for ERNIE model."""
    vocab_size: int = 30522  # BERT vocab size
    entity_vocab_size: int = 50000  # Knowledge graph entity vocab size
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    max_entity_length: int = 64
    type_vocab_size: int = 2
    initializer_range: float = 0.02     
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    position_embedding_type: str = "absolute"
    use_cache: bool = True
    classifier_dropout: Optional[float] = None