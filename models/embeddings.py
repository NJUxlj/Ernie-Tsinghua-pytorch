import torch
import torch.nn as nn
from typing import Optional, Tuple

class ErnieEmbeddings(nn.Module):
    """Construct the embeddings from word, position, token_type and entity embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.hidden_size, padding_idx=0)
        
        # Layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Position IDs (for position embeddings)
        self.position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
        self.token_type_ids = torch.zeros(self.position_ids.size(), dtype=torch.long)
        
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        entity_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        entity_embeddings = self.entity_embeddings(entity_ids) if entity_ids is not None else 0

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = inputs_embeds + position_embeddings + token_type_embeddings + entity_embeddings
        else:
            embeddings = inputs_embeds + token_type_embeddings + entity_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings