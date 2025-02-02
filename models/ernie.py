
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Tuple

class ErnieConfig(PretrainedConfig):
    """ERNIE模型的配置类"""
    model_type = "ernie"
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        entity_vocab_size: int = 50000,
        entity_emb_size: int = 768,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.entity_vocab_size = entity_vocab_size
        self.entity_emb_size = entity_emb_size
        self.layer_norm_eps = layer_norm_eps

class ErnieModel(PreTrainedModel):
    """ERNIE模型的主体架构"""
    config_class = ErnieConfig
    base_model_prefix = "ernie"
    
    def __init__(self, config: ErnieConfig):
        super().__init__(config)
        self.config = config
        
        # 文本编码器
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        # 实体编码器
        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size)
        self.entity_pos_embeddings = nn.Embedding(config.max_position_embeddings, config.entity_emb_size)
        
        # Layer Norm和Dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 多头注意力层
        self.encoder = nn.ModuleList([
            ErnieLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        self.init_weights()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        entity_ids: Optional[torch.Tensor] = None,
        entity_position_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        
        # 生成position_ids
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
            
        # 生成attention_mask
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=input_ids.device)
            
        # 文本嵌入
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # 合并文本嵌入
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # 实体嵌入
        if entity_ids is not None:
            entity_embeddings = self.entity_embeddings(entity_ids)
            entity_pos_embeddings = self.entity_pos_embeddings(entity_position_ids)
            entity_embeddings = entity_embeddings + entity_pos_embeddings
        else:
            entity_embeddings = None
            
        # 注意力mask处理
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # 通过编码器层
        hidden_states = embeddings
        all_hidden_states = ()
        
        for layer in self.encoder:
            all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer(
                hidden_states,
                extended_attention_mask,
                entity_embeddings
            )
            
        return hidden_states, all_hidden_states

class ErnieLayer(nn.Module):
    """ERNIE的编码器层，包含多头注意力和前馈网络"""
    def __init__(self, config):
        super().__init__()
        self.attention = ErnieAttention(config)
        self.intermediate = ErnieIntermediate(config)
        self.output = ErnieOutput(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask, entity_embeddings)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class ErnieAttention(nn.Module):
    """实现了ERNIE的多头注意力机制，包含实体融合"""
    def __init__(self, config):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 自注意力计算
        self_attention_output, _ = self.self_attention(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=attention_mask,
            need_weights=False
        )
        
        # 如果有实体嵌入，进行融合
        if entity_embeddings is not None:
            # 实体信息融合，这里使用简单的加法融合
            self_attention_output = self_attention_output + entity_embeddings
            
        attention_output = self.dense(self_attention_output)
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)
        
        return attention_output

class ErnieIntermediate(nn.Module):
    """ERNIE的前馈中间层"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class ErnieOutput(nn.Module):
    """ERNIE的输出层"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

