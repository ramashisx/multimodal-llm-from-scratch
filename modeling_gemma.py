import torch
import math
from dataclasses import dataclass
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Dict, List, Tuple, Union, Optional, Iterable
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel


class KVCache:
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        
        return self.key_cache[0].shape[-2]
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if layer_idx >= len(self.key_cache):
            # we never added cache for this layer
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # otherwise concat the new keys with the existing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        
        # and then we return the new cache
        return self.key_states[layer_idx], self.value_states[layer_idx]


@dataclass
class GemmaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int = 256
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id: int = None


@dataclass
class PaliGemmaConfig:
    vision_config: dict = {}
    text_config: dict = {}
    ignore_index: int = -100
    image_token_index: int = 256_000
    vocab_size: int = 257_152
    projection_dim: int = 2048
    hidden_size: int = 2048
    pad_token_id: int = None

    def __post_init__(self):
        self.vision_config = SiglipVisionConfig(**self.vision_config)
        self.text_config = GemmaConfig(**self.text_config, pad_token_id=self.pad_token_id)
        self.text_config.num_image_tokens = (
            self.vision_config.image_size // self.vision_config.patch_size
        ) ** 2
        self.vision_config.projection_dim = self.projection_dim


class PaliGemmaMultiModalProjection(nn.Module):
    
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.projection = nn.Linear(
            in_features=config.vision_config.hidden_size,
            out_features=config.vision_config.projection_dim,
            bias=True,
        )
    
    def forward(self, images_embed: torch.Tensor) -> torch.Tensor:
        # Shape (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, projection_dim)
        return self.projection(images_embed)


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.power(2).mean(dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float())
        # this is here for quantization and stuff
        output *= (1.0 + self.weight.float())
        return output.type_as(x)


class GemmaMLP(nn.Module):
    
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = nn.functional.gelu
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # shape (batch_size, sequence_length, intermediate_size)
        gate = self.act(self.gate_proj(hidden_states), approximate="tanh")
        # shape (batch_size, sequence_length, intermediate_size)
        up = self.up_proj(hidden_states)
        # shape (batch_size, sequence_length, hidden_size)
        return self.down_proj(gate * up)


def repeat_kv(
    hidden_states: torch.Tensor,
    n_repeats: int,
) -> torch.Tensor:
    batch_size, num_key_value_heads, sequence_length, head_dim = hidden_states.shape
    if n_repeats == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_key_value_heads, n_repeats, sequence_length, head_dim)
    return hidden_states.reshape(batch_size, num_key_value_heads * n_repeats, sequence_length, head_dim)


class GemmaAttention(nn.Module):
    
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        
        self.layer_idx = layer_idx
        self.is_causal = True
        self.attention_dropouts = config.attention_dropout
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        
        assert config.hidden_size % config.num_attention_heads == 0, "Hidden size must be divisible by the number of heads"
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=config.attention_bias)
        
        self.rotatory_embed = GemmaRotatoryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )        
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        
        batch_size, sequence_length, hidden_size = hidden_states.size()
        
        # shape (batch_size, sequence_length, num_attention_heads * head_dim)
        query_states = self.q_proj(hidden_states)
        # shape (batch_size, sequence_length, num_key_value_heads * head_dim)
        key_states = self.k_proj(hidden_states)
        # shape (batch_size, sequence_length, num_key_value_heads * head_dim)
        value_states = self.v_proj(hidden_states)
        
        # shape (batch_size, num_attention_heads, sequence_length, head_dim)
        query_states = query_states.view(
            batch_size, sequence_length, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        # shape (batch_size, num_key_value_heads, sequence_length, head_dim)
        key_states = key_states.view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        # shape (batch_size, num_key_value_heads, sequence_length, head_dim)
        value_states = value_states.view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        
        # shape (batch_size, sequence_length, head_dim)
        cos, sin = self.rotatory_embed(value_states, position_ids, seq_len=None)
        # shape (batch_size, num_query_heads, sequence_length, head_dim)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)
        
        # repeat the query states for each key-value head
        # this is a naive implementation and can be optimized
        # shape (batch_size, num_attention_heads, sequence_length, sequence_length)
        key_states = repeat_kv(key_states, self.num_key_value_heads)
        value_states = repeat_kv(value_states, self.num_key_value_heads)
        
        attn_weights = torch.einsum("bnqd,bnkd->bnqk", query_states, key_states)
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        assert attention_mask is not None, "Attention mask must be provided"
        attn_weights += attention_mask
        
        # Apply the softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(query_states)
        # Apply the dropout
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropouts, training=self.training)
        
        attn_output = torch.einsum("bnqk,bnkd->bnqd", attn_weights, value_states)
        
        if attn_output.shape != (batch_size, self.num_attention_heads, sequence_length, self.head_dim):
            raise ValueError(
                f"Attention output has the wrong shape. Expected {(batch_size, self.num_attention_heads, sequence_length, self.head_dim)}, got {attn_output.shape}"
            )
        
        # make sure to transpose back the attention output and make it contiguous
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, sequence_length, self.num_attention_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights        


class GemmaDecoderLayer(nn.Module):
    
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config=config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        
        residual = hidden_states
        
        # shape (batch_size, sequence_length, hidden_size)
        hidden_states = self.input_layernorm(hidden_states)
        
        # shape (batch_size, sequence_length, hidden_size)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        
        hidden_states += residual
        
        residual = hidden_states
        
        # shape (batch_size, sequence_length, hidden_size)
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # shape (batch_size, sequence_length, hidden_size)
        hidden_states = self.mlp(hidden_states)
        
        hidden_states += residual
        
        return hidden_states


class GemmaModel(nn.Module):
    
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        
        self.layers = nn.ModuleList([
            GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.layer_norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(
        self,
        inputs_embed: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        
        # shape (batch_size, sequence_length, hidden_size)
        hidden_states = inputs_embed
        
        # shape (batch_size, sequence_length, hidden_size)
        normalizer = torch.tensor(
            self.config.hidden_size ** 0.5,
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        hidden_states *= normalizer
        
        for layer in self.layers:
            # shape (batch_size, sequence_length, hidden_size)
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )
        
        hidden_states = self.layer_norm(hidden_states)
        
        return hidden_states


class GemmaForCausalLM(nn.Module):
    
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight
        
    def forward(
        self,
        inputs_embed: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Shape (batch_size, sequence_length, hidden_size)
        # output shape (batch_size, sequence_length, hidden_size)
        hidden_states = self.model(
            inputs_embed=inputs_embed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        
        # Shape (batch_size, sequence_length, vocab_size)
        logits = self.lm_head(hidden_states)
        
        return logits, kv_cache


class PaliGemmaForConditionalGeneration(nn.Module):
    
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionModel(config.vision_config)
        self.multi_model_projection = PaliGemmaMultiModalProjection(config)
        self.vocab_size = config.vocab_size
        self.language_model = GemmaForCausalLM(config.text_config)
        
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1
    
    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def _merge_text_and_images(
        self,
        images_embed: torch.Tensor,
        inputs_embed: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        _, _, embed_dim = inputs_embed.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embed.dtype, inputs_embed.device
        
        # Scale the image features
        # Shape (batch_size, sequence_length, hidden_size)
        scaled_images_embed  = images_embed / (self.config.hidden_size ** 0.5)
        
        # combine the embeddings of the text and images and mask out padding tokens
        embeds = torch.zeros(
            batch_size, sequence_length, embed_dim, dtype=dtype, device=device
        )
        
        # shape (batch_size, sequence_length) True for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # shape (batch_size, sequence_length) True for image tokens
        image_mask = input_ids == self.config.image_token_index
        # shape (batch_size, sequence_length) True for padding tokens
        padding_mask = input_ids == self.pad_token_id
        
        # we need to expand the masks to the embedding dimension otherwise we can't use torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand_as(inputs_embed)
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(scaled_images_embed)
        padding_mask_expanded = padding_mask.unsqueeze(-1).expand_as(inputs_embed)
        
        # add the text embeddings to the combined embeddings
        embeds = torch.where(text_mask_expanded, inputs_embed, embeds)
        # add the image embeddings to the combined embeddings
        # we will use mask scatter here as the scaled_images_embed has a different shape
        embeds = torch.masked_scatter(image_mask_expanded, scaled_images_embed)
        # mask out the padding tokens
        embeds = torch.where(padding_mask_expanded, torch.zeros_like(embeds), embeds)
        
        # Create the attention mask
        
        min_dtype = torch.finfo(dtype).min
        
        if kv_cache is None or kv_cache.num_items == 0:
            # do not mast any tokens, because we are in prefilling phase
            # this only works when we have no padding
            causal_mask = torch.full(
                (batch_size, sequence_length, sequence_length),
                fill_value=0,
                dtype=dtype,
                device=device,
            )
        else:
            # we are in generation phase so query must be one single token
            assert sequence_length == 1, "Query must be one single token"
            kv_len = kv_cache.num_items() + sequence_length
            # also in this case we do not mask any tokens, since it needs to attend to all previous tokens
            # this only works when we have no padding
            causal_mask = torch.full(
                (batch_size, sequence_length, kv_len),
                fill_value=0,
                dtype=dtype,
                device=device,
            )
        
        # Add the head dimension
        # [batch_size, q_len, kv_len] -> [batch_size, num_heads, q_len, kv_len]
        # unsqueeze(1) adds the extra dimension to do mat addition
        causal_mask = causal_mask.unsqueeze(1)
        
        if kv_cache is not None and kv_cache.num_items > 0:
            # the position of the query is just the last position
            position_ids = attention_mask.cumsum(dim=-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(-1)
        
        else:
            # create a position ids based on the size og the attention mask
            # for masked token, use the number 1 as the position
            position_ids = (attention_mask.cumsum(dim=-1)).masked_fill(
                (attention_mask == 0), 1
            ).to(device)
        
        return embeds, causal_mask, position_ids
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,

    ) -> Tuple:
        
        assert torch.all(attention_mask == 1), "Make sure that the attention mask is set to 1 for all tokens as input cannot be padded"
        
        # 1. Extract the language model input embeddings
        # Shape (batch_size, sequence_length, hidden_size)
        inputs_embed = self.language_model.get_input_embeddings()(input_ids)
        
        # 2. Merge text and images
        # Shape (batch_size, channels, height, width) -> (batch_size, num_patches, embedding_dim)
        images_embed = self.vision_model(pixel_values.to(inputs_embed.dtype))        
        images_embed = self.multi_model_projection(images_embed)
        
        # 3. Concatenate the text and image embeddings
        # Shape (batch_size, sequence_length, hidden_size)
        
        inputs_embed, attention_mask, position_ids = self._merge_text_and_images(
            images_embed=images_embed,
            inputs_embed=inputs_embed,
            attention_mask=attention_mask,
            input_ids=input_ids,
            kv_cache=kv_cache,
        )
        
        # 4. Forward pass through the language_model
        
        outputs = self.language_model(
            inputs_embed=inputs_embed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        
        return outputs
        
        