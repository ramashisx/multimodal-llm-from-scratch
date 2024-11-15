import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SiglipVisionConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    num_image_tokens: Optional[int] = None
    kwargs: dict = field(default_factory=dict)  # To store any additional keyword arguments


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid" # This indicates no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:

        _, _, height, width = pixel_values.shape # [Batch_Size, Channel, Height, Width]
        # Convolve the `patch_size` kernel over the image without any overlap
        # The output shape will be [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]`
        # Here Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeddings = self.patch_embedding(pixel_values)
        # Reshape the output to [Batch_Size, Encoded_Dim, Num_Patches]
        # Here Num_Patches = Num_Patches_H * Num_Patches_W
        # This is done to make the patches as tokens
        patch_embeddings = patch_embeddings.flatten(2)
        # Transpose the output from 
        # [Batch_Size, Encoded_Dim, Num_Patches] to [Batch_Size, Num_Patches, Encoded_Dim]
        embeddings = patch_embeddings.transpose(1, 2)
        # Add position embeddings to the patches
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fc1 = nn.Linear(self.embed_dim, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # hidden_states: [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_state: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _ = hidden_states.shape
        # query_state [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)
        # query_states: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # key_states: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # value_states: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        value_states = value_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # Calculate the attention scores Q * K^T / sqrt(Head_Dim) attention weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = torch.einsum('bhqd,bhkd->bhqk', query_states, key_states) * self.scale
        
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is {attn_weights.size()}"
            )
        
        # Apply the row-wise softmax to the attention weights
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(query_states)
        # Apply dropout to the attention weights
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # Apply the attention weights to the value_states
        attn_output = torch.matmul(attn_weights, value_states)
        
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention output should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is {attn_output.size()}"
            )
        
        # Concatenate the attention output
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads * Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        
        # Apply the output projection
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights
        

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual: [Batch_Size, Num_patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] 
        hidden_states = hidden_states + residual
        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = input_embeds
        for layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = layer(hidden_states=hidden_states)
        
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Channel, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_states = self.encoder(input_embeds=hidden_states)
        last_hidden_states = self.post_layernorm(last_hidden_states)

        return last_hidden_states


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channel, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)
        