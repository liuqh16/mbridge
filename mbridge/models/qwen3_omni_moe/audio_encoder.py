# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024 Alibaba PAI Team.

"""Qwen3OmniMoe Audio Encoder"""

import math
from typing import Optional

import torch
import torch.nn as nn
from megatron.core.transformer.module import MegatronModule

from mbridge.models.qwen3_omni_moe.transformer_config import Qwen3OmniMoeTransformerConfig


class SinusoidsPositionEmbedding(nn.Module):
    """Sinusoidal positional embeddings (fixed, not learnable)."""

    def __init__(self, length: int, channels: int, max_timescale: float = 10000.0):
        super().__init__()
        self.length = length
        self.channels = channels

        # Compute sinusoidal positional embeddings
        position = torch.arange(length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2, dtype=torch.float) * -(math.log(max_timescale) / channels))

        positional_embedding = torch.zeros(length, channels)
        positional_embedding[:, 0::2] = torch.sin(position * div_term)
        positional_embedding[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, won't be trained)
        self.register_buffer("positional_embedding", positional_embedding, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        """Return positional embeddings for sequence length."""
        return self.positional_embedding[:seqlen, :]


class Qwen3OmniMoeAudioEncoder(MegatronModule):
    """Audio encoder for Qwen3OmniMoe.

    Simplified but compatible implementation matching HF weight structure.
    Uses Conv2d and sinusoidal position embeddings like the original.

    Args:
        transformer_config: Configuration object
    """

    def __init__(
        self,
        transformer_config: Qwen3OmniMoeTransformerConfig,
    ) -> None:
        super().__init__(config=transformer_config)

        self.num_mel_bins = transformer_config.audio_num_mel_bins
        self.hidden_size = transformer_config.audio_hidden_size
        self.output_dim = transformer_config.audio_output_dim
        self.downsample_hidden_size = transformer_config.audio_downsample_hidden_size
        
        # Additional config attributes from HF
        self.dropout = getattr(transformer_config, 'audio_dropout', 0.0)
        self.n_window = getattr(transformer_config, 'audio_n_window', 32000)
        self.n_window_infer = getattr(transformer_config, 'audio_n_window_infer', 1600)
        self.conv_chunksize = getattr(transformer_config, 'audio_conv_chunksize', 32)

        # Conv2d layers for downsampling (matching HF implementation)
        # Input: [batch, 1, num_mel_bins, seq_len] - mel-spectrogram is 2D (freq x time)
        # NOTE: HF uses conv2d1/2/3, not conv1/2/3!
        self.conv2d1 = nn.Conv2d(
            1,  # Single channel input (mel-spectrogram)
            self.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.conv2d2 = nn.Conv2d(
            self.downsample_hidden_size,
            self.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.conv2d3 = nn.Conv2d(
            self.downsample_hidden_size,
            self.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )

        # Output projection after conv layers
        # CRITICAL: Calculate the flattened dimension after 3 stride-2 convolutions
        # freq_dim after conv: ((((num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        # flattened_dim = downsample_hidden_size * freq_dim_after_conv
        freq_dim_after_conv = ((((self.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2)
        conv_out_input_dim = self.downsample_hidden_size * freq_dim_after_conv
        self.conv_out = nn.Linear(conv_out_input_dim, self.hidden_size, bias=False)

        # Sinusoidal positional embedding (fixed, not learnable)
        self.pos_emb = SinusoidsPositionEmbedding(
            length=transformer_config.audio_max_source_positions,
            channels=self.hidden_size,
        )

        # Encoder layers (ModuleList to match HF structure exactly)
        # HF: self.layers = nn.ModuleList([Qwen3OmniMoeAudioEncoderLayer(...) for _ in range(num_layers)])
        # We use standard TransformerEncoderLayer but in a ModuleList for correct weight names
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=transformer_config.audio_num_attention_heads,
                dim_feedforward=transformer_config.audio_intermediate_size,
                dropout=transformer_config.audio_dropout,
                activation="gelu",
                batch_first=False,  # seq_len first
                norm_first=True,
            )
            for _ in range(transformer_config.audio_num_layers)
        ])

        # Layer norm
        self.ln_post = nn.LayerNorm(self.hidden_size)

        # Output projection to match language model hidden size
        # HF: proj1 (1280->1280), act (GELU), proj2 (1280->2048)
        self.proj1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.proj2 = nn.Linear(self.hidden_size, self.output_dim, bias=True)

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of audio encoder.

        Args:
            input_features: Audio mel-spectrogram features
                           Shape: [num_mel_bins, seq_len] or [batch, num_mel_bins, seq_len]
            feature_lens: Length of each audio sequence (optional)

        Returns:
            Audio embeddings [seq_len, hidden_dim] or [seq_len, batch, hidden_dim]
        """
        # Ensure input is 4D: [batch, 1, num_mel_bins, seq_len] for Conv2d
        if input_features.dim() == 2:
            # [num_mel_bins, seq_len] -> [1, 1, num_mel_bins, seq_len]
            x = input_features.unsqueeze(0).unsqueeze(0)
        elif input_features.dim() == 3:
            # [batch, num_mel_bins, seq_len] -> [batch, 1, num_mel_bins, seq_len]
            x = input_features.unsqueeze(1)
        else:
            x = input_features

        # Conv2d layers (with GELU activation)
        # [batch, 1, num_mel_bins, seq_len] -> [batch, downsample_hidden_size, freq//2, time//2]
        x = torch.nn.functional.gelu(self.conv2d1(x))
        # [batch, C, H//2, W//2] -> [batch, C, H//4, W//4]
        x = torch.nn.functional.gelu(self.conv2d2(x))
        # [batch, C, H//4, W//4] -> [batch, C, H//8, W//8]
        x = torch.nn.functional.gelu(self.conv2d3(x))

        # Reshape to match HF format: [batch, channels, freq, time] -> [batch, time, channels*freq]
        # This matches HF's: padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        batch_size, channels, height, width = x.shape
        
        # Permute: [batch, channels, height, width] -> [batch, width, channels, height]
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Flatten last two dims: [batch, width, channels, height] -> [batch, width, channels*height]
        x = x.view(batch_size, width, channels * height)
        
        # Transpose to sequence-first: [batch, width, channels*height] -> [width, batch, channels*height]
        x = x.transpose(0, 1).contiguous()

        # Project to hidden_size
        # [seq_len, batch, channels*height] -> [seq_len, batch, hidden_size]
        x = self.conv_out(x)

        # If batch size is 1, squeeze it out for compatibility
        if x.shape[1] == 1:
            x = x.squeeze(1)  # [seq_len//4, hidden_size]

        # Add sinusoidal positional embedding
        seq_len = x.shape[0]
        pos_emb = self.pos_emb(seq_len)

        if x.dim() == 2:
            # [seq_len, hidden_size]
            x = x + pos_emb
        else:
            # [seq_len, batch, hidden_size]
            x = x + pos_emb.unsqueeze(1)

        # Transformer encoder layers (manually iterate through ModuleList)
        # Input/Output: [seq_len, batch, hidden_size] or [seq_len, hidden_size]
        for layer in self.layers:
            x = layer(x)

        # Post layer norm
        x = self.ln_post(x)

        # Project to output dimension (language model hidden size)
        x = torch.nn.functional.gelu(self.proj1(x))
        x = self.proj2(x)

        return x
