# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024 Alibaba PAI Team.

"""Qwen3OmniMoe Transformer Configuration"""

from dataclasses import dataclass
from typing import Optional

from mbridge.models.qwen3_vl.transformer_config import Qwen3VLTransformerConfig


@dataclass
class Qwen3OmniMoeTransformerConfig(Qwen3VLTransformerConfig):
    """Configuration for Qwen3OmniMoe model.

    Extends Qwen3VLTransformerConfig with audio encoder parameters.
    """

    # Audio encoder config
    audio_hidden_size: int = 1280
    audio_num_layers: int = 32
    audio_num_attention_heads: int = 16
    audio_intermediate_size: int = 5120
    audio_dropout: float = 0.0
    audio_num_mel_bins: int = 128
    audio_max_source_positions: int = 3000
    audio_downsample_hidden_size: int = 1280
    audio_output_dim: int = 3584
    audio_n_window: int = 750
    audio_n_window_infer: int = 750
    audio_conv_chunksize: int = 3

    # Token IDs
    audio_token_id: int = 151664
