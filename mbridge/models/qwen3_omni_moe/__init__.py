# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024 Alibaba PAI Team.

"""Qwen3OmniMoe Thinker Bridge for MBridge"""

from copy import deepcopy
from typing import Callable, Optional

import torch

from mbridge.core import register_model
from mbridge.models.qwen3_vl.base_bridge import Qwen3VBaseBridge
from mbridge.models.qwen3_omni_moe.transformer_config import Qwen3OmniMoeTransformerConfig

# ========== Vision Model Weight Mappings ==========
# (Same as Qwen3VL but with 'thinker.visual' prefix)

_QWEN3OMNI_VIT_DIRECT_MAPPING = {
    "vision_model.patch_embed.proj.weight": "thinker.visual.patch_embed.proj.weight",
    "vision_model.patch_embed.proj.bias": "thinker.visual.patch_embed.proj.bias",
    "vision_model.pos_embed.weight": "thinker.visual.pos_embed.weight",
    # Vision merger (NOTE: Qwen3OmniMoe uses ln_q and mlp.0/2, not norm and linear_fc1/2!)
    "vision_model.merger.patch_norm.weight": "thinker.visual.merger.ln_q.weight",
    "vision_model.merger.patch_norm.bias": "thinker.visual.merger.ln_q.bias",
    "vision_model.merger.linear_fc1.weight": "thinker.visual.merger.mlp.0.weight",
    "vision_model.merger.linear_fc1.bias": "thinker.visual.merger.mlp.0.bias",
    "vision_model.merger.linear_fc2.weight": "thinker.visual.merger.mlp.2.weight",
    "vision_model.merger.linear_fc2.bias": "thinker.visual.merger.mlp.2.bias",
}

_QWEN3OMNI_VIT_ATTENTION_MAPPING = {
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_proj.weight": [
        "thinker.visual.blocks.{layer_number}.attn.proj.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_proj.bias": [
        "thinker.visual.blocks.{layer_number}.attn.proj.bias",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.bias": [
        "thinker.visual.blocks.{layer_number}.attn.qkv.bias",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
        "thinker.visual.blocks.{layer_number}.attn.qkv.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": [
        "thinker.visual.blocks.{layer_number}.norm1.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_bias": [
        "thinker.visual.blocks.{layer_number}.norm1.bias",
    ],
}

_QWEN3OMNI_VIT_MLP_MAPPING = {
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.weight": [
        "thinker.visual.blocks.{layer_number}.mlp.linear_fc1.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.bias": [
        "thinker.visual.blocks.{layer_number}.mlp.linear_fc1.bias",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc2.weight": [
        "thinker.visual.blocks.{layer_number}.mlp.linear_fc2.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc2.bias": [
        "thinker.visual.blocks.{layer_number}.mlp.linear_fc2.bias",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_weight": [
        "thinker.visual.blocks.{layer_number}.norm2.weight",
    ],
    "vision_model.decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_bias": [
        "thinker.visual.blocks.{layer_number}.norm2.bias",
    ],
}

_QWEN3OMNI_VIT_OTHER_MAPPING = {
    # Deepstack mergers (NOTE: Qwen3OmniMoe uses merger_list with ln_q and mlp.0/2!)
    "vision_model.decoder.deepstack_merger_list.{layer_number}.patch_norm.weight": [
        "thinker.visual.merger_list.{layer_number}.ln_q.weight",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.patch_norm.bias": [
        "thinker.visual.merger_list.{layer_number}.ln_q.bias",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.linear_fc1.weight": [
        "thinker.visual.merger_list.{layer_number}.mlp.0.weight",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.linear_fc1.bias": [
        "thinker.visual.merger_list.{layer_number}.mlp.0.bias",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.linear_fc2.weight": [
        "thinker.visual.merger_list.{layer_number}.mlp.2.weight",
    ],
    "vision_model.decoder.deepstack_merger_list.{layer_number}.linear_fc2.bias": [
        "thinker.visual.merger_list.{layer_number}.mlp.2.bias",
    ],
}


# ========== Audio Encoder Weight Mappings (NEW for OmniMoe) ==========

_QWEN3OMNI_AUDIO_DIRECT_MAPPING = {
    # Conv2d layers for downsampling (NOTE: HF uses conv2d1/2/3, not conv1/2/3!)
    "audio_encoder.conv2d1.weight": "thinker.audio_tower.conv2d1.weight",
    "audio_encoder.conv2d1.bias": "thinker.audio_tower.conv2d1.bias",
    "audio_encoder.conv2d2.weight": "thinker.audio_tower.conv2d2.weight",
    "audio_encoder.conv2d2.bias": "thinker.audio_tower.conv2d2.bias",
    "audio_encoder.conv2d3.weight": "thinker.audio_tower.conv2d3.weight",
    "audio_encoder.conv2d3.bias": "thinker.audio_tower.conv2d3.bias",
    # Output projection after conv
    "audio_encoder.conv_out.weight": "thinker.audio_tower.conv_out.weight",
    "audio_encoder.conv_out.bias": "thinker.audio_tower.conv_out.bias",
    # Sinusoidal positional embedding (fixed, not trainable - no weight mapping needed)
    # positional_embedding is a buffer, created at init time, not loaded from checkpoint
    # Layer norm
    "audio_encoder.ln_post.weight": "thinker.audio_tower.ln_post.weight",
    "audio_encoder.ln_post.bias": "thinker.audio_tower.ln_post.bias",
    # Final projection
    "audio_encoder.proj1.weight": "thinker.audio_tower.proj1.weight",
    "audio_encoder.proj1.bias": "thinker.audio_tower.proj1.bias",
    "audio_encoder.proj2.weight": "thinker.audio_tower.proj2.weight",
    "audio_encoder.proj2.bias": "thinker.audio_tower.proj2.bias",
}

# Audio encoder transformer layers mapping
# NOTE: PyTorch nn.TransformerEncoderLayer uses fused in_proj (Q+K+V), HF uses separate q/k/v_proj
# We use ModuleList so path is audio_encoder.layers.{layer_number}, not layers.layers
_QWEN3OMNI_AUDIO_OTHER_MAPPING = {
    # Self-attention: in_proj (Q+K+V fused) -> separate q/k/v projections
    "audio_encoder.layers.{layer_number}.self_attn.in_proj_weight": [
        "thinker.audio_tower.layers.{layer_number}.self_attn.q_proj.weight",
        "thinker.audio_tower.layers.{layer_number}.self_attn.k_proj.weight",
        "thinker.audio_tower.layers.{layer_number}.self_attn.v_proj.weight",
    ],
    "audio_encoder.layers.{layer_number}.self_attn.in_proj_bias": [
        "thinker.audio_tower.layers.{layer_number}.self_attn.q_proj.bias",
        "thinker.audio_tower.layers.{layer_number}.self_attn.k_proj.bias",
        "thinker.audio_tower.layers.{layer_number}.self_attn.v_proj.bias",
    ],
    # Self-attention: out_proj
    "audio_encoder.layers.{layer_number}.self_attn.out_proj.weight": [
        "thinker.audio_tower.layers.{layer_number}.self_attn.out_proj.weight",
    ],
    "audio_encoder.layers.{layer_number}.self_attn.out_proj.bias": [
        "thinker.audio_tower.layers.{layer_number}.self_attn.out_proj.bias",
    ],
    # Layer norms
    "audio_encoder.layers.{layer_number}.norm1.weight": [
        "thinker.audio_tower.layers.{layer_number}.self_attn_layer_norm.weight",
    ],
    "audio_encoder.layers.{layer_number}.norm1.bias": [
        "thinker.audio_tower.layers.{layer_number}.self_attn_layer_norm.bias",
    ],
    "audio_encoder.layers.{layer_number}.norm2.weight": [
        "thinker.audio_tower.layers.{layer_number}.final_layer_norm.weight",
    ],
    "audio_encoder.layers.{layer_number}.norm2.bias": [
        "thinker.audio_tower.layers.{layer_number}.final_layer_norm.bias",
    ],
    # FFN layers
    "audio_encoder.layers.{layer_number}.linear1.weight": [
        "thinker.audio_tower.layers.{layer_number}.fc1.weight",
    ],
    "audio_encoder.layers.{layer_number}.linear1.bias": [
        "thinker.audio_tower.layers.{layer_number}.fc1.bias",
    ],
    "audio_encoder.layers.{layer_number}.linear2.weight": [
        "thinker.audio_tower.layers.{layer_number}.fc2.weight",
    ],
    "audio_encoder.layers.{layer_number}.linear2.bias": [
        "thinker.audio_tower.layers.{layer_number}.fc2.bias",
    ],
}


@register_model("qwen3_omni_moe")
class Qwen3OmniMoeThinkerBridge(Qwen3VBaseBridge):
    """
    Bridge implementation for Qwen3OmniMoe Thinker models.

    Extends Qwen3VBaseBridge with audio encoder support.
    Weight mappings use 'thinker.*' prefix instead of 'model.*'.
    """

    TransformerConfigClass = Qwen3OmniMoeTransformerConfig

    def __init__(self, hf_config, **kwargs):
        """
        Initialize Qwen3OmniMoe Thinker Bridge.

        Note: We extract thinker_config from the top-level Qwen3OmniMoeConfig
        and use it as the working config, since we only train the thinker module.
        """
        # Extract thinker config and use it as the main config
        # This allows the base class methods to work correctly
        thinker_config = hf_config.thinker_config

        # Store the original config for reference if needed
        self._original_hf_config = hf_config

        # Initialize with thinker_config as the working config
        super().__init__(thinker_config, **kwargs)

    # All mappings explicitly use 'thinker.*' prefix
    _DIRECT_MAPPING = {
        **_QWEN3OMNI_VIT_DIRECT_MAPPING,
        **_QWEN3OMNI_AUDIO_DIRECT_MAPPING,
        # Language model mappings
        "language_model.embedding.word_embeddings.weight": "thinker.model.embed_tokens.weight",
        "language_model.decoder.final_layernorm.weight": "thinker.model.norm.weight",
        "language_model.output_layer.weight": "thinker.lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        **_QWEN3OMNI_VIT_ATTENTION_MAPPING,
        # Language model attention mappings
        "language_model.decoder.layers.{layer_number}.self_attention.linear_proj.weight": [
            "thinker.model.layers.{layer_number}.self_attn.o_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": [
            "thinker.model.layers.{layer_number}.input_layernorm.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.q_layernorm.weight": [
            "thinker.model.layers.{layer_number}.self_attn.q_norm.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.k_layernorm.weight": [
            "thinker.model.layers.{layer_number}.self_attn.k_norm.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
            "thinker.model.layers.{layer_number}.self_attn.q_proj.weight",
            "thinker.model.layers.{layer_number}.self_attn.k_proj.weight",
            "thinker.model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.bias": [
            "thinker.model.layers.{layer_number}.self_attn.q_proj.bias",
            "thinker.model.layers.{layer_number}.self_attn.k_proj.bias",
            "thinker.model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
    }

    _MLP_MAPPING = {
        **_QWEN3OMNI_VIT_MLP_MAPPING,
        # Language model MLP mappings (MoE)
        "language_model.decoder.layers.{layer_number}.pre_mlp_layernorm.weight": [
            "thinker.model.layers.{layer_number}.post_attention_layernorm.weight",
        ],
        "language_model.decoder.layers.{layer_number}.mlp.router.weight": [
            "thinker.model.layers.{layer_number}.mlp.gate.weight",
        ],
        # NOTE: Qwen3OmniMoe uses ModuleList for experts, each with separate gate_proj/up_proj
        # (unlike Qwen3VLMoe which uses fused gate_up_proj without expert_idx)
        "language_model.decoder.layers.{layer_number}.mlp.experts.linear_fc1.weight": [
            "thinker.model.layers.{layer_number}.mlp.experts.{expert_idx}.gate_proj.weight",
            "thinker.model.layers.{layer_number}.mlp.experts.{expert_idx}.up_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.mlp.experts.linear_fc2.weight": [
            "thinker.model.layers.{layer_number}.mlp.experts.{expert_idx}.down_proj.weight",
        ],
    }

    _OTHER_MAPPING = {
        **_QWEN3OMNI_VIT_OTHER_MAPPING,
        **_QWEN3OMNI_AUDIO_OTHER_MAPPING,
    }

    def _weight_name_mapping_other(self, name: str) -> list[str]:
        """
        Override to handle audio encoder weights which have different depth than vision/language.

        Audio encoder: audio_encoder.layers.{layer_number}.self_attn...
        Vision/Language: vision_model.decoder.layers.{layer_number}...
        """
        split_name = name.split(".")

        # Audio encoder has layer number at index 2
        if name.startswith("audio_encoder.layers."):
            layer_number = split_name[2]  # audio_encoder.layers.0 -> index 2
            split_name[2] = "{layer_number}"
            key = ".".join(split_name)
            mapping_names = self._OTHER_MAPPING[key]
            return [x.format(layer_number=layer_number) for x in mapping_names]

        # For vision/language, use base class logic (layer number at index 3)
        return super()._weight_name_mapping_other(name)

    def _adjust_mapping_for_shared_weights(self):
        """Adjust mappings for tied embeddings in Thinker model."""
        # self.hf_config is now thinker_config (set in __init__)
        text_config = self.hf_config.text_config
        if getattr(text_config, "tie_word_embeddings", False):
            self._DIRECT_MAPPING["language_model.output_layer.weight"] = "thinker.model.embed_tokens.weight"

    def _get_hf_shared_weight_keys(self):
        """Get shared weight keys for Thinker model."""
        # self.hf_config is now thinker_config (set in __init__)
        text_config = self.hf_config.text_config
        if getattr(text_config, "tie_word_embeddings", False):
            return ["thinker.model.embed_tokens.weight"]
        return []

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        """Handle MLP weight mapping including MoE experts."""
        # Vision model or standard language model layers
        if (
            name.startswith("vision_model.")
            or name.startswith("audio_encoder.")
            or ".pre_mlp_layernorm.weight" in name
            or ".mlp.router.weight" in name
        ):
            return super()._weight_name_mapping_mlp(name)

        # Language model MoE experts
        # NOTE: Qwen3OmniMoe uses ModuleList with separate expert weights
        # (unlike Qwen3VLMoe which uses fused experts)
        assert ".mlp.experts.linear_fc" in name
        split_name = name.split(".")
        layer_number = split_name[3]
        split_name[3] = "{layer_number}"
        key = ".".join(split_name)
        key = key.split(".weight")[0] + ".weight"

        # Get mapping template (contains {layer_number} and {expert_idx} placeholders)
        mapping_names = self._MLP_MAPPING[key]

        # Expand for all experts
        num_experts = self.hf_config.text_config.num_experts
        convert_names = []
        for expert_idx in range(num_experts):
            for pattern in mapping_names:
                # Replace both {layer_number} and {expert_idx}
                hf_name = pattern.format(layer_number=layer_number, expert_idx=expert_idx)
                convert_names.append(hf_name)

        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        """
        Convert HF weights to Megatron-Core format.

        Override to handle:
        1. Audio encoder attention (fused Q/K/V in_proj)
        2. Vision merger weights (transpose for TE)
        3. Qwen3OmniMoe's ModuleList expert structure
        """
        # Handle audio encoder attention: HF has separate q/k/v_proj, mbridge fuses into in_proj
        if "audio_encoder.layers." in mcore_weights_name and "self_attn.in_proj" in mcore_weights_name:
            # hf_weights = [q_proj, k_proj, v_proj] (each [hidden_size, hidden_size])
            # Concatenate along dim 0 to get [3 * hidden_size, hidden_size]
            assert len(hf_weights) == 3, f"Expected 3 weights for in_proj (Q/K/V), got {len(hf_weights)}"
            return torch.cat(hf_weights, dim=0).clone().contiguous()

        # Handle MoE experts specially for Qwen3OmniMoe
        if ".mlp.experts.linear_fc" in mcore_weights_name:
            # Extract expert index from Megatron weight name
            # Format: language_model.decoder.layers.X.mlp.experts.linear_fcN.weightM
            # where M is the local expert index
            local_experts_idx = int(mcore_weights_name.split(".weight")[-1])
            num_experts = self.config.num_moe_experts
            num_experts_per_rank = num_experts // self.mpu.ep_size
            # Calculate global expert index
            global_expert_idx = local_experts_idx + num_experts_per_rank * self.mpu.ep_rank

            # For linear_fc1: hf_weights contains [gate_0, up_0, gate_1, up_1, ..., gate_N, up_N]
            # For linear_fc2: hf_weights contains [down_0, down_1, ..., down_N]
            if "linear_fc1" in mcore_weights_name:
                # Each expert has 2 weights (gate_proj and up_proj)
                # Extract the corresponding gate and up projections for this expert
                gate_idx = global_expert_idx * 2  # gate_proj index
                up_idx = global_expert_idx * 2 + 1  # up_proj index
                gate_weight = hf_weights[gate_idx]
                up_weight = hf_weights[up_idx]
                # Concatenate gate and up (Megatron format)
                return torch.cat([gate_weight, up_weight], dim=0).clone().contiguous()
            else:  # linear_fc2
                # Each expert has 1 weight (down_proj)
                down_weight = hf_weights[global_expert_idx]
                return down_weight.clone().contiguous()

        # Handle vision merger and deepstack merger weights: TE Linear layers need transposed weights
        # MUST be before calling super() to intercept before parent class processes them
        if len(hf_weights) == 1 and ".weight" in mcore_weights_name:
            if (
                "vision_model.merger.linear_fc" in mcore_weights_name
                or "vision_model.decoder.deepstack_merger_list" in mcore_weights_name
            ):
                # Check if it's a linear layer weight (not norm/bias)
                hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
                # HF Qwen3OmniMoe: linear weights are in mlp.0 and mlp.2 (not named linear_fc)
                if any("mlp." in name for name in hf_names):
                    # TE Linear expects (in_features, out_features), HF stores (out_features, in_features)
                    return hf_weights[0].clone().contiguous()

        # For all other weights, use base class logic
        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weight: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        """
        Convert Megatron-Core weights to HF format.

        Override to handle:
        1. Audio encoder attention (split fused in_proj back to Q/K/V)
        2. Qwen3OmniMoe's ModuleList expert structure
        
        Returns:
            tuple: (hf_names, hf_weights) - lists of Hugging Face weight names and tensors
        """
        # Handle audio encoder attention: split fused in_proj back to separate q/k/v_proj
        if "audio_encoder.layers." in mcore_weights_name and "self_attn.in_proj" in mcore_weights_name:
            # Get the HF weight names for this Megatron weight
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            assert len(hf_names) == 3, f"Expected 3 HF names for in_proj (Q/K/V), got {len(hf_names)}"

            # mcore_weight shape: [3 * hidden_size, hidden_size]
            # Split into Q, K, V
            hidden_size = mcore_weight.shape[0] // 3
            q_weight = mcore_weight[:hidden_size].clone().contiguous()
            k_weight = mcore_weight[hidden_size : 2 * hidden_size].clone().contiguous()
            v_weight = mcore_weight[2 * hidden_size :].clone().contiguous()

            return hf_names, [q_weight, k_weight, v_weight]

        # Handle vision merger and deepstack merger weights: transpose back to HF format
        # MUST be before calling super() to intercept before parent class processes them
        if ".weight" in mcore_weights_name:
            if (
                "vision_model.merger.linear_fc" in mcore_weights_name
                or "vision_model.decoder.deepstack_merger_list" in mcore_weights_name
            ):
                hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
                # HF Qwen3OmniMoe: linear weights are in mlp.0 and mlp.2
                if any("mlp." in name for name in hf_names):
                    # Transpose back from TE format (in, out) to HF format (out, in)
                    return [hf_names[0]], [mcore_weight.clone().contiguous()]

        # Handle MoE experts specially for Qwen3OmniMoe
        if ".mlp.experts.linear_fc" in mcore_weights_name:
            # Get the HF weight names for this Megatron weight
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)

            # Extract expert index
            local_experts_idx = int(mcore_weights_name.split(".weight")[-1])
            num_experts = self.config.num_moe_experts
            num_experts_per_rank = num_experts // self.mpu.ep_size
            global_expert_idx = local_experts_idx + num_experts_per_rank * self.mpu.ep_rank

            if "linear_fc1" in mcore_weights_name:
                # Split Megatron's concatenated [gate; up] back to separate tensors
                # mcore_weight shape: [ffn_hidden_size * 2, hidden_size]
                gate_up = mcore_weight.clone().contiguous()
                mid = gate_up.shape[0] // 2
                gate_weight = gate_up[:mid]
                up_weight = gate_up[mid:]

                # hf_names contains [gate_0, up_0, gate_1, up_1, ..., gate_N, up_N]
                gate_idx = global_expert_idx * 2
                up_idx = global_expert_idx * 2 + 1

                return [hf_names[gate_idx], hf_names[up_idx]], [gate_weight, up_weight]
            else:  # linear_fc2
                # Just transpose back
                down_weight = mcore_weight.clone().contiguous()
                return [hf_names[global_expert_idx]], [down_weight]

        # For all other weights, use base class logic
        return super()._weight_to_hf_format(mcore_weights_name, mcore_weight)

    def _build_config(self):
        """Build configuration for Qwen3OmniMoe Thinker."""
        # self.hf_config is now thinker_config (set in __init__)
        text_config = self.hf_config.text_config
        vision_config = self.hf_config.vision_config
        audio_config = self.hf_config.audio_config

        config = self._build_base_config(
            text_config_key="text_config",
            layernorm_epsilon=text_config.rms_norm_eps,
            use_cpu_initialization=False,
            # MoE specific (CRITICAL for Qwen3OmniMoe)
            moe_ffn_hidden_size=text_config.moe_intermediate_size,
            moe_router_bias_update_rate=0.001,
            moe_router_topk=text_config.num_experts_per_tok,
            num_moe_experts=text_config.num_experts,
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
            moe_router_dtype="fp32",
            moe_router_load_balancing_type="none",  # default None for RL
            moe_grouped_gemm=True,
            moe_router_score_function="softmax",
            # Other optimizations
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            masked_softmax_fusion=False,
            deallocate_pipeline_outputs=True,
            async_tensor_model_parallel_allreduce=True,
            distribute_saved_activations=False,
            cp_comm_type="p2p",
            # Qwen specific
            moe_router_pre_softmax=False,
            qk_layernorm=True,
            # Qwen3VL specific
            mrope_section=text_config.rope_scaling.get(
                "mrope_section",
                [24, 20, 20],
            ),
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_channels=vision_config.in_channels,
            spatial_merge_size=vision_config.spatial_merge_size,
            num_position_embeddings=vision_config.num_position_embeddings,
            out_hidden_size=vision_config.out_hidden_size,
            deepstack_visual_indexes=deepcopy(vision_config.deepstack_visual_indexes),
        )

        # Add audio encoder parameters (NEW for OmniMoe)
        if audio_config is not None:
            config.audio_hidden_size = audio_config.d_model
            config.audio_num_layers = audio_config.encoder_layers
            config.audio_num_attention_heads = audio_config.encoder_attention_heads
            config.audio_intermediate_size = audio_config.encoder_ffn_dim
            config.audio_dropout = audio_config.dropout
            config.audio_num_mel_bins = audio_config.num_mel_bins
            config.audio_max_source_positions = audio_config.max_source_positions
            config.audio_downsample_hidden_size = audio_config.downsample_hidden_size
            config.audio_output_dim = audio_config.output_dim
            config.audio_n_window = audio_config.n_window
            config.audio_n_window_infer = audio_config.n_window_infer
            config.audio_conv_chunksize = audio_config.conv_chunksize

            # Token IDs (from thinker_config)
            config.audio_token_id = getattr(self.hf_config, "audio_token_id", 151646)

        return config

    def _model_provider(self, post_model_creation_callbacks: list[Callable[[torch.nn.Module], None]]):
        """
        Creates and returns a model provider function.

        Args:
            post_model_creation_callbacks: List of callbacks to be called after model creation

        Returns:
            function: A provider function that creates and returns a Qwen3OmniMoeThinkerModel instance
        """
        from mbridge.models.qwen3_vl.transformer_config import get_vision_model_config
        from mbridge.models.qwen3_vl.utils import PatchMergerSubmodules
        from mbridge.models.qwen3_omni_moe.model import Qwen3OmniMoeThinkerModel
        from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
        from megatron.core.extensions.transformer_engine import (
            TENorm,
            TEColumnParallelLinear,
            TERowParallelLinear,
        )

        # self.hf_config is now thinker_config (set in __init__)
        text_config = self.hf_config.text_config

        def provider(
            pre_process,
            post_process,
            add_decoder=True,
            add_encoder=True,
            vp_stage: Optional[int] = None,
        ):
            # Language model layer spec (use parent class method)
            language_transformer_layer_spec = self._get_transformer_layer_spec(vp_stage)

            # Vision model config (pass vision_config, not the full hf_config)
            vision_config = get_vision_model_config(deepcopy(self.config), self.hf_config.vision_config)
            vision_config.pipeline_model_parallel_size = 1
            vision_config.first_pipeline_num_layers = None

            vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()
            vision_patch_merger_spec = PatchMergerSubmodules(
                patch_norm=TENorm,
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            )

            setattr(self, "vision_config", vision_config)

            # Build model
            model = Qwen3OmniMoeThinkerModel(
                language_transformer_config=self.config,
                language_transformer_layer_spec=language_transformer_layer_spec,
                language_vocab_size=text_config.vocab_size,
                language_max_sequence_length=text_config.max_position_embeddings,
                vision_transformer_config=vision_config,
                vision_transformer_layer_spec=vision_transformer_layer_spec,
                vision_patch_merger_spec=vision_patch_merger_spec,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder,
                fp16_lm_cross_entropy=getattr(self.config, "fp16_lm_cross_entropy", False),
                language_rotary_base=text_config.rope_theta,
                language_share_embeddings_and_output_weights=text_config.tie_word_embeddings,
                image_token_id=self.hf_config.image_token_id,
                video_token_id=self.hf_config.video_token_id,
                vision_start_token_id=getattr(self.hf_config, "vision_start_token_id", 151652),
                audio_token_id=self.hf_config.audio_token_id,
            )

            # Execute callbacks
            for callback in post_model_creation_callbacks:
                callback(model)

            return model

        return provider
