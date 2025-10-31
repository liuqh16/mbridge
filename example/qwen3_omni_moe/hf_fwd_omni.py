import argparse

import torch

try:
    from transformers import Qwen3OmniMoeThinkerForConditionalGeneration
except:
    print(f"Please install transformers>=4.58.0 or install from source")

from example.qwen3_omni_moe.load_model_and_forward import get_sample_for_forward

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load Qwen3OmniMoe model and test forward pass")
    parser.add_argument(
        "--model_path", type=str, required=True, help="HuggingFace model path"
    )
    parser.add_argument(
        "--use_audio", action="store_true", help="Test with audio input"
    )
    parser.add_argument(
        "--use_image", action="store_true", default=True, help="Test with image input"
    )
    args = parser.parse_args()

    # Load the model on the available device(s)
    torch.set_grad_enabled(False)
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # Preparation for inference
    inputs = get_sample_for_forward(
        args.model_path, 
        use_audio=args.use_audio,
        use_image=args.use_image
    )

    # Inference: Generation of the output
    hf_output = model.forward(**inputs)
    print(f"HF Forward output logits shape: {hf_output.logits.shape}")
    torch.save(hf_output.logits, "/tmp/hf_qwen3_omni_moe.pt")
    print("✓ HF forward pass completed successfully!")

