#!/usr/bin/env python3
"""
Prepare INL-LLM checkpoint for HuggingFace distribution.

Usage:
    python prepare_hf_model.py checkpoint.pt --output pacific-prime_3.8B
"""
import os
import json
import shutil
import argparse
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer


# Model configurations
MODEL_CONFIGS = {
    "student_3.8b": {
        "architectures": ["UltraOptimizedIntegratorLanguageModel"],
        "model_type": "inl-llm-v3",
        "vocab_size": 50261,
        "d_model": 3072,
        "num_layers": 32,
        "num_heads": 24,
        "num_kv_heads": 6,
        "num_iterations_per_layer": 2,
        "feedforward_dim": 12288,
        "max_seq_len": 1024,
        "dropout": 0.1,
        "rope_base": 10000.0,
        "use_lowrank_embeddings": True,
        "lowrank_ratio": 0.125,
        "use_gradient_checkpointing": False,
        "use_shared_controllers": True,
        "use_adaptive_stopping": True,
        "adaptive_convergence_threshold": 0.001,
        "hierarchical_group_size": 64,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "pad_token_id": 50256,
        "torch_dtype": "bfloat16"
    },
    "teacher_500m": {
        "architectures": ["IntegratorLanguageModel"],
        "model_type": "inl-llm-v2",
        "vocab_size": 50261,
        "d_model": 1280,
        "num_layers": 18,
        "num_heads": 20,
        "num_iterations_per_layer": 2,
        "feedforward_dim": 5120,
        "max_seq_len": 1024,
        "dropout": 0.1,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "pad_token_id": 50256,
        "torch_dtype": "float32"
    }
}


def prepare_hf_model(checkpoint_path: str, output_dir: str, config_name: str = "student_3.8b"):
    """
    Prepare a checkpoint for HuggingFace distribution.

    Args:
        checkpoint_path: Path to .pt checkpoint
        output_dir: Output directory (e.g., "pacific-prime_3.8B")
        config_name: Model config to use
    """
    print(f"=" * 60)
    print(f"Preparing HuggingFace model: {output_dir}")
    print(f"=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and convert checkpoint to safetensors
    print(f"\n[1/4] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        step = ckpt.get('global_step', 'unknown')
        print(f"  Checkpoint from step: {step}")
    else:
        state_dict = ckpt
        step = 'unknown'

    # Count parameters
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"  Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")

    # Clone tensors to break shared memory
    print("  Cloning tensors...")
    state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}

    # Save as safetensors
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    print(f"  Saving to: {safetensors_path}")
    save_file(state_dict, safetensors_path)

    # 2. Save config.json
    print(f"\n[2/4] Creating config.json")
    config = MODEL_CONFIGS.get(config_name, MODEL_CONFIGS["student_3.8b"]).copy()
    config["_name_or_path"] = output_dir

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config.json")

    # 3. Copy tokenizer files from GPT-2
    print(f"\n[3/4] Setting up tokenizer (GPT-2)")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir)
    print(f"  Saved tokenizer files")

    # 4. Create generation_config.json
    print(f"\n[4/4] Creating generation_config.json")
    generation_config = {
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "pad_token_id": 50256,
        "max_length": 1024,
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1
    }
    with open(os.path.join(output_dir, "generation_config.json"), "w") as f:
        json.dump(generation_config, f, indent=2)
    print(f"  Saved generation_config.json")

    # Create README
    readme = f"""---
license: cc-by-nc-4.0
language:
- en
tags:
- code
- inl-llm
- integrator-neuron
---

# Pacific-Prime 3.8B (INL-LLM v3)

Integrator Neural Language Model - A novel architecture based on integrator dynamics.

## Model Details

| Parameter | Value |
|-----------|-------|
| Parameters | {total_params/1e9:.2f}B |
| Architecture | INL-LLM v3 |
| d_model | {config.get('d_model', 3072)} |
| Layers | {config.get('num_layers', 32)} |
| Heads | {config.get('num_heads', 24)} |
| KV Heads | {config.get('num_kv_heads', 6)} |
| Context | {config.get('max_seq_len', 1024)} |
| Training | Distillation from 500M teacher |

## Usage

```python
import torch
from transformers import AutoTokenizer
from inl_llm_v3 import UltraOptimizedIntegratorLanguageModel

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{output_dir}")

# Load model
model = UltraOptimizedIntegratorLanguageModel(
    vocab_size=50261,
    d_model={config.get('d_model', 3072)},
    num_layers={config.get('num_layers', 32)},
    num_heads={config.get('num_heads', 24)},
    num_kv_heads={config.get('num_kv_heads', 6)},
    feedforward_dim={config.get('feedforward_dim', 12288)},
    max_seq_len=1024
)

from safetensors.torch import load_file
model.load_state_dict(load_file("{output_dir}/model.safetensors"))
model.eval()

# Generate
prompt = "def fibonacci(n):"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## License

CC BY-NC 4.0
"""

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme)
    print(f"  Saved README.md")

    # Summary
    print(f"\n" + "=" * 60)
    print(f"SUCCESS! Model prepared in: {output_dir}/")
    print(f"=" * 60)
    print(f"\nFiles created:")
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f))
        if size > 1e9:
            print(f"  - {f} ({size/1e9:.2f} GB)")
        elif size > 1e6:
            print(f"  - {f} ({size/1e6:.2f} MB)")
        else:
            print(f"  - {f} ({size/1e3:.1f} KB)")

    print(f"\nTo upload to HuggingFace:")
    print(f"  huggingface-cli upload Pacific-Prime/{output_dir} {output_dir}/")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Prepare INL-LLM for HuggingFace")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--output", "-o", type=str, default="pacific-prime_3.8B",
                        help="Output directory name")
    parser.add_argument("--config", "-c", type=str, default="student_3.8b",
                        choices=["student_3.8b", "teacher_500m"],
                        help="Model configuration")

    args = parser.parse_args()
    prepare_hf_model(args.checkpoint, args.output, args.config)


if __name__ == "__main__":
    main()
