#!/usr/bin/env python3
"""
Convert PyTorch checkpoint to SafeTensors format for HuggingFace upload.

Usage:
    python convert_to_safetensors.py checkpoints/final_model/pytorch_model.pt output/
"""

import torch
import argparse
import os
import json
import shutil
from safetensors.torch import save_file

def convert_checkpoint(input_path: str, output_dir: str):
    """Convert .pt checkpoint to .safetensors format."""

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint from {input_path}...")
    checkpoint = torch.load(input_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DDP training)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        # SafeTensors requires contiguous tensors
        if isinstance(value, torch.Tensor):
            cleaned_state_dict[new_key] = value.contiguous()

    print(f"Found {len(cleaned_state_dict)} tensors")

    # Save as safetensors
    output_path = os.path.join(output_dir, 'model.safetensors')
    print(f"Saving to {output_path}...")
    save_file(cleaned_state_dict, output_path)

    # Copy config and tokenizer files if they exist
    input_dir = os.path.dirname(input_path)
    files_to_copy = [
        'config.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'vocab.json',
        'merges.txt',
        'special_tokens_map.json',
        'added_tokens.json',
        'generation_config.json'
    ]

    for fname in files_to_copy:
        src = os.path.join(input_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, fname))
            print(f"  Copied {fname}")

    # Calculate size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Conversion complete!")
    print(f"   Output: {output_path}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"\nTo upload to HuggingFace:")
    print(f"   huggingface-cli upload Pacific-Prime/pacific-prime {output_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PyTorch checkpoint to SafeTensors')
    parser.add_argument('input', help='Path to .pt checkpoint file')
    parser.add_argument('output', help='Output directory')
    args = parser.parse_args()

    convert_checkpoint(args.input, args.output)
