#!/usr/bin/env python3
"""Convert .pt checkpoint to safetensors format."""
import torch
from safetensors.torch import save_file
import os

def convert_checkpoint(pt_path, output_path=None):
    """Convert a .pt checkpoint to safetensors."""
    print(f"Loading checkpoint: {pt_path}")

    # Load checkpoint
    ckpt = torch.load(pt_path, map_location='cpu')

    # Extract model state dict
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        step = ckpt.get('global_step', 'unknown')
        print(f"Checkpoint from step: {step}")
    else:
        state_dict = ckpt
        step = 'unknown'

    # Count parameters
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"Total parameters: {total_params:,}")

    # Clone all tensors to break shared memory (required for safetensors)
    print("Cloning tensors to break shared memory...")
    state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}

    # Output path
    if output_path is None:
        output_path = pt_path.replace('.pt', '.safetensors')

    # Save as safetensors
    print(f"Saving to: {output_path}")
    save_file(state_dict, output_path)

    # Compare file sizes
    pt_size = os.path.getsize(pt_path) / (1024**3)
    st_size = os.path.getsize(output_path) / (1024**3)
    print(f"Original .pt: {pt_size:.2f} GB")
    print(f"Safetensors:  {st_size:.2f} GB")
    print("Done!")

    return output_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pt_path = sys.argv[1]
    else:
        pt_path = "checkpoints/checkpoint-step-230000.pt"

    convert_checkpoint(pt_path)
