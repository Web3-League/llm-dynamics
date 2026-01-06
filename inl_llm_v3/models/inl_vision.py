"""
INL-Vision: Image-to-Image model based on Integrator Neuron dynamics

Adapts the INL-LLM architecture for vision tasks by treating image patches
as tokens and using the same equilibrium-based dynamics.

Author: Boris Peyriguère
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from inl_llm_v3.optimizations.optimizations import (
    LowRankEmbedding,
    AdaptiveIntegratorNeuronLayer
)
from inl_llm_v3.core.integrator_neuron_layer import IntegratorNeuronLayer


class SimpleINLDynamics(nn.Module):
    """
    Simplified Integrator Neuron Layer for vision.

    Uses integrator dynamics without the full complexity of INL:
    - x_{t+1} = x_t + dt * MLP(x_t)
    - Iterated num_iterations times for equilibrium

    This gives similar dynamics but simpler implementation.
    """
    def __init__(
        self,
        d_model: int,
        num_iterations: int = 5,
        dt: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.num_iterations = num_iterations
        self.dt = dt

        # Simple MLP for dynamics
        self.dynamics_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input state (B, seq_len, d_model)

        Returns:
            Final state after iterations (B, seq_len, d_model)
        """
        # Iterate to refine representation
        state = x
        for _ in range(self.num_iterations):
            delta = self.dynamics_mlp(state)
            state = state + self.dt * delta

        return state


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.
    Similar to ViT (Vision Transformer) patch embedding.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Convolutional projection
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Project patches
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)

        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        return x


class INLVisionBlock(nn.Module):
    """
    Vision block using Integrator Neuron Layer dynamics.
    Applies equilibrium-based processing to image patch embeddings.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_iterations: int,
        layer_idx: int,
        feedforward_dim: int,
        dropout: float = 0.1,
        group_size: int = 64,
        excitation_sparsity: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.num_iterations = num_iterations
        self.layer_idx = layer_idx

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)

        # Multi-head attention (for patch-to-patch interactions)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout)
        )

        # Use simplified INL dynamics for vision
        self.inl_layer = SimpleINLDynamics(
            d_model=d_model,
            num_iterations=num_iterations,
            dt=0.1
        )

    def forward(self, x, return_trajectory=False):
        """
        Forward pass with integrator dynamics.

        Args:
            x: (B, num_patches, d_model)
            return_trajectory: Return full dynamics trajectory
        """
        trajectory = None

        # Self-attention on patches
        attn_out, _ = self.attention(
            self.norm_attn(x),
            self.norm_attn(x),
            self.norm_attn(x)
        )
        x = x + attn_out

        # Apply integrator dynamics to patch embeddings (iterate multiple times)
        x_normed = self.norm1(x)

        # Run integrator dynamics (wrapper handles iterations internally)
        inl_out = self.inl_layer(x_normed)
        x = x + inl_out

        trajectory = None  # Simplified: no trajectory tracking yet

        # Feedforward
        x = x + self.ffn(self.norm2(x))

        return (x, trajectory) if return_trajectory else x


class INLVisionModel(nn.Module):
    """
    Complete INL-Vision model for image-to-image tasks.

    Uses integrator neuron dynamics to process image patches iteratively,
    allowing the model to refine representations through equilibrium-based dynamics.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        out_channels: int = 3,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_iterations_per_layer: int = 5,
        feedforward_dim: int = None,
        dropout: float = 0.1,
        # Optimizations
        use_shared_controllers: bool = True,
        hierarchical_group_size: int = 64,
        excitation_sparsity: float = 0.1
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers

        if feedforward_dim is None:
            feedforward_dim = 4 * d_model

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=d_model
        )

        num_patches = self.patch_embed.num_patches

        # Positional encoding for patches
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, d_model) * 0.02
        )

        # Note: For simplicity in this vision model, we don't use shared controllers
        # Each block has its own integrator layer
        self.use_shared_controllers = use_shared_controllers
        if use_shared_controllers:
            print(f"ℹ️  Shared controllers disabled for INL-Vision (using per-layer controllers)")
        self.shared_controller = None

        # Vision blocks with integrator dynamics
        self.blocks = nn.ModuleList([
            INLVisionBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_iterations=num_iterations_per_layer,
                layer_idx=i,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                group_size=hierarchical_group_size,
                excitation_sparsity=excitation_sparsity
            )
            for i in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Decoder: patches back to image
        self.decoder = nn.Sequential(
            nn.Linear(d_model, patch_size * patch_size * out_channels),
            nn.Tanh()  # Output in [-1, 1]
        )

        self.out_channels = out_channels

    def forward(self, x, return_aux=False):
        """
        Forward pass.

        Args:
            x: Input image (B, C, H, W)
            return_aux: Return auxiliary information (trajectories)

        Returns:
            Output image (B, C, H, W)
            Optional: trajectories from all layers
        """
        B, C, H, W = x.shape

        # Embed patches
        x = self.patch_embed(x)  # (B, num_patches, d_model)

        # Add positional encoding
        x = x + self.pos_embedding

        # Apply vision blocks with integrator dynamics
        trajectories = []
        for block in self.blocks:
            if return_aux:
                x, traj = block(x, return_trajectory=True)
                trajectories.append(traj)
            else:
                x = block(x)

        # Final norm
        x = self.norm(x)

        # Decode patches back to image
        x = self.decoder(x)  # (B, num_patches, patch_size^2 * C)

        # Reshape to image
        num_patches_per_side = self.img_size // self.patch_size
        x = x.reshape(B, num_patches_per_side, num_patches_per_side,
                     self.patch_size, self.patch_size, self.out_channels)

        # Rearrange to (B, C, H, W)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, self.out_channels, self.img_size, self.img_size)

        if return_aux:
            return x, trajectories
        return x

    def get_num_params(self):
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_inl_vision_model(size='small', img_size=224, **kwargs):
    """
    Factory function to create INL-Vision models of different sizes.

    Args:
        size: 'tiny', 'small', 'base', 'large'
        img_size: Input image size
        **kwargs: Override default parameters
    """
    configs = {
        'tiny': {
            'd_model': 192,
            'num_layers': 12,
            'num_heads': 3,
            'feedforward_dim': 768
        },
        'small': {
            'd_model': 384,
            'num_layers': 12,
            'num_heads': 6,
            'feedforward_dim': 1536
        },
        'base': {
            'd_model': 768,
            'num_layers': 12,
            'num_heads': 12,
            'feedforward_dim': 3072
        },
        'large': {
            'd_model': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'feedforward_dim': 4096
        }
    }

    config = configs.get(size, configs['small'])
    config.update(kwargs)
    config['img_size'] = img_size

    return INLVisionModel(**config)
