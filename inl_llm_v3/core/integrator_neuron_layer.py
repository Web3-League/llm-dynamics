"""
IntegratorNeuronLayer (INL) - Learnable Dynamics Architecture

This module implements a neural network layer with learnable integrator/velocity dynamics.
Key features:
- Initial convergence towards 5 (configurable target)
- Learnable controller parameters (alpha, beta, gating)
- Soft constraints allowing deviation when data requires it
- Deterministic and fully differentiable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any

# Optional: safetensors support for fast/secure model saving
try:
    from safetensors.torch import save_file, load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


class IntegratorNeuronLayer(nn.Module):
    """
    Implements learnable integrator dynamics with velocity control.

    Equations:
        error = x_t - mu
        alpha = alpha_base * exp(-kappa * ||error||)  [if dynamic_alpha=True]
        v_{t+1} = alpha * v_t + (1 - alpha) * v_cand - beta * error + harmonic_noise
        x_{t+1} = x_t + (dt * velocity_scale) * g * v_{t+1}

    where alpha_base, beta, g, v_cand are context-dependent learnable parameters
    computed by a fused MLP controller from inputs [h, x, v].
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 1,
        target_value: float = 5.0,
        dt: float = 0.1,
        hidden_controller: int = 64,
        init_alpha: float = 0.8,
        init_beta: float = 0.5,
        init_gate: float = 0.5,
        velocity_scale: float = 1.0,
        excitation_amplitude: float = 0.03,
        learnable_mu: bool = True,
        dynamic_alpha: bool = True,
        alpha_kappa: float = 1.0
    ):
        """
        Args:
            hidden_dim: Dimension of context embedding h_t
            output_dim: Dimension of state x (typically 1 for scalar prediction)
            target_value: Initial target value (default 5.0)
            dt: Time step for integration
            hidden_controller: Hidden size for controller MLPs
            init_alpha: Initial inertia coefficient
            init_beta: Initial correction coefficient
            init_gate: Initial gating value
            velocity_scale: Scale factor for velocity
            excitation_amplitude: Amplitude of deterministic harmonic noise
            learnable_mu: Use learnable equilibrium attractor
            dynamic_alpha: Use dynamic integration gain (α-control)
            alpha_kappa: Sensitivity parameter for dynamic alpha
        """
        super().__init__()

        # Validate hyperparameters
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if hidden_controller <= 0:
            raise ValueError(f"hidden_controller must be positive, got {hidden_controller}")
        if not 0 <= init_alpha <= 1:
            raise ValueError(f"init_alpha must be in [0, 1], got {init_alpha}")
        if init_beta < 0:
            raise ValueError(f"init_beta must be non-negative, got {init_beta}")
        if not 0 <= init_gate <= 1:
            raise ValueError(f"init_gate must be in [0, 1], got {init_gate}")
        if velocity_scale <= 0:
            raise ValueError(f"velocity_scale must be positive, got {velocity_scale}")
        if excitation_amplitude < 0:
            raise ValueError(f"excitation_amplitude must be non-negative, got {excitation_amplitude}")
        if alpha_kappa < 0:
            raise ValueError(f"alpha_kappa must be non-negative, got {alpha_kappa}")

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dt = dt
        self.velocity_scale = velocity_scale
        self.dynamic_alpha = dynamic_alpha
        self.alpha_kappa = alpha_kappa

        # Pre-compute constant for performance
        self._dt_velocity_scale = dt * velocity_scale

        # Learnable equilibrium attractor
        if learnable_mu:
            self.mu = nn.Parameter(torch.full((output_dim,), target_value))
            self.learnable_mu = True
        else:
            self.register_buffer('mu', torch.full((output_dim,), target_value))
            self.learnable_mu = False

        # Deterministic harmonic excitation
        # Store as buffer so it can be modified dynamically (e.g., by scheduler)
        self.register_buffer('excitation_amplitude', torch.tensor(excitation_amplitude, dtype=torch.float32))
        # Learnable frequency and phase per dimension (deterministic initialization)
        # Use deterministic initialization for reproducibility
        gen = torch.Generator()
        gen.manual_seed(42)  # Fixed seed for reproducibility
        self.excitation_gamma = nn.Parameter(torch.randn(output_dim, generator=gen) * 0.1 + 1.0)
        self.excitation_phi = nn.Parameter(torch.randn(output_dim, generator=gen) * 2 * math.pi)

        # Fused controller MLP - outputs all 4 parameters at once for GPU efficiency
        # Uses 3 separate inputs to avoid concat overhead
        # Input: h (hidden_dim), x (output_dim), v (output_dim)
        self.controller_h = nn.Linear(hidden_dim, hidden_controller)
        self.controller_x = nn.Linear(output_dim, hidden_controller)
        self.controller_v = nn.Linear(output_dim, hidden_controller)
        self.controller_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_controller, 4 * output_dim),  # 4x output for all params
        )

        # Store output_dim for splitting
        self._controller_output_dim = output_dim

        # Initialize controller input layers
        with torch.no_grad():
            nn.init.xavier_uniform_(self.controller_h.weight)
            nn.init.xavier_uniform_(self.controller_x.weight)
            nn.init.xavier_uniform_(self.controller_v.weight)
            self.controller_h.bias.zero_()
            self.controller_x.bias.zero_()
            self.controller_v.bias.zero_()

            # Initialize output layer to produce desired initial values
            bias = self.controller_mlp[-1].bias
            alpha_bias = bias[0*output_dim:1*output_dim]
            beta_bias = bias[1*output_dim:2*output_dim]
            gate_bias = bias[2*output_dim:3*output_dim]
            v_cand_bias = bias[3*output_dim:4*output_dim]

            alpha_bias.fill_(self._inverse_sigmoid(init_alpha))
            beta_bias.fill_(self._inverse_softplus(init_beta))
            gate_bias.fill_(self._inverse_sigmoid(init_gate))
            v_cand_bias.fill_(0.0)

            # Small random initialization for symmetry breaking
            self.controller_mlp[-1].weight.normal_(0.0, 0.01)

    @staticmethod
    def _inverse_sigmoid(y: float) -> float:
        """Inverse of sigmoid function for initialization."""
        y = max(min(y, 0.999), 0.001)  # Clamp to avoid inf
        return torch.tensor(y / (1 - y)).log().item()

    @staticmethod
    def _inverse_softplus(y: float) -> float:
        """Inverse of softplus function for initialization."""
        y = max(y, 0.001)
        return torch.tensor(y).expm1().log().item()

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor,
        step: int = 0,
        return_aux: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass computing one integration step.

        Args:
            h: Context embedding [batch_size, hidden_dim]
            x: Current state [batch_size, output_dim]
            v: Current velocity [batch_size, output_dim]
            step: Current iteration step for deterministic excitation
            return_aux: If False, skip creating aux dict (performance optimization)

        Returns:
            x_next: Next state [batch_size, output_dim]
            v_next: Next velocity [batch_size, output_dim]
            aux: Dictionary with controller parameters for monitoring (None if return_aux=False)
        """
        # Process inputs separately then sum (avoids concat overhead)
        # Fuse additions for better performance
        controller_hidden = self.controller_h(h)
        controller_hidden = controller_hidden + self.controller_x(x)
        controller_hidden = controller_hidden + self.controller_v(v)

        # Compute all controller parameters in one forward pass (GPU efficient)
        controller_output = self.controller_mlp(controller_hidden)

        # Split into individual parameters using torch.split (more efficient than slicing)
        alpha_base_raw, beta_raw, gate_raw, v_cand = torch.split(
            controller_output, self._controller_output_dim, dim=1
        )

        # Apply activations (fused when possible with inplace for memory efficiency)
        alpha_base = torch.sigmoid(alpha_base_raw)
        beta = F.softplus(beta_raw)
        gate = torch.sigmoid(gate_raw)
        # v_cand has no activation (linear output)

        # Compute error once (used in both alpha and velocity update)
        error = x - self.mu

        # Dynamic integration gain (α-control)
        if self.dynamic_alpha:
            # Only compute when needed (avoid torch.where overhead)
            imbalance = torch.norm(error, dim=-1, keepdim=True)
            alpha = alpha_base * torch.exp(-self.alpha_kappa * imbalance)
        else:
            alpha = alpha_base

        # Update velocity with error correction term
        v_next = alpha * v + (1 - alpha) * v_cand - beta * error

        # Add deterministic harmonic excitation (only if amplitude > 0)
        if self.excitation_amplitude.item() > 0:
            # Deterministic noise based on iteration step
            t = float(step)
            # harmonic_noise shape: [output_dim]
            harmonic_noise = self.excitation_amplitude * torch.sin(
                self.excitation_gamma * t + self.excitation_phi
            )
            # Broadcast to [batch_size, output_dim] - implicit broadcasting is efficient
            v_next = v_next + harmonic_noise

        # Update state with gated velocity (use pre-computed constant)
        x_next = x + self._dt_velocity_scale * gate * v_next

        # Return auxiliary info for monitoring/loss (only if requested)
        if return_aux:
            aux = {
                'alpha': alpha,
                'alpha_base': alpha_base,
                'beta': beta,
                'gate': gate,
                'v_cand': v_cand,
                'error': error,
                'mu': self.mu
            }
        else:
            aux = None

        return x_next, v_next, aux

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize state x and velocity v.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            x0: Initial state [batch_size, output_dim] initialized to learned mu
            v0: Initial velocity [batch_size, output_dim] initialized to 0
        """
        # Initialize to current learned equilibrium, ensure correct device
        # Move to device before expand for efficiency
        mu_on_device = self.mu.to(device)
        x0 = mu_on_device.unsqueeze(0).expand(batch_size, -1)
        v0 = torch.zeros((batch_size, self.output_dim), device=device)
        return x0, v0

    def reset_parameters(self) -> None:
        """
        Reset all learnable parameters to their initial values.
        Standard PyTorch method for parameter reinitialization.
        """
        # Reset controller layers
        nn.init.xavier_uniform_(self.controller_h.weight)
        nn.init.xavier_uniform_(self.controller_x.weight)
        nn.init.xavier_uniform_(self.controller_v.weight)
        self.controller_h.bias.zero_()
        self.controller_x.bias.zero_()
        self.controller_v.bias.zero_()

        # Reset output layer with proper initialization
        output_dim = self._controller_output_dim
        bias = self.controller_mlp[-1].bias

        # Use stored init values if available, otherwise use defaults
        init_alpha = getattr(self, '_init_alpha', 0.8)
        init_beta = getattr(self, '_init_beta', 0.5)
        init_gate = getattr(self, '_init_gate', 0.5)

        bias[0*output_dim:1*output_dim].fill_(self._inverse_sigmoid(init_alpha))
        bias[1*output_dim:2*output_dim].fill_(self._inverse_softplus(init_beta))
        bias[2*output_dim:3*output_dim].fill_(self._inverse_sigmoid(init_gate))
        bias[3*output_dim:4*output_dim].zero_()

        self.controller_mlp[-1].weight.normal_(0.0, 0.01)

        # Reset excitation parameters
        with torch.no_grad():
            gen = torch.Generator()
            gen.manual_seed(42)
            self.excitation_gamma.copy_(torch.randn(output_dim, generator=gen) * 0.1 + 1.0)
            self.excitation_phi.copy_(torch.randn(output_dim, generator=gen) * 2 * math.pi)

    def __repr__(self) -> str:
        """String representation for debugging."""
        # Use .item() for scalar tensors in repr (acceptable in non-critical path)
        exc_amp = self.excitation_amplitude.item() if self.excitation_amplitude.numel() == 1 else self.excitation_amplitude
        return (
            f"{self.__class__.__name__}(\n"
            f"  hidden_dim={self.hidden_dim}, output_dim={self.output_dim},\n"
            f"  dt={self.dt}, velocity_scale={self.velocity_scale},\n"
            f"  excitation_amplitude={exc_amp:.4f},\n"
            f"  learnable_mu={self.learnable_mu}, dynamic_alpha={self.dynamic_alpha},\n"
            f"  alpha_kappa={self.alpha_kappa}\n"
            f")"
        )


class IntegratorModel(nn.Module):
    """
    Complete model: Backbone + IntegratorNeuronLayer + Readout
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_iterations: int = 10,
        output_dim: int = 1,
        target_value: float = 5.0,
        **inl_kwargs: Any
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for backbone and INL
            num_layers: Number of layers in backbone MLP
            num_iterations: Number of integration steps T
            output_dim: Output dimension (1 for scalar regression)
            target_value: Target value for convergence (default 5.0)
            **inl_kwargs: Additional arguments for IntegratorNeuronLayer
        """
        super().__init__()

        # Validate hyperparameters
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_iterations <= 0:
            raise ValueError(f"num_iterations must be positive, got {num_iterations}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.output_dim = output_dim

        # Backbone: simple MLP (can be replaced with Transformer)
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            current_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Integrator Neuron Layer
        self.inl = IntegratorNeuronLayer(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            target_value=target_value,
            **inl_kwargs
        )

        # Readout layer
        self.readout = nn.Linear(output_dim, output_dim)
        # Initialize readout to identity transformation (no bias shift)
        # Since x is already initialized to target_value, we just pass it through
        with torch.no_grad():
            # Only set diagonal if square matrix
            if self.readout.weight.shape[0] == self.readout.weight.shape[1]:
                self.readout.weight.fill_(0.0)
                self.readout.weight.diagonal().fill_(1.0)
            else:
                # For non-square, use Xavier/Glorot initialization
                nn.init.xavier_uniform_(self.readout.weight)
            self.readout.bias.fill_(0.0)  # No bias - x already at target_value

    def _run_dynamics(
        self,
        inputs: torch.Tensor,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Internal method to run INL dynamics.

        Args:
            inputs: Input features [batch_size, input_dim]
            return_trajectory: If True, return full trajectory and aux info

        Returns:
            x: Final state [batch_size, output_dim]
            v: Final velocity [batch_size, output_dim]
            trajectory: Optional dict with trajectory info if return_trajectory=True
        """
        batch_size = inputs.shape[0]
        device = inputs.device

        # Compute context from backbone
        h = self.backbone(inputs)  # [B, hidden_dim]

        # Initialize state and velocity
        x, v = self.inl.init_state(batch_size, device)

        # Store trajectory if requested (pre-allocate for efficiency)
        if return_trajectory:
            # Pre-allocate tensors with empty (no initialization overhead)
            x_traj = torch.empty(batch_size, self.num_iterations + 1, self.output_dim, device=device)
            v_traj = torch.empty(batch_size, self.num_iterations + 1, self.output_dim, device=device)
            x_traj[:, 0] = x
            v_traj[:, 0] = v
            # For aux, we still need a list (dict values vary)
            aux_traj = []

        # Run integration steps
        for t in range(self.num_iterations):
            # Skip aux creation if not needed (performance)
            x, v, aux = self.inl(h, x, v, step=t, return_aux=return_trajectory)

            if return_trajectory:
                # Store directly in pre-allocated tensors (no detach needed, done at the end)
                x_traj[:, t + 1] = x
                v_traj[:, t + 1] = v
                # Only store essential aux info (skip redundant fields)
                aux_traj.append({
                    'alpha': aux['alpha'].detach(),
                    'beta': aux['beta'].detach(),
                    'error': aux['error'].detach()
                })

        if return_trajectory:
            trajectory = {
                'x': x_traj.detach(),  # Already stacked, just detach
                'v': v_traj.detach(),  # Already stacked, just detach
                'aux': aux_traj
            }
            return x, v, trajectory

        return x, v, None

    def forward(
        self,
        inputs: torch.Tensor,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through complete model.

        Args:
            inputs: Input features [batch_size, input_dim]
            return_trajectory: If True, return full trajectory and aux info

        Returns:
            output: Final prediction [batch_size, output_dim]
            trajectory: Optional dict with trajectory info if return_trajectory=True
        """
        x, v, trajectory = self._run_dynamics(inputs, return_trajectory)
        output = self.readout(x)

        if return_trajectory:
            return output, trajectory

        return output, None

    def get_final_state(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get final state x_T before readout."""
        x, _, _ = self._run_dynamics(inputs, return_trajectory=False)
        return x

    def get_learned_mu(self) -> Optional[torch.Tensor]:
        """
        Get the learned equilibrium attractor.

        Returns:
            Learned mu tensor if learnable_mu enabled, else None
        """
        if hasattr(self.inl, 'learnable_mu') and self.inl.learnable_mu:
            return self.inl.mu
        return None

    def save_safetensors(self, path: str) -> None:
        """
        Save model state dict using safetensors format.

        Args:
            path: Path to save file (e.g., 'model.safetensors')

        Requires: pip install safetensors
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError(
                "safetensors not installed. Install with: pip install safetensors"
            )

        save_file(self.state_dict(), path)

    def load_safetensors(self, path: str, strict: bool = True) -> None:
        """
        Load model state dict from safetensors format.

        Args:
            path: Path to safetensors file
            strict: Whether to strictly enforce matching keys

        Requires: pip install safetensors
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError(
                "safetensors not installed. Install with: pip install safetensors"
            )

        state_dict = load_file(path)
        self.load_state_dict(state_dict, strict=strict)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  input_dim={self.input_dim}, hidden_dim={self.hidden_dim},\n"
            f"  output_dim={self.output_dim}, num_iterations={self.num_iterations}\n"
            f")"
        )
