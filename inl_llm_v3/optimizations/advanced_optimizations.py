"""
Advanced Optimizations for INL-LLM

Implements additional efficiency techniques:
1. Shared Controllers: Share control MLPs across layers (-15-20% params)
2. Sparse Harmonic Excitation: Only excite subset of dimensions (-10x compute)
3. Mixture of Integrators (MoI): Conditional computation like MoE
4. Hierarchical Equilibrium: Global + local offsets for μ

Author: Boris Peyriguère
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


class SharedController(nn.Module):
    """
    Shared controller MLP across multiple INL layers.

    Instead of each layer having its own controller (α, β, g, v_cand),
    we use ONE shared controller + small layer-specific modulation.

    Benefit: 15-20% parameter reduction on controller networks
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        hidden_controller: int = 64
    ):
        """
        Args:
            hidden_dim: Context dimension
            output_dim: State dimension
            num_layers: Number of layers sharing this controller
            hidden_controller: Hidden size for controller MLP
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Single shared controller (used by all layers)
        self.controller_h = nn.Linear(hidden_dim, hidden_controller)
        self.controller_x = nn.Linear(output_dim, hidden_controller)
        self.controller_v = nn.Linear(output_dim, hidden_controller)
        self.controller_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_controller, 4 * output_dim)
        )

        # Layer-specific modulation (tiny parameters)
        # Each layer gets 4 scalar multipliers (α, β, g, v_cand)
        self.layer_scalers = nn.Parameter(torch.ones(num_layers, 4))
        self.layer_biases = nn.Parameter(torch.zeros(num_layers, 4))

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize controller weights."""
        with torch.no_grad():
            nn.init.xavier_uniform_(self.controller_h.weight)
            nn.init.xavier_uniform_(self.controller_x.weight)
            nn.init.xavier_uniform_(self.controller_v.weight)
            self.controller_h.bias.zero_()
            self.controller_x.bias.zero_()
            self.controller_v.bias.zero_()
            self.controller_mlp[-1].weight.normal_(0.0, 0.01)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute controller parameters for specific layer.

        Args:
            h: Context [batch, hidden_dim]
            x: State [batch, output_dim]
            v: Velocity [batch, output_dim]
            layer_idx: Which layer is requesting control

        Returns:
            alpha, beta, gate, v_cand (all [batch, output_dim])
        """
        # Shared computation
        controller_hidden = self.controller_h(h) + self.controller_x(x) + self.controller_v(v)
        controller_output = self.controller_mlp(controller_hidden)

        # Split into components
        alpha_base, beta_base, gate_base, v_cand_base = torch.split(
            controller_output, self.output_dim, dim=1
        )

        # Layer-specific modulation
        scaler = self.layer_scalers[layer_idx]  # [4]
        bias = self.layer_biases[layer_idx]      # [4]

        alpha = torch.sigmoid(alpha_base * scaler[0] + bias[0])
        beta = F.softplus(beta_base * scaler[1] + bias[1])
        gate = torch.sigmoid(gate_base * scaler[2] + bias[2])
        v_cand = v_cand_base * scaler[3] + bias[3]

        return alpha, beta, gate, v_cand

    def num_parameters(self) -> int:
        """Count parameters."""
        shared = sum(p.numel() for p in [
            self.controller_h.weight, self.controller_h.bias,
            self.controller_x.weight, self.controller_x.bias,
            self.controller_v.weight, self.controller_v.bias
        ]) + sum(p.numel() for p in self.controller_mlp.parameters())

        layer_specific = self.layer_scalers.numel() + self.layer_biases.numel()

        return shared + layer_specific


class SparseHarmonicINL(nn.Module):
    """
    INL with Sparse Harmonic Excitation.

    Only applies harmonic noise to a subset of dimensions (e.g., 10%).
    Reduces compute by 10x while maintaining exploration.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        sparsity: float = 0.1,
        target_value: float = 5.0,
        dt: float = 0.1,
        excitation_amplitude: float = 0.03
    ):
        """
        Args:
            hidden_dim: Context dimension
            output_dim: State dimension
            sparsity: Fraction of dimensions to excite (0.1 = 10%)
            target_value: Initial equilibrium
            dt: Time step
            excitation_amplitude: Amplitude of excitation
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.dt = dt

        # Learnable μ
        self.mu = nn.Parameter(torch.full((output_dim,), target_value))

        # Excitation parameters (only for sparse subset)
        self.num_excited = max(1, int(output_dim * sparsity))

        # Fixed sparse indices (deterministic)
        indices = torch.linspace(0, output_dim - 1, self.num_excited).long()
        self.register_buffer('excited_indices', indices)

        # Learnable excitation params (only for excited dims)
        self.register_buffer('excitation_amplitude', torch.tensor(excitation_amplitude))
        self.excitation_gamma = nn.Parameter(torch.ones(self.num_excited))
        self.excitation_phi = nn.Parameter(torch.zeros(self.num_excited))

        # Simple controller (for demo - would use shared in practice)
        self.controller = nn.Sequential(
            nn.Linear(hidden_dim + 2 * output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * output_dim)  # α, β, g
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor,
        step: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Forward with sparse excitation."""
        batch_size = x.shape[0]

        # Compute controllers
        ctx = torch.cat([h, x, v], dim=-1)
        controller_out = self.controller(ctx)
        alpha_raw, beta_raw, gate_raw = torch.split(controller_out, self.output_dim, dim=1)

        alpha = torch.sigmoid(alpha_raw)
        beta = F.softplus(beta_raw)
        gate = torch.sigmoid(gate_raw)

        # Velocity update
        error = x - self.mu
        v_next = alpha * v - beta * error

        # Sparse harmonic excitation (only on subset of dims)
        if self.excitation_amplitude.item() > 0 and self.training:
            t = float(step)
            # Compute noise only for excited dimensions
            noise_sparse = self.excitation_amplitude * torch.sin(
                self.excitation_gamma * t + self.excitation_phi
            )  # [num_excited]

            # Apply to specific indices (sparse operation)
            v_next[:, self.excited_indices] += noise_sparse.unsqueeze(0)

        # State update
        x_next = x + self.dt * gate * v_next

        aux = {'alpha': alpha, 'beta': beta, 'gate': gate}
        return x_next, v_next, aux

    def init_state(self, batch_size: int, device: torch.device):
        """Initialize state."""
        x0 = self.mu.unsqueeze(0).expand(batch_size, -1).to(device)
        v0 = torch.zeros(batch_size, self.output_dim, device=device)
        return x0, v0


class MixtureOfIntegrators(nn.Module):
    """
    Mixture of Integrators (MoI) - like Mixture of Experts for INL.

    Routes each token to top-k integrator experts.
    Enables sparse, conditional computation.

    Benefit: Can scale capacity without scaling compute linearly
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        target_value: float = 5.0,
        dt: float = 0.1
    ):
        """
        Args:
            hidden_dim: Context dimension
            output_dim: State dimension
            num_experts: Number of INL experts
            top_k: Use top-k experts per token
            target_value: Initial equilibrium
            dt: Time step
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.dt = dt

        # Shared equilibrium (all experts share same μ)
        self.mu = nn.Parameter(torch.full((output_dim,), target_value))

        # Router: decides which expert(s) to use
        self.router = nn.Linear(hidden_dim, num_experts)

        # Expert-specific controllers
        self.expert_controllers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + 2 * output_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 3 * output_dim)  # α, β, g
            )
            for _ in range(num_experts)
        ])

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor,
        step: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward with expert routing.

        Args:
            h: Context [batch, hidden_dim]
            x: State [batch, output_dim]
            v: Velocity [batch, output_dim]
            step: Integration step

        Returns:
            x_next, v_next, aux_info
        """
        batch_size = x.shape[0]

        # Route: which experts to use?
        router_logits = self.router(h)  # [batch, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Renormalize

        # Compute outputs from selected experts
        x_next_combined = torch.zeros_like(x)
        v_next_combined = torch.zeros_like(v)

        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]  # [batch]
            weight = top_k_probs[:, k].unsqueeze(-1)  # [batch, 1]

            # Process each sample with its selected expert
            for i in range(batch_size):
                exp_id = expert_idx[i].item()

                # Get controller output from this expert
                ctx_i = torch.cat([h[i:i+1], x[i:i+1], v[i:i+1]], dim=-1)
                ctrl_out = self.expert_controllers[exp_id](ctx_i)
                alpha_raw, beta_raw, gate_raw = torch.split(ctrl_out, self.output_dim, dim=1)

                alpha = torch.sigmoid(alpha_raw)
                beta = F.softplus(beta_raw)
                gate = torch.sigmoid(gate_raw)

                # INL dynamics
                error = x[i:i+1] - self.mu
                v_next_i = alpha * v[i:i+1] - beta * error
                x_next_i = x[i:i+1] + self.dt * gate * v_next_i

                # Accumulate weighted contribution
                x_next_combined[i:i+1] += weight[i:i+1] * x_next_i
                v_next_combined[i:i+1] += weight[i:i+1] * v_next_i

        aux = {
            'router_probs': router_probs,
            'top_k_experts': top_k_indices,
            'expert_weights': top_k_probs
        }

        return x_next_combined, v_next_combined, aux

    def init_state(self, batch_size: int, device: torch.device):
        """Initialize state."""
        x0 = self.mu.unsqueeze(0).expand(batch_size, -1).to(device)
        v0 = torch.zeros(batch_size, self.output_dim, device=device)
        return x0, v0


class HierarchicalEquilibriumINL(nn.Module):
    """
    Hierarchical Equilibrium Learning.

    Instead of learning μ per dimension independently:
    - Learn global μ_global (1 parameter)
    - Learn local offsets per group (d_model // group_size parameters)

    Benefit: Fewer parameters, better generalization
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        group_size: int = 64,
        target_value: float = 5.0,
        dt: float = 0.1
    ):
        """
        Args:
            hidden_dim: Context dimension
            output_dim: State dimension
            group_size: Size of each group sharing offset
            target_value: Initial global equilibrium
            dt: Time step
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.group_size = group_size
        self.dt = dt

        # Global equilibrium (shared by all)
        self.mu_global = nn.Parameter(torch.tensor(target_value))

        # Local offsets per group
        num_groups = (output_dim + group_size - 1) // group_size
        self.mu_local_offsets = nn.Parameter(torch.zeros(num_groups))

        # Simple controller
        self.controller = nn.Sequential(
            nn.Linear(hidden_dim + 2 * output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * output_dim)
        )

    def get_mu(self) -> torch.Tensor:
        """
        Compute full μ from hierarchical representation.

        Returns:
            mu: [output_dim]
        """
        # Repeat each group offset
        mu_local = self.mu_local_offsets.repeat_interleave(self.group_size)

        # Trim to exact size
        mu_local = mu_local[:self.output_dim]

        # Combine global + local
        mu = self.mu_global + mu_local
        return mu

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor,
        step: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Forward with hierarchical equilibrium."""
        mu = self.get_mu()

        # Controller
        ctx = torch.cat([h, x, v], dim=-1)
        ctrl_out = self.controller(ctx)
        alpha_raw, beta_raw, gate_raw = torch.split(ctrl_out, self.output_dim, dim=1)

        alpha = torch.sigmoid(alpha_raw)
        beta = F.softplus(beta_raw)
        gate = torch.sigmoid(gate_raw)

        # Dynamics
        error = x - mu
        v_next = alpha * v - beta * error
        x_next = x + self.dt * gate * v_next

        aux = {'mu': mu, 'mu_global': self.mu_global, 'mu_offsets': self.mu_local_offsets}
        return x_next, v_next, aux

    def init_state(self, batch_size: int, device: torch.device):
        """Initialize state."""
        mu = self.get_mu()
        x0 = mu.unsqueeze(0).expand(batch_size, -1).to(device)
        v0 = torch.zeros(batch_size, self.output_dim, device=device)
        return x0, v0

    def num_mu_parameters(self) -> int:
        """Count parameters used for μ."""
        return 1 + self.mu_local_offsets.numel()


def compute_advanced_optimization_gains(
    d_model: int = 2048,
    num_layers: int = 24,
    hidden_controller: int = 64
):
    """
    Compute parameter savings from advanced optimizations.

    Args:
        d_model: Model dimension
        num_layers: Number of layers
        hidden_controller: Controller hidden size
    """
    print("=" * 70)
    print("ADVANCED OPTIMIZATION ANALYSIS")
    print("=" * 70)

    # 1. Shared Controllers
    print("\n1. SHARED CONTROLLERS")
    print("-" * 70)

    # Standard: each layer has own controller
    params_per_controller = (
        d_model * hidden_controller +  # h projection
        d_model * hidden_controller +  # x projection
        d_model * hidden_controller +  # v projection
        hidden_controller * (4 * d_model)  # output
    )
    standard_total = params_per_controller * num_layers

    # Shared: one controller + layer modulation
    shared_base = params_per_controller
    layer_modulation = num_layers * 8  # 4 scalers + 4 biases per layer
    shared_total = shared_base + layer_modulation

    reduction_pct = (1 - shared_total / standard_total) * 100

    print(f"  Standard (independent): {standard_total:,} params")
    print(f"  Shared + modulation:    {shared_total:,} params")
    print(f"  💾 REDUCTION: {reduction_pct:.1f}%")

    # 2. Sparse Harmonic
    print("\n2. SPARSE HARMONIC EXCITATION")
    print("-" * 70)
    sparsity = 0.1
    compute_reduction = 1 / sparsity
    print(f"  Sparsity: {sparsity*100:.0f}% of dimensions excited")
    print(f"  ⚡ COMPUTE REDUCTION: {compute_reduction:.0f}x less operations")

    # 3. Hierarchical μ
    print("\n3. HIERARCHICAL EQUILIBRIUM")
    print("-" * 70)
    group_size = 64
    num_groups = (d_model + group_size - 1) // group_size

    standard_mu = d_model
    hierarchical_mu = 1 + num_groups
    mu_reduction = (1 - hierarchical_mu / standard_mu) * 100

    print(f"  Standard μ:       {standard_mu:,} params")
    print(f"  Hierarchical μ:   {hierarchical_mu:,} params (global + {num_groups} groups)")
    print(f"  💾 REDUCTION: {mu_reduction:.1f}%")

    # 4. Combined impact
    print("\n4. COMBINED IMPACT")
    print("-" * 70)
    print(f"  Controller params saved:  {standard_total - shared_total:,}")
    print(f"  Harmonic compute:         {compute_reduction:.0f}x faster")
    print(f"  Equilibrium params saved: {standard_mu - hierarchical_mu}")
    print(f"  Overall controller reduction: {reduction_pct:.1f}%")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    print("\n")

    # Test 1: Shared Controllers
    print("=" * 70)
    print("TEST 1: Shared Controllers")
    print("=" * 70)

    shared_ctrl = SharedController(
        hidden_dim=512,
        output_dim=512,
        num_layers=12
    )

    h = torch.randn(2, 512)
    x = torch.randn(2, 512)
    v = torch.randn(2, 512)

    for layer_idx in range(3):
        alpha, beta, gate, v_cand = shared_ctrl(h, x, v, layer_idx)
        print(f"Layer {layer_idx}: alpha={alpha.mean().item():.3f}, beta={beta.mean().item():.3f}")

    print(f"✅ Shared controller parameters: {shared_ctrl.num_parameters():,}")

    # Test 2: Sparse Harmonic
    print("\n" + "=" * 70)
    print("TEST 2: Sparse Harmonic Excitation")
    print("=" * 70)

    sparse_inl = SparseHarmonicINL(
        hidden_dim=512,
        output_dim=512,
        sparsity=0.1
    )

    x0, v0 = sparse_inl.init_state(2, 'cpu')
    x_next, v_next, aux = sparse_inl(h, x0, v0, step=0)

    print(f"✅ Sparse excitation: {sparse_inl.num_excited}/{sparse_inl.output_dim} dims excited")
    print(f"   Sparsity: {sparse_inl.sparsity*100:.0f}%")

    # Test 3: Mixture of Integrators
    print("\n" + "=" * 70)
    print("TEST 3: Mixture of Integrators")
    print("=" * 70)

    moi = MixtureOfIntegrators(
        hidden_dim=512,
        output_dim=512,
        num_experts=8,
        top_k=2
    )

    x0, v0 = moi.init_state(2, 'cpu')
    x_next, v_next, aux = moi(h, x0, v0, step=0)

    print(f"✅ MoI: {moi.num_experts} experts, top-{moi.top_k} routing")
    print(f"   Expert distribution: {aux['top_k_experts']}")

    # Test 4: Hierarchical Equilibrium
    print("\n" + "=" * 70)
    print("TEST 4: Hierarchical Equilibrium")
    print("=" * 70)

    hier_inl = HierarchicalEquilibriumINL(
        hidden_dim=512,
        output_dim=512,
        group_size=64
    )

    x0, v0 = hier_inl.init_state(2, 'cpu')
    x_next, v_next, aux = hier_inl(h, x0, v0)

    print(f"✅ Hierarchical μ: {hier_inl.num_mu_parameters()} params (vs 512 standard)")
    print(f"   Global μ: {aux['mu_global'].item():.3f}")
    print(f"   Local offsets: {aux['mu_offsets'][:3].tolist()}")

    # Analysis
    print("\n")
    compute_advanced_optimization_gains(
        d_model=2048,
        num_layers=24,
        hidden_controller=64
    )
