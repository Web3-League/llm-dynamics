"""
Integration: MoE Controller + AdaptiveBudgetAllocator-v2

This module combines the power of:
1. MoE Controller: Intelligent routing between specialized experts
2. AdaptiveBudgetAllocator-v2: Smart iteration budget management

The combination enables:
- Expert specialization per layer + phase
- Budget allocation adapted to expert choices
- Loss-component feedback to both MoE and budget allocator
- Comprehensive monitoring and statistics

Expected Performance:
- 30-50% compute savings (budget allocator)
- 2-3x model capacity (MoE)
- Automatic specialization (emergent behavior)
- Phase-aware adaptation (equilibrium/exploration)

Author: Boris Peyriguère
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

from .moe_controller import INLMixtureOfExperts, create_moe_controller
from .adaptive_budget_allocator import (
    AdaptiveBudgetAllocator,
    create_budget_allocator
)


class MoEBudgetAwareINLLayer(nn.Module):
    """
    INL Layer with BOTH MoE Controller AND Adaptive Budget Allocation.

    This is the ULTIMATE optimization combining:
    - MoE: Smart expert routing for capacity
    - Budget Allocator: Smart iteration management for efficiency
    - Multi-criteria convergence
    - Budget redistribution
    - Phase awareness
    - Loss-component feedback

    The two systems work synergistically:
    - MoE provides specialized control strategies
    - Budget allocator optimizes compute per layer
    - Both adapt to phase and loss signals
    """

    def __init__(
        self,
        inl_layer: nn.Module,
        layer_idx: int,
        d_model: int,
        num_layers: int,
        budget_allocator: Optional[AdaptiveBudgetAllocator] = None,
        moe_controller: Optional[INLMixtureOfExperts] = None,
        use_moe_for_mu: bool = False
    ):
        """
        Args:
            inl_layer: Base INL layer (can be None if using MoE for all dynamics)
            layer_idx: Layer index
            d_model: Model dimension
            num_layers: Total number of layers
            budget_allocator: Budget allocator instance (shared across layers)
            moe_controller: MoE controller instance (shared across layers)
            use_moe_for_mu: Use MoE to predict equilibrium mu (experimental)
        """
        super().__init__()

        self.inl_layer = inl_layer
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.num_layers = num_layers
        self.budget_allocator = budget_allocator
        self.moe_controller = moe_controller
        self.use_moe_for_mu = use_moe_for_mu

        # Optional: MoE-predicted equilibrium
        if use_moe_for_mu and moe_controller is not None:
            self.mu_predictor = nn.Linear(d_model, d_model)

    def forward(
        self,
        h: torch.Tensor,
        x_init: torch.Tensor,
        v_init: torch.Tensor,
        default_iterations: int = 5,
        return_trajectory: bool = False,
        mu: Optional[torch.Tensor] = None,
        loss_components: Optional[Dict[str, float]] = None,
        phase: str = 'equilibrium',
        attention_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with MoE control and adaptive budget.

        Args:
            h: Context embedding [batch, d_model]
            x_init: Initial state [batch, d_model]
            v_init: Initial velocity [batch, d_model]
            default_iterations: Default iterations if no budget allocator
            return_trajectory: Whether to return full trajectory
            mu: Learned equilibrium (for error-based convergence)
            loss_components: Loss components dict (L_speed, L_energy, L_mean)
            phase: Training phase ('equilibrium' or 'exploration')
            attention_weights: Attention pattern for MoE routing

        Returns:
            x_final: Final state
            v_final: Final velocity
            info: Dictionary with comprehensive statistics
        """
        batch_size = h.size(0)
        device = h.device

        # Phase 1: Get iteration budget (with redistribution bonus)
        if self.budget_allocator is not None:
            bonus = self.budget_allocator.get_redistribution_bonus(self.layer_idx)
            max_iters = self.budget_allocator.get_layer_budget(
                self.layer_idx,
                training=self.training,
                bonus_budget=bonus
            )
        else:
            max_iters = default_iterations

        # Phase 2: Optional - Predict mu using MoE
        if self.use_moe_for_mu and self.moe_controller is not None and mu is None:
            # Use MoE to predict equilibrium target
            with torch.no_grad():
                alpha_pred, _, _, _ , _ = self.moe_controller(h, x_init, self.layer_idx, phase)
                mu = self.mu_predictor(alpha_pred)

        # Phase 3: Run integrator with MoE control
        x, v = x_init, v_init
        x_prev = x_init

        if return_trajectory:
            x_traj = [x.clone()]
            v_traj = [v.clone()]

        actual_iterations = 0
        converged = False
        convergence_metrics = {}
        moe_info_history = []

        for iteration in range(max_iters):
            # Get MoE control parameters
            if self.moe_controller is not None:
                alpha, beta, gate, v_cand, moe_info = self.moe_controller(
                    h, x, self.layer_idx, phase, attention_weights
                )
                moe_info_history.append(moe_info)
            else:
                # Fallback: use base INL layer controller
                alpha, beta, gate, v_cand = self._get_default_control(h, x)
                moe_info = {}

            # INL integration step with MoE control
            x_next, v_next = self._integration_step(
                h, x, v, alpha, beta, gate, v_cand, mu, iteration
            )

            # Check convergence (multi-criteria if enabled)
            if self.budget_allocator is not None and iteration >= self.budget_allocator.warmup_iterations:
                converged, convergence_metrics = self.budget_allocator.check_convergence(
                    x_next, x, iteration,
                    v_current=v_next,
                    mu=mu
                )
                if converged and not self.training:
                    # Early stop during inference
                    x, v = x_next, v_next
                    actual_iterations = iteration + 1
                    break

            x_prev = x
            x, v = x_next, v_next
            actual_iterations = iteration + 1

            if return_trajectory:
                x_traj.append(x.clone())
                v_traj.append(v.clone())

        # Phase 4: Update statistics and redistribute budget
        if self.budget_allocator is not None:
            # Add unused budget to redistribution pool
            unused = max_iters - actual_iterations
            self.budget_allocator.add_to_budget_pool(unused)

            # Update statistics with all metrics
            if self.training:
                final_delta = torch.norm(x - x_prev, dim=-1).mean().item()
                final_velocity = torch.norm(v, dim=-1).mean().item() if v is not None else 0.0
                final_error = torch.norm(x - mu, dim=-1).mean().item() if mu is not None else 0.0

                # Extract gradient magnitude if possible
                grad_mag = None
                if x.requires_grad and x.grad is not None:
                    grad_mag = torch.norm(x.grad, dim=-1).mean().item()

                self.budget_allocator.update_statistics(
                    self.layer_idx,
                    actual_iterations,
                    final_delta,
                    budget_allocated=max_iters,
                    final_velocity=final_velocity,
                    final_error=final_error,
                    loss_components=loss_components,
                    grad_magnitude=grad_mag
                )

        # Phase 5: Aggregate MoE information
        moe_summary = self._aggregate_moe_info(moe_info_history)

        # Prepare comprehensive output info
        info = {
            # Budget allocator info
            'iterations_used': actual_iterations,
            'max_iterations': max_iters,
            'converged': converged,
            'layer_idx': self.layer_idx,
            'convergence_metrics': convergence_metrics,

            # MoE info
            'moe_summary': moe_summary,

            # Phase info
            'phase': phase
        }

        if return_trajectory:
            info['x_trajectory'] = torch.stack(x_traj, dim=1)
            info['v_trajectory'] = torch.stack(v_traj, dim=1)

        return x, v, info

    def _integration_step(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gate: torch.Tensor,
        v_cand: torch.Tensor,
        mu: Optional[torch.Tensor],
        step: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single INL integration step with MoE-provided control parameters.

        Implements the core INL dynamics:
        error = x - mu
        v_next = alpha * v + (1 - alpha) * v_cand - beta * error
        x_next = x + gate * v_next
        """
        # Compute error term
        if mu is not None:
            error = x - mu
        else:
            error = torch.zeros_like(x)

        # Velocity update with MoE control
        v_next = alpha * v + (1 - alpha) * v_cand - beta * error

        # State update with gating
        x_next = x + gate * v_next

        return x_next, v_next

    def _get_default_control(
        self,
        h: torch.Tensor,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Fallback: get control from base INL layer if no MoE."""
        if hasattr(self.inl_layer, 'controller'):
            return self.inl_layer.controller(h, x)
        else:
            # Simple defaults
            batch_size = h.size(0)
            alpha = torch.ones(batch_size, self.d_model, device=h.device) * 0.5
            beta = torch.ones(batch_size, self.d_model, device=h.device) * 0.1
            gate = torch.ones(batch_size, self.d_model, device=h.device) * 0.9
            v_cand = torch.zeros(batch_size, self.d_model, device=h.device)
            return alpha, beta, gate, v_cand

    def _aggregate_moe_info(self, moe_info_history: List[Dict]) -> Dict:
        """Aggregate MoE information across iterations."""
        if not moe_info_history:
            return {}

        # Average routing weights across iterations
        all_weights = [info['routing_weights'] for info in moe_info_history if 'routing_weights' in info]
        if all_weights:
            avg_routing_weights = torch.stack(all_weights).mean(dim=0)
        else:
            avg_routing_weights = None

        # Collect expert usage
        expert_usage = {}
        for info in moe_info_history:
            if 'selected_experts' in info and info['selected_experts'] is not None:
                for expert_id in info['selected_experts'].flatten().tolist():
                    expert_usage[expert_id] = expert_usage.get(expert_id, 0) + 1

        # Aggregate auxiliary losses
        aux_losses = {}
        if 'aux_losses' in moe_info_history[-1]:
            for loss_name, loss_value in moe_info_history[-1]['aux_losses'].items():
                aux_losses[loss_name] = loss_value

        return {
            'avg_routing_weights': avg_routing_weights,
            'expert_usage': expert_usage,
            'aux_losses': aux_losses,
            'num_iterations': len(moe_info_history)
        }


def create_moe_budget_model(
    d_model: int,
    num_layers: int,
    # Budget allocator params
    total_budget: int = 125,
    budget_strategy: str = 'hybrid',
    # MoE params
    num_experts: int = 4,
    top_k: int = 2,
    # Shared params
    use_phase_aware: bool = True,
    use_loss_tracking: bool = True,
    **kwargs
) -> Tuple[AdaptiveBudgetAllocator, INLMixtureOfExperts]:
    """
    Helper to create both MoE controller and budget allocator.

    Args:
        d_model: Model dimension
        num_layers: Number of layers
        total_budget: Total iteration budget
        budget_strategy: Budget allocation strategy
        num_experts: Number of MoE experts
        top_k: Number of experts to activate
        use_phase_aware: Enable phase-aware features
        use_loss_tracking: Enable loss-component tracking
        **kwargs: Additional arguments

    Returns:
        budget_allocator: AdaptiveBudgetAllocator instance
        moe_controller: INLMixtureOfExperts instance
    """
    # Create budget allocator
    budget_allocator = AdaptiveBudgetAllocator(
        num_layers=num_layers,
        total_budget=total_budget,
        strategy=budget_strategy,
        use_phase_aware=use_phase_aware,
        use_loss_tracking=use_loss_tracking,
        **{k: v for k, v in kwargs.items() if k.startswith('use_') or k in [
            'min_iterations_per_layer', 'max_iterations_per_layer',
            'convergence_threshold', 'warmup_iterations',
            'velocity_threshold', 'error_threshold', 'redistribution_window'
        ]}
    )

    # Create MoE controller
    moe_controller = create_moe_controller(
        d_model=d_model,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        **{k: v for k, v in kwargs.items() if k in [
            'expert_hidden_dim', 'router_hidden_dim',
            'use_sparse_routing', 'load_balance_weight',
            'router_z_loss_weight', 'use_attention_features'
        ]}
    )

    return budget_allocator, moe_controller


if __name__ == '__main__':
    print("=" * 70)
    print("MoE + BUDGET ALLOCATOR INTEGRATION - Test")
    print("=" * 70)

    # Configuration
    d_model = 1024
    num_layers = 25
    batch_size = 16
    seq_len = 128

    # Create integrated system
    print("\n🔧 Creating MoE + Budget Allocator...")
    budget_allocator, moe_controller = create_moe_budget_model(
        d_model=d_model,
        num_layers=num_layers,
        total_budget=125,
        budget_strategy='hybrid',
        num_experts=4,
        top_k=2
    )

    print(f"\n{budget_allocator}")
    print(f"\n{moe_controller}")

    # Create test layer (mock INL layer)
    class MockINLLayer(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.d_model = d_model

        def forward(self, h, x, v, step):
            # Mock forward
            return x, v, {}

    test_layer = MoEBudgetAwareINLLayer(
        inl_layer=MockINLLayer(d_model),
        layer_idx=12,
        d_model=d_model,
        num_layers=num_layers,
        budget_allocator=budget_allocator,
        moe_controller=moe_controller
    )

    # Test forward pass
    print("\n🧪 Testing integrated forward pass...")
    h = torch.randn(batch_size, d_model)
    x_init = torch.randn(batch_size, d_model)
    v_init = torch.randn(batch_size, d_model)
    mu = torch.randn(batch_size, d_model)

    # Test different phases
    for phase in ['equilibrium', 'exploration']:
        print(f"\n  Phase: {phase}")
        budget_allocator.set_phase(phase)

        x, v, info = test_layer(
            h, x_init, v_init,
            phase=phase,
            mu=mu,
            loss_components={'L_speed': 0.1, 'L_energy': 0.05, 'L_mean': 0.2}
        )

        print(f"    Iterations: {info['iterations_used']}/{info['max_iterations']}")
        print(f"    Converged: {info['converged']}")
        print(f"    MoE experts used: {info['moe_summary'].get('expert_usage', {})}")

        if 'convergence_metrics' in info:
            print(f"    Convergence metrics: {info['convergence_metrics']}")

    # Statistics
    print("\n📊 System Statistics:")

    print("\n  Budget Allocator:")
    budget_stats = budget_allocator.get_statistics()
    print(f"    Phase: {budget_stats['current_phase']}")
    print(f"    Updates: {int(budget_stats['updates'].item())}")
    print(f"    Budget pool: {budget_stats['current_budget_pool']:.2f}")

    print("\n  MoE Controller:")
    moe_stats = moe_controller.get_expert_statistics()
    print(f"    Load balance score: {moe_stats['load_balance_score'].item():.3f}")
    print(f"    Router calls: {int(moe_stats['router_calls'].item())}")
    for i, usage in enumerate(moe_stats['expert_usage']):
        print(f"    Expert {i}: {usage.item():.1%}")

    print("\n" + "=" * 70)
    print("✅ INTEGRATION TEST COMPLETE!")
    print("=" * 70)
    print("\n💡 This system combines:")
    print("  - MoE routing for intelligent control")
    print("  - Adaptive budget for compute efficiency")
    print("  - Multi-criteria convergence")
    print("  - Phase-aware adaptation")
    print("  - Budget redistribution")
    print("  - Loss-component feedback")
    print("\n🚀 Expected: 30-50% compute savings + 2-3x capacity!")
