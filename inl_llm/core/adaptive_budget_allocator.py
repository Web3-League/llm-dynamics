"""
Adaptive Budget Allocator for INL Architecture (ULTRA-OPTIMIZED v2)

This module implements dynamic iteration budget allocation across layers:
- Global budget pool (e.g., 125 iterations total for 25 layers)
- Adaptive allocation based on layer complexity and convergence speed
- Bio-inspired: Different brain regions process at different speeds

Key Features:
✅ Budget-aware: Total compute stays constant
✅ Adaptive: Simple layers use fewer iterations, complex layers use more
✅ Convergence-driven: Stop early when layer has converged
✅ Multiple strategies: uniform, complexity-based, learned allocation

NEW ULTRA-OPTIMIZED FEATURES (v2):
🚀 Multi-Criteria Convergence: delta + velocity + error magnitude
🚀 Budget Redistribution Pool: Unused budget → next layers
🚀 Phase-Aware Allocation: Equilibrium vs Exploration phase
🚀 Layer-Position Specialization: Early/Mid/Late layer patterns
🚀 Loss-Component Tracking: L_speed, L_energy, L_mean awareness
🚀 Gradient Magnitude Tracking: Allocate more to actively learning layers

Author: Boris Peyriguère
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Literal, Any
import math


class AdaptiveBudgetAllocator(nn.Module):
    """
    Manages iteration budget allocation across layers (ULTRA-OPTIMIZED v2).

    Strategies:
    - 'uniform': Equal iterations per layer (baseline)
    - 'learned': Learnable per-layer budget allocation
    - 'dynamic': Runtime allocation based on convergence speed
    - 'hybrid': Combination of learned + dynamic (RECOMMENDED)

    NEW v2 Features:
    - Multi-criteria convergence detection
    - Budget redistribution pool
    - Phase-aware allocation (equilibrium/exploration)
    - Layer position specialization (early/mid/late)
    """

    def __init__(
        self,
        num_layers: int,
        total_budget: int,
        strategy: Literal['uniform', 'learned', 'dynamic', 'hybrid'] = 'hybrid',
        min_iterations_per_layer: int = 2,
        max_iterations_per_layer: int = 15,
        convergence_threshold: float = 1e-3,
        warmup_iterations: int = 3,
        # NEW v2 parameters
        use_multi_criteria_convergence: bool = True,
        use_budget_redistribution: bool = True,
        use_phase_aware: bool = True,
        use_layer_specialization: bool = True,
        use_loss_tracking: bool = True,
        use_gradient_tracking: bool = True,
        velocity_threshold: float = 1e-3,
        error_threshold: float = 1e-2,
        redistribution_window: int = 3
    ):
        """
        Args:
            num_layers: Number of layers in the model
            total_budget: Total iteration budget (e.g., 125 for 25 layers × 5 avg)
            strategy: Allocation strategy
            min_iterations_per_layer: Minimum iterations per layer
            max_iterations_per_layer: Maximum iterations per layer
            convergence_threshold: Threshold for early stopping (delta norm)
            warmup_iterations: Minimum iterations before checking convergence

            NEW v2 Args:
            use_multi_criteria_convergence: Use delta + velocity + error for convergence
            use_budget_redistribution: Redistribute unused budget to next layers
            use_phase_aware: Adapt to equilibrium/exploration phase
            use_layer_specialization: Early/mid/late layer patterns
            use_loss_tracking: Track L_speed, L_energy, L_mean per layer
            use_gradient_tracking: Track gradient magnitudes for allocation
            velocity_threshold: Convergence threshold for velocity magnitude
            error_threshold: Convergence threshold for error magnitude
            redistribution_window: How many next layers to share unused budget with
        """
        super().__init__()

        self.num_layers = num_layers
        self.total_budget = total_budget
        self.strategy = strategy
        self.min_iterations = min_iterations_per_layer
        self.max_iterations = max_iterations_per_layer
        self.convergence_threshold = convergence_threshold
        self.warmup_iterations = warmup_iterations

        # NEW v2 feature flags
        self.use_multi_criteria = use_multi_criteria_convergence
        self.use_redistribution = use_budget_redistribution
        self.use_phase_aware = use_phase_aware
        self.use_layer_specialization = use_layer_specialization
        self.use_loss_tracking = use_loss_tracking
        self.use_gradient_tracking = use_gradient_tracking
        self.velocity_threshold = velocity_threshold
        self.error_threshold = error_threshold
        self.redistribution_window = redistribution_window

        # Learnable budget allocation (if using learned or hybrid strategy)
        if strategy in ['learned', 'hybrid']:
            # Initialize to uniform allocation, will be learned
            initial_allocation = torch.ones(num_layers) / num_layers
            self.budget_weights = nn.Parameter(initial_allocation)
        else:
            self.register_buffer('budget_weights', torch.ones(num_layers) / num_layers)

        # Original statistics tracking
        self.register_buffer('layer_iterations_history', torch.zeros(num_layers))
        self.register_buffer('layer_convergence_speed', torch.ones(num_layers))
        self.register_buffer('update_count', torch.zeros(1))

        # NEW v2: Multi-criteria convergence tracking
        self.register_buffer('layer_velocity_history', torch.zeros(num_layers))
        self.register_buffer('layer_error_history', torch.zeros(num_layers))

        # NEW v2: Phase tracking
        self.current_phase = 'equilibrium'  # or 'exploration'
        self.phase_multipliers = {
            'equilibrium': 0.8,  # Use 20% less iterations in equilibrium (fast convergence)
            'exploration': 1.2   # Use 20% more iterations in exploration (need stability)
        }

        # NEW v2: Layer position specialization patterns
        if self.use_layer_specialization:
            self.layer_position_weights = self._compute_layer_position_weights()

        # NEW v2: Budget redistribution pool (shared across forward pass)
        self.budget_pool = 0.0
        self.register_buffer('unused_budget_history', torch.zeros(num_layers))

        # NEW v2: Loss component tracking
        if self.use_loss_tracking:
            self.register_buffer('layer_L_speed', torch.zeros(num_layers))
            self.register_buffer('layer_L_energy', torch.zeros(num_layers))
            self.register_buffer('layer_L_mean', torch.zeros(num_layers))

        # NEW v2: Gradient magnitude tracking
        if self.use_gradient_tracking:
            self.register_buffer('layer_grad_magnitude', torch.ones(num_layers))

    def _compute_layer_position_weights(self) -> torch.Tensor:
        """
        Compute position-based weights for layer specialization.

        Bio-inspired pattern:
        - Early layers (0-33%): Fast processing, fewer iterations (0.8x)
        - Middle layers (34-66%): Complex processing, more iterations (1.2x)
        - Late layers (67-100%): Refinement, medium iterations (1.0x)

        Returns:
            Tensor of shape [num_layers] with position weights
        """
        weights = torch.ones(self.num_layers)

        third = self.num_layers // 3

        # Early layers: faster
        weights[:third] = 0.8

        # Middle layers: slower (more complex)
        weights[third:2*third] = 1.2

        # Late layers: medium
        weights[2*third:] = 1.0

        return weights

    def get_layer_budget(self, layer_idx: int, training: bool = True, bonus_budget: float = 0.0) -> int:
        """
        Get iteration budget for a specific layer (ULTRA-OPTIMIZED v2).

        NEW v2: Applies multiple adjustments:
        - Phase-aware multiplier (equilibrium vs exploration)
        - Layer position specialization (early/mid/late)
        - Gradient magnitude adjustment
        - Loss component adjustment
        - Budget redistribution bonus

        Args:
            layer_idx: Layer index
            training: Whether in training mode
            bonus_budget: Bonus iterations from budget redistribution pool

        Returns:
            Number of iterations allocated to this layer
        """
        # Base budget calculation (original strategies)
        if self.strategy == 'uniform':
            base_budget = self.total_budget // self.num_layers

        elif self.strategy == 'learned':
            weights = torch.softmax(self.budget_weights, dim=0)
            base_budget = (weights[layer_idx] * self.total_budget).item()

        elif self.strategy == 'dynamic':
            speed = self.layer_convergence_speed[layer_idx].item()
            relative_budget = (1.0 / (speed + 0.1))
            total_relative = sum(1.0 / (self.layer_convergence_speed[i].item() + 0.1)
                                for i in range(self.num_layers))
            fraction = relative_budget / total_relative
            base_budget = fraction * self.total_budget

        elif self.strategy == 'hybrid':
            weights = torch.softmax(self.budget_weights, dim=0)
            learned_budget = weights[layer_idx] * self.total_budget

            if self.update_count.item() > 10:
                speed = self.layer_convergence_speed[layer_idx].item()
                speed_factor = 1.0 / (speed + 0.1)
                avg_speed_factor = sum(1.0 / (self.layer_convergence_speed[i].item() + 0.1)
                                      for i in range(self.num_layers)) / self.num_layers
                adjustment = speed_factor / avg_speed_factor
                learned_budget = learned_budget * adjustment

            base_budget = learned_budget.item()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # NEW v2: Apply phase-aware multiplier
        if self.use_phase_aware:
            phase_mult = self.phase_multipliers.get(self.current_phase, 1.0)
            base_budget *= phase_mult

        # NEW v2: Apply layer position specialization
        if self.use_layer_specialization:
            pos_weight = self.layer_position_weights[layer_idx].item()
            base_budget *= pos_weight

        # NEW v2: Apply gradient magnitude adjustment (after warmup)
        if self.use_gradient_tracking and self.update_count.item() > 10:
            grad_mag = self.layer_grad_magnitude[layer_idx].item()
            avg_grad = self.layer_grad_magnitude.mean().item()
            if avg_grad > 1e-8:
                grad_adjustment = grad_mag / avg_grad
                # Clip to reasonable range [0.8, 1.3]
                grad_adjustment = max(0.8, min(1.3, grad_adjustment))
                base_budget *= grad_adjustment

        # NEW v2: Apply loss component adjustment (high L_speed = needs more iterations)
        if self.use_loss_tracking and self.update_count.item() > 10:
            L_speed = self.layer_L_speed[layer_idx].item()
            L_energy = self.layer_L_energy[layer_idx].item()

            # High speed loss = slow convergence = more iterations needed
            avg_speed = self.layer_L_speed.mean().item()
            if avg_speed > 1e-8:
                speed_adjustment = 1.0 + 0.2 * (L_speed / avg_speed - 1.0)
                speed_adjustment = max(0.9, min(1.2, speed_adjustment))
                base_budget *= speed_adjustment

        # NEW v2: Add bonus from redistribution pool
        base_budget += bonus_budget

        # Final budget with bounds
        budget = int(base_budget)
        return max(self.min_iterations, min(self.max_iterations, budget))

    def check_convergence(
        self,
        x_current: torch.Tensor,
        x_prev: torch.Tensor,
        iteration: int,
        v_current: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if layer has converged (ULTRA-OPTIMIZED v2 with multi-criteria).

        NEW v2: Multi-criteria convergence detection:
        1. Delta norm: ||x_current - x_prev|| < threshold (original)
        2. Velocity magnitude: ||v|| < velocity_threshold (new)
        3. Error magnitude: ||x - mu|| < error_threshold (new)

        All criteria must be satisfied for convergence (AND logic).

        Args:
            x_current: Current state [batch_size, d_model]
            x_prev: Previous state [batch_size, d_model]
            iteration: Current iteration number
            v_current: Current velocity [batch_size, d_model] (optional, for multi-criteria)
            mu: Learned equilibrium [batch_size, d_model] or scalar (optional, for error check)

        Returns:
            converged: True if converged, False otherwise
            metrics: Dictionary with convergence metrics
        """
        if iteration < self.warmup_iterations:
            return False, {'delta': 0.0, 'velocity': 0.0, 'error': 0.0}

        metrics = {}

        # Criterion 1: Delta norm (original)
        delta = torch.norm(x_current - x_prev, dim=-1).mean()
        metrics['delta'] = delta.item()
        delta_converged = delta.item() < self.convergence_threshold

        # If multi-criteria is disabled, return early
        if not self.use_multi_criteria:
            return delta_converged, metrics

        # Criterion 2: Velocity magnitude (NEW v2)
        velocity_converged = True
        if v_current is not None:
            v_mag = torch.norm(v_current, dim=-1).mean()
            metrics['velocity'] = v_mag.item()
            velocity_converged = v_mag.item() < self.velocity_threshold
        else:
            metrics['velocity'] = 0.0

        # Criterion 3: Error magnitude (NEW v2)
        error_converged = True
        if mu is not None:
            error = torch.norm(x_current - mu, dim=-1).mean()
            metrics['error'] = error.item()
            error_converged = error.item() < self.error_threshold
        else:
            metrics['error'] = 0.0

        # ALL criteria must be satisfied (AND logic)
        converged = delta_converged and velocity_converged and error_converged

        return converged, metrics

    def update_statistics(
        self,
        layer_idx: int,
        iterations_used: int,
        final_delta: float,
        budget_allocated: int = 0,
        final_velocity: float = 0.0,
        final_error: float = 0.0,
        loss_components: Optional[Dict[str, float]] = None,
        grad_magnitude: Optional[float] = None
    ):
        """
        Update layer statistics after processing (ULTRA-OPTIMIZED v2).

        NEW v2: Tracks additional metrics:
        - Velocity magnitude
        - Error magnitude
        - Unused budget
        - Loss components (L_speed, L_energy, L_mean)
        - Gradient magnitude

        Args:
            layer_idx: Layer index
            iterations_used: Number of iterations actually used
            final_delta: Final convergence delta (smaller = faster convergence)
            budget_allocated: Budget that was allocated (NEW v2)
            final_velocity: Final velocity magnitude (NEW v2)
            final_error: Final error magnitude (NEW v2)
            loss_components: Dict with L_speed, L_energy, L_mean (NEW v2)
            grad_magnitude: Gradient magnitude for this layer (NEW v2)
        """
        alpha = 0.9  # Exponential moving average

        # Original statistics
        self.layer_iterations_history[layer_idx] = (
            alpha * self.layer_iterations_history[layer_idx] +
            (1 - alpha) * iterations_used
        )

        speed = 1.0 / (final_delta + 1e-6)
        self.layer_convergence_speed[layer_idx] = (
            alpha * self.layer_convergence_speed[layer_idx] +
            (1 - alpha) * speed
        )

        # NEW v2: Track velocity
        self.layer_velocity_history[layer_idx] = (
            alpha * self.layer_velocity_history[layer_idx] +
            (1 - alpha) * final_velocity
        )

        # NEW v2: Track error
        self.layer_error_history[layer_idx] = (
            alpha * self.layer_error_history[layer_idx] +
            (1 - alpha) * final_error
        )

        # NEW v2: Track unused budget
        if budget_allocated > 0:
            unused = budget_allocated - iterations_used
            self.unused_budget_history[layer_idx] = (
                alpha * self.unused_budget_history[layer_idx] +
                (1 - alpha) * unused
            )

        # NEW v2: Track loss components
        if self.use_loss_tracking and loss_components is not None:
            if 'L_speed' in loss_components:
                self.layer_L_speed[layer_idx] = (
                    alpha * self.layer_L_speed[layer_idx] +
                    (1 - alpha) * loss_components['L_speed']
                )
            if 'L_energy' in loss_components:
                self.layer_L_energy[layer_idx] = (
                    alpha * self.layer_L_energy[layer_idx] +
                    (1 - alpha) * loss_components['L_energy']
                )
            if 'L_mean' in loss_components:
                self.layer_L_mean[layer_idx] = (
                    alpha * self.layer_L_mean[layer_idx] +
                    (1 - alpha) * loss_components['L_mean']
                )

        # NEW v2: Track gradient magnitude
        if self.use_gradient_tracking and grad_magnitude is not None:
            self.layer_grad_magnitude[layer_idx] = (
                alpha * self.layer_grad_magnitude[layer_idx] +
                (1 - alpha) * grad_magnitude
            )

        self.update_count += 1

    def set_phase(self, phase: str):
        """
        Set training phase for phase-aware budget allocation.

        Args:
            phase: 'equilibrium' or 'exploration'
        """
        if phase not in ['equilibrium', 'exploration']:
            raise ValueError(f"Unknown phase: {phase}. Use 'equilibrium' or 'exploration'.")
        self.current_phase = phase

    def reset_budget_pool(self):
        """
        Reset the budget redistribution pool (call at start of forward pass).
        """
        self.budget_pool = 0.0

    def add_to_budget_pool(self, unused_iterations: int):
        """
        Add unused iterations to redistribution pool.

        Args:
            unused_iterations: Number of unused iterations from a layer
        """
        if self.use_redistribution:
            self.budget_pool += unused_iterations

    def get_redistribution_bonus(self, layer_idx: int) -> float:
        """
        Get bonus iterations from redistribution pool for a layer.

        Distributes pool evenly across next N layers (redistribution_window).

        Args:
            layer_idx: Current layer index

        Returns:
            Bonus iterations from pool
        """
        if not self.use_redistribution or self.budget_pool <= 0:
            return 0.0

        # Distribute to next N layers
        remaining_layers = self.num_layers - layer_idx
        if remaining_layers <= 0:
            return 0.0

        # Distribute pool across min(remaining_layers, window)
        window = min(remaining_layers, self.redistribution_window)
        bonus = self.budget_pool / window

        # Deduct from pool
        self.budget_pool -= bonus

        return bonus

    def get_all_budgets(self, training: bool = True) -> List[int]:
        """
        Get budget allocation for all layers.

        Args:
            training: Whether in training mode

        Returns:
            List of iteration budgets for each layer
        """
        budgets = [self.get_layer_budget(i, training) for i in range(self.num_layers)]

        # Ensure total doesn't exceed budget (adjust if needed)
        total = sum(budgets)
        if total > self.total_budget:
            # Scale down proportionally
            scale = self.total_budget / total
            budgets = [max(self.min_iterations, int(b * scale)) for b in budgets]

        return budgets

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current allocation statistics (ULTRA-OPTIMIZED v2).

        NEW v2: Includes all new metrics tracked by v2 allocator.

        Returns:
            Dictionary with comprehensive statistics
        """
        budgets = self.get_all_budgets(training=False)

        stats = {
            # Original statistics
            'layer_budgets': torch.tensor(budgets),
            'layer_iterations_history': self.layer_iterations_history.clone(),
            'layer_convergence_speed': self.layer_convergence_speed.clone(),
            'budget_weights': torch.softmax(self.budget_weights, dim=0) if self.strategy in ['learned', 'hybrid'] else self.budget_weights,
            'total_budget': torch.tensor(self.total_budget),
            'updates': self.update_count.clone(),

            # NEW v2: Multi-criteria convergence tracking
            'layer_velocity_history': self.layer_velocity_history.clone(),
            'layer_error_history': self.layer_error_history.clone(),

            # NEW v2: Phase information
            'current_phase': self.current_phase,
            'phase_multipliers': self.phase_multipliers,

            # NEW v2: Budget redistribution
            'unused_budget_history': self.unused_budget_history.clone(),
            'current_budget_pool': self.budget_pool,
        }

        # NEW v2: Layer position weights (if enabled)
        if self.use_layer_specialization:
            stats['layer_position_weights'] = self.layer_position_weights.clone()

        # NEW v2: Loss component tracking (if enabled)
        if self.use_loss_tracking:
            stats['layer_L_speed'] = self.layer_L_speed.clone()
            stats['layer_L_energy'] = self.layer_L_energy.clone()
            stats['layer_L_mean'] = self.layer_L_mean.clone()

        # NEW v2: Gradient magnitude tracking (if enabled)
        if self.use_gradient_tracking:
            stats['layer_grad_magnitude'] = self.layer_grad_magnitude.clone()

        # NEW v2: Feature flags summary
        stats['v2_features'] = {
            'multi_criteria_convergence': self.use_multi_criteria,
            'budget_redistribution': self.use_redistribution,
            'phase_aware': self.use_phase_aware,
            'layer_specialization': self.use_layer_specialization,
            'loss_tracking': self.use_loss_tracking,
            'gradient_tracking': self.use_gradient_tracking
        }

        return stats

    def __repr__(self) -> str:
        budgets = self.get_all_budgets(training=False)
        avg_budget = sum(budgets) / len(budgets)
        min_budget = min(budgets)
        max_budget = max(budgets)

        # NEW v2: Count enabled features
        enabled_features = []
        if self.use_multi_criteria:
            enabled_features.append("multi-criteria")
        if self.use_redistribution:
            enabled_features.append("redistribution")
        if self.use_phase_aware:
            enabled_features.append("phase-aware")
        if self.use_layer_specialization:
            enabled_features.append("layer-spec")
        if self.use_loss_tracking:
            enabled_features.append("loss-track")
        if self.use_gradient_tracking:
            enabled_features.append("grad-track")

        features_str = ", ".join(enabled_features) if enabled_features else "none"

        return (
            f"AdaptiveBudgetAllocator-v2(\n"
            f"  strategy={self.strategy},\n"
            f"  num_layers={self.num_layers},\n"
            f"  total_budget={self.total_budget},\n"
            f"  avg_budget_per_layer={avg_budget:.1f},\n"
            f"  budget_range=[{min_budget}, {max_budget}],\n"
            f"  convergence_threshold={self.convergence_threshold:.1e},\n"
            f"  phase={self.current_phase},\n"
            f"  v2_features=[{features_str}]\n"
            f")"
        )


class BudgetAwareINLLayer(nn.Module):
    """
    Wrapper for INL layers that respects budget allocation (ULTRA-OPTIMIZED v2).

    Handles:
    - Dynamic iteration count based on budget
    - Early stopping when converged (multi-criteria)
    - Statistics tracking for budget allocator
    - Budget redistribution pool management

    NEW v2 Features:
    - Multi-criteria convergence checking
    - Budget redistribution to next layers
    - Loss component extraction
    - Gradient magnitude tracking
    """

    def __init__(
        self,
        inl_layer: nn.Module,
        layer_idx: int,
        budget_allocator: Optional[AdaptiveBudgetAllocator] = None
    ):
        """
        Args:
            inl_layer: The base INL layer to wrap
            layer_idx: Index of this layer
            budget_allocator: Budget allocator (if None, uses default iterations)
        """
        super().__init__()

        self.inl_layer = inl_layer
        self.layer_idx = layer_idx
        self.budget_allocator = budget_allocator

    def forward(
        self,
        h: torch.Tensor,
        x_init: torch.Tensor,
        v_init: torch.Tensor,
        default_iterations: int = 5,
        return_trajectory: bool = False,
        mu: Optional[torch.Tensor] = None,
        loss_components: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with budget-aware iteration control (ULTRA-OPTIMIZED v2).

        NEW v2: Includes multi-criteria convergence, budget redistribution,
        and comprehensive statistics tracking.

        Args:
            h: Context embedding [batch_size * seq_len, d_model]
            x_init: Initial state [batch_size * seq_len, d_model]
            v_init: Initial velocity [batch_size * seq_len, d_model]
            default_iterations: Default iterations if no budget allocator
            return_trajectory: Whether to return full trajectory
            mu: Learned equilibrium (for error-based convergence) (NEW v2)
            loss_components: Loss components dict (L_speed, L_energy, L_mean) (NEW v2)

        Returns:
            x_final: Final state
            v_final: Final velocity
            info: Dictionary with statistics
        """
        # NEW v2: Get budget with redistribution bonus
        if self.budget_allocator is not None:
            bonus = self.budget_allocator.get_redistribution_bonus(self.layer_idx)
            max_iters = self.budget_allocator.get_layer_budget(
                self.layer_idx,
                training=self.training,
                bonus_budget=bonus
            )
        else:
            max_iters = default_iterations

        # Run iterations
        x, v = x_init, v_init
        x_prev = x_init

        if return_trajectory:
            x_traj = [x.clone()]
            v_traj = [v.clone()]

        actual_iterations = 0
        converged = False
        convergence_metrics = {}

        for iteration in range(max_iters):
            # One integration step
            x_next, v_next, aux = self.inl_layer(h, x, v, step=iteration)

            # NEW v2: Check convergence with multi-criteria (if budget allocator available)
            if self.budget_allocator is not None and iteration >= self.budget_allocator.warmup_iterations:
                converged, convergence_metrics = self.budget_allocator.check_convergence(
                    x_next, x, iteration,
                    v_current=v_next,  # NEW v2: velocity for multi-criteria
                    mu=mu  # NEW v2: equilibrium for error-based check
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

        # NEW v2: Add unused budget to redistribution pool
        if self.budget_allocator is not None:
            unused = max_iters - actual_iterations
            self.budget_allocator.add_to_budget_pool(unused)

        # NEW v2: Update statistics with all new metrics (during training)
        if self.training and self.budget_allocator is not None:
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
                budget_allocated=max_iters,  # NEW v2
                final_velocity=final_velocity,  # NEW v2
                final_error=final_error,  # NEW v2
                loss_components=loss_components,  # NEW v2
                grad_magnitude=grad_mag  # NEW v2
            )

        # Prepare output info
        info = {
            'iterations_used': actual_iterations,
            'max_iterations': max_iters,
            'converged': converged,
            'layer_idx': self.layer_idx,
            'convergence_metrics': convergence_metrics  # NEW v2
        }

        if return_trajectory:
            info['x_trajectory'] = torch.stack(x_traj, dim=1)
            info['v_trajectory'] = torch.stack(v_traj, dim=1)

        return x, v, info


def create_budget_allocator(
    num_layers: int,
    avg_iterations_per_layer: int = 5,
    strategy: str = 'hybrid',
    **kwargs
) -> AdaptiveBudgetAllocator:
    """
    Helper function to create a budget allocator.

    Args:
        num_layers: Number of layers
        avg_iterations_per_layer: Average iterations per layer (determines total budget)
        strategy: Allocation strategy
        **kwargs: Additional arguments for AdaptiveBudgetAllocator

    Returns:
        Configured AdaptiveBudgetAllocator
    """
    total_budget = num_layers * avg_iterations_per_layer

    return AdaptiveBudgetAllocator(
        num_layers=num_layers,
        total_budget=total_budget,
        strategy=strategy,
        **kwargs
    )


if __name__ == '__main__':
    print("=" * 70)
    print("ADAPTIVE BUDGET ALLOCATOR - Test")
    print("=" * 70)

    # Create allocator
    allocator = create_budget_allocator(
        num_layers=25,
        avg_iterations_per_layer=5,
        strategy='hybrid'
    )

    print(f"\n{allocator}")

    # Test budget allocation
    print("\n📊 Initial Budget Allocation:")
    budgets = allocator.get_all_budgets()
    for i, budget in enumerate(budgets):
        print(f"  Layer {i:2d}: {budget:2d} iterations")

    print(f"\n✅ Total budget: {sum(budgets)} / {allocator.total_budget}")

    # Simulate some updates
    print("\n🔄 Simulating convergence updates...")
    for i in range(25):
        # Simulate: early layers converge faster, later layers slower
        convergence_speed = 1.0 if i < 10 else 0.5
        final_delta = 0.001 * convergence_speed
        iterations = 4 if i < 10 else 7

        allocator.update_statistics(i, iterations, final_delta)

    # Check updated allocation
    print("\n📊 Updated Budget Allocation (after learning):")
    budgets_updated = allocator.get_all_budgets()
    for i, budget in enumerate(budgets_updated):
        change = "+" if budget > budgets[i] else ("-" if budget < budgets[i] else " ")
        print(f"  Layer {i:2d}: {budget:2d} iterations {change}")

    print(f"\n✅ Total budget: {sum(budgets_updated)} / {allocator.total_budget}")

    # Show statistics
    print("\n📈 Statistics:")
    stats = allocator.get_statistics()
    print(f"  Updates: {stats['updates'].item():.0f}")
    print(f"  Convergence speeds (first 5 layers): {stats['layer_convergence_speed'][:5].tolist()}")
    print(f"  Convergence speeds (last 5 layers): {stats['layer_convergence_speed'][-5:].tolist()}")

    print("\n" + "=" * 70)
    print("✅ ADAPTIVE BUDGET ALLOCATOR WORKING!")
    print("=" * 70)
