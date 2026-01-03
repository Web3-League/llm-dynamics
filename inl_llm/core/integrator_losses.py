"""
Adaptive Loss Functions for IntegratorNeuronLayer Training

Implements:
- L_task: Main task loss (MSE or CE)
- L_mean: Soft constraint to encourage convergence towards target
- L_speed: Penalizes slow convergence in early iterations
- L_energy: Regularizes velocity to prevent wild oscillations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class IntegratorLoss(nn.Module):
    """
    Combined loss function with adaptive weighting for curriculum learning.
    """

    def __init__(
        self,
        target_value: float = 5.0,
        lambda_mean_init: float = 1.0,
        lambda_speed: float = 0.1,
        lambda_energy: float = 0.01,
        energy_p: float = 2.0,
        annealing_schedule: str = 'exponential',
        annealing_factor: float = 0.1,
        annealing_epochs: int = 100,
        variance_weighted: bool = True,
        exploration_phase: bool = False,
        exploration_lambda_mean: float = 0.05,
        exploration_lambda_energy: float = 0.001,
        task_loss_type: str = 'mse'  # 'mse' for regression, 'ce' for classification (LM)
    ):
        """
        Args:
            target_value: Target value for convergence (default 5.0)
            lambda_mean_init: Initial weight for L_mean (will be annealed)
            lambda_speed: Weight for L_speed (convergence speed penalty)
            lambda_energy: Weight for L_energy (velocity regularization)
            energy_p: Power for energy loss (2.0 = L2, 1.0 = L1)
            annealing_schedule: 'exponential' or 'linear'
            annealing_factor: Target factor for lambda_mean after annealing
            annealing_epochs: Number of epochs to anneal over
            variance_weighted: Use variance-weighted regularization
            exploration_phase: Current phase (equilibrium=False, exploration=True)
            exploration_lambda_mean: Lambda mean during exploration phase
            exploration_lambda_energy: Lambda energy during exploration phase
            task_loss_type: 'mse' for regression, 'ce' for classification/language modeling
        """
        super().__init__()

        self.target_value = target_value
        self.lambda_mean_init = lambda_mean_init
        self.lambda_speed = lambda_speed
        self.lambda_energy = lambda_energy
        self.energy_p = energy_p
        self.annealing_schedule = annealing_schedule
        self.annealing_factor = annealing_factor
        self.annealing_epochs = annealing_epochs

        # Phase control and variance weighting
        self.variance_weighted = variance_weighted
        self.exploration_phase = exploration_phase
        self.exploration_lambda_mean = exploration_lambda_mean
        self.exploration_lambda_energy = exploration_lambda_energy

        # Task loss type: MSE for regression, CrossEntropy for classification (language models)
        self.task_loss_type = task_loss_type
        if task_loss_type == 'mse':
            self.task_loss = nn.MSELoss()
        elif task_loss_type == 'ce':
            self.task_loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown task_loss_type: {task_loss_type}. Use 'mse' or 'ce'.")

    def get_lambda_mean(self, epoch: int) -> float:
        """
        Compute current lambda_mean based on annealing schedule.

        Args:
            epoch: Current training epoch

        Returns:
            Current lambda_mean value
        """
        if epoch >= self.annealing_epochs:
            return self.lambda_mean_init * self.annealing_factor

        progress = epoch / self.annealing_epochs

        if self.annealing_schedule == 'exponential':
            # Exponential decay: lambda_mean = init * (factor)^progress
            lambda_mean = self.lambda_mean_init * (self.annealing_factor ** progress)
        elif self.annealing_schedule == 'linear':
            # Linear decay: lambda_mean = init * (1 - progress * (1 - factor))
            lambda_mean = self.lambda_mean_init * (1 - progress * (1 - self.annealing_factor))
        else:
            raise ValueError(f"Unknown annealing schedule: {self.annealing_schedule}")

        return lambda_mean

    def compute_L_task(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Main task loss: MSE between final prediction and target.

        Args:
            predictions: Model predictions [batch_size, output_dim]
            targets: Ground truth targets [batch_size, output_dim]

        Returns:
            Scalar loss
        """
        return self.task_loss(predictions, targets)

    def compute_L_mean(
        self,
        x_final: torch.Tensor,
        epoch: int,
        learned_mu: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Mean constraint loss: encourages batch mean to be close to target.
        Supports variance-weighted regularization and learned mu.

        Args:
            x_final: Final state x_T [batch_size, output_dim]
            epoch: Current epoch for annealing
            learned_mu: Learned equilibrium attractor, if None uses target_value

        Returns:
            Scalar loss
        """
        # Use exploration phase lambda if in exploration mode
        if self.exploration_phase:
            lambda_mean = self.exploration_lambda_mean
        else:
            lambda_mean = self.get_lambda_mean(epoch)

        # Use learned mu if provided, otherwise fixed target
        target = learned_mu if learned_mu is not None else self.target_value

        # Variance-weighted regularization
        if self.variance_weighted:
            # Compute per-neuron variance across batch
            x_var = torch.var(x_final, dim=0, keepdim=False)  # [output_dim]
            # Weight inversely proportional to variance (stable neurons penalized less)
            weights = 1.0 / (1.0 + x_var)  # [output_dim]
            # Normalize weights
            weights = weights / weights.sum() * weights.numel()
            # Weighted penalty
            deviations = (x_final - target) ** 2  # [batch_size, output_dim]
            loss = lambda_mean * (weights * deviations.mean(dim=0)).mean()
        else:
            # Uniform weighting
            batch_mean = x_final.mean(dim=0)  # [output_dim]
            loss = lambda_mean * ((batch_mean - target) ** 2).mean()

        return loss

    def compute_L_speed(
        self,
        x_trajectory: torch.Tensor
    ) -> torch.Tensor:
        """
        Speed loss: penalizes deviation from target in early iterations.

        Uses weighted sum: w_t = exp(-t / tau) to prioritize early steps.

        Args:
            x_trajectory: Trajectory of states [batch_size, T+1, output_dim]

        Returns:
            Scalar loss
        """
        T = x_trajectory.shape[1] - 1  # Exclude initial state
        if T == 0:
            return torch.tensor(0.0, device=x_trajectory.device)

        # Exponentially decaying weights: prioritize early iterations
        tau = T / 3.0  # Decay constant
        t_indices = torch.arange(1, T + 1, device=x_trajectory.device, dtype=torch.float32)
        weights = torch.exp(-t_indices / tau)
        weights = weights / weights.sum()  # Normalize

        # Compute weighted deviation from target
        deviations = torch.abs(x_trajectory[:, 1:, :] - self.target_value)  # [B, T, output_dim]
        weighted_dev = (deviations * weights.view(1, -1, 1)).sum(dim=1)  # [B, output_dim]

        loss = self.lambda_speed * weighted_dev.mean()
        return loss

    def compute_L_energy(
        self,
        v_trajectory: torch.Tensor
    ) -> torch.Tensor:
        """
        Energy loss: regularizes velocity to prevent oscillations.

        Args:
            v_trajectory: Trajectory of velocities [batch_size, T+1, output_dim]

        Returns:
            Scalar loss
        """
        # Average absolute velocity over time
        energy = torch.abs(v_trajectory) ** self.energy_p  # [B, T+1, output_dim]
        loss = self.lambda_energy * energy.mean()
        return loss

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        trajectory: Optional[Dict[str, torch.Tensor]] = None,
        epoch: int = 0,
        learned_mu: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and components.

        Args:
            predictions: Final predictions [batch_size, output_dim]
            targets: Ground truth [batch_size, output_dim]
            trajectory: Optional trajectory dict with 'x', 'v', and 'aux'
            epoch: Current epoch for annealing
            learned_mu: Learned equilibrium attractor (v2)

        Returns:
            Dictionary with total loss and components
        """
        losses = {}

        # Main task loss
        L_task = self.compute_L_task(predictions, targets)
        losses['L_task'] = L_task

        total_loss = L_task

        # Auxiliary losses (require trajectory)
        if trajectory is not None:
            x_traj = trajectory['x']  # [B, T+1, output_dim]
            v_traj = trajectory['v']  # [B, T+1, output_dim]

            # Mean constraint loss (v2: with learned_mu and variance weighting)
            x_final = x_traj[:, -1, :]  # [B, output_dim]
            L_mean = self.compute_L_mean(x_final, epoch, learned_mu)
            losses['L_mean'] = L_mean
            total_loss = total_loss + L_mean

            # Speed loss
            L_speed = self.compute_L_speed(x_traj)
            losses['L_speed'] = L_speed
            total_loss = total_loss + L_speed

            # Energy loss (reduced during exploration phase)
            lambda_energy = self.exploration_lambda_energy if self.exploration_phase else self.lambda_energy
            # Temporarily override for this computation
            original_lambda_energy = self.lambda_energy
            self.lambda_energy = lambda_energy
            L_energy = self.compute_L_energy(v_traj)
            self.lambda_energy = original_lambda_energy  # Restore
            losses['L_energy'] = L_energy
            total_loss = total_loss + L_energy

        losses['total'] = total_loss

        # Report current phase lambda values
        if self.exploration_phase:
            losses['lambda_mean'] = torch.tensor(self.exploration_lambda_mean)
        else:
            losses['lambda_mean'] = torch.tensor(self.get_lambda_mean(epoch))

        return losses

    def set_exploration_phase(self, is_exploration: bool):
        """
        Set the current training phase.

        Args:
            is_exploration: True for exploration phase, False for equilibrium phase
        """
        self.exploration_phase = is_exploration


def compute_convergence_metrics(
    x_trajectory: torch.Tensor,
    target_value: float = 5.0,
    epsilon: float = 0.1
) -> Dict[str, float]:
    """
    Compute metrics about convergence behavior.

    Args:
        x_trajectory: Trajectory [batch_size, T+1, output_dim]
        target_value: Target value for convergence
        epsilon: Tolerance for "converged" check

    Returns:
        Dictionary with metrics:
            - time_to_converge: Average time steps to reach epsilon-ball
            - final_rmse: RMSE at final time step
            - final_mean: Mean value at final time step
            - final_std: Std dev at final time step
            - fraction_converged: Fraction of samples within epsilon at end
    """
    batch_size, T_plus_1, output_dim = x_trajectory.shape
    T = T_plus_1 - 1

    # Final time step statistics
    x_final = x_trajectory[:, -1, :]  # [B, output_dim]
    final_rmse = torch.sqrt(((x_final - target_value) ** 2).mean()).item()
    final_mean = x_final.mean().item()
    final_std = x_final.std().item()

    # Fraction converged at final step
    is_converged = torch.abs(x_final - target_value) <= epsilon
    fraction_converged = is_converged.float().mean().item()

    # Time to converge (first time within epsilon-ball)
    # [B, T, output_dim]
    deviations = torch.abs(x_trajectory[:, 1:, :] - target_value)  # Skip initial state
    within_epsilon = deviations <= epsilon  # [B, T, output_dim]

    # For each sample, find first time it's converged (across all output dims)
    within_epsilon_all = within_epsilon.all(dim=-1)  # [B, T]

    # Find first True index for each batch element
    time_to_converge_list = []
    for b in range(batch_size):
        converged_times = torch.where(within_epsilon_all[b])[0]
        if len(converged_times) > 0:
            time_to_converge_list.append(converged_times[0].item() + 1)  # +1 because we skipped initial
        else:
            time_to_converge_list.append(T)  # Never converged

    avg_time_to_converge = sum(time_to_converge_list) / len(time_to_converge_list)

    return {
        'time_to_converge': avg_time_to_converge,
        'final_rmse': final_rmse,
        'final_mean': final_mean,
        'final_std': final_std,
        'fraction_converged': fraction_converged
    }
