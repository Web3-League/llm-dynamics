"""
Optimization modules for INL-LLM.

Level 1 (Production-ready):
- LowRankEmbedding: Reduces embedding parameters by 70-80%
- GradientCheckpointedINL: Reduces training memory by 50-70%
- AdaptiveIntegratorNeuronLayer: Speeds up inference by 30-50%

Level 2 (Research/Experimental):
- SharedController: Shares controllers across layers (-96% params)
- SparseHarmonicINL: Sparse excitation (10x less compute)
- HierarchicalEquilibriumINL: Hierarchical equilibrium learning (-98% params)
- MixtureOfIntegrators: Conditional computation (MoE-style)
"""

# Level 1 optimizations
from .optimizations import (
    LowRankEmbedding,
    AdaptiveIntegratorNeuronLayer,
    AdaptiveHierarchicalINL,
    GradientCheckpointedINL,
    compute_parameter_reduction,
    print_optimization_summary
)

# Level 2 optimizations
from .advanced_optimizations import (
    SharedController,
    SparseHarmonicINL,
    HierarchicalEquilibriumINL,
    MixtureOfIntegrators,
    compute_advanced_optimization_gains
)

__all__ = [
    # Level 1
    'LowRankEmbedding',
    'AdaptiveIntegratorNeuronLayer',
    'AdaptiveHierarchicalINL',
    'GradientCheckpointedINL',
    'compute_parameter_reduction',
    'print_optimization_summary',
    # Level 2
    'SharedController',
    'SparseHarmonicINL',
    'HierarchicalEquilibriumINL',
    'MixtureOfIntegrators',
    'compute_advanced_optimization_gains'
]
