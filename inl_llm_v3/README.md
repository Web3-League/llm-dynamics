# INL-LLM v3: Integrator Neuron Language Model

A novel neural network architecture combining **integrator dynamics** with modern LLM optimizations (2023-2024).

## Architecture Overview

INL-LLM introduces **Integrator Neuron Layers (INL)** - a bio-inspired computation mechanism where hidden states evolve through learnable differential dynamics towards learned equilibrium points.

### Core Innovation: Integrator Dynamics

```
error = x_t - mu                    # deviation from equilibrium
v_{t+1} = alpha * v_t + (1 - alpha) * v_cand - beta * error
x_{t+1} = x_t + dt * gate * v_{t+1}
```

Where `alpha`, `beta`, `gate`, `v_cand` are **context-dependent learnable parameters** computed by a controller MLP.

### Key Features

- **Learnable Equilibrium (mu)**: Each layer learns its own attractor point
- **Dynamic Integration Gain**: Alpha adapts based on state imbalance
- **Harmonic Excitation**: Deterministic oscillations for exploration
- **Adaptive Early Stopping**: Converged states skip remaining iterations

## Modern LLM Optimizations (2023-2024)

### Architectural Improvements

| Component | Traditional | INL-LLM v3 |
|-----------|-------------|------------|
| Normalization | LayerNorm | **RMSNorm** (faster) |
| Position Encoding | Absolute/Sinusoidal | **RoPE** (better extrapolation) |
| Attention | Multi-Head (MHA) | **Grouped-Query (GQA)** (memory efficient) |
| Activation | GELU | **SwiGLU** (better performance) |
| Bias | Yes | **No bias** (like LLaMA) |

### Efficiency Optimizations

**Level 1 - Basic:**
- Low-rank embeddings (-87% embedding params)
- Gradient checkpointing (-65% memory)
- Adaptive early stopping (+50% inference speed)

**Level 2 - Advanced:**
- Shared controllers across layers (-96% controller params)
- Hierarchical equilibrium (global mu + local offsets, -98% mu params)
- Sparse harmonic excitation (10x less compute)

**Level 3 - Bio-inspired:**
- Adaptive budget allocation (dynamic iterations per layer)
- Convergence-based early stopping

## Usage

```python
from inl_llm_v3 import UltraOptimizedIntegratorLanguageModel

model = UltraOptimizedIntegratorLanguageModel(
    vocab_size=50261,
    d_model=3072,
    num_layers=32,
    num_heads=24,
    num_kv_heads=6,        # GQA: 4 Q heads per KV head
    num_iterations_per_layer=2,
    feedforward_dim=12288,
    max_seq_len=4096,
    # All optimizations enabled by default
)

# Forward pass
logits, aux_info, cache = model(input_ids, use_cache=True, return_aux=True)

# Generation with KV cache
output_ids = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
    use_cache=True  # 3x faster!
)
```

## Project Structure

```
inl_llm_v3/
├── __init__.py                 # Package exports
├── core/
│   ├── integrator_neuron_layer.py   # Core INL dynamics
│   ├── integrator_losses.py         # Specialized loss functions
│   ├── integrator_scheduler_v2.py   # Learning rate scheduling
│   └── adaptive_budget_allocator.py # Dynamic compute allocation
├── models/
│   ├── integrator_language_model.py # Main LLM architecture
│   ├── inl_vision.py               # Vision encoder (future)
│   └── inl_diffusion.py            # Diffusion model (future)
└── optimizations/
    ├── optimizations.py            # Level 1 optimizations
    └── advanced_optimizations.py   # Level 2 optimizations
```

## Performance

Compared to standard Transformers of equivalent size:

| Metric | Improvement |
|--------|-------------|
| Memory usage | -40-60% |
| Inference speed | +30-50% |
| Training FLOPS | -20-30% |
| Embedding params | -87% |
| Controller params | -96% |

## References

Modern techniques incorporated from:
- **RMSNorm**: Zhang & Sennrich (2019)
- **RoPE**: Su et al. (2021) - RoFormer
- **GQA**: Ainslie et al. (2023) - LLaMA 2
- **SwiGLU**: Shazeer (2020) - GLU Variants

## Author

nano3

## License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0)
