---
language: en
license: cc-by-nc-4.0
tags:
- text-generation
- integrator-neuron
- custom-architecture
pipeline_tag: text-generation
---

# INL-LLM - Integrator Neuron Language Model

**Bio-inspired architecture** using iterative integrator dynamics instead of static FFN layers.

*Created by nano3*

**GitHub**: [Web3-League/llm-dynamics](https://github.com/Web3-League/llm-dynamics)

## Quick Start

```bash
pip install inl-llm
```

```python
import torch
from inl_llm import UltraOptimizedIntegratorLanguageModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Pacific-Prime/pacific-prime")

# Load model weights
weights_path = hf_hub_download("Pacific-Prime/pacific-prime", "model.safetensors")
state_dict = load_file(weights_path)

# Create model
model = UltraOptimizedIntegratorLanguageModel(
    vocab_size=50261,
    d_model=1280,
    num_layers=18,
    num_heads=20,
    num_iterations_per_layer=2,
    feedforward_dim=5120,
    max_seq_len=1024
)
model.load_state_dict(state_dict)
model.eval()

# Generate
input_ids = tokenizer("def fibonacci(n):", return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_new_tokens=100, temperature=0.8)
print(tokenizer.decode(outputs[0]))
```

## Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | ~500M |
| Training steps | 250K |
| d_model | 1280 |
| Layers | 18 |
| Heads | 20 |
| Iterations/layer | 2 |
| Context | 1024 |

**Key difference from standard transformers:**

```python
# Standard: static one-shot FFN
x = x + FFN(x)

# INL: iterative dynamics (2 iterations/layer)
for t in range(2):
    error = x - mu
    v = alpha * v - beta * error
    x = x + dt * gate * v
```

## Optimizations

- **Shared controllers**: 96% fewer controller params
- **Low-rank embeddings**: 87% fewer embedding params
- **Adaptive stopping**: Early exit when converged
- **Pure CrossEntropy**: No equilibrium regularization (optimized for LM)

## Training

```bash
python simple_training.py --streaming --dataset codeparrot --use-amp
```

## Citation

```bibtex
@misc{inl-llm-2025,
  author = {nano3},
  title = {INL-LLM: Integrator Neural Language Model},
  year = {2025}
}
```

**License**: CC BY-NC 4.0
