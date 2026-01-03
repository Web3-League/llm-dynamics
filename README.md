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

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Pacific-Prime/pacific-prime",
    trust_remote_code=True,
    torch_dtype="bfloat16"
)
tokenizer = AutoTokenizer.from_pretrained("Pacific-Prime/pacific-prime")

outputs = model.generate(
    tokenizer("def fibonacci(n):", return_tensors="pt").input_ids,
    max_new_tokens=100,
    temperature=0.8
)
print(tokenizer.decode(outputs[0]))
```

## Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | ~500M |
| d_model | 1280 |
| Layers | 18 |
| Heads | 20 |
| Iterations/layer | 2 |
| Context | 2048 |

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
