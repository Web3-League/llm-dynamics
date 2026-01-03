#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="inl-llm",
    version="2.0.0",
    author="Boris Peyriguère",
    author_email="boris@pacific-prime.ai",
    description="Integrator Neural Language Model - Novel LLM architecture",
    long_description="""# INL-LLM

Integrator Neural Language Model - A novel language model architecture based on integrator dynamics.

## Install

```bash
pip install inl-llm
```

## Features

- Low-rank embeddings (-87% params)
- Shared controllers (-96% params)
- Hierarchical equilibrium (-98% params)
- Adaptive early stopping (+50% speed)
""",
    long_description_content_type="text/markdown",
    url="https://huggingface.co/Pacific-Prime/pacific-prime",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="llm language-model transformer integrator neural-network",
)
