#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="inl-llm",
    version="3.1.0",
    author="nano3",
    author_email="",
    description="INL-LLM: Integrator Neuron Language Model with bio-inspired dynamics",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/anthropics/pacific-prime",
    project_urls={
        "HuggingFace": "https://huggingface.co/Pacific-Prime/pacific-prime",
        "GitHub": "https://github.com/anthropics/pacific-prime",
    },
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
    ],
    extras_require={
        "training": ["datasets", "tensorboard", "safetensors"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm language-model transformer integrator neural-network bio-inspired",
    license="CC BY-NC 4.0",
)
