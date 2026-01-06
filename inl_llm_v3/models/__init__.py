"""
Complete INL-LLM model with all optimizations (Level 1 + 2).

Single production-ready model with maximum efficiency.
"""

from .integrator_language_model import (
    UltraOptimizedIntegratorLanguageModel,
    create_ultra_optimized_model
)

# HuggingFace-compatible wrappers (for vLLM support)
from .modeling_inl_llm import (
    INLLLMConfig,
    INLLLMForCausalLM
)

# Aliases for simpler API
IntegratorLanguageModel = UltraOptimizedIntegratorLanguageModel
create_model = create_ultra_optimized_model

__all__ = [
    'IntegratorLanguageModel',
    'create_model',
    # Legacy aliases
    'UltraOptimizedIntegratorLanguageModel',
    'create_ultra_optimized_model',
    # HuggingFace compatibility
    'INLLLMConfig',
    'INLLLMForCausalLM'
]
