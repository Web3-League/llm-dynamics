"""
HuggingFace-compatible wrapper for INL-LLM to enable vLLM support.

This module registers the UltraOptimizedIntegratorLanguageModel with HuggingFace's
AutoModel system, making it compatible with vLLM and other HF-based serving frameworks.

Author: Boris Peyriguère
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from .integrator_language_model import UltraOptimizedIntegratorLanguageModel, INLCache


class INLLLMConfig(PretrainedConfig):
    """
    Configuration class for INL-LLM models.

    This is required for HuggingFace AutoModel integration and vLLM compatibility.
    """

    model_type = "inl-llm"

    def __init__(
        self,
        vocab_size: int = 50261,
        d_model: int = 1728,
        num_layers: int = 25,
        num_heads: int = 32,
        num_iterations_per_layer: int = 5,
        feedforward_dim: int = 6912,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        # Optimization settings
        use_lowrank_embeddings: bool = True,
        lowrank_ratio: float = 0.125,
        use_gradient_checkpointing: bool = True,
        use_shared_controllers: bool = True,
        use_adaptive_stopping: bool = True,
        adaptive_convergence_threshold: float = 0.001,
        hierarchical_group_size: int = 64,
        excitation_sparsity: float = 0.1,
        # Token IDs
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        pad_token_id: int = 50256,
        # Integration metadata
        integrator_type: str = "ultra_optimized",
        controller_type: str = "shared",
        equilibrium_type: str = "hierarchical",
        excitation_type: str = "sparse_harmonic",
        **kwargs
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs
        )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_iterations_per_layer = num_iterations_per_layer
        self.feedforward_dim = feedforward_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        # Optimizations
        self.use_lowrank_embeddings = use_lowrank_embeddings
        self.lowrank_ratio = lowrank_ratio
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_shared_controllers = use_shared_controllers
        self.use_adaptive_stopping = use_adaptive_stopping
        self.adaptive_convergence_threshold = adaptive_convergence_threshold
        self.hierarchical_group_size = hierarchical_group_size
        self.excitation_sparsity = excitation_sparsity

        # Metadata
        self.integrator_type = integrator_type
        self.controller_type = controller_type
        self.equilibrium_type = equilibrium_type
        self.excitation_type = excitation_type


class INLLLMForCausalLM(PreTrainedModel):
    """
    HuggingFace-compatible wrapper for UltraOptimizedIntegratorLanguageModel.

    This wrapper enables:
    - vLLM support
    - HuggingFace AutoModel.from_pretrained()
    - Compatibility with HF ecosystem (pipelines, etc.)
    """

    config_class = INLLLMConfig
    base_model_prefix = "inl_llm"
    supports_gradient_checkpointing = True
    _no_split_modules = ["UltraOptimizedINLBlock"]

    def __init__(self, config: INLLLMConfig):
        super().__init__(config)

        # Create the underlying INL-LLM model
        self.model = UltraOptimizedIntegratorLanguageModel(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_iterations_per_layer=config.num_iterations_per_layer,
            feedforward_dim=config.feedforward_dim,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            use_lowrank_embeddings=config.use_lowrank_embeddings,
            lowrank_ratio=config.lowrank_ratio,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            use_shared_controllers=config.use_shared_controllers,
            use_adaptive_stopping=config.use_adaptive_stopping,
            adaptive_convergence_threshold=config.adaptive_convergence_threshold,
            hierarchical_group_size=config.hierarchical_group_size,
            excitation_sparsity=config.excitation_sparsity
        )

        # Language model head (already part of UltraOptimizedIntegratorLanguageModel)
        # No need to add another lm_head

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self):
        """Required for HuggingFace compatibility."""
        return self.model.token_embedding

    def set_input_embeddings(self, value):
        """Required for HuggingFace compatibility."""
        self.model.token_embedding = value

    def get_output_embeddings(self):
        """Required for HuggingFace compatibility."""
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Required for HuggingFace compatibility."""
        self.model.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass compatible with HuggingFace's CausalLM interface.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask (currently not used by INL-LLM)
            labels: Labels for language modeling loss
            return_dict: Whether to return a ModelOutput object

        Returns:
            CausalLMOutputWithPast or tuple of (loss, logits)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through INL-LLM
        logits = self.model(input_ids)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # INL-LLM doesn't use KV cache
            hidden_states=None,
            attentions=None
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        **kwargs
    ):
        """Prepare inputs for generation (required for .generate())."""
        return {
            "input_ids": input_ids,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """Required for beam search (INL-LLM doesn't use cache)."""
        return past

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return self.model.get_num_params()


# Register the model with HuggingFace AutoModel
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("inl-llm", INLLLMConfig)
AutoModelForCausalLM.register(INLLLMConfig, INLLLMForCausalLM)
