"""
INL-LLM Inference Script

Supports both v2 (checkpoint 230K) and v3 architectures.
"""
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

# v2 architecture (for checkpoint 230K)
from inl_llm import IntegratorLanguageModel as V2Model

# v3 architecture (for future checkpoints)
from inl_llm_v3 import UltraOptimizedIntegratorLanguageModel as V3Model


def load_model(weights_path="pacific-prime_500M_500K/model.safetensors", device="cuda", version="v2"):
    """Load the trained model.

    Args:
        weights_path: Path to safetensors file
        device: 'cuda' or 'cpu'
        version: 'v2' for 500M teacher, 'v3' for 3.8B distilled, 'v3_3.8b' for full 3.8B config
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model weights...")
    state_dict = load_file(weights_path)

    print(f"Creating model (architecture: {version})...")

    if version == "v2":
        # v2 architecture - 500M teacher model
        model = V2Model(
            vocab_size=50261,
            d_model=1280,
            num_layers=18,
            num_heads=20,
            num_iterations_per_layer=2,
            feedforward_dim=5120,
            max_seq_len=1024
        )
    elif version == "v3_3.8b":
        # v3 architecture - 3.8B distilled model (full config)
        model = V3Model(
            vocab_size=50261,
            d_model=3072,
            num_layers=32,
            num_heads=24,
            num_kv_heads=6,
            num_iterations_per_layer=2,
            feedforward_dim=12288,
            max_seq_len=1024,
            use_gradient_checkpointing=False
        )
    else:
        # v3 architecture - default (smaller config for testing)
        model = V3Model(
            vocab_size=50261,
            d_model=1280,
            num_layers=18,
            num_heads=20,
            num_iterations_per_layer=2,
            feedforward_dim=5120,
            max_seq_len=1024
        )

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()

    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        print(f"Model loaded on CUDA")
    else:
        device = "cpu"
        print(f"Model loaded on CPU")

    return model, tokenizer, device

def generate(model, tokenizer, prompt, device="cuda", max_new_tokens=200,
             temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.2):
    """Generate text from a prompt.

    Args:
        model: The language model
        tokenizer: Tokenizer
        prompt: Input text prompt
        device: 'cuda' or 'cpu'
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more focused)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling (limits vocabulary)
        repetition_penalty: Penalty for repeating tokens (>1.0 discourages repetition)
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    if device == "cuda":
        input_ids = input_ids.cuda()

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="INL-LLM Inference")
    parser.add_argument("--weights", "-w", type=str, default="pacific-prime_500M_500K/model.safetensors",
                        help="Path to model weights (.safetensors)")
    parser.add_argument("--version", "-v", type=str, default="v2",
                        choices=["v2", "v3", "v3_3.8b"],
                        help="Model architecture version")
    args = parser.parse_args()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, device = load_model(
        weights_path=args.weights,
        device=device,
        version=args.version
    )

    print("\n" + "="*60)
    print("INL-LLM Inference")
    print("="*60)
    print("Type 'quit' to exit\n")

    while True:
        prompt = input("Prompt> ")
        if prompt.lower() == 'quit':
            break

        if not prompt.strip():
            continue

        print("\nGenerating...")
        output = generate(model, tokenizer, prompt, device=device)
        print(f"\n{output}\n")
        print("-"*60)

if __name__ == "__main__":
    main()
