"""
Universal Code Datasets for INL-LLM Training

Supports all major coding datasets:
- TheStack (BigCode) - 300+ languages
- StarCoder Data - Cleaned code data
- CodeParrot - Python focused
- The-Stack-Dedup - Deduplicated version
- CodeSearchNet - Code + docstrings
- MBPP / HumanEval - Code benchmarks
- Local files (Python, JS, etc.)

Author: Boris Peyriguere
"""

import os
import torch
from torch.utils.data import Dataset, IterableDataset
from typing import Optional, List, Dict, Any, Iterator
from pathlib import Path
import random

# Optional imports
try:
    from datasets import load_dataset, interleave_datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False


# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

CODE_DATASET_CONFIGS = {
    # TheStack V2 - Latest version (recommended)
    "the-stack-v2": {
        "path": "bigcode/the-stack-v2",
        "config": "Python",
        "text_column": "content",
        "streaming": True,
        "description": "The Stack V2 - 67TB of code"
    },

    # TheStack - Massive multilingual code dataset
    "the-stack": {
        "path": "bigcode/the-stack",
        "text_column": "content",
        "streaming": True,
        "languages": ["python", "javascript", "java", "c", "cpp", "go", "rust", "typescript"],
        "description": "3TB+ of permissively licensed code"
    },

    # TheStack Dedup - Deduplicated version
    "the-stack-dedup": {
        "path": "bigcode/the-stack-dedup",
        "text_column": "content",
        "streaming": True,
        "languages": ["python", "javascript", "java", "c", "cpp", "go", "rust"],
        "description": "Deduplicated version of TheStack"
    },

    # StarCoderData - High quality code data
    "starcoderdata": {
        "path": "bigcode/starcoderdata",
        "text_column": "content",
        "streaming": True,
        "description": "Curated code data for StarCoder"
    },

    # CodeParrot - Python focused
    "codeparrot": {
        "path": "codeparrot/codeparrot-clean",
        "text_column": "content",
        "streaming": True,
        "description": "Cleaned Python code from GitHub"
    },

    # The Stack Smol - Small subset (no approval needed, works immediately)
    "stack-smol": {
        "path": "bigcode/the-stack-smol",
        "config": "data/python",
        "text_column": "content",
        "streaming": True,
        "description": "Small subset of The Stack - Python"
    },

    # CodeSearchNet - Code with docstrings
    "codesearchnet": {
        "path": "code_search_net",
        "text_column": "func_code_string",
        "streaming": False,
        "languages": ["python", "javascript", "java", "go", "ruby", "php"],
        "description": "Functions with documentation"
    },

    # MBPP - Python programming problems
    "mbpp": {
        "path": "mbpp",
        "text_column": "code",
        "streaming": False,
        "description": "Python programming problems"
    },

    # RedPajama Code subset
    "redpajama-code": {
        "path": "togethercomputer/RedPajama-Data-1T",
        "config": "github",
        "text_column": "text",
        "streaming": True,
        "description": "GitHub code from RedPajama"
    },

    # Pile - Code subset (GitHub)
    "pile-github": {
        "path": "monology/pile-uncopyrighted",
        "text_column": "text",
        "streaming": True,
        "description": "GitHub code from The Pile"
    }
}


# ============================================================================
# STREAMING CODE DATASET
# ============================================================================

class StreamingCodeDataset(IterableDataset):
    """
    Streaming dataset for large-scale code training.

    Supports:
    - Multiple code datasets (TheStack, StarCoder, etc.)
    - Language filtering
    - On-the-fly tokenization
    - Automatic sequence packing for efficiency
    """

    def __init__(
        self,
        dataset_name: str = "the-stack-dedup",
        languages: Optional[List[str]] = None,
        seq_len: int = 2048,
        tokenizer=None,
        split: str = "train",
        max_samples: Optional[int] = None,
        pack_sequences: bool = True,
        add_code_tokens: bool = True
    ):
        """
        Args:
            dataset_name: Name of dataset (see CODE_DATASET_CONFIGS)
            languages: List of programming languages to include
            seq_len: Sequence length for training
            tokenizer: Tokenizer to use
            split: Dataset split
            max_samples: Maximum samples (for testing)
            pack_sequences: Pack multiple short samples into one sequence
            add_code_tokens: Add special <CODE> tokens
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required. Install: pip install datasets")

        if tokenizer is None:
            raise ValueError("Tokenizer is required")

        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.pack_sequences = pack_sequences
        self.add_code_tokens = add_code_tokens

        # Get dataset config
        if dataset_name not in CODE_DATASET_CONFIGS:
            available = list(CODE_DATASET_CONFIGS.keys())
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

        config = CODE_DATASET_CONFIGS[dataset_name]
        self.text_column = config["text_column"]

        # Filter languages if specified
        if languages is None and "languages" in config:
            languages = config["languages"][:3]  # Default: first 3 languages

        print(f"Loading {dataset_name}...")

        # Load dataset(s)
        if languages and "languages" in config:
            # Load multiple language subsets
            datasets_list = []
            for lang in languages:
                try:
                    ds = load_dataset(
                        config["path"],
                        data_dir=f"data/{lang}",
                        split=split,
                        streaming=config.get("streaming", True),
                        trust_remote_code=True
                    )
                    datasets_list.append(ds)
                    print(f"  Loaded {lang}")
                except Exception as e:
                    print(f"  Warning: Could not load {lang}: {e}")

            if datasets_list:
                # Interleave datasets
                self.dataset = interleave_datasets(datasets_list)
            else:
                raise ValueError(f"No languages could be loaded for {dataset_name}")
        else:
            # Single dataset
            self.dataset = load_dataset(
                config["path"],
                config.get("config"),
                split=split,
                streaming=config.get("streaming", True),
                trust_remote_code=True
            )

        print(f"Dataset {dataset_name} ready!")

        # Buffer for sequence packing
        self._token_buffer = []

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize code with optional special tokens."""
        if self.add_code_tokens and hasattr(self.tokenizer, 'encode'):
            # Add <CODE> markers if available
            text = f"<CODE>\n{text}\n</CODE>"

        tokens = self.tokenizer.encode(
            text,
            max_length=self.seq_len * 2,  # Allow longer for packing
            truncation=True,
            add_special_tokens=True
        )
        return tokens

    def _pack_tokens(self, tokens: List[int]) -> Optional[torch.Tensor]:
        """Pack tokens into fixed-length sequences."""
        self._token_buffer.extend(tokens)

        # Add separator between samples
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id:
            self._token_buffer.append(self.tokenizer.eos_token_id)

        # Return a sequence if we have enough tokens
        if len(self._token_buffer) >= self.seq_len + 1:
            seq = self._token_buffer[:self.seq_len + 1]
            self._token_buffer = self._token_buffer[self.seq_len + 1:]
            return torch.tensor(seq, dtype=torch.long)

        return None

    def __iter__(self) -> Iterator[tuple]:
        """Iterate over tokenized sequences."""
        sample_count = 0

        for example in self.dataset:
            # Check max samples
            if self.max_samples and sample_count >= self.max_samples:
                break

            # Get text content
            text = example.get(self.text_column, "")
            if not text or len(text.strip()) < 10:
                continue

            # Tokenize
            tokens = self._tokenize(text)

            if len(tokens) < 10:
                continue

            if self.pack_sequences:
                # Pack sequences
                seq = self._pack_tokens(tokens)
                if seq is not None:
                    sample_count += 1
                    yield seq[:-1], seq[1:]
            else:
                # Direct output (pad/truncate)
                if len(tokens) < self.seq_len + 1:
                    pad_len = self.seq_len + 1 - len(tokens)
                    pad_id = self.tokenizer.pad_token_id or 0
                    tokens = tokens + [pad_id] * pad_len
                else:
                    tokens = tokens[:self.seq_len + 1]

                seq = torch.tensor(tokens, dtype=torch.long)
                sample_count += 1
                yield seq[:-1], seq[1:]

        # Flush remaining buffer
        if self.pack_sequences and len(self._token_buffer) > 10:
            pad_len = self.seq_len + 1 - len(self._token_buffer)
            if pad_len > 0:
                pad_id = self.tokenizer.pad_token_id or 0
                self._token_buffer.extend([pad_id] * pad_len)
            seq = torch.tensor(self._token_buffer[:self.seq_len + 1], dtype=torch.long)
            yield seq[:-1], seq[1:]


# ============================================================================
# LOCAL CODE DATASET
# ============================================================================

class LocalCodeDataset(Dataset):
    """
    Dataset for local code files.

    Supports:
    - Python (.py)
    - JavaScript (.js, .ts)
    - Java (.java)
    - C/C++ (.c, .cpp, .h)
    - Go (.go)
    - Rust (.rs)
    - And more...
    """

    SUPPORTED_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx',
        '.java', '.c', '.cpp', '.h', '.hpp',
        '.go', '.rs', '.rb', '.php', '.cs',
        '.swift', '.kt', '.scala', '.r',
        '.sql', '.sh', '.bash', '.ps1',
        '.html', '.css', '.scss', '.json', '.yaml', '.yml',
        '.md', '.rst', '.txt'
    }

    def __init__(
        self,
        root_dir: str,
        seq_len: int = 2048,
        tokenizer=None,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
        max_file_size: int = 1_000_000,  # 1MB max
        min_file_size: int = 100  # 100 bytes min
    ):
        """
        Args:
            root_dir: Root directory containing code files
            seq_len: Sequence length
            tokenizer: Tokenizer to use
            extensions: File extensions to include (default: all supported)
            recursive: Search subdirectories
            max_file_size: Maximum file size in bytes
            min_file_size: Minimum file size in bytes
        """
        if tokenizer is None:
            raise ValueError("Tokenizer is required")

        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.extensions = set(extensions) if extensions else self.SUPPORTED_EXTENSIONS

        # Find all code files
        self.files = []
        root = Path(root_dir)

        if not root.exists():
            raise ValueError(f"Directory not found: {root_dir}")

        pattern = "**/*" if recursive else "*"
        for filepath in root.glob(pattern):
            if filepath.is_file() and filepath.suffix.lower() in self.extensions:
                size = filepath.stat().st_size
                if min_file_size <= size <= max_file_size:
                    self.files.append(filepath)

        print(f"Found {len(self.files)} code files in {root_dir}")

        # Pre-tokenize all files
        self.samples = []
        self._tokenize_files()

    def _tokenize_files(self):
        """Tokenize all code files."""
        print("Tokenizing files...")

        for filepath in self.files:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                if len(content.strip()) < 10:
                    continue

                # Add file path as context
                header = f"# File: {filepath.name}\n"
                text = header + content

                # Tokenize
                tokens = self.tokenizer.encode(
                    text,
                    max_length=self.seq_len + 1,
                    truncation=True
                )

                if len(tokens) > 20:
                    # Pad if needed
                    if len(tokens) < self.seq_len + 1:
                        pad_id = self.tokenizer.pad_token_id or 0
                        tokens = tokens + [pad_id] * (self.seq_len + 1 - len(tokens))

                    self.samples.append(torch.tensor(tokens[:self.seq_len + 1]))

            except Exception as e:
                print(f"  Warning: Could not process {filepath}: {e}")

        print(f"Tokenized {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        seq = self.samples[idx]
        return seq[:-1], seq[1:]


# ============================================================================
# MIXED CODE DATASET (Combines multiple sources)
# ============================================================================

class MixedCodeDataset(IterableDataset):
    """
    Combines multiple code datasets with configurable mixing ratios.

    Example:
        dataset = MixedCodeDataset(
            datasets=["the-stack-dedup", "codeparrot"],
            weights=[0.7, 0.3],
            tokenizer=tokenizer
        )
    """

    def __init__(
        self,
        datasets: List[str],
        weights: Optional[List[float]] = None,
        tokenizer=None,
        seq_len: int = 2048,
        languages: Optional[List[str]] = None
    ):
        """
        Args:
            datasets: List of dataset names
            weights: Sampling weights (default: uniform)
            tokenizer: Tokenizer to use
            seq_len: Sequence length
            languages: Languages to include
        """
        if tokenizer is None:
            raise ValueError("Tokenizer is required")

        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Set uniform weights if not specified
        if weights is None:
            weights = [1.0 / len(datasets)] * len(datasets)

        self.weights = weights

        # Load all datasets
        print(f"Loading {len(datasets)} datasets...")
        self.dataset_iters = []

        for name in datasets:
            ds = StreamingCodeDataset(
                dataset_name=name,
                languages=languages,
                seq_len=seq_len,
                tokenizer=tokenizer,
                pack_sequences=True
            )
            self.dataset_iters.append(iter(ds))

        print("All datasets loaded!")

    def __iter__(self) -> Iterator[tuple]:
        """Iterate with weighted sampling."""
        while True:
            # Sample a dataset based on weights
            idx = random.choices(range(len(self.dataset_iters)), weights=self.weights)[0]

            try:
                sample = next(self.dataset_iters[idx])
                yield sample
            except StopIteration:
                # Dataset exhausted, remove it
                self.dataset_iters.pop(idx)
                self.weights.pop(idx)

                if not self.dataset_iters:
                    break

                # Renormalize weights
                total = sum(self.weights)
                self.weights = [w / total for w in self.weights]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_code_dataset(
    name: str = "the-stack-dedup",
    tokenizer=None,
    seq_len: int = 2048,
    languages: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    **kwargs
) -> IterableDataset:
    """
    Create a code dataset by name.

    Args:
        name: Dataset name or "local:/path/to/code"
        tokenizer: Tokenizer to use
        seq_len: Sequence length
        languages: Languages to include
        max_samples: Max samples (for testing)

    Returns:
        Dataset ready for training
    """
    if name.startswith("local:"):
        # Local directory
        path = name[6:]
        return LocalCodeDataset(
            root_dir=path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            **kwargs
        )

    return StreamingCodeDataset(
        dataset_name=name,
        tokenizer=tokenizer,
        seq_len=seq_len,
        languages=languages,
        max_samples=max_samples,
        **kwargs
    )


def list_available_datasets() -> Dict[str, str]:
    """List all available code datasets."""
    return {name: cfg["description"] for name, cfg in CODE_DATASET_CONFIGS.items()}


def get_recommended_config(model_size: str = "small") -> Dict[str, Any]:
    """
    Get recommended training configuration for code models.

    Args:
        model_size: "small", "medium", "large"

    Returns:
        Dictionary with recommended settings
    """
    configs = {
        "small": {
            "dataset": "codeparrot",
            "languages": ["python"],
            "seq_len": 1024,
            "batch_size": 8,
            "learning_rate": 5e-4,
            "description": "Fast training on Python code"
        },
        "medium": {
            "dataset": "the-stack-dedup",
            "languages": ["python", "javascript", "java"],
            "seq_len": 2048,
            "batch_size": 4,
            "learning_rate": 3e-4,
            "description": "Balanced multilingual training"
        },
        "large": {
            "dataset": "the-stack-dedup",
            "languages": ["python", "javascript", "java", "c", "cpp", "go", "rust"],
            "seq_len": 4096,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "description": "Full multilingual code training"
        }
    }

    return configs.get(model_size, configs["medium"])


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CODE DATASETS - Test")
    print("=" * 70)

    # List available datasets
    print("\nAvailable datasets:")
    for name, desc in list_available_datasets().items():
        print(f"  {name}: {desc}")

    # Test with local tokenizer
    if TOKENIZER_AVAILABLE:
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Test recommended config
        config = get_recommended_config("small")
        print(f"\nRecommended config for 'small': {config}")

        print("\n" + "=" * 70)
        print("CODE DATASETS READY!")
        print("=" * 70)
    else:
        print("\nInstall transformers to test: pip install transformers")
