"""
Production-Ready Pre-Training Script for INL-LLM

Features:
✅ Multi-GPU training (DDP/FSDP) - Scale to 8+ GPUs
✅ Streaming datasets - Train on billions of tokens (C4, Pile, RedPajama)
✅ CODE DATASETS - TheStack, StarCoder, CodeParrot, etc.
✅ Robust checkpointing - Auto-resume after crashes
✅ Weights & Biases logging - Monitor training in real-time
✅ Gradient accumulation - Larger effective batch sizes
✅ Mixed precision (bf16/fp16) - 2x faster training

Usage:
  # Single GPU with code dataset:
  python simple_training.py --streaming --dataset codeparrot

  # Multi-GPU with TheStack:
  torchrun --nproc_per_node=8 simple_training.py --distributed --streaming --dataset the-stack-dedup

  # With specific languages:
  python simple_training.py --streaming --dataset the-stack-dedup --languages python,javascript

  # Local code directory:
  python simple_training.py --local-code ./my_code_folder

  # Resume from checkpoint:
  python simple_training.py --resume checkpoints/checkpoint-1000.pt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import argparse
import time
from pathlib import Path

# Import from the correct path
from inl_llm_v3.models.integrator_language_model import UltraOptimizedIntegratorLanguageModel
from inl_llm_v3.core.integrator_losses import IntegratorLoss
from inl_llm_v3.core.integrator_scheduler_v2 import create_cycle_scheduler

# Distributed training imports
try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

# Streaming datasets
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Weights & Biases logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import tokenizer
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("⚠️ transformers not installed. Install with: pip install transformers")

# Import code datasets
try:
    from code_datasets import (
        StreamingCodeDataset,
        LocalCodeDataset,
        create_code_dataset,
        list_available_datasets,
        get_recommended_config
    )
    CODE_DATASETS_AVAILABLE = True
except ImportError:
    CODE_DATASETS_AVAILABLE = False
    print("⚠️ code_datasets not found. Using basic datasets only.")


# ============================================================================
# DISTRIBUTED TRAINING UTILITIES
# ============================================================================

def setup_distributed():
    """Initialize distributed training."""
    if not DISTRIBUTED_AVAILABLE:
        return None, 0, 1

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("⚠️ Not running in distributed mode (missing RANK/WORLD_SIZE env vars)")
        return None, 0, 1

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    return local_rank, rank, world_size


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if this is the main process (rank 0)."""
    return rank == 0


# ============================================================================
# CHECKPOINTING UTILITIES
# ============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss, save_dir, rank=0, keep_last_n=5):
    """Save training checkpoint."""
    if not is_main_process(rank):
        return

    os.makedirs(save_dir, exist_ok=True)

    # Unwrap DDP model if needed
    model_to_save = model.module if hasattr(model, 'module') else model

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'timestamp': time.time()
    }

    checkpoint_path = os.path.join(save_dir, f'checkpoint-step-{global_step}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")

    # Cleanup old checkpoints (keep only last N)
    cleanup_old_checkpoints(save_dir, keep_last_n)


def cleanup_old_checkpoints(save_dir, keep_last_n=5):
    """Remove old checkpoints, keeping only the last N."""
    checkpoint_files = sorted(Path(save_dir).glob('checkpoint-step-*.pt'))

    if len(checkpoint_files) > keep_last_n:
        for old_checkpoint in checkpoint_files[:-keep_last_n]:
            old_checkpoint.unlink()
            print(f"🗑️  Removed old checkpoint: {old_checkpoint.name}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load checkpoint and resume training."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model state
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    loss = checkpoint.get('loss', 0.0)

    print(f"✅ Resumed from epoch {epoch}, step {global_step}, loss {loss:.4f}")
    return epoch, global_step


# ============================================================================
# STREAMING DATASET
# ============================================================================

class StreamingTextDataset:
    """
    Streaming dataset for large-scale pre-training.
    Supports C4, Pile, RedPajama, etc. from HuggingFace datasets.
    """

    def __init__(self, dataset_name='c4', dataset_config='en', split='train',
                 seq_len=2048, tokenizer=None, streaming=True):
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library not installed. Install with: pip install datasets")

        if tokenizer is None:
            raise ValueError("Tokenizer is required")

        self.seq_len = seq_len
        self.tokenizer = tokenizer

        print(f"📂 Loading streaming dataset: {dataset_name} ({dataset_config})")

        # Load dataset in streaming mode
        self.dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=streaming,
            trust_remote_code=True
        )

        # Apply tokenization on-the-fly
        self.dataset = self.dataset.map(
            self._tokenize_function,
            batched=False,
            remove_columns=self.dataset.column_names if not streaming else None
        )

        print(f"✅ Streaming dataset ready: {dataset_name}")

    def _tokenize_function(self, examples):
        """Tokenize text on-the-fly."""
        # Handle both single examples and batches
        text = examples.get('text', '')

        if not text or len(text.strip()) == 0:
            # Return dummy tokens for empty text
            return {
                'input_ids': [self.tokenizer.pad_token_id] * (self.seq_len + 1)
            }

        tokens = self.tokenizer.encode(
            text,
            max_length=self.seq_len + 1,
            truncation=True,
            padding='max_length'
        )

        return {'input_ids': tokens}

    def __iter__(self):
        """Iterate over dataset."""
        for example in self.dataset:
            if 'input_ids' in example and len(example['input_ids']) > 1:
                tokens = torch.tensor(example['input_ids'])
                yield tokens[:-1], tokens[1:]


# ============================================================================
# ORIGINAL PARQUET DATASET (for local testing)
# ============================================================================

class ParquetTextDataset(Dataset):
    """Dataset loader for parquet files with text data."""

    def __init__(self, parquet_path, seq_len=128, tokenizer=None):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        if tokenizer is None:
            raise ValueError("Tokenizer is required for ParquetTextDataset")

        print(f"  📂 Loading dataset from {parquet_path}...")
        df = pd.read_parquet(parquet_path)

        if 'text' not in df.columns:
            raise ValueError("Parquet file must contain a 'text' column")

        print(f"  📊 Dataset contains {len(df)} text samples")
        print(f"  🔤 Tokenizing text samples...")

        self.samples = []
        for idx, text in enumerate(df['text']):
            if pd.notna(text) and len(str(text).strip()) > 0:
                # Tokenize the text
                tokens = self.tokenizer.encode(
                    str(text),
                    max_length=self.seq_len + 1,
                    truncation=True
                )

                # Only keep samples with enough tokens
                if len(tokens) > 10:  # Minimum length
                    # Pad if needed
                    if len(tokens) < self.seq_len + 1:
                        tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_len + 1 - len(tokens))

                    self.samples.append(torch.tensor(tokens[:self.seq_len + 1]))

            if (idx + 1) % 100 == 0:
                print(f"    Processed {idx + 1}/{len(df)} samples...")

        print(f"  ✅ Dataset ready: {len(self.samples)} valid samples")
        self.num_samples = len(self.samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.samples[idx]
        return seq[:-1], seq[1:]


def train_streaming_epoch(model, dataset, loss_fn, optimizer, scheduler, device='cpu', epoch=0,
                          batch_size=2, gradient_accumulation_steps=1, rank=0, global_step=0,
                          log_interval=10, checkpoint_interval=500, checkpoint_dir='checkpoints',
                          use_amp=False, max_steps_per_epoch=10000):
    """
    Train for one epoch using streaming dataset.

    This function handles streaming datasets that don't have a fixed length.
    """
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    start_time = time.time()
    tokens_processed = 0

    # Collect batches from streaming dataset
    batch_inputs = []
    batch_targets = []

    for step, (inputs, targets) in enumerate(dataset):
        if step >= max_steps_per_epoch * batch_size:
            break

        batch_inputs.append(inputs)
        batch_targets.append(targets)

        # Process batch when full
        if len(batch_inputs) >= batch_size:
            inputs_batch = torch.stack(batch_inputs).to(device)
            targets_batch = torch.stack(batch_targets).to(device)
            batch_inputs = []
            batch_targets = []

            tokens_processed += inputs_batch.numel()

            # Forward pass
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16 if use_amp else torch.float32):
                logits, trajectory, _ = model(inputs_batch, return_aux=True, use_cache=False)

                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets_batch.view(-1)

                if loss_fn is not None and trajectory is not None:
                    last_layer_traj = trajectory[-1] if isinstance(trajectory, list) and len(trajectory) > 0 else trajectory
                    loss_components = loss_fn(
                        predictions=logits_flat,
                        targets=targets_flat,
                        trajectory=last_layer_traj,
                        epoch=epoch
                    )
                    loss = loss_components['total']
                else:
                    loss = nn.CrossEntropyLoss()(logits_flat, targets_flat)
                    loss_components = {'total': loss, 'L_task': loss}

                loss = loss / gradient_accumulation_steps

            # Backward
            if use_amp and scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            num_batches += 1

            # Update weights
            if num_batches % gradient_accumulation_steps == 0:
                if use_amp and scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                global_step += 1

                # Logging
                if is_main_process(rank) and global_step % log_interval == 0:
                    elapsed_time = time.time() - start_time
                    tokens_per_sec = tokens_processed / elapsed_time if elapsed_time > 0 else 0

                    L_task = loss_components.get('L_task', torch.tensor(0.0)).item() if isinstance(loss_components, dict) else loss.item()

                    print(f'  Step {global_step}, Loss: {loss.item() * gradient_accumulation_steps:.4f} '
                          f'[Task: {L_task:.4f}] '
                          f'| {tokens_per_sec:.0f} tokens/s | LR: {optimizer.param_groups[0]["lr"]:.2e}')

                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log({
                            'train/loss': loss.item() * gradient_accumulation_steps,
                            'train/loss_task': L_task,
                            'train/learning_rate': optimizer.param_groups[0]['lr'],
                            'train/tokens_per_sec': tokens_per_sec,
                            'train/epoch': epoch,
                            'global_step': global_step
                        })

                # Checkpointing
                if is_main_process(rank) and global_step % checkpoint_interval == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, global_step,
                        loss.item() * gradient_accumulation_steps, checkpoint_dir, rank
                    )

            total_loss += loss.item() * gradient_accumulation_steps

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


def train_epoch(model, dataloader, loss_fn, optimizer, scheduler, device='cpu', epoch=0,
                gradient_accumulation_steps=1, rank=0, global_step=0, log_interval=10,
                checkpoint_interval=500, checkpoint_dir='checkpoints', use_amp=False):
    """
    Train for one epoch with gradient accumulation and distributed support.

    Args:
        gradient_accumulation_steps: Accumulate gradients over N steps before updating
        rank: Process rank for distributed training
        global_step: Global training step counter
        use_amp: Use automatic mixed precision (bf16/fp16)
    """
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()

    # AMP scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    start_time = time.time()
    tokens_processed = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        tokens_processed += inputs.numel()

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16 if use_amp else torch.float32):
            logits, trajectory, _ = model(inputs, return_aux=True, use_cache=False)

            # Reshape for loss computation
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)

            # Compute loss
            if loss_fn is not None and trajectory is not None:
                last_layer_traj = trajectory[-1] if isinstance(trajectory, list) and len(trajectory) > 0 else trajectory
                loss_components = loss_fn(
                    predictions=logits_flat,
                    targets=targets_flat,
                    trajectory=last_layer_traj,
                    epoch=epoch
                )
                loss = loss_components['total']
            else:
                loss = nn.CrossEntropyLoss()(logits_flat, targets_flat)
                loss_components = {'total': loss, 'L_task': loss}

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

        # Backward pass
        if use_amp and scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights every N steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp and scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            global_step += 1

            # Logging
            if is_main_process(rank) and global_step % log_interval == 0:
                elapsed_time = time.time() - start_time
                tokens_per_sec = tokens_processed / elapsed_time if elapsed_time > 0 else 0

                L_task = loss_components.get('L_task', torch.tensor(0.0)).item() if isinstance(loss_components, dict) else loss.item()
                L_mean = loss_components.get('L_mean', torch.tensor(0.0)).item() if isinstance(loss_components, dict) else 0.0
                L_speed = loss_components.get('L_speed', torch.tensor(0.0)).item() if isinstance(loss_components, dict) else 0.0
                L_energy = loss_components.get('L_energy', torch.tensor(0.0)).item() if isinstance(loss_components, dict) else 0.0

                print(f'  Step {global_step}, Loss: {loss.item() * gradient_accumulation_steps:.4f} '
                      f'[Task: {L_task:.4f}, Mean: {L_mean:.4f}, Speed: {L_speed:.4f}, Energy: {L_energy:.4f}] '
                      f'| {tokens_per_sec:.0f} tokens/s | LR: {optimizer.param_groups[0]["lr"]:.2e}')

                # W&B logging
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({
                        'train/loss': loss.item() * gradient_accumulation_steps,
                        'train/loss_task': L_task,
                        'train/loss_mean': L_mean,
                        'train/loss_speed': L_speed,
                        'train/loss_energy': L_energy,
                        'train/learning_rate': optimizer.param_groups[0]['lr'],
                        'train/tokens_per_sec': tokens_per_sec,
                        'train/epoch': epoch,
                        'global_step': global_step
                    })

            # Checkpointing
            if is_main_process(rank) and global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step,
                    loss.item() * gradient_accumulation_steps, checkpoint_dir, rank
                )

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

    return total_loss / num_batches, global_step


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Production Pre-Training for INL-LLM')

    # Distributed training
    parser.add_argument('--distributed', action='store_true', help='Enable multi-GPU training')

    # Dataset
    parser.add_argument('--streaming', action='store_true', help='Use streaming dataset')
    parser.add_argument('--dataset', type=str, default='codeparrot',
                       help='Dataset name: codeparrot, the-stack-dedup, starcoderdata, c4, pile')
    parser.add_argument('--dataset-config', type=str, default='en', help='Dataset config')
    parser.add_argument('--seq-len', type=int, default=2048, help='Sequence length')
    parser.add_argument('--languages', type=str, default='python',
                       help='Programming languages (comma-separated): python,javascript,java')
    parser.add_argument('--local-code', type=str, default=None,
                       help='Path to local code directory for training')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max samples for testing (None = unlimited)')

    # Training
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--num-epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--use-amp', action='store_true', help='Use automatic mixed precision (bf16)')

    # Checkpointing
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--checkpoint-interval', type=int, default=500, help='Save checkpoint every N steps')
    parser.add_argument('--keep-last-n', type=int, default=5, help='Keep only last N checkpoints')

    # Logging
    parser.add_argument('--use-wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='inl-llm-pretraining', help='W&B project name')
    parser.add_argument('--log-interval', type=int, default=10, help='Log every N steps')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup distributed training
    local_rank, rank, world_size = setup_distributed() if args.distributed else (None, 0, 1)

    if is_main_process(rank):
        print("="*70)
        print("PRODUCTION PRE-TRAINING - INL-LLM")
        print("="*70)
        print(f"\n🚀 Configuration:")
        print(f"  Distributed: {args.distributed} (world_size={world_size})")
        print(f"  Streaming: {args.streaming}")
        if args.streaming:
            print(f"  Dataset: {args.dataset} ({args.dataset_config})")
        print(f"  Batch size: {args.batch_size} (per GPU)")
        print(f"  Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"  Sequence length: {args.seq_len}")
        print(f"  Mixed precision: {args.use_amp}")
        print(f"  Checkpointing: Every {args.checkpoint_interval} steps")
        print(f"  Resume from: {args.resume if args.resume else 'None (fresh start)'}")

    # Setup device
    if args.distributed and local_rank is not None:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_main_process(rank):
        print(f"  Device: {device}")

    # Load tokenizer (GPT-2 BPE tokenizer, same as used by many LLMs)
    if is_main_process(rank):
        print("\n Loading tokenizer...")

    if TOKENIZER_AVAILABLE:
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token

            # Add special tokens for chat format
            special_tokens = {
                'additional_special_tokens': ['<USER>', '<ASSISTANT>', '<SYSTEM>', '<ERROR>']
            }
            tokenizer.add_special_tokens(special_tokens)

            # Add Jinja chat template
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}<SYSTEM> {{ message['content'] }}"
                "{% elif message['role'] == 'user' %}<USER> {{ message['content'] }}"
                "{% elif message['role'] == 'assistant' %}<ASSISTANT> {{ message['content'] }}"
                "{% endif %}"
                "{% if not loop.last %}\n{% endif %}"
                "{% endfor %}"
            )

            vocab_size = tokenizer.vocab_size
            if is_main_process(rank):
                print(f"✅ Tokenizer loaded: GPT-2 BPE (vocab_size={vocab_size})")
        except Exception as e:
            if is_main_process(rank):
                print(f"❌ Failed to load tokenizer: {e}")
            tokenizer = None
            vocab_size = 50000
    else:
        tokenizer = None
        vocab_size = 50000
        if is_main_process(rank):
            print(f"⚠️ transformers not installed")

    # Initialize Weights & Biases
    if args.use_wandb and WANDB_AVAILABLE and is_main_process(rank):
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"inl-1.1b-{args.dataset if args.streaming else 'parquet'}"
        )
        print(f"✅ W&B logging enabled: {args.wandb_project}")

    # Create model
    if is_main_process(rank):
        print("\n🔧 Creating INL-LLM model (~500M parameters)...")

    actual_vocab_size = len(tokenizer) if tokenizer else vocab_size
    model = UltraOptimizedIntegratorLanguageModel(
        vocab_size=actual_vocab_size,
        d_model=1280,
        num_layers=18,
        num_heads=20,
        num_iterations_per_layer=2,
        feedforward_dim=5120,
        max_seq_len=args.seq_len,
        use_lowrank_embeddings=True,
        lowrank_ratio=0.125,
        use_gradient_checkpointing=True,
        use_shared_controllers=True,
        hierarchical_group_size=64,
        excitation_sparsity=0.1
    )
    model = model.to(device)

    if is_main_process(rank):
        print(f"✅ Model created: {model.get_num_params():,} parameters")

    # Wrap model with DDP
    if args.distributed and world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main_process(rank):
            print(f"✅ Model wrapped with DistributedDataParallel (world_size={world_size})")

    # Create dataset
    if is_main_process(rank):
        print("\n📦 Loading dataset...")

    # Parse languages
    languages = [lang.strip() for lang in args.languages.split(',')] if args.languages else None

    if args.local_code:
        # Local code directory
        if is_main_process(rank):
            print(f"  Loading local code from: {args.local_code}")

        if not CODE_DATASETS_AVAILABLE:
            raise ImportError("code_datasets required. Check code_datasets.py exists.")

        dataset = LocalCodeDataset(
            root_dir=args.local_code,
            seq_len=args.seq_len,
            tokenizer=tokenizer
        )

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if args.distributed else None
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=0,
            pin_memory=True
        )
        total_steps_estimate = args.num_epochs * len(dataloader) // args.gradient_accumulation_steps
        use_streaming = False

    elif args.streaming:
        # Streaming dataset for large-scale pre-training
        if CODE_DATASETS_AVAILABLE and args.dataset in ['codeparrot', 'the-stack-dedup', 'starcoderdata', 'the-stack', 'codesearchnet', 'mbpp', 'stack-smol', 'the-stack-v2']:
            # Use code datasets
            if is_main_process(rank):
                print(f"  Using CODE dataset: {args.dataset}")
                print(f"  Languages: {languages}")

            dataset = StreamingCodeDataset(
                dataset_name=args.dataset,
                languages=languages,
                seq_len=args.seq_len,
                tokenizer=tokenizer,
                max_samples=args.max_samples,
                pack_sequences=True
            )
        else:
            # Fallback to text datasets (C4, Pile, etc.)
            if not DATASETS_AVAILABLE:
                raise ImportError("datasets library required. Install: pip install datasets")

            if is_main_process(rank):
                print(f"  Using TEXT dataset: {args.dataset}")

            dataset = StreamingTextDataset(
                dataset_name=args.dataset,
                dataset_config=args.dataset_config,
                split='train',
                seq_len=args.seq_len,
                tokenizer=tokenizer,
                streaming=True
            )

        dataloader = None  # Streaming uses iter() directly
        total_steps_estimate = args.max_samples if args.max_samples else 100000
        use_streaming = True

    else:
        # Local parquet dataset for testing
        parquet_path = os.path.join(os.path.dirname(__file__), 'part_000000.parquet')
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Dataset not found: {parquet_path}")

        dataset = ParquetTextDataset(
            parquet_path=parquet_path,
            seq_len=args.seq_len,
            tokenizer=tokenizer
        )

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if args.distributed else None
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=0,
            pin_memory=True
        )
        total_steps_estimate = args.num_epochs * len(dataloader) // args.gradient_accumulation_steps
        use_streaming = False

    if is_main_process(rank):
        print(f"✅ Dataset loaded")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # Learning rate scheduler
    from torch.optim.lr_scheduler import OneCycleLR
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps_estimate,
        pct_start=0.1,
        anneal_strategy='cos'
    )

    if is_main_process(rank):
        print(f"✅ Optimizer: AdamW (lr={args.learning_rate}, total_steps={total_steps_estimate})")

    # Loss function - Pure CrossEntropy sans régularisations d'équilibre
    # Les régularisations (lambda_mean, lambda_speed, lambda_energy) sont incompatibles
    # avec la génération de texte/code car elles forcent une convergence vers un équilibre
    # au lieu de permettre des prédictions de tokens précises et diverses.
    integrator_loss_fn = IntegratorLoss(
        target_value=0.0,
        lambda_mean_init=0.0,   # Désactivé - pas d'équilibre forcé
        lambda_speed=0.0,       # Désactivé - pas de pénalité de vitesse
        lambda_energy=0.0,      # Désactivé - pas de régularisation d'énergie
        annealing_epochs=args.num_epochs,
        variance_weighted=False,
        task_loss_type='ce'     # Pure CrossEntropy pour language modeling
    )

    if is_main_process(rank):
        print(f"✅ Loss function: Pure CrossEntropy (optimized for language modeling)")

    # Create cycle scheduler
    cycle_scheduler = create_cycle_scheduler(
        preset='balanced',
        total_epochs=args.num_epochs
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0

    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, lr_scheduler
        )
        if is_main_process(rank):
            print(f"✅ Resumed from step {global_step}")

    # Training loop
    if is_main_process(rank):
        print("\n" + "="*70)
        print("🚀 STARTING TRAINING")
        print("="*70)

    try:
        for epoch in range(start_epoch, args.num_epochs):
            if is_main_process(rank):
                print(f"\n📍 Epoch {epoch+1}/{args.num_epochs}")

            # Update phase
            phase_info = cycle_scheduler.step(epoch)
            if is_main_process(rank):
                print(f"  Phase: {phase_info['phase_name']}")

            integrator_loss_fn.set_exploration_phase(phase_info['phase_name'] == 'exploration')

            # Set epoch for distributed sampler
            if args.distributed and hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
                dataloader.sampler.set_epoch(epoch)

            # Train epoch
            if use_streaming:
                # Streaming training (code datasets, etc.)
                avg_loss, global_step = train_streaming_epoch(
                    model=model,
                    dataset=dataset,
                    loss_fn=integrator_loss_fn,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    device=device,
                    epoch=epoch,
                    batch_size=args.batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    rank=rank,
                    global_step=global_step,
                    log_interval=args.log_interval,
                    checkpoint_interval=args.checkpoint_interval,
                    checkpoint_dir=args.checkpoint_dir,
                    use_amp=args.use_amp,
                    max_steps_per_epoch=total_steps_estimate // args.num_epochs
                )
            else:
                # DataLoader-based training (local files, parquet)
                avg_loss, global_step = train_epoch(
                    model=model,
                    dataloader=dataloader,
                    loss_fn=integrator_loss_fn,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    device=device,
                    epoch=epoch,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    rank=rank,
                    global_step=global_step,
                    log_interval=args.log_interval,
                    checkpoint_interval=args.checkpoint_interval,
                    checkpoint_dir=args.checkpoint_dir,
                    use_amp=args.use_amp
                )

            if is_main_process(rank):
                print(f"  Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}")

                # W&B epoch-level logging
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({
                        'epoch': epoch + 1,
                        'epoch_avg_loss': avg_loss,
                        'global_step': global_step
                    })

    except KeyboardInterrupt:
        if is_main_process(rank):
            print("\n⚠️ Training interrupted by user")

    finally:
        # Final checkpoint
        if is_main_process(rank):
            print("\n💾 Saving final checkpoint...")
            save_checkpoint(
                model, optimizer, lr_scheduler, args.num_epochs, global_step,
                0.0, args.checkpoint_dir, rank, keep_last_n=args.keep_last_n
            )

    if is_main_process(rank):
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)

    # Save final model in HuggingFace format
    if is_main_process(rank):
        save_dir = os.path.join(args.checkpoint_dir, 'final_model')
        os.makedirs(save_dir, exist_ok=True)

        # Unwrap DDP model
        model_to_save = model.module if hasattr(model, 'module') else model

        # Save model weights
        model_path = os.path.join(save_dir, 'pytorch_model.pt')
        torch.save(model_to_save.state_dict(), model_path)
        print(f"\n💾 Model saved to: {model_path}")

    # Save model config.json
    config = {
        "architectures": ["UltraOptimizedIntegratorLanguageModel"],
        "model_type": "inl-llm",
        "transformers_version": "4.57.0",

        # Architecture
        "vocab_size": actual_vocab_size,
        "d_model": 1280,
        "num_layers": 18,
        "num_heads": 20,
        "num_iterations_per_layer": 2,
        "feedforward_dim": 5120,
        "max_seq_len": 2048,
        "dropout": 0.1,

        # Token IDs (from tokenizer)
        "bos_token_id": tokenizer.bos_token_id if tokenizer else 1,
        "eos_token_id": tokenizer.eos_token_id if tokenizer else 2,
        "pad_token_id": tokenizer.pad_token_id if tokenizer else 0,
        "unk_token_id": tokenizer.unk_token_id if tokenizer else 3,

        # Optimizations
        "use_lowrank_embeddings": True,
        "lowrank_ratio": 0.125,
        "use_gradient_checkpointing": True,
        "use_shared_controllers": True,
        "use_adaptive_stopping": True,
        "adaptive_convergence_threshold": 0.001,
        "hierarchical_group_size": 64,
        "excitation_sparsity": 0.1,

        # Training config
        "dtype": "bfloat16",
        "use_cache": True,
        "initializer_range": 0.02,

        # INL-LLM specific
        "integrator_type": "ultra_optimized",
        "controller_type": "shared",
        "equilibrium_type": "hierarchical",
        "excitation_type": "sparse_harmonic"
    }

    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 Model config saved to: {config_path}")

    # Save tokenizer (if available)
    if tokenizer:
        tokenizer.save_pretrained(save_dir)
        print(f"💾 Tokenizer saved to: {save_dir}")
        print(f"   Files: vocab.json, merges.txt, tokenizer_config.json (with Jinja chat template)")

    print(f"\n✅ Complete checkpoint saved to: {save_dir}")
    print(f"   📦 Files: pytorch_model.pt, config.json, tokenizer files")

    # Test generation
    print("\nTesting generation...")
    model.eval()
    with torch.no_grad():
        if tokenizer:
            # Test 1: Simple text generation
            prompt_text = "Once upon a time"
            print(f"\n📝 Test 1 - Simple Prompt: '{prompt_text}'")

            # Tokenize
            prompt_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
            print(f"   Tokenized: {prompt_ids.shape}")

            # Generate (with KV caching for speed!)
            output = model.generate(
                prompt_ids,
                max_new_tokens=50,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                use_cache=True  # Enable KV cache for fast generation
            )

            # Decode
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"\n🎯 Generated text:")
            print(f"   {generated_text}")

            # Test 2: Chat template usage
            print(f"\n📝 Test 2 - Chat Template (Jinja):")
            messages = [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI that enables systems to learn from data."},
                {"role": "user", "content": "Give me an example"}
            ]

            # Apply chat template
            chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(f"\n🔤 Formatted conversation:")
            print(chat_text)

            # Tokenize and generate response (with KV caching)
            chat_ids = tokenizer.encode(chat_text, return_tensors='pt').to(device)
            chat_output = model.generate(
                chat_ids,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                use_cache=True  # Enable KV cache
            )

            # Decode the response
            response = tokenizer.decode(chat_output[0][chat_ids.shape[1]:], skip_special_tokens=True)
            print(f"\n🤖 Assistant response:")
            print(f"   {response}")
        else:
            # Test with synthetic data
            prompt = torch.randint(0, vocab_size, (1, 10)).to(device)
            output = model.generate(
                prompt,
                max_new_tokens=50,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                use_cache=True  # Enable KV cache
            )
            print(f"Generated {output.size(1)} tokens")
            print(f"Output shape: {output.shape}")

    if is_main_process(rank):
        print("\n✅ Training completed successfully!")

    # Cleanup
    if args.use_wandb and WANDB_AVAILABLE and is_main_process(rank):
        wandb.finish()

    if args.distributed:
        cleanup_distributed()


if __name__ == '__main__':
    main()
