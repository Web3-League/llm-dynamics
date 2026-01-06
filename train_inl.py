#!/usr/bin/env python3
"""
INL-LLM Training Script - Clean and Simple
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from inl_llm import IntegratorLanguageModel

def create_model(vocab_size=50261, max_seq_len=1024):
    """Create the INL-LLM v2 model (500M teacher)."""
    return IntegratorLanguageModel(
        vocab_size=50261,
        d_model=1280,
        num_layers=18,
        num_heads=20,
        num_iterations_per_layer=2,
        feedforward_dim=5120,
        max_seq_len=max_seq_len
    )

def load_checkpoint(model, optimizer, path):
    """Load checkpoint, return start_step."""
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return 0

    ckpt = torch.load(path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        start_step = ckpt.get('global_step', 0)
    else:
        # Direct state dict
        model.load_state_dict(ckpt)
        start_step = 0

    print(f"Loaded checkpoint from {path}, starting at step {start_step}")
    return start_step

def save_checkpoint(model, optimizer, step, loss, path):
    """Save checkpoint."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': step,
        'loss': loss
    }, path)
    print(f"Saved checkpoint at step {step} to {path}")

def get_batch(dataset_iter, tokenizer, batch_size, seq_len, device):
    """Get a batch from streaming dataset."""
    texts = []
    for _ in range(batch_size):
        try:
            sample = next(dataset_iter)
            text = sample.get('content', sample.get('text', ''))
            texts.append(text)
        except StopIteration:
            break

    if not texts:
        return None, None

    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=seq_len + 1,
        padding='max_length',
        return_tensors='pt'
    )

    input_ids = encodings['input_ids'][:, :-1].to(device)
    labels = encodings['input_ids'][:, 1:].to(device)

    return input_ids, labels

def train(args):
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create model
    print("Creating model...")
    model = create_model(vocab_size=50261, max_seq_len=args.seq_len)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Load checkpoint if resuming
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, optimizer, args.resume)

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps - start_step, eta_min=args.lr * 0.1)

    # Loss function (use -100 as ignore index for safety)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Load dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split='train',
        streaming=True,
        trust_remote_code=False
    )
    dataset_iter = iter(dataset)

    # Training loop
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING")
    print(f"  Start step: {start_step}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print(f"{'='*60}\n")

    # TensorBoard
    writer = SummaryWriter(f'runs/inl_500m_step{start_step}')
    print(f"TensorBoard logging to: runs/inl_500m_step{start_step}")

    model.train()
    scaler = torch.amp.GradScaler('cuda') if args.amp else None

    step = start_step
    running_loss = 0.0
    optimizer.zero_grad()

    while step < args.max_steps:
        # Get batch
        input_ids, labels = get_batch(dataset_iter, tokenizer, args.batch_size, args.seq_len, device)

        if input_ids is None:
            # Reset dataset iterator
            print("Resetting dataset iterator...")
            dataset_iter = iter(dataset)
            continue

        # Forward pass
        if args.amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(input_ids)
                # Handle different output formats
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                elif isinstance(outputs, tuple):
                    logits = outputs[0]  # First element is usually logits
                else:
                    logits = outputs.logits
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
        else:
            outputs = model(input_ids)
            # Handle different output formats
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            elif isinstance(outputs, tuple):
                logits = outputs[0]  # First element is usually logits
            else:
                logits = outputs.logits
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss / args.grad_accum
            loss.backward()

        running_loss += loss.item() * args.grad_accum

        # Gradient accumulation step
        if (step + 1) % args.grad_accum == 0:
            if args.amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        step += 1

        # Logging
        if step % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            lr = scheduler.get_last_lr()[0]
            print(f"Step {step:>7} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

            # TensorBoard logging
            writer.add_scalar('Loss/train', avg_loss, step)
            writer.add_scalar('LR', lr, step)

            running_loss = 0.0

        # Save checkpoint
        if step % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, step, avg_loss if 'avg_loss' in dir() else 0,
                os.path.join(args.checkpoint_dir, f'checkpoint-step-{step}.pt')
            )

    # Close TensorBoard
    writer.close()

    # Final save
    print("\nTraining complete!")
    save_checkpoint(
        model, optimizer, step, running_loss,
        os.path.join(args.checkpoint_dir, 'final_model.pt')
    )

    # Save for HuggingFace
    print("Saving final model...")
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'pytorch_model.pt'))

    # Convert to safetensors
    try:
        from safetensors.torch import save_file
        save_file(model.state_dict(), os.path.join(args.checkpoint_dir, 'model.safetensors'))
        print(f"Saved safetensors to {args.checkpoint_dir}/model.safetensors")
    except ImportError:
        print("safetensors not installed, skipping")

def main():
    parser = argparse.ArgumentParser(description='INL-LLM Training')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--dataset', type=str, default='bigcode/starcoderdata', help='Dataset name')
    parser.add_argument('--dataset-config', type=str, default='default', help='Dataset config')
    parser.add_argument('--seq-len', type=int, default=1024, help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--grad-accum', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max-steps', type=int, default=500000, help='Max training steps')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--log-interval', type=int, default=100, help='Log every N steps')
    parser.add_argument('--save-interval', type=int, default=5000, help='Save every N steps')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
