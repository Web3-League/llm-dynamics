#!/usr/bin/env python3
"""
INL-LLM v3 Knowledge Distillation Training

Train a larger student model (3.8B) using a smaller teacher model (500M @ 230K steps).
The teacher provides soft targets that help the student learn more efficiently.

Distillation loss = alpha * KL(student || teacher) + (1-alpha) * CE(student, labels)
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from datasets import load_dataset

# v3 for student (new architecture with GQA, RoPE, SwiGLU)
from inl_llm_v3 import UltraOptimizedIntegratorLanguageModel as StudentModel

# v2 for teacher (original architecture - checkpoint 230K)
from inl_llm import IntegratorLanguageModel as TeacherModel


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    # Teacher model (500M) - v2 architecture, trained for 230K steps
    "teacher_500m": {
        "vocab_size": 50261,
        "d_model": 1280,
        "num_layers": 18,
        "num_heads": 20,
        "num_iterations_per_layer": 2,
        "feedforward_dim": 5120,
        "max_seq_len": 1024,
        # v2 settings (no GQA, uses all heads for KV)
        "num_kv_heads": 20,  # MHA for v2
    },

    # Student models - v3 architecture with GQA
    "student_3.8b": {
        "vocab_size": 50261,
        "d_model": 3072,       # Wider
        "num_layers": 32,      # Deeper
        "num_heads": 24,
        "num_iterations_per_layer": 2,
        "feedforward_dim": 12288,  # 4 * d_model
        "max_seq_len": 1024,
        # v3: GQA with 6 KV heads (4 Q heads per KV head)
        "num_kv_heads": 6,
    },

    "student_7b": {
        "vocab_size": 50261,
        "d_model": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "num_iterations_per_layer": 2,
        "feedforward_dim": 14336,  # ~3.5 * d_model (LLaMA style)
        "max_seq_len": 1024,
        "num_kv_heads": 8,  # 4 Q heads per KV head
    },

    "student_13b": {
        "vocab_size": 50261,
        "d_model": 5120,
        "num_layers": 40,
        "num_heads": 40,
        "num_iterations_per_layer": 2,
        "feedforward_dim": 17920,
        "max_seq_len": 1024,
        "num_kv_heads": 8,
    },
}


def create_student(config_name: str, use_gradient_checkpointing: bool = True):
    """Create a student model (v3 architecture) from config."""
    config = MODEL_CONFIGS[config_name]
    return StudentModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        num_iterations_per_layer=config["num_iterations_per_layer"],
        feedforward_dim=config["feedforward_dim"],
        max_seq_len=config["max_seq_len"],
        num_kv_heads=config.get("num_kv_heads"),
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


def create_teacher():
    """Create the teacher model (v2 architecture) - matches checkpoint 230K."""
    config = MODEL_CONFIGS["teacher_500m"]
    return TeacherModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        num_iterations_per_layer=config["num_iterations_per_layer"],
        feedforward_dim=config["feedforward_dim"],
        max_seq_len=config["max_seq_len"],
    )


def count_parameters(model, trainable_only=False):
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def load_teacher(checkpoint_path: str, device: str = "cuda"):
    """Load the teacher model (v2 architecture) from checkpoint."""
    print(f"Loading teacher from {checkpoint_path}...")

    # Create teacher model (500M v2 architecture)
    teacher = create_teacher()

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in ckpt:
        teacher.load_state_dict(ckpt["model_state_dict"])
    else:
        teacher.load_state_dict(ckpt)

    teacher = teacher.to(device)
    teacher.eval()

    # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False

    print(f"Teacher loaded: {count_parameters(teacher):,} params")
    return teacher


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
    ignore_index: int = -100
):
    """
    Compute distillation loss.

    Args:
        student_logits: [B, S, V] student output logits
        teacher_logits: [B, S, V] teacher output logits
        labels: [B, S] ground truth labels
        temperature: Softmax temperature (higher = softer)
        alpha: Weight for distillation loss (1-alpha for CE loss)
        ignore_index: Index to ignore in CE loss

    Returns:
        total_loss, distill_loss, ce_loss
    """
    # Soft targets from teacher (with temperature)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL divergence loss (scaled by T^2 as per Hinton et al.)
    # KL(teacher || student) = sum(teacher * log(teacher/student))
    distill_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean"
    ) * (temperature ** 2)

    # Standard cross-entropy with hard labels
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index
    )

    # Combined loss
    total_loss = alpha * distill_loss + (1 - alpha) * ce_loss

    return total_loss, distill_loss, ce_loss


def get_batch(dataset_iter, tokenizer, batch_size, seq_len, device):
    """Get a batch from streaming dataset."""
    texts = []
    for _ in range(batch_size):
        try:
            sample = next(dataset_iter)
            text = sample.get("content", sample.get("text", ""))
            texts.append(text)
        except StopIteration:
            break

    if not texts:
        return None, None

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=seq_len + 1,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = encodings["input_ids"][:, :-1].to(device)
    labels = encodings["input_ids"][:, 1:].to(device)

    return input_ids, labels


def save_checkpoint(model, optimizer, step, loss, path):
    """Save checkpoint."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": step,
        "loss": loss
    }, path)
    print(f"Saved checkpoint at step {step} to {path}")


def load_checkpoint(model, optimizer, path):
    """Load checkpoint, return start_step."""
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return 0

    ckpt = torch.load(path, map_location="cpu")

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("global_step", 0)
    else:
        model.load_state_dict(ckpt)
        start_step = 0

    print(f"Loaded checkpoint from {path}, starting at step {start_step}")
    return start_step


def train(args):
    """Main distillation training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load teacher model
    print("\n" + "=" * 60)
    print("LOADING TEACHER MODEL")
    print("=" * 60)
    teacher = load_teacher(args.teacher_checkpoint, device)

    # Create student model
    print("\n" + "=" * 60)
    print(f"CREATING STUDENT MODEL: {args.student_config}")
    print("=" * 60)
    student = create_student(args.student_config, use_gradient_checkpointing=True)
    student = student.to(device)

    student_params = count_parameters(student)
    teacher_params = count_parameters(teacher)
    print(f"\nTeacher params: {teacher_params:,} ({teacher_params/1e6:.1f}M)")
    print(f"Student params: {student_params:,} ({student_params/1e9:.2f}B)")
    print(f"Student/Teacher ratio: {student_params/teacher_params:.1f}x")

    # Optimizer
    optimizer = AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )

    # Load checkpoint if resuming
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(student, optimizer, args.resume)

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.max_steps - start_step,
        eta_min=args.lr * 0.1
    )

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}...")
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split="train",
        streaming=True,
        trust_remote_code=False
    )
    dataset_iter = iter(dataset)

    # Training info
    print("\n" + "=" * 60)
    print("STARTING DISTILLATION TRAINING")
    print("=" * 60)
    print(f"  Start step: {start_step}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Alpha (distill weight): {args.alpha}")
    print("=" * 60 + "\n")

    # Training loop
    student.train()
    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    step = start_step
    running_loss = 0.0
    running_distill = 0.0
    running_ce = 0.0
    optimizer.zero_grad()

    while step < args.max_steps:
        # Get batch
        input_ids, labels = get_batch(
            dataset_iter, tokenizer, args.batch_size, args.seq_len, device
        )

        if input_ids is None:
            print("Resetting dataset iterator...")
            dataset_iter = iter(dataset)
            continue

        # Forward pass
        if args.amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_outputs = teacher(input_ids)
                    if isinstance(teacher_outputs, tuple):
                        teacher_logits = teacher_outputs[0]
                    else:
                        teacher_logits = teacher_outputs

                # Student forward
                student_outputs = student(input_ids)
                if isinstance(student_outputs, tuple):
                    student_logits = student_outputs[0]
                else:
                    student_logits = student_outputs

                # Distillation loss
                loss, distill_loss, ce_loss = distillation_loss(
                    student_logits,
                    teacher_logits,
                    labels,
                    temperature=args.temperature,
                    alpha=args.alpha
                )
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
        else:
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_outputs = teacher(input_ids)
                if isinstance(teacher_outputs, tuple):
                    teacher_logits = teacher_outputs[0]
                else:
                    teacher_logits = teacher_outputs

            # Student forward
            student_outputs = student(input_ids)
            if isinstance(student_outputs, tuple):
                student_logits = student_outputs[0]
            else:
                student_logits = student_outputs

            # Distillation loss
            loss, distill_loss, ce_loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=args.temperature,
                alpha=args.alpha
            )
            loss = loss / args.grad_accum
            loss.backward()

        running_loss += loss.item() * args.grad_accum
        running_distill += distill_loss.item()
        running_ce += ce_loss.item()

        # Gradient accumulation step
        if (step + 1) % args.grad_accum == 0:
            if args.amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        step += 1

        # Logging
        if step % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            avg_distill = running_distill / args.log_interval
            avg_ce = running_ce / args.log_interval
            lr = scheduler.get_last_lr()[0]

            print(f"Step {step:>7} | Loss: {avg_loss:.4f} | "
                  f"Distill: {avg_distill:.4f} | CE: {avg_ce:.4f} | LR: {lr:.2e}")

            running_loss = 0.0
            running_distill = 0.0
            running_ce = 0.0

        # Save checkpoint
        if step % args.save_interval == 0:
            save_checkpoint(
                student, optimizer, step,
                avg_loss if "avg_loss" in dir() else 0,
                os.path.join(args.checkpoint_dir, f"student-{args.student_config}-step-{step}.pt")
            )

    # Final save
    print("\nDistillation training complete!")
    save_checkpoint(
        student, optimizer, step, running_loss,
        os.path.join(args.checkpoint_dir, f"student-{args.student_config}-final.pt")
    )

    # Save for deployment
    print("Saving final model...")
    torch.save(
        student.state_dict(),
        os.path.join(args.checkpoint_dir, f"student-{args.student_config}-pytorch_model.pt")
    )

    try:
        from safetensors.torch import save_file
        save_file(
            student.state_dict(),
            os.path.join(args.checkpoint_dir, f"student-{args.student_config}-model.safetensors")
        )
        print(f"Saved safetensors model")
    except ImportError:
        print("safetensors not installed, skipping")


def main():
    parser = argparse.ArgumentParser(description="INL-LLM v3 Distillation Training")

    # Model config
    parser.add_argument("--teacher-checkpoint", type=str, required=True,
                        help="Path to teacher checkpoint (500M @ 230K)")
    parser.add_argument("--student-config", type=str, default="student_3.8b",
                        choices=["student_3.8b", "student_7b", "student_13b"],
                        help="Student model configuration")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume student from checkpoint")

    # Distillation params
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for distillation loss (1-alpha for CE)")

    # Data
    parser.add_argument("--dataset", type=str, default="bigcode/starcoderdata",
                        help="Dataset name")
    parser.add_argument("--dataset-config", type=str, default="default",
                        help="Dataset config")
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Sequence length")

    # Training
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size per GPU")
    parser.add_argument("--grad-accum", type=int, default=32,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=100000,
                        help="Max training steps")
    parser.add_argument("--amp", action="store_true",
                        help="Use mixed precision")

    # Logging/saving
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Log every N steps")
    parser.add_argument("--save-interval", type=int, default=10000,
                        help="Save every N steps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
