#!/usr/bin/env python3
"""
Parse training logs and create TensorBoard graphs retroactively.

Usage:
    python parse_logs_to_tensorboard.py training.log
    python parse_logs_to_tensorboard.py training.log --run-name inl_500m

Then view with:
    tensorboard --logdir=runs
"""
import re
import argparse
from torch.utils.tensorboard import SummaryWriter


def parse_log_file(log_path):
    """Parse training log file and extract metrics."""
    pattern = r"Step\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+LR:\s+([\d.e+-]+)"

    metrics = []
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                lr = float(match.group(3))
                metrics.append({
                    'step': step,
                    'loss': loss,
                    'lr': lr
                })

    return metrics


def write_to_tensorboard(metrics, run_name='inl_500m'):
    """Write metrics to TensorBoard."""
    writer = SummaryWriter(f'runs/{run_name}')

    for m in metrics:
        writer.add_scalar('Loss/train', m['loss'], m['step'])
        writer.add_scalar('LR', m['lr'], m['step'])

    writer.close()
    print(f"Written {len(metrics)} data points to runs/{run_name}")


def main():
    parser = argparse.ArgumentParser(description='Parse logs to TensorBoard')
    parser.add_argument('log_file', type=str, help='Path to training log file')
    parser.add_argument('--run-name', type=str, default='inl_500m',
                        help='TensorBoard run name')
    args = parser.parse_args()

    print(f"Parsing {args.log_file}...")
    metrics = parse_log_file(args.log_file)

    if not metrics:
        print("No metrics found in log file!")
        print("Expected format: 'Step  123456 | Loss: 0.5432 | LR: 5.00e-05'")
        return

    print(f"Found {len(metrics)} data points")
    print(f"  Steps: {metrics[0]['step']} -> {metrics[-1]['step']}")
    print(f"  Loss: {metrics[0]['loss']:.4f} -> {metrics[-1]['loss']:.4f}")

    write_to_tensorboard(metrics, args.run_name)
    print(f"\nRun: tensorboard --logdir=runs")


if __name__ == '__main__':
    main()
