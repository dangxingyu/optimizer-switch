"""
Train Llama models with AdamW optimizer - Single GPU version
Baseline version for comparison with Muon optimizer

Usage: python train_llama_adamw_single_gpu.py
"""

import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging

import time
from dataclasses import dataclass
from pathlib import Path

# Disable all network access
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

# Import custom Llama implementation
from llama_model import LlamaForCausalLM, LlamaConfig

# Import QA data utilities
from qa_data_utils import create_qa_dataloaders


# ============================================================================
# Training utilities
# ============================================================================

# Import evaluation utilities
from eval_utils import evaluate_gsm8k, evaluate_siqa, evaluate_loss_only


# ============================================================================
# Main training configuration and loop
# ============================================================================

@dataclass
class TrainingConfig:
    # Model and checkpoint
    checkpoint_path: str = "../checkpoints/adamw_130m_1"

    # Dataset - Choose one:
    # For GSM8K:
    dataset_name: str = "openai/gsm8k"
    dataset_type: str = "gsm8k"
    dataset_config: str = "main"

    # For SIQA:
    # dataset_name: str = "allenai/social_i_qa"
    # dataset_type: str = "siqa"
    # dataset_config: str = None

    # Training hyperparameters
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 512
    num_epochs: int = 3
    max_steps: int = -1  # -1 for full epochs

    # AdamW optimizer
    adamw_lr: float = 3e-5
    adamw_betas: tuple = (0.9, 0.95)
    adamw_weight_decay: float = 0.00
    adamw_eps: float = 1e-8

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Learning rate schedule
    warmup_steps: int = 50
    use_cosine_schedule: bool = True

    # Logging
    log_interval: int = 1  # Log every step
    eval_interval: int = 50
    save_interval: int = 500
    output_dir: str = "output_adamw"

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp32: bool = True  # Use FP32 instead of bfloat16 for stability


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_steps', type=int, default=None)
    args = parser.parse_args()

    config = TrainingConfig()

    # Override with command line arguments
    if args.checkpoint_path:
        config.checkpoint_path = args.checkpoint_path
    if args.lr:
        config.adamw_lr = args.lr
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_steps:
        config.max_steps = args.max_steps

    print("=" * 80)
    print(f"Training Llama with AdamW optimizer - Single GPU")
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {config.device}")
    print("=" * 80)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Create training log file
    log_file = os.path.join(config.output_dir, "training_log.csv")
    with open(log_file, 'w') as f:
        f.write("step,train_loss,grad_norm,val_loss\n")

    # ========================================================================
    # Load data
    # ========================================================================

    print(f"\nLoading dataset: {config.dataset_name}")
    print(f"Dataset type: {config.dataset_type}")

    # For single GPU, we don't need DDP samplers
    # Modify qa_data_utils to support single GPU
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if config.dataset_config:
        dataset = load_dataset(config.dataset_name, config.dataset_config)
    else:
        dataset = load_dataset(config.dataset_name)

    # Import formatting functions
    from qa_data_utils import format_gsm8k_example, format_siqa_example

    if config.dataset_type == 'gsm8k':
        format_fn = format_gsm8k_example
    elif config.dataset_type == 'siqa':
        format_fn = format_siqa_example
    else:
        raise ValueError(f"Unknown dataset_type: {config.dataset_type}")

    def tokenize_function(examples):
        if isinstance(examples['question'], list):
            texts = [format_fn({k: examples[k][i] for k in examples.keys()})
                    for i in range(len(examples['question']))]
        else:
            texts = [format_fn(examples)]

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_length,
            padding='max_length',
            return_tensors='pt'
        )

        labels = tokenized['input_ids'].clone()
        for i in range(len(labels)):
            labels[i][tokenized['attention_mask'][i] == 0] = -100

        tokenized['labels'] = labels
        return tokenized

    # Tokenize
    train_key = 'train'
    val_key = 'validation' if 'validation' in dataset else 'test'

    tokenized_train = dataset[train_key].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset[train_key].column_names,
    )

    if val_key in dataset:
        tokenized_val = dataset[val_key].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset[val_key].column_names,
        )
    else:
        tokenized_val = tokenized_train.select(range(min(500, len(tokenized_train))))

    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Create simple dataloaders (no DDP)
    train_loader = DataLoader(
        tokenized_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        tokenized_val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"✓ Dataset loaded")
    print(f"  Train: {len(tokenized_train)} examples ({len(train_loader)} batches)")
    print(f"  Val: {len(tokenized_val)} examples ({len(val_loader)} batches)")

    # ========================================================================
    # Load model
    # ========================================================================

    print(f"\nLoading model from {config.checkpoint_path}")
    model = LlamaForCausalLM.from_pretrained(config.checkpoint_path)
    model = model.to(config.device)

    # Convert to bfloat16 for memory efficiency
    model = model.bfloat16()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {total_params / 1e6:.1f}M parameters")

    # ========================================================================
    # Setup optimizer
    # ========================================================================

    # Use AdamW for all parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.adamw_lr,
        betas=config.adamw_betas,
        eps=config.adamw_eps,
        weight_decay=config.adamw_weight_decay,
    )

    print(f"\nOptimizer: AdamW")
    print(f"  Learning rate: {config.adamw_lr}")
    print(f"  Betas: {config.adamw_betas}")
    print(f"  Weight decay: {config.adamw_weight_decay}")
    print(f"  Epsilon: {config.adamw_eps}")

    # ========================================================================
    # Learning rate scheduler
    # ========================================================================

    # Calculate total training steps
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    if config.max_steps > 0:
        total_steps = config.max_steps
    else:
        total_steps = steps_per_epoch * config.num_epochs

    def get_lr_multiplier(step):
        """Get LR multiplier for warmup + cosine decay schedule."""
        if step < config.warmup_steps:
            # Linear warmup
            return step / config.warmup_steps
        elif config.use_cosine_schedule:
            # Cosine decay after warmup
            progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359))).item()
        else:
            # Constant LR after warmup
            return 1.0

    print(f"\nLearning rate schedule:")
    print(f"  Warmup steps: {config.warmup_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Cosine decay: {config.use_cosine_schedule}")
    print(f"  Initial LR: {config.adamw_lr}")

    # ========================================================================
    # Training loop
    # ========================================================================

    print("\n" + "=" * 80)
    print("Starting training")
    print("=" * 80)

    model.train()
    global_step = 0

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(config.device)
            labels = batch['labels'].to(config.device)

            # Forward pass
            loss, logits = model(input_ids=input_ids, labels=labels)

            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss at step {global_step}, batch {batch_idx}")
                print(f"  Logits: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
                print(f"  Labels: {labels[:5]}")
                print(f"  Skipping this batch...")
                continue

            # Backward pass
            loss.backward()

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Compute gradient norm before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

                # Gradient clipping
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Update learning rate with schedule
                lr_mult = get_lr_multiplier(global_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config.adamw_lr * lr_mult

                # Logging
                if global_step % config.log_interval == 0:
                    print(f"Step {global_step}: loss = {loss.item():.4f}, grad_norm = {grad_norm.item():.4f}")

                # Evaluation
                val_loss_for_log = ""
                if global_step % config.eval_interval == 0:
                    # Quick loss evaluation during training
                    val_loss = evaluate_loss_only(model, val_loader, config.device)
                    val_loss_for_log = f"{val_loss:.6f}"
                    print(f"Step {global_step}: val_loss = {val_loss:.4f}")

                # Write to log file (every step)
                with open(log_file, 'a') as f:
                    f.write(f"{global_step},{loss.item():.6f},{grad_norm.item():.6f},{val_loss_for_log}\n")

                # Full accuracy evaluation (slower, less frequent)
                if global_step % (config.eval_interval * 5) == 0:
                    if config.dataset_type == 'gsm8k':
                        results = evaluate_gsm8k(model, val_loader, tokenizer, config.device, max_samples=500)
                        print(f"Step {global_step}: GSM8K Accuracy = {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
                    elif config.dataset_type == 'siqa':
                        results = evaluate_siqa(model, val_loader, tokenizer, config.device, max_samples=500)
                        print(f"Step {global_step}: SIQA Accuracy = {results['accuracy']:.4f} ({results['correct']}/{results['total']})")

                # Check max steps
                if config.max_steps > 0 and global_step >= config.max_steps:
                    print(f"\nReached max_steps ({config.max_steps}). Stopping training.")
                    break

        # Save checkpoint at end of epoch
        epoch_checkpoint_path = f"{config.output_dir}/checkpoint_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), epoch_checkpoint_path)
        print(f"✓ Saved epoch {epoch + 1} checkpoint to {epoch_checkpoint_path}")

        if config.max_steps > 0 and global_step >= config.max_steps:
            break

    # Final evaluation
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    final_val_loss = evaluate_loss_only(model, val_loader, config.device)
    print(f"Final validation loss: {final_val_loss:.4f}")

    # Final accuracy evaluation
    if config.dataset_type == 'gsm8k':
        final_results = evaluate_gsm8k(model, val_loader, tokenizer, config.device, max_samples=500)
        print(f"Final GSM8K Accuracy: {final_results['accuracy']:.4f} ({final_results['correct']}/{final_results['total']})")
    elif config.dataset_type == 'siqa':
        final_results = evaluate_siqa(model, val_loader, tokenizer, config.device, max_samples=500)
        print(f"Final SIQA Accuracy: {final_results['accuracy']:.4f} ({final_results['correct']}/{final_results['total']})")

    # Save final model
    final_path = f"{config.output_dir}/final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"✓ Saved final model to {final_path}")


if __name__ == "__main__":
    main()
