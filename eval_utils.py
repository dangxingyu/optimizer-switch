"""
Evaluation utilities for different QA tasks
"""

import torch
import re
from typing import Dict, List, Tuple


def extract_answer_number(text: str) -> float:
    """
    Extract the final numerical answer from GSM8K format.

    GSM8K answers end with "#### <number>"
    Example: "Step 1... Step 2... #### 42"
    """
    # Try to find #### marker
    if "####" in text:
        answer_part = text.split("####")[-1].strip()
    else:
        answer_part = text.strip()

    # Remove commas from numbers
    answer_part = answer_part.replace(",", "")

    # Extract number (including decimals and negatives)
    numbers = re.findall(r'-?\d+\.?\d*', answer_part)

    if numbers:
        try:
            return float(numbers[0])
        except:
            return None
    return None


def extract_choice_letter(text: str) -> str:
    """
    Extract choice letter (A, B, or C) from SIQA format.

    SIQA answers end with "Answer: X) <text>"
    Example: "... Answer: A) go to the store"
    """
    # Try to find "Answer:" marker
    if "Answer:" in text:
        answer_part = text.split("Answer:")[-1].strip()
        # Extract first letter
        if answer_part and answer_part[0] in ['A', 'B', 'C', 'a', 'b', 'c']:
            return answer_part[0].upper()

    # Fallback: look for A), B), C) pattern
    match = re.search(r'\b([ABC])\)', text)
    if match:
        return match.group(1)

    return None


@torch.no_grad()
def evaluate_gsm8k(model, val_loader, tokenizer, device, max_new_tokens=256, max_samples=None):
    """
    Evaluate GSM8K by generating answers and comparing numbers.

    Args:
        model: The language model
        val_loader: Validation dataloader
        tokenizer: Tokenizer
        device: Device to run on
        max_new_tokens: Max tokens to generate per sample
        max_samples: Maximum number of samples to evaluate (None = all)

    Returns:
        dict with 'accuracy', 'correct', 'total', 'avg_loss'
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct = 0
    total = 0

    for batch in val_loader:
        # Stop if we've evaluated enough samples
        if max_samples is not None and total >= max_samples:
            break
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Compute loss
        loss, logits = model(input_ids=input_ids, labels=labels)
        num_tokens = (labels != -100).sum()
        total_loss += loss.item() * num_tokens.item()
        total_tokens += num_tokens.item()

        # Generate answers for accuracy calculation
        # Find where "Answer:" starts in the input
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            # Decode input to find the question part
            input_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)

            # Find where to start generation (after "Question: ... Answer:")
            if "Answer:" in input_text:
                question_part = input_text.split("Answer:")[0] + "Answer:"
                question_ids = tokenizer.encode(question_part, return_tensors='pt').to(device)

                # Generate answer
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=question_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Greedy decoding
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                # Extract predicted answer from #### marker (GSM8K raw format)
                # Generated text should contain "... #### <number>"
                pred_answer = extract_answer_number(generated_text)

                # Get ground truth from labels
                # Labels contain the full answer with "step by step ... #### <number>"
                label_ids = labels[i][labels[i] != -100]
                true_text = tokenizer.decode(label_ids, skip_special_tokens=True)
                true_answer = extract_answer_number(true_text)

                # Compare
                if pred_answer is not None and true_answer is not None:
                    if abs(pred_answer - true_answer) < 1e-3:  # Allow small floating point errors
                        correct += 1
                    total += 1

    avg_loss = total_loss / max(total_tokens, 1)
    accuracy = correct / max(total, 1)

    model.train()

    return {
        'avg_loss': avg_loss,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


@torch.no_grad()
def evaluate_siqa(model, val_loader, tokenizer, device, max_new_tokens=50, max_samples=None):
    """
    Evaluate SIQA by generating choice letters and comparing.

    Args:
        model: The language model
        val_loader: Validation dataloader
        tokenizer: Tokenizer
        device: Device to run on
        max_new_tokens: Max tokens to generate per sample
        max_samples: Maximum number of samples to evaluate (None = all)

    Returns:
        dict with 'accuracy', 'correct', 'total', 'avg_loss'
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct = 0
    total = 0

    for batch in val_loader:
        # Stop if we've evaluated enough samples
        if max_samples is not None and total >= max_samples:
            break
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Compute loss
        loss, logits = model(input_ids=input_ids, labels=labels)
        num_tokens = (labels != -100).sum()
        total_loss += loss.item() * num_tokens.item()
        total_tokens += num_tokens.item()

        # Generate answers for accuracy calculation
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            # Decode input to find the question part
            input_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)

            # Find where to start generation (after "... Answer:")
            if "Answer:" in input_text:
                question_part = input_text.split("Answer:")[0] + "Answer:"
                question_ids = tokenizer.encode(question_part, return_tensors='pt').to(device)

                # Generate answer
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=question_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                # Extract predicted choice
                pred_choice = extract_choice_letter(generated_text)

                # Get ground truth from labels
                label_ids = labels[i][labels[i] != -100]
                true_text = tokenizer.decode(label_ids, skip_special_tokens=True)
                true_choice = extract_choice_letter(true_text)

                # Compare
                if pred_choice is not None and true_choice is not None:
                    if pred_choice == true_choice:
                        correct += 1
                    total += 1

    avg_loss = total_loss / max(total_tokens, 1)
    accuracy = correct / max(total, 1)

    model.train()

    return {
        'avg_loss': avg_loss,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


@torch.no_grad()
def evaluate_loss_only(model, val_loader, device):
    """
    Simple loss-only evaluation (for quick checks during training).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        loss, _ = model(input_ids=input_ids, labels=labels)

        # Count non-padding tokens
        num_tokens = (labels != -100).sum()

        total_loss += loss.item() * num_tokens.item()
        total_tokens += num_tokens.item()

    avg_loss = total_loss / max(total_tokens, 1)
    model.train()
    return avg_loss
