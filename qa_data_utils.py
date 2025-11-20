"""
Data utilities for QA tasks (SIQA, GSM8K) with Llama + Muon training

This module provides data loading functions that format QA datasets correctly
for causal language modeling finetuning.
"""

from typing import Dict, List, Optional
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler


def format_siqa_example(example: Dict, add_eos: bool = True) -> str:
    """
    Format SIQA example as a text string for causal LM.

    SIQA format:
    - context: background situation
    - question: the question being asked
    - answerA, answerB, answerC: three answer choices
    - label: correct answer index (0, 1, or 2)

    Output format:
    Context: {context}
    Question: {question}
    A) {answerA}
    B) {answerB}
    C) {answerC}
    Answer: {correct_answer}<|end_of_text|>

    Args:
        example: Dict with SIQA fields
        add_eos: If True, append EOS token (default: True)
    """
    context = example.get('context', '')
    question = example['question']

    # Get answer choices (field names vary by dataset version)
    if 'answerA' in example:
        choices = [example['answerA'], example['answerB'], example['answerC']]
    else:
        choices = example['choices']

    # Get correct answer index
    if 'label' in example:
        label = int(example['label'])
    else:
        label = int(example['answerID'])

    # Format the text
    text = f"Context: {context}\nQuestion: {question}\n"
    text += f"A) {choices[0]}\n"
    text += f"B) {choices[1]}\n"
    text += f"C) {choices[2]}\n"

    # Map label to letter
    answer_letter = ['A', 'B', 'C'][label]
    text += f"Answer: {answer_letter}) {choices[label]}"

    # Add EOS marker for proper sequence termination
    if add_eos:
        text += "<|end_of_text|>"

    return text


def format_gsm8k_example(example: Dict, add_eos: bool = True) -> str:
    """
    Format GSM8K example as a text string for causal LM.

    GSM8K format:
    - question: math word problem
    - answer: step-by-step solution with final answer

    Output format:
    Question: {question}
    Answer: {answer}<|end_of_text|>

    Args:
        example: Dict with 'question' and 'answer' keys
        add_eos: If True, append EOS token (default: True)
    """
    question = example['question']
    answer = example['answer']

    text = f"Question: {question}\nAnswer: {answer}"

    # Add EOS marker for proper sequence termination
    # This teaches the model to generate EOS when the answer is complete
    if add_eos:
        text += "<|end_of_text|>"

    return text


def create_qa_dataloaders(
    config,
    rank: int,
    world_size: int,
    dataset_name: str,
    dataset_type: str = 'gsm8k',  # 'gsm8k' or 'siqa'
    dataset_config: Optional[str] = None,
):
    """
    Create training and validation dataloaders for QA tasks.

    Args:
        config: Training configuration object
        rank: Current process rank for DDP
        world_size: Total number of processes
        dataset_name: HuggingFace dataset name (e.g., 'openai/gsm8k' or 'allenai/social_i_qa')
        dataset_type: Type of QA dataset ('gsm8k' or 'siqa')
        dataset_config: Optional dataset configuration (e.g., 'main' for gsm8k)

    Returns:
        train_loader, val_loader
    """
    from transformers import AutoTokenizer

    # Load tokenizer from checkpoint
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        dataset = load_dataset(dataset_name)

    # Select formatting function
    if dataset_type.lower() == 'gsm8k':
        format_fn = format_gsm8k_example
    elif dataset_type.lower() == 'siqa':
        format_fn = format_siqa_example
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}. Use 'gsm8k' or 'siqa'.")

    def tokenize_function(examples):
        """Tokenize QA examples with proper label masking."""
        # Format each example
        if isinstance(examples['question'], list):
            # Batched processing
            texts = [format_fn(
                {k: examples[k][i] for k in examples.keys()}
            ) for i in range(len(examples['question']))]
        else:
            # Single example
            texts = [format_fn(examples)]

        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Create labels
        labels = tokenized['input_ids'].clone()

        # IMPORTANT: For QA finetuning, we only compute loss on the answer part
        # Find where "Answer:" starts and mask everything before it
        for i, input_ids in enumerate(tokenized['input_ids']):
            # Find "Answer:" token positions
            answer_str = "Answer:"
            answer_tokens = tokenizer.encode(answer_str, add_special_tokens=False)

            # Simple approach: find the last occurrence of answer_tokens[0]
            # (more robust implementations would search for the full sequence)
            input_ids_list = input_ids.tolist()

            # Mask everything except the answer part
            # For now, we'll compute loss on everything (simpler for causal LM)
            # If you want to mask the question, uncomment below:
            # try:
            #     answer_start = input_ids_list.index(answer_tokens[0])
            #     labels[i, :answer_start] = -100
            # except ValueError:
            #     pass  # Answer token not found, use full sequence

            # Mask padding tokens
            labels[i][tokenized['attention_mask'][i] == 0] = -100

        tokenized['labels'] = labels
        return tokenized

    # Tokenize datasets
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
        # Use a small portion of train as validation
        tokenized_val = tokenized_train.select(range(min(1000, len(tokenized_train))))

    # Set format for PyTorch
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Create distributed samplers
    train_sampler = DistributedSampler(
        tokenized_train,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    val_sampler = DistributedSampler(
        tokenized_val,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    # Create dataloaders
    train_loader = DataLoader(
        tokenized_train,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        tokenized_val,
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader


# Example usage:
if __name__ == "__main__":
    """Test the data loading functions."""
    from dataclasses import dataclass

    @dataclass
    class DummyConfig:
        checkpoint_path: str = "../checkpoints/adamw_130m_1"
        max_seq_length: int = 512
        batch_size: int = 4

    config = DummyConfig()

    print("=" * 80)
    print("Testing GSM8K data loading")
    print("=" * 80)

    # Test GSM8K
    try:
        train_loader, val_loader = create_qa_dataloaders(
            config=config,
            rank=0,
            world_size=1,
            dataset_name='openai/gsm8k',
            dataset_type='gsm8k',
            dataset_config='main',
        )

        print(f"✓ GSM8K loaded successfully")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")

        # Print first example
        batch = next(iter(train_loader))
        print(f"\nFirst batch shapes:")
        print(f"  - input_ids: {batch['input_ids'].shape}")
        print(f"  - labels: {batch['labels'].shape}")

    except Exception as e:
        print(f"✗ GSM8K failed: {e}")

    print("\n" + "=" * 80)
    print("Testing SIQA data loading")
    print("=" * 80)

    # Test SIQA
    try:
        train_loader, val_loader = create_qa_dataloaders(
            config=config,
            rank=0,
            world_size=1,
            dataset_name='allenai/social_i_qa',
            dataset_type='siqa',
        )

        print(f"✓ SIQA loaded successfully")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")

        # Print first example
        batch = next(iter(train_loader))
        print(f"\nFirst batch shapes:")
        print(f"  - input_ids: {batch['input_ids'].shape}")
        print(f"  - labels: {batch['labels'].shape}")

    except Exception as e:
        print(f"✗ SIQA failed: {e}")
