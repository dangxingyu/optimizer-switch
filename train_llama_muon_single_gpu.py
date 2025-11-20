"""
Train Llama models with Muon optimizer - Single GPU version
Simplified version for small datasets and models

Usage: python train_llama_muon_single_gpu.py
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

# Import Triton for kernels
import triton
import triton.language as tl

# ============================================================================
# Muon Optimizer and Triton Kernels
# Copied from train_gpt_muon_only.py (lines 116-676)
# ============================================================================

# -----------------------------------------------------------------------------
# Polar Express coefficients and kernel
# From: https://arxiv.org/pdf/2505.16932

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

# -----------------------------------------------------------------------------
# Triton kernel for symmetric matrix multiplication by @byronxu99

def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=1,
            num_warps=2,
        )
        for bm in [64, 128]
        for bn in [64, 128]
        for bk in [32, 64]
    ] + [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 0,
            },
            num_stages=1,
            num_warps=2,
        )
        for bm in [64, 128]
        for bn in [64, 128]
        for bk in [32, 64]
    ]


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def XXT_kernel(
    A_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    """Compute C = A @ A.T"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if LOWER_UPPER == 1:
        num_pid_in_group = GROUP_SIZE_M * (GROUP_SIZE_M + 1) // 2
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m_offset = pid % num_pid_in_group
        i = tl.sqrt(8 * pid_m_offset.to(tl.float32) + 1).to(tl.int32) - 1
        i = i // 2
        pid_m = first_pid_m + i
        pid_n = first_pid_m + (pid_m_offset - (i * (i + 1) // 2))
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % GROUP_SIZE_M)
        pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = A_ptr + (offs_bn[:, None] * stride_am + offs_k[None, :] * stride_ak)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, tl.trans(b), accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_ak

    c = accumulator.to(C_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def compute_XXT(A: Tensor) -> Tensor:
    """Compute A @ A.T using Triton kernel"""
    assert A.is_contiguous()
    M, K = A.shape
    C = torch.empty((M, M), device=A.device, dtype=A.dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(M, META["BLOCK_SIZE_N"]),
    )
    XXT_kernel[grid](
        A, C, M, M, K,
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def ba_plus_cAA_kernel(
    A_ptr,
    C_ptr,
    beta,
    alpha,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    """Compute C = beta * A + alpha * A @ A.T"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if LOWER_UPPER == 1:
        num_pid_in_group = GROUP_SIZE_M * (GROUP_SIZE_M + 1) // 2
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m_offset = pid % num_pid_in_group
        i = tl.sqrt(8 * pid_m_offset.to(tl.float32) + 1).to(tl.int32) - 1
        i = i // 2
        pid_m = first_pid_m + i
        pid_n = first_pid_m + (pid_m_offset - (i * (i + 1) // 2))
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % GROUP_SIZE_M)
        pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = A_ptr + (offs_bn[:, None] * stride_am + offs_k[None, :] * stride_ak)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, tl.trans(b), accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_ak

    # Add beta * A contribution
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if beta != 0:
        c_ptrs_read = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        old_c = tl.load(c_ptrs_read, mask=c_mask, other=0.0)
        result = beta * old_c + alpha * accumulator
    else:
        result = alpha * accumulator

    c = result.to(C_ptr.dtype.element_ty)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


def ba_plus_cAA(A: Tensor, beta: float, alpha: float) -> Tensor:
    """Compute beta * A + alpha * A @ A.T"""
    assert A.is_contiguous()
    M, K = A.shape
    C = A.clone() if beta != 0 else torch.empty((M, M), device=A.device, dtype=A.dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(M, META["BLOCK_SIZE_N"]),
    )
    ba_plus_cAA_kernel[grid](
        A, C, beta, alpha, M, M, K,
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


def polar_express(G: torch.Tensor):
    """
    Polar Express Sign Method: https://arxiv.org/pdf/2505.16932
    by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.
    Code adapted from https://github.com/NoahAmsel/PolarExpress/tree/main by @varunneal.
    """
    X = G.bfloat16() if G.dtype != torch.bfloat16 else G.clone()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    # Allocate buffers
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    # Perform the iterations
    for a, b, c in polar_express_coeffs:
        # A = X @ X.T
        if X.ndim == 2:
            torch.mm(X, X.mT, out=A)
        else:
            torch.bmm(X, X.mT, out=A)

        # B = b * A + c * A @ A
        if X.ndim == 2:
            torch.addmm(A, A, A, beta=b, alpha=c, out=B)
        else:
            torch.baddbmm(A, A, A, beta=b, alpha=c, out=B)

        # C = a * X + B @ X
        if X.ndim == 2:
            torch.addmm(X, B, X, beta=a, alpha=1.0, out=C)
        else:
            torch.baddbmm(X, B, X, beta=a, alpha=1.0, out=C)

        X, C = C, X  # Swap references

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    NorMuon - Normalized Muon with adaptive step size

    Matches train_gpt.py implementation exactly (lines 551-613).

    Features:
    - Lerp momentum (train_gpt.py lines 555-556)
    - LR scaling by max(1, d_out/d_in)**0.5 (line 575)
    - Second momentum for variance tracking (lines 567-602)
    - Adaptive step size normalization
    - Cautious weight decay (lines 609-611)
    """

    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.01,
        momentum=0.95,
        beta2=0.95,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            beta2=beta2,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                param_shape = p.shape

                # Initialize momentum buffer (train_gpt.py line 551-553)
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                momentum_buffer = state["momentum_buffer"]

                # Apply momentum update (train_gpt.py lines 555-556)
                momentum_buffer.lerp_(grad, 1 - group["momentum"])
                updated_grads = grad.lerp(momentum_buffer, group["momentum"])

                # Initialize param_lr and param_wd (train_gpt.py lines 573-583)
                if "param_lr" not in state:
                    if p.ndim >= 2:
                        scale = max(1., param_shape[-2] / param_shape[-1]) ** 0.5
                    else:
                        scale = 1.0
                    state["param_lr"] = scale * getattr(p, "lr_mul", 1.0)
                    state["param_wd"] = getattr(p, "wd_mul", 1.0)

                # Determine effective LR and WD (train_gpt.py lines 586-587)
                eff_lr = group["lr"] * state["param_lr"]
                eff_wd = group["lr"] * group["weight_decay"] * state["param_wd"]

                # Apply polar_express for 2D params (train_gpt.py line 593)
                if p.ndim >= 2:
                    v = polar_express(updated_grads)
                else:
                    v = updated_grads

                # NorMuon: adaptive step size with second momentum (train_gpt.py lines 567-602)
                if p.ndim >= 2:
                    # Initialize second_momentum_buffer
                    if "second_momentum_buffer" not in state:
                        if param_shape[-2] >= param_shape[-1]:
                            state["second_momentum_buffer"] = torch.zeros_like(v[..., :, :1])
                        else:
                            state["second_momentum_buffer"] = torch.zeros_like(v[..., :1, :])
                    second_momentum_buffer = state["second_momentum_buffer"]

                    # Track variance along one dimension (line 596-598)
                    v_norm = v.norm(dim=(-2, -1), keepdim=True)
                    if param_shape[-2] >= param_shape[-1]:
                        v_mean = v.square().mean(dim=-1, keepdim=True)
                    else:
                        v_mean = v.square().mean(dim=-2, keepdim=True)

                    # Update second momentum with lerp (line 598)
                    second_momentum_buffer.lerp_(v_mean.to(dtype=p.dtype), 1 - group["beta2"])

                    # Compute adaptive step size (line 599-600)
                    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
                    v.mul_(step_size)

                    # Renormalize to preserve magnitude (lines 601-602)
                    v_norm_new = v.norm(dim=(-2, -1), keepdim=True)
                    v.mul_(v_norm / v_norm_new.clamp_min_(1e-10))

                # Cautious weight decay: only apply when aligned (train_gpt.py lines 609-611)
                if eff_wd > 0:
                    mask = (v * p) >= 0
                    v.addcmul_(p, (eff_wd * mask).to(p.dtype))

                # Apply parameter update (train_gpt.py line 613)
                p.addcmul_(v, value=-eff_lr)

        return loss


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
    max_train_samples: int = -1  # -1 for all data, otherwise limit training samples

    # Muon optimizer (for 2D weights)
    muon_lr: float = 0.002  # Reduced from 0.02 for stability
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.0

    # AdamW optimizer (for 1D params, embeddings)
    adamw_lr: float = 3e-5  # Reduced from 1e-4 for stability
    adamw_betas: tuple = (0.9, 0.98)
    adamw_weight_decay: float = 0.01

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Learning rate schedule
    warmup_steps: int = 50
    use_cosine_schedule: bool = True

    # Logging
    log_interval: int = 1  # Log every step
    eval_interval: int = 50
    save_interval: int = 500
    output_dir: str = "output"

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
    parser.add_argument('--max_train_samples', type=int, default=None)
    args = parser.parse_args()

    config = TrainingConfig()

    # Override with command line arguments
    if args.checkpoint_path:
        config.checkpoint_path = args.checkpoint_path
    if args.lr:
        config.muon_lr = args.lr
        config.adamw_lr = args.lr / 10.0  # AdamW uses 1/10 of Muon LR
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.max_train_samples:
        config.max_train_samples = args.max_train_samples

    print("=" * 80)
    print(f"Training Llama with Muon optimizer - Single GPU")
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Triton version: {triton.__version__}")
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

    # Limit training samples if specified
    if config.max_train_samples > 0 and config.max_train_samples < len(tokenized_train):
        tokenized_train = tokenized_train.select(range(config.max_train_samples))
        print(f"  Limiting training to {config.max_train_samples} samples")

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
    # Setup optimizers
    # ========================================================================

    # Separate parameters for Muon (2D) and AdamW (1D, embeddings)
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim >= 2 and 'weight' in name:
            if 'embed' not in name.lower() and 'lm_head' not in name.lower():
                muon_params.append(param)
            else:
                adamw_params.append(param)
        else:
            adamw_params.append(param)

    print(f"\nParameter groups:")
    print(f"  Muon: {sum(p.numel() for p in muon_params) / 1e6:.1f}M params ({len(muon_params)} tensors)")
    print(f"  AdamW: {sum(p.numel() for p in adamw_params) / 1e6:.1f}M params ({len(adamw_params)} tensors)")

    muon_optimizer = Muon(
        muon_params,
        lr=config.muon_lr,
        momentum=config.muon_momentum,
        weight_decay=config.muon_weight_decay,
        beta2=0.95,  # NorMuon second momentum (for variance tracking)
    )

    adamw_optimizer = torch.optim.AdamW(
        adamw_params,
        lr=config.adamw_lr,
        betas=config.adamw_betas,
        weight_decay=config.adamw_weight_decay,
    )

    # ========================================================================
    # Learning rate schedulers
    # ========================================================================

    # Calculate total training steps
    total_steps = len(train_loader) * config.num_epochs
    if config.max_steps > 0:
        total_steps = config.max_steps

    def get_lr_multiplier(step):
        """Get learning rate multiplier for warmup + cosine decay."""
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
    print(f"  Initial Muon LR: {config.muon_lr}")
    print(f"  Initial AdamW LR: {config.adamw_lr}")

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

                muon_optimizer.step()
                adamw_optimizer.step()
                muon_optimizer.zero_grad()
                adamw_optimizer.zero_grad()
                global_step += 1

                # Update learning rates with schedule
                lr_mult = get_lr_multiplier(global_step)
                for param_group in muon_optimizer.param_groups:
                    param_group['lr'] = config.muon_lr * lr_mult
                for param_group in adamw_optimizer.param_groups:
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
