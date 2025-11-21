# -----------------------------------------------------------------------------
# Triton kernel for symmetric matrix multiplication by @byronxu99

import torch
try:
    import torch.distributed as dist
except ImportError:
    dist = None
import triton
import triton.language as tl

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
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]

@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def XXT_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def XXT(A: torch.Tensor, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    XXT_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
    )
    return out

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ba_plus_cAA_kernel(
    A_ptr, C_ptr,
    M,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    # This is mostly duplicated from XXT_kernel, but also loads and adds a block of A
    # Performance is slightly slower than XXT_kernel, so we use two separate kernels
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K, "Input matrix must be square"
    assert out.size(-2) == M
    assert out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ba_plus_cAA_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
    )
    return out

# Computed for num_iters=5, safety_factor=2e-2, cushion=2
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

@torch.compile(dynamic=False, fullgraph=True) # Must use dynamic=False or else it's much slower
def polar_express(G: torch.Tensor):
    """
    Polar Express Sign Method: https://arxiv.org/pdf/2505.16932
    by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.
    Code adapted from https://github.com/NoahAmsel/PolarExpress/tree/main by @varunneal.
    """
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    # Allocate buffers
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Perform the iterations
    for a, b, c in polar_express_coeffs:
        XXT(X, out=A)  # A = X @ X.mT
        ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
        aX_plus_BX(X, B, X, beta=a, out=C)  # C = a * X + B @ X
        X, C = C, X  # Swap references to avoid unnecessary copies

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

# -----------------------------------------------------------------------------
# Muon optimizer

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU. 
    Note: A later PR replaced Newton-Shulz with Polar Express for the orthogonalization step

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    Though empirically small 1D params perform efficiently here:
        NS approximately performs a magnitude normalization of the grad
        This hyper-optimized class has faster execution time than the current impl of Adam for small params

    Custom distributed sizing:
    The model stores all attn and mlp weights in the same shape, and then updates the view as 
    needed on the forward pass. This enables attn and mlp weights to be contained within the same 
    dist.reduce_scatter_tensor() call. The model architecture has been customized to enable 
    (n_attn_layers+n_mlp_layers*2)%8==0 for batching across 8 GPUs with zero padding on mlp and attn. 
    The scheduling is:
        1. reduce scatter smear_gate (1 param 7 padding params)
        2. reduce scatter attn_gate (10 params 6 padding params)
        3. reduce scatter attn/mlp round 1 (10 attn params 6 mlp params)
        4. reduce scatter attn/mlp round 2 (16 mlp params)
        5. wait on step 1, then compute update of 1 and schedule all gather
        6. wait on step 2, then compute update of 2 and schedule all gather
        7. wait on step 3, then compute update of 3 and schedule all gather
            GPUs receive [2 ATTN, 2 ATTN, 2 ATTN, 2 ATTN, 2 ATTN, 2 MLP, 2 MLP, 2 MLP]
            GPUs that receive params of type attn reshape before computing update
        8. wait on 4, then compute update of 4 and schedule all gather
        9. wait for each all gather to complete and update params
    Empirically, leading with small params provides an additional 0.2s improvement.
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, custom_sizing=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        # custom sizing requires 8 GPUs and distributed to be initialized
        if custom_sizing and dist and dist.is_initialized() and dist.get_world_size()==8:
            param_groups = self.generate_custom_param_groups(params)
        else:
            param_groups = self.generate_standard_param_groups(params)
        super().__init__(param_groups, defaults)

    def generate_standard_param_groups(self, params):
        """
        Use this method if running on less than 8 GPU or experimenting with additional attn or mlp modules.
        Creates one param group per size, while giving attn its own param group for resize op.
        """
        params = list(params)
        param_groups = []
        attn_subset = [p for p in params if getattr(p, 'label', None) == 'attn']
        non_attn_subset = [p for p in params if getattr(p, 'label', None) != 'attn']
        if attn_subset:
            param_groups.append(dict(params=attn_subset))

        sizes = {p.shape for p in non_attn_subset}
        for size in sizes:
            group_params = [p for p in non_attn_subset if p.shape == size]
            param_groups.append(dict(params=group_params))
        return param_groups
    
    def generate_custom_param_groups(self, params):
        """
        Implementation requires that a single GPU does not receive both attn 
        and mlp params when a param group is split across GPUs.
        """
        label_ranks = {
            'smear_gate': 1, # 1 param
            'attn_gate': 2, # 10 params
            'attn': 3, # 10 params
            'mlp': 4, # 22 params
        }
        params = list(params)
        params.sort(key=lambda x: label_ranks.get(x.label))
        idx = 0
        group_sizes = [1,10,16,16]
        assert len(params)==sum(group_sizes)
        param_groups = []
        for size in group_sizes:
            group_params = params[idx:idx+size]
            param_groups.append(dict(params=group_params))
            idx += size
        return param_groups

    @torch.no_grad()
    def step_single_gpu(self):
        """
        Simple single-GPU implementation without distributed operations.
        This is used when torch.distributed is not initialized.
        """
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue

            # Determine effective LR and WD once per group
            p_example = params[0]
            # Latest Muon (2025-06-15) uses: scale = max(1, d_out/d_in)**0.5
            # This provides more balanced scaling across different layer shapes
            scale = max(1, p_example.size(-2) / p_example.size(-1)) ** 0.5
            eff_lr_val = (
                group["lr"]
                * scale
                * getattr(p_example, "lr_mul", 1.0)
            )
            eff_weight_decay_val = (
                group["lr"]
                * group["weight_decay"]
                * getattr(p_example, "wd_mul", 1.0)
            )

            # Collect update_grads for batched processing
            update_grads_list = []
            params_list = []

            for param in params:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                # Initialize momentum buffer if not present
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                momentum_buffer = state["momentum_buffer"]

                # Apply momentum update: buf = buf * momentum + grad
                # Note: This is standard SGD momentum, NOT lerp!
                momentum_buffer.mul_(group["momentum"]).add_(grad)

                # For Nesterov-style momentum, use: update = grad + buf * momentum
                # For standard momentum, use: update = buf
                # Here we use Nesterov-style which is recommended for Muon
                update_grad = grad + momentum_buffer * group["momentum"]
                update_grads_list.append(update_grad)
                params_list.append(param)

            if not update_grads_list:
                continue

            # Stack for batched polar_express
            batched_update_grads = torch.stack(update_grads_list)

            # Handle attn reshaping
            if getattr(params_list[0], 'label', None) == 'attn':
                original_shape = batched_update_grads.shape
                batch = 4 * original_shape[0]
                d1 = original_shape[1]
                d2 = original_shape[2] // 4
                batched = batched_update_grads.view(batch, d1, d2)
                v_batch = polar_express(batched)
                v_batch = v_batch.view(original_shape)
            else:
                v_batch = polar_express(batched_update_grads)

            # Apply updates to parameters
            for i, param in enumerate(params_list):
                # Apply weight decay
                param.mul_(1 - eff_weight_decay_val)
                # Apply zeropower update
                param.add_(v_batch[i], alpha=-eff_lr_val)

    @torch.no_grad()
    def step(self):
        # Check if distributed is initialized
        if not dist or not dist.is_initialized():
            return self.step_single_gpu()

        # Efficient systems-wise implementation of step developed by @YouJiacheng,
        # @KonstantinWilleke, @alexrgilbert, @adricarda, @tuttyfrutyee, @vdlad,
        # @ryanyang0, and @vagrawal.
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        group_infos = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            if not params:
                continue

            num_params = len(params)
            padded_num_params = (
                (num_params + world_size - 1) // world_size * world_size
            )

            grads_to_stack = [p.grad for p in params]
            if padded_num_params > num_params:
                padding_grad = torch.zeros_like(params[0].grad)
                grads_to_stack.extend(
                    [padding_grad] * (padded_num_params - num_params)
                )

            stacked_grads = torch.stack(grads_to_stack)

            chunk_size = padded_num_params // world_size
            grad_chunk = torch.empty(
                (chunk_size, *params[0].grad.shape),
                dtype=stacked_grads.dtype,
                device=stacked_grads.device,
            )

            reduce_future = dist.reduce_scatter_tensor(
                grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
            ).get_future()

            group_infos.append(
                {
                    "params": params,
                    "grad_chunk": grad_chunk,
                    "reduce_future": reduce_future,
                    "chunk_size": chunk_size,
                    "padded_num_params": padded_num_params,
                }
            )

        all_gather_infos = []
        # Second pass: wait for gradients, compute updates for the local shard of parameters,
        # and launch all async all_gather operations.
        for group, info in zip(self.param_groups, group_infos):
            info["reduce_future"].wait()

            params = info["params"]
            grad_chunk = info["grad_chunk"]
            chunk_size = info["chunk_size"]
            start_idx = rank * chunk_size

            # Determine effective LR and WD once per group, assuming constant for same-shaped params.
            # This helps in vectorizing operations later.
            p_example = params[0]  # All params in a group have the same shape.
            # Latest Muon (2025-06-15) uses: scale = max(1, d_out/d_in)**0.5
            # This provides more balanced scaling across different layer shapes
            scale = max(1, p_example.size(-2) / p_example.size(-1)) ** 0.5
            eff_lr_val = (
                group["lr"]
                * scale
                * getattr(p_example, "lr_mul", 1.0)
            )
            eff_weight_decay_val = (
                group["lr"]
                * group["weight_decay"]
                * getattr(p_example, "wd_mul", 1.0)
            )

            # Prepare a contiguous buffer for the updated parameters for this rank's chunk.
            # This buffer will serve as the input_tensor for dist.all_gather_into_tensor.
            updated_param_chunk = torch.empty(
                (chunk_size, *p_example.shape),
                dtype=p_example.dtype,
                device=p_example.device,
            )

            # List to collect update_grad tensors for batched zeropower computation.
            update_grads_for_zeropower = []

            # Process each parameter in this rank's chunk.
            for i in range(chunk_size):
                param_idx = start_idx + i

                if param_idx >= len(params):
                    # For padding: Fill the corresponding part of the updated_param_chunk with zeros.
                    # These padded entries will not be used by other ranks in the all_gather, but
                    # initializing them prevents uninitialized memory access issues.
                    updated_param_chunk[i].zero_()
                    # Also append a zero tensor for zeropower input if it must be padded.
                    update_grads_for_zeropower.append(
                        torch.zeros_like(p_example.grad)
                    )
                    continue
                param = params[param_idx]
                grad = grad_chunk[
                    i
                ]  # This gradient corresponds to the current parameter param.
                state = self.state[param]

                # Initialize momentum buffer if not present
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                momentum_buffer = state["momentum_buffer"]

                # Apply momentum update: buf = buf * momentum + grad
                # Note: This is standard SGD momentum, NOT lerp!
                momentum_buffer.mul_(group["momentum"]).add_(grad)

                # For Nesterov-style momentum, use: update = grad + buf * momentum
                # For standard momentum, use: update = buf
                # Here we use Nesterov-style which is recommended for Muon
                update_grad = grad + momentum_buffer * group["momentum"]
                update_grads_for_zeropower.append(update_grad)

                # Copy the current parameter value into the temporary buffer.
                updated_param_chunk[i].copy_(param)

                # Apply weight decay directly to the buffer.
                updated_param_chunk[i].mul_(1 - eff_weight_decay_val)

            # Stack the individual `update_grad` tensors for efficient batched zeropower computation.
            batched_update_grads = torch.stack(update_grads_for_zeropower)

            # Compute zeropower for the entire chunk in a single, batched call.
            original_shape = batched_update_grads.shape
            # Reshape attn params from [hdim, dim*4] to [4,hdim,dim] to apply polar_express independently to Q,K,V,O
            param_idx = start_idx if start_idx < len(params) else 0
            if getattr(params[param_idx], 'label', None) == 'attn':
                for p in params[param_idx:param_idx+chunk_size]:
                    assert getattr(params[param_idx], 'label', None)=='attn', "GPU cannot mix attn and mlp params"
                batch = 4 * original_shape[0]
                d1 = original_shape[1]
                d2 = original_shape[2] // 4
                batched = batched_update_grads.view(batch, d1, d2)
                v_chunk = polar_express(batched)
                v_chunk = v_chunk.view(original_shape)
            else:
                v_chunk = polar_express(batched_update_grads)

            # Add the computed zeropower update to the parameters in the buffer.
            # This loop applies the zeropower output (v_chunk) to the `updated_param_chunk` buffer.
            for i in range(chunk_size):
                param_idx = start_idx + i
                if param_idx >= len(params):  # Skip padded entries again.
                    continue

                # Add the computed zeropower update to the parameter in the buffer.
                updated_param_chunk[i].add_(v_chunk[i], alpha=-eff_lr_val)

            stacked_params = torch.empty(
                (info["padded_num_params"], *params[0].shape),
                dtype=params[0].dtype,
                device=params[0].device,
            )
            gather_future = dist.all_gather_into_tensor(
                stacked_params, updated_param_chunk, async_op=True
            ).get_future()

            all_gather_infos.append(
                {
                    "gather_future": gather_future,
                    "stacked_params": stacked_params,
                    "orig_params": params,
                }
            )

        # Final pass: wait for all_gather to complete and copy results back into original parameter tensors.
        for info in all_gather_infos:
            info["gather_future"].wait()
            stacked_params = info["stacked_params"]
            orig_params = info["orig_params"]

            unstacked_params = torch.unbind(stacked_params)
            for i, p in enumerate(orig_params):
                p.copy_(unstacked_params[i], non_blocking=True)

# Muon optimizer extracted
