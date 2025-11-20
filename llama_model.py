"""
Custom Llama implementation with modded-nanogpt optimizations
Combines transformers' Llama architecture with:
- Flash Attention 3 from kernels library
- Optional FP8 linear layers
- Compatible with Muon optimizer

Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from kernels import get_kernel
    flash_attn_interface = get_kernel('varunneal/flash-attention-3').flash_attn_interface
    HAS_FLASH_ATTN_3 = True
except:
    HAS_FLASH_ATTN_3 = False
    print("Warning: Flash Attention 3 not available, using standard attention")


@dataclass
class LlamaConfig:
    """Llama model configuration"""
    vocab_size: int = 128256
    hidden_size: int = 1536
    intermediate_size: int = 6144
    num_hidden_layers: int = 32
    num_attention_heads: int = 24
    num_key_value_heads: int = 24  # For GQA, set to < num_attention_heads
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False

    # Performance optimizations
    use_fp8: bool = False
    use_flash_attn_3: bool = True


class LlamaRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Reference: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    Reference: https://arxiv.org/abs/2104.09864
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper with RoPE.
    Optionally uses Flash Attention 3 for efficiency.
    """
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Q, K, V projections (separate, unlike GPT's merged qkvo)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # RoPE
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # Label for Muon optimizer
        self.q_proj.weight.label = 'attn'
        self.k_proj.weight.label = 'attn'
        self.v_proj.weight.label = 'attn'
        self.o_proj.weight.label = 'attn'

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_flash_attn_3: bool = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [bsz, num_heads, seq_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if position_ids is None:
            position_ids = torch.arange(kv_seq_len - q_len, kv_seq_len, device=hidden_states.device).unsqueeze(0)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Concatenate with past key/value if exists
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # Grouped-query attention: repeat k/v heads if necessary
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Use Flash Attention 3 if available
        use_flash = use_flash_attn_3 if use_flash_attn_3 is not None else (self.config.use_flash_attn_3 and HAS_FLASH_ATTN_3)

        if use_flash and self.training:
            # Flash Attention 3 expects [bsz, seq_len, num_heads, head_dim]
            q = query_states.transpose(1, 2).contiguous()
            k = key_states.transpose(1, 2).contiguous()
            v = value_states.transpose(1, 2).contiguous()

            # Flatten batch dimension for varlen interface
            # For simplicity, assume no padding (adjust cu_seqlens for real batching)
            q_flat = q.view(-1, self.num_heads, self.head_dim)
            k_flat = k.view(-1, self.num_heads, self.head_dim)
            v_flat = v.view(-1, self.num_heads, self.head_dim)

            cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=q.device)

            attn_output = flash_attn_interface.flash_attn_varlen_func(
                q_flat, k_flat, v_flat,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=q_len,
                max_seqlen_k=q_len,
                causal=True,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
            )

            attn_output = attn_output.view(bsz, q_len, self.num_heads, self.head_dim)
        else:
            # Standard attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            # Causal mask (only for prefill or when kv_seq_len == q_len)
            if self.is_causal and kv_seq_len == q_len:
                causal_mask = torch.triu(torch.ones(q_len, q_len, device=query_states.device, dtype=torch.bool), diagonal=1)
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            # Transpose to [bsz, seq_len, num_heads, head_dim]
            attn_output = attn_output.transpose(1, 2).contiguous()

        # Reshape and project output
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # Save key/value for next iteration if use_cache
        past_key_value_out = None
        if use_cache:
            # Save the ungrouped key/value states (before repeat_interleave)
            if self.num_key_value_groups > 1:
                # Get the original key/value states before repeat
                key_cache = key_states[:, ::self.num_key_value_groups, :, :]
                value_cache = value_states[:, ::self.num_key_value_groups, :, :]
            else:
                key_cache = key_states
                value_cache = value_states
            past_key_value_out = (key_cache, value_cache)

        return attn_output, past_key_value_out


class LlamaMLP(nn.Module):
    """
    Llama MLP with SwiGLU activation
    Reference: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

        # Activation function
        if config.hidden_act == "silu":
            self.act_fn = F.silu
        elif config.hidden_act == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {config.hidden_act}")

        # Label for Muon optimizer
        self.gate_proj.weight.label = 'mlp'
        self.up_proj.weight.label = 'mlp'
        self.down_proj.weight.label = 'mlp'

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU: (silu(gate) * up) @ down
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        down = self.down_proj(gate * up)
        return down


class LlamaDecoderLayer(nn.Module):
    """Single Llama decoder layer with pre-normalization"""
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        residual = hidden_states

        # Self Attention with pre-norm
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value_out = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value_out


class LlamaModel(nn.Module):
    """
    Transformer decoder consisting of multiple LlamaDecoderLayer layers.
    Compatible with modded-nanogpt's Muon optimizer and training infrastructure.
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = None
        self.vocab_size = config.vocab_size

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Decoder layers
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Final norm
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tuple[Tensor, Tensor], ...]]]:
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Prepare past_key_values if needed
        if past_key_values is None and use_cache:
            past_key_values = tuple([None] * len(self.layers))

        # Pass through decoder layers
        next_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            hidden_states, past_key_value_out = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

            if use_cache:
                next_cache += (past_key_value_out,)

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states, next_cache


class LlamaForCausalLM(nn.Module):
    """
    Llama model with a language modeling head.
    Compatible with modded-nanogpt's training loop.
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)

        # LM head (not tied with embeddings by default)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following Llama conventions"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
        labels: Optional[Tensor] = None,
        use_cache: bool = False,
    ):
        # Get hidden states from model
        hidden_states, past_key_values_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # Compute logits
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )

        # Return format depends on whether we're using cache and/or labels
        if use_cache:
            return (loss, logits, past_key_values_out) if loss is not None else (logits, past_key_values_out)
        else:
            return (loss, logits) if loss is not None else logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Keep tokens with cumulative probability >= top_p (nucleus sampling)
            pad_token_id: Token ID for padding (optional)
            eos_token_id: Token ID for end of sequence (optional)

        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Start with input sequence
        generated = input_ids.clone()

        # KV cache for faster generation
        past_key_values = None

        # First forward pass with full input (prefill)
        outputs = self(input_ids=input_ids, use_cache=True)
        logits, past_key_values = outputs

        # Get logits for the last position
        next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

        # Apply temperature
        if do_sample and temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Apply top-k filtering
        if do_sample and top_k is not None:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

        # Apply top-p (nucleus) filtering
        if do_sample and top_p is not None:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')

        # Sample or take argmax
        if do_sample:
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Append first generated token
        generated = torch.cat([generated, next_token], dim=-1)

        # Check for EOS
        if eos_token_id is not None and (next_token == eos_token_id).all():
            return generated

        # Continue generation using cache (decode phase)
        for _ in range(max_new_tokens - 1):
            # Only pass the last token with cache
            outputs = self(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            logits, past_key_values = outputs

            # Get logits for the last (only) position
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if do_sample and temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if do_sample and top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if do_sample and top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample or take argmax
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config: Optional[LlamaConfig] = None):
        """
        Load model from Hugging Face checkpoint.
        Converts weight names from transformers format to our custom format.
        """
        from transformers import AutoConfig, AutoModelForCausalLM as HFLlamaForCausalLM
        import os

        # Load config from checkpoint if not provided
        if config is None:
            hf_config = AutoConfig.from_pretrained(checkpoint_path)
            config = LlamaConfig(
                vocab_size=hf_config.vocab_size,
                hidden_size=hf_config.hidden_size,
                intermediate_size=hf_config.intermediate_size,
                num_hidden_layers=hf_config.num_hidden_layers,
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads),
                max_position_embeddings=hf_config.max_position_embeddings,
                rope_theta=getattr(hf_config, 'rope_theta', 10000.0),
                rms_norm_eps=hf_config.rms_norm_eps,
            )

        # Create our model
        model = cls(config)

        # Load weights from HF checkpoint
        hf_model = HFLlamaForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
        hf_state_dict = hf_model.state_dict()

        # Map HF keys to our keys
        our_state_dict = model.state_dict()

        for our_key in our_state_dict.keys():
            # Convert our key to HF key
            # Our format: model.embed_tokens.weight
            # HF format: model.embed_tokens.weight (same for most keys)

            # Simply prepend 'model.' if not already there
            if our_key.startswith('model.'):
                hf_key = our_key
            else:
                hf_key = 'model.' + our_key

            # Handle lm_head specially (it's not under model. in HF)
            if 'lm_head' in our_key:
                hf_key = our_key

            if hf_key in hf_state_dict:
                our_state_dict[our_key] = hf_state_dict[hf_key]
            else:
                print(f"Warning: Key {our_key} not found in HF checkpoint (looked for {hf_key})")

        model.load_state_dict(our_state_dict, strict=False)
        print(f"Loaded model from {checkpoint_path}")

        return model
