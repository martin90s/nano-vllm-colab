import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except (ImportError, OSError):
    flash_attn_varlen_func = flash_attn_with_kvcache = None

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def _repeat_kv(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    num_kv_heads = x.size(1)
    if num_kv_heads == num_heads:
        return x
    assert num_heads % num_kv_heads == 0
    return x.repeat_interleave(num_heads // num_kv_heads, dim=1)


def _read_paged_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    block_size = cache.size(1)
    num_blocks = (seq_len + block_size - 1) // block_size
    block_ids = block_table[:num_blocks].long()
    return cache.index_select(0, block_ids).flatten(0, 1)[:seq_len]


def _sdpa_one(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    q_len, num_heads, _ = q.shape
    k_len = k.size(0)
    k = _repeat_kv(k, num_heads)
    v = _repeat_kv(v, num_heads)

    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)

    attn_mask = None
    is_causal = causal and q_len == k_len
    if causal and not is_causal:
        prefix_len = k_len - q_len
        q_positions = torch.arange(q_len, device=q.device) + prefix_len
        k_positions = torch.arange(k_len, device=q.device)
        attn_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

    output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
    )
    return output.squeeze(0).transpose(0, 1)


def _sdpa_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float,
    block_table: torch.Tensor | None,
) -> torch.Tensor:
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    outputs = []
    for i in range(len(cu_q) - 1):
        q_i = q[cu_q[i]:cu_q[i + 1]]
        k_len = cu_k[i + 1] - cu_k[i]
        if block_table is None:
            k_i = k[cu_k[i]:cu_k[i + 1]]
            v_i = v[cu_k[i]:cu_k[i + 1]]
        else:
            k_i = _read_paged_cache(k, block_table[i], k_len)
            v_i = _read_paged_cache(v, block_table[i], k_len)
        outputs.append(_sdpa_one(q_i, k_i, v_i, scale, causal=True))
    return torch.cat(outputs)


def _sdpa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    outputs = []
    for i, seq_len in enumerate(context_lens.tolist()):
        k_i = _read_paged_cache(k_cache, block_table[i], seq_len)
        v_i = _read_paged_cache(v_cache, block_table[i], seq_len)
        outputs.append(_sdpa_one(q[i:i + 1], k_i, v_i, scale, causal=False))
    return torch.cat(outputs)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        use_flash_attn = (
            flash_attn_varlen_func is not None
            and torch.cuda.get_device_capability(q.device)[0] >= 8
        )
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            if use_flash_attn:
                return flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                )
            return _sdpa_varlen(
                q,
                k,
                v,
                context.cu_seqlens_q,
                context.cu_seqlens_k,
                self.scale,
                context.block_tables,
            )

        if use_flash_attn:
            return flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
        return _sdpa_decode(
            q,
            k_cache,
            v_cache,
            context.context_lens,
            context.block_tables,
            self.scale,
        )
