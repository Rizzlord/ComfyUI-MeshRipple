# This file is modified from the original implementation (implemented by Xunhao Lai)

import torch
from einops import rearrange
from flash_attn import flash_attn_varlen_func

from nsa.module.rope import RopeConfig, RotaryEmbedding
from nsa.module.hrope import Rotary
from nsa.ops.compressed_attention import compressed_attention
from nsa.ops.linear_compress import linear_compress
from nsa.ops.topk_sparse_attention import topk_sparse_attention


class NativeSparseAttention(torch.nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_q_heads: int,
            num_kv_heads: int,
            head_dim: int,
            kernel_size: int,
            kernel_stride: int,
            block_size: int,
            topk: int,
            init_blocks: int,
            local_blocks: int,
            window_size: int,
            rope_config: RopeConfig,  # useless
            debug_mode: bool = False,
    ):
        super().__init__()
        # configs
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.block_size = block_size
        self.topk = topk
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.window_size = window_size
        self.rope_config = rope_config
        self.debug_mode = debug_mode

        # qkv proj and o proj
        self.proj_q = torch.nn.Linear(self.hidden_size, self.num_q_heads * self.head_dim, bias=False)
        self.proj_k = torch.nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.proj_v = torch.nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.proj_o = torch.nn.Linear(self.num_q_heads * self.head_dim, self.hidden_size, bias=False)

        # nsa parameteres
        self.compress_key = torch.nn.Parameter(
            torch.zeros(self.num_kv_heads, self.head_dim * self.kernel_size, self.head_dim)
        )
        self.compress_value = torch.nn.Parameter(
            torch.zeros(self.num_kv_heads, self.head_dim * self.kernel_size, self.head_dim)
        )
        self.intra_block_pe = torch.nn.Parameter(torch.zeros(self.num_kv_heads, self.kernel_size, self.head_dim))

        # gate function
        self.gate = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, 3, bias=False), torch.nn.Sigmoid())

        # rope
        if rope_config is not None:
            self.rope = RotaryEmbedding(self.rope_config)
        else:
            # fixme: 50000 is the max seqlen, which is enough for our use case
            self.rope = Rotary(self.head_dim, max_seq_len=self.kernel_stride * 9 * 50001)

        # init parameters
        self.init_params()

    def init_params(self):
        # todo: something wrong with init
        return
        for p in self.parameters():
            if not self.debug_mode:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.constant_(p, 1.0)

    def forward(
            self,
            x_q: torch.Tensor,  # shape: [total_len, hidden_size]
            x_k: torch.Tensor,  # shape: [k_total_len, hidden_size]
            x_v: torch.Tensor,  # shape: [v_total_len, hidden_size]
            q_cu_seqlens: torch.Tensor,  # shape: [batch_size + 1]
            kv_cu_seqlens: torch.Tensor,  # shape: [batch_size + 1]
            q_start: int = 0,
            kv_start: int = 0,
            causal: bool = True,
            return_three_gates: bool = False,
    ):
        # dtype and shape check
        assert x_q.dtype == torch.bfloat16 or x_q.dtype == torch.float16, x_q.dtype
        assert x_k.dtype == torch.bfloat16 or x_k.dtype == torch.float16, x_k.dtype
        assert x_v.dtype == torch.bfloat16 or x_v.dtype == torch.float16, x_v.dtype
        assert x_q.shape[-1] == self.hidden_size
        assert x_k.shape[-1] == self.hidden_size
        assert x_v.shape[-1] == self.hidden_size

        q_cu_seqlens = q_cu_seqlens.to(torch.int32)
        q_seqlens = q_cu_seqlens[1:] - q_cu_seqlens[:-1]
        kv_cu_seqlens = kv_cu_seqlens.to(torch.int32)
        kv_seqlens = kv_cu_seqlens[1:] - kv_cu_seqlens[:-1]

        # qkv proj
        q = self.proj_q(x_q).view(-1, self.num_q_heads, self.head_dim)
        k = self.proj_k(x_k).view(-1, self.num_kv_heads, self.head_dim)
        v = self.proj_v(x_v).view(-1, self.num_kv_heads, self.head_dim)

        # compressed key and value before rope
        compressed_k, compressed_k_cu_seqlens = linear_compress(
            k,
            self.compress_key,
            kv_cu_seqlens,
            self.kernel_size,
            self.kernel_stride,
            self.intra_block_pe,
        )
        compressed_v, _ = linear_compress(
            v,
            self.compress_value,
            kv_cu_seqlens,
            self.kernel_size,
            self.kernel_stride,
            None,
        )

        # do rope for query and compressed key
        q = self.rope(q, q_cu_seqlens, offset=q_start)
        compressed_k = self.rope(compressed_k, compressed_k_cu_seqlens, offset=kv_start, stride=self.kernel_stride)

        # attention between query and compressed key value
        compressed_k_seqlens = compressed_k_cu_seqlens[1:] - compressed_k_cu_seqlens[:-1]
        compressed_attn_output, topk_idx = compressed_attention(
            q,
            compressed_k,
            compressed_v,
            self.kernel_size,
            self.kernel_stride,
            self.block_size,
            self.topk,
            q_cu_seqlens,
            compressed_k_cu_seqlens,
            kv_cu_seqlens,
            q_seqlens.max().item(),
            compressed_k_seqlens.max().item(),
            None,
            self.init_blocks,
            self.local_blocks,
            parallel_topk_compute=False,
            causal=causal,
        )
        if self.debug_mode:
            print("compressed_attn_output:", compressed_attn_output, compressed_attn_output.shape)
            print("topk_idx:", topk_idx, topk_idx.shape)

        # do rope for original key
        k = self.rope(k, kv_cu_seqlens, offset=kv_start)

        # topk sparse attention
        sparse_attn_output = topk_sparse_attention(
            q, k, v, topk_idx, self.block_size, q_cu_seqlens, kv_cu_seqlens, None, causal=causal
        )
        if self.debug_mode:
            print("sparse_attn_output:", sparse_attn_output, sparse_attn_output.shape)

        # sliding window attention
        sliding_attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            q_cu_seqlens,
            kv_cu_seqlens,
            q_seqlens.max().item(),
            kv_seqlens.max().item(),
            causal=causal,
            window_size=(self.window_size, -1),
        )
        if self.debug_mode:
            print("sliding_attn_output:", sliding_attn_output, sliding_attn_output.shape)

        # gate average
        gate = self.gate(x_q)
        attn_output = (
                gate[:, 0:1, None] * compressed_attn_output
                + gate[:, 1:2, None] * sparse_attn_output
                + gate[:, 2:3, None] * sliding_attn_output
        )

        # rearrange and output proj
        attn_output = rearrange(attn_output, "n h d -> n (h d)")
        attn_output = self.proj_o(attn_output)
        if return_three_gates:
            return attn_output, compressed_attn_output, sparse_attn_output, sliding_attn_output
        return attn_output
