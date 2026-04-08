# This file is modified from the original implementation (implemented by Xunhao Lai)

import torch
from typing import Optional
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
            past_kv: Optional[dict] = None,
    ):
        # dtype and shape check
        assert x_q.dtype == torch.bfloat16 or x_q.dtype == torch.float16, x_q.dtype
        
        # When using cache, x_k and x_v only contain the new tokens
        if past_kv is not None:
            # q_start is used as the current sequence length in the cache management logic elsewhere,
            # but here we need the total length for indexing.
            pass

        q_cu_seqlens = q_cu_seqlens.to(torch.int32)
        q_seqlens = q_cu_seqlens[1:] - q_cu_seqlens[:-1]
        kv_cu_seqlens = kv_cu_seqlens.to(torch.int32)
        kv_seqlens = kv_cu_seqlens[1:] - kv_cu_seqlens[:-1]

        # qkv proj
        q = self.proj_q(x_q).view(-1, self.num_q_heads, self.head_dim)
        k = self.proj_k(x_k).view(-1, self.num_kv_heads, self.head_dim)
        v = self.proj_v(x_v).view(-1, self.num_kv_heads, self.head_dim)

        if past_kv is not None:
            # Incremental KV Cache Update
            # In MeshRipple, k and v here are the new tokens [batch * seq_len, heads, dim]
            # Since we assume single-token generation (seq_len=1 per batch), we can simplify.
            # However, for robustness, we'll handle multiple tokens.
            
            k_cache = past_kv['k_raw']
            v_cache = past_kv['v_raw']
            
            # Append new raw k,v to cache
            # Expected k_cache shape: [total_len, heads, dim]
            # Assuming contiguous batching for now as per nsa design
            k_cache = torch.cat([k_cache, k], dim=0)
            v_cache = torch.cat([v_cache, v], dim=0)
            past_kv['k_raw'] = k_cache
            past_kv['v_raw'] = v_cache
            
            full_k = k_cache
            full_v = v_cache
            
            # total raw length
            total_raw_len = full_k.shape[0]
            # current batch size (assuming 1 for now or fixed)
            bsz = q_cu_seqlens.shape[0] - 1
            
            # Update compressed cache
            k_comp_cache = past_kv['k_compressed']
            v_comp_cache = past_kv['v_compressed']
            
            # Logic: check if we hit a new compression window
            # N = floor((L - K) / S) + 1
            new_comp_len = max(0, (total_raw_len - self.kernel_size) // self.kernel_stride + 1)
            old_comp_len = k_comp_cache.shape[0]
            
            if new_comp_len > old_comp_len:
                # We need to compute new compressed tokens
                # We can just run linear_compress on the relevant window of full_k/full_v
                # or for simplicity (since it's only 1 new token usually), run it on the whole thing and take last
                # But running it on the whole thing defeats the O(1) purpose.
                # However, linear_compress kernel is fast.
                
                # To keep it O(1), we only compress the NEW window
                for i in range(old_comp_len, new_comp_len):
                    w_start = i * self.kernel_stride
                    w_end = w_start + self.kernel_size
                    
                    # Manual compression for high performance on single window
                    # window_k: [1, kernel_size, heads, dim]
                    window_k = full_k[w_start:w_end].reshape(1, self.kernel_size, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
                    window_k = rearrange(window_k, 'b h k d -> b h (k d)')
                    # window_k: [1, heads, kernel_size*dim]
                    # weight: [heads, kernel_size*dim, dim]
                    new_k_c = torch.einsum('b h d, h d D -> b h D', window_k, self.compress_key)
                    # apply PE if exists
                    pe = rearrange(self.intra_block_pe, 'h k d -> h (k d)')
                    bias_k = torch.einsum('h D, h D d -> h d', pe, self.compress_key)
                    new_k_c = new_k_c + bias_k.unsqueeze(0)
                    
                    k_comp_cache = torch.cat([k_comp_cache, new_k_c.view(-1, self.num_kv_heads, self.head_dim)], dim=0)
                    
                    # Same for V (no PE usually)
                    window_v = full_v[w_start:w_end].reshape(1, self.kernel_size, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
                    window_v = rearrange(window_v, 'b h k d -> b h (k d)')
                    new_v_c = torch.einsum('b h d, h d D -> b h D', window_v, self.compress_value)
                    v_comp_cache = torch.cat([v_comp_cache, new_v_c.view(-1, self.num_kv_heads, self.head_dim)], dim=0)
                
                past_kv['k_compressed'] = k_comp_cache
                past_kv['v_compressed'] = v_comp_cache

            compressed_k = k_comp_cache
            compressed_v = v_comp_cache
            
            # Recalculate cu_seqlens for the full cache
            full_kv_cu_seqlens = torch.tensor([0, total_raw_len], device=q.device, dtype=torch.int32)
            compressed_k_cu_seqlens = torch.tensor([0, compressed_k.shape[0]], device=q.device, dtype=torch.int32)
            
            k = full_k
            v = full_v
            kv_cu_seqlens = full_kv_cu_seqlens
        else:
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
