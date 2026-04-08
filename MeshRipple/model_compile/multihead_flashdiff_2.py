import math
import torch
import torch.nn.functional as F
from torch import nn
from .rms_norm import RMSNorm
from einops import rearrange

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len=9*30000):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.bfloat16)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.bfloat16)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, x_BTHD, offset):
        """
        Args:
            x_BTHD: Input tensor of shape [batch, seq_len, num_heads, dim]
            offset: Position offset for the sequence
        """
        batch_size, seq_len = x_BTHD.size(0), x_BTHD.size(-3)

        # Get the appropriate slice of cos/sin based on offset
        offset = offset.to(device=x_BTHD.device, dtype=torch.long).view(-1)
        positions = torch.arange(seq_len, device=x_BTHD.device).expand(batch_size, -1) + offset.unsqueeze(-1)
        cos = self.cos[positions].unsqueeze(-2)
        sin = self.sin[positions].unsqueeze(-2)

        x1, x2 = x_BTHD.to(dtype=torch.bfloat16).chunk(2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.cat((y1, y2), dim=-1)
    
class MultiheadFlashrope(nn.Module):
    def __init__(self, args, embed_dim, num_heads):
        super().__init__()
        self.args = args
        kvcache_method, kv_cache_window_size, window_stride, depth_scale = args
        self.depth_scale = depth_scale
        self.embed_dim = embed_dim
        self.num_heads = num_heads  
        self.num_kv_heads = num_heads
        self.head_dim = embed_dim // num_heads 
        self.scaling = self.head_dim ** -0.5
        self.kv_cache_window_size = kv_cache_window_size
        self.window_stride = window_stride
        self.kvcache_method = kvcache_method
        # KV cache related attributes
        self.kv_cache_enabled = False
        self.k_cache = None
        self.v_cache = None
        self.cache_pos = 0  # Track position in cache
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.register_buffer(
            "sliding_window_cur_len", 
            torch.tensor(0, dtype=torch.long), 
            persistent=False
        )
    
    def forward(self, x, start_pos, context=None, attention_mask=None, causal=False, move_sliding_window=False):
        """
        Args:
            x: [batch_size, tgt_len, embed_dim]
            start_pos: [batch_size]
            context: Optional context for cross-attention [batch_size, src_len, embed_dim]
        """
        bsz, tgt_len, embed_dim = x.size()
        
        # Project inputs to q, k, v
        q = self.q_proj(x)
        k = self.k_proj(context if context is not None else x)
        v = self.v_proj(context if context is not None else x)

        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        q = q.view(bsz, -1, self.num_heads, self.head_dim)
        k = k.view(bsz, -1, self.num_heads, self.head_dim)
        v = v.view(bsz, -1, self.num_heads, self.head_dim)
        
        q = self.rotary(q, offset=start_pos)
        
        # Handle KV cache if enabled
        if self.kv_cache_enabled:
            # Apply rotary embeddings with correct position offset for query
            k = self.rotary(k, offset=start_pos)
            
            if self.k_cache is not None:
                bsz, k_len = k.size(0), k.size(1)
                
                start = start_pos[0]    
                end = start + k_len
                window = self.kv_cache_window_size
                
                stride = getattr(self, 'window_stride', window // 2)

                # ring write
                write_positions = start + torch.arange(k_len, dtype=torch.long, device=k.device)
                write_indices = write_positions % window

                self.k_cache[:, write_indices, :, :] = k
                self.v_cache[:, write_indices, :, :] = v

                keep_start = torch.clamp(end - stride, min=0) // stride * stride
                
                cur_seq_len = end - keep_start

                full_read_positions = end - window + torch.arange(window, dtype=torch.long, device=k.device)
                full_read_indices = full_read_positions % window
                
                start_idx = window - cur_seq_len
                read_indices = full_read_indices[start_idx:]

                k = self.k_cache.index_select(1, read_indices)
                v = self.v_cache.index_select(1, read_indices)

                if attention_mask is None and causal and tgt_len == 1:
                    causal = False
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :, -read_indices.shape[0]:]
        else:
            # Normal operation without cache
            k = self.rotary(k, offset=start_pos)
            
        if attention_mask is not None:
            attn_output = self.attention_pytorch_optimized(q, k, v, mask=attention_mask)
        else:
            attn_output = self.attention_pytorch_optimized(q, k, v, causal=causal)

        # Reshape output
        attn_output = attn_output.contiguous().view(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output

    def attention_pytorch_optimized(self, q, k, v, mask=None, causal=False):
        b, seq_q, head_num, c = q.shape
        seq_k = k.shape[1]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if mask is not None:
            atten_mask_npu = mask.unsqueeze(1).expand(b, head_num, seq_q, seq_k)
            atten_mask_npu = torch.logical_not(atten_mask_npu)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=atten_mask_npu)
        else:
            if causal:
                attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                attn_output = F.scaled_dot_product_attention(q, k, v)

        return attn_output.permute(0, 2, 1, 3)

    def empty_kv_cache(self, batch_size: int, dtype: torch.dtype):
        """Initialize empty KV cache"""
        self.kv_cache_enabled=True

        device = next(self.parameters()).device
        kv_cache_maxlen = self.kv_cache_window_size
        # Initialize empty caches
        k_cache = torch.zeros(
            batch_size, 
            kv_cache_maxlen, 
            self.num_heads,
            self.head_dim,
            dtype=dtype,
            device=device
        )
        self.k_cache=k_cache

        v_cache = torch.zeros(
            batch_size,
            kv_cache_maxlen,
            self.num_heads,
            self.head_dim,
            dtype=dtype,
            device=device
        )       
        self.v_cache=v_cache
        self.cache_pos = 0  # Reset cache position

        element_size = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }[dtype]
        k_cache_size = (batch_size * 
                    kv_cache_maxlen * 
                    self.num_heads * 
                    self.head_dim * 
                    element_size)
        v_cache_size = k_cache_size 
        total_cache_size = k_cache_size + v_cache_size
        cache_size_gb = total_cache_size / (1024**3)
        return cache_size_gb
        

    def reset_cache(self):
        """Reset KV cache state"""
        self.k_cache = None
        self.v_cache = None
        self.cache_pos = 0


class MultiheadCrossFlashrope(nn.Module):
    def __init__(self, args, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads  
        self.num_kv_heads = num_heads
        self.head_dim = embed_dim // num_heads 
        kvcache_method, kv_cache_window_size, window_stride = args
        self.scaling = self.head_dim ** -0.5
        self.kv_cache_window_size = kv_cache_window_size
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rotary = Rotary(self.head_dim)

        self.kv_cache_enabled = False
        self.k_cache = None
        self.v_cache = None
        # self.attention_scores = None

    def forward(self, x, start_pos, context, attention_mask=None, causal=False, move_sliding_window=False):
        """
        Args:
            x: [batch_size, tgt_len, embed_dim]
            start_pos:
            context: Optional context for cross-attention [batch_size, src_len, embed_dim]
        """
        bsz, tgt_len, embed_dim = x.size()
        # Project inputs to q, k, v
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        q = q.view(bsz, -1, self.num_heads, self.head_dim)
        k = k.view(bsz, -1, self.num_heads, self.head_dim)
        v = v.view(bsz, -1, self.num_heads, self.head_dim)

        # Apply rotary embeddings with correct position offset for query
        q = self.rotary(q, offset=start_pos)
        
        # k = self.rotary(k, offset=k_start_pos)
        if self.kv_cache_enabled:
            # Apply rotary embeddings with correct position offset for query
            k = self.rotary(k, offset=start_pos)
            start = start_pos[0]
            if self.k_cache is not None:
                bsz, k_len = k.size(0), k.size(1)  # cache_len is the new sequence length
                q_len = q.size(1)
                end = start + q_len
                if move_sliding_window:
                    start = end-1
                input_pos_index = torch.arange(0, end, dtype=torch.long, device=self.v_cache.device) 
                self.k_cache[:, start: end, :, :] = k
                self.v_cache[:, start: end, :, :] = v

                k = self.k_cache.index_select(1, input_pos_index)
                v = self.v_cache.index_select(1, input_pos_index)
            if attention_mask is None and causal and tgt_len == 1:
                causal = False

        else:
            # Normal operation without cache
            k_start_pos = torch.full((bsz,), 0).to(q.device)
            k = self.rotary(k, offset=k_start_pos)

        if attention_mask is not None:
            attn_output = self.attention_pytorch_optimized(q, k, v, mask=attention_mask)
        else:
            attn_output = self.attention_pytorch_optimized(q, k, v, causal=causal)

        # Reshape output
        attn_output = attn_output.contiguous().view(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output
    
    def attention_pytorch_optimized(self, q, k, v, mask=None, causal=False):
        b, seq_q, head_num, c = q.shape
        seq_k = k.shape[1]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if mask is not None:
            atten_mask_npu = mask.unsqueeze(1).expand(b, head_num, seq_q, seq_k)
            atten_mask_npu = torch.logical_not(atten_mask_npu)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=atten_mask_npu)
        else:
            if causal:
                attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                attn_output = F.scaled_dot_product_attention(q, k, v)

        return attn_output.permute(0, 2, 1, 3)
    
    def empty_kv_cache(self, batch_size: int, dtype: torch.dtype):
        """Initialize empty KV cache"""
        self.kv_cache_enabled=True

        device = next(self.parameters()).device
        kv_cache_maxlen = self.kv_cache_window_size
        # Initialize empty caches
        k_cache = torch.zeros(
            batch_size, 
            kv_cache_maxlen, 
            self.num_heads,
            self.head_dim,
            dtype=dtype,
            device=device
        )
        self.k_cache=k_cache

        v_cache = torch.zeros(
            batch_size,
            kv_cache_maxlen,
            self.num_heads,
            self.head_dim,
            dtype=dtype,
            device=device
        )       
        self.v_cache=v_cache
        self.cache_pos = 0  # Reset cache position
        v_cache = torch.zeros(
            batch_size,
            kv_cache_maxlen,
            self.num_heads,
            self.head_dim,
            dtype=dtype,
            device=device
        )       
        self.v_cache=v_cache
        self.cache_pos = 0  # Reset cache position

        element_size = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }[dtype]
        k_cache_size = (batch_size * 
                    kv_cache_maxlen * 
                    self.num_heads * 
                    self.head_dim * 
                    element_size)
        v_cache_size = k_cache_size 
        total_cache_size = k_cache_size + v_cache_size
        cache_size_gb = total_cache_size / (1024**3)
        return cache_size_gb

    def reset_cache(self):
        """Reset KV cache state"""
        self.k_cache = None
        self.v_cache = None
        self.cache_pos = 0
        
class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(context_dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        # x: [batch, seq_len, dim], context: [batch, context_len, context_dim]
        B, N, C = x.shape
        _, M, _ = context.shape
        H = self.n_heads

        # Linear projections
        q = self.q_proj(x).view(B, N, H, C // H)  # [B, H, seq_len, dim//H]
        k, v = self.kv_proj(context).chunk(2, dim=-1)
        k = k.view(B, M, H, C // H)  # [B, context_len, H, dim//H]
        v = v.view(B, M, H, C // H)  # [B, context_len, H, dim//H]
        
        attn_output = self.attention_pytorch_optimized(q,k,v)
        attn_output = attn_output.contiguous().view(B, N, C)
        
        out = self.out_proj(attn_output)
        out = self.dropout(out)

        return out
    
    def attention_pytorch_optimized(self, q, k, v, mask=None, causal=False):
        b, seq_q, head_num, c = q.shape
        seq_k = k.shape[1]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if mask is not None:
            atten_mask_npu = mask.unsqueeze(1).expand(b, head_num, seq_q, seq_k)
            atten_mask_npu = torch.logical_not(atten_mask_npu)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=atten_mask_npu)
        else:
            if causal:
                attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                attn_output = F.scaled_dot_product_attention(q, k, v)

        return attn_output.permute(0, 2, 1, 3)