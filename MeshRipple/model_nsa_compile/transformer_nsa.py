from .multihead_flashdiff_2 import MultiheadFlashrope, MultiheadCrossFlashrope, CrossAttention
from .rms_norm import RMSNorm
import time  # cal_time new at 10.21
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
import torch.nn.functional as F
from tqdm import tqdm
import tqdm as tqdm1
import timm

from utils.utils import update_eos_mask
from torch.optim.lr_scheduler import LambdaLR
import subprocess
import math
from einops import rearrange, repeat, pack
from .miche_conditioner import PointConditioner
from nsa import RopeConfig, NativeSparseAttention


class LinearDownsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim * shorten_factor, dim)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = rearrange(x, 'b (n s) d -> b n (s d)', s=self.shorten_factor)
        return self.proj(x)


class LinearUpsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim, dim * shorten_factor)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, 'b n (s d) -> b (n s) d', s=self.shorten_factor)


class SwiGLU(nn.Module):
    def __init__(self, embed_dim):
        super(SwiGLU, self).__init__()
        self.proj1 = nn.Linear(embed_dim, int(8 / 3 * embed_dim))  # XWG
        self.proj2 = nn.Linear(embed_dim, int(8 / 3 * embed_dim))  # XW1
        self.proj3 = nn.Linear(int(8 / 3 * embed_dim), embed_dim)  # W2

    def forward(self, x):
        # Apply Swish (SiLU in PyTorch) to the first projection
        x_proj1 = F.silu(self.proj1(x))  # Swish(XWG)
        # Apply the second projection
        x_proj2 = self.proj2(x)  # XW1
        # Element-wise multiplication
        x_glu = x_proj1 * x_proj2  # (swish(XWG) ⊙ XW1)
        # Final projection back to the original dimension
        output = self.proj3(x_glu)  # (swish(XWG) ⊙ XW1)W2

        return output

def shift_sequence(x, shift_amount):
    if shift_amount == 0:
        return x
    else:
        shifted_x = torch.zeros_like(x)
        if shift_amount < x.size(1):
            shifted_x[:, shift_amount:, :] = x[:, :-shift_amount, :]
        return shifted_x


def pad_to_multiple(tensor, multiple, dim=-1, pad_value=0):
    current_size = tensor.size(dim)
    remainder = current_size % multiple
    if remainder == 0:
        return tensor
    pad_size = multiple - remainder
    pad = [0] * (2 * tensor.dim())
    pad_start = (tensor.dim() - 1 - dim) * 2
    pad[pad_start + 1] = pad_size
    padded_tensor = F.pad(tensor, pad, mode='constant', value=pad_value)
    return padded_tensor


class MLA(nn.Module):
    def __init__(self, embed_dim, num_heads, args, cross_embed_dim, causal=True, dropout=0.):
        """
        Differential Transformer Block with optional causal self-attention or cross-attention
        and automatic application of ROPE.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            args (Namespace): Arguments for attention settings.
            causal (bool): Whether to use causal masking by default for self-attention.
        """
        super(MLA, self).__init__()
        # Multihead differential attention module
        self.attn = MultiheadFlashrope(args, embed_dim, num_heads)
        self.causal = causal

        self.feed_forward = SwiGLU(embed_dim)
        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm2 = RMSNorm(embed_dim, eps=1e-5)
        self.attn_dropout = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        if cross_embed_dim != -1:
            self.cross_norm = RMSNorm(embed_dim, eps=1e-5)
            self.cross_attn = CrossAttention(embed_dim, cross_embed_dim, num_heads)

    def forward(
            self, x, start, conds=None, attention_mask=None, use_cache=False
    ):
        """
        Args:
            x: Input tensor
            use_cache: Whether to use KV cache
        """
        # Enable/disable KV cache in attention module
        self.attn.kv_cache_enabled = use_cache

        residual = x
        x = self.norm1(x)
        attn_out = self.attn(
            x,
            start_pos=start,
            attention_mask=attention_mask,
            causal=self.causal,
        )
        attn_out = self.attn_dropout(attn_out)
        x = residual + attn_out
        if conds is not None:
            x_skip = x
            x = self.cross_norm(x)
            x = self.cross_attn(x, conds) + x_skip

        residual = x
        ff_out = self.ff_dropout((self.feed_forward(self.norm2(x))))
        x = residual + ff_out

        return x

    def init_kv_cache(self, batch_size, dtype=torch.bfloat16):
        """Initialize KV cache for this transformer block"""
        return self.attn.empty_kv_cache(
            batch_size=batch_size, dtype=dtype
        )

    def reset_kv_cache(self):
        """Reset KV cache for this transformer block"""
        self.attn.reset_cache()


class MLCA(nn.Module):
    def __init__(
        self, embed_dim, num_heads, args, causal=True, dropout=0.,
        head_dim:int = 16,
        kernel_size:int = 32,
        kernel_stride:int = 16,
        block_size:int = 64,
        topk:int = 16,
        window_size:int = 512,
    ):
        """
        multi head cross attention

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            args (Namespace): Arguments for attention settings.
            causal (bool): Whether to use causal masking by default for self-attention.
        """
        super(MLCA, self).__init__()

        # Multihead differential attention module
        self.attn = NativeSparseAttention(
            hidden_size=embed_dim,
            num_q_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            topk=topk,
            init_blocks=1,
            local_blocks=2,
            window_size=window_size,
            rope_config=None,
        ).to(torch.bfloat16)
        self.causal = causal
        self.feed_forward = SwiGLU(embed_dim)
        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm2 = RMSNorm(embed_dim, eps=1e-5)

        self.attn_dropout = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)

    @staticmethod
    def get_attn_input(x: torch.Tensor, seqlens: Optional[torch.Tensor]):
        bsz, seq_len, hidden_size = x.shape
        
        if seqlens is not None:
            seqlens = seqlens.to(torch.int32)
            x_list = [x[i, :seqlens[i], :].reshape(-1, hidden_size) for i in range(bsz)] # [(<=seqlens[i], hidden_size)]
            x_list = [
                torch.cat(
                    [x[i], x[i,-1:].repeat(seqlens[i] - x[i].shape[0], 1)],
                    dim=0
                ) if seqlens[i] > x[i].shape[0] else x[i, :seqlens[i], :] for i in range(bsz)
            ] # [(seqlens[i], hidden_size)]
            x = torch.cat(x_list, dim=0) # (sum(seqlens), hidden_size)
        else:
            # seqlens is None means all x are not padding, and have the same length
            seqlens = torch.tensor([seq_len] * bsz, device=x.device, dtype=torch.int32)
            x = x.reshape(-1, hidden_size)
        
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=x.device),
                torch.cumsum(seqlens, dim=0),
            ],
            dim=0,
        ).to(torch.int32)

        return x.to(torch.bfloat16).contiguous(), cu_seqlens

    def forward(
            self, x, start, context, attention_mask=None, use_cache=False, causal=False,
    ):
        """
        Args:
            x: Input tensor
            context: context for cross-attention
            use_cache: Whether to use KV cache
        """
        # self.attn.kv_cache_enabled = use_cache
        
        # First residual path: cross-attention + dropout
        residual = x
        bsz, seq_len, hidden_size = x.shape
        x_norm = self.norm1(x)
        
        # q is always not contains padding or it is not a problem with loss mask
        q, q_cu_seqlens = self.get_attn_input(x_norm, None)
        
        if causal:
            # kv = history+q
            # which means k_len = q_len + q_start
            kv_seqlens = torch.tensor(
                [start[i]+seq_len for i in range(bsz)], device=x.device, dtype=torch.int32
            )
            kv, kv_cu_seqlens = self.get_attn_input(context, kv_seqlens)
        else:
            # kv not contains padding when causal=False
            kv, kv_cu_seqlens = self.get_attn_input(context, None)
        
        attn_out = self.attn(
            q,
            kv,
            kv,
            q_cu_seqlens,
            kv_cu_seqlens,
            q_start=start,
            kv_start=0,
            causal=causal,
        )
        
        attn_out = self.attn_dropout(attn_out)
        attn_out = attn_out.reshape(bsz, seq_len, hidden_size)
        x = residual + attn_out

        # Second residual path: feed-forward network + dropout
        residual = x
        x_norm = self.norm2(x)
        ff_out = self.feed_forward(x_norm)
        ff_out = self.ff_dropout(ff_out)
        x = residual + ff_out

        return x

    def init_kv_cache(self, batch_size, dtype=torch.bfloat16):
        """Initialize KV cache for this transformer block"""
        # return self.attn.empty_kv_cache(
        #     batch_size=batch_size, dtype=dtype
        # )

        # todo: nsa-kv-cache
        return 0

    def reset_kv_cache(self):
        """Reset KV cache for this transformer block"""
        # self.attn.reset_cache()

        # todo: nsa-kv-cache
        pass


class NSAFaceBoundary(nn.Module):
    def __init__(
            self,
            args,
            embed_dim,
            num_heads,
            max_len,
            num_categories,
            num_classify,
            context_embedding_dim,
            hourglass_depth=[4, 4, 8],
            input_embe_pos="inner",
            root_pred_depth=1,
            conditioned_on_pc=True,
            encoder_name='miche-256-feature',
            encoder_freeze=True,
            nsa_params={},
    ):
        super(NSAFaceBoundary, self).__init__()

        cache_method, kv_cache_window_size, cache_stride = args
        self.cache_method = cache_method
        if cache_method == "random":
            kv_cache_window_size = kv_cache_window_size - 1
        # kv_cache_window_size = kv_cache_window_size
        kv_cache_window_size = kv_cache_window_size * 9
        cache_stride = cache_stride * 9
        assert kv_cache_window_size % 9 == 0, "kv_cache_window_size error"
        self.depth = 2
        # kv_cache_window_size, kv_cache_window_size//3
        encoder_args = [[cache_method, kv_cache_window_size // int(math.pow(3, i)),
                         cache_stride // int(math.pow(3, i)), int(math.pow(3, self.depth - i))]
                        for i in range(self.depth)]

        deocker_args = [[cache_method, kv_cache_window_size // int(math.pow(3, self.depth - 1 - i)),
                         cache_stride // int(math.pow(3, self.depth - 1 - i)), int(math.pow(3, i + 1))]
                        for i in range(self.depth)]

        # kv_cache_window_size//3, kv_cache_window_size
        bottle_args = [cache_method, kv_cache_window_size // int(math.pow(3, self.depth)),
                       cache_stride // int(math.pow(3, self.depth)), 1]
        bottle_cross_args = cache_method, max_len, cache_stride // int(math.pow(3, self.depth))
        self.embed_dim = embed_dim

        self.skip_weights2 = nn.Parameter(torch.ones(2))
        # load point_cloud encoder
        if conditioned_on_pc:
            print(f'Point cloud encoder: {encoder_name} | freeze: {encoder_freeze}')
            self.conditioner = PointConditioner(model_name=encoder_name, freeze=encoder_freeze, feature_dim=embed_dim)
            cross_attn_dim = self.conditioner.dim_latent
            # self.token_len_conditioner = nn.Embedding(max_len, cross_attn_dim)
        else:
            cross_attn_dim = -1
        self.conditioned_on_pc = conditioned_on_pc
        self.embedding = nn.Embedding(num_categories, embed_dim)
        self.context_embedding = nn.Embedding(num_categories, context_embedding_dim)
        self.context_proj = nn.Linear(context_embedding_dim * 9, embed_dim)

        self.downsamplers = nn.ModuleList([
            LinearDownsample(embed_dim, 3),
            LinearDownsample(embed_dim, 3)
        ])
        self.upsamplers = nn.ModuleList([
            LinearUpsample(embed_dim, 3),
            LinearUpsample(embed_dim, 3)
        ])

        self.encoder_blocks = nn.ModuleList([
            nn.ModuleList([
                MLA(embed_dim, num_heads, cross_embed_dim=cross_attn_dim, args=encoder_args[0], causal=True)
                for i in range(hourglass_depth[0])
            ]),
            nn.ModuleList([
                MLA(embed_dim, num_heads, cross_embed_dim=cross_attn_dim, args=encoder_args[1], causal=True)
                for i in range(hourglass_depth[1])
            ])
        ])

        self.bottlenecke = nn.ModuleList([
            MLA(embed_dim, num_heads, cross_embed_dim=cross_attn_dim, args=bottle_args, causal=True) if i % 3 == 0
            else MLA(embed_dim, num_heads, cross_embed_dim=cross_attn_dim, args=bottle_args, causal=True) if i % 3 == 1
            else MLCA(embed_dim, num_heads, args=bottle_cross_args, causal=True, **nsa_params)  # Added branch
            for i in range(hourglass_depth[2])
        ])
        self.bottleneckd = nn.ModuleList([
            MLA(embed_dim, num_heads, cross_embed_dim=cross_attn_dim, args=bottle_args, causal=True) if i % 3 == 0
            else MLA(embed_dim, num_heads, cross_embed_dim=cross_attn_dim, args=bottle_args, causal=True) if i % 3 == 1
            else MLCA(embed_dim, num_heads, args=bottle_cross_args, causal=True, **nsa_params)  # Added branch
            for i in range(hourglass_depth[2])
        ])

        self.decoder_blocks = nn.ModuleList([
            nn.ModuleList([
                MLA(embed_dim, num_heads, cross_embed_dim=cross_attn_dim, args=deocker_args[0], causal=True)
                for i in range(hourglass_depth[1])
            ]),
            nn.ModuleList([
                MLA(embed_dim, num_heads, cross_embed_dim=cross_attn_dim, args=deocker_args[1], causal=True)
                for i in range(hourglass_depth[0])
            ])
        ])
        self.root_pred_depth = root_pred_depth

        self.root_pred_bottleneckd = nn.ModuleList([
            MLA(embed_dim, num_heads, cross_embed_dim=cross_attn_dim, args=bottle_args, causal=True) if i % 3 == 0
            else MLA(embed_dim, num_heads, cross_embed_dim=cross_attn_dim, args=bottle_args, causal=True) if i % 3 == 1
            else MLCA(embed_dim, num_heads, args=bottle_cross_args, causal=True, **nsa_params)  # Added branch
            for i in range(self.root_pred_depth)
        ])
        self.to_root = nn.Linear(embed_dim, num_classify)
        self.output_proj = nn.Linear(embed_dim, num_categories)

        self.factor = [3, 3]
        self.norm = RMSNorm(embed_dim, eps=1e-5)

    def forward(self, x, context, pc, token_len, boundary_mask, cross_mask, start, input_root=None):
        assert torch.all(start % 9 == 0), "hourglass start must be divided by 9"
        if x.shape[1] % 9 != 0:
            x = pad_to_multiple(x, 9, 1)
        if context is not None:
            context = self.context_embedding(context).reshape(context.shape[0], context.shape[1], -1)
            context = self.context_proj(context)
        x = self.embedding(x)
        encoder_start = [start, start // 3]
        decoder_start = [start // 3, start]
        bottle_start = start // 9
        x = self.norm(x)
        encoder_outputs = []
        if pc is not None and self.conditioned_on_pc:
            conds = self.conditioner(pc)  # b,257,1024
            # token_len_embed = self.token_len_conditioner(token_len).unsqueeze(1)
            # conds = torch.cat([conds, token_len_embed.expand(-1, conds.shape[1], -1)], dim=-1)
        else:
            conds = None
        # Compression stage
        for scale in range(self.depth):
            for block in self.encoder_blocks[scale]:
                x = block(x, conds=conds, start=encoder_start[scale])  # Self-attention

            encoder_outputs.append(x)
            x = self.downsamplers[scale](x)

        # Bottleneck stage
        for i, block in enumerate(self.bottlenecke):
            
            if i % 3 == 0:
                x = block(x, conds=conds, attention_mask=boundary_mask, start=bottle_start)
            elif i % 3 == 1:
                x = block(x, conds=conds, start=bottle_start)
            else:
                x = block(x, context=context, causal=True, start=bottle_start)
        inner_output = x.clone()
        for i, block in enumerate(self.bottleneckd):
            
            if i % 3 == 0:
                x = block(x, conds=conds, attention_mask=boundary_mask, start=bottle_start)
            elif i % 3 == 1:
                x = block(x, conds=conds, start=bottle_start)
            else:
                x = block(x, context=context, causal=True, start=bottle_start)
        # Decompression stage
        for scale in range(self.depth):
            x = self.upsamplers[scale](x)
            skip = encoder_outputs[-(scale + 1)]

            x = shift_sequence(x, self.factor[scale] - 1)
            x = self.skip_weights2[scale] * x + skip

            for block in self.decoder_blocks[scale]:
                x = block(x, conds=conds, start=decoder_start[scale])  # Self-attention

        x = self.output_proj(x)

        # Predict the next root-node offset
        root_pred = inner_output
        for i, block in enumerate(self.root_pred_bottleneckd):
            if i % 3 == 0:
                root_pred = block(root_pred, conds=conds, attention_mask=boundary_mask, start=bottle_start)
            elif i % 3 == 1:
                root_pred = block(root_pred, conds=conds, start=bottle_start)
            else:
                root_pred = block(root_pred, causal=True, context=context, start=bottle_start)
        root_pred = self.to_root(root_pred)
        return x, root_pred

    def init_kv_cache(self, batch_size, max_len=90, dtype=torch.bfloat16):

        Gb = 0
        self.use_cache = True
        self.inference_state = {
            'cache_initialized': False,
            'cur_pos': 0,
            # 'encoder_outputs': [],
            'dtype': dtype,
            'batch_size': batch_size,
            'max_len': max_len,

            'layer_states': {
                'encoder_0': None,
                'encoder_1': None,
                # 'bottleneck': None,  
            },

            'upsampled_states': {
                'decoder_0': None,
                'decoder_1': None
            }
        }

        for scale in range(self.depth):
            for i, block in enumerate(self.encoder_blocks[scale]):
                Gb += block.init_kv_cache(batch_size, dtype)

        for i, block in enumerate(self.bottlenecke):
            Gb += block.init_kv_cache(batch_size, dtype)

        for i, block in enumerate(self.bottleneckd):
            Gb += block.init_kv_cache(batch_size, dtype)

        for i, block in enumerate(self.root_pred_bottleneckd):
            Gb += block.init_kv_cache(batch_size, dtype)

        for scale in range(self.depth):
            for i, block in enumerate(self.decoder_blocks[scale]):
                Gb += block.init_kv_cache(batch_size, dtype)
        return Gb

    def reset_kv_cache(self):

        if hasattr(self, 'inference_state'):

            self.inference_state['cur_pos'] = 0
            # self.inference_state['encoder_outputs'] = []
            self.inference_state['cache_initialized'] = False
            self.inference_state['layer_states'] = {
                'encoder_0': None,
                'encoder_1': None,
                'bottleneck': None,
            }
            self.inference_state['upsampled_states'] = {
                'decoder_0': None,
                'decoder_1': None
            }

            for scale in range(self.depth):
                for block in self.encoder_blocks[scale]:
                    block.reset_kv_cache()

            for block in self.bottlenecke:
                block.reset_kv_cache()

            for block in self.bottleneckd:
                block.reset_kv_cache()

            for block in self.root_pred_bottleneckd:
                block.reset_kv_cache()

            for scale in range(self.depth):
                for block in self.decoder_blocks[scale]:
                    block.reset_kv_cache()

    def _process_first_tokens(self, x, context, boundary_mask, start, conds=None):
        if context is not None:
            context = self.context_embedding(context).reshape(context.shape[0], context.shape[1], -1)
            context = self.context_proj(context)
        x = self.embedding(x)
        x = self.norm(x)

        encoder_outputs = []

        # First compression stage
        for block in self.encoder_blocks[0]:
            x = block(x, conds=conds, start=start, use_cache=True)
        encoder_outputs.append(x)
        self.inference_state['layer_states']['encoder_0'] = x[:, -3:]

        x_downsampled = self.downsamplers[0](x)

        # Second compression stage
        for block in self.encoder_blocks[1]:
            x_downsampled = block(x_downsampled, conds=conds, start=start // 3, use_cache=True)
        encoder_outputs.append(x_downsampled)
        self.inference_state['layer_states']['encoder_1'] = x_downsampled[:, -3:]

        x_bottleneck = self.downsamplers[1](x_downsampled)

        # Bottleneck stage
        for i, block in enumerate(self.bottlenecke):
            if i % 3 == 0:
                x_bottleneck = block(x_bottleneck, conds=conds, attention_mask=boundary_mask, start=start // 9,
                                     use_cache=True)
            elif i % 3 == 1:
                x_bottleneck = block(x_bottleneck, conds=conds, start=start // 9, use_cache=True)
            else:
                x_bottleneck = block(x_bottleneck, context=context, start=start // 9, use_cache=True)
        inner_output = x_bottleneck.clone()
        for i, block in enumerate(self.bottleneckd):
            if i % 3 == 0:
                x_bottleneck = block(x_bottleneck, conds=conds, attention_mask=boundary_mask, start=start // 9,
                                     use_cache=True)
            elif i % 3 == 1:
                x_bottleneck = block(x_bottleneck, conds=conds, start=start // 9, use_cache=True)
            else:
                x_bottleneck = block(x_bottleneck, context=context, start=start // 9, use_cache=True)

        # First decoding stage
        x_upsampled = self.upsamplers[0](x_bottleneck)
        self.inference_state['upsampled_states']['decoder_0'] = x_upsampled[:, -3:]

        skip = encoder_outputs[1]

        x_upsampled = shift_sequence(x_upsampled, self.factor[0] - 1)
        x_upsampled = self.skip_weights2[0] * x_upsampled + skip

        for block in self.decoder_blocks[0]:
            x_upsampled = block(x_upsampled, conds=conds, start=start // 3, use_cache=True)

        # Second decoding stage
        x_final = self.upsamplers[1](x_upsampled)
        self.inference_state['upsampled_states']['decoder_1'] = x_final[:, -3:]

        skip = encoder_outputs[0]

        x_final = shift_sequence(x_final, self.factor[1] - 1)
        x_final = self.skip_weights2[1] * x_final + skip

        for block in self.decoder_blocks[1]:
            x_final = block(x_final, conds=conds, start=start, use_cache=True)

        logits = self.output_proj(x_final)

        # Predict the next root-node offset
        root_pred = inner_output
        for i, block in enumerate(self.root_pred_bottleneckd):
            if i % 3 == 0:
                root_pred = block(root_pred, conds=conds, attention_mask=boundary_mask, start=start // 9,
                                  use_cache=True)
            elif i % 3 == 1:
                root_pred = block(root_pred, conds=conds, start=start // 9, use_cache=True)
            else:
                root_pred = block(root_pred, context=context, start=start // 9, use_cache=True)
        root_pred = self.to_root(root_pred)
        self.inference_state['cur_pos'] = x.shape[1]
        self.inference_state['cache_initialized'] = True

        return logits, root_pred

    def _process_muti_tokens(self, x, context, boundary_mask, start, conds=None):
        if context is not None:
            context = self.context_embedding(context).reshape(context.shape[0], context.shape[1], -1)
            context = self.context_proj(context)
        x = self.embedding(x)
        x = self.norm(x)

        encoder_outputs = []

        # First compression stage
        for block in self.encoder_blocks[0]:
            x = block(x, conds=conds, start=start, use_cache=True)
        encoder_outputs.append(x)
        self.inference_state['layer_states']['encoder_0'] = x[:, -3:]

        x_downsampled = self.downsamplers[0](x)

        # Second compression stage
        for block in self.encoder_blocks[1]:
            x_downsampled = block(x_downsampled, conds=conds, start=start // 3, use_cache=True)
        encoder_outputs.append(x_downsampled)
        self.inference_state['layer_states']['encoder_1'] = x_downsampled[:, -3:]

        x_bottleneck = self.downsamplers[1](x_downsampled)

        # Bottleneck stage
        for i, block in enumerate(self.bottlenecke):
            if i % 3 == 0:
                x_bottleneck = block(x_bottleneck, conds=conds, attention_mask=boundary_mask, start=start // 9,
                                     use_cache=True)
            elif i % 3 == 1:
                x_bottleneck = block(x_bottleneck, conds=conds, start=start // 9, use_cache=True)
            else:
                x_bottleneck = block(x_bottleneck, context=context, start=start // 9, use_cache=True)
        inner_output = x_bottleneck.clone()
        for i, block in enumerate(self.bottleneckd):
            if i % 3 == 0:
                x_bottleneck = block(x_bottleneck, conds=conds, attention_mask=boundary_mask, start=start // 9,
                                     use_cache=True)
            elif i % 3 == 1:
                x_bottleneck = block(x_bottleneck, conds=conds, start=start // 9, use_cache=True)
            else:
                x_bottleneck = block(x_bottleneck, context=context, start=start // 9, use_cache=True)

        # First decoding stage
        x_upsampled = self.upsamplers[0](x_bottleneck)
        self.inference_state['upsampled_states']['decoder_0'] = x_upsampled[:, -3:]

        skip = encoder_outputs[1]

        x_upsampled = shift_sequence(x_upsampled, self.factor[0] - 1)
        x_upsampled = self.skip_weights2[0] * x_upsampled + skip

        for block in self.decoder_blocks[0]:
            x_upsampled = block(x_upsampled, conds=conds, start=start // 3, use_cache=True)

        # Second decoding stage
        x_final = self.upsamplers[1](x_upsampled)
        self.inference_state['upsampled_states']['decoder_1'] = x_final[:, -3:]

        skip = encoder_outputs[0]

        x_final = shift_sequence(x_final, self.factor[1] - 1)
        x_final = self.skip_weights2[1] * x_final + skip

        for block in self.decoder_blocks[1]:
            x_final = block(x_final, conds=conds, start=start, use_cache=True)

        logits = self.output_proj(x_final)

        # Predict the next root-node offset
        root_pred = inner_output
        for i, block in enumerate(self.root_pred_bottleneckd):
            if i % 3 == 0:
                root_pred = block(root_pred, conds=conds, attention_mask=boundary_mask, start=start // 9,
                                  use_cache=True)
            elif i % 3 == 1:
                root_pred = block(root_pred, conds=conds, start=start // 9, use_cache=True)
            else:
                root_pred = block(root_pred, context=context, start=start // 9, use_cache=True)
        root_pred = self.to_root(root_pred)
        self.inference_state['cur_pos'] += 1

        return logits, root_pred

    def _math_scale_1(self, x, start_pos: int, conds, dec1_state):
        x = self.embedding(x)
        x = self.norm(x)

        enc0_out = x
        i = 0
        n_enc0 = len(self.encoder_blocks[0])
        while i < n_enc0:
            enc0_out = self.encoder_blocks[0][i](enc0_out, conds=conds, start=start_pos, use_cache=True)
            i += 1

        x_final = self.skip_weights2[1] * dec1_state + enc0_out
        i = 0
        n_dec1 = len(self.decoder_blocks[1])
        while i < n_dec1:
            x_final = self.decoder_blocks[1][i](x_final, conds=conds, start=start_pos, use_cache=True)
            i += 1

        logits = self.output_proj(x_final)
        return enc0_out, logits

    def _math_scale_3(self, x, start_pos: int, conds, enc0_prev_buf, dec0_state):
        x = self.embedding(x)
        x = self.norm(x)

        enc0_out = x
        i = 0
        n_enc0 = len(self.encoder_blocks[0])
        while i < n_enc0:
            enc0_out = self.encoder_blocks[0][i](enc0_out, conds=conds, start=start_pos, use_cache=True)
            i += 1

        enc0_buffer = torch.cat([enc0_prev_buf, enc0_out], dim=1)
        x_downsampled = self.downsamplers[0](enc0_buffer)
        i = 0
        n_enc1 = len(self.encoder_blocks[1])
        while i < n_enc1:
            x_downsampled = self.encoder_blocks[1][i](x_downsampled, conds=conds, start=start_pos // 3, use_cache=True)
            i += 1
        enc1_out = x_downsampled

        x_upsampled = self.skip_weights2[0] * dec0_state + enc1_out
        i = 0
        n_dec0 = len(self.decoder_blocks[0])
        while i < n_dec0:
            x_upsampled = self.decoder_blocks[0][i](x_upsampled, conds=conds, start=start_pos // 3, use_cache=True)
            i += 1

        dec1_buf_new = self.upsamplers[1](x_upsampled)
        dec1_state = dec1_buf_new[:, 0:1, :]
        x_final = self.skip_weights2[1] * dec1_state + enc0_out

        i = 0
        n_dec1 = len(self.decoder_blocks[1])
        while i < n_dec1:
            x_final = self.decoder_blocks[1][i](x_final, conds=conds, start=start_pos, use_cache=True)
            i += 1

        logits = self.output_proj(x_final)
        return enc0_out, enc1_out, dec1_buf_new, logits

    @torch.compiler.disable
    def _call_context_block(self, block, x, **kwargs):
        return block(x, **kwargs)

    def _math_scale_9(self, x, context, boundary_mask, start_pos: int, conds, enc0_prev_buf, enc1_prev_buf):
        if context is not None:
            context = self.context_embedding(context).reshape(context.shape[0], context.shape[1], -1)
            context = self.context_proj(context)

        x = self.embedding(x)
        x = self.norm(x)

        enc0_out = x
        i = 0
        n_enc0 = len(self.encoder_blocks[0])
        while i < n_enc0:
            enc0_out = self.encoder_blocks[0][i](enc0_out, conds=conds, start=start_pos, use_cache=True)
            i += 1

        enc0_buffer = torch.cat([enc0_prev_buf, enc0_out], dim=1)
        x_downsampled = self.downsamplers[0](enc0_buffer)
        i = 0
        n_enc1 = len(self.encoder_blocks[1])
        while i < n_enc1:
            x_downsampled = self.encoder_blocks[1][i](x_downsampled, conds=conds, start=start_pos // 3, use_cache=True)
            i += 1
        enc1_out = x_downsampled

        enc1_buffer = torch.cat([enc1_prev_buf, enc1_out], dim=1)
        x_bottleneck = self.downsamplers[1](enc1_buffer)

        i = 0
        n_botte = len(self.bottlenecke)
        while i < n_botte:
            block = self.bottlenecke[i]
            if i % 3 == 0:
                x_bottleneck = block(x_bottleneck, conds=conds, attention_mask=boundary_mask, start=start_pos // 9, use_cache=True)
            elif i % 3 == 1:
                x_bottleneck = block(x_bottleneck, conds=conds, start=start_pos // 9, use_cache=True)
            else:
                x_bottleneck = self._call_context_block(block, x_bottleneck, context=context, start=start_pos // 9, use_cache=True)
            i += 1

        inner_output = x_bottleneck.clone()

        i = 0
        n_bottd = len(self.bottleneckd)
        while i < n_bottd:
            block = self.bottleneckd[i]
            if i % 3 == 0:
                x_bottleneck = block(x_bottleneck, conds=conds, attention_mask=boundary_mask, start=start_pos // 9, use_cache=True)
            elif i % 3 == 1:
                x_bottleneck = block(x_bottleneck, conds=conds, start=start_pos // 9, use_cache=True)
            else:
                x_bottleneck = self._call_context_block(block, x_bottleneck, context=context, start=start_pos // 9, use_cache=True)
            i += 1

        dec0_buf_new = self.upsamplers[0](x_bottleneck)
        dec0_state = dec0_buf_new[:, 0:1, :]
        x_upsampled = self.skip_weights2[0] * dec0_state + enc1_out

        i = 0
        n_dec0 = len(self.decoder_blocks[0])
        while i < n_dec0:
            x_upsampled = self.decoder_blocks[0][i](x_upsampled, conds=conds, start=start_pos // 3, use_cache=True)
            i += 1

        dec1_buf_new = self.upsamplers[1](x_upsampled)
        dec1_state = dec1_buf_new[:, 0:1, :]
        x_final = self.skip_weights2[1] * dec1_state + enc0_out

        i = 0
        n_dec1 = len(self.decoder_blocks[1])
        while i < n_dec1:
            x_final = self.decoder_blocks[1][i](x_final, conds=conds, start=start_pos, use_cache=True)
            i += 1

        logits = self.output_proj(x_final)

        root_pred = inner_output
        i = 0
        n_root = len(self.root_pred_bottleneckd)
        while i < n_root:
            block = self.root_pred_bottleneckd[i]
            if i % 3 == 0:
                root_pred = block(root_pred, conds=conds, attention_mask=boundary_mask, start=start_pos // 9, use_cache=True)
            elif i % 3 == 1:
                root_pred = block(root_pred, conds=conds, start=start_pos // 9, use_cache=True)
            else:
                root_pred = self._call_context_block(block, root_pred, context=context, start=start_pos // 9, use_cache=True)
            i += 1
        root_pred = self.to_root(root_pred)

        return enc0_out, enc1_out, dec0_buf_new, dec1_buf_new, logits, root_pred
    
    def _process_single_token(self, x, context, boundary_mask, start_pos: int, conds=None):
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        cur_pos = start_pos[0]
        update_9 = (cur_pos + 1) % 9 == 0
        update_3 = (cur_pos + 1) % 3 == 0

        enc0_states = self.inference_state['layer_states']['encoder_0']
        enc1_states = self.inference_state['layer_states']['encoder_1']
        dec0_states = self.inference_state['upsampled_states']['decoder_0']
        dec1_states = self.inference_state['upsampled_states']['decoder_1']

        if update_9:
            enc0_prev = enc0_states[:, :2, :].clone()
            enc1_prev = enc1_states[:, :2, :].clone()

            enc0_out, enc1_out, dec0_buf, dec1_buf, logits, root_pred = self._math_scale_9(
                x, context, boundary_mask, start_pos, conds, enc0_prev, enc1_prev
            )

            self.inference_state['layer_states']['encoder_0'][:, cur_pos % 3 : cur_pos % 3 + 1] = enc0_out
            self.inference_state['layer_states']['encoder_1'][:, (cur_pos - 2) % 9 // 3 : (cur_pos - 2) % 9 // 3 + 1] = enc1_out
            self.inference_state['upsampled_states']['decoder_0'].copy_(dec0_buf.detach().clone())
            self.inference_state['upsampled_states']['decoder_1'].copy_(dec1_buf.detach().clone())
            logits = logits.detach().clone()

        elif update_3:
            enc0_prev = enc0_states[:, :2, :].clone()
            dec0_idx = ((cur_pos - 2) % 9 // 3 - 2) % 3
            dec0_state = dec0_states[:, dec0_idx:dec0_idx + 1].clone()

            enc0_out, enc1_out, dec1_buf, logits = self._math_scale_3(
                x, start_pos, conds, enc0_prev, dec0_state
            )

            self.inference_state['layer_states']['encoder_0'][:, cur_pos % 3 : cur_pos % 3 + 1] = enc0_out
            self.inference_state['layer_states']['encoder_1'][:, (cur_pos - 2) % 9 // 3 : (cur_pos - 2) % 9 // 3 + 1] = enc1_out
            self.inference_state['upsampled_states']['decoder_1'].copy_(dec1_buf.detach().clone())
            root_pred = None
            logits = logits.detach().clone()

        else:
            dec1_idx = (cur_pos - 2) % 3
            dec1_state = dec1_states[:, dec1_idx:dec1_idx + 1].clone()

            enc0_out, logits = self._math_scale_1(
                x, start_pos, conds, dec1_state
            )

            self.inference_state['layer_states']['encoder_0'][:, cur_pos % 3 : cur_pos % 3 + 1] = enc0_out
            root_pred = None
            logits = logits.detach().clone()

        return logits, root_pred

    def compile_math_kernels(self, compile_mode="max-autotune-no-cudagraphs"):
        """
        Compile static math kernels (like _math_scale_x) to accelerate inference before generating.
        """
        if not hasattr(torch, "compile"):
            print("Warning: torch.compile is not supported in this version of PyTorch.")
            return

        self._math_scale_1 = torch.compile(
            self._math_scale_1,
            mode=compile_mode,
            fullgraph=False,
            dynamic=False,
        )
        self._math_scale_3 = torch.compile(
            self._math_scale_3,
            mode=compile_mode,
            fullgraph=False,
            dynamic=False,
        )
        self._math_scale_9 = torch.compile(
            self._math_scale_9,
            mode=compile_mode,
            fullgraph=False,
            dynamic=False,
        )
        print(f"[{self.__class__.__name__}] Math kernels successfully compiled with mode: {compile_mode}")
    
    def generate(
        self,
        initial_input,
        init_context,
        init_attention_mask,
        init_cur_root_index,
        token_map,
        max_seq_len,
        device,
        accelerator,
        init_cur_root_index_total,
        pc=None,
        top_k: Optional[int] = 20,
        top_p: Optional[float] = 0.80,
        temperature: float = 1.0,
        return_cur_root=False,
        eos_aug = False,
        root_connect_constrain = True,
        wr_fix = True,
    ):
        """
        Generate sequence using the same interface as the original generate_sequence function
        """
        self.eval()
        input_len = initial_input.shape[1]
        generated = initial_input.to(device)
        generated_context = init_context.to(device)
        generated_attention_mask = init_attention_mask.to(device)
        generated_cur_root_index = init_cur_root_index_total.to(device)
        eos_token = token_map["eos"].to(device)
        token_map_gpu = {k: v.to(device) for k, v in token_map.items()}
        match = (generated.view(generated.shape[0], -1, 9) == eos_token).all(dim=-1)
        eos_mask = match.any(dim=1)  # shape: (b,)
        start = torch.full((generated.shape[0],), 0, device=device)
        # Initialize KV cache
        batch_size = generated.size(0)
        total_attention_mask = torch.ones(batch_size, max_seq_len//9, max_seq_len//9, device=device).bool()
        total_attention_mask[:, :init_attention_mask.shape[1], :init_attention_mask.shape[2]] = init_attention_mask.to(device)
        self.cache_size=self.init_kv_cache(batch_size, max_seq_len, dtype=torch.bfloat16)
        last_next_root = torch.max(init_cur_root_index,torch.tensor(0))
        cur_root_index = init_cur_root_index
        state = [{} for _ in range(batch_size)]
        stats = {
            'correction_count': 0,
            'total_steps': 0
        }
        total_predict = []
        wrong_data = [[] for _ in range(batch_size)]
        with torch.no_grad():
            if pc is not None and self.conditioned_on_pc:
                conds = self.conditioner(pc)
            else:
                conds = None  
            # First forward pass with the entire initial sequence
            with torch.amp.autocast(device_type="cuda",dtype=torch.bfloat16):
                output, class_next = self._process_first_tokens(
                            generated, 
                            generated_context, 
                            generated_attention_mask, 
                            start, 
                            conds=conds)

            start += generated.shape[1]

            recent_wr_per_batch = torch.zeros(batch_size, device=device)
            batch_wrong_counts = torch.zeros(batch_size, device=device)
            # Store the most recent 100 steps in a list (1 means wrong/corrected, 0 means correct)
            batch_recent_errors = [[] for _ in range(batch_size)]

            # Then generate one token at a time
            generate_tqdm = tqdm(
                range(max_seq_len - generated.size(1)),
                desc=f"Generate input_len {input_len}",
                disable=not accelerator.is_local_main_process
            )
            for _ in generate_tqdm:
                if eos_mask.all():
                    break

                # Get logits for the last position
                last_logits = output[:, -1, :]
                total_predict.append(last_logits)
                # Update root-node state
                if (start[0]) % 9 == 0:

                    max_root = (start[0] // 9).clone().detach()
                    max_root_move = max_root - last_next_root
                    recent_wr = torch.tensor([
                        sum(errors) / len(errors) if len(errors) > 0 else 0 
                        for errors in batch_recent_errors
                    ])
                    class_next_sampled = self.sample_with_constraints(
                            class_next[:,-1], 
                            max_values=max_root_move,
                            temperature=1,
                            top_k=1,
                            top_p=1
                    ).squeeze(1)    
                    if wr_fix:
                        condition = recent_wr_per_batch > 0.01
                        class_next = torch.clamp(torch.where(condition, max_root_move - 1, class_next_sampled), min=0)

                        intervened_batches = torch.where(condition)[0].tolist()
                        if intervened_batches and start[0] > 1000: # If any batch needs intervention
                            # print(f"WR Intervention! Batches {intervened_batches} forced to max_root_move.")
                            
                            # Reset all entries in the corresponding error histories to 0
                            for batch_idx in intervened_batches:
                                batch_recent_errors[batch_idx] = [0] * len(batch_recent_errors[batch_idx])
                    else:
                        class_next = class_next_sampled
                    cur_root_index = last_next_root + class_next
                    
                    last_next_root = cur_root_index

                    generated_cur_root_index = torch.cat([generated_cur_root_index, cur_root_index.unsqueeze(1)], dim=1)
                    
                # Apply temperature
                for batch_idx in range(generated.shape[0]):
                    if eos_mask[batch_idx]:
                        continue
                    original_prediction = torch.argmax(last_logits[batch_idx])
                    allowed_tokens = self.prefix_allowed_tokens_fn_with_state_non_manifold(
                        batch_idx, generated[batch_idx], token_map_gpu, cur_root_index[batch_idx].item(), state[batch_idx])

                    is_wrong = original_prediction.item() not in set(allowed_tokens)
                    if is_wrong:
                        wrong_data[batch_idx].append({
                            "batch_idx": batch_idx,
                            "idx": start[0],
                            "argmax_pred": original_prediction.item(),
                            "allowed_pred": allowed_tokens
                        })
                        stats['correction_count'] += 1
                        # print(wrong_data[batch_idx][-1])

                        batch_wrong_counts[batch_idx] += 1
                    batch_recent_errors[batch_idx].append(1 if is_wrong else 0)
                    if len(batch_recent_errors[batch_idx]) > 1000:
                        batch_recent_errors[batch_idx].pop(0)
                    recent_wr_per_batch[batch_idx] = sum(batch_recent_errors[batch_idx]) / len(batch_recent_errors[batch_idx])
                    stats['total_steps'] += 1
                    # 2. Create a mask and set all token logits to -inf first
                    # We only restore the original logits at allowed token positions
                    filtered_logits = torch.full_like(last_logits[batch_idx], -float("inf"))
                    
                    # 3. Fill the mask with the original logits of the allowed tokens
                    if root_connect_constrain:
                        filtered_logits[allowed_tokens] = last_logits[batch_idx][allowed_tokens]
                        if eos_aug and filtered_logits[token_map["eos"][0].item()] > -float("inf"):
                            # print(f"[batch:{batch_idx}, pos:{start[0]}] logits n: {filtered_logits[token_map["n"][0].item()]}")
                            # print(f"[batch:{batch_idx}, pos:{start[0]}] logits eos: {filtered_logits[token_map["eos"][0].item()]}")
                            # Add ln(5) at the logit level, which is equivalent to multiplying the post-softmax probability by 5
                            filtered_logits[eos_token[0].item()] += math.log(1.0)
                            if filtered_logits[token_map["eos"][0].item()] >  filtered_logits[token_map["n"][0].item()]:
                                filtered_logits[token_map["n"][0].item()] = -float("inf")

                    
                        # 4. Update the logits for this sequence
                        last_logits[batch_idx] = filtered_logits
                
                next_token = self.sample(last_logits, temperature, top_k, top_p)

                generated = torch.cat([generated, next_token], dim=1)
                wrong_rate = stats['correction_count'] / stats['total_steps']
                generate_tqdm.set_postfix({
                            'w_r': f'{wrong_rate:.4f}',      # Format as a float with 4 decimal places
                        })
                
                # udpate boundary mask
                if (start[0]+1) % 9 == 0:
                    eos_mask = update_eos_mask(
                            generated[:,-9:].view(generated.shape[0], 9),
                            eos_mask,
                            token_map
                        )
                    generated_context = torch.cat([generated_context, generated[:,-9:].reshape([generated.shape[0],1,9])], dim=1)
                    
                if (start[0]) % 9 == 0:
                    
                    cur_unflatten_len = start[0]//9
                    generated_attention_mask = torch.ones(batch_size, 1, cur_unflatten_len+1, device=device, dtype=torch.bool)
                    for i in range(generated_attention_mask.shape[0]):
                        cur_next_root_i = cur_root_index[i].item()
                        if eos_mask[i]:
                            generated_attention_mask[i, 0, -1] = False
                        else:
                            generated_attention_mask[i, 0, cur_next_root_i:] = False
                    total_attention_mask[:, cur_unflatten_len:cur_unflatten_len+1, :generated_attention_mask.shape[2]] = generated_attention_mask

                if self.cache_method=="sliding_window" and start[0]+1 >= 9000 and (start[0]+1-9000)%4500 == 0:
                    start_pos = torch.full((generated.shape[0],), 4500 + ((start[0]+1 - 9000)//4500)*4500, device=device)
                    with torch.amp.autocast(device_type="cuda",dtype=torch.bfloat16):
                        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                            torch.compiler.cudagraph_mark_step_begin()
                        output, class_next = self._process_muti_tokens(
                                    generated[:, start_pos[0]:start[0]+1], 
                                    generated_context, 
                                    total_attention_mask[:,start_pos[0]//9:start[0]//9+1,start_pos[0]//9:start[0]//9+1], 
                                    start_pos, 
                                    conds=conds
                        )

                else:
                    with torch.amp.autocast(device_type="cuda",dtype=torch.bfloat16):
                        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                            torch.compiler.cudagraph_mark_step_begin()
                        output, class_next = self._process_single_token(
                                    next_token, 
                                    generated_context, 
                                    generated_attention_mask[:,-1:], 
                                    start, 
                                    conds=conds)

                start = start + 1
        # Reset cache after generation
        self.reset_kv_cache()
        return generated.cpu()

    def sample_with_constraints(self, last_logits, max_values, temperature, top_k, top_p):
        """
        Apply Top-K sampling to model output logits with a dynamic upper-bound constraint.

        Args:
        class_next_logits: torch.Tensor, raw model logits with shape (b, seq_len, vocab_size)
        max_root: torch.Tensor, shape (b,)
        last_next_root: torch.Tensor, shape (b,)
        top_k: int, k value used in Top-K sampling
        
        Returns:
        torch.Tensor, sampled token indices with shape (b,)
        """
        
        # 1. Last-step logits
        b, vocab_size = last_logits.shape

        # Create a tensor of all possible indices with shape (b, vocab_size)
        # Example: [[0, 1, 2, ...], [0, 1, 2, ...], ...]
        indices = torch.arange(vocab_size, device=last_logits.device).expand(b, -1)
        
        # Create the mask using broadcasting.
        # For each row i, mask[i, j] is True when index j > max_values[i]
        # max_values.unsqueeze(1) reshapes it to (b, 1) for broadcasting
        mask = indices > max_values.unsqueeze(1)
        
        # Set logits that exceed the limit to a very small value
        # This makes their softmax probabilities approach 0
        last_logits[mask] = -float('inf')

        return self.sample(last_logits, temperature, top_k, top_p)


    def prefix_allowed_tokens_fn_with_state_non_manifold(self, batch_id, input_ids, token_map, root_idx, state: dict):
        """
        An autoregressive decoding helper that dynamically restricts the next allowed token
        according to the given topological constraints.
        """
        idx = input_ids.shape[0]
        pos = idx % 9

        # =================================================================
        # === 1. Initialization and special-case handling
        # =================================================================
        # End of a connected component; predict n or eos
        if root_idx >= idx//9 and pos == 0:
            all_allowed_tokens = []
            all_allowed_tokens.append(token_map['eos'][0].item())
            all_allowed_tokens.append(token_map['n'][0].item())
            return all_allowed_tokens
            
        # At the start of each new face (pos=0), find the root face in the generated sequence and store it in state
        if idx > 0 and pos == 0:
            # input_ids.view(-1, 9) reshapes all tokens into a list of faces
            # root_idx indicates which face is the root of the current new face
            root_token = input_ids.view(-1, 9)[root_idx]
            state["root_token"] = root_token
            # Reshape the 9 tokens of the root face into 3 vertices (vj1, vj2, vj3), each with 3 coordinates
            state["available_root_token"] = root_token.view(3, 3)

        # The first 9 tokens must be <s> (BOS)
        # Note: this assumes the BOS token sequence length is 8 and the 9th token is n.
        # If the first face is also a normal face, this may need adjustment.
        # Based on your code, I assume the first 8 tokens are BOS and the 9th starts n.
        if idx <= 7: 
            return [token_map['s'][0].item()]

        # Tokens 9 through 17 must be <n> (the start of a new connected component)
        if idx >= 8 and idx <= 16: 
            return [token_map['n'][0].item()]
        
        # Handle some special global tokens: if the previous token is special, all following coordinates must match it
        # (This logic may need fine-tuning based on your specific requirements)
        special_tokens = [t[0].item() for t in token_map.values() if t[0].item() > token_map['s'][0].item()]
        if pos > 0 and input_ids[-1].item() in special_tokens:
            return [input_ids[-1].item()]

        # Create a list of all valid vertex tokens and special tokens for later use
        all_allowed_tokens = list(range(0, token_map['s'][0].item()))
        all_allowed_tokens.append(token_map['eos'][0].item())
        all_allowed_tokens.append(token_map['n'][0].item())

        all_cord = list(range(0, token_map['s'][0].item()))

        # If the root token is <n>, this is the first face of a new connected component, so restrict to coordinates only
        if "root_token" in state and torch.equal(token_map["n"].flatten(), state["root_token"]):
            return all_cord

        # If state has not been initialized yet (for example, when generating the first token), allow any token
        if "available_root_token" not in state:
            return all_allowed_tokens

        # =================================================================
        # === 2. Core constraint logic
        # =================================================================
        
        # Get the three vertices of the root face from state
        vj1, vj2, vj3 = state["available_root_token"]

        # --- Constrain v_i1 (pos in [0, 1, 2]) ---
        # vi1 must be vj1 or vj3
        if pos == 0:  # Generate vi1_x
            # Allowed x coordinates are the x coordinates of vj1 and vj3
            allowed = list(set([vj1[0].item(), vj2[0].item(), vj3[0].item()]))
            # allowed.append(token_map['eos'][0].item())
            # all_allowed_tokens.append(token_map['n'][0].item())
            return allowed
        
        if pos == 1:  # Generate vi1_y
            # The allowed y coordinate depends on which vertex the previous x coordinate came from
            prev_token = input_ids[-1]
            allowed = []
            if prev_token == vj1[0]:  # If the previous step chose vj1's x
                allowed.append(vj1[1].item())
            if prev_token == vj2[0]:  # If the previous step chose vj2's x
                allowed.append(vj2[1].item())
            if prev_token == vj3[0]:  # If the previous step chose vj3's x
                allowed.append(vj3[1].item())
            return list(set(allowed)) # Use set to deduplicate in case vj1 and vj3 share the same x/y values

        if pos == 2:  # Generate vi1_z
            # The allowed z coordinate depends on which vertex the previous xy coordinates came from
            vi1_xy = input_ids[-2:]
            allowed = []
            if torch.equal(vi1_xy, vj1[:2]):  # If the previous two steps form vj1's xy
                allowed.append(vj1[2].item())
            if torch.equal(vi1_xy, vj2[:2]):  # If the previous two steps form vj3's xy
                allowed.append(vj2[2].item())
            if torch.equal(vi1_xy, vj3[:2]):  # If the previous two steps form vj3's xy
                allowed.append(vj3[2].item())
            
            return list(set(allowed))

        # --- Constrain v_i2 (pos in [3, 4, 5]) ---
        # The choice of vi2 depends entirely on the final result of vi1
        if pos == 3:  # Generate vi2_x
            chosen_vi1 = input_ids[-3:]
            possible_vi2 = []
            # v2 v1 | v1 v3 | v3 v2
            if torch.equal(chosen_vi1, vj2):  # If the previous two steps form vj2's xy
                possible_vi2.append(vj1)
            if torch.equal(chosen_vi1, vj1):  # If the previous two steps form vj3's xy
                possible_vi2.append(vj3)
            if torch.equal(chosen_vi1, vj3):  # If the previous two steps form vj3's xy
                possible_vi2.append(vj2)
                
            state['possible_vi2'] = possible_vi2
            allowed_x = list(set([v[0].item() for v in possible_vi2]))
            return allowed_x

        if pos == 4 or pos == 5:  # Generate vi2_y or vi2_z
            if 'possible_vi2' not in state:
                print("Warning: possible_vi2 not found in state when generating vi2_y or vi2_z.")
                # Fallback safely in this exceptional case
                return all_allowed_tokens
                
            # Get all possible vi2 vertices from state
            possible_vi2_vertices = state['possible_vi2']
            
            allowed_coords = []
            if pos == 4: # Generate y
                prev_token_x = input_ids[-1]
                # Filter candidate vertices whose x coordinate matches
                for v in possible_vi2_vertices:
                    if v[0] == prev_token_x:
                        allowed_coords.append(v[1].item())
            else: # pos == 5, generate z
                prev_tokens_xy = input_ids[-2:]
                # Filter candidate vertices whose xy coordinates match
                for v in possible_vi2_vertices:
                    if torch.equal(v[:2], prev_tokens_xy):
                        allowed_coords.append(v[2].item())
            return allowed_coords

        # --- Constrain v_i3 (pos in [6, 7, 8]) ---
        # vi3 can be sampled freely, except for special symbols
        if 6 <= pos and pos <= 7:
            return list(range(0, token_map['s'][0].item()))

        if pos == 8:
            # Constrain vi3 so it cannot match vi1/vi2
            # input_ids[-8:] = xyz xyz xy
            prev_tokens_xy = input_ids[-2:]
            chosen_vi1 = input_ids[-8:-5]
            chosen_vi2 = input_ids[-5:-2]

            not_allowed_coords = set()
            if torch.equal(chosen_vi1[:2], prev_tokens_xy):
                not_allowed_coords.add(chosen_vi1[2].item())
            if torch.equal(chosen_vi2[:2], prev_tokens_xy):
                not_allowed_coords.add(chosen_vi2[2].item())
            
            return list(id for id in range(0, token_map['s'][0].item()) if id not in not_allowed_coords)


        # In theory all positions should be covered, but provide a final fallback for safety
        print("wrong")
        return all_allowed_tokens


    def sample(self, last_logits, temperature, top_k, top_p):
        """
        A more robust sampling function that correctly handles the mutual exclusivity of Top-K and Top-P.
        """
        # 1. Apply temperature
        # Ensure temperature > 0 to avoid division by zero
        if temperature > 0:
            logits = last_logits / temperature
        else:
            logits = last_logits  # temperature=0 is equivalent to greedy search

        # Compute the probability distribution
        probs = F.softmax(logits, dim=-1)

        # 2. [Core change] Apply Top-K or Top-P filtering mutually exclusively
        # Prioritize Top-K; if Top-K is not active, then consider Top-P
        if top_k is not None and top_k > 0:
            # Ensure k does not exceed the vocabulary size
            top_k = min(top_k, probs.size(-1))

            # Zero out all probabilities except top_k
            # topk() returns sorted values, so we can use them directly
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)

            # Create a new all-zero tensor and fill probabilities only at topk positions
            filtered_probs = torch.zeros_like(probs)
            filtered_probs.scatter_(1, topk_indices, topk_probs)
            probs = filtered_probs

        elif top_p is not None and 0.0 < top_p < 1.0:
            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

            # Compute cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Identify tokens to remove: those beyond cumulative probability top_p
            # Also, keep at least one token
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep the first token marked for removal to ensure at least one candidate remains
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Create a mask to mark tokens that should be removed
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

            # Zero out the probabilities of tokens to remove
            probs = probs.masked_fill(indices_to_remove, 0.0)

        # 3. [Improvement] Handle the all-zero probability edge case
        # If all probabilities are zero, fall back to greedy sampling to avoid multinomial errors (pick max original logit)
        if torch.all(probs == 0):
            # Return the token index with the highest original logit
            return torch.argmax(last_logits, dim=-1, keepdim=True)

        # 4. [Improvement] Re-normalize probabilities
        probs = probs / probs.sum(-1, keepdim=True)


        # 5. Sample from the final probability distribution
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token