import time

import torch

from nsa import RopeConfig
from nsa.module import RotaryEmbedding
from nsa.module.hrope import Rotary

if __name__ == '__main__':
    N = 25600
    dim = 256
    head_dim = 12
    test_step = 256

    rope = Rotary(dim=dim, max_seq_len=16 * 9 * 50000).cuda()
    rope_triton = RotaryEmbedding(
        RopeConfig(
            max_position_embeddings=131072,
            head_dim=dim,
            rope_theta=500000,
            rope_scaling={
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            })
    ).cuda()

    x = torch.randn(N, head_dim, dim).cuda()
    cu_seqlens = torch.Tensor([0, N]).long().cuda()
    stride = 16
    offset = 16
    offset_tensor = torch.Tensor([offset]).long().cuda()

    for _ in range(test_step):
        rope(x, cu_seqlens, offset_tensor, stride)
        rope_triton(x, cu_seqlens, offset, stride)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    for _ in range(test_step):
        rope(x, cu_seqlens, offset_tensor, stride)
    print(f"rope: {time.time() - start_time:.3f}s")
    print(torch.cuda.max_memory_allocated())

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    for _ in range(test_step):
        rope_triton(x, cu_seqlens, offset, stride)
    print(f"triton_rope: {time.time() - start_time:.3f}s")
    print(torch.cuda.max_memory_allocated())

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    for _ in range(test_step):
        rope_triton.forward_dev(x, cu_seqlens, offset_tensor, stride)
    print(f"triton_rope: {time.time() - start_time:.3f}s")
    print(torch.cuda.max_memory_allocated())