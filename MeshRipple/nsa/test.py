# This file is modified from the original implementation (implemented by Xunhao Lai)
import argparse

import torch

from nsa.module import NativeSparseAttention as NSA_t, RopeConfig as RopeConfig_t

if __name__ == "__main__":
    torch.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--seqlens", nargs="+", type=int, default=[256])
    parser.add_argument("--kv-seqlen", type=int, default=256)
    parser.add_argument("--kv-seqlens", nargs="+", type=int, default=[512])
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--kv-heads", type=int, default=-1)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--gqa-deg", type=int, default=1)
    parser.add_argument('--topk', type=int, default=64)
    parser.add_argument('--attn-mode', type=str, default="NSA", choices=["NSA"])
    parser.add_argument("--kernel-stride", type=int, default=16)
    parser.add_argument("--nseqs", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--benchmark-iters", type=int, default=5)
    parser.add_argument("--benchmark-bwd", action="store_true")
    parser.add_argument("--dtype", type=str, default="float16", choices=["bfloat16", "float16", "float32"])

    args = parser.parse_args()
    DTYPE = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32)[args.dtype]
    seqlen = args.seqlen
    kv_seqlen = args.kv_seqlen

    if args.kv_heads > 0:
        q_heads = args.kv_heads * args.gqa_deg
        kv_heads = args.kv_heads
    else:
        q_heads = args.heads
        kv_heads = args.heads // args.gqa_deg
    assert q_heads % args.gqa_deg == 0

    NativeSparseAttention = None
    RopeConfig = None
    if args.attn_mode == "NSA":
        NativeSparseAttention = NSA_t
        RopeConfig = RopeConfig_t
    else:
        raise ValueError("attn_mode must be NSA or NSA_pt")

    sparse_attn = (
        NativeSparseAttention(
            hidden_size=args.hidden_size,
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            head_dim=args.head_dim,
            kernel_size=32,
            kernel_stride=args.kernel_stride,
            block_size=args.block_size,
            topk=args.topk,
            init_blocks=1,
            local_blocks=2,
            window_size=512,
            rope_config=RopeConfig(
                max_position_embeddings=131072,
                head_dim=args.head_dim,
                rope_theta=500000,
                rope_scaling={
                    "factor": 8.0,
                    "high_freq_factor": 4.0,
                    "low_freq_factor": 1.0,
                    "original_max_position_embeddings": 8192,
                    "rope_type": "llama3",
                },
            ),
            debug_mode=True,
        )
        .cuda()
        .to(DTYPE)
    )
    print(f"======= Num Heads: {args.attn_mode} =======\n")

    print(f"q_heads={q_heads}, kv_heads={kv_heads}\n")

    print(f"======= Init Moduel: {args.attn_mode} =======\n")
    for name, param in sparse_attn.named_parameters():
        print(f"{args.attn_mode} Parameters, {name}, shape: {param.shape}\n")

    # random input
    if args.nseqs > 1:
        seqlens = torch.LongTensor([seqlen] * args.nseqs).int().cuda()
        kv_seqlens = torch.LongTensor([kv_seqlen] * args.nseqs).int().cuda()
    else:
        seqlens = torch.LongTensor(args.seqlens).int().cuda()
        kv_seqlens = torch.LongTensor(args.kv_seqlens).int().cuda()

    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    kv_cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(kv_seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)

    q = torch.randn(cu_seqlens[-1].item(), args.hidden_size, device="cuda", dtype=DTYPE)
    k = torch.randn(kv_cu_seqlens[-1].item(), args.hidden_size, device="cuda", dtype=DTYPE)
    v = torch.randn(kv_cu_seqlens[-1].item(), args.hidden_size, device="cuda", dtype=DTYPE)

    # warmup
    print(f"======= {args.attn_mode} Forward & Backward Performance Test =======\n")
    for i in range(4):
        if args.attn_mode == "NSA_o":
            y = sparse_attn(q, cu_seqlens)
        else:
            y = sparse_attn(q, k, v, cu_seqlens, kv_cu_seqlens)
        if args.benchmark_bwd:
            loss = (y * torch.randn_like(y)).sum(-1).mean()
            loss.backward()

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    num_iters = args.benchmark_iters
    for i in range(num_iters):
        if args.attn_mode == "NSA_o":
            y = sparse_attn(q, cu_seqlens)
        else:
            y = sparse_attn(q, k, v, cu_seqlens, kv_cu_seqlens)
        if args.benchmark_bwd:
            loss = (y * torch.randn_like(y)).sum(-1).mean()
            loss.backward()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / num_iters

    if args.benchmark_bwd:
        benchmark_mode = "Fwd+Bwd"
    else:
        benchmark_mode = "Fwd"
    print(f"[{args.attn_mode} E2E ({benchmark_mode})] Time: {elapsed_ms:.3f} ms\n")

    print(f"======= {args.attn_mode} Forward & Backward Output Test =======\n")
    if args.attn_mode == "NSA_o":
        y = sparse_attn(q, cu_seqlens)
    else:
        y = sparse_attn(q, k, v, cu_seqlens, kv_cu_seqlens)
    print(f"Forward, output shape: {y.shape}, output norm: {y.norm()}\n")

    # backward test
    loss = (y * torch.randn_like(y)).sum(-1).mean()
    loss.backward()
    for name, param in sparse_attn.named_parameters():
        print(f"Backward, {name}, grad shape: {param.grad.shape}, grad norm: {param.grad.norm()}\n")

    print('[Max allocated]:', torch.cuda.max_memory_allocated() / 1024 ** 3)
