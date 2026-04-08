import torch


class Rotary(torch.nn.Module):
    def __init__(self, dim: int, max_seq_len=9 * 500000):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.bfloat16)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.bfloat16)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, x, x_cu_seqlen, offset, stride: int = 1):
        seq_lens = x_cu_seqlen[1:] - x_cu_seqlen[:-1]
        
        if type(offset) == int:
            offset = torch.tensor([offset] * len(seq_lens), device=x.device).long()

        assert seq_lens.size(0) == offset.size(0)

        # positions = torch.cat([
        #     torch.arange(off, seq_len * stride + off, stride, device=x.device, dtype=torch.long)
        #     for seq_len, off in zip(seq_lens, offset)
        # ])
        # assert (seq_lens == seq_lens[0]).all()
        positions = offset.view(-1, 1) + stride * torch.arange(
            seq_lens[0], device=x.device, dtype=torch.long
        ).view(1, -1)
        positions = positions.flatten()

        cos = self.cos[positions].unsqueeze(-2)
        sin = self.sin[positions].unsqueeze(-2)

        x1, x2 = x.chunk(2, dim=-1)
        y1 = torch.addcmul(x1 * cos, x2, sin)
        y2 = torch.addcmul(x2 * cos, x1, -sin)
        return torch.cat((y1, y2), dim=-1)


if __name__ == '__main__':
    net = Rotary(dim=8)
    print(net(torch.randn(7, 4, 8), torch.Tensor([0, 2, 7]), offset=torch.Tensor([1, 3]).int(), stride=3).shape)
