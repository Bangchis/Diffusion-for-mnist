# unet.py
# Lightweight UNet for MNIST:
# - Optional outer attention disabled (Identity)
# - Full attention only at bottleneck


import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def divisible_by(x, y):
    return x % y == 0


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.weight * self.scale


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=device)
                          * -(torch.log(torch.tensor(self.theta)) / (half - 1)))
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.):
        super().__init__()

        self.project = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x, shift_scale=None):
        x = self.project(x)
        x = self.norm(x)
        if shift_scale is not None:
            s, b = shift_scale
            x = x * (s + 1) + b
        x = self.dropout(self.activation(x))

        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(
            time_emb_dim, dim_out * 2)) if time_emb_dim else None
        self.b1 = Block(dim, dim_out, dropout=dropout)
        self.b2 = Block(dim_out, dim_out, dropout=0.)
        self.skip = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, t=None):
        scale_shift = None
        if self.mlp is not None and t is not None:
            emb = self.mlp(t).view(t.size(0), -1, 1, 1)
            scale_shift = emb.chunk(2, dim=1)
        h = self.b1(x, scale_shift)
        h = self.b2(h)
        return h + self.skip(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=2, dim_head=16):
        super().__init__()
        self.heads = heads
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim_head * heads * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(dim_head * heads, dim, 1), RMSNorm(dim))
        self.scale = dim_head ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = rearrange(q, 'b (h d) x y -> b h d (x y)', h=self.heads)
        k = rearrange(k, 'b (h d) x y -> b h d (x y)', h=self.heads)
        v = rearrange(v, 'b (h d) x y -> b h d (x y)', h=self.heads)
        q = torch.softmax(q, dim=-2) * self.scale
        k = torch.softmax(k, dim=-1)
        ctx = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', ctx, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', x=h, y=w)
        return self.to_out(out)


class FullAttention(nn.Module):
    def __init__(self, dim, heads=2, dim_head=16):
        super().__init__()
        self.heads = heads
        inner = heads * dim_head
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, inner * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner, dim, 1)
        self.scale = dim_head ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.heads)
        k = rearrange(k, 'b (h d) x y -> b h (x y) d', h=self.heads)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=self.heads)
        attn = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)
        out = attn @ v
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out) + x


class UNet(nn.Module):
    """
    Minimal UNet for MNIST 32x32.
    - outer_attn=False -> use Identity in outer levels
    - FullAttention at bottleneck only
    """

    def __init__(self, dim=32, init_dim=None, out_dim=None, dim_mults=(1, 2, 4),
                 channels=1, dropout=0.0, attn_heads=2, attn_dim_head=16,
                 self_condition=False, learned_variance=False, outer_attn=False):
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        self.learned_variance = learned_variance

        in_ch = channels * (2 if self_condition else 1)
        init_dim = init_dim or dim
        self.init_conv = nn.Conv2d(in_ch, init_dim, 7, padding=3)

        dims = [init_dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for i, (d_in, d_out) in enumerate(in_out):
            is_last = i == (len(in_out) - 1)
            attn_mod = LinearAttention(
                d_in, heads=attn_heads, dim_head=attn_dim_head) if outer_attn else nn.Identity()
            self.downs.append(nn.ModuleList([
                ResnetBlock(d_in, d_in, time_emb_dim=time_dim,
                            dropout=dropout),
                ResnetBlock(d_in, d_in, time_emb_dim=time_dim,
                            dropout=dropout),
                attn_mod,
                (nn.Conv2d(d_in, d_out, 3, padding=1) if is_last else
                 nn.Sequential(nn.Conv2d(d_in, d_in, 4, stride=2, padding=1),
                               nn.Conv2d(d_in, d_out, 3, padding=1)))
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, dropout=dropout)
        self.mid_attn = FullAttention(
            mid_dim, heads=attn_heads, dim_head=attn_dim_head)  # bottleneck
        self.mid_block2 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, dropout=dropout)

        for i, (d_in, d_out) in enumerate(reversed(in_out)):
            is_last = i == (len(in_out) - 1)
            attn_mod_up = LinearAttention(
                d_out, heads=attn_heads, dim_head=attn_dim_head) if outer_attn else nn.Identity()
            self.ups.append(nn.ModuleList([
                ResnetBlock(d_out + d_in, d_out,
                            time_emb_dim=time_dim, dropout=dropout),
                ResnetBlock(d_out + d_in, d_out,
                            time_emb_dim=time_dim, dropout=dropout),
                attn_mod_up,
                (nn.Conv2d(d_out, d_in, 3, padding=1) if is_last else
                 nn.Sequential(nn.ConvTranspose2d(d_out, d_out, 4, stride=2, padding=1),
                               nn.Conv2d(d_out, d_in, 3, padding=1)))
            ]))

        self.out_dim = out_dim or channels  # learned_variance=False for MNIST
        self.final_res_block = ResnetBlock(
            init_dim * 2, init_dim, time_emb_dim=time_dim, dropout=dropout)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(
        self): return 2 ** (len(self.downs) - 1)  # (len=3) -> 4

    def forward(self, x, time, x_self_cond=None):
        assert all(divisible_by(d, self.downsample_factor)
                   for d in x.shape[-2:])
        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat([x_self_cond, x], dim=1)

        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)

        hs = []
        for b1, b2, attn, down in self.downs:
            x = b1(x, t)
            hs.append(x)
            x = b2(x, t)
            x = attn(x) + x
            hs.append(x)
            x = down(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for b1, b2, attn, up in self.ups:
            x = torch.cat([x, hs.pop()], dim=1)
            x = b1(x, t)
            x = torch.cat([x, hs.pop()], dim=1)
            x = b2(x, t)
            x = attn(x) + x
            x = up(x)

        x = torch.cat([x, r], dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)
