from torch import nn
import torch
import math


class Block(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.ff = nn.Linear(channels, channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(
        self, channels_data=2, layers=5, channels=512, channels_t=512, t_dof=1
    ):
        super().__init__()
        self.channels_t = channels_t * t_dof
        self.in_projection = nn.Linear(channels_data, channels)
        self.t_projection = nn.Linear(self.channels_t, channels)
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(layers)])
        self.out_projection = nn.Linear(channels, channels_data * t_dof)
        self.t_dof = t_dof

    def forward(self, x, t, verbose=False):
        B, C = x.shape
        if verbose:
            print("x", x.shape)
            print("t", t.shape)
        x = self.in_projection(x)
        if verbose:
            print("x proj", x.shape)
        t = gen_t_embedding(t)
        if verbose:
            print("t emb", t.shape)
        t = self.t_projection(t)
        if verbose:
            print("t proj", t.shape)
        x = x + t
        x = self.blocks(x)
        if verbose:
            print("x block", x.shape)
        x = self.out_projection(x)
        if verbose:
            print("x out proj", x.shape)
        return x.view(B, C, self.t_dof)


def gen_t_embedding(t, channels_t=512, max_positions=10000):
    B = t.shape[0]
    t = t * max_positions
    # print(t)
    half_dim = channels_t // 2
    # print(half_dim)
    emb = math.log(max_positions) / (half_dim - 1)
    # print("math", emb)
    emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
    # print("arange", emb.shape, emb.min(), emb.max())
    emb = t[..., None] * emb[None, None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    # print(emb.shape)
    if channels_t % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), mode="constant")
    # print(emb.min(), emb.max(), emb.shape)
    return emb.view(B, -1)
