# -*- coding: utf-8 -*-
"""Temporal fusion modules: Res2Fusion (BEVNeXt) and ConvGRU (StreamMapNet)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Res2Fusion(nn.Module):
    """Res2Net-style multi-scale temporal aggregation over k BEV frames."""

    def __init__(self, channels: int = 64, k: int = 4, window: int = 2):
        super().__init__()
        self.k, self.w = k, window
        self.g = (k + window - 1) // window
        self.group = nn.ModuleList([nn.Conv2d(window * channels, channels, 1) for _ in range(self.g)])
        self.ms = nn.ModuleList([nn.Conv2d(channels, channels, 3, 1, 1) for _ in range(self.g)])
        self.final = nn.Conv2d(self.g * channels, channels, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, feats):
        # feats: list, oldest -> newest, each (B,C,H,W)
        feats = list(feats)
        while len(feats) < self.k:
            feats.insert(0, torch.zeros_like(feats[0]))
        feats = feats[-self.k:]
        k = len(feats)
        # group newest first: group 0 = [t, t-1], group 1 = [t-2, t-3]
        groups = []
        for gi in range(self.g):
            chunk = feats[max(0, k - (gi + 1) * self.w): k - gi * self.w]
            chunk = chunk[::-1]  # newest first inside group
            while len(chunk) < self.w:
                chunk.insert(0, torch.zeros_like(feats[0]))
            groups.append(torch.cat(chunk, dim=1))
        b = [self.act(self.group[i](groups[i])) for i in range(self.g)]
        # hierarchical multi-scale, oldest (g-1) -> newest (0)
        out = [None] * self.g
        out[self.g - 1] = self.act(self.ms[self.g - 1](b[self.g - 1]))
        for i in range(self.g - 2, -1, -1):
            out[i] = self.act(self.ms[i](b[i] + out[i + 1]))
        return self.final(torch.cat(out, dim=1))


class ConvGRU(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.gz = nn.Conv2d(2 * channels, channels, 3, 1, 1)
        self.gr = nn.Conv2d(2 * channels, channels, 3, 1, 1)
        self.gc = nn.Conv2d(2 * channels, channels, 3, 1, 1)

    def forward(self, h, x):
        if h is None:
            h = torch.zeros_like(x)
        z = torch.sigmoid(self.gz(torch.cat([h, x], 1)))
        r = torch.sigmoid(self.gr(torch.cat([h, x], 1)))
        c = torch.tanh(self.gc(torch.cat([r * h, x], 1)))
        return (1 - z) * h + z * c


def warp_bev(bev: torch.Tensor, dx_cells: int):
    """Shift a previous BEV (x = rows) backwards by ego forward motion."""
    if dx_cells == 0:
        return bev
    out = torch.zeros_like(bev)
    if dx_cells > 0:
        if dx_cells < bev.shape[-2]:
            out[:, :, :-dx_cells] = bev[:, :, dx_cells:]
    else:
        s = -dx_cells
        if s < bev.shape[-2]:
            out[:, :, s:] = bev[:, :, :-s]
    return out


def compress_bev(bev: torch.Tensor, unc: torch.Tensor, budget: float = 0.25,
                 mode: str = "uncertainty", pool: int = 4):
    """Salient-cell + background-downsample compression.

    Returns a dict with reconstruction, byte estimate and keep mask.
    """
    b, c, h, w = bev.shape
    n = h * w
    keep_n = max(1, int(round(budget * n)))
    if mode == "uncertainty":
        score = unc.reshape(b, -1) * bev.detach().abs().mean(1).reshape(b, -1)
    elif mode == "magnitude":
        score = bev.detach().abs().mean(1).reshape(b, -1)
    else:
        score = torch.rand(b, n, device=bev.device)
    top = score.topk(keep_n, dim=1).indices  # B,K
    bg = F.avg_pool2d(bev, pool, pool, ceil_mode=True)
    rec = F.interpolate(bg, size=(h, w), mode="bilinear", align_corners=False)
    flat = rec.reshape(b, c, n)
    vals = torch.gather(bev.reshape(b, c, n), 2, top[:, None, :].expand(b, c, keep_n))
    flat.scatter_(2, top[:, None, :].expand(b, c, keep_n), vals)
    rec = flat.reshape(b, c, h, w)
    # memory bytes (fp16): salient cells + pooled background
    sal_bytes = keep_n * (c + 1) * 2
    bg_bytes = c * (h // pool) * (w // pool) * 2
    return {"rec": rec, "bytes": sal_bytes + bg_bytes, "keep": top,
            "full_bytes": c * h * w * 2}
