# -*- coding: utf-8 -*-
"""Depth network and CRF-modulated depth refinement (BEVNeXt component)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG, Config


def conv_bn(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DepthNet(nn.Module):
    """Shared multi-view backbone + per-pixel depth distribution at stride 2."""

    def __init__(self, cfg: Config = CFG, channels: int = 48):
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Sequential(
            conv_bn(5, 32, 3, 2, 1),   # 24 x 40, RGB + pixel-coordinate prior
            conv_bn(32, 64, 3, 1, 1),
            conv_bn(64, channels, 3, 1, 1),
        )
        self.cam_emb = nn.Embedding(cfg.n_cams, channels)
        self.depth_head = nn.Sequential(
            conv_bn(channels, channels, 3, 1, 1),
            nn.Conv2d(channels, cfg.depth_bins, 1),
        )

    def forward(self, images: torch.Tensor, cam_ids: torch.Tensor | None = None):
        # images: (B, Ncam, 3, H, W)
        b, n, c, h, w = images.shape
        x = images.reshape(b * n, c, h, w)
        # normalised pixel-coordinate channels: a camera-aware positional prior
        u = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b * n, 1, h, w)
        v = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b * n, 1, h, w)
        x = torch.cat([x, u, v], 1)
        f = self.stem(x)
        if cam_ids is None:
            cam_ids = torch.arange(n, device=images.device).repeat(b)
        f = f + self.cam_emb(cam_ids).view(b * n, -1, 1, 1)
        logits = self.depth_head(f)
        return f, logits


class CRFMeanField(nn.Module):
    """A small mean-field CRF (Zheng et al., 2015) used to modulate depth logits.

    The pairwise term uses image colour similarity and a local spatial Gaussian,
    with a learnable label-compatibility matrix over depth bins.
    """

    def __init__(self, bins: int = 32, iters: int = 3, radius: int = 2,
                 depth_min: float = 2.0, depth_max: float = 50.0):
        super().__init__()
        self.bins = bins
        self.iters = iters
        self.radius = radius
        centers = torch.linspace(depth_min, depth_max, bins)
        step = (depth_max - depth_min) / max(bins - 1, 1)
        compat = -torch.abs(centers[:, None] - centers[None, :]) / (step * bins)
        self.compat = nn.Parameter(compat)
        self.theta_a = nn.Parameter(torch.tensor(0.12))
        self.theta_b = nn.Parameter(torch.tensor(1.8))

    @staticmethod
    def _shift(x: torch.Tensor, di: int, dj: int, pad: int):
        xp = F.pad(x, (pad, pad, pad, pad))
        h, w = x.shape[-2:]
        return xp[:, :, pad + di:pad + di + h, pad + dj:pad + dj + w]

    def forward(self, logits: torch.Tensor, image: torch.Tensor):
        # logits: (B, D, H, W); image: (B, 3, H, W)
        pad = self.radius
        q = torch.softmax(logits, dim=1)
        theta_a = F.softplus(self.theta_a) + 1e-3
        theta_b = F.softplus(self.theta_b) + 1e-3
        for _ in range(self.iters):
            msg = torch.zeros_like(q)
            for di in range(-pad, pad + 1):
                for dj in range(-pad, pad + 1):
                    if di == 0 and dj == 0:
                        continue
                    img_j = self._shift(image, di, dj, pad)
                    q_j = self._shift(q, di, dj, pad)
                    col = torch.exp(-((image - img_j) ** 2).sum(1, keepdim=True) / (2 * theta_a ** 2))
                    spa = torch.exp(-(di * di + dj * dj) / (2 * theta_b ** 2))
                    msg = msg + col * spa * q_j
            msg = msg / ((2 * pad + 1) ** 2 - 1)
            msg = torch.einsum("bdhw,dc->bchw", msg, self.compat)
            q = torch.softmax(logits - msg, dim=1)
        return q
