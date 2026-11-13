# -*- coding: utf-8 -*-
"""Lift-Splat-Shoot view transformer with uncertainty export."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import CFG, Config


def _rays_at(cfg: Config, h: int, w: int):
    """Rays for a down-sampled grid (stride = img_h/h)."""
    sy, sx = cfg.img_h / h, cfg.img_w / w
    vs, us = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    u = ((us + 0.5) * sx - cfg.img_w / 2.0) / cfg.focal
    v = ((vs + 0.5) * sy - cfg.img_h / 2.0) / cfg.focal
    orig, dirs = [], []
    for yaw_deg in cfg.cam_yaws:
        a = math.radians(yaw_deg)
        p = math.radians(cfg.cam_pitch_deg)
        fwd = torch.tensor([math.cos(a) * math.cos(p), math.sin(a) * math.cos(p), math.sin(p)])
        right = torch.tensor([math.sin(a), -math.cos(a), 0.0])
        down = torch.cross(fwd, right, dim=-1)
        d = fwd[None, None, :] + u[..., None] * right[None, None, :] + v[..., None] * down[None, None, :]
        d = d / d.norm(dim=-1, keepdim=True)
        orig.append(torch.tensor([0.0, 0.0, cfg.cam_height]).expand(h, w, 3))
        dirs.append(d)
    return torch.stack(orig), torch.stack(dirs)  # (N,H,W,3)


class LSSViewTransform(nn.Module):
    def __init__(self, cfg: Config = CFG, h: int = 24, w: int = 40):
        super().__init__()
        self.cfg = cfg
        self.h, self.w = h, w
        orig, dirs = _rays_at(cfg, h, w)
        depth = torch.linspace(cfg.depth_min, cfg.depth_max, cfg.depth_bins)
        # use bin centres
        step = (cfg.depth_max - cfg.depth_min) / cfg.depth_bins
        depth = cfg.depth_min + (torch.arange(cfg.depth_bins) + 0.5) * step
        pts = orig[..., None, :] + dirs[..., None, :] * depth[None, None, None, :, None]
        ix = torch.floor((pts[..., 0] - cfg.x_min) / cfg.res).long()
        iy = torch.floor((pts[..., 1] - cfg.y_min) / cfg.res).long()
        iz = pts[..., 2]
        valid = (ix >= 0) & (ix < cfg.bev_h) & (iy >= 0) & (iy < cfg.bev_w) & (iz > -0.3) & (iz < 4.0)
        idx = (ix.clamp(0, cfg.bev_h - 1) * cfg.bev_w + iy.clamp(0, cfg.bev_w - 1))
        self.register_buffer("bev_idx", idx.reshape(cfg.n_cams, h * w, cfg.depth_bins))
        self.register_buffer("valid", valid.reshape(cfg.n_cams, h * w, cfg.depth_bins))
        self.register_buffer("depth_centers", depth)
        self.out_hw = cfg.bev_h * cfg.bev_w

    def forward(self, feat: torch.Tensor, prob: torch.Tensor, unc: torch.Tensor | None = None):
        # feat: (B,N,C,H,W), prob: (B,N,D,H,W), unc: (B,N,H,W)
        b, n, c, h, w = feat.shape
        d = prob.shape[2]
        hw = h * w
        f = feat.reshape(b, n, c, hw)
        p = prob.reshape(b, n, d, hw)
        bev = feat.new_zeros(b, c, self.out_hw)
        cnt = feat.new_zeros(b, 1, self.out_hw)
        valid = self.valid[None].to(feat.dtype)  # (1,N,HW,D)
        for di in range(d):
            wd = p[:, :, di] * valid[:, :, :, di]           # B,N,HW
            # index layout must match the (camera, pixel) flattening of wd
            idx = self.bev_idx[:, :, di].reshape(1, -1).expand(b, c, -1)
            bev.scatter_add_(2, idx, (f * wd[:, :, None, :]).reshape(b, c, -1))
            cnt.scatter_add_(2, idx[:, :1], wd.reshape(b, 1, -1))
        bev = bev / cnt.clamp_min(1.0)
        bev = bev.reshape(b, c, self.cfg.bev_h, self.cfg.bev_w)
        if unc is None:
            return bev, None
        # map per-pixel entropy to the BEV cell of the most likely depth bin
        with torch.no_grad():
            am = p.argmax(dim=2)  # B,N,HW
            idx_pix = torch.gather(self.bev_idx[None].expand(b, -1, -1, -1), 3,
                                   am[:, :, None, :]).squeeze(2)  # B,N,HW
            ok = torch.gather(valid.expand(b, -1, -1, -1), 3,
                              am[:, :, None, :]).squeeze(2)  # B,N,HW
            u = unc.reshape(b, n, hw) * ok
            ubev = feat.new_zeros(b, 1, self.out_hw)
            ucnt = feat.new_zeros(b, 1, self.out_hw)
            ubev.scatter_add_(2, idx_pix.reshape(b, 1, -1), u.reshape(b, 1, -1))
            ucnt.scatter_add_(2, idx_pix.reshape(b, 1, -1), ok.reshape(b, 1, -1))
            ubev = ubev / ucnt.clamp_min(1e-4)
        return bev, ubev.reshape(b, 1, self.cfg.bev_h, self.cfg.bev_w)
