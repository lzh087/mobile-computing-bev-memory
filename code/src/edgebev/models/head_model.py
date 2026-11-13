# -*- coding: utf-8 -*-
"""Head-only temporal model operating on cached BEV features.

This isolates the temporal-fusion + detection/map-head training from the depth
backbone, which is trained separately in stage 1 (see pretrain_depth.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import CFG, Config
from .depth import conv_bn
from .fusion import ConvGRU, Res2Fusion, compress_bev
from .heads import DetectionHead, MapDecoder, OccHead


class HeadModel(nn.Module):
    def __init__(self, cfg: Config = CFG, temporal: str = "res2", map_temporal: str = "gru",
                 query_prop: int = 10):
        super().__init__()
        self.cfg = cfg
        self.temporal = temporal
        self.map_temporal = map_temporal
        self.query_prop = query_prop
        self.res2 = Res2Fusion(64, k=4, window=2)
        self.gru = ConvGRU(64)
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                conv_bn(64, 48, 3, 1))
        self.det = DetectionHead(48, 2)
        self.occ = OccHead(48)
        self.map = MapDecoder(48, 128, 40, 12, 3, 2, 4)
        self.map.cfg = cfg
        self.q_trans = nn.Linear(1, 128)

    def fuse_det(self, bevs, uncs, umc: dict | None = None):
        if self.temporal == "none":
            return bevs[-1]
        if self.temporal == "gru":
            h = None
            for b in bevs[:-1]:
                h = self.gru(h, b.detach())
            return self.gru(h, bevs[-1])
        k = min(4, len(bevs))
        feats = [b.detach() for b in bevs[-k - 1:-1]] + [bevs[-1]]
        if len(feats) < 4:
            feats = [torch.zeros_like(feats[0])] * (4 - len(feats)) + feats
        if umc is not None and umc.get("budget", 1.0) < 0.999:
            hist = feats[:-1]
            u_hist = list(uncs[-4:-1])
            if len(u_hist) < len(hist):
                u_hist = [torch.zeros_like(hist[0][:, :1])] * (len(hist) - len(u_hist)) + u_hist
            recs = [compress_bev(f, u, umc["budget"], umc.get("mode", "uncertainty"))["rec"]
                    for f, u in zip(hist, u_hist[-len(hist):])]
            feats = recs + [feats[-1]]
        return self.res2(feats)

    def fuse_map(self, bev, h, umc: dict | None = None, unc: torch.Tensor | None = None):
        if self.map_temporal == "none":
            return bev
        if h is not None:
            h = h.detach()
        if umc is not None and umc.get("budget", 1.0) < 0.999 and h is not None:
            u = unc if unc is not None else torch.zeros_like(h[:, :1])
            h = compress_bev(h, u, umc["budget"], umc.get("mode", "magnitude"))["rec"]
        return self.gru(h, bev)

    def map_step(self, feat, q_mem=None, p_mem=None, dx: float = 0.0,
                 k_prop: int | None = None, umc: dict | None = None):
        k = self.query_prop if k_prop is None else k_prop
        if umc is not None:
            k = max(1, int(round(umc.get("query_budget", 1.0) * self.query_prop)))
        if q_mem is not None and p_mem is not None and k > 0:
            q_mem = q_mem + self.q_trans(torch.tensor([[dx]], device=q_mem.device, dtype=q_mem.dtype))
            p_mem = p_mem.clone()
            p_mem[..., 0] = p_mem[..., 0] - dx
        logits, pts, q = self.map(feat, q_mem, p_mem, k_prop=k)
        with torch.no_grad():
            conf = torch.softmax(logits, -1)[..., :3].max(-1).values
            kk = min(self.query_prop, conf.shape[1])
            idx = conf.topk(kk, dim=1).indices
            q_new = torch.gather(q, 1, idx[..., None].expand(-1, -1, q.shape[-1])).detach()
            p_new = torch.gather(pts, 1, idx[..., None, None].expand(-1, -1, pts.shape[2], 2)).detach()
        return logits, pts, q_new, p_new
