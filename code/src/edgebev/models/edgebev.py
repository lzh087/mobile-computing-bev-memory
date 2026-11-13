# -*- coding: utf-8 -*-
"""EdgeBEV: a compact temporal BEV perception testbed.

It contains the reproduced components of BEVNeXt (CRF-modulated depth + Res2Fusion)
and StreamMapNet (streaming BEV memory + query propagation + multi-point attention),
plus the proposed uncertainty-aware memory compression (UMC) plug-in.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG, Config
from .depth import CRFMeanField, DepthNet, conv_bn
from .fusion import ConvGRU, Res2Fusion, compress_bev, warp_bev
from .heads import DetectionHead, MapDecoder
from .view_transform import LSSViewTransform


class EdgeBEV(nn.Module):
    def __init__(self, cfg: Config = CFG, use_crf: bool = True, temporal: str = "res2",
                 map_temporal: str = "gru", use_map: bool = True, query_prop: int = 10):
        super().__init__()
        self.cfg = cfg
        self.use_crf = use_crf
        self.temporal = temporal
        self.map_temporal = map_temporal
        self.use_map = use_map
        self.query_prop = query_prop
        self.depth = DepthNet(cfg, 48)
        self.crf = CRFMeanField(cfg.depth_bins, 3, 2, cfg.depth_min, cfg.depth_max)
        self.lss = LSSViewTransform(cfg, 24, 40)
        self.down = conv_bn(48, 64, 3, 2)
        self.res2 = Res2Fusion(64, k=4, window=2)
        self.gru = ConvGRU(64)
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                conv_bn(64, 48, 3, 1))
        self.det = DetectionHead(48, 2)
        self.map = MapDecoder(48, 128, 40, 12, 3, 2, 4)
        self.map.cfg = cfg
        self.q_trans = nn.Linear(1, 128)

    # ------------------------------------------------------------------ encoding
    def encode_frame(self, images: torch.Tensor):
        """images: (B,N,3,H,W) -> BEV feature (B,64,50,50), uncertainty, depth outputs."""
        b, n, c, h, w = images.shape
        feat, logits = self.depth(images)
        rgb = F.interpolate(images.reshape(b * n, c, h, w), size=(24, 40), mode="area")
        prob = torch.softmax(logits, 1)
        refined = self.crf(logits, rgb) if self.use_crf else prob
        ent = -(refined.clamp_min(1e-6).log() * refined).sum(1)  # B*N,H,W
        cfeat = feat.shape[1]
        feat = feat.reshape(b, n, cfeat, 24, 40)
        logits = logits.reshape(b, n, self.cfg.depth_bins, 24, 40)
        refined = refined.reshape(b, n, self.cfg.depth_bins, 24, 40)
        ent = ent.reshape(b, n, 24, 40)
        bev, unc = self.lss(feat, refined, ent)
        bev_d = self.down(bev)
        unc_d = F.avg_pool2d(unc, 2)
        return bev_d, unc_d, logits, refined

    def fuse_det(self, bevs, uncs, umc: dict | None = None):
        """Temporal fusion for the dense detection branch."""
        if self.temporal == "none":
            return bevs[-1]
        if self.temporal == "gru":
            h = None
            for b in bevs[:-1]:
                h = self.gru(h, b.detach())
            return self.gru(h, bevs[-1])
        # Res2Fusion over the last 4 frames
        k = min(4, len(bevs))
        feats = [b.detach() for b in bevs[-k - 1:-1]] + [bevs[-1]]
        if len(feats) < 4:
            feats = [torch.zeros_like(feats[0])] * (4 - len(feats)) + feats
        if umc is not None and umc.get("budget", 1.0) < 0.999:
            # compress the historical frames (not the current one)
            hist = feats[:-1]
            u_hist = list(uncs[-4:-1])
            if len(u_hist) < len(hist):
                u_hist = [torch.zeros_like(hist[0][:, :1])] * (len(hist) - len(u_hist)) + u_hist
            recs = [compress_bev(f, u, umc["budget"], umc.get("mode", "uncertainty"))["rec"]
                    for f, u in zip(hist, u_hist[-len(hist):])]
            feats = recs + [feats[-1]]
        return self.res2(feats)

    def fuse_map(self, bev, h, umc: dict | None = None):
        if self.map_temporal == "none":
            return bev
        if h is not None:
            h = h.detach()
        if umc is not None and umc.get("budget", 1.0) < 0.999:
            # budgeted memory write: compress the previous recurrent state
            u = torch.zeros_like(bev[:, :1])
            h = compress_bev(h, u, umc["budget"], "magnitude")["rec"] if h is not None else h
        return self.gru(h, bev)

    def map_step(self, feat, q_mem=None, p_mem=None, dx: float = 0.0,
                 k_prop: int | None = None, memory: dict | None = None,
                 umc: dict | None = None):
        k = self.query_prop if k_prop is None else k_prop
        if umc is not None:
            k = max(1, int(round(umc.get("query_budget", 1.0) * self.query_prop)))
        if q_mem is not None and p_mem is not None and k > 0:
            q_mem = q_mem + self.q_trans(torch.tensor([[dx]], device=q_mem.device, dtype=q_mem.dtype))
            p_mem = p_mem.clone()
            p_mem[..., 0] = p_mem[..., 0] - dx
        logits, pts, q = self.map(feat, q_mem, p_mem, k_prop=k)
        # update memory with the most confident queries
        with torch.no_grad():
            conf = torch.softmax(logits, -1)[..., :3].max(-1).values
            kk = min(self.query_prop, conf.shape[1])
            idx = conf.topk(kk, dim=1).indices
            q_new = torch.gather(q, 1, idx[..., None].expand(-1, -1, q.shape[-1])).detach()
            p_new = torch.gather(pts, 1, idx[..., None, None].expand(-1, -1, pts.shape[2], 2)).detach()
        return logits, pts, q_new, p_new

    # ------------------------------------------------------------------ sequence inference
    def forward_sequence(self, images, ego=None, umc: dict | None = None, return_depth: bool = False):
        """images (B,T,N,3,H,W) -> per-frame det/map outputs."""
        b, t = images.shape[:2]
        det_outs, map_outs, depth_out = [], [], []
        q_mem = p_mem = None
        h_map = None
        bevs, uncs = [], []
        for ti in range(t):
            bev, unc, logits, prob = self.encode_frame(images[:, ti])
            bevs.append(bev)
            uncs.append(unc)
            fused = self.fuse_det(bevs, uncs, umc)
            det_outs.append(self.det(self.up(fused)))
            if self.use_map:
                h_map = self.fuse_map(bev, h_map, umc)
                dx = 0.0 if (ego is None or ti == 0) else float(ego[:, ti, 0].mean() - ego[:, ti - 1, 0].mean())
                logits_m, pts, q_mem, p_mem = self.map_step(self.up(h_map), q_mem, p_mem, dx,
                                                            umc=umc)
                map_outs.append((logits_m, pts))
            else:
                map_outs.append(None)
            if return_depth:
                depth_out.append((logits.detach(), prob.detach()))
        return {"det": det_outs, "map": map_outs, "bev": bevs, "unc": uncs, "depth": depth_out}

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
