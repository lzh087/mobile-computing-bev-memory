# -*- coding: utf-8 -*-
"""Detection head (CenterPoint-style) and vectorised map decoder (StreamMapNet-style)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..config import CFG, Config


class OccHead(nn.Module):
    """Dense BEV occupancy head (vehicle/pedestrian footprint)."""

    def __init__(self, channels: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def occ_loss(logits, target, pos_weight: float = 8.0):
    target = target.float().unsqueeze(1)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=torch.tensor(pos_weight, device=logits.device))
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + target.sum() + 1)
    return bce + dice, {"bce": float(bce.detach()), "dice": float(dice.detach())}


# ------------------------------------------------------------------ detection
def gaussian_heatmap(centers, classes, cfg: Config, ncls: int = 2, radius: int = 2):
    hm = np.zeros((ncls, cfg.bev_h, cfg.bev_w), np.float32)
    for (x, y), c in zip(centers, classes):
        ix = int((x - cfg.x_min) / cfg.res)
        iy = int((y - cfg.y_min) / cfg.res)
        if not (0 <= ix < cfg.bev_h and 0 <= iy < cfg.bev_w):
            continue
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < cfg.bev_h and 0 <= jy < cfg.bev_w:
                    hm[c, jx, jy] = max(hm[c, jx, jy], math.exp(-(dx * dx + dy * dy) / (2 * 1.0)))
    return hm


class DetectionHead(nn.Module):
    def __init__(self, channels: int = 48, ncls: int = 2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.hm = nn.Conv2d(64, ncls, 1)
        self.hm.bias.data.fill_(-4.6)
        self.off = nn.Conv2d(64, 2, 1)
        self.size = nn.Conv2d(64, 2, 1)
        self.yaw = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        f = self.shared(x)
        return {"hm": self.hm(f), "off": self.off(f), "size": self.size(f), "yaw": self.yaw(f)}


def focal_loss(pred_logits, target, alpha=2.0, beta=4.0):
    pred = torch.sigmoid(pred_logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = target.lt(1).float()
    neg_w = torch.pow(1 - target, beta)
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * neg_w * neg
    npos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / npos


def det_loss(out, hm_target, centers, sizes, yaws, cfg: Config):
    """centers/sizes/yaws: (B,M,2/2/1) padded, M>=1; empty padded with mask."""
    dev = out["hm"].device
    hm_t = torch.from_numpy(hm_target).to(dev).float()
    l_hm = focal_loss(out["hm"], hm_t)
    b, ncls, h, w = out["hm"].shape
    ix = ((centers[..., 0] - cfg.x_min) / cfg.res).long().clamp(0, h - 1)
    iy = ((centers[..., 1] - cfg.y_min) / cfg.res).long().clamp(0, w - 1)
    mask = (centers.abs().sum(-1) > 0).float()
    off_t = torch.stack([(centers[..., 0] - cfg.x_min) / cfg.res - ix.float(),
                         (centers[..., 1] - cfg.y_min) / cfg.res - iy.float()], -1)
    size_t = torch.log(sizes.clamp(min=1e-2))
    yaw_t = torch.stack([torch.sin(yaws), torch.cos(yaws)], -1)
    idx = (torch.arange(b, device=dev)[:, None] * (h * w)
           + ix * w + iy).reshape(-1)
    pred_off = out["off"].permute(0, 2, 3, 1).reshape(-1, 2)[idx].reshape(b, -1, 2)
    pred_size = out["size"].permute(0, 2, 3, 1).reshape(-1, 2)[idx].reshape(b, -1, 2)
    pred_yaw = out["yaw"].permute(0, 2, 3, 1).reshape(-1, 2)[idx].reshape(b, -1, 2)
    m = mask[..., None]
    l_off = (F.l1_loss(pred_off, off_t, reduction="none") * m).sum() / m.sum().clamp(min=1)
    l_size = (F.l1_loss(pred_size, size_t, reduction="none") * m).sum() / m.sum().clamp(min=1)
    l_yaw = (F.l1_loss(pred_yaw, yaw_t, reduction="none") * m).sum() / m.sum().clamp(min=1)
    return l_hm + 2.0 * (l_off + l_size + l_yaw), {"hm": float(l_hm.detach()), "off": float(l_off.detach()),
                                                    "size": float(l_size.detach()), "yaw": float(l_yaw.detach())}


@torch.no_grad()
def decode_det(out, cfg: Config, topk: int = 64, thresh: float = 0.05):
    hm = torch.sigmoid(out["hm"])
    b, ncls, h, w = hm.shape
    pooled = F.max_pool2d(hm, 3, 1, 1)
    peaks = (hm == pooled) & (hm > thresh)
    results = []
    for bi in range(b):
        dets = []
        for c in range(ncls):
            ys, xs = torch.nonzero(peaks[bi, c], as_tuple=True)
            scores = hm[bi, c, ys, xs]
            if scores.numel() == 0:
                continue
            k = min(topk, scores.numel())
            scores, sel = scores.topk(k)
            ys, xs = ys[sel], xs[sel]
            off = out["off"][bi, :, ys, xs].T
            size = torch.exp(out["size"][bi, :, ys, xs].T)
            yaw = out["yaw"][bi, :, ys, xs].T
            cx = (xs.float() + off[:, 0]) * cfg.res + cfg.x_min
            cy = (ys.float() + off[:, 1]) * cfg.res + cfg.y_min
            ang = torch.atan2(yaw[:, 0], yaw[:, 1])
            for i in range(scores.numel()):
                dets.append([float(cx[i]), float(cy[i]), float(size[i, 0]), float(size[i, 1]),
                             float(ang[i]), float(scores[i]), c])
        results.append(dets)
    return results


# ------------------------------------------------------------------ map decoder
def pos_encoding(points: torch.Tensor, d: int, cfg: Config):
    # points (B,N,P,2) metres -> (B,N,P,d)
    x = (points[..., 0:1] - cfg.x_min) / (cfg.x_max - cfg.x_min)
    y = (points[..., 1:2] - cfg.y_min) / (cfg.y_max - cfg.y_min)
    freqs = torch.linspace(1.0, 8.0, d // 4, device=points.device)
    ex = torch.cat([torch.sin(x * freqs), torch.cos(x * freqs)], -1)
    ey = torch.cat([torch.sin(y * freqs), torch.cos(y * freqs)], -1)
    return torch.cat([ex, ey], -1)[..., :d]


class MapLayer(nn.Module):
    def __init__(self, d: int = 128, heads: int = 4, npts: int = 12, n_off: int = 4):
        super().__init__()
        self.npts, self.n_off = npts, n_off
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.off_w = nn.Linear(d, npts * n_off)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm3 = nn.LayerNorm(d)
        self.reg = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, npts * 2))
        self.offsets = nn.Parameter(torch.tensor(
            [[0.0, 0.0], [0.6, 0.0], [-0.6, 0.0], [0.0, 0.6]], dtype=torch.float32)[:n_off])

    def forward(self, q, p, feat, cfg: Config):
        b, nq, d = q.shape
        q = q + pos_encoding(p, d, cfg).mean(2)
        qa, _ = self.self_attn(q, q, q, need_weights=False)
        q = self.norm1(q + qa)
        # sample K offsets around each of the npts reference points
        pts = p[:, :, :, None, :] + self.offsets[None, None, None, :, :]  # B,N,P,K,2
        gx = pts[..., 0] / (cfg.x_max - cfg.x_min) * 2 - 1
        gy = pts[..., 1] / (cfg.y_max - cfg.y_min)  # [-25,25] -> [-1,1]
        grid = torch.stack([gx, gy], -1).reshape(b, nq * self.npts * self.n_off, 1, 2)
        samp = F.grid_sample(feat, grid, align_corners=False, padding_mode="zeros")
        samp = samp.squeeze(-1).reshape(b, d, nq, self.npts, self.n_off).permute(0, 2, 3, 4, 1)
        w = torch.softmax(self.off_w(q).reshape(b, nq, self.npts, self.n_off), -1)
        msg = (samp * w[..., None]).sum(3).sum(2)  # B,N,d
        q = self.norm2(q + self.proj(msg))
        q = self.norm3(q + self.ffn(q))
        dp = self.reg(q).reshape(b, nq, self.npts, 2)
        p = torch.sigmoid(dp) * torch.tensor(
            [cfg.x_max - cfg.x_min, cfg.y_max - cfg.y_min], device=q.device) + torch.tensor(
            [cfg.x_min, cfg.y_min], device=q.device)
        return q, p


class MapDecoder(nn.Module):
    def __init__(self, cin: int = 48, d: int = 128, nq: int = 40, npts: int = 12,
                 ncls: int = 3, layers: int = 2, heads: int = 4):
        super().__init__()
        self.nq, self.npts, self.ncls = nq, npts, ncls
        self.inp = nn.Conv2d(cin, d, 1)
        self.query = nn.Embedding(nq, d)
        init = torch.zeros(nq, npts, 2)
        xs = torch.linspace(3, 47, npts)
        for i in range(nq):
            init[i, :, 0] = xs
            init[i, :, 1] = -12 + 24 * i / max(nq - 1, 1)
        self.ref = nn.Parameter(init)
        self.layers = nn.ModuleList([MapLayer(d, heads, npts) for _ in range(layers)])
        self.cls = nn.Linear(d, ncls + 1)
        self.pos = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.cfg = CFG

    def forward(self, bev, q_init=None, p_init=None, k_prop: int = 0):
        b = bev.shape[0]
        feat = self.inp(bev)
        q = self.query.weight[None].expand(b, -1, -1).clone()
        p = self.ref[None].expand(b, -1, -1, -1).clone()
        if q_init is not None and k_prop > 0:
            k = min(k_prop, q_init.shape[1])
            q[:, :k] = q_init[:, :k]
            p[:, :k] = p_init[:, :k]
        for layer in self.layers:
            q, p = layer(q, p, feat, self.cfg)
        logits = self.cls(q)
        return logits, p, q


# ------------------------------------------------------------------ map matching / loss
def map_loss(logits, pred_pts, gt_cls, gt_pts, gt_mask, cfg: Config):
    """logits (B,Nq,4); pred_pts (B,Nq,P,2); gt_cls (B,M); gt_pts (B,M,P,2); gt_mask (B,M)."""
    b, nq, ncls1 = logits.shape
    total_cls, total_pts = 0.0, 0.0
    n_match = 0
    for bi in range(b):
        m = int(gt_mask[bi].sum().item())
        if m == 0:
            total_cls = total_cls + F.cross_entropy(logits[bi], torch.full((nq,), ncls1 - 1,
                                                                          dtype=torch.long,
                                                                          device=logits.device))
            continue
        gc = gt_cls[bi, :m].long()
        gp = gt_pts[bi, :m]
        with torch.no_grad():
            prob = torch.softmax(logits[bi], -1)[:, :ncls1 - 1].clamp_min(1e-6)
            cls_cost = -torch.log(prob[:, gc].T + 1e-6)  # M,Nq
            pt_cost = torch.cdist(pred_pts[bi].reshape(nq, -1), gp.reshape(m, -1), p=1).reshape(nq, m) / (pred_pts.shape[2] * 2)
            cost = cls_cost.T + 2.0 * pt_cost
        ri, ci = linear_sum_assignment(cost.detach().cpu().numpy())
        tgt = torch.full((nq,), ncls1 - 1, dtype=torch.long, device=logits.device)
        tgt[ri] = gc[ci]
        total_cls = total_cls + F.cross_entropy(logits[bi], tgt)
        total_pts = total_pts + F.l1_loss(pred_pts[bi, ri], gp[ci])
        n_match += len(ri)
    return total_cls / b + 2.0 * total_pts / max(n_match, 1), {"cls": float(total_cls / b),
                                                              "pts": float(total_pts / max(n_match, 1))}


@torch.no_grad()
def decode_map(logits, pts, cfg: Config, score_thr: float = 0.25, topk: int = 40):
    prob = torch.softmax(logits, -1)[..., :3]
    out = []
    for b in range(prob.shape[0]):
        items = []
        for q in range(prob.shape[1]):
            c = int(prob[b, q].argmax())
            s = float(prob[b, q, c])
            if s < score_thr:
                continue
            items.append((s, c, pts[b, q].cpu().numpy()))
        items.sort(key=lambda x: -x[0])
        out.append(items[:topk])
    return out
