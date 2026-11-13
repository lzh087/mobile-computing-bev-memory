# -*- coding: utf-8 -*-
"""Evaluation metrics for depth, 3D detection, vectorised mapping and temporal consistency."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .config import CFG, Config


# ------------------------------------------------------------------ depth
def depth_metrics(prob: np.ndarray, depth: np.ndarray, valid: np.ndarray, cfg: Config = CFG):
    """prob (N,D,H,W), depth (N,H,W), valid (N,H,W)."""
    centers = cfg.depth_min + (np.arange(cfg.depth_bins) + 0.5) * (
        (cfg.depth_max - cfg.depth_min) / cfg.depth_bins)
    pred = (prob * centers[None, :, None, None]).sum(1)
    m = (valid > 0) & (depth > cfg.depth_min) & (depth < cfg.depth_max)
    if m.sum() == 0:
        return {"mae": 0.0, "rmse": 0.0, "delta1": 0.0}
    err = np.abs(pred[m] - depth[m])
    ratio = np.maximum(pred[m] / depth[m], depth[m] / pred[m])
    return {"mae": float(err.mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "delta1": float((ratio < 1.25).mean())}


# ------------------------------------------------------------------ detection
def _ap_11(scores, tp, npos):
    if npos == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tp)[order]
    cum = np.cumsum(tp)
    rec = cum / npos
    prec = cum / (np.arange(len(tp)) + 1)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = prec[rec >= t]
        ap += (p.max() if len(p) else 0.0) / 11.0
    return float(ap)


def detection_map(preds, gts, thresholds=(0.5, 1.0, 2.0, 4.0), ncls: int = 2):
    """preds/gts: list over frames; each [[x,y,l,w,yaw,score,cls], ...]."""
    aps, ates = [], []
    for thr in thresholds:
        per_cls = []
        for c in range(ncls):
            scores, tp = [], []
            npos = 0
            for pr, gt in zip(preds, gts):
                g = [x for x in gt if x[5] == c]
                npos += len(g)
                used = [False] * len(g)
                for p in sorted([x for x in pr if x[6] == c], key=lambda z: -z[5]):
                    best, bd = -1, 1e9
                    for j, q in enumerate(g):
                        if used[j]:
                            continue
                        d = np.hypot(p[0] - q[0], p[1] - q[1])
                        if d < bd:
                            bd, best = d, j
                    ok = best >= 0 and bd < thr
                    if ok:
                        used[best] = True
                        ates.append(bd)
                    scores.append(p[5])
                    tp.append(1.0 if ok else 0.0)
            per_cls.append(_ap_11(scores, tp, npos))
        aps.append(float(np.mean(per_cls)))
    return {"mAP": float(np.mean(aps)),
            "AP@0.5": aps[0], "AP@1": aps[1], "AP@2": aps[2], "AP@4": aps[3],
            "mATE": float(np.mean(ates)) if ates else 0.0}


# ------------------------------------------------------------------ map
def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(0.5 * (d.min(1).mean() + d.min(0).mean()))


def map_ap(preds, gts, thresholds=(0.5, 1.0, 1.5), ncls: int = 3):
    """preds: list over frames of [(score, cls, pts[P,2])]; gts: [(cls, pts[P,2])]."""
    aps = []
    for thr in thresholds:
        per_cls = []
        for c in range(ncls):
            scores, tp = [], []
            npos = 0
            for pr, gt in zip(preds, gts):
                g = [pts for cc, pts in gt if cc == c]
                npos += len(g)
                used = [False] * len(g)
                for score, pc, pts in sorted([z for z in pr if z[1] == c], key=lambda z: -z[0]):
                    best, bd = -1, 1e9
                    for j, q in enumerate(g):
                        if used[j]:
                            continue
                        d = chamfer(pts, q)
                        if d < bd:
                            bd, best = d, j
                    ok = best >= 0 and bd < thr
                    if ok:
                        used[best] = True
                    scores.append(score)
                    tp.append(1.0 if ok else 0.0)
            per_cls.append(_ap_11(scores, tp, npos))
        aps.append(float(np.mean(per_cls)))
    return {"mAP": float(np.mean(aps)), "AP@0.5": aps[0], "AP@1": aps[1], "AP@1.5": aps[2]}


def rasterize_lines(lines, cfg: Config = CFG):
    img = Image.new("L", (cfg.bev_w, cfg.bev_h), 0)
    dr = ImageDraw.Draw(img)
    for pts in lines:
        xy = [((p[1] - cfg.y_min) / cfg.res, (p[0] - cfg.x_min) / cfg.res) for p in pts]
        dr.line(xy, fill=1, width=1)
    return np.array(img, np.uint8)


def temporal_consistency(preds, ego, cfg: Config = CFG):
    """IoU of rasterised predictions between consecutive frames after ego alignment."""
    ious = []
    for t in range(1, len(preds)):
        a = rasterize_lines([p[2] for p in preds[t - 1]], cfg)
        b = rasterize_lines([p[2] for p in preds[t]], cfg)
        dx = int(round((ego[t, 0] - ego[t - 1, 0]) / cfg.res))
        a = np.roll(a, -dx, axis=0)
        inter = np.logical_and(a, b).sum()
        uni = np.logical_or(a, b).sum()
        if uni > 0:
            ious.append(inter / uni)
    return float(np.mean(ious)) if ious else 0.0
