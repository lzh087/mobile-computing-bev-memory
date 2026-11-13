# -*- coding: utf-8 -*-
"""Cached synthetic sequence dataset for EdgeBEV."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import CFG, Config
from .render import bev_occupancy, detection_targets, map_targets, render_scene
from .scene import make_sequence

MAX_OBJ = 16
MAX_MAP = 14
N_POINTS = 12


def _pad_objects(cls, ctr, size, yaw):
    n = min(len(cls), MAX_OBJ)
    c = np.zeros(MAX_OBJ, np.int64); c[:n] = cls[:n]
    ce = np.zeros((MAX_OBJ, 2), np.float32); ce[:n] = ctr[:n]
    sz = np.zeros((MAX_OBJ, 2), np.float32); sz[:n] = size[:n]
    ya = np.zeros(MAX_OBJ, np.float32); ya[:n] = yaw[:n]
    mask = np.zeros(MAX_OBJ, np.float32); mask[:n] = 1.0
    return c, ce, sz, ya, mask


def _pad_map(cls, lines):
    n = min(len(cls), MAX_MAP)
    c = np.zeros(MAX_MAP, np.int64); c[:n] = cls[:n]
    ln = np.zeros((MAX_MAP, N_POINTS, 2), np.float32); ln[:n] = lines[:n]
    mask = np.zeros(MAX_MAP, np.float32); mask[:n] = 1.0
    return c, ln, mask


def generate_sequence(seed: int, cfg: Config = CFG):
    scenes = make_sequence(seed, cfg.frames, cfg.dt, cfg.ego_speed)
    imgs, deps, vals, objs = [], [], [], []
    occs, dcls, dctr, dsize, dyaw, dmask = [], [], [], [], [], []
    mcls, mlines, mmask = [], [], []
    ego = []
    objids = []
    for sc in scenes:
        r = render_scene(sc, cfg, rng=np.random.default_rng(seed * 100 + int(sc.time * 10)))
        imgs.append(r["image"])
        deps.append(r["depth"].astype(np.float16))
        vals.append(r["valid"])
        objids.append(r["objid"])
        occs.append(bev_occupancy(sc.ego_boxes(), cfg))
        c, ce, sz, ya, m = _pad_objects(*detection_targets(sc, cfg))
        dcls.append(c); dctr.append(ce); dsize.append(sz); dyaw.append(ya); dmask.append(m)
        mc, ml = map_targets(sc, cfg, N_POINTS)
        c2, l2, m2 = _pad_map(mc, ml)
        mcls.append(c2); mlines.append(l2); mmask.append(m2)
        ego.append([sc.ego_x, 0.0, 0.0])
    return {
        "image": np.stack(imgs),
        "depth": np.stack(deps),
        "valid": np.stack(vals),
        "objid": np.stack(objids),
        "occ": np.stack(occs),
        "det_cls": np.stack(dcls),
        "det_center": np.stack(dctr),
        "det_size": np.stack(dsize),
        "det_yaw": np.stack(dyaw),
        "det_mask": np.stack(dmask),
        "map_cls": np.stack(mcls),
        "map_line": np.stack(mlines),
        "map_mask": np.stack(mmask),
        "ego": np.array(ego, np.float32),
    }


def build_cache(root: str, n_train: int, n_val: int, cfg: Config = CFG, seed0: int = 1000):
    for split, n in (("train", n_train), ("val", n_val)):
        out = os.path.join(root, split)
        os.makedirs(out, exist_ok=True)
        for i in range(n):
            path = os.path.join(out, f"{i:04d}.npz")
            if os.path.exists(path):
                continue
            d = generate_sequence(seed0 + i + (0 if split == "train" else 100000), cfg)
            np.savez_compressed(path, **d)
        print(f"{split}: {n} sequences -> {out}")


class SeqDataset(Dataset):
    def __init__(self, root: str, split: str, cfg: Config = CFG, augment: bool = False):
        self.dir = os.path.join(root, split)
        self.files = sorted(os.listdir(self.dir))
        self.cfg = cfg
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(os.path.join(self.dir, self.files[idx]))
        out = {k: torch.from_numpy(np.asarray(v)) for k, v in d.items()}
        out["image"] = out["image"].float() / 255.0
        if self.augment:
            # photometric noise
            out["image"] = (out["image"] + 0.03 * torch.randn_like(out["image"])).clamp(0, 1)
            # random camera dropout (simulates sensor failure / occlusion of a view)
            if torch.rand(()) < 0.5:
                drop = torch.randint(0, self.cfg.n_cams, (1,)).item()
                out["image"][:, drop] = 0.0
        return out


def collate(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], 0) for k in keys}
