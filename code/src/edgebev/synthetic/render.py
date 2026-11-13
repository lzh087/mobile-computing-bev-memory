# -*- coding: utf-8 -*-
"""Differentiable-friendly synthetic multi-view renderer (ray casting on CPU).

The renderer produces:
  * 6 RGB images (uint8) with distance shading and lane markings,
  * per-pixel Euclidean depth (float32) and valid mask,
  * BEV occupancy / detection / vector-map targets.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ..config import CFG, Config
from .scene import CROSSWALKS, LANE_CENTERS, LANE_WIDTH, ROAD_HALF, Box, Scene, lane_y, road_offset


PALETTE = np.array([
    [0.82, 0.20, 0.18],  # red
    [0.16, 0.38, 0.78],  # blue
    [0.18, 0.62, 0.32],  # green
    [0.88, 0.70, 0.14],  # yellow
    [0.55, 0.28, 0.66],  # purple
    [0.90, 0.45, 0.12],  # orange
    [0.15, 0.65, 0.68],  # cyan
    [0.70, 0.70, 0.72],  # silver
], dtype=np.float32)


def camera_rays(cfg: Config = CFG):
    """Return ray origins (M,3) and directions (M,3) in the ego frame."""
    f = cfg.focal
    us, vs = np.meshgrid(np.arange(cfg.img_w), np.arange(cfg.img_h))
    u = (us.reshape(-1) - cfg.img_w / 2.0) / f
    v = (vs.reshape(-1) - cfg.img_h / 2.0) / f
    orig, dirs = [], []
    for yaw_deg in cfg.cam_yaws:
        a = math.radians(yaw_deg)
        p = math.radians(cfg.cam_pitch_deg)
        fwd = np.array([math.cos(a) * math.cos(p), math.sin(a) * math.cos(p), math.sin(p)])
        right = np.array([math.sin(a), -math.cos(a), 0.0])
        down = np.cross(fwd, right)
        d = fwd[None, :] + u[:, None] * right[None, :] + v[:, None] * down[None, :]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        orig.append(np.tile(np.array([0.0, 0.0, cfg.cam_height]), (d.shape[0], 1)))
        dirs.append(d)
    return np.concatenate(orig, 0), np.concatenate(dirs, 0)


def _box_arrays(boxes: List[Box]):
    c = np.array([[b.x, b.y, b.z] for b in boxes], np.float32)
    half = np.array([[b.l / 2, b.w / 2, b.h / 2] for b in boxes], np.float32)
    yaw = np.array([b.yaw for b in boxes], np.float32)
    return c, half, yaw


def ray_box_intersection(orig, dirs, boxes: List[Box]):
    """Vectorised slab test. Returns (N,) nearest t and (N,) box index (-1 = miss)."""
    n = orig.shape[0]
    if not boxes:
        return np.full(n, np.inf, np.float32), np.full(n, -1, np.int32)
    c, half, yaw = _box_arrays(boxes)
    cos, sin = np.cos(yaw), np.sin(yaw)
    # world -> box frame: R^T (p - c), R = [[cos,-sin,0],[sin,cos,0],[0,0,1]]
    px = orig[:, None, 0] - c[None, :, 0]
    py = orig[:, None, 1] - c[None, :, 1]
    pz = orig[:, None, 2] - c[None, :, 2]
    lx = cos[None, :] * px + sin[None, :] * py
    ly = -sin[None, :] * px + cos[None, :] * py
    lz = pz
    dx = cos[None, :] * dirs[:, None, 0] + sin[None, :] * dirs[:, None, 1]
    dy = -sin[None, :] * dirs[:, None, 0] + cos[None, :] * dirs[:, None, 1]
    dz = dirs[:, None, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t1x = (-half[None, :, 0] - lx) / dx
        t2x = (half[None, :, 0] - lx) / dx
        t1y = (-half[None, :, 1] - ly) / dy
        t2y = (half[None, :, 1] - ly) / dy
        t1z = (-half[None, :, 2] - lz) / dz
        t2z = (half[None, :, 2] - lz) / dz
    tmin = np.maximum(np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y)), np.minimum(t1z, t2z))
    tmax = np.minimum(np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y)), np.maximum(t1z, t2z))
    hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin > 0.0)
    tmin = np.where(hit, tmin, np.inf)
    idx = np.argmin(tmin, axis=1)
    best = tmin[np.arange(n), idx]
    best = np.where(np.isfinite(best), best, np.inf)
    idx = np.where(np.isfinite(best), idx, -1)
    return best.astype(np.float32), idx.astype(np.int32)


def _lane_markings(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a boolean mask of white lane / crosswalk markings at ego-frame points."""
    off = road_offset(x)
    mask = np.zeros_like(x, dtype=bool)
    if np.any((x > -10) & (x < 70)):
        for yc in LANE_CENTERS:
            for s in (-LANE_WIDTH / 2, LANE_WIDTH / 2):
                mask |= np.abs(y - (yc + s + off)) < 0.16
        for yb in (-ROAD_HALF, ROAD_HALF):
            mask |= np.abs(y - (yb + off)) < 0.18
    for xc in CROSSWALKS:
        in_cw = (np.abs(x - xc) < 2.0) & (np.abs(y - off) < ROAD_HALF)
        stripe = (np.floor((x - (xc - 2.0)) / 0.8) % 2) == 0
        mask |= in_cw & stripe
    return mask


def render_scene(scene: Scene, cfg: Config = CFG, noise: float = 0.02,
                 rng: np.random.Generator | None = None) -> Dict[str, np.ndarray]:
    rng = rng or np.random.default_rng(0)
    ego_boxes = scene.ego_boxes()
    orig, dirs = camera_rays(cfg)
    n = orig.shape[0]
    t_box, idx = ray_box_intersection(orig, dirs, ego_boxes)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_ground = -orig[:, 2] / dirs[:, 2]
    ground_ok = (dirs[:, 2] < -1e-4) & (t_ground > 0.5)
    t_ground = np.where(ground_ok, t_ground, np.inf)

    use_box = t_box < t_ground
    t = np.where(use_box, t_box, t_ground)
    hit = np.isfinite(t)
    depth = np.where(hit, t, 0.0).astype(np.float32)

    pts = orig + dirs * np.where(hit, t, 0.0)[:, None]
    # ---- colours
    sky_top = np.array([0.42, 0.62, 0.85], np.float32)
    sky_bot = np.array([0.80, 0.87, 0.93], np.float32)
    sky_t = np.clip((dirs[:, 2] + 0.15) / 0.6, 0, 1)[:, None]
    rgb = sky_top[None, :] * (1 - sky_t) + sky_bot[None, :] * sky_t

    # ground
    gpos = pts
    white = _lane_markings(gpos[:, 0], gpos[:, 1])
    asphalt = np.array([0.33, 0.34, 0.36], np.float32)
    ground_col = np.where(white[:, None], np.array([0.93, 0.93, 0.90], np.float32)[None, :],
                          asphalt[None, :] * (0.9 + 0.1 * np.sin(gpos[:, 0] * 2.1)[:, None]))
    fog = np.exp(-np.clip(depth, 0, 80) / 70.0)[:, None]
    ground_col = ground_col * fog + sky_bot[None, :] * (1 - fog)
    rgb = np.where((ground_ok & ~use_box)[:, None], ground_col, rgb)

    # objects
    if ego_boxes:
        obj_col = np.zeros((n, 3), np.float32)
        for i, b in enumerate(ego_boxes):
            m = idx == i
            if not m.any():
                continue
            base = PALETTE[i % len(PALETTE)]
            # local longitudinal coordinate for a simple stripe texture
            c, s = math.cos(b.yaw), math.sin(b.yaw)
            lx = c * (pts[m, 0] - b.x) + s * (pts[m, 1] - b.y)
            stripe = 0.92 + 0.08 * np.sin(lx * 6.0)
            shade = (0.55 + 0.45 * np.exp(-depth[m] / 40.0)) * stripe
            obj_col[m] = base[None, :] * shade[:, None]
            if b.cls == 1:  # pedestrian: darker, add head/legs band
                obj_col[m] *= np.where((pts[m, 2] > 1.1)[:, None], 1.05, 0.85)
        rgb = np.where(use_box[:, None], obj_col, rgb)

    rgb = np.clip(rgb + rng.normal(0, noise, rgb.shape).astype(np.float32), 0, 1)
    imgs = rgb.reshape(cfg.n_cams, cfg.img_h, cfg.img_w, 3).transpose(0, 3, 1, 2)
    depth = depth.reshape(cfg.n_cams, cfg.img_h, cfg.img_w)
    objid = idx.reshape(cfg.n_cams, cfg.img_h, cfg.img_w)
    valid = (depth > 0).astype(np.float32)
    return {"image": (imgs * 255).astype(np.uint8), "depth": depth.astype(np.float32),
            "objid": objid.astype(np.int16), "valid": valid.astype(np.uint8)}


# --------------------------------------------------------------------- BEV targets
def bev_occupancy(boxes: List[Box], cfg: Config = CFG, classes=(0, 1)) -> np.ndarray:
    occ = Image.new("L", (cfg.bev_w, cfg.bev_h), 0)
    draw = ImageDraw.Draw(occ)
    for b in boxes:
        if b.cls not in classes:
            continue
        pts = b.corners_bev()
        # BEV image: col = y, row = x
        xy = [((p[1] - cfg.y_min) / cfg.res, (p[0] - cfg.x_min) / cfg.res) for p in pts]
        draw.polygon(xy, fill=1)
    return np.array(occ, np.uint8)


def map_targets(scene: Scene, cfg: Config = CFG, n_points: int = 12,
                min_len: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Clip ego-frame map polylines to the BEV range and resample to n_points."""
    classes, lines = [], []
    for cls, pts in scene.ego_map():
        # clip by x range first (all elements are monotone in one axis)
        m = (pts[:, 0] >= cfg.x_min - 1.0) & (pts[:, 0] <= cfg.x_max + 1.0)
        q = pts[m]
        if q.shape[0] < 2:
            continue
        # arc-length resampling
        d = np.linalg.norm(np.diff(q, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])
        if s[-1] < min_len:
            continue
        si = np.linspace(0, s[-1], n_points)
        x = np.interp(si, s, q[:, 0])
        y = np.interp(si, s, q[:, 1])
        # require most points inside BEV
        inside = (x >= cfg.x_min) & (x <= cfg.x_max) & (y >= cfg.y_min) & (y <= cfg.y_max)
        if inside.mean() < 0.6:
            continue
        classes.append(cls)
        lines.append(np.stack([x, y], -1))
    if not lines:
        return np.zeros((0,), np.int64), np.zeros((0, n_points, 2), np.float32)
    return np.array(classes, np.int64), np.stack(lines).astype(np.float32)


def detection_targets(scene: Scene, cfg: Config = CFG):
    cls, centers, sizes, yaws = [], [], [], []
    for b in scene.ego_boxes():
        if not (cfg.x_min <= b.x <= cfg.x_max and cfg.y_min <= b.y <= cfg.y_max):
            continue
        cls.append(b.cls)
        centers.append([b.x, b.y])
        sizes.append([b.l, b.w])
        yaws.append(b.yaw)
    if not centers:
        return (np.zeros((0,), np.int64), np.zeros((0, 2), np.float32),
                np.zeros((0, 2), np.float32), np.zeros((0,), np.float32))
    return (np.array(cls, np.int64), np.array(centers, np.float32),
            np.array(sizes, np.float32), np.array(yaws, np.float32))


def save_debug_png(scene: Scene, out: str, cfg: Config = CFG):
    r = render_scene(scene, cfg)
    canvas = np.zeros((cfg.img_h, cfg.img_w * 3, 3), np.uint8)
    for i in range(6):
        row, col = divmod(i, 3)
        canvas[(1 - row) * 0 + 0:cfg.img_h, col * cfg.img_w:(col + 1) * cfg.img_w] = 0
    tiles = []
    for i in range(6):
        tiles.append(r["image"][i].transpose(1, 2, 0))
    top = np.concatenate(tiles[:3], 1)
    bot = np.concatenate(tiles[3:], 1)
    Image.fromarray(np.concatenate([top, bot], 0)).save(out)
