# -*- coding: utf-8 -*-
"""Synthetic driving-scene generator: road geometry, dynamic objects, map elements."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# ----------------------------------------------------------------------------- road
LANE_CENTERS = (-3.5, 0.0, 3.5)
LANE_WIDTH = 3.5
ROAD_HALF = 5.25
CROSSWALKS = (18.0, 36.0)
CURVE_AMP = 0.6
CURVE_LEN = 18.0


def road_offset(x: np.ndarray | float) -> np.ndarray | float:
    """Lateral offset of the road centreline (shared by all lanes)."""
    return CURVE_AMP * np.sin(np.asarray(x) / CURVE_LEN)


def lane_y(x, lane_center: float):
    return lane_center + road_offset(x)


def road_heading(x):
    # d(offset)/dx
    return np.arctan(CURVE_AMP / CURVE_LEN * np.cos(np.asarray(x) / CURVE_LEN))


@dataclass
class Box:
    cls: int          # 0 vehicle, 1 pedestrian
    x: float
    y: float
    z: float
    l: float
    w: float
    h: float
    yaw: float
    vx: float = 0.0
    vy: float = 0.0

    def corners_bev(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx = np.array([-self.l / 2, self.l / 2, self.l / 2, -self.l / 2])
        dy = np.array([-self.w / 2, -self.w / 2, self.w / 2, self.w / 2])
        x = self.x + c * dx - s * dy
        y = self.y + s * dx + c * dy
        return np.stack([x, y], axis=-1)


@dataclass
class Scene:
    boxes: List[Box]
    ego_x: float
    time: float
    map_elements: List[Tuple[int, np.ndarray]]  # (class, points[N,2]) in world frame

    def ego_boxes(self) -> List[Box]:
        out = []
        for b in self.boxes:
            out.append(Box(b.cls, b.x - self.ego_x, b.y, b.z, b.l, b.w, b.h, b.yaw, b.vx, b.vy))
        return out

    def ego_map(self) -> List[Tuple[int, np.ndarray]]:
        out = []
        for cls, pts in self.map_elements:
            q = pts.copy()
            q[:, 0] -= self.ego_x
            out.append((cls, q))
        return out


def _sample_polyline(x0: float, x1: float, yfun, step: float = 0.5) -> np.ndarray:
    xs = np.arange(x0, x1 + 1e-6, step)
    ys = yfun(xs)
    return np.stack([xs, ys], axis=-1)


def build_map_elements(rng: np.random.Generator) -> List[Tuple[int, np.ndarray]]:
    """Static world-frame map elements. class 0: lane divider, 1: boundary, 2: crosswalk."""
    elems: List[Tuple[int, np.ndarray]] = []
    for yc in LANE_CENTERS:
        for off in (-LANE_WIDTH / 2, LANE_WIDTH / 2):
            y0 = yc + off
            pts = _sample_polyline(-25.0, 85.0, lambda x, y0=y0: lane_y(x, y0))
            elems.append((0, pts))
    # outer boundaries
    for y0 in (-ROAD_HALF, ROAD_HALF):
        pts = _sample_polyline(-25.0, 85.0, lambda x, y0=y0: lane_y(x, y0))
        elems.append((1, pts))
    # crosswalks: straight segments across the road
    for xc in CROSSWALKS:
        ys = np.linspace(-ROAD_HALF, ROAD_HALF, 24)
        xs = np.full_like(ys, xc)
        elems.append((2, np.stack([xs, ys], axis=-1)))
    return elems


def sample_scene(rng: np.random.Generator, t: float, ego_speed: float,
                 n_vehicles: int = 6, n_peds: int = 2) -> Scene:
    ego_x = ego_speed * t
    boxes: List[Box] = []
    lanes = list(LANE_CENTERS)
    for i in range(n_vehicles):
        lane = lanes[int(rng.integers(0, len(lanes)))]
        x0 = float(rng.uniform(6.0, 55.0))
        speed = float(rng.uniform(-1.0, 8.0))  # relative to world; ego moves +4 m/s
        # small lane-change for some vehicles
        lc = 0.0
        if rng.random() < 0.25:
            lc = float(rng.choice([-LANE_WIDTH, LANE_WIDTH])) * min(1.0, max(0.0, (t - 0.5) / 1.5))
        x = x0 + speed * t
        y = lane_y(x, lane) + lc
        yaw = float(road_heading(x)) + (0.12 if lc > 0 else (-0.12 if lc < 0 else 0.0))
        if x - ego_x < -5 or x - ego_x > 60:
            x = float(rng.uniform(15.0, 45.0)) + ego_x
            y = lane_y(x, lane)
        boxes.append(Box(0, x, y, 0.75, 4.4, 1.85, 1.5, yaw, speed, 0.0))
    # pedestrians near crosswalks, crossing laterally
    for i in range(n_peds):
        xc = float(rng.choice(CROSSWALKS)) + float(rng.uniform(-0.6, 0.6))
        y = float(rng.uniform(-4.5, 4.5))
        vy = float(rng.choice([-0.8, 0.8]))
        boxes.append(Box(1, xc, y + vy * t, 0.85, 0.55, 0.55, 1.7, 0.0, 0.0, vy))
    return Scene(boxes=boxes, ego_x=ego_x, time=t, map_elements=build_map_elements(rng))


def make_sequence(seed: int, frames: int, dt: float, ego_speed: float):
    rng = np.random.default_rng(seed)
    # randomise road parameters per sequence a little by shifting crosswalks? keep global for simplicity
    scenes = []
    # sample object roster once (same objects evolve across frames)
    n_veh = int(rng.integers(4, 8))
    n_ped = int(rng.integers(1, 3))
    # initial states
    lanes = list(LANE_CENTERS)
    states = []
    for i in range(n_veh):
        lane = lanes[int(rng.integers(0, len(lanes)))]
        states.append(dict(kind=0, lane=lane, x0=float(rng.uniform(8.0, 50.0)),
                           speed=float(rng.uniform(0.0, 9.0)),
                           lc=float(rng.choice([0.0, 0.0, -LANE_WIDTH, LANE_WIDTH])) if rng.random() < 0.3 else 0.0))
    peds = []
    for i in range(n_ped):
        xc = float(rng.choice(CROSSWALKS))
        peds.append(dict(x=xc + float(rng.uniform(-0.5, 0.5)), y=float(rng.uniform(-4.0, 4.0)),
                         vy=float(rng.choice([-0.7, 0.7]))))
    map_elems = build_map_elements(rng)
    for f in range(frames):
        t = f * dt
        ego_x = ego_speed * t
        boxes = []
        for s in states:
            x = s["x0"] + s["speed"] * t
            prog = min(1.0, max(0.0, (t - 0.5) / 1.5)) if s["lc"] != 0 else 0.0
            y = lane_y(x, s["lane"]) + s["lc"] * prog
            yaw = float(road_heading(x)) + (0.10 if s["lc"] > 0 else (-0.10 if s["lc"] < 0 else 0.0)) * prog
            boxes.append(Box(0, x, y, 0.75, 4.4, 1.85, 1.5, yaw, s["speed"], 0.0))
        for p in peds:
            boxes.append(Box(1, p["x"], p["y"] + p["vy"] * t, 0.85, 0.55, 0.55, 1.7, 0.0, 0.0, p["vy"]))
        scenes.append(Scene(boxes=boxes, ego_x=ego_x, time=t, map_elements=map_elems))
    return scenes
