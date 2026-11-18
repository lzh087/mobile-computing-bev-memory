"""Synthetic BEV observations with scene-disjoint seeds and known ego shifts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


LENGTH, WIDTH = 48, 24
FAR_START = 24
OFFSETS = (0, 4, 8)
CLASSES = ('divider', 'boundary', 'crossing')
SPLIT_SEEDS = {'train': 100_000, 'validation': 200_000, 'test': 300_000}


@dataclass(frozen=True)
class Split:
    name: str
    scene_ids: list[int]
    views: torch.Tensor
    labels: torch.Tensor


def rasterize_scene(rng: np.random.Generator) -> np.ndarray:
    """Make a static world map covering the union of the three view windows."""
    x = np.arange(LENGTH + OFFSETS[-1])[:, None] + 0.5
    y = np.arange(WIDTH)[None, :] - WIDTH / 2 + 0.5
    slope = rng.uniform(-0.11, 0.11)
    amplitude = rng.uniform(0.3, 2.0)
    phase = rng.uniform(0, 2 * np.pi)
    center = rng.uniform(-2.8, 2.8) + slope * (x - 26)
    center += amplitude * np.sin(x / rng.uniform(8, 18) + phase)
    half_width = rng.uniform(2.8, 4.8) + rng.uniform(-0.025, 0.025) * (x - 26)
    local_y = y - center
    divider = np.abs(local_y) < 0.65
    boundary = np.abs(np.abs(local_y) - half_width) < 0.65
    crossing = np.zeros_like(divider)
    for position in rng.choice(np.arange(7, 54, 4), int(rng.integers(1, 4)), replace=False):
        tilt = rng.uniform(-0.22, 0.22)
        crossing |= (np.abs(x - position - tilt * local_y) < 0.85) & (np.abs(local_y) <= half_width)
    return np.stack((divider, boundary, crossing)).astype(np.float32)


def observe(rng: np.random.Generator, local_map: np.ndarray) -> np.ndarray:
    """Emulate distance-dependent weak measurements and structured occlusion."""
    distance = np.linspace(0, 1, LENGTH, dtype=np.float32)[:, None]
    visibility = rng.random((LENGTH, WIDTH)) < (0.97 - 0.38 * distance)
    for _ in range(int(rng.integers(1, 4))):
        x = int(rng.integers(12, LENGTH))
        y = int(rng.integers(2, WIDTH - 2))
        visibility[x:x + int(rng.integers(5, 13)), max(0, y - 4):y + 5] = False
    sharp = gaussian_filter(local_map, sigma=(0, 0.45, 0.45))
    blurred = gaussian_filter(local_map, sigma=(0, 1.25, 1.25))
    signal = ((1 - distance) * sharp + distance * blurred) * (0.98 - 0.26 * distance)
    noise = rng.normal(0, 0.10 + 0.17 * distance, size=local_map.shape)
    clutter = (rng.random(local_map.shape) < 0.018) * rng.uniform(0.25, 0.7, local_map.shape)
    evidence = np.clip(signal + noise + clutter, 0, 1) * visibility[None]
    return np.concatenate((evidence, visibility[None])).astype(np.float32)


def make_split(name: str, size: int) -> Split:
    if not 1 <= size <= 100_000:
        raise ValueError('Each split must have 1 to 100,000 unique scene IDs.')
    scene_ids = list(range(SPLIT_SEEDS[name], SPLIT_SEEDS[name] + size))
    observations, labels = [], []
    for scene_id in scene_ids:
        rng = np.random.default_rng(scene_id)
        gt = rasterize_scene(rng)
        observations.append(np.stack([observe(rng, gt[:, offset:offset + LENGTH]) for offset in OFFSETS]))
        labels.append(gt[:, :LENGTH].copy())
    return Split(name, scene_ids, torch.from_numpy(np.stack(observations)), torch.from_numpy(np.stack(labels)))


def align_future(view: torch.Tensor, offset: int) -> torch.Tensor:
    """Warp the future frame into current coordinates, zeroing non-overlap."""
    if not 0 <= offset < view.shape[-2]:
        raise ValueError('The shift must fit inside the longitudinal grid.')
    aligned = torch.zeros_like(view)
    aligned[..., offset:, :] = view[..., :view.shape[-2] - offset, :]
    return aligned


def aligned_inputs(views: torch.Tensor, extra_shift: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    future = [align_future(views[:, i], offset + extra_shift) for i, offset in enumerate(OFFSETS[1:], 1)]
    visible = torch.stack([frame[:, 3:4] for frame in future]).amax(dim=0)
    return torch.cat([views[:, 0], *future], dim=1), visible


def mask_far_observation(current: torch.Tensor) -> torch.Tensor:
    """Remove current evidence and its visibility flag in a fixed far corridor."""
    masked = current.clone()
    masked[..., 28:44, 5:19] = 0
    return masked
