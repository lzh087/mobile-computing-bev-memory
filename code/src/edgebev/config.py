# -*- coding: utf-8 -*-
"""Global configuration for the EdgeBEV synthetic benchmark."""
import math
from dataclasses import dataclass, field


@dataclass
class Config:
    # BEV grid (ego frame: x forward, y left)
    x_min: float = 0.0
    x_max: float = 50.0
    y_min: float = -25.0
    y_max: float = 25.0
    res: float = 0.5
    # camera
    img_h: int = 48
    img_w: int = 80
    fov_deg: float = 90.0
    cam_height: float = 1.55
    cam_pitch_deg: float = -8.0
    cam_yaws = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
    # depth
    depth_min: float = 2.0
    depth_max: float = 50.0
    depth_bins: int = 32
    # sequence
    frames: int = 4
    dt: float = 0.5
    ego_speed: float = 4.0
    # training
    seed: int = 0

    @property
    def bev_h(self) -> int:
        return int(round((self.x_max - self.x_min) / self.res))

    @property
    def bev_w(self) -> int:
        return int(round((self.y_max - self.y_min) / self.res))

    @property
    def focal(self) -> float:
        return self.img_w / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))

    @property
    def n_cams(self) -> int:
        return len(self.cam_yaws)


CFG = Config()
