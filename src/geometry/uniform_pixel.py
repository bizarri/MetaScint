from __future__ import annotations
import numpy as np
from config.schema import GeometryConfig, IrradiationConfig


def distance_to_box_mm(position_mm: np.ndarray, direction: np.ndarray, geometry: GeometryConfig) -> float:
    hx = geometry.pixel_x_mm / 2.0
    hy = geometry.pixel_y_mm / 2.0
    z0 = 0.0
    z1 = geometry.pixel_z_mm
    x, y, z = position_mm
    dx, dy, dz = direction
    ds = []
    if dx > 1e-12:
        ds.append((hx - x) / dx)
    elif dx < -1e-12:
        ds.append((-hx - x) / dx)
    if dy > 1e-12:
        ds.append((hy - y) / dy)
    elif dy < -1e-12:
        ds.append((-hy - y) / dy)
    if dz > 1e-12:
        ds.append((z1 - z) / dz)
    elif dz < -1e-12:
        ds.append((z0 - z) / dz)
    ds = [d for d in ds if d > 1e-12]
    return min(ds) if ds else 0.0


def sample_entry_xy(rng: np.random.Generator, geometry: GeometryConfig, irradiation: IrradiationConfig):
    hx = geometry.pixel_x_mm / 2.0
    hy = geometry.pixel_y_mm / 2.0
    scale = (max(min(irradiation.front_face_area_fraction, 1.0), 0.0)) ** 0.5
    return rng.uniform(-hx * scale, hx * scale), rng.uniform(-hy * scale, hy * scale)


def sample_entry_direction(rng: np.random.Generator, irradiation: IrradiationConfig) -> np.ndarray:
    """Sample a photon entry direction within a cone of half-angle max_angle_deg around +z.

    Uses the spherical cap formula: cos_theta drawn uniformly from
    [cos(max_angle_deg), 1], phi drawn uniformly from [0, 2*pi].
    When max_angle_deg == 0 this always returns exactly [0, 0, 1].
    """
    max_angle_deg = irradiation.max_angle_deg
    if max_angle_deg <= 0.0:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    cos_min = np.cos(np.radians(max_angle_deg))
    cos_theta = rng.uniform(cos_min, 1.0)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], dtype=float)
