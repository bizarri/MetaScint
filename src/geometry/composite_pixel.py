from __future__ import annotations
import bisect
import numpy as np
from typing import Optional
from config.schema import RegionConfig, IrradiationConfig
from geometry.uniform_pixel import sample_entry_direction as _sample_entry_direction


class CompositeGeometry:
    def __init__(
        self,
        pixel_x_mm: float,
        pixel_y_mm: float,
        pixel_z_mm: float,
        regions: tuple[RegionConfig, ...],
        default_material_role: str,
    ):
        self.hx = pixel_x_mm / 2.0
        self.hy = pixel_y_mm / 2.0
        self.z0 = 0.0
        self.z1 = pixel_z_mm
        self.regions = regions
        self.default_material_role = default_material_role

        # Pre-build a sorted axis index for fast lookup when ALL regions are
        # axis-aligned (tilt_deg == 0) and partition along a single axis.
        # Detects manufacturing-style configs where all regions span full y/z
        # and are arranged along x (or y), enabling O(log N) lookup.
        self._fast_axis: str | None = None
        self._fast_boundaries: list[float] = []
        self._fast_roles: list[str] = []
        if regions:
            self._build_fast_index()

    def _build_fast_index(self) -> None:
        """Build a sorted boundary list for O(log N) point lookup.

        Enabled only when all regions are axis-aligned and tile a single axis
        (x or y) with non-overlapping, contiguous intervals that each span the
        full extent of the other two axes. This matches the output of
        generate_regions() for manufacturing configs.
        """
        if any(abs(r.tilt_deg) >= 1e-8 for r in self.regions):
            return  # has tilted regions — fall back to linear scan

        # Try x-axis partitioning first, then y-axis
        for axis in ('x', 'y'):
            sorted_r = sorted(self.regions, key=lambda r: r.xmin if axis == 'x' else r.ymin)
            ok = True
            for r in sorted_r:
                # Must span full depth (z) and the perpendicular axis
                if axis == 'x':
                    if not (abs(r.ymin - (-self.hy)) < 1e-9 and abs(r.ymax - self.hy) < 1e-9
                            and abs(r.zmin - self.z0) < 1e-9 and abs(r.zmax - self.z1) < 1e-9):
                        ok = False; break
                else:
                    if not (abs(r.xmin - (-self.hx)) < 1e-9 and abs(r.xmax - self.hx) < 1e-9
                            and abs(r.zmin - self.z0) < 1e-9 and abs(r.zmax - self.z1) < 1e-9):
                        ok = False; break
            if ok and sorted_r:
                self._fast_axis = axis
                self._fast_boundaries = [r.xmin if axis == 'x' else r.ymin for r in sorted_r]
                self._fast_roles = [r.material_role for r in sorted_r]
                self._fast_upper = [r.xmax if axis == 'x' else r.ymax for r in sorted_r]
                return

    # ------------------------------------------------------------------
    # Point-in-region test (handles rotated boxes)
    # ------------------------------------------------------------------
    def _point_in_region(self, pos: np.ndarray, r: RegionConfig) -> bool:
        if abs(r.tilt_deg) < 1e-8:
            return (r.xmin <= pos[0] < r.xmax and
                    r.ymin <= pos[1] < r.ymax and
                    r.zmin <= pos[2] < r.zmax)
        cx = (r.xmin + r.xmax) / 2.0
        cy = (r.ymin + r.ymax) / 2.0
        cz = (r.zmin + r.zmax) / 2.0
        theta = np.radians(r.tilt_deg)
        c, s = np.cos(theta), np.sin(theta)
        dx = pos[0] - cx
        dy = pos[1] - cy
        dz = pos[2] - cz
        lx = dx * c + dz * s
        ly = dy
        lz = -dx * s + dz * c
        hx = (r.xmax - r.xmin) / 2.0
        hy = (r.ymax - r.ymin) / 2.0
        hz = (r.zmax - r.zmin) / 2.0
        return (-hx <= lx < hx and -hy <= ly < hy and -hz <= lz < hz)

    def material_role_at(self, pos: np.ndarray) -> str:
        # Fast O(log N) path for axis-aligned single-axis configs (manufacturing layouts)
        if self._fast_axis is not None:
            coord = pos[0] if self._fast_axis == 'x' else pos[1]
            idx = bisect.bisect_right(self._fast_boundaries, coord) - 1
            if 0 <= idx < len(self._fast_roles) and coord < self._fast_upper[idx]:
                return self._fast_roles[idx]
            return self.default_material_role
        # General O(N) path for mixed/tilted geometries
        for r in self.regions:
            if self._point_in_region(pos, r):
                return r.material_role
        return self.default_material_role

    # ------------------------------------------------------------------
    # Distance to bounding-box exit
    # ------------------------------------------------------------------
    def distance_to_box_mm(self, pos: np.ndarray, dir: np.ndarray) -> float:
        x, y, z = pos
        dx, dy, dz = dir
        ds = []
        if dx > 1e-12:
            ds.append((self.hx - x) / dx)
        elif dx < -1e-12:
            ds.append((-self.hx - x) / dx)
        if dy > 1e-12:
            ds.append((self.hy - y) / dy)
        elif dy < -1e-12:
            ds.append((-self.hy - y) / dy)
        if dz > 1e-12:
            ds.append((self.z1 - z) / dz)
        elif dz < -1e-12:
            ds.append((self.z0 - z) / dz)
        ds = [d for d in ds if d > 1e-12]
        return min(ds) if ds else 0.0

    # ------------------------------------------------------------------
    # Ray–AABB intersection (axis-aligned)
    # ------------------------------------------------------------------
    def _ray_aabb_intersect(
        self, pos: np.ndarray, dir: np.ndarray,
        xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float,
    ) -> tuple[Optional[float], Optional[float]]:
        x, y, z = pos
        dx, dy, dz = dir

        # For each axis: if the ray is parallel (abs(d) < eps), the slab either
        # imposes no constraint (ray inside slab → [-∞, +∞]) or makes the
        # intersection impossible (ray outside slab → [+∞, -∞]).
        # Do NOT swap the impossible case — that would turn it into [-∞, +∞].
        if abs(dx) > 1e-12:
            tx1 = (xmin - x) / dx
            tx2 = (xmax - x) / dx
            if tx1 > tx2:
                tx1, tx2 = tx2, tx1
        elif xmin <= x < xmax:
            tx1, tx2 = -1e30, 1e30   # inside slab: no x constraint
        else:
            tx1, tx2 = 1e30, -1e30   # outside slab: impossible

        if abs(dy) > 1e-12:
            ty1 = (ymin - y) / dy
            ty2 = (ymax - y) / dy
            if ty1 > ty2:
                ty1, ty2 = ty2, ty1
        elif ymin <= y < ymax:
            ty1, ty2 = -1e30, 1e30
        else:
            ty1, ty2 = 1e30, -1e30

        if abs(dz) > 1e-12:
            tz1 = (zmin - z) / dz
            tz2 = (zmax - z) / dz
            if tz1 > tz2:
                tz1, tz2 = tz2, tz1
        elif zmin <= z < zmax:
            tz1, tz2 = -1e30, 1e30
        else:
            tz1, tz2 = 1e30, -1e30

        t_entry = max(tx1, ty1, tz1)
        t_exit = min(tx2, ty2, tz2)

        if t_entry >= t_exit or t_exit <= 0:
            return None, None

        t_entry = max(t_entry, 0.0)
        return t_entry, t_exit

    # ------------------------------------------------------------------
    # Ray–box intersection (handles rotation)
    # ------------------------------------------------------------------
    def _ray_box_intersect(
        self, pos: np.ndarray, dir: np.ndarray, r: RegionConfig,
    ) -> tuple[Optional[float], Optional[float]]:
        if abs(r.tilt_deg) < 1e-8:
            return self._ray_aabb_intersect(
                pos, dir, r.xmin, r.xmax, r.ymin, r.ymax, r.zmin, r.zmax,
            )
        cx = (r.xmin + r.xmax) / 2.0
        cy = (r.ymin + r.ymax) / 2.0
        cz = (r.zmin + r.zmax) / 2.0
        theta = np.radians(r.tilt_deg)
        c, s = np.cos(theta), np.sin(theta)

        dx = pos[0] - cx
        dy = pos[1] - cy
        dz = pos[2] - cz
        lpos = np.array([dx * c + dz * s, dy, -dx * s + dz * c])
        ldir = np.array([dir[0] * c + dir[2] * s, dir[1], -dir[0] * s + dir[2] * c])

        hx = (r.xmax - r.xmin) / 2.0
        hy = (r.ymax - r.ymin) / 2.0
        hz = (r.zmax - r.zmin) / 2.0
        return self._ray_aabb_intersect(lpos, ldir, -hx, hx, -hy, hy, -hz, hz)

    # ------------------------------------------------------------------
    # Distance to next material boundary (handles rotated regions)
    # ------------------------------------------------------------------
    def distance_to_next_material_change(self, pos: np.ndarray, dir: np.ndarray) -> tuple[float, Optional[str]]:
        box_dist = self.distance_to_box_mm(pos, dir)

        current_material = self.material_role_at(pos)
        boundary_dist = box_dist
        next_material = None

        for r in self.regions:
            entry_d, exit_d = self._ray_box_intersect(pos, dir, r)

            if r.material_role == current_material:
                if exit_d is not None and 1e-12 < exit_d < boundary_dist:
                    probe = pos + dir * (exit_d + 1e-6)
                    probe_mat = self.material_role_at(probe)
                    if probe_mat != current_material:
                        boundary_dist = exit_d
                        next_material = probe_mat
            else:
                if entry_d is not None and 1e-12 < entry_d < boundary_dist:
                    boundary_dist = entry_d
                    next_material = r.material_role

        return boundary_dist, next_material

    def sample_entry_xy(self, rng: np.random.Generator, irradiation: IrradiationConfig):
        scale = (max(min(irradiation.front_face_area_fraction, 1.0), 0.0)) ** 0.5
        return rng.uniform(-self.hx * scale, self.hx * scale), rng.uniform(-self.hy * scale, self.hy * scale)

    def sample_entry_direction(self, rng: np.random.Generator, irradiation: IrradiationConfig) -> np.ndarray:
        return _sample_entry_direction(rng, irradiation)

    # ------------------------------------------------------------------
    # Geometry consistency: trace a ray through all material domains
    # ------------------------------------------------------------------
    def _distance_to_enter_box_mm(self, pos: np.ndarray, dir: np.ndarray) -> float:
        x, y, z = pos
        dx, dy, dz = dir
        tmin = -1e30
        tmax = 1e30
        for dim in range(3):
            if dim == 0:
                lo, hi, p, d = -self.hx, self.hx, x, dx
            elif dim == 1:
                lo, hi, p, d = -self.hy, self.hy, y, dy
            else:
                lo, hi, p, d = self.z0, self.z1, z, dz
            if abs(d) < 1e-12:
                if p < lo or p > hi:
                    return 1e30
            else:
                t1 = (lo - p) / d
                t2 = (hi - p) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax or tmax < 0:
                    return 1e30
        return max(0.0, tmin) if tmax > 0 else 1e30

    def trace_ray(self, origin: np.ndarray, direction: np.ndarray, max_segments: int = 200) -> list[dict]:
        dir_unit = direction / np.linalg.norm(direction)
        pos = np.array(origin, dtype=float)

        entry_d = self._distance_to_enter_box_mm(pos, dir_unit)
        if entry_d >= 1e29:
            return []
        pos = pos + dir_unit * entry_d

        segments = []
        for _ in range(max_segments):
            dist_to_box = self.distance_to_box_mm(pos, dir_unit)

            if dist_to_box <= 1e-12:
                break

            current_mat = self.material_role_at(pos)
            dist_to_change, next_mat = self.distance_to_next_material_change(pos, dir_unit)
            seg_dist = min(dist_to_change, dist_to_box)

            if seg_dist <= 1e-12:
                break

            end_pos = pos + dir_unit * seg_dist
            exit = next_mat is None and seg_dist >= dist_to_box - 1e-12

            segments.append({
                'material_role': current_mat,
                'start_mm': pos.tolist(),
                'end_mm': end_pos.tolist(),
                'length_mm': float(seg_dist),
                'exits_pixel': exit,
            })

            if exit:
                break

            pos = end_pos + dir_unit * 1e-10

        return segments
