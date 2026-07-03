from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

Strategy = Literal["adjust_walls", "adjust_grooves", "adjust_matrix_edges", "minimize_change"]


@dataclass(frozen=True)
class ManufacturingConstraint:
    axis: str
    n_channels: int
    groove_width_mm: float
    groove_material_role: str
    wall_width_mm: float
    matrix_material_role: str
    matrix_edge_width_mm: float
    strategy: str = "adjust_walls"
    tilt_deg: float = 0.0


def solve_constraint(
    mc: ManufacturingConstraint,
    pixel_width_mm: float,
) -> ManufacturingConstraint:
    s = mc.strategy
    N = mc.n_channels
    W = pixel_width_mm
    m0 = mc.matrix_edge_width_mm
    g0 = mc.groove_width_mm
    w0 = mc.wall_width_mm

    if s == "adjust_walls":
        if N == 1:
            m = (W - N * g0) / 2.0
            if m <= 0:
                raise ValueError(
                    f"adjust_walls (N=1): computed matrix_edge_width={m:.6f} <= 0"
                )
            return ManufacturingConstraint(
                axis=mc.axis, n_channels=N,
                groove_width_mm=g0, groove_material_role=mc.groove_material_role,
                wall_width_mm=w0, matrix_material_role=mc.matrix_material_role,
                matrix_edge_width_mm=m, strategy=s, tilt_deg=mc.tilt_deg,
            )
        w = (W - 2 * m0 - N * g0) / (N - 1)
        if w <= 0:
            raise ValueError(
                f"adjust_walls: computed wall_width={w:.6f} <= 0. "
                f"pixel={W} matrix_edge={m0} groove={g0} N={N}"
            )
        return ManufacturingConstraint(
            axis=mc.axis, n_channels=N,
            groove_width_mm=g0, groove_material_role=mc.groove_material_role,
            wall_width_mm=w, matrix_material_role=mc.matrix_material_role,
            matrix_edge_width_mm=m0, strategy=s, tilt_deg=mc.tilt_deg,
        )

    if s == "adjust_grooves":
        g = (W - 2 * m0 - (N - 1) * w0) / N
        if g <= 0:
            raise ValueError(
                f"adjust_grooves: computed groove_width={g:.6f} <= 0"
            )
        return ManufacturingConstraint(
            axis=mc.axis, n_channels=N,
            groove_width_mm=g, groove_material_role=mc.groove_material_role,
            wall_width_mm=w0, matrix_material_role=mc.matrix_material_role,
            matrix_edge_width_mm=m0, strategy=s, tilt_deg=mc.tilt_deg,
        )

    if s == "adjust_matrix_edges":
        m = (W - N * g0 - (N - 1) * w0) / 2.0
        if m <= 0:
            raise ValueError(
                f"adjust_matrix_edges: computed matrix_edge_width={m:.6f} <= 0"
            )
        return ManufacturingConstraint(
            axis=mc.axis, n_channels=N,
            groove_width_mm=g0, groove_material_role=mc.groove_material_role,
            wall_width_mm=w0, matrix_material_role=mc.matrix_material_role,
            matrix_edge_width_mm=m, strategy=s, tilt_deg=mc.tilt_deg,
        )

    if s == "minimize_change":
        # Minimise unweighted L2: (m-m0)^2 + (g-g0)^2 + (w-w0)^2
        # subject to: 2m + N*g + (N-1)*w = W
        #
        # Lagrangian solution:
        #   lambda = (2*m0 + N*g0 + (N-1)*w0 - W) / (4 + N^2 + (N-1)^2)
        #   m = m0 - 2*lambda
        #   g = g0 - N*lambda
        #   w = w0 - (N-1)*lambda
        total0 = 2 * m0 + N * g0 + (N - 1) * w0
        denom = 4.0 + N * N + (N - 1) ** 2
        lam = (total0 - W) / denom
        m = m0 - 2.0 * lam
        g = g0 - N * lam
        w = w0 - (N - 1) * lam
        if m <= 0 or g <= 0 or w <= 0:
            raise ValueError(
                f"minimize_change produced non-positive widths: "
                f"matrix_edge={m:.6f} groove={g:.6f} wall={w:.6f}"
            )
        return ManufacturingConstraint(
            axis=mc.axis, n_channels=N,
            groove_width_mm=g, groove_material_role=mc.groove_material_role,
            wall_width_mm=w, matrix_material_role=mc.matrix_material_role,
            matrix_edge_width_mm=m, strategy=s, tilt_deg=mc.tilt_deg,
        )

    raise ValueError(f"Unknown strategy: {s}")


def generate_regions(
    mc: ManufacturingConstraint,
    pixel_height_mm: float,
    pixel_depth_mm: float,
    pixel_width_mm: float,
) -> list[dict]:
    regions: list[dict] = []
    axis = mc.axis
    N = mc.n_channels
    g = mc.groove_width_mm
    w = mc.wall_width_mm
    m = mc.matrix_edge_width_mm
    groove_mat = mc.groove_material_role
    matrix_mat = mc.matrix_material_role
    tilt = mc.tilt_deg

    half_h = pixel_height_mm / 2.0
    z0, z1 = 0.0, pixel_depth_mm
    half_w = pixel_width_mm / 2.0

    def add_region(name, mat, x1, x2, local_tilt=0.0):
        if axis == "x":
            r = dict(name=name, material_role=mat,
                     xmin=x1, xmax=x2,
                     ymin=-half_h, ymax=half_h,
                     zmin=z0, zmax=z1)
        else:
            r = dict(name=name, material_role=mat,
                     xmin=-half_h, xmax=half_h,
                     ymin=x1, ymax=x2,
                     zmin=z0, zmax=z1)
        if local_tilt:
            r['tilt_deg'] = local_tilt
        regions.append(r)

    cur = -half_w

    if m > 0:
        add_region("matrix_left", matrix_mat, cur, cur + m)
        cur += m

    for i in range(N):
        add_region(f"groove_{i}", groove_mat, cur, cur + g, local_tilt=tilt)
        cur += g
        if i < N - 1 and w > 0:
            add_region(f"wall_{i}", matrix_mat, cur, cur + w)
            cur += w

    if m > 0:
        add_region("matrix_right", matrix_mat, cur, cur + m)

    return regions


def check_manufacturing_rules(config: dict) -> list[str]:
    mfg = config.get("manufacturing")
    if mfg is None:
        return []

    issues: list[str] = []
    axis = mfg.get("axis", "x")
    N = int(mfg.get("n_channels", 0))
    g = mfg.get("groove_width_mm", 0)
    w = mfg.get("wall_width_mm", 0)
    m = mfg.get("matrix_edge_width_mm", 0)
    strategy = mfg.get("strategy", "adjust_walls")

    if N < 1:
        issues.append("manufacturing.n_channels must be >= 1")
    if g <= 0:
        issues.append("manufacturing.groove_width_mm must be positive")
    if w <= 0:
        issues.append("manufacturing.wall_width_mm must be positive")
    if m <= 0:
        issues.append("manufacturing.matrix_edge_width_mm must be positive")
    if axis not in ("x", "y"):
        issues.append('manufacturing.axis must be "x" or "y"')
    if strategy not in ("adjust_walls", "adjust_grooves", "adjust_matrix_edges", "minimize_change"):
        issues.append(f"manufacturing.strategy '{strategy}' not supported")

    if issues:
        return issues

    pixel_w = config.get("pixel_x_mm" if axis == "x" else "pixel_y_mm", 0)
    if pixel_w <= 0:
        issues.append(f"pixel_{axis}_mm must be positive for manufacturing constraint")
        return issues

    total_initial = 2 * m + N * g + (N - 1) * w
    tol = 1e-9

    if abs(total_initial - pixel_w) < tol:
        return issues

    try:
        mc_obj = ManufacturingConstraint(
            axis=axis, n_channels=N,
            groove_width_mm=g, groove_material_role=mfg.get("groove_material_role", ""),
            wall_width_mm=w, matrix_material_role=mfg.get("matrix_material_role", ""),
            matrix_edge_width_mm=m, strategy=strategy,
            tilt_deg=mfg.get("tilt_deg", 0.0),
        )
        solve_constraint(mc_obj, pixel_w)
    except ValueError as e:
        issues.append(f"manufacturing constraint cannot be satisfied: {e}")

    return issues


def apply_manufacturing(config: dict) -> dict:
    mfg = config.get("manufacturing")
    if mfg is None:
        return config

    axis = mfg["axis"]
    pixel_w = config.get("pixel_x_mm" if axis == "x" else "pixel_y_mm", 0)
    pixel_h = config.get("pixel_y_mm" if axis == "x" else "pixel_x_mm", 0)
    pixel_d = config.get("pixel_z_mm", 0)

    mc = ManufacturingConstraint(
        axis=axis,
        n_channels=int(mfg["n_channels"]),
        groove_width_mm=mfg["groove_width_mm"],
        groove_material_role=mfg.get("groove_material_role", ""),
        wall_width_mm=mfg["wall_width_mm"],
        matrix_material_role=mfg.get("matrix_material_role", ""),
        matrix_edge_width_mm=mfg["matrix_edge_width_mm"],
        strategy=mfg.get("strategy", "adjust_walls"),
        tilt_deg=mfg.get("tilt_deg", 0.0),
    )

    solved = solve_constraint(mc, pixel_w)
    regions = generate_regions(solved, pixel_h, pixel_d, pixel_w)

    out = dict(config)
    out["regions"] = regions
    out["manufacturing_resolved"] = {
        "matrix_edge_width_mm": solved.matrix_edge_width_mm,
        "groove_width_mm": solved.groove_width_mm,
        "wall_width_mm": solved.wall_width_mm,
        "strategy": solved.strategy,
        "total_mm": pixel_w,
    }
    return out
