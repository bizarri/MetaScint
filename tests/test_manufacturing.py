from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import json
import math
import pytest
from geometry.manufacturing import (
    ManufacturingConstraint,
    solve_constraint,
    generate_regions,
    check_manufacturing_rules,
    apply_manufacturing,
)
from config.loader import load_benchmark_config


# ── solve_constraint ──────────────────────────────────────────────────────────

def test_adjust_walls():
    mc = ManufacturingConstraint('x', 12, 0.2, 'BaF2', 0.05, 'BGO', 0.025, 'adjust_walls')
    solved = solve_constraint(mc, 3.0)
    assert abs(solved.matrix_edge_width_mm - 0.025) < 1e-9
    assert abs(solved.groove_width_mm - 0.2) < 1e-9
    assert abs(solved.wall_width_mm - 0.05) < 1e-9
    total = 2 * solved.matrix_edge_width_mm + 12 * solved.groove_width_mm + 11 * solved.wall_width_mm
    assert abs(total - 3.0) < 1e-9


def test_adjust_walls_fixes_off_by_one():
    mc = ManufacturingConstraint('x', 12, 0.2, 'BaF2', 0.04, 'BGO', 0.025, 'adjust_walls')
    solved = solve_constraint(mc, 3.0)
    expected_wall = (3.0 - 2 * 0.025 - 12 * 0.2) / 11
    assert abs(solved.wall_width_mm - expected_wall) < 1e-9
    assert abs(solved.groove_width_mm - 0.2) < 1e-9
    assert abs(solved.matrix_edge_width_mm - 0.025) < 1e-9


def test_adjust_grooves():
    mc = ManufacturingConstraint('x', 10, 0.3, 'BaF2', 0.06, 'BGO', 0.04, 'adjust_grooves')
    solved = solve_constraint(mc, 5.0)
    expected_groove = (5.0 - 2 * 0.04 - 9 * 0.06) / 10
    assert abs(solved.groove_width_mm - expected_groove) < 1e-9
    assert abs(solved.wall_width_mm - 0.06) < 1e-9
    assert abs(solved.matrix_edge_width_mm - 0.04) < 1e-9


def test_adjust_matrix_edges():
    mc = ManufacturingConstraint('x', 8, 0.25, 'BaF2', 0.05, 'BGO', 0.03, 'adjust_matrix_edges')
    solved = solve_constraint(mc, 4.0)
    expected_edge = (4.0 - 8 * 0.25 - 7 * 0.05) / 2
    assert abs(solved.matrix_edge_width_mm - expected_edge) < 1e-9
    assert abs(solved.groove_width_mm - 0.25) < 1e-9
    assert abs(solved.wall_width_mm - 0.05) < 1e-9


def test_minimize_change():
    mc = ManufacturingConstraint('x', 10, 0.25, 'BaF2', 0.06, 'BGO', 0.03, 'minimize_change')
    total0 = 2 * 0.03 + 10 * 0.25 + 9 * 0.06
    assert abs(total0 - 3.1) < 1e-9
    solved = solve_constraint(mc, 3.0)
    total = 2 * solved.matrix_edge_width_mm + 10 * solved.groove_width_mm + 9 * solved.wall_width_mm
    assert abs(total - 3.0) < 1e-9
    assert solved.matrix_edge_width_mm < 0.03
    assert solved.groove_width_mm < 0.25
    assert solved.wall_width_mm < 0.06


def test_already_satisfied_no_change():
    mc = ManufacturingConstraint('x', 12, 0.2, 'BaF2', 0.05, 'BGO', 0.025, 'adjust_walls')
    solved = solve_constraint(mc, 3.0)
    assert abs(solved.matrix_edge_width_mm - 0.025) < 1e-9
    assert abs(solved.groove_width_mm - 0.2) < 1e-9
    assert abs(solved.wall_width_mm - 0.05) < 1e-9


def test_single_channel():
    mc = ManufacturingConstraint('x', 1, 0.5, 'BaF2', 0.1, 'BGO', 0.2, 'adjust_matrix_edges')
    solved = solve_constraint(mc, 1.0)
    total = 2 * solved.matrix_edge_width_mm + 1 * solved.groove_width_mm
    assert abs(total - 1.0) < 1e-9
    assert abs(solved.groove_width_mm - 0.5) < 1e-9
    expected_edge = (1.0 - 0.5) / 2.0
    assert abs(solved.matrix_edge_width_mm - expected_edge) < 1e-9


def test_negative_wall_raises():
    mc = ManufacturingConstraint('x', 12, 0.2, 'BaF2', 0.05, 'BGO', 0.025, 'adjust_walls')
    with pytest.raises(ValueError, match='<= 0'):
        solve_constraint(mc, 0.5)


def test_unknown_strategy_raises():
    mc = ManufacturingConstraint('x', 5, 0.2, 'BaF2', 0.05, 'BGO', 0.02, 'bogus')
    with pytest.raises(ValueError, match='Unknown strategy'):
        solve_constraint(mc, 3.0)


# ── generate_regions ─────────────────────────────────────────────────────────

def test_generate_regions_count():
    mc = ManufacturingConstraint('x', 12, 0.2, 'BaF2', 0.05, 'BGO', 0.025, 'adjust_walls')
    solved = solve_constraint(mc, 3.0)
    regions = generate_regions(solved, 3.0, 15.0, 3.0)
    assert len(regions) == 2 + 12 + 11  # matrix edges + grooves + walls


def test_generate_regions_boundaries():
    mc = ManufacturingConstraint('x', 6, 0.3, 'BaF2', 0.05, 'BGO', 0.05, 'adjust_walls')
    solved = solve_constraint(mc, 3.0)
    regions = generate_regions(solved, 3.0, 15.0, 3.0)
    assert abs(regions[0]['xmin'] + 1.5) < 1e-9
    assert abs(regions[-1]['xmax'] - 1.5) < 1e-9
    for r in regions:
        assert abs(r['ymin'] + 1.5) < 1e-9
        assert abs(r['ymax'] - 1.5) < 1e-9
        assert abs(r['zmin'] - 0.0) < 1e-9
        assert abs(r['zmax'] - 15.0) < 1e-9


def test_generate_regions_contiguous():
    mc = ManufacturingConstraint('x', 8, 0.2, 'BaF2', 0.06, 'BGO', 0.04, 'adjust_walls')
    solved = solve_constraint(mc, 3.0)
    regions = generate_regions(solved, 3.0, 15.0, 3.0)
    for i in range(len(regions) - 1):
        assert abs(regions[i]['xmax'] - regions[i + 1]['xmin']) < 1e-9


def test_generate_regions_y_axis():
    mc = ManufacturingConstraint('y', 5, 0.3, 'BaF2', 0.05, 'BGO', 0.1, 'adjust_walls')
    solved = solve_constraint(mc, 3.0)
    regions = generate_regions(solved, 3.0, 15.0, 3.0)
    for r in regions:
        assert abs(r['xmin'] + 1.5) < 1e-9
        assert abs(r['xmax'] - 1.5) < 1e-9
        assert r['name'].startswith('matrix_') or r['name'].startswith('groove_') or r['name'].startswith('wall_')


def test_generate_regions_single_channel():
    mc = ManufacturingConstraint('x', 1, 0.5, 'BaF2', 0.1, 'BGO', 0.2, 'adjust_matrix_edges')
    solved = solve_constraint(mc, 0.9)
    regions = generate_regions(solved, 3.0, 15.0, 0.9)
    assert len(regions) == 3  # matrix_left, groove_0, matrix_right
    assert regions[0]['material_role'] == 'BGO'
    assert regions[1]['material_role'] == 'BaF2'
    assert regions[2]['material_role'] == 'BGO'


# ── check_manufacturing_rules ─────────────────────────────────────────────────

def test_check_rules_valid():
    config = {
        'pixel_x_mm': 3.0,
        'manufacturing': {
            'axis': 'x', 'n_channels': 12,
            'groove_width_mm': 0.2, 'groove_material_role': 'BaF2',
            'wall_width_mm': 0.05, 'matrix_material_role': 'BGO',
            'matrix_edge_width_mm': 0.025, 'strategy': 'adjust_walls',
        },
    }
    issues = check_manufacturing_rules(config)
    assert len(issues) == 0


def test_check_rules_already_satisfied():
    config = {
        'pixel_x_mm': 3.0,
        'manufacturing': {
            'axis': 'x', 'n_channels': 12,
            'groove_width_mm': 0.2, 'groove_material_role': 'BaF2',
            'wall_width_mm': 0.05, 'matrix_material_role': 'BGO',
            'matrix_edge_width_mm': 0.025, 'strategy': 'adjust_walls',
        },
    }
    issues = check_manufacturing_rules(config)
    assert len(issues) == 0


def test_check_rules_mismatch():
    config = {
        'pixel_x_mm': 3.0,
        'manufacturing': {
            'axis': 'x', 'n_channels': 12,
            'groove_width_mm': 0.3, 'groove_material_role': 'BaF2',
            'wall_width_mm': 0.05, 'matrix_material_role': 'BGO',
            'matrix_edge_width_mm': 0.025, 'strategy': 'adjust_walls',
        },
    }
    issues = check_manufacturing_rules(config)
    assert len(issues) >= 1
    assert 'cannot be satisfied' in issues[-1]


def test_check_rules_missing_keys():
    config = {'pixel_x_mm': 3.0, 'manufacturing': {'axis': 'x'}}
    issues = check_manufacturing_rules(config)
    assert len(issues) >= 4


def test_check_rules_non_positive():
    config = {
        'pixel_x_mm': 3.0,
        'manufacturing': {
            'axis': 'x', 'n_channels': 0,
            'groove_width_mm': -1, 'groove_material_role': 'BaF2',
            'wall_width_mm': 0, 'matrix_material_role': 'BGO',
            'matrix_edge_width_mm': 0.025, 'strategy': 'adjust_walls',
        },
    }
    issues = check_manufacturing_rules(config)
    assert len(issues) >= 3


def test_check_rules_no_manufacturing():
    assert check_manufacturing_rules({'pixel_x_mm': 3.0}) == []


def test_check_rules_y_axis():
    config = {
        'pixel_y_mm': 5.0, 'pixel_x_mm': 3.0,
        'manufacturing': {
            'axis': 'y', 'n_channels': 8,
            'groove_width_mm': 0.4, 'groove_material_role': 'BaF2',
            'wall_width_mm': 0.08, 'matrix_material_role': 'BGO',
            'matrix_edge_width_mm': 0.06, 'strategy': 'adjust_grooves',
        },
    }
    issues = check_manufacturing_rules(config)
    assert len(issues) == 0


# ── apply_manufacturing ──────────────────────────────────────────────────────

def test_apply_manufacturing_no_mfg():
    config = {'pixel_x_mm': 3.0, 'regions': []}
    result = apply_manufacturing(config)
    assert result is config


def test_apply_manufacturing_generates_regions():
    config = {
        'pixel_x_mm': 3.0, 'pixel_y_mm': 3.0, 'pixel_z_mm': 15.0,
        'manufacturing': {
            'axis': 'x', 'n_channels': 4,
            'groove_width_mm': 0.4, 'groove_material_role': 'BaF2',
            'wall_width_mm': 0.08, 'matrix_material_role': 'BGO',
            'matrix_edge_width_mm': 0.14, 'strategy': 'adjust_walls',
        },
    }
    result = apply_manufacturing(config)
    assert 'regions' in result
    assert len(result['regions']) == 2 + 4 + 3
    assert 'manufacturing_resolved' in result
    res = result['manufacturing_resolved']
    total_res = 2 * res['matrix_edge_width_mm'] + 4 * res['groove_width_mm'] + 3 * res['wall_width_mm']
    assert abs(total_res - 3.0) < 1e-9


# ── Pipeline integration ──────────────────────────────────────────────────────

def test_manufacturing_benchmark_loads():
    cfg = load_benchmark_config(
        ROOT / 'configs' / 'benchmarks' / 'composite_bgo_baf2_mfg_511keV.json'
    )
    assert len(cfg.geometry.regions) == 25
    assert cfg.default_material_role == 'BGO'
    assert cfg.geometry.pixel_x_mm == 3.0


def test_manufacturing_benchmark_runs():
    from simulation.composite_benchmark import run_composite_benchmark
    from config.schema import SimulationConfig, IrradiationConfig, BenchmarkConfig
    cfg = load_benchmark_config(
        ROOT / 'configs' / 'benchmarks' / 'composite_bgo_baf2_mfg_511keV.json'
    )
    geo = __import__('geometry.composite_pixel', fromlist=['CompositeGeometry']).CompositeGeometry(
        cfg.geometry.pixel_x_mm, cfg.geometry.pixel_y_mm, cfg.geometry.pixel_z_mm,
        cfg.geometry.regions, cfg.default_material_role,
    )
    sim = SimulationConfig(gamma_energy_keV=511.0, n_events=30, seed=42)
    irr = IrradiationConfig(front_face_area_fraction=0.95)
    bc = BenchmarkConfig(materials=cfg.materials, geometry=cfg.geometry,
                         irradiation=irr, simulation=sim, default_material_role=cfg.default_material_role)
    result = run_composite_benchmark(cfg.materials, geo, bc)
    assert result['total_deposited_keV'] > 0
    assert result['deposition_by_material_keV']['BGO'] > 0
    assert result['deposition_by_material_keV']['BaF2'] > 0


def test_manufacturing_region_count_matches_explicit_grooves():
    mfg_cfg = load_benchmark_config(
        ROOT / 'configs' / 'benchmarks' / 'composite_bgo_baf2_mfg_511keV.json'
    )
    exp_cfg = load_benchmark_config(
        ROOT / 'configs' / 'benchmarks' / 'composite_bgo_baf2_511keV.json'
    )
    mfg_grooves = [r for r in mfg_cfg.geometry.regions if r.material_role == 'BaF2']
    exp_grooves = list(exp_cfg.geometry.regions)
    assert len(mfg_grooves) == len(exp_grooves) == 12
    for i in range(12):
        assert abs(mfg_grooves[i].xmin - exp_grooves[i].xmin) < 1e-9
        assert abs(mfg_grooves[i].xmax - exp_grooves[i].xmax) < 1e-9
