from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import numpy as np
from config.schema import RegionConfig
from geometry.composite_pixel import CompositeGeometry
from config.loader import load_benchmark_config


def test_uniform_geometry_no_regions():
    cfg = load_benchmark_config(ROOT / 'configs' / 'benchmarks' / 'uniform_bgo_lso_511keV.json')
    geo = CompositeGeometry(cfg.geometry.pixel_x_mm, cfg.geometry.pixel_y_mm, cfg.geometry.pixel_z_mm, (), 'BGO')
    pos = np.array([0.0, 0.0, 5.0])
    assert geo.material_role_at(pos) == 'BGO'


def test_composite_channel_geometry_load():
    geo_cfg = load_benchmark_config(ROOT / 'configs' / 'benchmarks' / 'composite_bgo_baf2_511keV.json')
    assert len(geo_cfg.geometry.regions) == 12
    for r in geo_cfg.geometry.regions:
        assert r.material_role == 'BaF2'
    assert geo_cfg.default_material_role == 'BGO'


def test_composite_material_lookup():
    geo = CompositeGeometry(
        3.0, 3.0, 15.0,
        (
            RegionConfig('chan', 'BaF2', -0.125, 0.125, -1.5, 1.5, 0, 15),
        ),
        'BGO',
    )
    assert geo.material_role_at(np.array([0.0, 0.0, 5.0])) == 'BaF2'
    assert geo.material_role_at(np.array([1.0, 0.0, 5.0])) == 'BGO'


def test_distance_to_next_material_change_enters_region():
    geo = CompositeGeometry(
        3.0, 3.0, 15.0,
        (
            RegionConfig('chan', 'BaF2', -0.5, 0.5, -1.5, 1.5, 0, 15),
        ),
        'BGO',
    )
    pos = np.array([-1.0, 0.0, 5.0])
    dir_vec = np.array([1.0, 0.0, 0.0])
    dist, next_mat = geo.distance_to_next_material_change(pos, dir_vec)
    assert abs(dist - 0.5) < 1e-6
    assert next_mat == 'BaF2'


def test_distance_to_next_material_change_exits_region():
    geo = CompositeGeometry(
        3.0, 3.0, 15.0,
        (
            RegionConfig('chan', 'BaF2', -0.5, 0.5, -1.5, 1.5, 0, 15),
        ),
        'BGO',
    )
    pos = np.array([0.0, 0.0, 5.0])
    dir_vec = np.array([1.0, 0.0, 0.0])
    dist, next_mat = geo.distance_to_next_material_change(pos, dir_vec)
    assert abs(dist - 0.5) < 1e-6
    assert next_mat == 'BGO'


def test_distance_to_box_mm():
    geo = CompositeGeometry(3.0, 3.0, 15.0, (), 'BGO')
    pos = np.array([0.0, 0.0, 0.0])
    dir_vec = np.array([0.0, 0.0, 1.0])
    assert abs(geo.distance_to_box_mm(pos, dir_vec) - 15.0) < 1e-6


def test_electron_transport_crosses_material_boundary():
    from simulation.composite_benchmark import transport_electron_csda
    from physics.electron_tabata import TabataRangeModel
    mats = {
        'BGO': load_benchmark_config(ROOT / 'configs' / 'benchmarks' / 'uniform_bgo_lso_511keV.json').materials['BGO'],
        'BaF2': load_benchmark_config(ROOT / 'configs' / 'benchmarks' / 'composite_bgo_baf2_511keV.json').materials['BaF2'],
    }
    range_models = {role: TabataRangeModel(m.formula, m.density_g_cm3) for role, m in mats.items()}
    geo = CompositeGeometry(
        3.0, 3.0, 15.0,
        (RegionConfig('chan', 'BaF2', -0.1, 0.1, -1.5, 1.5, 0, 15),),
        'BGO',
    )
    dist_to_boundary = 0.1
    pos = np.array([-0.2, 0.0, 7.5])
    dir_vec = np.array([1.0, 0.0, 0.0])
    E = 300.0
    rm_bgo = range_models['BGO']
    assert rm_bgo.range_mm(E) > dist_to_boundary, 'need enough range to cross boundary'
    result = transport_electron_csda(range_models, pos, dir_vec, E, geo)
    assert 'BGO' in result['deposition_by_material_keV']
    assert 'BaF2' in result['deposition_by_material_keV']
    assert result['deposition_by_material_keV']['BaF2'] > 0.0
    assert result['deposited_keV'] + result['escaped_keV'] <= E + 1e-9


def test_composite_benchmark_smoke_with_per_material_deposition():
    import numpy as np
    from simulation.composite_benchmark import run_composite_benchmark
    from config.schema import BenchmarkConfig, SimulationConfig, IrradiationConfig
    geo = CompositeGeometry(
        3.0, 3.0, 15.0,
        (RegionConfig('right_half', 'BaF2', 0.0, 1.5, -1.5, 1.5, 0, 15),),
        'BGO',
    )
    mats = {
        'BGO': load_benchmark_config(ROOT / 'configs' / 'benchmarks' / 'uniform_bgo_lso_511keV.json').materials['BGO'],
        'BaF2': load_benchmark_config(ROOT / 'configs' / 'benchmarks' / 'composite_bgo_baf2_511keV.json').materials['BaF2'],
    }
    sim = SimulationConfig(gamma_energy_keV=511.0, n_events=50, seed=42)
    irr = IrradiationConfig(front_face_area_fraction=0.95)
    cfg = BenchmarkConfig(materials=mats, geometry=geo, irradiation=irr, simulation=sim, default_material_role='BGO')
    result = run_composite_benchmark(mats, geo, cfg)
    dep_by_mat = result['deposition_by_material_keV']
    assert 'BGO' in dep_by_mat
    assert 'BaF2' in dep_by_mat
    assert dep_by_mat['BGO'] + dep_by_mat['BaF2'] == result['total_deposited_keV']
    assert result['total_deposited_keV'] > 0.0
    assert result['deposition_by_material_fraction']['BGO'] > 0.0
    assert result['deposition_by_material_fraction']['BaF2'] > 0.0
    for ex in result['examples']:
        assert 'deposited_by_material_keV' in ex
        assert abs(sum(ex['deposited_by_material_keV'].values()) - ex['deposited_keV']) < 1e-9


def test_photon_exits_pixel_when_no_next_material():
    geo = CompositeGeometry(
        3.0, 3.0, 15.0,
        (
            RegionConfig('chan', 'BaF2', -0.5, 0.5, -1.5, 1.5, 0, 15),
        ),
        'BGO',
    )
    pos = np.array([0.0, 0.0, 14.0])
    dir_vec = np.array([0.0, 0.0, 1.0])
    dist, next_mat = geo.distance_to_next_material_change(pos, dir_vec)
    assert abs(dist - 1.0) < 1e-6
    assert next_mat is None


def test_tilted_region_point_containment():
    geo = CompositeGeometry(
        3.0, 3.0, 15.0,
        (RegionConfig('groove', 'BaF2', -2.0, 2.0, -0.1, 0.1, 0, 15, tilt_deg=45.0),),
        'BGO',
    )
    # Tilt is around y-axis (xz-plane). y-bounds [-0.1, 0.1] are unaffected.
    assert geo.material_role_at(np.array([0.0, 0.0, 7.5])) == 'BaF2'
    assert geo.material_role_at(np.array([0.5, 0.0, 7.5])) == 'BaF2'
    assert geo.material_role_at(np.array([0.0, 0.2, 7.5])) == 'BGO'
    assert geo.material_role_at(np.array([-0.5, -0.2, 7.5])) == 'BGO'


def test_tilted_region_ray_intersection():
    geo = CompositeGeometry(
        3.0, 3.0, 15.0,
        (RegionConfig('groove', 'BaF2', -2.0, 2.0, -0.1, 0.1, 0, 15, tilt_deg=45.0),),
        'BGO',
    )
    pos = np.array([0.0, -1.0, 7.5])
    dir_vec = np.array([0.0, 1.0, 0.0])
    dist, next_mat = geo.distance_to_next_material_change(pos, dir_vec)
    assert next_mat == 'BaF2'
    assert abs(dist - 0.9) < 0.01


def test_tilted_region_smoke_benchmark():
    from simulation.composite_benchmark import run_composite_benchmark
    from config.schema import BenchmarkConfig, SimulationConfig, IrradiationConfig
    geo = CompositeGeometry(
        3.0, 3.0, 15.0,
        (RegionConfig('groove', 'BaF2', -2.0, 2.0, -0.2, 0.2, 0, 15, tilt_deg=30.0),),
        'BGO',
    )
    mats = {
        'BGO': load_benchmark_config(ROOT / 'configs' / 'benchmarks' / 'uniform_bgo_lso_511keV.json').materials['BGO'],
        'BaF2': load_benchmark_config(ROOT / 'configs' / 'benchmarks' / 'composite_bgo_baf2_511keV.json').materials['BaF2'],
    }
    sim = SimulationConfig(gamma_energy_keV=511.0, n_events=20, seed=42)
    irr = IrradiationConfig(front_face_area_fraction=0.95)
    cfg = BenchmarkConfig(materials=mats, geometry=geo, irradiation=irr, simulation=sim, default_material_role='BGO')
    result = run_composite_benchmark(mats, geo, cfg)
    assert result['total_deposited_keV'] > 0.0
    assert result['deposition_by_material_keV']['BaF2'] > 0.0
    assert result['deposition_by_material_keV']['BGO'] > 0.0


def test_ray_trace_vertical():
    geo = CompositeGeometry(3.0, 3.0, 15.0, (), 'BGO')
    segs = geo.trace_ray(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    assert len(segs) == 1
    assert segs[0]['material_role'] == 'BGO'
    assert abs(segs[0]['length_mm'] - 15.0) < 1e-6
    assert segs[0]['exits_pixel']


def test_ray_trace_starts_outside():
    geo = CompositeGeometry(3.0, 3.0, 15.0, (), 'BGO')
    segs = geo.trace_ray(np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0]))
    assert len(segs) == 1
    assert abs(segs[0]['length_mm'] - 15.0) < 1e-6
    assert segs[0]['exits_pixel']


def test_ray_trace_crosses_channel():
    geo = CompositeGeometry(
        3.0, 3.0, 15.0,
        (RegionConfig('chan', 'BaF2', -0.5, 0.5, -1.5, 1.5, 0, 15),),
        'BGO',
    )
    segs = geo.trace_ray(np.array([-1.0, 0.0, 7.5]), np.array([1.0, 0.0, 0.0]))
    assert len(segs) >= 3
    assert segs[0]['material_role'] == 'BGO'
    assert segs[1]['material_role'] == 'BaF2'
    assert segs[-1]['material_role'] == 'BGO'
    assert segs[-1]['exits_pixel']
    total = sum(s['length_mm'] for s in segs)
    assert abs(total - 2.5) < 1e-6



