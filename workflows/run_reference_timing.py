from __future__ import annotations
import sys
from dataclasses import replace
from _bootstrap import ROOT

from config.loader import load_benchmark_config
from simulation.uniform_benchmark import run_uniform_material
from signal.timing import ScintillationPulse
from signal.analysis import compute_markers


def build_pulse(materials):
    return {
        role: ScintillationPulse(mat.scintillation)
        for role, mat in materials.items()
        if mat.scintillation is not None
    }


def main(n_events: int = 50000, threshold: int = 10):
    cfg = load_benchmark_config(
        ROOT / "configs" / "benchmarks" / "uniform_bgo_lso_511keV.json"
    )

    if n_events != cfg.simulation.n_events:
        cfg = replace(cfg, simulation=replace(cfg.simulation, n_events=n_events))

    pulse_by_material = build_pulse(cfg.materials)

    for role in ["BGO", "LSO"]:
        print(f"\n{'='*60}")
        print(f"  Uniform {role} reference")
        print(f"{'='*60}")

        mat = cfg.materials[role]
        if mat.scintillation:
            p = pulse_by_material[role]
            comps = ", ".join(
                f"{c.amplitude_fraction*100:.2g}%x{c.decay_time_ns:.4g}ns"
                for c in p.components
            )
            print(f"  Scintillation: LY={p.ly:.1f}/keV, tr={p.tau_r:.2f}ns, decays=[{comps}]")

        result = run_uniform_material(
            role, mat, cfg, seed_offset=1000,
            collect_per_event_data=True,
        )

        single_pulse = {role: pulse_by_material[role]}
        markers = compute_markers(result, single_pulse, cfg.materials, threshold)

        print(f"  Benchmark:")
        print(f"    Full-absorption events:   {markers['n_full_absorption']} / {result['n_events']}")
        print(f"    Full-absorption fraction: {markers['full_fraction']:.5f}")
        print(f"    Interacted fraction:      {result['interacted_fraction']:.5f}")
        print(f"  Timing (threshold = {threshold} photons):")
        print(f"    Avg fast-component keV:   {markers['avg_fast_component_energy_keV']:.2f}")
        print(f"    Avg detection time (ns):  {markers['avg_detection_time_ns']:.2f}")
        print(f"    Median detection time:    {markers['median_detection_time_ns']:.2f}")
        print(f"    Std detection time:       {markers['std_detection_time_ns']:.2f}")
        print(f"  Figure of merit:")
        fss = markers['full_fraction'] ** 2
        adt = markers['avg_detection_time_ns']
        print(f"    FOM = ({markers['full_fraction']:.4f})² / {adt:.2f} = {markers['figure_of_merit']:.6f}")

    from io_utils.jsonio import dump_json
    # also run composite for comparison
    print(f"\n{'='*60}")
    print(f"  Composite BGO/BaF₂ dense (for comparison)")
    print(f"{'='*60}")
    from config.loader import resolved_dict_to_benchmark_config
    from config.merge import resolve_run_config_dict
    from geometry.composite_pixel import CompositeGeometry
    from geometry.manufacturing import apply_manufacturing
    from simulation.composite_benchmark import run_composite_benchmark
    import json, copy
    base_benchmark = json.loads((ROOT / "configs" / "benchmarks" / "composite_bgo_baf2_dense_511keV.json").read_text())
    base_geo = json.loads((ROOT / "configs" / "geometry" / "bgo_baf2_dense_manufacturing.json").read_text())
    raw = dict(base_benchmark)
    raw["_config_dir"] = str(ROOT / "configs")
    resolved = resolve_run_config_dict(raw)
    resolved["default_material_role"] = raw.get("default_material_role", "")
    resolved["geometry"] = dict(base_geo)
    resolved["geometry"] = apply_manufacturing(resolved["geometry"])
    cfg2 = resolved_dict_to_benchmark_config(resolved)
    sim_override = replace(cfg2.simulation, n_events=n_events)
    cfg2 = replace(cfg2, simulation=sim_override)
    geometry = CompositeGeometry(
        pixel_x_mm=cfg2.geometry.pixel_x_mm, pixel_y_mm=cfg2.geometry.pixel_y_mm,
        pixel_z_mm=cfg2.geometry.pixel_z_mm, regions=cfg2.geometry.regions,
        default_material_role=cfg2.default_material_role,
    )
    pulse2 = build_pulse(cfg2.materials)
    result2 = run_composite_benchmark(cfg2.materials, geometry, cfg2, seed_offset=2000, collect_per_event_data=True)
    markers2 = compute_markers(result2, pulse2, cfg2.materials, threshold)
    print(f"  Benchmark:")
    print(f"    Full-absorption events:   {markers2['n_full_absorption']} / {result2['n_events']}")
    print(f"    Full-absorption fraction: {markers2['full_fraction']:.5f}")
    print(f"    Interacted fraction:      {result2['interacted_fraction']:.5f}")
    print(f"  Timing:")
    print(f"    Avg fast-component keV:   {markers2['avg_fast_component_energy_keV']:.2f}")
    print(f"    Avg detection time (ns):  {markers2['avg_detection_time_ns']:.2f}")
    print(f"    Median detection time:    {markers2['median_detection_time_ns']:.2f}")
    print(f"  FOM = ({markers2['full_fraction']:.4f})² / {markers2['avg_detection_time_ns']:.2f} = {markers2['figure_of_merit']:.6f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    th = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    main(n, th)
