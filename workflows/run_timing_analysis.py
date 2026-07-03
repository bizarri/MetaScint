from __future__ import annotations
import sys
from dataclasses import replace
import numpy as np
from _bootstrap import ROOT

from config.loader import load_benchmark_config
from geometry.composite_pixel import CompositeGeometry
from simulation.composite_benchmark import run_composite_benchmark
from signal.timing import ScintillationPulse
from signal.analysis import compute_markers


def build_pulse_by_material(materials):
    pulse_by_material = {}
    for role, mat in materials.items():
        if mat.scintillation is not None:
            pulse_by_material[role] = ScintillationPulse(mat.scintillation)
    return pulse_by_material


def main(config_path: str, threshold: int = 10, n_events_override: int | None = None):
    cfg = load_benchmark_config(config_path)

    pulse_by_material = build_pulse_by_material(cfg.materials)
    if not pulse_by_material:
        print("No materials have scintillation data. Nothing to do.")
        return

    print(f"Materials with scintillation data: {list(pulse_by_material.keys())}")
    for role, p in pulse_by_material.items():
        comps = ", ".join(
            f"{c.amplitude_fraction*100:.2g}%x{c.decay_time_ns:.4g}ns"
            for c in p.components
        )
        print(f"  {role}: LY={p.ly:.1f}/keV, tr={p.tau_r:.2f}ns, decays=[{comps}]")

    if n_events_override is not None:
        cfg = replace(cfg, simulation=replace(cfg.simulation, n_events=n_events_override))

    run_evts = cfg.simulation.n_events
    print(f"\nRunning {run_evts} events with per-event data collection...")

    geometry = CompositeGeometry(
        pixel_x_mm=cfg.geometry.pixel_x_mm,
        pixel_y_mm=cfg.geometry.pixel_y_mm,
        pixel_z_mm=cfg.geometry.pixel_z_mm,
        regions=cfg.geometry.regions,
        default_material_role=cfg.default_material_role,
    )

    result = run_composite_benchmark(
        cfg.materials, geometry, cfg, seed_offset=0,
        collect_per_event_data=True,
    )

    markers = compute_markers(result, pulse_by_material, cfg.materials, threshold)

    print(f"\n  Optimization markers (threshold = {threshold} photons):")
    print(f"    Full-absorption events:    {markers['n_full_absorption']} / {run_evts}")
    print(f"    Full-absorption fraction:  {markers['full_fraction']:.5f}")
    print(f"    Avg fast-component energy: {markers['avg_fast_component_energy_keV']:.2f} keV")
    print(f"    Avg detection time (full): {markers['avg_detection_time_ns']:.2f} ns")
    print(f"    Median detection time:     {markers['median_detection_time_ns']:.2f} ns")
    print(f"    Std detection time:        {markers['std_detection_time_ns']:.2f} ns")
    print(f"    Figure of merit:           {markers['figure_of_merit']:.6f}")
    print(f"      FOM = ({markers['full_fraction']:.5f})^2 / {markers['avg_detection_time_ns']:.2f}")

    out = {**markers, "threshold_photons": threshold, "n_events": run_evts}
    out_path = ROOT / "outputs" / "timing_analysis.json"
    from io_utils.jsonio import dump_json
    dump_json(out, out_path)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else str(
        ROOT / "configs" / "benchmarks" / "composite_bgo_baf2_dense_511keV.json"
    )
    n_threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    n_events_override = int(sys.argv[3]) if len(sys.argv) > 3 else None
    main(cfg_path, n_threshold, n_events_override)
