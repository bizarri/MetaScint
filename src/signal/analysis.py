from __future__ import annotations
import numpy as np
from config.schema import MaterialConfig
from signal.timing import ScintillationPulse, compute_detection_time

_FAST_COMPONENT_THRESHOLD_NS = 10.0


def fast_component_energy_keV(dep_by_mat: dict[str, float], materials: dict[str, MaterialConfig]) -> float:
    total = 0.0
    for role, energy in dep_by_mat.items():
        sc = materials[role].scintillation
        if sc is not None and energy > 0.0:
            for comp in sc.decay_components:
                if comp.decay_time_ns < _FAST_COMPONENT_THRESHOLD_NS:
                    total += energy * comp.amplitude_fraction
    return total


def compute_markers(
    result: dict,
    pulse_by_material: dict[str, ScintillationPulse],
    materials: dict[str, MaterialConfig],
    threshold_photons: int = 10,
) -> dict:
    full_fraction = result.get("full_fraction", 0.0)
    events_data = result.get("events_data", [])

    fast_energies = []
    detect_times = []

    for ev in events_data:
        if ev.get("classification") == "full":
            dep = ev.get("deposition_by_material_keV", {})
            fast_energies.append(fast_component_energy_keV(dep, materials))
            t = compute_detection_time(dep, pulse_by_material, threshold_photons)
            detect_times.append(t)

    n_full = len(fast_energies)
    finite_times = np.array([t for t in detect_times if np.isfinite(t)], dtype=float)

    avg_fast_keV = float(np.mean(fast_energies)) if fast_energies else 0.0
    avg_detect_ns = float(np.mean(finite_times)) if len(finite_times) > 0 else float("inf")
    median_detect_ns = float(np.median(finite_times)) if len(finite_times) > 0 else float("inf")
    std_detect_ns = float(np.std(finite_times)) if len(finite_times) > 1 else 0.0

    if avg_detect_ns > 0.0 and np.isfinite(avg_detect_ns):
        fom = (full_fraction ** 2) / avg_detect_ns
    else:
        fom = 0.0

    return {
        "n_full_absorption": n_full,
        "full_fraction": full_fraction,
        "avg_fast_component_energy_keV": avg_fast_keV,
        "avg_detection_time_ns": avg_detect_ns,
        "median_detection_time_ns": median_detect_ns,
        "std_detection_time_ns": std_detect_ns,
        "detection_times_ns": [float(t) if np.isfinite(t) else None for t in detect_times],
        "figure_of_merit": fom,
        "fom_components": {
            "full_fraction_squared": float(full_fraction ** 2),
            "avg_detection_time_ns": avg_detect_ns,
        },
    }
