from __future__ import annotations
import numpy as np
from config.schema import ScintillationConfig


class ScintillationPulse:
    def __init__(self, config: ScintillationConfig):
        self.ly = config.light_yield_per_keV
        self.tau_r = config.rise_time_ns
        self.components = config.decay_components

    def total_photons(self, energy_keV: float) -> float:
        return self.ly * energy_keV

    def cumulative_photons(self, t_ns: float | np.ndarray, energy_keV: float) -> float | np.ndarray:
        if energy_keV <= 0.0:
            return np.zeros_like(t_ns) if isinstance(t_ns, np.ndarray) else 0.0

        total = np.zeros_like(t_ns) if isinstance(t_ns, np.ndarray) else 0.0
        for comp in self.components:
            tau_d = comp.decay_time_ns
            amp = comp.amplitude_fraction
            if comp.pulse_model == "exponential":
                integral = 1.0 - np.exp(-t_ns / tau_d)
            else:
                if abs(tau_d - self.tau_r) > 1e-12:
                    integral = 1.0 - (
                        tau_d * np.exp(-t_ns / tau_d) - self.tau_r * np.exp(-t_ns / self.tau_r)
                    ) / (tau_d - self.tau_r)
                else:
                    integral = 1.0 - (1.0 + t_ns / tau_d) * np.exp(-t_ns / tau_d)
            total += amp * np.maximum(integral, 0.0)
        return self.ly * energy_keV * total

    def photon_rate_ns1(self, t_ns: float | np.ndarray, energy_keV: float) -> float | np.ndarray:
        if energy_keV <= 0.0:
            return np.zeros_like(t_ns) if isinstance(t_ns, np.ndarray) else 0.0

        total = np.zeros_like(t_ns) if isinstance(t_ns, np.ndarray) else 0.0
        for comp in self.components:
            tau_d = comp.decay_time_ns
            amp = comp.amplitude_fraction
            if comp.pulse_model == "exponential":
                total += amp * np.exp(-t_ns / tau_d) / tau_d
            else:
                if abs(tau_d - self.tau_r) > 1e-12:
                    norm = 1.0 / (tau_d - self.tau_r)
                else:
                    norm = 1.0 / tau_d
                total += amp * norm * (np.exp(-t_ns / tau_d) - np.exp(-t_ns / self.tau_r))
        return self.ly * energy_keV * np.maximum(total, 0.0)


def compute_detection_time(
    energy_by_material_keV: dict[str, float],
    pulse_by_material: dict[str, ScintillationPulse],
    threshold_photons: int = 10,
    t_max_ns: float = 10000.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    if threshold_photons <= 0:
        return 0.0

    total_photons = sum(
        pulse_by_material[mat].total_photons(e)
        for mat, e in energy_by_material_keV.items()
        if mat in pulse_by_material
    )
    if total_photons < threshold_photons:
        return float("inf")

    def cum(t: float) -> float:
        c = 0.0
        for mat, e in energy_by_material_keV.items():
            if mat in pulse_by_material and e > 0.0:
                c += pulse_by_material[mat].cumulative_photons(t, e)
        return c

    hi = 1e-6
    while cum(hi) < threshold_photons and hi < t_max_ns:
        hi *= 2.0
    if hi >= t_max_ns and cum(t_max_ns) < threshold_photons:
        return float("inf")

    lo = 0.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if cum(mid) < threshold_photons:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0
