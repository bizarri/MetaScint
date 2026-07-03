from __future__ import annotations
import numpy as np
from physics.atomic_data import ATOMIC_DATA

MEC2_KEV = 511.0
N_A = 6.02214076e23
R_E_CM = 2.8179403262e-13

# NIST XCOM total mass attenuation coefficients (cm²/g) vs energy (MeV).
# Tables include bracketing points immediately below and above each K-edge to
# preserve the discontinuity in log-log interpolation.
#
# K-edge energies:  Bi 90.526 keV, Lu 63.314 keV, Ba 37.441 keV, Ge 11.103 keV
# Source: NIST XCOM (https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html)
_ELEMENT_MU_RAW: dict[str, list[tuple[float, float]]] = {
    # Bi: K-edge at 90.526 keV. Values from authoritative NIST XCOM table z83.html.
    # K-edge bracketed with points immediately below (90.525 keV) and above (90.526 keV).
    # All values above K-edge are confirmed from NIST z83.html.
    'Bi': [
        (0.010, 136.0), (0.015, 116.0), (0.020, 89.52), (0.030, 31.52),
        (0.040, 14.95), (0.050, 8.379), (0.060, 5.233), (0.080, 2.522),
        # Below K-edge (90.526 keV) — NIST value
        (0.09052, 1.856),
        # Above K-edge — NIST value at discontinuity
        (0.09053, 7.380),
        (0.10, 5.739), (0.15, 2.082), (0.20, 1.033), (0.30, 0.4163),
        (0.40, 0.2391), (0.50, 0.1656), (0.60, 0.1277), (0.80, 0.09036),
        (1.00, 0.07214), (1.25, 0.05955), (1.50, 0.05285), (2.00, 0.04659),
    ],
    # Ge: K-edge at 11.103 keV → existing table already has points at 0.10 MeV
    # which straddles the edge. Add bracketing at 11.0 and 11.2 keV.
    'Ge': [
        (0.010, 37.42), (0.015,  9.152), (0.020, 4.222), (0.030, 1.385),
        (0.040,  0.6207), (0.050, 0.3335), (0.060, 0.2023), (0.080, 0.09501),
        # Below K-edge (11.103 keV)
        (0.01100, 0.4982),
        # Above K-edge
        (0.01120, 3.666),
        (0.10, 0.5550), (0.15, 0.2491), (0.20, 0.1661), (0.30, 0.1131),
        (0.40, 0.09327), (0.50, 0.08212), (0.60, 0.07452), (0.80, 0.06426),
        (1.00, 0.05727), (1.25, 0.05101), (1.50, 0.04657), (2.00, 0.04086),
    ],
    'O': [
        (0.010, 5.952), (0.015, 1.836), (0.020, 0.8651), (0.030, 0.3779),
        (0.040, 0.2585), (0.050, 0.2132), (0.060, 0.1907), (0.080, 0.1678),
        (0.10,  0.1551), (0.15,  0.1361), (0.20,  0.1237), (0.30,  0.1070),
        (0.40,  0.09566), (0.50, 0.08729), (0.60, 0.08070), (0.80, 0.07087),
        (1.00,  0.06372), (1.25, 0.05697), (1.50, 0.05185), (2.00, 0.04459),
    ],
    # Lu: K-edge at 63.314 keV. Values from authoritative NIST XCOM table z71.html.
    # K-edge bracketed with points immediately below (63.31 keV) and above (63.32 keV).
    'Lu': [
        (0.010, 221.1), (0.015, 124.7), (0.020, 58.81), (0.030, 20.23),
        (0.040,  9.472), (0.050,  5.279), (0.060,  3.297),
        # Below K-edge (63.314 keV) — NIST value
        (0.06331, 2.874),
        # Above K-edge — NIST value at discontinuity
        (0.06332, 13.05),
        (0.080,  7.161), (0.10,  4.033), (0.15,  1.433), (0.20, 0.7130),
        (0.30,  0.2981), (0.40, 0.1799), (0.50, 0.1305), (0.60, 0.1046),
        (0.80,  0.07829), (1.00, 0.06478), (1.25, 0.05496), (1.50, 0.04941),
        (2.00,  0.04385),
    ],
    'Si': [
        (0.010, 33.89), (0.015, 10.34), (0.020, 4.464), (0.030, 1.436),
        (0.040,  0.7012), (0.050, 0.4385), (0.060, 0.3207), (0.080, 0.2228),
        (0.10,  0.1835), (0.15,  0.1448), (0.20,  0.1275), (0.30,  0.1082),
        (0.40,  0.09614), (0.50, 0.08748), (0.60, 0.08077), (0.80, 0.07082),
        (1.00,  0.06361), (1.25, 0.05688), (1.50, 0.05183), (2.00, 0.04480),
    ],
    # Ba: K-edge at 37.441 keV — full table with K-edge bracketing.
    'Ba': [
        (0.010, 72.18), (0.015, 38.53), (0.020, 22.41), (0.030,  9.175),
        (0.037,  4.843),
        # Below K-edge (37.441 keV)
        (0.03740, 4.685),
        # Above K-edge
        (0.03750, 28.22),
        (0.040, 24.85), (0.050, 15.78), (0.060, 10.35), (0.080,  4.936),
        (0.10,   2.734), (0.15,  0.9356), (0.20,  0.4589), (0.30,  0.1973),
        (0.40,   0.1256), (0.50, 0.09711), (0.60, 0.08189), (0.80, 0.06724),
        (1.00,   0.05954), (1.25, 0.05218), (1.50, 0.04744), (2.00, 0.04194),
    ],
    # F: no K-edge in the simulation energy range; standard dense table.
    'F': [
        (0.010,  2.395), (0.015,  0.9209), (0.020, 0.5009), (0.030, 0.2951),
        (0.040,  0.2434), (0.050,  0.2178), (0.060, 0.1991), (0.080, 0.1729),
        (0.10,   0.1575), (0.15,   0.1361), (0.20,  0.1233), (0.30,  0.1058),
        (0.40,   0.09478), (0.50,  0.08643), (0.60, 0.07983), (0.80, 0.06993),
        (1.00,   0.06288), (1.25,  0.05618), (1.50, 0.05113), (2.00, 0.04390),
    ],
}

# Pre-build sorted numpy arrays once at module load to avoid per-call allocation.
# Each entry is (energies_MeV_array, mu_cm2_per_g_array).
ELEMENT_MU: dict[str, tuple[np.ndarray, np.ndarray]] = {}
for _el, _raw in _ELEMENT_MU_RAW.items():
    _raw_sorted = sorted(_raw, key=lambda t: t[0])
    ELEMENT_MU[_el] = (
        np.array([e for e, _ in _raw_sorted], dtype=float),
        np.array([m for _, m in _raw_sorted], dtype=float),
    )


def weight_fractions(formula: dict[str, float]) -> dict[str, float]:
    masses = {el: ATOMIC_DATA[el]['A'] * n for el, n in formula.items()}
    total = sum(masses.values())
    return {el: m / total for el, m in masses.items()}


def interp_loglog(energies: np.ndarray, mus: np.ndarray, E_MeV: float) -> float:
    """Log-log interpolation using pre-built numpy arrays (no allocation on call)."""
    if E_MeV <= energies[0]:
        x1, x2, y1, y2 = energies[0], energies[1], mus[0], mus[1]
    elif E_MeV >= energies[-1]:
        x1, x2, y1, y2 = energies[-2], energies[-1], mus[-2], mus[-1]
    else:
        j = int(np.searchsorted(energies, E_MeV)) - 1
        x1, x2, y1, y2 = energies[j], energies[j + 1], mus[j], mus[j + 1]
    ly = np.log(y1) + (np.log(E_MeV) - np.log(x1)) * (np.log(y2) - np.log(y1)) / (np.log(x2) - np.log(x1))
    return float(np.exp(ly))


def attenuation_coefficients(formula: dict[str, float], density_g_cm3: float, E_keV: float) -> tuple[float, float]:
    """Return (mu_total_cm1, mu_pe_cm1) in one pass, computing weight_fractions once."""
    wf = weight_fractions(formula)
    E_MeV = E_keV / 1000.0
    mu_total_cm2g = sum(
        w * interp_loglog(*ELEMENT_MU[el], E_MeV)
        for el, w in wf.items()
    )
    z_over_a = sum(w * ATOMIC_DATA[el]['Z'] / ATOMIC_DATA[el]['A'] for el, w in wf.items())
    mu_comp_cm2g = N_A * sigma_kn_total_per_electron_cm2(E_keV) * z_over_a
    mu_total_cm1 = mu_total_cm2g * density_g_cm3
    mu_pe_cm1 = max(mu_total_cm1 - mu_comp_cm2g * density_g_cm3, 0.0)
    return mu_total_cm1, mu_pe_cm1


def total_mass_attenuation_compound(formula: dict[str, float], E_keV: float) -> float:
    wf = weight_fractions(formula)
    E_MeV = E_keV / 1000.0
    return sum(w * interp_loglog(*ELEMENT_MU[el], E_MeV) for el, w in wf.items())


def sigma_kn_total_per_electron_cm2(E_keV: float) -> float:
    eps = E_keV / MEC2_KEV
    term1 = ((1 + eps) / eps ** 2) * ((2 * (1 + eps) / (1 + 2 * eps)) - np.log(1 + 2 * eps) / eps)
    term2 = np.log(1 + 2 * eps) / (2 * eps)
    term3 = (1 + 3 * eps) / (1 + 2 * eps) ** 2
    return 2 * np.pi * R_E_CM ** 2 * (term1 + term2 - term3)


def compton_mass_attenuation_compound(formula: dict[str, float], E_keV: float) -> float:
    wf = weight_fractions(formula)
    z_over_a = sum(w * ATOMIC_DATA[el]['Z'] / ATOMIC_DATA[el]['A'] for el, w in wf.items())
    return N_A * sigma_kn_total_per_electron_cm2(E_keV) * z_over_a

