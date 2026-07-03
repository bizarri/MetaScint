from __future__ import annotations
from physics.atomic_data import ATOMIC_DATA
from physics.photon_kinematics import random_unit_vector


def fluorescence_yield_k(z):
    return (z**4) / (1_000_000.0 + z**4)


def post_photoelectric_secondaries(dominant_photo_element: str | None, photon_energy_keV: float, pos, rng):
    # Guard: if no element is specified or it is not in the data table, deposit
    # all energy locally rather than crashing with a KeyError.
    if dominant_photo_element is None or dominant_photo_element not in ATOMIC_DATA:
        return {
            'mode': 'no_element_fallback',
            'local_relaxation_keV': photon_energy_keV,
            'photoelectron_energy_keV': None,
            'photoelectron_direction': None,
            'secondary_photons': [],
            'secondary_electrons': [],
        }
    dom = ATOMIC_DATA[dominant_photo_element]
    out = {'mode':'fallback_local','local_relaxation_keV':0.0,'photoelectron_energy_keV':None,'photoelectron_direction':None,'secondary_photons':[],'secondary_electrons':[]}
    K = dom.get('K_edge_keV'); L3 = dom.get('L3_edge_keV'); Kalpha1 = dom.get('Kalpha1_keV')
    if K is None or photon_energy_keV < K:
        out.update({'mode':'subK_fallback','photoelectron_energy_keV':photon_energy_keV,'photoelectron_direction':random_unit_vector(rng).tolist()})
        return out
    out['photoelectron_energy_keV'] = max(photon_energy_keV - K, 0.0)
    out['photoelectron_direction'] = random_unit_vector(rng).tolist()
    omega_k = fluorescence_yield_k(dom['Z'])
    if rng.random() < omega_k:
        out.update({'mode':'fluorescence','fluorescence_yield':omega_k,'local_relaxation_keV':max(K-Kalpha1,0.0),'secondary_photons':[{'energy_keV':Kalpha1,'direction':random_unit_vector(rng).tolist(),'position_mm':list(pos)}]})
    else:
        out.update({'mode':'auger','fluorescence_yield':omega_k,'local_relaxation_keV':max(2.0*L3,0.0),'secondary_electrons':[{'energy_keV':max(K-2.0*L3,0.0),'direction':random_unit_vector(rng).tolist(),'position_mm':list(pos)}]})
    return out
