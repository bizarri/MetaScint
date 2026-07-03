# Physics Modules

## Photon transport

- **Attenuation** (`physics/attenuation.py`): energy-dependent mass attenuation coefficients from tabulated NIST data. Computes total, Compton, and photoelectric cross sections per compound via mixture rule.
- **Photon kinematics** (`physics/photon_kinematics.py`): Compton scattering via Klein-Nishina differential cross section, with rejection-sampled polar angle and uniform azimuthal rotation.
- **Recursive Compton tracking**: photons scatter in the current material until they escape the pixel or fall below the energy cutoff. Below-cutoff photons deposit their remaining energy locally.

## Photoelectric post-processing

`physics/relaxation.py` — `post_photoelectric_secondaries()` handles K-shell vacancy relaxation:
- Fluorescence photon emission (with direction) — tracked as a secondary photon in the event loop.
- Auger electron emission — deposited locally.
- Photoelectron emission — transported via CSDA straight-line model.

Whether fluorescence or Auger occurs depends on the K-shell fluorescence yield of the dominant photo-element.

## Electron transport

`physics/electron_tabata.py` — Tabata 1994 CSDA range model:
- Energy-dependent continuous-slowing-down approximation range for electrons.
- Range computed from material composition and density (compound mean method with harmonic Z/A).

### Cross-material transport

`simulation/composite_benchmark.py` — `transport_electron_csda()` segments the straight-line CSDA path at each material boundary:
1. At each step, the current material's range model determines the maximum possible path.
2. The distance to the next material boundary is found via `geometry.distance_to_next_material_change()`.
3. If the remaining range fits within the current domain, all remaining energy is deposited in the current material.
4. If the path crosses a boundary, the fraction of energy lost in the current domain is computed via the CSDA range fraction (energy lost = E_lost / R_total * path_here), and the particle continues in the next material.

### Per-material deposition

Energy is attributed to the **material the electron is passing through at that moment**, not the emission material. The output `deposition_by_material_keV` dictionary sums per-material deposited energy across all events.

## Output

Results include:
- `full_count` / `partial_count` / `none_count` — absorption tallies.
- `total_deposited_keV` — sum of all deposited energy across events.
- `deposition_by_material_keV` — dict of material role → total deposited keV.
- `deposition_by_material_fraction` — per-material fraction of total incident gamma energy (n_events × gamma_energy_keV).
- `examples` — detailed event histories with step-by-step tracking.

## Next extension points
- More detailed shell-resolved relaxation.
- Optical / photonic correction modules.
