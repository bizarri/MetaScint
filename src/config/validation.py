from __future__ import annotations
from config.schema import BenchmarkConfig
from physics.atomic_data import ATOMIC_DATA

# Explicit sets of supported mode/strategy strings.
# Update these when new modes are implemented.
_SUPPORTED_IRRADIATION_MODES = {'centered_rectangle_equal_scaling'}
_SUPPORTED_ELECTRON_TRANSPORT = {'csda_straight_line'}
_SUPPORTED_ELECTRON_RANGE_MODELS = {'tabata1994_compound_mean_ZA'}
_SUPPORTED_PE_POSTPROCESS = {'fluorescence_or_auger_minimal'}


def validate_benchmark_config(cfg: BenchmarkConfig) -> list[str]:
    """Validate the fully resolved runtime benchmark configuration.

    Important:
    - this validator assumes references have already been resolved
    - it validates the final internal BenchmarkConfig consumed by simulation
    """
    issues: list[str] = []

    # -------------------------------------------------------------------------
    # Geometry checks
    # -------------------------------------------------------------------------
    if cfg.geometry.pixel_x_mm <= 0:
        issues.append('geometry.pixel_x_mm must be positive.')
    if cfg.geometry.pixel_y_mm <= 0:
        issues.append('geometry.pixel_y_mm must be positive.')
    if cfg.geometry.pixel_z_mm <= 0:
        issues.append('geometry.pixel_z_mm must be positive.')

    hx, hy = cfg.geometry.pixel_x_mm / 2.0, cfg.geometry.pixel_y_mm / 2.0
    z0, z1 = 0.0, cfg.geometry.pixel_z_mm
    for r in cfg.geometry.regions:
        if r.material_role not in cfg.materials:
            issues.append(f'region "{r.name}" references unknown material_role "{r.material_role}"')
        if abs(r.tilt_deg) < 1e-8:
            if r.xmin < -hx or r.xmax > hx or r.ymin < -hy or r.ymax > hy:
                issues.append(f'region "{r.name}" extends outside pixel xy bounds')
        if r.zmin < z0 or r.zmax > z1:
            issues.append(f'region "{r.name}" extends outside pixel z bounds')

    if cfg.geometry.regions and not cfg.default_material_role:
        issues.append('default_material_role must be set when geometry has regions')

    if cfg.default_material_role and cfg.default_material_role not in cfg.materials:
        issues.append(
            f'default_material_role "{cfg.default_material_role}" '
            f'is not defined in materials dict. '
            f'Available roles: {sorted(cfg.materials.keys())}'
        )

    # -------------------------------------------------------------------------
    # Irradiation checks
    # -------------------------------------------------------------------------
    if not (0.0 < cfg.irradiation.front_face_area_fraction <= 1.0):
        issues.append('irradiation.front_face_area_fraction must be in the interval (0, 1].')

    if cfg.irradiation.mode not in _SUPPORTED_IRRADIATION_MODES:
        issues.append(
            f'irradiation.mode "{cfg.irradiation.mode}" is not supported. '
            f'Supported: {sorted(_SUPPORTED_IRRADIATION_MODES)}'
        )

    if not (0.0 <= cfg.irradiation.max_angle_deg <= 25.0):
        issues.append('irradiation.max_angle_deg must be in [0, 25] deg (covers all realistic PET scanner geometries).')

    # -------------------------------------------------------------------------
    # Simulation checks
    # -------------------------------------------------------------------------
    if cfg.simulation.gamma_energy_keV <= 0:
        issues.append('simulation.gamma_energy_keV must be positive.')

    if cfg.simulation.n_events <= 0:
        issues.append('simulation.n_events must be positive.')

    if cfg.simulation.photon_energy_cutoff_keV < 0:
        issues.append('simulation.photon_energy_cutoff_keV must be non-negative.')

    if cfg.simulation.electron_transport not in _SUPPORTED_ELECTRON_TRANSPORT:
        issues.append(
            f'simulation.electron_transport "{cfg.simulation.electron_transport}" is not supported. '
            f'Supported: {sorted(_SUPPORTED_ELECTRON_TRANSPORT)}'
        )

    if cfg.simulation.electron_range_model not in _SUPPORTED_ELECTRON_RANGE_MODELS:
        issues.append(
            f'simulation.electron_range_model "{cfg.simulation.electron_range_model}" is not supported. '
            f'Supported: {sorted(_SUPPORTED_ELECTRON_RANGE_MODELS)}'
        )

    if cfg.simulation.photoelectric_postprocess not in _SUPPORTED_PE_POSTPROCESS:
        issues.append(
            f'simulation.photoelectric_postprocess "{cfg.simulation.photoelectric_postprocess}" is not supported. '
            f'Supported: {sorted(_SUPPORTED_PE_POSTPROCESS)}'
        )

    if cfg.simulation.n_example_histories < 0:
        issues.append('simulation.n_example_histories must be >= 0.')

    # -------------------------------------------------------------------------
    # Material checks
    # -------------------------------------------------------------------------
    if len(cfg.materials) == 0:
        issues.append('At least one material must be defined in the resolved configuration.')

    for role, mat in cfg.materials.items():
        if mat.density_g_cm3 <= 0:
            issues.append(f'materials[{role}].density_g_cm3 must be positive.')

        if len(mat.formula) == 0:
            issues.append(f'materials[{role}].formula cannot be empty.')

        for element, stoich in mat.formula.items():
            if stoich <= 0:
                issues.append(f'materials[{role}].formula[{element}] must be positive.')

        if mat.dominant_photo_element is None:
            issues.append(
                f'materials[{role}].dominant_photo_element is not set. '
                f'Photoelectric interactions will deposit energy locally without '
                f'fluorescence/Auger secondary generation.'
            )
        elif mat.dominant_photo_element not in ATOMIC_DATA:
            issues.append(
                f'materials[{role}].dominant_photo_element '
                f'"{mat.dominant_photo_element}" is not in the atomic data table. '
                f'Available elements: {sorted(ATOMIC_DATA.keys())}'
            )

        if mat.scintillation is not None:
            amp_sum = sum(c.amplitude_fraction for c in mat.scintillation.decay_components)
            if abs(amp_sum - 1.0) > 1e-6:
                issues.append(
                    f'materials[{role}].scintillation.decay_components: '
                    f'amplitude_fractions sum to {amp_sum:.8f}, expected 1.0'
                )

    return issues
