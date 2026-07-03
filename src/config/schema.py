from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from config.defaults import (
    DEFAULT_FRONT_FACE_AREA_FRACTION,
    DEFAULT_PHOTON_CUTOFF_KEV,
    DEFAULT_ELECTRON_TRANSPORT,
    DEFAULT_ELECTRON_RANGE_MODEL,
    DEFAULT_MAX_ANGLE_DEG,
)


# -----------------------------------------------------------------------------
# Resolved runtime configuration objects
# These are the fully expanded objects consumed by the simulation code.
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ScintillationComponent:
    amplitude_fraction: float
    decay_time_ns: float
    pulse_model: str = "standard"


@dataclass(frozen=True)
class ScintillationConfig:
    light_yield_per_keV: float
    rise_time_ns: float
    decay_components: Tuple[ScintillationComponent, ...]


@dataclass(frozen=True)
class MaterialConfig:
    density_g_cm3: float
    formula: Dict[str, float]
    dominant_photo_element: Optional[str] = None
    scintillation: Optional[ScintillationConfig] = None


@dataclass(frozen=True)
class RegionConfig:
    name: str
    material_role: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    tilt_deg: float = 0.0


@dataclass(frozen=True)
class GeometryConfig:
    pixel_x_mm: float
    pixel_y_mm: float
    pixel_z_mm: float
    regions: Tuple[RegionConfig, ...] = ()


@dataclass(frozen=True)
class IrradiationConfig:
    front_face_area_fraction: float = DEFAULT_FRONT_FACE_AREA_FRACTION
    mode: str = 'centered_rectangle_equal_scaling'
    max_angle_deg: float = DEFAULT_MAX_ANGLE_DEG


@dataclass(frozen=True)
class SimulationConfig:
    gamma_energy_keV: float
    n_events: int
    seed: int
    photon_energy_cutoff_keV: float = DEFAULT_PHOTON_CUTOFF_KEV
    electron_transport: str = DEFAULT_ELECTRON_TRANSPORT
    electron_range_model: str = DEFAULT_ELECTRON_RANGE_MODEL
    photoelectric_postprocess: str = 'fluorescence_or_auger_minimal'
    n_example_histories: int = 2
    max_photon_steps: int = 1000


@dataclass(frozen=True)
class BenchmarkConfig:
    materials: Dict[str, MaterialConfig]
    geometry: GeometryConfig
    irradiation: IrradiationConfig
    simulation: SimulationConfig
    default_material_role: str = ''


# -----------------------------------------------------------------------------
# Reference-based run configuration objects
# These are the lightweight user-facing configs that assemble canonical config
# files from configs/materials, configs/geometry, configs/photonics, etc.
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    material_refs: Dict[str, str]
    geometry_ref: str
    photonics_ref: Optional[str] = None
    optimization_ref: Optional[str] = None
    irradiation: Optional[Dict] = None
    simulation: Optional[Dict] = None
    material_overrides: Optional[Dict[str, Dict]] = None
    geometry_overrides: Optional[Dict] = None
    photonics_overrides: Optional[Dict] = None
    optimization_overrides: Optional[Dict] = None
