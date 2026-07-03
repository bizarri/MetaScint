from __future__ import annotations
import numpy as np
from physics.attenuation import MEC2_KEV


def random_unit_vector(rng):
    mu = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2*np.pi)
    s = np.sqrt(max(0.0, 1.0 - mu*mu))
    return np.array([s*np.cos(phi), s*np.sin(phi), mu], dtype=float)


def random_unit_perp(v):
    a = np.array([0.0,0.0,1.0]) if abs(v[2]) < 0.9 else np.array([1.0,0.0,0.0])
    u = np.cross(v, a)
    n = np.linalg.norm(u)
    if n < 1e-12:
        a = np.array([0.0,1.0,0.0])
        u = np.cross(v, a)
        n = np.linalg.norm(u)
    return u / n


def rotate_direction(v, cos_theta, phi):
    # v is already a unit vector from the caller; no re-normalization needed.
    # The rotation formula produces a unit vector when v, u, w are orthonormal
    # and cos_theta^2 + sin_theta^2 = 1, so no output normalization needed either.
    u = random_unit_perp(v)
    w = np.cross(v, u)
    s = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    return cos_theta * v + s * (np.cos(phi) * u + np.sin(phi) * w)


def klein_nishina_weight(E_keV, cos_theta):
    alpha = E_keV / MEC2_KEV
    ratio = 1.0/(1.0 + alpha*(1.0-cos_theta))
    return ratio*ratio*(ratio + 1.0/ratio - (1.0-cos_theta*cos_theta))


def sample_klein_nishina_cosine(rng, E_keV):
    while True:
        mu = rng.uniform(-1.0, 1.0)
        if rng.uniform(0.0, 2.1) < klein_nishina_weight(E_keV, mu):
            return mu
