"""Heliocentric point-mass gravity in any consistent system of units."""

import numpy as np


def gravity(r, bodies, mu_sun, mu_bodies):
    """Solar gravity plus the direct and indirect third-body terms."""
    relative = bodies - r
    return -mu_sun * r / np.linalg.norm(r)**3 + mu_bodies @ (
        relative / np.linalg.norm(relative, axis=1)[:, None]**3
        - bodies / np.linalg.norm(bodies, axis=1)[:, None]**3)


def outbound_radius_crossing(r, v, mu, radius):
    """Time and position of the next outward Kepler crossing of a larger radius."""
    start_radius = np.linalg.norm(r)
    momentum = np.cross(r, v)
    h = np.linalg.norm(momentum)
    radial_speed = r @ v / start_radius
    if radius <= start_radius or radial_speed <= 0 or h == 0:
        return None
    p = h**2 / mu
    e_vec = np.cross(v, momentum) / mu - r / start_radius
    e = np.linalg.norm(e_vec)
    if e < 1e-10 or abs(e - 1) < 1e-8:
        return None
    cosine = (p / radius - 1) / e
    if not -1 <= cosine <= 1:
        return None
    start = np.arctan2(radial_speed * h / mu, p / start_radius - 1)
    end = np.arccos(cosine)
    if e < 1:
        def mean(nu):
            anomaly = np.arctan2(np.sqrt(1 - e**2) * np.sin(nu), e + np.cos(nu))
            return anomaly - e * np.sin(anomaly)
    else:
        def mean(nu):
            anomaly = np.arcsinh(np.sqrt(e**2 - 1) * np.sin(nu) / (1 + e * np.cos(nu)))
            return e * np.sinh(anomaly) - anomaly
    time = (mean(end) - mean(start)) * np.sqrt(abs(p / (1 - e**2))**3 / mu)
    periapsis = e_vec / e
    transverse = np.cross(momentum / h, periapsis)
    return time, radius * (np.cos(end) * periapsis + np.sin(end) * transverse)
