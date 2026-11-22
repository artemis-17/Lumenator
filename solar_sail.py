from math import cos, sin

import numpy as np

AU = 149597870700.0  # m
P_SRP = 4.56e-6       # N/m^2 at 1 AU


def solar_sail_accel(r, v, alpha, delta, A, m, rho):
    """Solar-sail acceleration in m/s^2, in heliocentric J2000 axes."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    er = r / np.linalg.norm(r)
    h = np.cross(r, v)
    eh = h / np.linalg.norm(h)
    et = np.cross(eh, er)

    n = (np.cos(alpha) * er
         + np.sin(alpha) * np.cos(delta) * et
         + np.sin(alpha) * np.sin(delta) * eh)

    return (P_SRP * A / m) * (AU / np.linalg.norm(r))**2 * (
        (1 - rho) * np.cos(alpha) * er
        + 2 * rho * np.cos(alpha)**2 * n
    )


def rtn_basis(r, v):
    """Rows are outward radial, prograde transverse, and orbital normal."""
    radial = r / np.linalg.norm(r)
    normal = np.cross(radial, v)
    if np.linalg.norm(normal) <= 1e-12:
        normal = np.cross(radial, np.eye(3)[np.argmin(np.abs(radial))])
    normal /= np.linalg.norm(normal)
    return np.array([radial, np.cross(normal, radial), normal])


class SailThrust:
    """Map a normalized RTN request onto the force surface of a flat sail."""

    def __init__(self, area, mass, reflectivity):
        if not np.isfinite([area, mass, reflectivity]).all() or area <= 0 or mass <= 0 or not 0 <= reflectivity <= 1:
            raise ValueError("Sail area and mass must be positive and reflectivity must be between zero and one")
        self.mass = mass
        self.scale = P_SRP * area * (1 + reflectivity)
        self.absorbed = (1 - reflectivity) / (1 + reflectivity)
        self.reflected = 2 * reflectivity / (1 + reflectivity)
        self.cones = np.linspace(0, np.pi / 2, 513)
        cosine = np.cos(self.cones)
        radial = self.absorbed * cosine + self.reflected * cosine**3
        lateral = self.reflected * cosine**2 * np.sin(self.cones)
        self.surface = np.column_stack((radial, lateral))
        self.surface[-1] = 0

    def project(self, request):
        request = np.asarray(request, dtype=float)
        if request.shape != (3,) or not np.all(np.isfinite(request)):
            raise ValueError("Expected three finite RTN thrust components")
        request = np.clip(request, [-1, -1, -1], 1)
        lateral = np.linalg.norm(request[1:])
        index = np.argmin(np.sum((self.surface - [request[0], lateral])**2, axis=1))
        radial, magnitude = self._refine_projection(index, request[0], lateral)
        direction = request[1:] / lateral if lateral else np.array([1., 0.])
        return np.r_[radial, magnitude * direction]

    def _refine_projection(self, index, target_radial, target_lateral):
        # Minimize squared force error near the best sampled cone. Compute the
        # final force analytically so the result stays on the flat-sail surface.
        lower = self.cones[max(0, index - 1)]
        upper = self.cones[min(len(self.cones) - 1, index + 1)]
        cone = self.cones[index]
        a, b = self.absorbed, self.reflected
        for _ in range(3):
            c, s = cos(cone), sin(cone)
            r, t = a * c + b * c**3, b * c**2 * s
            dr, dt = -s * (a + 3 * b * c**2), b * c * (c**2 - 2 * s**2)
            ddr = -c * (a + 3 * b * c**2) + 6 * b * c * s**2
            ddt = b * s * (2 * s**2 - 7 * c**2)
            curvature = dr**2 + dt**2 + (r - target_radial) * ddr + (t - target_lateral) * ddt
            if curvature <= 0:
                break
            gradient = (r - target_radial) * dr + (t - target_lateral) * dt
            cone = max(lower, min(upper, cone - gradient / curvature))
        if cone == self.cones[-1]:
            return 0., 0.
        c = cos(cone)
        return a * c + b * c**3, b * c**2 * sin(cone)

    def command_at(self, cone, clock):
        index = round(np.clip(cone / (np.pi / 2), 0., 1.) * (len(self.surface) - 1))
        radial, lateral = self.surface[index]
        return np.array([radial, lateral * np.cos(clock), lateral * np.sin(clock)])

    def continuous_command_at(self, cone, clock):
        """Evaluate the same sail-force surface between sampled cone angles."""
        if cone == np.pi / 2:
            return np.zeros(3)
        c = cos(cone)
        radial = self.absorbed * c + self.reflected * c**3
        lateral = self.reflected * c**2 * sin(cone)
        return np.array([radial, lateral * cos(clock), lateral * sin(clock)])

    def force(self, x, command):
        r, v = x[:3], x[3:]
        return self.scale * (AU / np.linalg.norm(r))**2 * (command @ rtn_basis(r, v))
