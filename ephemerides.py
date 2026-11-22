import json
from pathlib import Path
import numpy as np
import spiceypy as spice
from scipy.interpolate import CubicHermiteSpline

from flyby import BODY_LIMITS, PLANETS

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
PLANET_IDS = tuple(range(1, 9))  # Mercury, Venus, Earth-Moon barycenter, Mars, ..., Neptune


def load_kernels():
    for name in ("naif0012.tls", "de440s.bsp", "gm_de440.tpc"):
        path = KERNEL_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"Download {name} into {KERNEL_DIR}")
        spice.furnsh(str(path))


def gm_si(body_id):
    _, values = spice.bodvcd(body_id, "GM", 1)
    return float(values[0]) * 1e9  # km^3/s^2 -> m^3/s^2


def planet_positions(et):
    # Geometric heliocentric positions, J2000; m
    return np.array([
        spice.spkgps(body_id, et, "J2000", 10)[0]
        for body_id in PLANET_IDS
    ]) * 1e3


def earth_state(et):
    state, _ = spice.spkezr("EARTH", et, "J2000", "NONE", "SUN")
    return np.asarray(state) * 1e3  # m, m/s


def planet_state(index, et):
    """Heliocentric geometric state of the propagated planetary system, in SI."""
    state, _ = spice.spkgeo(PLANET_IDS[index], et, "J2000", 10)
    return np.asarray(state) * 1000


def check_coverage(start_et, end_et):
    """Raise a SPICE error if the mission endpoints exceed ephemeris coverage."""
    earth_state(start_et)
    earth_state(end_et)
    planet_positions(start_et)
    planet_positions(end_et)


class Ephemeris:
    """Named SPICE planets and optional heliocentric state tables."""

    def __init__(self, bodies=()):
        self.names = PLANETS + tuple(body["name"] for body in bodies)
        if any(not isinstance(name, str) or not name for name in self.names) or len(set(self.names)) != len(self.names):
            raise ValueError("Ephemeris body names must be nonempty and unique")
        self.mu = np.r_[[gm_si(i) for i in PLANET_IDS], [1e9 * float(body.get("gm_km3_s2", 0.)) for body in bodies]]
        limits = [1000 * float(body["safety_radius_km"]) for body in bodies]
        self.limits = np.r_[BODY_LIMITS, limits]
        self.fixed_gates = np.r_[np.zeros(len(PLANETS)),
                                 [1000 * float(body["encounter_radius_km"]) for body in bodies]]
        if (not np.isfinite(np.r_[self.mu, self.limits, self.fixed_gates]).all()
                or np.any(self.mu < 0) or np.any(self.limits <= 0)
                or any(gate <= limit for gate, limit in zip(self.fixed_gates[len(PLANETS):], limits))):
            raise ValueError("Use finite nonnegative GM, a positive safety radius, and a larger encounter radius")
        self.tables = []
        for body in bodies:
            data = np.asarray(body["states"], dtype=float)
            if (data.ndim != 2 or data.shape[1] != 7 or len(data) < 2 or not np.isfinite(data).all()
                    or np.any(np.diff(data[:, 0]) <= 0) or np.any(np.linalg.norm(data[:, 1:4], axis=1) == 0)):
                raise ValueError(f"{body['name']} requires sorted, unique [ET, x, y, z, vx, vy, vz] rows")
            self.tables.append(CubicHermiteSpline(data[:, 0], data[:, 1:4] * 1000,
                                                 data[:, 4:] * 1000, extrapolate=False))

    def positions(self, et):
        planets = planet_positions(et)
        return np.vstack([planets, *(table(et) for table in self.tables)]) if self.tables else planets

    def state(self, index, et):
        if index < len(PLANETS):
            return planet_state(index, et)
        table = self.tables[index - len(PLANETS)]
        return np.r_[table(et), table(et, 1)]

    def gate(self, index, radius, mu_sun):
        fixed = self.fixed_gates[index]
        return fixed if fixed else .2 * radius * (self.mu[index] / (3 * mu_sun))**(1 / 3)

    def check_coverage(self, start_et, end_et):
        check_coverage(start_et, end_et)
        for name, table in zip(self.names[len(PLANETS):], self.tables):
            if start_et < table.x[0] or end_et > table.x[-1]:
                raise ValueError(f"{name} ephemeris must cover the entire launch window and flight deadline")


def read_body_tables(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("bodies"), list):
        raise ValueError("Expected an ephemeris JSON object containing a bodies list")
    return data["bodies"]
