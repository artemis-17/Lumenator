"""Flyby geometry and encounter diagnostics for trajectory planning."""

from dataclasses import dataclass

import numpy as np

PLANETS = ("Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune")
BODY_LIMITS = 1000 * np.array([2600, 6300, 7000, 3600, 75000, 65000, 28000, 27000])
GEOMETRY_HELP = "PLANET:PERIAPSIS_KM[:TOL_KM[:boost|brake|either[:CLOCK_DEG[:CLOCK_TOL_DEG]]]]"


def focused_impact_parameter(periapsis, mu, excess_speed):
    """Two-body impact parameter in the same distance units as periapsis."""
    return np.sqrt(periapsis**2 + 2 * mu * periapsis / excess_speed**2)


def incoming_asymptote(r, v, mu, *, ballistic=False):
    speed = np.linalg.norm(v)
    if speed == 0:
        return None
    h = np.cross(r, v)
    speed2 = speed**2 - 2 * mu / np.linalg.norm(r) if mu and not ballistic else speed**2
    if speed2 <= 0 or not np.linalg.norm(h):
        return None
    incoming = v / speed
    if mu and not ballistic:
        e_vec = np.cross(v, h) / mu - r / np.linalg.norm(r)
        e = np.linalg.norm(e_vec)
        if e <= 1:
            return None
        incoming = (e_vec + np.sqrt(e**2 - 1) * np.cross(h / np.linalg.norm(h), e_vec)) / e**2
    excess = np.sqrt(speed2)
    return incoming, np.cross(incoming, h) / excess, excess


@dataclass(frozen=True)
class Encounter:
    planet: str
    entry_time: float
    periapsis_time: float
    exit_time: float
    periapsis_km: float
    periapsis_position_m: tuple[float, float, float]
    energy_change_km2_s2: float
    b_plane_clock_deg: float | None
    turn_deg: float | None
    relative_entry_speed_km_s: float
    relative_v_inf_km_s: float | None
    impact_parameter_km: float | None


def measure_encounter(planet, entry, closest, exit_state, mu_planet, mu_sun):
    t_in, x_in, p_in = entry
    t_ca, distance, position = closest
    t_out, x_out, p_out = exit_state
    r = x_in[:3] - p_in[:3]
    v = x_in[3:] - p_in[3:]
    h = np.cross(r, v)
    clock = turn = None
    eccentricity = np.cross(v, h) / mu_planet - r / np.linalg.norm(r) if mu_planet else np.zeros(3)
    e = np.linalg.norm(eccentricity)
    if e > 1 and np.linalg.norm(h) > 0:
        peri = eccentricity / e
        h_unit = h / np.linalg.norm(h)
        incoming = (peri + np.sqrt(e**2 - 1) * np.cross(h_unit, peri)) / e
        b = np.cross(incoming, h_unit)
        pole = np.cross(p_in[:3], p_in[3:])
        pole_norm = np.linalg.norm(pole)
        reference = np.cross(incoming, pole / pole_norm) if pole_norm else np.zeros(3)
        if np.linalg.norm(reference) > 1e-10:
            reference /= np.linalg.norm(reference)
            clock = float(np.degrees(np.arctan2(b @ np.cross(incoming, reference), b @ reference)) % 360)
        turn = float(np.degrees(2 * np.arcsin(1 / e)))

    def heliocentric_energy(x):
        return (x[3:] @ x[3:]) / 2 - mu_sun / np.linalg.norm(x[:3])

    asymptote = incoming_asymptote(r, v, mu_planet)
    return Encounter(planet, t_in, t_ca, t_out, distance / 1000, tuple(position),
                     (heliocentric_energy(x_out) - heliocentric_energy(x_in)) / 1e6,
                     clock, turn, np.linalg.norm(v) / 1000,
                     asymptote[2] / 1000 if asymptote else None,
                     np.linalg.norm(asymptote[1]) / 1000 if asymptote else None)
@dataclass(frozen=True)
class FlybyTarget:
    periapsis_km: float
    tolerance_km: float
    direction: str
    clock_deg: float | None = None
    clock_tolerance_deg: float = 10.

    @property
    def energy_sign(self):
        return {"boost": 1, "brake": -1, "either": 0}[self.direction]

    def error(self, encounter):
        excess = abs(encounter.periapsis_km - self.periapsis_km) - self.tolerance_km
        distance_error = max(0., excess) / self.periapsis_km
        sign = self.energy_sign
        energy_error = max(0., .001 - sign * encounter.energy_change_km2_s2) / 100 if sign else 0.
        clock_error = 0.
        if self.clock_deg is not None:
            if encounter.b_plane_clock_deg is None:
                clock_error = 1.
            else:
                angle = abs((encounter.b_plane_clock_deg - self.clock_deg + 180) % 360 - 180)
                clock_error = max(0., angle - self.clock_tolerance_deg) / 180
        return distance_error, energy_error, clock_error


def parse_flyby_geometry(specs, route, ephemeris=None):
    targets = {}
    for spec in specs:
        fields = spec.split(":")
        if not 2 <= len(fields) <= 6:
            raise ValueError(f"Use {GEOMETRY_HELP}: {spec}")
        planet = fields[0]
        if planet not in route or planet in targets:
            raise ValueError(f"Geometry must name a planet in --flybys exactly once: {planet}")
        try:
            radius = float(fields[1])
            tolerance = float(fields[2]) if len(fields) >= 3 else .1 * radius
            clock = float(fields[4]) if len(fields) >= 5 else None
            clock_tolerance = float(fields[5]) if len(fields) == 6 else 10.
        except ValueError as error:
            raise ValueError(f"Invalid numerical geometry for {planet}") from error
        direction = fields[3] if len(fields) >= 4 else "either"
        names = PLANETS if ephemeris is None else ephemeris.names
        if planet not in names:
            raise ValueError(f"Unknown flyby body: {planet}")
        index = names.index(planet)
        limit = (BODY_LIMITS if ephemeris is None else ephemeris.limits)[index] / 1000
        if not np.isfinite([radius, tolerance]).all() or radius <= 0 or tolerance <= 0:
            raise ValueError(f"{planet}: periapsis and tolerance must be positive and finite")
        if radius + tolerance <= limit:
            raise ValueError(f"{planet}: periapsis band has no safe point above {limit:,.0f} km")
        if ephemeris is not None and ephemeris.fixed_gates[index]:
            gate = ephemeris.fixed_gates[index] / 1000
            if radius - tolerance >= gate:
                raise ValueError(f"{planet}: periapsis band lies outside the {gate:,.0f} km encounter gate")
        if direction not in ("boost", "brake", "either"):
            raise ValueError(f"{planet}: expected boost, brake, or either")
        if ephemeris is not None and not ephemeris.mu[index] and (direction != "either" or clock is not None):
            raise ValueError(f"{planet}: a massless waypoint supports periapsis geometry only")
        if clock is not None and (not np.isfinite(clock) or not 0 < clock_tolerance <= 180):
            raise ValueError(f"{planet}: use a finite clock angle and a tolerance in (0, 180] degrees")
        targets[planet] = FlybyTarget(radius, tolerance, direction,
                                     None if clock is None else clock % 360, clock_tolerance)
    return targets



