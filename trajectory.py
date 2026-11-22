"""Propagate a solar-sail trajectory with ordered gravity assists."""

from copy import copy
from dataclasses import asdict

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import config
from dynamics import gravity
from ephemerides import Ephemeris, earth_state, gm_si, load_kernels
from flyby import focused_impact_parameter, incoming_asymptote, measure_encounter
from guidance import SolarDiveGuide
from solar_sail import AU, SailThrust

DAY = 86400.
YEAR = 365.25 * DAY
V_REF = 30000.
ESCAPE_ENERGY_MARGIN = 1.  # m^2/s^2 above the parabolic boundary for numerical verification


def launch_state(epoch):
    state = earth_state(epoch)
    state[:3] *= 1 + .02 * AU / np.linalg.norm(state[:3])
    return state


def make_initial_state(epoch=config.DEFAULT_EPOCH, bodies=()):
    """Build the heliocentric post-Earth-departure state and spacecraft parameters."""
    import spiceypy as spice

    load_kernels()
    et0 = spice.str2et(epoch)
    x0 = launch_state(et0)

    ephemeris = Ephemeris(bodies)
    params = {
        "et0": et0,
        "mu_sun": gm_si(10),
        "mu_planets": ephemeris.mu,
        "ephemeris": ephemeris,
        "sail_params": config.SAIL,
    }
    return x0, params


class SolarSailTrajectory:
    """Hold a normalized RTN force command over each guidance interval."""

    def __init__(self, x0, params, dt_days=config.GUIDANCE_DAYS, horizon_years=config.MAX_YEARS,
                 min_au=config.MIN_SOLAR_AU, accurate=False, record=False, flybys=(), targets=None,
                 launch_window_years=0., guide=None, launch_anchor=None, leg_controls=None):
        self.x0 = np.asarray(x0, dtype=float)
        self.params = params
        self.ephemeris = params["ephemeris"]
        self.flybys = tuple(flybys)
        unknown = set(self.flybys) - set(self.ephemeris.names)
        if unknown:
            raise ValueError(f"Unknown flyby bodies: {', '.join(sorted(unknown))}")
        self.route = tuple(self.ephemeris.names.index(name) for name in self.flybys)
        self.targets = targets or {}
        self.leg_controls = {}
        for body, angles in (leg_controls or {}).items():
            if (body not in self.flybys[1:] or len(angles) != 2 or not np.isfinite(angles).all()
                    or not 0 <= angles[0] <= np.pi / 2):
                raise ValueError("Leg controls must specify feasible cone and clock angles for a later flyby")
            self.leg_controls[body] = tuple(angles)
        self.dt, self.T = dt_days * DAY, horizon_years * YEAR
        self.r_limit = min_au * AU
        self.launch_window = launch_window_years * YEAR
        self.launch_anchor = launch_anchor
        if launch_anchor is not None and not 0 <= launch_anchor <= 1:
            raise ValueError("Reference departure fraction must be in [0, 1]")
        if not np.isfinite([self.dt, self.T, self.r_limit, self.launch_window]).all():
            raise ValueError("Mission parameters must be finite")
        if min(self.dt, self.T, self.r_limit) <= 0 or self.launch_window < 0:
            raise ValueError("Invalid propagation duration, launch window, or thermal limit")
        if len(set(self.route)) != len(self.route) or set(self.targets) - set(self.flybys):
            raise ValueError("Geometry targets require distinct planets in the flyby route")
        self.record = record
        self.guide = SolarDiveGuide(**guide) if guide is not None else None
        self.mu_sun = params["mu_sun"]
        self.mu_planets = params["mu_planets"]
        self.massive = self.mu_planets > 0
        self.sail = SailThrust(*params["sail_params"])
        self.rtol = 1e-10 if accurate else 2e-7
        self.atol = ([.1] * 3 + [1e-6] * 3) if accurate else ([20.] * 3 + [2e-4] * 3)
        self.max_step = 12 * DAY

    def _orbit(self, state=None):
        state = self.x if state is None else state
        r, v = state[:3], state[3:]
        radius = np.linalg.norm(r)
        vr = r @ v / radius
        h = np.cross(r, v)
        eccentricity = np.linalg.norm(np.cross(v, h) / self.mu_sun - r / radius)
        q = (h @ h) / (self.mu_sun * (1 + eccentricity))
        return radius, vr, np.linalg.norm(h) / radius, (v @ v) / 2 - self.mu_sun / radius, q, eccentricity

    def _escape_margin(self, state):
        r, v = state[:3], state[3:]
        radius = np.linalg.norm(r)
        energy = (v @ v) / 2 - self.mu_sun / radius
        return min((energy - ESCAPE_ENERGY_MARGIN) / V_REF**2, (r @ v) / (radius * V_REF))

    @property
    def next_body(self):
        return self.route[self.flyby_count] if self.flyby_count < len(self.route) else None

    def _body(self, t, index):
        return self.ephemeris.state(index, self.launch_et + t)

    def _gate(self, index, state):
        return self.ephemeris.gate(index, np.linalg.norm(state[:3]), self.mu_sun)

    def _initialize_guide(self):
        self.guide_phase = 0
        self.reference_request = np.zeros(3)
        if self.guide is not None:
            radius, vr, _, _, perihelion, _ = self._orbit()
            if perihelion <= self.guide.dive_perihelion_au * AU:
                self.guide_phase = self.guide.outbound_phase(radius) if vr >= 0 else 1
            self.reference_request = self.guide.command(self.guide_phase, self.sail)

    def _leg_control(self):
        return self.leg_controls.get(self.flybys[self.flyby_count]) if self.next_body is not None else None

    def _reference_command(self):
        angles = self._leg_control()
        if angles is not None:
            return self.sail.continuous_command_at(*angles)
        return self.guide.command(self.guide_phase, self.sail) if self.guide else np.zeros(3)

    def fork_at_exit(self):
        """Copy a numerically propagated exit so leg trials share an identical prefix."""
        if not self.flyby_passes or abs(self.t - self.flyby_passes[-1].exit_time) > 1e-6:
            raise ValueError("Leg trials must start at a completed encounter's exact exit")
        branch = copy(self)
        branch.x = self.x.copy()
        for name in ("leg_controls", "flyby_passes", "approaches", "approach_states",
                     "gate_entries", "gate_exits", "exit_states"):
            setattr(branch, name, getattr(self, name).copy())
        branch.record = False
        branch.history_u, branch.trace_t, branch.trace_x = [], [], []
        return branch

    def _add_guide_events(self, add):
        guide = self.guide

        def burn_radius(t, x):
            return np.linalg.norm(x[:3]) - guide.burn_radius_au * AU

        def apsis(t, x):
            return x[:3] @ x[3:]

        if self.guide_phase == 0:
            add("guide_dive", lambda t, x: self._orbit(x)[4] - guide.dive_perihelion_au * AU, -1, True)
        elif self.guide_phase == 1:
            add("guide_perihelion", apsis, 1, True)
        elif self.guide_phase == 2:
            add("guide_outbound", burn_radius, 1, True)
        elif self.guide_phase == 3:
            add("guide_inbound", apsis, -1, True)
        else:
            add("guide_burn", burn_radius, -1, True)
            add("guide_perihelion", apsis, 1, True)

    def _update_guide(self, crossings):
        for name, (times, _) in crossings.items():
            if not name.startswith("guide_") or not len(times):
                continue
            radius = np.linalg.norm(self.x[:3])
            if name == "guide_dive":
                self.guide_phase = self.guide.outbound_phase(radius) if self.x[:3] @ self.x[3:] >= 0 else 1
            elif name == "guide_perihelion":
                self.guide_phase = self.guide.outbound_phase(radius)
            else:
                self.guide_phase = {"guide_burn": 2, "guide_outbound": 3, "guide_inbound": 4}[name]

    def reset(self):
        self.launch_et = self.params["et0"]
        self.pending_launch = bool(self.launch_window)
        self.t = 0.
        self.x = self.x0.copy()
        self.r_min = np.linalg.norm(self.x[:3])
        self.flyby_count = 0
        self.flyby_passes = []
        self.approaches = {}
        self.approach_states = {}
        self.gate_entries = {}
        self.gate_exits = set()
        self.exit_states = {}
        self.entry = self.periapsis = None
        self.escape_reached = False
        self.planet_violation = self.thermal_violation = False
        self.history_u = []
        self.trace_t = [0.] if self.record else []
        self.trace_x = [self.x.copy()] if self.record else []
        self._initialize_guide()

    def _geometry_ok(self):
        return all(not any(self.targets[p.planet].error(p))
                   for p in self.flyby_passes if p.planet in self.targets)

    def result(self, diagnostics=False):
        radius, _, _, energy, _, _ = self._orbit()
        geometry_ok = self._geometry_ok() and all(
            any(p.planet == name for p in self.flyby_passes) for name in self.targets)
        safe = not (self.planet_violation or self.thermal_violation)
        success = self.escape_reached and self.flyby_count == len(self.route) and geometry_ok and safe
        # The work available from holding the sail face-on on the outbound arc.
        radial_sail_work = self.sail.scale * AU**2 / (self.sail.mass * radius)
        failure = ("thermal_violation" if self.thermal_violation else "collision" if self.planet_violation
                   else None if success else "missed_planet" if self.next_body is not None
                   else "invalid_geometry" if not geometry_ok else "escape_not_reached")
        info = dict(mission_success=bool(success),
                    escape_reached=bool(self.escape_reached), day=self.t / DAY, flight_years=self.t / YEAR,
                    r_au=radius / AU, perihelion_au=self.r_min / AU, energy_m2s2=energy,
                    v_inf_km_s=np.sqrt(max(0., 2 * energy)) / 1000,
                    projected_sail_v_inf_km_s=np.sqrt(max(0., 2 * (energy + radial_sail_work))) / 1000
                    if self.escape_reached else None,
                    escape_et=self.launch_et + self.t if success else None,
                    final_flyby_exit_et=self.launch_et + self.flyby_passes[-1].exit_time
                    if self.flyby_passes and self.next_body is None else None,
                    flyby_count=self.flyby_count, valid_flyby_count=sum(
                        p.planet not in self.targets or not any(self.targets[p.planet].error(p))
                        for p in self.flyby_passes),
                    encounters=[asdict(p) for p in self.flyby_passes], launch_et=self.launch_et,
                    thermal_violation=self.thermal_violation, planet_violation=self.planet_violation,
                    geometry_ok=geometry_ok, failure_reason=failure)
        if diagnostics:
            info["flyby_diagnostics"] = self.flyby_diagnostics()
        return info

    def flyby_diagnostics(self):
        completed = {encounter.planet: encounter for encounter in self.flyby_passes}
        output = []
        for index in self.route:
            name = self.ephemeris.names[index]
            target = self.targets.get(name)
            flyby = completed.get(name)
            entry = self.gate_entries.get(index)
            closest = self.approach_states.get(index)
            sample = entry or closest
            relative_speed = impact = expected = excess = None
            if sample is not None:
                _, craft, body = sample
                r, v = (craft - body).reshape(2, 3) / 1000
                relative_speed = float(np.linalg.norm(v))
                mu = self.mu_planets[index] / 1e9
                asymptote = incoming_asymptote(r, v, mu, ballistic=entry is None)
                if asymptote:
                    _, b, excess = asymptote
                    impact = float(np.linalg.norm(b))
                    if target:
                        expected = float(focused_impact_parameter(target.periapsis_km, mu, excess))
            clock_error = None
            if flyby and target and flyby.b_plane_clock_deg is not None and target.clock_deg is not None:
                clock_error = abs((flyby.b_plane_clock_deg - target.clock_deg + 180) % 360 - 180)
            output.append(dict(
                body=name, closest_approach_km=self.approaches[index][1] / 1000 if closest is not None else None,
                actual_periapsis_km=flyby.periapsis_km if flyby else None,
                requested_periapsis_km=target.periapsis_km if target else None,
                tolerance_km=target.tolerance_km if target else None,
                absolute_periapsis_error_km=abs(flyby.periapsis_km - target.periapsis_km) if flyby and target else None,
                requested_clock_deg=target.clock_deg if target else None,
                absolute_clock_error_deg=clock_error,
                requested_energy_direction=target.direction if target else None,
                periapsis_speed_km_s=float(np.linalg.norm(closest[1][3:] - closest[2][3:]) / 1000)
                if flyby and closest is not None else None,
                relative_speed_km_s=relative_speed, excess_speed_km_s=excess,
                estimated_impact_parameter_km=impact, required_impact_parameter_km=expected,
                focusing_ratio=expected / target.periapsis_km if expected is not None else None,
                impact_estimate_basis="gate_entry_two_body" if entry else "ballistic_at_closest" if closest else None,
                gate_entered=entry is not None, gate_exited=index in self.gate_exits,
                exit_state_j2000_si=(self.exit_states[index].tolist() if index in self.exit_states else None),
                flyby_completed=flyby is not None,
                geometry_satisfied=bool(flyby and (target is None or not any(target.error(flyby)))),
                failure_reason=(None if flyby and (target is None or not any(target.error(flyby)))
                                else "invalid_geometry" if flyby else "invalid_or_incomplete_pass" if entry
                                else "missed_planet" if closest else "not_reached")))
        return output

    def _propagate(self, force, end, *, open_loop=False):
        index = self.next_body
        events = {}
        max_step = self.max_step
        limits = self.ephemeris.limits
        recover_crossings = False
        gm = self.mu_planets[self.massive]

        def add(name, function, direction, terminal=False):
            function.direction, function.terminal = direction, terminal
            events[name] = function

        def rhs(t, x):
            r, v = x[:3], x[3:]
            planets = self.ephemeris.positions(self.launch_et + t)[self.massive]
            acceleration = gravity(r, planets, self.mu_sun, gm)
            return np.r_[v, acceleration + force(x) / self.sail.mass]

        add("thermal", lambda t, x: np.linalg.norm(x[:3]) - self.r_limit, -1, True)
        add("solar_peri", lambda t, x: x[:3] @ x[3:], 1)
        if self.guide is not None and not open_loop:
            self._add_guide_events(add)
        if index is None and self._escape_margin(self.x) <= 0.:
            add("escape", lambda t, x: self._escape_margin(x), 1, True)
        for body, pos in enumerate(self.ephemeris.positions(self.launch_et + self.t)):
            distance = np.linalg.norm(self.x[:3] - pos)
            # Resolve safety crossings when a body could be reached this interval.
            if distance < 200000 * (end - self.t) + limits[body]:
                add(f"body_{body}", lambda t, x, b=body:
                    np.linalg.norm(x[:3] - self._body(t, b)[:3]) - limits[b], -1, True)
                def body_periapsis(t, x, b=body):
                    rel = x - self._body(t, b)
                    return rel[:3] @ rel[3:]

                add(f"peri_{body}", body_periapsis, 1)
                if distance < 12 * limits[body] and not self.ephemeris.fixed_gates[body]:
                    max_step = min(max_step, limits[body] / 200000)
        if index is not None:
            state = self._body(self.t, index)
            gate = self._gate(index, state)
            if np.linalg.norm(self.x[:3] - state[:3]) < gate + 200000 * (end - self.t):
                def gate_distance(t, x):
                    p = self._body(t, index)
                    rel = x - p
                    distance = np.linalg.norm(rel[:3]) - self._gate(index, p)
                    if t == self.t and self.entry is None and rel[:3] @ rel[3:] > 0:
                        return max(distance, 1e-3) if abs(distance) < 1e-3 else distance
                    return distance

                def approach(t, x):
                    rel = x - self._body(t, index)
                    return rel[:3] @ rel[3:]

                add("entry", gate_distance, -1)
                add("exit", lambda t, x: gate_distance(t, x), 1, True)
                add("closest", approach, 1)
                recover_crossings = bool(self.ephemeris.fixed_gates[index])
                if not recover_crossings:
                    max_step = min(max_step, gate / 400000)
        sol = solve_ivp(rhs, (self.t, end), self.x, method="DOP853", events=list(events.values()),
                        rtol=self.rtol, atol=self.atol, max_step=max_step,
                        dense_output=recover_crossings,
                        first_step=min(end - self.t, max_step))
        if not sol.success:
            raise RuntimeError(sol.message)
        crossings = dict(zip(events, zip(sol.t_events, sol.y_events)))
        if recover_crossings and len(crossings["closest"][0]):
            # A small waypoint gate can be crossed twice in one solver step.
            # Bracket that transit around the localized closest approach.
            for closest_time, closest_state in zip(*crossings["closest"]):
                if gate_distance(closest_time, closest_state) >= 0:
                    continue
                for name, endpoint in (("entry", sol.t[0]), ("exit", sol.t[-1])):
                    if len(crossings[name][0]) or gate_distance(endpoint, sol.sol(endpoint)) <= 0:
                        continue
                    bounds = sorted((endpoint, closest_time))
                    time = brentq(lambda value: gate_distance(value, sol.sol(value)), *bounds)
                    crossings[name] = ([time], [sol.sol(time)])
                break
        # A dense-output recovery can locate the exit inside a solver step.
        # Stop there too, so the next commanded leg always starts at the exit.
        end_time, end_state = sol.t[-1], sol.y[:, -1]
        samples = sol.y
        trace_t, trace_x = sol.t, sol.y.T
        if recover_crossings and len(crossings["exit"][0]) and crossings["exit"][0][0] < end_time:
            end_time, end_state = crossings["exit"][0][0], crossings["exit"][1][0]
            mask = sol.t < end_time
            trace_t = np.r_[sol.t[mask], end_time]
            trace_x = np.vstack([sol.y.T[mask], end_state])
            samples = trace_x.T
            crossings = {name: ([t for t in times if t <= end_time],
                                [x for t, x in zip(times, states) if t <= end_time])
                         for name, (times, states) in crossings.items()}
        if self.record:
            self.trace_t.extend(trace_t[1:])
            self.trace_x.extend(trace_x[1:])
        self.t, self.x = float(end_time), end_state.copy()
        self.r_min = min(self.r_min, np.min(np.linalg.norm(samples[:3], axis=0)),
                         *(np.linalg.norm(x[:3]) for x in crossings["solar_peri"][1]))
        self.planet_violation = any(len(times) for name, (times, _) in crossings.items() if name.startswith("body_"))
        self.thermal_violation = bool(len(crossings["thermal"][0]))
        for name, (times, states) in crossings.items():
            if name.startswith("peri_"):
                body = int(name.split("_")[1])
                self.planet_violation |= any(
                    np.linalg.norm(x[:3] - self._body(t, body)[:3]) <= limits[body]
                    for t, x in zip(times, states))
        return crossings

    def _encounter(self, index, crossings):
        if index is None:
            return
        planet = self.ephemeris.names[index]
        for t, x in zip(*crossings.get("entry", ([], []))):
            p = self._body(t, index)
            self.entry = (t, x, p)
            self.gate_entries[index] = self.entry
            self.periapsis = (t, np.linalg.norm(x[:3] - p[:3]), x[:3])
        samples = [*zip(*crossings.get("closest", ([], []))), (self.t, self.x)]
        for t, x in samples:
            body_state = self._body(t, index)
            distance = np.linalg.norm(x[:3] - body_state[:3])
            if index not in self.approaches or distance < self.approaches[index][1]:
                self.approaches[index] = (t, distance)
                self.approach_states[index] = (t, x, body_state)
            if self.entry is not None and t >= self.entry[0] and distance < self.periapsis[1]:
                self.periapsis = (t, distance, x[:3])
        if self.entry is None or not len(crossings.get("exit", ([], []))[0]):
            return
        t_out, x_out = crossings["exit"][0][0], crossings["exit"][1][0]
        self.gate_exits.add(index)
        encounter = measure_encounter(planet, self.entry, self.periapsis,
                                      (t_out, x_out, self._body(t_out, index)),
                                      self.mu_planets[index], self.mu_sun)
        self.entry = self.periapsis = None
        if (encounter.turn_deg is None and self.mu_planets[index] > 0) or self.planet_violation:
            return
        self.flyby_passes.append(encounter)
        self.exit_states[index] = x_out.copy()
        self.flyby_count += 1

    def step_reference(self):
        """Advance one guidance interval, selecting departure on the first step."""
        if self.pending_launch:
            # Preserve the rounding used to calibrate saved departure fractions.
            centered = 0. if self.launch_anchor is None else 2 * self.launch_anchor - 1
            self.select_launch((centered + 1) / 2)
            return False, False
        return self.advance(self.reference_request)

    def select_launch(self, fraction):
        self.launch_et += fraction * self.launch_window
        self.x = launch_state(self.launch_et)
        self.r_min = np.linalg.norm(self.x[:3])
        self.pending_launch = False
        self._initialize_guide()
        if self.record:
            self.trace_x[0] = self.x.copy()

    def command_interval(self):
        radius = np.linalg.norm(self.x[:3]) / AU
        index = self.next_body
        if index is not None and np.linalg.norm(self._body(self.t, index)[:3] - self.x[:3]) < .35 * AU:
            return min(self.dt, .5 * DAY)
        if radius < .45:
            return min(self.dt, .25 * DAY)
        if radius < 1.5:
            return min(self.dt, DAY)
        if radius > 12 and self.x[:3] @ self.x[3:] > 0:
            return max(self.dt, 12 * DAY)
        return self.dt

    def advance(self, request, end=None, *, open_loop=False):
        start = self.t
        end = min(self.t + self.command_interval(), self.T) if end is None else min(end, self.T)
        index = self.next_body
        command = self.sail.project(request)

        def force(x):
            return self.sail.force(x, command)

        initial_force = force(self.x) if self.record else None
        crossings = self._propagate(force, end, open_loop=open_loop)
        self._encounter(index, crossings)
        self.escape_reached = bool(self._escape_margin(self.x) > 0. or len(crossings.get("escape", ((),))[0]))
        complete = self.next_body is None and self.escape_reached
        unsafe = self.thermal_violation or self.planet_violation
        terminated = bool(unsafe or complete)
        truncated = bool(self.t >= self.T - 1e-6 and not terminated)
        if self.record:
            self.history_u.append((start, self.t, request.copy(), initial_force, force(self.x), command))
        if not open_loop:
            if self.guide is not None:
                self._update_guide(crossings)
            self.reference_request = self._reference_command()
        return terminated, truncated
