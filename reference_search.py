"""Target-aware search over a small family of solar photonic assists."""

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from math import sqrt

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, least_squares
from spiceypy.utils.exceptions import SpiceyError

from dynamics import gravity, outbound_radius_crossing
from guidance import SolarDiveGuide
from flyby import FlybyTarget, focused_impact_parameter, incoming_asymptote
from replay import mission_rank
from solar_sail import SailThrust, rtn_basis
from trajectory import AU, DAY, YEAR, launch_state


def scout_turn(env, first, second, epoch, planet, incoming, speed, deadline=None):
    """Rank reachable outward turns within the requested first-flyby geometry."""
    mu = env.mu_planets[first]
    if not mu or env.ephemeris.fixed_gates[second]:
        return None
    name = env.ephemeris.names[first]
    limits = env.targets.get(name)
    low = 1.1 * env.ephemeris.limits[first] / 1000
    high = env._gate(first, planet) / 2000
    if limits:
        safety = env.ephemeris.limits[first] / 1000
        low = max(safety * (1 + 1e-6), limits.periapsis_km - .95 * limits.tolerance_km)
        high = limits.periapsis_km + limits.tolerance_km
    if low >= high:
        return None
    if limits:
        # A numerically fitted periapsis at the exact upper edge can cross it
        # during full-gravity replay. Aim just inside the accepted band.
        high -= min(1., .05 * (high - low))
    axis = np.cross(incoming, np.cross(planet[:3], planet[3:]))
    if not np.linalg.norm(axis):
        return None
    axis /= np.linalg.norm(axis)
    normal = np.cross(incoming, axis)
    mu_sun_sail = env.mu_sun - env.sail.scale * AU**2 / env.sail.mass
    if deadline is None:
        deadline = env.params["et0"] + env.launch_window + env.T
    follow = env.ephemeris.state(second, epoch)
    if np.linalg.norm(follow[:3]) <= np.linalg.norm(planet[:3]):
        return None
    constrained_clock = limits is not None and limits.clock_deg is not None
    center = limits.clock_deg if constrained_clock else 180.
    clock_range = .95 * limits.clock_tolerance_deg if constrained_clock else 180.
    bounds = ([np.log(low), np.radians(center - clock_range)],
              [np.log(high), np.radians(center + clock_range)])

    def residual(point):
        rp, clock = np.exp(point[0]), point[1]
        turning_angle = 2 * np.arcsin(1 / (1 + rp * 1000 * speed**2 / mu))
        direction = np.cos(clock) * axis + np.sin(clock) * normal
        outgoing = speed * (np.cos(turning_angle) * incoming - np.sin(turning_angle) * direction)
        if limits and limits.energy_sign * ((outgoing - speed * incoming) @ planet[3:]) < 0:
            return np.full(3, 50.)
        velocity = planet[3:] + outgoing
        destination = follow
        for _ in range(3):
            crossing = outbound_radius_crossing(planet[:3], velocity, mu_sun_sail,
                                                np.linalg.norm(destination[:3]))
            if crossing is None:
                return np.full(3, 50.)
            time, position = crossing
            if time <= 0 or epoch + time > deadline:
                return np.full(3, 50.)
            destination = env.ephemeris.state(second, epoch + time)
        return (position - destination[:3]) / AU

    points = [(rp, clock) for rp in np.linspace(bounds[0][0], bounds[1][0], 3)
              for clock in np.linspace(bounds[0][1], bounds[1][1], 13)]
    start = min(points, key=lambda point: np.linalg.norm(residual(point)))
    miss = np.linalg.norm(residual(start))
    if miss >= 50.:
        return None
    # For far misses the coarse turns suffice to rank the launch. Refine
    # plausible branches only, keeping the planetary geometry bounds intact.
    if miss < 2.:
        if not constrained_clock:
            # A full circle has no physical edge at 0/360 degrees.
            bounds = ([bounds[0][0], start[1] - np.pi], [bounds[1][0], start[1] + np.pi])
        fit = least_squares(residual, start, bounds=bounds, max_nfev=8, diff_step=1e-4)
        start, miss = fit.x, np.linalg.norm(fit.fun)
    rp, clock = np.exp(start[0]), np.degrees(start[1]) % 360
    tolerance = min(limits.tolerance_km, .01 * rp) if limits else .01 * rp
    return float(miss), FlybyTarget(float(rp), float(tolerance), limits.direction if limits else "either",
                                    float(clock), .2)


def safe_aim_radius(env, index, target):
    """Pick a safe interior aim while leaving the user's acceptance band intact."""
    safety = env.ephemeris.limits[index] / 1000
    if target.periapsis_km <= safety:
        return .5 * (safety + target.periapsis_km + target.tolerance_km)
    gate = env.ephemeris.fixed_gates[index] / 1000
    if gate and target.periapsis_km >= gate:
        return .5 * (gate + target.periapsis_km - target.tolerance_km)
    return target.periapsis_km


def past_outbound_approach(env, index):
    """Avoid propagating an already receding outward miss to the full deadline."""
    if env.ephemeris.fixed_gates[index] or index not in env.approaches:
        return False
    closest_time, _ = env.approaches[index]
    radius, radial_speed, _, energy, _, _ = env._orbit()
    if radial_speed <= 0 or energy <= 0 or env.t - closest_time <= .4 * YEAR:
        return False
    return radius > 1.25 * np.linalg.norm(env._body(env.t, index)[:3])


def fit_departing_leg(exit_env, evaluations=8, aim=None):
    """Fit a sail cone and clock from the preceding full-gravity exit state."""
    index = exit_env.next_body
    name = exit_env.ephemeris.names[index]
    target = aim or exit_env.targets.get(name)
    mu = exit_env.mu_planets[index] / 1e9

    @lru_cache(maxsize=48)
    def propagate(cone, clock):
        env = exit_env.fork_at_exit()
        env.leg_controls[name] = (cone, clock)
        env.reference_request = env._reference_command()
        while True:
            done, truncated = env.step_reference()
            if done or truncated or env.next_body != index:
                break
            # An outward hyperbolic spacecraft past a planet's orbit cannot
            # regain the earlier intercept by coasting to the mission deadline.
            if past_outbound_approach(env, index):
                break
        return env

    def measured(env):
        entry = env.gate_entries.get(index)
        sample = entry or env.approach_states.get(index)
        if sample is None:
            return None
        _, craft, planet = sample
        r, v = (craft - planet).reshape(2, 3) / 1000
        asymptote = incoming_asymptote(r, v, mu, ballistic=entry is None)
        if asymptote is None:
            return None
        incoming, impact, speed = asymptote
        if env.next_body != index:
            actual = env.flyby_passes[-1].periapsis_km
            impact *= focused_impact_parameter(actual, mu, speed) / np.linalg.norm(impact)
        return incoming, impact, speed, planet

    # Try several transverse/normal directions before using local derivatives.
    starts = [(0., 0.), *((.32, angle) for angle in (0., np.pi / 2, np.pi, -np.pi / 2))]
    stored = exit_env.leg_controls.get(name)
    if stored is not None:
        starts.insert(0, stored)
    viable = []
    misses = []

    def leg_rank(env):
        rank = mission_rank((env, env.result()))
        if aim is None:
            return rank
        tracking_error = sum(aim.error(env.flyby_passes[-1])) if env.next_body != index else np.inf
        return (rank[0], tracking_error, *rank[1:])

    for point in starts:
        env = propagate(*point)
        if index in env.approaches and not (env.thermal_violation or env.planet_violation):
            misses.append(env.approaches[index][1])
        if env.next_body != index and not (env.thermal_violation or env.planet_violation):
            encounter = env.flyby_passes[-1]
            if target is None or not any(target.error(encounter)):
                return list(point), dict(body=name, propagations=propagate.cache_info().misses,
                                         previous_exit_s=exit_env.t)
        if measured(env) is not None:
            viable.append((leg_rank(env), point))
    if not viable:
        return None, dict(body=name, status="no_usable_approach", previous_exit_s=exit_env.t)

    _, start = min(viable)
    incoming, impact, _, planet = measured(propagate(*start))
    gate = exit_env._gate(index, planet) / 1000
    if misses and min(misses) / 1000 > gate + 5 * max(gate, (max(misses) - min(misses)) / 1000):
        return None, dict(body=name, status="far_outside_sampled_leg_approaches",
                          closest_sample_km=min(misses) / 1000,
                          propagations=propagate.cache_info().misses, previous_exit_s=exit_env.t)
    pole = np.cross(planet[:3], planet[3:])
    axis = np.cross(incoming, pole)
    axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) else impact / np.linalg.norm(impact)
    second = np.cross(incoming, axis)
    direction = impact / np.linalg.norm(impact)
    if target and target.clock_deg is not None:
        clock = np.radians(target.clock_deg)
        direction = np.cos(clock) * axis + np.sin(clock) * second
    elif target and target.energy_sign * (direction @ planet[3:]) > 0:
        direction = -direction
    scale = target.tolerance_km if target else .25 * gate
    tried = set(starts)

    def residual(point):
        key = tuple(point)
        tried.add(key)
        arrival = measured(propagate(*key))
        if arrival is None:
            return np.full(2, 1e5)
        _, actual, speed, _ = arrival
        wanted = focused_impact_parameter(safe_aim_radius(env=exit_env, index=index, target=target),
                                           mu, speed) if target else .4 * gate
        difference = actual - direction * wanted
        return np.array([difference @ axis, difference @ second]) / scale

    bounds = ([0., -2 * np.pi], [np.pi / 2, 2 * np.pi])
    fit = least_squares(residual, [max(.02, start[0]), start[1]], bounds=bounds,
                        jac=bounded_jacobian(residual, bounds, (.002, .008)),
                        max_nfev=evaluations, x_scale=[.3, 1.])
    tried.add(tuple(fit.x))
    best = min(sorted(tried), key=lambda point: leg_rank(propagate(*point)))
    return list(best), dict(body=name, status=str(fit.message), propagations=propagate.cache_info().misses,
                            previous_exit_s=exit_env.t, previous_exit_state_j2000_si=exit_env.x.tolist())


def screen_previous_turn(exit_env):
    """Screen allowable turns from the previous encounter's measured entry."""
    previous, following = exit_env.route[exit_env.flyby_count - 1], exit_env.next_body
    if following is None:
        return None
    _, craft, planet = exit_env.gate_entries[previous]
    asymptote = incoming_asymptote(craft[:3] - planet[:3], craft[3:] - planet[3:],
                                   exit_env.mu_planets[previous])
    if asymptote is None:
        return None
    incoming, _, speed = asymptote
    epoch = exit_env.launch_et + exit_env.flyby_passes[-1].periapsis_time
    planet = exit_env.ephemeris.state(previous, epoch)
    result = scout_turn(exit_env, previous, following, epoch, planet, incoming, speed,
                        deadline=exit_env.launch_et + exit_env.T)
    if result is None or result[0] > .7:
        return None
    miss, geometry = result
    return geometry, miss * AU / 1000


def screen_first_fit(flight):
    """Rank first-encounter refits by their downstream potential."""
    first, second = flight.route[:2]
    closest = flight.approach_states.get(first)
    if closest is None or flight.approaches[first][1] > .5 * AU:
        return None
    entry = flight.gate_entries.get(first)
    _, craft, planet = entry or closest
    asymptote = incoming_asymptote(craft[:3] - planet[:3], craft[3:] - planet[3:],
                                   flight.mu_planets[first], ballistic=entry is None)
    if asymptote is None:
        return None
    incoming, _, speed = asymptote
    epoch = flight.launch_et + closest[0]
    planet = flight.ephemeris.state(first, epoch)
    return scout_turn(flight, first, second, epoch, planet, incoming, speed,
                      deadline=flight.launch_et + flight.T)


def suggest_later_departure(flight):
    """Estimate a later launch season without changing the requested search window."""
    if not flight.route:
        return None
    if len(flight.route) == 1:
        return screen_later_single_flyby(flight)
    first, second = flight.route[:2]
    ephemeris = flight.ephemeris
    if ephemeris.fixed_gates[first] or ephemeris.fixed_gates[second]:
        return None
    entry = flight.gate_entries.get(first)
    closest = flight.approach_states.get(first)
    if entry is None and (closest is None or flight.approaches[first][1] > .5 * AU):
        return None
    age, craft, planet = entry or closest
    asymptote = incoming_asymptote(craft[:3] - planet[:3], craft[3:] - planet[3:],
                                   flight.mu_planets[first], ballistic=entry is None)
    if asymptote is None:
        return None
    incoming, _, speed = asymptote
    components = rtn_basis(planet[:3], planet[3:]) @ incoming
    if flight.flyby_passes:
        age = flight.flyby_passes[0].periapsis_time

    # Estimate a full cycle of relative longitude, subject to ephemeris coverage.
    # A fixed 20-year scan can miss outer-planet alignment seasons.
    second_state = ephemeris.state(second, flight.launch_et + age)
    first_motion = np.sqrt(flight.mu_sun / np.linalg.norm(planet[:3])**3)
    second_motion = np.sqrt(flight.mu_sun / np.linalg.norm(second_state[:3])**3)
    rate = abs(first_motion - second_motion)
    lookahead = min(100., max(20., 2 * np.pi / (rate * YEAR))) if rate else 100.
    end = flight.params["et0"] + flight.launch_window
    best = None
    for shift in np.arange(.15, lookahead, .15):
        departure = end + shift * YEAR
        deadline = departure + flight.T
        try:
            ephemeris.check_coverage(departure, deadline)
        except (ValueError, SpiceyError):
            break
        epoch = departure + age
        future = ephemeris.state(first, epoch)
        direction = components @ rtn_basis(future[:3], future[3:])
        result = scout_turn(flight, first, second, epoch, future, direction, speed, deadline=deadline)
        if result and (best is None or result[0] < best[0]):
            best = result[0], departure, epoch
    if best is None or best[0] > .5:
        return None
    return dict(projected_miss_km=best[0] * AU / 1000, approximate_departure_et=best[1],
                approximate_first_flyby_et=best[2], assumed_first_leg_years=age / YEAR)


def target_reference(make_env, fraction, guide, index, evaluations=9, aim=None):
    """Fit an incoming B-plane vector using full-gravity propagation.

    Inside the gate, calibrate its magnitude with the integrated periapsis.
    The optimizer can also use a flyby miss or an unsafe trial as a shooting
    residual. Those trials are never accepted as reference solutions.
    """
    baseline = make_env(fraction, guide)
    target = aim or baseline.targets[baseline.ephemeris.names[index]]
    fit_epoch = bool(baseline.launch_window)
    initial = np.array([fraction if fit_epoch else guide["burn_radius_au"], guide["burn_clock_rad"]])
    width = .035 if fit_epoch else .045
    lower = np.array([max(0. if fit_epoch else guide["dive_perihelion_au"] + .015, initial[0] - width), -2.])
    upper = np.array([min(1. if fit_epoch else .65, initial[0] + width), 2.])
    mu = baseline.mu_planets[index] / 1e9  # km^3/s^2

    def settings(point):
        result = dict(guide, burn_clock_rad=float(point[1]))
        if not fit_epoch:
            result["burn_radius_au"] = float(point[0])
        return (float(point[0]) if fit_epoch else fraction), result

    @lru_cache(maxsize=64)
    def propagate(first, clock):
        f, g = settings((first, clock))
        env = make_env(f, g)
        env.reset()
        while True:
            done, truncated = env.step_reference()
            if index in env.gate_exits or done or truncated:
                break
            if past_outbound_approach(env, index):
                break
            if (index in env.approaches and env.approaches[index][1] < .5 * AU
                    and env.t - env.approaches[index][0] > .3 * YEAR):
                break
        entry = env.gate_entries.get(index)
        state = entry or env.approach_states.get(index)
        if state is None:
            return None
        _, spacecraft, planet = state
        relative = (spacecraft - planet) / 1000
        asymptote = incoming_asymptote(relative[:3], relative[3:], mu, ballistic=entry is None)
        if asymptote is None:
            return None
        incoming, impact, speed = asymptote
        encounter = next((item for item in env.flyby_passes if item.planet == env.ephemeris.names[index]), None)
        if encounter:
            impact *= focused_impact_parameter(encounter.periapsis_km, mu, speed) / np.linalg.norm(impact)
        return incoming, impact, speed, planet

    origin = propagate(*initial)
    if origin is None:
        return fraction, guide, dict(status="no_incoming_solution")
    incoming, impact, _, planet = origin
    pole = np.cross(planet[:3], planet[3:])
    first_axis = np.cross(incoming, pole)
    if not np.linalg.norm(first_axis):
        first_axis = impact
    first_axis /= np.linalg.norm(first_axis)
    second_axis = np.cross(incoming, first_axis)
    if target.clock_deg is None:
        direction = impact / np.linalg.norm(impact)
        if target.energy_sign and target.energy_sign * (direction @ planet[3:]) > 0:
            direction = -direction
    else:
        clock = np.radians(target.clock_deg)
        direction = np.cos(clock) * first_axis + np.sin(clock) * second_axis

    def residual(point):
        state = propagate(*point)
        if state is None:
            return np.full(2, 1e5)
        _, impact, speed, _ = state
        error = impact - direction * focused_impact_parameter(safe_aim_radius(baseline, index, target), mu, speed)
        return np.array([error @ first_axis, error @ second_axis]) / max(target.tolerance_km, 1.)

    initial_error = float(np.linalg.norm(residual(initial)))
    derivative = bounded_jacobian(residual, (lower, upper), (3e-7 if fit_epoch else 2e-7, 3e-6))
    fit = least_squares(residual, initial, bounds=(lower, upper), jac=derivative,
                        max_nfev=evaluations, x_scale=[.003, .025], ftol=1e-4)
    f, g = settings(fit.x)
    return f, g, dict(status=str(fit.message), propagations=propagate.cache_info().misses,
                      initial_residual_tolerances=initial_error,
                      final_residual_tolerances=float(np.linalg.norm(fit.fun)))


@dataclass(frozen=True)
class Crossing:
    relative: np.ndarray  # AU
    target: np.ndarray  # AU and AU/year
    velocity: np.ndarray  # relative velocity, AU/year
    at_closest: bool = False

    @property
    def distance(self):
        return float(np.linalg.norm(self.relative))

    def offsets(self):
        radial, transverse, normal = rtn_basis(self.target[:3], self.target[3:])
        if self.at_closest:
            first, second = self.axes()
            return np.array([self.relative @ first, self.relative @ second])
        r = self.relative + self.target[:3]
        return np.array([np.arctan2(r @ transverse, r @ radial) * np.linalg.norm(self.target[:3]),
                         self.relative @ normal])

    def axes(self):
        _, transverse, normal = rtn_basis(self.target[:3], self.target[3:])
        speed = np.linalg.norm(self.velocity)
        if self.at_closest and speed > 1e-10:
            incoming = self.velocity / speed
            lateral = np.cross(incoming, normal)
            if np.linalg.norm(lateral) > 1e-10:
                lateral /= np.linalg.norm(lateral)
                return lateral, np.cross(incoming, lateral)
        return transverse, normal


class ReferenceSearch:
    """Screen osculating-radius crossings with continuous sail guidance.

    The scout omits gravity from bodies on the requested route to keep the
    conjunction search smooth. Candidates must then pass full dynamics replay.
    """

    def __init__(self, env, *, epoch=None, launch_window=None):
        self.env = env
        self.epoch = env.params["et0"] if epoch is None else epoch
        self.launch_window = env.launch_window if launch_window is None else launch_window
        self.mu = env.mu_sun * YEAR**2 / AU**3
        self.planet_mu = env.mu_planets * YEAR**2 / AU**3
        self.planet_mu[list(env.route)] = 0.
        self.massive = self.planet_mu > 0
        self.planet_mu = self.planet_mu[self.massive]
        # Pre-tabulate the exact model's ephemerides. Cubic interpolation is used
        # only for the scout, and verification uses the original provider.
        stop = (env.T + self.launch_window) / YEAR
        times = np.linspace(0., stop, max(5, int(np.ceil(stop * 365.25)) + 1))
        positions = np.array([env.ephemeris.positions(self.epoch + t * YEAR)
                              for t in times]) / AU
        self.outer_radius = 1.2 * np.max(np.linalg.norm(positions[:, env.route], axis=2))
        self.positions = CubicSpline(times, positions, extrapolate=False)
        self.velocities = self.positions.derivative()
        self.sail = SailThrust(*env.params["sail_params"])

    def trace(self, fraction, guide):
        offset = fraction * self.launch_window / YEAR
        start = launch_state(self.epoch + offset * YEAR)
        x = np.r_[start[:3] / AU, start[3:] * YEAR / AU]
        t, phase = 0., 0
        result = {index: [] for index in self.env.route}
        horizon = self.env.T / YEAR

        def perihelion(t, state):
            radius = np.linalg.norm(state[:3])
            energy = (state[3:] @ state[3:]) / 2 - self.mu / radius
            h2 = np.cross(state[:3], state[3:])
            h2 = h2 @ h2
            e = sqrt(max(0., 1 + 2 * energy * h2 / self.mu**2))
            return h2 / (self.mu * (1 + e)) - guide.dive_perihelion_au

        def radial(t, state):
            return state[:3] @ state[3:]

        def burn(t, state):
            return np.linalg.norm(state[:3]) - guide.burn_radius_au

        def thermal(t, state):
            return np.linalg.norm(state[:3]) - self.env.r_limit / AU

        def energy(state):
            return (state[3:] @ state[3:]) / 2 - self.mu / np.linalg.norm(state[:3])

        def outbound(t, state):
            # Once unbound and past the targets' radial range, the scout has
            # collected the outbound crossings needed to screen encounters.
            return min(np.linalg.norm(state[:3]) - self.outer_radius, energy(state) / self.mu, radial(t, state))

        def target_radius(index):
            def crossing(t, state):
                query = min(self.positions.x[-1], offset + t)
                target = self.positions(query)[index]
                if self.env.ephemeris.fixed_gates[index]:
                    return (state[:3] - target) @ (state[3:] - self.velocities(query)[index])
                return np.linalg.norm(state[:3]) - np.linalg.norm(target)
            if self.env.ephemeris.fixed_gates[index]:
                crossing.direction = 1
            return crossing

        targets = [target_radius(index) for index in self.env.route]
        for function, direction in ((thermal, -1), (outbound, 1)):
            function.direction, function.terminal = direction, True
        # Use separate event functions for opposite crossings of the same radius.
        def event(function, direction, next_phase):
            def crossing(t, state):
                return function(t, state)
            crossing.direction, crossing.terminal = direction, True
            return crossing, next_phase

        switches = (
            (event(perihelion, -1, 1),),
            (event(radial, 1, None),),
            (event(burn, 1, 3),),
            (event(radial, -1, 4),),
            (event(burn, -1, 2), event(radial, 1, 3)),
        )
        for _ in range(100):
            command = guide.command(phase, self.sail)
            if np.any(command):
                thrust_scale = self.sail.scale / self.sail.mass * YEAR**2 / AU
            else:
                thrust_scale = 0.

            def rhs(time, state):
                r, v = state[:3], state[3:]
                radius = np.linalg.norm(r)
                p = self.positions(min(self.positions.x[-1], offset + time))[self.massive]
                acceleration = gravity(r, p, self.mu, self.planet_mu)
                if thrust_scale:
                    if command[1] == command[2] == 0.:
                        acceleration += thrust_scale * command[0] * r / radius**3
                    else:
                        acceleration += thrust_scale / radius**2 * (command @ rtn_basis(r, v))
                return np.r_[v, acceleration]

            phase_events = switches[phase]
            functions = [thermal, outbound, *targets, *(item[0] for item in phase_events)]
            solution = solve_ivp(rhs, (t, horizon), x, method="DOP853", rtol=2e-9, atol=1e-10,
                                 max_step=.10, events=functions)
            if not solution.success:
                return result, -self.mu
            for index, times, states in zip(self.env.route, solution.t_events[2:], solution.y_events[2:]):
                for time, state in zip(times, states):
                    query = min(self.positions.x[-1], offset + time)
                    p = np.r_[self.positions(query)[index], self.velocities(query)[index]]
                    result[index].append((time, Crossing(state[:3] - p[:3], p, state[3:] - p[3:],
                                                          bool(self.env.ephemeris.fixed_gates[index]))))
            t, x = solution.t[-1], solution.y[:, -1]
            if len(solution.t_events[0]) or t >= horizon:
                break
            if len(solution.t_events[1]):
                break
            for (_, next_phase), times in zip(phase_events, solution.t_events[2 + len(targets):]):
                if len(times):
                    phase = guide.outbound_phase(np.linalg.norm(x[:3]) * AU) if next_phase is None else next_phase
                    if phase == 1 and radial(t, x) >= 0:
                        phase = guide.outbound_phase(np.linalg.norm(x[:3]) * AU)
                    break
            else:
                break
            # Move to the outgoing side of an event before changing the event set.
            t += 1e-11
        return result, float(energy(x))

    def aim(self, index, crossing, normal_side=None, target=None):
        radius = np.linalg.norm(crossing.target[:3])
        gate = self.env.ephemeris.gate(index, radius * AU, self.env.mu_sun) / AU
        target = target or self.env.targets.get(self.env.ephemeris.names[index])
        if target is None:
            distance = .32 * gate
        else:
            periapsis = safe_aim_radius(self.env, index, target) * 1000 / AU
            speed2 = max(.01, crossing.velocity @ crossing.velocity)
            mu = self.env.mu_planets[index] * YEAR**2 / AU**3
            distance = min(.85 * gate, sqrt(periapsis**2 + 2 * mu * periapsis / speed2))
        speed = np.linalg.norm(crossing.velocity)
        if target is not None and speed > 1e-10 and (target.clock_deg is not None or target.energy_sign):
            incoming = crossing.velocity / speed
            _, _, pole = rtn_basis(crossing.target[:3], crossing.target[3:])
            reference = np.cross(incoming, pole)
            if np.linalg.norm(reference) > 1e-10:
                reference /= np.linalg.norm(reference)
                clock = target.clock_deg if target.clock_deg is not None else (180. if target.energy_sign > 0 else 0.)
                angle = np.deg2rad(clock)
                offset = distance * (np.cos(angle) * reference + np.sin(angle) * np.cross(incoming, reference))
                axes = crossing.axes()
                return np.array([offset @ axis for axis in axes])
        if normal_side is None:
            normal_side = -1. if crossing.offsets()[1] < 0 else 1.
        return np.array([0., normal_side * distance])


def bounded_jacobian(residual, bounds, strides):
    """Use physical step sizes so finite differences resolve phase switches."""
    lower, upper = bounds

    def derivative(point):
        columns = []
        for axis, stride in enumerate(strides):
            left, right = point.copy(), point.copy()
            left[axis] = max(lower[axis], left[axis] - stride)
            right[axis] = min(upper[axis], right[axis] + stride)
            columns.append((residual(right) - residual(left)) / (right[axis] - left[axis]))
        return np.column_stack(columns)

    return derivative


def screen_later_single_flyby(flight):
    """Use the launch and sail scout to suggest a later single flyby season."""
    if flight.guide is None:
        return None
    start = flight.params["et0"] + flight.launch_window + DAY
    for years in (6., 3., 1.):
        window = years * YEAR
        try:
            flight.ephemeris.check_coverage(start, start + window + flight.T)
        except (ValueError, SpiceyError):
            continue
        break
    else:
        return None
    search = ReferenceSearch(flight, epoch=start, launch_window=window)
    candidate = next(scout_candidates(flight, int(12 * years) + 1,
                                      search=search, profiles=(flight.guide,)), None)
    if candidate is None:
        return None
    fraction, guide = candidate
    tracks, _ = search.trace(fraction, SolarDiveGuide(**guide))
    index = flight.route[0]
    approaches = [(float(np.linalg.norm(crossing.offsets() - search.aim(index, crossing))), age)
                  for age, crossing in tracks[index]]
    if not approaches:
        return None
    miss, age = min(approaches)
    if miss > .5:
        return None
    departure = start + fraction * window
    return dict(projected_miss_km=miss * AU / 1000, approximate_departure_et=departure,
                approximate_first_flyby_et=departure + age * YEAR, assumed_first_leg_years=age)


def scout_candidates(env, samples, on_profile=None, on_turn=None, *, search=None, profiles=None):
    """Yield screened launch and guide settings, with no success claims."""
    search = ReferenceSearch(env) if search is None else search
    first = env.route[0]

    fractions = np.linspace(0., 1., samples) if search.launch_window else np.array([0.])
    records = []
    # Alternate propulsion arcs move the arrival longitude and escape energy.
    if profiles is None:
        guide = SolarDiveGuide()
        burns = (guide.burn_radius_au, .355, .365, .397, .405, .425, .445)
        profiles = [replace(guide, burn_radius_au=burn) for burn in burns]
    @lru_cache(maxsize=512)
    def trace(profile, fraction, clock, burn):
        return search.trace(fraction, replace(profiles[profile], burn_clock_rad=clock, burn_radius_au=burn))

    def turn_data(fraction, time, crossing):
        if len(env.route) < 2 or crossing.distance > 1.5:
            return None
        planet = np.r_[crossing.target[:3] * AU, crossing.target[3:] * AU / YEAR]
        velocity = crossing.velocity * AU / YEAR
        speed = np.linalg.norm(velocity)
        if speed <= 0:
            return None
        departure = search.epoch + fraction * search.launch_window
        epoch = departure + time * YEAR
        return scout_turn(env, first, env.route[1], epoch, planet, velocity / speed, speed,
                          deadline=departure + env.T)

    @lru_cache(maxsize=512)
    def best_turn(profile, fraction, clock, burn):
        tracks, _ = trace(profile, fraction, clock, burn)
        turns = [turn for time, crossing in tracks[first]
                 if (turn := turn_data(fraction, time, crossing)) is not None]
        return min(turns, key=lambda turn: turn[0]) if turns else None

    for profile, template in enumerate(profiles):
        burn = template.burn_radius_au
        clock = template.burn_clock_rad
        previous = {}
        for fraction in fractions:
            fraction = float(fraction)
            tracks, energy = trace(profile, fraction, clock, burn)
            current = {}
            for number, (time, crossing) in enumerate(tracks[first]):
                along, _ = crossing.offsets()
                desired = search.aim(first, crossing)[0]
                current[number] = (fraction, along - desired, time)
                old = previous.get(number)
                records.append((crossing.distance, profile, fraction, number, burn, energy))
                if (old and old[1] * (along - desired) < 0 and abs(time - old[2]) < 1.2
                        and max(abs(old[1]), abs(along)) < np.linalg.norm(crossing.target[:3])):
                    def longitude(value):
                        found, _ = trace(profile, float(value), clock, burn)
                        if len(found[first]) <= number:
                            raise ValueError("Encounter branch disappeared")
                        event = found[first][number][1]
                        return event.offsets()[0] - search.aim(first, event)[0]
                    try:
                        root = brentq(longitude, old[0], fraction, xtol=3e-7, maxiter=16)
                    except (ValueError, RuntimeError):
                        continue
                    found, energy_at_root = trace(profile, root, clock, burn)
                    records.append((found[first][number][1].distance, profile, root, number, burn, energy_at_root))
            previous = current
        if on_profile is not None:
            on_profile(profile + 1, len(profiles))

    def score(candidate):
        profile, fraction, clock, burn = candidate
        tracks, energy = trace(profile, float(fraction), float(clock), float(burn))
        # Keep the best prefix at each encounter time. A slightly closer
        # crossing can be too late for the next body in the requested order.
        prefixes = {-1.: (0, 0., 0.)}
        for index in env.route:
            advanced = {time: (misses + 1, max(worst, 1000.), total + 1000.)
                        for time, (misses, worst, total) in prefixes.items()}
            for time, crossing in tracks[index]:
                gate = env.ephemeris.gate(index, np.linalg.norm(crossing.target[:3]) * AU, env.mu_sun) / AU
                ratio = crossing.distance / gate
                geometry = np.linalg.norm(crossing.offsets() - search.aim(index, crossing)) / gate
                eligible = [(misses + int(ratio >= 1), max(worst, geometry), total + ratio)
                            for after, (misses, worst, total) in prefixes.items() if after < time]
                if eligible:
                    advanced[time] = min([advanced[time], *eligible]) if time in advanced else min(eligible)
            prefixes = advanced
        misses, worst, total = min(prefixes.values())
        first_complete = min((time for time, rank in prefixes.items() if rank[0] == 0), default=np.inf)
        completion = fraction * search.launch_window / YEAR + first_complete
        escape_penalty = max(0., -energy / search.mu)
        first_inside = any(crossing.distance * AU < env.ephemeris.gate(
            first, np.linalg.norm(crossing.target[:3]) * AU, env.mu_sun)
            for _, crossing in tracks[first])
        if len(env.route) > 1:
            turn = best_turn(*candidate)
            return (not turn or turn[0] > .7, int(not first_inside), turn[0] if turn else np.inf, misses,
                    escape_penalty, worst, total, completion, -energy)
        return (int(not first_inside), misses, escape_penalty, worst, total, completion, -energy)

    candidates, fit_seeds = [], []
    separation = .5 / max(1, samples - 1)
    seen = []
    screened_turns = []

    def coarse_rank(item):
        distance, profile, fraction, number, burn, energy = item
        priority = distance + 2 * max(0., -energy / search.mu)
        if len(env.route) > 1 and distance <= 1.5:
            tracks, _ = trace(profile, fraction, profiles[profile].burn_clock_rad, burn)
            turn = turn_data(fraction, *tracks[first][number])
            if turn:
                screened_turns.append((turn[0], fraction, tracks[first][number][0]))
                return (False, turn[0] + priority)
        return (len(env.route) > 1, priority)

    order = sorted(records, key=coarse_rank)
    if on_turn is not None and screened_turns:
        miss, fraction, age = min(screened_turns)
        on_turn(dict(body=env.flybys[1], projected_miss_km=miss * AU / 1000,
                     departure_fraction=fraction, first_encounter_year=age,
                     screened_crossings=len(screened_turns)))
    for _, profile, fraction, number, burn, _ in order:
        if any(abs(fraction - old) < separation and number == old_number and profile == old_profile
               for old, old_number, old_profile in seen):
            continue
        seen.append((fraction, number, profile))
        if len(seen) > 6:
            break
        clock = profiles[profile].burn_clock_rad
        base = np.array([fraction, clock, burn])
        fit_epoch = bool(search.launch_window)
        tracks, _ = trace(profile, float(fraction), clock, float(burn))
        side = -1. if tracks[first][number][1].offsets()[1] < 0 else 1.
        turn = turn_data(fraction, *tracks[first][number])
        aimed_turn = turn[1] if turn and turn[0] < 1. else None
        # Hold burn radius for the first solve. A free launch date and sail
        # clock steer the azimuth and latitude of the encounter independently.
        def unpack(parameters):
            return (parameters[0], parameters[1], base[2]) if fit_epoch else (0., parameters[1], parameters[0])

        def residual(parameters):
            trial = unpack(parameters)
            tracks, _ = trace(profile, *(float(value) for value in trial))
            if len(tracks[first]) <= number:
                return np.array([10., 10.])
            crossing = tracks[first][number][1]
            return crossing.offsets() - search.aim(first, crossing, normal_side=side, target=aimed_turn)

        initial = [fraction if fit_epoch else base[2], clock]
        width = 1.5 / max(1, samples - 1)
        bounds = ([max(0., fraction - width), clock - 1.1], [min(1., fraction + width), clock + 1.1]) if fit_epoch else (
            [max(.31, profiles[profile].dive_perihelion_au + .02), clock - 1.1], [.50, clock + 1.1])

        nearby = [initial, [initial[0], clock - .08], [initial[0], clock + .08]]
        initial = min(nearby, key=lambda point: np.linalg.norm(residual(point)))
        jacobian = bounded_jacobian(residual, bounds, (2e-5 if fit_epoch else 1e-5, .004))
        fit = least_squares(residual, initial, bounds=bounds, jac=jacobian, max_nfev=12, ftol=2e-5)
        fitted = (profile, *unpack(fit.x))
        candidates.extend([(profile, *base), fitted])
        fit_seeds.append((fitted, number, side))
    # When the fixed-profile fits still miss, free the burn radius as an extra
    # degree of freedom. The first two residuals aim the first encounter and
    # the last penalizes a heliocentrically bound outbound trajectory.
    miss_index = 1
    if search.launch_window and fit_seeds and min(score(candidate)[miss_index] for candidate in candidates) > 0:
        for fitted, number, side in sorted(fit_seeds, key=lambda seed: score(seed[0]))[:3]:
            profile, *initial = fitted
            def combined(point):
                tracks, energy = trace(profile, *(float(value) for value in point))
                if len(tracks[first]) <= number:
                    return np.array([10., 10., 1.])
                crossing = tracks[first][number][1]
                offsets = crossing.offsets() - search.aim(first, crossing, side)
                return np.r_[offsets, .3 * max(0., -energy / search.mu)]

            fraction, _, burn = initial
            clock = profiles[profile].burn_clock_rad
            lower = np.array([max(0., fraction - .04), clock - 1.1,
                              max(.31, burn - .018, profiles[profile].dive_perihelion_au + .02)])
            upper = np.array([min(1., fraction + .04), clock + 1.1, min(.50, burn + .018)])
            if np.any(np.asarray(initial) < lower) or np.any(np.asarray(initial) > upper):
                continue
            derivative = bounded_jacobian(combined, (lower, upper), (2e-5, .003, 1e-5))
            fit = least_squares(combined, initial, bounds=(lower, upper), jac=derivative,
                                max_nfev=10, x_scale=[.01, .10, .01])
            candidates.append((profile, *fit.x))
    for candidate in sorted(set(candidates), key=score):
        profile, fraction, clock, burn = candidate
        yield float(fraction), asdict(replace(profiles[profile], burn_clock_rad=float(clock),
                                            burn_radius_au=float(burn)))
