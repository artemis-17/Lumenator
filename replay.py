"""Replay reference guidance, verify fixed thrust commands, and rank trajectories."""

import numpy as np


def replay_reference(env, on_exit=None):
    env.reset()
    while True:
        previous_count = env.flyby_count
        done, timeout = env.step_reference()
        if (on_exit is not None and env.flyby_count > previous_count
                and not (env.thermal_violation or env.planet_violation)):
            on_exit(env)
        if done or timeout:
            return env, env.result(diagnostics=True)


def replay_commands(env, launch_fraction, schedule):
    """Replay controls at fixed mission elapsed times, independent of event step splits."""
    env.reset()
    if env.pending_launch:
        env.select_launch(launch_fraction)
    return _continue_commands(env, schedule)


def _continue_commands(env, schedule):
    for end, request in schedule:
        while env.t < end - 1e-6:
            done, timeout = env.advance(request, end, open_loop=True)
            if done or timeout:
                return env, env.result(diagnostics=True)
    # A tighter integration can reach escape just after the final command.
    while env.t < env.T - 1e-6:
        done, timeout = env.advance(schedule[-1][1], open_loop=True)
        if done or timeout:
            break
    return env, env.result(diagnostics=True)


def command_schedule(env):
    return [(end, request) for _, end, request, *_ in env.history_u]


def launch_fraction(env):
    return (env.launch_et - env.params["et0"]) / env.launch_window if env.launch_window else 0.


def verify_commands(make_env, fraction, schedule):
    env = make_env()
    env.rtol = 5e-13
    env.atol = [.005] * 3 + [2.5e-8] * 3
    env.max_step /= 2
    return replay_commands(env, fraction, schedule)


def mission_rank(result):
    env, info = result
    unsafe = info["thermal_violation"] or info["planet_violation"]
    if info["mission_success"]:
        return (unsafe, False, info["escape_et"], -info["projected_sail_v_inf_km_s"], info["day"])
    errors = [sum(env.targets[p.planet].error(p)) for p in env.flyby_passes if p.planet in env.targets]
    prefix = 0
    for encounter in env.flyby_passes:
        target = env.targets.get(encounter.planet)
        if target is not None and any(target.error(encounter)):
            break
        prefix += 1
    miss = env.approaches.get(env.next_body, (0, np.inf))[1]
    return (unsafe, True, -prefix, -env.flyby_count, sum(e > 0 for e in errors),
            round(sum(errors), 8), round(miss, 1), not info["escape_reached"],
            -round(info["energy_m2s2"], 1))


def print_mission(label, info):
    speed = info["projected_sail_v_inf_km_s"]
    escape = f" | Projected escape speed {speed:.3f} km/s" if info["mission_success"] else ""
    print(f"{label} | Success {info['mission_success']} | Geometry-valid flybys {info['valid_flyby_count']}"
          f" | {info['r_au']:.3f} AU | {info['day'] / 365.25:.3f} years"
          f"{escape} | Status {info['failure_reason'] or 'success'}", flush=True)
    for row in info["flyby_diagnostics"]:
        value = row["actual_periapsis_km"]
        field = "periapsis" if value is not None else "closest approach"
        if value is None:
            value = row["closest_approach_km"]
        amount = f"{value:,.0f} km" if value is not None else "unobserved"
        error = row["absolute_periapsis_error_km"]
        print(f"  {row['body']} {field} {amount}"
              + (f" | periapsis error {error:,.0f} km" if error is not None else "")
              + f" | {row['failure_reason'] or 'valid flyby'}", flush=True)


