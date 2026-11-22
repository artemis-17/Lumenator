"""Replay a saved reference and export its trajectory and thrust schedule."""

import argparse
import csv
import json
from pathlib import Path
import webbrowser

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import spiceypy as spice
from spiceypy.utils.exceptions import SpiceSPKINSUFFDATA, SpiceyError

from dynamics import gravity
from ephemerides import PLANET_IDS, planet_positions, read_body_tables
from flyby import parse_flyby_geometry
from flyby_camera import CAMERA_MARKER, save_flyby_camera
from plotting import save_animation
from replay import command_schedule, launch_fraction, print_mission, replay_reference, verify_commands
from trajectory import AU, DAY, YEAR, SolarSailTrajectory, make_initial_state


def planet_coverage_end(start_et, end_et):
    """Find the end of the continuous planetary coverage used for the replay."""
    try:
        planet_positions(end_et)
    except SpiceSPKINSUFFDATA:
        # The installed planetary kernel covers the verified escape state.
        lower, upper = start_et, end_et
        while True:
            midpoint = lower + (upper - lower) / 2
            if midpoint == lower or midpoint == upper:
                return lower
            try:
                planet_positions(midpoint)
            except SpiceSPKINSUFFDATA:
                upper = midpoint
            else:
                lower = midpoint
    return end_et


def propagate_after_escape(flight, years):
    """Continue the verified state under its final projected sail command."""
    start, end = flight.t, flight.t + years * YEAR
    start_et, end_et = flight.launch_et + start, flight.launch_et + end
    if not np.isfinite(end_et) or end_et <= start_et:
        raise ValueError("Post-escape duration is outside the supported time range")
    planet_end = planet_coverage_end(start_et, end_et)
    planet_count = len(PLANET_IDS)
    tables = flight.ephemeris.tables
    cutoffs = np.array([planet_end] * planet_count + [table.x[-1] for table in tables])

    def positions_at(et):
        planets = planet_positions(et) if et <= planet_end else np.full((planet_count, 3), np.nan)
        return np.vstack([planets, *(table(et) for table in tables)]) if tables else planets

    command = flight.history_u[-1][5]
    times, states = [], []
    state = flight.x
    boundaries = sorted({end_et, *cutoffs[(cutoffs > start_et) & (cutoffs < end_et)]})
    for boundary_et in boundaries:
        boundary = end if boundary_et == end_et else boundary_et - flight.launch_et
        active = flight.massive & (cutoffs >= boundary_et)
        gm = flight.mu_planets[active]

        def rhs(t, x):
            bodies = positions_at(min(flight.launch_et + t, boundary_et))[active]
            acceleration = gravity(x[:3], bodies, flight.mu_sun, gm)
            return np.r_[x[3:], acceleration + flight.sail.force(x, command) / flight.sail.mass]

        sol = solve_ivp(rhs, (start, boundary), state, method="DOP853",
                        rtol=flight.rtol, atol=flight.atol,
                        max_step=flight.max_step if gm.size else 30 * DAY)
        if not sol.success:
            raise RuntimeError(f"Post-escape integration failed. {sol.message}")
        times.extend(sol.t[1:])
        states.extend(sol.y.T[1:])
        start, state = boundary, sol.y[:, -1]

    coverage_ends = {name: float(cutoff) for name, cutoff in zip(flight.ephemeris.names, cutoffs)
                     if cutoff < end_et}
    info = dict(years=years, start_et=start_et, end_et=end_et,
                end_distance_au=float(np.linalg.norm(state[:3]) / AU),
                held_command_rtn=command.tolist(), ephemeris_cutoffs_et=coverage_ends)
    return np.array(times), np.array(states), positions_at, info


def inclination_summary(env):
    states = np.asarray(env.trace_x)
    rotation = spice.pxform("J2000", "ECLIPJ2000", env.launch_et)
    angular_momentum = np.cross(states[:, :3], states[:, 3:]) @ rotation.T
    inclination = np.degrees(np.arctan2(np.linalg.norm(angular_momentum[:, :2], axis=1),
                                         angular_momentum[:, 2]))
    return dict(launch=float(inclination[0]), final=float(inclination[-1]), max=float(np.max(inclination)))


def save_results(folder, env, report, show, camera_flight=None, extension=None):
    folder.mkdir(parents=True, exist_ok=True)
    t, x = np.array(env.trace_t), np.array(env.trace_x)
    starts = np.array([row[0] for row in env.history_u])
    ends = np.array([row[1] for row in env.history_u])
    requests = np.array([row[2] for row in env.history_u])
    thrust_start = np.array([row[3] for row in env.history_u])
    thrust_end = np.array([row[4] for row in env.history_u])
    normalized_rtn = np.array([row[5] for row in env.history_u])
    force_times, forces = starts, thrust_start
    escape_time, positions_at = None, None
    if extension is not None:
        more_t, more_x, positions_at, info = extension
        escape_time = env.t
        t, x = np.r_[t, more_t], np.vstack([x, more_x])
        starts, ends = np.r_[starts, escape_time], np.r_[ends, more_t[-1]]
        requests = np.vstack([requests, requests[-1]])
        command = normalized_rtn[-1]
        normalized_rtn = np.vstack([normalized_rtn, command])
        more_force = np.array([env.sail.force(state, command) for state in more_x])
        force_times, forces = np.r_[starts, more_t], np.vstack([thrust_start, thrust_end[-1], more_force])
        thrust_start = np.vstack([thrust_start, thrust_end[-1]])
        thrust_end = np.vstack([thrust_end, more_force[-1]])
        report["post_escape_extension"] = info
    np.savez_compressed(folder / "trajectory.npz", t=t, X=x.T, launch_et=env.launch_et,
                        command_start_s=starts, command_end_s=ends, command_request_rtn=requests,
                        thrust_start_j2000_n=thrust_start, thrust_end_j2000_n=thrust_end,
                        thrust_normalized_rtn=normalized_rtn, max_thrust_at_1au_n=env.sail.scale)
    with (folder / "thrust_vectors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["elapsed_start_s", "elapsed_end_s", "start_et_tdb_s", "end_et_tdb_s",
                         "start_fx_n", "start_fy_n", "start_fz_n",
                         "end_fx_n", "end_fy_n", "end_fz_n",
                         "radial_1au_n", "transverse_1au_n", "normal_1au_n"])
        for start, end, first, last, rtn in zip(starts, ends, thrust_start, thrust_end, normalized_rtn):
            writer.writerow([start, end, env.launch_et + start, env.launch_et + end, *first, *last, *(env.sail.scale * rtn)])
    camera_path = folder / "flyby_camera.html"
    has_camera = camera_flight is not None and bool(camera_flight.flyby_passes)
    open_camera = show and has_camera and extension is None
    if has_camera:
        save_flyby_camera(camera_path, camera_flight)
        report["flyby_camera"] = camera_path.name
    elif camera_path.is_file():
        with camera_path.open("rb") as handle:
            generated = handle.read(len(CAMERA_MARKER)) == CAMERA_MARKER.encode("ascii")
        if generated:
            camera_path.unlink()
    (folder / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save_animation(folder / "animation.html", t, x[:, :3], env.launch_et, env.flyby_passes,
                   show=show and not open_camera, ephemeris=env.ephemeris,
                   escape_time=escape_time, positions_at=positions_at)

    radius = np.linalg.norm(x[:, :3], axis=1)
    energy = np.sum(x[:, 3:]**2, axis=1) / 2 - env.mu_sun / radius
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    time_unit, time_label = (YEAR, "years") if extension is not None else (DAY, "days")
    axes[0].plot(t / time_unit, radius / AU)
    axes[0].axhline(env.r_limit / AU, linestyle="--", label="Thermal limit")
    if escape_time is not None:
        for axis in axes:
            axis.axvline(escape_time / time_unit, linestyle=":", color="0.45",
                         label="Verified escape" if axis is axes[0] else None)
    axes[0].legend()
    axes[0].set_ylabel("Distance (AU)")
    axes[1].plot(t / time_unit, energy / 1e6)
    axes[1].axhline(0., linestyle=":", color="0.5", label="Parabolic boundary")
    axes[1].legend()
    axes[1].set_ylabel("Solar energy (km²/s²)")
    for i, name in enumerate(("X", "Y", "Z")):
        axes[2].plot(force_times / time_unit, forces[:, i], label=name)
    axes[2].set_ylabel("J2000 thrust (N)")
    axes[2].set_xlabel(f"Mission elapsed time ({time_label})")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(folder / "trajectory.png", dpi=160)
    plt.close(fig)
    if open_camera:
        webbrowser.open(camera_path.resolve().as_uri())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("runs/latest/reference_search.json"),
                        help="Search JSON containing the selected trajectory to replay")
    parser.add_argument("--ephemeris", type=Path, help="Body tables for a legacy reference without its ephemeris")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--extend-years", type=float, default=0., metavar="YEARS",
                        help="Project this many years past verified escape with the final sail command held. "
                             "Bodies are omitted after their ephemeris coverage ends.")
    parser.add_argument("--show", action="store_true",
                        help="Open the trajectory overview for an extended run, or the flyby camera when available")
    args = parser.parse_args()
    if not np.isfinite(args.extend_years) or args.extend_years < 0:
        parser.error("--extend-years must be finite and nonnegative")

    try:
        saved = json.loads(args.reference.read_text(encoding="utf-8-sig"))
        mission, selected = saved["mission"], saved["selected"]
        mission = {key: value for key, value in mission.items() if key != "goal_au"}
        mission["objective"] = "escape"
        if not isinstance(selected, dict) or selected.get("guide") is None or "launch_anchor" not in selected:
            raise ValueError("The saved search does not contain a selected reference")
        keys = ("flybys", "dt_days", "horizon_years", "launch_window_years")
        settings = {key: mission[key] for key in keys if key in mission}
        controls = {key: selected[key] for key in ("guide", "launch_anchor", "leg_controls") if key in selected}
        bodies = read_body_tables(args.ephemeris) if args.ephemeris else mission.get("body_tables", [])
        if args.ephemeris:
            mission = dict(mission, body_tables=bodies)
        x0, params = make_initial_state(mission["epoch"], bodies)
        ephemeris = params["ephemeris"]
        targets = parse_flyby_geometry(mission.get("geometry", []), settings.get("flybys", []), ephemeris)

        def make_trajectory(record=False):
            return SolarSailTrajectory(x0, params, **settings, **controls,
                                       targets=targets, accurate=True, record=record)

        env = make_trajectory(record=True)
        ephemeris.check_coverage(params["et0"], params["et0"] + env.launch_window + env.T)
        flight, reference = replay_reference(env)
        verified_flight, verified = verify_commands(
            lambda: make_trajectory(record=True), launch_fraction(flight), command_schedule(flight))
        report = dict(mission=mission, selected=controls,
                      reference=dict(reference, ecliptic_inclination_deg=inclination_summary(flight)),
                      fixed_command_verification=verified,
                      verified_mission_success=reference["mission_success"] and verified["mission_success"])
        output = args.output or args.reference.parent / "evaluation"
        camera_flight = verified_flight if report["verified_mission_success"] else None
        extension = None
        if args.extend_years and report["verified_mission_success"]:
            extension = propagate_after_escape(verified_flight, args.extend_years)
        elif args.extend_years:
            print("Post-escape extension skipped because escape did not verify.")
        display_flight = verified_flight if extension is not None else flight
        save_results(output, display_flight, report, args.show, camera_flight=camera_flight, extension=extension)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, SpiceyError) as exc:
        parser.error(str(exc))

    print_mission("Reference", reference)
    print_mission("Fixed-command verification", verified)
    if extension is not None:
        info = extension[-1]
        print(f"Projected {info['years']:g} years past verified escape to {info['end_distance_au']:.3f} AU")
        if info["ephemeris_cutoffs_et"]:
            print("Some body ephemerides end during the projection. See post_escape_extension in report.json.")
    print(f"Saved trajectory, thrust, animation, and report to {output.resolve()}")


if __name__ == "__main__":
    main()
