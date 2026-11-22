"""Search for a solar-sail reference trajectory with optional flybys."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import spiceypy as spice
from spiceypy.utils.exceptions import SpiceyError

import config
from ephemerides import load_kernels, read_body_tables
from flyby import FlybyTarget, GEOMETRY_HELP, parse_flyby_geometry
from guidance import SolarDiveGuide
from reference_search import (fit_departing_leg, scout_candidates, screen_first_fit,
                              screen_previous_turn, suggest_later_departure, target_reference)
from replay import mission_rank, print_mission, replay_reference
from trajectory import AU, YEAR, SolarSailTrajectory, make_initial_state


def choose_reference(make_env, seed=None, save_progress=None):
    env = make_env()
    template = asdict(SolarDiveGuide())
    turn_screen = []
    verified, report = [], []

    def snapshot(status):
        best = min(verified, key=lambda row: row[0]) if verified else None
        fastest = max(
            (index for index, candidate in enumerate(report)
             if candidate.get("result", {}).get("mission_success")),
            key=lambda index: (report[index]["result"]["projected_sail_v_inf_km_s"],
                               -report[index]["result"]["escape_et"]),
            default=None,
        )
        return dict(status=status, selected=best[1] if best else None,
                    result=best[2][1] if best else None, candidates=report,
                    fastest_candidate_index=fastest,
                    feasibility=("full_dynamics_feasible" if best and best[2][1]["mission_success"]
                                 else "not_found" if status == "complete" else "searching"),
                    departure_turn_screen=turn_screen[0] if turn_screen else None)

    def describe_turn(screen):
        turn_screen.append(screen)
        print(f"Best screened {screen['body']} miss after the first turn "
              f"{screen['projected_miss_km']:,.0f} km (approximate)", flush=True)
        if save_progress:
            save_progress(snapshot("verifying"))

    if seed:
        candidates = [(seed["launch_anchor"], seed["guide"])]
    elif env.route:
        print("Screening departure and sail profiles before full-gravity replay", flush=True)
        candidates = scout_candidates(
            env, config.REFERENCE_DEPARTURES,
            on_profile=lambda complete, total: print(f"Screened sail profiles {complete}/{total}", flush=True),
            on_turn=describe_turn)
    else:
        fractions = (0., .5, 1.) if env.launch_window else (0.,)
        candidates = ((fraction, dict(template, burn_radius_au=burn))
                      for fraction in fractions for burn in (.377453, .405))
    def verify(fraction, guide, fit=None, leg_controls=None):
        candidate = dict(guide=guide, launch_anchor=fraction)
        if leg_controls:
            candidate["leg_controls"] = leg_controls
        exits = {}

        def capture(state):
            if state.next_body is not None and state._geometry_ok():
                exits[state.flyby_count] = state.fork_at_exit()

        try:
            flight = replay_reference(make_env(**candidate), on_exit=capture)
        except RuntimeError as exc:
            report.append(dict(candidate, optimization=fit, error=str(exc)))
            return
        info = flight[1]
        report.append(dict(candidate, optimization=fit, result=info))
        print_mission(f"Reference candidate {len(report)}", info)
        verified.append((mission_rank(flight), candidate, flight, exits))
        if save_progress:
            save_progress(snapshot("verifying"))
        return info

    if save_progress:
        save_progress(snapshot("screening"))
    for _, (fraction, guide) in zip(range(config.REFERENCE_REPLAYS), candidates):
        verify(fraction, guide, leg_controls=seed.get("leg_controls") if seed else None)
    feasible = any(row[2][1]["mission_success"] for row in verified)
    if (env.targets or len(env.route) > 1) and not feasible:
        has_valid_first_exit = any(1 in row[3] for row in verified)
        trials = [(row, screen_first_fit(row[2][0]) if len(env.route) > 1 else None) for row in verified]
        trials.sort(key=lambda trial: (trial[0][0][0],
                                       trial[1][0] if trial[1] and trial[1][0] < 1. else np.inf,
                                       trial[0][0]))
        for (_, candidate, (flight, info), _), turn in trials[:config.REFERENCE_TARGET_FITS]:
            pending = next((row for row in info["flyby_diagnostics"]
                            if row["requested_periapsis_km"] and not row["geometry_satisfied"]), None)
            aim = None
            if turn is not None and turn[0] < 1.:
                index, aim = env.route[0], turn[1]
            elif (len(env.route) > 1 and not has_valid_first_exit and flight.flyby_count == 0
                  and env.flybys[0] not in env.targets):
                index = env.route[0]
                approach = flight.approach_states.get(index)
                if approach is None or flight.approaches[index][1] > .5 * AU:
                    continue
                gate_km = flight._gate(index, approach[2]) / 1000
                aim = FlybyTarget(.35 * gate_km, .05 * gate_km, "either")
            elif pending is not None:
                index = env.ephemeris.names.index(pending["body"])
            else:
                continue
            # Fit the insertion from Earth only for the first body. Later
            # encounters are fitted from the actual, geometry-valid exit.
            if len(env.route) > 1 and index != env.route[0]:
                continue
            print(f"Fitting {env.ephemeris.names[index]} geometry with full planetary gravity", flush=True)
            try:
                fraction, guide, fit = target_reference(
                    lambda f, g: make_env(guide=g, launch_anchor=f, leg_controls=candidate.get("leg_controls")),
                    candidate["launch_anchor"], candidate["guide"], index, config.REFERENCE_FIT_STEPS, aim=aim)
            except (ValueError, RuntimeError) as exc:
                report.append(dict(candidate, error=str(exc)))
                continue
            if turn is not None and aim is turn[1]:
                fit.update(screened_next_miss_km=turn[0] * AU / 1000,
                           aimed_periapsis_km=aim.periapsis_km, aimed_clock_deg=aim.clock_deg)
            verify(fraction, guide, fit, candidate.get("leg_controls"))
            if verified[-1][2][1]["mission_success"]:
                break
    feasible = any(row[2][1]["mission_success"] for row in verified)
    if len(env.route) > 1 and not feasible:
        for _, candidate, _, exits in sorted(verified, key=lambda row: row[0])[:4]:
            if 1 not in exits:
                continue
            exit_env = exits[1]
            print(f"Screening allowable {exit_env.flyby_passes[0].planet} turns toward {env.flybys[1]}", flush=True)
            try:
                hint = screen_previous_turn(exit_env)
                if hint is None:
                    report.append(dict(candidate, optimization=dict(
                        body=env.flybys[1], status="no_promising_preceding_turn")))
                    continue
                desired, miss = hint
                fraction, guide, details = target_reference(
                    lambda f, g: make_env(guide=g, launch_anchor=f, leg_controls=candidate.get("leg_controls")),
                    candidate["launch_anchor"], candidate["guide"], env.route[0],
                    config.REFERENCE_FIT_STEPS, aim=desired)
                details.update(next_body=env.flybys[1], screened_miss_km=miss,
                               aimed_periapsis_km=desired.periapsis_km, aimed_clock_deg=desired.clock_deg)
                verify(fraction, guide, details, candidate.get("leg_controls"))
            except (ValueError, RuntimeError) as exc:
                report.append(dict(candidate, error=str(exc), failed_leg=env.flybys[1]))
            if any(row[2][1]["mission_success"] for row in verified):
                break
    attempted = set()
    for _ in range(config.REFERENCE_LEG_BRANCHES * max(0, len(env.route) - 1)):
        if any(row[2][1]["mission_success"] for row in verified):
            break
        branch = None
        for _, candidate, (_, previous_result), exits in sorted(verified, key=lambda row: row[0]):
            for count in sorted(exits, reverse=True):
                key = json.dumps([candidate, count], sort_keys=True)
                if key not in attempted:
                    branch = candidate, exits[count], exits.get(count - 1), key, previous_result
                    break
            if branch is not None:
                break
        if branch is None:
            break
        candidate, exit_env, preceding_exit, key, previous_result = branch
        attempted.add(key)
        name = exit_env.flybys[exit_env.flyby_count]
        print(f"Targeting {name} from the full-gravity exit of {exit_env.flyby_passes[-1].planet}", flush=True)
        try:
            angles, details = fit_departing_leg(exit_env, config.REFERENCE_LEG_FIT_STEPS)
        except (ValueError, RuntimeError) as exc:
            report.append(dict(candidate, error=str(exc), failed_leg=name))
            continue
        if angles is None:
            report.append(dict(candidate, optimization=details))
            print(f"  Leg search stopped with status {details['status']}", flush=True)
            if preceding_exit is None:
                continue
            # If this leg cannot be acquired from the existing exit, try
            # changing the preceding turn without changing its own prefix.
            try:
                hint = screen_previous_turn(exit_env)
                if hint is not None:
                    desired, miss = hint
                    revised, turn_fit = fit_departing_leg(
                        preceding_exit, config.REFERENCE_LEG_FIT_STEPS, aim=desired)
                    if revised is not None:
                        turn_fit.update(next_body=name, screened_miss_km=miss)
                        controls = dict(candidate.get("leg_controls", {}),
                                        **{exit_env.flyby_passes[-1].planet: revised})
                        verify(candidate["launch_anchor"], candidate["guide"], turn_fit, controls)
            except (ValueError, RuntimeError) as exc:
                report.append(dict(candidate, error=str(exc), failed_leg=name))
            continue
        controls = dict(candidate.get("leg_controls", {}), **{name: angles})
        info = verify(candidate["launch_anchor"], candidate["guide"], details, controls)
        target = env.targets.get(name)
        # Continue a useful near-band fit with its new controls. Avoid spending
        # another trial on the same exit when it failed to improve the geometry.
        polished = False
        if info is not None and target:
            old = next(row for row in previous_result["flyby_diagnostics"] if row["body"] == name)
            new = next(row for row in info["flyby_diagnostics"] if row["body"] == name)
            old_error = old["absolute_periapsis_error_km"]
            new_error = new["absolute_periapsis_error_km"]
            polished = (new_error is not None and not new["geometry_satisfied"]
                        and new_error <= max(4 * target.tolerance_km, .5 * target.periapsis_km)
                        and (old_error is None or new_error < .9 * old_error))
        if not polished:
            attempted.add(json.dumps([dict(candidate, leg_controls=controls), exit_env.flyby_count], sort_keys=True))
    if not verified:
        verify(.5 if env.launch_window else 0., template)
    if not verified:
        raise RuntimeError("No reference could be propagated")
    final = snapshot("complete")
    if env.route and not final["result"]["mission_success"]:
        _, _, (best_flight, _), _ = min(verified, key=lambda row: row[0])
        if len(env.route) == 1:
            print(f"Screening later departure seasons for {env.flybys[0]}", flush=True)
        try:
            later = suggest_later_departure(best_flight)
        except (ValueError, RuntimeError, SpiceyError) as exc:
            print(f"Later departure screening stopped. {exc}", flush=True)
            later = None
        if later:
            later["approximate_departure_utc"] = spice.et2utc(later["approximate_departure_et"], "ISOC", 0)
            final["later_departure_screen"] = later
            print(f"Approximate later departure season {later['approximate_departure_utc']} UTC "
                  f"(outside requested window, not full-gravity verified)", flush=True)
    return final


def mission_for_later_season(report):
    """Reuse the mission and search around its screened departure date."""
    mission = report["mission"]
    hint = (report.get("later_departure_screen") or {}).get("approximate_departure_et")
    if hint is None or not np.isfinite(hint):
        raise ValueError("The saved search does not contain a later departure hint")
    load_kernels()
    end = spice.str2et(mission["epoch"]) + mission["launch_window_years"] * YEAR
    if hint <= end:
        raise ValueError("The saved departure hint must be after the original launch window")
    start = max(end, hint - 1.5 * YEAR)
    duration = (hint + 1.5 * YEAR - start) / YEAR
    return dict(mission, epoch=spice.et2utc(start, "ISOC", 3), launch_window_years=duration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flybys", nargs="+", metavar="BODY")
    parser.add_argument("--geometry", action="append", metavar=GEOMETRY_HELP)
    parser.add_argument("--max-years", type=float)
    parser.add_argument("--epoch")
    parser.add_argument("--launch-window-years", type=float)
    parser.add_argument("--dt-days", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ephemeris", type=Path, help="JSON state tables for additional flyby bodies")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--reference", type=Path, help="Start reference selection from an existing reference JSON")
    inputs.add_argument("--retry-season", type=Path, metavar="SEARCH_JSON",
                        help="Search around the later departure hint in a saved mission")
    args = parser.parse_args()
    if args.retry_season and any(value is not None for value in (
            args.flybys, args.geometry, args.max_years, args.epoch,
            args.launch_window_years, args.dt_days, args.ephemeris)):
        parser.error("--retry-season uses the saved mission settings. Specify only --output to change where it saves.")
    try:
        if args.retry_season:
            source = json.loads(args.retry_season.read_text(encoding="utf-8-sig"))
            mission = mission_for_later_season(source)
            output = args.output or args.retry_season.parent / "later"
            print(f"Retrying near the later departure hint from {mission['epoch']} UTC "
                  f"over {mission['launch_window_years']:.3f} years", flush=True)
        else:
            mission = dict(flybys=args.flybys or [], geometry=args.geometry or [],
                           epoch=args.epoch if args.epoch is not None else config.DEFAULT_EPOCH,
                           horizon_years=args.max_years if args.max_years is not None else config.MAX_YEARS,
                           launch_window_years=args.launch_window_years if args.launch_window_years is not None
                           else config.LAUNCH_WINDOW_YEARS,
                           dt_days=args.dt_days if args.dt_days is not None else config.GUIDANCE_DAYS)
            if args.ephemeris:
                mission["body_tables"] = read_body_tables(args.ephemeris)
            output = args.output or Path("runs/latest")
        settings = {key: mission[key] for key in ("flybys", "horizon_years", "dt_days", "launch_window_years")}
        mission.pop("goal_au", None)
        mission["objective"] = "escape"
        x0, params = make_initial_state(mission["epoch"], mission.get("body_tables", []))
        ephemeris = params["ephemeris"]
        targets = parse_flyby_geometry(mission.get("geometry", []), settings["flybys"], ephemeris)

        def make_env(**overrides):
            return SolarSailTrajectory(x0, params, **dict(settings, **overrides), targets=targets, accurate=True)

        env = make_env()
        ephemeris.check_coverage(params["et0"], params["et0"] + env.launch_window + env.T)
        seed = None
        if args.reference:
            source = json.loads(args.reference.read_text(encoding="utf-8-sig"))
            if not isinstance(source, dict):
                raise ValueError("A reference seed must be a JSON object")
            seed = source.get("selected", source)
        report_path = output / "reference_search.json"
        if args.retry_season and report_path.resolve() == args.retry_season.resolve():
            raise ValueError("--retry-season requires an output directory separate from the original search")
        output.mkdir(parents=True, exist_ok=True)

        def save_reference(reference):
            pending = report_path.with_suffix(".tmp")
            pending.write_text(json.dumps(dict(reference, mission=mission), indent=2, allow_nan=False),
                               encoding="utf-8")
            pending.replace(report_path)

        reference = choose_reference(make_env, seed, save_progress=save_reference)
        save_reference(reference)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, SpiceyError) as exc:
        parser.error(str(exc))
    print_mission("Selected reference", reference["result"])
    print(f"Saved reference to {report_path.resolve()}")


if __name__ == "__main__":
    main()
