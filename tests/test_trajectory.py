"""Regression checks for sail thrust, flyby targeting, and reference replay."""

import json
import unittest
from unittest.mock import Mock, patch

import numpy as np
import spiceypy as spice
from scipy.integrate import solve_ivp

from dynamics import gravity, outbound_radius_crossing
from ephemerides import planet_state
from flyby import FlybyTarget, focused_impact_parameter, incoming_asymptote, parse_flyby_geometry
from guidance import JUPITER_REFERENCE_EPOCH
from plan import mission_for_later_season
from replay import command_schedule, launch_fraction, mission_rank, replay_reference, verify_commands
from reference_search import fit_departing_leg, suggest_later_departure
from solar_sail import SailThrust, solar_sail_accel
from trajectory import AU, DAY, YEAR, SolarSailTrajectory, launch_state, make_initial_state


class SolarSailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x0, cls.params = make_initial_state()

    def test_projected_thrust_matches_sail_and_inverse_square_scaling(self):
        sail = SailThrust(*self.params["sail_params"])
        x = np.array([AU, 0, 0, 0, 30000, 0])
        for vector in ([2, 1, -1], [.5, -.2, .3], [0, 0, 0], [-1, 0, 0], [1, 0, 0]):
            command = sail.project(vector)
            force = sail.force(x, command)
            self.assertGreaterEqual(force[0], 0.)
            self.assertLessEqual(np.linalg.norm(force), sail.scale * (1 + 1e-12))
            np.testing.assert_allclose(sail.force(x * [2, 2, 2, 1, 1, 1], command), force / 4.)
        with self.assertRaises(ValueError):
            sail.project([np.inf, 0, 0])

    def test_reference_selects_departure_inside_launch_window(self):
        env = SolarSailTrajectory(self.x0, self.params, launch_window_years=5, launch_anchor=.75)
        env.reset()
        self.assertEqual(env.step_reference(), (False, False))
        epoch = self.params["et0"] + 3.75 * YEAR
        self.assertEqual(env.launch_et, epoch)
        np.testing.assert_allclose(env.x, launch_state(epoch))
        self.assertEqual(env.t, 0.)
        self.assertFalse(env.pending_launch)

    def test_smooth_projection_recovers_feasible_vector_between_grid_points(self):
        sail = SailThrust(*self.params["sail_params"])
        x = np.array([AU, 0., 0., 0., 30000., 0.])
        for cone in (.002, .617, 1.099, 1.569):
            force = sail.mass * solar_sail_accel(x[:3], x[3:], cone, .75, *self.params["sail_params"])
            request = force / sail.scale
            np.testing.assert_allclose(sail.project(request), request, rtol=1e-9, atol=1e-12)

    def test_thermal_and_planet_constraints(self):
        env = SolarSailTrajectory([.201 * AU, 0, 0, -10000, 30000, 0], self.params)
        env.reset()
        done, timeout = env.step_reference()
        info = env.result()
        self.assertTrue(done and not timeout and info["thermal_violation"])
        self.assertAlmostEqual(info["perihelion_au"], .2)

        venus = planet_state(1, self.params["et0"])
        radial = venus[:3] / np.linalg.norm(venus[:3])
        start = np.r_[venus[:3] + 10e6 * radial, venus[3:] - 8000 * radial]
        env = SolarSailTrajectory(start, self.params, flybys=("Venus",), accurate=True)
        env.reset()
        done, _ = env.step_reference()
        info = env.result()
        self.assertTrue(done and info["planet_violation"])
        self.assertEqual(info["flyby_count"], 0)

    def jupiter_env(self, targets=None, route=("Jupiter",)):
        p = planet_state(4, self.params["et0"])
        er = p[:3] / np.linalg.norm(p[:3])
        eh = np.cross(p[:3], p[3:])
        eh /= np.linalg.norm(eh)
        gate = .2 * np.linalg.norm(p[:3]) * (self.params["mu_planets"][4] / (3 * self.params["mu_sun"]))**(1/3)
        start = np.r_[p[:3] + gate * (1.2 * er + .5 * eh), p[3:] - 30000 * er]
        return SolarSailTrajectory(start, self.params, flybys=route, dt_days=3,
                              horizon_years=.1, accurate=True, record=True, targets=targets)

    def test_flyby_requires_exit_advances_route_and_reports_geometry(self):
        env = self.jupiter_env(route=("Jupiter", "Saturn"))
        env.reset()
        for _ in range(10):
            env.step_reference()
            if env.entry is not None:
                break
        self.assertIsNotNone(env.entry)
        self.assertEqual(env.flyby_count, 0)
        result, info = replay_reference(env)
        self.assertEqual(info["flyby_count"], 1)
        self.assertEqual(result.next_body, 5)
        self.assertFalse(info["mission_success"])
        encounter = result.flyby_passes[0]
        self.assertLess(encounter.entry_time, encounter.periapsis_time)
        self.assertLess(encounter.periapsis_time, encounter.exit_time)
        self.assertGreater(encounter.turn_deg, 0)
        self.assertAlmostEqual(encounter.b_plane_clock_deg, 270, places=2)
        self.assertLess(encounter.energy_change_km2_s2, 0)

    def test_geometry_failure_and_fixed_command_verification(self):
        baseline, _ = replay_reference(self.jupiter_env())
        target = FlybyTarget(baseline.flyby_passes[0].periapsis_km + 10, 1, "either")

        def make_env():
            return self.jupiter_env({"Jupiter": target})

        env, info = replay_reference(make_env())
        row = info["flyby_diagnostics"][0]
        self.assertTrue(row["gate_entered"] and row["gate_exited"] and row["flyby_completed"])
        self.assertFalse(info["geometry_ok"] or row["geometry_satisfied"] or info["mission_success"])
        self.assertAlmostEqual(row["absolute_periapsis_error_km"], 10.)
        self.assertGreater(row["focusing_ratio"], 1.)
        same, diagnostics = verify_commands(make_env, launch_fraction(env), command_schedule(env))
        np.testing.assert_allclose(env.x, same.x, rtol=1e-9)
        json.dumps(diagnostics, allow_nan=False)

    def test_focusing_maps_hyperbolic_periapsis_to_impact_parameter(self):
        mu, periapsis, excess = 5.7939e6, 80_000., 22.6
        _, vector, speed = incoming_asymptote(
            np.array([periapsis, 0., 0.]), np.array([0., np.sqrt(excess**2 + 2 * mu / periapsis), 0.]), mu)
        self.assertAlmostEqual(speed, excess)
        self.assertAlmostEqual(np.linalg.norm(vector), focused_impact_parameter(periapsis, mu, excess))
        self.assertAlmostEqual(np.linalg.norm(vector), 90_638., delta=2.)

    def test_geometry_parser_checks_safety_and_wraps_clock(self):
        targets = parse_flyby_geometry(["Venus:10000:1000:boost:-1:4"], ["Venus"])
        self.assertEqual(targets["Venus"].clock_deg, 359)
        self.assertEqual(parse_flyby_geometry(["Venus:6500:1000:boost"], ["Venus"])["Venus"].periapsis_km, 6500.)
        with self.assertRaises(ValueError):
            parse_flyby_geometry(["Venus:6100:100:boost"], ["Venus"])

    def test_reference_command_records_applied_feasible_force(self):
        env = SolarSailTrajectory(self.x0, self.params, guide={}, record=True)
        env.reset()
        planned = env.reference_request.copy()
        env.step_reference()
        _, _, requested, force, _, applied = env.history_u[0]
        np.testing.assert_array_equal(requested, planned)
        np.testing.assert_allclose(applied, env.sail.project(requested))
        np.testing.assert_allclose(force, env.sail.force(env.x0, applied))

    def test_escape_requires_outbound_positive_energy(self):
        for radius, speed, success in ((2., 40_000., True), (120., 1000., False), (2., -40_000., False)):
            start = np.array([radius * AU, 0., 0., speed, 100., 0.])
            env = SolarSailTrajectory(start, self.params, horizon_years=1e-5)
            env.reset()
            done, timeout = env.step_reference()
            self.assertEqual(done, success)
            self.assertEqual(timeout, not success)
            info = env.result()
            self.assertEqual(info["mission_success"], success)
            if success:
                self.assertLess(info["r_au"], 3.)
                self.assertGreater(info["energy_m2s2"], 0.)
                self.assertIsNotNone(info["escape_et"])
            else:
                self.assertIsNone(info["escape_et"])

    def test_escape_is_localized_when_sail_crosses_the_energy_boundary(self):
        radius = 2 * AU
        speed = np.sqrt(2 * self.params["mu_sun"] / radius) - 1
        env = SolarSailTrajectory([radius, 0., 0., speed, 0., 0.], self.params, accurate=True)
        env.reset()
        done, timeout = env.advance(np.array([1., 0., 0.]))
        self.assertTrue(done and not timeout)
        self.assertLess(env.t, env.dt)
        self.assertAlmostEqual(env.result()["energy_m2s2"], 1., places=3)

    def test_ranking_prefers_earlier_escape_then_higher_escape_speed(self):
        slower = dict(mission_success=True, thermal_violation=False, planet_violation=False,
                      escape_et=1000., projected_sail_v_inf_km_s=12., day=300.)
        later = dict(slower, escape_et=1001., projected_sail_v_inf_km_s=20., day=200.)
        faster = dict(slower, projected_sail_v_inf_km_s=18.)
        self.assertLess(mission_rank((None, slower)), mission_rank((None, later)))
        self.assertLess(mission_rank((None, faster)), mission_rank((None, slower)))

    def test_reference_completes_jupiter_flyby_on_an_escape_trajectory(self):
        x0, params = make_initial_state(JUPITER_REFERENCE_EPOCH)
        env = SolarSailTrajectory(x0, params, flybys=("Jupiter",), horizon_years=30.,
                              dt_days=4.,
                              guide={}, accurate=True)
        phases = [0]
        env.reset()
        while True:
            done, timeout = env.step_reference()
            if env.guide_phase != phases[-1]:
                phases.append(env.guide_phase)
            if done or timeout:
                break
        info = env.result()
        self.assertEqual(phases[:4], [0, 1, 2, 3])
        self.assertTrue(info["mission_success"])
        self.assertAlmostEqual(env.t, env.flyby_passes[0].exit_time)
        self.assertGreater(env.flyby_passes[0].energy_change_km2_s2, 0.)
        self.assertLess(env.t, 30 * YEAR)

    def test_full_gravity_matches_spice_third_body_sum(self):
        r = self.x0[:3]
        ephemeris = self.params["ephemeris"]
        epoch = self.params["et0"]
        expected = -self.params["mu_sun"] * r / np.linalg.norm(r)**3
        positions = ephemeris.positions(epoch)
        for index, mu in enumerate(ephemeris.mu):
            planet = planet_state(index, epoch)[:3]
            np.testing.assert_allclose(positions[index], planet)
            relative = planet - r
            expected += mu * (relative / np.linalg.norm(relative)**3 - planet / np.linalg.norm(planet)**3)
        np.testing.assert_allclose(gravity(r, positions, self.params["mu_sun"], ephemeris.mu), expected)

    def test_outbound_turn_screen_matches_integrated_kepler_crossings(self):
        def rhs(t, state):
            r = state[:3]
            return np.r_[state[3:], -r / np.linalg.norm(r)**3]

        def event(t, state):
            return np.linalg.norm(state[:3]) - 1.5

        event.terminal = True
        for transverse in (1.2, 1.6):
            r = np.array([1., 0., 0.])
            v = np.array([.3, transverse, 0.])
            time, position = outbound_radius_crossing(r, v, 1., 1.5)
            sol = solve_ivp(rhs, (0., 10.), np.r_[r, v], rtol=1e-11, atol=1e-12, events=event)
            self.assertAlmostEqual(time, sol.t[-1], places=8)
            np.testing.assert_allclose(position, sol.y[:3, -1], atol=1e-8)
        self.assertIsNone(outbound_radius_crossing(r, np.array([.1, .9, 0.]), 1., 2.))

    def test_uranus_precise_reference_and_numerical_verification(self):
        guide = dict(burn_radius_au=.405, burn_clock_rad=.16681741083922835)
        targets = {"Uranus": FlybyTarget(80_000., 8_000., "either")}

        def make_env():
            return SolarSailTrajectory(self.x0, self.params, flybys=("Uranus",), targets=targets,
                                  guide=guide, launch_window_years=2., launch_anchor=.16934407788666775,
                                  accurate=True, record=True)

        env, info = replay_reference(make_env())
        self.assertTrue(info["mission_success"])
        self.assertAlmostEqual(info["encounters"][0]["periapsis_km"], 80_000., delta=10.)
        _, verified = verify_commands(make_env, launch_fraction(env), command_schedule(env))
        self.assertTrue(verified["mission_success"])

    def test_tabulated_massless_flyby_recovers_small_gate_crossings(self):
        epoch = self.params["et0"]
        # The moving massless waypoint passes the ship inside a short gate
        # transit at mid-step. Its small safety boundary stays clear.
        start = np.array([1.5 * AU, 0., 0., 10000., 32000., 0.])
        age = 10.
        center = start[:3] + age * start[3:] + np.array([0., 0., 4000.])
        waypoint_velocity = np.array([-5000., 32000., 0.])
        states = [[epoch + t, *((center + (t - age) * waypoint_velocity) / 1000),
                   *(waypoint_velocity / 1000)] for t in (-60., DAY)]
        _, params = make_initial_state(bodies=[dict(name="Test comet", safety_radius_km=1.,
                                                   encounter_radius_km=12., states=states)])
        with self.assertRaisesRegex(ValueError, "outside the .* encounter gate"):
            parse_flyby_geometry(["Test comet:15:1"], ["Test comet"], params["ephemeris"])
        env = SolarSailTrajectory(start, params, flybys=("Test comet",), dt_days=.003,
                              horizon_years=.00001)
        env.reset()
        env.step_reference()
        info = env.result()
        self.assertEqual(info["flyby_count"], 1)
        self.assertFalse(info["planet_violation"])
        self.assertAlmostEqual(info["encounters"][0]["periapsis_km"], 4., places=2)
        self.assertIsNone(info["encounters"][0]["turn_deg"])
        with self.assertRaises(ValueError):
            params["ephemeris"].check_coverage(epoch, epoch + 2 * DAY)

    def test_two_geometry_flybys_targeted_from_the_first_gravity_exit(self):
        epoch = self.params["et0"]
        start = np.array([2 * AU, 0., 0., 18000., 32000., 0.])
        # Prescribed moving massive bodies produce resolved turns on both legs.
        positions = [
            [[299235741.4, -2764800., 2000.], [299235741.4, 55296000., 2000.]],
            [[306128497.6542354, -2759590.2171265543, -96626.00993129844],
             [308513253.3070725, 55275857.53946738, 1915739.649139882]],
        ]
        velocities = [[0., 32., 0.], [1.3143494559287356, 31.986027202708296, 1.1091080572482255]]
        names = ("First", "Second")
        bodies = [dict(name=name, gm_km3_s2=20000., safety_radius_km=120., encounter_radius_km=6000.,
                       states=[[epoch + t, *position, *velocity]
                               for t, position in zip((-DAY, 20 * DAY), row)])
                  for name, row, velocity in zip(names, positions, velocities)]
        _, params = make_initial_state(bodies=bodies)
        targets = {"First": FlybyTarget(1942.10845, 1., "either"),
                   "Second": FlybyTarget(1418.95218, .8, "either")}

        def make_env(controls=None):
            return SolarSailTrajectory(start, params, flybys=names, targets=targets, accurate=True,
                                  record=True, horizon_years=.05, dt_days=.25,
                                  leg_controls=controls)

        exits = []
        baseline, info = replay_reference(make_env(), on_exit=lambda env: exits.append(env.fork_at_exit()))
        self.assertFalse(info["mission_success"])
        self.assertEqual(info["valid_flyby_count"], 1)
        self.assertEqual(len(exits), 2)
        first = exits[0]
        self.assertEqual(first.t, baseline.flyby_passes[0].exit_time)
        np.testing.assert_array_equal(first.x, baseline.exit_states[first.route[0]])
        self.assertGreater(first.flyby_passes[0].turn_deg, 3.)
        angles, fit = fit_departing_leg(first)
        self.assertEqual(fit["previous_exit_s"], first.t)
        self.assertEqual(first.leg_controls, {})
        controls = {"Second": angles}
        selected, info = replay_reference(make_env(controls))
        self.assertTrue(info["mission_success"])
        self.assertEqual(info["valid_flyby_count"], 2)
        self.assertEqual(selected.flyby_passes[0], baseline.flyby_passes[0])
        self.assertLess(selected.flyby_passes[0].exit_time, selected.flyby_passes[1].entry_time)
        _, verified = verify_commands(lambda: make_env(controls), 0., command_schedule(selected))
        self.assertTrue(verified["mission_success"])
        self.assertTrue(all(row["geometry_satisfied"] for row in verified["flyby_diagnostics"]))

    def test_retry_season_keeps_the_saved_route_and_geometry(self):
        mission = dict(epoch="2035-01-01T00:00:00", launch_window_years=2., flybys=["Saturn", "Uranus"],
                       geometry=["Saturn:800000:100000:either", "Uranus:80000:8000:either"],
                       horizon_years=50., dt_days=4.)
        departure = spice.str2et("2073-11-25T05:24:00")
        saved = dict(mission=mission, later_departure_screen=dict(approximate_departure_et=departure))
        retry = mission_for_later_season(saved)
        self.assertAlmostEqual(spice.str2et(retry["epoch"]) + 1.5 * YEAR, departure, places=3)
        self.assertEqual(retry["launch_window_years"], 3.)
        self.assertEqual(retry, dict(mission, epoch=retry["epoch"], launch_window_years=3.))
        self.assertEqual(mission["launch_window_years"], 2.)
        with self.assertRaisesRegex(ValueError, "does not contain a later departure hint"):
            mission_for_later_season(dict(mission=mission))

    def test_later_departure_can_screen_a_slow_pair_from_a_close_miss(self):
        env = SolarSailTrajectory(self.x0, self.params, flybys=("Saturn", "Uranus"),
                                  launch_window_years=2.)
        env.reset()
        env.step_reference()
        first = env.route[0]
        age = 7 * YEAR
        planet = env.ephemeris.state(first, env.launch_et + age)
        craft = planet + [0., .04 * AU, 0., 10_000., 0., 0.]
        env.approach_states[first] = (age, craft, planet)
        env.approaches[first] = (age, .04 * AU)
        window_end = self.params["et0"] + env.launch_window

        def screen(flight, first, second, epoch, planet, incoming, speed, deadline):
            shift_years = (epoch - age - window_end) / YEAR
            if abs(shift_years - 25.05) < .001:
                return .0001, None
            return None

        with patch("reference_search.scout_turn", side_effect=screen):
            hint = suggest_later_departure(env)
        self.assertIsNotNone(hint)
        self.assertAlmostEqual((hint["approximate_departure_et"] - window_end) / YEAR, 25.05)
        self.assertEqual(hint["assumed_first_leg_years"], 7.)

    def test_single_flyby_hint_uses_a_scouted_departure_outside_the_window(self):
        env = SolarSailTrajectory(self.x0, self.params, flybys=("Neptune",),
                                  launch_window_years=2., guide={})
        window_end = self.params["et0"] + env.launch_window
        with patch("reference_search.ReferenceSearch") as search, patch("reference_search.scout_candidates") as scout:
            candidate = dict(burn_radius_au=.405)
            scout.return_value = iter([(.25, candidate)])
            crossing = Mock()
            crossing.offsets.return_value = np.array([.001, 0.])
            search.return_value.aim.return_value = np.zeros(2)
            search.return_value.trace.return_value = ({env.route[0]: [(12., crossing)]}, 0.)
            hint = suggest_later_departure(env)
        search.assert_called_once_with(env, epoch=window_end + DAY, launch_window=6. * YEAR)
        self.assertEqual(scout.call_args.kwargs["profiles"], (env.guide,))
        self.assertAlmostEqual(hint["approximate_departure_et"], window_end + DAY + 1.5 * YEAR)
        self.assertAlmostEqual(hint["projected_miss_km"], .001 * AU / 1000)

    def test_real_jupiter_uranus_reference_escapes_after_both_flybys(self):
        x0, params = make_initial_state("2042-01-01T00:00:00")
        route = ("Jupiter", "Uranus")
        targets = parse_flyby_geometry(
            ["Jupiter:600000:100000:either", "Uranus:80000:8000:either"], route, params["ephemeris"])

        def make_env():
            return SolarSailTrajectory(
                x0, params, flybys=route, targets=targets, launch_window_years=10.,
                launch_anchor=.480216001111974, record=True, accurate=True,
                guide=dict(burn_radius_au=.355, burn_clock_rad=.15980011801881017),
                leg_controls={"Uranus": (.07509759502052098, -.0008377786241839996)})

        flight, result = replay_reference(make_env())
        self.assertTrue(result["mission_success"])
        _, checked = verify_commands(make_env, launch_fraction(flight), command_schedule(flight))
        self.assertTrue(checked["mission_success"])
        self.assertEqual(tuple(p["planet"] for p in checked["encounters"]), route)
        self.assertLess(checked["encounters"][0]["exit_time"], checked["encounters"][1]["entry_time"])
        self.assertLess(checked["flight_years"], 50.)

    def test_real_jupiter_saturn_reference_escapes_after_both_flybys(self):
        route = ("Jupiter", "Saturn")
        targets = parse_flyby_geometry(
            ["Jupiter:600000:100000:either", "Saturn:180000:30000:either"], route, self.params["ephemeris"])

        def make_env():
            return SolarSailTrajectory(
                self.x0, self.params, flybys=route, targets=targets, launch_window_years=16.,
                horizon_years=65., launch_anchor=.04330369115684194, record=True, accurate=True,
                guide=dict(burn_radius_au=.397, burn_clock_rad=.16897626087369857),
                leg_controls={"Saturn": (.24777901687641818, 3.12367369160493)})

        flight, result = replay_reference(make_env())
        self.assertTrue(result["mission_success"])
        _, checked = verify_commands(make_env, launch_fraction(flight), command_schedule(flight))
        self.assertTrue(checked["mission_success"])
        self.assertEqual(tuple(p["planet"] for p in checked["encounters"]), route)
        self.assertLess(checked["encounters"][0]["exit_time"], checked["encounters"][1]["entry_time"])


if __name__ == "__main__":
    unittest.main()
