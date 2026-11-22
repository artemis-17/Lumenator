# LUMEN solar-sail trajectory planner

LUMEN searches for the earliest outbound solar escape that completes an optional ordered flyby route. The mission starts from the modeled post-Earth-departure state, with 50 flight years available by default. The model uses a 340 m² sail, a 5 kg spacecraft, 0.9 specular reflectivity, a 0.2 AU solar limit, and SPICE solar and planetary gravity. The Earth launch maneuver and attitude dynamics are outside the model.

## Plan and evaluate

Run from the repository root. Dependencies are in `requirements.txt` and the SPICE kernels are in `kernels/`.

```powershell
python -m pip install -r requirements.txt
python plan.py --flybys Uranus --geometry Uranus:80000:8000:either
python evaluate.py --reference runs/latest/reference_search.json --show
```

Use `python plan.py` for direct escape. Use `--flybys Jupiter Saturn` to check encounters in order. Repeat `--geometry` to specify targets for multiple bodies. Geometry uses **planet-centered distance**, in km, with the form `BODY:PERIAPSIS_KM[:TOL_KM[:boost|brake|either[:CLOCK_DEG[:CLOCK_TOL_DEG]]]]`. The defaults are ±10% for periapsis, `either` for energy change, and ±10° when a B-plane clock is specified.

| Planning option | Purpose and default |
| --- | --- |
| `--flybys BODY [BODY ...]` | Ordered route. Omit for direct escape. |
| `--geometry SPEC` | Requested flyby geometry. Repeat for each target body. |
| `--max-years YEARS` | Time allowed to complete the route and reach escape, 50 years by default. |
| `--epoch UTC` | Start of the launch window, 2035-01-01 UTC by default. |
| `--launch-window-years YEARS` | Departure search window, two years by default. Use zero to fix departure at the epoch. |
| `--dt-days DAYS` | Nominal guidance interval, four days by default. |
| `--output DIR` | Reference output directory, `runs/latest` by default. |
| `--reference PATH` | Seed the search with an existing reference and check it against the current mission. |
| `--retry-season PATH` | Reuse a saved mission and search up to 1.5 years on either side of its later departure hint, outside the original window. Saves under `later/` beside the source JSON by default. |
| `--ephemeris PATH` | Load state tables for additional flyby bodies. |

Evaluation uses `--reference PATH` to select the saved search file, defaulting to `runs/latest/reference_search.json`. Use `--output DIR` to change the export directory, `--show` to open the animation, and `--ephemeris PATH` to supply body tables for an older reference that lacks them. Both commands support `--help`. Numerical defaults are in [config.py](config.py).

## Output folders

| Folder | Contents |
| --- | --- |
| `runs/` | Mission plans and their evaluation exports. A default planning run rewrites `runs/latest/reference_search.json`. Replay it to update the exports, or use `--output runs/<name>` to save a mission separately. |
| `test_runs/` | Outputs from manual smoke tests and regression checks. For an experiment, pass `--output test_runs/<name>` explicitly. The automated tests live in `tests/`. |
| `archive/run_cleanup_2026-11-21/` | Older runs, test outputs, and training artifacts organized under their original folder names. |

## Reference search and flybys

The analytical reference uses retrograde thrust to lower perihelion, then applies prograde thrust on close solar passes to build energy. The initial search samples departure and burn profiles against the moving bodies. This screening approximation omits gravity from the requested flyby bodies. Promising references are replayed with full gravity. For unmet precise geometry, a bounded targeting fit varies departure or burn radius and sail clock using the same numerical propagation as evaluation.

Candidate comparison happens in `plan.py`, which uses the search routines in `reference_search.py` and saves its choice in the `selected` field of `reference_search.json`. Among the feasible replays, the planner prefers the earliest calendar date at which the flybys and escape criteria are both satisfied. Higher projected escape speed breaks a tie. The `candidates` field records the alternatives, and `fastest_candidate_index` is the zero-based index of the feasible candidate with the highest projected escape speed. The bounded search samples a finite set of launch and sail profiles. Evaluation reads the mission settings and replays only `selected`, including for older saved plans. Its `reference` and `fixed_command_verification` results describe the same trajectory, first with its reference guidance and then with the recorded commands at tighter integration tolerances.

The targeting fit uses the incoming planet-relative asymptote. In two-body motion, gravitational focusing gives `b² = rp² + 2 μ rp / v∞²`, where `b` is the asymptotic impact parameter and `rp` is periapsis. Near Uranus, an 80,000 km periapsis at about 22.6 km/s excess speed corresponds to an impact parameter near 90,638 km. The fit uses the integrated periapsis after an actual pass.

A planetary encounter requires an inbound gate crossing, a measured closest approach, and an outbound gate crossing on a safe hyperbolic pass. The default gate is 20% of the planet's modeled Hill radius. The full simulation checks escape once all requested flybys have exited in order. A successful trajectory meets the flyby geometry, travels outward, and has positive Sun-only specific orbital energy `v²/2 − μ☉/r` before the deadline. This check uses the integrated state with a 1 m²/s² numerical margin above the parabolic boundary. The output ends at the escape check.

The report gives the escape epoch and the final flyby exit epoch as TDB seconds past J2000. It also gives `v_inf_km_s` for a Sun-only coast and `projected_sail_v_inf_km_s` for an ideal face-on sail held from that state with inverse-square radiation pressure. Both are asymptotic speed estimates from the escape state.

For successive flybys, the search saves the exact position, velocity, and time at each geometry-valid exit. It fits feasible sail steering for the next leg from that state using full planetary gravity. When a preceding turn may help, a fast screen proposes an alternative periapsis and B-plane clock within the requested limits, followed by a full-gravity refit. The selected sequence is replayed from Earth and every requested geometry is checked again. Reference and evaluation reports include the exit states and identify which encounters passed their geometry checks. The sail feasibility projection and inverse-square radiation pressure determine the applied force.

For outward planet pairs, departure screening ranks candidates by where the first planet can send them within its requested geometry. It looks up the next planet at the projected arrival time. The `departure_turn_screen` field in `reference_search.json` records the best approximate miss found in the launch window. The full-gravity candidate results determine mission success.

A `not_found` result means the bounded search found no feasible trajectory in the requested window. A periapsis band wholly below the safety radius or wholly outside a fixed encounter gate is rejected as a constraint conflict. The planner computes later departure hints from the modeled trajectory and body ephemerides. For a single flyby, it screens launches over up to six more years with the selected sail profile. For a planetary pair, it uses a resolved first encounter or a close approach to screen the next alignment, which can take more than 20 years for slow-moving pairs. Both screens require ephemeris coverage and a promising projected approach. Use `python plan.py --retry-season runs/latest/reference_search.json` to test the suggested season with full gravity and the saved flyby geometry. Then evaluate `runs/latest/later/reference_search.json` to refresh the plots for that search.

These saved plans passed the tighter fixed-command replay under the escape objective. All requested flybys were within tolerance.

| Saved route | Departure | Time from launch to escape check | Sun-only excess speed |
| --- | --- | --- | --- |
| Jupiter → Uranus (`runs/jupiter_uranus_escape/`) | Sep 2035 | 11.756 years | 5.06 km/s |
| Jupiter → Uranus (`runs/jupiter_uranus_calibrated/`) | Oct 2046 | 8.973 years | 18.53 km/s |
| Jupiter → Saturn (`runs/jupiter_saturn_calibrated/`) | Sep 2035 | 5.632 years | 21.42 km/s |
| Saturn → Uranus (`runs/saturn_uranus_verified/`) | Jan 2071 | 9.114 years | 29.40 km/s |
| Saturn → Uranus, 800,000 km Saturn target (`runs/saturn_uranus_800k_verified/`) | Feb 2073 | 8.426 years | 20.21 km/s |
| Neptune (`runs/neptune_long_verification/`) | Apr 2037 | 11.447 years | 12.51 km/s |
| Neptune, 270° B-plane clock (`runs/neptune_270_verified/`) | Apr 2037 | 11.938 years | 10.96 km/s |

## Output

- `plan.py` writes `reference_search.json` with the selected trajectory inputs, mission settings, full-gravity candidates, targeting convergence, closest distances, geometry, and failure reasons.
- `evaluate.py` writes `report.json` with the reference replay and tighter fixed-command verification. It also exports `animation.html`, `trajectory.png`, `trajectory.npz`, and `thrust_vectors.csv` under the saved reference's `evaluation/` directory by default. The CSV includes applied J2000 forces in newtons and the held RTN force referenced to 1 AU. Evaluation also exports trajectories that time out or miss a flyby.
- A successful evaluation with flybys also exports `flyby_camera.html`. It opens offline in a WebGL 2 browser. Use the encounter picker, timeline, and field-of-view controls to review each pass. `--show` opens the camera when available. The camera is centered on the simulated target throughout, uses the tighter replay with its localized periapsis, and lights the globe and rings from the simulated Sun direction. Planet spin axes and radii come from the bundled SPICE orientation kernel. The readout gives the camera's planetocentric latitude in that body-fixed frame. Surface maps, ring material, and atmosphere appearance are representative. A custom body is drawn as a sphere at its modeled safety radius. Map and ring sources are in [the asset credits](assets/flyby/CREDITS.md).

## Additional ephemerides

Use `--ephemeris bodies.json --flybys "My comet"`. The JSON contains a `bodies` list. Each body has `name`, `safety_radius_km`, `encounter_radius_km`, optional `gm_km3_s2`, and `states`. Each state row is `[ET, x, y, z, vx, vy, vz]` in heliocentric J2000, with SPICE TDB seconds past J2000, km, and km/s. The table must cover the entire launch window and flight deadline. The planner embeds these tables in the saved reference so evaluation can use the same states. A zero-GM body uses geometric gate and periapsis checks. SPICE outer-planet trajectories use system barycenters.

Run the regression tests with `python -m unittest discover -s tests`.
