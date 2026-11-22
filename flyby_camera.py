"""Export planet-centered camera clips from the verified trajectory."""

import base64
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline
import spiceypy as spice

from flyby import PLANETS


ASSETS = Path(__file__).resolve().parent / "assets" / "flyby"
CAMERA_MARKER = "<!-- LUMEN flyby camera -->"

# Inner edge, outer edge, normal optical depth, and representative reflectance.
# The rings are circular render models centered on the body's equator.
RING_BANDS = {
    "Jupiter": [(89400, 123000, .000003, .015), (123000, 128940, .000005, .015),
                (128940, 181350, .0000001, .015), (181350, 221900, .0000001, .015),
                (221900, 280000, .0000001, .010)],
    "Saturn": [(66900, 74510, .00001, .18), (74658, 91975, .13, .35),
               (91975, 117507, 1.2, .68), (117507, 122340, .035, .30),
               (122340, 133410, .63, .60), (133735, 136487, .55, .57),
               (136529, 136780, .55, .57), (139796, 139856, .1, .60),
               (166000, 173000, .000001, .35), (180000, 480000, .000001, .28)],
    "Uranus": [(r - w / 2, r + w / 2, tau, albedo) for r, w, tau, albedo in
               [(41837, 1.5, .3, .015), (42234, 2, .5, .015), (42571, 2, .3, .015),
                (44718, 7, .4, .015), (45661, 8, .3, .015), (47176, 1.6, .4, .015),
                (47627, 3, .3, .015), (48300, 5, .5, .015), (50024, 2, .1, .015),
                (51149, 60, 1.1, .018)]] + [(66100, 69900, .00005, .04), (86000, 103000, .0000085, .04)],
    "Neptune": [(40900, 42900, .0001, .015), (53160, 53240, .01, .015),
                (53240, 57160, .0001, .015), (57160, 57240, .001, .015),
                (62925.5, 62940.5, .025, .015)],
}
MAIN_RING_RADIUS_KM = {"Jupiter": 128940, "Saturn": 139856, "Uranus": 51179, "Neptune": 62940.5}

ATMOSPHERES = {
    "Venus": [.030, [.95, .65, .30]],
    "Earth": [.017, [.18, .49, .98]],
    "Mars": [.019, [.89, .49, .29]],
    "Jupiter": [.013, [.63, .50, .36]],
    "Saturn": [.014, [.80, .62, .40]],
    "Uranus": [.017, [.22, .62, .82]],
    "Neptune": [.015, [.20, .43, .90]],
}


def _sample_times(encounter):
    # Keep both gate crossings and concentrate playback around periapsis.
    distance = np.linspace(0., 1., 121)**3
    peri = encounter.periapsis_time
    return np.unique(np.r_[peri - (peri - encounter.entry_time) * distance[::-1],
                           peri + (encounter.exit_time - peri) * distance])


def _camera_axes(forward):
    # Start with ecliptic north up. Transport the camera roll as the target
    # moves to avoid a flip when the sightline crosses a reference pole.
    up = spice.pxform("ECLIPJ2000", "J2000", 0.)[:, 2]
    axes = []
    for line in forward:
        right = np.cross(line, up)
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(line, np.eye(3)[np.argmin(np.abs(line))])
        right /= np.linalg.norm(right)
        up = np.cross(right, line)
        axes.append([right, up, line])
    return np.array(axes)


def _clip(flight, positions, encounter):
    name = encounter.planet
    body = flight.ephemeris.names.index(name)
    planet = name in PLANETS
    if planet:
        radii_m = 1000 * spice.bodvrd(name, "RADII", 3)[1]
        frame = f"IAU_{name.upper()}"
    else:
        radii_m = np.full(3, flight.ephemeris.limits[body])
    scale = radii_m[0]

    times = _sample_times(encounter)
    peri = int(np.searchsorted(times, encounter.periapsis_time))
    epochs = flight.launch_et + times
    body_states = np.array([flight.ephemeris.state(body, et) for et in epochs])
    craft_positions = positions(times)
    craft_positions[peri] = encounter.periapsis_position_m
    relative = craft_positions - body_states[:, :3]
    ranges = np.linalg.norm(relative, axis=1)
    speeds = np.linalg.norm(positions(times, 1) - body_states[:, 3:], axis=1)
    sun = -body_states[:, :3]
    phase = np.degrees(np.arccos(np.clip(np.sum(sun * relative, axis=1) /
                                       (np.linalg.norm(sun, axis=1) * ranges), -1., 1.)))
    axes = _camera_axes(-relative / ranges[:, None])
    frames = []
    for i, et in enumerate(epochs):
        rotation = spice.pxform("J2000", frame, et) if planet else np.eye(3)
        observer = rotation @ relative[i] / scale
        frames.append(dict(offset=float(times[i] - encounter.periapsis_time),
                           utc=spice.et2utc(et, "ISOC", 0).replace("T", " "),
                           camera=observer.tolist(),
                           sun=(rotation @ sun[i] / scale).tolist(),
                           axes=(axes[i] @ rotation.T).tolist(), inertial=axes[i].tolist(),
                           range_km=float(ranges[i] / 1000), speed_km_s=float(speeds[i] / 1000),
                           phase_deg=float(phase[i]), latitude_deg=float(np.degrees(np.arctan2(
                               observer[2], np.linalg.norm(observer[:2])))) if planet else None))

    bands = RING_BANDS.get(name, ())
    fov = 1.4 * np.degrees(2 * np.arcsin(min(1., scale / ranges[peri])))
    ring_fov = None
    if bands:
        angles = np.linspace(0., 2 * np.pi, 361)
        edge = MAIN_RING_RADIUS_KM[name] * 1000 / scale * np.c_[np.cos(angles), np.sin(angles), np.zeros_like(angles)]
        closest = frames[peri]
        projection = (edge - closest["camera"]) @ np.asarray(closest["axes"]).T
        if np.all(projection[:, 2] > 0.):
            slopes = np.abs(projection[:, :2] / projection[:, 2, None]) * [9 / 16, 1.]
            ring_fov = float(np.clip(np.degrees(2 * np.arctan(1.1 * np.max(slopes))), 4., 120.))
        else:
            ring_fov = 120.
    if name == "Saturn":
        fov = max(fov, ring_fov)
    image = ASSETS / f"{name.lower()}.jpg"
    texture = None
    if planet:
        texture = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode("ascii")
    return dict(name=name, frames=frames, periapsis=peri, radius=(radii_m / scale).tolist(),
                fov=float(np.clip(fov, 10., 115.)), ring_fov=ring_fov, texture=texture,
                sun_radius=695700000 / scale,
                surface_model=None if planet else "Sphere at the modeled safety radius",
                atmosphere=ATMOSPHERES.get(name, [0., [0., 0., 0.]]),
                rings=[[inner * 1000 / scale, outer * 1000 / scale, tau, albedo]
                       for inner, outer, tau, albedo in bands])


def save_flyby_camera(path, flight):
    """Write a self-contained browser viewer for all verified encounters."""
    times, indices = np.unique(flight.trace_t, return_index=True)
    states = np.asarray(flight.trace_x)[indices]
    positions = CubicHermiteSpline(times, states[:, :3], states[:, 3:], extrapolate=False)
    kernel = str(ASSETS / "pck00011.tpc")
    spice.furnsh(kernel)
    try:
        clips = [_clip(flight, positions, encounter) for encounter in flight.flyby_passes]
    finally:
        spice.unload(kernel)
    data = json.dumps(clips, allow_nan=False, separators=(",", ":")).replace("<", "\u003c")
    template = (ASSETS / "view.html").read_text(encoding="utf-8")
    path.write_text(template.replace("__FLYBY_DATA__", data, 1), encoding="utf-8")
