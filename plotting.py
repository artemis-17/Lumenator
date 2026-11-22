import numpy as np
import plotly.graph_objects as go
import spiceypy as spice

from solar_sail import AU

from flyby import PLANETS

PLANET_COLORS = ["#A0A0A0", "#E5BD79", "#4285F4", "#DE7755",
                 "#D1A984", "#DCC894", "#77D5DA", "#648CF0"]


def save_animation(path, times, positions, launch_et, encounters=(), show=False, ephemeris=None,
                   *, escape_time=None, positions_at=None):
    """Save an interactive animation in heliocentric ecliptic axes."""
    from ephemerides import planet_positions

    rotation = spice.pxform("J2000", "ECLIPJ2000", launch_et)
    positions = np.asarray(positions) @ rotation.T / AU
    times = np.asarray(times)
    split = None if escape_time is None else int(np.searchsorted(times, escape_time))
    if split is None:
        indices = np.linspace(0, len(times) - 1, min(240, len(times)), dtype=int)
    else:
        indices = np.unique(np.r_[np.linspace(0, split, min(145, split + 1), dtype=int),
                                  np.linspace(split, len(times) - 1, min(96, len(times) - split), dtype=int)])
    if positions_at is None:
        positions_at = planet_positions if ephemeris is None else ephemeris.positions
    names = PLANETS if ephemeris is None else ephemeris.names
    colors = PLANET_COLORS + ["#e9b5ff"] * (len(names) - len(PLANETS))
    planets = np.array([positions_at(launch_et + times[i]) for i in indices]) @ rotation.T / AU

    def points(values, **kwargs):
        return go.Scatter3d(x=values[:, 0], y=values[:, 1], z=values[:, 2], **kwargs)

    fig = go.Figure()
    mission_positions = positions if split is None else positions[:split + 1]
    fig.add_trace(points(mission_positions, mode="lines", name="LUMEN trajectory",
                         line=dict(color="#22d3ee", width=4)))
    if split is not None:
        fig.add_trace(points(positions[split:], mode="lines", name="Post-escape projection",
                             line=dict(color="#c4a5ff", dash="dash", width=4)))
        fig.add_trace(points(positions[split:split + 1], mode="markers", name="Verified escape",
                             marker=dict(color="#c4a5ff", size=6, symbol="diamond")))
    fig.add_trace(points(np.zeros((1, 3)), mode="markers", name="Sun", marker=dict(color="orange", size=8)))
    for encounter in encounters:
        point = np.array([encounter.periapsis_position_m]) @ rotation.T / AU
        fig.add_trace(points(point, mode="markers", name=f"{encounter.planet} flyby",
                             marker=dict(size=5, symbol="diamond")))

    def markers(frame):
        visible = np.isfinite(planets[frame]).all(axis=1)
        return [
            points(positions[indices[frame]:indices[frame] + 1], mode="markers", name="LUMEN",
                   marker=dict(color="cyan", size=6)),
            points(planets[frame][visible], mode="markers", name="Flyby bodies",
                   text=[name for name, show_body in zip(names, visible) if show_body],
                   hovertemplate="%{text}<extra></extra>",
                   marker=dict(color=[color for color, show_body in zip(colors, visible) if show_body], size=4)),
        ]

    animated = list(range(len(fig.data), len(fig.data) + 2))
    fig.add_traces(markers(0))
    fig.frames = [go.Frame(name=str(k), data=markers(k), traces=animated) for k in range(len(indices))]
    time_unit, time_label = (86400, "day") if split is None else (365.25 * 86400, "year")
    fig.update_layout(
        title="LUMEN trajectory", template="plotly_dark",
        scene=dict(xaxis_title="Ecliptic X (AU)", yaxis_title="Ecliptic Y (AU)",
                   zaxis_title="Ecliptic Z (AU)", aspectmode="data",
                   camera=dict(eye=dict(x=1.25, y=1.25, z=1.7), projection=dict(type="orthographic"))),
        updatemenus=[dict(type="buttons", buttons=[
            dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=55, redraw=True),
                                                                  fromcurrent=True)]),
            dict(label="Pause", method="animate", args=[[None], dict(mode="immediate")]),
        ])],
        sliders=[dict(currentvalue=dict(prefix=f"Mission {time_label} "), steps=[
            dict(label=f"{times[i] / time_unit:.1f}", method="animate",
                 args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
            for k, i in enumerate(indices)
        ])],
    )
    fig.write_html(path, auto_open=show)
