"""Solar-dive guidance for the reference trajectory."""

from dataclasses import dataclass

import numpy as np

from solar_sail import AU

JUPITER_REFERENCE_EPOCH = "2035-01-09T12:14:52.079754"


@dataclass(frozen=True)
class SolarDiveGuide:
    dive_perihelion_au: float = .23844907
    burn_radius_au: float = .377453
    burn_cone_rad: float = .629835
    dive_clock_rad: float = 0.
    burn_clock_rad: float = 0.

    def __post_init__(self):
        values = (self.dive_perihelion_au, self.burn_radius_au, self.burn_cone_rad,
                  self.dive_clock_rad, self.burn_clock_rad)
        if not np.isfinite(values).all() or not 0 < self.dive_perihelion_au < self.burn_radius_au:
            raise ValueError("Invalid solar-dive reference parameters")

    def command(self, phase, sail):
        if phase == 0:
            return sail.command_at(np.arcsin(1 / np.sqrt(3)), np.pi + self.dive_clock_rad)
        if phase == 2:
            return sail.command_at(self.burn_cone_rad, self.burn_clock_rad)
        return np.array([float(phase == 3), 0., 0.])

    def outbound_phase(self, radius):
        return 2 if radius < self.burn_radius_au * AU else 3
