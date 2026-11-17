"""Illustrative blade geometry for a generic 10 MW-class rotor.

The geometry is ILLUSTRATIVE. It is generated from the Schmitz (optimum rotor with
wake rotation) chord and twist formulas for a chosen design tip-speed ratio, then
capped at a maximum chord. It is not the blade of the DTU 10 MW reference turbine
and not the geometry studied in the author's paper. One airfoil family is used
along the whole span and the cylindrical root section is left out: the aerodynamic
blade starts at 15 percent of the radius.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from wtsurrogate.airfoils import AirfoilPolar, get_airfoil

AIR_DENSITY = 1.225  # kg/m^3
AIR_VISCOSITY = 1.81e-5  # Pa s


@dataclass(frozen=True)
class Rotor:
    """Blade element discretisation of a rotor. Lengths in metres, angles in radians."""

    radius: float
    root_radius: float
    n_blades: int
    r: np.ndarray
    dr: float
    chord: np.ndarray
    twist: np.ndarray
    airfoil: AirfoilPolar
    design_tsr: float

    @property
    def swept_area(self) -> float:
        return float(np.pi * self.radius**2)

    @property
    def solidity(self) -> float:
        """Blade planform area (modelled span only) divided by the swept area."""
        return float(self.n_blades * np.sum(self.chord) * self.dr / self.swept_area)

    def scaled(self, chord_scale: float = 1.0, twist_scale: float = 1.0) -> Rotor:
        """Return a copy with chord and twist distributions multiplied by constants."""
        if chord_scale <= 0.0:
            raise ValueError("chord_scale must be positive")
        return replace(self, chord=self.chord * chord_scale, twist=self.twist * twist_scale)

    def at_span(self, fraction: float) -> tuple[float, float]:
        """Chord (m) and twist (rad) interpolated at ``fraction`` of the radius."""
        position = fraction * self.radius
        return (
            float(np.interp(position, self.r, self.chord)),
            float(np.interp(position, self.r, self.twist)),
        )


def reference_rotor(
    airfoil: str = "naca4415",
    n_elements: int = 30,
    radius: float = 89.0,
    root_fraction: float = 0.15,
    n_blades: int = 3,
    design_tsr: float = 7.5,
    max_chord: float = 6.2,
) -> Rotor:
    """Build the illustrative reference rotor for one airfoil family.

    Schmitz design at local speed ratio ``lambda_r = design_tsr * r / R``:
    ``phi = (2 / 3) * atan(1 / lambda_r)``,
    ``chord = 8 * pi * r * (1 - cos(phi)) / (B * cl_design)`` and
    ``twist = phi - alpha_design``, where the design point of the airfoil is its
    best glide ratio.
    """
    if n_elements < 5:
        raise ValueError("need at least 5 blade elements")
    polar = get_airfoil(airfoil)
    root = root_fraction * radius
    dr = (radius - root) / n_elements
    r = root + dr * (np.arange(n_elements) + 0.5)
    alpha_design, cl_design, _ = polar.best_glide()
    lam_r = design_tsr * r / radius
    phi = (2.0 / 3.0) * np.arctan(1.0 / lam_r)
    chord = 8.0 * np.pi * r * (1.0 - np.cos(phi)) / (n_blades * cl_design)
    chord = np.minimum(chord, max_chord)
    twist = phi - alpha_design
    return Rotor(radius, root, n_blades, r, dr, chord, twist, polar, design_tsr)
