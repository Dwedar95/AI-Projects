"""Steady Blade Element Momentum (BEM) solver, vectorised over operating points.

The solver follows the residual formulation of Ning (2014, "A simple solution
method for the blade element momentum equations with guaranteed convergence",
Wind Energy 17). For every blade element one scalar equation in the inflow angle
``phi`` is solved by bracketing and bisection, so there is no fixed-point iteration
that can diverge. Included corrections:

* Prandtl tip loss and root (hub) loss
* Buhl's empirical correction for heavily loaded elements (axial induction > 0.4)
* drag in the induction equations

Not included: yaw, shear, tower shadow, unsteady or 3-D rotational effects,
structural deflection. The Reynolds number of each element is estimated from the
relative speed without induction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wtsurrogate.airfoils import AirfoilPolar
from wtsurrogate.rotor import AIR_DENSITY, AIR_VISCOSITY, Rotor

_EPS = 1.0e-6


@dataclass(frozen=True)
class BemResult:
    """Rotor quantities for a batch of ``n`` operating points.

    Integrated quantities have shape ``(n,)``; spanwise quantities (``phi``, the
    inductions, ``alpha`` and the loads per unit span of one blade in N/m) have
    shape ``(n, n_elements)``.
    """

    cp: np.ndarray
    ct: np.ndarray
    power: np.ndarray
    thrust: np.ndarray
    torque: np.ndarray
    flap_moment: np.ndarray
    converged: np.ndarray
    phi: np.ndarray
    axial_induction: np.ndarray
    tangential_induction: np.ndarray
    alpha: np.ndarray
    normal_load: np.ndarray
    tangential_load: np.ndarray


def prandtl_loss(
    phi: np.ndarray, r: np.ndarray, radius: float, root_radius: float, n_blades: int
) -> np.ndarray:
    """Combined Prandtl tip and root loss factor ``F`` in (0, 1]."""
    sin_phi = np.maximum(np.abs(np.sin(phi)), 1.0e-9)
    f_tip = 0.5 * n_blades * (radius - r) / (r * sin_phi)
    f_root = 0.5 * n_blades * (r - root_radius) / (root_radius * sin_phi)
    tip = 2.0 / np.pi * np.arccos(np.clip(np.exp(-f_tip), 0.0, 1.0))
    root = 2.0 / np.pi * np.arccos(np.clip(np.exp(-f_root), 0.0, 1.0))
    return np.maximum(tip * root, 1.0e-6)


def _element_state(
    phi: np.ndarray,
    theta: np.ndarray,
    sigma: np.ndarray,
    lam_r: np.ndarray,
    reynolds: np.ndarray,
    r: np.ndarray,
    polar: AirfoilPolar,
    radius: float,
    root_radius: float,
    n_blades: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Residual, inductions and force coefficients for trial inflow angles ``phi``."""
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    cl, cd = polar.coefficients(phi - theta, reynolds)
    cn = cl * cos_phi + cd * sin_phi
    ct = cl * sin_phi - cd * cos_phi
    loss = prandtl_loss(phi, r, radius, root_radius, n_blades)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = sigma * cn / (4.0 * loss * sin_phi**2)
        kp = sigma * ct / (4.0 * loss * sin_phi * cos_phi)
        # momentum and empirical (Buhl) region for phi > 0
        a_mom = k / (1.0 + k)
        g1 = 2.0 * loss * k - (10.0 / 9.0 - loss)
        g2 = np.maximum(2.0 * loss * k - loss * (4.0 / 3.0 - loss), 0.0)
        g3 = 2.0 * loss * k - (25.0 / 9.0 - 2.0 * loss)
        a_buhl = np.where(
            np.abs(g3) < 1.0e-6, 1.0 - 0.5 / np.sqrt(np.maximum(g2, 1e-12)),
            (g1 - np.sqrt(g2)) / np.where(np.abs(g3) < 1.0e-6, 1.0, g3),
        )  # fmt: skip
        a_pos = np.where(k <= 2.0 / 3.0, a_mom, a_buhl)
        # propeller brake region for phi < 0
        a_neg = np.where(k > 1.0, k / (k - 1.0), 0.0)
        a = np.where(phi > 0.0, a_pos, a_neg)
        ap = kp / (1.0 - kp)
        res_pos = sin_phi / (1.0 - a) - cos_phi / lam_r * (1.0 - kp)
        res_neg = sin_phi * (1.0 - k) - cos_phi / lam_r * (1.0 - kp)
        residual = np.where(phi > 0.0, res_pos, res_neg)
    return residual, a, ap, cn, ct


def solve_bem(
    rotor: Rotor,
    wind_speed: np.ndarray | float,
    tsr: np.ndarray | float,
    pitch: np.ndarray | float = 0.0,
    chord_scale: np.ndarray | float = 1.0,
    twist_scale: np.ndarray | float = 1.0,
    density: float = AIR_DENSITY,
    viscosity: float = AIR_VISCOSITY,
    n_bisect: int = 48,
) -> BemResult:
    """Solve the steady BEM equations for a batch of operating points.

    Parameters broadcast against each other to a common batch size ``n``.

    Parameters
    ----------
    wind_speed:
        Free-stream wind speed in m/s.
    tsr:
        Tip-speed ratio ``omega * R / U``.
    pitch:
        Collective blade pitch in radians (positive pitches to feather).
    chord_scale, twist_scale:
        Multipliers applied to the chord and twist distributions of ``rotor``.
    """
    u, lam, beta, cs, ts = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(v, dtype=float))
          for v in (wind_speed, tsr, pitch, chord_scale, twist_scale))
    )  # fmt: skip
    if np.any(u <= 0.0) or np.any(lam <= 0.0) or np.any(cs <= 0.0):
        raise ValueError("wind_speed, tsr and chord_scale must be positive")
    n = u.size
    r = rotor.r[None, :]
    chord = rotor.chord[None, :] * cs[:, None]
    theta = rotor.twist[None, :] * ts[:, None] + beta[:, None]
    omega = lam * u / rotor.radius
    lam_r = lam[:, None] * r / rotor.radius
    sigma = rotor.n_blades * chord / (2.0 * np.pi * r)
    w_free = u[:, None] * np.sqrt(1.0 + lam_r**2)
    reynolds = density * w_free * chord / viscosity
    shape = (n, rotor.r.size)
    r_full = np.broadcast_to(r, shape)

    def state(phi: np.ndarray) -> tuple[np.ndarray, ...]:
        return _element_state(phi, theta, sigma, lam_r, reynolds, r_full, rotor.airfoil,
                              rotor.radius, rotor.root_radius, rotor.n_blades)  # fmt: skip

    def sign_change(lo: float, hi: float) -> np.ndarray:
        res_lo = state(np.full(shape, lo))[0]
        res_hi = state(np.full(shape, hi))[0]
        return np.isfinite(res_lo) & np.isfinite(res_hi) & (res_lo * res_hi < 0.0)

    # bracket selection in the order proposed by Ning (2014)
    lo = np.full(shape, _EPS)
    hi = np.full(shape, 0.5 * np.pi)
    found = sign_change(_EPS, 0.5 * np.pi)
    brake = ~found & sign_change(-0.25 * np.pi, -_EPS)
    lo[brake], hi[brake] = -0.25 * np.pi, -_EPS
    found |= brake
    reverse = ~found & sign_change(0.5 * np.pi, np.pi - _EPS)
    lo[reverse], hi[reverse] = 0.5 * np.pi, np.pi - _EPS
    found |= reverse

    res_lo = state(lo)[0]
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        res_mid = state(mid)[0]
        same = np.sign(res_mid) == np.sign(res_lo)
        lo = np.where(same, mid, lo)
        res_lo = np.where(same, res_mid, res_lo)
        hi = np.where(same, hi, mid)
    phi = 0.5 * (lo + hi)
    residual, a, ap, cn, ct = state(phi)
    ok = found & np.isfinite(residual) & (np.abs(residual) < 1.0e-5)

    w2 = (u[:, None] * (1.0 - a)) ** 2 + (omega[:, None] * r * (1.0 + ap)) ** 2
    q = 0.5 * density * w2 * chord
    normal = np.where(ok, cn * q, 0.0)
    tangential = np.where(ok, ct * q, 0.0)
    thrust = rotor.n_blades * normal.sum(axis=1) * rotor.dr
    torque = rotor.n_blades * (tangential * r).sum(axis=1) * rotor.dr
    flap = (normal * (r - rotor.root_radius)).sum(axis=1) * rotor.dr
    power = torque * omega
    q_inf = 0.5 * density * rotor.swept_area * u**2
    return BemResult(
        cp=power / (q_inf * u),
        ct=thrust / q_inf,
        power=power,
        thrust=thrust,
        torque=torque,
        flap_moment=flap,
        converged=ok.all(axis=1),
        phi=phi,
        axial_induction=a,
        tangential_induction=ap,
        alpha=phi - theta,
        normal_load=normal,
        tangential_load=tangential,
    )
