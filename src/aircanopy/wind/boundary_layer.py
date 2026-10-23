"""A neutral atmospheric boundary layer over rough ground: the test of the wall model.

Air in a box, periodic in both horizontal directions, over a rough floor (walls.py) and under a
free-slip lid at height H, pushed along x by a constant pressure gradient, written as a body
acceleration g. Once the flow is steady on average, the floor must hold back the whole push on
the column, rho u*^2 = rho g H, so the friction velocity is known exactly, u* = sqrt(g H), and
the law of the wall predicts the mean wind near the floor with no free parameter:

    U(z) = (u* / kappa) ln(z / z0).

Heights z are measured from the floor plane, half a cell below the centre of the first layer
of air. The air starts on that profile with random gusts added, which grow into turbulence.
Time is counted in eddy turnovers, H / u*. After `spin_up` turnovers, the wind is sampled for
`average` more and averaged over time and over each horizontal layer.

Momentum in the box changes only by the push and the floor's friction, so the friction can
differ from the push only while the column speeds up or slows down. The mean wind of the whole
column is therefore recorded once per turnover during the spin-up and at every sample, to show
whether the run has settled (JOURNAL.md, Step 11). Two measures of noise at the scale of the
grid are recorded for each layer (Step 12): the share of the vertical wind's variance carried
by waves shorter than 4 cells across, and the correlation of the vertical wind between
neighbouring cells along x, which is near 1 for smooth turbulence and negative for noise that
flips sign from one cell to the next.
"""

from __future__ import annotations

import math

import numpy as np

from . import walls
from .lattice import VelocitySet
from .solver import Params, Solver


def run(
    lat: VelocitySet,
    nx: int,
    ny: int,
    h: int,
    z0: float = 0.05,
    u_star: float = 0.005,
    smagorinsky: float = 0.1,
    collision: str = "recursive",
    match_cells: int = 0,
    tau0: float = 0.500003,
    spin_up: float = 10.0,
    average: float = 10.0,
    samples_per_turnover: int = 10,
    gusts: float = 0.3,
    seed: int = 0,
    backend: str = "jax",
    dtype=np.float32,
) -> dict:
    """Run the boundary layer; return the mean profiles and the balance of forces.

    All lengths are in cells and speeds in cells per step. tau0 sets the molecular viscosity,
    (tau0 - 1/2) / 3: the default 1e-6 is negligible, as for real air on metre-sized cells.
    `gusts` is the size of the random start-up gusts, as a share of the local wind. With
    `match_cells` = k > 0, the wall model reads the wind k cells above the first cell
    (walls.matched_floor_drag), at the height k + 1/2.
    """
    solid = np.zeros((h + 1, ny, nx), bool)
    solid[0] = True
    g = u_star**2 / h
    params = Params(tau0=tau0, smagorinsky=smagorinsky, g=(g, 0.0, 0.0), free_slip_top=True,
                    slip_floors=True, collision=collision, match_cells=match_cells)  # fmt: skip
    if match_cells:
        _, drag = walls.matched_floor_drag(solid, z0, match_cells)
        s = Solver(lat, solid, params, backend, dtype, match_drag=drag)
    else:
        drag = walls.rough_floor_drag(solid, z0)
        s = Solver(lat, solid, params, backend, dtype, drag=drag)
    read = 1 + match_cells  # the layer whose wind the wall model reads
    z = np.arange(h + 1) - 0.5  # height of each layer's centre above the floor plane
    z[0] = np.nan
    log_law = u_star / walls.KAPPA * np.log(z / z0)
    rng = np.random.default_rng(seed)
    shape = solid.shape
    profile = np.broadcast_to(np.nan_to_num(log_law)[:, None, None], shape)
    noise = [gusts * profile * rng.standard_normal(shape) for _ in range(3)]
    s.set_equilibrium(np.ones(shape), (profile + noise[0], noise[1], noise[2]))
    turnover = h / u_star  # steps
    spin_steps, spin_up_column_u = round(spin_up * turnover), []
    while s.time < spin_steps:
        s.run(min(round(turnover), spin_steps - s.time))
        spin_up_column_u.append(float(s.fields()[1][0][1:].mean()))
    every = max(1, round(turnover / samples_per_turnover))
    samples = round(average * samples_per_turnover)
    mean_u = np.zeros(h + 1)
    mean_uw = np.zeros(h + 1)
    mean_uu = np.zeros(h + 1)
    mean_ww = np.zeros(h + 1)
    floor_stress = 0.0
    lam = 2 * walls.drag_coefficient(z0, height=read - 0.5)
    column_u = [spin_up_column_u[-1]] if spin_up_column_u else []
    # Noise at the scale of the grid in the vertical wind, layer by layer (module docstring).
    kx, ky = (np.abs(np.fft.fftfreq(m) * m) for m in (nx, ny))
    short = (kx[None, :] > nx / 4) | (ky[:, None] > ny / 4)
    w_short, w_all, w_lag, w_sq = (np.zeros(h) for _ in range(4))
    floor_push, floor_u = 0.0, 0.0
    for _ in range(samples):
        s.run(every)
        rho, (ux, uy, uz) = s.fields()
        layer_u = ux.mean(axis=(1, 2))
        du = ux - layer_u[:, None, None]
        mean_u += layer_u
        mean_uw += (du * uz).mean(axis=(1, 2))
        mean_uu += (du * du).mean(axis=(1, 2))
        mean_ww += (uz * uz).mean(axis=(1, 2))
        # The floor's friction acts on the first layer, driven by the wind of the layer read.
        speed = np.sqrt(ux[read] ** 2 + uy[read] ** 2 + uz[read] ** 2)
        floor_stress += float((0.5 * lam * rho[1] * speed * ux[read]).mean())
        floor_push += float((speed * ux[read]).mean())
        floor_u += float(ux[read].mean())
        column_u.append(float(ux[1:].mean()))
        w = uz[1:]
        spectrum = np.abs(np.fft.fft2(w)) ** 2
        w_short += (spectrum * short).sum(axis=(1, 2))
        w_all += spectrum.sum(axis=(1, 2))
        w_lag += (w * np.roll(w, 1, axis=2)).sum(axis=(1, 2))
        w_sq += (w * w).sum(axis=(1, 2))
    mean_u, mean_uw = mean_u / samples, mean_uw / samples
    mean_uu, mean_ww = mean_uu / samples, mean_ww / samples
    floor_stress /= samples
    # phi = (kappa z / u*) dU/dz between layer centres: 1 on the law of the wall.
    z_mid = 0.5 * (z[1:-1] + z[2:])
    phi = walls.KAPPA * z_mid / u_star * np.diff(mean_u[1:])
    return {
        "nx": nx,
        "ny": ny,
        "h": h,
        "z0": z0,
        "u_star": u_star,
        "g": g,
        "smagorinsky": smagorinsky,
        "collision": collision,
        "match_cells": match_cells,
        "wall_model_height": read - 0.5,
        "tau0": tau0,
        "lattice": lat.name,
        "steps": s.time,
        "turnover_steps": turnover,
        "z": [float(v) for v in z[1:]],
        "mean_u": [float(v) for v in mean_u[1:]],
        "log_law": [float(v) for v in log_law[1:]],
        "resolved_uw": [float(v) for v in mean_uw[1:]],
        "variance_u": [float(v) for v in mean_uu[1:]],
        "variance_w": [float(v) for v in mean_ww[1:]],
        "z_phi": [float(v) for v in z_mid],
        "phi": [float(v) for v in phi],
        "floor_stress_over_push": floor_stress / (g * h),
        "finite": bool(np.isfinite(mean_u).all()),
        "max_speed": float(np.max(mean_u[1:])),
        "drag_lambda": lam,
        "mach_of_fastest_mean": float(np.max(mean_u[1:]) / math.sqrt(1 / 3)),
        # The column's mean wind over u*, once per turnover of the spin-up, then at each sample.
        "spin_up_column_u": [v / u_star for v in spin_up_column_u],
        "column_u": [v / u_star for v in column_u],
        # In the layer the wall model reads, mean(|U| u) / mean(u)^2: how much the gusts raise
        # the friction above what the mean wind alone would give.
        "floor_gust_factor": (floor_push / samples) / (floor_u / samples) ** 2,
        "w_grid_scale_share": [float(v) for v in w_short / w_all],
        "w_neighbour_correlation_x": [float(v) for v in w_lag / w_sq],
    }
