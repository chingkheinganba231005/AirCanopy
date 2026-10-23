"""One time step of the 3D lattice Boltzmann method, written once for NumPy and JAX.

Every function takes the array module `xp` (numpy or jax.numpy) as its first argument, so the
same lines run in the NumPy version, slow but easy to inspect, and in the compiled JAX
version, and the tests check that the two agree to round-off. The loops over directions run in
Python: they are unrolled when JAX compiles the step, and zero coefficients are skipped.

The state is the populations f, shape (Q, nz, ny, nx): f[i] is the amount of air in each cell
moving in direction c_i. One step:

1. *Stream* (pull): each cell gathers the packets that left its neighbours towards it,
   f_i(x) = f_post_i(x - c_i). A packet that would come from a solid cell is instead the one
   that left x towards that cell and bounced back off a wall halfway between the two cells
   (halfway bounce-back): f_i(x) = f_post_opp(i)(x). The sides are periodic. The top can be
   free-slip: a frictionless lid that mirrors packets, like a mirror for light.
2. *Density and velocity*: rho = sum_i f_i and rho u = sum_i f_i c_i + F / 2, with the body
   force F = rho g (Guo, Zheng and Shi 2002).
3. *Relaxation time*: tau0 from the molecular viscosity, raised by the Smagorinsky model where
   the flow is sheared,
       tau = (tau0 + sqrt(tau0^2 + 18 sqrt(2) Cs^2 |P| / rho)) / 2,
   with P the non-equilibrium momentum flux sum_i c_i c_i (f_i - f_eq_i), corrected for the
   force. This is the exact solution of nu = nu0 + Cs^2 |S| when the strain rate is read from
   the packets themselves, S = -P / (2 rho c_s^2 tau), so no finite differences are needed
   (Hou et al. 1994).
4. *Collision*, regularised (Latt and Chopard 2006): the packets are rebuilt from the
   equilibrium and the trace-free part P_d of P alone,
       f_post_i = f_eq_i + (1 - 1/tau) w_i (9/2) c_i c_i : P_d + S_i / 2,
   with S_i the Guo forcing term. Shear stresses relax as in the plain BGK collision. The
   higher moments, which BGK lets ring when tau is close to 1/2, and the bulk stress, which
   carries sound, relax fully every step (Dellar 2001). This keeps the scheme stable at the
   tiny viscosity of air on a city-sized grid.

Solid cells keep the values streamed into them; no fluid cell ever reads them.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from .lattice import CS2, VelocitySet

SMAG = 18.0 * math.sqrt(2.0)  # 2 sqrt(2) / c_s^4, with c_s^2 = 1/3
#: Components of a symmetric 3 x 3 tensor, in the order they are stored.
PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _combine(terms):
    """sum of k * v over (k, v) pairs, for small integer k, skipping zeros and unit products."""
    out = None
    for k, v in terms:
        if k == 0:
            continue
        t = v if k == 1 else (-v if k == -1 else k * v)
        out = t if out is None else out + t
    return out


def moment(f, coef):
    """sum_i coef_i f_i for integer coefficients."""
    return _combine((int(k), f[i]) for i, k in enumerate(coef))


def _dot(c, u):
    """c . u for an integer direction c; None when c is the rest direction."""
    return _combine((int(c[a]), u[a]) for a in range(3))


def equilibrium(xp, lat: VelocitySet, rho, u):
    """Second-order equilibrium, shape (Q, *rho.shape)."""
    ux, uy, uz = u
    usq = 1.5 * (ux * ux + uy * uy + uz * uz)
    out = []
    for i in range(lat.q):
        cu = _dot(lat.c[i], u)
        if cu is None:
            out.append(float(lat.w[i]) * rho * (1.0 - usq))
        else:
            out.append(float(lat.w[i]) * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - usq))
    return xp.stack(out)


def density_velocity(xp, lat: VelocitySet, f, g=None):
    """rho and u = (sum_i f_i c_i + F / 2) / rho, for a body acceleration g (F = rho g)."""
    rho = moment(f, [1] * lat.q)
    u = [moment(f, lat.c[:, a]) / rho for a in range(3)]
    if g is not None:
        u = [u[a] + 0.5 * g[a] for a in range(3)]
    return rho, tuple(u)


def flux(xp, lat: VelocitySet, f, feq, u, force=None):
    """Non-equilibrium momentum flux sum_i c_i c_i (f_i - feq_i) + (F u + u F) / 2.

    The Guo forcing leaves -(F u + u F) / 2 in the raw flux. Adding it back makes P measure the
    strain rate alone, for the turbulence model and the collision. Returned in PAIRS order.
    """
    fneq = f - feq
    p = []
    for a, b in PAIRS:
        m = moment(fneq, lat.c[:, a] * lat.c[:, b])
        if force is not None:
            m = m + (force[a] * u[a] if a == b else 0.5 * (force[a] * u[b] + force[b] * u[a]))
        p.append(m)
    return p


def smagorinsky_tau(xp, p, rho, tau0, cs):
    """Relaxation time with the Smagorinsky model; exactly tau0 when cs is 0."""
    pxx, pyy, pzz, pxy, pxz, pyz = p
    q = xp.sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz))
    return 0.5 * (tau0 + xp.sqrt(tau0 * tau0 + SMAG * cs * cs * q / rho))


def regularised_neq(xp, lat: VelocitySet, p):
    """The non-equilibrium packets rebuilt from P: w_i (9/2) c_i c_i : P_d, shape (Q, ...).

    P_d = P - tr(P) I / 3 is the trace-free part. For the rest direction the product is 0.
    """
    pxx, pyy, pzz, pxy, pxz, pyz = p
    third = (pxx + pyy + pzz) / 3.0
    dxx, dyy, dzz = pxx - third, pyy - third, pzz - third
    out = []
    for i in range(lat.q):
        cx, cy, cz = (int(v) for v in lat.c[i])
        term = _combine(
            [
                (cx * cx, dxx),
                (cy * cy, dyy),
                (cz * cz, dzz),
                (2 * cx * cy, pxy),
                (2 * cx * cz, pxz),
                (2 * cy * cz, pyz),
            ]
        )
        out.append(0.0 * pxx if term is None else (4.5 * float(lat.w[i])) * term)
    return xp.stack(out)


def guo_source(xp, lat: VelocitySet, pre, u, force):
    """pre w_i [3 (c_i - u) + 9 (c_i . u) c_i] . F, shape (Q, ...).

    Its moments are 0, F and F u + u F times `pre`: it adds the force's momentum and the
    stress the force does work against (Guo, Zheng and Shi 2002).
    """
    out = []
    for i in range(lat.q):
        c = lat.c[i]
        cu = _dot(c, u)
        total = None
        for a in range(3):
            factor = 3.0 * (int(c[a]) - u[a])
            if cu is not None and c[a] != 0:
                factor = factor + 9.0 * int(c[a]) * cu
            term = factor * force[a]
            total = term if total is None else total + term
        out.append(pre * float(lat.w[i]) * total)
    return xp.stack(out)


def _roll(xp, a, shift):
    """Periodic shift of a (nz, ny, nx) array by shift = (sz, sy, sx) cells."""
    axes = [k for k, s in enumerate(shift) if s]
    if not axes:
        return a
    return xp.roll(a, [shift[k] for k in axes], axis=axes)


def stream(xp, lat: VelocitySet, f_post, solid, free_slip_top: bool = False):
    """Pull streaming with halfway bounce-back on solid cells; periodic elsewhere.

    With `free_slip_top`, a packet arriving at the top layer from above is the mirror image
    (cz -> -cz) of the one that left the top layer upwards from the cell it passed over. The top
    layer must then be fluid everywhere (the solver checks).
    """
    out = []
    for i in range(lat.q):
        cx, cy, cz = (int(v) for v in lat.c[i])
        g = _roll(xp, f_post[i], (cz, cy, cx))
        if free_slip_top and cz < 0:
            top = _roll(xp, f_post[lat.mirror_z[i]][-1:], (0, cy, cx))
            g = xp.concatenate([g[:-1], top], axis=0)
        wall = _roll(xp, solid, (cz, cy, cx))
        if free_slip_top and cz < 0:
            wall = xp.concatenate([wall[:-1], xp.zeros_like(wall[-1:])], axis=0)
        out.append(xp.where(wall, f_post[lat.opp[i]], g))
    return xp.stack(out)


def collide(xp, lat: VelocitySet, f, cfg, regularised: bool = True, forced: bool = False):
    """The collision of step 3 and 4 above, at every cell; solid cells keep f."""
    g = cfg["g"] if forced else None
    rho, u = density_velocity(xp, lat, f, g)
    feq = equilibrium(xp, lat, rho, u)
    force = tuple(rho * g[a] for a in range(3)) if forced else None
    p = flux(xp, lat, f, feq, u, force)
    tau = smagorinsky_tau(xp, p, rho, cfg["tau0"], cfg["cs"])
    if regularised:
        # With the force-corrected flux, the forcing enters at half weight whatever the rate
        # the stress relaxes at: m_post = m_eq + (1 - 1/tau) m_neq + F_m / 2.
        neq, pre = regularised_neq(xp, lat, p), 0.5
    else:
        neq, pre = f - feq, 1.0 - 0.5 / tau
    post = feq + (1.0 - 1.0 / tau) * neq
    if forced:
        post = post + guo_source(xp, lat, pre, u, force)
    return xp.where(cfg["fluid"], post, f)


@functools.cache
def moment_matrices(lat: VelocitySet) -> tuple[np.ndarray, np.ndarray]:
    """M (10, Q) takes the packets to the moments rho, rho u (3) and sum f c c (6, in PAIRS
    order); H (Q, 10) builds packets back from a set of those moments:

        f_i = w_i [A0 + c_i . A1 / c_s^2 + (c_i c_i - c_s^2 I) : A2 / (2 c_s^4)],

    the expansion of the packets in Hermite polynomials up to second order. The equilibrium,
    the regularised non-equilibrium part and the Guo source are all of this form.
    """
    c = lat.c.astype(float)
    rows = [np.ones(lat.q), c[:, 0], c[:, 1], c[:, 2]]
    rows += [c[:, a] * c[:, b] for a, b in PAIRS]
    m = np.array(rows)
    cols = [np.ones(lat.q), 3.0 * c[:, 0], 3.0 * c[:, 1], 3.0 * c[:, 2]]
    cols += [4.5 * (c[:, a] * c[:, b] - (CS2 if a == b else 0.0)) * (1 if a == b else 2)
             for a, b in PAIRS]  # fmt: skip
    h = lat.w[:, None] * np.array(cols).T
    return m, h


def collide_moments(xp, lat: VelocitySet, f, cfg, regularised: bool = True, forced: bool = False):
    """The same collision as `collide`, through the 10 moments: f -> M f -> H A.

    Two small matrix products per cell replace the sums over directions, which processors and
    graphics cards do much faster. The result is the same to round-off (tests/test_wind_step.py).
    """
    m_mat, h_mat = moment_matrices(lat)
    cells = f.shape[1:]
    # One flat matrix product over all cells: twice as fast as einsum on a processor.
    m = (xp.asarray(m_mat, dtype=f.dtype) @ f.reshape(lat.q, -1)).reshape(10, *cells)
    rho = m[0]
    g = cfg["g"]
    u = [m[1 + a] / rho + (0.5 * g[a] if forced else 0.0) for a in range(3)]
    force = [rho * g[a] for a in range(3)] if forced else None
    p, uu, uf = [], [], []
    for k, (a, b) in enumerate(PAIRS):
        uu.append(rho * u[a] * u[b])
        pk = m[4 + k] - uu[k] - (rho * CS2 if a == b else 0.0)
        if forced:
            uf.append(force[a] * u[b] + force[b] * u[a])
            pk = pk + 0.5 * uf[k]
        p.append(pk)
    tau = smagorinsky_tau(xp, p, rho, cfg["tau0"], cfg["cs"])
    h = xp.asarray(h_mat, dtype=f.dtype)

    def build(moments):
        return (h @ xp.stack(moments).reshape(10, -1)).reshape(lat.q, *cells)

    if regularised:
        omega = 1.0 - 1.0 / tau
        third = (p[0] + p[1] + p[2]) / 3.0
        a1 = [rho * u[a] + (0.5 * force[a] if forced else 0.0) for a in range(3)]
        a2 = []
        for k in range(6):
            pd = p[k] - third if k < 3 else p[k]
            a2.append(uu[k] + omega * pd + (0.5 * uf[k] if forced else 0.0))
        post = build([rho, *a1, *a2])
    else:
        inv, pre = 1.0 / tau, 1.0 - 0.5 / tau
        b1 = [rho * u[a] * inv + (pre * force[a] if forced else 0.0) for a in range(3)]
        b2 = [uu[k] * inv + (pre * uf[k] if forced else 0.0) for k in range(6)]
        post = (1.0 - inv) * f + build([rho * inv, *b1, *b2])
    return xp.where(cfg["fluid"], post, f)


def step(xp, lat: VelocitySet, f_post, cfg, regularised=True, free_slip_top=False, forced=False):
    """One time step: stream, then collide. Takes and returns post-collision populations."""
    f = stream(xp, lat, f_post, cfg["solid"], free_slip_top)
    return collide_moments(xp, lat, f, cfg, regularised, forced)
