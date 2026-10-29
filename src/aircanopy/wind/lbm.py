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
4. *Collision*, one of three (COLLISIONS):
   - "recursive" (the default; D3Q27 only): recursive regularisation with the extended
     equilibrium (Malaspinas 2015; Coreixas et al. 2017, 2019). On D3Q27 the equilibrium is a
     product of one factor per axis, f_eq_i = rho phi(c_ix, ux) phi(c_iy, uy) phi(c_iz, uz),
     with phi(0, u) = 2/3 - u^2 and phi(+-1, u) = (1/3 +- u + u^2) / 2; it has the right
     moments up to third order, where the usual second-order equilibrium does not. The
     non-equilibrium part is rebuilt from P alone, at every order the lattice holds, with the
     recursive formulas of Coreixas et al. (2019, Eq. 89). In product form this is
         f_neq_i = sum_a P_aa (phi''_a / 2) phi_b phi_c + sum_{a<b} P_ab phi'_a phi'_b phi_c,
     with ' = d/du, and f_post = f_eq + (1 - 1/tau) f_neq + F . d f_eq / d(rho u) / 2.
     In a uniform wind at Mach 0.17 with almost no viscosity, noise dies away (JOURNAL.md,
     Step 12).
   - "regularised" (Latt and Chopard 2006): the second-order equilibrium, and the packets
     rebuilt from it and the trace-free part P_d of P alone,
         f_post_i = f_eq_i + (1 - 1/tau) w_i (9/2) c_i c_i : P_d + S_i / 2,
     with S_i the Guo forcing term; the bulk stress, which carries sound, relaxes fully every
     step (Dellar 2001). Step 12 found it unstable in a fast wind at almost no viscosity:
     noise one cell long grows until the run fails. Kept for comparison, and for D3Q19.
   - "bgk": plain BGK with the second-order equilibrium, for comparison.

Solid cells keep the values streamed into them; no fluid cell ever reads them.
"""

from __future__ import annotations

import functools
import itertools
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


COLLISIONS = ("recursive", "regularised", "bgk")


def check_collision(lat: VelocitySet, collision: str) -> None:
    if collision not in COLLISIONS:
        raise ValueError(f"unknown collision {collision!r}; choose one of {COLLISIONS}")
    if collision == "recursive" and lat.name != "D3Q27":
        raise ValueError("the recursive collision needs D3Q27, whose equilibrium is a product "
                         "of one factor per axis")  # fmt: skip


def _axis_factors(u):
    """phi(c, u), phi'(c, u) and phi''(c, u) / 2 for c = -1, 0, 1 (module docstring)."""
    usq = u * u
    phi = {0: 2.0 / 3.0 - usq, 1: 0.5 * (1.0 / 3.0 + u + usq), -1: 0.5 * (1.0 / 3.0 - u + usq)}
    d1 = {0: -2.0 * u, 1: 0.5 + u, -1: u - 0.5}
    half_d2 = {0: -1.0, 1: 0.5, -1: 0.5}
    return phi, d1, half_d2


def product_terms(xp, lat: VelocitySet, rho, u, p=None, force=None):
    """The extended equilibrium rho phi_x phi_y phi_z, and, if given, the non-equilibrium part
    rebuilt from the flux p (PAIRS order) and the force term F . grad_u(phi_x phi_y phi_z), each
    of shape (Q, ...). D3Q27 only; None for the parts not asked for.

    The nine directions that share (c_y, c_z) share most of the work: each part is a sum of
    phi_x, phi'_x and phi''_x / 2, each times a factor that depends on y and z only, computed
    once for the nine."""
    fx, fy, fz = (_axis_factors(u[a]) for a in range(3))
    eq, neq, src = {}, {}, {}
    for cy, cz in itertools.product((-1, 0, 1), repeat=2):
        py, pz = fy[0][cy], fz[0][cz]
        dy, dz = fy[1][cy], fz[1][cz]
        yz = py * pz
        rho_yz = rho * yz
        if p is not None:
            # f_neq = phi_x (...) + phi'_x (...) + (phi''_x / 2) (...), from the module docstring.
            times_phi = fy[2][cy] * p[1] * pz + fz[2][cz] * p[2] * py + p[5] * dy * dz
            times_d1 = p[3] * dy * pz + p[4] * py * dz
            times_half_d2 = p[0] * yz
        if force is not None:
            src_phi = force[1] * dy * pz + force[2] * py * dz
            src_d1 = force[0] * yz
        for cx in (-1, 0, 1):
            px, dx = fx[0][cx], fx[1][cx]
            eq[cx, cy, cz] = px * rho_yz
            if p is not None:
                neq[cx, cy, cz] = px * times_phi + dx * times_d1 + fx[2][cx] * times_half_d2
            if force is not None:
                src[cx, cy, cz] = px * src_phi + dx * src_d1
    order = [(int(a), int(b), int(c)) for a, b, c in lat.c]
    return (xp.stack([eq[k] for k in order]),
            xp.stack([neq[k] for k in order]) if p is not None else None,
            xp.stack([src[k] for k in order]) if force is not None else None)  # fmt: skip


@functools.cache
def hermite_basis(lat: VelocitySet) -> tuple[tuple[tuple[int, int, int], ...], np.ndarray]:
    """The 27 tensor-product Hermite polynomials of D3Q27, H_pqr = H_p(cx) H_q(cy) H_r(cz) with
    H_0 = 1, H_1 = c, H_2 = c^2 - c_s^2, and the matrix (Q, 27) that builds packets from
    Hermite coefficients a_pqr: f_i = w_i sum a_pqr H_pqr(c_i) / (N_p N_q N_r), N = 1, c_s^2,
    2 c_s^4. Used by the reference form of the recursive collision (`collide`)."""
    check_collision(lat, "recursive")
    orders = tuple(itertools.product(range(3), repeat=3))
    norm = (1.0, CS2, 2.0 * CS2 * CS2)
    c = lat.c.astype(float)

    def h1(n, v):
        return (1.0, v, v * v - CS2)[n]

    build = np.array([[lat.w[i] * np.prod([h1(o[k], c[i, k]) / norm[o[k]] for k in range(3)])
                       for o in orders] for i in range(lat.q)])  # fmt: skip
    return orders, build


def _recursive_coefficient(order, u, p):
    """a(1)_pqr from the second-order a(1) = p: the sum, over each pair of the order's indices,
    of p for that pair times u for each remaining index (Coreixas et al. 2019, Eq. 89)."""
    index = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5}
    axes = [0] * order[0] + [1] * order[1] + [2] * order[2]
    total = None
    for i, j in itertools.combinations(range(len(axes)), 2):
        term = p[index[tuple(sorted((axes[i], axes[j])))]]
        for k, a in enumerate(axes):
            if k not in (i, j):
                term = term * u[a]
        total = term if total is None else total + term
    return total


def _roll(xp, a, shift):
    """Periodic shift of a (nz, ny, nx) array by shift = (sz, sy, sx) cells."""
    axes = [k for k, s in enumerate(shift) if s]
    if not axes:
        return a
    return xp.roll(a, [shift[k] for k in axes], axis=axes)


def stream(
    xp, lat: VelocitySet, f_post, solid, free_slip_top: bool = False, slip_floors: bool = False
):
    """Pull streaming with halfway bounce-back on solid cells; periodic elsewhere.

    With `free_slip_top`, a packet arriving at the top layer from above is the mirror image
    (cz -> -cz) of the one that left the top layer upwards from the cell it passed over. The top
    layer must then be fluid everywhere (the solver checks).

    With `slip_floors`, floors are free-slip: a packet arriving from below at an air cell that
    stands on a solid cell is the mirror image of the one that left the air cell beside it
    downwards, as if reflected off the floor halfway down. Only on a flat stretch of floor: where
    that neighbour is solid too (a wall, an edge), the packet bounces back as before. The
    friction of a rough floor is then added as a drag in the air just above it (walls.py).
    """
    out = []
    below = _roll(xp, solid, (1, 0, 0)) if slip_floors else None
    for i in range(lat.q):
        cx, cy, cz = (int(v) for v in lat.c[i])
        g = _roll(xp, f_post[i], (cz, cy, cx))
        if free_slip_top and cz < 0:
            top = _roll(xp, f_post[lat.mirror_z[i]][-1:], (0, cy, cx))
            g = xp.concatenate([g[:-1], top], axis=0)
        wall = _roll(xp, solid, (cz, cy, cx))
        if free_slip_top and cz < 0:
            wall = xp.concatenate([wall[:-1], xp.zeros_like(wall[-1:])], axis=0)
        bounced = f_post[lat.opp[i]]
        if slip_floors and cz > 0:
            beside = _roll(xp, solid, (0, cy, cx))
            mirrored = _roll(xp, f_post[lat.mirror_z[i]], (0, cy, cx))
            bounced = xp.where(below & ~beside, mirrored, bounced)
        out.append(xp.where(wall, bounced, g))
    return xp.stack(out)


def velocity_and_force(xp, rho, momentum, cfg, forced: bool, dragged: bool, matched: int = 0):
    """The velocity u = (momentum + F / 2) / rho and the force F on each cell (None if none).

    F is rho g for a body acceleration g (`forced`), plus a drag -(lambda / 2) rho |u| u
    (`dragged`): the friction of a rough floor (walls.py) or, later, of leaves. The drag depends
    on the velocity it changes. Writing u0 = momentum / rho + g / 2, the velocity is u = k u0
    with k = 2 / (1 + sqrt(1 + lambda |u0|)), the exact solution of u = u0 - (lambda / 4) |u| u,
    which stays bounded however large lambda is (as in the 2D solver; Guo and Zhao 2002).

    With `matched` = k > 0, floor cells also feel -(lambda_m / 2) rho |u_k| u_k, where u_k is
    u0 of the cell k above and lambda_m = cfg["match_drag"] (walls.matched_floor_drag). It does
    not depend on the cell's own velocity, so it is applied as it is.
    """
    g = cfg["g"]
    u = [momentum[a] / rho + (0.5 * g[a] if forced else 0.0) for a in range(3)]
    if not (forced or dragged or matched):
        return u, None
    force = [rho * g[a] if forced else 0.0 * rho for a in range(3)]
    if matched:
        up = [_roll(xp, c, (-matched, 0, 0)) for c in u]
        speed_up = xp.sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2])
        pull = 0.5 * cfg["match_drag"] * rho * speed_up
        force = [force[a] - pull * up[a] for a in range(3)]
        u = [u[a] - 0.5 * pull * up[a] / rho for a in range(3)]
    if dragged:
        lam = cfg["drag"]
        speed0 = xp.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
        k = 2.0 / (1.0 + xp.sqrt(1.0 + lam * speed0))
        u = [k * c for c in u]
        pull = 0.5 * lam * rho * k * speed0
        force = [force[a] - pull * u[a] for a in range(3)]
    return u, force


def _force_coefficient(order, u, force):
    """Hermite coefficient of F . grad_u(u_x^p u_y^q u_z^r): the force term of `collide`."""
    total = None
    for k in range(3):
        if order[k] == 0:
            continue
        term = order[k] * force[k]
        for a in range(3):
            for _ in range(order[a] - (1 if a == k else 0)):
                term = term * u[a]
        total = term if total is None else total + term
    return total


def collide(
    xp,
    lat: VelocitySet,
    f,
    cfg,
    collision: str = "recursive",
    forced: bool = False,
    dragged: bool = False,
    matched: int = 0,
):
    """The collision of step 3 and 4 above, at every cell; solid cells keep f.

    Written for checking, not speed: the recursive collision goes through the 27 Hermite
    coefficients one by one, as Coreixas et al. (2019) write it, which makes it an independent
    check of the product form in `collide_moments`.
    """
    check_collision(lat, collision)
    rho = moment(f, [1] * lat.q)
    momentum = [moment(f, lat.c[:, a]) for a in range(3)]
    u, force = velocity_and_force(xp, rho, momentum, cfg, forced, dragged, matched)
    forced = force is not None
    # The second-order equilibrium and the extended one share rho, rho u and the momentum flux
    # rho u u + rho c_s^2 I, so P is the same for both.
    feq = equilibrium(xp, lat, rho, u)
    p = flux(xp, lat, f, feq, u, force)
    tau = smagorinsky_tau(xp, p, rho, cfg["tau0"], cfg["cs"])
    if collision == "recursive":
        orders, build = hermite_basis(lat)
        omega = 1.0 - 1.0 / tau
        coeffs = []
        for o in orders:
            a = rho * u[0] ** o[0] * u[1] ** o[1] * u[2] ** o[2]
            if sum(o) >= 2:
                a = a + omega * _recursive_coefficient(o, u, p)
            if forced and sum(o) >= 1:
                a = a + 0.5 * _force_coefficient(o, u, force)
            coeffs.append(a)
        cells = f.shape[1:]
        post = (xp.asarray(build) @ xp.stack(coeffs).reshape(27, -1)).reshape(lat.q, *cells)
        return xp.where(cfg["fluid"], post, f)
    if collision == "regularised":
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


def collide_moments(
    xp,
    lat: VelocitySet,
    f,
    cfg,
    collision: str = "recursive",
    forced: bool = False,
    dragged: bool = False,
    matched: int = 0,
):
    """The same collision as `collide`, through the 10 moments: f -> M f -> H A.

    Two small matrix products per cell replace the sums over directions, which processors and
    graphics cards do much faster. The recursive collision is built in its product form
    instead of the second matrix product. The result is the same to round-off
    (tests/test_wind_step.py).
    """
    m_mat, h_mat = moment_matrices(lat)
    cells = f.shape[1:]
    # One flat matrix product over all cells: twice as fast as einsum on a processor.
    m = (xp.asarray(m_mat, dtype=f.dtype) @ f.reshape(lat.q, -1)).reshape(10, *cells)
    rho = m[0]
    u, force = velocity_and_force(xp, rho, [m[1], m[2], m[3]], cfg, forced, dragged, matched)
    forced = force is not None
    p, uu, uf = [], [], []
    for k, (a, b) in enumerate(PAIRS):
        uu.append(rho * u[a] * u[b])
        pk = m[4 + k] - uu[k] - (rho * CS2 if a == b else 0.0)
        if forced:
            uf.append(force[a] * u[b] + force[b] * u[a])
            pk = pk + 0.5 * uf[k]
        p.append(pk)
    tau = smagorinsky_tau(xp, p, rho, cfg["tau0"], cfg["cs"])
    if collision == "recursive":
        eq, neq, src = product_terms(xp, lat, rho, u, p, force if forced else None)
        post = eq + (1.0 - 1.0 / tau) * neq
        if forced:
            post = post + 0.5 * src
        return xp.where(cfg["fluid"], post, f)
    h = xp.asarray(h_mat, dtype=f.dtype)

    def build(moments):
        return (h @ xp.stack(moments).reshape(10, -1)).reshape(lat.q, *cells)

    if collision == "regularised":
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


def _rebuild(xp, lat: VelocitySet, collision: str, rho, u, p):
    """Packets from a density, a velocity and a non-equilibrium flux, built as the collision
    builds them: the extended equilibrium and the recursive part for "recursive", the
    second-order equilibrium and the regularised part otherwise."""
    if collision == "recursive":
        eq, neq, _ = product_terms(xp, lat, rho, u, p)
        return eq + neq
    return equilibrium(xp, lat, rho, u) + regularised_neq(xp, lat, p)


def _open_column(xp, lat: VelocitySet, col, collision: str, side: int, g, rho=None, u=None):
    """Rebuild the packets of a boundary column (Q, nz, ny) after streaming.

    side = -1 is an inlet at x = 0, with the velocity u given; side = +1 an outlet at the last
    x, with the density rho given and the velocity u given only across the flow (its x part
    is replaced). The packets moving into the box from outside (c_x = -side) did not come from
    the air and are unknown. Mass and momentum along x fix the missing one of rho and u_x from
    the known packets (Zou and He 1997):
        inlet:  rho (1 - u_x + g_x / 2) = S_0 + 2 S_-,   outlet:  u_x = (S_0 + 2 S_+) / rho - 1
    + g_x / 2, with S_0, S_-, S_+ the sums of the packets with c_x = 0, -1, +1. Each unknown packet
    is then the equilibrium plus the non-equilibrium part of the packet opposite to it (their
    bounce-back of the non-equilibrium part), and the column is rebuilt from rho, u and the
    flux of that set (as in the regularised boundary of Latt et al. 2008), so the collision
    that follows sees a consistent state. The equilibrium is taken at u - g / 2, the velocity
    the packets carry before the force's half step.

    The equilibrium must be the collision's own. The extended one differs from the
    second-order one in moments of third order and up, such as u_x^2 u_y, which are odd in
    c_x: bouncing back the difference would turn it into a false stress wherever the wind
    crosses the boundary at an angle.
    """
    cx = lat.c[:, 0]
    s0 = _combine((1, col[i]) for i in range(lat.q) if cx[i] == 0)
    s_in = _combine((1, col[i]) for i in range(lat.q) if cx[i] == side)  # moving out, known
    if side < 0:
        rho = (s0 + 2.0 * s_in) / (1.0 - u[0] + 0.5 * g[0])
    else:
        u = [(s0 + 2.0 * s_in) / rho - 1.0 + 0.5 * g[0], u[1], u[2]]
    u_pre = [u[a] - 0.5 * g[a] for a in range(3)]
    if collision == "recursive":
        feq = product_terms(xp, lat, rho, u_pre)[0]
    else:
        feq = equilibrium(xp, lat, rho, u_pre)
    full = [col[i] if cx[i] != -side else feq[i] + col[lat.opp[i]] - feq[lat.opp[i]]
            for i in range(lat.q)]  # fmt: skip
    p = flux(xp, lat, xp.stack(full), feq, u_pre)
    return _rebuild(xp, lat, collision, rho, u_pre, p)


def open_boundaries(xp, lat: VelocitySet, f, cfg, collision: str, forced: bool):
    """A velocity inlet at x = 0 and a pressure outlet at the last x (`_open_column`).

    cfg["inlet_u"]: the inlet's velocity, three arrays (nz, ny); cfg["outlet_rho"]: the outlet's
    density. Across the flow, the outlet takes the velocity of the column before it. Solid
    cells are left as they are.
    """
    g = cfg["g"] if forced else (0.0, 0.0, 0.0)
    fluid = cfg["fluid"]
    inlet = _open_column(xp, lat, f[..., 0], collision, -1, g, u=cfg["inlet_u"])
    before = f[..., -2]
    rho_b = moment(before, [1] * lat.q)
    u_before = [moment(before, lat.c[:, a]) / rho_b + 0.5 * g[a] for a in range(3)]
    outlet = _open_column(xp, lat, f[..., -1], collision, +1, g, rho=cfg["outlet_rho"] + 0 * rho_b,
                          u=u_before)  # fmt: skip
    inlet = xp.where(fluid[..., 0], inlet, f[..., 0])
    outlet = xp.where(fluid[..., -1], outlet, f[..., -1])
    return xp.concatenate([inlet[..., None], f[..., 1:-1], outlet[..., None]], axis=-1)


def absorb(xp, f, cfg, cells: int):
    """The absorbing layer of the last `cells` columns: f - sigma (f - f_target), with the
    shares cfg["absorb_sigma"] (cells,) and the target packets cfg["absorb_target"]
    (Q, nz, ny, cells). Solid cells keep their packets."""
    tail = f[..., -cells:]
    relaxed = tail - cfg["absorb_sigma"] * (tail - cfg["absorb_target"])
    tail = xp.where(cfg["fluid"][..., -cells:], relaxed, tail)
    return xp.concatenate([f[..., :-cells], tail], axis=-1)


def step(
    xp,
    lat: VelocitySet,
    f_post,
    cfg,
    collision="recursive",
    free_slip_top=False,
    forced=False,
    dragged=False,
    slip_floors=False,
    matched=0,
    open_x=False,
    absorbing=0,
):
    """One time step: stream, then collide. Takes and returns post-collision populations.

    With `open_x`, the box is open along x: an inlet at x = 0 and an outlet at the last x
    (`open_boundaries`) instead of periodic sides; `absorbing` > 0 adds an absorbing layer of
    that many columns before the outlet (`absorb`).
    """
    f = stream(xp, lat, f_post, cfg["solid"], free_slip_top, slip_floors)
    if open_x:
        f = open_boundaries(xp, lat, f, cfg, collision, forced)
    f = collide_moments(xp, lat, f, cfg, collision, forced, dragged, matched)
    return absorb(xp, f, cfg, absorbing) if absorbing else f
