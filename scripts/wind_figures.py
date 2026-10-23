"""Pictures of the wind solver's tests, for the journal.

    uv run --group docs python scripts/wind_figures.py

writes

- `docs/images/wind_convergence.png`: error against grid size for the flows with exact
  answers (from results/wind_exact.json, D3Q19 with the regularised collision), on log-log
  axes with a line of slope -2 for second order;
- `docs/images/wind_taylor_green.png`: the Taylor-Green vortices at the time their speed has
  halved, 64 cells per period at the planned speed, and what is left of the error: a pattern
  at twice the vortices' wavenumber, where their pressure lives.

Colours: the first slots of the reference categorical palette of the dataviz guidance, on a
light surface; text stays in neutral ink and every line is labelled directly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aircanopy.wind.lattice import CS2, D3Q19
from aircanopy.wind.solver import Params, Solver
from aircanopy.wind.walls import KAPPA

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]  # blue, orange, aqua, violet


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)


def convergence(out: Path) -> None:
    data = json.loads((ROOT / "results" / "wind_exact.json").read_text())
    runs = data["cases"]["D3Q19_regularised"]
    lines = [
        ("Shear wave", runs["shear_wave"]),
        ("Poiseuille flow", runs["poiseuille"]),
        ("Taylor-Green, slow (speed ÷ 16)", runs["taylor_green_speed_divided_by_16"]),
        ("Taylor-Green, planned speed", runs["taylor_green"]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150, facecolor=SURFACE)
    _style(ax)
    for (label, r), colour in zip(lines, SERIES, strict=True):
        n, e = r["sizes"], r["relative_errors"]
        ax.plot(n, e, color=colour, linewidth=2, marker="o", markersize=6, label=label)
        # Poiseuille's last point sits among the others: label it at its first point.
        at, offset, ha = (
            (0, (-8, 0), "right") if label.startswith("Poiseuille") else (-1, (8, 0), "left")
        )
        ax.annotate(label, (n[at], e[at]), xytext=offset, textcoords="offset points",
                    va="center", ha=ha, fontsize=8.5, color=INK)  # fmt: skip
    # Slope -2: halving the cell size divides the error by four.
    n_ref = np.array([8, 64])
    ax.plot(n_ref, 0.012 * (8 / n_ref) ** 2, color=MUTED, linestyle="--", linewidth=1.2,
            label="second order: error ∝ 1/n²")  # fmt: skip
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([8, 16, 32, 64], ["8", "16", "32", "64"])
    ax.set_xlim(3.2, 260)
    ax.set_xlabel("cells across the flow (per wavelength, period or channel)", color=INK)
    ax.set_ylabel("relative error", color=INK)
    ax.set_title("Wind solver against exact solutions (D3Q19, regularised)", color=INK,
                 fontsize=11, loc="left")  # fmt: skip
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def taylor_green_fields(out: Path, n: int = 64, tau: float = 0.8) -> None:
    nu = CS2 * (tau - 0.5)
    k = 2 * math.pi / n
    u0 = 0.04 * 16 / n
    y, x = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")

    def exact(t: float):
        d = math.exp(-2 * nu * k * k * t)
        ux = u0 * np.sin(k * x) * np.cos(k * y) * d
        uy = -u0 * np.cos(k * x) * np.sin(k * y) * d
        rho = 1 + 0.25 * u0**2 * (np.cos(2 * k * x) + np.cos(2 * k * y)) * d * d / CS2
        return rho, ux, uy

    shape = (1, n, n)
    s = Solver(D3Q19, np.zeros(shape, bool), Params(tau0=tau), "jax", np.float64)
    rho, ux, uy = exact(0)
    s.set_flow(rho.reshape(shape), (ux.reshape(shape), uy.reshape(shape), np.zeros(shape)),
               settle=True)  # fmt: skip
    steps = round(math.log(2.0) / (2 * nu * k * k))
    s.run(steps)
    _, (a, b, _) = s.fields()
    _, ex, ey = exact(steps)
    speed = np.hypot(a[0], b[0]) / u0
    error = 100 * np.hypot(a[0] - ex, b[0] - ey) / u0
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), dpi=150, facecolor=SURFACE)
    panels = [
        (speed, "Speed of the vortices (÷ starting speed)", "Blues"),
        (error, "Error that is left (% of the starting speed)", "Blues"),
    ]
    for ax, (field, title, cmap) in zip(axes, panels, strict=True):
        im = ax.imshow(field, origin="lower", cmap=cmap, extent=(0, n, 0, n))
        ax.set_title(title, color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("x (cells)", color=INK, fontsize=8.5)
        ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.ax.tick_params(labelsize=7.5, colors=MUTED, labelcolor=INK)
    axes[0].set_ylabel("y (cells)", color=INK, fontsize=8.5)
    fig.suptitle(f"Taylor-Green vortices after {steps} steps, {n} cells per period",
                 color=INK, fontsize=10.5, x=0.02, ha="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def lattices(out: Path) -> None:
    """The directions of D3Q19 and D3Q27 as arrows from the centre of a cell."""
    from aircanopy.wind.lattice import D3Q27

    kinds = [(1, "6 faces", SERIES[0]), (2, "12 edges", SERIES[1]), (3, "8 corners", SERIES[3])]
    fig = plt.figure(figsize=(8.4, 4.2), dpi=150, facecolor=SURFACE)
    for k, lat in enumerate((D3Q19, D3Q27)):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.set_facecolor(SURFACE)
        norm2 = (lat.c**2).sum(axis=1)
        # The cube around the cell, for orientation.
        for s in (-1, 1):
            for t in (-1, 1):
                for axis in range(3):
                    p = np.zeros((2, 3))
                    p[:, axis] = (-1, 1)
                    p[:, (axis + 1) % 3] = s
                    p[:, (axis + 2) % 3] = t
                    ax.plot(*p.T, color=GRID, linewidth=0.8)
        for n2, label, colour in kinds:
            c = lat.c[norm2 == n2]
            if len(c):
                ax.quiver(0, 0, 0, c[:, 0], c[:, 1], c[:, 2], color=colour, linewidth=1.6,
                          arrow_length_ratio=0.12, label=label)  # fmt: skip
        ax.scatter([0], [0], [0], color=INK, s=18, label="1 at rest")
        ax.set_title(f"{lat.name}: {lat.q} directions", color=INK, fontsize=10, loc="left")
        ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
        (
            ax.set_xlabel("x", color=MUTED),
            ax.set_ylabel("y", color=MUTED),
            ax.set_zlabel("z", color=MUTED),
        )
        ax.view_init(elev=18, azim=-58)
        ax.set_box_aspect((1, 1, 1))
        ax.legend(
            frameon=False, fontsize=8, labelcolor=INK, loc="upper left", bbox_to_anchor=(0.0, 0.95)
        )
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _label_ends(ax, items) -> None:
    """Label each line at its right end, nudged apart so the labels do not overlap.

    items: (x, y, text, colour) for each line's last point.
    """
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData.transform
    gap = 8.5 / 72 * fig.dpi * 1.15  # one line of 8.5-point text, in pixels
    placed = []
    for x, y, text, _ in sorted(items, key=lambda it: it[1]):
        px = to_px((x, y))[1]
        if placed and px - placed[-1] < gap:
            px = placed[-1] + gap
        placed.append(px)
        y_label = ax.transData.inverted().transform((0, px))[1]
        ax.annotate(text, (x, y), xytext=(x + 0.4, y_label), textcoords="data", va="center",
                    fontsize=8.5, color=INK, annotation_clip=False)  # fmt: skip


def tgv_dissipation(out: Path) -> None:
    """The 3D Taylor-Green vortex at Re 1600: dissipation rate against time, every grid run
    without a turbulence model, and the 512^3 pseudo-spectral reference (not stored in the
    repository: read from the cache, see benchmarks/wind_tgv1600.py)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("tgv", ROOT / "benchmarks" / "wind_tgv1600.py")
    tgv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tgv)
    ref = tgv.reference()
    runs = []
    for device in ("cpu", "gpu"):
        path = ROOT / "results" / f"wind_tgv1600_{device}.json"
        if path.is_file():
            runs += [r for r in json.loads(path.read_text())["runs"] if r["smagorinsky"] == 0.0]
    best = {}
    for r in runs:  # one run per grid size and lattice; the graphics card's if there are two
        best[(r["n"], r["lattice"])] = r
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150, facecolor=SURFACE)
    _style(ax)
    ax.plot(ref[:, 0], ref[:, 2], color=INK, linewidth=2.2, label="reference: spectral, 512³")
    ends = [(20.0, float(ref[-1, 2]), "reference", INK)]
    for k, ((n, lattice), r) in enumerate(sorted(best.items())):
        colour = SERIES[k % len(SERIES)]
        label = f"{n}³, {lattice}"
        ax.plot(r["times"], r["dissipation"], color=colour, linewidth=1.6, label=label)
        ends.append((r["times"][-1], r["dissipation"][-1], label, colour))
    ax.set_xlim(0, 24.5)
    ax.set_xlabel("time, in units of L / V0", color=INK)
    ax.set_ylabel("dissipation rate -dE/dt (in V0³ / L)", color=INK)
    ax.set_title("Taylor-Green vortex at Re 1600: how fast the kinetic energy is lost",
                 color=INK, fontsize=11, loc="left")  # fmt: skip
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    _label_ends(ax, ends)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def abl_profiles(out: Path) -> None:
    """The neutral boundary layer: mean wind against height over the roughness length (log
    scale, on which the law of the wall is one straight line for every run), and
    phi = (kappa z / u*) dU/dz, which is 1 on the law of the wall."""
    path = ROOT / "results" / "wind_abl_cpu.json"
    runs = sorted(json.loads(path.read_text())["runs"], key=lambda r: (-r["h"], r["smagorinsky"]))
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.6, 4.6), dpi=150, facecolor=SURFACE,
                                      sharey=True)  # fmt: skip
    for ax in (left, right):
        _style(ax)
        ax.set_yscale("log")
    z0 = runs[0]["z0"]
    tallest = max(r["h"] for r in runs)
    z = np.geomspace(0.5, tallest, 50)
    left.plot(np.log(z / z0) / KAPPA, z / z0, color=INK, linewidth=2.2, label="law of the wall")
    for k, r in enumerate(runs):
        colour = SERIES[k % len(SERIES)]
        label = f"Cs = {r['smagorinsky']}, H = {r['h']} cells"
        left.plot(np.asarray(r["mean_u"]) / r["u_star"], np.asarray(r["z"]) / r["z0"],
                  color=colour, linewidth=1.8, marker="o", markersize=3, label=label)  # fmt: skip
        right.plot(r["phi"], np.asarray(r["z_phi"]) / r["z0"], color=colour, linewidth=1.8,
                   marker="o", markersize=3, label=label)  # fmt: skip
    right.axvline(1.0, color=INK, linewidth=1.6)
    # The curves' lowest points sit near phi = 1, so a legend there would hide them: the left
    # panel's legend serves both, and the line is labelled where it is.
    right.text(1.03, 0.5 / z0 * 1.15, "law of the wall", color=INK, fontsize=8, va="bottom")
    for ax in (left, right):  # where the criterion applies, for the tallest box
        ax.axhspan(2 / z0, 0.2 * tallest / z0, color=GRID, alpha=0.6, zorder=0)
    left.set_xlabel("mean wind U / u*", color=INK)
    left.set_ylabel("height z / z0 (log scale)", color=INK)
    right.set_xlabel("phi = (kappa z / u*) dU/dz", color=INK)
    left.set_title("Mean wind", color=INK, fontsize=10, loc="left")
    right.set_title("Shear, scaled: 1 on the law of the wall", color=INK, fontsize=10,
                    loc="left")  # fmt: skip
    left.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    title = (f"Neutral boundary layer over a rough floor (band: where the criterion applies for "
             f"H = {tallest} cells)")  # fmt: skip
    fig.suptitle(title, color=INK, fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    images = ROOT / "docs" / "images"
    lattices(images / "wind_lattices.png")
    convergence(images / "wind_convergence.png")
    taylor_green_fields(images / "wind_taylor_green.png")
    tgv_dissipation(images / "wind_tgv1600.png")
    abl_profiles(images / "wind_abl.png")


if __name__ == "__main__":
    main()
