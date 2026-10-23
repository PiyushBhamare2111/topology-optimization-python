"""Re-create every figure in docs/ (benchmarks, C-clamp study, showcase media).

    python scripts/generate_docs.py

Runs all cases, writes per-case figures to docs/<case>/, a results table to
docs/results.md / docs/results.json, and 1920x1080 / 1080x1080 showcase PNGs
plus per-iteration frames to docs/media/.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from topopt import Settings, make_case, optimize  # noqa: E402
from topopt.cli import save_outputs  # noqa: E402
from topopt.visualize import (ACCENT, BG, DENSITY_CMAP, MUTED, TEXT, _style, density_rgb,  # noqa: E402
                              display_image, plot_density, plot_stress, render_stl)
from topopt.export import density_to_triangles, mirror_image  # noqa: E402

DOCS = ROOT / "docs"
MEDIA = DOCS / "media"

RUNS = [
    # key, case, nelx, nely, settings
    ("mbb_60x20_sens", "mbb", 60, 20, Settings(volfrac=0.5, penal=3, rmin=1.5, filter="sensitivity", maxiter=300)),
    ("mbb_60x20_dens", "mbb", 60, 20, Settings(volfrac=0.5, penal=3, rmin=1.5, filter="density", maxiter=300)),
    ("mbb", "mbb", 120, 40, Settings(volfrac=0.5, penal=3, rmin=1.5, filter="density", maxiter=300)),
    ("mbb_heaviside", "mbb", 120, 40, Settings(volfrac=0.5, penal=3, rmin=1.5, filter="heaviside", maxiter=400)),
    ("cantilever", "cantilever", 80, 40, Settings(volfrac=0.4, penal=3, rmin=1.5, filter="density", maxiter=300)),
    ("lbracket", "lbracket", 100, 100, Settings(volfrac=0.4, penal=3, rmin=1.5, filter="density", maxiter=300)),
    ("cclamp", "cclamp", 90, None, Settings(volfrac=0.326, penal=3, rmin=1.5, filter="density", maxiter=300)),
]
FULL_OUTPUT = {"mbb", "mbb_heaviside", "cantilever", "lbracket", "cclamp"}


def run_all():
    results = {}
    rows = []
    for key, case, nelx, nely, s in RUNS:
        p = make_case(case, nelx, nely)
        print(f"running {key} ({p.nelx}x{p.nely}, {s.filter}) ...", flush=True)
        res = optimize(p, s)
        results[key] = res
        sm = res.summary()
        if key in FULL_OUTPUT:
            out = DOCS / key
            save_outputs(res, out, gif=True, stl=True, gif_width=480)
            # STL files/renders are big; keep only the C-clamp ones in the repo
            if key != "cclamp":
                (out / "design.stl").unlink()
                (out / "stl_render.png").unlink()
            (out / "density.npy").unlink()
        rows.append({"key": key, **sm})
        print(f"   c = {sm['compliance']:.5g}, vol = {sm['volume']:.4f}, it = {sm['iterations']}, "
              f"t = {sm['runtime_s']} s", flush=True)
    (DOCS / "results.json").write_text(json.dumps(rows, indent=2))
    lines = ["| Run | Case | Mesh | Filter | volfrac | Final compliance | Iterations | Converged | Runtime [s] |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        unit = r["units"]["compliance"]
        lines.append(f"| `{r['key']}` | {r['case']} | {r['nelx']}x{r['nely']} | {r['filter']} | {r['volfrac']} | "
                     f"{r['compliance']:.4g}{'' if unit == '-' else ' ' + unit} | {r['iterations']} | "
                     f"{'yes' if r['converged'] else 'no (max it.)'} | {r['runtime_s']} |")
    (DOCS / "results.md").write_text("\n".join(lines) + "\n")
    return results


# ----------------------------------------------------------------------------- media
def _canvas(w=1920, h=1080, dpi=100):
    _style()
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_facecolor(BG)
    return fig


def _clean(ax):
    ax.set_axis_off()
    ax.set_facecolor(BG)


def _headline(fig, title, subtitle):
    fig.text(0.04, 0.93, title, fontsize=40, fontweight="bold", color=TEXT, va="top")
    fig.text(0.04, 0.855, subtitle, fontsize=20, color=MUTED, va="top")
    fig.add_artist(plt.Line2D([0.04, 0.12], [0.80, 0.80], color=ACCENT, lw=5, transform=fig.transFigure))


def media_hero(results):
    r = results["cclamp"]
    s = r.summary()
    fig = _canvas()
    _headline(fig, "Topology Optimization in Python",
              "2D SIMP optimiser from scratch  -  Q4 FEA  -  density filter  -  OC update  -  STL export")
    ax1 = fig.add_axes([0.03, 0.08, 0.27, 0.68])
    ax1.imshow(display_image(r), cmap=DENSITY_CMAP, vmin=0, vmax=1, interpolation="nearest")
    _clean(ax1)
    ax1.set_title("optimised density", color=MUTED, fontsize=16)
    ax2 = fig.add_axes([0.31, 0.08, 0.27, 0.68])
    vm = display_image(r, r.von_mises)
    rho = display_image(r)
    ax2.imshow(np.ma.masked_where(rho < 0.5, vm), cmap=plt.get_cmap("inferno").with_extremes(bad=BG),
               vmin=0, vmax=np.percentile(vm[rho > 0.5], 99.5), interpolation="nearest")
    _clean(ax2)
    ax2.set_title(f"von Mises stress  (peak {s['max_von_mises']:.0f} MPa)", color=MUTED, fontsize=16)
    ax3 = fig.add_axes([0.59, 0.04, 0.39, 0.66], projection="3d")
    tris = density_to_triangles(r.image, r.problem.h, r.problem.t, upsample=2)
    render_stl(tris, ax=ax3, title="", elev=32, azim=-58, zoom=1.3)
    fig.text(0.785, 0.735, "extruded STL, ready to slice", color=MUTED, fontsize=16, ha="center", fontweight="bold")
    fig.text(0.96, 0.93, "C-clamp frame  |  PLA  |  100 N", fontsize=20, color=ACCENT, ha="right", va="top")
    fig.text(0.96, 0.875, f"volume {100 * s['volume']:.1f} %  |  {s['mass_g']:.1f} g  |  peak/yield {s['max_von_mises'] / 50:.2f}",
             fontsize=17, color=TEXT, ha="right", va="top")
    fig.savefig(MEDIA / "01_hero_cclamp.png", dpi=100)
    plt.close(fig)


def media_gallery(results):
    fig = _canvas()
    _headline(fig, "Classic benchmarks", "Minimum-compliance designs found by the same 2D SIMP code")
    items = [("mbb", "MBB beam  120x40 (mirrored)"), ("cantilever", "Cantilever  80x40"),
             ("lbracket", "L-bracket  100x100"), ("cclamp", "C-clamp  90x120")]
    boxes = [[0.03, 0.42, 0.45, 0.36], [0.52, 0.42, 0.45, 0.36], [0.03, 0.02, 0.45, 0.36], [0.52, 0.02, 0.45, 0.36]]
    for (key, label), b in zip(items, boxes):
        ax = fig.add_axes(b)
        r = results[key]
        ax.imshow(display_image(r), cmap=DENSITY_CMAP, vmin=0, vmax=1, interpolation="nearest")
        _clean(ax)
        ax.set_title(f"{label}   c = {r.final_compliance:.4g}", color=TEXT, fontsize=16)
    fig.savefig(MEDIA / "02_benchmark_gallery.png", dpi=100)
    plt.close(fig)


def media_evolution(results, key="mbb", name="03_mbb_evolution.png"):
    r = results[key]
    fig = _canvas()
    _headline(fig, "How the design emerges", f"{r.problem.title}: density field at selected iterations")
    picks = [1, 5, 10, 20, 40, r.iterations]
    frames = {it: x for it, x in r.frames}
    from topopt.fem import FEModel  # noqa: F401  (frames hold flat densities)
    for k, it in enumerate(picks):
        it_near = min(frames, key=lambda i: abs(i - it))
        ax = fig.add_axes([0.03 + (k % 3) * 0.32, 0.43 - (k // 3) * 0.38, 0.30, 0.33])
        ax.imshow(display_image(r, frames[it_near]), cmap=DENSITY_CMAP, vmin=0, vmax=1, interpolation="nearest")
        _clean(ax)
        ax.set_title(f"iteration {it_near}", color=ACCENT if k == len(picks) - 1 else TEXT, fontsize=16)
    fig.savefig(MEDIA / name, dpi=100)
    plt.close(fig)


def media_convergence(results):
    r = results["mbb"]
    fig = _canvas()
    _headline(fig, "Convergence", "Compliance falls by an order of magnitude while the volume constraint stays active")
    it = np.arange(1, len(r.compliance) + 1)
    a1 = fig.add_axes([0.07, 0.40, 0.88, 0.38])
    a1.plot(it, r.compliance, color=ACCENT, lw=3)
    a1.set_yscale("log")
    a1.set_ylabel("compliance  F$^T$u", fontsize=16)
    a1.tick_params(labelsize=14, labelbottom=False)
    a2 = fig.add_axes([0.07, 0.08, 0.88, 0.25], sharex=a1)
    a2.plot(it, r.volume, color=TEXT, lw=3)
    a2.axhline(r.settings.volfrac, color=MUTED, ls="--")
    a2.set_ylim(0.45, 0.55)
    a2.set_ylabel("volume fraction", fontsize=16)
    a2.set_xlabel("iteration", fontsize=16)
    a2.tick_params(labelsize=14)
    fig.savefig(MEDIA / "04_convergence.png", dpi=100)
    plt.close(fig)


def media_square_stl(results):
    r = results["cclamp"]
    fig = _canvas(1080, 1080)
    fig.text(0.06, 0.94, "From density field to printable STL", fontsize=30, fontweight="bold", color=TEXT, va="top")
    fig.text(0.06, 0.885, "marching squares  ->  polygons  ->  extrusion  (watertight)", fontsize=17,
             color=MUTED, va="top")
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.86], projection="3d")
    tris = density_to_triangles(r.image, r.problem.h, r.problem.t, upsample=2)
    render_stl(tris, ax=ax, title="", elev=38, azim=-62, zoom=1.15)
    fig.savefig(MEDIA / "05_stl_render_square.png", dpi=100)
    plt.close(fig)


def media_frames(results, key="cclamp"):
    """Per-iteration 1080x1080 frames (for video editing) and a high-res GIF."""
    r = results[key]
    d = MEDIA / f"frames_{key}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    imgs = []
    for it, flat in r.frames:
        img = display_image(r, flat)
        scale = max(1, 900 // max(img.shape))
        tile = Image.fromarray(density_rgb(img, scale))
        canvas = Image.new("RGB", (1080, 1080), BG)
        canvas.paste(tile, ((1080 - tile.width) // 2, (1080 - tile.height) // 2))
        canvas.save(d / f"iter_{it:04d}.png", optimize=True)
        imgs.append(canvas.resize((540, 540), Image.NEAREST))
    imgs[0].save(MEDIA / f"{key}_optimization_540.gif", save_all=True, append_images=imgs[1:],
                 duration=[90] * (len(imgs) - 1) + [1500], loop=0, optimize=True)


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    results = run_all()
    media_hero(results)
    media_gallery(results)
    media_evolution(results)
    media_convergence(results)
    media_square_stl(results)
    media_frames(results)
    print("docs regenerated")


if __name__ == "__main__":
    main()
