"""Command-line interface:  python -m topopt run --case mbb --nelx 120 --nely 40 --volfrac 0.5"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .cases import CASES, DEFAULT_MESH, make_case
from .optimizer import Settings, optimize


def save_outputs(res, out: Path, gif=True, stl=True, thickness=None, upsample=2, gif_width=640) -> dict:
    """Write every artefact of a run into ``out`` and return {name: path}."""
    from . import export, visualize

    out.mkdir(parents=True, exist_ok=True)
    files = {}
    visualize.plot_density(res, out / "density.png")
    files["density"] = out / "density.png"
    visualize.plot_convergence(res, out / "convergence.png")
    files["convergence"] = out / "convergence.png"
    visualize.plot_stress(res, out / "stress.png")
    files["stress"] = out / "stress.png"
    np.save(out / "density.npy", res.image)
    files["density_npy"] = out / "density.npy"
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "compliance", "volume", "change"])
        for i, row in enumerate(zip(res.compliance, res.volume, res.change), 1):
            w.writerow([i, *row])
    files["history"] = out / "history.csv"
    if gif:
        visualize.save_gif(res, out / "optimization.gif", max_width=gif_width)
        files["gif"] = out / "optimization.gif"
    summary = res.summary()
    if stl:
        tris = export.export_stl(res, out / "design.stl", thickness=thickness, upsample=upsample)
        visualize.render_stl(tris, out / "stl_render.png", title=f"{res.problem.title} - extruded STL")
        files["stl"] = out / "design.stl"
        files["stl_render"] = out / "stl_render.png"
        summary["stl_triangles"] = int(len(tris))
        summary["stl_volume"] = export.triangle_volume(tris)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    files["summary"] = out / "summary.json"
    return files


def _cmd_run(a) -> int:
    kw = {}
    if a.thickness is not None and a.case == "cclamp":
        kw["thickness"] = a.thickness
    problem = make_case(a.case, a.nelx, a.nely, **kw)
    s = Settings(volfrac=a.volfrac if a.volfrac is not None else problem.volfrac,
                 penal=a.penal, rmin=a.rmin, filter=a.filter, move=a.move,
                 maxiter=a.maxiter, tol=a.tol)
    print(f"[topopt] {problem.title}: {problem.nelx}x{problem.nely} elements, volfrac={s.volfrac}, "
          f"penal={s.penal}, rmin={s.rmin}, filter={s.filter}")

    def cb(it, x, c, vol, ch):
        if not a.quiet and (it == 1 or it % 10 == 0):
            print(f"  it {it:4d}  c = {c:12.5g}  vol = {vol:.4f}  change = {ch:.4f}")

    res = optimize(problem, s, callback=cb)
    out = Path(a.out or f"results/{a.case}")
    save_outputs(res, out, gif=not a.no_gif, stl=not a.no_stl, thickness=a.thickness)
    sm = res.summary()
    print(f"[topopt] done in {sm['runtime_s']} s, {sm['iterations']} iterations "
          f"({'converged' if sm['converged'] else 'max iterations reached'})")
    print(f"[topopt] compliance = {sm['compliance']:.5g} {problem.units['compliance']}, volume = {sm['volume']:.4f}")
    if "mass_g" in sm:
        print(f"[topopt] mass = {sm['mass_g']:.2f} g, peak von Mises = {sm['max_von_mises']:.3g} MPa "
              f"(SF vs {sm['yield_strength']:g} MPa yield = {sm['safety_factor']:.2f})")
    print(f"[topopt] results written to {out.resolve()}")
    return 0


def _cmd_list(_a) -> int:
    for name, fn in CASES.items():
        doc = (fn.__doc__ or "").strip().splitlines()[0]
        print(f"{name:11s} default mesh {DEFAULT_MESH[name][0]}x{DEFAULT_MESH[name][1]}  {doc}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m topopt", description="2D SIMP topology optimisation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run an optimisation and save results")
    r.add_argument("--case", choices=sorted(CASES), default="mbb")
    r.add_argument("--nelx", type=int, default=None, help="elements in x (default depends on case)")
    r.add_argument("--nely", type=int, default=None, help="elements in y (default keeps the case aspect ratio)")
    r.add_argument("--volfrac", type=float, default=None, help="target volume fraction (default per case)")
    r.add_argument("--penal", type=float, default=3.0, help="SIMP penalisation exponent p")
    r.add_argument("--rmin", type=float, default=1.5, help="filter radius in elements")
    r.add_argument("--filter", choices=["density", "sensitivity", "heaviside"], default="density")
    r.add_argument("--move", type=float, default=0.2, help="OC move limit")
    r.add_argument("--maxiter", type=int, default=200)
    r.add_argument("--tol", type=float, default=0.01, help="stop when max density change < tol")
    r.add_argument("--thickness", type=float, default=None, help="STL extrusion thickness (cclamp: part thickness, mm)")
    r.add_argument("--out", default=None, help="output folder (default results/<case>)")
    r.add_argument("--no-gif", action="store_true")
    r.add_argument("--no-stl", action="store_true")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=_cmd_run)
    lst = sub.add_parser("list", help="list available load cases")
    lst.set_defaults(func=_cmd_list)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
