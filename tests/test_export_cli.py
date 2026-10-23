import json

import numpy as np
import pytest

from topopt import Settings, make_case, optimize
from topopt.cli import main
from topopt.export import density_to_triangles, export_stl, is_watertight, marching_squares, triangle_volume


def test_square_extrusion_volume_and_watertight():
    img = np.zeros((6, 6))
    img[1:5, 1:5] = 1.0
    tris = density_to_triangles(img, h=2.0, thickness=3.0, upsample=1)
    assert is_watertight(tris)
    # iso 0.5 lies half-way between pixel centres -> an 8 x 8 mm square whose four
    # corners are chamfered by marching squares (legs of 1 mm, area 0.5 mm^2 each)
    assert triangle_volume(tris) == pytest.approx((8.0 * 8.0 - 4 * 0.5) * 3.0, rel=1e-9)


def test_ring_with_hole_and_saddle_is_watertight():
    yy, xx = np.mgrid[0:40, 0:40]
    r = np.hypot(xx - 19.5, yy - 19.5)
    img = ((r > 7) & (r < 15)).astype(float)
    img[3, 3] = img[4, 4] = 1.0          # diagonal pixels -> saddle cell
    tris = density_to_triangles(img, thickness=2.0, upsample=2)
    assert is_watertight(tris)
    assert triangle_volume(tris) > 0
    polys, segs = marching_squares(img)
    assert len(segs) > 0 and len(polys) > 0


def test_export_optimised_design(tmp_path):
    res = optimize(make_case("mbb", 40, 14), Settings(volfrac=0.5, maxiter=40))
    tris = export_stl(res, tmp_path / "mbb.stl", thickness=2.0)
    assert (tmp_path / "mbb.stl").stat().st_size > 1000
    assert is_watertight(tris)
    # extruded volume ~ solid area of the (mirrored) design * thickness
    area = (res.image > 0.5).sum() * 2
    assert triangle_volume(tris) == pytest.approx(area * 2.0, rel=0.1)


def test_cli_run_writes_everything(tmp_path, capsys):
    out = tmp_path / "run"
    assert main(["run", "--case", "cantilever", "--nelx", "30", "--maxiter", "15", "--out", str(out), "--quiet"]) == 0
    for name in ["density.png", "convergence.png", "stress.png", "optimization.gif", "design.stl",
                 "stl_render.png", "history.csv", "summary.json", "density.npy"]:
        assert (out / name).exists(), name
    s = json.loads((out / "summary.json").read_text())
    assert s["case"] == "cantilever" and s["iterations"] == 15
    assert "compliance" in capsys.readouterr().out


def test_cli_list(capsys):
    assert main(["list"]) == 0
    assert "cclamp" in capsys.readouterr().out
