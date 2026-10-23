import numpy as np
import pytest

from topopt import Settings, make_case, optimize
from topopt.filters import build_filter, density_filter, heaviside


@pytest.fixture(scope="module")
def mbb60():
    """The textbook setup of the 88/99-line codes: 60x20, volfrac 0.5, p = 3, rmin = 1.5."""
    return optimize(make_case("mbb", 60, 20), Settings(volfrac=0.5, penal=3.0, rmin=1.5, filter="sensitivity"))


def test_filter_preserves_uniform_field():
    H, Hs = build_filter(17, 9, 2.3)
    np.testing.assert_allclose(density_filter(H, Hs, np.full(17 * 9, 0.37)), 0.37)
    assert abs(H - H.T).max() < 1e-15


def test_heaviside_endpoints():
    for beta in (1, 8, 64):
        np.testing.assert_allclose(heaviside(np.array([0.0, 1.0]), beta), [0.0, 1.0], atol=1e-12)


def test_mbb_volume_constraint_satisfied(mbb60):
    assert mbb60.final_volume == pytest.approx(0.5, abs=1e-3)
    assert max(mbb60.volume) <= 0.5 + 1e-3


def test_mbb_compliance_decreases(mbb60):
    c = np.array(mbb60.compliance)
    assert mbb60.final_compliance < 0.5 * c[0]
    # after the initial transient the objective is (essentially) monotone
    assert np.all(np.diff(c[5:]) <= 1e-3 * c[5:-1])


def test_mbb_matches_published_range(mbb60):
    # Our element matrix, BCs and OC update are identical to top88; its classic
    # MBB result for this setup has a compliance of roughly 200-220.
    assert 190.0 < mbb60.final_compliance < 230.0
    assert mbb60.converged
    assert mbb60.iterations < 200


def test_mbb_density_filter_range():
    res = optimize(make_case("mbb", 60, 20), Settings(volfrac=0.5, filter="density", maxiter=300))
    assert 190.0 < res.final_compliance < 240.0
    assert res.final_volume == pytest.approx(0.5, abs=1e-3)


def test_mbb_topology_is_qualitatively_right(mbb60):
    img = mbb60.image                         # half beam, symmetry at left edge
    assert img[0, :3].mean() > 0.9            # material under the load (top-left)
    assert img[-1, -3:].mean() > 0.9          # material at the support (bottom-right)
    assert img[0, :].mean() > 0.7             # top chord
    assert img[-1, :].mean() > 0.4            # bottom chord
    assert np.mean(4 * img * (1 - img)) < 0.25  # mostly black and white


def test_heaviside_gives_crisp_design():
    res = optimize(make_case("mbb", 60, 20), Settings(volfrac=0.5, filter="heaviside", maxiter=250))
    assert res.final_volume == pytest.approx(0.5, abs=2e-3)
    assert np.mean(4 * res.x * (1 - res.x)) < 0.05


def test_passive_regions_respected():
    p = make_case("cclamp", 30)
    res = optimize(p, Settings(volfrac=p.volfrac, maxiter=15))
    assert np.all(res.x[p.passive_void] == 0.0)
    assert np.all(res.x[p.passive_solid] == 1.0)
    assert res.final_volume == pytest.approx(p.volfrac, abs=2e-3)


@pytest.mark.parametrize("case", ["cantilever", "lbracket"])
def test_other_cases_run(case):
    p = make_case(case, 40)
    res = optimize(p, Settings(volfrac=p.volfrac, maxiter=30))
    assert res.final_compliance < res.compliance[0]
    assert res.final_volume == pytest.approx(p.volfrac, abs=2e-3)
    if p.passive_void.any():
        assert np.all(res.x[p.passive_void] == 0.0)


def test_cclamp_load_is_self_equilibrated():
    p = make_case("cclamp", 60)
    Fy = p.F[1::2]
    assert Fy.sum() == pytest.approx(100.0)
    assert np.allclose(p.F[0::2], 0.0)


def test_callback_can_stop_early():
    res = optimize(make_case("mbb", 30, 10), Settings(volfrac=0.5), callback=lambda it, *a: it < 5)
    assert res.iterations == 5
