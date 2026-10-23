import numpy as np
import pytest

from topopt.fem import FEModel, element_stiffness, node_id, plane_stress_D, strain_displacement_B


def top88_KE(nu):
    """Closed-form element matrix printed in Andreassen et al. (2011), E = 1."""
    k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
                  -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
    idx = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 7, 6, 5, 4, 3, 2], [2, 7, 0, 5, 6, 3, 4, 1],
                    [3, 6, 5, 0, 7, 2, 1, 4], [4, 5, 6, 7, 0, 1, 2, 3], [5, 4, 3, 2, 1, 0, 7, 6],
                    [6, 3, 4, 1, 2, 7, 0, 5], [7, 2, 1, 4, 3, 6, 5, 0]])
    return k[idx] / (1 - nu**2)


@pytest.mark.parametrize("nu", [0.0, 0.3, 0.36, 0.45])
def test_element_stiffness_matches_88_line_code(nu):
    np.testing.assert_allclose(element_stiffness(1.0, nu), top88_KE(nu), atol=1e-14)


def test_element_stiffness_symmetric_psd_with_three_rigid_body_modes():
    KE = element_stiffness(2.5, 0.3, h=0.7, t=3.0)
    np.testing.assert_allclose(KE, KE.T, atol=1e-12)
    ev = np.linalg.eigvalsh(KE)
    tol = 1e-10 * ev.max()
    assert ev.min() > -tol
    assert np.sum(np.abs(ev) < tol) == 3
    # translations x, y and an infinitesimal rotation produce zero force
    xy = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    tx = np.tile([1.0, 0.0], 4)
    ty = np.tile([0.0, 1.0], 4)
    rot = np.column_stack([-xy[:, 1], xy[:, 0]]).ravel()
    for mode in (tx, ty, rot):
        assert np.abs(KE @ mode).max() < 1e-12


def test_constant_strain_patch():
    """A linear displacement field gives exactly the constant stress D*eps at any point."""
    D = plane_stress_D(1.0, 0.3)
    xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    ex, ey, gxy = 1e-3, -2e-4, 5e-4
    u = np.column_stack([ex * xy[:, 0] + gxy * xy[:, 1], ey * xy[:, 1]]).ravel()
    for xi, eta in [(0, 0), (0.3, -0.7), (-1, 1)]:
        np.testing.assert_allclose(strain_displacement_B(xi, eta, 1.0) @ u, [ex, ey, gxy], atol=1e-15)
        np.testing.assert_allclose(D @ strain_displacement_B(xi, eta, 1.0) @ u, D @ [ex, ey, gxy])


def solid_cantilever_tip(nelx, nely, E=1.0, nu=0.3, P=1.0):
    m = FEModel(nelx, nely, E, nu)
    F = np.zeros(m.ndof)
    iy = np.arange(nely + 1)
    w = np.ones(nely + 1)
    w[[0, -1]] = 0.5                       # uniform shear traction on the free end
    F[2 * node_id(nely, nelx, iy) + 1] = -P * w / w.sum()
    left = node_id(nely, 0, iy)
    fixed = np.union1d(2 * left, 2 * left + 1)
    free = np.setdiff1d(np.arange(m.ndof), fixed)
    u = m.solve(np.ones(m.nel), 3.0, F, free)
    return -u[2 * node_id(nely, nelx, nely // 2) + 1]


@pytest.mark.parametrize("nelx,nely", [(100, 10), (200, 20)])
def test_cantilever_tip_deflection_vs_beam_theory(nelx, nely):
    E, nu, P = 1.0, 0.3, 1.0
    L, H = float(nelx), float(nely)
    I = H**3 / 12.0
    G = E / (2 * (1 + nu))
    euler = P * L**3 / (3 * E * I)
    timoshenko = euler + P * L / (5 / 6 * G * H)
    tip = solid_cantilever_tip(nelx, nely, E, nu, P)
    assert tip == pytest.approx(euler, rel=0.02)
    assert tip == pytest.approx(timoshenko, rel=0.02)


def test_stiffness_scales_with_modulus_and_thickness():
    a = solid_cantilever_tip(40, 4)
    m = FEModel(40, 4, 10.0, 0.3, t=2.0)
    assert m.KE.max() == pytest.approx(2.0 * element_stiffness(1.0, 0.3).max())
    assert a > 0


def test_von_mises_uniaxial_tension():
    """Bar in uniaxial tension: sigma_vm = applied stress everywhere."""
    nelx, nely, P = 20, 4, 8.0
    m = FEModel(nelx, nely, 200.0, 0.3, h=0.5, t=2.0)
    iy = np.arange(nely + 1)
    w = np.ones(nely + 1)
    w[[0, -1]] = 0.5
    F = np.zeros(m.ndof)
    F[2 * node_id(nely, nelx, iy)] = P * w / w.sum()
    left = node_id(nely, 0, iy)
    fixed = np.union1d(2 * left, [2 * left[-1] + 1])   # roller left edge + pin
    free = np.setdiff1d(np.arange(m.ndof), fixed)
    u = m.solve(np.ones(m.nel), 3.0, F, free)
    vm = m.von_mises(u, np.ones(m.nel), 3.0)
    area = nely * 0.5 * 2.0
    np.testing.assert_allclose(vm, P / area, rtol=1e-8)
