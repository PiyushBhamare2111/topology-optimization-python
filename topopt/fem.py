"""Linear-elastic finite element analysis on a structured grid of Q4 elements.

Conventions
-----------
* The design domain is a rectangle of ``nelx`` x ``nely`` square elements of side
  ``h`` and out-of-plane thickness ``t`` (plane stress).
* Physical axes: x points right, y points **up**.
* Nodes are numbered column by column, top to bottom (the same convention as the
  88-line MATLAB code of Andreassen et al., 2011)::

      node(ix, iy) = ix * (nely + 1) + iy      # iy = 0 is the TOP row

  Node ``n`` owns DOFs ``2n`` (u_x) and ``2n + 1`` (u_y).
* Elements are numbered the same way: ``elem(ex, ey) = ex * nely + ey``. A flat
  element array reshapes to an image with ``a.reshape(nelx, nely).T``
  (row 0 = top of the part).
* Local node order inside an element is counter-clockwise starting at the
  lower-left corner: LL, LR, UR, UL.

Units are whatever you feed in, as long as they are consistent. The C-clamp case
uses N, mm and MPa, so displacements come out in mm and compliance in N*mm.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# Natural coordinates of the four local nodes (LL, LR, UR, UL).
_XI = np.array([-1.0, 1.0, 1.0, -1.0])
_ETA = np.array([-1.0, -1.0, 1.0, 1.0])


def plane_stress_D(E: float, nu: float) -> np.ndarray:
    """3x3 plane-stress constitutive matrix relating [sx, sy, txy] to [ex, ey, gxy]."""
    return E / (1.0 - nu**2) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
    )


def strain_displacement_B(xi: float, eta: float, h: float = 1.0) -> np.ndarray:
    """3x8 strain-displacement matrix of a square Q4 element of side ``h``.

    Bilinear shape functions N_i = (1 + xi_i xi)(1 + eta_i eta) / 4. For a square
    element the Jacobian is diagonal, dx/dxi = dy/deta = h/2.
    """
    dN_dxi = 0.25 * _XI * (1.0 + _ETA * eta)
    dN_deta = 0.25 * _ETA * (1.0 + _XI * xi)
    dN_dx = dN_dxi * (2.0 / h)
    dN_dy = dN_deta * (2.0 / h)
    B = np.zeros((3, 8))
    B[0, 0::2] = dN_dx
    B[1, 1::2] = dN_dy
    B[2, 0::2] = dN_dy
    B[2, 1::2] = dN_dx
    return B


def element_stiffness(E: float = 1.0, nu: float = 0.3, h: float = 1.0, t: float = 1.0) -> np.ndarray:
    """8x8 stiffness matrix of a square bilinear (Q4) plane-stress element.

    Integrated exactly with 2x2 Gauss quadrature: KE = sum_gp B^T D B |J| t.
    For a square element the result does not depend on ``h`` (B ~ 1/h, |J| ~ h^2).
    """
    D = plane_stress_D(E, nu)
    g = 1.0 / np.sqrt(3.0)
    detJ = (h / 2.0) ** 2
    KE = np.zeros((8, 8))
    for xi in (-g, g):
        for eta in (-g, g):
            B = strain_displacement_B(xi, eta, h)
            KE += B.T @ D @ B * detJ * t
    return 0.5 * (KE + KE.T)


def node_id(nely: int, ix, iy):
    """Global node number of grid node (ix, iy); iy = 0 is the top row."""
    return np.asarray(ix) * (nely + 1) + np.asarray(iy)


def element_dofs(nelx: int, nely: int) -> np.ndarray:
    """(nelx*nely, 8) array with the global DOFs of every element (LL, LR, UR, UL)."""
    ex, ey = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    ul = node_id(nely, ex, ey).ravel()
    ll = ul + 1
    ur = ul + nely + 1
    lr = ur + 1
    return np.column_stack(
        [2 * ll, 2 * ll + 1, 2 * lr, 2 * lr + 1, 2 * ur, 2 * ur + 1, 2 * ul, 2 * ul + 1]
    )


class FEModel:
    """Sparse FE model of the grid with SIMP-interpolated element stiffness.

    Young's modulus of element e is  E_e = Emin + x_e^p (E0 - Emin)  (modified SIMP).
    """

    def __init__(self, nelx, nely, E0=1.0, nu=0.3, h=1.0, t=1.0, Emin_ratio=1e-9):
        self.nelx, self.nely = int(nelx), int(nely)
        self.E0, self.nu, self.h, self.t = float(E0), float(nu), float(h), float(t)
        self.Emin = Emin_ratio * self.E0
        self.nel = self.nelx * self.nely
        self.ndof = 2 * (self.nelx + 1) * (self.nely + 1)
        self.KE = element_stiffness(1.0, nu, h, t)  # unit modulus; scaled per element
        self.edof = element_dofs(self.nelx, self.nely)
        # Triplet index arrays for COO assembly: entry (i, j) of KE for element e
        # goes to (edof[e, i], edof[e, j]); KE.ravel() is row-major, so i repeats.
        self.iK = np.repeat(self.edof, 8, axis=1).ravel()
        self.jK = np.tile(self.edof, (1, 8)).ravel()
        self._KE_flat = self.KE.ravel()

    def young(self, xPhys: np.ndarray, penal: float) -> np.ndarray:
        return self.Emin + xPhys**penal * (self.E0 - self.Emin)

    def assemble(self, xPhys: np.ndarray, penal: float) -> sp.csc_matrix:
        sK = (self._KE_flat[None, :] * self.young(xPhys, penal)[:, None]).ravel()
        K = sp.coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()
        return K

    def solve(self, xPhys: np.ndarray, penal: float, F: np.ndarray, free: np.ndarray) -> np.ndarray:
        """Solve K(x) u = F with u = 0 on the fixed DOFs; returns the full u vector."""
        K = self.assemble(xPhys, penal)
        Kff = K[free][:, free]
        u = np.zeros(self.ndof)
        u[free] = spsolve(Kff, F[free], permc_spec="MMD_AT_PLUS_A")
        return u

    def element_energy(self, u: np.ndarray) -> np.ndarray:
        """u_e^T KE u_e for every element (unit modulus)."""
        ue = u[self.edof]
        return np.einsum("ij,jk,ik->i", ue, self.KE, ue)

    def von_mises(self, u: np.ndarray, xPhys: np.ndarray, penal: float) -> np.ndarray:
        """Element-centre von Mises stress of the SIMP-interpolated material.

        sigma_e = (E_e / E0) * D0 B(0,0) u_e. For solid elements (x = 1) this is the
        true stress; for void it tends to zero.
        """
        D = plane_stress_D(self.E0, self.nu)
        B0 = strain_displacement_B(0.0, 0.0, self.h)
        s = (u[self.edof] @ (D @ B0).T) * (self.young(xPhys, penal) / self.E0)[:, None]
        sx, sy, txy = s[:, 0], s[:, 1], s[:, 2]
        return np.sqrt(sx**2 - sx * sy + sy**2 + 3.0 * txy**2)


def to_image(a: np.ndarray, nelx: int, nely: int) -> np.ndarray:
    """Flat element array -> (nely, nelx) image, row 0 at the top of the part."""
    return np.asarray(a).reshape(nelx, nely).T


def from_image(img: np.ndarray) -> np.ndarray:
    """(nely, nelx) image -> flat element array (inverse of :func:`to_image`)."""
    return np.asarray(img).T.ravel()
