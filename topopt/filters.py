"""Regularisation filters and Heaviside projection.

Without filtering, SIMP produces checkerboards and mesh-dependent results. A
filter of radius ``rmin`` (in elements) imposes a minimum length scale:

* **density filter**     x~_e = sum_i H_ei x_i / sum_i H_ei           (Bruns & Tortorelli 2001)
* **sensitivity filter** dc~_e = sum_i H_ei x_i dc_i / (max(1e-3, x_e) sum_i H_ei)  (Sigmund 1997)

with the linear "hat" weight H_ei = max(0, rmin - dist(e, i)).

**Heaviside projection** (Guest 2004; tanh form of Wang, Lazarov & Sigmund 2011)
pushes the filtered field towards crisp 0/1 values:

    x_phys = [tanh(beta*eta) + tanh(beta*(x~ - eta))] / [tanh(beta*eta) + tanh(beta*(1 - eta))]
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_filter(nelx: int, nely: int, rmin: float):
    """Return the sparse weight matrix H (csr, symmetric) and its row sums Hs."""
    r = int(np.ceil(rmin)) - 1
    ex, ey = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    ex, ey = ex.ravel(), ey.ravel()
    e = ex * nely + ey
    rows, cols, vals = [], [], []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            w = rmin - np.hypot(dx, dy)
            if w <= 0:
                continue
            nx, ny = ex + dx, ey + dy
            ok = (nx >= 0) & (nx < nelx) & (ny >= 0) & (ny < nely)
            rows.append(e[ok])
            cols.append(nx[ok] * nely + ny[ok])
            vals.append(np.full(ok.sum(), w))
    n = nelx * nely
    H = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    ).tocsr()
    Hs = np.asarray(H.sum(axis=1)).ravel()
    return H, Hs


def density_filter(H, Hs, x):
    return (H @ x) / Hs


def density_filter_chain(H, Hs, dfdxt):
    """Chain rule through the density filter: d f / d x = H^T (df/dx~ / Hs)."""
    return H.T @ (dfdxt / Hs)


def sensitivity_filter(H, Hs, x, dc):
    return (H @ (x * dc)) / Hs / np.maximum(1e-3, x)


def heaviside(xt, beta, eta=0.5):
    num = np.tanh(beta * eta) + np.tanh(beta * (xt - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return num / den


def heaviside_grad(xt, beta, eta=0.5):
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return beta * (1.0 - np.tanh(beta * (xt - eta)) ** 2) / den
