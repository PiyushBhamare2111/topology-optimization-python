"""SIMP compliance minimisation with an optimality-criteria (OC) update.

    minimise    c(x) = F^T u(x)
    subject to  K(x) u = F
                (1/|Omega|) sum_e x_phys,e  <=  volfrac
                0 <= x_e <= 1,  passive elements fixed to 0 or 1

The structure follows the 88-line MATLAB code of Andreassen et al. (2011),
rewritten in vectorised NumPy/SciPy with passive regions, three filter options
and progress callbacks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import filters
from .cases import Problem
from .fem import FEModel, to_image


@dataclass
class Settings:
    volfrac: float = 0.5
    penal: float = 3.0
    rmin: float = 1.5
    filter: str = "density"        # "density" | "sensitivity" | "heaviside"
    move: float = 0.2
    maxiter: int = 200
    tol: float = 0.01              # stop when max |x_new - x| < tol
    beta_max: float = 32.0         # Heaviside continuation limit
    eta: float = 0.5
    n_frames: int = 60             # density snapshots kept for the GIF


@dataclass
class Result:
    problem: Problem
    settings: Settings
    x: np.ndarray                  # final physical densities, flat
    u: np.ndarray
    von_mises: np.ndarray          # element von Mises stress, flat
    compliance: list = field(default_factory=list)
    volume: list = field(default_factory=list)
    change: list = field(default_factory=list)
    frames: list = field(default_factory=list)   # (iteration, flat density)
    iterations: int = 0
    runtime: float = 0.0
    converged: bool = False
    final_compliance: float = float("nan")   # compliance of the returned design
    final_volume: float = float("nan")

    @property
    def image(self) -> np.ndarray:
        return to_image(self.x, self.problem.nelx, self.problem.nely)


    def summary(self) -> dict:
        p, s = self.problem, self.settings
        solid = self.x > 0.5
        out = {
            "case": p.name, "nelx": p.nelx, "nely": p.nely, "volfrac": s.volfrac,
            "penal": s.penal, "rmin": s.rmin, "filter": s.filter,
            "compliance": float(self.final_compliance), "volume": float(self.final_volume),
            "iterations": self.iterations, "converged": self.converged,
            "runtime_s": round(self.runtime, 2),
            "max_von_mises": float(self.von_mises[solid].max()) if solid.any() else 0.0,
            "grayness": float(np.mean(4 * self.x * (1 - self.x))),
            "units": p.units,
        }
        if p.mass_density is not None:
            out["mass_g"] = float(self.x.sum() * p.h**2 * p.t * p.mass_density)
        if p.yield_strength is not None and out["max_von_mises"] > 0:
            out["yield_strength"] = p.yield_strength
            out["safety_factor"] = p.yield_strength / out["max_von_mises"]
        return out


def optimize(problem: Problem, settings: Settings | None = None, callback=None) -> Result:
    """Run the optimisation. ``callback(it, xPhys, c, vol, change)`` is called every iteration;
    if it returns ``False`` the run stops early."""
    s = settings or Settings(volfrac=problem.volfrac)
    if s.filter not in ("density", "sensitivity", "heaviside"):
        raise ValueError("filter must be 'density', 'sensitivity' or 'heaviside'")
    t0 = time.perf_counter()
    p = problem
    fem = FEModel(p.nelx, p.nely, p.E0, p.nu, p.h, p.t)
    H, Hs = filters.build_filter(p.nelx, p.nely, s.rmin)
    free = p.free

    solid, void = p.passive_solid, p.passive_void
    active = ~(solid | void)
    counted = ~void                  # volume is measured over the non-void domain
    n_counted = counted.sum()

    def enforce(x):
        x = x.copy()
        x[solid] = 1.0
        x[void] = 0.0
        return x

    beta = 1.0
    x = enforce(np.full(fem.nel, s.volfrac))

    def physical(x, beta):
        if s.filter == "sensitivity":
            xt = x
        else:
            xt = filters.density_filter(H, Hs, x)
        xp = filters.heaviside(xt, beta, s.eta) if s.filter == "heaviside" else xt
        return enforce(xp), xt

    xPhys, xTilde = physical(x, beta)
    dv_raw = np.where(counted, 1.0, 0.0) / n_counted
    res = Result(p, s, xPhys, np.zeros(fem.ndof), np.zeros(fem.nel))
    frame_every = max(1, s.maxiter // max(1, s.n_frames))
    loop_beta = 0
    it = 0
    change = 1.0
    converged = False

    while it < s.maxiter:
        it += 1
        loop_beta += 1
        u = fem.solve(xPhys, s.penal, p.F, free)
        ce = fem.element_energy(u)
        E = fem.young(xPhys, s.penal)
        c = float(np.sum(E * ce))
        dc = -s.penal * (fem.E0 - fem.Emin) * xPhys ** (s.penal - 1) * ce
        dv = dv_raw.copy()

        # Chain rule back to the design variables
        if s.filter == "sensitivity":
            dc = filters.sensitivity_filter(H, Hs, x, dc)
        else:
            if s.filter == "heaviside":
                dx = filters.heaviside_grad(xTilde, beta, s.eta)
                dc, dv = dc * dx, dv * dx
            dc = filters.density_filter_chain(H, Hs, dc)
            dv = filters.density_filter_chain(H, Hs, dv)
        dc[~active] = 0.0
        dc = np.minimum(dc, 0.0)     # compliance sensitivities are non-positive
        dv = np.maximum(dv, 1e-12)

        # Optimality-criteria update: bisection on the Lagrange multiplier
        l1, l2 = 0.0, 1e9 * max(1.0, abs(c))
        lo, hi = np.maximum(0.0, x - s.move), np.minimum(1.0, x + s.move)
        while (l2 - l1) / (l1 + l2 + 1e-300) > 1e-4:
            lmid = 0.5 * (l1 + l2)
            xnew = enforce(np.clip(x * np.sqrt(-dc / dv / lmid), lo, hi))
            xPnew, _ = physical(xnew, beta)
            if xPnew[counted].mean() > s.volfrac:
                l1 = lmid
            else:
                l2 = lmid
        change = float(np.max(np.abs(xnew - x)))
        x = xnew
        xPhys, xTilde = physical(x, beta)
        vol = float(xPhys[counted].mean())

        res.compliance.append(c)
        res.volume.append(vol)
        res.change.append(change)
        if it == 1 or it % frame_every == 0:
            res.frames.append((it, xPhys.copy()))
        if callback is not None and callback(it, xPhys, c, vol, change) is False:
            break

        if s.filter == "heaviside" and beta < s.beta_max and (loop_beta >= 50 or change <= s.tol):
            beta *= 2.0
            loop_beta = 0
            xPhys, xTilde = physical(x, beta)
            continue
        if change < s.tol and (s.filter != "heaviside" or beta >= s.beta_max):
            converged = True
            break

    # Final analysis of the final design (compliance/stress consistent with the reported x)
    u = fem.solve(xPhys, s.penal, p.F, free)
    res.final_compliance = float(p.F @ u)
    res.final_volume = float(xPhys[counted].mean())
    res.frames.append((it, xPhys.copy()))
    res.x, res.u = xPhys, u
    res.von_mises = fem.von_mises(u, xPhys, s.penal)
    res.iterations = it
    res.converged = converged
    res.runtime = time.perf_counter() - t0
    return res
