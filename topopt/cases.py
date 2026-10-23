"""Benchmark load cases.

Every case is a :class:`Problem`: mesh size, material, loads, supports and
passive (non-designable) regions. Three are classic textbook benchmarks in
dimensionless units (E0 = 1, nu = 0.3, unit load) so their compliance can be
compared with the literature; ``cclamp`` is a physical-unit (N, mm, MPa) 2D
plane-stress analogue of a 3D-printed PLA C-clamp frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fem import node_id


@dataclass
class Problem:
    name: str
    title: str
    nelx: int
    nely: int
    F: np.ndarray                  # global load vector (ndof,)
    fixed: np.ndarray              # constrained DOF indices
    passive_solid: np.ndarray      # bool (nel,) forced to x = 1
    passive_void: np.ndarray       # bool (nel,) forced to x = 0
    E0: float = 1.0
    nu: float = 0.3
    h: float = 1.0                 # element size
    t: float = 1.0                 # out-of-plane thickness
    volfrac: float = 0.5           # suggested default
    mirror: str | None = None      # "left" -> image is mirrored about its left edge for display
    units: dict = field(default_factory=lambda: {"length": "-", "force": "-", "stress": "-", "compliance": "-"})
    yield_strength: float | None = None
    mass_density: float | None = None  # mass per unit volume (g/mm^3 for cclamp)
    description: str = ""

    @property
    def ndof(self) -> int:
        return 2 * (self.nelx + 1) * (self.nely + 1)

    @property
    def free(self) -> np.ndarray:
        return np.setdiff1d(np.arange(self.ndof), self.fixed)


def _elem_centres(nelx, nely):
    """Element centre coordinates (x right, y up) in element units, flat order."""
    ex, ey = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    return (ex + 0.5).ravel(), (nely - ey - 0.5).ravel()


def _no_passive(nelx, nely):
    return np.zeros(nelx * nely, bool), np.zeros(nelx * nely, bool)


def mbb(nelx: int = 120, nely: int = 40, **_) -> Problem:
    """Half MBB beam (symmetry on the left edge, roller at bottom-right, unit load top-left)."""
    ndof = 2 * (nelx + 1) * (nely + 1)
    F = np.zeros(ndof)
    F[2 * node_id(nely, 0, 0) + 1] = -1.0
    left_x = 2 * node_id(nely, 0, np.arange(nely + 1))
    roller_y = 2 * node_id(nely, nelx, nely) + 1
    fixed = np.union1d(left_x, [roller_y])
    ps, pv = _no_passive(nelx, nely)
    return Problem("mbb", "MBB beam (half, symmetric)", nelx, nely, F, fixed, ps, pv,
                   volfrac=0.5, mirror="left",
                   description="Simply supported beam with a central point load; only the right half is modelled.")


def cantilever(nelx: int = 80, nely: int = 40, **_) -> Problem:
    """Short cantilever: left edge clamped, unit downward load at mid-height of the right edge."""
    ndof = 2 * (nelx + 1) * (nely + 1)
    F = np.zeros(ndof)
    F[2 * node_id(nely, nelx, nely // 2) + 1] = -1.0
    left = node_id(nely, 0, np.arange(nely + 1))
    fixed = np.union1d(2 * left, 2 * left + 1)
    ps, pv = _no_passive(nelx, nely)
    return Problem("cantilever", "Cantilever beam", nelx, nely, F, fixed, ps, pv, volfrac=0.4,
                   description="Clamped left edge, point load at the middle of the free end.")


def lbracket(nelx: int = 100, nely: int | None = None, **_) -> Problem:
    """L-bracket: top of the vertical arm clamped, downward load on the tip of the horizontal arm."""
    nely = nelx if nely is None else nely
    ndof = 2 * (nelx + 1) * (nely + 1)
    xc, yc = _elem_centres(nelx, nely)
    arm_x = int(round(0.4 * nelx))   # width of vertical arm
    arm_y = int(round(0.4 * nely))   # height of horizontal arm
    pv = (xc > arm_x) & (yc > arm_y)
    ps = np.zeros_like(pv)
    fixed_nodes = node_id(nely, np.arange(arm_x + 1), 0)
    fixed = np.union1d(2 * fixed_nodes, 2 * fixed_nodes + 1)
    # Load spread over a few nodes on the right edge, just below the re-entrant level.
    iy_top = nely - arm_y
    n_load = max(2, nely // 25)
    iys = np.arange(iy_top, iy_top + n_load)
    F = np.zeros(ndof)
    F[2 * node_id(nely, nelx, iys) + 1] = -1.0 / len(iys)
    return Problem("lbracket", "L-bracket", nelx, nely, F, fixed, ps, pv, volfrac=0.4,
                   description="Classic stress-concentration benchmark with a re-entrant corner (void region passive).")


def cclamp(nelx: int = 90, nely: int | None = None, *, width: float = 60.0, height: float = 80.0,
           spine: float = 15.0, jaw: float = 15.0, pad: float = 10.0, thickness: float = 5.0,
           fillet: float = 6.0,
           load: float = 100.0, **_) -> Problem:
    """2D plane-stress analogue of a PLA C-clamp frame (units: N, mm, MPa).

    Design domain: a ``width`` x ``height`` rectangle minus the throat of the C
    (passive void). The workpiece pushes the two jaws apart with ``load`` newtons:

    * lower jaw (anvil): nodes on the anvil face are supported vertically
      (u_y = 0, plus u_x = 0 at the pad centre to remove rigid-body sliding);
    * upper jaw (screw boss): ``load`` N upward, distributed over the pad.

    Because the load is self-equilibrated through the frame, this is equivalent
    to the free-floating clamp. Thin solid pads under both contact faces are
    passive so the load always has material to act on.
    PLA: E = 3500 MPa, nu = 0.36, yield ~ 50 MPa, density 1.24 g/cm^3.
    """
    h = width / nelx
    nely = int(round(height / h)) if nely is None else nely
    ndof = 2 * (nelx + 1) * (nely + 1)
    xc, yc = _elem_centres(nelx, nely)
    xc, yc = xc * h, yc * h
    throat = (xc > spine) & (yc > jaw) & (yc < height - jaw)
    # Round the two inner corners of the throat (fillet radius ``fillet``) to avoid
    # an artificial re-entrant-corner stress singularity.
    for cy in (jaw + fillet, height - jaw - fillet):
        corner = (xc < spine + fillet) & (np.abs(yc - cy) < fillet) & ((yc - cy) * (cy - height / 2) > 0)
        throat &= ~(corner & (np.hypot(xc - spine - fillet, yc - cy) > fillet))
    pad_t = 2.0
    pad_x = xc > width - pad
    lower_pad = pad_x & (yc < jaw) & (yc > jaw - pad_t)
    upper_pad = pad_x & (yc > height - jaw) & (yc < height - jaw + pad_t)
    ps = (lower_pad | upper_pad) & ~throat
    pv = throat

    ix0 = int(round((width - pad) / h))
    ixs = np.arange(ix0, nelx + 1)
    iy_anvil = nely - int(round(jaw / h))      # node row of the anvil face (y = jaw)
    iy_boss = int(round(jaw / h))              # node row of the upper jaw face (y = H - jaw)
    anvil = node_id(nely, ixs, iy_anvil)
    fixed = np.union1d(2 * anvil + 1, [2 * anvil[len(anvil) // 2]])
    w = np.ones(len(ixs))
    w[[0, -1]] = 0.5                           # consistent nodal loads for uniform traction
    F = np.zeros(ndof)
    F[2 * node_id(nely, ixs, iy_boss) + 1] = load * w / w.sum()
    return Problem("cclamp", "C-clamp frame (PLA, 2D plane stress)", nelx, nely, F, fixed, ps, pv,
                   E0=3500.0, nu=0.36, h=h, t=thickness, volfrac=0.326,
                   units={"length": "mm", "force": "N", "stress": "MPa", "compliance": "N*mm"},
                   yield_strength=50.0, mass_density=1.24e-3,
                   description=(f"{width:g} x {height:g} mm C-frame, {thickness:g} mm thick, "
                                f"{load:g} N clamp load prying the jaws apart."))


CASES = {"mbb": mbb, "cantilever": cantilever, "lbracket": lbracket, "cclamp": cclamp}

DEFAULT_MESH = {"mbb": (120, 40), "cantilever": (80, 40), "lbracket": (100, 100), "cclamp": (90, 120)}


def make_case(name: str, nelx: int | None = None, nely: int | None = None, **kw) -> Problem:
    """Build a case by name. ``nely`` may be omitted for cases with a fixed aspect ratio."""
    if name not in CASES:
        raise ValueError(f"unknown case {name!r}; choose from {sorted(CASES)}")
    dx, dy = DEFAULT_MESH[name]
    nelx = dx if nelx is None else int(nelx)
    if nely is None and name in ("mbb", "cantilever"):
        nely = max(1, int(round(nelx * dy / dx)))
    return CASES[name](nelx=nelx, nely=nely, **kw)
