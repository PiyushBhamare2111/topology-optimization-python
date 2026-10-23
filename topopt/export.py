"""Export a 2D density field to a watertight, extruded STL for 3D printing.

Pipeline: density image -> (optional bilinear upsampling) -> marching squares at
the 0.5 iso-level -> per-cell solid polygons (caps) + contour segments (side
walls) -> extrude by the part thickness -> binary STL via numpy-stl.

Each marching-squares cell contributes the convex polygon of its "inside"
region (corner points above the iso-level plus edge crossings). Neighbouring
cells compute shared edge crossings from the same two samples in the same
order, so vertices coincide exactly and the mesh closes. Saddle cells use the
"connected" interpretation.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def _crossing(pa, va, pb, vb, iso):
    # Canonical order (by coordinates) so both neighbouring cells get identical floats.
    if (pa[0], pa[1]) > (pb[0], pb[1]):
        pa, va, pb, vb = pb, vb, pa, va
    s = (iso - va) / (vb - va)
    return (pa[0] + s * (pb[0] - pa[0]), pa[1] + s * (pb[1] - pa[1]))


def marching_squares(img: np.ndarray, iso: float = 0.5, h: float = 1.0):
    """Return (polygons, segments) of the region ``img > iso``.

    ``img`` is (rows, cols) with row 0 at the top; samples sit at pixel centres.
    Output coordinates are physical (x right, y up) with pixel pitch ``h``; the
    image is padded with zeros so every contour is closed. Polygons are CCW.
    Segments are oriented with the solid on their left.
    """
    a = np.pad(np.asarray(img, float), 1, constant_values=0.0)
    if np.any(a == iso):            # avoid degenerate zero-length edges at exact iso values
        iso = iso + 1e-7
    rows, cols = a.shape
    # sample (r, c) -> physical point (x, y), pixel centres of the unpadded image at 0.5h ...
    X = (np.arange(cols) - 0.5) * h
    Y = (rows - 2 - np.arange(rows) + 0.5) * h
    polys, segs = [], []
    inside = a > iso
    for r in range(rows - 1):
        for c in range(cols - 1):
            # corners CCW: LL, LR, UR, UL  (row r+1 is lower)
            idx = ((r + 1, c), (r + 1, c + 1), (r, c + 1), (r, c))
            ins = [inside[i] for i in idx]
            n_in = sum(ins)
            if n_in == 0:
                continue
            pts = [(X[i[1]], Y[i[0]]) for i in idx]
            vals = [a[i] for i in idx]
            poly, is_cross = [], []
            for k in range(4):
                if ins[k]:
                    poly.append(pts[k])
                    is_cross.append(False)
                k2 = (k + 1) % 4
                if ins[k] != ins[k2]:
                    poly.append(_crossing(pts[k], vals[k], pts[k2], vals[k2], iso))
                    is_cross.append(True)
            polys.append(poly)
            if n_in < 4:
                m = len(poly)
                for k in range(m):
                    if is_cross[k] and is_cross[(k + 1) % m]:
                        segs.append((poly[k], poly[(k + 1) % m]))
    return polys, segs


def extrude(polys, segs, thickness: float) -> np.ndarray:
    """Triangles (n, 3, 3) of the extruded solid with outward normals."""
    tris = []
    for poly in polys:
        p0 = poly[0]
        for k in range(1, len(poly) - 1):
            p1, p2 = poly[k], poly[k + 1]
            tris.append([(*p0, thickness), (*p1, thickness), (*p2, thickness)])  # top, CCW
            tris.append([(*p0, 0.0), (*p2, 0.0), (*p1, 0.0)])                    # bottom, CW
    for p, q in segs:
        # solid on the left of p->q, so the outward side is the right
        tris.append([(*p, 0.0), (*q, 0.0), (*q, thickness)])
        tris.append([(*p, 0.0), (*q, thickness), (*p, thickness)])
    return np.asarray(tris, float)


def density_to_triangles(img: np.ndarray, h: float = 1.0, thickness: float = 1.0,
                         iso: float = 0.5, upsample: int = 2) -> np.ndarray:
    """Density image (row 0 = top) -> triangle array of the extruded part."""
    img = np.asarray(img, float)
    if upsample and upsample > 1:
        img = zoom(img, upsample, order=1)
        h = h / upsample
    polys, segs = marching_squares(img, iso, h)
    return extrude(polys, segs, thickness)


def save_stl(triangles: np.ndarray, path) -> None:
    from stl import mesh as stl_mesh

    m = stl_mesh.Mesh(np.zeros(len(triangles), dtype=stl_mesh.Mesh.dtype))
    m.vectors[:] = triangles
    m.update_normals()
    m.save(str(path))


def mirror_image(img: np.ndarray, mirror: str | None) -> np.ndarray:
    """Reconstruct the full part from a symmetric half model."""
    if mirror == "left":
        return np.hstack([img[:, ::-1], img])
    return img


def export_stl(result, path, thickness: float | None = None, upsample: int = 2, full: bool = True):
    """Write the final design of an optimisation :class:`~topopt.optimizer.Result` to STL.

    Thickness defaults to the problem thickness for physical cases, otherwise
    to 10 % of the part height. Returns the triangle array.
    """
    p = result.problem
    img = result.image
    if full:
        img = mirror_image(img, p.mirror)
    if thickness is None:
        thickness = p.t if p.units.get("length") == "mm" else 0.1 * p.nely * p.h
    tris = density_to_triangles(img, p.h, thickness, 0.5, upsample)
    save_stl(tris, path)
    return tris


def is_watertight(triangles: np.ndarray, decimals: int = 9) -> bool:
    """Every undirected edge is shared by exactly two triangles, with opposite orientation."""
    from collections import Counter

    v = np.round(triangles, decimals)
    directed = Counter()
    for t in v:
        for i in range(3):
            a, b = tuple(t[i]), tuple(t[(i + 1) % 3])
            directed[(a, b)] += 1
    for (a, b), n in directed.items():
        if n != 1 or directed.get((b, a), 0) != 1:
            return False
    return True


def triangle_volume(triangles: np.ndarray) -> float:
    """Enclosed volume via the divergence theorem (positive for outward normals)."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
