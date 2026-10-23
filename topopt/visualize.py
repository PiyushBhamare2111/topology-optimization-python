"""Plots, animations and renders (dark theme, accent #ff8a3d)."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

from .export import mirror_image  # noqa: E402
from .fem import to_image  # noqa: E402

BG = "#0b0c0f"
PANEL = "#14161b"
GRID = "#2a2d35"
TEXT = "#e8e6e3"
MUTED = "#9a9ca3"
ACCENT = "#ff8a3d"

#: void -> solid: near-black to the accent orange (single-hue sequential ramp)
DENSITY_CMAP = LinearSegmentedColormap.from_list("density", [BG, "#5a2c10", ACCENT, "#ffd2ad"], N=256)
STRESS_CMAP = plt.get_cmap("inferno")


def _style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 11,
        "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    })


def display_image(result, flat=None, full=True) -> np.ndarray:
    """(rows, cols) image of a flat element field, mirrored for symmetric half models."""
    p = result.problem
    img = to_image(result.x if flat is None else flat, p.nelx, p.nely)
    return mirror_image(img, p.mirror) if full else img


def density_rgb(img: np.ndarray, scale: int = 1) -> np.ndarray:
    rgb = (DENSITY_CMAP(np.clip(img, 0, 1))[..., :3] * 255).astype(np.uint8)
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, 0), scale, 1)
    return rgb


def _extent(result, img):
    h = result.problem.h
    return [0, img.shape[1] * h, 0, img.shape[0] * h]


def plot_density(result, path=None, title=None, ax=None):
    _style()
    img = display_image(result)
    own = ax is None
    if own:
        aspect = img.shape[0] / img.shape[1]
        fig, ax = plt.subplots(figsize=(8, max(2.5, 8 * aspect + 0.8)))
    ax.imshow(img, cmap=DENSITY_CMAP, vmin=0, vmax=1, extent=_extent(result, img), interpolation="nearest")
    ax.grid(False)
    ax.set_title(title or f"{result.problem.title}  |  c = {result.final_compliance:.4g}", color=TEXT)
    unit = result.problem.units.get("length", "-")
    ax.set_xlabel(f"x [{unit}]" if unit != "-" else "x [elements]")
    ax.set_ylabel(f"y [{unit}]" if unit != "-" else "y [elements]")
    if own:
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=150)
            plt.close(fig)
        return fig


def plot_convergence(result, path=None):
    """Compliance and volume fraction vs iteration (two panels, shared x axis)."""
    _style()
    it = np.arange(1, len(result.compliance) + 1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(it, result.compliance, color=ACCENT, lw=2)
    a1.set_ylabel(f"compliance [{result.problem.units['compliance']}]")
    a1.set_yscale("log")
    a1.set_title(f"Convergence - {result.problem.title}")
    a1.annotate(f"final {result.final_compliance:.4g}", xy=(it[-1], result.compliance[-1]),
                xytext=(-10, 18), textcoords="offset points", ha="right", color=TEXT)
    a2.plot(it, result.volume, color=TEXT, lw=2)
    a2.axhline(result.settings.volfrac, color=MUTED, ls="--", lw=1)
    a2.set_ylabel("volume fraction")
    a2.set_xlabel("iteration")
    lo = min(min(result.volume), result.settings.volfrac)
    hi = max(max(result.volume), result.settings.volfrac)
    pad = max(0.02, 0.2 * (hi - lo))
    a2.set_ylim(lo - pad, hi + pad)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def plot_stress(result, path=None, ax=None, clip_percentile=99.5):
    """von Mises stress of the final design; void (x < 0.5) is blanked."""
    _style()
    p = result.problem
    vm = display_image(result, result.von_mises)
    rho = display_image(result)
    vmax = np.percentile(vm[rho > 0.5], clip_percentile) if (rho > 0.5).any() else 1.0
    masked = np.ma.masked_where(rho < 0.5, vm)
    own = ax is None
    if own:
        aspect = vm.shape[0] / vm.shape[1]
        fig, ax = plt.subplots(figsize=(8.8, max(2.5, 8 * aspect + 0.8)))
    cmap = STRESS_CMAP.with_extremes(bad=BG)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax, extent=_extent(result, vm), interpolation="nearest")
    ax.grid(False)
    unit = p.units.get("stress", "-")
    peak = result.von_mises[result.x > 0.5].max() if (result.x > 0.5).any() else 0
    ax.set_title(f"von Mises stress  |  peak {peak:.3g} {unit if unit != '-' else ''}".strip())
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(f"von Mises [{unit}]" if unit != "-" else "von Mises (dimensionless)")
    cb.outline.set_edgecolor(GRID)
    if own:
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=150)
            plt.close(fig)
        return fig


def gif_frames(result, max_width=640):
    imgs = []
    for _, flat in result.frames:
        img = display_image(result, flat)
        scale = max(1, max_width // img.shape[1])
        imgs.append(Image.fromarray(density_rgb(img, scale)))
    return imgs


def save_gif(result, path, max_width=640, ms=90):
    frames = gif_frames(result, max_width)
    durations = [ms] * (len(frames) - 1) + [1500]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)


def gif_bytes(result, max_width=480, ms=90) -> bytes:
    buf = io.BytesIO()
    frames = gif_frames(result, max_width)
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=[ms] * (len(frames) - 1) + [1500], loop=0, optimize=True)
    return buf.getvalue()


def render_stl(triangles, path=None, title="Exported STL", ax=None, elev=28, azim=-60, figsize=(8, 6), zoom=1.35):
    """Shaded 3D render of a triangle array with matplotlib."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    _style()
    own = ax is None
    if own:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    n = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    light = np.array([-0.4, -0.6, 0.9])
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
    base = np.array(matplotlib.colors.to_rgb(ACCENT))
    colors = np.clip(base[None, :] * shade[:, None], 0, 1)
    pc = Poly3DCollection(triangles, facecolors=colors, edgecolors="none", linewidths=0)
    ax.add_collection3d(pc)
    mins, maxs = triangles.reshape(-1, 3).min(0), triangles.reshape(-1, 3).max(0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(maxs - mins, zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, color=TEXT)
    if own:
        fig.subplots_adjust(0, 0, 1, 0.94)
        if path:
            fig.savefig(path, dpi=150)
            plt.close(fig)
        return fig
