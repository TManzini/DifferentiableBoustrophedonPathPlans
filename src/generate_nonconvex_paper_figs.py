"""
Figure generation for the paper's non-convex extension: triangulate (ear
clipping), sum per-triangle closed-form chord lengths.

Produces three figures into src/figs_nonconvex/:

  1. nonconvex_shapes_triangulation.png
     Four edge-case concave polygons (L-shape, U-shape/horseshoe, plus/cross,
     5-point star) with their ear-clip triangulation overlaid and reflex
     vertices marked -- establishes the test shapes and shows the
     triangulation that makes the closed form work on each.

  2. nonconvex_multisegment_transect.png
     The one geometric phenomenon a convex polygon cannot produce: a single
     transect line intersecting the polygon in TWO disjoint segments (the
     U-shape's two arms). Shows that summing per-triangle chord lengths
     reproduces shapely's exact multi-segment length with no special-casing.

  3. nonconvex_smoothing_surface.png
     X-offset x Theta score surface on the plus/cross polygon, one 3D panel
     per smoothness-beta value (low to high, exact last), plus a schematic
     panel showing the polygon and transects at one representative point.

Run:  python src/generate_nonconvex_paper_figs.py
"""

import os
import sys

import numpy as np
import torch
import shapely
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from shape_utils import recenter_centroid_at_origin, rescale_polygon
from boustrophedon.triangulate import ear_clip_triangulate, signed_area
from boustrophedon.closed_form import triangulated_face_groups, rotate_triangulated_faces, \
    nonconvex_closed_form_transect_length
from boustrophedon.differentiable import get_transect_points, score_polygon_orientation_nonconvex
from boustrophedon.discrete import rotate_polygon_discrete, make_polygon_from_points

X, Y = 0, 1
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_nonconvex")
os.makedirs(OUT_DIR, exist_ok=True)

L_SHAPE = [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]]
U_SHAPE = [[0, 0], [6, 0], [6, 4], [4, 4], [4, 1.5], [2, 1.5], [2, 4], [0, 4]]
PLUS = [[2, 0], [4, 0], [4, 2], [6, 2], [6, 4], [4, 4], [4, 6], [2, 6],
        [2, 4], [0, 4], [0, 2], [2, 2]]


def star_polygon(n_points=5, r_outer=1.0, r_inner=0.4):
    pts = []
    for i in range(2 * n_points):
        r = r_outer if i % 2 == 0 else r_inner
        a = np.pi / 2 + i * np.pi / n_points
        pts.append([r * np.cos(a), r * np.sin(a)])
    return pts


STAR = star_polygon()


def reflex_vertices(coords, eps=1e-9):
    n = len(coords)
    ccw = signed_area(coords) > 0
    reflex = []
    for i in range(n):
        a, b, c = coords[(i - 1) % n], coords[i], coords[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        convex_here = (cross > eps) if ccw else (cross < -eps)
        if not convex_here:
            reflex.append(i)
    return reflex


def fig1_shapes_and_triangulation():
    shapes = [("L-shape (1 reflex vertex)", L_SHAPE),
              ("U-shape / horseshoe (2 reflex vertices)", U_SHAPE),
              ("Plus / cross (4 reflex vertices)", PLUS),
              ("5-point star (5 reflex vertices)", STAR)]

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for ax, (title, coords) in zip(axes.flat, shapes):
        coords = np.array(coords, dtype=float)
        tris = ear_clip_triangulate(coords.tolist())
        reflex = reflex_vertices(coords.tolist())

        ax.add_patch(MplPolygon(coords, closed=True, facecolor="#cde6f7",
                                edgecolor="none", zorder=1))
        for tri in tris:
            tri = np.array(tri)
            ax.add_patch(MplPolygon(tri, closed=True, facecolor="none",
                                    edgecolor="#4a90d9", linewidth=0.8, zorder=2))
        ax.plot(np.append(coords[:, 0], coords[0, 0]),
                np.append(coords[:, 1], coords[0, 1]), color="black", linewidth=1.5, zorder=3)
        ax.scatter(coords[:, 0], coords[:, 1], color="black", s=12, zorder=4)
        if reflex:
            ax.scatter(coords[reflex, 0], coords[reflex, 1], color="red", s=40,
                      zorder=5, label="reflex vertex")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{title}\n{len(tris)} triangles", fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle("Ear-clipping triangulation of non-convex test polygons", fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "nonconvex_shapes_triangulation.png")
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")


def fig2_multisegment_transect():
    theta, x = 143.0, -0.17  # found by grid search: two clean disjoint segments

    centered = recenter_centroid_at_origin(U_SHAPE)
    scaled, scale = rescale_polygon(centered, 1.0)
    discrete_polygon = make_polygon_from_points(scaled)
    rotated_polygon = rotate_polygon_discrete(discrete_polygon, theta)

    line = shapely.LineString([(x, -2), (x, 2)])
    inter = rotated_polygon.intersection(line)
    seg_lengths = sorted(float(g.length) for g in inter.geoms)
    truth_total = sum(seg_lengths)

    face_groups = triangulated_face_groups(scaled)
    rotated_faces = rotate_triangulated_faces(face_groups, torch.tensor([theta]))
    cf_total = float(nonconvex_closed_form_transect_length(
        torch.tensor(float(x)), rotated_faces, y_min=-2, y_max=2))

    fig, ax = plt.subplots(figsize=(6, 6))
    rx, ry = rotated_polygon.exterior.xy
    ax.add_patch(MplPolygon(np.column_stack([rx, ry]), closed=True,
                            facecolor="#cde6f7", edgecolor="black", linewidth=1.5, zorder=1))
    ax.axvline(x, color="gray", linestyle=":", linewidth=1, zorder=2)
    for g in inter.geoms:
        gx, gy = g.xy
        ax.plot(gx, gy, color="crimson", linewidth=4, solid_capstyle="butt", zorder=3)
    ax.scatter([x], [(inter.geoms[0].coords[0][1] + inter.geoms[0].coords[1][1]) / 2],
              s=0)  # no-op, keeps autoscale consistent

    ax.set_title(
        f"Single transect, two disjoint chord segments\n"
        f"segments = {seg_lengths[0]:.4f} + {seg_lengths[1]:.4f} = {truth_total:.4f} (shapely)\n"
        f"triangulated closed form (sum over triangles) = {cf_total:.4f}",
        fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "nonconvex_multisegment_transect.png")
    fig.savefig(out, dpi=160)
    print(f"Saved {out}  (truth={truth_total:.6f}, closed_form={cf_total:.6f}, "
          f"|diff|={abs(truth_total-cf_total):.2e})")


CACHE_PATH = os.path.join(OUT_DIR, "smoothing_surface_cache.npz")

# beta=inf (exact) last, so reading left to right / top to bottom goes from
# smoothest to sharpest, ending on the true surface. None means hard/exact.
BETA_CONFIGS = [("Beta=20", 20), ("Beta=100", 100), ("Beta=1000", 1000),
                ("Beta=10000", 10000), ("Beta=inf (exact)", None)]


def compute_smoothing_surfaces(force_recompute=False):
    """Computes (or loads a cached) (theta, x-offset) score surface for each
    beta in BETA_CONFIGS. This grid is expensive (multiple minutes), while the
    plotting/styling around it tends to need several cosmetic iterations --
    cache to disk so only a genuine grid/shape/beta change pays the recompute.
    """
    if not force_recompute and os.path.exists(CACHE_PATH):
        data = np.load(CACHE_PATH)
        thetas_mesh, x_offsets_mesh = data["thetas_mesh"], data["x_offsets_mesh"]
        all_scores = [data[f"scores_{k}"] for k in range(len(BETA_CONFIGS))]
        return thetas_mesh, x_offsets_mesh, all_scores

    transect_spacing, ppt, rescale = 0.5, 400, 1.0
    centered = recenter_centroid_at_origin(PLUS)
    scaled, scale = rescale_polygon(centered, rescale)
    face_groups = triangulated_face_groups(scaled)
    spacing = transect_spacing * scale
    transects = get_transect_points(-rescale / 2, rescale / 2, -rescale / 2,
                                    rescale / 2, spacing, ppt)

    theta_vals = np.arange(0, 180, 1.5)
    x_offset_vals = np.arange(-spacing / 2, spacing / 2, spacing / 120)
    thetas_mesh, x_offsets_mesh = np.meshgrid(theta_vals, x_offset_vals)

    def surface(beta):
        falloff = beta if beta is not None else 10000.0
        scores = np.zeros_like(thetas_mesh)
        for i in range(thetas_mesh.shape[0]):
            for j in range(thetas_mesh.shape[1]):
                th = thetas_mesh[i, j]
                xo = x_offsets_mesh[i, j]
                rotated_faces = rotate_triangulated_faces(face_groups, torch.tensor([float(th)]))
                sc = score_polygon_orientation_nonconvex(
                    transects, rotated_faces, torch.tensor([float(xo)]), 1.0, falloff,
                    use_closed_form=True, closed_form_beta=beta)
                scores[i, j] = float(sc)
        return scores

    all_scores = [surface(beta) for _, beta in BETA_CONFIGS]
    cache = {"thetas_mesh": thetas_mesh, "x_offsets_mesh": x_offsets_mesh}
    for k, scores in enumerate(all_scores):
        cache[f"scores_{k}"] = scores
    np.savez(CACHE_PATH, **cache)
    return thetas_mesh, x_offsets_mesh, all_scores


def fig3_smoothing_surface(force_recompute=False):
    thetas_mesh, x_offsets_mesh, all_scores = compute_smoothing_surfaces(force_recompute)

    centered = recenter_centroid_at_origin(PLUS)
    scaled, scale = rescale_polygon(centered, 1.0)
    discrete_polygon = make_polygon_from_points(scaled)
    spacing = 0.5 * scale
    transects = get_transect_points(-0.5, 0.5, -0.5, 0.5, spacing, 400)
    n_vertices = len(PLUS)
    n_triangles = len(ear_clip_triangulate(scaled))
    n_transects = len(transects)

    fig = plt.figure(figsize=(15, 9))
    beta_axes = []
    for k, (label, beta) in enumerate(BETA_CONFIGS):
        scores = all_scores[k]
        ax = fig.add_subplot(2, 3, k + 2, projection="3d")
        ax.plot_surface(thetas_mesh, x_offsets_mesh, scores, cmap=cm.coolwarm,
                        rstride=1, cstride=1, edgecolor="none")
        ax.set_zlim(0.55, 0.70)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlabel("Theta", fontsize=8)
        ax.set_ylabel("X-offset", fontsize=8)
        ax.set_zlabel("Score", fontsize=8)
        ax.view_init(30, 135)
        ax.tick_params(labelsize=6)
        beta_axes.append((ax, label, k))

    # 1st panel (top-left): the polygon itself at one representative (theta,
    # x_offset) point on the grid, with the transect lines drawn -- lets the
    # reader map the two surface axes directly onto a concrete geometric
    # configuration. The beta panels shift down to fill slots 2-6.
    demo_theta, demo_x_offset = 45.0, 0.0
    rotated_polygon = rotate_polygon_discrete(discrete_polygon, demo_theta)
    ax6 = fig.add_subplot(2, 3, 1)
    rx, ry = rotated_polygon.exterior.xy
    ax6.add_patch(MplPolygon(np.column_stack([rx, ry]), closed=True,
                             facecolor="#cde6f7", edgecolor="black", linewidth=1.5, zorder=1))
    # add_patch() does not trigger axis autoscaling, and axvline()'s y-span is
    # axes-relative (not data), so without explicit limits the y-axis defaults
    # to (0,1) and clips the polygon's lower half -- set limits from the
    # rotated polygon's actual bounds (with a margin) instead.
    minx, miny, maxx, maxy = rotated_polygon.bounds
    margin = 0.15
    ax6.set_xlim(minx - margin, maxx + margin)
    ax6.set_ylim(miny - margin, maxy + margin)
    for tr in transects:
        tx = float(tr[0][X]) + demo_x_offset
        # Thin gray line for the full transect, red only on the segment(s)
        # actually inside the polygon -- same convention as
        # nonconvex_multisegment_transect.png, and correctly shows multiple
        # disjoint interior segments if the transect crosses a concave notch.
        ax6.axvline(tx, color="gray", linestyle=":", linewidth=0.8, zorder=2)
        line = shapely.LineString([(tx, miny - margin), (tx, maxy + margin)])
        inter = rotated_polygon.intersection(line)
        if inter.is_empty:
            continue
        geoms = inter.geoms if hasattr(inter, "geoms") else [inter]
        for g in geoms:
            if g.length == 0:
                continue
            gx, gy = g.xy
            ax6.plot(gx, gy, color="crimson", linewidth=2.5, solid_capstyle="butt", zorder=3)
    ax6.set_title(f"Polygon at Theta={demo_theta:.0f}, X-offset={demo_x_offset:.2f}", fontsize=10)
    ax6.set_aspect("equal")
    ax6.axis("off")

    fig.suptitle(f"Score vs X-offset vs Theta on the \"Cross\" Polygon\n"
                f"{n_vertices} Vertices, {n_triangles} Triangles, {n_transects} Transects",
                fontsize=14)
    fig.tight_layout()

    # Titles are (re-)applied AFTER tight_layout() so manual positioning is not
    # overridden by its spacing pass. Use the title's y in axes-fraction coords
    # (1.0 = top of the axes bbox, lower = further down) rather than pad, which
    # behaves unintuitively on 3D axes. Even at y=1.0 the 3D panels sit lower
    # than the polygon (2D) panel's default title height, so the top row needs
    # to go above 1.0 to match it; the bottom row gets a smaller upward nudge
    # than the top row so it stays clearly lower in the cascade.
    for ax, label, k in beta_axes:
        is_bottom_row = k >= 2
        ax.set_title(label, fontsize=10, y=1 if is_bottom_row else 1.1)

    out = os.path.join(OUT_DIR, "nonconvex_smoothing_surface.png")
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    fig1_shapes_and_triangulation()
    fig2_multisegment_transect()
    fig3_smoothing_surface()
