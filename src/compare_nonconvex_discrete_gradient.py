"""
Non-convex counterpart to compare_discrete_gradient.py: validates the
triangulated (ear-clipping) length implementations -- both the trapezoidal sum
(approximate_nonconvex_transect_length) and the closed-form sum
(nonconvex_closed_form_transect_length), hard and smoothed -- against the
discrete/shapely score on randomly generated CONCAVE polygons.

Random non-convex polygon generation: vertices are placed at strictly
increasing angles around the origin with i.i.d. random radii ("star-shaped"
about the origin). Strictly increasing angle guarantees a simple
(non-self-intersecting) polygon for any radii, and radius variance makes reflex
(concave) vertices likely -- this is the non-convex analogue of
random_convex_polygon (convex hull of random points) used in
compare_discrete_gradient.py / compare_closed_form_scores.py.

Run:  python src/compare_nonconvex_discrete_gradient.py
"""

import os
import sys
import time

import numpy as np
import torch
import shapely

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from shape_utils import recenter_centroid_at_origin, rescale_polygon
from boustrophedon.triangulate import signed_area, ear_clip_triangulate
from boustrophedon.closed_form import triangulated_face_groups, rotate_triangulated_faces
from boustrophedon.differentiable import get_transect_points, score_polygon_orientation_nonconvex
from boustrophedon.discrete import (
    get_transects, rotate_polygon_discrete, discrete_score_polygon_orientation,
    make_polygon_from_points)

X, Y = 0, 1


def random_nonconvex_polygon(rng, n_points, min_radius=0.2, max_radius=1.0, max_attempts=50):
    """Random simple polygon, vertices placed at strictly increasing angles
    around the origin with i.i.d. random radii. Varying radius makes reflex
    (concave) vertices likely.

    NOTE: angularly-sorted vertices do NOT guarantee a simple polygon in
    general -- if the radii vary enough, a chord between two angularly distant
    vertices (especially the wrap-around closing edge, when one angular gap is
    large) can swing close to the origin and cross an unrelated edge. So this
    explicitly validates simplicity (shapely) and resamples on failure, the
    same rejection-sampling pattern compare_discrete_gradient.py already uses
    for its (convex) random polygon generator.
    """
    for _ in range(max_attempts):
        angles = np.sort(rng.random(n_points) * 2 * np.pi)
        angles = angles + np.arange(n_points) * 1e-9  # break exact ties
        radii = rng.uniform(min_radius, max_radius, n_points)
        coords = [[float(r * np.cos(a)), float(r * np.sin(a))] for a, r in zip(angles, radii)]
        ring = shapely.LinearRing(coords + [coords[0]])
        if shapely.is_simple(ring):
            return coords
    raise RuntimeError("Failed to generate a simple random non-convex polygon.")


def is_concave(coords, eps=1e-9):
    """True if any vertex is reflex (i.e. the polygon is not convex)."""
    n = len(coords)
    ccw = signed_area(coords) > 0
    for i in range(n):
        a, b, c = coords[(i - 1) % n], coords[i], coords[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        convex_here = (cross > eps) if ccw else (cross < -eps)
        if not convex_here:
            return True
    return False


def scores_for_polygon(polygon_coords, theta, x_offset, ppt, temp,
                       transect_spacing=0.5, polygon_rescale=1.0):
    beta = temp  # tie smoothness beta to the same value as the falloff temp
    centered = recenter_centroid_at_origin(polygon_coords)
    scaled, scale = rescale_polygon(centered, polygon_rescale)

    face_groups = triangulated_face_groups(scaled)
    discrete_polygon = make_polygon_from_points(scaled)

    grad_transects = get_transect_points(
        -polygon_rescale / 2, polygon_rescale / 2,
        -polygon_rescale / 2, polygon_rescale / 2, transect_spacing * scale, ppt)
    disc_transects = get_transects(
        -polygon_rescale / 2, polygon_rescale / 2,
        -polygon_rescale / 2, polygon_rescale / 2, transect_spacing * scale)

    rotated_faces = rotate_triangulated_faces(face_groups, torch.tensor([theta]))
    rotated_polygon = rotate_polygon_discrete(discrete_polygon, theta)
    x_off_t = torch.tensor([x_offset * scale])

    t0 = time.perf_counter()
    discrete = float(discrete_score_polygon_orientation(disc_transects, rotated_polygon, x_offset * scale))
    t_discrete = time.perf_counter() - t0

    t0 = time.perf_counter()
    trap = float(score_polygon_orientation_nonconvex(
        grad_transects, rotated_faces, x_off_t, temp, temp, use_closed_form=False))
    t_trap = time.perf_counter() - t0

    t0 = time.perf_counter()
    hard_cf = float(score_polygon_orientation_nonconvex(
        grad_transects, rotated_faces, x_off_t, temp, temp, use_closed_form=True, closed_form_beta=None))
    t_hard = time.perf_counter() - t0

    soft_cf = float(score_polygon_orientation_nonconvex(
        grad_transects, rotated_faces, x_off_t, temp, temp, use_closed_form=True, closed_form_beta=beta))

    return discrete, trap, hard_cf, soft_cf, t_trap, t_hard, t_discrete


def main():
    rng = np.random.default_rng(0)
    num_polygons = 60
    ppt_range = [100, 1000]
    temp_range = [100, 1000, 10000]
    # closed_form_beta must be tied to the SAME value as transect_falloff_temp
    # (as optimize_gd.py / demo_smooth_landscape.py do): a sharp falloff sigmoid
    # amplifies even the tiny, expected O(1/beta) near-zero ghost-length residual
    # of the soft chord into a much bigger "shown" count error if beta is left
    # low/fixed while the falloff temperature is swept up independently.

    # --- 0. sanity: confirm the generator actually produces concave shapes ---
    n_concave = 0
    n_triangles_total = 0
    sample_polys = []
    for _ in range(200):
        n = rng.integers(5, 12)
        coords = random_nonconvex_polygon(rng, n)
        sample_polys.append(coords)
        if is_concave(coords):
            n_concave += 1
    print(f"Random polygon generator: {n_concave}/200 ({n_concave/2:.0f}%) sampled "
          f"polygons are concave (have >=1 reflex vertex).")

    # partition sanity check (area of triangles == polygon area)
    max_area_diff = 0.0
    for coords in sample_polys[:50]:
        tris_area = sum(abs(signed_area(t)) for t in ear_clip_triangulate(coords))
        poly_area = abs(signed_area(coords))
        n_triangles_total += 1
        max_area_diff = max(max_area_diff, abs(tris_area - poly_area))
    print(f"Triangulation partition check over 50 polygons: max |area diff| = {max_area_diff:.2e}\n")

    # --- 1. trapezoidal-nonconvex error vs discrete, across Temp x PPT ---
    trap_errors = {p: {t: [] for t in temp_range} for p in ppt_range}
    hard_errors = []
    soft_errors = {t: [] for t in temp_range}
    timings = {"trap": [], "hard": [], "discrete": []}

    n = 0
    while n < num_polygons:
        n_pts = rng.integers(5, 12)
        coords = random_nonconvex_polygon(rng, n_pts)
        theta = float(rng.random() * 360.0)
        x_offset = float(rng.random() * 0.5 - 0.25)
        try:
            for ppt in ppt_range:
                for temp in temp_range:
                    d, tr, hard, soft, t_tr, t_h, t_d = scores_for_polygon(
                        coords, theta, x_offset, ppt, temp)
                    if any(np.isnan(v) for v in (d, tr, hard, soft)):
                        raise ValueError("NaN encountered")
                    trap_errors[ppt][temp].append(abs(tr - d))
                    if ppt == ppt_range[-1]:
                        soft_errors[temp].append(abs(soft - d))
                        if temp == temp_range[-1]:
                            hard_errors.append(abs(hard - d))
                            timings["trap"].append(t_tr)
                            timings["hard"].append(t_h)
                            timings["discrete"].append(t_d)
        except Exception:
            continue
        n += 1

    print(f"Score comparison over N={num_polygons} random CONCAVE polygons "
          f"(triangulated trapezoidal vs. triangulated closed form vs. discrete truth)")
    print("=" * 72)
    print("Trapezoidal-nonconvex mean |error| vs discrete (Temp x PPT):")
    header = "Temp \\ PPT".ljust(14) + "".join(f"{p:>14}" for p in ppt_range)
    print(header)
    for temp in temp_range:
        row = f"{temp}".ljust(14)
        for ppt in ppt_range:
            row += f"{np.mean(trap_errors[ppt][temp]):>14.6f}"
        print(row)

    print(f"\nClosed-form-nonconvex (smoothed, beta=falloff_temp tied together) mean |error| vs discrete:")
    for temp in temp_range:
        print(f"  beta=falloff_temp={temp:<8} mean|err|={np.mean(soft_errors[temp]):.6f}")

    print(f"\nClosed-form-nonconvex (hard / exact chord) mean |error| vs discrete:")
    print(f"  mean|err|={np.mean(hard_errors):.6f}   max|err|={np.max(hard_errors):.6f}")
    print("=" * 72)

    print(f"\nPer-polygon timing (PPT={ppt_range[-1]}, Temp={temp_range[-1]}):")
    print(f"  trapezoidal-nonconvex : {np.mean(timings['trap'])*1e3:7.2f} ms")
    print(f"  closed-form-nonconvex : {np.mean(timings['hard'])*1e3:7.2f} ms")
    print(f"  discrete (shapely)    : {np.mean(timings['discrete'])*1e3:7.2f} ms")
    print(f"  speedup (trap -> closed form): "
          f"{np.mean(timings['trap'])/np.mean(timings['hard']):.1f}x")


if __name__ == "__main__":
    main()
