"""
Compare full fitness *scores* (Equation 1) computed three ways:

    discrete            -- shapely chord lengths            (ground truth)
    differentiable+trap -- trapezoidal approx of Eq. 6      (use_closed_form=False)
    differentiable+cf   -- closed-form chord length         (use_closed_form=True)

This extends src/compare_discrete_gradient.py from comparing transect *lengths*
to comparing the end-to-end scores, and shows the closed form reproduces the
discrete score at least as well as the trapezoidal path -- with no temperature
or points-per-transect to tune.

Run:  python src/compare_closed_form_scores.py
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
from boustrophedon.differentiable import (
    generate_polygon_weights_and_biases_from_coordinates,
    get_transect_points, rotate_polygon, score_polygon_orientation)
from boustrophedon.discrete import (
    get_transects, rotate_polygon_discrete, discrete_score_polygon_orientation,
    make_polygon_from_points)

X, Y = 0, 1


def random_convex_polygon(rng, n_points=12):
    """CCW convex hull of random points (interior on the all-v_i>0 side)."""
    pts = rng.random((n_points, 2))
    hull = shapely.geometry.polygon.orient(
        shapely.MultiPoint(pts).convex_hull, sign=1.0)
    return [list(c) for c in hull.exterior.coords[:-1]]


def scores_for_polygon(polygon_coords, theta, x_offset, ppt, temp,
                       transect_spacing=0.5, polygon_rescale=1.0):
    centered = recenter_centroid_at_origin(polygon_coords)
    scaled, scale = rescale_polygon(centered, polygon_rescale)

    line_components = generate_polygon_weights_and_biases_from_coordinates(scaled)
    discrete_polygon = make_polygon_from_points(scaled)

    grad_transects = get_transect_points(
        -polygon_rescale / 2, polygon_rescale / 2,
        -polygon_rescale / 2, polygon_rescale / 2, transect_spacing * scale, ppt)
    disc_transects = get_transects(
        -polygon_rescale / 2, polygon_rescale / 2,
        -polygon_rescale / 2, polygon_rescale / 2, transect_spacing * scale)

    w_rot, b_rot = rotate_polygon(line_components, torch.tensor([theta], requires_grad=True))
    rotated_polygon = rotate_polygon_discrete(discrete_polygon, theta)
    x_off_t = torch.tensor([x_offset * scale])

    discrete = float(discrete_score_polygon_orientation(
        disc_transects, rotated_polygon, x_offset * scale))

    t0 = time.perf_counter()
    trap = float(score_polygon_orientation(
        grad_transects, w_rot, b_rot, x_off_t, temp, temp, use_closed_form=False))
    t_trap = time.perf_counter() - t0

    t0 = time.perf_counter()
    closed = float(score_polygon_orientation(
        grad_transects, w_rot, b_rot, x_off_t, temp, temp, use_closed_form=True))
    t_cf = time.perf_counter() - t0

    return discrete, trap, closed, t_trap, t_cf


def main():
    rng = np.random.default_rng(0)
    num_polygons = 200
    ppt = 1000
    temp = 10000

    e_trap, e_cf, e_trap_cf = [], [], []
    tt, tc = [], []
    n = 0
    while n < num_polygons:
        coords = random_convex_polygon(rng, n_points=rng.integers(3, 10))
        coords, _ = rescale_polygon(coords, 1.0)
        theta = float(rng.random() * 360.0)
        x_offset = float(rng.random() * 0.5 - 0.25)
        try:
            d, tr, cf, t_tr, t_cf = scores_for_polygon(coords, theta, x_offset, ppt, temp)
            if any(np.isnan(v) for v in (d, tr, cf)):
                continue
        except Exception:
            continue
        e_trap.append(abs(tr - d))
        e_cf.append(abs(cf - d))
        e_trap_cf.append(abs(tr - cf))
        tt.append(t_tr)
        tc.append(t_cf)
        n += 1

    print(f"\nScore comparison over N={num_polygons} random convex polygons "
          f"(Temp={temp}, PPT={ppt})")
    print("=" * 60)
    print("error vs discrete truth        mean        max")
    print("-" * 60)
    print(f"  trapezoidal (Eq.6 approx){np.mean(e_trap):>12.6f}{np.max(e_trap):>11.6f}")
    print(f"  closed form  (chord len) {np.mean(e_cf):>12.6f}{np.max(e_cf):>11.6f}")
    print("-" * 60)
    print(f"  |trap - closed form|     {np.mean(e_trap_cf):>12.6f}{np.max(e_trap_cf):>11.6f}")
    print("=" * 60)
    print(f"per-polygon time   trapezoidal={np.mean(tt)*1e3:7.2f} ms   "
          f"closed form={np.mean(tc)*1e3:7.2f} ms   "
          f"speedup={np.mean(tt)/np.mean(tc):.1f}x")


if __name__ == "__main__":
    main()
