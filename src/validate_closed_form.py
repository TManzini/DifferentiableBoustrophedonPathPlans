"""
Validate the closed-form transect-length solution (boustrophedon.closed_form)
against the trapezoidal approximation of Equation 6 implemented in
boustrophedon.differentiable.approximate_transect_length.

For a sweep of random convex polygons, random orientations and random transect
positions, we compare:

    trapezoidal L(x)  =  trapz of P_sigma([x, y], F, Temp) over y      (existing)
    closed-form  L(x)  =  geometric chord length at x                   (new)

and report the mean absolute difference as a function of Temperature and points
per transect (PPT). The closed form is the Temp -> inf limit, so the gap should
shrink as Temperature and PPT grow -- exactly mirroring Table I in the paper.

We also (1) sanity-check the closed form against a *very* fine direct integration
of P_sigma, and (2) confirm gradients flow through the closed form w.r.t. angle
and x-offset.

Run:  python src/validate_closed_form.py
"""

import os
import sys

import numpy as np
import torch
import shapely

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from shape_utils import recenter_centroid_at_origin, rescale_polygon
from boustrophedon.differentiable import (
    generate_polygon_weights_and_biases_from_coordinates,
    get_transect_points,
    rotate_polygon,
    approximate_transect_length,
)
from boustrophedon.closed_form import closed_form_transect_length

X, Y = 0, 1


def random_convex_polygon(rng, n_points=12):
    """Convex polygon = convex hull of random points (no extra deps).

    Forced to counter-clockwise winding so the polygon representation has the
    interior on the all-v_i>0 side (the convention the differentiable pipeline
    assumes). Clockwise rings make P_sigma ~= 0 everywhere -- a degenerate case
    where the trapezoidal integral and the closed form both return 0.
    """
    pts = rng.random((n_points, 2))
    hull = shapely.MultiPoint(pts).convex_hull
    hull = shapely.geometry.polygon.orient(hull, sign=1.0)  # CCW
    return [list(c) for c in hull.exterior.coords[:-1]]


def setup_polygon(polygon_coords, theta, polygon_rescale=1.0):
    centered = recenter_centroid_at_origin(polygon_coords)
    scaled, scale_factor = rescale_polygon(centered, polygon_rescale)
    line_components = generate_polygon_weights_and_biases_from_coordinates(scaled)
    weights, biases = rotate_polygon(
        line_components, torch.tensor([theta], requires_grad=True))
    return weights, biases, scale_factor


def fine_reference_length(x, weights, biases, temp, n=2_000_000):
    """Direct high-resolution numerical integral of P_sigma at one x (truth)."""
    y = torch.linspace(-0.5, 0.5, n, dtype=torch.double)
    W = torch.stack([w.double() for w in weights])          # (F, 2)
    B = torch.tensor([float(b) for b in biases], dtype=torch.double)
    v = W[:, X] * float(x) + W[:, Y] * y[:, None] + B       # (n, F)
    P = torch.sigmoid(temp * v).prod(dim=1)                 # (n,)
    Psig = torch.sigmoid(temp * (P - 0.5))
    return torch.trapz(Psig, y).item()


def main():
    rng = np.random.default_rng(0)

    ppt_range = [100, 1000, 10000]
    temp_range = [1, 10, 100, 1000, 10000]
    num_polygons = 40
    polygon_rescale = 1.0
    transect_spacing = 0.5

    # errors[ppt][temp] = list of |trapezoidal - closed_form|
    errors = {p: {t: [] for t in temp_range} for p in ppt_range}
    closed_lengths = []  # all closed-form lengths (to show they're non-trivial)

    for _ in range(num_polygons):
        coords = random_convex_polygon(rng, n_points=rng.integers(3, 10))
        coords, _ = rescale_polygon(coords, polygon_rescale)
        theta = float(rng.random() * 360.0)
        x_offset = float(rng.random() * transect_spacing - transect_spacing / 2)

        weights, biases, scale = setup_polygon(coords, theta, polygon_rescale)
        x_off_t = torch.tensor([x_offset * scale])

        for ppt in ppt_range:
            transects = get_transect_points(
                -polygon_rescale / 2, polygon_rescale / 2,
                -polygon_rescale / 2, polygon_rescale / 2,
                transect_spacing * scale, ppt)
            for temp in temp_range:
                for transect in transects:
                    _, trap_len = approximate_transect_length(
                        transect, weights, biases, temp, x_off_t)
                    x = transect[0][X] + x_off_t[0]
                    cf_len = closed_form_transect_length(x, weights, biases)
                    errors[ppt][temp].append(
                        abs(float(trap_len) - float(cf_len)))
                    if ppt == ppt_range[-1] and temp == temp_range[-1]:
                        closed_lengths.append(float(cf_len))

    print("\nMean |trapezoidal - closed_form|  (lower = better agreement)")
    print("=" * 64)
    header = "Temp \\ PPT".ljust(14) + "".join(f"{p:>16}" for p in ppt_range)
    print(header)
    print("-" * 64)
    for temp in temp_range:
        row = f"Temperature={temp}".ljust(14)
        for ppt in ppt_range:
            row += f"{np.mean(errors[ppt][temp]):>16.6f}"
        print(row)
    print("=" * 64)
    cl = np.array(closed_lengths)
    print(f"closed-form lengths: N={cl.size}, mean={cl.mean():.4f}, "
          f"max={cl.max():.4f}, non-zero={(cl > 1e-9).mean()*100:.0f}%")

    # ---- (1) closed form vs. a near-exact direct integral at high Temp ----
    print("\nClosed form vs. fine direct integration of P_sigma (Temp=10000):")
    coords = random_convex_polygon(rng, n_points=6)
    coords, _ = rescale_polygon(coords, 1.0)
    weights, biases, scale = setup_polygon(coords, 37.0, 1.0)
    max_abs = 0.0
    for x in np.linspace(-0.45, 0.45, 9):
        ref = fine_reference_length(x, weights, biases, temp=10000.0)
        cf = float(closed_form_transect_length(torch.tensor(float(x)), weights, biases))
        max_abs = max(max_abs, abs(ref - cf))
        print(f"  x={x:+.3f}   fine={ref:.6f}   closed={cf:.6f}   |diff|={abs(ref-cf):.2e}")
    print(f"  -> max |diff| across x = {max_abs:.2e}")

    # ---- (2) gradients flow through the closed form (angle & x-offset) ----
    print("\nGradient check (closed form is differentiable):")
    theta = torch.tensor([41.0], requires_grad=True)
    x_off = torch.tensor([0.05], requires_grad=True)
    line_components = generate_polygon_weights_and_biases_from_coordinates(
        rescale_polygon(recenter_centroid_at_origin(coords), 1.0)[0])
    w_rot, b_rot = rotate_polygon(line_components, theta)
    total = torch.stack([
        closed_form_transect_length(t[0][X] + x_off[0], w_rot, b_rot)
        for t in get_transect_points(-0.5, 0.5, -0.5, 0.5, 0.25, 100)
    ]).sum()
    total.backward()
    print(f"  total length        = {float(total):.6f}")
    print(f"  d(length)/d(theta)  = {float(theta.grad):.6e}")
    print(f"  d(length)/d(xoffset)= {float(x_off.grad):.6e}")


if __name__ == "__main__":
    main()
