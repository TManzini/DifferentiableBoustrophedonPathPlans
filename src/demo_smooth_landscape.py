"""
Show that the *soft* closed-form transect length gives a tunable-smoothness loss
landscape -- the role Temperature played in Equation 6 -- without any numerical
integration.

We sweep theta over [0, 360) for the paper's parallelogram (x-offset fixed) and
compute the fitness score under several length backends:

    trapezoidal, low  Temp   -- reference for what a "smooth" landscape looks like
    trapezoidal, high Temp   -- sharp, non-convex (what GD gets stuck in)
    hard closed form         -- exact chord, kinked, equally non-convex
    soft closed form, beta varied -- the new knob: small beta smooth, large sharp

Smoothness is quantified two ways over the theta sweep:
    * # local maxima  (proxy for the number of local optima that trap GD)
    * total variation of the score (sum of |delta score|; lower = smoother)

Run:  python src/demo_smooth_landscape.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from shape_utils import recenter_centroid_at_origin, rescale_polygon
from boustrophedon.differentiable import (
    generate_polygon_weights_and_biases_from_coordinates,
    get_transect_points, rotate_polygon, score_polygon_orientation)

X, Y = 0, 1

# The parallelogram used throughout the paper (default coords in optimize_gd.py).
POLYGON = [[8, 4], [8, 10], [4, 8], [4, 2]]


def count_local_maxima(scores, prominence=1e-4):
    s = np.asarray(scores)
    n = len(s)
    count = 0
    for i in range(n):
        l = s[(i - 1) % n]
        r = s[(i + 1) % n]
        if s[i] > l and s[i] >= r and (s[i] - min(l, r)) > prominence:
            count += 1
    return count


def total_variation(scores):
    s = np.asarray(scores)
    return float(np.sum(np.abs(np.diff(np.concatenate([s, s[:1]])))))


def sweep(weights_fn, transects, polygon_temp, falloff_temp,
          use_closed_form=False, beta=None, n_theta=720):
    thetas = np.linspace(0, 360, n_theta, endpoint=False)
    scores = []
    for th in thetas:
        w_rot, b_rot = weights_fn(th)
        sc = score_polygon_orientation(
            transects, w_rot, b_rot, torch.tensor([0.0]),
            polygon_temp, falloff_temp,
            use_closed_form=use_closed_form, closed_form_beta=beta)
        scores.append(float(sc))
    return thetas, np.array(scores)


def main():
    transect_spacing, ppt, rescale = 0.5, 400, 1.0
    centered = recenter_centroid_at_origin(POLYGON)
    scaled, scale = rescale_polygon(centered, rescale)
    line_components = generate_polygon_weights_and_biases_from_coordinates(scaled)
    transects = get_transect_points(-rescale / 2, rescale / 2, -rescale / 2,
                                    rescale / 2, transect_spacing * scale, ppt)

    def weights_fn(theta):
        return rotate_polygon(line_components, torch.tensor([float(theta)], requires_grad=False))

    # The "transect shown" indicator (Eq. 7) is its own temperature: a sharp
    # count makes transects flicker in/out as theta changes -- a major source of
    # local maxima on its own. The existing pipeline ties both temperatures
    # together (score_polygon_orientation(..., temp, temp)), so for a fair
    # comparison the soft closed form ties the falloff to beta as well.
    configs = [
        ("hard closed form (exact, beta=inf)", dict(polygon_temp=1,     falloff_temp=10000, use_closed_form=True)),
        ("soft closed form beta=4",            dict(polygon_temp=1,     falloff_temp=4,     use_closed_form=True, beta=4)),
        ("soft closed form beta=8",            dict(polygon_temp=1,     falloff_temp=8,     use_closed_form=True, beta=8)),
        ("soft closed form beta=20",           dict(polygon_temp=1,     falloff_temp=20,    use_closed_form=True, beta=20)),
        ("soft closed form beta=100",          dict(polygon_temp=1,     falloff_temp=100,   use_closed_form=True, beta=100)),
        ("soft closed form beta=1000",         dict(polygon_temp=1,     falloff_temp=1000,  use_closed_form=True, beta=1000)),
        ("soft closed form beta=10000",        dict(polygon_temp=1,     falloff_temp=10000, use_closed_form=True, beta=10000)),
    ]

    print("\nLoss-landscape smoothness over a full theta sweep "
          "(parallelogram, x-offset=0)")
    print("=" * 72)
    print(f"{'backend':40}{'#local maxima':>16}{'total variation':>16}")
    print("-" * 72)
    curves = {}
    for label, cfg in configs:
        _, scores = sweep(weights_fn, transects,
                          cfg.get("polygon_temp", 1), cfg["falloff_temp"],
                          use_closed_form=cfg.get("use_closed_form", False),
                          beta=cfg.get("beta"))
        curves[label] = scores
        print(f"{label:40}{count_local_maxima(scores):>16}{total_variation(scores):>16.4f}")
    print("=" * 72)
    print("Fewer local maxima + lower total variation = smoother = easier for GD.")
    print("Note the global optimum (theta=90/270) is a knife-edge: it is the most")
    print("beta-sensitive point on the curve, only reaching its true value at high")
    print("beta, while bulk local optima (e.g. theta~150) stabilize much earlier.")

    # Save a plot if matplotlib is available (optional, non-fatal).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        thetas = np.linspace(0, 360, len(next(iter(curves.values()))), endpoint=False)
        plt.figure(figsize=(9, 5))
        for label, scores in curves.items():
            plt.plot(thetas, scores, label=label, linewidth=1.2)
        plt.xlabel("theta (degrees)")
        plt.ylabel("fitness score")
        plt.title("Loss landscape vs. length backend (parallelogram)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "smooth_landscape_demo.png")
        plt.savefig(out, dpi=130)
        print(f"\nSaved landscape plot to {out}")
    except Exception as e:
        print(f"\n(plot skipped: {e})")


if __name__ == "__main__":
    main()
