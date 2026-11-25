"""
Validate the triangulated (ear-clipping) closed-form transect length against
shapely's exact intersection on CONCAVE polygons -- the extension discussed for
non-convex shapes: triangulate once, sum each triangle's closed-form chord
length (boustrophedon.closed_form.nonconvex_closed_form_transect_length).

Ground truth: shapely.intersection of a vertical line with the (possibly
concave) polygon, which is exact for any simple polygon, convex or not.

Run:  python src/validate_nonconvex_closed_form.py
"""

import os
import sys

import numpy as np
import torch
import shapely

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from boustrophedon.triangulate import ear_clip_triangulate, signed_area
from boustrophedon.closed_form import (
    triangulated_face_groups, rotate_triangulated_faces,
    nonconvex_closed_form_transect_length)
from boustrophedon.discrete import rotate_polygon_discrete, make_polygon_from_points

X, Y = 0, 1

# An L-shape: concave (one reflex vertex), CCW winding.
L_SHAPE = [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]]

# A 5-point star: concave, several reflex vertices.
def star_polygon(n_points=5, r_outer=1.0, r_inner=0.4):
    pts = []
    for i in range(2 * n_points):
        r = r_outer if i % 2 == 0 else r_inner
        a = np.pi / 2 + i * np.pi / n_points
        pts.append([r * np.cos(a), r * np.sin(a)])
    return pts


def shapely_chord_length(poly, x):
    line = shapely.LineString([(x, -10), (x, 10)])
    return poly.intersection(line).length


def check_partition(coords, label):
    triangles = ear_clip_triangulate(coords)
    poly_area = abs(signed_area(coords))
    tri_area = sum(abs(signed_area(t)) for t in triangles)
    print(f"  [{label}] {len(triangles)} triangles, "
          f"polygon area={poly_area:.6f}, sum of triangle areas={tri_area:.6f}, "
          f"diff={abs(poly_area - tri_area):.2e}")


def main():
    print("Triangulation partition sanity check (areas must match exactly):")
    check_partition(L_SHAPE, "L-shape")
    check_partition(star_polygon(), "5-point star")

    print("\nChord length vs shapely ground truth, across rotation angles "
          "(concave polygons):")
    print("=" * 70)
    for label, coords in [("L-shape", L_SHAPE), ("5-point star", star_polygon())]:
        max_err = 0.0
        errs = []
        discrete_polygon = make_polygon_from_points(coords)
        face_groups = triangulated_face_groups(coords)
        # Offset the sweep slightly off axis-alignment: at theta a multiple of
        # 90deg the (axis-aligned) L-shape's edges land exactly on a sampled x,
        # a measure-zero "transect on the boundary" tie that shapely (closed
        # region) and the strict-inequality closed form resolve differently --
        # the same kind of vertex/edge-tangency caveat already documented for
        # the convex closed form, not an error in the triangulated extension.
        for theta in np.linspace(0.13, 350.13, 36):
            # rotate_polygon_discrete adds the +90deg convention offset that
            # the differentiable face construction implies (see
            # boustrophedon.discrete.rotate_polygon_discrete); matching it here
            # is what the rest of the repo's own discrete-vs-differentiable
            # comparisons (e.g. compare_discrete_gradient.py) do too.
            rotated_polygon = rotate_polygon_discrete(discrete_polygon, theta)
            rotated_faces = rotate_triangulated_faces(
                face_groups, torch.tensor([float(theta)]))

            xs = np.linspace(-1.5, 1.5, 13)
            for x in xs:
                truth = shapely_chord_length(rotated_polygon, x)
                cf = float(nonconvex_closed_form_transect_length(
                    torch.tensor(float(x)), rotated_faces,
                    y_min=-10, y_max=10))
                err = abs(truth - cf)
                errs.append(err)
                max_err = max(max_err, err)
        print(f"  {label:16} mean|err|={np.mean(errs):.2e}   max|err|={max_err:.2e}   N={len(errs)}")
    print("=" * 70)

    print("\nSoft (smoothed) version convergence to hard form as beta -> inf "
          "(L-shape, theta=37deg, x=1.0):")
    face_groups = triangulated_face_groups(L_SHAPE)
    rotated_faces = rotate_triangulated_faces(face_groups, torch.tensor([37.0]))
    x = torch.tensor(1.0)
    hard = float(nonconvex_closed_form_transect_length(x, rotated_faces, beta=None,
                                                        y_min=-10, y_max=10))
    for beta in [1, 10, 100, 1000, 10000]:
        soft = float(nonconvex_closed_form_transect_length(x, rotated_faces, beta=beta,
                                                            y_min=-10, y_max=10))
        print(f"  beta={beta:>6}: soft={soft:.6f}   |soft-hard|={abs(soft-hard):.2e}")
    print(f"  hard (exact) ={hard:.6f}")


if __name__ == "__main__":
    main()
