"""
Ear-clipping triangulation of a simple polygon (convex OR concave, no
self-intersections). This is the geometric primitive that extends the
differentiable / closed-form transect-length machinery from convex-only to
arbitrary simple polygons: triangulate once, treat each triangle as its own
tiny convex polygon (3 faces), and reuse every existing convex routine
unchanged on each piece. See boustrophedon.closed_form.nonconvex_* for how the
per-triangle results are recombined.

Pure-Python, no extra dependency (shapely's triangulate() does an unconstrained
Delaunay triangulation of the *point set*, which ignores concave edges and is
wrong here; ear clipping respects the polygon boundary).
"""

X = 0
Y = 1


def _cross(o, a, b):
    return (a[X] - o[X]) * (b[Y] - o[Y]) - (a[Y] - o[Y]) * (b[X] - o[X])


def signed_area(coordinates):
    """Twice-the-area sign convention: positive => counter-clockwise winding."""
    n = len(coordinates)
    s = 0.0
    for i in range(n):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def _point_in_triangle(p, a, b, c):
    d1 = _cross(a, b, p)
    d2 = _cross(b, c, p)
    d3 = _cross(c, a, p)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def ear_clip_triangulate(coordinates, eps=1e-12):
    """Triangulate a simple polygon via ear clipping.

    Args:
        coordinates: ordered list of [x, y] vertices of a simple (non
            self-intersecting) polygon. May be convex or concave.
        eps: numerical tolerance for the convexity test.

    Returns:
        A list of triangles, each a list of 3 [x, y] points, with the same
        winding orientation as the input. The triangles partition the polygon
        (disjoint interiors, union = the polygon) -- this is what lets
        per-triangle quantities (area, chord length) be summed directly.
    """
    poly = [list(c) for c in coordinates]
    n = len(poly)
    if n < 3:
        raise ValueError("Polygon must have at least 3 vertices.")
    if n == 3:
        return [poly]

    ccw = signed_area(poly) > 0
    indices = list(range(n))
    triangles = []

    guard = 0
    max_guard = n * n + 10

    while len(indices) > 3 and guard < max_guard:
        guard += 1
        m = len(indices)
        ear_found = False
        for k in range(m):
            i_prev = indices[(k - 1) % m]
            i_curr = indices[k]
            i_next = indices[(k + 1) % m]
            a, b, c = poly[i_prev], poly[i_curr], poly[i_next]

            cross = _cross(a, b, c)
            is_convex_vertex = (cross > eps) if ccw else (cross < -eps)
            if not is_convex_vertex:
                continue

            contains_other = False
            for idx in indices:
                if idx in (i_prev, i_curr, i_next):
                    continue
                if _point_in_triangle(poly[idx], a, b, c):
                    contains_other = True
                    break

            if not contains_other:
                triangles.append([a, b, c])
                del indices[k]
                ear_found = True
                break

        if not ear_found:
            # Degenerate/near-collinear input: clip the first vertex rather
            # than loop forever. Keeps the function total for malformed input.
            i_prev = indices[-1]
            i_curr = indices[0]
            i_next = indices[1 % m]
            triangles.append([poly[i_prev], poly[i_curr], poly[i_next]])
            del indices[0]

    if len(indices) == 3:
        triangles.append([poly[indices[0]], poly[indices[1]], poly[indices[2]]])

    return triangles
