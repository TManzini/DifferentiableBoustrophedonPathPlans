"""
Closed-form solution to the transect-length integral (Equation 6 in the paper).

Equation 6 approximates the length of a transect as

    L(x) = integral_{-0.5}^{0.5} P_sigma([x, y], F, Temp) dy

where, for a fixed transect x, every face is a linear function of y,

    v_i(y) = W_{x,i} * x + W_{y,i} * y + B_i = m_i * y + k_i ,
    m_i = W_{y,i},   k_i = W_{x,i} * x + B_i ,

and

    P(y)        = prod_i sigmoid(Temp * v_i(y))
    P_sigma(y)  = sigmoid(Temp * (P(y) - 0.5)).

No elementary antiderivative exists for P_sigma (it is a sigmoid of a product of
sigmoids). However, the *value* of the definite integral has a clean closed form.

Near the lower boundary only the binding lower face is transitioning, so locally
P(y) ~= sigmoid(Temp * m_lo * (y - y_lo)) with y_lo = -k_lo / m_lo, and the
integrand becomes

    g(y) = sigmoid(Temp * (sigmoid(Temp * m_lo * (y - y_lo)) - 0.5)).

Because sigmoid(z) - 0.5 is odd and sigmoid(-w) = 1 - sigmoid(w), this satisfies
g(y_lo + t) + g(y_lo - t) = 1 for every finite Temp: the smooth transition is
*perfectly antisymmetric* about y_lo. The same holds at the upper boundary y_hi.
Antisymmetric transitions contribute zero net area relative to a hard step, so the
integral equals the geometric chord length of the convex polygon at x:

    y_lo = max_{m_i > 0} ( -k_i / m_i )   (highest "turns on going up" boundary)
    y_hi = min_{m_i < 0} ( -k_i / m_i )   (lowest  "turns off going up" boundary)
    L(x) = clip(y_hi, -0.5, 0.5) - clip(y_lo, -0.5, 0.5)   (0 if the chord is empty)

The result is exact as Temp -> inf; the finite-Temp residual is the
antisymmetry-violating term that appears only when two faces transition together
(i.e. the transect grazes a vertex), and is exponentially small in Temp.

This matches the convention used in boustrophedon.differentiable: a point is
inside the polygon when every v_i > 0 (consistent polygon winding), and transects
run parallel to the Y axis from y_min=-0.5 to y_max=0.5.
"""

import torch

from boustrophedon.triangulate import ear_clip_triangulate

X = 0
Y = 1


def closed_form_transect_length(x, line_weights, line_biases,
                                y_min=-0.5, y_max=0.5):
    """Closed-form length of a single transect at horizontal position ``x``.

    Args:
        x: scalar (python float or 0-/1-d tensor) horizontal position of the
            transect, already including any x_offset.
        line_weights: list of length-2 tensors [W_x, W_y], one per polygon face
            (same objects produced by ``rotate_polygon`` / generated from
            coordinates).
        line_biases: list of scalar biases B, one per face.
        y_min, y_max: integration limits (the rescaled transect runs -0.5..0.5).

    Returns:
        A scalar tensor with the transect length. Differentiable w.r.t. ``x`` and
        w.r.t. the weights/biases (so gradients flow to x-offset and angle).
    """
    W = torch.stack([torch.as_tensor(w) for w in line_weights]).double()  # (F, 2)
    B = torch.stack([torch.as_tensor(b) for b in line_biases]).double().reshape(-1)  # (F,)
    x = torch.as_tensor(x).double().reshape(())

    m = W[:, Y]                 # slope of v_i in y
    k = W[:, X] * x + B         # constant part of v_i at this x

    # Boundary in y where the face flips sign: v_i = 0 -> y = -k/m.
    # Guard the (measure-zero) horizontal-face case m == 0 separately.
    horizontal = m == 0
    m_safe = torch.where(horizontal, torch.ones_like(m), m)
    crossing = -k / m_safe

    pos = m > 0     # face turns ON as y increases  -> lower bound on y
    neg = m < 0     # face turns OFF as y increases  -> upper bound on y

    # Lower envelope: the largest "turn-on" crossing. Faces that are not lower
    # bounds are filled with y_min so they never win the max.
    lo_candidates = torch.where(pos, crossing, torch.full_like(crossing, y_min))
    y_lo = torch.maximum(lo_candidates.max(), torch.as_tensor(float(y_min)).double())

    # Upper envelope: the smallest "turn-off" crossing.
    hi_candidates = torch.where(neg, crossing, torch.full_like(crossing, y_max))
    y_hi = torch.minimum(hi_candidates.min(), torch.as_tensor(float(y_max)).double())

    # Clip the chord to the integration window and take the (non-negative) length.
    lo = torch.clamp(y_lo, min=y_min, max=y_max)
    hi = torch.clamp(y_hi, min=y_min, max=y_max)
    length = torch.clamp(hi - lo, min=0.0)

    # A horizontal face with k <= 0 means the whole transect lies outside that
    # half-plane -> zero length. (Rare; transects are vertical, faces generic.)
    if bool(horizontal.any()):
        outside = (horizontal & (k <= 0)).any()
        if bool(outside):
            length = length * 0.0

    return length


def soft_closed_form_transect_length(x, line_weights, line_biases, beta,
                                     y_min=-0.5, y_max=0.5, slope_eps=1e-6):
    """Smoothed closed-form transect length with a tunable smoothness ``beta``.

    Same chord as ``closed_form_transect_length``, but every hard operation is
    replaced by its smooth (LogSumExp-family) analogue so the resulting loss
    landscape is C-infinity with no kinks:

        max over lower faces  -> soft-max   (1/beta) logsumexp(beta * c)
        min over upper faces  -> soft-min  -(1/beta) logsumexp(-beta * c)
        clip(., -0.5, 0.5)    -> folded in by adding y_min / y_max to each set
        clamp(hi - lo, min=0) -> softplus   (1/beta) log(1 + exp(beta * (hi-lo)))

    where c_i = -k_i / m_i is the y at which face i crosses zero. ``beta`` is the
    smoothness knob and plays the role Temperature did in Equation 6: small beta
    "melts" the polygon (smooth landscape, a transect just outside the polygon
    keeps a small ghost length); large beta -> the exact chord. As beta -> inf
    this converges to ``closed_form_transect_length``.

    Args:
        beta: smoothness/sharpness (> 0). Anneal from small to large during
            optimization for the same effect as raising Temperature.
        slope_eps: faces with |m_i| <= slope_eps (near-horizontal) are excluded;
            they define no y-crossing for a vertical transect.
    """
    W = torch.stack([torch.as_tensor(w) for w in line_weights]).double()
    B = torch.stack([torch.as_tensor(b) for b in line_biases]).double().reshape(-1)
    x = torch.as_tensor(x).double().reshape(())
    beta = torch.as_tensor(beta).double().reshape(())

    m = W[:, Y]
    k = W[:, X] * x + B

    # A near-horizontal face (m ~= 0) does not define a y-crossing for a vertical
    # transect; in the integral its sigmoid sigma(beta * k) is ~constant in y and
    # factors out of the length. So such faces are handled as a multiplicative
    # *presence* gate, not as part of the y-envelope. (Exactly at symmetric angles
    # like theta=90 the polygon edges become axis-aligned and m hits 0 -- without
    # this gate the soft form would hallucinate absent transects into existence.)
    valid = m.abs() > slope_eps
    m_safe = torch.where(valid, m, torch.ones_like(m))
    crossing = -k / m_safe        # c_i = -k_i / m_i

    neg_inf = torch.full_like(crossing, float("-inf"))
    pos_inf = torch.full_like(crossing, float("inf"))

    # Lower endpoint = soft-max over { c_i : m_i > 0 } and the window floor y_min.
    lo_set = torch.where(valid & (m > 0), crossing, neg_inf)
    lo_set = torch.cat([lo_set, torch.tensor([y_min], dtype=crossing.dtype)])
    y_lo = torch.logsumexp(beta * lo_set, dim=0) / beta

    # Upper endpoint = soft-min over { c_i : m_i < 0 } and the window ceil y_max.
    hi_set = torch.where(valid & (m < 0), crossing, pos_inf)
    hi_set = torch.cat([hi_set, torch.tensor([y_max], dtype=crossing.dtype)])
    y_hi = -torch.logsumexp(-beta * hi_set, dim=0) / beta

    # Soft, non-negative length (softplus replaces clamp(., min=0)).
    diff = y_hi - y_lo
    length = torch.nn.functional.softplus(diff, beta=float(beta))

    # Soft presence gate from any near-horizontal faces: inside <=> k > 0.
    if bool((~valid).any()):
        gate = torch.sigmoid(beta * k[~valid]).prod()
        length = length * gate

    return length


def triangulated_face_groups(coordinates):
    """Ear-clip ``coordinates`` (any simple polygon, convex or concave) and
    build the (W, B) face list for each triangle via the same point-pair
    construction used for a single convex polygon.

    Returns a list of (line_weights, line_biases) tuples, one per triangle
    (each with exactly 3 faces). Compute once per polygon shape; rotate every
    triangle's faces together each step (see ``rotate_triangulated_faces``).
    """
    from boustrophedon.differentiable import generate_polygon_weights_and_biases_from_coordinates
    triangles = ear_clip_triangulate(coordinates)
    groups = []
    for tri in triangles:
        line_components = generate_polygon_weights_and_biases_from_coordinates(tri)
        weights = [w for w, _ in line_components]
        biases = [b for _, b in line_components]
        groups.append((weights, biases))
    return groups


def rotate_triangulated_faces(face_groups, angle):
    """Apply the existing rigid rotation to every triangle's faces uniformly.

    ``angle`` is rotated identically for all triangles (it is one rigid
    transform of the whole polygon), so this is just
    ``rotate_polygon`` mapped over each triangle's face list.
    """
    from boustrophedon.differentiable import rotate_polygon
    rotated = []
    for weights, biases in face_groups:
        line_components = list(zip(weights, biases))
        w_rot, b_rot = rotate_polygon(line_components, angle)
        rotated.append((w_rot, b_rot))
    return rotated


def nonconvex_closed_form_transect_length(x, rotated_face_groups, beta=None,
                                          y_min=-0.5, y_max=0.5):
    """Exact (or smoothed) chord length through a non-convex polygon at ``x``.

    ``rotated_face_groups`` is the per-triangle (weights, biases) list from
    ``rotate_triangulated_faces``. Because a triangulation partitions the
    polygon into pieces with disjoint interiors, the chord through the whole
    (possibly concave) polygon at any x is the *sum* of the per-triangle chord
    lengths -- no union/inclusion-exclusion needed, and no double counting,
    since for almost every x the triangles' y-intervals at that x are disjoint.

    ``beta=None`` sums the exact ``closed_form_transect_length`` per triangle
    (matches the convex closed form exactly when the polygon happens to be
    convex / triangulates into pieces sharing every transect uniformly).
    A finite ``beta`` sums the smoothed ``soft_closed_form_transect_length``
    per triangle instead, for the same tunable-smoothness benefit as the
    convex case, now over a concave shape.
    """
    total = None
    for weights, biases in rotated_face_groups:
        if beta is None:
            l = closed_form_transect_length(x, weights, biases, y_min=y_min, y_max=y_max)
        else:
            l = soft_closed_form_transect_length(x, weights, biases, beta, y_min=y_min, y_max=y_max)
        total = l if total is None else total + l
    return total


def nonconvex_closed_form_transect_lengths(transects, rotated_face_groups, x_offset,
                                           beta=None, y_min=-0.5, y_max=0.5):
    """Batch version mirroring ``closed_form_transect_lengths`` for non-convex
    polygons: sums per-triangle chord lengths for every transect in ``transects``.
    """
    x_offset = torch.as_tensor(x_offset).reshape(-1)[0]
    lengths = []
    for transect in transects:
        x = transect[0][X] + x_offset
        lengths.append(nonconvex_closed_form_transect_length(
            x, rotated_face_groups, beta=beta, y_min=y_min, y_max=y_max))
    return torch.stack(lengths)


def closed_form_transect_lengths(transects, line_weights, line_biases, x_offset,
                                 beta=None, y_min=-0.5, y_max=0.5):
    """Closed-form lengths for a batch of transects.

    Mirrors the inputs of ``boustrophedon.differentiable.approximate_transect_length``
    so it can be dropped into the scoring pipeline. ``transects`` is the tensor
    returned by ``get_transect_points``; each transect shares a constant x, to
    which ``x_offset`` is added.

    ``beta`` selects the variant: ``None`` -> exact (hard) chord length; a finite
    value -> the smoothed chord (``soft_closed_form_transect_length``) for a
    smoother loss landscape.
    """
    x_offset = torch.as_tensor(x_offset).reshape(-1)[0]
    lengths = []
    for transect in transects:
        x = transect[0][X] + x_offset
        if beta is None:
            lengths.append(closed_form_transect_length(
                x, line_weights, line_biases, y_min=y_min, y_max=y_max))
        else:
            lengths.append(soft_closed_form_transect_length(
                x, line_weights, line_biases, beta, y_min=y_min, y_max=y_max))
    return torch.stack(lengths)
