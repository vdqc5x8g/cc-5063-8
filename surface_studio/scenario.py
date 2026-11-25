"""Teapot decal scenario and canonical solver diagnostic cases."""
from __future__ import annotations

import math

from .surfaces import (
    FoldbackSurface,
    PlaneSurface,
    SpherePatch,
    make_curved_patch,
    make_teapot_body,
    make_teapot_spout,
    yaw_transform,
    TransformedSurface,
)
from .projection import TransitionPoint, catmull_rom_chain


def build_teapot(yaw_rad: float = 0.0):
    body0 = make_teapot_body()
    spout0 = make_teapot_spout()
    if abs(yaw_rad) < 1e-15:
        return [body0, spout0]
    tr = yaw_transform(yaw_rad)
    return [TransformedSurface(body0, tr), TransformedSurface(spout0, tr)]


def teapot_frame(yaw_rad: float = 0.0):
    """Surfaces plus the rigid transform used to build design geometry."""
    body = make_teapot_body()
    spout = make_teapot_spout()
    if abs(yaw_rad) < 1e-15:
        return body, spout, lambda p: p
    tr = yaw_transform(yaw_rad)
    return TransformedSurface(body, tr), TransformedSurface(spout, tr), tr


def design_control_points(yaw_rad: float = 0.0, lift: float = 0.05):
    """Same spatial pick/design curve is reused by both edits.

    Runs along the front stripe of the spout from tip to root, then continues
    across the junction onto the body belly.
    """
    body, spout, tr = teapot_frame(yaw_rad)
    pts = []
    spout_s = [0.98, 0.86, 0.72, 0.58, 0.44, 0.30, 0.17, 0.06]
    theta = math.pi / 2.0
    for s in spout_s:
        p = spout.point(s, theta)
        n = spout.normal(s, theta)
        q = tuple(p[k] + lift * n[k] for k in range(3)) if n else p
        pts.append(tr(q) if not isinstance(spout, TransformedSurface) else q)
    body_uv = [
        (0.34, 0.34),
        (0.36, 0.52),
        (0.39, 0.78),
        (0.41, 1.08),
        (0.43, 1.38),
        (0.45, 1.62),
    ]
    for t, phi in body_uv:
        p = body.point(t, phi)
        n = body.normal(t, phi)
        q = tuple(p[k] + lift * n[k] for k in range(3)) if n else p
        pts.append(q)
    return pts


def sampled_design_curve(yaw_rad: float = 0.0, samples_per_span: int = 9):
    cps = design_control_points(yaw_rad)
    return cps, catmull_rom_chain(cps, samples_per_span)


def design_curve_samples(
    yaw_rad: float = 0.0,
    lift: float = 0.06,
    s_j: float = 0.12,
    theta_j: float = 2.95,
):
    """Ordered short design curve used by both edits.

    Spout stripe down to the best-fit junction band (s ~ 0.22), then a
    body-space UV walk whose first direction is the incoming spout tangent
    projected onto the body tangent plane, so the route (and decal frame)
    crosses the seam continuously. Edit A forces the same input onto the
    spout only and is clipped at its root boundary; edit B switches at the
    user transition line. Input positions are identical for both edits.
    """
    from .solver import closest as _closest
    from . import vecmath as vm

    body, spout, _tr = teapot_frame(yaw_rad)
    pts = []
    n_spout = 22
    for k in range(n_spout):
        s = 0.985 + (s_j - 0.985) * k / (n_spout - 1)
        theta = theta_j
        p = spout.point(s, theta)
        n = spout.normal(s, theta)
        pts.append(tuple(p[j] + lift * n[j] for j in range(3)) if n else p)

    contact = spout.point(s_j, theta_j)
    br = _closest(body, contact)
    u0, v0 = br.best.u, br.best.v

    eps = 2e-3
    t_in = vm.normalize(vm.sub(
        spout.point(s_j - eps, theta_j),
        spout.point(s_j + eps, theta_j),
    ))
    _p, st, sv = body.derivs(u0, v0)
    n_body = vm.normalize(vm.cross(st, sv))
    t_proj = vm.normalize(vm.sub(t_in, vm.mul(n_body, vm.dot(t_in, n_body))))

    a11, a12, a22 = vm.dot(st, st), vm.dot(st, sv), vm.dot(sv, sv)
    r1, r2 = vm.dot(t_proj, st), vm.dot(t_proj, sv)
    det = a11 * a22 - a12 * a12
    dt0 = (r1 * a22 - a12 * r2) / det
    dv0 = (a11 * r2 - a12 * r1) / det

    uv = [u0, v0]
    step_uv_scale = 0.075
    n_body_steps = 18
    for k in range(1, n_body_steps + 1):
        q = k / n_body_steps
        dt = dt0 * (1 - 0.7 * q) + (-0.12) * 0.7 * q
        dv = dv0 * (1 - 0.7 * q) + (1.25) * 0.7 * q
        norm_uv = math.hypot(dt, dv) or 1.0
        uv[0] += dt / norm_uv * step_uv_scale
        uv[1] += dv / norm_uv * step_uv_scale
        u = min(max(uv[0], body.u0 + 1e-3), body.u1 - 1e-3)
        v = uv[1]
        p = body.point(u, v)
        n = body.normal(u, v)
        pts.append(tuple(p[j] + lift * n[j] for j in range(3)) if n else p)
    return pts


def preset_transition_uv():
    """User-drawn transition line expressed as picked surface UVs.

    Placed at the best-fit junction band where the two patches are ~0.02
    apart. Order matters: spout anchors first, then adjacent body picks.
    """
    return [
        {"surfaceId": "spout", "u": 0.14, "v": 2.95},
        {"surfaceId": "spout", "u": 0.12, "v": 2.95},
        {"surfaceId": "spout", "u": 0.10, "v": 2.95},
        {"surfaceId": "body", "u": 0.3094, "v": 0.0482},
        {"surfaceId": "body", "u": 0.3214, "v": 0.0342},
        {"surfaceId": "body", "u": 0.3004, "v": 0.0642},
    ]


def preset_transition(yaw_rad: float = 0.0):
    surfs = {s.id: s for s in build_teapot(yaw_rad)}
    line = []
    for item in preset_transition_uv():
        s = surfs[item["surfaceId"]]
        u, v = item["u"], item["v"]
        line.append(TransitionPoint(item["surfaceId"], u, v, tuple(s.point(u, v))))
    return line


def singularity_design_curve(samples: int = 31, phi0: float = 1.0):
    """A straight-in-space line passing exactly over the sphere pole."""
    sph = SpherePatch(radius=1.0)
    pts = []
    # Travel from theta=0.9 on meridian phi0, through the pole, out the
    # opposite meridian phi0+pi, theta growing again.
    for k in range(samples):
        x = -0.9 + 1.8 * k / (samples - 1)  # coordinate along travel axis
        # Great-circle-like path in the plane spanned by meridians.
        angle = x  # from -0.9..0.9
        if angle <= 0:
            theta = -angle
            phi = phi0
        else:
            theta = angle
            phi = phi0 + math.pi
        pts.append(sph.point(theta, phi))
    return sph, pts


def foldback_tracking_case(samples: int = 30, half_span: float = 0.6, height: float = 2.0):
    """Design curve above a hairpin fold that crosses the symmetry plane.

    For every sample away from the fold crest a point on the open side has two
    near-equal closest candidates (the two sheets).  Nearest-point selection
    therefore flips sheets at the symmetry plane; branch tracking must keep one
    branch and report the break where that branch genuinely ends.
    """
    surf = FoldbackSurface()
    pts = [
        (-half_span + 2.0 * half_span * k / (samples - 1), height, 0.0)
        for k in range(samples)
    ]
    return surf, pts


def equidistant_candidates_case(samples: int = 21):
    """Straight line exactly above the fold crest: two symmetric candidates.

    Every sample has mirrored closest points with identical distance, which is
    the cleanest test that equal distance alone never changes the branch.
    """
    surf = FoldbackSurface()
    pts = [(0.0, 2.0, -0.6 + 1.2 * k / (samples - 1)) for k in range(samples)]
    return surf, pts


def boundary_tracking_case(samples: int = 25):
    """Design curve sweeping sideways out of the fold's parameter domain.

    Used to check that branch identity and boundary-exit reporting still work
    when the route runs along the parameter boundary.
    """
    surf = FoldbackSurface()
    pts = [
        (1.0 + 2.0 * k / (samples - 1), 2.0, 0.5)
        for k in range(samples)
    ]
    return surf, pts
