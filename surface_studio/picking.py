"""Screen ray picking with analytic UV refinement.

The display mesh is used only to obtain a ray hit and a warm UV seed. The
returned parameter coordinates are refined using surface derivatives (a
3-equation ray intersection Newton solve, followed by the analytic closest
point solver), so coarse tessellation vertices are never reported as results.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from . import vecmath as vm
from .solver import closest
from .surfaces import ParamSurface


def _ray_triangle(orig, direction, a, b, c):
    e1 = vm.sub(b, a)
    e2 = vm.sub(c, a)
    pvec = vm.cross(direction, e2)
    det = vm.dot(e1, pvec)
    if abs(det) < 1e-12:
        return None
    inv = 1.0 / det
    tvec = vm.sub(orig, a)
    bu = vm.dot(tvec, pvec) * inv
    if bu < -1e-7 or bu > 1.0 + 1e-7:
        return None
    qvec = vm.cross(tvec, e1)
    bv = vm.dot(direction, qvec) * inv
    if bv < -1e-7 or bu + bv > 1.0 + 1e-7:
        return None
    t = vm.dot(e2, qvec) * inv
    if t < -1e-7:
        return None
    return t, bu, bv


def _solve3(matrix, rhs):
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        pv = a[col][col]
        for j in range(col, 4):
            a[col][j] /= pv
        for r in range(3):
            if r == col:
                continue
            factor = a[r][col]
            for j in range(col, 4):
                a[r][j] -= factor * a[col][j]
    return [a[i][3] for i in range(3)]


def refine_ray_hit(
    surf: ParamSurface,
    orig: Sequence[float],
    direction: Sequence[float],
    u: float,
    v: float,
    ray_t: float,
    iterations: int = 18,
):
    """Solve S(u,v) = origin + t*direction with analytic derivatives."""
    d = vm.normalize(direction)
    if d is None:
        return None
    t = ray_t
    for _ in range(iterations):
        p, su, sv = surf.derivs(u, v)
        target = vm.add(orig, vm.mul(d, t))
        residual = vm.sub(p, target)
        rn = vm.norm(residual)
        if rn < 2e-8:
            return {
                "u": u, "v": v, "t": t, "xyz": p,
                "rayResidual": rn, "converged": True,
            }
        # J * [du, dv, dt] = -residual; columns are Su, Sv, -D.
        matrix = [
            [su[0], sv[0], -d[0]],
            [su[1], sv[1], -d[1]],
            [su[2], sv[2], -d[2]],
        ]
        delta = _solve3(matrix, [-residual[0], -residual[1], -residual[2]])
        if delta is None:
            return None
        du, dv, dt = delta
        scale = 1.0
        candidate = None
        for _ls in range(12):
            nu, nv, nt = u + scale * du, v + scale * dv, t + scale * dt
            np = surf.point(nu, nv)
            nr = vm.norm(vm.sub(np, vm.add(orig, vm.mul(d, nt))))
            if math.isfinite(nr) and nr < rn:
                candidate = (nu, nv, nt, nr)
                break
            scale *= 0.5
        if candidate is None:
            return None
        u, v, t, rn_new = candidate
        if abs(scale * du) + abs(scale * dv) < 1e-12 and rn_new < 1e-6:
            p = surf.point(u, v)
            return {
                "u": u, "v": v, "t": t, "xyz": p,
                "rayResidual": rn_new, "converged": True,
            }
    p = surf.point(u, v)
    residual = vm.norm(vm.sub(p, vm.add(orig, vm.mul(d, t))))
    return {
        "u": u, "v": v, "t": t, "xyz": p,
        "rayResidual": residual, "converged": residual < 1e-5,
    }


def pick_surface_ray(
    surf: ParamSurface,
    orig: Sequence[float],
    direction: Sequence[float],
    mesh_nu: int = 56,
    mesh_nv: int = 38,
) -> Optional[dict]:
    d = vm.normalize(direction)
    if d is None:
        return None
    mesh = surf.mesh(mesh_nu, mesh_nv)
    pos = mesh["positions"]
    uvs = mesh["uvs"]
    best = None

    for quad in mesh["indices"]:
        ia, ib, ic, idd = quad
        triangles = ((ia, ib, ic), (ia, ic, idd))
        for i0, i1, i2 in triangles:
            hit = _ray_triangle(orig, d, pos[i0], pos[i1], pos[i2])
            if hit is None:
                continue
            t, bu, bv = hit
            if best is not None and t >= best["meshT"]:
                continue
            uv0, uv1, uv2 = uvs[i0], uvs[i1], uvs[i2]
            warm_u = uv0[0] * (1 - bu - bv) + uv1[0] * bu + uv2[0] * bv
            warm_v = uv0[1] * (1 - bu - bv) + uv1[1] * bu + uv2[1] * bv
            xyz = vm.add(orig, vm.mul(d, t))
            best = {
                "meshT": t, "warmU": warm_u, "warmV": warm_v,
                "meshHit": xyz,
            }

    if best is None:
        return None

    refined = refine_ray_hit(
        surf, orig, d, best["warmU"], best["warmV"], best["meshT"]
    )
    if refined is not None and refined["converged"]:
        u, v = surf.wrap(refined["u"], refined["v"])
        xyz = tuple(surf.point(u, v))
        return {
            "surfaceId": surf.id,
            "u": u,
            "v": v,
            "xyz": list(xyz),
            "source": "analytic-ray-derivative-refinement",
            "rayResidual": refined["rayResidual"],
            "meshUv": [best["warmU"], best["warmV"]],
        }

    res = closest(surf, best["meshHit"], warm_uv=(best["warmU"], best["warmV"]))
    b = res.best
    return {
        "surfaceId": surf.id,
        "u": b.u,
        "v": b.v,
        "xyz": list(b.xyz),
        "distance": b.distance,
        "orthoRel": b.ortho_rel,
        "location": b.location,
        "oobReasons": list(b.oob_reasons),
        "source": "analytic-closest-after-mesh-seed",
        "meshUv": [best["warmU"], best["warmV"]],
        "solverWarnings": res.warnings,
    }


def pick_ray(surfaces: Sequence[ParamSurface], orig, direction) -> dict:
    hits = []
    for surf in surfaces:
        hit = pick_surface_ray(surf, orig, direction)
        if hit is not None:
            hits.append(hit)
    if not hits:
        return {"hit": False, "picks": []}
    origin = tuple(orig)
    d = vm.normalize(direction)
    for hit in hits:
        hit["rayDistance"] = vm.dist(origin, hit["xyz"]) if d else None
    hits.sort(key=lambda h: h.get("rayDistance", float("inf")))
    return {"hit": True, "pick": hits[0], "picks": hits}
