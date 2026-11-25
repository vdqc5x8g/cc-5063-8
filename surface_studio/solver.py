"""Analytic closest-point inverse solver for parametric surfaces.

Pipeline for one query point P:

1. Initial-value strategy: warm start (previous UV) plus a coarse domain
   grid whose best, well-separated points seed different basins.
2. Interior solve: Levenberg-Marquardt / Gauss-Newton on the stationarity
   equations (P-S).S_u = 0, (P-S).S_v = 0 using analytic first derivatives.
   Steps that leave the parameter domain are line-searched back; a result is
   never blindly clamped and accepted.
3. Boundary check: every edge gets bracketed 1-D minimization and all four
   corners are tested. Out-of-domain intent is confirmed from the outward
   directional derivative (KKT sign), producing an explicit reason code.
4. Singularities: when the metric loses rank (e.g. a sphere pole) several
   equivalent inverse branches are retained instead of one arbitrary UV.
5. Search limitation: a finite multi-start grid cannot prove global
   optimality on arbitrary non-convex surfaces, so the result carries
   ``global_guaranteed=False`` and a warning unless the surface is a plane.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from . import vecmath as vm
from .surfaces import ParamSurface

EPS = 1e-12
SING_AREA_EPS = 1e-7


@dataclass
class Candidate:
    u: float
    v: float
    xyz: tuple
    distance: float
    ortho: float
    ortho_rel: float
    location: str = "interior"
    oob_reasons: list = field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    rank_deficient: bool = False
    normal_degenerate: bool = False
    seed_kind: str = "grid"

    @property
    def oob_reason(self) -> Optional[str]:
        return "|".join(self.oob_reasons) if self.oob_reasons else None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["oobReason"] = self.oob_reason
        d.pop("oob_reasons", None)
        return d


@dataclass
class ClosestResult:
    surface_id: str
    query: tuple
    best: Candidate
    candidates: list
    branches: list
    degenerate: bool
    global_guaranteed: bool
    search_strategy: str
    warnings: list
    seed_count: int

    def to_dict(self) -> dict:
        return {
            "surfaceId": self.surface_id,
            "query": list(self.query),
            "best": self.best.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "branches": [b.to_dict() for b in self.branches],
            "degenerate": self.degenerate,
            "globalGuaranteed": self.global_guaranteed,
            "searchStrategy": self.search_strategy,
            "warnings": self.warnings,
            "seedCount": self.seed_count,
        }


def _dist2(surf: ParamSurface, p, u: float, v: float) -> float:
    q = surf.point(u, v)
    d = vm.sub(q, p)
    return vm.dot(d, d)


def _ortho(surf, p, u, v):
    q, pu, pv = surf.derivs(u, v)
    r = vm.sub(q, p)
    fu, fv = vm.dot(r, pu), vm.dot(r, pv)
    scale = max(
        1e-12,
        vm.norm(r) * max(vm.norm(pu), vm.norm(pv)),
    )
    return max(abs(fu), abs(fv)), max(abs(fu), abs(fv)) / scale, fu, fv


def _interior_solve(
    surf: ParamSurface,
    p,
    u: float,
    v: float,
    seed_kind: str,
    max_iter: int = 60,
) -> Candidate:
    u0, u1, v0, v1 = surf.u0, surf.u1, surf.v0, surf.v1
    ur, vr = u1 - u0, v1 - v0
    # Start seeds slightly inside so derivatives at exact boundaries are fine.
    u = vm.clamp(u, u0 + 1e-10 * ur, u1 - 1e-10 * ur)
    v = vm.clamp(v, v0 + 1e-10 * vr, v1 - 1e-10 * vr)
    lam = 1e-4
    iterations = 0
    rank_def = False
    wanted_oob: list[str] = []

    for it in range(max_iter):
        iterations = it + 1
        q, pu, pv = surf.derivs(u, v)
        r = vm.sub(q, p)
        fu, fv = vm.dot(r, pu), vm.dot(r, pv)
        ortho = max(abs(fu), abs(fv))
        npu, npv = vm.norm(pu), vm.norm(pv)
        ortho_rel = ortho / max(1e-12, vm.norm(r) * max(npu, npv))
        if ortho_rel < 1e-12 or ortho < 1e-11:
            break

        a11 = vm.dot(pu, pu)
        a12 = vm.dot(pu, pv)
        a22 = vm.dot(pv, pv)
        det = a11 * a22 - a12 * a12
        if det < 1e-16 * max(1.0, a11 * a22):
            rank_def = True

        fcur = vm.dot(r, r)
        accepted = False
        local_lam = lam
        du = dv = 0.0
        for _lam_try in range(18):
            diag_u = a11 + local_lam * max(a11, 1e-8)
            diag_v = a22 + local_lam * max(a22, 1e-8)
            sol = vm.solve2(diag_u, a12, a12, diag_v, -fu, -fv)
            local_lam *= 6.0
            if sol is None:
                continue
            du, dv = sol
            # Largest step fraction that stays inside the open domain.
            alpha = 1.0
            if du > 0:
                alpha = min(alpha, (u1 - 1e-11 * ur - u) / du)
            elif du < 0:
                alpha = min(alpha, (u0 + 1e-11 * ur - u) / du)
            if dv > 0:
                alpha = min(alpha, (v1 - 1e-11 * vr - v) / dv)
            elif dv < 0:
                alpha = min(alpha, (v0 + 1e-11 * vr - v) / dv)
            alpha = max(0.0, min(1.0, alpha))
            if alpha < 1e-12:
                # Step points straight into a wall.
                if abs(du) > abs(dv) * ur / max(vr, 1e-12):
                    wanted_oob.append("u")
                else:
                    wanted_oob.append("v")
                break
            # Backtracking for sufficient decrease.
            beta = alpha
            for _bt in range(24):
                nu = u + beta * du
                nv = v + beta * dv
                fnew = _dist2(surf, p, nu, nv)
                if math.isfinite(fnew) and fnew < fcur - 1e-17:
                    u, v = nu, nv
                    accepted = True
                    lam = local_lam * 0.2 if local_lam > 1e-11 else 0.0
                    break
                beta *= 0.5
            if accepted:
                break
        if not accepted:
            break
        step = math.hypot(du, dv)
        if step < 1e-13:
            break

    q = surf.point(u, v)
    dist = vm.dist(q, p)
    ortho, ortho_rel, _fu, _fv = _ortho(surf, p, u, v)
    metric = surf.metric(u, v)
    ndeg = surf.normal(u, v) is None or metric["areaScale"] < 1e-9
    converged = ortho_rel < 1e-8 or ortho < 1e-7
    loc, reasons = _classify_boundary(surf, p, u, v)
    if wanted_oob and loc == "interior":
        # Step wanted outside but line search held us inside; surface verdict
        # is authoritative, but keep the hint if classification disagrees.
        pass
    return Candidate(
        u=u,
        v=v,
        xyz=tuple(q),
        distance=dist,
        ortho=ortho,
        ortho_rel=ortho_rel,
        location=loc,
        oob_reasons=reasons,
        iterations=iterations,
        converged=converged,
        rank_deficient=rank_def or metric["areaScale"] < 1e-8,
        normal_degenerate=ndeg,
        seed_kind=seed_kind,
    )


def _classify_boundary(surf: ParamSurface, p, u: float, v: float):
    """Return (location, outward-reason list) using KKT sign checks."""
    u0, u1, v0, v1 = surf.u0, surf.u1, surf.v0, surf.v1
    ru, rv = u1 - u0, v1 - v0
    bu = min(u - u0, u1 - u) <= 2e-8 * ru + 1e-11
    bv = min(v - v0, v1 - v) <= 2e-8 * rv + 1e-11
    reasons: list[str] = []
    loc = "interior"
    if bu or bv:
        at_u0 = (u - u0) <= (u1 - u)
        at_v0 = (v - v0) <= (v1 - v)
        tags = []
        if bu:
            tags.append("u_min" if at_u0 else "u_max")
        if bv:
            tags.append("v_min" if at_v0 else "v_max")
        loc = "+".join(tags)
        _o, _pu, _pv = surf.derivs(u, v)
        r = vm.sub(_o, p)
        fu, fv = vm.dot(r, _pu), vm.dot(r, _pv)
        tu = 1e-7 * max(1e-9, vm.norm(r) * vm.norm(_pu)) + 1e-10
        tv = 1e-7 * max(1e-9, vm.norm(r) * vm.norm(_pv)) + 1e-10
        if bu and at_u0 and fu > tu:
            reasons.append("BEYOND_U_MIN")
        if bu and not at_u0 and fu < -tu:
            reasons.append("BEYOND_U_MAX")
        if bv and at_v0 and fv > tv:
            reasons.append("BEYOND_V_MIN")
        if bv and not at_v0 and fv < -tv:
            reasons.append("BEYOND_V_MAX")
    return loc, reasons


def _golden_min(f, lo: float, hi: float, tol: float = 1e-12) -> tuple[float, float]:
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(80):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = f(d)
        if hi - lo < tol:
            break
    x = 0.5 * (lo + hi)
    return x, f(x)


def _edge_candidates(surf: ParamSurface, p, edge_samples: int = 56) -> list[Candidate]:
    out: list[Candidate] = []
    u0, u1, v0, v1 = surf.u0, surf.u1, surf.v0, surf.v1

    def edge_fixed_u(ufix):
        xs = [v0 + (v1 - v0) * i / edge_samples for i in range(edge_samples + 1)]
        ys = [_dist2(surf, p, ufix, x) for x in xs]
        brackets = []
        for i in range(1, edge_samples):
            if ys[i] <= ys[i - 1] and ys[i] <= ys[i + 1]:
                brackets.append((xs[i - 1], xs[i + 1]))
        if not brackets:
            brackets = [(v0, v1)]
        for lo, hi in brackets:
            vv, _ = _golden_min(lambda x: _dist2(surf, p, ufix, x), lo, hi)
            out.append(_finalize_edge(surf, p, ufix, vv, "edge"))

    def edge_fixed_v(vfix):
        xs = [u0 + (u1 - u0) * i / edge_samples for i in range(edge_samples + 1)]
        ys = [_dist2(surf, p, x, vfix) for x in xs]
        brackets = []
        for i in range(1, edge_samples):
            if ys[i] <= ys[i - 1] and ys[i] <= ys[i + 1]:
                brackets.append((xs[i - 1], xs[i + 1]))
        if not brackets:
            brackets = [(u0, u1)]
        for lo, hi in brackets:
            uu, _ = _golden_min(lambda x: _dist2(surf, p, x, vfix), lo, hi)
            out.append(_finalize_edge(surf, p, uu, vfix, "edge"))

    edge_fixed_u(u0)
    edge_fixed_u(u1)
    edge_fixed_v(v0)
    edge_fixed_v(v1)

    for uu in (u0, u1):
        for vv in (v0, v1):
            out.append(_finalize_edge(surf, p, uu, vv, "corner"))
    return out


def _finalize_edge(surf, p, u: float, v: float, kind: str) -> Candidate:
    q = surf.point(u, v)
    dist = vm.dist(q, p)
    ortho, ortho_rel, _fu, _fv = _ortho(surf, p, u, v)
    loc, reasons = _classify_boundary(surf, p, u, v)
    metric = surf.metric(u, v)
    return Candidate(
        u=u,
        v=v,
        xyz=tuple(q),
        distance=dist,
        ortho=ortho,
        ortho_rel=ortho_rel,
        location=loc if loc != "interior" else kind,
        oob_reasons=reasons,
        iterations=0,
        converged=ortho_rel < 1e-7,
        normal_degenerate=surf.normal(u, v) is None or metric["areaScale"] < 1e-9,
        rank_deficient=metric["areaScale"] < 1e-8,
        seed_kind=kind,
    )


def _select_seeds(
    surf: ParamSurface,
    p,
    grid_n: int,
    max_seeds: int,
    warm: Optional[tuple[float, float]],
) -> list[tuple[float, float, str]]:
    u0, u1, v0, v1 = surf.u0, surf.u1, surf.v0, surf.v1
    ur, vr = max(u1 - u0, 1e-12), max(v1 - v0, 1e-12)
    scored = []
    for i in range(grid_n + 1):
        u = u0 + (u1 - u0) * i / grid_n
        for j in range(grid_n + 1):
            v = v0 + (v1 - v0) * j / grid_n
            scored.append((_dist2(surf, p, u, v), u, v))
    scored.sort(key=lambda z: z[0])
    chosen: list[tuple[float, float, str]] = []
    if warm is not None and surf.contains(warm[0], warm[1]):
        chosen.append((warm[0], warm[1], "warm"))
    sep = 0.13
    for _d, u, v in scored:
        nu, nv = (u - u0) / ur, (v - v0) / vr
        ok = True
        for cu, cv, _kind in chosen:
            cnu, cnv = (cu - u0) / ur, (cv - v0) / vr
            if (nu - cnu) ** 2 + (nv - cnv) ** 2 < sep * sep:
                ok = False
                break
        if ok:
            chosen.append((u, v, "grid"))
        if len(chosen) >= max_seeds:
            break
    return chosen


def _dedupe(surf: ParamSurface, cands: list[Candidate]) -> list[Candidate]:
    u0, u1, v0, v1 = surf.u0, surf.u1, surf.v0, surf.v1
    ur, vr = max(u1 - u0, 1e-12), max(v1 - v0, 1e-12)
    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda z: z.distance):
        dup = False
        for k in kept:
            du = abs(c.u - k.u) / ur
            dv = abs(c.v - k.v) / vr
            if du * du + dv * dv < 4e-10:
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


def _singularity_branches(
    surf: ParamSurface, p, cands: list[Candidate]
) -> tuple[bool, list[Candidate]]:
    """Consolidate equivalent branches at a rank-deficient parameter point."""
    best = min(cands, key=lambda c: c.distance)
    m = surf.metric(best.u, best.v)
    degenerate = best.normal_degenerate or m["areaScale"] < SING_AREA_EPS
    if not degenerate:
        return False, []

    u0, u1, v0, v1 = surf.u0, surf.u1, surf.v0, surf.v1
    # Snap to a collapsed boundary when the solve converged next to it.
    snap_tol = 2e-3 * max(u1 - u0, v1 - v0)

    u = best.u
    if abs(best.u - u0) < snap_tol:
        u = u0
    elif abs(best.u - u1) < snap_tol:
        u = u1

    # Retain a spread of inverse starting parameters plus every distinct
    # converged value. At a coordinate collapse all of these map to the same
    # world point but lead to different branches after the singularity.
    n_branch = 12
    branch_vs = [best.v]
    period = v1 - v0
    if surf.periodic_v:
        for k in range(n_branch):
            branch_vs.append(v0 + period * k / n_branch)
    else:
        for k in range(n_branch + 1):
            branch_vs.append(v0 + period * k / n_branch)
    for c in cands:
        if c.seed_kind in ("grid", "warm", "verify-grid") and (
            c.rank_deficient or c.normal_degenerate
        ):
            branch_vs.append(c.v)

    branches: list[Candidate] = []
    seen = set()
    for v in branch_vs:
        vv = surf.wrap_v(v) if surf.periodic_v else vm.clamp(v, v0, v1)
        key = round((vv - v0) / max(period, 1e-12), 4)
        if key in seen:
            continue
        seen.add(key)
        q = surf.point(u, vv)
        dist = vm.dist(q, p)
        ortho, rel, _f1, _f2 = _ortho(surf, p, u, vv)
        mm = surf.metric(u, vv)
        loc, reasons = _classify_boundary(surf, p, u, vv)
        branches.append(
            Candidate(
                u=u,
                v=vv,
                xyz=tuple(q),
                distance=dist,
                ortho=ortho,
                ortho_rel=rel,
                location=loc or "singularity",
                oob_reasons=reasons,
                converged=True,
                rank_deficient=True,
                normal_degenerate=True,
                seed_kind="singular-branch",
            )
        )
    branches.sort(key=lambda b: b.v)
    return True, branches


def closest(
    surf: ParamSurface,
    point,
    warm_uv: Optional[tuple[float, float]] = None,
    grid_n: int = 16,
    max_seeds: int = 7,
    run_boundaries: bool = True,
) -> ClosestResult:
    """Closest-point inverse solve on ``surf`` for world point ``point``."""
    warnings: list[str] = []
    seeds = _select_seeds(surf, point, grid_n, max_seeds, warm_uv)
    cands: list[Candidate] = []
    for u, v, kind in seeds:
        cands.append(_interior_solve(surf, point, u, v, kind))
    if run_boundaries:
        cands.extend(_edge_candidates(surf, point))
    cands = _dedupe(surf, cands)

    # Adaptive verification scan: a finer grid may expose a missed basin.
    verify_n = 26
    bu, bv, bd = None, None, float("inf")
    u0, u1, v0, v1 = surf.u0, surf.u1, surf.v0, surf.v1
    for i in range(verify_n + 1):
        u = u0 + (u1 - u0) * i / verify_n
        for j in range(verify_n + 1):
            v = v0 + (v1 - v0) * j / verify_n
            d = _dist2(surf, point, u, v)
            if d < bd:
                bd, bu, bv = d, u, v
    current_best_d2 = min(c.distance ** 2 for c in cands)
    if bd < current_best_d2 - 1e-9:
        extra = _interior_solve(surf, point, bu, bv, "verify-grid")
        cands.append(extra)
        cands = _dedupe(surf, cands)
        warnings.append("verification_scan_found_extra_basin")

    degenerate, branches = _singularity_branches(surf, point, cands)
    if branches:
        cands.extend(branches)
        cands = _dedupe(surf, cands)

    best = min(cands, key=lambda c: c.distance)
    if degenerate:
        # Prefer a branch object carrying the explicit singular flags.
        for b in branches:
            if abs(b.u - best.u) < 1e-8 and abs(b.v - best.v) < 1e-6:
                best = b
                break
        warnings.append(
            "parameter_singularity_detected_multiple_inverse_branches_retained"
        )

    guaranteed = surf.kind == "plane"
    strategy = (
        "analytic-convex-plane+multistart-LM+edge-golden+corners"
        if guaranteed
        else "restricted-search:coarse-grid-multistart-LM+edge-bracketing+"
        "corners+verification-scan"
    )
    if not guaranteed:
        warnings.append(
            "GLOBAL_OPTIMUM_NOT_GUARANTEED_RESTRICTED_SEARCH:"
            "finite seeds may miss a basin on a non-convex surface"
        )
    if not best.converged and best.location == "interior":
        warnings.append("solver_did_not_meet_tolerance_from_best_seed")
    return ClosestResult(
        surface_id=surf.id,
        query=tuple(point),
        best=best,
        candidates=sorted(cands, key=lambda c: c.distance),
        branches=branches,
        degenerate=degenerate,
        global_guaranteed=guaranteed,
        search_strategy=strategy,
        warnings=warnings,
        seed_count=len(seeds),
    )


def select_singular_branch(
    surf: ParamSurface,
    result: ClosestResult,
    travel_dir_xyz=None,
    prev_uv: Optional[tuple[float, float]] = None,
    area_ref: Optional[float] = None,
    continuity_weight: float = 1.0,
    distortion_weight: float = 0.35,
) -> dict:
    """Choose among equivalent singularity branches.

    Continuity is measured in physical space when an incoming travel
    direction is available (so crossing a sphere pole correctly selects the
    opposite meridian), with UV proximity as a fallback. Distortion cost uses
    the local metric at each branch. All scores are returned so the choice is
    auditable.
    """
    if not result.branches:
        b = result.best
        return {"chosen": b.to_dict(), "scores": [], "ambiguous": False}

    branches = result.branches
    area_ref = area_ref or 1.0
    scored = []
    u0, u1 = surf.u0, surf.u1
    for b in branches:
        cont = 0.0
        cont_kind = "none"
        if travel_dir_xyz is not None:
            # Probe a small step into the domain along +u to read the
            # outgoing physical tangent of this branch.
            du = max(1e-5, 1e-3 * (u1 - u0))
            u_probe = b.u + du if b.u < 0.5 * (u0 + u1) else b.u - du
            p0 = surf.point(b.u, b.v)
            p1 = surf.point(u_probe, b.v)
            tout = vm.normalize(vm.sub(p1, p0))
            td = vm.normalize(travel_dir_xyz)
            if tout is not None and td is not None:
                cont = 1.0 - max(-1.0, min(1.0, vm.dot(tout, td)))
                cont_kind = "physical-tangent"
        elif prev_uv is not None:
            ur = max(u1 - u0, 1e-12)
            vr = max(surf.v1 - surf.v0, 1e-12)
            du = (b.u - prev_uv[0]) / ur
            dv = b.v - prev_uv[1]
            if surf.periodic_v:
                period = surf.v1 - surf.v0
                dv = (dv + 0.5 * period) % period - 0.5 * period
                dv /= vr
            cont = math.hypot(du, dv)
            cont_kind = "uv-proximity"

        u_probe = b.u
        if abs(b.u - u0) < 1e-12:
            u_probe = b.u + 1e-4 * (u1 - u0)
        elif abs(b.u - u1) < 1e-12:
            u_probe = b.u - 1e-4 * (u1 - u0)
        m = surf.metric(u_probe, b.v)
        area = max(m["areaScale"], 1e-10)
        area_cost = abs(math.log(area / max(area_ref, 1e-10)))
        shape_cost = math.log1p(min(m["shapeRatio"], 1e6) - 1.0) * 0.25
        dist_cost = area_cost + shape_cost
        total = continuity_weight * cont + distortion_weight * dist_cost
        scored.append(
            {
                "branch": b.to_dict(),
                "continuityCost": cont,
                "continuityKind": cont_kind,
                "distortionCost": dist_cost,
                "areaScale": m["areaScale"],
                "shapeRatio": min(m["shapeRatio"], 1e6),
                "total": total,
            }
        )
    scored.sort(key=lambda z: z["total"])
    chosen = scored[0]

    # Continuous refinement of the chosen branch representative: at a
    # coordinate collapse v is physically free, so continuity/distortion
    # determines its proper value instead of grid discretization.
    b_u = chosen["branch"]["u"]
    v0d, v1d = surf.v0, surf.v1
    period = v1d - v0d

    def total_at(vv: float) -> float:
        cont = 0.0
        if travel_dir_xyz is not None:
            du = max(1e-5, 1e-3 * (u1 - u0))
            u_probe = b_u + du if b_u < 0.5 * (u0 + u1) else b_u - du
            p0 = surf.point(b_u, vv)
            p1 = surf.point(u_probe, vv)
            tout = vm.normalize(vm.sub(p1, p0))
            td = vm.normalize(travel_dir_xyz)
            if tout is not None and td is not None:
                cont = 1.0 - max(-1.0, min(1.0, vm.dot(tout, td)))
        elif prev_uv is not None:
            dv = vv - prev_uv[1]
            if surf.periodic_v:
                dv = (dv + 0.5 * period) % period - 0.5 * period
            cont = abs(dv) / max(period, 1e-12)
        u_probe = b_u + 1e-4 * (u1 - u0) if abs(b_u - u0) < 1e-12 else b_u
        m = surf.metric(u_probe, vv)
        area_cost = abs(math.log(max(m["areaScale"], 1e-10) / max(area_ref or 1.0, 1e-10)))
        shape_cost = math.log1p(min(m["shapeRatio"], 1e6) - 1.0) * 0.25
        return continuity_weight * cont + distortion_weight * (area_cost + shape_cost)

    vc = chosen["branch"]["v"]
    half = 0.6 * period / 12.0 if surf.periodic_v else 0.12 * period
    lo, hi = vc - half, vc + half
    if not surf.periodic_v:
        lo, hi = max(v0d, lo), min(v1d, hi)
    if hi - lo > 1e-12:
        vr = _golden_min(lambda x: total_at(x), lo, hi, tol=1e-10)[0]
        vr = surf.wrap_v(vr) if surf.periodic_v else vm.clamp(vr, v0d, v1d)
        refined_total = total_at(vr)
        if refined_total <= chosen["total"] + 1e-12:
            chosen["branch"]["v"] = vr
            q = surf.point(b_u, vr)
            chosen["branch"]["xyz"] = list(q)
            chosen["total"] = refined_total
            chosen["refined"] = True

    ambiguous = len(scored) > 1 and abs(scored[0]["total"] - scored[1]["total"]) < 1e-6
    return {
        "chosen": chosen["branch"],
        "scores": scored,
        "ambiguous": ambiguous,
        "selectionRule": "min(continuityWeight*continuity + distortionWeight*distortion)",
    }
