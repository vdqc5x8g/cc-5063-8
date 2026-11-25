"""Path-level parameter-branch tracking for multi-solution projection.

A single closest-point solve can return several well-separated candidates on
the same surface (for example the two sheets of a fold-back).  Choosing the
nearest candidate independently at every sample yields the globally nearest
single points but not a continuous, usable curve.  This module keeps the
candidate branches alive along the whole design curve and selects the route
with an explicit distance / continuity cost instead of smoothing jumps later.

Two strategies are implemented and can be measured against each other:

``neighbor``
    Greedy continuation: among candidates within a distance guard of the
    nearest one, follow the smallest parameter step.

``path``
    Viterbi dynamic programming over the full candidate graph.  The emission
    cost is the real attachment distance, the transition cost is the unwrapped
    parameter jump plus a branch-change penalty.  Look-ahead lets the route
    cross a ridge only when the accumulated distance gain justifies it.

``auto`` runs both and keeps the lower total cost, recording the totals so the
choice is auditable.

Rules on top of the graph:

* User branch anchors are hard constraints: the anchored candidate is pinned
  at its sample and its branch identity is locked for the following samples.
* If a locked branch is lost (typically after a seam) the tracker resumes at
  the nearest feasible candidate and records the recovery.
* If an anchor can only be honoured by jumping over a disconnected region the
  tracker reports a break instead of inventing an in-air connector.

Every chosen point carries the abandoned candidates and the selection reason
so the decision can be audited in the UI.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional, Sequence

from . import vecmath as vm
from .solver import Candidate, ClosestResult, _interior_solve, closest
from .surfaces import ParamSurface

STRATEGIES = ("neighbor", "path")
DEFAULT_STRATEGY = "auto"


@dataclass
class TrackCandidate:
    u: float
    v: float
    xyz: tuple
    distance: float
    ortho_rel: float
    location: str = "interior"
    oob_reasons: list = field(default_factory=list)
    converged: bool = False
    rank_deficient: bool = False
    normal_degenerate: bool = False
    seed_kind: str = "grid"

    @classmethod
    def from_candidate(cls, c: Candidate) -> "TrackCandidate":
        return cls(
            u=float(c.u), v=float(c.v), xyz=tuple(c.xyz), distance=float(c.distance),
            ortho_rel=float(c.ortho_rel), location=c.location,
            oob_reasons=list(c.oob_reasons), converged=bool(c.converged),
            rank_deficient=bool(c.rank_deficient),
            normal_degenerate=bool(c.normal_degenerate), seed_kind=c.seed_kind,
        )

    @property
    def oob_reason(self) -> Optional[str]:
        return "|".join(self.oob_reasons) if self.oob_reasons else None

    @property
    def usable(self) -> bool:
        """Can this candidate carry a continuous interior route?

        A parameter singularity (rank-deficient metric) is still a real point
        on the surface and stays usable; it is flagged separately.  A verified
        boundary exit is not usable for the interior route.
        """
        return bool(self.converged) and not self.oob_reasons

    def to_dict(self) -> dict:
        return {
            "u": self.u,
            "v": self.v,
            "xyz": list(self.xyz),
            "distance": self.distance,
            "orthoRel": self.ortho_rel,
            "location": self.location,
            "oobReason": self.oob_reason,
            "oobReasons": list(self.oob_reasons),
            "converged": self.converged,
            "rankDeficient": self.rank_deficient,
            "normalDegenerate": self.normal_degenerate,
            "seedKind": self.seed_kind,
        }


@dataclass
class TrackFrame:
    index: int
    candidate: TrackCandidate
    candidates: list
    u_unwrapped: float
    v_unwrapped: float
    jump_param: Optional[float]
    jump_xyz: Optional[float]
    continuity_ok: bool
    status: str
    reason: str
    alternatives: list
    anchor: Optional[dict] = None
    locked: bool = False
    recovery: Optional[dict] = None
    break_before: Optional[dict] = None

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "chosen": self.candidate.to_dict(),
            "candidateCount": len(self.candidates),
            "candidates": [c.to_dict() for c in self.candidates],
            "uUnwrapped": self.u_unwrapped,
            "vUnwrapped": self.v_unwrapped,
            "jumpParam": self.jump_param,
            "jumpXyz": self.jump_xyz,
            "continuityOk": self.continuity_ok,
            "status": self.status,
            "reason": self.reason,
            "abandoned": self.alternatives,
            "anchor": self.anchor,
            "locked": self.locked,
            "recovery": self.recovery,
            "breakBefore": self.break_before,
        }


@dataclass
class BranchTrack:
    surface_id: str
    strategy: str
    frames: list
    breaks: list
    warnings: list
    stats: dict
    comparison: dict
    real_attachment_error: dict

    def to_dict(self) -> dict:
        return {
            "surfaceId": self.surface_id,
            "strategy": self.strategy,
            "frames": [f.to_dict() for f in self.frames],
            "breaks": list(self.breaks),
            "warnings": list(self.warnings),
            "stats": self.stats,
            "strategyComparison": self.comparison,
            "attachmentError": self.real_attachment_error,
        }


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------
def _periods(surf: ParamSurface) -> tuple[float, float]:
    pu = surf.u1 - surf.u0 if surf.periodic_u else 0.0
    pv = surf.v1 - surf.v0 if surf.periodic_v else 0.0
    return pu, pv


def unwrap_near(prev: Optional[float], raw: float, period: float) -> float:
    if prev is None or period <= 0.0:
        return raw
    return raw + round((prev - raw) / period) * period


def param_delta(surf: ParamSurface, a_u, a_v, b_u, b_v) -> float:
    """Normalised parameter distance; periodic coordinates take the short way."""
    ur = max(surf.u1 - surf.u0, 1e-12)
    vr = max(surf.v1 - surf.v0, 1e-12)
    pu, pv = _periods(surf)
    du = b_u - a_u
    dv = b_v - a_v
    if pu > 0:
        du -= round(du / pu) * pu
    if pv > 0:
        dv -= round(dv / pv) * pv
    return math.hypot(du / ur, dv / vr)


def _surface_scale(surf: ParamSurface) -> float:
    m = surf.metric(0.5 * (surf.u0 + surf.u1), 0.5 * (surf.v0 + surf.v1))
    return max(math.sqrt(max(m["areaScale"], 1e-14)), 1e-6)


def _distance_guard(surface_scale: float, nearest: float) -> float:
    """How much worse than the nearest candidate a continuation may be.

    A purely relative guard collapses at a seam where the nearest distance is
    near zero; combine it with a floor scaled by the surface size.
    """
    return max(0.02, 0.15 * nearest + 0.02 * surface_scale)


# ---------------------------------------------------------------------------
# Candidate collection
# ---------------------------------------------------------------------------
def collect_candidates(
    surf: ParamSurface,
    query,
    warm_starts: Sequence[tuple[float, float]] = (),
    full: bool = True,
) -> tuple[list[TrackCandidate], str, list[str]]:
    """Candidates for one sample, retaining neighbouring solutions.

    Every carried branch is first continued from its previous parameter value
    (a local solve), so a branch whose basin the global seed grid would miss is
    still available.  A full multi-start search is merged in when requested or
    when no carried branch survives, which also surfaces newly appearing
    basins.  The returned mode is ``full`` when the global search ran.
    """
    warnings: list[str] = []
    cands: list[TrackCandidate] = []
    for uv in warm_starts:
        if not surf.contains(*uv):
            continue
        c = _interior_solve(surf, query, uv[0], uv[1], "warm-track")
        tc = TrackCandidate.from_candidate(c)
        if tc.usable and tc.location == "interior":
            cands.append(tc)
    ran_full = full or not cands
    if ran_full:
        warm = warm_starts[0] if warm_starts else None
        res: ClosestResult = closest(surf, query, warm_uv=warm)
        warnings.extend(res.warnings)
        cands.extend(TrackCandidate.from_candidate(c) for c in res.candidates)
        cands.extend(TrackCandidate.from_candidate(b) for b in res.branches)
    cands.sort(key=lambda c: c.distance)
    kept: list[TrackCandidate] = []
    for c in cands:
        if any(
            abs(c.u - k.u) < 1e-9
            and abs(c.v - k.v) < 1e-9
            and abs(c.distance - k.distance) < 1e-9
            for k in kept
        ):
            continue
        kept.append(c)
    return kept, ("full" if ran_full else "warm"), warnings


def _nearest(cands: list) -> TrackCandidate:
    return min(cands, key=lambda c: c.distance)


def design_gaps(design_points: Sequence) -> list[float]:
    gaps = [0.0]
    for a, b in zip(design_points, design_points[1:]):
        gaps.append(vm.dist(a, b))
    return gaps


def _carry_uvs(cands: list, limit: int = 8) -> list[tuple[float, float]]:
    """Interior branch positions to continue into the next sample.

    Boundary and degenerate parameter branches are not carried: they are not
    valid starts for a local interior solve and are recovered by the full scan.
    """
    usable = [
        c for c in cands
        if c.usable and c.location == "interior"
        and not c.rank_deficient and not c.normal_degenerate
    ]
    usable.sort(key=lambda c: c.distance)
    return [(c.u, c.v) for c in usable[:limit]]


def solve_chain(
    surf: ParamSurface,
    design_points: Sequence,
    warm_every: int,
) -> tuple[list[list[TrackCandidate]], list[str], dict]:
    """Solve every sample, carrying every live branch forward."""
    n = len(design_points)
    cands_per: list[list[TrackCandidate]] = []
    warnings: list[str] = []
    full_count = 0
    warm_count = 0
    carries: list[tuple[float, float]] = []
    prev_ambiguous = True
    for i, q in enumerate(design_points):
        full = (
            i == 0
            or prev_ambiguous
            or not carries
            or i % max(1, warm_every) == 0
        )
        cands, mode, w = collect_candidates(surf, q, carries, full=full)
        if not cands:
            raise RuntimeError(f"no candidate for sample {i} on {surf.id}")
        if mode == "full":
            full_count += 1
        else:
            warm_count += 1
        for msg in w:
            stamped = f"[{surf.id}] {msg}"
            if stamped not in warnings:
                warnings.append(stamped)
        cands_per.append(cands)
        carries = _carry_uvs(cands)
        prev_ambiguous = len(cands) > 1
    return cands_per, warnings, {"full": full_count, "warm": warm_count}


# ---------------------------------------------------------------------------
# User anchors
# ---------------------------------------------------------------------------
def parse_anchors(
    raw: Optional[Sequence[dict]], surf: ParamSurface, n_samples: int
) -> list[dict]:
    """Normalise user branch anchors for one surface.

    Accepted keys: ``index`` (sample index) or ``t`` (normalised ``[0, 1]``
    position); ``u``/``v`` (parameter identity of the branch) or ``xyz`` (world
    position used to match a candidate).  Anchors of other surfaces are ignored.
    """
    anchors: list[dict] = []
    if not raw or n_samples <= 0:
        return anchors
    for item in raw:
        if item.get("surfaceId", surf.id) != surf.id:
            continue
        if item.get("index") is not None:
            index = int(round(float(item["index"])))
        elif item.get("t") is not None:
            index = int(round(float(item["t"]) * (n_samples - 1)))
        else:
            continue
        index = max(0, min(n_samples - 1, index))
        u, v, xyz = item.get("u"), item.get("v"), item.get("xyz")
        anchors.append({
            "index": index,
            "u": None if u is None else float(u),
            "v": None if v is None else float(v),
            "xyz": None if xyz is None else tuple(float(k) for k in xyz),
            "note": item.get("note"),
        })
    anchors.sort(key=lambda a: a["index"])
    return anchors


def _resolve_anchor(anchor: dict, cands: list) -> Optional[int]:
    usable = [j for j, c in enumerate(cands) if c.usable]
    pool = usable or list(range(len(cands)))
    if anchor.get("u") is not None and anchor.get("v") is not None:
        a_u, a_v = anchor["u"], anchor["v"]
        return min(pool, key=lambda j: abs(cands[j].u - a_u) + abs(cands[j].v - a_v))
    if anchor.get("xyz") is not None:
        ax = anchor["xyz"]
        return min(pool, key=lambda j: vm.dist(cands[j].xyz, ax))
    return min(pool, key=lambda j: cands[j].distance)


def _lock_chains(
    surf: ParamSurface,
    cands_per: list,
    anchors: list,
    max_jump: float,
) -> tuple[dict, list, dict]:
    """Pin anchors and follow each pinned branch forward until it is lost."""
    forced: dict[int, int] = {}
    losses: list[dict] = []
    anchor_info: dict[int, dict] = {}
    for ai, anchor in enumerate(anchors):
        i0 = anchor["index"]
        ci = _resolve_anchor(anchor, cands_per[i0])
        if ci is None:
            continue
        forced[i0] = ci
        anchor_info[i0] = {
            "u": cands_per[i0][ci].u,
            "v": cands_per[i0][ci].v,
            "distance": cands_per[i0][ci].distance,
            "candidateIndex": ci,
            "note": anchor.get("note"),
        }
        next_index = anchors[ai + 1]["index"] if ai + 1 < len(anchors) else len(cands_per)
        cur = ci
        for i in range(i0 + 1, next_index):
            prev = cands_per[i - 1][cur]
            feasible = [
                j for j, c in enumerate(cands_per[i])
                if c.usable and param_delta(surf, prev.u, prev.v, c.u, c.v) <= max_jump
            ]
            if not feasible:
                losses.append({
                    "anchorIndex": i0,
                    "lostAt": i,
                    "lastU": prev.u,
                    "lastV": prev.v,
                    "lastXyz": list(prev.xyz),
                    "resumedAt": None,
                })
                break
            cur = min(
                feasible,
                key=lambda j: (param_delta(surf, prev.u, prev.v,
                                           cands_per[i][j].u, cands_per[i][j].v),
                               cands_per[i][j].distance),
            )
            forced[i] = cur
    return forced, losses, anchor_info


# ---------------------------------------------------------------------------
# Route solvers
# ---------------------------------------------------------------------------
INF = float("inf")


def _emission(cand: TrackCandidate, scale: float, distance_weight: float) -> float:
    return distance_weight * max(cand.distance, 0.0) / max(scale, 1e-9)


def _transition(
    surf: ParamSurface,
    a: TrackCandidate,
    b: TrackCandidate,
    continuity_weight: float,
    switch_penalty: float,
) -> float:
    jump = param_delta(surf, a.u, a.v, b.u, b.v)
    changed = 0.0 if (abs(a.u - b.u) < 1e-9 and abs(a.v - b.v) < 1e-9) else switch_penalty
    return continuity_weight * jump + changed


def _path_cost(surf, cands_per, path, scale, weights) -> float:
    total = sum(_emission(cands_per[i][path[i]], scale, weights["distance"])
                for i in range(len(path)))
    total += sum(
        _transition(surf, cands_per[i - 1][path[i - 1]], cands_per[i][path[i]],
                    weights["continuity"], weights["switch"])
        for i in range(1, len(path))
    )
    return total


def _usable_sets(cands_per: list, forced: dict) -> tuple[list, dict]:
    """Usable candidate indices per sample for route optimisation."""
    sets = []
    for i, cands in enumerate(cands_per):
        usable = [j for j, c in enumerate(cands) if c.usable]
        if not usable:
            # No usable candidate: keep the nearest as the sole node so the
            # route can still report an explicit oob frame.
            usable = [int(min(range(len(cands)), key=lambda j: cands[j].distance))]
        sets.append(usable)
    forced_local = {
        i: sets[i].index(ci) for i, ci in forced.items() if ci in sets[i]
    }
    return sets, forced_local


def _viterbi(surf, cands_per, scale, weights, forced) -> tuple[list[int], float]:
    sets, forced_local = _usable_sets(cands_per, forced)
    n = len(cands_per)
    if n == 0:
        return [], 0.0

    cost: list[list[float]] = []
    back: list[list[int]] = []
    for i in range(n):
        nodes = sets[i]
        row = [INF] * len(nodes)
        brow = [-1] * len(nodes)
        allowed = [forced_local[i]] if i in forced_local else list(range(len(nodes)))
        if i == 0:
            for a in allowed:
                row[a] = _emission(cands_per[i][nodes[a]], scale, weights["distance"])
        else:
            for a in allowed:
                b_global = cands_per[i][nodes[a]]
                em = _emission(b_global, scale, weights["distance"])
                best_k, best_v = -1, INF
                for k, a_global in enumerate(sets[i - 1]):
                    if cost[i - 1][k] == INF:
                        continue
                    v = cost[i - 1][k] + _transition(
                        surf, cands_per[i - 1][a_global], b_global,
                        weights["continuity"], weights["switch"],
                    )
                    if v < best_v:
                        best_k, best_v = k, v
                row[a] = em + best_v
                brow[a] = best_k
        cost.append(row)
        back.append(brow)

    last_row = cost[-1]
    end = min(range(len(last_row)), key=lambda a: last_row[a])
    local_path = [end]
    for i in range(n - 1, 0, -1):
        k = back[i][local_path[-1]]
        if k < 0:
            k = int(min(range(len(sets[i - 1])),
                        key=lambda a: cands_per[i - 1][sets[i - 1][a]].distance))
        local_path.append(k)
    local_path.reverse()
    path = [sets[i][local_path[i]] for i in range(n)]
    return path, _path_cost(surf, cands_per, path, scale, weights)


def _greedy(surf, cands_per, scale, weights, forced, max_jump) -> tuple[list[int], float]:
    sets, forced_local = _usable_sets(cands_per, forced)
    n = len(cands_per)
    if n == 0:
        return [], 0.0
    path: list[int] = []
    for i in range(n):
        nodes = sets[i]
        if i == 0:
            local = forced_local.get(0, int(min(
                range(len(nodes)), key=lambda a: cands_per[0][nodes[a]].distance
            )))
        elif i in forced_local:
            local = forced_local[i]
        else:
            prev = cands_per[i - 1][path[-1]]
            nearest = min((cands_per[i][g] for g in nodes), key=lambda c: c.distance)
            guard = _distance_guard(scale, nearest.distance)
            acceptable = [
                a for a, g in enumerate(nodes)
                if cands_per[i][g].distance <= nearest.distance + guard
                and param_delta(surf, prev.u, prev.v,
                                cands_per[i][g].u, cands_per[i][g].v) <= max_jump
            ]
            if acceptable:
                local = min(
                    acceptable,
                    key=lambda a: (param_delta(surf, prev.u, prev.v,
                                               cands_per[i][nodes[a]].u,
                                               cands_per[i][nodes[a]].v),
                                   cands_per[i][nodes[a]].distance),
                )
            else:
                local = int(min(range(len(nodes)),
                                key=lambda a: cands_per[i][nodes[a]].distance))
        path.append(nodes[local])
    return path, _path_cost(surf, cands_per, path, scale, weights)


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------
def _build_frames(
    surf: ParamSurface,
    cands_per: list,
    path: list,
    forced: dict,
    anchor_info: dict,
    losses: list,
    design_gap: list,
    max_jump: float,
    xyz_floor: float,
    xyz_design_ratio: float,
) -> tuple[list[TrackFrame], list[dict]]:
    frames: list[TrackFrame] = []
    breaks: list[dict] = []
    prev: Optional[TrackFrame] = None
    loss_at = {loss["lostAt"]: loss for loss in losses}
    pending_loss = None
    pu, pv = _periods(surf)

    for i, cidx in enumerate(path):
        cands = cands_per[i]
        chosen = cands[cidx]
        if pending_loss is None and i in loss_at:
            pending_loss = loss_at[i]

        jump_param = jump_xyz = None
        continuity = True
        if prev is None:
            u_un, v_un = chosen.u, chosen.v
        else:
            u_un = unwrap_near(prev.u_unwrapped, chosen.u, pu)
            v_un = unwrap_near(prev.v_unwrapped, chosen.v, pv)
            jump_param = param_delta(surf, prev.u_unwrapped, prev.v_unwrapped, u_un, v_un)
            jump_xyz = vm.dist(prev.candidate.xyz, chosen.xyz)
            # The physical gap a sample may cover scales with how far the design
            # curve itself moved, so a sparse but smooth design is not flagged.
            xyz_limit = max(xyz_floor, xyz_design_ratio * design_gap[i])
            continuity = jump_param <= max_jump and jump_xyz <= xyz_limit

        status, reason = "attached", "route_start"
        recovery = None
        break_before = None

        if not chosen.usable:
            status = "oob" if chosen.oob_reasons else "singular-fallback"
            reason = chosen.oob_reason or "candidate_not_usable"
        elif prev is None:
            reason = "user_branch_anchor_at_start" if i in anchor_info else "route_start"
        elif jump_xyz is not None and jump_xyz > max(xyz_floor, xyz_design_ratio * design_gap[i]):
            status = "break"
            reason = "PHYSICAL_DISCONTINUITY_BEYOND_TOLERANCE"
            break_before = {"index": prev.index, "reason": reason}
            breaks.append({
                "beforeIndex": prev.index, "afterIndex": i,
                "xyzJump": jump_xyz, "paramJump": jump_param,
                "kind": "anchor-not-continuable" if i in anchor_info else "disconnected",
                "atAnchor": bool(i in anchor_info),
            })
        elif jump_param is not None and jump_param > max_jump:
            if i in anchor_info:
                status = "break"
                reason = "ANCHOR_FORCES_NON_CONTINUOUS_PARAMETER_JUMP"
                break_before = {"index": prev.index, "reason": reason}
                breaks.append({
                    "beforeIndex": prev.index, "afterIndex": i,
                    "xyzJump": jump_xyz, "paramJump": jump_param,
                    "kind": "anchor-not-continuable", "atAnchor": True,
                })
            elif pending_loss is not None:
                status = "recovered"
                reason = "NEAREST_FEASIBLE_AFTER_LOCKED_BRANCH_LOST"
                recovery = {
                    "fromIndex": pending_loss["anchorIndex"],
                    "lostAt": pending_loss["lostAt"],
                    "resumedAt": i,
                    "xyzJump": jump_xyz,
                    "paramJump": jump_param,
                    "accepted": True,
                }
                pending_loss["resumedAt"] = i
                pending_loss = None
            else:
                status = "branch-switch"
                reason = "PARAMETER_BRANCH_CHANGED_CONTINUOUS_PHYSICAL_PATH"
        elif i in anchor_info:
            status, reason = "anchored", "user_branch_anchor"
        elif i in forced:
            status, reason = "locked", "branch_locked_by_anchor"
        elif chosen.rank_deficient or chosen.normal_degenerate:
            status, reason = "singular", "singular_parameter_branch"

        frames.append(TrackFrame(
            index=i,
            candidate=chosen,
            candidates=cands,
            u_unwrapped=u_un,
            v_unwrapped=v_un,
            jump_param=jump_param,
            jump_xyz=jump_xyz,
            continuity_ok=continuity,
            status=status,
            reason=reason,
            alternatives=[c.to_dict() for j, c in enumerate(cands) if j != cidx],
            anchor=anchor_info.get(i),
            locked=i in forced,
            recovery=recovery,
            break_before=break_before,
        ))
        prev = frames[-1]
    return frames, breaks


def _attachment_error(surf: ParamSurface, design_points, frames) -> dict:
    """Real attachment error of the final route, not just solver residuals.

    ``ortho_rel`` is normalised by the residual length, so it is undefined when
    the query itself lies on the surface.  Those samples are excluded from the
    relative figure and reported separately; the reprojection error (distance
    from the reported point back to ``surface(u, v)``) is always meaningful.
    """
    max_ortho = 0.0
    max_interior_ortho = 0.0
    max_reproj = 0.0
    on_surface = 0
    boundary = 0
    for f in frames:
        if f.candidate.distance > 1e-9:
            max_ortho = max(max_ortho, f.candidate.ortho_rel)
            # A boundary minimum is orthogonal only in the projected sense; its
            # residual along the free direction is legitimately nonzero.
            if f.candidate.location == "interior":
                max_interior_ortho = max(max_interior_ortho, f.candidate.ortho_rel)
            else:
                boundary += 1
        else:
            on_surface += 1
        q = surf.point(f.candidate.u, f.candidate.v)
        max_reproj = max(max_reproj, vm.dist(q, f.candidate.xyz))
    dists = [f.candidate.distance for f in frames]
    return {
        "maxOrthoRel": max_ortho,
        "maxInteriorOrthoRel": max_interior_ortho,
        "maxSurfaceReprojectionError": max_reproj,
        "onSurfaceSamples": on_surface,
        "boundarySamples": boundary,
        "maxDistance": max(dists) if dists else None,
        "meanDistance": (sum(dists) / len(dists)) if dists else None,
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def track_candidates(
    surf: ParamSurface,
    cands_per: list,
    design_points: Sequence,
    strategy: str = DEFAULT_STRATEGY,
    anchors: Optional[Sequence[dict]] = None,
    continuity_weight: float = 1.0,
    distance_weight: float = 1.0,
    switch_penalty: float = 1.0,
    param_jump: float = 0.28,
    xyz_floor: float = 0.18,
    xyz_design_ratio: float = 2.0,
) -> BranchTrack:
    """Select one candidate per sample with path-level continuity.

    ``cands_per`` is the pre-solved candidate list for each sample, so callers
    that already solved the points do not pay for a second search.
    """
    if strategy not in STRATEGIES + ("auto",):
        raise ValueError(f"unknown strategy {strategy!r}")
    if len(cands_per) != len(design_points):
        raise ValueError("candidate lists and design points must align")

    n = len(design_points)
    scale = _surface_scale(surf)
    warnings: list[str] = []
    anchors_norm = parse_anchors(anchors, surf, n)
    forced, losses, anchor_info = _lock_chains(
        surf, cands_per, anchors_norm, max_jump=param_jump
    )
    weights = {
        "continuity": continuity_weight,
        "distance": distance_weight,
        "switch": switch_penalty,
    }

    results: dict[str, tuple[list[int], float]] = {}
    if strategy in ("neighbor", "auto"):
        results["neighbor"] = _greedy(surf, cands_per, scale, weights, forced, param_jump)
    if strategy in ("path", "auto"):
        results["path"] = _viterbi(surf, cands_per, scale, weights, forced)

    if strategy == "auto":
        chosen_strategy = min(results, key=lambda k: results[k][1])
        comparison = {
            "mode": "measured-min-total-cost",
            "totals": {k: v[1] for k, v in results.items()},
            "chosen": chosen_strategy,
            "routesAgree": results["neighbor"][0] == results["path"][0],
        }
    else:
        chosen_strategy = strategy
        comparison = {
            "mode": "requested",
            "totals": {strategy: results[strategy][1]},
            "chosen": strategy,
            "routesAgree": True,
        }

    path, total = results[chosen_strategy]
    gaps = design_gaps(design_points)
    frames, breaks = _build_frames(
        surf, cands_per, path, forced, anchor_info, losses, gaps,
        param_jump, xyz_floor, xyz_design_ratio,
    )

    # A loss that was never resumed is still a lost branch, reported as an
    # unrecovered break in the tracked route.
    unresolved = [l for l in losses if l["resumedAt"] is None]
    jump_params = [f.jump_param for f in frames if f.jump_param is not None]
    stats = {
        "samples": n,
        "candidateTotal": sum(len(c) for c in cands_per),
        "ambiguousSamples": sum(1 for c in cands_per if len(c) > 1),
        "maxParamStep": max(jump_params) if jump_params else 0.0,
        "medianParamStep": statistics.median(jump_params) if jump_params else 0.0,
        "maxXyzStep": max((f.jump_xyz for f in frames if f.jump_xyz is not None),
                          default=0.0),
        "branchSwitches": sum(1 for f in frames if f.status == "branch-switch"),
        "recoveries": sum(1 for f in frames if f.status == "recovered"),
        "oobSamples": sum(1 for f in frames if f.status == "oob"),
        "breaks": len(breaks),
        "unresolvedLosses": len(unresolved),
        "anchoredSamples": sum(1 for f in frames if f.anchor is not None),
        "lockedSamples": sum(1 for f in frames if f.locked),
        "totalCost": total,
        "distanceSum": sum(f.candidate.distance for f in frames),
    }
    if breaks:
        warnings.append(
            "ROUTE_NOT_CONTINUOUS: physical gap exceeds tolerance at samples "
            + ", ".join(str(b["afterIndex"]) for b in breaks)
            + "; no in-air connector was generated"
        )
    if unresolved:
        warnings.append(
            "ANCHORED_BRANCH_LOST_WITHOUT_RESUME: "
            + ", ".join(str(l["lostAt"]) for l in unresolved)
        )

    return BranchTrack(
        surface_id=surf.id,
        strategy=chosen_strategy,
        frames=frames,
        breaks=breaks,
        warnings=warnings,
        stats=stats,
        comparison=comparison,
        real_attachment_error=_attachment_error(surf, design_points, frames),
    )


def track_surface(
    surf: ParamSurface,
    design_points: Sequence,
    strategy: str = DEFAULT_STRATEGY,
    anchors: Optional[Sequence[dict]] = None,
    continuity_weight: float = 1.0,
    distance_weight: float = 1.0,
    switch_penalty: float = 1.0,
    param_jump: float = 0.28,
    xyz_floor: float = 0.18,
    xyz_design_ratio: float = 2.0,
    warm_every: int = 4,
) -> BranchTrack:
    """Solve and track the closest-point branch route on one surface."""
    cands_per, solve_warnings, solve_stats = solve_chain(
        surf, design_points, warm_every
    )
    track = track_candidates(
        surf,
        cands_per,
        design_points,
        strategy=strategy,
        anchors=anchors,
        continuity_weight=continuity_weight,
        distance_weight=distance_weight,
        switch_penalty=switch_penalty,
        param_jump=param_jump,
        xyz_floor=xyz_floor,
        xyz_design_ratio=xyz_design_ratio,
    )
    for msg in solve_warnings:
        if msg not in track.warnings:
            track.warnings.append(msg)
    track.stats["fullSolves"] = solve_stats["full"]
    track.stats["warmSolves"] = solve_stats["warm"]
    return track
