"""Curve projection with surface identity, continuous intervals and switching.

Rules implemented here:

* Every design sample is inverted with the analytic closest-point solver on
  the active surface (warm start) and, for comparison, on every other surface.
* Surface identity is sticky. Equal nearest distance never triggers an
  automatic face switch; a switch is authorized only at a user-drawn
  transition line in ``mode="transition"`` (crossing the line or hitting the
  verified parameter-domain exit next to it).
* An interval ends when the inverse solver verifies that the unconstrained
  target lies beyond the parameter domain. The verified reason code is kept
  and later handed to the decal clipper.
* Periodic coordinates are unwrapped along the route so a theta=0/2pi seam
  cannot create fake jumps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from . import vecmath as vm
from .solver import (
    Candidate,
    ClosestResult,
    _interior_solve,
    closest,
    select_singular_branch,
)
from .surfaces import ParamSurface


# ---------------------------------------------------------------------------
# Design curve utilities
# ---------------------------------------------------------------------------
def catmull_rom_chain(points: Sequence[vm.Vec3], samples_per_span: int = 10):
    """Dense uniform Catmull-Rom sampling through all control points."""
    pts = [tuple(p) for p in points]
    if len(pts) == 1:
        return pts[:]
    if len(pts) == 2:
        return [
            tuple(pts[0][k] * (1 - i / samples_per_span) + pts[1][k] * (i / samples_per_span)
                  for k in range(3))
            for i in range(samples_per_span + 1)
        ]
    ext = [pts[0]] + pts + [pts[-1]]
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for j in range(samples_per_span):
            t = j / samples_per_span
            t2, t3 = t * t, t * t * t
            q = []
            for k in range(3):
                qk = 0.5 * (
                    2 * p1[k]
                    + (-p0[k] + p2[k]) * t
                    + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
                    + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3
                )
                q.append(qk)
            out.append(tuple(q))
    out.append(pts[-1])
    return out


# ---------------------------------------------------------------------------
# Transition line (user drawn; points carry real picked UV identity)
# ---------------------------------------------------------------------------
@dataclass
class TransitionPoint:
    surface_id: str
    u: float
    v: float
    xyz: tuple = (0.0, 0.0, 0.0)

    def to_dict(self):
        return {
            "surfaceId": self.surface_id,
            "u": self.u,
            "v": self.v,
            "xyz": list(self.xyz),
        }


def make_transition_line(raw: Sequence[dict], surfaces: dict[str, ParamSurface]):
    line = []
    for item in raw:
        sid = item["surfaceId"]
        s = surfaces[sid]
        u, v = float(item["u"]), float(item["v"])
        line.append(TransitionPoint(sid, u, v, tuple(s.point(u, v))))
    return line


def authorize_switch(
    line: Sequence[TransitionPoint],
    current_sid: str,
    near_xyz: vm.Vec3,
    match_tol: float = 0.25,
    index_window: int = 4,
):
    """A drawn current-face anchor plus a neighboring other-face pick."""
    if not line:
        return None
    best_i, best_d = None, float("inf")
    for i, tp in enumerate(line):
        if tp.surface_id != current_sid:
            continue
        d = vm.dist(tp.xyz, near_xyz)
        if d < best_d:
            best_d, best_i = d, i
    if best_i is None or best_d > match_tol:
        return None
    candidates = []
    for j in range(max(0, best_i - index_window), min(len(line), best_i + index_window + 1)):
        tp = line[j]
        if tp.surface_id != current_sid:
            candidates.append((vm.dist(tp.xyz, line[best_i].xyz), j, tp))
    if not candidates:
        return None
    _d, j, target = min(candidates, key=lambda z: z[0])
    return {
        "from_index": best_i,
        "to_index": j,
        "match_distance": best_d,
        "target": target,
    }


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------
@dataclass
class RoutePoint:
    index: int
    t: float
    query: tuple
    status: str
    surface_id: Optional[str] = None
    u: Optional[float] = None
    v: Optional[float] = None
    u_unwrapped: Optional[float] = None
    v_unwrapped: Optional[float] = None
    xyz: Optional[tuple] = None
    distance: Optional[float] = None
    ortho_rel: Optional[float] = None
    location: Optional[str] = None
    oob_reasons: list = field(default_factory=list)
    degenerate: bool = False
    branch_selection: Optional[dict] = None
    distances_by_surface: dict = field(default_factory=dict)
    tie_hold: bool = False
    switch: Optional[dict] = None
    terminal: bool = False
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return {
            "index": self.index,
            "t": self.t,
            "query": list(self.query),
            "status": self.status,
            "surfaceId": self.surface_id,
            "u": self.u,
            "v": self.v,
            "uUnwrapped": self.u_unwrapped,
            "vUnwrapped": self.v_unwrapped,
            "xyz": None if self.xyz is None else list(self.xyz),
            "distance": self.distance,
            "orthoRel": self.ortho_rel,
            "location": self.location,
            "oobReason": "|".join(self.oob_reasons) if self.oob_reasons else None,
            "oobReasons": list(self.oob_reasons),
            "degenerate": self.degenerate,
            "branchSelection": self.branch_selection,
            "distancesBySurface": {
                k: float(v) for k, v in self.distances_by_surface.items()
            },
            "tieHold": self.tie_hold,
            "switch": self.switch,
            "terminal": self.terminal,
            "warnings": list(self.warnings),
        }


@dataclass
class Interval:
    surface_id: str
    index0: int
    index1: int
    t0: float
    t1: float
    arc0: float
    arc1: float
    uv_start: tuple
    uv_end: tuple
    xyz_start: Optional[tuple] = None
    xyz_end: Optional[tuple] = None
    end_reason: Optional[str] = None
    start_reason: str = "initial-pick"
    switch_info: Optional[dict] = None

    def to_dict(self):
        return {
            "surfaceId": self.surface_id,
            "index0": self.index0,
            "index1": self.index1,
            "t0": self.t0,
            "t1": self.t1,
            "arc0": self.arc0,
            "arc1": self.arc1,
            "uvStart": list(self.uv_start),
            "uvEnd": list(self.uv_end),
            "xyzStart": None if self.xyz_start is None else list(self.xyz_start),
            "xyzEnd": None if self.xyz_end is None else list(self.xyz_end),
            "endReason": self.end_reason,
            "startReason": self.start_reason,
            "switchInfo": self.switch_info,
        }


@dataclass
class ProjectionOutcome:
    mode: str
    points: list
    intervals: list
    oob_records: list
    switches: list
    warnings: list
    tie_holds: list
    surface_order: list

    def to_dict(self):
        return {
            "mode": self.mode,
            "points": [p.to_dict() for p in self.points],
            "intervals": [i.to_dict() for i in self.intervals],
            "oobRecords": [o.to_dict() for o in self.oob_records],
            "switches": self.switches,
            "warnings": self.warnings,
            "tieHolds": self.tie_holds,
            "surfaceOrder": self.surface_order,
        }


def _unwrap(prev: Optional[float], raw: float, period: float):
    if prev is None or period <= 0:
        return raw
    k = round((prev - raw) / period)
    return raw + k * period


def project_curve(
    surfaces: Sequence[ParamSurface],
    design_points: Sequence[vm.Vec3],
    mode: str = "spout_only",
    initial_surface_id: Optional[str] = None,
    transition: Optional[Sequence[TransitionPoint]] = None,
    tie_tol: float = 2e-3,
    switch_max_distance: float = 0.8,
    transition_band: float = 0.04,
) -> ProjectionOutcome:
    smap = {s.id: s for s in surfaces}
    order = [s.id for s in surfaces]
    n = len(design_points)

    first_results = {s.id: closest(s, design_points[0]) for s in surfaces}
    if initial_surface_id and initial_surface_id in smap:
        current_sid = initial_surface_id
    else:
        current_sid = min(first_results, key=lambda sid: first_results[sid].best.distance)

    last_uv: dict[str, tuple[float, float]] = {}
    last_unwrapped: dict[str, tuple[float, float]] = {}
    records: list[RoutePoint] = []
    intervals: list[Interval] = []
    switches: list[dict] = []
    tie_holds: list[dict] = []
    warnings_all: list[str] = []
    oob_records: list[RoutePoint] = []

    interval_start = 0
    interval_open = True
    switched = False
    arc_global = 0.0
    interval_arc0 = 0.0
    prev_attached_xyz = None

    for i, query in enumerate(design_points):
        t = i / max(1, n - 1)
        if 0 < i < n - 1:
            travel = vm.normalize(vm.sub(design_points[i + 1], design_points[i - 1]))
        elif n > 1:
            travel = vm.normalize(vm.sub(design_points[1], design_points[0]))
        else:
            travel = None

        results = {}
        dists = {}

        def solve_one(s: ParamSurface):
            warm = last_uv.get(s.id)
            run_full = warm is None or i == 0 or i % 7 == 0
            if run_full:
                return closest(s, query, warm_uv=warm)
            cand = _interior_solve(s, query, warm[0], warm[1], "warm")
            if cand.location != "interior" or not cand.converged or cand.rank_deficient:
                return closest(s, query, warm_uv=warm)
            return ClosestResult(
                surface_id=s.id, query=tuple(query), best=cand, candidates=[cand],
                branches=[], degenerate=False, global_guaranteed=s.kind == "plane",
                search_strategy="warm-lm", warnings=[], seed_count=1,
            )

        for s in surfaces:
            res = solve_one(s)
            results[s.id] = res
            dists[s.id] = res.best.distance
            last_uv[s.id] = (res.best.u, res.best.v)
            for warning in res.warnings:
                stamped = f"[{s.id}] {warning}"
                if stamped not in warnings_all:
                    warnings_all.append(stamped)
        cur_res = results[current_sid]
        global_best = cur_res.best

        # Continuity-aware candidate selection (no basin jumps for a hair).
        best = global_best
        warm_cur = None
        if records:
            warm_cur = (records[-1].u, records[-1].v)
        if warm_cur is not None and len(cur_res.candidates) > 1:
            ur = max(smap[current_sid].u1 - smap[current_sid].u0, 1e-12)
            vr = max(smap[current_sid].v1 - smap[current_sid].v0, 1e-12)
            vper = smap[current_sid].v1 - smap[current_sid].v0 if smap[current_sid].periodic_v else 0.0

            def jump_of(c: Candidate):
                cv = _unwrap(warm_cur[1], c.v, vper)
                wv = _unwrap(warm_cur[1], warm_cur[1], vper)
                return math.hypot((c.u - warm_cur[0]) / ur, (cv - wv) / vr)

            best_jump = jump_of(global_best)
            tol_d = max(0.03, 0.15 * global_best.distance + 1e-6)
            near = [
                c for c in cur_res.candidates
                if c.distance <= global_best.distance + tol_d and jump_of(c) < best_jump - 0.02
            ]
            if near:
                cont = min(near, key=jump_of)
                if jump_of(cont) < 0.5:
                    best = cont

        rec = RoutePoint(
            index=i, t=t, query=tuple(query),
            status="attached" if interval_open else "oob",
            surface_id=current_sid,
            distances_by_surface={k: float(v) for k, v in dists.items()},
        )

        other = min(
            ((sid, d) for sid, d in dists.items() if sid != current_sid),
            key=lambda z: z[1], default=None,
        )
        if other is not None:
            osid, od = other
            if abs(od - best.distance) <= tie_tol:
                rec.tie_hold = True
                tie_holds.append({
                    "index": i, "heldSurface": current_sid, "tiedSurface": osid,
                    "distanceCurrent": best.distance, "distanceTied": od,
                    "action": "identity_held_equal_distance_no_auto_switch",
                })
            elif od + tie_tol < best.distance:
                w = (f"index {i}: {osid} closer but switch not authorized; "
                     f"keeping {current_sid}")
                rec.warnings.append(w)
                warnings_all.append(w)

        chosen_cand = best
        if cur_res.degenerate and cur_res.branches:
            sel = select_singular_branch(
                smap[current_sid], cur_res,
                travel_dir_xyz=travel, prev_uv=warm_cur,
            )
            rec.branch_selection = {
                "ambiguous": sel["ambiguous"],
                "candidateCount": len(sel["scores"]),
                "chosenV": sel["chosen"]["v"],
                "scores": [
                    {"v": sc["branch"]["v"], "continuityCost": sc["continuityCost"],
                     "distortionCost": sc["distortionCost"], "total": sc["total"]}
                    for sc in sel["scores"]
                ],
            }
            ch = sel["chosen"]
            chosen_cand = Candidate(
                u=ch["u"], v=ch["v"], xyz=tuple(ch["xyz"]),
                distance=ch["distance"], ortho=ch["ortho"], ortho_rel=ch["ortho_rel"],
                location=ch.get("location", "singularity"),
                oob_reasons=ch.get("oob_reasons", []) or [],
                normal_degenerate=True, rank_deficient=True, seed_kind="selected-branch",
            )
            rec.degenerate = True

        u_c, v_c = chosen_cand.u, chosen_cand.v
        rec.u, rec.v = u_c, v_c
        rec.xyz = tuple(chosen_cand.xyz)
        rec.distance = chosen_cand.distance
        rec.ortho_rel = chosen_cand.ortho_rel
        rec.location = chosen_cand.location
        rec.oob_reasons = list(chosen_cand.oob_reasons)

        surf = smap[current_sid]
        prev_un = last_unwrapped.get(current_sid)
        if prev_un is not None:
            uu = _unwrap(prev_un[0], u_c, surf.u1 - surf.u0 if surf.periodic_u else 0.0)
            vv = _unwrap(prev_un[1], v_c, surf.v1 - surf.v0 if surf.periodic_v else 0.0)
        else:
            uu, vv = u_c, v_c
        rec.u_unwrapped, rec.v_unwrapped = uu, vv
        last_uv[current_sid] = (u_c, v_c)
        last_unwrapped[current_sid] = (uu, vv)

        if interval_open and prev_attached_xyz is not None:
            arc_global += vm.dist(prev_attached_xyz, rec.xyz)
        if interval_open:
            prev_attached_xyz = rec.xyz

        if not interval_open:
            rec.status = "oob"
            if not rec.oob_reasons:
                rec.oob_reasons = ["CONTINUATION_BEYOND_CLOSED_INTERVAL"]
            oob_records.append(rec)
            records.append(rec)
            continue

        def build_interval(end_i: int, end_u: float, end_v: float, reason, sw_info):
            start_rec = records[interval_start] if interval_start < len(records) else rec
            return Interval(
                surface_id=current_sid,
                index0=interval_start, index1=end_i,
                t0=start_rec.t, t1=t,
                arc0=interval_arc0, arc1=arc_global,
                uv_start=(start_rec.u if start_rec.u is not None else end_u,
                          start_rec.v if start_rec.v is not None else end_v),
                uv_end=(end_u, end_v),
                xyz_start=None if start_rec.xyz is None else tuple(start_rec.xyz),
                xyz_end=tuple(surf.point(end_u, end_v)),
                end_reason=reason, switch_info=sw_info,
            )

        def try_transition(reason_context):
            nonlocal current_sid, interval_start, interval_arc0, switched, prev_attached_xyz
            if mode != "transition" or switched or not transition:
                return False
            auth = authorize_switch(transition, current_sid, rec.xyz, transition_band)
            if auth is None:
                return False
            tp: TransitionPoint = auth["target"]
            anchor_xyz = tuple(rec.xyz) if rec.xyz is not None else tuple(query)
            b_res = closest(smap[tp.surface_id], anchor_xyz, warm_uv=(tp.u, tp.v))
            if b_res.best.distance > switch_max_distance:
                return False
            old_sid = current_sid
            sw = {
                "index": i,
                "fromSurface": old_sid,
                "toSurface": tp.surface_id,
                "reason": "USER_TRANSITION_LINE",
                "context": reason_context,
                "previousOobReason": reason_context if reason_context not in (None, "LINE_CROSSING") else None,
                "matchDistance": auth["match_distance"],
                "targetUv": [b_res.best.u, b_res.best.v],
                "targetDistance": b_res.best.distance,
            }
            safe_switch_info = {
                "fromIndex": auth["from_index"],
                "toIndex": auth["to_index"],
                "matchDistance": auth["match_distance"],
                "target": auth["target"].to_dict(),
            }
            intervals.append(build_interval(i, u_c, v_c, reason_context or "TRANSITION_LINE", safe_switch_info))
            switches.append(sw)
            nb = b_res.best
            current_sid = tp.surface_id
            rec.surface_id = current_sid
            rec.u, rec.v = nb.u, nb.v
            rec.xyz = tuple(nb.xyz)
            rec.distance = nb.distance
            rec.ortho_rel = nb.ortho_rel
            rec.location = nb.location
            rec.oob_reasons = list(nb.oob_reasons)
            rec.switch = sw
            last_uv[current_sid] = (nb.u, nb.v)
            last_unwrapped.pop(current_sid, None)
            interval_arc0 = arc_global
            interval_start = i
            switched = True
            prev_attached_xyz = rec.xyz
            return True

        # 1) Voluntary switch while crossing the user line (interior route).
        if not chosen_cand.oob_reasons and try_transition("LINE_CROSSING"):
            records.append(rec)
            continue

        # 2) Verified domain exit: authorized line next to it, or close up.
        if chosen_cand.oob_reasons:
            reason = chosen_cand.oob_reason
            if try_transition(reason):
                records.append(rec)
                continue
            rec.terminal = True
            intervals.append(build_interval(i, u_c, v_c, reason, None))
            interval_open = False
            oob_records.append(rec)
            w = (f"index {i}: {current_sid} domain exit {reason}; "
                 + ("no matching transition line -> clipped" if mode == "transition"
                    else "continuous-only edit, no switch allowed -> clipped"))
            rec.warnings.append(w)
            warnings_all.append(w)

        records.append(rec)

    if interval_open and interval_start < n:
        first = records[interval_start]
        last = records[-1]
        intervals.append(Interval(
            surface_id=current_sid,
            index0=interval_start, index1=n - 1,
            t0=first.t, t1=last.t,
            arc0=interval_arc0, arc1=arc_global,
            uv_start=(first.u, first.v), uv_end=(last.u, last.v),
            xyz_start=None if first.xyz is None else tuple(first.xyz),
            xyz_end=None if last.xyz is None else tuple(last.xyz),
            end_reason=None,
        ))

    return ProjectionOutcome(
        mode=mode, points=records, intervals=intervals,
        oob_records=oob_records, switches=switches, warnings=warnings_all,
        tie_holds=tie_holds, surface_order=order,
    )
