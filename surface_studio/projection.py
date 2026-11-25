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
from .branches import (
    DEFAULT_STRATEGY,
    STRATEGIES,
    BranchTrack,
    TrackFrame,
    track_surface,
)
from .solver import Candidate, ClosestResult, _interior_solve, closest


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
    branch: Optional[dict] = None

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
            "branch": self.branch,
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
    tracks: dict = field(default_factory=dict)
    strategy: str = DEFAULT_STRATEGY
    branch_summary: dict = field(default_factory=dict)

    def to_dict(self, compact: bool = False):
        if compact:
            # Sheet exports do not need the full candidate set per sample or the
            # per-frame track duplicate; the audit lives in report.json and the
            # live API.  Keep the chosen branch and an abandoned count.
            def _compacted(point):
                d = point.to_dict()
                b = d.get("branch")
                if b:
                    kept = {
                        k: b.get(k) for k in (
                            "surfaceId", "strategy", "candidateCount", "status",
                            "reason", "uUnwrapped", "vUnwrapped", "jumpParam",
                            "jumpXyz", "continuityOk", "anchored", "locked",
                            "recovery", "distanceCost",
                        )
                    }
                    kept["abandonedCount"] = len(b.get("abandoned") or [])
                    d["branch"] = kept
                return d
            return {
                "mode": self.mode,
                "points": [_compacted(p) for p in self.points],
                "intervals": [i.to_dict() for i in self.intervals],
                "oobRecords": [_compacted(o) for o in self.oob_records],
                "switches": self.switches,
                "warnings": self.warnings,
                "tieHolds": self.tie_holds,
                "surfaceOrder": self.surface_order,
                "strategy": self.strategy,
                "branchSummary": self.branch_summary,
            }
        return {
            "mode": self.mode,
            "points": [p.to_dict() for p in self.points],
            "intervals": [i.to_dict() for i in self.intervals],
            "oobRecords": [o.to_dict() for o in self.oob_records],
            "switches": self.switches,
            "warnings": self.warnings,
            "tieHolds": self.tie_holds,
            "surfaceOrder": self.surface_order,
            "strategy": self.strategy,
            "tracks": {sid: t.to_dict() for sid, t in self.tracks.items()},
            "branchSummary": self.branch_summary,
        }


def _unwrap(prev: Optional[float], raw: float, period: float):
    if prev is None or period <= 0:
        return raw
    k = round((prev - raw) / period)
    return raw + k * period


def _frame_candidate(frame: TrackFrame) -> Candidate:
    c = frame.candidate
    return Candidate(
        u=c.u, v=c.v, xyz=tuple(c.xyz), distance=c.distance,
        ortho=c.ortho_rel, ortho_rel=c.ortho_rel, location=c.location,
        oob_reasons=list(c.oob_reasons), converged=c.converged,
        rank_deficient=c.rank_deficient, normal_degenerate=c.normal_degenerate,
        seed_kind=c.seed_kind,
    )


def _branch_record(frame: TrackFrame, track: BranchTrack) -> dict:
    """Compact, auditable summary of the branch decision for one sample."""
    return {
        "surfaceId": track.surface_id,
        "strategy": track.strategy,
        "candidateCount": len(frame.candidates),
        "status": frame.status,
        "reason": frame.reason,
        "uUnwrapped": frame.u_unwrapped,
        "vUnwrapped": frame.v_unwrapped,
        "jumpParam": frame.jump_param,
        "jumpXyz": frame.jump_xyz,
        "continuityOk": frame.continuity_ok,
        "anchored": frame.anchor is not None,
        "locked": frame.locked,
        "recovery": frame.recovery,
        "distanceCost": frame.candidate.distance,
        "abandoned": frame.alternatives,
    }


def _track_surfaces(
    surfaces: Sequence[ParamSurface],
    design_points: Sequence,
    strategy: str,
    anchors: Optional[Sequence[dict]],
) -> dict[str, BranchTrack]:
    """Solve and track every surface once, before any surface-identity walk."""
    per_surface_anchors = _anchors_by_surface(anchors)
    return {
        s.id: track_surface(
            s, design_points, strategy=strategy,
            anchors=per_surface_anchors.get(s.id),
        )
        for s in surfaces
    }


def _anchors_by_surface(anchors: Optional[Sequence[dict]]) -> dict[str, list]:
    out: dict[str, list] = {}
    if not anchors:
        return out
    for a in anchors:
        sid = a.get("surfaceId")
        if sid is None:
            continue
        out.setdefault(sid, []).append(a)
    return out


def project_curve(
    surfaces: Sequence[ParamSurface],
    design_points: Sequence[vm.Vec3],
    mode: str = "spout_only",
    initial_surface_id: Optional[str] = None,
    transition: Optional[Sequence[TransitionPoint]] = None,
    tie_tol: float = 2e-3,
    switch_max_distance: float = 0.8,
    transition_band: float = 0.04,
    branch_strategy: str = DEFAULT_STRATEGY,
    branch_anchors: Optional[Sequence[dict]] = None,
) -> ProjectionOutcome:
    """Project the design curve with per-surface branch tracking.

    Each surface is solved and tracked independently first (see
    :mod:`surface_studio.branches`); the surface-identity walk below then uses
    the tracked route of the current surface, so the reported point is already
    a continuous branch choice rather than the nearest single point.
    """
    if branch_strategy not in STRATEGIES + ("auto",):
        raise ValueError(f"unknown branch strategy {branch_strategy!r}")
    smap = {s.id: s for s in surfaces}
    order = [s.id for s in surfaces]
    n = len(design_points)
    if n == 0:
        return ProjectionOutcome(
            mode=mode, points=[], intervals=[], oob_records=[], switches=[],
            warnings=[], tie_holds=[], surface_order=order,
            strategy=branch_strategy,
        )

    tracks = _track_surfaces(surfaces, design_points, branch_strategy, branch_anchors)

    first_frame = {sid: t.frames[0] for sid, t in tracks.items()}
    if initial_surface_id and initial_surface_id in smap:
        current_sid = initial_surface_id
    else:
        current_sid = min(
            first_frame, key=lambda sid: first_frame[sid].candidate.distance
        )

    records: list[RoutePoint] = []
    intervals: list[Interval] = []
    switches: list[dict] = []
    tie_holds: list[dict] = []
    warnings_all: list[str] = []
    oob_records: list[RoutePoint] = []

    for sid, track in tracks.items():
        for w in track.warnings:
            if w not in warnings_all:
                warnings_all.append(w)

    interval_start = 0
    interval_open = True
    switched = False
    arc_global = 0.0
    interval_arc0 = 0.0
    prev_attached_xyz = None

    for i, query in enumerate(design_points):
        t = i / max(1, n - 1)
        cur_track = tracks[current_sid]
        frame = cur_track.frames[i]
        chosen_cand = _frame_candidate(frame)

        dists = {
            sid: tracks[sid].frames[i].candidate.distance for sid in order
        }

        rec = RoutePoint(
            index=i, t=t, query=tuple(query),
            status="attached" if interval_open else "oob",
            surface_id=current_sid,
            distances_by_surface={k: float(v) for k, v in dists.items()},
            branch=_branch_record(frame, cur_track),
        )

        other = min(
            ((sid, d) for sid, d in dists.items() if sid != current_sid),
            key=lambda z: z[1], default=None,
        )
        if other is not None:
            osid, od = other
            if abs(od - chosen_cand.distance) <= tie_tol:
                rec.tie_hold = True
                tie_holds.append({
                    "index": i, "heldSurface": current_sid, "tiedSurface": osid,
                    "distanceCurrent": chosen_cand.distance, "distanceTied": od,
                    "action": "identity_held_equal_distance_no_auto_switch",
                })
            elif od + tie_tol < chosen_cand.distance:
                w = (f"index {i}: {osid} closer but switch not authorized; "
                     f"keeping {current_sid}")
                rec.warnings.append(w)
                warnings_all.append(w)

        if frame.candidate.rank_deficient or frame.candidate.normal_degenerate:
            rec.degenerate = True
            rec.branch_selection = {
                "ambiguous": frame.ambiguous,
                "candidateCount": len(frame.candidates),
                "chosenV": frame.candidate.v,
                "strategy": cur_track.strategy,
            }

        u_c, v_c = chosen_cand.u, chosen_cand.v
        rec.u, rec.v = u_c, v_c
        rec.xyz = tuple(chosen_cand.xyz)
        rec.distance = chosen_cand.distance
        rec.ortho_rel = chosen_cand.ortho_rel
        rec.location = chosen_cand.location
        rec.oob_reasons = list(chosen_cand.oob_reasons)
        rec.u_unwrapped = frame.u_unwrapped
        rec.v_unwrapped = frame.v_unwrapped

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

        def build_interval(end_i, end_u, end_v, reason, sw_info, surf_id=None):
            sid = surf_id or current_sid
            surf = smap[sid]
            start_rec = records[interval_start] if interval_start < len(records) else rec
            return Interval(
                surface_id=sid,
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
            nonlocal current_sid, interval_start, interval_arc0, switched
            nonlocal prev_attached_xyz
            if mode != "transition" or switched or not transition:
                return False
            auth = authorize_switch(transition, current_sid, rec.xyz, transition_band)
            if auth is None:
                return False
            tp: TransitionPoint = auth["target"]
            anchor_xyz = tuple(rec.xyz) if rec.xyz is not None else tuple(query)
            new_track = tracks[tp.surface_id]
            new_surf = smap[tp.surface_id]
            # Honour the user-drawn landing UV: solve from that parameter value
            # so the switch lands on the branch the transition line specifies,
            # then keep the matching tracked frame for reporting.
            from .solver import closest as _closest
            b_res = _closest(new_surf, anchor_xyz, warm_uv=(tp.u, tp.v))
            if b_res.best.distance > switch_max_distance:
                return False
            nb = b_res.best
            b_frame = min(
                new_track.frames,
                key=lambda f: min(
                    abs(f.candidate.u - nb.u),
                    abs(f.candidate.v - nb.v),
                ),
            )
            old_sid = current_sid
            sw = {
                "index": i,
                "fromSurface": old_sid,
                "toSurface": tp.surface_id,
                "reason": "USER_TRANSITION_LINE",
                "context": reason_context,
                "previousOobReason": reason_context if reason_context not in (None, "LINE_CROSSING") else None,
                "matchDistance": auth["match_distance"],
                "targetUv": [nb.u, nb.v],
                "targetDistance": nb.distance,
                "targetBranchStatus": b_frame.status,
            }
            safe_switch_info = {
                "fromIndex": auth["from_index"],
                "toIndex": auth["to_index"],
                "matchDistance": auth["match_distance"],
                "target": auth["target"].to_dict(),
            }
            intervals.append(build_interval(
                i, u_c, v_c, reason_context or "TRANSITION_LINE", safe_switch_info
            ))
            switches.append(sw)
            current_sid = tp.surface_id
            rec.surface_id = current_sid
            rec.u, rec.v = nb.u, nb.v
            rec.xyz = tuple(nb.xyz)
            rec.distance = nb.distance
            rec.ortho_rel = nb.ortho_rel
            rec.location = nb.location
            rec.oob_reasons = list(nb.oob_reasons)
            rec.u_unwrapped = b_frame.u_unwrapped
            rec.v_unwrapped = b_frame.v_unwrapped
            rec.branch = _branch_record(b_frame, new_track)
            rec.switch = sw
            interval_arc0 = arc_global
            interval_start = i
            switched = True
            prev_attached_xyz = rec.xyz
            return True

        # 1) Voluntary switch while crossing the user line (interior route).
        if not chosen_cand.oob_reasons and try_transition("LINE_CROSSING"):
            records.append(rec)
            continue

        # 2) Verified domain exit or a physical/parameter break from tracking.
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
        elif frame.status == "break":
            # The tracked route could not stay continuous. Close the interval
            # at the previous sample (the break sample itself is not attached)
            # and report the break instead of an in-air connector.
            reason = frame.reason
            rec.status = "oob"
            rec.oob_reasons = ["ROUTE_" + reason]
            rec.terminal = True
            end_i = max(interval_start, i - 1)
            prev_rec = records[end_i]
            intervals.append(Interval(
                surface_id=current_sid,
                index0=interval_start, index1=end_i,
                t0=records[interval_start].t, t1=prev_rec.t,
                arc0=interval_arc0, arc1=arc_global,
                uv_start=(records[interval_start].u, records[interval_start].v),
                uv_end=(prev_rec.u, prev_rec.v),
                xyz_start=None if records[interval_start].xyz is None else tuple(records[interval_start].xyz),
                xyz_end=None if prev_rec.xyz is None else tuple(prev_rec.xyz),
                end_reason="ROUTE_" + reason,
            ))
            interval_open = False
            oob_records.append(rec)
            w = (f"index {i}: {current_sid} route not continuable ({reason}); "
                 "interval closed, no connector added")
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

    summary = {
        "strategy": branch_strategy,
        "surfaces": {
            sid: {
                "chosenStrategy": t.strategy,
                "strategyComparison": t.comparison,
                "stats": t.stats,
                "breaks": t.breaks,
                "attachmentError": t.real_attachment_error,
            }
            for sid, t in tracks.items()
        },
        "totalBreaks": sum(len(t.breaks) for t in tracks.values()),
        "ambiguousSamples": {sid: t.stats["ambiguousSamples"] for sid, t in tracks.items()},
        "branchSwitches": {sid: t.stats["branchSwitches"] for sid, t in tracks.items()},
        "recoveries": {sid: t.stats["recoveries"] for sid, t in tracks.items()},
    }

    return ProjectionOutcome(
        mode=mode, points=records, intervals=intervals,
        oob_records=oob_records, switches=switches, warnings=warnings_all,
        tie_holds=tie_holds, surface_order=order,
        tracks=tracks, strategy=branch_strategy, branch_summary=summary,
    )
