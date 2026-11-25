"""Pattern/decal construction, surface mapping, clipping and distortion.

The decal is a real vector stroke pattern. Every dense stroke sample is
mapped onto a continuous surface route frame and then re-projected with the
analytic inverse solver. Samples outside a continuous interval (or rejected
by the solver outside a parameter boundary) are clipped with the surface
identity and the solver-verified reason handed to the clipper.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from . import vecmath as vm
from .projection import ProjectionOutcome
from .solver import _interior_solve, closest
from .surfaces import ParamSurface


# ---------------------------------------------------------------------------
# Tiny stroke font (unit cells, y up)
# ---------------------------------------------------------------------------
def _ellipse(cx=0.5, cy=0.5, rx=0.42, ry=0.5, n=20):
    return [
        (cx + rx * math.cos(2 * math.pi * k / n), cy + ry * math.sin(2 * math.pi * k / n))
        for k in range(n + 1)
    ]


GLYPHS = {
    "T": [[(0.02, 1.0), (0.98, 1.0)], [(0.5, 1.0), (0.5, 0.0)]],
    "E": [
        [(0.98, 1.0), (0.02, 1.0), (0.02, 0.0), (0.98, 0.0)],
        [(0.02, 0.52), (0.74, 0.52)],
    ],
    "A": [
        [(0.02, 0.0), (0.5, 1.0), (0.98, 0.0)],
        [(0.26, 0.42), (0.74, 0.42)],
    ],
    "P": [
        [(0.02, 0.0), (0.02, 1.0), (0.7, 1.0), (0.96, 0.82), (0.96, 0.62),
         (0.7, 0.5), (0.02, 0.5)],
    ],
    "O": [_ellipse()],
}

ADVANCE = 1.15


def build_pattern(text: str = "TEAPOT") -> dict:
    letters = []
    strokes = []
    for k, ch in enumerate(text):
        cx = k * ADVANCE + 0.5
        for gi, poly in enumerate(GLYPHS.get(ch, [])):
            moved = [(x + k * ADVANCE, y) for x, y in poly]
            strokes.append({"id": f"{ch}{k}-{gi}", "char": ch, "letterIndex": k, "points": moved})
        letters.append({"char": ch, "index": k, "cx": cx, "cy": 0.5})
    width = (len(text) - 1) * ADVANCE + 1.0
    # Baseline + two registration ticks help judge seam read-across.
    guides = [
        {"id": "baseline", "char": None, "points": [(-0.15, -0.12), (width + 0.15, -0.12)]},
        {"id": "topline", "char": None, "points": [(-0.15, 1.12), (width + 0.15, 1.12)]},
    ]
    return {"text": text, "width": width, "height": 1.0, "letters": letters,
            "strokes": strokes + guides, "advance": ADVANCE}


# ---------------------------------------------------------------------------
# Route frames
# ---------------------------------------------------------------------------
@dataclass
class FrameSample:
    arc: float
    sid: str
    uv: tuple
    xyz: tuple
    tangent: tuple
    normal: tuple
    side: tuple


def _surface_normal_or_fallback(surf, uv, prev_n):
    n = surf.normal(uv[0], uv[1])
    if n is not None:
        return n
    for d in (1e-3, -1e-3, 1e-2, -1e-2):
        for du, dv in ((d, 0.0), (0.0, d)):
            u = vm.clamp(uv[0] + du, surf.u0, surf.u1)
            v = uv[1] + dv
            nn = surf.normal(u, v)
            if nn is not None:
                return nn
    return prev_n or (0.0, 1.0, 0.0)


def build_interval_frames(outcome: ProjectionOutcome, surfaces: dict):
    points = outcome.points
    frames: list[list[FrameSample]] = []
    prev_last_side = None

    for k, iv in enumerate(outcome.intervals):
        surf = surfaces[iv.surface_id]
        uvs = []
        for j in range(iv.index0, iv.index1 + 1):
            r = points[j]
            if r.surface_id == iv.surface_id and r.u is not None:
                uvs.append((r.u, r.v))
        if iv.switch_info is not None:
            # Shared endpoint record was overwritten by the new face.
            end_diff = math.hypot(uvs[-1][0] - iv.uv_end[0], uvs[-1][1] - iv.uv_end[1]) if uvs else 1.0
            if not uvs or end_diff > 1e-8:
                uvs.append(tuple(iv.uv_end))
        if len(uvs) < 2:
            uvs = [tuple(iv.uv_start), tuple(iv.uv_end)]

        xyz = [tuple(surf.point(u, v)) for u, v in uvs]
        cum = [0.0]
        for a, b in zip(xyz, xyz[1:]):
            cum.append(cum[-1] + vm.dist(a, b))
        total = cum[-1]
        span = max(iv.arc1 - iv.arc0, 1e-9)
        arcs = [iv.arc0 + (c / total) * span for c in cum] if total > 1e-10 else [
            iv.arc0 + span * j / max(1, len(uvs) - 1) for j in range(len(uvs))
        ]

        fs: list[FrameSample] = []
        prev_n = None
        sides_raw = []
        for j, uv in enumerate(uvs):
            if j == 0:
                t = vm.sub(xyz[1], xyz[0]) if len(xyz) > 1 else (1, 0, 0)
            elif j == len(xyz) - 1:
                t = vm.sub(xyz[-1], xyz[-2])
            else:
                t = vm.sub(xyz[j + 1], xyz[j - 1])
            tangent = vm.normalize(t) or (1.0, 0.0, 0.0)
            n = _surface_normal_or_fallback(surf, uv, prev_n)
            prev_n = n
            side = vm.normalize(vm.cross(n, tangent))
            if side is None:
                side = (0.0, 0.0, 1.0)
            sides_raw.append(side)
            fs.append(FrameSample(arcs[j], iv.surface_id, uv, xyz[j], tangent, n, side))

        # Parallel transport the side vector along the interval.
        if k > 0 and prev_last_side is not None:
            s = prev_last_side
            n0 = fs[0].normal
            s = vm.sub(s, vm.mul(n0, vm.dot(s, n0)))
            s = vm.sub(s, vm.mul(fs[0].tangent, vm.dot(s, fs[0].tangent)))
            s = vm.normalize(s) or fs[0].side
            fs[0].side = s
        for j in range(1, len(fs)):
            s = fs[j - 1].side
            n = fs[j].normal
            s = vm.sub(s, vm.mul(n, vm.dot(s, n)))
            s = vm.sub(s, vm.mul(fs[j].tangent, vm.dot(s, fs[j].tangent)))
            ns = vm.normalize(s)
            if ns is not None:
                fs[j].side = ns
        prev_last_side = fs[-1].side
        frames.append(fs)
    return frames


def frame_at_arc(interval_frames, intervals, s: float):
    if not intervals:
        return None
    if s < intervals[0].arc0 - 1e-9 or s > intervals[-1].arc1 + 1e-9:
        return None
    for k, iv in enumerate(intervals):
        if s <= iv.arc1 + 1e-9 or k == len(intervals) - 1:
            fs = interval_frames[k]
            if s <= fs[0].arc:
                a, b = fs[0], fs[0]
                t = 0.0
            elif s >= fs[-1].arc:
                a, b = fs[-1], fs[-1]
                t = 0.0
            else:
                lo, hi = 0, len(fs) - 1
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if fs[mid].arc <= s:
                        lo = mid
                    else:
                        hi = mid
                a, b = fs[lo], fs[hi]
                t = 0.0 if b.arc == a.arc else (s - a.arc) / (b.arc - a.arc)
            xyz = tuple(a.xyz[j] * (1 - t) + b.xyz[j] * t for j in range(3))
            uv = (a.uv[0] * (1 - t) + b.uv[0] * t, a.uv[1] * (1 - t) + b.uv[1] * t)
            side = vm.normalize(tuple(a.side[j] * (1 - t) + b.side[j] * t for j in range(3))) or a.side
            tang = vm.normalize(tuple(a.tangent[j] * (1 - t) + b.tangent[j] * t for j in range(3))) or a.tangent
            return {"sid": iv.surface_id, "uv": uv, "xyz": xyz, "side": side, "tangent": tang}
    return None


# ---------------------------------------------------------------------------
# Pattern placement and mapping
# ---------------------------------------------------------------------------
@dataclass
class Attachment:
    offset: float = 1.70
    lateral: float = 0.0
    scale: float = 0.24
    angle_deg: float = 0.0

    @property
    def angle(self):
        return math.radians(self.angle_deg)

    def to_dict(self):
        return {
            "offset": self.offset,
            "lateral": self.lateral,
            "scale": self.scale,
            "angleDeg": self.angle_deg,
        }


def pattern_to_st(pt, pattern: dict, att: Attachment):
    x, y = pt
    x0 = x - pattern["width"] / 2.0
    y0 = y - 0.5
    ca, sa = math.cos(att.angle), math.sin(att.angle)
    xr = x0 * ca - y0 * sa
    yr = x0 * sa + y0 * ca
    return att.offset + xr * att.scale, att.lateral + yr * att.scale


def _project_world(surf: ParamSurface, q, warm_uv):
    cand = _interior_solve(surf, q, warm_uv[0], warm_uv[1], "warm-map")
    if cand.location != "interior" or not cand.converged:
        res = closest(surf, q, warm_uv=warm_uv, grid_n=10, max_seeds=4)
        cand = res.best
    return cand


def map_st_world(surfaces: dict, frames, intervals, s: float, lateral: float):
    fr = frame_at_arc(frames, intervals, s)
    if fr is None:
        return None
    q = vm.add(fr["xyz"], vm.mul(fr["side"], lateral))
    cand = _project_world(surfaces[fr["sid"]], q, fr["uv"])
    return {
        "sid": fr["sid"],
        "uv": (cand.u, cand.v),
        "xyz": tuple(cand.xyz),
        "distance": cand.distance,
        "oob": cand.oob_reason,
        "frame": fr,
    }


def _clip_reason_for_s(intervals, s: float):
    if s < intervals[0].arc0:
        return "PATTERN_BEYOND_ROUTE_START", intervals[0].surface_id
    for iv in intervals:
        if s <= iv.arc1:
            return None, iv.surface_id
    last = intervals[-1]
    if last.end_reason:
        return "DECAL_CLIPPED_AFTER_INTERVAL_END:" + last.end_reason, last.surface_id
    return "DECAL_CLIPPED_BEYOND_DESIGN_CURVE_END", last.surface_id


def map_decal(
    pattern: dict,
    outcome: ProjectionOutcome,
    surfaces: list,
    att: Attachment,
    densify_step: float = 0.07,
):
    smap = {s.id: s for s in surfaces}
    intervals = outcome.intervals
    frames = build_interval_frames(outcome, smap)

    runs = []
    clip_records = []
    clip_counts: dict = {}

    def map_one_pattern_point(x, y, char):
        s, lat = pattern_to_st((x, y), pattern, att)
        reason, owner_sid = _clip_reason_for_s(intervals, s)
        if reason:
            return None, {"x": x, "y": y, "s": s, "reason": reason, "surfaceId": owner_sid, "char": char}
        mw = map_st_world(smap, frames, intervals, s, lat)
        if mw is None:
            return None, {"x": x, "y": y, "s": s, "reason": "FRAME_UNAVAILABLE", "surfaceId": owner_sid, "char": char}
        if mw["oob"]:
            return None, {"x": x, "y": y, "s": s, "reason": "MAPPED_POINT_" + mw["oob"],
                          "surfaceId": mw["sid"], "char": char}
        return {"x": x, "y": y, "s": s, "lateral": lat, **mw}, None

    for stroke in pattern["strokes"]:
        char = stroke["char"] or "guide"
        dense = []
        poly = stroke["points"]
        for a, b in zip(poly, poly[1:]):
            seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
            m = max(1, int(math.ceil(seg_len / densify_step)))
            for k in range(m):
                t = k / m
                dense.append((a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t))
        dense.append(poly[-1])

        cur_run = None
        for x, y in dense:
            mp, clip = map_one_pattern_point(x, y, stroke["char"])
            if clip:
                key = clip["reason"].split(":")[0]
                clip_counts[key] = clip_counts.get(key, 0) + 1
                clip_records.append(clip)
                if cur_run is not None:
                    runs.append(cur_run)
                    cur_run = None
                continue
            if mp is None:
                continue
            if cur_run is None or cur_run["surfaceId"] != mp["sid"]:
                if cur_run is not None:
                    runs.append(cur_run)
                cur_run = {
                    "strokeId": stroke["id"],
                    "char": char,
                    "surfaceId": mp["sid"],
                    "points2d": [],
                    "s": [],
                    "lateral": [],
                    "xyz": [],
                    "uv": [],
                }
            cur_run["points2d"].append([x, y])
            cur_run["s"].append(mp["s"])
            cur_run["lateral"].append(mp["lateral"])
            cur_run["xyz"].append(list(mp["xyz"]))
            cur_run["uv"].append([mp["uv"][0], mp["uv"][1]])
        if cur_run is not None:
            runs.append(cur_run)

    # Per-letter landing face + local mapping distortion.
    letter_reports = []
    for L in pattern["letters"]:
        mp, clip = map_one_pattern_point(L["cx"], L["cy"], L["char"])
        rep = {"char": L["char"], "index": L["index"], "cx": L["cx"]}
        if clip:
            rep.update({"status": "clipped", "surfaceId": clip["surfaceId"], "reason": clip["reason"]})
        else:
            eps = 0.06

            def world_at(dx, dy):
                s, lat = pattern_to_st((L["cx"] + dx, L["cy"] + dy), pattern, att)
                mw = map_st_world(smap, frames, intervals, s, lat)
                return mw["xyz"] if mw else mp["xyz"]

            wx = vm.sub(world_at(eps, 0), world_at(-eps, 0))
            wy = vm.sub(world_at(0, eps), world_at(0, -eps))
            dxdx = vm.mul(wx, 1.0 / (2 * eps))
            dxdy = vm.mul(wy, 1.0 / (2 * eps))
            cross = vm.cross(dxdx, dxdy)
            area_world = vm.norm(cross)
            expected = att.scale * att.scale
            area_ratio = area_world / max(expected, 1e-12)
            lx, ly = vm.norm(dxdx), vm.norm(dxdy)
            anis = max(lx, ly) / max(min(lx, ly), 1e-12)
            rep.update({
                "status": "attached",
                "surfaceId": mp["sid"],
                "xyz": list(mp["xyz"]),
                "uv": list(mp["uv"]),
                "areaRatio": area_ratio,
                "anisotropy": anis,
            })
        letter_reports.append(rep)

    seam = _seam_report(outcome, smap, frames, intervals, att)
    distortion = _decal_distortion(letter_reports)

    face_letters: dict = {}
    clipped_letters: list = []
    for r in letter_reports:
        if r.get("status") == "attached":
            face_letters.setdefault(r["surfaceId"], []).append(r["char"])
        else:
            clipped_letters.append({
                "char": r["char"],
                "surfaceId": r.get("surfaceId"),
                "reason": r.get("reason"),
            })

    return {
        "attachment": att.to_dict(),
        "runs": runs,
        "clipRecords": clip_records,
        "clipCounts": clip_counts,
        "letters": letter_reports,
        "faceLetters": face_letters,
        "clippedLetters": clipped_letters,
        "seam": seam,
        "distortion": distortion,
        "frameCount": [len(f) for f in frames],
    }


def _seam_report(outcome, smap, frames, intervals, att: Attachment):
    if len(intervals) < 2:
        return {"hasSeam": False}
    iv0, iv1 = intervals[0], intervals[1]
    a = frames[0][-1]
    b = frames[1][0]
    gap = vm.dist(a.xyz, b.xyz)
    td = vm.normalize(vm.sub(b.tangent, (0, 0, 0)))
    cos_t = max(-1.0, min(1.0, vm.dot(a.tangent, b.tangent)))
    tangent_angle = math.degrees(math.acos(cos_t))
    cos_s = max(-1.0, min(1.0, vm.dot(a.side, b.side)))
    side_angle = math.degrees(math.acos(cos_s))
    # Baseline registration points 4 cm on each pattern side of the seam.
    eps_arc = 0.05
    pa = map_st_world(smap, frames, intervals, iv1.arc0 - eps_arc, att.lateral - 0.12 * att.scale)
    pb = map_st_world(smap, frames, intervals, iv1.arc0 + eps_arc, att.lateral - 0.12 * att.scale)
    reg_gap = vm.dist(pa["xyz"], pb["xyz"]) if pa and pb else None
    readable = gap <= 0.14 and tangent_angle <= 35.0 and side_angle <= 35.0
    return {
        "hasSeam": True,
        "fromSurface": iv0.surface_id,
        "toSurface": iv1.surface_id,
        "seamArc": iv1.arc0,
        "physicalGap": gap,
        "tangentAngleDeg": tangent_angle,
        "sideAngleDeg": side_angle,
        "registrationGap": reg_gap,
        "readable": readable,
        "thresholds": {"gap": 0.14, "tangentAngleDeg": 35, "sideAngleDeg": 35},
        "switch": outcome.switches[0] if outcome.switches else None,
    }


def _decal_distortion(letter_reports):
    areas = [r["areaRatio"] for r in letter_reports if r.get("areaRatio") is not None]
    anis = [r["anisotropy"] for r in letter_reports if r.get("anisotropy") is not None]
    if not areas:
        return {"available": False}
    return {
        "available": True,
        "areaRatioMin": min(areas),
        "areaRatioMax": max(areas),
        "areaRatioMean": sum(areas) / len(areas),
        "anisotropyMax": max(anis),
        "perLetter": [
            {"char": r["char"], "surfaceId": r.get("surfaceId"),
             "areaRatio": r.get("areaRatio"), "anisotropy": r.get("anisotropy")}
            for r in letter_reports
        ],
    }
