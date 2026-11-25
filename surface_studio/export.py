"""Export real attachment sheets, solved curves and machine-readable reports.

The SVG is the flattened pattern sheet (original strokes, face-colored
attached runs, clipped samples and seam registration mark). The JSON/CSV
files carry the actual surface identity, UV and XYZ curves produced by the
analytic projection pipeline.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Optional

from .decal import Attachment, build_pattern, map_decal
from .projection import project_curve
from .scenario import build_teapot, design_curve_samples, preset_transition

DEFAULT_EXPORT_DIR = Path(__file__).resolve().parent.parent / "exports"

SURFACE_COLORS = {
    "spout": "#2563eb",
    "body": "#ea580c",
    "plane": "#16a34a",
    "curved": "#7c3aed",
    "foldback": "#0891b2",
    "sphere": "#db2777",
}
CLIP_COLOR = "#dc2626"
GUIDE_COLOR = "#94a3b8"


def surface_color(sid: str) -> str:
    return SURFACE_COLORS.get(sid, "#475569")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_decal_csv(path: Path, mode: str, yaw_rad: float, decal: dict) -> None:
    fields = [
        "mode", "yawRad", "strokeId", "char", "surfaceId", "runIndex",
        "point2dX", "point2dY", "s", "lateral", "u", "v", "x", "y", "z",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for ri, run in enumerate(decal["runs"]):
            for k, (p2, uv, xyz) in enumerate(zip(run["points2d"], run["uv"], run["xyz"])):
                writer.writerow({
                    "mode": mode,
                    "yawRad": yaw_rad,
                    "strokeId": run["strokeId"],
                    "char": run["char"],
                    "surfaceId": run["surfaceId"],
                    "runIndex": ri,
                    "point2dX": p2[0],
                    "point2dY": p2[1],
                    "s": run.get("s", [None] * len(run["points2d"]))[k],
                    "lateral": run.get("lateral", [None] * len(run["points2d"]))[k],
                    "u": uv[0],
                    "v": uv[1],
                    "x": xyz[0],
                    "y": xyz[1],
                    "z": xyz[2],
                })


def _write_route_csv(path: Path, bundle: dict, only_edit: Optional[str] = None) -> None:
    fields = [
        "edit", "index", "status", "surfaceId", "u", "v",
        "uUnwrapped", "vUnwrapped", "x", "y", "z", "distance",
        "orthoRel", "location", "oobReason", "tieHold",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for edit in ("A", "B"):
            if only_edit is not None and edit != only_edit:
                continue
            outcome = bundle[edit]["outcome"]
            for p in outcome.points:
                writer.writerow({
                    "edit": edit,
                    "index": p.index,
                    "status": p.status,
                    "surfaceId": p.surface_id,
                    "u": p.u,
                    "v": p.v,
                    "uUnwrapped": p.u_unwrapped,
                    "vUnwrapped": p.v_unwrapped,
                    "x": None if p.xyz is None else p.xyz[0],
                    "y": None if p.xyz is None else p.xyz[1],
                    "z": None if p.xyz is None else p.xyz[2],
                    "distance": p.distance,
                    "orthoRel": p.ortho_rel,
                    "location": p.location,
                    "oobReason": "|".join(p.oob_reasons),
                    "tieHold": p.tie_hold,
                })


def _xml(text) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _polyline(points, xmap, ymap, **attrs) -> str:
    coords = " ".join(f"{xmap(x):.2f},{ymap(y):.2f}" for x, y in points)
    attr = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<polyline points="{coords}" {attr}/>'


def write_svg_sheet(path: Path, bundle: dict, edit: str) -> None:
    pattern = bundle["pattern"]
    decal = bundle[edit]["decal"]
    yaw_deg = math.degrees(bundle["yawRad"])
    title = bundle[edit]["title"]
    width = pattern["width"]

    vx0, vx1 = -0.45, width + 0.45
    vy0, vy1 = -0.34, 1.34
    plot_x, plot_y, plot_w, plot_h = 42, 76, 850, 360

    def xmap(x):
        return plot_x + (x - vx0) / (vx1 - vx0) * plot_w

    def ymap(y):
        return plot_y + plot_h - (y - vy0) / (vy1 - vy0) * plot_h

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="620" viewBox="0 0 1200 620">',
        '<rect width="1200" height="620" fill="#f8fafc"/>',
        f'<text x="42" y="38" font-family="sans-serif" font-size="24" font-weight="700" fill="#0f172a">{_xml(title)}</text>',
        f'<text x="42" y="61" font-family="sans-serif" font-size="14" fill="#475569">yaw={yaw_deg:.1f}° · real analytic surface projection · {_xml(bundle[edit]["mode"])}</text>',
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" rx="10" fill="#ffffff" stroke="#cbd5e1"/>',
    ]

    for gy in (0.0, 0.5, 1.0):
        parts.append(
            f'<line x1="{xmap(vx0):.1f}" y1="{ymap(gy):.1f}" '
            f'x2="{xmap(vx1):.1f}" y2="{ymap(gy):.1f}" stroke="#e2e8f0" stroke-width="1"/>'
        )

    for stroke in pattern["strokes"]:
        is_guide = stroke["char"] is None
        parts.append(_polyline(
            stroke["points"], xmap, ymap,
            fill="none",
            stroke=GUIDE_COLOR if is_guide else "#cbd5e1",
            **{"stroke-width": 1.4 if is_guide else 2.2,
               "stroke-dasharray": "7 5" if is_guide else "none",
               "stroke-linecap": "round", "stroke-linejoin": "round",
               "opacity": 0.95},
        ))

    for run in decal["runs"]:
        color = surface_color(run["surfaceId"])
        parts.append(_polyline(
            run["points2d"], xmap, ymap,
            fill="none", stroke=color,
            **{"stroke-width": 4.2, "stroke-linecap": "round",
               "stroke-linejoin": "round"},
        ))

    for rec in decal["clipRecords"]:
        x, y = rec["x"], rec["y"]
        parts.append(
            f'<circle cx="{xmap(x):.2f}" cy="{ymap(y):.2f}" r="2.1" '
            f'fill="{CLIP_COLOR}" opacity="0.42"/>'
        )

    seam = decal.get("seam", {})
    if seam.get("hasSeam"):
        att = decal["attachment"]
        ca, sa = math.cos(math.radians(att["angleDeg"])), math.sin(math.radians(att["angleDeg"]))
        xr = (seam["seamArc"] - att["offset"]) / max(att["scale"], 1e-9)
        line_pts = []
        for yr in (-0.82, 0.82):
            x0 = xr * ca + yr * sa
            y0 = -xr * sa + yr * ca
            line_pts.append((width / 2 + x0, y0 + 0.5))
        parts.append(_polyline(
            line_pts, xmap, ymap, fill="none", stroke="#111827",
            **{"stroke-width": 2.2, "stroke-dasharray": "10 6 2 6"},
        ))
        mx = sum(x for x, _ in line_pts) / 2
        my = sum(y for _, y in line_pts) / 2
        parts.append(
            f'<text x="{xmap(mx) + 8:.1f}" y="{ymap(my) - 8:.1f}" '
            f'font-family="sans-serif" font-size="13" font-weight="700" fill="#111827">seam</text>'
        )

    # Legend and report panel.
    px = 930
    parts.append('<rect x="918" y="76" width="244" height="430" rx="10" fill="#ffffff" stroke="#cbd5e1"/>')
    y = 105
    parts.append(f'<text x="{px}" y="{y}" font-family="sans-serif" font-size="17" font-weight="700" fill="#0f172a">Landing faces</text>')
    y += 27
    for sid, chars in decal["faceLetters"].items():
        parts.append(f'<line x1="{px}" y1="{y-5}" x2="{px+24}" y2="{y-5}" stroke="{surface_color(sid)}" stroke-width="5" stroke-linecap="round"/>')
        parts.append(f'<text x="{px+34}" y="{y}" font-family="monospace" font-size="15" fill="#0f172a">{_xml(sid)}: {_xml("".join(chars))}</text>')
        y += 25
    clipped = decal.get("clippedLetters", [])
    if clipped:
        parts.append(f'<circle cx="{px+12}" cy="{y-5}" r="5" fill="{CLIP_COLOR}"/>')
        parts.append(f'<text x="{px+34}" y="{y}" font-family="monospace" font-size="15" fill="#0f172a">clipped: {_xml("".join(c["char"] for c in clipped))}</text>')
        y += 28

    parts.append(f'<text x="{px}" y="{y}" font-family="sans-serif" font-size="17" font-weight="700" fill="#0f172a">Clipping</text>')
    y += 24
    if decal["clipCounts"]:
        for reason, count in list(decal["clipCounts"].items())[:5]:
            short = reason.replace("DECAL_CLIPPED_AFTER_INTERVAL_END:", "")
            parts.append(f'<text x="{px}" y="{y}" font-family="monospace" font-size="11.5" fill="#b91c1c">{_xml(short[:31])} ×{count}</text>')
            y += 18
    else:
        parts.append(f'<text x="{px}" y="{y}" font-family="sans-serif" font-size="13" fill="#15803d">none — all strokes attached</text>')
        y += 18

    if seam.get("hasSeam"):
        y += 12
        parts.append(f'<text x="{px}" y="{y}" font-family="sans-serif" font-size="17" font-weight="700" fill="#0f172a">Seam check</text>')
        y += 24
        rows = [
            f"gap {seam['physicalGap']:.3f}",
            f"tangent {seam['tangentAngleDeg']:.1f}°",
            f"frame {seam['sideAngleDeg']:.1f}°",
            f"register {seam['registrationGap']:.3f}" if seam.get("registrationGap") is not None else "register n/a",
            "READABLE" if seam.get("readable") else "NOT READABLE",
        ]
        for row in rows:
            color = "#15803d" if row == "READABLE" else ("#b91c1c" if row == "NOT READABLE" else "#334155")
            weight = "700" if row in ("READABLE", "NOT READABLE") else "400"
            parts.append(f'<text x="{px}" y="{y}" font-family="monospace" font-size="13.5" font-weight="{weight}" fill="{color}">{_xml(row)}</text>')
            y += 21

    dist = decal.get("distortion", {})
    if dist.get("available"):
        y += 10
        parts.append(f'<text x="{px}" y="{y}" font-family="sans-serif" font-size="17" font-weight="700" fill="#0f172a">Distortion</text>')
        y += 23
        parts.append(f'<text x="{px}" y="{y}" font-family="monospace" font-size="12.5" fill="#334155">area {dist["areaRatioMin"]:.3f}–{dist["areaRatioMax"]:.3f}</text>')
        y += 19
        parts.append(f'<text x="{px}" y="{y}" font-family="monospace" font-size="12.5" fill="#334155">anisotropy max {dist["anisotropyMax"]:.2f}</text>')

    warnings = bundle[edit]["outcome"].warnings
    if warnings:
        w = next((x for x in warnings if "GLOBAL_OPTIMUM_NOT_GUARANTEED" in x), None)
        if w:
            parts.append(f'<text x="42" y="585" font-family="sans-serif" font-size="12.5" fill="#92400e">{_xml("Restricted search: global optimality is not guaranteed on non-convex surfaces.")}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _rotation_check(att_a: Attachment, att_b: Attachment) -> dict:
    surfs0 = build_teapot(0.0)
    pts0 = design_curve_samples(0.0)
    tr0 = preset_transition(0.0)
    out0 = project_curve(
        surfs0, pts0, mode="transition", initial_surface_id="spout", transition=tr0
    )

    yaw = 0.61
    surfs = build_teapot(yaw)
    pts = design_curve_samples(yaw)
    tr = preset_transition(yaw)
    out = project_curve(
        surfs, pts, mode="transition", initial_surface_id="spout", transition=tr
    )
    max_uv = 0.0
    max_dist = 0.0
    for a, b in zip(out0.points, out.points):
        if a.u is not None and b.u is not None:
            max_uv = max(max_uv, abs(a.u - b.u), abs(a.v - b.v))
            max_dist = max(max_dist, abs((a.distance or 0.0) - (b.distance or 0.0)))
    dec0 = map_decal(build_pattern(), out0, surfs0, att_b)
    dec = map_decal(build_pattern(), out, surfs, att_b)
    return {
        "yawRadChecked": yaw,
        "maxUvDifference": max_uv,
        "maxDistanceDifference": max_dist,
        "sameFaceLetters": dec0["faceLetters"] == dec["faceLetters"],
        "seamReadable": bool(dec["seam"].get("readable")),
        "uvInvariant": max_uv < 1e-7 and max_dist < 1e-8,
    }


def build_export_bundle(
    yaw_rad: float = 0.61,
    attachment_a: Optional[Attachment] = None,
    attachment_b: Optional[Attachment] = None,
) -> dict:
    att_a = attachment_a or Attachment(offset=1.70, scale=0.24)
    att_b = attachment_b or Attachment(offset=1.70, scale=0.24)
    surfaces = build_teapot(yaw_rad)
    points = design_curve_samples(yaw_rad)
    transition = preset_transition(yaw_rad)
    pattern = build_pattern("TEAPOT")

    outcome_a = project_curve(
        surfaces, points, mode="spout_only", initial_surface_id="spout"
    )
    outcome_b = project_curve(
        surfaces, points, mode="transition", initial_surface_id="spout",
        transition=transition,
    )
    decal_a = map_decal(pattern, outcome_a, surfaces, att_a)
    decal_b = map_decal(pattern, outcome_b, surfaces, att_b)

    bundle = {
        "yawRad": yaw_rad,
        "pattern": pattern,
        "surfaces": surfaces,
        "designPoints": [list(p) for p in points],
        "transition": [tp.to_dict() for tp in transition],
        "A": {
            "title": "A · continuous spout-only edit",
            "mode": "spout_only",
            "outcome": outcome_a,
            "decal": decal_a,
            "attachment": att_a.to_dict(),
        },
        "B": {
            "title": "B · authorized transition-line patch switch",
            "mode": "transition",
            "outcome": outcome_b,
            "decal": decal_b,
            "attachment": att_b.to_dict(),
        },
    }
    bundle["rotationCheck"] = _rotation_check(att_a, att_b)

    face_b = decal_b["faceLetters"]
    checks = {
        "bTeaOnSpout": face_b.get("spout") == ["T", "E", "A"],
        "bPotOnBody": face_b.get("body") == ["P", "O", "T"],
        "bSeamReadable": bool(decal_b["seam"].get("readable")),
        "aHasVerifiedClipping": bool(decal_a["clipCounts"]) and any(
            i.end_reason and "BEYOND_" in i.end_reason for i in outcome_a.intervals
        ),
        "rotationUvInvariant": bool(bundle["rotationCheck"]["uvInvariant"]),
        "equalDistanceNeverSwitches": all(
            t["action"] == "identity_held_equal_distance_no_auto_switch"
            for t in outcome_b.tie_holds
        ),
    }
    bundle["checks"] = checks
    bundle["correctnessPass"] = all(checks.values())
    return bundle


def export_all(
    outdir: Path | str = DEFAULT_EXPORT_DIR,
    yaw_rad: float = 0.61,
    attachment_a: Optional[Attachment] = None,
    attachment_b: Optional[Attachment] = None,
) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bundle = build_export_bundle(
        yaw_rad=yaw_rad,
        attachment_a=attachment_a,
        attachment_b=attachment_b,
    )

    written = {}
    for edit, stem in (("A", "pattern_A_spout_only"), ("B", "pattern_B_transition")):
        item = bundle[edit]
        svg = outdir / f"{stem}.svg"
        json_path = outdir / f"{stem}.json"
        csv_path = outdir / f"{stem}.csv"
        route_path = outdir / f"{stem}.route.csv"
        write_svg_sheet(svg, bundle, edit)
        _write_decal_csv(csv_path, edit, yaw_rad, item["decal"])
        _write_route_csv(route_path, bundle, only_edit=edit)
        payload = {
            "schema": "surface-studio-sheet-v1",
            "edit": edit,
            "title": item["title"],
            "mode": item["mode"],
            "yawRad": yaw_rad,
            "yawDeg": math.degrees(yaw_rad),
            "attachment": item["attachment"],
            "pattern": bundle["pattern"],
            "designPoints": bundle["designPoints"],
            "transitionLine": bundle["transition"],
            "projection": item["outcome"].to_dict(),
            "decal": item["decal"],
            "checks": bundle["checks"],
            "rotationCheck": bundle["rotationCheck"],
        }
        _write_json(json_path, payload)
        written[edit] = {
            "svg": str(svg), "json": str(json_path),
            "csv": str(csv_path), "routeCsv": str(route_path),
        }

    _write_route_csv(outdir / "projection_routes.csv", bundle)
    report = {
        "schema": "surface-studio-report-v1",
        "yawRad": yaw_rad,
        "yawDeg": math.degrees(yaw_rad),
        "correctnessPass": bundle["correctnessPass"],
        "checks": bundle["checks"],
        "rotationCheck": bundle["rotationCheck"],
        "A": {
            "faceLetters": bundle["A"]["decal"]["faceLetters"],
            "clippedLetters": bundle["A"]["decal"]["clippedLetters"],
            "clipCounts": bundle["A"]["decal"]["clipCounts"],
            "intervals": [i.to_dict() for i in bundle["A"]["outcome"].intervals],
            "warnings": bundle["A"]["outcome"].warnings,
        },
        "B": {
            "faceLetters": bundle["B"]["decal"]["faceLetters"],
            "clippedLetters": bundle["B"]["decal"]["clippedLetters"],
            "clipCounts": bundle["B"]["decal"]["clipCounts"],
            "seam": bundle["B"]["decal"]["seam"],
            "distortion": {
                k: v for k, v in bundle["B"]["decal"]["distortion"].items()
                if k != "perLetter"
            },
            "intervals": [i.to_dict() for i in bundle["B"]["outcome"].intervals],
            "switches": bundle["B"]["outcome"].switches,
            "warnings": bundle["B"]["outcome"].warnings,
        },
        "files": written,
        "routeCsv": str(outdir / "projection_routes.csv"),
    }
    _write_json(outdir / "report.json", report)
    return report
