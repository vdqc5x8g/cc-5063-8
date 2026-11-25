from __future__ import annotations

import json
import math
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .decal import Attachment, build_pattern, map_decal
from .export import DEFAULT_EXPORT_DIR, export_all
from .picking import pick_ray
from .projection import make_transition_line, project_curve
from .scenario import (
    build_teapot,
    design_curve_samples,
    preset_transition,
    preset_transition_uv,
    singularity_design_curve,
)
from .solver import closest, select_singular_branch
from .surfaces import FoldbackSurface, PlaneSurface, make_curved_patch

WEB_DIR = Path(__file__).resolve().parent / "web"


def _attachment(data: dict | None) -> Attachment:
    data = data or {}
    allowed = {"offset", "lateral", "scale", "angle_deg"}
    kwargs = {}
    for key in allowed:
        if key in data:
            kwargs[key] = float(data[key])
    if "angleDeg" in data:
        kwargs["angle_deg"] = float(data["angleDeg"])
    if "scale" in kwargs:
        kwargs["scale"] = max(0.05, min(0.8, kwargs["scale"]))
    return Attachment(**kwargs)


def state_payload(yaw_rad: float) -> dict:
    surfaces = build_teapot(yaw_rad)
    return {
        "yawRad": yaw_rad,
        "yawDeg": math.degrees(yaw_rad),
        "surfaces": [
            {
                "id": s.id,
                "label": s.label,
                "kind": s.kind,
                "uDomain": [s.u0, s.u1],
                "vDomain": [s.v0, s.v1],
                "periodicU": s.periodic_u,
                "periodicV": s.periodic_v,
                "mesh": s.mesh(42, 28),
                "distortion": s.distortion_grid(30, 20),
            }
            for s in surfaces
        ],
        "designPoints": [list(p) for p in design_curve_samples(yaw_rad)],
        "presetTransitionUv": preset_transition_uv(),
        "transition": [p.to_dict() for p in preset_transition(yaw_rad)],
        "pattern": build_pattern("TEAPOT"),
        "attachment": Attachment().to_dict(),
        "camera": {"yaw": 0.52, "pitch": 0.30, "zoom": 170, "target": [0.05, 0.16, -0.05]},
    }


def project_payload(body: dict) -> dict:
    yaw = float(body.get("yawRad", body.get("yaw", 0.61)))
    mode = body.get("mode", "transition")
    surfaces = build_teapot(yaw)
    smap = {s.id: s for s in surfaces}
    points = design_curve_samples(yaw)
    att = _attachment(body.get("attachment"))

    transition = None
    if mode == "transition":
        raw_line = body.get("transitionUv")
        if raw_line is None and body.get("usePreset", True):
            transition = preset_transition(yaw)
        elif raw_line:
            transition = make_transition_line(raw_line, smap)

    outcome = project_curve(
        surfaces,
        points,
        mode=mode,
        initial_surface_id=body.get("initialSurfaceId", "spout"),
        transition=transition,
        tie_tol=float(body.get("tieTol", 2e-3)),
    )
    decal = map_decal(build_pattern("TEAPOT"), outcome, surfaces, att)
    return {
        "yawRad": yaw,
        "mode": mode,
        "attachment": att.to_dict(),
        "projection": outcome.to_dict(),
        "decal": decal,
    }


def pick_payload(body: dict) -> dict:
    yaw = float(body.get("yawRad", body.get("yaw", 0.61)))
    origin = body["origin"]
    direction = body["direction"]
    surfaces = build_teapot(yaw)
    result = pick_ray(surfaces, origin, direction)
    result["yawRad"] = yaw
    return result


def diagnose_payload() -> dict:
    items = []

    plane = PlaneSurface()
    b = closest(plane, (0.4, 1.2, -0.7)).best
    items.append({
        "id": "plane",
        "label": "平面：内部最近点/正交",
        "pass": abs(b.u - 0.4) < 1e-6 and abs(b.distance - 1.2) < 1e-6 and b.ortho_rel < 1e-6,
        "metrics": {"u": b.u, "v": b.v, "distance": b.distance, "orthoRel": b.ortho_rel},
    })

    edge = closest(plane, (3.0, 0.4, 3.0)).best
    items.append({
        "id": "boundary",
        "label": "边界/角点：不夹紧，求解并复核 KKT 原因",
        "pass": "BEYOND_U_MAX" in edge.oob_reasons and "BEYOND_V_MAX" in edge.oob_reasons,
        "metrics": {"u": edge.u, "v": edge.v, "location": edge.location, "reasons": edge.oob_reasons},
    })

    curved = make_curved_patch()
    u0, v0 = 0.37, 0.61
    p = curved.point(u0, v0)
    n = curved.normal(u0, v0)
    q = tuple(p[i] + 0.18 * n[i] for i in range(3))
    cb = closest(curved, q).best
    items.append({
        "id": "curved",
        "label": "弯曲 Bézier 面：法向偏移参数恢复",
        "pass": abs(cb.u - u0) < 1e-5 and abs(cb.v - v0) < 1e-5,
        "metrics": {"u": cb.u, "v": cb.v, "distance": cb.distance, "orthoRel": cb.ortho_rel},
    })

    fold = FoldbackSurface()
    fr = closest(fold, (0.0, 2.0, 0.0))
    fb = fr.best
    items.append({
        "id": "foldback",
        "label": "折返非凸面：双局部极小与受限搜索警告",
        "pass": abs(fb.distance - math.sqrt(3.0)) < 1e-5 and not fr.global_guaranteed,
        "metrics": {
            "u": fb.u, "distance": fb.distance,
            "candidateCount": len(fr.candidates),
            "globalGuaranteed": fr.global_guaranteed,
            "warning": next((w for w in fr.warnings if "GLOBAL_OPTIMUM" in w), None),
        },
    })

    sphere, sphere_points = singularity_design_curve()
    sr = closest(sphere, sphere_points[len(sphere_points) // 2])
    sel = select_singular_branch(sphere, sr, travel_dir_xyz=(-math.cos(1.0), 0.0, -math.sin(1.0)))
    items.append({
        "id": "spherePole",
        "label": "参数奇点：保留多个反求分支并按连续性/畸变选择",
        "pass": sr.degenerate and len(sr.branches) >= 12 and sel["chosen"]["u"] < 1e-5,
        "metrics": {
            "branchCount": len(sr.branches),
            "chosenU": sel["chosen"]["u"],
            "chosenV": sel["chosen"]["v"],
            "ambiguous": sel["ambiguous"],
        },
    })
    return {"items": items, "allPass": all(x["pass"] for x in items)}


def export_payload(body: dict | None = None) -> dict:
    body = body or {}
    yaw = float(body.get("yawRad", body.get("yaw", 0.61)))
    att = _attachment(body.get("attachment")) if body.get("attachment") else None
    report = export_all(
        DEFAULT_EXPORT_DIR,
        yaw_rad=yaw,
        attachment_a=att,
        attachment_b=att,
    )
    urls = {}
    for edit, files in report["files"].items():
        urls[edit] = {
            key: "/exports/" + Path(path).name
            for key, path in files.items()
        }
    urls["routeCsv"] = "/exports/projection_routes.csv"
    urls["reportJson"] = "/exports/report.json"
    return {"report": report, "urls": urls}


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "SurfaceStudio/1.0"

    def log_message(self, fmt, *args):
        return

    def _send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path):
        if not path.is_file():
            self.send_error(404, "Not found")
            return
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".js":
            ctype = "text/javascript; charset=utf-8"
        elif path.suffix == ".css":
            ctype = "text/css; charset=utf-8"
        elif path.suffix == ".svg":
            ctype = "image/svg+xml"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            self._send_json({"ok": True, "service": "surface-studio"})
            return
        if path == "/api/state":
            qs = parse_qs(parsed.query)
            yaw = float(qs.get("yaw", ["0.61"])[0])
            self._send_json(state_payload(yaw))
            return
        if path.startswith("/exports/"):
            name = Path(path.removeprefix("/exports/")).name
            self._send_file(DEFAULT_EXPORT_DIR / name)
            return
        if path == "/" or path == "/index.html":
            self._send_file(WEB_DIR / "index.html")
            return
        static_name = Path(path).name
        candidate = WEB_DIR / static_name
        if path in ("/app.js", "/style.css") and candidate.is_file():
            self._send_file(candidate)
            return
        self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception as exc:
            self._send_json({"error": f"invalid JSON request: {exc}"}, status=400)
            return

        try:
            if parsed.path == "/api/project":
                self._send_json(project_payload(body))
            elif parsed.path == "/api/pick":
                self._send_json(pick_payload(body))
            elif parsed.path == "/api/diagnose":
                self._send_json(diagnose_payload())
            elif parsed.path == "/api/export":
                self._send_json(export_payload(body))
            else:
                self._send_json({"error": "unknown endpoint"}, status=404)
        except Exception as exc:
            self._send_json({"error": type(exc).__name__, "message": str(exc)}, status=500)


def create_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), StudioHandler)
