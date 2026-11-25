"""Multi-solution branch regression runner (command-line, auxiliary).

The graphical editing window remains the primary interface; this module only
freezes the numerical branch cases into a repeatable check::

    python3 -m surface_studio.branch_regression
    python3 -m surface_studio.branch_regression --json out.json

It exercises the cases the editing window relies on -- fold-back sheets,
exactly equidistant candidates, the sphere seam/pole and boundary sweeps --
and prints the chosen strategy, abandoned candidates and real attachment error
per sample.  Exit code is 0 only when every case passes.
"""
from __future__ import annotations

import argparse
import json
import sys

from .branches import track_surface
from .scenario import (
    boundary_tracking_case,
    equidistant_candidates_case,
    foldback_tracking_case,
    singularity_design_curve,
)
from .surfaces import FoldbackSurface


def _foldback_case() -> dict:
    surf, pts = foldback_tracking_case()
    track = track_surface(surf, pts, strategy="auto")
    first_sign = track.frames[0].candidate.u < 0
    end = track.breaks[0]["afterIndex"] if track.breaks else len(track.frames)
    one_sheet = all(
        (track.frames[i].candidate.u < 0) == first_sign for i in range(end)
    )
    return {
        "case": "foldback",
        "pass": (
            track.stats["ambiguousSamples"] >= len(pts) - 2
            and one_sheet
            and bool(track.breaks)
            and track.real_attachment_error["maxOrthoRel"] < 1e-5
        ),
        "chosenStrategy": track.strategy,
        "strategyTotals": track.comparison["totals"],
        "ambiguousSamples": track.stats["ambiguousSamples"],
        "branchSwitches": track.stats["branchSwitches"],
        "breaks": track.breaks,
        "maxOrthoRel": track.real_attachment_error["maxOrthoRel"],
        "abandonedPerSample": [len(f.alternatives) for f in track.frames],
    }


def _equidistant_case() -> dict:
    surf, pts = equidistant_candidates_case()
    track = track_surface(surf, pts, strategy="path")
    signs = {f.candidate.u < 0 for f in track.frames}
    return {
        "case": "equidistant",
        "pass": len(signs) == 1 and track.stats["branchSwitches"] == 0,
        "branchSwitches": track.stats["branchSwitches"],
        "distinctSigns": len(signs),
        "maxOrthoRel": track.real_attachment_error["maxOrthoRel"],
    }


def _sphere_case() -> dict:
    surf, pts = singularity_design_curve(phi0=1.0)
    track = track_surface(surf, pts, strategy="path")
    v_un = [f.v_unwrapped for f in track.frames]
    monotone = all(b >= a - 1e-6 for a, b in zip(v_un, v_un[1:]))
    return {
        "case": "sphereSeam",
        "pass": monotone and not track.breaks
        and track.real_attachment_error["maxOrthoRel"] < 1e-4,
        "breaks": len(track.breaks),
        "monotoneUnwrappedV": monotone,
        "vStart": v_un[0],
        "vEnd": v_un[-1],
    }


def _boundary_case() -> dict:
    surf, pts = boundary_tracking_case()
    track = track_surface(surf, pts, strategy="path")
    return {
        "case": "boundarySweep",
        "pass": track.real_attachment_error["maxOrthoRel"] < 1e-4,
        "breaks": len(track.breaks),
        "maxOrthoRel": track.real_attachment_error["maxOrthoRel"],
        "maxParamStep": track.stats["maxParamStep"],
        "attachmentError": track.real_attachment_error,
    }


def _anchor_case() -> dict:
    surf, pts = foldback_tracking_case()
    free = track_surface(surf, pts, strategy="path")
    idx = 16
    negative = [
        c for c in free.frames[idx].candidates if c.u < 0 and c.usable
    ]
    if not negative:
        return {"case": "anchor", "pass": False, "reason": "no opposite branch"}
    target = min(negative, key=lambda c: c.distance)
    track = track_surface(
        surf, pts, strategy="path",
        anchors=[{"surfaceId": "fold", "index": idx, "u": target.u, "v": 0.0}],
    )
    anchored = track.frames[idx]
    return {
        "case": "anchor",
        "pass": (
            anchored.status == "anchored"
            and abs(anchored.candidate.u - target.u) < 1e-3
            and (any(f.status == "recovered" for f in track.frames)
                 or track.stats["unresolvedLosses"] >= 1
                 or any(b["kind"] == "anchor-not-continuable" for b in track.breaks))
        ),
        "anchoredU": anchored.candidate.u,
        "targetU": target.u,
        "statusAtAnchor": anchored.status,
        "recoveries": track.stats["recoveries"],
        "breaks": track.breaks,
    }


CASES = (_foldback_case, _equidistant_case, _sphere_case, _boundary_case, _anchor_case)


def run() -> dict:
    results = [case() for case in CASES]
    return {
        "schema": "surface-studio-branch-regression-v1",
        "pass": all(r["pass"] for r in results),
        "cases": results,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-solution branch regression (auxiliary to the editor UI)"
    )
    parser.add_argument("--json", metavar="PATH", help="write the full report here")
    args = parser.parse_args(argv)

    report = run()
    print("Surface Studio branch regression")
    print("=" * 40)
    for r in report["cases"]:
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {r['case']}")
        print(f"         {json.dumps({k: v for k, v in r.items() if k != 'case'}, ensure_ascii=False)}")
    print(f"\noverall: {'PASS' if report['pass'] else 'FAIL'}")
    print("Graphical entry point: python3 -m surface_studio")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"report written to {args.json}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
