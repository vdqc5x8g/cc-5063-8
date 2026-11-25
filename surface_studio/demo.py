"""Headless teapot demonstration and sheet export.

Run with::

    python3 -m surface_studio.demo
"""
from __future__ import annotations

from .export import export_all


def main() -> int:
    report = export_all()
    print("Surface Studio headless demonstration")
    print("=" * 48)
    print(f"correctness pass: {report['correctnessPass']}")
    for key, ok in report["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {key}")
    rot = report["rotationCheck"]
    print(
        f"rotation UV max diff: {rot['maxUvDifference']:.3e}; "
        f"face letters stable: {rot['sameFaceLetters']}"
    )
    print(f"A faces: {report['A']['faceLetters']}")
    print(f"A clipped: {[x['char'] for x in report['A']['clippedLetters']]}")
    seam = report["B"]["seam"]
    print(
        "B seam: "
        f"gap={seam['physicalGap']:.4f}, "
        f"tangent={seam['tangentAngleDeg']:.2f}°, "
        f"side={seam['sideAngleDeg']:.2f}°, readable={seam['readable']}"
    )
    print("\nExported files:")
    for edit, files in report["files"].items():
        print(f"  {edit}:")
        for kind, path in files.items():
            print(f"    {kind:8s} {path}")
    print(f"  report   {report['files']['A']['svg'].rsplit('/', 1)[0]}/report.json")
    print("\nStart the editing window with: python3 -m surface_studio")
    return 0 if report["correctnessPass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
