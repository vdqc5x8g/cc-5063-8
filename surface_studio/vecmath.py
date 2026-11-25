"""Small 3D vector and 2x2 linear algebra helpers (pure stdlib)."""
from __future__ import annotations

import math
from typing import Sequence

Vec3 = tuple[float, float, float] | Sequence[float]


def add(a: Vec3, b: Vec3) -> tuple[float, ...]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> tuple[float, ...]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a: Vec3, s: float) -> tuple[float, ...]:
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def dist(a: Vec3, b: Vec3) -> float:
    return norm(sub(a, b))


def normalize(a: Vec3, eps: float = 1e-14) -> tuple[float, ...] | None:
    n = norm(a)
    if n < eps:
        return None
    return mul(a, 1.0 / n)


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def solve2(a: float, b: float, c: float, d: float, rhs1: float, rhs2: float):
    """Solve [[a,b],[c,d]] x = rhs; returns None when singular."""
    det = a * d - b * c
    scale = max(1.0, abs(a), abs(b), abs(c), abs(d))
    if abs(det) <= 1e-18 * scale * scale:
        return None
    x = (rhs1 * d - b * rhs2) / det
    y = (a * rhs2 - rhs1 * c) / det
    return x, y


def rotate_y(p: Vec3, rad: float) -> tuple[float, float, float]:
    """Right handed rotation about +Y."""
    c, s = math.cos(rad), math.sin(rad)
    x, y, z = p
    return (c * x + s * z, y, -s * x + c * z)
