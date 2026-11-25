"""Parametric surfaces with analytic first derivatives.

Every surface exposes:
  point(u, v) -> XYZ
  derivs(u, v) -> (P, dP/du, dP/dv)
  domain, periodic flags
and shared helpers for normals / metric / area distortion / tessellation.

The inverse solver (:mod:`surface_studio.solver`) relies on these analytic
derivatives; the tessellated mesh produced here is only for display and
ray picking, never as the attachment answer.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

from . import vecmath as vm

_DEG_EPS = 1e-11


class ParamSurface:
    kind = "parametric"

    def __init__(
        self,
        sid: str,
        label: str,
        u_domain: tuple[float, float],
        v_domain: tuple[float, float],
        periodic_u: bool = False,
        periodic_v: bool = False,
    ):
        self.id = sid
        self.label = label
        self.u0, self.u1 = u_domain
        self.v0, self.v1 = v_domain
        self.periodic_u = periodic_u
        self.periodic_v = periodic_v

    # --- to override -------------------------------------------------------
    def point(self, u: float, v: float):
        raise NotImplementedError

    def derivs(self, u: float, v: float):
        raise NotImplementedError

    # --- shared analysis ---------------------------------------------------
    def normal(self, u: float, v: float, eps: float = _DEG_EPS):
        p, pu, pv = self.derivs(u, v)
        n = vm.cross(pu, pv)
        ln = vm.norm(n)
        if ln < eps:
            return None
        return vm.mul(n, 1.0 / ln)

    def metric(self, u: float, v: float) -> dict:
        _p, pu, pv = self.derivs(u, v)
        E = vm.dot(pu, pu)
        F = vm.dot(pu, pv)
        G = vm.dot(pv, pv)
        det = max(0.0, E * G - F * F)
        area = math.sqrt(det)
        tr = E + G
        disc = max(0.0, tr * tr - 4.0 * det)
        lmax = 0.5 * (tr + math.sqrt(disc))
        lmin = max(0.0, 0.5 * (tr - math.sqrt(disc)))
        shape_ratio = (lmax / lmin) if lmin > 1e-14 else float("inf")
        return {"E": E, "F": F, "G": G, "areaScale": area, "shapeRatio": shape_ratio}

    def wrap_u(self, u: float) -> float:
        if not self.periodic_u:
            return u
        period = self.u1 - self.u0
        return self.u0 + (u - self.u0) % period

    def wrap_v(self, v: float) -> float:
        if not self.periodic_v:
            return v
        period = self.v1 - self.v0
        return self.v0 + (v - self.v0) % period

    def wrap(self, u: float, v: float) -> tuple[float, float]:
        return self.wrap_u(u), self.wrap_v(v)

    def contains(self, u: float, v: float, tol: float = 1e-9) -> bool:
        return (
            self.u0 - tol <= u <= self.u1 + tol
            and self.v0 - tol <= v <= self.v1 + tol
        )

    def mesh(self, nu: int = 36, nv: int = 24) -> dict:
        positions: list[list[float]] = []
        uvs: list[list[float]] = []
        indices: list[list[int]] = []
        for i in range(nu + 1):
            u = self.u0 + (self.u1 - self.u0) * i / nu
            for j in range(nv + 1):
                v = self.v0 + (self.v1 - self.v0) * j / nv
                p = self.point(u, v)
                positions.append([p[0], p[1], p[2]])
                uvs.append([u, v])
        row = nv + 1
        for i in range(nu):
            for j in range(nv):
                a = i * row + j
                b = (i + 1) * row + j
                c = (i + 1) * row + j + 1
                d = i * row + j + 1
                indices.append([a, b, c, d])
        return {
            "surfaceId": self.id,
            "nu": nu,
            "nv": nv,
            "positions": positions,
            "uvs": uvs,
            "indices": indices,
        }

    def distortion_grid(self, nu: int = 42, nv: int = 30) -> dict:
        values: list[list[float]] = []
        ratios: list[list[float]] = []
        us, vs = [], []
        areas: list[float] = []
        for i in range(nu + 1):
            u = self.u0 + (self.u1 - self.u0) * i / nu
            us.append(u)
            row_a, row_r = [], []
            for j in range(nv + 1):
                v = self.v0 + (self.v1 - self.v0) * j / nv
                m = self.metric(u, v)
                row_a.append(m["areaScale"])
                row_r.append(min(m["shapeRatio"], 1e6))
                if m["areaScale"] > 1e-9:
                    areas.append(m["areaScale"])
            values.append(row_a)
            ratios.append(row_r)
        for j in range(nv + 1):
            v = self.v0 + (self.v1 - self.v0) * j / nv
            vs.append(v)
        if areas:
            srt = sorted(areas)
            median = srt[len(srt) // 2]
            amin, amax = min(areas), max(areas)
            mean = sum(areas) / len(areas)
        else:
            median = amin = amax = mean = 0.0
        return {
            "surfaceId": self.id,
            "us": us,
            "vs": vs,
            "area": values,
            "shapeRatio": ratios,
            "areaMin": amin,
            "areaMax": amax,
            "areaMean": mean,
            "areaMedian": median,
            "uDomain": [self.u0, self.u1],
            "vDomain": [self.v0, self.v1],
        }


class TransformedSurface(ParamSurface):
    """Rigid transform wrapper; identity (id) of the surface is preserved."""

    def __init__(self, base: ParamSurface, transform: Callable[[vm.Vec3], vm.Vec3]):
        super().__init__(
            base.id,
            base.label,
            (base.u0, base.u1),
            (base.v0, base.v1),
            base.periodic_u,
            base.periodic_v,
        )
        self.base = base
        self._t = transform

    def point(self, u, v):
        return self._t(self.base.point(u, v))

    def derivs(self, u, v):
        p, pu, pv = self.base.derivs(u, v)
        # Rotation-only rigid transforms keep derivative vectors rotated.
        return self._t(p), _rotate_only(self._t, p, pu), _rotate_only(self._t, p, pv)

    def normal(self, u, v, eps: float = _DEG_EPS):
        # Delegate orientation to the base surface: subclasses may override the
        # normal's sign (e.g. a swept spout points outward), and computing from
        # cross(du, dv) here would silently flip it after a rigid transform.
        n = self.base.normal(u, v, eps)
        return None if n is None else _rotate_only(self._t, self.base.point(u, v), n)


def _rotate_only(t: Callable, origin_p, vec):
    """Apply the rotational part of a rigid transform to a derivative vector."""
    a = t(origin_p)
    b = t(vm.add(origin_p, vec))
    return vm.sub(b, a)


def yaw_transform(yaw_rad: float):
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)

    def tr(p):
        x, y, z = p
        return (c * x + s * z, y, -s * x + c * z)

    return tr


# ---------------------------------------------------------------------------
# Plane
# ---------------------------------------------------------------------------
class PlaneSurface(ParamSurface):
    kind = "plane"

    def __init__(
        self,
        sid: str = "plane",
        label: str = "平面",
        origin: vm.Vec3 = (0.0, 0.0, 0.0),
        e_u: vm.Vec3 = (1.0, 0.0, 0.0),
        e_v: vm.Vec3 = (0.0, 0.0, 1.0),
        u_domain=(-2.0, 2.0),
        v_domain=(-2.0, 2.0),
    ):
        super().__init__(sid, label, u_domain, v_domain)
        self.o = tuple(origin)
        self.eu = tuple(e_u)
        self.ev = tuple(e_v)

    def point(self, u, v):
        return vm.add(vm.add(self.o, vm.mul(self.eu, u)), vm.mul(self.ev, v))

    def derivs(self, u, v):
        return self.point(u, v), self.eu, self.ev


# ---------------------------------------------------------------------------
# Bicubic Bezier patch
# ---------------------------------------------------------------------------
def _bernstein3(t: float) -> tuple[float, float, float, float]:
    q = 1.0 - t
    return (q ** 3, 3.0 * t * q * q, 3.0 * t * t * q, t ** 3)


def _dbernstein3(t: float) -> tuple[float, float, float, float]:
    q = 1.0 - t
    return (-3.0 * q * q, 3.0 * q * (1.0 - 3.0 * t), 3.0 * t * (2.0 - 3.0 * t), 3.0 * t * t)


class BezierSurface(ParamSurface):
    kind = "bezier"

    def __init__(self, sid: str, label: str, control_points: Sequence[Sequence[vm.Vec3]]):
        super().__init__(sid, label, (0.0, 1.0), (0.0, 1.0))
        assert len(control_points) == 4 and all(len(r) == 4 for r in control_points)
        self.cp = [[tuple(p) for p in row] for row in control_points]

    def _eval(self, bu, bv):
        p = [0.0, 0.0, 0.0]
        for i in range(4):
            for j in range(4):
                w = bu[i] * bv[j]
                for k in range(3):
                    p[k] += w * self.cp[i][j][k]
        return tuple(p)

    def point(self, u, v):
        return self._eval(_bernstein3(u), _bernstein3(v))

    def derivs(self, u, v):
        bu, bv = _bernstein3(u), _bernstein3(v)
        du, dv = _dbernstein3(u), _dbernstein3(v)
        return self._eval(bu, bv), self._eval(du, bv), self._eval(bu, dv)


def make_curved_patch(sid: str = "curved", label: str = "弯曲曲面") -> BezierSurface:
    """A gently twisted bent sheet with a clear curved cross section."""
    cps = []
    for i in range(4):
        row = []
        u = i / 3.0
        for j in range(4):
            v = j / 3.0
            x = (u - 0.5) * 3.2
            z = (v - 0.5) * 2.2
            y = 0.55 * math.sin(u * math.pi) + 0.28 * (v - 0.5) + 0.18 * x * z
            row.append((x, y, z))
        cps.append(row)
    return BezierSurface(sid, label, cps)


# ---------------------------------------------------------------------------
# Fold-back / hairpin surface: extrusion of y = a u^2 along z
# ---------------------------------------------------------------------------
class FoldbackSurface(ParamSurface):
    """Hairpin fold: S(u,v) = (u, a*u^2, v).

    A point on the open side of the fold has two symmetric local minima,
    which makes this a genuine non-convex closest-point test case.
    """

    kind = "foldback"

    def __init__(
        self,
        sid: str = "fold",
        label: str = "折返面",
        a: float = 0.5,
        u_domain=(-2.5, 2.5),
        v_domain=(-1.2, 1.2),
    ):
        super().__init__(sid, label, u_domain, v_domain)
        self.a = a

    def point(self, u, v):
        return (u, self.a * u * u, v)

    def derivs(self, u, v):
        return (
            (u, self.a * u * u, v),
            (1.0, 2.0 * self.a * u, 0.0),
            (0.0, 0.0, 1.0),
        )


# ---------------------------------------------------------------------------
# Sphere patch including a parameter pole (singularity)
# ---------------------------------------------------------------------------
class SpherePatch(ParamSurface):
    """Lat-long sphere patch with theta=0 north pole included.

    At the pole dP/dphi = 0 and the normal is undefined: every phi is the
    same 3D point. The inverse solver must retain multiple phi branches
    instead of returning one arbitrary parameter pair.
    """

    kind = "sphere"

    def __init__(
        self,
        sid: str = "sphere",
        label: str = "奇点球面",
        radius: float = 1.0,
        theta_domain=(0.0, 1.45),
        phi_domain=(0.0, 2.0 * math.pi),
    ):
        super().__init__(sid, label, theta_domain, phi_domain, periodic_v=True)
        self.r = radius

    def point(self, theta, phi):
        st, ct = math.sin(theta), math.cos(theta)
        return (
            self.r * st * math.cos(phi),
            self.r * ct,
            self.r * st * math.sin(phi),
        )

    def derivs(self, theta, phi):
        st, ct = math.sin(theta), math.cos(theta)
        p = (self.r * st * math.cos(phi), self.r * ct, self.r * st * math.sin(phi))
        pt = (self.r * ct * math.cos(phi), -self.r * st, self.r * ct * math.sin(phi))
        pp = (-self.r * st * math.sin(phi), 0.0, self.r * st * math.cos(phi))
        return p, pt, pp


# ---------------------------------------------------------------------------
# Natural cubic spline for lathe profiles
# ---------------------------------------------------------------------------
class CubicSpline1D:
    def __init__(self, knots: Sequence[float], values: Sequence[float]):
        self.xs = list(knots)
        self.ys = list(values)
        n = len(knots)
        h = [self.xs[i + 1] - self.xs[i] for i in range(n - 1)]
        # tridiagonal system for second derivatives
        A = [[0.0] * n for _ in range(n)]
        rhs = [0.0] * n
        A[0][0] = 1.0
        A[-1][-1] = 1.0
        for i in range(1, n - 1):
            A[i][i - 1] = h[i - 1]
            A[i][i] = 2.0 * (h[i - 1] + h[i])
            A[i][i + 1] = h[i]
            rhs[i] = 6.0 * (
                (self.ys[i + 1] - self.ys[i]) / h[i]
                - (self.ys[i] - self.ys[i - 1]) / h[i - 1]
            )
        m = _tridiagonal_solve(A, rhs)
        self.m = m
        self.h = h

    def _segment(self, x: float) -> int:
        if x <= self.xs[0]:
            return 0
        if x >= self.xs[-1]:
            return len(self.xs) - 2
        i = 0
        while x > self.xs[i + 1]:
            i += 1
        return i

    def eval(self, x: float) -> float:
        i = self._segment(x)
        x0, x1 = self.xs[i], self.xs[i + 1]
        h = self.h[i]
        a = (x1 - x) / h
        b = (x - x0) / h
        return (
            a * self.ys[i]
            + b * self.ys[i + 1]
            + ((a ** 3 - a) * self.m[i] + (b ** 3 - b) * self.m[i + 1]) * h * h / 6.0
        )

    def deriv(self, x: float) -> float:
        i = self._segment(x)
        x0, x1 = self.xs[i], self.xs[i + 1]
        h = self.h[i]
        return (
            (self.ys[i + 1] - self.ys[i]) / h
            + (-(3.0 * (x1 - x) ** 2 - h * h) * self.m[i]
               + (3.0 * (x - x0) ** 2 - h * h) * self.m[i + 1])
            / (6.0 * h)
        )


def _tridiagonal_solve(A, b):
    n = len(b)
    # Generic Gaussian elimination on small dense matrix (n ~ 8).
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        d = M[col][col]
        if abs(d) < 1e-18:
            continue
        for r in range(col + 1, n):
            f = M[r][col] / d
            if f:
                for k in range(col, n + 1):
                    M[r][k] -= f * M[col][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        val = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = val / M[i][i] if abs(M[i][i]) > 1e-18 else 0.0
    return x


# ---------------------------------------------------------------------------
# Lathe surface (teapot body)
# ---------------------------------------------------------------------------
class LatheSurface(ParamSurface):
    kind = "lathe"

    def __init__(
        self,
        sid: str,
        label: str,
        profile: Sequence[tuple[float, float]],
        phi_max: float = 2.0 * math.pi,
    ):
        super().__init__(sid, label, (0.0, 1.0), (0.0, phi_max), periodic_v=True)
        ts = [i / (len(profile) - 1) for i in range(len(profile))]
        self.r_spline = CubicSpline1D(ts, [p[0] for p in profile])
        self.z_spline = CubicSpline1D(ts, [p[1] for p in profile])

    def radius_z(self, t):
        return self.r_spline.eval(t), self.z_spline.eval(t)

    def point(self, t, phi):
        r, z = self.radius_z(t)
        return (r * math.cos(phi), z, r * math.sin(phi))

    def derivs(self, t, phi):
        r = self.r_spline.eval(t)
        z = self.z_spline.eval(t)
        dr = self.r_spline.deriv(t)
        dz = self.z_spline.deriv(t)
        c, s = math.cos(phi), math.sin(phi)
        return (
            (r * c, z, r * s),
            (dr * c, dz, dr * s),
            (-r * s, 0.0, r * c),
        )


def make_teapot_body(sid: str = "body", label: str = "壶身") -> LatheSurface:
    profile = [
        (0.46, -0.86),
        (0.78, -0.80),
        (1.02, -0.52),
        (1.13, -0.12),
        (1.10, 0.22),
        (0.96, 0.55),
        (0.74, 0.78),
        (0.58, 0.90),
        (0.56, 1.02),
    ]
    return LatheSurface(sid, label, profile)


# ---------------------------------------------------------------------------
# Swept spout along a cubic Bezier centerline in the XY plane
# ---------------------------------------------------------------------------
class SweptSpout(ParamSurface):
    kind = "spout"

    def __init__(
        self,
        sid: str = "spout",
        label: str = "壶嘴",
        p0=(-0.2, 0.0, 0.0),
        p1=(0.3, 0.1, 0.0),
        p2=(0.4, 0.9, 0.0),
        p3=(1.1, 1.0, 0.0),
        r_root: float = 0.30,
        r_tip: float = 0.13,
    ):
        super().__init__(sid, label, (0.0, 1.0), (0.0, 2.0 * math.pi), periodic_v=True)
        self.c = [(p[0], p[1]) for p in (p0, p1, p2, p3)]
        self.r_root = r_root
        self.r_tip = r_tip

    def radius(self, s: float) -> float:
        return self.r_root * (1.0 - s) + self.r_tip * s

    def _center_frame(self, s):
        c = self.c
        q = 1.0 - s
        cx = q ** 3 * c[0][0] + 3 * q * q * s * c[1][0] + 3 * q * s * s * c[2][0] + s ** 3 * c[3][0]
        cy = q ** 3 * c[0][1] + 3 * q * q * s * c[1][1] + 3 * q * s * s * c[2][1] + s ** 3 * c[3][1]
        dx = 3 * q * q * (c[1][0] - c[0][0]) + 6 * q * s * (c[2][0] - c[1][0]) + 3 * s * s * (c[3][0] - c[2][0])
        dy = 3 * q * q * (c[1][1] - c[0][1]) + 6 * q * s * (c[2][1] - c[1][1]) + 3 * s * s * (c[3][1] - c[2][1])
        ddx = 6 * q * (c[2][0] - 2 * c[1][0] + c[0][0]) + 6 * s * (c[3][0] - 2 * c[2][0] + c[1][0])
        ddy = 6 * q * (c[2][1] - 2 * c[1][1] + c[0][1]) + 6 * s * (c[3][1] - 2 * c[2][1] + c[1][1])
        qn = math.hypot(dx, dy)
        tx, ty = dx / qn, dy / qn
        qdot = (dx * ddx + dy * ddy) / qn
        tpx = ddx / qn - dx * qdot / (qn * qn)
        tpy = ddy / qn - dy * qdot / (qn * qn)
        nx, ny = -ty, tx
        npx, npy = -tpy, tpx
        return (cx, cy, 0.0), (dx, dy, 0.0), (nx, ny, 0.0), (npx, npy, 0.0)

    def point(self, s, theta):
        cen, _tan, nrm, _np = self._center_frame(s)
        r = self.radius(s)
        ct, st = math.cos(theta), math.sin(theta)
        off = vm.add(vm.mul(nrm, r * ct), (0.0, 0.0, r * st))
        return vm.add(cen, off)

    def derivs(self, s, theta):
        cen, cp, nrm, npr = self._center_frame(s)
        r = self.radius(s)
        rp = self.r_tip - self.r_root
        ct, st = math.cos(theta), math.sin(theta)
        radial = vm.add(vm.mul(nrm, ct), (0.0, 0.0, st))
        p = vm.add(cen, vm.mul(radial, r))
        # dP/ds = C' + r'(cosθ N + sinθ B) + r cosθ N'  (B is constant)
        ps = vm.add(cp, vm.mul(radial, rp))
        ps = vm.add(ps, vm.mul(npr, r * ct))
        ptheta = vm.add(vm.mul(nrm, -r * st), (0.0, 0.0, r * ct))
        return p, ps, ptheta

    def normal(self, s: float, theta: float, eps: float = _DEG_EPS):
        n = super().normal(s, theta, eps)
        return None if n is None else vm.mul(n, -1.0)


def make_teapot_spout(sid: str = "spout", label: str = "壶嘴") -> SweptSpout:
    # Root is embedded in the body belly (+X side); tip rises and points out.
    return SweptSpout(
        sid,
        label,
        p0=(0.82, -0.18, 0.0),
        p1=(1.28, -0.10, 0.0),
        p2=(1.42, 0.92, 0.0),
        p3=(2.08, 1.16, 0.0),
        r_root=0.31,
        r_tip=0.125,
    )
