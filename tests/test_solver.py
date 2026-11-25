import math
import unittest

from surface_studio.solver import closest, select_singular_branch
from surface_studio.projection import project_curve
from surface_studio.surfaces import (
    FoldbackSurface,
    PlaneSurface,
    SpherePatch,
    make_curved_patch,
)
from surface_studio.scenario import singularity_design_curve


class ClosestPointSolverTests(unittest.TestCase):
    def test_plane_interior_distance_and_orthogonality(self):
        surf = PlaneSurface()
        q = (0.4, 1.2, -0.7)
        res = closest(surf, q)
        b = res.best

        self.assertTrue(res.global_guaranteed)
        self.assertAlmostEqual(b.u, 0.4, places=8)
        self.assertAlmostEqual(b.v, -0.7, places=8)
        self.assertAlmostEqual(b.distance, 1.2, places=8)
        self.assertLess(b.ortho_rel, 1e-6)

        p, pu, pv = surf.derivs(b.u, b.v)
        d = tuple(p[i] - q[i] for i in range(3))
        self.assertLess(abs(sum(d[i] * pu[i] for i in range(3))), 1e-8)
        self.assertLess(abs(sum(d[i] * pv[i] for i in range(3))), 1e-8)

    def test_plane_boundary_and_corner_reasons_are_verified(self):
        surf = PlaneSurface()

        edge = closest(surf, (3.0, 0.4, 0.0)).best
        self.assertAlmostEqual(edge.u, 2.0, places=8)
        self.assertEqual(edge.location, "u_max")
        self.assertIn("BEYOND_U_MAX", edge.oob_reasons)

        corner = closest(surf, (3.0, 0.4, 3.0)).best
        self.assertAlmostEqual(corner.u, 2.0, places=8)
        self.assertAlmostEqual(corner.v, 2.0, places=8)
        self.assertEqual(corner.location, "u_max+v_max")
        self.assertIn("BEYOND_U_MAX", corner.oob_reasons)
        self.assertIn("BEYOND_V_MAX", corner.oob_reasons)

    def test_curved_surface_recovers_normal_offset_parameter(self):
        surf = make_curved_patch()
        u0, v0, offset = 0.37, 0.61, 0.18
        p, _pu, _pv = surf.derivs(u0, v0)
        n = surf.normal(u0, v0)
        q = tuple(p[i] + offset * n[i] for i in range(3))

        b = closest(surf, q).best
        self.assertLess(abs(b.u - u0), 1e-5)
        self.assertLess(abs(b.v - v0), 1e-5)
        self.assertAlmostEqual(b.distance, offset, places=6)
        self.assertLess(b.ortho_rel, 1e-6)

    def test_curved_surface_boundary_minimum_is_solved_and_rechecked(self):
        b = closest(make_curved_patch(), (-2.4, 0.0, 0.0)).best
        self.assertAlmostEqual(b.u, 0.0, places=7)
        self.assertEqual(b.location, "u_min")
        self.assertIn("BEYOND_U_MIN", b.oob_reasons)
        self.assertAlmostEqual(b.distance, 0.8, places=6)

    def test_foldback_boundary_minimum(self):
        b = closest(FoldbackSurface(), (4.0, 3.125, 0.0)).best
        self.assertAlmostEqual(b.u, 2.5, places=7)
        self.assertEqual(b.location, "u_max")
        self.assertIn("BEYOND_U_MAX", b.oob_reasons)
        self.assertAlmostEqual(b.distance, 1.5, places=6)

    def test_foldback_has_two_minima_and_reports_restricted_search(self):
        surf = FoldbackSurface()
        res = closest(surf, (0.0, 2.0, 0.0))
        b = res.best

        self.assertAlmostEqual(b.distance, math.sqrt(3.0), places=6)
        self.assertAlmostEqual(abs(b.u), math.sqrt(2.0), places=5)
        self.assertAlmostEqual(b.v, 0.0, places=6)
        self.assertLess(b.ortho_rel, 1e-6)
        self.assertFalse(res.global_guaranteed)
        self.assertTrue(any(
            "GLOBAL_OPTIMUM_NOT_GUARANTEED_RESTRICTED_SEARCH" in w
            for w in res.warnings
        ))
        converged_u = {round(c.u, 4) for c in res.candidates if c.converged}
        self.assertTrue(any(abs(u - math.sqrt(2.0)) < 1e-3 for u in converged_u))
        self.assertTrue(any(abs(u + math.sqrt(2.0)) < 1e-3 for u in converged_u))

    def test_sphere_pole_retains_branches_and_selects_continuing_meridian(self):
        surf = SpherePatch()
        phi0 = 1.0
        _sphere, points = singularity_design_curve(phi0=phi0)
        res = closest(surf, points[len(points) // 2])

        self.assertTrue(res.degenerate)
        self.assertGreaterEqual(len(res.branches), 12)

        expected_phi = phi0 + math.pi
        travel = (math.cos(expected_phi), 0.0, math.sin(expected_phi))
        selected = select_singular_branch(
            surf, res, travel_dir_xyz=travel
        )["chosen"]
        self.assertLess(selected["u"], 1e-5)
        delta = abs((selected["v"] - expected_phi + math.pi) % (2 * math.pi) - math.pi)
        self.assertLess(delta, 0.08)

    def test_sphere_pole_projection_records_explained_branch_and_no_split(self):
        surf, points = singularity_design_curve(phi0=1.0)
        out = project_curve([surf], points, initial_surface_id="sphere")

        self.assertEqual(len(out.intervals), 1)
        pole = out.points[len(points) // 2]
        self.assertTrue(pole.degenerate)
        self.assertIsNotNone(pole.branch_selection)
        self.assertGreaterEqual(pole.branch_selection["candidateCount"], 12)
        after = out.points[len(points) // 2 + 2]
        self.assertAlmostEqual(after.v, 1.0 + math.pi, delta=0.08)


if __name__ == "__main__":
    unittest.main()
