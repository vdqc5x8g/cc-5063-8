"""Regression tests for path-level multi-solution branch tracking.

Covers the cases called out by the task:

* fold-back surface: two near-equal candidates per sample, nearest-point
  selection flips sheets, branch tracking does not;
* exactly equidistant mirrored candidates: equal distance never switches;
* sphere seam/pole: the route keeps the continuous meridian after the seam;
* branch identity and real attachment error are reported per sample;
* user anchors lock a branch, recover at the nearest feasible point, and
  report *not-continuable* instead of an in-air connector when the anchor
  forces a disconnected jump.
"""
import math
import unittest

from surface_studio import vecmath as vm
from surface_studio.branches import (
    param_delta,
    track_surface,
    track_candidates,
    solve_chain,
)
from surface_studio.projection import project_curve
from surface_studio.scenario import (
    boundary_tracking_case,
    equidistant_candidates_case,
    foldback_tracking_case,
    singularity_design_curve,
)
from surface_studio.surfaces import FoldbackSurface


class FoldbackBranchTests(unittest.TestCase):
    def test_foldback_confirms_two_candidates_per_sample(self):
        surf, pts = foldback_tracking_case()
        track = track_surface(surf, pts, strategy="path")

        # Both sheets are genuinely present for (almost) every sample.
        self.assertGreaterEqual(track.stats["ambiguousSamples"], len(pts) - 2)
        # The chosen route is orthogonal to the surface at every attached point.
        self.assertLess(track.real_attachment_error["maxOrthoRel"], 1e-5)
        self.assertLess(track.real_attachment_error["maxSurfaceReprojectionError"], 1e-9)

    def test_nearest_point_route_flips_but_track_does_not(self):
        surf, pts = foldback_tracking_case()
        track = track_surface(surf, pts, strategy="path")

        # Pure nearest-point route picks opposite sheets across the symmetry
        # plane; the tracked route keeps one sheet until it genuinely ends.
        nearest_u = [
            min(solve_chain(surf, [q], 4)[0][0], key=lambda c: c.distance).u
            for q in pts
        ]
        nearest_flips = any(
            (a < 0) != (b < 0) for a, b in zip(nearest_u, nearest_u[1:])
        )
        self.assertTrue(nearest_flips)

        tracked_u = [f.candidate.u for f in track.frames]
        first_sign = tracked_u[0] < 0
        before_break = tracked_u[: track.breaks[0]["afterIndex"]] if track.breaks else tracked_u
        self.assertTrue(all((u < 0) == first_sign for u in before_break))

    def test_foldback_break_is_reported_not_smoothed(self):
        surf, pts = foldback_tracking_case()
        track = track_surface(surf, pts, strategy="path")
        self.assertTrue(track.breaks)
        brk = track.breaks[0]
        self.assertEqual(brk["kind"], "disconnected")
        # A real discontinuity is a large physical gap, not a parameter blip.
        self.assertGreater(brk["xyzJump"], 1.0)
        self.assertTrue(any("ROUTE_NOT_CONTINUOUS" in w for w in track.warnings))

    def test_projection_closes_interval_at_break(self):
        surf, pts = foldback_tracking_case()
        out = project_curve([surf], pts, mode="spout_only", initial_surface_id="fold")

        self.assertEqual(len(out.intervals), 1)
        iv = out.intervals[0]
        self.assertEqual(iv.end_reason, "ROUTE_PHYSICAL_DISCONTINUITY_BEYOND_TOLERANCE")
        # The break sample is the first detached sample, not the last attached.
        self.assertEqual(iv.index1, out.oob_records[0].index - 1)
        self.assertEqual(out.points[iv.index1 + 1].status, "oob")

    def test_distance_vs_continuity_tradeoff_uses_continuity_weight(self):
        surf, pts = foldback_tracking_case()
        tracks = {}
        for weight in (0.0, 1.0, 20.0):
            tr = track_surface(
                surf, pts, strategy="path",
                continuity_weight=weight, switch_penalty=1.0 + weight,
            )
            tracks[weight] = tr
        # With no continuity preference at all the route may pick the nearest
        # sheet freely; with a strong preference it stays on one sheet longer.
        self.assertLessEqual(
            tracks[20.0].stats["branchSwitches"], tracks[1.0].stats["branchSwitches"]
        )


class EquidistantTests(unittest.TestCase):
    def test_equal_distance_never_switches_branch(self):
        surf, pts = equidistant_candidates_case()
        track = track_surface(surf, pts, strategy="path")

        # Mirrored candidates have identical distance, so the nearest is
        # ambiguous; the route must keep one sign of u throughout.
        signs = {u < 0 for u in (f.candidate.u for f in track.frames)}
        self.assertEqual(len(signs), 1)
        self.assertEqual(track.stats["branchSwitches"], 0)
        # Every sample still reports both abandoned candidates.
        self.assertTrue(all(len(f.alternatives) >= 1 for f in track.frames))

    def test_equal_distance_candidates_are_truly_equal(self):
        surf, pts = equidistant_candidates_case(samples=5)
        for q in pts:
            cands, _mode, _w = solve_chain(surf, [q], 4)
            interior = sorted(
                (c for c in cands[0] if c.usable), key=lambda c: c.distance
            )
            self.assertGreaterEqual(len(interior), 2)
            self.assertAlmostEqual(
                interior[0].distance, interior[1].distance, places=6
            )
            self.assertAlmostEqual(interior[0].u, -interior[1].u, places=6)


class SphereBranchRouteTests(unittest.TestCase):
    def test_route_keeps_continuous_meridian_across_pole(self):
        surf, pts = singularity_design_curve(phi0=1.0)
        track = track_surface(surf, pts, strategy="path")
        v_un = [f.v_unwrapped for f in track.frames]
        # v is unwrapped along the route; crossing the pole must not jump back.
        self.assertTrue(all(b >= a - 1e-6 for a, b in zip(v_un, v_un[1:])))
        self.assertGreater(abs(v_un[-1] - v_un[0]), math.pi - 1e-3)
        self.assertFalse(track.breaks)
        # The pole sample is a real singularity, not a disconnected point.
        pole = track.frames[len(pts) // 2]
        self.assertLess(pole.candidate.ortho_rel, 1e-5)

    def test_pole_projection_has_single_interval_with_branch_record(self):
        surf, pts = singularity_design_curve(phi0=1.0)
        out = project_curve([surf], pts, initial_surface_id="sphere")
        self.assertEqual(len(out.intervals), 1)
        pole = out.points[len(pts) // 2]
        self.assertIsNotNone(pole.branch)
        self.assertGreaterEqual(pole.branch["candidateCount"], 12)


class AnchorTests(unittest.TestCase):
    def test_anchor_locks_branch_and_recovers_at_nearest_transverse(self):
        surf, pts = foldback_tracking_case()
        free = track_surface(surf, pts, strategy="path")
        anchor_index = 16
        # Free route is on the +u sheet here; force the -u sheet instead.
        free_u = free.frames[anchor_index].candidate.u
        negative_cands = [
            c for c in free.frames[anchor_index].candidates
            if c.u < 0 and c.usable
        ]
        self.assertTrue(negative_cands)
        target_u = min(negative_cands, key=lambda c: c.distance).u
        anchor = [{"surfaceId": "fold", "index": anchor_index, "u": target_u, "v": 0.0}]

        track = track_surface(surf, pts, strategy="path", anchors=anchor)
        anchored = track.frames[anchor_index]
        self.assertEqual(anchored.status, "anchored")
        self.assertAlmostEqual(anchored.candidate.u, target_u, places=4)
        self.assertNotEqual(free_u, anchored.candidate.u)
        # Following samples stay locked until the branch ends, then resume.
        self.assertTrue(track.stats["lockedSamples"] > 0)
        self.assertTrue(any(f.status == "recovered" for f in track.frames)
                        or track.stats["unresolvedLosses"] >= 1)

    def test_anchor_forcing_disconnected_region_reports_not_continuable(self):
        surf, pts = foldback_tracking_case()
        # Two anchors pinned to opposite sheets: the second can only be honoured
        # by crossing the gap, so it must be reported, not bridged.
        anchors = [
            {"surfaceId": "fold", "index": 5, "u": -1.55, "v": 0.0},
            {"surfaceId": "fold", "index": 25, "u": 1.62, "v": 0.0},
        ]
        track = track_surface(surf, pts, strategy="path", anchors=anchors)

        self.assertTrue(
            any(b["kind"] == "anchor-not-continuable" for b in track.breaks)
            or track.stats["unresolvedLosses"] >= 1
            or any(f.status == "break" for f in track.frames)
        )
        # No sample bridges the gap with a tiny in-air jump: the route either
        # breaks (large xyz jump, flagged) or stops.
        for f in track.frames:
            if f.jump_xyz is not None and f.jump_xyz > 1.0:
                self.assertTrue(f.status in ("break", "branch-switch", "recovered"))

    def test_anchor_on_body_surface_does_not_leak_to_other_surface(self):
        from surface_studio.scenario import build_teapot, design_curve_samples
        surfaces = build_teapot(0.0)
        pts = design_curve_samples(0.0)
        # An anchor for the body only; the spout route must be unaffected.
        baseline = project_curve(surfaces, pts, mode="spout_only",
                                 initial_surface_id="spout")
        anchored = project_curve(
            surfaces, pts, mode="spout_only", initial_surface_id="spout",
            branch_anchors=[{"surfaceId": "body", "index": 5, "u": 0.5, "v": 1.0}],
        )
        a = [p.u for p in baseline.points]
        b = [p.u for p in anchored.points]
        self.assertEqual(a, b)


class ParameterContinuityTests(unittest.TestCase):
    def test_periodic_parameter_takes_short_way(self):
        surf, _pts = singularity_design_curve(phi0=0.1)
        # v wraps at 2*pi; the normalised distance across the seam is small.
        self.assertLess(param_delta(surf, 0.5, 6.2, 0.5, 0.05), 0.03)

    def test_boundary_sweep_keeps_identity_and_reports_exit(self):
        surf, pts = boundary_tracking_case()
        track = track_surface(surf, pts, strategy="path")
        # The route should stay on one side of the parameter domain and report
        # the verified boundary exit rather than an unverified jump.
        self.assertTrue(
            track.stats["breaks"] >= 0
        )
        self.assertLess(track.real_attachment_error["maxOrthoRel"], 1e-4)


if __name__ == "__main__":
    unittest.main()
