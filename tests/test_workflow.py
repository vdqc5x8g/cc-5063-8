import json
import unittest

from surface_studio.decal import Attachment, build_pattern, map_decal
from surface_studio.export import build_export_bundle
from surface_studio.projection import project_curve
from surface_studio.scenario import (
    build_teapot,
    design_curve_samples,
    preset_transition,
)
from surface_studio.surfaces import PlaneSurface


class IdentityAndTransitionTests(unittest.TestCase):
    def test_equal_nearest_distance_never_switches_identity(self):
        a = PlaneSurface(sid="A")
        b = PlaneSurface(sid="B", origin=(0.0, 2.0, 0.0))
        points = [(-0.5, 1.0, -0.4), (0.0, 1.0, 0.0), (0.5, 1.0, 0.4)]

        out = project_curve([a, b], points, initial_surface_id="A")

        self.assertTrue(out.tie_holds)
        self.assertTrue(all(p.surface_id == "A" for p in out.points))
        self.assertTrue(all(p.tie_hold for p in out.points))
        self.assertEqual(
            out.tie_holds[0]["action"],
            "identity_held_equal_distance_no_auto_switch",
        )

    def test_user_transition_line_authorizes_real_patch_switch(self):
        surfaces = build_teapot(0.0)
        out = project_curve(
            surfaces,
            design_curve_samples(0.0),
            mode="transition",
            initial_surface_id="spout",
            transition=preset_transition(0.0),
        )

        self.assertEqual(len(out.intervals), 2)
        self.assertEqual([i.surface_id for i in out.intervals], ["spout", "body"])
        self.assertEqual(len(out.switches), 1)
        self.assertEqual(out.switches[0]["reason"], "USER_TRANSITION_LINE")
        self.assertLess(out.switches[0]["targetDistance"], 0.08)
        json.dumps(out.to_dict())


class TeapotSheetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = build_export_bundle(yaw_rad=0.61)

    def test_expected_letter_faces_and_clipping(self):
        decal_b = self.bundle["B"]["decal"]
        decal_a = self.bundle["A"]["decal"]

        self.assertEqual(decal_b["faceLetters"]["spout"], ["T", "E", "A"])
        self.assertEqual(decal_b["faceLetters"]["body"], ["P", "O", "T"])
        self.assertIn("spout", decal_a["faceLetters"])
        self.assertTrue(decal_a["clippedLetters"])
        self.assertTrue(decal_a["clipCounts"])
        self.assertTrue(any(
            i.end_reason and "BEYOND_" in i.end_reason
            for i in self.bundle["A"]["outcome"].intervals
        ))

    def test_seam_is_geometrically_registerable(self):
        seam = self.bundle["B"]["decal"]["seam"]
        self.assertTrue(seam["hasSeam"])
        self.assertLess(seam["physicalGap"], 0.14)
        self.assertLess(seam["tangentAngleDeg"], 35.0)
        self.assertLess(seam["sideAngleDeg"], 35.0)
        self.assertTrue(seam["readable"])

    def test_rotation_preserves_parameters_and_verdict(self):
        check = self.bundle["rotationCheck"]
        self.assertTrue(check["uvInvariant"])
        self.assertLess(check["maxUvDifference"], 1e-6)
        self.assertTrue(check["sameFaceLetters"])
        self.assertTrue(self.bundle["correctnessPass"])
        json.dumps(self.bundle["B"]["outcome"].to_dict())
        json.dumps(self.bundle["B"]["decal"])


if __name__ == "__main__":
    unittest.main()
