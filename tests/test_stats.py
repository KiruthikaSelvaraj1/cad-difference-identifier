import unittest

from backend.stats import compute_statistics


class StatsTests(unittest.TestCase):
    def test_assigns_severity_and_confidence(self):
        stats = compute_statistics(
            regions=[
                {"bbox": [0, 0, 200, 200], "area": 60000, "location": "center", "change_type": "modification"},
            ],
            image_shape=(1000, 1000),
            text_changes=[{"old_text": "400mm", "new_text": "420mm", "change_type": "modified"}],
            has_ssim_signal=True,
            has_absdiff_signal=True,
        )

        self.assertEqual(stats["change_severity"], "major_revision")
        self.assertEqual(stats["regions"][0]["severity"], "critical")
        self.assertGreaterEqual(stats["confidence_score"], 80)
        self.assertEqual(stats["change_breakdown"], {
            "additions": 0,
            "removals": 0,
            "modifications": 1,
            "positional_shifts": 0,
        })
        self.assertGreaterEqual(stats["impact_score"], 0)
        self.assertIn(stats["impact_label"], {"Low Impact", "Moderate Impact", "High Impact"})


if __name__ == "__main__":
    unittest.main()
