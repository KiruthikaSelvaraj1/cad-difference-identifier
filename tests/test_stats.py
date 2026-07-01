import unittest

from backend.stats import compute_statistics


class StatsTests(unittest.TestCase):
    def test_assigns_severity_and_confidence(self):
        stats = compute_statistics(
            regions=[
                {"bbox": [0, 0, 200, 200], "area": 60000, "location": "center"},
            ],
            image_shape=(1000, 1000),
            text_changes=[{"old_text": "400mm", "new_text": "420mm", "change_type": "modified"}],
            has_ssim_signal=True,
            has_absdiff_signal=True,
        )

        self.assertEqual(stats["change_severity"], "major_revision")
        self.assertEqual(stats["regions"][0]["severity"], "critical")
        self.assertGreaterEqual(stats["confidence_score"], 80)


if __name__ == "__main__":
    unittest.main()
