import unittest

from backend.summarizer import generate_difference_explanation


class SummarizerTests(unittest.TestCase):
    def test_difference_explanation_mentions_regions_and_locations(self):
        stats = {
            "region_count": 2,
            "percent_changed": 7.5,
            "regions": [
                {"location": "top-left", "bbox": [10, 20, 30, 40], "area": 1200},
                {"location": "bottom-right", "bbox": [100, 150, 50, 40], "area": 800},
            ],
        }

        explanation = generate_difference_explanation(stats)

        self.assertIn("2", explanation)
        self.assertIn("top-left", explanation)
        self.assertIn("bottom-right", explanation)
        self.assertIn("7.5%", explanation)


if __name__ == "__main__":
    unittest.main()
