import unittest
from unittest.mock import patch

import numpy as np

from backend.text_diff import detect_text_changes


class TextDiffTests(unittest.TestCase):
    @patch("backend.text_diff._read_text_from_image")
    def test_detects_modified_text_as_single_change(self, mock_read_text):
        mock_read_text.side_effect = [
            [{"text": "400mm", "location": [10, 20, 30, 40]}],
            [{"text": "420mm", "location": [10, 20, 30, 40]}],
        ]

        image_a = np.zeros((64, 64, 3), dtype=np.uint8)
        image_b = np.zeros((64, 64, 3), dtype=np.uint8)

        changes = detect_text_changes(image_a, image_b)

        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["change_type"], "modified")
        self.assertEqual(changes[0]["old_text"], "400mm")
        self.assertEqual(changes[0]["new_text"], "420mm")


if __name__ == "__main__":
    unittest.main()
