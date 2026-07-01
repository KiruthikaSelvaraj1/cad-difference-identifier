import difflib
import os
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import pytesseract


class OCRUnavailableError(RuntimeError):
    """Raised when OCR dependencies are not available."""


def _read_text_from_image(image: np.ndarray) -> List[Dict]:
    """Extract OCR text from a single image with bounding boxes."""
    if not image.size:
        return []

    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception as exc:
        raise OCRUnavailableError(str(exc)) from exc

    results: List[Dict] = []
    for idx, text in enumerate(data.get("text", [])):
        conf = int(float(data.get("conf", [0])[idx])) if idx < len(data.get("conf", [])) else 0
        if not text or conf < 20:
            continue
        x = int(data["left"][idx])
        y = int(data["top"][idx])
        w = int(data["width"][idx])
        h = int(data["height"][idx])
        results.append(
            {
                "text": text.strip(),
                "location": [x, y, w, h],
            }
        )
    return results


def detect_text_changes(image_a: np.ndarray, image_b: np.ndarray) -> List[Dict]:
    """Compare OCR text between two images and return text-level change records."""
    try:
        texts_a = _read_text_from_image(image_a)
        texts_b = _read_text_from_image(image_b)
    except OCRUnavailableError:
        return []

    normalized_a = [item for item in texts_a if item.get("text")]
    normalized_b = [item for item in texts_b if item.get("text")]

    changes: List[Dict] = []
    used_b_indices = set()

    for item_a in normalized_a:
        text_a = item_a["text"].lower()
        best_match = None
        best_score = 0.0
        best_index = None

        for idx, item_b in enumerate(normalized_b):
            if idx in used_b_indices:
                continue
            text_b = item_b["text"].lower()
            if text_a == text_b:
                best_match = item_b
                best_index = idx
                best_score = 1.0
                break
            if _is_similar_text(text_a, text_b):
                score = difflib.SequenceMatcher(None, text_a, text_b).ratio()
                if score > best_score:
                    best_score = score
                    best_match = item_b
                    best_index = idx

        if best_match is None:
            changes.append(
                {
                    "old_text": item_a["text"],
                    "new_text": "",
                    "change_type": "removed",
                    "location": item_a["location"],
                }
            )
            continue

        if best_index is not None:
            used_b_indices.add(best_index)

        if text_a == best_match["text"].lower():
            continue

        if _is_similar_text(text_a, best_match["text"].lower()):
            changes.append(
                {
                    "old_text": item_a["text"],
                    "new_text": best_match["text"],
                    "change_type": "modified",
                    "location": item_a["location"],
                }
            )
        else:
            changes.append(
                {
                    "old_text": item_a["text"],
                    "new_text": "",
                    "change_type": "removed",
                    "location": item_a["location"],
                }
            )

    for idx, item_b in enumerate(normalized_b):
        if idx in used_b_indices:
            continue
        changes.append(
            {
                "old_text": "",
                "new_text": item_b["text"],
                "change_type": "added",
                "location": item_b["location"],
            }
        )

    return changes


def _best_text_match(text: str, candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None
    return max(candidates, key=lambda cand: difflib.SequenceMatcher(None, text, cand).ratio())


def _is_similar_text(a: str, b: str) -> bool:
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio >= 0.7
