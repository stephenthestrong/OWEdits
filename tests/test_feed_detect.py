from pathlib import Path

import numpy as np

from owedits.feed_detect import KillEvent, _dedupe, red_mask


def test_dedupe_drops_within_window():
    hits = [10.0, 10.4, 10.9, 14.0, 14.5, 25.0]
    out = _dedupe(Path("a.mp4"), hits, window_s=2.0)
    assert [k.t for k in out] == [10.0, 14.0, 25.0]
    assert all(isinstance(k, KillEvent) for k in out)


def test_dedupe_empty():
    assert _dedupe(Path("a.mp4"), [], window_s=2.0) == []


def test_red_mask_detects_saturated_red():
    # Construct a BGR image with a red square in the center.
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[20:30, 20:30] = (0, 0, 255)  # BGR red
    mask = red_mask(img)
    # Only the red region should be non-zero
    assert mask[25, 25] > 0
    assert mask[0, 0] == 0
    # Count should be roughly the area of the square
    import cv2
    assert cv2.countNonZero(mask) == 100
