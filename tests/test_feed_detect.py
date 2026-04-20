from pathlib import Path

import cv2
import numpy as np
import pytest

from owedits.feed_detect import KillEvent, _dedupe_by_row, _nms_matches, red_mask


def test_dedupe_by_row_drops_within_window():
    # Same row bucket (y=0 → bucket 0), third hit is too close to second
    hits = [(10.0, 0), (10.4, 0), (10.9, 0), (14.0, 0), (14.5, 0), (25.0, 0)]
    out = _dedupe_by_row(Path("a.mp4"), hits, window_s=2.0, row_h=20)
    assert [k.t for k in out] == [10.0, 14.0, 25.0]
    assert all(isinstance(k, KillEvent) for k in out)


def test_dedupe_by_row_different_rows_not_merged():
    # Two simultaneous kills in different rows should both be kept
    hits = [(10.0, 0), (10.0, 25)]  # row 0 and row 1 (row_h=20)
    out = _dedupe_by_row(Path("a.mp4"), hits, window_s=2.0, row_h=20)
    assert len(out) == 2


def test_dedupe_by_row_empty():
    assert _dedupe_by_row(Path("a.mp4"), [], window_s=2.0, row_h=20) == []


def test_red_mask_detects_saturated_red():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[20:30, 20:30] = (0, 0, 255)  # BGR red
    mask = red_mask(img)
    assert mask[25, 25] > 0
    assert mask[0, 0] == 0
    assert cv2.countNonZero(mask) == 100


def test_nms_matches_finds_multiple_peaks():
    # Build a result map with two clear peaks
    res = np.zeros((100, 100), dtype=np.float32)
    res[10, 10] = 0.9
    res[50, 50] = 0.8
    matches = _nms_matches(res, threshold=0.7, suppress_h=15, suppress_w=15)
    assert len(matches) == 2
    scores = [s for s, _, _ in matches]
    assert scores[0] > scores[1]  # highest first


def test_nms_matches_suppresses_nearby():
    # Two peaks close together — NMS should suppress the weaker one
    res = np.zeros((100, 100), dtype=np.float32)
    res[10, 10] = 0.9
    res[12, 12] = 0.85  # within suppress radius
    matches = _nms_matches(res, threshold=0.7, suppress_h=15, suppress_w=15)
    assert len(matches) == 1
    assert matches[0][0] == pytest.approx(0.9)


def test_nms_matches_none_above_threshold():
    res = np.full((50, 50), 0.5, dtype=np.float32)
    assert _nms_matches(res, threshold=0.7, suppress_h=10, suppress_w=10) == []
