from pathlib import Path

import cv2
import numpy as np
import pytest

from owedits.feed_detect import (
    KillEvent,
    _dedupe_by_row,
    _nms_matches,
    _row_fingerprint,
    count_kill_rows,
)


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


def _make_roi(h: int = 300, w: int = 400) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _draw_row(roi: np.ndarray, y: int, row_h: int, width_px: int = 200) -> None:
    # Bright horizontal band to simulate a kill-feed text row
    roi[y:y + row_h, 10:10 + width_px] = 255


def test_count_kill_rows_empty():
    assert count_kill_rows(_make_roi(), row_h_px=42) == 0


def test_count_kill_rows_single_row():
    roi = _make_roi()
    _draw_row(roi, y=20, row_h=42)
    assert count_kill_rows(roi, row_h_px=42) == 1


def test_count_kill_rows_multiple_rows():
    roi = _make_roi()
    _draw_row(roi, y=20, row_h=42)
    _draw_row(roi, y=20 + 42 + 8, row_h=42)  # gap between rows
    _draw_row(roi, y=20 + 2 * (42 + 8), row_h=42)
    assert count_kill_rows(roi, row_h_px=42) == 3


def test_count_kill_rows_noise_below_threshold_ignored():
    roi = _make_roi()
    # Sparse noise: a few bright pixels scattered, below _MIN_BRIGHT_PX_PER_ROW
    roi[50, 10:20] = 255
    assert count_kill_rows(roi, row_h_px=42) == 0


def test_row_fingerprint_identical_when_rows_match():
    # Row-shift detection requires that the *same* row content produces a very
    # similar fingerprint wherever it appears. Build an identical strip twice
    # and verify their fingerprints are near-zero apart.
    a = _make_roi(h=42, w=400)
    b = _make_roi(h=42, w=400)
    for r in (a, b):
        r[5:37, 20:220] = 255  # same row drawn in both strips
    bc_a, fp_a = _row_fingerprint(a)
    bc_b, fp_b = _row_fingerprint(b)
    assert bc_a == bc_b
    assert float(np.abs(fp_a - fp_b).mean()) < 1.0


def test_row_fingerprint_differs_when_content_shifts():
    a = _make_roi(h=42, w=400)
    b = _make_roi(h=42, w=400)
    a[5:37, 20:220] = 255   # content in columns 20-220
    b[5:37, 120:320] = 255  # shifted content in columns 120-320
    _, fp_a = _row_fingerprint(a)
    _, fp_b = _row_fingerprint(b)
    # Different horizontal position → fingerprint differs meaningfully.
    assert float(np.abs(fp_a - fp_b).mean()) > 20.0
