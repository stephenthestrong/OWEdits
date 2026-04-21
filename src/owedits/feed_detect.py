from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from .config import Config
from .video import VideoMeta, crop, iter_frames, probe



@dataclass
class KillEvent:
    video: Path
    t: float  # seconds into the video


@runtime_checkable
class Detector(Protocol):
    def detect(self, video_path: Path, cfg: Config, meta: VideoMeta) -> list[KillEvent]:
        ...


def _has_kf_content(roi: np.ndarray) -> bool:
    """Quick pre-filter: True if the ROI has enough variation to warrant template matching."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(gray.std()) > 12.0


def _nms_matches(
    res: np.ndarray, threshold: float, suppress_h: int, suppress_w: int
) -> list[tuple[float, int, int]]:
    """Iterative NMS on a matchTemplate result map. Returns list of (score, x, y)."""
    work = res.copy()
    out: list[tuple[float, int, int]] = []
    while True:
        _, max_v, _, (mx, my) = cv2.minMaxLoc(work)
        if max_v < threshold:
            break
        out.append((float(max_v), mx, my))
        x0 = max(0, mx - suppress_w // 2)
        y0 = max(0, my - suppress_h // 2)
        x1 = min(work.shape[1], mx + suppress_w)
        y1 = min(work.shape[0], my + suppress_h)
        work[y0:y1, x0:x1] = -1.0
    return out


def match_score(image: np.ndarray, template: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Return (best_score, (x,y) of best match) for TM_CCOEFF_NORMED."""
    if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
        return -1.0, (0, 0)
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(res)
    return float(max_v), max_l


def detect_kill_frames(
    video_path: Path,
    cfg: Config,
    elim_x_tpl: np.ndarray,
    player_icon_tpl: np.ndarray | None,
    meta: VideoMeta | None = None,
) -> tuple[list[KillEvent], VideoMeta]:
    """Scan a video and return (deduped kill events, video metadata)."""
    if meta is None:
        meta = probe(video_path)
    fx, fy, fw, fh = cfg.feed_region.as_pixels(meta.width, meta.height)

    raw_hits: list[tuple[float, int]] = []  # (timestamp_s, y_pixel in result map)
    tpl_h, tpl_w = elim_x_tpl.shape[:2]

    for t, frame in iter_frames(video_path, cfg.sample_fps):
        roi = crop(frame, fx, fy, fw, fh)
        if roi.size == 0:
            continue

        # cheap pre-filter: skip frames where the kill-feed zone has no content
        if not _has_kf_content(roi):
            continue

        res = cv2.matchTemplate(roi, elim_x_tpl, cv2.TM_CCOEFF_NORMED)
        for _score, mx, my in _nms_matches(res, cfg.match_thresholds.elim_x, tpl_h, tpl_w):
            raw_hits.append((t, my))

    return _dedupe_by_row(video_path, raw_hits, cfg.dedupe_window_s, tpl_h), meta


def _dedupe_by_row(
    video: Path, hits: list[tuple[float, int]], window_s: float, row_h: int
) -> list[KillEvent]:
    """Dedupe (timestamp, y_pixel) hits per kill-feed row bucket."""
    hits_sorted = sorted(hits, key=lambda h: h[0])
    last_by_bucket: dict[int, float] = {}
    out: list[KillEvent] = []
    for t, y in hits_sorted:
        bucket = y // max(row_h, 1)
        if t - last_by_bucket.get(bucket, -1e9) < window_s:
            continue
        out.append(KillEvent(video=video, t=t))
        last_by_bucket[bucket] = t
    return out


class KillFeedDetector:
    """Kill-feed detector: matches elim-X and (optionally) player icon templates."""

    def __init__(self, elim_x_tpl: np.ndarray, player_icon_tpl: np.ndarray | None = None):
        self.elim_x_tpl = elim_x_tpl
        self.player_icon_tpl = player_icon_tpl

    def detect(self, video_path: Path, cfg: Config, meta: VideoMeta) -> list[KillEvent]:
        kills, _ = detect_kill_frames(video_path, cfg, self.elim_x_tpl, self.player_icon_tpl, meta)
        return kills


def load_template(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"template not found or unreadable: {path}")
    return img


# My-team kill detector: watches the TOP of the kill-feed for rows matching
# the blue-left (killer = your team) + red-right (victim = enemy) pattern.
#
# OW2 color-codes the kill-feed: your team's name/portrait background is a
# cyan-blue, the enemy team's is red. A kill by your team therefore produces
# a row where blue pixels cluster on the LEFT (killer) and red pixels cluster
# on the RIGHT (victim). This signal is both specific (rejects enemy kills
# and non-kill UI) and robust to slide/fade animation (colors are consistent
# throughout the row's lifetime).
#
# All tunable constants live in Config.color_pattern so the GUI can adjust
# them without a code change. Module-level names below are kept as fallback
# defaults for the few callers that operate without a Config (count_kill_rows
# tests, has_myteam_kill_pattern default-arg path).

_ROW_BRIGHT_THRESH = 200        # grayscale threshold for "bright" text pixels (used by count_kill_rows)
_MIN_BRIGHT_PX_PER_ROW = 25     # used by count_kill_rows
_FP_W, _FP_H = 32, 8            # downsampled fingerprint of a row strip

# Module-level defaults, mirrored in ColorPatternCfg for easy standalone use.
_ROW_HEIGHT_FRAC = 0.039
_BLUE_H_LOW, _BLUE_H_HIGH = 85, 100
_RED_H_HIGH_LOW = 170
_RED_H_LOW_HIGH = 10
_COLOR_S_MIN = 130
_COLOR_V_MIN = 80
_MIN_COLOR_PX_FRAC = 0.01
_MIN_CENTER_SEP_FRAC = 0.10
_ROW_SHIFT_THRESH = 12.0
_ROW_DEDUPE_S = 0.6


def _row_fingerprint(strip: np.ndarray) -> tuple[int, np.ndarray]:
    """Return (bright_px_count, 32×8 fingerprint) of a row-height strip."""
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    bright_count = int(np.count_nonzero(gray > _ROW_BRIGHT_THRESH))
    fp = cv2.resize(gray, (_FP_W, _FP_H)).astype(np.float32)
    return bright_count, fp


def has_myteam_kill_pattern(row_bgr: np.ndarray, cp: "ColorPatternCfg | None" = None) -> bool:
    """True if the row shows a my-team kill: blue (killer) left of red (victim).

    Computes weighted x-centers of saturated blue and red pixel clusters in
    the row, requires both to be meaningfully present, and requires the blue
    center to sit left of the red center by at least min_center_sep_frac of
    the row width. Rejects enemy kills (red-left / blue-right), empty rows
    (no saturated colors), and non-kill UI.

    If `cp` is not given, uses the module-level defaults.
    """
    if row_bgr.size == 0:
        return False
    if cp is None:
        from .config import ColorPatternCfg
        cp = ColorPatternCfg()
    hsv = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    color_ok = (s_ch > cp.color_s_min) & (v_ch > cp.color_v_min)
    blue = ((h_ch >= cp.blue_h_low) & (h_ch <= cp.blue_h_high) & color_ok)
    red = (((h_ch <= cp.red_h_low_high) | (h_ch >= cp.red_h_high_low)) & color_ok)

    min_px = int(row_bgr.shape[0] * row_bgr.shape[1] * cp.min_color_px_frac)
    b_sum = int(blue.sum())
    r_sum = int(red.sum())
    if b_sum < min_px or r_sum < min_px:
        return False

    col_idx = np.arange(row_bgr.shape[1])
    b_center = float((blue.sum(axis=0) * col_idx).sum() / b_sum)
    r_center = float((red.sum(axis=0) * col_idx).sum() / r_sum)
    min_sep = row_bgr.shape[1] * cp.min_center_sep_frac
    return b_center < r_center - min_sep


def count_kill_rows(roi: np.ndarray, row_h_px: int) -> int:
    """Count horizontal kill-feed rows in the ROI by projecting bright pixels onto the y-axis.

    Kept for inspection/tests; the detector uses top-strip change detection instead.
    """
    if roi.size == 0 or row_h_px <= 0:
        return 0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright = gray > _ROW_BRIGHT_THRESH
    per_y = bright.sum(axis=1)
    y_has_text = per_y > _MIN_BRIGHT_PX_PER_ROW

    min_run = max(1, int(row_h_px * 0.4))
    count = 0
    run_start: int | None = None
    for i, has in enumerate(y_has_text):
        if has and run_start is None:
            run_start = i
        elif not has and run_start is not None:
            run_len = i - run_start
            if run_len >= min_run:
                count += max(1, round(run_len / row_h_px))
            run_start = None
    if run_start is not None:
        run_len = len(y_has_text) - run_start
        if run_len >= min_run:
            count += max(1, round(run_len / row_h_px))
    return count


def detect_kill_rows(
    video_path: Path,
    cfg: Config,
    meta: VideoMeta | None = None,
) -> tuple[list[KillEvent], VideoMeta]:
    """Detect my-team kills by watching the top kill-feed row.

    Fires when the top row matches the my-team color pattern
    (blue-left / red-right) AND is a *new* row, where "new" means:
      (a) the previous sample had no my-team row (empty / enemy → my-team), OR
      (b) the current second row matches the previous top row — i.e. the
          previous top got pushed down into the second slot because a new
          row was inserted at the top.

    The color gate rejects enemy kills and non-kill UI. The row-shift gate
    separates consecutive multikills (back-to-back my-team rows with
    different content) from the same kill staying on top.
    """
    if meta is None:
        meta = probe(video_path)
    cp = cfg.color_pattern
    fx, fy, fw, fh = cfg.feed_region.as_pixels(meta.width, meta.height)
    row_h = max(1, int(round(meta.height * cp.row_height_frac)))

    events: list[KillEvent] = []
    prev_top_fp: np.ndarray | None = None  # None when previous sample wasn't a my-team row
    last_event_t = -1e9

    for t, frame in iter_frames(video_path, cfg.sample_fps):
        roi = crop(frame, fx, fy, fw, fh)
        if roi.size == 0 or roi.shape[0] < 2 * row_h:
            continue
        top = roi[:row_h]
        sec = roi[row_h:2 * row_h]

        if not has_myteam_kill_pattern(top, cp):
            prev_top_fp = None
            continue

        _, top_fp = _row_fingerprint(top)
        _, sec_fp = _row_fingerprint(sec)

        if prev_top_fp is None:
            do_fire = True  # empty/enemy → my-team
        else:
            # New row inserted iff the previous top now appears in the second row.
            shift_match = float(np.abs(sec_fp - prev_top_fp).mean())
            do_fire = shift_match < cp.row_shift_thresh

        if do_fire and (t - last_event_t) >= cp.row_dedupe_s:
            events.append(KillEvent(video=video_path, t=t))
            last_event_t = t
        prev_top_fp = top_fp

    return events, meta


class RowCountDetector:
    """Player-agnostic detector: fires on any new kill-feed row."""

    def detect(self, video_path: Path, cfg: Config, meta: VideoMeta) -> list[KillEvent]:
        kills, _ = detect_kill_rows(video_path, cfg, meta)
        return kills
