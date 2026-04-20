from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from .config import Config
from .video import VideoMeta, crop, iter_frames, probe


# OW's elim-X is a saturated red. Two hue bands wrap around 0/180 in HSV.
RED_HSV_LOW_1 = np.array([0, 120, 120], dtype=np.uint8)
RED_HSV_HIGH_1 = np.array([10, 255, 255], dtype=np.uint8)
RED_HSV_LOW_2 = np.array([170, 120, 120], dtype=np.uint8)
RED_HSV_HIGH_2 = np.array([180, 255, 255], dtype=np.uint8)


@dataclass
class KillEvent:
    video: Path
    t: float  # seconds into the video


@runtime_checkable
class Detector(Protocol):
    def detect(self, video_path: Path, cfg: Config, meta: VideoMeta) -> list[KillEvent]:
        ...


def red_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, RED_HSV_LOW_1, RED_HSV_HIGH_1)
    m2 = cv2.inRange(hsv, RED_HSV_LOW_2, RED_HSV_HIGH_2)
    return cv2.bitwise_or(m1, m2)


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

        # cheap pre-filter: require some red in the ROI
        mask = red_mask(roi)
        if cv2.countNonZero(mask) < tpl_h * tpl_w // 4:
            continue

        res = cv2.matchTemplate(roi, elim_x_tpl, cv2.TM_CCOEFF_NORMED)
        for _score, mx, my in _nms_matches(res, cfg.match_thresholds.elim_x, tpl_h, tpl_w):
            if player_icon_tpl is not None:
                # player icon sits just LEFT of the X marker; restrict search to this row
                pi_h, pi_w = player_icon_tpl.shape[:2]
                left_x0 = max(0, mx - pi_w - 4)
                left_x1 = mx + 4
                row_y0 = my
                row_y1 = min(roi.shape[0], my + tpl_h)
                left_roi = roi[row_y0:row_y1, left_x0:left_x1]
                p_score, _ = match_score(left_roi, player_icon_tpl)
                if p_score < cfg.match_thresholds.player_icon:
                    continue
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
