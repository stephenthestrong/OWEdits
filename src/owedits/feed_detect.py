from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .video import crop, iter_frames, probe


# OW's elim-X is a saturated red. Two hue bands wrap around 0/180 in HSV.
RED_HSV_LOW_1 = np.array([0, 120, 120], dtype=np.uint8)
RED_HSV_HIGH_1 = np.array([10, 255, 255], dtype=np.uint8)
RED_HSV_LOW_2 = np.array([170, 120, 120], dtype=np.uint8)
RED_HSV_HIGH_2 = np.array([180, 255, 255], dtype=np.uint8)


@dataclass
class KillEvent:
    video: Path
    t: float  # seconds into the video


def red_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, RED_HSV_LOW_1, RED_HSV_HIGH_1)
    m2 = cv2.inRange(hsv, RED_HSV_LOW_2, RED_HSV_HIGH_2)
    return cv2.bitwise_or(m1, m2)


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
) -> list[KillEvent]:
    """Scan a video and return deduped kill-event timestamps."""
    meta = probe(video_path)
    fx, fy, fw, fh = cfg.feed_region.as_pixels(meta.width, meta.height)

    raw_hits: list[float] = []
    tpl_h, tpl_w = elim_x_tpl.shape[:2]

    for t, frame in iter_frames(video_path, cfg.sample_fps):
        roi = crop(frame, fx, fy, fw, fh)
        if roi.size == 0:
            continue

        # cheap pre-filter: require some red in the ROI
        mask = red_mask(roi)
        if cv2.countNonZero(mask) < tpl_h * tpl_w // 4:
            continue

        score, (mx, _my) = match_score(roi, elim_x_tpl)
        if score < cfg.match_thresholds.elim_x:
            continue

        if player_icon_tpl is not None:
            # player icon sits just LEFT of the X marker in the same row
            pi_h, pi_w = player_icon_tpl.shape[:2]
            left_x0 = max(0, mx - pi_w - 4)
            left_x1 = mx + 4
            row_y0 = 0
            row_y1 = roi.shape[0]
            left_roi = roi[row_y0:row_y1, left_x0:left_x1]
            p_score, _ = match_score(left_roi, player_icon_tpl)
            if p_score < cfg.match_thresholds.player_icon:
                continue

        raw_hits.append(t)

    return _dedupe(video_path, raw_hits, cfg.dedupe_window_s)


def _dedupe(video: Path, hits: list[float], window_s: float) -> list[KillEvent]:
    hits.sort()
    out: list[KillEvent] = []
    last = -1e9
    for t in hits:
        if t - last < window_s:
            continue
        out.append(KillEvent(video=video, t=t))
        last = t
    return out


def load_template(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"template not found or unreadable: {path}")
    return img
