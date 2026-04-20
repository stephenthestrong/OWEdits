from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

from .config import Config
from .video import crop, iter_frames, probe


# Circular ability icon at 1080p is roughly 40-45px diameter.
ICON_PATCH = (44, 44)

# Kill-feed rows appear below the team scoreboard; skip the top 25% of the ROI.
KF_ZONE_FRAC = 0.25

# Hough circle parameters tuned for 1080p ability icons.
_HOUGH_PARAMS = dict(
    method=cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=25,
    param1=60,
    param2=25,
    minRadius=14,
    maxRadius=30,
)

# Larger phash threshold accounts for animated/spinning icons across frames.
_PHASH_DIST = 20


@dataclass
class ExtractResult:
    elim_x_path: Path
    player_icon_path: Path | None
    kill_rows_seen: int
    icon_cluster_size: int


def _find_ability_icons(roi: np.ndarray) -> list[tuple[int, int]]:
    """Detect circular ability icons in the kill-feed zone. Returns (cx, cy) in ROI coords."""
    kf_y = int(roi.shape[0] * KF_ZONE_FRAC)
    zone = roi[kf_y:, :]
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, **_HOUGH_PARAMS)
    if circles is None:
        return []
    return [(int(x), int(y) + kf_y) for x, y, _r in np.round(circles[0]).astype(int)]


def _crop_centered(img: np.ndarray, cx: int, cy: int, ph: int, pw: int) -> np.ndarray | None:
    h, w = img.shape[:2]
    x0, y0 = cx - pw // 2, cy - ph // 2
    if x0 < 0 or y0 < 0 or x0 + pw > w or y0 + ph > h:
        return None
    return img[y0:y0 + ph, x0:x0 + pw].copy()


def extract(sample_video: Path, cfg: Config) -> ExtractResult:
    """Scan a sample video, find the player's circular ability icon in the kill feed, and save:

    - templates_dir/elim-x.png : sharpest icon crop from the largest phash cluster
                                  (the player's most-used hero ability icon)

    In OW2 the kill-feed row is: [killer name] [circular ability icon] > [victim portrait] [victim name].
    The circular icon uniquely identifies the killer's hero, so it doubles as the player filter —
    no separate player-icon.png is needed.
    """
    meta = probe(sample_video)
    fx, fy, fw, fh = cfg.feed_region.as_pixels(meta.width, meta.height)

    ph, pw = ICON_PATCH
    icon_crops: list[np.ndarray] = []
    icon_hashes: list[imagehash.ImageHash] = []
    kill_rows = 0

    for _t, frame in iter_frames(sample_video, max(cfg.sample_fps, 2.0)):
        roi = crop(frame, fx, fy, fw, fh)
        if roi.size == 0:
            continue

        for cx, cy in _find_ability_icons(roi):
            patch = _crop_centered(roi, cx, cy, ph, pw)
            if patch is None:
                continue
            kill_rows += 1
            icon_crops.append(patch)
            icon_hashes.append(
                imagehash.phash(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)))
            )

    if not icon_crops:
        raise RuntimeError(
            "No ability icons found in the sample. "
            "Try a longer sample video or adjust feed_region in your config."
        )

    # Cluster by phash. Larger threshold (_PHASH_DIST=20) handles animated/spinning icons.
    clusters: list[list[int]] = []
    for i, hi in enumerate(icon_hashes):
        for cl in clusters:
            if hi - icon_hashes[cl[0]] <= _PHASH_DIST:
                cl.append(i)
                break
        else:
            clusters.append([i])

    biggest = max(clusters, key=len)
    best_icon = max(
        (icon_crops[i] for i in biggest),
        key=lambda im: cv2.Laplacian(im, cv2.CV_64F).var(),
    )

    cfg.templates_dir.mkdir(parents=True, exist_ok=True)
    elim_x_path = cfg.templates_dir / "elim-x.png"
    cv2.imwrite(str(elim_x_path), best_icon)

    # Remove stale player-icon.png from any previous extraction (it is no longer used).
    stale = cfg.templates_dir / "player-icon.png"
    if stale.exists():
        stale.unlink()

    return ExtractResult(
        elim_x_path=elim_x_path,
        player_icon_path=None,
        kill_rows_seen=kill_rows,
        icon_cluster_size=len(biggest),
    )
