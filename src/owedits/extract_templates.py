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

# Minimum mean brightness of the inner region of a detected circle.
# Stats-screen rounded-card corners are dark inside; ability icons have content.
_MIN_INNER_MEAN = 70

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

# Reject clusters that appear in more than this fraction of sampled frames.
# Persistent UI elements (health-bar portraits) appear every frame; kill-feed
# icons appear only during kills (~5-30% of frames).
_MAX_FRAME_FREQ = 0.50


@dataclass
class ExtractResult:
    elim_x_path: Path
    player_icon_path: Path | None
    kill_rows_seen: int
    icon_cluster_size: int


def _find_ability_icons(roi: np.ndarray) -> list[tuple[int, int]]:
    """Detect circular ability icons anywhere in the ROI. Returns (cx, cy) in ROI coords."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, **_HOUGH_PARAMS)
    if circles is None:
        return []
    ph, pw = ICON_PATCH
    results = []
    for x, y, _r in np.round(circles[0]).astype(int):
        cx, cy = int(x), int(y)
        patch = _crop_centered(roi, cx, cy, ph, pw)
        if patch is None:
            continue
        # Reject stats-screen card corners: their interiors are near-black.
        inner = patch[ph // 4 : 3 * ph // 4, pw // 4 : 3 * pw // 4]
        if float(inner.mean()) < _MIN_INNER_MEAN:
            continue
        results.append((cx, cy))
    return results


def _crop_centered(img: np.ndarray, cx: int, cy: int, ph: int, pw: int) -> np.ndarray | None:
    h, w = img.shape[:2]
    x0, y0 = cx - pw // 2, cy - ph // 2
    if x0 < 0 or y0 < 0 or x0 + pw > w or y0 + ph > h:
        return None
    return img[y0:y0 + ph, x0:x0 + pw].copy()


def extract(sample_video: Path, cfg: Config) -> ExtractResult:
    """Scan a sample video, find the player's circular ability icon in the kill feed, and save:

    - templates_dir/elim-x.png : sharpest icon crop from the best phash cluster
                                  (the player's most-used hero ability icon)

    In OW2 the kill-feed row is: [killer name] [circular ability icon] > [victim portrait] [victim name].
    The circular icon uniquely identifies the killer's hero, so it doubles as the player filter.

    Persistent circular UI elements (e.g. health-bar hero portraits) are rejected because they
    appear in the majority of frames, whereas kill-feed icons only appear during kills.
    """
    meta = probe(sample_video)
    fx, fy, fw, fh = cfg.feed_region.as_pixels(meta.width, meta.height)

    ph, pw = ICON_PATCH
    icon_crops: list[np.ndarray] = []
    icon_hashes: list[imagehash.ImageHash] = []
    icon_frame_ts: list[float] = []   # which sample timestamp each patch came from
    kill_rows = 0
    all_ts: list[float] = []

    for t, frame in iter_frames(sample_video, max(cfg.sample_fps, 2.0)):
        roi = crop(frame, fx, fy, fw, fh)
        if roi.size == 0:
            continue
        all_ts.append(t)

        for cx, cy in _find_ability_icons(roi):
            patch = _crop_centered(roi, cx, cy, ph, pw)
            if patch is None:
                continue
            kill_rows += 1
            icon_crops.append(patch)
            icon_hashes.append(
                imagehash.phash(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)))
            )
            icon_frame_ts.append(t)

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

    # Reject clusters that appear in too many frames — those are persistent UI elements,
    # not transient kill-feed icons.
    total_frames = max(len(set(all_ts)), 1)
    valid_clusters = [
        cl for cl in clusters
        if len(set(icon_frame_ts[i] for i in cl)) / total_frames <= _MAX_FRAME_FREQ
    ]

    if not valid_clusters:
        # Fallback: use all clusters if the frequency filter is too aggressive
        valid_clusters = clusters

    best_cluster = max(valid_clusters, key=len)
    best_icon = max(
        (icon_crops[i] for i in best_cluster),
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
        icon_cluster_size=len(best_cluster),
    )
