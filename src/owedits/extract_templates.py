from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

from .config import Config
from .feed_detect import red_mask
from .video import crop, iter_frames, probe


# Tuned for 1920x1080; ROI already scales with the frame so these stay fixed.
ELIM_X_PATCH = (24, 24)  # (h, w)
ICON_PATCH = (40, 40)    # (h, w)


@dataclass
class ExtractResult:
    elim_x_path: Path
    player_icon_path: Path
    kill_rows_seen: int
    icon_cluster_size: int


def _largest_red_blob(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return bounding box (x, y, w, h) of the biggest red blob, or None."""
    num, _labels, stats, _cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None
    # stats[0] is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(areas)) + 1
    x, y, w, h, area = stats[k]
    if area < 30:  # too small to be a real marker
        return None
    return int(x), int(y), int(w), int(h)


def _crop_centered(img: np.ndarray, cx: int, cy: int, ph: int, pw: int) -> np.ndarray | None:
    h, w = img.shape[:2]
    x0 = cx - pw // 2
    y0 = cy - ph // 2
    if x0 < 0 or y0 < 0 or x0 + pw > w or y0 + ph > h:
        return None
    return img[y0 : y0 + ph, x0 : x0 + pw].copy()


def extract(sample_video: Path, cfg: Config) -> ExtractResult:
    """Scan a sample video, find red elim markers in the kill feed, and save:

    - templates_dir/elim-x.png : cleanest red-X crop found
    - templates_dir/player-icon.png : most frequent icon crop left of the X
                                      (assumed to be the player's most-played hero)
    """
    meta = probe(sample_video)
    fx, fy, fw, fh = cfg.feed_region.as_pixels(meta.width, meta.height)

    best_x_crop: np.ndarray | None = None
    best_x_score = -1.0
    icon_crops: list[np.ndarray] = []
    icon_hashes: list[imagehash.ImageHash] = []
    kill_rows = 0

    ph_x, pw_x = ELIM_X_PATCH
    ph_i, pw_i = ICON_PATCH

    for _t, frame in iter_frames(sample_video, max(cfg.sample_fps, 2.0)):
        roi = crop(frame, fx, fy, fw, fh)
        if roi.size == 0:
            continue
        mask = red_mask(roi)

        # clean noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        bbox = _largest_red_blob(mask)
        if bbox is None:
            continue

        bx, by, bw, bh = bbox
        # OW's X is ~square and ~22-30 px on a 1080p feed. Filter out long red bars.
        if not (10 <= bw <= 60 and 10 <= bh <= 60):
            continue
        aspect = bw / max(bh, 1)
        if not (0.6 <= aspect <= 1.6):
            continue

        kill_rows += 1

        cx_x = bx + bw // 2
        cy_y = by + bh // 2

        x_crop = _crop_centered(roi, cx_x, cy_y, ph_x, pw_x)
        if x_crop is not None:
            # score = how saturated-red this crop is; keep the best
            score = float(cv2.countNonZero(red_mask(x_crop)))
            if score > best_x_score:
                best_x_score = score
                best_x_crop = x_crop

        icon_cx = cx_x - (pw_x // 2 + pw_i // 2 + 2)
        icon_cy = cy_y
        icon = _crop_centered(roi, icon_cx, icon_cy, ph_i, pw_i)
        if icon is not None:
            icon_crops.append(icon)
            icon_hashes.append(imagehash.phash(Image.fromarray(cv2.cvtColor(icon, cv2.COLOR_BGR2RGB))))

    if best_x_crop is None:
        raise RuntimeError(
            "no kill-feed rows found in the sample. Try a longer sample video or "
            "adjust feed_region in your config."
        )

    # Cluster icons by phash Hamming distance <= 6; biggest cluster wins
    # because the user's most-played hero shows up in the kill-feed most often.
    clusters: list[list[int]] = []
    for i, hi in enumerate(icon_hashes):
        for cl in clusters:
            if hi - icon_hashes[cl[0]] <= 6:
                cl.append(i)
                break
        else:
            clusters.append([i])

    if not clusters:
        raise RuntimeError("found kill rows but could not crop any icons.")

    biggest = max(clusters, key=len)
    # pick median-sharpest crop from the cluster
    best_icon = max(
        (icon_crops[i] for i in biggest),
        key=lambda im: cv2.Laplacian(im, cv2.CV_64F).var(),
    )

    cfg.templates_dir.mkdir(parents=True, exist_ok=True)
    elim_x_path = cfg.templates_dir / "elim-x.png"
    player_icon_path = cfg.templates_dir / "player-icon.png"
    cv2.imwrite(str(elim_x_path), best_x_crop)
    cv2.imwrite(str(player_icon_path), best_icon)

    return ExtractResult(
        elim_x_path=elim_x_path,
        player_icon_path=player_icon_path,
        kill_rows_seen=kill_rows,
        icon_cluster_size=len(biggest),
    )
