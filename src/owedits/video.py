from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe(path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    try:
        return VideoMeta(
            path=Path(path),
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 0),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )
    finally:
        cap.release()


def iter_frames(
    path: str | Path, sample_fps: float
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp_s, BGR frame) sampled at roughly `sample_fps`.

    Uses frame-index stepping rather than CAP_PROP_POS_MSEC seeking so it stays
    fast on long recordings.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        if src_fps <= 0:
            raise RuntimeError(f"video reports fps={src_fps}: {path}")
        step = max(1, int(round(src_fps / max(sample_fps, 0.1))))

        idx = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield idx / src_fps, frame
            idx += step
    finally:
        cap.release()


def crop(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    fh, fw = frame.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(fw, x + w)
    y1 = min(fh, y + h)
    return frame[y0:y1, x0:x1]
