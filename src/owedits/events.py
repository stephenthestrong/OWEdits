from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import Config
from .feed_detect import KillEvent

EventKind = Literal["multikill", "team_wipe"]


@dataclass
class HighlightEvent:
    video: Path
    kind: EventKind
    kill_count: int
    first_kill_t: float
    last_kill_t: float
    start_s: float  # clamped clip start
    end_s: float    # clamped clip end


def group(kills: list[KillEvent], cfg: Config, video_duration_s: float) -> list[HighlightEvent]:
    """Group a single video's kill timestamps into multikill events.

    Rules:
      - Scan kills with a sliding window.
      - A burst qualifies as `multikill` if >= multikill.min_kills fit within multikill.window_s.
      - Each kill belongs to at most one event — advance past the window after emitting.
    """
    if not kills:
        return []
    kills = sorted(kills, key=lambda k: k.t)
    video = kills[0].video

    out: list[HighlightEvent] = []
    i = 0
    n = len(kills)
    while i < n:
        j_mk = _farthest_within(kills, i, cfg.multikill.window_s)
        if (j_mk - i + 1) >= cfg.multikill.min_kills:
            out.append(_make_event(video, kills, i, j_mk, "multikill", cfg, video_duration_s))
            i = j_mk + 1
            continue

        i += 1

    return out


def _farthest_within(kills: list[KillEvent], i: int, window_s: float) -> int:
    """Largest j >= i such that kills[j].t - kills[i].t <= window_s."""
    start_t = kills[i].t
    j = i
    while j + 1 < len(kills) and kills[j + 1].t - start_t <= window_s:
        j += 1
    return j


def _make_event(
    video: Path,
    kills: list[KillEvent],
    i: int,
    j: int,
    kind: EventKind,
    cfg: Config,
    video_duration_s: float,
) -> HighlightEvent:
    first_t = kills[i].t
    last_t = kills[j].t
    start = max(0.0, first_t - cfg.pre_roll)
    end = min(video_duration_s, last_t + cfg.post_roll) if video_duration_s > 0 else last_t + cfg.post_roll
    return HighlightEvent(
        video=video,
        kind=kind,
        kill_count=j - i + 1,
        first_kill_t=first_t,
        last_kill_t=last_t,
        start_s=start,
        end_s=end,
    )
