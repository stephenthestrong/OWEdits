from pathlib import Path

from owedits import config as cfg_mod
from owedits.events import group
from owedits.feed_detect import KillEvent


BASE_CFG = {
    "input_dir": "./in",
    "output_dir": "./out",
    "templates_dir": "./templates",
    "pre_roll": 10,
    "post_roll": 5,
    "feed_region": {"x": 0.7, "y": 0.08, "w": 0.29, "h": 0.25},
    "sample_fps": 4,
    "dedupe_window_s": 2.0,
    "multikill": {"min_kills": 2, "window_s": 4.0},
    "team_wipe": {"min_kills": 4, "window_s": 8.0},
    "clip_method": "copy",
}


def _cfg():
    return cfg_mod.from_dict(BASE_CFG, base_dir=Path("/tmp"))


def _kills(video: str, times: list[float]) -> list[KillEvent]:
    return [KillEvent(video=Path(video), t=t) for t in times]


def test_empty_list_returns_no_events():
    assert group([], _cfg(), video_duration_s=60) == []


def test_single_kill_is_not_an_event():
    kills = _kills("a.mp4", [30.0])
    assert group(kills, _cfg(), video_duration_s=120) == []


def test_multikill_detected():
    kills = _kills("a.mp4", [30.0, 32.0])  # 2 kills in 2s
    evs = group(kills, _cfg(), video_duration_s=120)
    assert len(evs) == 1
    e = evs[0]
    assert e.kind == "multikill"
    assert e.kill_count == 2
    assert e.first_kill_t == 30.0
    assert e.last_kill_t == 32.0
    assert e.start_s == 20.0  # 30 - pre_roll
    assert e.end_s == 37.0    # 32 + post_roll


def test_burst_exceeding_window_splits_into_multiple_multikills():
    # 4 kills in 4.5s: first 3 fit in the 4s window, the 4th is outside.
    # Only multikills are emitted now; the 4th kill is left ungrouped.
    kills = _kills("a.mp4", [10.0, 11.5, 13.0, 14.5])
    evs = group(kills, _cfg(), video_duration_s=60)
    assert len(evs) == 1
    assert evs[0].kind == "multikill"
    assert evs[0].kill_count == 3


def test_pair_outside_window_is_ignored():
    # 5s between kills exceeds multikill.window_s (4.0)
    kills = _kills("a.mp4", [10.0, 15.5])
    evs = group(kills, _cfg(), video_duration_s=60)
    assert evs == []


def test_two_separate_multikills():
    kills = _kills("a.mp4", [10.0, 12.0, 60.0, 62.0])
    evs = group(kills, _cfg(), video_duration_s=120)
    assert len(evs) == 2
    assert all(e.kind == "multikill" and e.kill_count == 2 for e in evs)
    assert evs[0].first_kill_t == 10.0
    assert evs[1].first_kill_t == 60.0


def test_clip_bounds_clamped_to_video_duration():
    kills = _kills("a.mp4", [1.0, 2.0])  # near start
    evs = group(kills, _cfg(), video_duration_s=5.0)  # tiny video
    assert len(evs) == 1
    e = evs[0]
    assert e.start_s == 0.0     # clamped (cannot go below 0)
    assert e.end_s == 5.0       # clamped (cannot exceed duration)
