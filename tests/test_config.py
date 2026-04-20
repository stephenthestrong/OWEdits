from pathlib import Path

import pytest

from owedits import config as cfg_mod


SAMPLE = {
    "input_dir": "./in",
    "output_dir": "./out",
    "templates_dir": "./templates",
    "pre_roll": 8,
    "post_roll": 4,
    "feed_region": {"x": 0.7, "y": 0.08, "w": 0.29, "h": 0.25},
    "sample_fps": 4,
    "match_thresholds": {"elim_x": 0.7, "player_icon": 0.6},
    "dedupe_window_s": 2.0,
    "multikill": {"min_kills": 2, "window_s": 4.0},
    "team_wipe": {"min_kills": 4, "window_s": 8.0},
    "clip_method": "copy",
    "montage": {"enabled": True, "filename": "highlights.mp4"},
}


def test_from_dict_defaults(tmp_path):
    cfg = cfg_mod.from_dict(SAMPLE, base_dir=tmp_path)
    assert cfg.pre_roll == 8
    assert cfg.multikill.min_kills == 2
    assert cfg.team_wipe.window_s == 8.0
    assert cfg.input_dir == (tmp_path / "in").resolve()
    assert cfg.clip_method == "copy"


def test_feed_region_pixels():
    fr = cfg_mod.FeedRegion(0.5, 0.1, 0.4, 0.2)
    assert fr.as_pixels(1000, 500) == (500, 50, 400, 100)


def test_rejects_bad_clip_method():
    bad = dict(SAMPLE, clip_method="nope")
    with pytest.raises(ValueError):
        cfg_mod.from_dict(bad, base_dir=Path("/tmp"))
