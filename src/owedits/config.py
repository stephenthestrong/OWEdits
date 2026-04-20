from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class FeedRegion:
    x: float
    y: float
    w: float
    h: float

    def as_pixels(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        px = int(self.x * frame_w)
        py = int(self.y * frame_h)
        pw = int(self.w * frame_w)
        ph = int(self.h * frame_h)
        return px, py, pw, ph


@dataclass
class EventRule:
    min_kills: int
    window_s: float


@dataclass
class Thresholds:
    elim_x: float = 0.70
    player_icon: float = 0.65


@dataclass
class MontageCfg:
    enabled: bool = True
    filename: str = "highlights.mp4"


@dataclass
class Config:
    input_dir: Path
    output_dir: Path
    templates_dir: Path
    pre_roll: float
    post_roll: float
    feed_region: FeedRegion
    sample_fps: float
    match_thresholds: Thresholds
    dedupe_window_s: float
    multikill: EventRule
    team_wipe: EventRule
    clip_method: str
    montage: MontageCfg = field(default_factory=MontageCfg)


def load(path: str | Path) -> Config:
    data = yaml.safe_load(Path(path).read_text())
    return from_dict(data, base_dir=Path(path).parent)


def from_dict(data: dict, base_dir: Path | None = None) -> Config:
    base_dir = base_dir or Path.cwd()

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (base_dir / pp).resolve()

    fr = data["feed_region"]
    mk = data["multikill"]
    tw = data["team_wipe"]
    th = data.get("match_thresholds", {})
    mont = data.get("montage", {})

    clip_method = data.get("clip_method", "copy")
    if clip_method not in ("copy", "reencode"):
        raise ValueError(f"clip_method must be 'copy' or 'reencode', got {clip_method!r}")

    return Config(
        input_dir=_resolve(data["input_dir"]),
        output_dir=_resolve(data["output_dir"]),
        templates_dir=_resolve(data.get("templates_dir", "./templates")),
        pre_roll=float(data.get("pre_roll", 10)),
        post_roll=float(data.get("post_roll", 5)),
        feed_region=FeedRegion(float(fr["x"]), float(fr["y"]), float(fr["w"]), float(fr["h"])),
        sample_fps=float(data.get("sample_fps", 4)),
        match_thresholds=Thresholds(
            elim_x=float(th.get("elim_x", 0.70)),
            player_icon=float(th.get("player_icon", 0.65)),
        ),
        dedupe_window_s=float(data.get("dedupe_window_s", 2.0)),
        multikill=EventRule(int(mk["min_kills"]), float(mk["window_s"])),
        team_wipe=EventRule(int(tw["min_kills"]), float(tw["window_s"])),
        clip_method=clip_method,
        montage=MontageCfg(
            enabled=bool(mont.get("enabled", True)),
            filename=str(mont.get("filename", "highlights.mp4")),
        ),
    )
