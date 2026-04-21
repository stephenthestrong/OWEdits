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
class ColorPatternCfg:
    """Knobs for the blue-left / red-right my-team-kill detector."""
    row_height_frac: float = 0.039      # kill-feed row height as fraction of frame height
    blue_h_low: int = 85                # OW UI blue hue range (OpenCV 0-180 scale)
    blue_h_high: int = 100
    red_h_low_high: int = 10            # red covers 0..low_high
    red_h_high_low: int = 170           # ...and high_low..180
    color_s_min: int = 130              # saturation floor (rejects sky ~S=110)
    color_v_min: int = 80               # value floor (rejects shadows)
    min_color_px_frac: float = 0.01     # each of blue/red must cover ≥ frac of row
    min_center_sep_frac: float = 0.10   # blue x-center must be left of red by ≥ frac of row width
    row_shift_thresh: float = 12.0      # prev-top-in-current-second must match within this
    row_dedupe_s: float = 0.6           # suppress repeat fires during slide-in animation


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
    color_pattern: ColorPatternCfg = field(default_factory=ColorPatternCfg)


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
    cp = data.get("color_pattern", {})

    clip_method = data.get("clip_method", "copy")
    if clip_method not in ("copy", "reencode"):
        raise ValueError(f"clip_method must be 'copy' or 'reencode', got {clip_method!r}")

    cp_defaults = ColorPatternCfg()
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
        color_pattern=ColorPatternCfg(
            row_height_frac=float(cp.get("row_height_frac", cp_defaults.row_height_frac)),
            blue_h_low=int(cp.get("blue_h_low", cp_defaults.blue_h_low)),
            blue_h_high=int(cp.get("blue_h_high", cp_defaults.blue_h_high)),
            red_h_low_high=int(cp.get("red_h_low_high", cp_defaults.red_h_low_high)),
            red_h_high_low=int(cp.get("red_h_high_low", cp_defaults.red_h_high_low)),
            color_s_min=int(cp.get("color_s_min", cp_defaults.color_s_min)),
            color_v_min=int(cp.get("color_v_min", cp_defaults.color_v_min)),
            min_color_px_frac=float(cp.get("min_color_px_frac", cp_defaults.min_color_px_frac)),
            min_center_sep_frac=float(cp.get("min_center_sep_frac", cp_defaults.min_center_sep_frac)),
            row_shift_thresh=float(cp.get("row_shift_thresh", cp_defaults.row_shift_thresh)),
            row_dedupe_s=float(cp.get("row_dedupe_s", cp_defaults.row_dedupe_s)),
        ),
    )


def to_dict(cfg: Config) -> dict:
    """Convert a Config back to a plain-dict form suitable for yaml.safe_dump."""
    return {
        "input_dir": str(cfg.input_dir),
        "output_dir": str(cfg.output_dir),
        "templates_dir": str(cfg.templates_dir),
        "pre_roll": cfg.pre_roll,
        "post_roll": cfg.post_roll,
        "feed_region": {
            "x": cfg.feed_region.x,
            "y": cfg.feed_region.y,
            "w": cfg.feed_region.w,
            "h": cfg.feed_region.h,
        },
        "sample_fps": cfg.sample_fps,
        "match_thresholds": {
            "elim_x": cfg.match_thresholds.elim_x,
            "player_icon": cfg.match_thresholds.player_icon,
        },
        "dedupe_window_s": cfg.dedupe_window_s,
        "multikill": {"min_kills": cfg.multikill.min_kills, "window_s": cfg.multikill.window_s},
        "team_wipe": {"min_kills": cfg.team_wipe.min_kills, "window_s": cfg.team_wipe.window_s},
        "clip_method": cfg.clip_method,
        "montage": {"enabled": cfg.montage.enabled, "filename": cfg.montage.filename},
        "color_pattern": {
            "row_height_frac": cfg.color_pattern.row_height_frac,
            "blue_h_low": cfg.color_pattern.blue_h_low,
            "blue_h_high": cfg.color_pattern.blue_h_high,
            "red_h_low_high": cfg.color_pattern.red_h_low_high,
            "red_h_high_low": cfg.color_pattern.red_h_high_low,
            "color_s_min": cfg.color_pattern.color_s_min,
            "color_v_min": cfg.color_pattern.color_v_min,
            "min_color_px_frac": cfg.color_pattern.min_color_px_frac,
            "min_center_sep_frac": cfg.color_pattern.min_center_sep_frac,
            "row_shift_thresh": cfg.color_pattern.row_shift_thresh,
            "row_dedupe_s": cfg.color_pattern.row_dedupe_s,
        },
    }


def save(cfg: Config, path: str | Path) -> None:
    """Serialize Config to YAML at the given path."""
    Path(path).write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False, indent=2))
