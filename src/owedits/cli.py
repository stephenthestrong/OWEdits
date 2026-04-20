from __future__ import annotations

from pathlib import Path

import click
from tqdm import tqdm

from . import config as cfg_mod
from . import clip as clip_mod
from . import events as events_mod
from . import extract_templates as templates_mod
from . import feed_detect


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi"}


@click.group()
@click.version_option()
def main() -> None:
    """Auto-clip Overwatch multi-kills and team wipes."""


def _load_config(path: str) -> cfg_mod.Config:
    return cfg_mod.load(path)


def _iter_videos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def _load_templates(cfg: cfg_mod.Config):
    elim_x = feed_detect.load_template(cfg.templates_dir / "elim-x.png")
    player_icon_path = cfg.templates_dir / "player-icon.png"
    player_icon = feed_detect.load_template(player_icon_path) if player_icon_path.exists() else None
    return elim_x, player_icon


def _scan_one(
    video: Path,
    cfg: cfg_mod.Config,
    elim_x,
    player_icon,
) -> list[events_mod.HighlightEvent]:
    kills, meta = feed_detect.detect_kill_frames(video, cfg, elim_x, player_icon)
    return events_mod.group(kills, cfg, meta.duration_s)


@main.command("extract-templates")
@click.argument("sample", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--config", "config_path", default="config.yaml",
              type=click.Path(dir_okay=False), show_default=True,
              help="Path to config.yaml. Used to resolve templates_dir and feed_region.")
def extract_templates_cmd(sample: Path, config_path: str) -> None:
    """Scan SAMPLE.mp4 and save templates/elim-x.png and templates/player-icon.png."""
    cfg = _load_config(config_path)
    click.echo(f"Scanning {sample} for kill-feed rows ...")
    res = templates_mod.extract(sample, cfg)
    click.echo(f"  kill rows seen: {res.kill_rows_seen}")
    click.echo(f"  player-icon cluster size: {res.icon_cluster_size}")
    click.echo(f"  wrote {res.elim_x_path}")
    click.echo(f"  wrote {res.player_icon_path}")
    click.echo("Open both and spot-check they look right before running `owedits clip`.")


@main.command("scan")
@click.option("--config", "config_path", default="config.yaml",
              type=click.Path(exists=True, dir_okay=False), show_default=True)
def scan_cmd(config_path: str) -> None:
    """Scan every video in input_dir and print detected events without cutting."""
    cfg = _load_config(config_path)
    elim_x, player_icon = _load_templates(cfg)

    videos = _iter_videos(cfg.input_dir)
    if not videos:
        click.echo(f"No videos found in {cfg.input_dir}")
        return

    for v in tqdm(videos, desc="videos"):
        evs = _scan_one(v, cfg, elim_x, player_icon)
        for e in evs:
            click.echo(f"{v.name}  {e.kind:10s}  kills={e.kill_count}  "
                       f"t={e.first_kill_t:.1f}-{e.last_kill_t:.1f}s  "
                       f"clip={e.start_s:.1f}-{e.end_s:.1f}s")


@main.command("clip")
@click.option("--config", "config_path", default="config.yaml",
              type=click.Path(exists=True, dir_okay=False), show_default=True)
def clip_cmd(config_path: str) -> None:
    """Scan videos, cut per-event clips, and (if enabled) build the montage."""
    cfg = _load_config(config_path)
    elim_x, player_icon = _load_templates(cfg)

    videos = _iter_videos(cfg.input_dir)
    if not videos:
        click.echo(f"No videos found in {cfg.input_dir}")
        return

    all_clips: list[Path] = []
    for v in tqdm(videos, desc="videos"):
        evs = _scan_one(v, cfg, elim_x, player_icon)
        for e in evs:
            out = clip_mod.cut(e, cfg.output_dir, cfg.clip_method)
            all_clips.append(out)
            click.echo(f"  -> {out.name}")

    if cfg.montage.enabled and all_clips:
        mpath = cfg.output_dir / cfg.montage.filename
        clip_mod.montage(all_clips, mpath)
        click.echo(f"montage: {mpath}")


@main.command("montage")
@click.option("--config", "config_path", default="config.yaml",
              type=click.Path(exists=True, dir_okay=False), show_default=True)
def montage_cmd(config_path: str) -> None:
    """Concatenate every .mp4 already in output_dir (except the montage itself) into the reel."""
    cfg = _load_config(config_path)
    name = cfg.montage.filename
    clips = sorted(
        p for p in cfg.output_dir.glob("*.mp4")
        if p.name != name
    )
    if not clips:
        click.echo(f"No clips in {cfg.output_dir}")
        return
    out = clip_mod.montage(clips, cfg.output_dir / name)
    click.echo(f"montage: {out}")


if __name__ == "__main__":
    main()
