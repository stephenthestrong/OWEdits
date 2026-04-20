from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
from tqdm import tqdm

from . import config as cfg_mod
from . import clip as clip_mod
from . import events as events_mod
from . import extract_templates as templates_mod
from . import feed_detect


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi"}

_NATIVE_AR = 16 / 9


def _load_config(path: str) -> cfg_mod.Config:
    return cfg_mod.load(path)


def _iter_videos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def _load_templates(cfg: cfg_mod.Config):
    elim_x = feed_detect.load_template(cfg.templates_dir / "elim-x.png")
    player_icon_path = cfg.templates_dir / "player-icon.png"
    player_icon = feed_detect.load_template(player_icon_path) if player_icon_path.exists() else None
    return elim_x, player_icon


def _warn_aspect_ratio(video: Path) -> None:
    try:
        meta = feed_detect.probe(video)
    except Exception:
        return
    if abs(meta.aspect_ratio - _NATIVE_AR) > 0.05:
        click.echo(
            f"Warning: {video.name} is {meta.width}x{meta.height} "
            f"(AR {meta.aspect_ratio:.2f}, expected ~{_NATIVE_AR:.2f}). "
            "Default feed_region is calibrated for 16:9 — adjust in config.yaml if kills are missed.",
            err=True,
        )


# Module-level so ProcessPoolExecutor (spawn) can pickle it.
def _scan_worker(
    args: tuple[Path, cfg_mod.Config, list[feed_detect.KillFeedDetector]],
) -> list[events_mod.HighlightEvent]:
    video, cfg, detectors = args
    meta = feed_detect.probe(video)
    all_kills: list[feed_detect.KillEvent] = []
    for d in detectors:
        all_kills.extend(d.detect(video, cfg, meta))
    all_kills.sort(key=lambda k: k.t)
    return events_mod.group(all_kills, cfg, meta.duration_s)


@click.group()
@click.version_option()
def main() -> None:
    """Auto-clip Overwatch multi-kills and team wipes."""


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
    if res.player_icon_path:
        click.echo(f"  wrote {res.player_icon_path}")
    click.echo("Open elim-x.png and spot-check it looks like the circular ability icon before running `owedits clip`.")


@main.command("scan")
@click.option("--config", "config_path", default="config.yaml",
              type=click.Path(exists=True, dir_okay=False), show_default=True)
def scan_cmd(config_path: str) -> None:
    """Scan every video in input_dir and print detected events without cutting."""
    cfg = _load_config(config_path)
    elim_x, player_icon = _load_templates(cfg)
    detectors = [feed_detect.KillFeedDetector(elim_x, player_icon)]

    videos = _iter_videos(cfg.input_dir)
    if not videos:
        click.echo(f"No videos found in {cfg.input_dir}")
        return

    _warn_aspect_ratio(videos[0])

    n_workers = min(len(videos), max(1, os.cpu_count() or 1))
    args = [(v, cfg, detectors) for v in videos]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(tqdm(ex.map(_scan_worker, args), total=len(videos), desc="videos"))

    for v, evs in zip(videos, results):
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
    detectors = [feed_detect.KillFeedDetector(elim_x, player_icon)]

    videos = _iter_videos(cfg.input_dir)
    if not videos:
        click.echo(f"No videos found in {cfg.input_dir}")
        return

    _warn_aspect_ratio(videos[0])

    n_workers = min(len(videos), max(1, os.cpu_count() or 1))
    args = [(v, cfg, detectors) for v in videos]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(tqdm(ex.map(_scan_worker, args), total=len(videos), desc="scanning"))

    all_clips: list[Path] = []
    for evs in results:
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
