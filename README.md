# OWEdits

Auto-clip multi-kills and team wipes from a folder of Overwatch gameplay videos.

## How it works

1. Samples frames from each video and looks at the kill-feed region (top-right).
2. Template-matches the red elimination "X" and your player icon to detect *your* kills.
3. Groups nearby kills into **multikill** (2+ in 4s) and **team wipe** (4+ in 8s) events.
4. Cuts each event out with ffmpeg, then concatenates them into a `highlights.mp4` reel.

## Install (Windows)

```powershell
# Requires Python 3.10+ and ffmpeg on PATH
winget install Gyan.FFmpeg     # or download from https://ffmpeg.org

git clone <this repo>
cd OWEdits
pip install -e .
```

## First-time setup

Pick one of your recorded videos that has several of your kills in it and extract templates:

```powershell
owedits extract-templates "C:\Videos\Overwatch\sample.mp4"
```

This writes `templates/elim-x.png` (the red elim marker) and `templates/player-icon.png` (your most-played hero's feed icon, auto-clustered from the sample). Open them and eyeball that they look right.

## Run

```powershell
copy config.example.yaml config.yaml
# edit input_dir / output_dir

owedits clip --config config.yaml
```

This scans every `.mp4` in `input_dir`, writes per-event clips to `output_dir`, and (if enabled) builds `highlights.mp4`.

Use `owedits scan --dry-run` first to print detected events without cutting anything.

## CLI reference

```
owedits extract-templates SAMPLE.mp4 [--feed-region X,Y,W,H]
owedits scan   [--config config.yaml] [--dry-run]
owedits clip   [--config config.yaml]        # scan + cut
owedits montage [--config config.yaml]       # combine existing clips
```

## Tuning

If detection is off:

- Adjust `feed_region` if your HUD is at a non-16:9 resolution.
- Lower `match_thresholds.elim_x` / `player_icon` for more recall, raise for more precision.
- Increase `dedupe_window_s` if a single kill registers twice (feed rows linger a few seconds).
