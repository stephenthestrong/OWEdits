# OWEdits — Project memory

Auto-clip good moments (multi-kills, team wipes) from a folder of Overwatch gameplay videos recorded by SteelSeries Moments.

## Stack
- Python 3.10+, `src/` layout, entry point `owedits` (click).
- CV: `opencv-python`, `numpy`, `Pillow`, `imagehash` (perceptual-hash clustering).
- Video cutting: external `ffmpeg` on PATH; `subprocess.run` wraps it.
- Tests: `pytest` (run with `python -m pytest -q`).

## Layout
```
src/owedits/
  cli.py               # click commands: extract-templates | scan | clip | montage
  config.py            # YAML -> Config dataclasses (FeedRegion, EventRule, Thresholds, ...)
  video.py             # probe() -> VideoMeta, iter_frames(), crop()
  feed_detect.py       # red_mask(), detect_kill_frames() -> (list[KillEvent], VideoMeta)
  extract_templates.py # extract() builds templates/elim-x.png + templates/player-icon.png
  events.py            # group() kills -> HighlightEvent(kind=multikill|team_wipe)
  clip.py              # cut() per event, montage() concatenation (copy with reencode fallback)
tests/                 # 15 tests, all passing
config.example.yaml    # user copies to config.yaml and edits
```

## Detection pipeline (how it works)
1. `iter_frames` opens the video once and alternates `cap.read()` / `cap.grab()` to sample at `sample_fps` without seeking — seeking per sample was ~10× slower on long recordings.
2. For each sampled frame, crop the kill-feed ROI (top-right; `feed_region` in config, stored as fractions so resolution-agnostic).
3. HSV red prefilter (`red_mask`) skips frames with no red in the ROI.
4. `cv2.matchTemplate` (TM_CCOEFF_NORMED) against `templates/elim-x.png`. Iterative NMS (`_nms_matches`) finds **all** peaks above threshold — one per kill-feed row — so stacked multi-kills in a single frame all count.
5. If `templates/player-icon.png` exists, template-match the region immediately **left** of the X (restricted to that row's y-slice) against the player icon — filters *your* kills from teammates'.
6. Dedupe per row-bucket within `dedupe_window_s` (`_dedupe_by_row`): same row seen again within the window is skipped; different rows at the same timestamp each produce a kill.
7. `events.group()` slides a window over sorted kills: ≥ `team_wipe.min_kills` in `team_wipe.window_s` → `team_wipe`; else ≥ `multikill.min_kills` in `multikill.window_s` → `multikill`. Team wipe takes precedence.
8. Each event becomes `[first_kill - pre_roll, last_kill + post_roll]`, clamped to video duration.

## Template auto-extraction
`owedits extract-templates SAMPLE.mp4` scans the sample, finds square red blobs (the X marker), crops the icon patch immediately left of each, `imagehash.phash`-clusters those crops by Hamming distance ≤ 6, and writes the biggest cluster's sharpest crop as `player-icon.png` (the user's most-played hero). Writes the cleanest red-X crop as `elim-x.png`.

## Clipping
- `clip_method: "copy"` — `ffmpeg -ss <start> -i <in> -t <dur> -c copy`. Fast, snaps to keyframes. Default.
- `clip_method: "reencode"` — decode + libx264/aac for exact cuts. Slower.
- Montage: concat demuxer (`-c copy`); falls back to re-encode with `-filter_complex concat` if codecs/timebase differ.

## Config (`config.yaml`)
All knobs live in `config.example.yaml`. The tune-first knobs:
- `feed_region` — if HUD isn't 16:9 or OW moved the feed, adjust.
- `match_thresholds.elim_x` / `player_icon` — lower for recall, higher for precision.
- `dedupe_window_s` — raise if single kills register twice.
- `pre_roll` / `post_roll` — clip window around detected event.

## CLI
```
owedits extract-templates SAMPLE.mp4 [--config config.yaml]
owedits scan   [--config config.yaml]
owedits clip   [--config config.yaml]     # scan + cut + montage
owedits montage [--config config.yaml]    # combine existing clips in output_dir
```

## Parallel scanning (v0.2)
`cli.py` uses `ProcessPoolExecutor` (workers = cpu_count, capped at video count) to scan videos in parallel. The module-level `_scan_worker` function must stay at module level for pickle compatibility on Windows (spawn start method).

## Git
- Develop on branch: `claude/overwatch-highlight-clipper-ij02Z` (pushed to `origin`).
- Initial commit `4bb71c7`; cleanup commit `007807e`; v0.2 commit adds multi-match NMS, parallel scanning, `Detector` protocol, AR warning.

## Detector interface (v0.2)
`feed_detect.Detector` is a `Protocol` with a single method:
```python
def detect(self, video_path: Path, cfg: Config, meta: VideoMeta) -> list[KillEvent]: ...
```
`KillFeedDetector` implements it. Add new event sources (POTG, ult, sound cues) by implementing `Detector` and appending to the `detectors` list built in `cli.py`.

## Known limitations
- No POTG / ult / sound-cue detection. Implement `Detector` and add to the list in `cli.py`'s scan/clip commands.
- `_warn_aspect_ratio` only probes the first video; mixed-AR input folders won't all be caught.

## Development commands
```bash
pip install -e ".[dev]"
python -m pytest -q
owedits --help
```
