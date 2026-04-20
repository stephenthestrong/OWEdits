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
4. `cv2.matchTemplate` (TM_CCOEFF_NORMED) against `templates/elim-x.png`. Above threshold → candidate kill.
5. If `templates/player-icon.png` exists, template-match the region immediately **left** of the X against the player icon — this filters *your* kills out from teammates' kills.
6. Dedupe candidates within `dedupe_window_s` (feed rows linger ~4s).
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

## Git
- Develop on branch: `claude/overwatch-highlight-clipper-ij02Z` (pushed to `origin`).
- Initial commit `4bb71c7`; cleanup commit `007807e` (sequential frame reads, drop redundant probe, remove dead code, extract `_load_templates`).

## Known limitations (v0.1, candidates for next iteration)
- Single-frame template match returns only the best match → three elim rows stacked in one frame count as one kill. True multi-kills within `dedupe_window_s` can undercount. Fix: find all matches per frame above threshold, dedupe by (y-bucket, time).
- Videos are scanned sequentially. `ProcessPoolExecutor` across videos would ~N-core speedup template matching.
- No POTG / ult / sound-cue detection. Add as alternative detectors behind the `KillEvent` / `HighlightEvent` interface.
- Detection assumes 16:9 gameplay with the kill-feed in the top-right.

## Development commands
```bash
pip install -e ".[dev]"
python -m pytest -q
owedits --help
```
