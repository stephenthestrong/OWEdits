# OWEdits

**Auto-clip your Overwatch multikills from a folder of gameplay recordings.**

Drop your SteelSeries Moments (or any other) recordings into the app, click **Clip**, and get back a folder of individually-cut multikill highlights plus a concatenated `highlights.mp4` reel. No manual scrubbing.

---

## What it does

OWEdits watches the top-right **kill-feed** in your Overwatch gameplay recordings and detects every my-team kill. When 2+ kills happen within an 8-second window, it cuts that moment out as a clip.

The detector is **player-agnostic** — it fires on any kill your team gets, not just yours. If you or a teammate kills an enemy, it counts.

## How it does it

Overwatch color-codes its kill-feed: **your team's name/portrait is blue-teal on the left, the enemy's is red on the right.** A my-team kill therefore produces a row with a specific color pattern. OWEdits exploits this with a two-stage detector:

1. **Color-pattern gate** — For the top row of the kill-feed ROI, compute the weighted x-centers of saturated blue and red pixel clusters. If the blue cluster sits meaningfully left of the red cluster, it's a my-team kill. This rejects enemy kills (red-left / blue-right), empty frames, and non-kill UI noise like the objective bar.

2. **Row-shift gate** — A matching row is only a *new* kill when either the previous sample wasn't a my-team row (transition from empty/enemy) or the previous top row has slid down into the current second-row slot (a new row was inserted above). This prevents the same kill from being counted multiple times during its 4-second on-screen lifetime.

Detected timestamps are grouped into multikills (configurable: default is 2+ kills in 8 seconds), each group becomes a clip via an `ffmpeg` keyframe-snap cut, and optionally every clip is concatenated into a single highlights reel.

### Why this detector

Earlier iterations used template matching on the red elimination "X" and a per-player hero icon, which broke across POVs and game patches. A threshold-only fingerprint diff on the kill-feed region then over-fired because the top strip's content shifts continuously during slide-in animation and when other HUD elements (like the top-center objective progress bar with its own blue-left / red-right pattern) leak into the ROI. The color-gated row-shift approach solves all three: it's tied to OW2's permanent UI language, it's specific to my-team kills, and it only fires when the feed actually inserts a new row.

---

## Install

### Option A — prebuilt Windows exe (recommended)

[Download `OWEdits.exe`](#) (127 MB, bundles `ffmpeg`). Double-click to launch. No Python, no external deps.

Config is read from / written to `config.yaml` next to the exe. First launch writes sensible defaults if none exists.

### Option B — from source

Requires **Python 3.10+** and **ffmpeg** on `PATH` (or let the build step bundle it).

```powershell
winget install Gyan.FFmpeg   # if you don't already have ffmpeg

git clone <this repo>
cd OWEdits
pip install -e ".[dev]"
```

Run the GUI:
```powershell
python launch_gui.py
```

Or the CLI:
```powershell
python -m owedits.cli clip --config config.yaml
```

Build the single-file exe yourself:
```powershell
python build_exe.py            # produces dist/OWEdits.exe
python build_exe.py --onedir   # folder build, faster cold start
python build_exe.py --no-ffmpeg  # don't bundle ffmpeg (user must have it on PATH)
```

---

## GUI

The GUI is the primary interface. Four tabs:

- **Main** — Drag & drop videos into the queue (or set an Input folder). Tweak pre/post-roll, multikill grouping, montage toggle. Hit **▶ Clip videos**. Progress bar shows per-video progress; log shows what's being cut.
- **Feed Region** — Live preview of where the detector is looking. Load a frame from your first video and tweak `x/y/w/h` with the overlay updating in real time.
- **Color Detection** — Mostly leave alone. Behind a "Show advanced" toggle: hue/saturation/value ranges for the blue + red detection, plus pattern strictness. Hover any label for a plain-English tooltip.
- **Clips** — Thumbnail grid of every clip already in your output folder. Click any card to play it.

Drag-and-drop: drop video files (`.mp4` / `.mkv` / `.mov` / `.avi`) or entire folders from Explorer onto the queue list. Folders are expanded recursively.

## CLI

```
python -m owedits.cli scan   --config config.yaml  # list detections, no files cut
python -m owedits.cli clip   --config config.yaml  # scan + cut + montage
python -m owedits.cli montage --config config.yaml # concat whatever's in output/
```

## Config

A gitignored `config.yaml` lives next to the app (or at the repo root in dev). All of these knobs are also exposed in the GUI.

| Key | What it does | Default |
|---|---|---|
| `input_dir` / `output_dir` | Source recordings / where clips go | `./input` / `./output` |
| `pre_roll` / `post_roll` | Seconds of footage before first / after last kill in a group | 8 / 4 |
| `sample_fps` | How often the detector samples each video (fps) | 4 |
| `clip_method` | `copy` (fast, keyframe-snap) or `reencode` (exact, slower) | `copy` |
| `multikill.min_kills` / `window_s` | Minimum kills in a window to produce a clip | 2 / 8 |
| `montage.enabled` / `filename` | Build `highlights.mp4` after clipping | `true` / `highlights.mp4` |
| `feed_region` | Kill-feed ROI as fractions of the frame (x, y, w, h) | `0.72, 0.01, 0.27, 0.32` |
| `color_pattern.blue_h_low` / `_high` | OpenCV hue range that counts as team-blue (0-180 scale) | `85` / `100` |
| `color_pattern.red_h_low_high` / `_high_low` | Red wraps — these bracket the two red bands | `10` / `170` |
| `color_pattern.color_s_min` / `color_v_min` | Saturation / value floors (filters out sky, shadows) | `130` / `80` |
| `color_pattern.min_color_px_frac` | Each of blue/red must cover at least this fraction of the row | `0.01` |
| `color_pattern.min_center_sep_frac` | Blue center must be left of red center by at least this fraction of row width | `0.10` |
| `color_pattern.row_shift_thresh` | How closely the prev top must match the current second row to count as insertion | `12.0` |
| `color_pattern.row_dedupe_s` | Minimum gap between fires | `0.6` |

## Troubleshooting

- **"No kills detected" on clips that clearly have multikills**
  Most likely the `feed_region` is wrong for your resolution or HUD scale. Open the **Feed Region** tab, load a preview, and adjust x/y/w/h until the green box hugs the kill-feed rows without overlapping the top-center objective bar.
- **False fires on non-kill frames** (e.g. clips that are just gameplay)
  Usually the ROI is including scene content (sky, rooftops) that matches the blue-left/red-right pattern. Tighten the region, or raise `color_s_min` — scene colors are less saturated than OW's UI.
- **Colorblind team colors** (CVD modes 1/2) don't detect
  OW2's colorblind options swap blue/red for yellow/magenta or similar. Edit the hue ranges on the Color Detection tab to match your CVD palette.
- **Clips are cut a second off from the real kill**
  `clip_method: copy` snaps to the nearest keyframe for speed. Switch to `reencode` for exact cuts at the cost of a slower run.

## Project status

The template-based detector (`KillFeedDetector`, `extract_templates`) is retained but dormant — the color-gated row-shift path (`RowCountDetector`) is the default. `team_wipe` grouping was removed in favor of a single configurable multikill window.

## License

Personal tool, no license declared. If you want to reuse anything, ask.
