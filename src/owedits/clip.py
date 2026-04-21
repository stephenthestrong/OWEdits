from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import Config
from .events import HighlightEvent


class FFmpegMissingError(RuntimeError):
    pass


def _require_ffmpeg() -> str:
    """Resolve the ffmpeg binary.

    Priority:
      1. PyInstaller bundle (sys._MEIPASS/ffmpeg[.exe]) — bundled alongside the exe.
      2. Next to the frozen executable (sys.executable/../ffmpeg[.exe]).
      3. `shutil.which('ffmpeg')` — the user's PATH.
    """
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / name)
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).parent / name)
    for c in candidates:
        if c.exists():
            return str(c)
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegMissingError(
            "ffmpeg not found. Install it (e.g. `winget install Gyan.FFmpeg`) or bundle "
            "it next to the exe."
        )
    return path


def clip_name(ev: HighlightEvent) -> str:
    stem = ev.video.stem
    mm = int(ev.first_kill_t // 60)
    ss = int(ev.first_kill_t % 60)
    return f"{stem}__{ev.kind}_{ev.kill_count}x_{mm:02d}-{ss:02d}.mp4"


def cut(ev: HighlightEvent, out_dir: Path, method: str) -> Path:
    ffmpeg = _require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / clip_name(ev)
    duration = max(0.1, ev.end_s - ev.start_s)

    if method == "copy":
        # Fast path: -ss before -i seeks via demuxer, and -c copy avoids re-encoding.
        # Cuts snap to the nearest keyframe, which is what we want for speed.
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{ev.start_s:.3f}",
            "-i", str(ev.video),
            "-t", f"{duration:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(out),
        ]
    elif method == "reencode":
        # Exact cuts: -ss after -i decodes up to the keyframe then trims.
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(ev.video),
            "-ss", f"{ev.start_s:.3f}",
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k",
            str(out),
        ]
    else:
        raise ValueError(f"unknown clip method: {method}")

    subprocess.run(cmd, check=True)
    return out


def montage(clips: list[Path], out_path: Path, reencode_fallback: bool = True) -> Path:
    """Concatenate clips into a single reel.

    First attempts the fast concat-demuxer path (-c copy). If that fails
    (typically because codecs/timebase differ between clips) and reencode_fallback
    is set, falls back to a filter-graph concat with re-encoding.
    """
    ffmpeg = _require_ffmpeg()
    if not clips:
        raise ValueError("no clips to concatenate")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    list_file = out_path.with_suffix(".concat.txt")
    list_file.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in clips) + "\n"
    )

    try:
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        return out_path
    except subprocess.CalledProcessError:
        if not reencode_fallback:
            raise

    # Re-encode fallback
    inputs: list[str] = []
    for p in clips:
        inputs += ["-i", str(p)]
    n = len(clips)
    filter_parts = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(n))
    filter_complex = f"{filter_parts}concat=n={n}:v=1:a=1[outv][outa]"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path
