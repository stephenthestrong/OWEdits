"""Build script for the OWEdits GUI executable.

Usage:
    python build_exe.py           # single-file exe in dist/OWEdits.exe
    python build_exe.py --onedir  # onedir build in dist/OWEdits/ (faster start)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENTRY = ROOT / "launch_gui.py"

# Candidate locations to bundle ffmpeg.exe from (first hit wins).
FFMPEG_CANDIDATES = [
    Path("C:/ffmpeg/bin/ffmpeg.exe"),
    Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
    Path(Path.home() / "scoop/shims/ffmpeg.exe"),
]


def _locate_ffmpeg(cli_override: str | None) -> Path | None:
    if cli_override:
        p = Path(cli_override)
        return p if p.exists() else None
    for c in FFMPEG_CANDIDATES:
        if c.exists():
            return c
    from shutil import which

    w = which("ffmpeg")
    return Path(w) if w else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onedir",
        action="store_true",
        help="Produce a folder rather than a single file (faster cold-start).",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Remove build/ and dist/ first."
    )
    parser.add_argument(
        "--ffmpeg",
        metavar="PATH",
        help="Path to ffmpeg.exe to bundle. If omitted, auto-detects from common locations.",
    )
    parser.add_argument(
        "--no-ffmpeg",
        action="store_true",
        help="Don't bundle ffmpeg (user must have it on PATH).",
    )
    args = parser.parse_args()

    if args.clean:
        for d in ("build", "dist"):
            p = ROOT / d
            if p.exists():
                shutil.rmtree(p)

    mode = "--onedir" if args.onedir else "--onefile"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        mode,
        "--windowed",
        "--name",
        "OWEdits",
        "--paths",
        str(ROOT / "src"),
        "--collect-submodules",
        "owedits",
        "--collect-submodules",
        "PIL",
        # tkinterdnd2 ships TCL extensions + platform binaries that PyInstaller
        # won't find on its own — --collect-all grabs the whole package tree.
        "--collect-all",
        "tkinterdnd2",
    ]

    if not args.no_ffmpeg:
        ffmpeg = _locate_ffmpeg(args.ffmpeg)
        if ffmpeg is None:
            print("WARNING: ffmpeg.exe not found. Building without bundled ffmpeg "
                  "(user will need it on PATH). Pass --ffmpeg PATH to override, "
                  "or --no-ffmpeg to silence this warning.")
        else:
            # "<src>;." drops ffmpeg.exe into the root of the bundle (sys._MEIPASS).
            cmd += ["--add-binary", f"{ffmpeg};."]
            print(f"Bundling ffmpeg from: {ffmpeg}")

    cmd.append(str(ENTRY))
    print(">>", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
