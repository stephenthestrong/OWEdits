"""Tkinter GUI for OWEdits — expose every knob, run scan/clip from buttons."""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import (
    BooleanVar,
    DoubleVar,
    END,
    IntVar,
    Listbox,
    StringVar,
    filedialog,
    messagebox,
    ttk,
)
from tkinter.scrolledtext import ScrolledText
from typing import Callable

import cv2
from PIL import Image, ImageDraw, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD

from . import clip as clip_mod
from . import config as cfg_mod
from . import events as events_mod
from . import feed_detect


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi"}

# Visual tokens — one source of truth so restyles only touch this block.
# Dark theme inspired by SteelSeries Moments.
FONT_FAMILY = "Segoe UI"
FONT_BODY = (FONT_FAMILY, 10)
FONT_LABEL = (FONT_FAMILY, 10)
FONT_HINT = (FONT_FAMILY, 9)
FONT_HINT_CAPS = (FONT_FAMILY, 8, "bold")  # small-caps style metadata
FONT_SECTION = (FONT_FAMILY, 11, "bold")
FONT_TITLE = (FONT_FAMILY, 12, "bold")
FONT_MONO = ("Consolas", 9)

COLOR_BG = "#0f1419"              # window background (near-black with blue tint)
COLOR_SURFACE = "#1a2028"          # card / panel background
COLOR_SURFACE_2 = "#222b36"        # entry fields, nested cards
COLOR_BORDER = "#2a333f"           # subtle dividers + card borders
COLOR_TEXT = "#e4e6eb"             # primary text
COLOR_TEXT_DIM = "#8b95a1"         # secondary text, hints, metadata
COLOR_TEXT_MUTED = "#5a6573"       # tertiary (captions, timestamps)
COLOR_HINT = COLOR_TEXT_DIM        # legacy alias for existing callers
COLOR_ERROR_BG = "#4a1c1c"         # invalid-input highlight (red-tinted surface)
COLOR_ERROR_FG = "#ff6b6b"
COLOR_ACCENT = "#6366f1"           # primary button purple
COLOR_ACCENT_HOVER = "#818cf8"
COLOR_ACCENT_ACTIVE = "#4f46e5"
COLOR_DIVIDER_STRONG = "#3a4450"


class TaskCancelled(Exception):
    """Raised inside worker threads when the user hits Cancel."""


# ──────────────────────────────────────────────────────────────────────────
# Tooltip helper — tk-native, no external deps.
# ──────────────────────────────────────────────────────────────────────────

class Tooltip:
    """Hover tooltip that works with any tk widget."""

    def __init__(self, widget: tk.Widget, text: str, *, delay_ms: int = 400) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay_ms
        self._tip: tk.Toplevel | None = None
        self._after_id: str | None = None
        widget.bind("<Enter>", self._on_enter)
        widget.bind("<Leave>", self._on_leave)
        widget.bind("<ButtonPress>", self._on_leave)

    def _on_enter(self, _event: tk.Event) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _on_leave(self, _event: tk.Event) -> None:
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self) -> None:
        if self._tip is not None:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(
            tip,
            text=self.text,
            background=COLOR_SURFACE_2,
            foreground=COLOR_TEXT,
            relief="solid",
            borderwidth=1,
            highlightbackground=COLOR_BORDER,
            font=FONT_HINT,
            justify="left",
            wraplength=360,
        )
        lbl.pack(ipadx=8, ipady=5)
        self._tip = tip


# ──────────────────────────────────────────────────────────────────────────
# Config file resolution
# ──────────────────────────────────────────────────────────────────────────

def _app_dir() -> Path:
    """Directory that holds the config.yaml next to the exe (frozen) or repo root (dev)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    return _app_dir() / "config.yaml"


def _ensure_default_config() -> Path:
    """If config.yaml doesn't exist next to the app, write one with sensible defaults."""
    path = _default_config_path()
    if path.exists():
        return path
    base = _app_dir()
    cfg = cfg_mod.Config(
        input_dir=base / "input",
        output_dir=base / "output",
        templates_dir=base / "templates",
        pre_roll=8.0,
        post_roll=4.0,
        feed_region=cfg_mod.FeedRegion(x=0.72, y=0.01, w=0.27, h=0.32),
        sample_fps=4.0,
        match_thresholds=cfg_mod.Thresholds(),
        dedupe_window_s=2.0,
        multikill=cfg_mod.EventRule(min_kills=2, window_s=8.0),
        team_wipe=cfg_mod.EventRule(min_kills=4, window_s=8.0),
        clip_method="copy",
        montage=cfg_mod.MontageCfg(enabled=True, filename="highlights.mp4"),
        color_pattern=cfg_mod.ColorPatternCfg(),
    )
    cfg_mod.save(cfg, path)
    return path


def _iter_videos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def _human_size(bytes_: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if bytes_ < 1024 or unit == "GB":
            return f"{bytes_:.0f} {unit}" if unit == "B" else f"{bytes_:.1f} {unit}"
        bytes_ /= 1024  # type: ignore[assignment]
    return f"{bytes_:.1f} TB"


# ──────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────

class App(TkinterDnD.Tk):
    _PREVIEW_W = 640  # preview display width; height scales to source aspect

    def __init__(self) -> None:
        super().__init__()
        self.title("OWEdits — Overwatch Multikill Clipper")
        self.geometry("880x840")
        self.minsize(720, 640)

        self.config_path = _ensure_default_config()
        self.cfg = cfg_mod.load(self.config_path)

        self._log_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._vars: dict[str, StringVar | IntVar | DoubleVar | BooleanVar] = {}

        # ROI preview state
        self._preview_base: Image.Image | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._preview_after_id: str | None = None

        # DnD queue of specific videos; falls back to input_dir when empty.
        self._queue: list[Path] = []

        # Per-run state for the post-run summary.
        self._run_started_at: float = 0.0
        self._run_clip_count: int = 0
        self._run_video_count: int = 0
        self._run_montage_path: Path | None = None

        # Validation state: widget → bool (is_valid)
        self._invalid_widgets: set[tk.Widget] = set()

        self._apply_theme()
        self._build_ui()
        self._load_vars_from_cfg()
        self._bind_region_preview()
        self._show_log_empty_state()
        self._show_queue_empty_state()
        self._poll_log_queue()

    # ── Theme ─────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        # Root window background
        self.configure(background=COLOR_BG)

        style = ttk.Style(self)
        # 'clam' is the most customizable built-in ttk theme — required for
        # dark-mode color overrides because 'vista'/'winnative' hardcode chrome.
        style.theme_use("clam")

        # Global defaults
        style.configure(".",
                        font=FONT_BODY,
                        background=COLOR_BG,
                        foreground=COLOR_TEXT,
                        fieldbackground=COLOR_SURFACE_2,
                        bordercolor=COLOR_BORDER,
                        lightcolor=COLOR_SURFACE,
                        darkcolor=COLOR_BG,
                        troughcolor=COLOR_SURFACE,
                        focuscolor=COLOR_ACCENT,
                        selectbackground=COLOR_ACCENT,
                        selectforeground="white",
                        insertcolor=COLOR_TEXT)

        # Frame + notebook
        style.configure("TFrame", background=COLOR_BG)
        style.configure("Card.TFrame", background=COLOR_SURFACE, borderwidth=1, relief="solid")
        style.configure("TNotebook", background=COLOR_BG, borderwidth=0, tabmargins=(0, 4, 0, 0))
        style.configure("TNotebook.Tab",
                        background=COLOR_BG,
                        foreground=COLOR_TEXT_DIM,
                        padding=(14, 8),
                        borderwidth=0,
                        font=(FONT_FAMILY, 10))
        style.map("TNotebook.Tab",
                  background=[("selected", COLOR_SURFACE)],
                  foreground=[("selected", COLOR_TEXT), ("active", COLOR_TEXT)])

        # Labels
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=FONT_LABEL)
        style.configure("Section.TLabel", font=FONT_SECTION, foreground=COLOR_TEXT)
        style.configure("Title.TLabel", font=FONT_TITLE, foreground=COLOR_TEXT)
        style.configure("Hint.TLabel", font=FONT_HINT, foreground=COLOR_TEXT_DIM)
        style.configure("Caps.TLabel", font=FONT_HINT_CAPS, foreground=COLOR_TEXT_DIM)
        style.configure("Error.TLabel", font=FONT_HINT, foreground=COLOR_ERROR_FG)
        style.configure("Card.TLabel", background=COLOR_SURFACE, foreground=COLOR_TEXT,
                        font=FONT_LABEL)
        style.configure("CardHint.TLabel", background=COLOR_SURFACE, foreground=COLOR_TEXT_DIM,
                        font=FONT_HINT)
        style.configure("CardCaps.TLabel", background=COLOR_SURFACE, foreground=COLOR_TEXT_MUTED,
                        font=FONT_HINT_CAPS)

        # Entries + combobox
        style.configure("TEntry",
                        fieldbackground=COLOR_SURFACE_2,
                        foreground=COLOR_TEXT,
                        bordercolor=COLOR_BORDER,
                        lightcolor=COLOR_BORDER,
                        darkcolor=COLOR_BORDER,
                        insertcolor=COLOR_TEXT,
                        padding=4)
        style.map("TEntry",
                  bordercolor=[("invalid", COLOR_ERROR_FG),
                               ("focus", COLOR_ACCENT)],
                  lightcolor=[("invalid", COLOR_ERROR_FG),
                              ("focus", COLOR_ACCENT)],
                  fieldbackground=[("invalid", COLOR_ERROR_BG)])
        style.configure("TCombobox",
                        fieldbackground=COLOR_SURFACE_2,
                        background=COLOR_SURFACE_2,
                        foreground=COLOR_TEXT,
                        arrowcolor=COLOR_TEXT_DIM,
                        bordercolor=COLOR_BORDER,
                        padding=3)
        style.map("TCombobox",
                  fieldbackground=[("readonly", COLOR_SURFACE_2)],
                  selectbackground=[("readonly", COLOR_SURFACE_2)],
                  selectforeground=[("readonly", COLOR_TEXT)])
        # Dropdown list (Listbox popup) inherits from root; recolor globally.
        self.option_add("*TCombobox*Listbox.background", COLOR_SURFACE_2)
        self.option_add("*TCombobox*Listbox.foreground", COLOR_TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", COLOR_ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", "white")

        # Checkbutton — flat dark with accent tick
        style.configure("TCheckbutton",
                        background=COLOR_BG,
                        foreground=COLOR_TEXT,
                        focuscolor=COLOR_ACCENT,
                        font=FONT_LABEL)
        style.map("TCheckbutton",
                  background=[("active", COLOR_BG)],
                  foreground=[("active", COLOR_TEXT)])

        # Buttons — secondary style (default) + primary accent
        style.configure("TButton",
                        background=COLOR_SURFACE_2,
                        foreground=COLOR_TEXT,
                        bordercolor=COLOR_BORDER,
                        lightcolor=COLOR_SURFACE_2,
                        darkcolor=COLOR_SURFACE_2,
                        padding=(10, 6),
                        relief="flat",
                        focusthickness=1)
        style.map("TButton",
                  background=[("active", COLOR_DIVIDER_STRONG),
                              ("pressed", COLOR_BORDER),
                              ("disabled", COLOR_SURFACE)],
                  foreground=[("disabled", COLOR_TEXT_MUTED)])

        style.configure("Primary.TButton",
                        font=(FONT_FAMILY, 11, "bold"),
                        padding=(18, 9),
                        background=COLOR_ACCENT,
                        foreground="white",
                        bordercolor=COLOR_ACCENT,
                        lightcolor=COLOR_ACCENT,
                        darkcolor=COLOR_ACCENT,
                        relief="flat")
        style.map("Primary.TButton",
                  background=[("active", COLOR_ACCENT_HOVER),
                              ("pressed", COLOR_ACCENT_ACTIVE),
                              ("disabled", "#3a3f52")],
                  foreground=[("disabled", COLOR_TEXT_MUTED)])

        # Progressbar
        style.configure("TProgressbar",
                        background=COLOR_ACCENT,
                        troughcolor=COLOR_SURFACE,
                        bordercolor=COLOR_BORDER,
                        lightcolor=COLOR_ACCENT,
                        darkcolor=COLOR_ACCENT)

        # Separator
        style.configure("TSeparator", background=COLOR_BORDER)

        # Scrollbar
        style.configure("Vertical.TScrollbar",
                        background=COLOR_SURFACE_2,
                        troughcolor=COLOR_BG,
                        bordercolor=COLOR_BG,
                        arrowcolor=COLOR_TEXT_DIM,
                        relief="flat")
        style.map("Vertical.TScrollbar",
                  background=[("active", COLOR_DIVIDER_STRONG)])

    # ── UI layout ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=False, padx=10, pady=(10, 0))

        self._build_main_tab(notebook)
        self._build_region_tab(notebook)
        self._build_color_tab(notebook)
        self._build_clips_tab(notebook)

        # Action bar — Clip is the primary action; others are secondary.
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=10, pady=(10, 6))

        self.clip_btn = ttk.Button(bar, text="▶  Clip videos", command=self._run_clip,
                                   style="Primary.TButton")
        self.clip_btn.pack(side="left", padx=(0, 10))

        self.scan_btn = ttk.Button(bar, text="Scan only", command=self._run_scan)
        self.scan_btn.pack(side="left", padx=(0, 6))
        self.montage_btn = ttk.Button(bar, text="Build montage", command=self._run_montage)
        self.montage_btn.pack(side="left", padx=(0, 6))
        self.cancel_btn = ttk.Button(bar, text="Cancel", command=self._cancel_task, state="disabled")
        self.cancel_btn.pack(side="left", padx=(0, 16))

        self.open_btn = ttk.Button(bar, text="Open output folder", command=self._open_output)
        self.open_btn.pack(side="right", padx=(6, 0))
        self.save_btn = ttk.Button(bar, text="Save config", command=self._save_cfg)
        self.save_btn.pack(side="right", padx=(6, 0))

        Tooltip(self.clip_btn, "Detect every my-team multikill in your videos "
                               "and cut a clip for each one.")
        Tooltip(self.scan_btn, "List detected multikills without cutting any files. "
                               "Useful for previewing before a real run.")
        Tooltip(self.montage_btn,
                "Stitch every .mp4 already in the output folder into a single highlights reel.")
        Tooltip(self.save_btn, "Write the current settings to config.yaml so they persist "
                               "between launches.")

        # Progress bar — packed only while a task runs.
        self._progress_container = ttk.Frame(self)
        self._progress_container.pack(fill="x", padx=10)
        self.progress = ttk.Progressbar(self._progress_container, mode="determinate", maximum=100)
        # Not packed initially; shown on task start, hidden on DONE.

        # Status label + scrolling log.
        self.status_var = StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status_var, anchor="w", style="Hint.TLabel").pack(
            fill="x", padx=10, pady=(4, 2)
        )

        self.log = ScrolledText(self, height=12, wrap="word", font=FONT_MONO,
                                background=COLOR_SURFACE, foreground=COLOR_TEXT,
                                insertbackground=COLOR_TEXT,
                                borderwidth=1, relief="flat",
                                highlightbackground=COLOR_BORDER,
                                highlightthickness=1)
        self.log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log.configure(state="disabled")

    # ── Field helpers ─────────────────────────────────────────────────────

    def _section_header(self, parent: ttk.Frame, row: int, title: str,
                        hint: str | None = None) -> int:
        """Place a section header + optional muted hint. Returns the next row index."""
        ttk.Label(parent, text=title, style="Section.TLabel").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(4, 0)
        )
        if hint:
            ttk.Label(parent, text=hint, style="Hint.TLabel", wraplength=780).grid(
                row=row + 1, column=0, columnspan=3, sticky="w", pady=(0, 4)
            )
            return row + 2
        return row + 1

    def _separator(self, parent: ttk.Frame, row: int) -> int:
        ttk.Separator(parent).grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        return row + 1

    def _field(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        key: str,
        var,
        *,
        width: int = 16,
        browse: str | None = None,
        hint: str | None = None,
        tooltip: str | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
        is_int: bool = False,
    ) -> int:
        """Place a labeled Entry with optional browse button, helper text, tooltip, and range validation.

        Returns the next row index after this field (accounting for any hint row below)."""
        label_w = ttk.Label(parent, text=label)
        label_w.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky="w", padx=(0, 8), pady=3)
        self._vars[key] = var

        if browse:
            cmd = self._browse_dir if browse == "dir" else self._browse_file
            ttk.Button(parent, text="Browse…", command=lambda v=var: cmd(v)).grid(
                row=row, column=2, sticky="w", padx=(0, 8), pady=3
            )

        # Validate on blur if a numeric range was supplied.
        if min_val is not None or max_val is not None:
            entry.bind(
                "<FocusOut>",
                lambda _e, v=var, mn=min_val, mx=max_val, ent=entry, is_int=is_int:
                    self._validate_numeric(v, mn, mx, ent, is_int),
            )

        if tooltip:
            Tooltip(label_w, tooltip)
            Tooltip(entry, tooltip)

        next_row = row + 1
        if hint:
            ttk.Label(parent, text=hint, style="Hint.TLabel", wraplength=640).grid(
                row=next_row, column=1, columnspan=2, sticky="w", padx=(0, 8)
            )
            next_row += 1
        return next_row

    def _checkbox(self, parent: ttk.Frame, row: int, label: str, key: str,
                  var: BooleanVar, *, tooltip: str | None = None) -> int:
        cb = ttk.Checkbutton(parent, text=label, variable=var)
        cb.grid(row=row, column=0, columnspan=3, sticky="w", padx=(0, 8), pady=3)
        self._vars[key] = var
        if tooltip:
            Tooltip(cb, tooltip)
        return row + 1

    def _combo(self, parent: ttk.Frame, row: int, label: str, key: str, var: StringVar,
               values: list[str], *, tooltip: str | None = None, hint: str | None = None) -> int:
        label_w = ttk.Label(parent, text=label)
        label_w.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=14)
        combo.grid(row=row, column=1, sticky="w", padx=(0, 8), pady=3)
        self._vars[key] = var
        if tooltip:
            Tooltip(label_w, tooltip)
            Tooltip(combo, tooltip)
        next_row = row + 1
        if hint:
            ttk.Label(parent, text=hint, style="Hint.TLabel", wraplength=640).grid(
                row=next_row, column=1, columnspan=2, sticky="w", padx=(0, 8)
            )
            next_row += 1
        return next_row

    def _validate_numeric(self, var, min_val: float | None, max_val: float | None,
                          entry: ttk.Entry, is_int: bool) -> None:
        """Flag entries whose current value is out of range. Red-ish background when bad."""
        raw = str(var.get()).strip()
        ok = True
        try:
            value = int(raw) if is_int else float(raw)
        except (ValueError, tk.TclError):
            ok = False
        else:
            if min_val is not None and value < min_val:
                ok = False
            if max_val is not None and value > max_val:
                ok = False

        if ok:
            # Clearing style revert — restore the themed fieldbackground.
            entry.state(["!invalid"])
            self._invalid_widgets.discard(entry)
            self.status_var.set("Ready.")
        else:
            entry.state(["invalid"])
            self._invalid_widgets.add(entry)
            bounds = []
            if min_val is not None:
                bounds.append(f"≥ {min_val}")
            if max_val is not None:
                bounds.append(f"≤ {max_val}")
            self.status_var.set("Out-of-range value — expected " + " and ".join(bounds))

    # ── Tabs ──────────────────────────────────────────────────────────────

    def _build_main_tab(self, nb: ttk.Notebook) -> None:
        f = ttk.Frame(nb, padding=12)
        nb.add(f, text="Main")

        # Video queue (DnD)
        row = self._section_header(
            f, 0,
            "Video queue",
            "Drag videos or folders from File Explorer into the list. "
            "Leave the queue empty to process every video in the Input folder instead.",
        )
        self.queue_list = Listbox(
            f, height=6, width=72, selectmode="extended", activestyle="dotbox",
            font=FONT_BODY,
            background=COLOR_SURFACE, foreground=COLOR_TEXT,
            selectbackground=COLOR_ACCENT, selectforeground="white",
            borderwidth=1, relief="flat", highlightthickness=1,
            highlightbackground=COLOR_BORDER, highlightcolor=COLOR_ACCENT,
        )
        self.queue_list.grid(row=row, column=0, columnspan=2, sticky="we", padx=(0, 8), pady=4)
        self.queue_list.drop_target_register(DND_FILES)
        self.queue_list.dnd_bind("<<Drop>>", self._on_queue_drop)

        qbtns = ttk.Frame(f)
        qbtns.grid(row=row, column=2, sticky="nw", pady=4)
        ttk.Button(qbtns, text="Add files…", command=self._queue_add_files).pack(fill="x", pady=2)
        ttk.Button(qbtns, text="Add folder…", command=self._queue_add_folder).pack(fill="x", pady=2)
        ttk.Button(qbtns, text="Remove", command=self._queue_remove_selected).pack(fill="x", pady=2)
        ttk.Button(qbtns, text="Clear", command=self._queue_clear).pack(fill="x", pady=2)

        row = self._separator(f, row + 1)

        # Folders
        row = self._section_header(
            f, row,
            "Folders",
            "Where videos come from and where clips go. Input is only used when the queue is empty.",
        )
        row = self._field(f, row, "Input folder", "input_dir", StringVar(),
                          width=48, browse="dir",
                          hint="Source .mp4/.mkv recordings live here.")
        row = self._field(f, row, "Output folder", "output_dir", StringVar(),
                          width=48, browse="dir",
                          hint="Clips and the highlights.mp4 montage are written here.")
        row = self._field(f, row, "Templates folder", "templates_dir", StringVar(),
                          width=48, browse="dir",
                          hint="Holds optional kill-feed templates. Usually leave as default.")

        row = self._separator(f, row)

        # Clip timing
        row = self._section_header(
            f, row,
            "Clip timing",
            "How much footage to include around each detected multikill.",
        )
        row = self._field(f, row, "Pre-roll (seconds)", "pre_roll", DoubleVar(),
                          hint="How many seconds of footage before the first kill.",
                          min_val=0, max_val=60)
        row = self._field(f, row, "Post-roll (seconds)", "post_roll", DoubleVar(),
                          hint="How many seconds of footage after the last kill.",
                          min_val=0, max_val=60)
        row = self._field(f, row, "Sample rate (fps)", "sample_fps", DoubleVar(),
                          hint="Detection samples per second. 4 is a good balance of "
                               "accuracy and speed.",
                          min_val=1, max_val=30)
        row = self._combo(f, row, "Clip cut mode", "clip_method", StringVar(),
                          ["copy", "reencode"],
                          hint="copy = fast, cuts snap to keyframes. "
                               "reencode = exact timing but slower.")

        row = self._separator(f, row)

        # Multikill grouping
        row = self._section_header(
            f, row,
            "Multikill grouping",
            "What counts as a multikill — minimum kills packed into a time window.",
        )
        row = self._field(f, row, "Minimum kills", "mk_min_kills", IntVar(),
                          hint="At least this many kills in the window to produce a clip.",
                          min_val=2, max_val=10, is_int=True)
        row = self._field(f, row, "Window (seconds)", "mk_window_s", DoubleVar(),
                          hint="Kills must all happen within this many seconds of each other.",
                          min_val=1, max_val=30)

        row = self._separator(f, row)

        # Montage
        row = self._section_header(
            f, row,
            "Montage",
            "Optionally stitch all produced clips into a single highlights reel.",
        )
        row = self._checkbox(
            f, row, "Build highlights reel after clipping", "montage_enabled", BooleanVar(),
            tooltip="When enabled, the final step concatenates every clip into one .mp4.",
        )
        row = self._field(f, row, "Reel filename", "montage_filename", StringVar(),
                          width=28,
                          hint="Written inside the output folder.")

    def _build_region_tab(self, nb: ttk.Notebook) -> None:
        f = ttk.Frame(nb, padding=12)
        nb.add(f, text="Feed Region")

        row = self._section_header(
            f, 0,
            "Where the kill feed lives on screen",
            "All four values are fractions of the frame (0 = left/top, 1 = right/bottom). "
            "Defaults target OW2 at 1920×1080 with the kill-feed in the top-right. "
            "Load a preview below to calibrate visually.",
        )
        row = self._field(
            f, row, "Left edge (x)", "fr_x", DoubleVar(),
            hint="Fraction from left edge to where the ROI starts.",
            tooltip="Example: 0.72 = start 72% from the left of the frame.",
            min_val=0, max_val=1,
        )
        row = self._field(
            f, row, "Top edge (y)", "fr_y", DoubleVar(),
            hint="Fraction from top to where the ROI starts.",
            min_val=0, max_val=1,
        )
        row = self._field(
            f, row, "Width (w)", "fr_w", DoubleVar(),
            hint="Fraction of frame width covered by the ROI.",
            min_val=0.01, max_val=1,
        )
        row = self._field(
            f, row, "Height (h)", "fr_h", DoubleVar(),
            hint="Fraction of frame height covered by the ROI.",
            min_val=0.01, max_val=1,
        )

        ttk.Button(f, text="Load preview from first video",
                   command=self._load_preview).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(8, 4)
        )
        row += 1
        self.preview_label = tk.Label(f, background=COLOR_SURFACE, anchor="center",
                                      text="Drop a video in the queue or set an input folder,\n"
                                           "then click “Load preview from first video”.",
                                      foreground=COLOR_TEXT_DIM, font=FONT_BODY, padx=40, pady=40,
                                      relief="flat", borderwidth=1,
                                      highlightbackground=COLOR_BORDER,
                                      highlightthickness=1)
        self.preview_label.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 4))
        row += 1
        self.preview_hint = ttk.Label(
            f,
            text="The green box is your feed region. The yellow box is the top row "
                 "used for kill detection.",
            style="Hint.TLabel",
            wraplength=720,
        )
        self.preview_hint.grid(row=row, column=0, columnspan=3, sticky="w")
        row = self._separator(f, row + 1)

        # Row-level tuning (power-user, with tooltips)
        row = self._section_header(
            f, row,
            "Row-level tuning",
            "Advanced knobs for how rows are detected inside the feed. "
            "The defaults work on standard OW2 output — hover each label for detail.",
        )
        row = self._field(
            f, row, "Row height (fraction of frame)", "row_height_frac", DoubleVar(),
            tooltip="How tall a single kill-feed row is, as a fraction of the frame height. "
                    "OW2 rows are roughly 3.9% of a 1080p frame (~42 pixels).",
            min_val=0.005, max_val=0.2,
        )
        row = self._field(
            f, row, "Row-shift match threshold", "row_shift_thresh", DoubleVar(),
            tooltip="When a new kill appears, the previous top row slides down to the second row. "
                    "The detector confirms this by comparing the previous top's fingerprint to "
                    "the current second row's. Lower = stricter match required.",
            min_val=0, max_val=100,
        )
        row = self._field(
            f, row, "Row dedupe (seconds)", "row_dedupe_s", DoubleVar(),
            tooltip="Minimum gap between two consecutive fires. Prevents the slide-in animation "
                    "from being counted as multiple kills.",
            min_val=0, max_val=5,
        )

    def _build_color_tab(self, nb: ttk.Notebook) -> None:
        f = ttk.Frame(nb, padding=12)
        nb.add(f, text="Color Detection")

        row = self._section_header(
            f, 0,
            "Color detection",
            "OW2 color-codes the kill-feed: your team is blue/teal on the left, the enemy is "
            "red on the right. The detector looks for this pattern to filter out enemy kills "
            "and UI noise.",
        )
        ttk.Label(
            f,
            text="The defaults work for the standard OW2 colors. Only touch the advanced "
                 "settings if detection seems off — for example, if you're using a "
                 "colorblind-friendly team-color scheme.",
            style="Hint.TLabel",
            wraplength=780,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        # Progressive disclosure: advanced section is hidden by default.
        self._color_advanced_open = BooleanVar(value=False)
        toggle = ttk.Checkbutton(
            f,
            text="Show advanced color tuning",
            variable=self._color_advanced_open,
            command=self._toggle_color_advanced,
        )
        toggle.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        # Everything below this line is inside the collapsible frame.
        self._color_advanced_frame = ttk.Frame(f)
        self._color_advanced_frame.grid(row=row, column=0, columnspan=3, sticky="ew")
        # Don't grid the frame's contents yet; they live inside the frame itself.
        self._color_advanced_grid_row = row

        adv = self._color_advanced_frame
        arow = 0
        arow = self._section_header(
            adv, arow,
            "Team color ranges",
            "Hue uses OpenCV's 0–180 scale. OW blue is around 92; red wraps around 0 and 180.",
        )
        arow = self._field(
            adv, arow, "Blue hue, low end", "blue_h_low", IntVar(),
            hint="Pixels with hue ≥ this count as “blue”. Lower = catches more.",
            tooltip="OW UI blue sits around H=92. Sky blue creeps in above H=100.",
            min_val=0, max_val=180, is_int=True,
        )
        arow = self._field(
            adv, arow, "Blue hue, high end", "blue_h_high", IntVar(),
            hint="Pixels with hue ≤ this count as “blue”. Higher = catches more.",
            min_val=0, max_val=180, is_int=True,
        )
        arow = self._field(
            adv, arow, "Red hue (lower band)", "red_h_low_high", IntVar(),
            hint="Red in OpenCV wraps: this is the top of the 0–X band.",
            tooltip="Red covers H=0–10 (lower band) and H=170–180 (upper band).",
            min_val=0, max_val=30, is_int=True,
        )
        arow = self._field(
            adv, arow, "Red hue (upper band)", "red_h_high_low", IntVar(),
            hint="Bottom of the high band (red wraps past 180).",
            min_val=150, max_val=180, is_int=True,
        )
        arow = self._field(
            adv, arow, "Minimum saturation", "color_s_min", IntVar(),
            hint="Pastel/grey pixels get filtered below this. UI colors are highly saturated.",
            tooltip="Sky blue sits around S=110; OW UI blue around S=180. "
                    "A higher floor keeps the detector from firing on sky.",
            min_val=0, max_val=255, is_int=True,
        )
        arow = self._field(
            adv, arow, "Minimum brightness (value)", "color_v_min", IntVar(),
            hint="Dark/shadowed pixels get filtered below this.",
            min_val=0, max_val=255, is_int=True,
        )

        arow = self._separator(adv, arow)
        arow = self._section_header(
            adv, arow,
            "Pattern strictness",
            "How strongly a row must match the blue-left / red-right shape before it counts.",
        )
        arow = self._field(
            adv, arow, "Min coloured pixel fraction", "min_color_px_frac", DoubleVar(),
            hint="Each of blue and red must cover this fraction of the row. 0.01 = 1%.",
            min_val=0, max_val=1,
        )
        arow = self._field(
            adv, arow, "Min colour-centre separation", "min_center_sep_frac", DoubleVar(),
            hint="The blue cluster's centre must sit this far left of the red cluster's, as a "
                 "fraction of the row width.",
            min_val=0, max_val=1,
        )

        # Start collapsed.
        self._color_advanced_frame.grid_remove()

    def _toggle_color_advanced(self) -> None:
        if self._color_advanced_open.get():
            self._color_advanced_frame.grid()
        else:
            self._color_advanced_frame.grid_remove()

    # ── Clips browser tab ─────────────────────────────────────────────────

    _CARD_W = 240          # thumbnail width (card inner image width)
    _CARD_H = 135          # 16:9 thumbnail height
    _CARD_PAD = 12
    _CARDS_PER_ROW = 3

    def _build_clips_tab(self, nb: ttk.Notebook) -> None:
        f = ttk.Frame(nb, padding=12)
        nb.add(f, text="Clips")

        # Header row: title · count · refresh
        header = ttk.Frame(f)
        header.pack(fill="x", pady=(0, 8))
        self._clips_title_var = StringVar(value="ALL CLIPS (0)")
        ttk.Label(header, textvariable=self._clips_title_var, style="Caps.TLabel").pack(side="left")
        ttk.Button(header, text="Refresh", command=self._refresh_clips).pack(side="right")
        ttk.Button(header, text="Open folder", command=self._open_output).pack(side="right", padx=(0, 6))

        # Scrollable canvas holding the card grid.
        self._clips_canvas = tk.Canvas(
            f, background=COLOR_BG, highlightthickness=0, borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(f, orient="vertical", command=self._clips_canvas.yview)
        self._clips_canvas.configure(yscrollcommand=scrollbar.set)
        self._clips_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._clips_inner = ttk.Frame(self._clips_canvas)
        self._clips_canvas.create_window((0, 0), window=self._clips_inner, anchor="nw",
                                         tags="inner")
        self._clips_inner.bind(
            "<Configure>",
            lambda e: self._clips_canvas.configure(scrollregion=self._clips_canvas.bbox("all")),
        )
        # Mouse-wheel scrolling while over the canvas
        self._clips_canvas.bind("<Enter>",
                                lambda _e: self._clips_canvas.bind_all("<MouseWheel>",
                                                                       self._on_clips_scroll))
        self._clips_canvas.bind("<Leave>",
                                lambda _e: self._clips_canvas.unbind_all("<MouseWheel>"))

        # Thumbnail cache: path -> ImageTk.PhotoImage (must outlive cards).
        self._thumb_cache: dict[str, ImageTk.PhotoImage] = {}
        # Empty-state placeholder shown when no clips are available.
        self._clips_empty_label: ttk.Label | None = None

        # Refresh whenever the Clips tab becomes visible.
        nb.bind("<<NotebookTabChanged>>", self._on_tab_changed, add="+")

    def _on_tab_changed(self, _event: tk.Event) -> None:
        try:
            idx = self._clips_canvas.winfo_toplevel().nametowidget(
                _event.widget.select()
            )
        except Exception:
            return
        # If the currently selected tab contains our canvas, refresh.
        if str(self._clips_canvas) in str(idx) or idx.winfo_name() == "!frame4":
            self._refresh_clips()

    def _on_clips_scroll(self, event: tk.Event) -> None:
        self._clips_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _refresh_clips(self) -> None:
        """Re-enumerate clips in the output folder and rebuild the card grid."""
        # Clear prior children
        for child in self._clips_inner.winfo_children():
            child.destroy()
        self._clips_empty_label = None

        out_dir = Path(self._vars["output_dir"].get())
        montage_name = str(self._vars["montage_filename"].get())
        clips: list[Path] = []
        if out_dir.exists():
            clips = sorted(
                (p for p in out_dir.glob("*.mp4") if p.name != montage_name),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        self._clips_title_var.set(f"ALL CLIPS ({len(clips)})")

        if not clips:
            self._clips_empty_label = ttk.Label(
                self._clips_inner,
                text="No clips yet — drop videos on the Main tab and hit Clip videos.",
                style="Hint.TLabel",
                padding=40,
            )
            self._clips_empty_label.grid(row=0, column=0, sticky="w")
            return

        # Build a 3-column grid of cards.
        for i, p in enumerate(clips):
            r, c = divmod(i, self._CARDS_PER_ROW)
            self._build_clip_card(self._clips_inner, p, r, c)

    def _build_clip_card(self, parent: ttk.Frame, path: Path, row: int, col: int) -> None:
        card = tk.Frame(
            parent,
            background=COLOR_SURFACE,
            highlightbackground=COLOR_BORDER,
            highlightthickness=1,
            borderwidth=0,
        )
        card.grid(row=row, column=col, padx=self._CARD_PAD, pady=self._CARD_PAD, sticky="nw")

        # Thumbnail + duration badge (badge is a label on top of thumbnail)
        thumb_frame = tk.Frame(card, background=COLOR_SURFACE)
        thumb_frame.pack(fill="x")
        thumb_img = self._clip_thumbnail(path)
        duration = self._clip_duration(path)

        thumb_lbl = tk.Label(thumb_frame, image=thumb_img, background=COLOR_BG, borderwidth=0)
        thumb_lbl.image = thumb_img  # keep a ref
        thumb_lbl.pack()

        if duration:
            badge = tk.Label(
                thumb_frame, text=duration,
                background="#000000", foreground="#ffffff",
                font=(FONT_FAMILY, 9, "bold"), padx=6, pady=1,
            )
            # Place badge in the top-right of the thumbnail
            badge.place(in_=thumb_lbl, relx=1.0, rely=0.0, x=-6, y=6, anchor="ne")

        # Metadata row under the thumbnail
        meta = tk.Frame(card, background=COLOR_SURFACE, padx=10, pady=8)
        meta.pack(fill="x")

        tk.Label(meta, text="OVERWATCH", background=COLOR_SURFACE,
                 foreground=COLOR_TEXT_DIM, font=FONT_HINT_CAPS).pack(anchor="w")

        title = self._clip_friendly_title(path)
        tk.Label(meta, text=title, background=COLOR_SURFACE, foreground=COLOR_TEXT,
                 font=FONT_LABEL, anchor="w", justify="left",
                 wraplength=self._CARD_W).pack(anchor="w", pady=(2, 4))

        age = self._clip_age(path)
        tk.Label(meta, text=age, background=COLOR_SURFACE, foreground=COLOR_TEXT_MUTED,
                 font=FONT_HINT_CAPS).pack(anchor="w")

        # Click anywhere on the card (or thumbnail) to play
        for w in (card, thumb_frame, thumb_lbl, meta):
            w.bind("<Button-1>", lambda _e, pth=path: self._play_clip(pth))
            w.bind("<Enter>", lambda _e, c=card: c.configure(
                highlightbackground=COLOR_ACCENT))
            w.bind("<Leave>", lambda _e, c=card: c.configure(
                highlightbackground=COLOR_BORDER))
        # Children of meta share hover behaviour too
        for child in meta.winfo_children():
            child.bind("<Button-1>", lambda _e, pth=path: self._play_clip(pth))

    def _play_clip(self, path: Path) -> None:
        try:
            os.startfile(path)
        except Exception as e:
            messagebox.showerror("Play clip", f"Could not open {path.name}:\n{e}")

    def _clip_thumbnail(self, path: Path) -> ImageTk.PhotoImage:
        key = f"{path}:{path.stat().st_mtime}"
        cached = self._thumb_cache.get(key)
        if cached is not None:
            return cached
        # Grab a frame ~25% in for a more representative still.
        cap = cv2.VideoCapture(str(path))
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.25))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            # Fallback: solid dark tile
            img = Image.new("RGB", (self._CARD_W, self._CARD_H), color=(26, 32, 40))
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            # Cover-fit into the card dimensions (letterbox if needed).
            img = img.resize((self._CARD_W, self._CARD_H), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._thumb_cache[key] = photo
        return photo

    @staticmethod
    def _clip_duration(path: Path) -> str:
        try:
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            if fps <= 0 or frames <= 0:
                return ""
            seconds = int(round(frames / fps))
            return f"{seconds // 60:02d}:{seconds % 60:02d}"
        except Exception:
            return ""

    @staticmethod
    def _clip_friendly_title(path: Path) -> str:
        """Extract a human-readable title from clip filename.

        Filenames look like: `<stem>__<kind>_<N>x_<MM>-<SS>.mp4`.
        Return "Multikill 3× at 02:15" style.
        """
        stem = path.stem
        parts = stem.rsplit("__", 1)
        if len(parts) == 2:
            tail = parts[1]
            try:
                kind, count_x, ts = tail.split("_")
                count = count_x.rstrip("x")
                mm, ss = ts.split("-")
                return f"{kind.replace('_', ' ').title()} {count}× at {mm}:{ss}"
            except ValueError:
                pass
        return stem

    @staticmethod
    def _clip_age(path: Path) -> str:
        try:
            delta = time.time() - path.stat().st_mtime
        except OSError:
            return ""
        if delta < 60:
            return "JUST NOW"
        if delta < 3600:
            return f"{int(delta // 60)} MIN AGO"
        if delta < 86400:
            hrs = int(delta // 3600)
            return f"{hrs} HOUR{'S' if hrs != 1 else ''} AGO"
        days = int(delta // 86400)
        return f"{days} DAY{'S' if days != 1 else ''} AGO"

    # ── Config ↔ vars ─────────────────────────────────────────────────────

    def _load_vars_from_cfg(self) -> None:
        c = self.cfg
        self._vars["input_dir"].set(str(c.input_dir))
        self._vars["output_dir"].set(str(c.output_dir))
        self._vars["templates_dir"].set(str(c.templates_dir))
        self._vars["pre_roll"].set(c.pre_roll)
        self._vars["post_roll"].set(c.post_roll)
        self._vars["sample_fps"].set(c.sample_fps)
        self._vars["clip_method"].set(c.clip_method)
        self._vars["mk_min_kills"].set(c.multikill.min_kills)
        self._vars["mk_window_s"].set(c.multikill.window_s)
        self._vars["montage_enabled"].set(c.montage.enabled)
        self._vars["montage_filename"].set(c.montage.filename)
        self._vars["fr_x"].set(c.feed_region.x)
        self._vars["fr_y"].set(c.feed_region.y)
        self._vars["fr_w"].set(c.feed_region.w)
        self._vars["fr_h"].set(c.feed_region.h)
        cp = c.color_pattern
        self._vars["row_height_frac"].set(cp.row_height_frac)
        self._vars["row_shift_thresh"].set(cp.row_shift_thresh)
        self._vars["row_dedupe_s"].set(cp.row_dedupe_s)
        self._vars["blue_h_low"].set(cp.blue_h_low)
        self._vars["blue_h_high"].set(cp.blue_h_high)
        self._vars["red_h_low_high"].set(cp.red_h_low_high)
        self._vars["red_h_high_low"].set(cp.red_h_high_low)
        self._vars["color_s_min"].set(cp.color_s_min)
        self._vars["color_v_min"].set(cp.color_v_min)
        self._vars["min_color_px_frac"].set(cp.min_color_px_frac)
        self._vars["min_center_sep_frac"].set(cp.min_center_sep_frac)

    def _build_cfg_from_vars(self) -> cfg_mod.Config:
        c = self.cfg
        return cfg_mod.Config(
            input_dir=Path(self._vars["input_dir"].get()),
            output_dir=Path(self._vars["output_dir"].get()),
            templates_dir=Path(self._vars["templates_dir"].get()),
            pre_roll=float(self._vars["pre_roll"].get()),
            post_roll=float(self._vars["post_roll"].get()),
            feed_region=cfg_mod.FeedRegion(
                x=float(self._vars["fr_x"].get()),
                y=float(self._vars["fr_y"].get()),
                w=float(self._vars["fr_w"].get()),
                h=float(self._vars["fr_h"].get()),
            ),
            sample_fps=float(self._vars["sample_fps"].get()),
            match_thresholds=c.match_thresholds,
            dedupe_window_s=c.dedupe_window_s,
            multikill=cfg_mod.EventRule(
                min_kills=int(self._vars["mk_min_kills"].get()),
                window_s=float(self._vars["mk_window_s"].get()),
            ),
            team_wipe=c.team_wipe,
            clip_method=str(self._vars["clip_method"].get()),
            montage=cfg_mod.MontageCfg(
                enabled=bool(self._vars["montage_enabled"].get()),
                filename=str(self._vars["montage_filename"].get()),
            ),
            color_pattern=cfg_mod.ColorPatternCfg(
                row_height_frac=float(self._vars["row_height_frac"].get()),
                row_shift_thresh=float(self._vars["row_shift_thresh"].get()),
                row_dedupe_s=float(self._vars["row_dedupe_s"].get()),
                blue_h_low=int(self._vars["blue_h_low"].get()),
                blue_h_high=int(self._vars["blue_h_high"].get()),
                red_h_low_high=int(self._vars["red_h_low_high"].get()),
                red_h_high_low=int(self._vars["red_h_high_low"].get()),
                color_s_min=int(self._vars["color_s_min"].get()),
                color_v_min=int(self._vars["color_v_min"].get()),
                min_color_px_frac=float(self._vars["min_color_px_frac"].get()),
                min_center_sep_frac=float(self._vars["min_center_sep_frac"].get()),
            ),
        )

    # ── Video queue (drag-and-drop) ───────────────────────────────────────

    _QUEUE_EMPTY_HINT = "— drop videos or folders here (or click “Add files…”) —"

    def _show_queue_empty_state(self) -> None:
        if self._queue:
            return
        self.queue_list.delete(0, END)
        self.queue_list.insert(END, self._QUEUE_EMPTY_HINT)
        self.queue_list.itemconfig(0, foreground=COLOR_TEXT_MUTED)
        # Prevent accidental selection of the placeholder row
        self.queue_list.selection_clear(0, END)

    def _clear_queue_empty_state(self) -> None:
        if self.queue_list.size() == 1 and self.queue_list.get(0) == self._QUEUE_EMPTY_HINT:
            self.queue_list.delete(0)

    @staticmethod
    def _parse_dnd_paths(data: str) -> list[Path]:
        """Parse TkDnD's file list, which uses {braces} around paths that contain spaces."""
        paths: list[Path] = []
        buf: list[str] = []
        in_brace = False
        for ch in data:
            if ch == "{":
                in_brace = True
            elif ch == "}":
                in_brace = False
                if buf:
                    paths.append(Path("".join(buf)))
                    buf = []
            elif ch == " " and not in_brace:
                if buf:
                    paths.append(Path("".join(buf)))
                    buf = []
            else:
                buf.append(ch)
        if buf:
            paths.append(Path("".join(buf)))
        return paths

    def _expand_to_videos(self, paths: list[Path]) -> list[Path]:
        out: list[Path] = []
        for p in paths:
            if p.is_dir():
                out.extend(_iter_videos(p))
            elif p.suffix.lower() in VIDEO_EXTS and p.is_file():
                out.append(p)
        return out

    def _queue_add(self, videos: list[Path]) -> None:
        added = 0
        existing = {str(p) for p in self._queue}
        self._clear_queue_empty_state()
        for v in videos:
            key = str(v.resolve())
            if key in existing:
                continue
            self._queue.append(v.resolve())
            existing.add(key)
            self.queue_list.insert(END, v.name)
            added += 1
        if added:
            self._log("INFO", f"queued {added} video(s) (total {len(self._queue)})")
        if not self._queue:
            self._show_queue_empty_state()

    def _on_queue_drop(self, event) -> None:  # type: ignore[no-untyped-def]
        dropped = self._parse_dnd_paths(event.data)
        videos = self._expand_to_videos(dropped)
        if not videos:
            messagebox.showinfo("Drop", "No video files found in what you dropped.")
            return
        self._queue_add(videos)

    def _queue_add_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="Choose videos",
            filetypes=[("Video files", "*.mp4 *.mkv *.mov *.avi"), ("All files", "*.*")],
        )
        if not files:
            return
        self._queue_add([Path(f) for f in files])

    def _queue_add_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose folder of videos")
        if not folder:
            return
        self._queue_add(_iter_videos(Path(folder)))

    def _queue_remove_selected(self) -> None:
        # Guard against the placeholder row being selected.
        if not self._queue:
            return
        for idx in reversed(self.queue_list.curselection()):
            self.queue_list.delete(idx)
            del self._queue[idx]
        if not self._queue:
            self._show_queue_empty_state()

    def _queue_clear(self) -> None:
        self.queue_list.delete(0, END)
        self._queue.clear()
        self._show_queue_empty_state()

    def _videos_to_process(self, cfg: cfg_mod.Config) -> list[Path]:
        if self._queue:
            return list(self._queue)
        return _iter_videos(cfg.input_dir)

    # ── ROI preview ───────────────────────────────────────────────────────

    def _bind_region_preview(self) -> None:
        for key in ("fr_x", "fr_y", "fr_w", "fr_h", "row_height_frac"):
            self._vars[key].trace_add("write", lambda *_: self._schedule_preview_refresh())

    def _schedule_preview_refresh(self) -> None:
        if self._preview_after_id is not None:
            try:
                self.after_cancel(self._preview_after_id)
            except Exception:
                pass
        self._preview_after_id = self.after(120, self._refresh_preview)

    def _load_preview(self) -> None:
        if self._queue:
            videos = list(self._queue)
        else:
            in_dir = Path(self._vars["input_dir"].get())
            videos = _iter_videos(in_dir) if in_dir.exists() else []
        if not videos:
            messagebox.showinfo(
                "Load preview",
                "No videos found.\nDrop a video onto the queue or set an input folder first.",
            )
            return
        v = videos[0]
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 3))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            messagebox.showerror("Load preview", f"Could not decode a frame from {v.name}.")
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._preview_base = Image.fromarray(rgb)
        self.preview_hint.configure(
            text=f"Loaded from {v.name} · {self._preview_base.width}×{self._preview_base.height}. "
                 "Green = feed region · Yellow = top row used for detection."
        )
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        self._preview_after_id = None
        if self._preview_base is None:
            return
        try:
            x = float(self._vars["fr_x"].get())
            y = float(self._vars["fr_y"].get())
            w = float(self._vars["fr_w"].get())
            h = float(self._vars["fr_h"].get())
            rh = float(self._vars["row_height_frac"].get())
        except Exception:
            return
        img = self._preview_base.copy()
        iw, ih = img.size
        draw = ImageDraw.Draw(img)
        box = (
            max(0, int(x * iw)),
            max(0, int(y * ih)),
            min(iw - 1, int((x + w) * iw)),
            min(ih - 1, int((y + h) * ih)),
        )
        draw.rectangle(box, outline="#00ff00", width=4)
        top = (box[0], box[1], box[2], min(ih - 1, box[1] + int(rh * ih)))
        draw.rectangle(top, outline="#ffff00", width=3)
        pw = self._PREVIEW_W
        ph = int(ih * pw / iw)
        img = img.resize((pw, ph), Image.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self._preview_photo, text="", padding=0)

    # ── Actions ───────────────────────────────────────────────────────────

    def _browse_dir(self, var: StringVar) -> None:
        initial = var.get() or str(_app_dir())
        chosen = filedialog.askdirectory(initialdir=initial)
        if chosen:
            var.set(chosen)

    def _browse_file(self, var: StringVar) -> None:
        initial = var.get() or str(_app_dir())
        chosen = filedialog.askopenfilename(initialdir=initial)
        if chosen:
            var.set(chosen)

    def _open_output(self) -> None:
        out = Path(self._vars["output_dir"].get())
        out.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(out)
        except Exception as e:
            messagebox.showerror("Open folder", f"Could not open {out}:\n{e}")

    def _save_cfg(self) -> None:
        try:
            cfg = self._build_cfg_from_vars()
        except Exception as e:
            messagebox.showerror("Save Config", f"Config is invalid:\n{e}")
            return
        try:
            cfg_mod.save(cfg, self.config_path)
            self.cfg = cfg
            self._log("INFO", f"Saved config to {self.config_path}")
            self.status_var.set(f"Saved {self.config_path.name}.")
        except Exception as e:
            messagebox.showerror("Save Config", f"Write failed:\n{e}")

    def _run_scan(self) -> None:
        self._run_task("Scan", self._scan_task)

    def _run_clip(self) -> None:
        self._run_task("Clip", self._clip_task)

    def _run_montage(self) -> None:
        self._run_task("Montage", self._montage_task)

    def _run_task(self, name: str, fn: Callable[[cfg_mod.Config], None]) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showinfo(name, "Another operation is already running.")
            return
        if self._invalid_widgets:
            messagebox.showwarning(
                name, "One or more fields are out of range. Fix the highlighted values first."
            )
            return
        try:
            cfg = self._build_cfg_from_vars()
        except Exception as e:
            messagebox.showerror(name, f"Config is invalid:\n{e}")
            return

        # Reset per-run counters + show the progress bar.
        self._run_started_at = time.monotonic()
        self._run_clip_count = 0
        self._run_video_count = 0
        self._run_montage_path = None
        self._stop_event.clear()
        self._set_buttons_enabled(False)
        self.cancel_btn.configure(state="normal")
        self.progress.configure(value=0, maximum=100, mode="determinate")
        self.progress.pack(fill="x", pady=(2, 4))
        self.status_var.set(f"{name}: starting…")
        self._log("INFO", f"{name} started")

        def _thread() -> None:
            try:
                fn(cfg)
                self._log_queue.put(("DONE", name))
            except TaskCancelled:
                self._log_queue.put(("INFO", f"{name} cancelled by user"))
                self._log_queue.put(("DONE", name))
            except Exception:
                self._log_queue.put(("ERR", traceback.format_exc()))
                self._log_queue.put(("DONE", name))

        self._worker = threading.Thread(target=_thread, daemon=True)
        self._worker.start()

    def _cancel_task(self) -> None:
        if self._worker and self._worker.is_alive():
            self._stop_event.set()
            self.status_var.set("Cancelling — finishing current step…")
            self.cancel_btn.configure(state="disabled")

    def _check_cancelled(self) -> None:
        if self._stop_event.is_set():
            raise TaskCancelled()

    def _post_progress(self, current: int, total: int) -> None:
        self._log_queue.put(("PROGRESS", (current, total)))

    def _post_indeterminate(self, on: bool) -> None:
        self._log_queue.put(("INDETERMINATE", on))

    # ── Worker bodies ─────────────────────────────────────────────────────

    def _scan_task(self, cfg: cfg_mod.Config) -> None:
        detectors = [feed_detect.RowCountDetector()]
        videos = self._videos_to_process(cfg)
        if not videos:
            src = "queue" if self._queue else f"folder {cfg.input_dir}"
            self._log_queue.put(("INFO", f"No videos found ({src} is empty)."))
            return
        total = len(videos)
        self._run_video_count = total
        self._post_progress(0, total)
        for i, v in enumerate(videos, start=1):
            self._check_cancelled()
            self._log_queue.put(("INFO", f"scanning {v.name}  ({i}/{total})"))
            meta = feed_detect.probe(v)
            kills: list[feed_detect.KillEvent] = []
            for d in detectors:
                kills.extend(d.detect(v, cfg, meta))
            kills.sort(key=lambda k: k.t)
            events = events_mod.group(kills, cfg, meta.duration_s)
            for e in events:
                self._log_queue.put((
                    "INFO",
                    f"  multikill {e.kill_count}× at {e.first_kill_t:5.1f}s "
                    f"(clip {e.start_s:.1f}s → {e.end_s:.1f}s)",
                ))
            if not events:
                self._log_queue.put(("INFO", "  (no multikills detected)"))
            self._post_progress(i, total)

    def _clip_task(self, cfg: cfg_mod.Config) -> None:
        detectors = [feed_detect.RowCountDetector()]
        videos = self._videos_to_process(cfg)
        if not videos:
            src = "queue" if self._queue else f"folder {cfg.input_dir}"
            self._log_queue.put(("INFO", f"No videos found ({src} is empty)."))
            return
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        all_clips: list[Path] = []
        self._run_video_count = len(videos)
        total = len(videos) + (1 if cfg.montage.enabled else 0)
        self._post_progress(0, total)
        for i, v in enumerate(videos, start=1):
            self._check_cancelled()
            self._log_queue.put(("INFO", f"processing {v.name}  ({i}/{len(videos)})"))
            meta = feed_detect.probe(v)
            kills: list[feed_detect.KillEvent] = []
            for d in detectors:
                kills.extend(d.detect(v, cfg, meta))
            kills.sort(key=lambda k: k.t)
            events = events_mod.group(kills, cfg, meta.duration_s)
            for e in events:
                self._check_cancelled()
                out = clip_mod.cut(e, cfg.output_dir, cfg.clip_method)
                all_clips.append(out)
                self._log_queue.put(("INFO", f"  → {out.name}"))
            self._post_progress(i, total)
        self._run_clip_count = len(all_clips)
        if cfg.montage.enabled and all_clips:
            self._check_cancelled()
            self._log_queue.put(("INFO", "building montage…"))
            self._post_indeterminate(True)
            mpath = cfg.output_dir / cfg.montage.filename
            clip_mod.montage(all_clips, mpath)
            self._run_montage_path = mpath
            self._post_indeterminate(False)
            self._log_queue.put(("INFO", f"montage: {mpath.name}"))
            self._post_progress(total, total)
        elif cfg.montage.enabled and not all_clips:
            self._log_queue.put(("INFO", "No clips produced; skipping montage."))
            self._post_progress(total, total)

    def _montage_task(self, cfg: cfg_mod.Config) -> None:
        name = cfg.montage.filename
        clips = sorted(p for p in cfg.output_dir.glob("*.mp4") if p.name != name)
        if not clips:
            self._log_queue.put(("INFO", f"No clips in {cfg.output_dir}."))
            return
        self._post_indeterminate(True)
        out = clip_mod.montage(clips, cfg.output_dir / name)
        self._run_montage_path = out
        self._post_indeterminate(False)
        self._log_queue.put(("INFO", f"montage: {out.name}"))
        self._post_progress(1, 1)

    # ── Log pump + UI state ───────────────────────────────────────────────

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for b in (self.scan_btn, self.clip_btn, self.montage_btn):
            b.configure(state=state)

    def _finalize_task(self, name: str) -> None:
        """Called on DONE: hide progress, re-enable buttons, write summary."""
        self._set_buttons_enabled(True)
        self.cancel_btn.configure(state="disabled")
        # Hide the progress bar when idle.
        try:
            self.progress.pack_forget()
        except tk.TclError:
            pass
        self.progress.configure(value=0, mode="determinate")

        if self._stop_event.is_set():
            self.status_var.set(f"{name}: cancelled.")
            return

        elapsed = time.monotonic() - self._run_started_at
        parts: list[str] = [f"{name} finished in {elapsed:.1f}s"]
        if name == "Clip":
            parts.append(
                f"{self._run_clip_count} clip(s) from {self._run_video_count} video(s)"
            )
            if self._run_montage_path and self._run_montage_path.exists():
                try:
                    size = self._run_montage_path.stat().st_size
                    parts.append(
                        f"montage: {self._run_montage_path.name} ({_human_size(size)})"
                    )
                except OSError:
                    pass
        elif name == "Scan":
            parts.append(f"{self._run_video_count} video(s) scanned")
        elif name == "Montage" and self._run_montage_path:
            try:
                size = self._run_montage_path.stat().st_size
                parts.append(f"{self._run_montage_path.name} ({_human_size(size)})")
            except OSError:
                pass

        summary = " · ".join(parts) + "."
        self._log("DONE", summary)
        self.status_var.set(summary)

    def _poll_log_queue(self) -> None:
        try:
            while True:
                kind, msg = self._log_queue.get_nowait()
                if kind == "DONE":
                    self._finalize_task(str(msg))
                elif kind == "ERR":
                    self._log("ERROR", str(msg))
                    self.status_var.set("Error — see log.")
                elif kind == "PROGRESS":
                    current, total = msg  # type: ignore[misc]
                    if total > 0:
                        if self.progress.cget("mode") == "indeterminate":
                            # Leave indeterminate alone; determinate update ignored.
                            pass
                        else:
                            self.progress.configure(maximum=total, value=current)
                            self.status_var.set(f"Working… step {current}/{total}")
                elif kind == "INDETERMINATE":
                    if bool(msg):
                        self.progress.configure(mode="indeterminate")
                        self.progress.start(80)
                    else:
                        self.progress.stop()
                        self.progress.configure(mode="determinate")
                else:
                    self._log(str(kind), str(msg))
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    def _show_log_empty_state(self) -> None:
        self.log.configure(state="normal")
        self.log.insert(
            END,
            "Logs will appear here once you run Scan or Clip.\n"
            "Tip: drag a video into the queue on the Main tab, then click “Clip videos”.\n",
        )
        self.log.configure(state="disabled")
        self._log_is_empty = True

    def _log(self, kind: str, msg: str) -> None:
        self.log.configure(state="normal")
        if getattr(self, "_log_is_empty", False):
            self.log.delete("1.0", END)
            self._log_is_empty = False
        self.log.insert(END, f"[{kind}] {msg}\n")
        self.log.see(END)
        self.log.configure(state="disabled")


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
