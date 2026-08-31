"""
titlekit.py — shared helpers for the on-screen title layer.

A *title* is one line of text rendered in the same turquoise pill as the
subtitles, sitting directly above them, shown for the first ~10 seconds of a
clip and then faded out.

Source of truth:  transcripts/<stem>/title.txt   (a single line of text)

  - auto-created on the first prepare / subtitle run, pre-filled with a title
    derived from the source file name
  - edit it to change the on-screen title
  - an empty file (whitespace only) means "no title for this clip"
  - shared by a horizontal video and its portrait reel crop (same stem)

Also exposes detect_face_vspan(), used by the burn step to keep the title from
covering the speaker's face.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

# Trailing camera-export cruft, e.g. "..., Film am 19.03.26 um 13.52 #2"
_STAMP_RE = re.compile(
    r"[,.]?\s*(?:Film\s+)?am\s+\d{1,2}\.\d{1,2}\.\d{2,4}\s+um\s+"
    r"\d{1,2}[.:]\d{2}\s*(?:#\d+)?\s*$",
    re.IGNORECASE,
)
_TRAILING_TAKE_RE = re.compile(r"\s*#\d+\s*$")


def default_title(stem: str) -> str:
    """Best-effort readable title from a source file's stem."""
    t = _STAMP_RE.sub("", stem)
    t = _TRAILING_TAKE_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip(" ,;.-–—")
    return t or stem


def title_file(transcripts_dir: "str | Path", stem: str) -> Path:
    return Path(transcripts_dir) / stem / "title.txt"


def ensure_title_file(transcripts_dir: "str | Path", stem: str) -> Path:
    """Create transcripts/<stem>/title.txt with the file-name default if missing."""
    p = title_file(transcripts_dir, stem)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(default_title(stem) + "\n", encoding="utf-8")
        print(f"  🏷  Title created: {p}  ← edit this (empty file = no title)")
    return p


def load_title(transcripts_dir: "str | Path", stem: str) -> str:
    """Return the clip's title text, or '' when unset / file missing."""
    p = title_file(transcripts_dir, stem)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8").strip()


def title_mtime(transcripts_dir: "str | Path", stem: str) -> float:
    p = title_file(transcripts_dir, stem)
    return p.stat().st_mtime if p.exists() else 0.0


def detect_face_vspan(video_path: "str | Path",
                      n_frames: int = 10) -> Optional[Tuple[float, float]]:
    """
    Median vertical span (y0, y1) of the speaker's face across n_frames sampled
    frames, normalised to [0, 1] of the frame height. Returns None if opencv is
    unavailable or no face is detected — callers then place the title without a
    face constraint (safe-zone clamp only).
    """
    try:
        import cv2
        import numpy
    except ImportError:
        return None

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        data = json.loads(probe.stdout)
        vs = next(s for s in data["streams"] if s.get("codec_type") == "video")
        w, h = int(vs["width"]), int(vs["height"])
        duration = float(data["format"]["duration"])
    except (ValueError, StopIteration, KeyError):
        return None

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    start, end = duration * 0.1, duration * 0.9
    tops, bottoms = [], []
    for i in range(n_frames):
        t = start + (end - start) * i / max(n_frames - 1, 1)
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}",
             "-i", str(video_path), "-frames:v", "1", "-f", "image2pipe",
             "-pix_fmt", "gray", "-vcodec", "rawvideo", "-"],
            capture_output=True,
        )
        if out.returncode != 0 or len(out.stdout) < w * h:
            continue
        frame = numpy.frombuffer(out.stdout[: w * h], dtype=numpy.uint8).reshape((h, w))
        faces = cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(60, 60))
        for (_, y, _, fh) in faces:
            tops.append(y / h)
            bottoms.append((y + fh) / h)

    if not tops:
        return None
    tops.sort()
    bottoms.sort()
    return (tops[len(tops) // 2], bottoms[len(bottoms) // 2])
