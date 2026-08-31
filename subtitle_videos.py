#!/usr/bin/env python3
"""
subtitle_videos.py — Automated subtitle generator for horizontal videos.
Uses OpenAI Whisper for transcription and ffmpeg for burning subtitles.

Workflow:
    1. python subtitle_videos.py --step transcribe   # → .srt  .words.json  .txt
    2.  edit  transcripts/<name>/transcript.txt       # fix typos, punctuation, etc.
    3. python subtitle_videos.py --step apply        # apply edits back to .srt
    4. python subtitle_videos.py --step burn         # render video

    python subtitle_videos.py                        # shortcut: transcribe + burn (skips step 2-3)
    python subtitle_videos.py --model large --lang fr
    python subtitle_videos.py --max-words 5

When a shared transcript (transcripts/<name>/transcript.words.json, written by
prepare_videos.py) exists, step 1 just reformats it into an SRT — Whisper is not
re-run, so --model / --lang / --max-words have no effect on the text there.
An edited transcript.txt newer than the SRT is applied automatically before burn.
"""

import argparse
import glob as _glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import titlekit


# ── Config ────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

TITLE_FONT      = "League Spartan ExtraBold"  # weight 800 instance, bundled in assets/
TITLE_FONT_MEASURE = "LeagueSpartan-ExtraBold"
BODY_FONT       = "Karla"                     # subtitle body text (see assets/Karla)

# Title type — mirrors the homepage H1 (League Spartan 800, tight)
TITLE_LINE_HEIGHT = 0.95
TITLE_TRACKING    = -0.035  # homepage letter-spacing, in em

# ASS colors (ABGR format: &HAABBGGRR — AA: 00 = opaque, FF = transparent)
TURQUOISE = "&H1EE6E15C"   # #5CE1E6 — title pill, ~88 % opaque (matches social posts)
WHITE     = "&H00FFFFFF"   # subtitle / title text
YELLOW    = "&H005BF6FF"   # #fff65b — *asterisk* emphasis
BLACK     = "&H00000000"   # subtitle outline
SHADOW    = "&HC0000000"   # subtitle drop shadow — soft + transparent (~25 % opaque)
NONE      = "&H00000000"

# Title layer
TITLE_MAX_SECONDS   = 10.0
TITLE_FADE_OUT_MS   = 400
TITLE_COVER_MS      = 1200   # title holds a low "cover" position this long …
TITLE_MOVE_MS       = 400    # … then slides up to sit above the subtitle
COVER_HOLD_SECONDS  = (TITLE_COVER_MS + TITLE_MOVE_MS) / 1000  # subtitles start after
BODY_SCALE          = 0.70   # subtitle size relative to the title
SAFE_BOTTOM_FRAC    = 0.92   # both layers sit this low (block moved down for the cover)


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_dependencies(need_whisper: bool = True):
    errors = []
    if need_whisper:
        try:
            import whisper  # noqa
        except ImportError:
            errors.append("openai-whisper  →  pip install openai-whisper")
    for tool in ("ffmpeg", "ffprobe"):
        if subprocess.run([tool, "-version"], capture_output=True).returncode != 0:
            errors.append(f"{tool}  →  https://ffmpeg.org/download.html")
    if errors:
        print("❌ Missing dependencies:\n")
        for e in errors:
            print(f"   {e}")
        sys.exit(1)


def get_video_duration(video_path: Path) -> float:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", str(video_path)]
    try:
        data = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
        return float(data["format"]["duration"])
    except (ValueError, KeyError):
        return 0.0


def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", str(video_path)]
    data = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    for s in data["streams"]:
        if s.get("codec_type") == "video":
            return s["width"], s["height"]
    raise RuntimeError(f"Cannot read video dimensions: {video_path}")


def format_srt_time(seconds: float) -> str:
    ms = int((seconds % 1) * 1000)
    return f"{int(seconds//3600):02d}:{int(seconds//60)%60:02d}:{int(seconds)%60:02d},{ms:03d}"


def parse_srt_time(ts: str) -> float:
    h, m, rest = ts.strip().split(":")
    s, ms = rest.split(",")
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000


def format_ass_time(seconds: float) -> str:
    cs = int((seconds % 1) * 100)
    return f"{int(seconds//3600)}:{int(seconds//60)%60:02d}:{int(seconds)%60:02d}.{cs:02d}"


def find_videos(path_or_glob: str) -> list[Path]:
    if any(c in path_or_glob for c in ('*', '?', '[')):
        return sorted(
            Path(m) for m in _glob.glob(path_or_glob)
            if Path(m).suffix.lower() in SUPPORTED_EXTENSIONS
        )
    p = Path(path_or_glob)
    if p.is_file():
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [p]
        print(f"❌ Unsupported file type: {p.suffix}")
        sys.exit(1)
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS)


def srt_name_for(video_path: Path) -> str:
    return video_path.with_suffix(".srt").name.replace(" ", "_")


def txt_name_for(video_path: Path) -> str:
    return video_path.with_suffix(".txt").name.replace(" ", "_")


# ── Segment splitting ─────────────────────────────────────────────────────────

def split_segment(start: float, end: float, text: str,
                  max_words: int) -> list[tuple[float, float, str]]:
    """Text-only split — used as fallback when word timestamps are unavailable."""
    words = text.split()
    if len(words) <= max_words:
        return [(start, end, text)]
    chunks = [words[i:i+max_words] for i in range(0, len(words), max_words)]
    duration, total, result, t = end - start, len(words), [], start
    for chunk in chunks:
        dur = duration * len(chunk) / total
        result.append((t, t + dur, " ".join(chunk)))
        t += dur
    return result


def split_segment_words(start: float, end: float, words: list[dict],
                        max_words: int) -> list[tuple[float, float, str, list[dict]]]:
    """
    Split a Whisper word list into chunks of at most max_words.
    Returns: [(chunk_start, chunk_end, chunk_text, chunk_words), ...]
    chunk_words: [{"word": str, "start": float, "end": float}, ...]
    """
    if len(words) <= max_words:
        text = " ".join(w["word"].strip() for w in words)
        return [(start, end, text, words)]

    chunks = []
    for i in range(0, len(words), max_words):
        cw = words[i:i + max_words]
        c_start = cw[0].get("start", start)
        c_end   = words[i + max_words]["start"] if i + max_words < len(words) else end
        c_text  = " ".join(w["word"].strip() for w in cw)
        chunks.append((c_start, c_end, c_text, cw))
    return chunks


# ── ASS generation ────────────────────────────────────────────────────────────

def parse_markup(text: str) -> list[tuple[str, bool]]:
    """
    Split text into (word, is_highlighted) pairs.
    Words inside *asterisks* are marked is_highlighted=True.
    """
    tokens = []
    for i, chunk in enumerate(re.split(r'\*', text)):
        for word in chunk.split():
            tokens.append((word, i % 2 == 1))
    return tokens


def ass_rounded_rect(bw: float, bh: float, r: float) -> str:
    """ASS \\p1 drawing path for a rounded rectangle from (0,0) to (bw, bh)."""
    k = 0.5523
    i = lambda v: round(v)
    x1, y1, x2, y2 = 0, 0, bw, bh
    return (
        f"m {i(x1+r)} {i(y1)} "
        f"l {i(x2-r)} {i(y1)} "
        f"b {i(x2-r+r*k)} {i(y1)} {i(x2)} {i(y1+r-r*k)} {i(x2)} {i(y1+r)} "
        f"l {i(x2)} {i(y2-r)} "
        f"b {i(x2)} {i(y2-r+r*k)} {i(x2-r+r*k)} {i(y2)} {i(x2-r)} {i(y2)} "
        f"l {i(x1+r)} {i(y2)} "
        f"b {i(x1+r-r*k)} {i(y2)} {i(x1)} {i(y2-r+r*k)} {i(x1)} {i(y2-r)} "
        f"l {i(x1)} {i(y1+r)} "
        f"b {i(x1)} {i(y1+r-r*k)} {i(x1+r-r*k)} {i(y1)} {i(x1+r)} {i(y1)}"
    )


def srt_to_ass(srt_path: Path, video_width: int, video_height: int,
               title: str = "", title_seconds: float = TITLE_MAX_SECONDS,
               face_vspan: "tuple[float, float] | None" = None) -> Path:
    """
    Convert SRT → ASS. Two bottom-anchored, centre-aligned layers:

      • Subtitle — single line of Karla body text, no pill, white with a black
        outline + drop shadow, shown for that line's full duration.
      • Title    — optional line in the turquoise League Spartan pill, directly
        above the subtitle. Shown from the *first frame* (cover, no fade-in)
        until `title_seconds`, then fades out. Nudged to avoid `face_vspan`.

    The block sits low (SAFE_BOTTOM_FRAC) so the title reads on the cover; the
    first ~COVER_HOLD_SECONDS stay subtitle-free.
    """
    ass_path = srt_path.with_suffix(".ass")

    # Title pill — League Spartan 800, tight leading + negative tracking (homepage H1).
    # `title_size` is nominal (drives body + margins); the title itself may shrink to fit.
    title_size = max(34, min(52, int(video_width * 0.041)))
    t_spacing  = round(TITLE_TRACKING * title_size)   # Style fallback
    box_w      = int(video_width * 0.90)

    # Subtitle body — Karla, no pill, thin outline + soft wide shadow
    body_size   = max(24, int(title_size * BODY_SCALE))
    body_line_h = int(body_size * 1.30)
    b_outline   = max(2, int(body_size * 0.10))
    b_shadow    = max(4, int(body_size * 0.16))
    b_blur      = max(2, int(body_size * 0.08))

    safe_bottom = int(video_height * SAFE_BOTTOM_FRAC)
    safe_top    = int(video_height * 0.10)

    cx = video_width // 2
    x1 = max(0, cx - box_w // 2)
    x2 = min(video_width, x1 + box_w)
    box_w = x2 - x1
    cx = x1 + box_w // 2

    pad_x    = int(title_size * 0.65)
    margin_l = x1 + pad_x
    margin_r = (video_width - x2) + pad_x
    max_text_width = box_w - 2 * pad_x

    body_bottom = safe_bottom - int(body_size * 0.25)   # breathing room at the edge
    body_cy     = body_bottom - body_line_h // 2
    body_top    = body_bottom - body_line_h

    def markup_parts(text: str) -> list[str]:
        return [f"{{\\c{YELLOW if emph else WHITE}}}{w}" for w, emph in parse_markup(text)]

    # Parse SRT
    subtitles = []
    for block in srt_path.read_text(encoding="utf-8").strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        start, end = (parse_srt_time(t) for t in lines[1].split(" --> "))
        if start < COVER_HOLD_SECONDS < end:      # keep the cover subtitle-free
            start = COVER_HOLD_SECONDS
        if start >= end:
            continue
        subtitles.append((start, end, " ".join(lines[2:])))

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            f"PlayResX: {video_width}\n"
            f"PlayResY: {video_height}\n"
            "WrapStyle: 1\n"
            "ScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Body,{BODY_FONT},{body_size},{WHITE},{NONE},{BLACK},{SHADOW},"
            f"0,0,0,0,100,100,0,0,1,{b_outline},{b_shadow},5,0,0,0,1\n"
            f"Style: TitleText,{TITLE_FONT},{title_size},{WHITE},{NONE},{NONE},{NONE},"
            f"0,0,0,0,100,100,{t_spacing},0,1,0,0,5,0,0,0,1\n"
            f"Style: TitleBG,{TITLE_FONT},{title_size},{TURQUOISE},{NONE},{NONE},{NONE},"
            f"0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        # ── Title (optional) ───────────────────────────────────────────────
        title = title.strip()
        if title and title_seconds > 0:
            # rough glyph advance for League Spartan 800 ≈ 0.50·title_size
            words = title.split()
            est_w = len(title) * title_size * 0.50
            ts    = title_size
            floor = max(24, int(title_size * 0.60))
            while ts > floor and (est_w * ts / title_size) / max_text_width > 2.0:
                ts = max(floor, int(ts * 0.92))
            tpad, tlh   = int(ts * 0.34), int(ts * TITLE_LINE_HEIGHT)
            tcorn, tsp  = int(ts * 0.40), round(TITLE_TRACKING * ts)

            n_lines = min(3, max(1, -(-int(est_w * ts / title_size) // max_text_width)))
            if n_lines > 1 and len(words) > 1:
                per = -(-len(words) // n_lines)
                tlines = [" ".join(words[i:i + per]) for i in range(0, len(words), per)]
            else:
                tlines = [title]

            t_box_h = ts + (len(tlines) - 1) * tlh + 2 * tpad
            corner  = min(tcorn, t_box_h // 2 - 2)
            gap = int(ts * 0.35)
            ty2 = body_top - gap
            ty1 = ty2 - t_box_h

            if face_vspan:
                fy0   = face_vspan[0] * video_height
                fy1   = face_vspan[1] * video_height
                fcrit = fy0 + 0.55 * (fy1 - fy0)   # eyes/mouth — chin may be grazed
                m     = int(ts * 0.35)
                if ty1 < fcrit + m and ty2 > fy0 - m:
                    below_y1 = int(fy1 + m)
                    above_y2 = int(fy0 - m)
                    if below_y1 + t_box_h <= body_top - gap:
                        ty1, ty2 = below_y1, below_y1 + t_box_h
                    elif above_y2 - t_box_h >= safe_top:
                        ty1, ty2 = above_y2 - t_box_h, above_y2
                    else:
                        ty1, ty2 = safe_top, safe_top + t_box_h

            ty1 = max(ty1, safe_top)                           # running position (above body)
            cover_ty1 = max(ty1, safe_bottom - t_box_h - int(ts * 0.15))
            t_shape = ass_rounded_rect(box_w, t_box_h, corner)
            fade   = f"\\fad(0,{TITLE_FADE_OUT_MS})"           # no fade-in → visible on frame 0
            fs     = f"\\fs{ts}\\fsp{tsp}"
            t1, t2 = TITLE_COVER_MS, TITLE_COVER_MS + TITLE_MOVE_MS
            s0, s1 = format_ass_time(0.0), format_ass_time(title_seconds)
            f.write(
                f"Dialogue: 0,{s0},{s1},TitleBG,,0,0,0,,"
                f"{{\\an7\\move({x1},{cover_ty1},{x1},{ty1},{t1},{t2})\\p1{fade}}}"
                f"{t_shape}{{\\p0}}\n"
            )
            base_cy = tpad + int(ts * 0.52)
            for i, line in enumerate(tlines):
                dy = base_cy + i * tlh
                f.write(
                    f"Dialogue: 1,{s0},{s1},TitleText,,{margin_l},{margin_r},0,,"
                    f"{{\\an5\\move({cx},{cover_ty1 + dy},{cx},{ty1 + dy},{t1},{t2})"
                    f"{fs}{fade}}}{' '.join(markup_parts(line))}\n"
                )

        # ── Subtitles (body text, no pill) ─────────────────────────────────
        for start, end, text in subtitles:
            s_bg, e_bg = format_ass_time(start), format_ass_time(end)
            f.write(
                f"Dialogue: 0,{s_bg},{e_bg},Body,,{margin_l},{margin_r},0,,"
                f"{{\\an5\\pos({cx},{body_cy})\\blur{b_blur}}}{' '.join(markup_parts(text))}\n"
            )

    return ass_path


# ── Pipeline ──────────────────────────────────────────────────────────────────

def transcribe(video_path: Path, output_dir: Path, model_name: str,
               language: Optional[str], max_words: int,
               force: bool = False, force_all: bool = False) -> Path:
    """
    Transcribe with Whisper (word_timestamps=True).
    Writes a .srt file and a .words.json sidecar for per-word highlighting.
    """
    import whisper
    srt_path   = output_dir / srt_name_for(video_path)
    words_path = srt_path.with_suffix(".words.json")

    if srt_path.exists() and not force and not force_all:
        print(f"⏭️  Skipping (SRT exists): {srt_path.name}")
        return srt_path

    print(f"🎙  Transcribing: {video_path.name}  (model={model_name})")
    model   = whisper.load_model(model_name)
    options = {"task": "transcribe", "word_timestamps": True}
    if language:
        options["language"] = language

    result     = model.transcribe(str(video_path), **options)
    idx        = 1
    words_data = []

    with open(srt_path, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            raw_words = seg.get("words", [])

            if raw_words:
                chunks = split_segment_words(
                    seg["start"], seg["end"], raw_words, max_words
                )
            else:
                chunks = [
                    (s, e, t, [])
                    for s, e, t in split_segment(seg["start"], seg["end"],
                                                 seg["text"].strip(), max_words)
                ]

            for c_start, c_end, c_text, c_words in chunks:
                f.write(f"{idx}\n"
                        f"{format_srt_time(c_start)} --> {format_srt_time(c_end)}\n"
                        f"{c_text}\n\n")
                words_data.append({
                    "index": idx,
                    "start": c_start,
                    "end":   c_end,
                    "text":  c_text,
                    "words": c_words,
                })
                idx += 1

    words_path.write_text(json.dumps(words_data, indent=2), encoding="utf-8")

    # Plain-text transcript — one subtitle per line.
    # Never overwrite an existing TXT so manual edits are preserved.
    txt_path  = srt_path.with_suffix(".txt")
    fresh_txt = "\n".join(entry["text"].strip() for entry in words_data) + "\n"
    if not txt_path.exists():
        txt_path.write_text(fresh_txt, encoding="utf-8")
        print(f"✏️  TXT saved:    {txt_path.name}  ← edit this, then run --step apply")
    elif force_all:
        stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = txt_path.with_name(f"{txt_path.stem}_backup_{stamp}.txt")
        shutil.copy2(txt_path, backup)
        txt_path.write_text(fresh_txt, encoding="utf-8")
        print(f"✏️  TXT replaced: {txt_path.name}  (backup → {backup.name})")
    else:
        print(f"✏️  TXT kept:     {txt_path.name}  (not overwritten — use --forceall to replace)")

    print(f"✅ SRT saved:    {srt_path.name}  ({idx - 1} lines, word timing included)")
    return srt_path


def build_srt_from_transcript(video_path: Path, output_dir: Path,
                              transcripts_dir: Path,
                              force: bool = False) -> Path:
    """
    Build .srt and .words.json in output_dir from the shared transcript.
    Skips if SRT already exists and not force.
    Does not create a local .txt — transcript.txt lives in transcripts/{stem}/.
    """
    srt_path   = output_dir / srt_name_for(video_path)
    words_path = srt_path.with_suffix(".words.json")
    src_words  = transcripts_dir / video_path.stem / "transcript.words.json"

    if srt_path.exists() and not force:
        print(f"⏭️  SRT exists: {srt_path.name}")
        return srt_path

    if not src_words.exists():
        print(f"❌ Transcript not found: {src_words}")
        print(f"   Run the prepare step first.")
        sys.exit(1)

    words_data = json.loads(src_words.read_text(encoding="utf-8"))
    with open(srt_path, "w", encoding="utf-8") as f:
        for entry in words_data:
            f.write(f"{entry['index']}\n"
                    f"{format_srt_time(entry['start'])} --> {format_srt_time(entry['end'])}\n"
                    f"{entry['text']}\n\n")

    words_path.write_text(json.dumps(words_data, indent=2), encoding="utf-8")
    print(f"✅ SRT built from transcript: {srt_path.name}")
    return srt_path


def apply_transcript(video_path: Path, output_dir: Path,
                     transcripts_dir: Optional[Path] = None) -> Path:
    """
    Apply an edited .txt file back to the .srt, preserving all timestamps.
    The .txt must have the same number of paragraphs as the .srt has entries.
    Looks for transcript.txt in transcripts_dir/{stem}/ first, then locally.
    """
    srt_path = output_dir / srt_name_for(video_path)

    # Prefer shared transcript.txt; fall back to local copy
    shared_txt = (transcripts_dir / video_path.stem / "transcript.txt"
                  if transcripts_dir else None)
    if shared_txt and shared_txt.exists():
        txt_path = shared_txt
    else:
        txt_path = output_dir / txt_name_for(video_path)

    if not srt_path.exists():
        print(f"❌ SRT not found: {srt_path}  (run --step transcribe first)")
        sys.exit(1)
    if not txt_path.exists():
        print(f"❌ TXT not found: {txt_path}  (run --step transcribe first)")
        sys.exit(1)

    srt_entries = []
    for block in srt_path.read_text(encoding="utf-8").strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        srt_entries.append(lines[1])

    # Parse edited TXT (one subtitle entry per line)
    txt_entries = [line.strip() for line in
                   txt_path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]

    if len(txt_entries) != len(srt_entries):
        print(f"❌ Mismatch: TXT has {len(txt_entries)} entries, "
              f"SRT has {len(srt_entries)} — they must match line-for-line.")
        print("   Tip: do not add or remove lines — one line per subtitle entry.")
        sys.exit(1)

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (timing, text) in enumerate(zip(srt_entries, txt_entries), 1):
            f.write(f"{i}\n{timing}\n{text}\n\n")

    print(f"✅ SRT updated:  {srt_path.name}  ({len(srt_entries)} entries)")
    return srt_path


def burn_subtitles(video_path: Path, srt_path: Path, output_dir: Path,
                   force: bool = False, title: str = "") -> Path:
    out_path = output_dir / video_path.name
    if out_path.exists() and not force:
        print(f"⏭️  Skipping (video exists): {out_path.name}")
        return out_path
    width, height = get_video_dimensions(video_path)
    duration      = get_video_duration(video_path)
    face_vspan    = titlekit.detect_face_vspan(video_path) if title.strip() else None
    if title.strip():
        where = f"face at {face_vspan[0]:.2f}–{face_vspan[1]:.2f}" if face_vspan else "no face detected"
        print(f"🏷  Title: “{title.strip()}”  ({where})")
    ass_path = srt_to_ass(
        srt_path, width, height,
        title=title,
        title_seconds=min(TITLE_MAX_SECONDS, duration) if duration else TITLE_MAX_SECONDS,
        face_vspan=face_vspan,
    )

    # Copy ASS to a temp path with no special characters so ffmpeg's
    # filter parser doesn't trip over commas, spaces, or # in the filename.
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".ass")
    os.close(tmp_fd)
    tmp_ass = Path(tmp_name)
    shutil.copy2(ass_path, tmp_ass)
    try:
        ass_str = str(tmp_ass).replace("\\", "/")
        vf = f"ass=f={ass_str}"
        fontsdir = titlekit.ass_fontsdir()
        if fontsdir:
            vf += f":fontsdir={fontsdir}"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", vf,
            "-c:a", "copy",
            str(out_path),
        ]
        print(f"🎬 Burning subs: {out_path.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ ffmpeg error:\n", result.stderr[-800:])
            sys.exit(1)
        print(f"✅ Video saved:  {out_path.name}")
    finally:
        tmp_ass.unlink(missing_ok=True)
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate subtitles for horizontal videos using Whisper + ffmpeg"
    )
    parser.add_argument("--input",     default="video_c_sources")
    parser.add_argument("--output",    default="video_d_subtitles")
    parser.add_argument("--step",      default="all",
                        choices=["transcribe", "apply", "burn", "all"],
                        help="transcribe = SRT+TXT | apply = apply edited TXT to SRT | "
                             "burn = render video | all = transcribe+burn")
    parser.add_argument("--model",     default="base",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--lang",      default=None,
                        help="Force language, e.g. 'en', 'de' (auto-detect if omitted)")
    parser.add_argument("--max-words", type=int, default=8,
                        help="Max words per subtitle line (default: 8)")
    parser.add_argument("--force",    action="store_true",
                        help="Reprocess existing outputs (never overwrites TXT)")
    parser.add_argument("--forceall", action="store_true",
                        help="Reprocess everything + overwrite TXT (backup created first)")
    parser.add_argument("--transcripts", default="transcripts",
                        help="Shared transcripts folder from prepare step (default: transcripts)")
    args = parser.parse_args()

    force     = args.force or args.forceall
    force_all = args.forceall
    transcripts_dir = Path(args.transcripts)

    check_dependencies(need_whisper=(args.step in ("transcribe", "all")))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = find_videos(args.input)
    if not videos:
        print("❌ No supported video files found.")
        sys.exit(1)

    print(f"\n📂 Found {len(videos)} video(s)  [step={args.step}]\n{'─'*40}")
    for video in videos:
        print(f"\n▶  {video}")
        video_dir = output_dir / video.stem
        video_dir.mkdir(parents=True, exist_ok=True)

        # Evaluate txt-changed state NOW, before any SRT rebuild that would reset its mtime.
        # When --force is set, build_srt_from_transcript() writes a fresh SRT whose mtime
        # is always newer than the txt — so the normal mtime check would silently drop edits.
        shared_txt    = transcripts_dir / video.stem / "transcript.txt"
        local_txt     = video_dir / txt_name_for(video)
        existing_srt  = video_dir / srt_name_for(video)
        txt_has_edits = any(
            txt.exists() and existing_srt.exists()
            and txt.stat().st_mtime > existing_srt.stat().st_mtime
            for txt in [shared_txt, local_txt]
        ) or (force and any(txt.exists() for txt in [shared_txt, local_txt]))

        # Title layer — transcripts/<stem>/title.txt (default = cleaned file name)
        titlekit.ensure_title_file(transcripts_dir, video.stem)
        title = titlekit.load_title(transcripts_dir, video.stem)
        burned = video_dir / video.name
        title_changed = (
            burned.exists()
            and titlekit.title_mtime(transcripts_dir, video.stem) > burned.stat().st_mtime
        )

        if args.step == "apply":
            srt = apply_transcript(video, video_dir, transcripts_dir)

        elif args.step in ("transcribe", "all"):
            shared = transcripts_dir / video.stem / "transcript.words.json"
            if shared.exists():
                srt = build_srt_from_transcript(video, video_dir, transcripts_dir,
                                                force=force)
            else:
                srt = transcribe(video, video_dir, args.model, args.lang, args.max_words,
                                 force=force, force_all=force_all)

        else:  # burn
            srt = video_dir / srt_name_for(video)
            if not srt.exists():
                print(f"❌ SRT not found: {srt}  (run --step transcribe first)")
                sys.exit(1)

        if args.step in ("burn", "all"):
            if txt_has_edits:
                print(f"📝 TXT is newer than SRT — applying edits first...")
                srt = apply_transcript(video, video_dir, transcripts_dir)
            if title_changed and not (force or txt_has_edits):
                print(f"🏷  title.txt is newer than the burned video — re-burning...")
            burn_subtitles(video, srt, video_dir,
                           force=force or txt_has_edits or title_changed,
                           title=title)

    print(f"\n🎉 Done! Processed {len(videos)} video(s).")


if __name__ == "__main__":
    main()
