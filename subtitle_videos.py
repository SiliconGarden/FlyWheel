#!/usr/bin/env python3
"""
subtitle_videos.py — Automated subtitle generator for horizontal videos.
Uses OpenAI Whisper for transcription and ffmpeg for burning subtitles.

Workflow:
    1. python subtitle_videos.py --step transcribe   # → .srt  .words.json  .txt
    2.  edit  video_subtitles/<name>.txt             # fix typos, punctuation, etc.
    3. python subtitle_videos.py --step apply        # apply edits back to .srt
    4. python subtitle_videos.py --step burn         # render video

    python subtitle_videos.py                        # shortcut: transcribe + burn (skips step 2-3)
    python subtitle_videos.py --model large --lang fr
    python subtitle_videos.py --max-words 5
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


# ── Config ────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
FONT_NAME = "League Spartan"

# ASS colors (ABGR format: &HAABBGGRR)
TURQUOISE = "&H1EE6E15C"   # #5CE1E6 — pill background, 88 % opaque (matches social posts)
BLACK     = "&H00000000"   # active / spoken word
WHITE     = "&H00FFFFFF"   # inactive words
YELLOW    = "&H005BF6FF"   # #fff65b — *asterisk* emphasis
NONE      = "&H00000000"


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


def srt_to_ass(srt_path: Path, video_width: int, video_height: int) -> Path:
    """
    Convert SRT → ASS with turquoise rounded-rectangle backgrounds.
    If a .words.json sidecar exists, active words are highlighted in black
    as they are spoken (karaoke-style). Otherwise plain white text is used.

    Each subtitle produces:
      Layer 0 — turquoise rounded-rect background (full line duration)
      Layer 1 — text events:
                  • with sidecar: one event per word showing active word in black
                  • without sidecar: one event per line in white
    """
    ass_path = srt_path.with_suffix(".ass")

    # Layout — horizontal video uses smaller font (~half reel size) to fit more
    # text across the wider frame while remaining legible on YouTube.
    font_size = max(28, min(44, int(video_width * 0.033)))
    pad_y     = int(font_size * 0.45)
    line_h    = int(font_size * 1.35)
    box_w     = int(video_width * 0.90)
    box_h     = line_h + 2 * pad_y   # single line
    corner_r  = int(font_size * 0.40)

    # YouTube safe area: bottom of box sits at 93 %, leaving a 7 % margin
    # above the player UI. Box is constrained to the bottom 10 % of the frame.
    safe_bottom = int(video_height * 0.90)
    safe_top_of_third = int(video_height * 0.86)

    cx = video_width // 2
    # Place box flush against safe bottom, with a small breathing gap
    y2 = safe_bottom - int(font_size * 0.20)
    y1 = y2 - box_h
    # If box top exceeds the bottom-third boundary, clamp it down
    if y1 < safe_top_of_third:
        y1 = safe_top_of_third
        y2 = y1 + box_h
    x1 = max(0, cx - box_w // 2)
    x2 = min(video_width, x1 + box_w)
    box_w, box_h = x2 - x1, y2 - y1
    cx = x1 + box_w // 2
    cy = y1 + box_h // 2

    bg_shape = ass_rounded_rect(box_w, box_h, corner_r)

    # Internal horizontal padding keeps text away from the pill edges
    pad_x    = int(font_size * 0.65)
    margin_l = x1 + pad_x
    margin_r = (video_width - x2) + pad_x

    # Load word timing sidecar (generated by --step transcribe)
    words_path = srt_path.with_suffix(".words.json")
    words_lookup: dict[int, list[dict]] = {}
    if words_path.exists():
        for entry in json.loads(words_path.read_text(encoding="utf-8")):
            if entry.get("words"):
                words_lookup[entry["index"]] = entry["words"]

    # Parse SRT (capture index for sidecar lookup)
    subtitles = []
    for block in srt_path.read_text(encoding="utf-8").strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        idx   = int(lines[0])
        start, end = (parse_srt_time(t) for t in lines[1].split(" --> "))
        text  = " ".join(lines[2:])
        subtitles.append((idx, start, end, text))

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
            f"Style: Text,{FONT_NAME},{font_size},{WHITE},{NONE},{NONE},{NONE},"
            f"-1,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1\n"
            f"Style: BG,{FONT_NAME},{font_size},{TURQUOISE},{NONE},{NONE},{NONE},"
            f"0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        pos_tag = f"{{\\an5\\pos({cx},{cy})}}"

        for idx, start, end, text in subtitles:
            # Layer 0: turquoise pill background (full subtitle duration)
            s_bg, e_bg = format_ass_time(start), format_ass_time(end)
            f.write(
                f"Dialogue: 0,{s_bg},{e_bg},BG,,0,0,0,,"
                f"{{\\an7\\pos({x1},{y1})\\p1}}{bg_shape}{{\\p0}}\n"
            )

            # Parse markup: *word* → yellow; plain words → white
            tokens = parse_markup(text)   # [(word, is_highlighted), ...]
            display_words = [w for w, _ in tokens]

            def word_color(j: int, active_k: int, is_marked: bool) -> str:
                if is_marked:
                    return YELLOW
                return BLACK if j == active_k else WHITE

            line_words = words_lookup.get(idx)
            if line_words:
                n_disp, n_timed = len(display_words), len(line_words)

                if n_disp == n_timed:
                    timed = [
                        {"start": lw.get("start", start), "end": lw.get("end", end)}
                        for lw in line_words
                    ]
                else:
                    # Word count changed — spread timing evenly
                    duration = end - start
                    timed = [
                        {"start": start + i * duration / n_disp,
                         "end":   start + (i + 1) * duration / n_disp}
                        for i in range(n_disp)
                    ]

                # Layer 1: one event per word — active=black, inactive=white, marked=yellow
                for k, wi in enumerate(timed):
                    w_start = wi["start"]
                    w_end   = timed[k + 1]["start"] if k < len(timed) - 1 else end
                    s, e_w  = format_ass_time(w_start), format_ass_time(w_end)
                    parts = [
                        f"{{\\c{word_color(j, k, marked)}}}{word}"
                        for j, (word, marked) in enumerate(tokens)
                    ]
                    f.write(
                        f"Dialogue: 1,{s},{e_w},Text,,{margin_l},{margin_r},0,,"
                        f"{pos_tag}{' '.join(parts)}\n"
                    )
            else:
                # No word timing — show line with markup colors (no karaoke)
                parts = [
                    f"{{\\c{YELLOW if marked else WHITE}}}{word}"
                    for word, marked in tokens
                ]
                f.write(
                    f"Dialogue: 1,{s_bg},{e_bg},Text,,{margin_l},{margin_r},0,,"
                    f"{pos_tag}{' '.join(parts)}\n"
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
                   force: bool = False) -> Path:
    out_path = output_dir / video_path.name
    if out_path.exists() and not force:
        print(f"⏭️  Skipping (video exists): {out_path.name}")
        return out_path
    width, height = get_video_dimensions(video_path)
    ass_path = srt_to_ass(srt_path, width, height)

    # Copy ASS to a temp path with no special characters so ffmpeg's
    # filter parser doesn't trip over commas, spaces, or # in the filename.
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".ass")
    os.close(tmp_fd)
    tmp_ass = Path(tmp_name)
    shutil.copy2(ass_path, tmp_ass)
    try:
        ass_str = str(tmp_ass).replace("\\", "/")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"ass=f={ass_str}",
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
            shared_txt = transcripts_dir / video.stem / "transcript.txt"
            local_txt  = video_dir / txt_name_for(video)
            txt_changed = any(
                txt.exists() and srt.exists() and txt.stat().st_mtime > srt.stat().st_mtime
                for txt in [shared_txt, local_txt]
            )
            if txt_changed:
                print(f"📝 TXT is newer than SRT — applying edits first...")
                srt = apply_transcript(video, video_dir, transcripts_dir)
            burn_subtitles(video, srt, video_dir, force=force or txt_changed)

    print(f"\n🎉 Done! Processed {len(videos)} video(s).")


if __name__ == "__main__":
    main()
