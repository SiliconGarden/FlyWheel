#!/usr/bin/env python3
"""
prepare_reels.py — Preprocess raw recordings before subtitle generation.

Takes videos from reel_a_originals/, applies voice audio optimisation and
loudness normalisation, scales to 720 px wide, and saves to reel_c_sources/.
Also transcribes each clip with Whisper into the shared transcripts/{stem}/
folder and extracts candidate thumbnails into reel_a_thumbnails/staging/.

Usage:
    python3 prepare_reels.py
    python3 prepare_reels.py --input reel_a_originals --output reel_c_sources
    python3 prepare_reels.py --force       # overwrite existing outputs (keeps edited TXT)
    python3 prepare_reels.py --forceall    # also refresh transcript.txt (backup first)
    python3 prepare_reels.py --no-transcribe --no-thumbnails   # audio/video only

Audio chain:
    1. highpass=f=80          remove low-frequency rumble
    2. acompressor            gentle dynamic compression for voice
    3. loudnorm=I=-16         EBU R128 normalisation to -16 LUFS
"""

import argparse
import glob as _glob
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import titlekit


# ── Config ─────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

# Audio filter chain — tuned for voice/speech content
AUDIO_FILTERS = ",".join([
    "highpass=f=80",                                          # remove rumble
    "acompressor=threshold=-18dB:ratio=3:attack=5:release=50:makeup=2",  # compress
    "loudnorm=I=-16:TP=-1.5:LRA=11",                         # EBU R128 -16 LUFS
])


# ── Helpers ────────────────────────────────────────────────────────────────────

def check_dependencies():
    errors = []
    for tool in ("ffmpeg", "ffprobe"):
        if subprocess.run([tool, "-version"], capture_output=True).returncode != 0:
            errors.append(f"{tool}  →  https://ffmpeg.org/download.html")
    if errors:
        print("❌ Missing dependencies:\n")
        for e in errors:
            print(f"   {e}")
        sys.exit(1)


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


def get_video_info(video_path: Path) -> dict:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", str(video_path)]
    data = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    info = {}
    for s in data["streams"]:
        if s.get("codec_type") == "video":
            info["width"]  = s["width"]
            info["height"] = s["height"]
    return info


def get_video_duration(video_path: Path) -> float:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", str(video_path)]
    data = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    return float(data["format"]["duration"])


def _score_candidates(candidates: "list[tuple[Path, float]]") -> "list[tuple[Path, float, float]]":
    """Score each candidate frame by face quality (primary) and sharpness (secondary).

    Returns list of (path, timestamp, combined_score) sorted best-first.
    Face score rewards open eyes and a forward-facing head.
    Sharpness (Laplacian variance) is used as a secondary signal.
    Frames with no detected face are ranked below those with a face.
    """
    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    # Import face scoring from social_linkedin (same model already cached on disk)
    try:
        from social_linkedin import _init_face_landmarker, _face_scores
        face_mesh = _init_face_landmarker()
    except Exception:
        face_mesh = None

    results = []
    sharpness_values = []

    for path, t in candidates:
        if not has_cv2:
            results.append((path, t, 0.0, 1.0))   # (path, t, face, sharpness)
            sharpness_values.append(1.0)
            continue

        img = cv2.imread(str(path))
        if img is None:
            results.append((path, t, 0.0, 0.0))
            sharpness_values.append(0.0)
            continue

        h, w = img.shape[:2]
        gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_values.append(sharpness)

        if face_mesh is not None:
            sc  = _face_scores(face_mesh, img, w, h)
            ear = sc["ear"]
            yaw = sc["yaw"]
            # Reward open eyes + forward-facing head; -1 means no face detected
            face_score = ear * max(0.1, 1.0 - 2.0 * yaw) if ear >= 0 else -1.0
        else:
            face_score = 0.0

        results.append((path, t, face_score, sharpness))

    if face_mesh is not None:
        face_mesh.close()

    # Normalise sharpness to [0, 1] so it combines cleanly with face score
    max_sharp = max(sharpness_values, default=1.0) or 1.0
    scored = []
    for path, t, face, sharp in results:
        sharp_n = sharp / max_sharp
        if face >= 0:
            # Face detected: face quality is primary (80 %), sharpness secondary (20 %)
            combined = face * 0.8 + sharp_n * 0.2
        else:
            # No face: sharpness only, capped below any frame that has a face
            combined = sharp_n * 0.2
        scored.append((path, t, combined))

    return sorted(scored, key=lambda x: x[2], reverse=True)


def extract_thumbnails(video_path: Path, thumbnails_dir: Path,
                       count: int = 20, oversample: int = 5,
                       force: bool = False) -> None:
    """Extract `count` quality thumbnails by sampling `count * oversample` candidate
    frames, scoring each by face expression quality (primary) and sharpness (secondary),
    and keeping the top `count` re-ordered chronologically.
    """
    import shutil
    import tempfile

    staging = thumbnails_dir / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    existing = list(staging.glob(f"{stem}_thumb_*.jpg"))
    if existing and not force:
        print(f"  ⏭️  Thumbnails exist: {stem}  ({len(existing)} frames) — skipping")
        return

    duration = get_video_duration(video_path)
    margin   = min(1.0, duration * 0.05)
    span     = duration - 2 * margin
    n_cand   = count * oversample
    times    = [margin + span * i / (n_cand - 1) for i in range(n_cand)]

    print(f"  📸 Sampling {n_cand} candidate frames for: {stem}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path   = Path(tmp)
        candidates: list[tuple[Path, float]] = []
        for i, t in enumerate(times):
            out = tmp_path / f"cand_{i:04d}.jpg"
            cmd = ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(video_path),
                   "-vframes", "1", "-q:v", "2", str(out)]
            if subprocess.run(cmd, capture_output=True).returncode == 0 and out.exists():
                candidates.append((out, t))

        print(f"  🔍 Scoring {len(candidates)} frames (face expression + sharpness) …")
        scored   = _score_candidates(candidates)
        selected = sorted(scored[:count], key=lambda x: x[1])  # re-sort chronologically

        for old in staging.glob(f"{stem}_thumb_*.jpg"):
            old.unlink()
        for i, (src, _, _) in enumerate(selected, 1):
            shutil.copy2(src, staging / f"{stem}_thumb_{i:02d}.jpg")

    n = len(selected)
    print(f"  ✅ {n} thumbnail(s) → {staging}  (face+sharpness scored, {n}/{len(candidates)} kept)")


# ── Transcription ──────────────────────────────────────────────────────────────

def split_segment(start: float, end: float, text: str,
                  max_words: int) -> list[tuple[float, float, str]]:
    words = text.split()
    if len(words) <= max_words:
        return [(start, end, text)]
    chunks = [words[i:i + max_words] for i in range(0, len(words), max_words)]
    duration, total, result, t = end - start, len(words), [], start
    for chunk in chunks:
        dur = duration * len(chunk) / total
        result.append((t, t + dur, " ".join(chunk)))
        t += dur
    return result


def split_segment_words(start: float, end: float, words: list[dict],
                        max_words: int) -> list[tuple[float, float, str, list[dict]]]:
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


def transcribe_to_folder(video_path: Path, transcripts_dir: Path,
                         model_name: str, language: Optional[str],
                         max_words: int, force: bool = False,
                         force_all: bool = False) -> None:
    """
    Transcribe video audio with Whisper → transcripts/{stem}/.
    Writes transcript.words.json and transcript.txt.
    - Skips entirely if transcript.words.json already exists and neither
      force nor force_all is set.
    - transcript.txt is never overwritten unless force_all is set, in which
      case a timestamped backup is created first.
    This handles the shared case: if prepare_videos.py already transcribed the
    stem, this call is a no-op (unless forced).
    """
    import shutil
    import whisper
    from datetime import datetime
    stem       = video_path.stem
    out        = transcripts_dir / stem
    out.mkdir(parents=True, exist_ok=True)
    words_path = out / "transcript.words.json"
    txt_path   = out / "transcript.txt"

    if words_path.exists() and not force and not force_all:
        print(f"  ⏭️  Transcript exists: {stem}  — skipping")
        return

    print(f"  🎙  Transcribing: {video_path.name}  (model={model_name})")
    model   = whisper.load_model(model_name)
    options = {"task": "transcribe", "word_timestamps": True}
    if language:
        options["language"] = language

    result     = model.transcribe(str(video_path), **options)
    words_data = []
    idx        = 1
    for seg in result["segments"]:
        raw_words = seg.get("words", [])
        if raw_words:
            chunks = split_segment_words(seg["start"], seg["end"], raw_words, max_words)
        else:
            chunks = [(s, e, t, []) for s, e, t in
                      split_segment(seg["start"], seg["end"], seg["text"].strip(), max_words)]
        for c_start, c_end, c_text, c_words in chunks:
            words_data.append({"index": idx, "start": c_start, "end": c_end,
                               "text": c_text, "words": c_words})
            idx += 1

    words_path.write_text(json.dumps(words_data, indent=2), encoding="utf-8")

    fresh_txt = "\n".join(e["text"].strip() for e in words_data) + "\n"
    if not txt_path.exists():
        txt_path.write_text(fresh_txt, encoding="utf-8")
        print(f"  ✏️  TXT created: {txt_path}  ← edit this to fix transcription errors")
    elif force_all:
        stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = txt_path.with_name(f"{txt_path.stem}_backup_{stamp}.txt")
        shutil.copy2(txt_path, backup)
        txt_path.write_text(fresh_txt, encoding="utf-8")
        print(f"  ✏️  TXT replaced: {txt_path}  (backup → {backup.name})")
    else:
        print(f"  ✏️  TXT kept:    {txt_path}  (not overwritten — use --forceall to replace)")

    print(f"  ✅ Transcript: {stem}/  ({idx - 1} entries)")


# ── Processing ─────────────────────────────────────────────────────────────────

def process_video(video_path: Path, output_dir: Path, force: bool) -> Path:
    out_path = output_dir / video_path.name

    if out_path.exists() and not force:
        print(f"⏭️  Skipping (exists): {out_path.name}  — use --force to overwrite")
        return out_path

    info = get_video_info(video_path)
    w, h = info.get("width", "?"), info.get("height", "?")
    print(f"🔊 Processing: {video_path.name}  ({w}×{h})")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", "scale=720:-2",          # 720 px wide, height keeps aspect ratio
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-af", AUDIO_FILTERS,
        "-c:a", "aac",
        "-b:a", "192k",
        str(out_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ ffmpeg error:\n", result.stderr[-800:])
        sys.exit(1)

    out_info = get_video_info(out_path)
    ow, oh = out_info.get("width", "?"), out_info.get("height", "?")
    print(f"✅ Saved: {out_path.name}  ({ow}×{oh})")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw reels: voice optimisation, loudness normalisation, 720 px"
    )
    parser.add_argument("--input",  default="reel_a_originals",
                        help="Folder with raw source videos (default: reel_a_originals)")
    parser.add_argument("--output", default="reel_c_sources",
                        help="Output folder (default: reel_c_sources)")
    parser.add_argument("--force",  action="store_true",
                        help="Overwrite existing output files (never overwrites edited TXT)")
    parser.add_argument("--forceall", action="store_true",
                        help="Like --force, but also refresh the shared transcript.txt "
                             "(a timestamped backup is created first)")
    parser.add_argument("--no-transcribe",  action="store_true",
                        help="Skip Whisper transcription (audio/video processing only)")
    parser.add_argument("--transcripts",    default="transcripts",
                        help="Output folder for shared transcripts (default: transcripts)")
    parser.add_argument("--model",  default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--lang",   default=None,
                        help="Force transcription language, e.g. 'en', 'de' (auto-detect if omitted)")
    parser.add_argument("--max-words", type=int, default=5,
                        help="Max words per subtitle line (default: 5)")
    parser.add_argument("--thumbnails", default="reel_a_thumbnails",
                        help="Output folder for thumbnails (default: reel_a_thumbnails)")
    parser.add_argument("--thumb-count", type=int, default=20,
                        help="Number of thumbnail frames to keep per video (default: 20)")
    parser.add_argument("--thumb-oversample", type=int, default=5,
                        help="Candidate multiplier for sharpness filtering: "
                             "extracts count×oversample frames, keeps sharpest count (default: 5)")
    parser.add_argument("--no-thumbnails", action="store_true",
                        help="Skip thumbnail extraction")
    args = parser.parse_args()
    force     = args.force or args.forceall
    force_all = args.forceall

    check_dependencies()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcripts_dir = None if args.no_transcribe else Path(args.transcripts)
    if transcripts_dir:
        transcripts_dir.mkdir(parents=True, exist_ok=True)

    thumbnails_dir = None if args.no_thumbnails else Path(args.thumbnails)

    videos = find_videos(args.input)
    if not videos:
        print("❌ No supported video files found.")
        sys.exit(1)

    print(f"\n📂 Found {len(videos)} video(s)\n{'─' * 40}")
    for video in videos:
        print(f"\n▶  {video}")
        out_path = process_video(video, output_dir, force)
        if transcripts_dir and out_path.exists():
            transcribe_to_folder(out_path, transcripts_dir,
                                 args.model, args.lang, args.max_words,
                                 force=force, force_all=force_all)
            titlekit.ensure_title_file(transcripts_dir, video.stem)
        if thumbnails_dir and out_path.exists():
            extract_thumbnails(out_path, thumbnails_dir,
                               args.thumb_count, args.thumb_oversample, force)

    print(f"\n🎉 Done! Processed {len(videos)} video(s).")


if __name__ == "__main__":
    main()
