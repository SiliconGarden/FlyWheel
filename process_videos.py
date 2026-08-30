#!/usr/bin/env python3
"""
process_videos.py — Full horizontal video pipeline: prepare → subtitle → final (+ outro).

Folders:
    video_a_originals/  raw recordings
    video_b_outro/      outro clip(s) to append
    video_c_sources/    prepared (audio-optimised, 1280×720)
    video_d_subtitles/  subtitled videos
    video_e_final/      subtitled + outro concatenated

Usage:
    python3 process_videos.py                   # full pipeline
    python3 process_videos.py --step prepare    # prepare only
    python3 process_videos.py --step subtitles  # subtitle only
    python3 process_videos.py --step final      # append outro only
    python3 process_videos.py --force           # overwrite existing prepared videos
    python3 process_videos.py --model large --lang en
"""

import argparse
import glob as _glob
import subprocess
import sys
from pathlib import Path

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def run(cmd: list[str]):
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def find_videos(path_or_glob: str) -> list[Path]:
    if any(c in path_or_glob for c in ('*', '?', '[')):
        return sorted(
            Path(m) for m in _glob.glob(path_or_glob)
            if Path(m).suffix.lower() in SUPPORTED_EXTENSIONS
        )
    p = Path(path_or_glob)
    if p.is_file():
        return [p] if p.suffix.lower() in SUPPORTED_EXTENSIONS else []
    if not p.exists():
        return []
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS)


def _step_input(user_input: "str | None", default_folder: str) -> str:
    """Map the user's --input to the equivalent path for a downstream pipeline step."""
    if user_input is None:
        return default_folder
    p = Path(user_input)
    if p.is_dir():
        return default_folder
    # Extract filename (or filename-glob) part and attach the step's folder
    return str(Path(default_folder) / p.name)


def append_outro(subtitled_dir: str, outro_dir: str, final_dir: Path,
                 force: bool = False):
    outros = find_videos(outro_dir)
    if not outros:
        print(f"⚠️  No outro found in {outro_dir} — skipping.")
        return
    outro = outros[0]

    videos = find_videos(subtitled_dir)
    if not videos:
        print(f"⚠️  No videos in {subtitled_dir} — skipping.")
        return

    final_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📂 Appending outro to {len(videos)} video(s)\n{'─' * 40}")
    for video in videos:
        out_path = final_dir / video.name
        if out_path.exists() and not force:
            print(f"⏭️  Skipping (exists): {out_path.name}")
            continue
        print(f"\n▶  {video.name}")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video), "-i", str(outro),
            "-filter_complex", "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ ffmpeg error:\n", result.stderr[-800:])
            sys.exit(1)
        print(f"✅ Final saved: {out_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Full horizontal video pipeline: prepare + subtitles + append outro"
    )
    parser.add_argument("--step", default="all",
                        choices=["all", "prepare", "subtitles", "final"],
                        help="all = prepare+subtitles+final | prepare | subtitles | final")
    parser.add_argument("--force",    action="store_true",
                        help="Reprocess existing outputs (never overwrites TXT)")
    parser.add_argument("--forceall", action="store_true",
                        help="Reprocess everything + overwrite TXT (backup created first)")
    parser.add_argument("--model",  default="base",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--lang",   default=None,
                        help="Force language, e.g. 'en', 'de' (auto-detect if omitted)")
    parser.add_argument("--max-words", type=int, default=5,
                        help="Max words per subtitle line (default: 5)")
    parser.add_argument("--no-outro", action="store_true",
                        help="Skip the outro step even when running --step all")
    parser.add_argument("--no-reel", action="store_true",
                        help="Skip portrait reel crop during the prepare step")
    parser.add_argument("--input", default=None,
                        help="Single file, glob pattern, or folder to process "
                             "(default: full originals folder)")
    args = parser.parse_args()
    force     = args.force or args.forceall
    force_all = args.forceall

    if args.step in ("all", "prepare"):
        print("━" * 40)
        print("  Step 1 / 3 — Prepare")
        print("━" * 40)
        cmd = [sys.executable, "prepare_videos.py"]
        if force:
            cmd.append("--force")
        if args.no_reel:
            cmd.append("--no-reel")
        if args.input:
            cmd += ["--input", args.input]
        run(cmd)

    if args.step in ("all", "subtitles"):
        print()
        print("━" * 40)
        print("  Step 2 / 3 — Subtitles")
        print("━" * 40)
        cmd = [sys.executable, "subtitle_videos.py",
               "--model", args.model,
               "--max-words", str(args.max_words),
               "--input", _step_input(args.input, "video_c_sources")]
        if args.lang:
            cmd += ["--lang", args.lang]
        if force_all:
            cmd.append("--forceall")
        elif force:
            cmd.append("--force")
        run(cmd)

    if args.step in ("all", "final") and not args.no_outro:
        print()
        print("━" * 40)
        print("  Step 3 / 3 — Final (+ outro)")
        print("━" * 40)
        append_outro(
            subtitled_dir=_step_input(args.input, "video_d_subtitles"),
            outro_dir="video_b_outro",
            final_dir=Path("video_e_final"),
            force=force,
        )

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
