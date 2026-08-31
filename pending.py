#!/usr/bin/env python3
"""
pending.py — list clips that still need work in the reel / video pipelines.

Read-only. Scans both tracks and reports, per clip, the earliest pipeline step
that is missing or stale:

  prepare    raw original not yet in *_c_sources/
  subtitle   prepared clip has no burned subtitle output yet
  re-burn    transcript.txt (vs the SRT) or title.txt (vs the burned clip) is
             newer  → subtitles and the final must be rebuilt
  final      burned clip not concatenated with the outro yet

Matches what a plain `process_reels.py` / `process_videos.py` run would actually
redo — outputs that merely have an older mtime than an input are NOT flagged,
because the pipeline only rebuilds those with `--force`.

Usage:
    python3 pending.py            # human-readable list + suggested commands
    python3 pending.py --json     # machine-readable
"""

import argparse
import json
import sys
from pathlib import Path

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

TRANSCRIPTS = Path("transcripts")

TRACKS = [
    {
        "name": "Reels",
        "command": "python3 process_reels.py",
        "originals": Path("reel_a_originals"),
        "sources":   Path("reel_c_sources"),
        "subtitles": Path("reel_d_subtitles"),
        "final":     Path("reel_e_final"),
        "outro":     Path("reel_b_outro"),
    },
    {
        "name": "Videos",
        "command": "python3 process_videos.py",
        "originals": Path("video_a_originals"),
        "sources":   Path("video_c_sources"),
        "subtitles": Path("video_d_subtitles"),
        "final":     Path("video_e_final"),
        "outro":     Path("video_b_outro"),
    },
]

# earliest-first, so a clip is reported at the step it's actually blocked on
STEP_ORDER = ["prepare", "subtitle", "re-burn", "final"]


def clips_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def srt_name_for(name: Path) -> str:
    return name.with_suffix(".srt").name.replace(" ", "_")


def analyse_track(track: dict) -> list[dict]:
    originals_dir = track["originals"]
    sources_dir   = track["sources"]
    subtitles_dir = track["subtitles"]
    final_dir     = track["final"]

    original_names = {c.name for c in clips_in(originals_dir)}
    pending: list[dict] = []

    # 1) raw originals not prepared yet
    for clip in clips_in(originals_dir):
        if not (sources_dir / clip.name).exists():
            pending.append({"step": "prepare", "clip": clip.name, "note": ""})

    # 2) prepared clips (incl. video-derived reel crops) — subtitle / re-burn / final
    for clip in clips_in(sources_dir):
        stem = clip.stem
        note = ("from video" if track["name"] == "Reels"
                and clip.name not in original_names else "")

        burned    = subtitles_dir / stem / clip.name
        srt       = subtitles_dir / stem / srt_name_for(clip)
        t_txt     = TRANSCRIPTS / stem / "transcript.txt"
        ti_txt    = TRANSCRIPTS / stem / "title.txt"
        final_out = final_dir / clip.name

        if not burned.exists():
            pending.append({"step": "subtitle", "clip": clip.name, "note": note})
        elif mtime(t_txt) > mtime(srt):
            pending.append({"step": "re-burn", "clip": clip.name, "note": "transcript edited"})
        elif mtime(ti_txt) > mtime(burned):
            pending.append({"step": "re-burn", "clip": clip.name, "note": "title changed"})
        elif not final_out.exists():
            pending.append({"step": "final", "clip": clip.name, "note": note})

    pending.sort(key=lambda e: (STEP_ORDER.index(e["step"]), e["clip"].lower()))
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    report = {t["name"]: analyse_track(t) for t in TRACKS}
    total  = sum(len(v) for v in report.values())

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        sys.exit(0)

    if total == 0:
        print("✅ Everything is up to date — nothing to produce.")
        sys.exit(0)

    for track in TRACKS:
        entries = report[track["name"]]
        if not entries:
            continue
        print(f"\n━━━━━━ {track['name']} ━━━━━━")
        for e in entries:
            note = f"   ({e['note']})" if e["note"] else ""
            print(f"  {e['step']:<9} {e['clip']}{note}")

    print(f"\n── {total} clip(s) pending ──")
    for track in TRACKS:
        if report[track["name"]]:
            print(f"  {track['name']:<7} {track['command']}")
    print("  (add --no-outro to skip the final step)")


if __name__ == "__main__":
    main()
