# FlyWheel — Video & Reel Processing Pipeline

Automated pipeline that turns raw talking-head recordings into subtitled social
clips. Two formats: **vertical reels** (9:16, 720 px wide) and **horizontal
videos** (16:9, 720 px tall). `README.md` has the user-facing docs; this file is
the operator's cheat sheet.

## Environment

- **Run with system `python3`** (`/usr/bin/python3`, currently 3.9.6). Its
  user site-packages (`~/Library/Python/3.9/lib/python/site-packages`) has the
  full set: whisper, torch, opencv-python, mediapipe, pillow, numpy.
  `ffmpeg`/`ffprobe` are on PATH (homebrew).
- **Ignore `./.venv/`.** It's an incomplete Python 3.13 env (whisper + torch
  only — no cv2/mediapipe/pillow) living in OneDrive, where "online-only" files
  make `import whisper` hang for 10+ minutes on first use. Running the pipeline
  through it silently degrades output (centre-crop instead of face-crop,
  unscored thumbnails, fixed 2-line subtitle pills). Delete it, or rebuild from
  `requirements.txt` **outside** OneDrive.
- `requirements.txt` pins the known-good versions.
- macOS. `git` yes, `gh` no. Remote: `github.com/SiliconGarden/FlyWheel`.

## How the pipeline is wired

Two wrappers, `process_reels.py` and `process_videos.py`, each run 3 steps:

1. **prepare** (`prepare_reels.py` / `prepare_videos.py`)
   - ffmpeg: voice audio chain + loudness norm + scale → `*_c_sources/`
   - **Whisper transcription** → `transcripts/<stem>/transcript.words.json` + `transcript.txt`
   - **`transcripts/<stem>/title.txt`** — created with a cleaned-filename default (see `titlekit.py`)
   - thumbnail extraction → `*_a_thumbnails/staging/`
   - videos only: face-centred portrait crop → `reel_c_sources/<name>`
2. **subtitles** (`subtitle_reels.py` / `subtitle_videos.py`)
   - builds `.srt` + `.ass` **from the shared transcript** (`build_srt_from_transcript`),
     re-applying an edited `transcript.txt` if it's newer than the SRT, then burns
     the subtitles **and the title** (`srt_to_ass` emits both layers)
   - only falls back to running Whisper itself if `transcripts/<stem>/` is missing
   - output: `*_d_subtitles/<stem>/` (one folder per clip)
   - re-burns when `transcript.txt` **or** `title.txt` is newer than the burned clip
3. **final** — appends `*_b_outro/`'s clip → `*_e_final/`

**Whisper runs in step 1, not step 2.** `--model` / `--lang` / `--max-words`
only matter to the *prepare* step. The wrappers forward them; calling
`subtitle_*.py` with them when a transcript already exists does nothing.

`transcripts/` is keyed by filename stem and **shared** between a horizontal
video and its reel crop — so a video-derived reel inherits the video's
`--max-words` (8, not the reel default 5). `--max-words` is frozen when the
transcript is first written; changing it later needs `--forceall`.

## Common commands

```bash
python3 pending.py                       # list clips needing prepare/subtitle/re-burn/final
python3 process_reels.py                 # full reel pipeline (skips existing)
python3 process_videos.py                # full horizontal pipeline
python3 process_reels.py --step prepare  # just prepare (→ transcripts to review)
python3 process_videos.py --model large --lang de
python3 process_reels.py --force         # redo outputs, keep edited transcript.txt
python3 process_reels.py --forceall      # redo + refresh transcript.txt (backs it up)
python3 process_reels.py --no-outro
```

Editing transcripts: fix `transcripts/<name>/transcript.txt` (one line per
subtitle entry — never add/remove lines). `*word*` → yellow emphasis in the
burn. Re-running the pipeline auto-applies edits before burning.

Titles: `transcripts/<name>/title.txt`, one line, empty = no title. Rendered in
the turquoise pill above the subtitle for the first 10 s, kept off the face.
Subtitles are white on a ~75 %-opaque black pill (no karaoke word-highlight).
Geometry + face-avoidance live in each `srt_to_ass`; text helpers in `titlekit.py`.

## Folder map

```
reel_a_originals/ video_a_originals/   raw drops
reel_b_outro/     video_b_outro/       single outro clip
reel_c_sources/   video_c_sources/     prepared clips (reel_c also holds video-derived crops)
reel_d_subtitles/<stem>/  video_d_subtitles/<stem>/   .srt/.ass/.words.json + burned clip
reel_e_final/     video_e_final/       clip + outro
reel_a_thumbnails/ video_a_thumbnails/ staging/ + approved/
transcripts/<stem>/  transcript.words.json + transcript.txt + title.txt   ← subtitle/title source of truth
titlekit.py          title-file helpers (default_title / ensure_title_file / load_title) + detect_face_vspan
pending.py           read-only: what still needs prepare / subtitle / re-burn / final (also --json)
social_assets/ social_posts/  social_post.py / social_linkedin.py  (separate "social" tooling)
```

## Git

- Never commit media. `.gitignore` keeps only scripts + structure (`.gitkeep`
  markers) + `README.md`/`CLAUDE.md`/`.claude/settings.json`.
- `transcripts/` contents are gitignored too (regenerate-able + hand-edited);
  flip that in `.gitignore` if the manual corrections should be versioned.
- Branch off `main` for changes; `main` tracks `origin/main`.

## Known rough edges (not yet fixed)

- `extract_thumbnails` has two divergent implementations (reels: staging-only;
  videos: manifest + approved/ + rejected-timestamps). Fixes don't port between them.
- Wrappers can't pass `--no-thumbnails` / `--no-transcribe` through to prepare.
- `subtitle_*.py` require `whisper` importable for `--step all` even when a
  shared transcript makes it unused.
