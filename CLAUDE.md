# FlyWheel — Video & Reel Processing Pipeline

Automated pipeline that turns raw talking-head recordings into subtitled social
clips. Two formats: **vertical reels** (9:16, 720 px wide) and **horizontal
videos** (16:9, 720 px tall). `README.md` has the user-facing docs; this file is
the operator's cheat sheet.

## Environment

- **Always use `.venv/bin/python`**, never bare `python3` (system Python lacks
  whisper/mediapipe).
- `.venv/` currently lives inside this OneDrive folder and its files are
  "online-only" → `import whisper` can hang for 10+ minutes on first use while
  OneDrive hydrates ~23k torch files. If a run sits at 0 % CPU during import,
  that's why. Fix: hydrate the venv (open it in Finder / "Always keep on this
  device") or rebuild it **outside** OneDrive. Deps: `openai-whisper`, `numpy`,
  `opencv-python`, `mediapipe`, `pillow`, plus system `ffmpeg`/`ffprobe`.
- macOS. `git` yes, `gh` no. Remote: `github.com/SiliconGarden/FlyWheel`.

## How the pipeline is wired

Two wrappers, `process_reels.py` and `process_videos.py`, each run 3 steps:

1. **prepare** (`prepare_reels.py` / `prepare_videos.py`)
   - ffmpeg: voice audio chain + loudness norm + scale → `*_c_sources/`
   - **Whisper transcription** → `transcripts/<stem>/transcript.words.json` + `transcript.txt`
   - thumbnail extraction → `*_a_thumbnails/staging/`
   - videos only: face-centred portrait crop → `reel_c_sources/<name>`
2. **subtitles** (`subtitle_reels.py` / `subtitle_videos.py`)
   - builds `.srt` + `.ass` **from the shared transcript** (`build_srt_from_transcript`),
     re-applying an edited `transcript.txt` if it's newer than the SRT, then burns subs
   - only falls back to running Whisper itself if `transcripts/<stem>/` is missing
   - output: `*_d_subtitles/<stem>/` (one folder per clip)
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
.venv/bin/python process_reels.py                 # full reel pipeline (skips existing)
.venv/bin/python process_videos.py                # full horizontal pipeline
.venv/bin/python process_reels.py --step prepare  # just prepare (→ transcripts to review)
.venv/bin/python process_videos.py --model large --lang de
.venv/bin/python process_reels.py --force         # redo outputs, keep edited transcript.txt
.venv/bin/python process_reels.py --forceall      # redo + refresh transcript.txt (backs it up)
.venv/bin/python process_reels.py --no-outro
```

Editing transcripts: fix `transcripts/<name>/transcript.txt` (one line per
subtitle entry — never add/remove lines). `*word*` → yellow emphasis in the
burn. Re-running the pipeline auto-applies edits before burning.

## Folder map

```
reel_a_originals/ video_a_originals/   raw drops
reel_b_outro/     video_b_outro/       single outro clip
reel_c_sources/   video_c_sources/     prepared clips (reel_c also holds video-derived crops)
reel_d_subtitles/<stem>/  video_d_subtitles/<stem>/   .srt/.ass/.words.json + burned clip
reel_e_final/     video_e_final/       clip + outro
reel_a_thumbnails/ video_a_thumbnails/ staging/ + approved/
transcripts/<stem>/  transcript.words.json + transcript.txt   ← subtitle source of truth
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
