# ReelPrc — Video & Reel Processing Pipeline

Automated pipeline for preparing, subtitling, and finalising social media videos.
Supports two formats: **vertical reels** (9:16, 720 px wide) and **horizontal videos** (16:9, 720 px tall).

---

## Requirements

- Python 3.9+
- [ffmpeg + ffprobe](https://ffmpeg.org/download.html)
- [openai-whisper](https://github.com/openai/whisper) — `pip install openai-whisper`
- **numpy** — `pip install numpy` _(required by `prepare_videos.py`)_
- [opencv-python](https://pypi.org/project/opencv-python/) — `pip install opencv-python` _(face detection for reel crop + thumbnail scoring; falls back to centre crop / unscored if missing)_
- **mediapipe** — `pip install mediapipe` _(optional: face-expression scoring for thumbnails)_
- **Pillow** — `pip install pillow` _(optional: line-count prediction for reel subtitle/title wrapping)_
- Fonts, all bundled in `assets/` and passed to ffmpeg via `fontsdir` (also
  installed to `~/Library/Fonts` on this machine): **League Spartan ExtraBold**
  (title — a weight-800 instance of the variable font, `assets/LeagueSpartan/`)
  and **Karla** (subtitle body, `assets/Karla/`)

Pinned versions are in `requirements.txt`. The pipeline currently runs under the
macOS **system `python3`** (3.9) with its user site-packages. The `.venv/` in
this folder is incomplete and OneDrive-synced — don't use it; rebuild from
`requirements.txt` **outside** any cloud-synced folder if you need a venv
(on-demand file hydration otherwise makes `import whisper` take many minutes).

---

## Folder Structure

### Reels (vertical)

```
reel_a_originals/       ← drop raw recordings here
reel_b_outro/           ← drop a single outro clip here
reel_c_sources/         ← prepared clips (audio-optimised, 720 px wide)
reel_d_subtitles/<name>/ ← one folder per clip: <name>.srt / .ass / .words.json + burned <name>.<ext>
reel_e_final/           ← subtitled clip + outro concatenated
reel_a_thumbnails/       ← candidate thumbnail frames (staging/ + approved/)
```

### Videos (horizontal)

```
video_a_originals/       ← drop raw recordings here
video_b_outro/           ← drop a single outro clip here
video_c_sources/         ← prepared clips (audio-optimised, 720 px tall)
video_d_subtitles/<name>/ ← one folder per clip: <name>.srt / .ass / .words.json + burned <name>.<ext>
video_e_final/           ← subtitled clip + outro concatenated
video_a_thumbnails/       ← candidate thumbnail frames (staging/ + approved/)
```

### Shared

```
transcripts/<name>/
    transcript.words.json   ← word timing (generated)
    transcript.txt          ← subtitle text — edit this!
    title.txt               ← on-screen title — edit this! (empty = no title)
```

The `transcripts/` folder is the single source of truth for subtitle + title
text. It is written by the **prepare** step and keyed by the clip's file name
(stem), so a horizontal video and its auto-cropped reel share one transcript
and one title.

> **Auto reel crop:** The prepare step also produces a portrait reel version of each horizontal video (720×1280) and saves it to `reel_c_sources/`. Face detection is used to centre the crop on the speaker. Use `--no-reel` to skip this.
>
> Because the reel crop shares the horizontal video's transcript, its subtitles use the **video** `--max-words` (8), not the reel default (5).

---

## Quick Start

Drop your raw recordings into `reel_a_originals/` or `video_a_originals/`, then run the full pipeline:

```bash
python3 process_reels.py    # reels
python3 process_videos.py   # horizontal videos
```

By default, **existing outputs are skipped** — only new files are processed.

To see what's outstanding first (new clips, plus any where `transcript.txt` or
`title.txt` changed since the last burn):

```bash
python3 pending.py          # add --json for machine-readable output
```

---

## Pipeline Steps

Each pipeline runs three steps in order:

| Step | Script | What it does |
|------|--------|--------------|
| **1 — prepare** | `prepare_reels.py` / `prepare_videos.py` | Audio optimisation + loudness normalisation + scale to target resolution · **Whisper transcription + `title.txt` → `transcripts/<name>/`** · thumbnail extraction · (videos) portrait reel crop |
| **2 — subtitles** | `subtitle_reels.py` / `subtitle_videos.py` | Build SRT from the shared transcript (re-applying any edited `transcript.txt`) + burn subtitles **and the title** into the video. Falls back to running Whisper itself only if no shared transcript exists. |
| **3 — final** | _(built into process script)_ | Concatenate subtitled clip + outro |

> Whisper runs in **step 1**, not step 2. `--model` / `--lang` therefore only
> affect transcription when passed to the prepare step (the `process_*.py`
> wrappers forward them for you).

### Run a single step

```bash
python3 process_reels.py --step prepare     # step 1 only
python3 process_reels.py --step subtitles   # step 2 only
python3 process_reels.py --step final       # step 3 only
```

---

## Force Flags

By default all steps skip files that already exist.

**Reprocess flags:**

| Flag         | Effect |
|--------------|--------|
| _(none)_     | Skip any output that already exists |
| `--force`    | Reprocess all outputs; **never** overwrites manually edited `transcript.txt` |
| `--forceall` | Reprocess all outputs **and** refresh `transcript.txt` — a timestamped backup (`transcript_backup_YYYYMMDD_HHMMSS.txt`) is created first |

**Skip flags:**

| Flag         | Effect |
|--------------|--------|
| `--no-outro` | Skip the outro step, even when running `--step all` |
| `--no-reel`  | Skip portrait reel crop during the prepare step _(videos only)_ |

```bash
python3 process_reels.py --force          # re-run everything, keep edited transcripts
python3 process_reels.py --forceall       # re-run everything, refresh transcripts (backup created)
python3 process_reels.py --no-outro       # full pipeline without appending the outro
python3 process_videos.py --no-reel       # skip reel crop during prepare
```

---

## Editing Transcripts

After the **prepare** step, each clip gets a plain-text transcript at:

```
transcripts/myvideo/transcript.txt
```

One line per subtitle entry. You can fix typos, punctuation, or emphasis here. The next time you run the pipeline, an edited transcript (newer than the burned SRT) is automatically applied back to the `.srt` before burning — no manual `--step apply` needed.

**`transcript.txt` is never overwritten unless you explicitly use `--forceall`** (which backs it up first).

### Word emphasis

Wrap a word in `*asterisks*` to highlight it in **yellow** — works in both the subtitles and the title:

```
This is a *really* important point.
```

---

## Titles

Each clip can show a **title** in a turquoise League Spartan pill, sitting
directly above the subtitle. It is visible from the **first frame** (so it
works as the cover / thumbnail — the first ~0.3 s stay subtitle-free) through
**10 seconds**, then fades out. The whole block sits low in the frame and the
title is nudged clear of the speaker's face.

Titles live next to the transcript:

```
transcripts/myvideo/title.txt      ← one line of text
```

- Auto-created on the **prepare** step, pre-filled with a title derived from the
  file name (camera timestamps like `, am 30.08.26 um 21.34` are stripped).
- Edit it to change the on-screen title. An **empty file = no title**.
- Shared by a horizontal video and its portrait reel crop (same file name).
- Editing it re-burns the clip on the next run (same as editing `transcript.txt`).

---

## Whisper Options

```bash
python3 process_reels.py --model large --lang en
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `base` | Whisper model size: `tiny` `base` `small` `medium` `large` |
| `--lang` | _(auto)_ | Force language, e.g. `en`, `de`, `fr` |
| `--max-words` | `5` (reels) / `8` (videos) | Max words per subtitle line — **fixed when the transcript is first created**; changing it later has no effect unless you re-transcribe with `--forceall` |

Larger models are slower but more accurate. Use `large` for tricky accents or technical vocabulary.

All three options are consumed by the **prepare** step. Run it (or the full
`process_*.py` pipeline) with the options set — passing them only to
`subtitle_*.py` when a shared transcript already exists does nothing.

---

## Subtitle & Title Style

| | Subtitle (body) | Title (first 10 s) |
|---|---|---|
| Font | **Karla** regular | **League Spartan ExtraBold** (weight 800, tight leading + `-0.035em` tracking — matches the homepage H1) |
| Background | none — white text, thin outline + soft blurred shadow | turquoise rounded pill |
| Size | ~82 % of the title | reels ≈ 60 px · videos ≈ 35 px |
| Text | white (`*emphasis*` → yellow) | white (`*emphasis*` → yellow) |
| Karaoke word-highlight | no | no |
| Position | low, ~92 % of frame height | directly above the subtitle |
| Lines | reels grow to fit · videos 1 | reels ≤ 2 · videos 1 |

The title shows from the first frame with **no fade-in** (so the cover frame
carries it) and fades out at 10 s. It is nudged vertically to avoid the
speaker's eyes/mouth (Haar-cascade detection, sampled at burn time — falls back
to a safe-zone clamp if opencv is missing or no face is found).

---

## Running Steps Individually

The individual scripts can also be called directly for more control:

```bash
# Prepare only (audio/video + Whisper transcript + thumbnails)
python3 prepare_reels.py --force
python3 prepare_reels.py --model large --lang en    # transcription options live here

# Build SRT from the shared transcript
python3 subtitle_reels.py --step transcribe

# Apply edited transcript.txt back to SRT manually
python3 subtitle_reels.py --step apply

# Burn subtitles only
python3 subtitle_reels.py --step burn

# Same commands work for videos
python3 prepare_videos.py --model large --lang en
python3 subtitle_videos.py --step transcribe
python3 subtitle_videos.py --step burn --force
```

---

## Supported Input Formats

`.mp4` `.mov` `.mkv` `.avi` `.webm` `.m4v`
