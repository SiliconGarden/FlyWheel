# ReelPrc — Video & Reel Processing Pipeline

Automated pipeline for preparing, subtitling, and finalising social media videos.
Supports two formats: **vertical reels** (9:16, 720 px wide) and **horizontal videos** (16:9, 720 px tall).

---

## Requirements

- Python 3.9+
- [ffmpeg + ffprobe](https://ffmpeg.org/download.html)
- [openai-whisper](https://github.com/openai/whisper) — `pip install openai-whisper`
- [opencv-python](https://pypi.org/project/opencv-python/) — `pip install opencv-python` _(face detection for reel crop; falls back to centre crop if missing)_
- Font: **League Spartan** installed on the system

---

## Folder Structure

### Reels (vertical)

```
reel_a_originals/   ← drop raw recordings here
reel_b_outro/       ← drop a single outro clip here
reel_c_sources/     ← prepared clips (audio-optimised, 720 px wide)
reel_d_subtitles/   ← subtitled clips + .srt / .txt / .words.json sidecars
reel_e_final/       ← subtitled clip + outro concatenated
```

### Videos (horizontal)

```
video_a_originals/  ← drop raw recordings here
video_b_outro/      ← drop a single outro clip here
video_c_sources/    ← prepared clips (audio-optimised, 1280×720)
video_d_subtitles/  ← subtitled clips + .srt / .txt / .words.json sidecars
video_e_final/      ← subtitled clip + outro concatenated
```

> **Auto reel crop:** The prepare step also produces a portrait reel version of each horizontal video (720×1280) and saves it to `reel_c_sources/`. Face detection is used to centre the crop on the speaker. Use `--no-reel` to skip this.

---

## Quick Start

Drop your raw recordings into `reel_a_originals/` or `video_a_originals/`, then run the full pipeline:

```bash
python3 process_reels.py    # reels
python3 process_videos.py   # horizontal videos
```

By default, **existing outputs are skipped** — only new files are processed.

---

## Pipeline Steps

Each pipeline runs three steps in order:

| Step | Script | What it does |
|------|--------|--------------|
| **1 — prepare** | `prepare_reels.py` / `prepare_videos.py` | Audio optimisation + loudness normalisation + scale to target resolution |
| **2 — subtitles** | `subtitle_reels.py` / `subtitle_videos.py` | Whisper transcription → SRT + burn subtitles into video |
| **3 — final** | _(built into process script)_ | Concatenate subtitled clip + outro |

### Run a single step

```bash
python3 process_reels.py --step prepare     # step 1 only
python3 process_reels.py --step subtitles   # step 2 only
python3 process_reels.py --step final       # step 3 only
```

---

## Force Flags

By default all steps skip files that already exist. Use these flags to reprocess:

| Flag         | Effect |
|--------------|--------|
| _(none)_     | Skip any output that already exists |
| `--force`    | Reprocess all outputs; **never** overwrites manually edited `.txt` transcript files |
| `--forceall` | Reprocess all outputs **and** overwrite `.txt` files — a timestamped backup (`name_backup_YYYYMMDD_HHMMSS.txt`) is created first |
| `--no-outro` | Skip the outro step, even when running `--step all` |
| `--no-reel` | Skip portrait reel crop during the prepare step _(videos only)_ |

```bash
python3 process_reels.py --force          # re-run everything, keep edited transcripts
python3 process_reels.py --forceall       # re-run everything, refresh transcripts (backup created)
python3 process_reels.py --no-outro       # full pipeline without appending the outro
python3 process_videos.py --no-reel       # skip reel crop during prepare
```

---

## Editing Transcripts

After the subtitle step, each video gets a plain-text transcript in the output folder:

```
reel_d_subtitles/myvideo.txt
```

One line per subtitle entry. You can fix typos, punctuation, or emphasis here. The next time you run the pipeline, edited transcripts are automatically applied back to the `.srt` before burning — no manual `--step apply` needed.

**The `.txt` file is never overwritten unless you explicitly use `--forceall`.**

### Word emphasis

Wrap a word in `*asterisks*` to highlight it in **yellow** in the burned subtitles:

```
This is a *really* important point.
```

---

## Whisper Options

```bash
python3 process_reels.py --model large --lang en
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `base` | Whisper model size: `tiny` `base` `small` `medium` `large` |
| `--lang` | _(auto)_ | Force language, e.g. `en`, `de`, `fr` |
| `--max-words` | `5` (reels) / `8` (videos) | Max words per subtitle line |

Larger models are slower but more accurate. Use `large` for tricky accents or technical vocabulary.

---

## Subtitle Style

- **Font:** League Spartan
- **Background:** turquoise rounded pill
- **Active (spoken) word:** black
- **Inactive words:** white
- **`*Emphasised*` words:** yellow
- Reels: two lines of text, larger font
- Videos: single line, smaller font, YouTube safe area

---

## Running Steps Individually

The individual scripts can also be called directly for more control:

```bash
# Prepare only
python3 prepare_reels.py --force

# Transcribe only
python3 subtitle_reels.py --step transcribe --model large

# Apply edited TXT back to SRT manually
python3 subtitle_reels.py --step apply

# Burn subtitles only
python3 subtitle_reels.py --step burn

# Same commands work for videos
python3 subtitle_videos.py --step transcribe
python3 subtitle_videos.py --step burn --force
```

---

## Supported Input Formats

`.mp4` `.mov` `.mkv` `.avi` `.webm` `.m4v`
