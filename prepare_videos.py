#!/usr/bin/env python3
"""
prepare_videos.py — Preprocess raw horizontal recordings before subtitle generation.

Takes videos from video_a_originals/, applies voice audio optimisation and
loudness normalisation, scales to 720 px tall (landscape 1280×720), and
saves to video_c_sources/.

Also produces a portrait reel crop (720×1280) saved to reel_c_sources/ by
detecting the speaker's face position in sampled frames. Falls back to centre
crop if no face is found or opencv is not installed.

Usage:
    python3 prepare_videos.py
    python3 prepare_videos.py --force       # overwrite existing outputs
    python3 prepare_videos.py --no-reel     # skip portrait reel output

Audio chain:
    1. highpass=f=80          remove low-frequency rumble
    2. acompressor            gentle dynamic compression for voice
    3. loudnorm=I=-16         EBU R128 normalisation to -16 LUFS
"""

import argparse
import glob as _glob
import json
import numpy
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ── Config ─────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

# Audio filter chain — tuned for voice/speech content
AUDIO_FILTERS = ",".join([
    "highpass=f=80",                                          # remove rumble
    "acompressor=threshold=-18dB:ratio=3:attack=5:release=50:makeup=2",  # compress
    "loudnorm=I=-16:TP=-1.5:LRA=11",                         # EBU R128 -16 LUFS
])

# Reel crop config
REEL_OUTPUT_DIR    = "reel_c_sources"
REEL_SAMPLE_FRAMES = 10

# Thumbnail extraction config
THUMBNAILS_OUTPUT_DIR      = "video_a_thumbnails"
REEL_THUMBNAILS_OUTPUT_DIR = "reel_a_thumbnails"
THUMBNAIL_COUNT       = 20   # thumbnails to keep per video
THUMBNAIL_SAMPLE      = 50   # candidate frames to evaluate

# MediaPipe expression-quality thresholds
THUMB_EAR_MIN = 0.20   # eye aspect ratio below this → eyes closed → discard
THUMB_MAR_MAX = 0.55   # mouth aspect ratio above this → too wide open → penalise
THUMB_YAW_MAX = 0.28   # nose-to-cheek asymmetry above this → head turned → penalise

# MediaPipe Face Mesh landmark indices for EAR (6-point method per eye)
_MP_LEFT_EYE  = [33, 160, 158, 133, 153, 144]
_MP_RIGHT_EYE = [362, 385, 387, 263, 380, 373]

# MediaPipe face landmarker model (Tasks API) — downloaded once, then cached
_LANDMARKER_CACHE = Path.home() / ".cache" / "reelprc" / "face_landmarker.task"
_LANDMARKER_URL   = (
    "https://storage.googleapis.com/mediapipe-models"
    "/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


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
            info["width"]    = s["width"]
            info["height"]   = s["height"]
            info["duration"] = float(s.get("duration", 0))
    return info


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
        print(f"  ⏭️  Transcript exists: {stem}  — skipping (use --force to redo)")
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


# ── Face detection ─────────────────────────────────────────────────────────────

def detect_face_crop_x(
    video_path: Path,
    src_width: int,
    src_height: int,
    duration: float,
    n_frames: int = REEL_SAMPLE_FRAMES,
) -> "int | None":
    """
    Sample n_frames evenly from the video, run Haar cascade face detection,
    and return the average horizontal centre of detected faces (source pixels).
    Returns None if no faces are found or if opencv is unavailable.
    """
    try:
        import cv2
    except ImportError:
        print("  ⚠️  opencv-python not installed — using centre crop for reel.")
        return None

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)

    start = duration * 0.05
    end   = duration * 0.95
    timestamps = [
        start + i * (end - start) / (n_frames - 1)
        for i in range(n_frames)
    ]

    centres = []
    for ts in timestamps:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{ts:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-f", "image2pipe",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 or not result.stdout:
            continue
        expected = src_width * src_height * 3
        if len(result.stdout) < expected:
            continue
        buf   = numpy.frombuffer(result.stdout[:expected], dtype=numpy.uint8)
        frame = buf.reshape((src_height, src_width, 3))
        grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            grey, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        for (x, y, w, h) in faces:
            centres.append(x + w // 2)

    if not centres:
        return None
    return int(sum(centres) / len(centres))


def make_reel_vfilter(src_width: int, src_height: int, face_cx: "int | None") -> str:
    """Return an ffmpeg -vf string that crops a 9:16 portrait window and scales to 720 wide."""
    crop_h = src_height
    crop_w = int(crop_h * 9 / 16)
    max_x  = src_width - crop_w

    if face_cx is not None:
        crop_x = max(0, min(face_cx - crop_w // 2, max_x))
    else:
        crop_x = (src_width - crop_w) // 2

    return f"crop={crop_w}:{crop_h}:{crop_x}:0,scale=720:1280"


def produce_reel(video_path: Path, reel_dir: Path, src_info: dict, force: bool) -> None:
    """Detect face position and write a 720×1280 portrait crop to reel_dir."""
    reel_path = reel_dir / video_path.name

    if reel_path.exists() and not force:
        print(f"  ⏭️  Reel exists: {reel_path.name}  — use --force to overwrite")
        return

    src_w    = src_info["width"]
    src_h    = src_info["height"]
    duration = src_info.get("duration", 0)

    print(f"  🔍 Detecting face position…")
    face_cx = detect_face_crop_x(video_path, src_w, src_h, duration)

    if face_cx is not None:
        print(f"  👤 Face centre at x={face_cx}px")
    else:
        print(f"  ↔️  No face detected — using centre crop")

    vf = make_reel_vfilter(src_w, src_h, face_cx)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-af", AUDIO_FILTERS,
        "-c:a", "aac",
        "-b:a", "192k",
        str(reel_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  ❌ Reel ffmpeg error:\n", result.stderr[-800:])
        return  # non-fatal — landscape output already succeeded

    reel_info = get_video_info(reel_path)
    rw, rh = reel_info.get("width", "?"), reel_info.get("height", "?")
    print(f"  ✅ Reel saved: {reel_path.name}  ({rw}×{rh})")


# ── Thumbnail extraction ────────────────────────────────────────────────────────

def _face_quality(face_mesh, frame_bgr: numpy.ndarray, img_w: int, img_h: int) -> float:
    """
    Score a frame's expression quality using MediaPipe Face Mesh.
    Returns a multiplier in [0, 1]: 0 = discard, 1 = no penalty.
    Penalises closed eyes, wide-open mouth, and strong head turns.
    """
    import cv2, mediapipe as mp

    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return 1.0  # no mesh detected — don't penalise

    lm = result.face_landmarks[0]

    def pt(idx):
        return numpy.array([lm[idx].x * img_w, lm[idx].y * img_h])

    def dist(a, b):
        return float(numpy.linalg.norm(pt(a) - pt(b)))

    # Eye Aspect Ratio: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    def ear(indices):
        v = dist(indices[1], indices[5]) + dist(indices[2], indices[4])
        h = dist(indices[0], indices[3])
        return v / (2.0 * h + 1e-6)

    ear_avg = (ear(_MP_LEFT_EYE) + ear(_MP_RIGHT_EYE)) / 2.0

    # Mouth Aspect Ratio: vertical opening / horizontal width
    mar = dist(13, 14) / (dist(61, 291) + 1e-6)

    # Yaw asymmetry: |nose-to-left-cheek − nose-to-right-cheek| / total
    d_l = dist(1, 234)
    d_r = dist(1, 454)
    yaw = abs(d_l - d_r) / (d_l + d_r + 1e-6)

    # Hard gate: eyes clearly closed → discard
    if ear_avg < THUMB_EAR_MIN:
        return 0.0

    quality = 1.0

    # Partially closing eyes → heavy penalty
    if ear_avg < THUMB_EAR_MIN * 1.4:
        quality *= 0.2

    # Mouth too wide open → scale penalty by excess above threshold
    if mar > THUMB_MAR_MAX:
        excess   = (mar - THUMB_MAR_MAX) / THUMB_MAR_MAX
        quality *= max(0.05, 1.0 - 1.5 * excess)

    # Head turned too far → scale penalty by excess above threshold
    if yaw > THUMB_YAW_MAX:
        excess   = (yaw - THUMB_YAW_MAX) / (1.0 - THUMB_YAW_MAX)
        quality *= max(0.1, 1.0 - 2.0 * excess)

    return quality


def _load_thumb_manifest(manifest_path: Path) -> dict:
    """Load per-video thumbnail manifest; returns empty structure if missing or corrupt."""
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            data.setdefault("generated", {})
            data.setdefault("rejected_timestamps", [])
            return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"generated": {}, "rejected_timestamps": []}


def _save_thumb_manifest(manifest_path: Path, manifest: dict) -> None:
    """Write the manifest dict as pretty-printed JSON."""
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def extract_thumbnails(
    video_path: Path,
    thumbnails_dir: Path,
    src_info: dict,
    n_thumbs: int = THUMBNAIL_COUNT,
    n_sample: int = THUMBNAIL_SAMPLE,
    force: bool = False,
) -> None:
    """
    Score n_sample evenly-spaced frames and save the top n_thumbs into
    thumbnails_dir/staging/. Approved frames live in thumbnails_dir/approved/.
    Frames deleted from staging are recorded as rejected in a per-video manifest
    and excluded from all future runs.
    """
    try:
        import cv2
    except ImportError:
        print("  ⚠️  opencv-python not installed — skipping thumbnail extraction.")
        return

    # ── Derive paths ──────────────────────────────────────────────────────────
    stem          = video_path.stem
    staging_dir   = thumbnails_dir / "staging"
    approved_dir  = thumbnails_dir / "approved"
    manifest_path = staging_dir / f"{stem}_thumbs.json"
    staging_dir.mkdir(parents=True, exist_ok=True)
    approved_dir.mkdir(parents=True, exist_ok=True)

    # ── Skip check ────────────────────────────────────────────────────────────
    existing_staging = sorted(staging_dir.glob(f"{stem}_thumb_*.jpg"))
    if len(existing_staging) >= n_thumbs and not force:
        print(f"  ⏭️  Thumbnails exist ({len(existing_staging)} in staging) — use --force to redo")
        return

    # ── Load manifest ─────────────────────────────────────────────────────────
    manifest = _load_thumb_manifest(manifest_path)

    # ── On --force: detect deleted staging frames → mark as rejected ──────────
    if force and manifest["generated"]:
        approved_names = {f.name for f in approved_dir.glob(f"{stem}_thumb_*.jpg")}
        staging_names  = {f.name for f in existing_staging}
        still_present  = staging_names | approved_names
        for fname, ts in manifest["generated"].items():
            if fname not in still_present and ts not in manifest["rejected_timestamps"]:
                manifest["rejected_timestamps"].append(ts)

    # ── Build exclusion set ───────────────────────────────────────────────────
    # Exclude rejected timestamps and already-approved timestamps
    approved_ts = {
        manifest["generated"][f.name]
        for f in approved_dir.glob(f"{stem}_thumb_*.jpg")
        if f.name in manifest["generated"]
    }
    excluded_ts = set(manifest["rejected_timestamps"]) | approved_ts

    # ── Video metrics & candidate timestamps ──────────────────────────────────
    src_w    = src_info["width"]
    src_h    = src_info["height"]
    duration = src_info.get("duration", 0)
    start    = duration * 0.05
    end      = duration * 0.95

    sample_interval  = (end - start) / max(n_sample - 1, 1)
    exclusion_radius = sample_interval / 2.0
    timestamps       = [start + i * sample_interval for i in range(n_sample)]

    # ── MediaPipe face landmarker (optional) ──────────────────────────────────
    face_mesh = None
    try:
        import mediapipe as mp, urllib.request
        if not _LANDMARKER_CACHE.exists():
            _LANDMARKER_CACHE.parent.mkdir(parents=True, exist_ok=True)
            print("  ⬇️  Downloading face landmarker model (~11 MB, cached for future runs)…")
            urllib.request.urlretrieve(_LANDMARKER_URL, _LANDMARKER_CACHE)
        face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(_LANDMARKER_CACHE)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
            )
        )
    except ImportError:
        print("  ℹ️  mediapipe not installed — pip install mediapipe for expression filtering")
    except Exception as e:
        print(f"  ⚠️  Could not initialise face landmarker: {e}")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade      = cv2.CascadeClassifier(cascade_path)
    frame_area   = src_w * src_h

    print(f"  🖼️  Sampling {n_sample} frames for thumbnails…")

    scored = []  # (score, timestamp, frame_bgr)

    for ts in timestamps:
        # Skip excluded regions without extracting the frame
        if any(abs(ts - excl) <= exclusion_radius for excl in excluded_ts):
            continue

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{ts:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-f", "image2pipe",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 or not result.stdout:
            continue
        expected = src_w * src_h * 3
        if len(result.stdout) < expected:
            continue

        buf   = numpy.frombuffer(result.stdout[:expected], dtype=numpy.uint8)
        frame = buf.reshape((src_h, src_w, 3))

        grey      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(grey, cv2.CV_64F).var()

        faces     = cascade.detectMultiScale(
            grey, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        face_area = sum(w * h for (_, _, w, h) in faces) if len(faces) else 0

        quality = _face_quality(face_mesh, frame, src_w, src_h) if face_mesh else 1.0
        score   = sharpness * (1.0 + 10.0 * face_area / frame_area) * quality
        scored.append((score, ts, frame))

    if face_mesh is not None:
        face_mesh.close()

    # ── How many new thumbnails are needed ────────────────────────────────────
    n_needed = n_thumbs - len(existing_staging)
    if n_needed <= 0 or not scored:
        if not scored:
            print("  ⚠️  No scorable frames available — skipping thumbnails.")
        _save_thumb_manifest(manifest_path, manifest)
        return

    scored.sort(key=lambda x: x[0], reverse=True)

    # ── Greedy diverse selection ──────────────────────────────────────────────
    min_gap  = (end - start) / (n_thumbs * 1.5)
    selected = []
    for entry in scored:
        if len(selected) >= n_needed:
            break
        _, ts, _ = entry
        if all(abs(ts - s[1]) >= min_gap for s in selected):
            selected.append(entry)

    # Top-up if diversity constraint left gaps
    if len(selected) < n_needed:
        picked_ts = {s[1] for s in selected}
        for entry in scored:
            if len(selected) >= n_needed:
                break
            if entry[1] not in picked_ts:
                selected.append(entry)
                picked_ts.add(entry[1])

    selected.sort(key=lambda x: x[1])  # chronological order

    # ── Determine next filename index (continue from highest existing) ─────────
    used_indices = set()
    for f in existing_staging:
        try:
            used_indices.add(int(f.stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    next_idx = max(used_indices, default=0) + 1

    # ── Write files and update manifest ──────────────────────────────────────
    new_entries = {}
    for i, (_, ts, frame) in enumerate(selected):
        filename = f"{stem}_thumb_{next_idx + i:02d}.jpg"
        cv2.imwrite(str(staging_dir / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        new_entries[filename] = ts

    manifest["generated"].update(new_entries)
    _save_thumb_manifest(manifest_path, manifest)

    total = len(existing_staging) + len(selected)
    rejected_count = len(manifest["rejected_timestamps"])
    suffix = f"  ({rejected_count} timestamp(s) permanently excluded)" if rejected_count else ""
    print(f"  ✅ {len(selected)} new thumbnails → staging/  ({total}/{n_thumbs}){suffix}")


# ── Processing ─────────────────────────────────────────────────────────────────

def process_video(video_path: Path, output_dir: Path, force: bool,
                  reel_dir: "Path | None" = None,
                  thumbnails_dir: "Path | None" = None,
                  reel_thumbnails_dir: "Path | None" = None) -> Path:
    out_path  = output_dir / video_path.name
    reel_path = (reel_dir / video_path.name) if reel_dir else None

    landscape_exists   = out_path.exists()
    reel_exists        = reel_path.exists() if reel_path else True
    thumbs_exist       = (
        len(list((thumbnails_dir / "staging").glob(f"{video_path.stem}_thumb_*.jpg"))) >= THUMBNAIL_COUNT
        if thumbnails_dir else True
    )
    reel_thumbs_exist  = (
        len(list((reel_thumbnails_dir / "staging").glob(f"{video_path.stem}_thumb_*.jpg"))) >= THUMBNAIL_COUNT
        if reel_thumbnails_dir else True
    )

    if landscape_exists and reel_exists and thumbs_exist and reel_thumbs_exist and not force:
        print(f"⏭️  Skipping (exists): {out_path.name}  — use --force to overwrite")
        return out_path

    info = get_video_info(video_path)
    w, h = info.get("width", "?"), info.get("height", "?")
    print(f"🔊 Processing: {video_path.name}  ({w}×{h})")

    if not landscape_exists or force:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", "scale=-2:720",
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

    if reel_dir is not None:
        reel_dir.mkdir(parents=True, exist_ok=True)
        produce_reel(video_path, reel_dir, info, force)
        if reel_thumbnails_dir is not None and reel_path is not None and reel_path.exists():
            reel_thumbnails_dir.mkdir(parents=True, exist_ok=True)
            extract_thumbnails(reel_path, reel_thumbnails_dir, get_video_info(reel_path), force=force)

    if thumbnails_dir is not None:
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        extract_thumbnails(video_path, thumbnails_dir, info, force=force)

    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw horizontal videos: voice optimisation, loudness normalisation, 720 px"
    )
    parser.add_argument("--input",       default="video_a_originals",
                        help="Folder with raw source videos (default: video_a_originals)")
    parser.add_argument("--output",      default="video_c_sources",
                        help="Output folder (default: video_c_sources)")
    parser.add_argument("--force",       action="store_true",
                        help="Overwrite existing output files (never overwrites edited TXT)")
    parser.add_argument("--forceall",    action="store_true",
                        help="Like --force, but also refresh the shared transcript.txt "
                             "(a timestamped backup is created first)")
    parser.add_argument("--no-reel",           action="store_true",
                        help="Skip portrait reel crop output")
    parser.add_argument("--reel-output",       default=REEL_OUTPUT_DIR,
                        help=f"Folder for portrait reel outputs (default: {REEL_OUTPUT_DIR})")
    parser.add_argument("--no-thumbnails",          action="store_true",
                        help="Skip still-image thumbnail extraction")
    parser.add_argument("--thumbnails-output",      default=THUMBNAILS_OUTPUT_DIR,
                        help=f"Folder for extracted thumbnails (default: {THUMBNAILS_OUTPUT_DIR})")
    parser.add_argument("--no-reel-thumbnails",     action="store_true",
                        help="Skip still-image thumbnail extraction for reels")
    parser.add_argument("--reel-thumbnails-output", default=REEL_THUMBNAILS_OUTPUT_DIR,
                        help=f"Folder for reel thumbnails (default: {REEL_THUMBNAILS_OUTPUT_DIR})")
    parser.add_argument("--no-transcribe",  action="store_true",
                        help="Skip Whisper transcription (audio/video processing only)")
    parser.add_argument("--transcripts",    default="transcripts",
                        help="Output folder for shared transcripts (default: transcripts)")
    parser.add_argument("--model",  default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--lang",   default=None,
                        help="Force transcription language, e.g. 'en', 'de' (auto-detect if omitted)")
    parser.add_argument("--max-words", type=int, default=8,
                        help="Max words per subtitle line (default: 8)")
    args = parser.parse_args()
    force     = args.force or args.forceall
    force_all = args.forceall

    check_dependencies()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    reel_dir = None if args.no_reel else Path(args.reel_output)
    if args.no_thumbnails:
        thumbnails_dir = None
    else:
        thumbnails_dir = Path(args.thumbnails_output)
        (thumbnails_dir / "staging").mkdir(parents=True, exist_ok=True)
        (thumbnails_dir / "approved").mkdir(parents=True, exist_ok=True)
    if args.no_reel_thumbnails or args.no_reel:
        reel_thumbnails_dir = None
    else:
        reel_thumbnails_dir = Path(args.reel_thumbnails_output)
        (reel_thumbnails_dir / "staging").mkdir(parents=True, exist_ok=True)
        (reel_thumbnails_dir / "approved").mkdir(parents=True, exist_ok=True)

    transcripts_dir = None if args.no_transcribe else Path(args.transcripts)
    if transcripts_dir:
        transcripts_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(args.input)
    if not videos:
        print("❌ No supported video files found.")
        sys.exit(1)

    print(f"\n📂 Found {len(videos)} video(s)\n{'─' * 40}")
    for video in videos:
        print(f"\n▶  {video}")
        out_path = process_video(video, output_dir, force, reel_dir=reel_dir,
                                 thumbnails_dir=thumbnails_dir,
                                 reel_thumbnails_dir=reel_thumbnails_dir)
        if transcripts_dir and out_path.exists():
            transcribe_to_folder(out_path, transcripts_dir,
                                 args.model, args.lang, args.max_words,
                                 force=force, force_all=force_all)

    print(f"\n🎉 Done! Processed {len(videos)} video(s).")


if __name__ == "__main__":
    main()
