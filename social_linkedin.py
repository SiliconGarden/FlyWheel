#!/usr/bin/env python3
"""
social_linkedin.py — Burn manually-written LinkedIn post text onto thumbnails.

Workflow:
  1. python3 social_linkedin.py --scaffold [--stem "..."]
       Creates numbered .txt template files — one per image post, one per carousel.
       Fill in the hook, post body, and slides (any text editor).

  2. Edit the generated .txt files.

  3. python3 social_linkedin.py [--stem "..."]
       Reads filled-in templates, assigns thumbnails (auto via MediaPipe or named),
       burns hook text onto images, writes carousel PDFs.

Output:
  social_posts/{stem}/linkedin/
    image_posts/
      01.txt   ← edit this            01/  post.txt  image.jpg
      02.txt                           02/  post.txt  image.jpg
      ...
    carousel_posts/
      01.txt   ← edit this            01/  post.txt  slide_01.jpg  ...  carousel.pdf
      ...

Usage:
    python3 social_linkedin.py --scaffold
    python3 social_linkedin.py --scaffold --stem "Work Life Balance..." --image-posts 10
    python3 social_linkedin.py
    python3 social_linkedin.py --stem "Work Life Balance..."
    python3 social_linkedin.py --force
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────

TURQUOISE = (92, 225, 230)   # #5CE1E6 — pill background (matches subtitle style)
WHITE     = (255, 255, 255)  # text colour — matches subtitle resting-word colour
BLACK     = (0,   0,   0)

_FONT_CACHE = Path.home() / ".cache" / "reelprc" / "LeagueSpartan-Bold.ttf"
_FONT_URL   = (
    "https://cdn.jsdelivr.net/gh/theleagueof/league-spartan@master"
    "/fonts/static/TTF/LeagueSpartan-Bold.ttf"
)

_MP_LEFT_EYE  = [33, 160, 158, 133, 153, 144]
_MP_RIGHT_EYE = [362, 385, 387, 263, 380, 373]
_LANDMARKER_CACHE = Path.home() / ".cache" / "reelprc" / "face_landmarker.task"
_LANDMARKER_URL   = (
    "https://storage.googleapis.com/mediapipe-models"
    "/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

DEFAULT_TRANSCRIPTS_DIR = "transcripts"
DEFAULT_THUMBNAILS_DIR  = "reel_a_thumbnails"
DEFAULT_OUTPUT_DIR      = "social_assets"

# Per-channel canvas dimensions.  Templates always live in the "linkedin" subfolder;
# other channels render from those same templates into their own output subfolders.
CHANNEL_CONFIGS: dict[str, dict] = {
    "linkedin":  {"image_size": (1080, 1080), "carousel_size": (1080, 1350)},
    "instagram": {"image_size": (1080, 1350), "carousel_size": (1080, 1350)},
    "facebook":  {"image_size": (1080, 1350), "carousel_size": (1080, 1350)},
}

# Sentinel text used in scaffold templates — files containing this are "unfilled"
_PLACEHOLDER_HOOK    = "Write your hook here."
_PLACEHOLDER_BODY    = "Write your LinkedIn post body here."
_PLACEHOLDER_SLIDES  = {
    "Write slide 2 key point here.",
    "Write slide 3 key point here.",
    "Write slide 4 key point here.",
    "Write slide 5 key point here.",
    "Write closing CTA here.",
}


# ── Dependency check ──────────────────────────────────────────────────────────

def check_dependencies() -> None:
    errors = []
    try:
        from PIL import Image  # noqa
    except ImportError:
        errors.append("Pillow  →  pip install Pillow")
    if errors:
        print("❌ Missing dependencies:\n")
        for e in errors:
            print(f"   {e}")
        sys.exit(1)


# ── Font resolution ───────────────────────────────────────────────────────────

def _find_system_font(needle: str) -> Optional[Path]:
    dirs = [
        Path.home() / "Library" / "Fonts",
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
        Path("/usr/share/fonts"),
    ]
    needle_clean = needle.lower().replace(" ", "").replace("-", "")
    for d in dirs:
        if not d.exists():
            continue
        for ext in ("*.ttf", "*.otf"):
            for f in d.glob(ext):
                if needle_clean in f.stem.lower().replace(" ", "").replace("-", ""):
                    return f
    return None


def get_font_path() -> Path:
    if _FONT_CACHE.exists():
        return _FONT_CACHE
    for needle in ("LeagueSpartanBold", "LeagueSpartanVariable", "LeagueSpartan"):
        found = _find_system_font(needle)
        if found:
            return found
    _FONT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print("  ⬇️  Downloading League Spartan Bold font…")
    try:
        urllib.request.urlretrieve(_FONT_URL, _FONT_CACHE)
        return _FONT_CACHE
    except Exception as e:
        print(f"❌ Font download failed: {e}")
        print(f"   Place LeagueSpartan-Bold.ttf at {_FONT_CACHE}")
        sys.exit(1)


# ── Stem discovery ────────────────────────────────────────────────────────────

def find_stems(transcripts_dir: Path, stem: Optional[str]) -> list[str]:
    if stem:
        if not (transcripts_dir / stem / "transcript.words.json").exists():
            print(f"❌ No transcript found for stem: {stem}")
            sys.exit(1)
        return [stem]
    stems = sorted(
        d.name for d in transcripts_dir.iterdir()
        if d.is_dir() and (d / "transcript.words.json").exists()
    )
    if not stems:
        print(f"❌ No transcript stems found in {transcripts_dir}")
        sys.exit(1)
    return stems


# ── Thumbnail loading + MediaPipe scoring ─────────────────────────────────────

def load_thumbnails(thumbnails_dir: Path, stem: str) -> list[Path]:
    """Return thumbnails for stem — approved first, then staging. Falls back to video dir."""
    results: list[Path] = []
    for sub in ("approved", "staging"):
        d = thumbnails_dir / sub
        if d.exists():
            results.extend(sorted(d.glob(f"{stem}_thumb_*.jpg")))
    if not results:
        alt = Path(str(thumbnails_dir).replace("reel_a_thumbnails", "video_a_thumbnails"))
        if alt != thumbnails_dir:
            for sub in ("approved", "staging"):
                d = alt / sub
                if d.exists():
                    results.extend(sorted(d.glob(f"{stem}_thumb_*.jpg")))
    return results


def _init_face_landmarker():
    try:
        import mediapipe as mp
        if not _LANDMARKER_CACHE.exists():
            _LANDMARKER_CACHE.parent.mkdir(parents=True, exist_ok=True)
            print("  ⬇️  Downloading face landmarker model…")
            urllib.request.urlretrieve(_LANDMARKER_URL, _LANDMARKER_CACHE)
        return mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(_LANDMARKER_CACHE)
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
            )
        )
    except Exception:
        return None


def _face_scores(face_mesh, img_bgr, w: int, h: int) -> dict:
    import mediapipe as mp
    import numpy as np
    import cv2
    rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return {"ear": -1.0, "mar": -1.0, "yaw": -1.0}
    lm = result.face_landmarks[0]

    def pt(i):      return np.array([lm[i].x * w, lm[i].y * h])
    def dist(a, b): return float(np.linalg.norm(pt(a) - pt(b)))
    def ear(idx):
        return (dist(idx[1], idx[5]) + dist(idx[2], idx[4])) / (2.0 * dist(idx[0], idx[3]) + 1e-6)

    d_l = dist(1, 234)
    d_r = dist(1, 454)
    return {
        "ear": (ear(_MP_LEFT_EYE) + ear(_MP_RIGHT_EYE)) / 2.0,
        "mar": dist(13, 14) / (dist(61, 291) + 1e-6),
        "yaw": abs(d_l - d_r) / (d_l + d_r + 1e-6),
    }


def score_thumbnails(thumb_paths: list[Path],
                     cache_path: Optional[Path] = None) -> list[dict]:
    """Score thumbnails via MediaPipe. Caches results in cache_path (keyed by
    filename + mtime) so unchanged thumbnails are not re-scored on future runs."""
    # ── Load cache ────────────────────────────────────────────────────────────
    cache: dict = {}
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    face_mesh  = None   # initialised lazily — only if a cache miss requires it
    results: list[dict] = []
    n_fresh = 0

    for p in thumb_paths:
        key   = p.name
        mtime = p.stat().st_mtime

        # ── Cache hit ─────────────────────────────────────────────────────────
        cached = cache.get(key)
        if cached and cached.get("mtime") == mtime:
            results.append({"path": p,
                             "ear": cached["ear"],
                             "mar": cached["mar"],
                             "yaw": cached["yaw"]})
            continue

        # ── Cache miss — score fresh ───────────────────────────────────────────
        if has_cv2 and face_mesh is None:
            face_mesh = _init_face_landmarker()

        img = cv2.imread(str(p)) if has_cv2 else None
        if img is None:
            sc = {"ear": 0.5, "mar": 0.1, "yaw": 0.0}
        else:
            h_px, w_px = img.shape[:2]
            sc = (_face_scores(face_mesh, img, w_px, h_px)
                  if face_mesh else {"ear": 0.5, "mar": 0.1, "yaw": 0.0})

        cache[key] = {**sc, "mtime": mtime}
        n_fresh += 1
        results.append({"path": p, **sc})

    if face_mesh:
        face_mesh.close()

    # ── Persist cache if anything was scored fresh ────────────────────────────
    if cache_path and n_fresh:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    n_cached = len(thumb_paths) - n_fresh
    if n_fresh and n_cached:
        print(f"  🖼️  Scored {n_fresh} new thumbnail(s), {n_cached} loaded from cache")
    elif n_fresh:
        print(f"  🖼️  Scored {n_fresh} thumbnail(s)")
    else:
        print(f"  🖼️  Loaded {n_cached} thumbnail score(s) from cache")

    return results


def _tone_score(info: dict, tone: str) -> float:
    ear, mar, yaw = info["ear"], info["mar"], info["yaw"]
    if ear < 0:
        return 0.5
    if tone == "energetic":
        return ear * (1.0 + 0.4 * min(mar, 0.5)) * max(0.1, 1.0 - 2.0 * yaw)
    if tone == "authoritative":
        return ear * max(0.1, 1.0 - 1.5 * mar) * max(0.1, 1.0 - 3.0 * yaw)
    return ear * max(0.1, 1.0 - mar) * (0.6 + 0.8 * yaw)   # thoughtful


def _pick_thumbnail(tone: str, thumb_infos: list[dict], used: set[int]) -> Optional[Path]:
    if not thumb_infos:
        return None
    pool = [(i, _tone_score(ti, tone)) for i, ti in enumerate(thumb_infos) if i not in used]
    if not pool:
        used.clear()
        pool = [(i, _tone_score(ti, tone)) for i, ti in enumerate(thumb_infos)]
    best = max(pool, key=lambda x: x[1])[0]
    used.add(best)
    return thumb_infos[best]["path"]


def _resolve_thumbnail(spec: str, thumb_paths: list[Path], thumb_infos: list[dict],
                        tone: str, used: set[int]) -> Optional[Path]:
    if spec == "auto":
        return _pick_thumbnail(tone, thumb_infos, used)
    for p in thumb_paths:
        if p.name == spec:
            return p
    print(f"  ⚠️  Thumbnail '{spec}' not found — falling back to auto")
    return _pick_thumbnail(tone, thumb_infos, used)


# ── Template scaffolding ──────────────────────────────────────────────────────

def _image_post_template(idx: int, stem: str, thumb_list: str) -> str:
    return (
        f"# ── Image Post {idx:02d} ─────────────────────────────────────────────\n"
        f"# Stem: {stem}\n"
        f"# Available thumbnails: {thumb_list}\n"
        f"#\n"
        f"# HOOK  (1–2 punchy sentences, max 130 chars — burned onto the image)\n"
        f"\n"
        f"{_PLACEHOLDER_HOOK}\n"
        f"\n"
        f"# POST BODY  (LinkedIn caption, 3–5 sentences)\n"
        f"\n"
        f"{_PLACEHOLDER_BODY}\n"
        f"\n"
        f"# thumbnail: auto\n"
        f"# tone: energetic\n"
    )


def _carousel_post_template(idx: int, stem: str, thumb_list: str) -> str:
    slides = "\n".join(sorted(_PLACEHOLDER_SLIDES))
    return (
        f"# ── Carousel Post {idx:02d} ──────────────────────────────────────────\n"
        f"# Stem: {stem}\n"
        f"# Available thumbnails: {thumb_list}\n"
        f"#\n"
        f"# HOOK  (cover slide text, max 80 chars — also shown as first slide)\n"
        f"\n"
        f"{_PLACEHOLDER_HOOK}\n"
        f"\n"
        f"# POST BODY  (LinkedIn caption introducing the carousel, 3–5 sentences)\n"
        f"\n"
        f"{_PLACEHOLDER_BODY}\n"
        f"\n"
        f"# slides\n"
        f"# One key point per line. Cover slide is auto-generated from the hook above.\n"
        f"{slides}\n"
        f"\n"
        f"# thumbnail: auto\n"
        f"# tone: thoughtful\n"
    )


def scaffold_stem(stem: str, thumbnails_dir: Path, output_dir: Path,
                  n_image: int, n_carousel: int, force: bool) -> None:
    thumb_names = [p.name for p in load_thumbnails(thumbnails_dir, stem)]
    if thumb_names:
        shown      = thumb_names[:6]
        thumb_list = ", ".join(shown)
        if len(thumb_names) > 6:
            thumb_list += f" … ({len(thumb_names)} total)"
    else:
        thumb_list = "(none found — run prepare step first)"

    stem_out = output_dir / stem / "linkedin"
    created = skipped = 0

    for subdir, n, tmpl_fn in [
        ("image_posts",    n_image,    _image_post_template),
        ("carousel_posts", n_carousel, _carousel_post_template),
    ]:
        d = stem_out / subdir
        d.mkdir(parents=True, exist_ok=True)
        for i in range(1, n + 1):
            p = d / f"{i:02d}.txt"
            if p.exists() and not force:
                skipped += 1
                continue
            p.write_text(tmpl_fn(i, stem, thumb_list), encoding="utf-8")
            created += 1

    print(f"  ✅ {created} template(s) created, {skipped} skipped (already exist)")
    print(f"  📝 Edit files in: {stem_out}")
    print(f"     Then run without --scaffold to render images.")


# ── Template parsing ──────────────────────────────────────────────────────────

def _parse_template(raw: str) -> tuple[list[str], list[str], dict[str, str]]:
    """
    Split raw template text into:
      paragraphs   — list of non-comment text blocks (blank-line separated)
      slides_lines — lines after '# slides' (until next directive)
      directives   — {key: value} from '# key: value' lines
    """
    lines        = raw.splitlines()
    directives: dict[str, str] = {}
    content: list[str] = []
    slides: list[str] = []
    in_slides = False

    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "# slides":
            in_slides = True
            continue
        if stripped.startswith("#"):
            # Only a "# key: value" directive ends the slides section;
            # plain comments ("# some note") are skipped but don't close it.
            if ":" in stripped:
                key, _, val = stripped[2:].partition(":")
                key_clean = key.strip().lower()
                # Guard: if the key looks like a sentence (spaces, long) it's a
                # plain comment that happens to contain a colon, not a directive.
                if " " not in key_clean and len(key_clean) <= 20:
                    directives[key_clean] = val.strip()
                    in_slides = False   # a real directive ends the slides block
            continue
        if in_slides:
            if stripped:
                slides.append(stripped)
        else:
            content.append(line)

    # Split content into paragraphs on blank lines
    paragraphs: list[str] = []
    current: list[str] = []
    for line in content:
        if line.strip():
            current.append(line.strip())
        else:
            if current:
                paragraphs.append(" ".join(current))
                current = []
    if current:
        paragraphs.append(" ".join(current))

    return paragraphs, slides, directives


def load_image_post(txt_path: Path) -> Optional[dict]:
    """
    Parse an image-post template. Returns None if the file is still unfilled.
    Returns {hook, post_text, thumbnail, tone}.
    """
    paragraphs, _, directives = _parse_template(
        txt_path.read_text(encoding="utf-8")
    )
    if not paragraphs or paragraphs[0] == _PLACEHOLDER_HOOK:
        return None
    return {
        "hook":      paragraphs[0],
        "post_text": paragraphs[1] if len(paragraphs) > 1 else "",
        "thumbnail": directives.get("thumbnail", "auto"),
        "tone":      directives.get("tone", "energetic"),
    }


def load_carousel_post(txt_path: Path) -> Optional[dict]:
    """
    Parse a carousel-post template. Returns None if still unfilled.
    Returns {hook, post_text, slides, thumbnail, tone}.
    """
    paragraphs, slide_lines, directives = _parse_template(
        txt_path.read_text(encoding="utf-8")
    )
    if not paragraphs or paragraphs[0] == _PLACEHOLDER_HOOK:
        return None
    hook   = paragraphs[0]
    filled = [s for s in slide_lines if s not in _PLACEHOLDER_SLIDES]
    slides = [hook] + filled if filled else [hook]
    return {
        "hook":      hook,
        "post_text": paragraphs[1] if len(paragraphs) > 1 else "",
        "slides":    slides,
        "thumbnail": directives.get("thumbnail", "auto"),
        "tone":      directives.get("tone", "thoughtful"),
    }


# ── Image rendering ───────────────────────────────────────────────────────────

def _wrap_text(text: str, font, max_px: int) -> list[str]:
    from PIL import ImageDraw, Image as _I
    _d = ImageDraw.Draw(_I.new("RGBA", (1, 1)))  # font already bold-loaded by caller
    words, lines, cur = text.split(), [], []
    for w in words:
        if _d.textbbox((0, 0), " ".join(cur + [w]), font=font)[2] <= max_px:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]


def _draw_pill(draw, x1, y1, x2, y2, r, fill) -> None:
    draw.rectangle([x1 + r, y1, x2 - r, y2], fill=fill)
    draw.rectangle([x1, y1 + r, x2, y2 - r], fill=fill)
    draw.ellipse([x1,       y1,       x1 + 2*r, y1 + 2*r], fill=fill)
    draw.ellipse([x2 - 2*r, y1,       x2,       y1 + 2*r], fill=fill)
    draw.ellipse([x1,       y2 - 2*r, x1 + 2*r, y2      ], fill=fill)
    draw.ellipse([x2 - 2*r, y2 - 2*r, x2,       y2      ], fill=fill)


def _load_font(font_path: Path, size: int):
    """Load font at the given size. For variable fonts, sets weight=700 (Bold)
    to match the ASS subtitle style (Bold=-1)."""
    from PIL import ImageFont
    font = ImageFont.truetype(str(font_path), size)
    try:
        font.set_variation_by_axes([700])   # wght=700 → Bold
    except (OSError, AttributeError):
        pass   # non-variable font or old Pillow — use as-is
    return font


def _composite_pill_text(base, text: str, font_path: Path,
                          font_size: int, y_frac: float):
    from PIL import Image, ImageDraw
    img   = base.convert("RGBA")
    w, h  = img.size
    font  = _load_font(font_path, font_size)
    pad_x = int(font_size * 0.65)
    pad_y = int(font_size * 0.45)
    max_w = int(w * 0.84) - 2 * pad_x
    lines = _wrap_text(text, font, max_w)
    lh    = int(font_size * 1.35)
    pw    = max_w + 2 * pad_x
    ph    = len(lines) * lh + 2 * pad_y
    r     = int(font_size * 0.40)
    px    = (w - pw) // 2
    py    = max(0, min(int(h * y_frac) - ph // 2, h - ph))
    ov    = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(ov)
    _draw_pill(draw, px, py, px + pw, py + ph, r, (*TURQUOISE, 225))
    y = py + pad_y
    for line in lines:
        bw = draw.textbbox((0, 0), line, font=font)[2]
        draw.text(((w - bw) // 2, y), line, font=font, fill=(*WHITE, 255))
        y += lh
    return Image.alpha_composite(img, ov).convert("RGB")


def _add_slide_indicator(img_rgb, n: int, total: int, font_path: Path, w: int, h: int):
    from PIL import Image, ImageDraw
    img  = img_rgb.convert("RGBA")
    draw = ImageDraw.Draw(img)
    font = _load_font(font_path, max(18, int(w * 0.026)))
    text = f"{n} / {total}"
    x, y = w - int(w * 0.04), int(h * 0.03)
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, 140), anchor="rt")
    draw.text((x,     y),     text, font=font, fill=(255, 255, 255, 190), anchor="rt")
    return img.convert("RGB")


def render_image_post(thumbnail: Path, hook: str, font_path: Path,
                       output_path: Path,
                       size: tuple[int, int] = (1080, 1080)) -> None:
    from PIL import Image, ImageOps
    img = Image.open(thumbnail).convert("RGBA")
    img = ImageOps.fit(img, size, Image.LANCZOS)
    fs  = max(60, min(88, int(size[0] * 0.065)))
    _composite_pill_text(img, hook, font_path, fs, y_frac=0.80).save(
        str(output_path), quality=95
    )


def render_carousel_slide(thumbnail: Path, text: str, slide_num: int, total: int,
                           is_cover: bool, font_path: Path, output_path: Path,
                           size: tuple[int, int] = (1080, 1350)) -> None:
    from PIL import Image, ImageOps
    img = Image.open(thumbnail).convert("RGBA")
    img = ImageOps.fit(img, size, Image.LANCZOS)
    w, h = img.size
    fs   = max(60, min(88, int(w * (0.088 if is_cover else 0.065))))
    result = _composite_pill_text(img, text, font_path, fs, y_frac=0.80)
    if not is_cover and total > 1:
        result = _add_slide_indicator(result, slide_num, total, font_path, w, h)
    result.save(str(output_path), quality=95)


def slides_to_pdf(slide_paths: list[Path], pdf_path: Path) -> None:
    from PIL import Image
    pages = [Image.open(p).convert("RGB") for p in slide_paths]
    if pages:
        pages[0].save(
            str(pdf_path),
            format="PDF",
            save_all=True,
            append_images=pages[1:],
            resolution=150,
        )


# ── Output helpers ────────────────────────────────────────────────────────────

def _post_txt(hook: str, post_text: str, slides: Optional[list[str]] = None) -> str:
    parts = [hook, "", post_text]
    if slides:
        parts += ["", "---", "Slides:"]
        parts += [f"{i}. {s}" for i, s in enumerate(slides, 1)]
    return "\n".join(parts)


# ── Per-stem render ───────────────────────────────────────────────────────────

def process_stem(stem: str, thumbnails_dir: Path, output_dir: Path,
                 n_image: int, n_carousel: int, force: bool,
                 channel: str = "linkedin",
                 image_size: tuple[int, int] = (1080, 1080),
                 carousel_size: tuple[int, int] = (1080, 1350)) -> None:
    font_path    = get_font_path()
    template_dir = output_dir / stem / "linkedin"   # templates always live here
    stem_out     = output_dir / stem / channel       # rendered output per channel

    thumb_paths = load_thumbnails(thumbnails_dir, stem)
    if thumb_paths:
        cache_path  = thumbnails_dir / f"{stem}_scores.json"
        thumb_infos = score_thumbnails(thumb_paths, cache_path)
    else:
        print("  ⚠️  No thumbnails found — images will not be rendered.")
        thumb_infos = []

    # ── Image posts ────────────────────────────────────────────────────────
    tmpl_img = template_dir / "image_posts"
    out_img  = stem_out     / "image_posts"
    used_img: set[int] = set()
    rendered = missing = unfilled = already = 0

    for i in range(1, n_image + 1):
        txt = tmpl_img / f"{i:02d}.txt"
        if not txt.exists():
            missing += 1
            continue
        post = load_image_post(txt)
        if post is None:
            unfilled += 1
            continue
        post_dir = out_img / f"{i:02d}"
        if (post_dir / "image.jpg").exists() and not force:
            already += 1
            continue
        post_dir.mkdir(parents=True, exist_ok=True)
        (post_dir / "post.txt").write_text(
            _post_txt(post["hook"], post["post_text"]), encoding="utf-8"
        )
        thumb = _resolve_thumbnail(post["thumbnail"], thumb_paths, thumb_infos,
                                    post["tone"], used_img)
        if thumb:
            render_image_post(thumb, post["hook"], font_path,
                               post_dir / "image.jpg", size=image_size)
        label = post["hook"][:55] + ("…" if len(post["hook"]) > 55 else "")
        print(f"  ✅ Image post {i:02d}  [{post['tone']}]  {label}")
        rendered += 1

    if missing:
        print(f"  ℹ️  {missing} image post template(s) missing — run --scaffold first")
    if unfilled:
        print(f"  ✏️  {unfilled} image post(s) not yet filled in — edit the .txt files")
    if already:
        print(f"  ⏭️  {already} image post(s) already rendered — use --force to redo")

    # ── Carousel posts ─────────────────────────────────────────────────────
    tmpl_car = template_dir / "carousel_posts"
    out_car  = stem_out     / "carousel_posts"
    used_car: set[int] = set()
    rendered = missing = unfilled = already = 0

    for i in range(1, n_carousel + 1):
        txt = tmpl_car / f"{i:02d}.txt"
        if not txt.exists():
            missing += 1
            continue
        post = load_carousel_post(txt)
        if post is None:
            unfilled += 1
            continue
        post_dir = out_car / f"{i:02d}"
        if (post_dir / "carousel.pdf").exists() and not force:
            already += 1
            continue
        post_dir.mkdir(parents=True, exist_ok=True)
        slides = post["slides"]
        (post_dir / "post.txt").write_text(
            _post_txt(post["hook"], post["post_text"], slides), encoding="utf-8"
        )
        thumb = _resolve_thumbnail(post["thumbnail"], thumb_paths, thumb_infos,
                                    post["tone"], used_car)
        slide_paths: list[Path] = []
        if thumb:
            for j, slide_text in enumerate(slides, 1):
                out = post_dir / f"slide_{j:02d}.jpg"
                render_carousel_slide(
                    thumb, slide_text, j, len(slides),
                    is_cover=(j == 1), font_path=font_path, output_path=out,
                    size=carousel_size,
                )
                slide_paths.append(out)
            if slide_paths:
                slides_to_pdf(slide_paths, post_dir / "carousel.pdf")
        label = post["hook"][:55] + ("…" if len(post["hook"]) > 55 else "")
        print(f"  ✅ Carousel {i:02d} [{len(slides)} slides, {post['tone']}]  {label}")
        rendered += 1

    if missing:
        print(f"  ℹ️  {missing} carousel template(s) missing — run --scaffold first")
    if unfilled:
        print(f"  ✏️  {unfilled} carousel(s) not yet filled in — edit the .txt files")
    if already:
        print(f"  ⏭️  {already} carousel(s) already rendered — use --force to redo")

    print(f"  📂 Output → {stem_out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    all_channels = list(CHANNEL_CONFIGS.keys())

    parser = argparse.ArgumentParser(
        description="Burn manually-written social post text onto thumbnails"
    )
    parser.add_argument("--transcripts",    default=DEFAULT_TRANSCRIPTS_DIR,
                        help=f"Transcripts root — used to discover stems (default: {DEFAULT_TRANSCRIPTS_DIR})")
    parser.add_argument("--stem",           default=None,
                        help="Process a single stem (default: all stems in --transcripts)")
    parser.add_argument("--thumbnails",     default=DEFAULT_THUMBNAILS_DIR,
                        help=f"Thumbnails folder (default: {DEFAULT_THUMBNAILS_DIR})")
    parser.add_argument("--output",         default=DEFAULT_OUTPUT_DIR,
                        help=f"Output root folder (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--channel",        default=None,
                        help=f"Channel(s) to render, comma-separated "
                             f"(default: all — {', '.join(all_channels)})")
    parser.add_argument("--image-posts",    type=int, default=10,
                        help="Number of image post templates to scaffold (default: 10)")
    parser.add_argument("--carousel-posts", type=int, default=5,
                        help="Number of carousel templates to scaffold (default: 5)")
    parser.add_argument("--scaffold",       action="store_true",
                        help="Create blank .txt template files for manual editing")
    parser.add_argument("--force",          action="store_true",
                        help="Re-render existing outputs / overwrite existing scaffold templates")
    args = parser.parse_args()

    check_dependencies()

    transcripts_dir = Path(args.transcripts)
    thumbnails_dir  = Path(args.thumbnails)
    output_dir      = Path(args.output)

    if not transcripts_dir.exists():
        print(f"❌ Transcripts folder not found: {transcripts_dir}")
        sys.exit(1)

    # Resolve channels
    if args.channel:
        requested = [c.strip().lower() for c in args.channel.split(",")]
        unknown   = [c for c in requested if c not in CHANNEL_CONFIGS]
        if unknown:
            print(f"❌ Unknown channel(s): {', '.join(unknown)}")
            print(f"   Valid channels: {', '.join(all_channels)}")
            sys.exit(1)
        channels = requested
    else:
        channels = all_channels

    stems = find_stems(transcripts_dir, args.stem)

    if args.scaffold:
        print(f"\n📂 Scaffolding {len(stems)} stem(s)")
        for stem in stems:
            print(f"\n{'─' * 50}")
            print(f"📝  {stem}")
            print(f"{'─' * 50}")
            scaffold_stem(stem, thumbnails_dir, output_dir,
                          args.image_posts, args.carousel_posts, args.force)
        print("\n✅ Scaffold complete.")
        print("   Fill in the .txt files, then run without --scaffold to render images.")
    else:
        print(f"\n📂 Processing {len(stems)} stem(s) × {len(channels)} channel(s)")
        for channel in channels:
            cfg = CHANNEL_CONFIGS[channel]
            print(f"\n{'━' * 50}")
            print(f"  Channel: {channel}  "
                  f"(image {cfg['image_size'][0]}×{cfg['image_size'][1]}, "
                  f"carousel {cfg['carousel_size'][0]}×{cfg['carousel_size'][1]})")
            print(f"{'━' * 50}")
            for stem in stems:
                print(f"\n{'─' * 50}")
                print(f"📝  {stem}")
                print(f"{'─' * 50}")
                process_stem(
                    stem           = stem,
                    thumbnails_dir = thumbnails_dir,
                    output_dir     = output_dir,
                    n_image        = args.image_posts,
                    n_carousel     = args.carousel_posts,
                    force          = args.force,
                    channel        = channel,
                    image_size     = cfg["image_size"],
                    carousel_size  = cfg["carousel_size"],
                )
        print("\n✅ Done.")


if __name__ == "__main__":
    main()
