#!/usr/bin/env python3
"""
social_post.py — Ad-hoc social image post generator.

Takes a plain text file (hook + body + optional tone directive), selects the
best matching images by tone from an image folder and/or existing stem
thumbnails, and renders image posts across all channels.

Workflow:
  1. python3 social_post.py my_hook.txt --stem "new breed of founders"
       → Scores images, renders 5 square (1080×1080) previews, opens preview
         folder, then asks which image to use.

  2. You inspect the previews and type a number (1–5).

  3. The script renders that one image across all channels at their native sizes.

Re-pick later (skips re-rendering previews):
  python3 social_post.py my_hook.txt --stem "..." --pick 3

Input file format (same as scaffold image-post templates):

    Your hook here — 1-2 punchy sentences, max 130 chars.

    Your post body here. 3-5 sentences with a clear takeaway.

    # tone: energetic

Output:
    social_assets/{name}/preview/
        01/  image.jpg          ← square preview
        02/  image.jpg
        manifest.json           ← maps 1–N to source image paths

    social_assets/{name}/{channel}/image_posts/
        01/  image.jpg  post.txt   ← final render at channel native size
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ── Import shared rendering logic from social_linkedin ────────────────────────
# social_linkedin.py has a `if __name__ == "__main__":` guard so this is safe.
from social_linkedin import (
    CHANNEL_CONFIGS,
    DEFAULT_THUMBNAILS_DIR,
    DEFAULT_OUTPUT_DIR,
    check_dependencies,
    get_font_path,
    load_thumbnails,
    score_thumbnails,
    _tone_score,
    render_image_post,
    _parse_template,
    _post_txt,
)

PREVIEW_SIZE = (1080, 1080)


# ── Text-post loading ─────────────────────────────────────────────────────────

def load_text_post(txt_path: Path) -> dict:
    """Parse the plain text file into {hook, post_text, tone}.

    Exits with an error message if the file cannot be parsed or has no hook.
    """
    raw = txt_path.read_text(encoding="utf-8")
    paragraphs, _, directives = _parse_template(raw)

    if not paragraphs or not paragraphs[0].strip():
        print(f"❌ No hook found in {txt_path}")
        print("   The file must have at least one non-empty paragraph before the first blank line.")
        sys.exit(1)

    return {
        "hook":      paragraphs[0],
        "post_text": paragraphs[1] if len(paragraphs) > 1 else "",
        "tone":      directives.get("tone", "energetic"),
    }


# ── Image collection ──────────────────────────────────────────────────────────

def collect_images(images_dir: Optional[Path],
                   stem: Optional[str],
                   thumbnails_dir: Path) -> list[Path]:
    """Gather image paths from --images folder and/or --stem thumbnails.

    Deduplicates by resolved path. Exits with an error if the pool is empty.
    """
    seen:   set[Path]  = set()
    result: list[Path] = []

    if images_dir:
        if not images_dir.exists():
            print(f"❌ Images folder not found: {images_dir}")
            sys.exit(1)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for p in sorted(images_dir.glob(ext)):
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    result.append(p)

    if stem:
        for p in load_thumbnails(thumbnails_dir, stem):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                result.append(p)

    if not result:
        sources = []
        if images_dir:
            sources.append(f"--images {images_dir}")
        if stem:
            sources.append(f"--stem {stem!r} (in {thumbnails_dir})")
        print(f"❌ No images found from: {', '.join(sources)}")
        sys.exit(1)

    return result


# ── Image selection ───────────────────────────────────────────────────────────

def select_top_images(scored: list[dict], tone: str, count: int) -> list[Path]:
    """Return the top `count` image paths ranked by tone score (highest first)."""
    ranked = sorted(scored, key=lambda info: _tone_score(info, tone), reverse=True)
    return [info["path"] for info in ranked[:count]]


# ── Preview rendering ─────────────────────────────────────────────────────────

def render_previews(images:    list[Path],
                    hook:      str,
                    name:      str,
                    output_dir: Path,
                    font_path:  Path,
                    force:     bool) -> Path:
    """Render square (1080×1080) preview posts and save a manifest.

    Returns the preview directory path.
    Skips images already rendered unless --force.
    """
    preview_dir = output_dir / name / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, str] = {}
    rendered = skipped = 0

    for i, img_path in enumerate(images, 1):
        post_dir = preview_dir / f"{i:02d}"
        out_img  = post_dir / "image.jpg"
        manifest[str(i)] = str(img_path.resolve())

        if out_img.exists() and not force:
            skipped += 1
            continue

        post_dir.mkdir(parents=True, exist_ok=True)
        render_image_post(img_path, hook, font_path, out_img, size=PREVIEW_SIZE)
        print(f"    {i}. {img_path.name}")
        rendered += 1

    (preview_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if skipped and not rendered:
        print(f"    (all {skipped} preview(s) already exist — use --force to redo)")
    elif skipped:
        print(f"    ({skipped} already existed, {rendered} rendered)")

    return preview_dir


def load_manifest(preview_dir: Path) -> dict[int, Path]:
    """Load manifest.json from a preview directory.

    Returns {1: Path, 2: Path, ...}. Exits with error if missing.
    """
    manifest_path = preview_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"❌ No preview manifest found at {manifest_path}")
        print("   Run without --pick first to generate previews.")
        sys.exit(1)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {int(k): Path(v) for k, v in raw.items()}


# ── Interactive pick ──────────────────────────────────────────────────────────

def prompt_pick(count: int) -> int:
    """Prompt the user to pick an image number. Returns the chosen int (1-based).

    Accepts 'q' or empty input to quit.
    """
    while True:
        try:
            raw = input(f"\nPick an image (1–{count}, or q to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if raw.lower() in ("q", "quit", ""):
            print("No image picked — exiting.")
            sys.exit(0)

        if raw.isdigit() and 1 <= int(raw) <= count:
            return int(raw)

        print(f"   Please enter a number between 1 and {count}.")


# ── Final rendering ───────────────────────────────────────────────────────────

def render_final(hook:      str,
                 post_text: str,
                 image:     Path,
                 name:      str,
                 channels:  dict[str, dict],
                 output_dir: Path,
                 font_path:  Path,
                 force:     bool) -> None:
    """Render the picked image at native size for every channel."""
    rendered = skipped = 0

    for channel, cfg in channels.items():
        size    = cfg["image_size"]
        post_dir = output_dir / name / channel / "image_posts" / "01"
        out_img  = post_dir / "image.jpg"

        if out_img.exists() and not force:
            skipped += 1
            print(f"  ⏭️  {channel}  already exists — use --force to redo")
            continue

        post_dir.mkdir(parents=True, exist_ok=True)
        render_image_post(image, hook, font_path, out_img, size=size)
        (post_dir / "post.txt").write_text(
            _post_txt(hook, post_text), encoding="utf-8"
        )
        print(f"  ✅ {channel}  ({size[0]}×{size[1]})  →  {out_img}")
        rendered += 1

    print(f"\n  📊 {rendered} rendered, {skipped} skipped")
    print(f"  📂 Output → {output_dir / name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    all_channels = list(CHANNEL_CONFIGS.keys())

    parser = argparse.ArgumentParser(
        description="Render social image posts from a text file + image pool"
    )
    parser.add_argument("text_file",
                        help="Plain text file with hook, post body, and optional # tone directive")
    parser.add_argument("--images",     default=None,
                        help="Folder of images (jpg/png) to use as the pool")
    parser.add_argument("--stem",       default=None,
                        help="Use existing thumbnails from --thumbnails for this stem")
    parser.add_argument("--name",       default=None,
                        help="Output folder name slug (default: text_file name without extension)")
    parser.add_argument("--channel",    default=None,
                        help=f"Channel(s) to render, comma-separated "
                             f"(default: all — {', '.join(all_channels)})")
    parser.add_argument("--thumbnails", default=DEFAULT_THUMBNAILS_DIR,
                        help=f"Thumbnails root directory (default: {DEFAULT_THUMBNAILS_DIR})")
    parser.add_argument("--output",     default=DEFAULT_OUTPUT_DIR,
                        help=f"Output root directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--count",      type=int, default=5,
                        help="Number of preview images to select (default: 5)")
    parser.add_argument("--pick",       type=int, default=None, metavar="N",
                        help="Skip preview step; render image N from the existing preview manifest")
    parser.add_argument("--force",      action="store_true",
                        help="Re-render existing outputs")
    args = parser.parse_args()

    check_dependencies()

    # ── Validate args ──────────────────────────────────────────────────────────
    txt_path = Path(args.text_file)
    if not txt_path.exists():
        print(f"❌ Text file not found: {txt_path}")
        sys.exit(1)

    thumbnails_dir = Path(args.thumbnails)
    output_dir     = Path(args.output)
    images_dir     = Path(args.images) if args.images else None
    name           = args.name or txt_path.stem
    preview_dir    = output_dir / name / "preview"

    # ── Resolve channels ───────────────────────────────────────────────────────
    if args.channel:
        requested = [c.strip().lower() for c in args.channel.split(",")]
        unknown   = [c for c in requested if c not in CHANNEL_CONFIGS]
        if unknown:
            print(f"❌ Unknown channel(s): {', '.join(unknown)}")
            print(f"   Valid channels: {', '.join(all_channels)}")
            sys.exit(1)
        channels = {c: CHANNEL_CONFIGS[c] for c in requested}
    else:
        channels = dict(CHANNEL_CONFIGS)

    # ── Load text post ─────────────────────────────────────────────────────────
    post = load_text_post(txt_path)
    print(f"\n📄 Post: {txt_path.name}")
    print(f"   Hook : {post['hook'][:80]}{'…' if len(post['hook']) > 80 else ''}")
    print(f"   Tone : {post['tone']}")

    font_path = get_font_path()

    # ── --pick: skip preview, use existing manifest ───────────────────────────
    if args.pick is not None:
        manifest = load_manifest(preview_dir)
        if args.pick not in manifest:
            keys = sorted(manifest.keys())
            print(f"❌ --pick {args.pick} is out of range. Available: {keys}")
            sys.exit(1)
        picked = manifest[args.pick]
        if not picked.exists():
            print(f"❌ Source image no longer exists: {picked}")
            sys.exit(1)
        print(f"\n  Using image {args.pick}: {picked.name}")
        print(f"\n🎨 Rendering across {len(channels)} channel(s)…")
        render_final(post["hook"], post["post_text"], picked, name,
                     channels, output_dir, font_path, args.force)
        print("\n✅ Done.")
        return

    # ── Normal flow: collect → score → preview → pick → render ────────────────
    if not args.images and not args.stem:
        parser.error("At least one of --images or --stem must be supplied.")

    print(f"\n🖼️  Collecting images…")
    images = collect_images(images_dir, args.stem, thumbnails_dir)
    print(f"   {len(images)} image(s) in pool")

    cache_path: Optional[Path] = None
    if args.stem:
        cache_path = thumbnails_dir / f"{args.stem}_scores.json"

    scored   = score_thumbnails(images, cache_path)
    count    = min(args.count, len(scored))
    selected = select_top_images(scored, post["tone"], count)

    print(f"\n   Top {count} for tone '{post['tone']}':")
    for i, p in enumerate(selected, 1):
        print(f"     {i}. {p.name}")

    # ── Render square previews ────────────────────────────────────────────────
    print(f"\n🔍 Rendering {count} square preview(s)…")
    render_previews(selected, post["hook"], name, output_dir, font_path, args.force)

    # Open preview folder so the user can inspect the images
    try:
        subprocess.run(["open", str(preview_dir)], check=False)
    except FileNotFoundError:
        pass  # non-macOS — no-op

    print(f"\n   Previews saved to: {preview_dir}")

    # ── Interactive pick ──────────────────────────────────────────────────────
    choice = prompt_pick(count)
    picked = selected[choice - 1]
    print(f"\n  Picked {choice}: {picked.name}")

    # ── Render final across all channels ─────────────────────────────────────
    print(f"\n🎨 Rendering across {len(channels)} channel(s)…")
    render_final(post["hook"], post["post_text"], picked, name,
                 channels, output_dir, font_path, args.force)
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
