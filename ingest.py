"""
Input resolver for Local Clipper.

The whole pipeline takes a single --input that can be EITHER a local video file
OR a video URL (YouTube, etc.). This module is the one front door that turns
either into a local file path the rest of the pipeline can use.

Standalone:
    python ingest.py --input https://www.youtube.com/watch?v=...
    python ingest.py --input C:\\path\\to\\video.mp4

As a library:
    from ingest import resolve_input
    local_path = resolve_input(src)   # downloads if src is a URL
"""

import argparse
import os
import re
import sys

MEDIA_DIR = "media"


def is_url(src: str) -> bool:
    return bool(re.match(r"^https?://", src.strip(), re.IGNORECASE))


def _safe_stem(text: str) -> str:
    text = re.sub(r"[^\w\-]+", "_", text).strip("_")
    return text[:80] or "video"


def download_url(url: str, media_dir: str = MEDIA_DIR) -> str:
    """Download a video URL to media_dir and return the local file path."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError as e:
        raise RuntimeError(
            "yt-dlp is not installed. Run: pip install yt-dlp"
        ) from e

    os.makedirs(media_dir, exist_ok=True)

    # Prefer a single progressive/merged mp4 up to 1080p so we don't waste
    # disk or time on 4K sources we only clip to 1080x1920 anyway.
    ydl_opts = {
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(media_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
        # merge_output_format may have changed the extension to .mp4
        if not os.path.exists(path):
            base = os.path.splitext(path)[0]
            alt = base + ".mp4"
            if os.path.exists(alt):
                path = alt
    if not os.path.exists(path):
        raise RuntimeError(f"yt-dlp reported success but no file found at {path}")
    return path


def resolve_input(src: str, media_dir: str = MEDIA_DIR) -> str:
    """Return a local video path for src, downloading first if src is a URL."""
    if is_url(src):
        return download_url(src, media_dir)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Input file not found: {src}")
    return src


def main():
    ap = argparse.ArgumentParser(description="Resolve a video file path or URL to a local file")
    ap.add_argument("--input", required=True, help="local video path or a video URL")
    ap.add_argument("--media-dir", default=MEDIA_DIR, help="where to save downloaded videos")
    args = ap.parse_args()

    try:
        path = resolve_input(args.input, args.media_dir)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    size_mb = os.path.getsize(path) / 1024**2
    print(f"Resolved input -> {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
