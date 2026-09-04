"""
Cutting + reframing stage for Local Clipper.

Takes a single --input (a local video path OR a video URL), a --start and --end
in seconds, cuts that segment, reframes it to a 9:16 vertical 1080x1920 frame,
and encodes with the GPU (h264_nvenc) to /output.

Reframe modes (--fit):
    cover  (default) scale to fill 1080x1920, then center-crop the overflow.
            This is the usual look for vertical social clips (a landscape
            source loses its left/right edges). A later phase replaces the
            static center-crop with face-tracking reframe.
    contain          scale to fit inside 1080x1920, then pad (letterbox) the
            gaps with black. Loses no content.

Standalone:
    venv\\Scripts\\python.exe cut.py --input C:\\path\\to\\video.mp4 --start 12 --end 45
    venv\\Scripts\\python.exe cut.py --input https://www.youtube.com/watch?v=... --start 12 --end 45 --fit contain

As a library:
    from cut import cut_segment
    out_path = cut_segment(input_path, start=12.0, end=45.0)  # -> output/<stem>_12.0-45.0.mp4
"""

import argparse
import os
import shutil
import subprocess
import sys

from ingest import resolve_input

OUTPUT_DIR = "output"
TARGET_W = 1080
TARGET_H = 1920

# CPU scale/crop/pad filters, then hand the frames to NVENC. Keeping the geometry
# on the CPU is simplest and plenty fast for short clips; setsar=1 avoids players
# stretching the result because of a leftover sample aspect ratio.
FILTERS = {
    "cover": (
        f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_W}:{TARGET_H},setsar=1"
    ),
    "contain": (
        f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_W}:{TARGET_H}:-1:-1:color=black,setsar=1"
    ),
}


def _safe_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0] or "clip"


def cut_segment(input_path: str, start: float, end: float,
                fit: str = "cover", output_dir: str = OUTPUT_DIR,
                out_path: str = None) -> str:
    """Resolve input, cut [start, end], reframe to 9:16, encode to out_path.

    Returns the path to the written clip.
    """
    if fit not in FILTERS:
        raise ValueError(f"fit must be one of {sorted(FILTERS)}, got {fit!r}")
    if start < 0:
        raise ValueError(f"--start must be >= 0, got {start}")
    if end <= start:
        raise ValueError(f"--end ({end}) must be greater than --start ({start})")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH. On Windows: winget install Gyan.FFmpeg")

    local_path = resolve_input(input_path)
    print(f"Input resolved -> {local_path}", flush=True)

    duration = end - start
    if out_path is None:
        os.makedirs(output_dir, exist_ok=True)
        stem = _safe_stem(local_path)
        out_path = os.path.join(output_dir, f"{stem}_{start:g}-{end:g}_{fit}.mp4")
    else:
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # -ss before -i is a fast seek; because we re-encode, ffmpeg still lands on
    # an accurate frame at `start`. -t is the clip length (end - start).
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start}",
        "-i", local_path,
        "-t", f"{duration}",
        "-vf", FILTERS[fit],
        "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "23", "-b:v", "0",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        out_path,
    ]

    print(
        f"Cutting {start:g}s -> {end:g}s ({duration:g}s), reframe={fit}, "
        f"encoding {TARGET_W}x{TARGET_H} h264_nvenc ...",
        flush=True,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(out_path):
        tail = (proc.stderr or "").strip().splitlines()[-8:]
        raise RuntimeError("ffmpeg cut/encode failed:\n" + "\n".join(tail))

    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Cut a segment and reframe it to a 9:16 (1080x1920) vertical clip"
    )
    ap.add_argument("--input", required=True, help="local video path or a video URL (e.g. YouTube)")
    ap.add_argument("--start", type=float, required=True, help="segment start in seconds")
    ap.add_argument("--end", type=float, required=True, help="segment end in seconds")
    ap.add_argument("--fit", choices=sorted(FILTERS), default="cover",
                    help="cover = fill+crop (default), contain = fit+pad")
    ap.add_argument("--output-dir", default=OUTPUT_DIR, help="where to write the clip")
    ap.add_argument("--output", default=None, help="explicit output file path (overrides --output-dir)")
    args = ap.parse_args()

    try:
        out = cut_segment(args.input, args.start, args.end, args.fit,
                          args.output_dir, args.output)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    size_mb = os.path.getsize(out) / 1024**2
    print(f"Wrote {out} ({size_mb:.1f} MB, {TARGET_W}x{TARGET_H})", flush=True)


if __name__ == "__main__":
    main()
