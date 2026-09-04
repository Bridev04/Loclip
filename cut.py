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
import json
import os
import shutil
import subprocess
import sys
import tempfile

from ingest import resolve_input
from captions import load_style, write_ass, STYLE_PATH
from transcribe import parse_hms

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


def _ass_filter_path(ass_path: str) -> str:
    """Escape an .ass path for use inside an ffmpeg -vf filtergraph.

    The Windows drive-letter colon is painful here: it is a separator both to
    the filtergraph parser AND to the ass filter's own option parser. Easiest
    to sidestep -- hand ffmpeg a path relative to the cwd (no colon at all) when
    we can, and only fall back to an escaped, quoted absolute path (different
    drive from cwd) where a colon is unavoidable.
    """
    try:
        p = os.path.relpath(ass_path).replace("\\", "/")
    except ValueError:  # different drive on Windows -> no relative path exists
        p = os.path.abspath(ass_path).replace("\\", "/")
    # Escape the colon (if any) and single-quote so neither parser splits on it.
    p = p.replace(":", "\\:")
    return f"'{p}'"


def cut_segment(input_path: str, start: float, end: float,
                fit: str = "cover", output_dir: str = OUTPUT_DIR,
                out_path: str = None, captions: bool = False,
                words: list = None, style: dict = None,
                reframe: bool = False, reframe_cfg: dict = None,
                layout: str = None, facecam: str = None,
                facecam_frac: float = 0.4, loudnorm: bool = True) -> str:
    """Resolve input, cut [start, end], reframe to 9:16, encode to out_path.

    When layout="split", the clip becomes a streamer-style vertical: the facecam
    on top and the gameplay on the bottom (see reframe.build_split_filtergraph).
    The facecam is auto-detected, or set with `facecam` ('x,y,w,h', fractions,
    or a corner name); `facecam_frac` is the top share (default 0.4). Split
    overrides `reframe`/`fit`; it falls back to the static crop if no facecam.

    When reframe=True, a face-tracking dynamic crop follows the main speaker
    (see reframe.py) instead of the static `fit` crop; it falls back to the
    static center-crop when no/too-few faces are detected. --reframe overrides
    --fit (it always produces a cover-style vertical crop).

    When captions=True, an ASS caption file is generated from `words` (a list of
    {start, end, word} spanning the source video) and burned into the clip.
    The subtitle filter is applied AFTER the crop/scale so captions land inside
    the 9:16 frame. `style` defaults to caption_style.json.

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

    # Build the video filters. The split layout needs a filter_complex (two
    # streams stacked), everything else a simple -vf chain. Reframe (dynamic
    # face-tracking crop) or the static `fit` crop produce the 1080x1920 frame.
    # Captions come LAST either way so the ass filter draws on the finished frame.
    complex_graph = None   # set for split layout
    final_label = None
    base_vf = None
    layout_desc = f"fit={fit}"

    if layout == "split":
        from reframe import build_split_filtergraph  # lazy: pulls in cv2/numpy
        res = build_split_filtergraph(local_path, start, end, TARGET_W, TARGET_H,
                                      facecam=facecam, facecam_frac=facecam_frac,
                                      overrides=reframe_cfg)
        if res is None:
            print("Split: no facecam detected -> static center crop.", flush=True)
        else:
            complex_graph, final_label, rect = res
            layout_desc = f"split facecam={rect} top={facecam_frac:g}"

    if complex_graph is None:
        if reframe:
            from reframe import build_reframe_vf  # lazy: pulls in cv2/numpy
            base_vf = build_reframe_vf(local_path, start, end, TARGET_W, TARGET_H,
                                       reframe_cfg)
            if base_vf is None:
                print("Reframe: no/too-few faces detected -> static center crop.", flush=True)
            else:
                layout_desc = "reframe"
        if base_vf is None:
            base_vf = FILTERS[fit]

    ass_path = None
    if captions:
        if not words:
            raise ValueError("captions=True requires `words` (word timestamps)")
        if style is None:
            style = load_style()
        # Temp .ass on the same drive as the output; removed after the encode.
        fd, ass_path = tempfile.mkstemp(suffix=".ass", dir=os.path.dirname(out_path) or ".")
        os.close(fd)
        write_ass(words, start, end, style, ass_path)
        ass_filter = f"ass={_ass_filter_path(ass_path)}"
        if complex_graph is not None:
            complex_graph = f"{complex_graph};{final_label}{ass_filter}[vout]"
            final_label = "[vout]"
        else:
            base_vf = f"{base_vf},{ass_filter}"

    # -ss before -i is a fast seek; because we re-encode, ffmpeg still lands on
    # an accurate frame at `start`. -t is the clip length (end - start).
    cmd = [ffmpeg, "-y", "-ss", f"{start}", "-i", local_path, "-t", f"{duration}"]
    if complex_graph is not None:
        # -map the composited video and the source audio (if any).
        cmd += ["-filter_complex", complex_graph, "-map", final_label, "-map", "0:a?"]
    else:
        cmd += ["-vf", base_vf]
    # Normalize loudness to the social-standard ~-14 LUFS so clips don't swing
    # between quiet and blaring. Single-pass loudnorm is plenty for short clips.
    if loudnorm:
        cmd += ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11"]
    cmd += [
        "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "23", "-b:v", "0",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        out_path,
    ]

    print(
        f"Cutting {start:g}s -> {end:g}s ({duration:g}s), {layout_desc}, "
        f"captions={'on' if captions else 'off'}, "
        f"encoding {TARGET_W}x{TARGET_H} h264_nvenc ...",
        flush=True,
    )
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(out_path):
            tail = (proc.stderr or "").strip().splitlines()[-8:]
            raise RuntimeError("ffmpeg cut/encode failed:\n" + "\n".join(tail))
    finally:
        if ass_path and os.path.exists(ass_path):
            os.remove(ass_path)

    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Cut a segment and reframe it to a 9:16 (1080x1920) vertical clip"
    )
    ap.add_argument("--input", required=True, help="local video path or a video URL (e.g. YouTube)")
    ap.add_argument("--start", type=parse_hms, required=True,
                    help="segment start (seconds or MM:SS / HH:MM:SS)")
    ap.add_argument("--end", type=parse_hms, required=True,
                    help="segment end (seconds or MM:SS / HH:MM:SS)")
    ap.add_argument("--fit", choices=sorted(FILTERS), default="cover",
                    help="cover = fill+crop (default), contain = fit+pad")
    ap.add_argument("--output-dir", default=OUTPUT_DIR, help="where to write the clip")
    ap.add_argument("--output", default=None, help="explicit output file path (overrides --output-dir)")
    ap.add_argument("--reframe", action="store_true",
                    help="face-tracking dynamic crop that follows the speaker (overrides --fit)")
    ap.add_argument("--layout", choices=["split"], default=None,
                    help="split = facecam on top, gameplay on bottom (streamer style)")
    ap.add_argument("--facecam", default=None,
                    help="facecam region for --layout split: 'x,y,w,h' (pixels or fractions) "
                         "or a corner (top-left/tr/bottom-right/...); omit to auto-detect")
    ap.add_argument("--facecam-frac", type=float, default=0.4,
                    help="top share of the frame for the facecam in split layout (default 0.4)")
    ap.add_argument("--no-loudnorm", dest="loudnorm", action="store_false",
                    help="skip audio loudness normalization (on by default, ~-14 LUFS)")
    ap.add_argument("--captions", action="store_true",
                    help="burn TikTok-style captions in (needs --transcript for word timestamps)")
    ap.add_argument("--transcript", default=None,
                    help="transcript.json with word timestamps (required with --captions)")
    ap.add_argument("--style", default=STYLE_PATH, help="caption style config json")
    args = ap.parse_args()

    words = None
    style = None
    if args.captions:
        if not args.transcript:
            print("ERROR: --captions requires --transcript with word timestamps.", file=sys.stderr)
            sys.exit(2)
        with open(args.transcript, encoding="utf-8") as f:
            words = json.load(f)["words"]
        style = load_style(args.style)

    try:
        out = cut_segment(args.input, args.start, args.end, args.fit,
                          args.output_dir, args.output,
                          captions=args.captions, words=words, style=style,
                          reframe=args.reframe, layout=args.layout,
                          facecam=args.facecam, facecam_frac=args.facecam_frac,
                          loudnorm=args.loudnorm)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    size_mb = os.path.getsize(out) / 1024**2
    print(f"Wrote {out} ({size_mb:.1f} MB, {TARGET_W}x{TARGET_H})", flush=True)


if __name__ == "__main__":
    main()
