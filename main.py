"""
Local Clipper pipeline entry point.

For now this exposes one end-to-end mode so the whole pipeline can be run start
to finish in a single command:

    --dumb   Transcribe the input, then blindly cut the first 45 seconds into a
             9:16 vertical clip. No moment scoring yet -- this is the thin
             end-to-end slice (dumb cutter) the build order calls for; smarter
             moment detection replaces the "first 45s" rule in a later phase.

Standalone:
    venv\\Scripts\\python.exe main.py --input C:\\path\\to\\video.mp4 --dumb
    venv\\Scripts\\python.exe main.py --input https://www.youtube.com/watch?v=... --dumb
"""

import argparse
import json
import sys

from transcribe import transcribe, OUTPUT as TRANSCRIPT_OUT, _fmt_hms
from cut import cut_segment

DUMB_CLIP_SECONDS = 45.0


def run_dumb(input_path: str, fit: str = "cover") -> str:
    """Transcribe input, then cut the first DUMB_CLIP_SECONDS into a vertical clip."""
    # transcribe() resolves the input (downloading a URL if needed) and returns
    # the resolved local path in result["input"], which we reuse for the cut so
    # a URL isn't downloaded twice.
    print("== Transcribing ==", flush=True)
    result = transcribe(input_path)
    with open(TRANSCRIPT_OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(
        f"Wrote {TRANSCRIPT_OUT}: duration {_fmt_hms(result['duration'])} "
        f"({result['duration']:.1f}s), {result['word_count']} words, "
        f"lang={result['language']}",
        flush=True,
    )

    local_path = result["input"]
    end = min(DUMB_CLIP_SECONDS, result["duration"])

    print("\n== Cutting first clip ==", flush=True)
    out = cut_segment(local_path, start=0.0, end=end, fit=fit)
    return out


def main():
    ap = argparse.ArgumentParser(description="Local Clipper pipeline")
    ap.add_argument("--input", required=True, help="local video path or a video URL (e.g. YouTube)")
    ap.add_argument("--dumb", action="store_true",
                    help="thin end-to-end slice: transcribe, then cut the first 45s")
    ap.add_argument("--fit", choices=["cover", "contain"], default="cover",
                    help="9:16 reframe: cover = fill+crop (default), contain = fit+pad")
    args = ap.parse_args()

    if not args.dumb:
        print("ERROR: no mode selected. Pass --dumb to run the end-to-end slice.",
              file=sys.stderr)
        sys.exit(2)

    try:
        out = run_dumb(args.input, args.fit)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. Clip -> {out}", flush=True)


if __name__ == "__main__":
    main()
