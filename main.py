"""
Local Clipper pipeline entry point.

Modes:
    --n N    Full pipeline: transcribe -> generate candidate segments -> score
             them with Claude -> cut the top N into 9:16 vertical clips.
    --dumb   Thin end-to-end slice: transcribe, then blindly cut the first 45s.
             No scoring -- kept as a plumbing smoke test.

Standalone:
    venv\\Scripts\\python.exe main.py --input C:\\path\\to\\video.mp4 --n 5
    venv\\Scripts\\python.exe main.py --input https://www.youtube.com/watch?v=... --n 5
    venv\\Scripts\\python.exe main.py --input C:\\path\\to\\video.mp4 --dumb
"""

import argparse
import json
import sys

from transcribe import transcribe, OUTPUT as TRANSCRIPT_OUT, _fmt_hms
from segments import generate_segments
from score import score_segments, select_top_distinct, DEFAULT_MODEL
from cut import cut_segment, OUTPUT_DIR, _safe_stem
from captions import load_style

DUMB_CLIP_SECONDS = 45.0


def _transcribe_and_save(input_path: str) -> dict:
    """Transcribe input and persist transcript.json; return the result dict."""
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
    return result


def run_dumb(input_path: str, fit: str = "cover") -> list:
    """Transcribe, then cut the first DUMB_CLIP_SECONDS into a vertical clip."""
    result = _transcribe_and_save(input_path)
    local_path = result["input"]
    end = min(DUMB_CLIP_SECONDS, result["duration"])

    print("\n== Cutting first clip (dumb) ==", flush=True)
    out = cut_segment(local_path, start=0.0, end=end, fit=fit)
    return [out]


def run_pipeline(input_path: str, n: int, fit: str = "cover",
                 model: str = DEFAULT_MODEL, min_len: float = None,
                 max_len: float = None, overlap: float = 0.5,
                 transcript_path: str = None, captions: bool = True) -> list:
    """Full pipeline: transcribe -> segment -> score -> cut top N clips.

    If transcript_path is given, that transcript is reused instead of
    re-transcribing -- the fast loop for tuning scoring/selection on a video
    you've already transcribed.

    When captions=True (default) each clip gets TikTok-style burned-in captions
    generated from the transcript's word timestamps.
    """
    if transcript_path:
        print(f"== Reusing transcript {transcript_path} ==", flush=True)
        with open(transcript_path, encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = _transcribe_and_save(input_path)
    local_path = result["input"]

    print("\n== Generating candidate segments ==", flush=True)
    seg_kwargs = {}
    if min_len is not None:
        seg_kwargs["min_len"] = min_len
    if max_len is not None:
        seg_kwargs["max_len"] = max_len
    candidates = generate_segments(result, **seg_kwargs)
    print(f"{len(candidates)} candidates.", flush=True)
    if not candidates:
        raise RuntimeError("No candidate segments (transcript too short or empty).")

    print(f"\n== Scoring with {model} ==", flush=True)
    ranked = score_segments(candidates, model)
    # Greedy overlap suppression so the top N are distinct moments, not several
    # cuts of the same hot moment. --overlap 1.0 disables it (pure top-N).
    top = select_top_distinct(ranked, n, overlap)
    print(f"Top picks (from {len(ranked)} scored, overlap<={overlap:g}):", flush=True)
    for i, r in enumerate(top, 1):
        print(f"  #{i}  score {r['score']:>3}  {r['start']:>7.2f}-{r['end']:>7.2f}s  {r['reason']}",
              flush=True)

    words = result["words"] if captions else None
    style = load_style() if captions else None

    print(f"\n== Cutting top {len(top)} clips (captions={'on' if captions else 'off'}) ==",
          flush=True)
    stem = _safe_stem(local_path)
    outs = []
    for i, r in enumerate(top, 1):
        out_path = (f"{OUTPUT_DIR}/{stem}_rank{i:02d}_score{r['score']:02d}"
                    f"_{r['start']:g}-{r['end']:g}.mp4")
        out = cut_segment(local_path, start=r["start"], end=r["end"], fit=fit,
                          out_path=out_path, captions=captions, words=words,
                          style=style)
        outs.append(out)
    return outs


def main():
    ap = argparse.ArgumentParser(description="Local Clipper pipeline")
    ap.add_argument("--input", default=None, help="local video path or a video URL (e.g. YouTube)")
    ap.add_argument("--n", type=int, default=None,
                    help="cut the top N scored clips (full pipeline)")
    ap.add_argument("--dumb", action="store_true",
                    help="plumbing slice: transcribe, then cut the first 45s (no scoring)")
    ap.add_argument("--fit", choices=["cover", "contain"], default="cover",
                    help="9:16 reframe: cover = fill+crop (default), contain = fit+pad")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Claude model id for scoring")
    ap.add_argument("--min", type=float, default=None, help="min candidate window length (s)")
    ap.add_argument("--max", type=float, default=None, help="max candidate window length (s)")
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="max allowed overlap between chosen clips, 0..1 (1.0 = disable dedup)")
    ap.add_argument("--transcript", default=None,
                    help="reuse this transcript.json instead of re-transcribing (--n only)")
    ap.add_argument("--captions", dest="captions", action="store_true", default=True,
                    help="burn TikTok-style captions into each clip (default on)")
    ap.add_argument("--no-captions", dest="captions", action="store_false",
                    help="skip caption burn-in")
    args = ap.parse_args()

    if args.dumb and args.n is not None:
        print("ERROR: choose one of --dumb or --n, not both.", file=sys.stderr)
        sys.exit(2)
    if not args.dumb and args.n is None:
        print("ERROR: no mode selected. Pass --n N for the full pipeline, or --dumb.",
              file=sys.stderr)
        sys.exit(2)
    if not args.input and not args.transcript:
        print("ERROR: pass --input (video/URL), or --transcript to reuse an existing one.",
              file=sys.stderr)
        sys.exit(2)

    try:
        if args.dumb:
            outs = run_dumb(args.input, args.fit)
        else:
            outs = run_pipeline(args.input, args.n, args.fit, args.model,
                                args.min, args.max, args.overlap, args.transcript,
                                args.captions)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. {len(outs)} clip(s) in ./{OUTPUT_DIR}:", flush=True)
    for o in outs:
        print(f"  {o}", flush=True)


if __name__ == "__main__":
    main()
