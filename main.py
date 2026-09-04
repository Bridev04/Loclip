"""
Local Clipper pipeline entry point.

Modes:
    --n N    Full pipeline: transcribe -> generate candidate segments -> score
             them with Claude -> reframe + caption -> cut the top N into 9:16
             vertical clips. Add --suggest for a title/description/hashtags .txt
             next to each clip.
    --dumb   Thin end-to-end slice: transcribe, then blindly cut the first 45s.
             No scoring -- kept as a plumbing smoke test.

--input may be a single video, a video URL, OR a folder of videos (batch: every
video in the folder is run through the pipeline).

Standalone:
    venv\\Scripts\\python.exe main.py --input C:\\path\\to\\video.mp4 --n 5
    venv\\Scripts\\python.exe main.py --input https://www.youtube.com/watch?v=... --n 5
    venv\\Scripts\\python.exe main.py --input C:\\path\\to\\videos_folder --n 5 --suggest
    venv\\Scripts\\python.exe main.py --input C:\\path\\to\\video.mp4 --dumb
"""

import argparse
import glob
import json
import os
import re
import sys

from transcribe import (transcribe, OUTPUT as TRANSCRIPT_OUT, _fmt_hms,
                        parse_hms, BATCH_SIZE)
from segments import generate_segments
from score import (score_segments_2stage, select_top_distinct, DEFAULT_MODEL,
                   DEFAULT_RANK_MODEL, DEFAULT_REFINE_TOP)
from energy import compute_energies, blend_scores, DEFAULT_ENERGY_WEIGHT
from cut import cut_segment, OUTPUT_DIR, _safe_stem
from captions import load_style

DUMB_CLIP_SECONDS = 45.0
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".wmv"}


def _gather_inputs(input_path: str) -> list:
    """Expand --input into a list of inputs. A folder -> every video file in it
    (sorted); a URL or single file -> itself; None -> [None] (transcript reuse)."""
    if not input_path:
        return [None]
    if re.match(r"^https?://", input_path, re.IGNORECASE):
        return [input_path]
    if os.path.isdir(input_path):
        files = [p for p in sorted(glob.glob(os.path.join(input_path, "*")))
                 if os.path.splitext(p)[1].lower() in VIDEO_EXTS]
        if not files:
            raise RuntimeError(f"No video files found in folder: {input_path}")
        return files
    return [input_path]


def _transcribe_and_save(input_path: str, start: float = None,
                         end: float = None, batch_size: int = BATCH_SIZE,
                         cache: bool = True) -> dict:
    """Transcribe input and persist transcript.json; return the result dict."""
    print("== Transcribing ==", flush=True)
    result = transcribe(input_path, start=start, end=end, batch_size=batch_size,
                        cache=cache)
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
                 transcript_path: str = None, captions: bool = True,
                 reframe: bool = True, suggest: bool = False,
                 energy_weight: float = DEFAULT_ENERGY_WEIGHT,
                 start: float = None, end: float = None,
                 layout: str = None, facecam: str = None,
                 facecam_frac: float = 0.4, batch_size: int = BATCH_SIZE,
                 loudnorm: bool = True, rank_model: str = DEFAULT_RANK_MODEL,
                 refine_top: int = DEFAULT_REFINE_TOP, cache: bool = True) -> list:
    """Full pipeline: transcribe -> segment -> score -> cut top N clips.

    If transcript_path is given, that transcript is reused instead of
    re-transcribing -- the fast loop for tuning scoring/selection on a video
    you've already transcribed.

    When reframe=True (default) each clip's 9:16 crop follows the main speaker
    (face tracking, with a static-crop fallback). When captions=True (default)
    each clip gets TikTok-style burned-in captions from the word timestamps.
    When suggest=True, a title/description/hashtags .txt is written next to each
    clip (one extra Haiku call per clip; a failure is warned, not fatal).
    """
    if transcript_path:
        print(f"== Reusing transcript {transcript_path} ==", flush=True)
        with open(transcript_path, encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = _transcribe_and_save(input_path, start=start, end=end,
                                      batch_size=batch_size, cache=cache)
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

    stage2 = refine_top > 0 and rank_model and rank_model != model
    print(f"\n== Scoring with {model}"
          f"{f' + {rank_model} (top {refine_top})' if stage2 else ''} ==", flush=True)
    ranked = score_segments_2stage(candidates, model, rank_model, refine_top)

    # Phase 5: blend a local audio-energy signal into the ranking BEFORE overlap
    # suppression, so a genuinely punchy beat (laughs, emphatic delivery) can
    # surface even when the transcript text alone read as unremarkable. Energy
    # is CPU/librosa perception -- the VRAM stays free. Best-effort: a decode or
    # librosa failure falls back to the pure LLM order rather than losing clips.
    if energy_weight > 0:
        print(f"\n== Blending audio energy (weight {energy_weight:g}) ==", flush=True)
        try:
            energies = compute_energies(local_path, candidates)
            ranked = blend_scores(ranked, energies, energy_weight)
            print(f"Re-ranked {len(ranked)} candidates by blended score.", flush=True)
        except Exception as e:
            print(f"WARNING: energy blend skipped ({type(e).__name__}: {e}); "
                  f"using LLM order.", file=sys.stderr, flush=True)

    # Greedy overlap suppression so the top N are distinct moments, not several
    # cuts of the same hot moment. --overlap 1.0 disables it (pure top-N).
    top = select_top_distinct(ranked, n, overlap)
    print(f"Top picks (from {len(ranked)} scored, overlap<={overlap:g}):", flush=True)
    for i, r in enumerate(top, 1):
        extra = ""
        if "blended" in r:
            extra = f" energy {r['energy']:.2f} blended {r['blended']:.2f}"
        print(f"  #{i}  score {r['score']:>3}{extra}  "
              f"{r['start']:>7.2f}-{r['end']:>7.2f}s  {r['reason']}", flush=True)

    words = result["words"] if captions else None
    style = load_style() if captions else None
    text_by_id = {c["id"]: c["text"] for c in candidates}

    crop_desc = "split(facecam/gameplay)" if layout == "split" else \
                ("reframe" if reframe else "static")
    print(f"\n== Cutting top {len(top)} clips "
          f"(crop={crop_desc}, "
          f"captions={'on' if captions else 'off'}, "
          f"suggest={'on' if suggest else 'off'}) ==", flush=True)
    stem = _safe_stem(local_path)
    outs = []
    for i, r in enumerate(top, 1):
        out_path = (f"{OUTPUT_DIR}/{stem}_rank{i:02d}_score{r['score']:02d}"
                    f"_{r['start']:g}-{r['end']:g}.mp4")
        out = cut_segment(local_path, start=r["start"], end=r["end"], fit=fit,
                          out_path=out_path, captions=captions, words=words,
                          style=style, reframe=reframe, layout=layout,
                          facecam=facecam, facecam_frac=facecam_frac,
                          loudnorm=loudnorm)
        outs.append(out)
        _write_clip_meta(out, i, r)
        if suggest:
            _write_suggestion(out, text_by_id.get(r["id"], ""))
    return outs


def _write_clip_meta(clip_path: str, rank: int, r: dict):
    """Write a <clip>.meta.json sidecar so the gallery can show why it was picked
    (score, the scorer's reason, and the audio-energy/blended figures)."""
    meta = {"rank": rank, "score": r.get("score"), "start": r.get("start"),
            "end": r.get("end"), "reason": r.get("reason", "")}
    if "energy" in r:
        meta["energy"] = r["energy"]
        meta["blended"] = r.get("blended")
    try:
        path = os.path.splitext(clip_path)[0] + ".meta.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
    except OSError:
        pass  # best-effort; never lose a clip over its sidecar


def _write_suggestion(clip_path: str, clip_text: str):
    """Best-effort per-clip title/description/hashtags .txt. Never fatal."""
    from suggest import suggest_metadata, write_suggestion
    try:
        meta = suggest_metadata(clip_text)
        txt = write_suggestion(clip_path, meta)
        print(f"    suggest -> {txt}: {meta['title']}", flush=True)
    except Exception as e:
        print(f"    suggest skipped ({type(e).__name__}: {e})", flush=True)


def main():
    # --suggest copy can include emoji; keep the Windows console from choking.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Local Clipper pipeline")
    ap.add_argument("--input", default=None, help="local video path or a video URL (e.g. YouTube)")
    ap.add_argument("--n", type=int, default=None,
                    help="cut the top N scored clips (full pipeline)")
    ap.add_argument("--dumb", action="store_true",
                    help="plumbing slice: transcribe, then cut the first 45s (no scoring)")
    ap.add_argument("--fit", choices=["cover", "contain"], default="cover",
                    help="9:16 reframe: cover = fill+crop (default), contain = fit+pad")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Claude model id for the first-pass scorer (shortlist)")
    ap.add_argument("--rank-model", default=DEFAULT_RANK_MODEL,
                    help="model that re-ranks the shortlist (two-stage; default Sonnet)")
    ap.add_argument("--refine-top", type=int, default=DEFAULT_REFINE_TOP,
                    help=f"how many top candidates the rank-model re-scores "
                         f"(default {DEFAULT_REFINE_TOP}; 0 = single-stage, shortlist only)")
    ap.add_argument("--min", type=float, default=None, help="min candidate window length (s)")
    ap.add_argument("--max", type=float, default=None, help="max candidate window length (s)")
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="max allowed overlap between chosen clips, 0..1 (1.0 = disable dedup)")
    ap.add_argument("--transcript", default=None,
                    help="reuse this transcript.json instead of re-transcribing (--n only)")
    ap.add_argument("--reframe", dest="reframe", action="store_true", default=True,
                    help="face-track the speaker when cropping to 9:16 (default on)")
    ap.add_argument("--no-reframe", dest="reframe", action="store_false",
                    help="use a static center crop instead of face tracking")
    ap.add_argument("--split", dest="layout", action="store_const", const="split", default=None,
                    help="streamer layout: facecam on top, gameplay on bottom (overrides reframe)")
    ap.add_argument("--facecam", default=None,
                    help="facecam region for --split: 'x,y,w,h' (pixels or fractions) or a "
                         "corner (top-left/tr/bottom-left/br/...); omit to auto-detect")
    ap.add_argument("--facecam-frac", type=float, default=0.4,
                    help="top share of the frame for the facecam with --split (default 0.4)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help=f"transcription batch size (default {BATCH_SIZE}; lower if you hit "
                         f"GPU out-of-memory, 1 = sequential)")
    ap.add_argument("--no-loudnorm", dest="loudnorm", action="store_false",
                    help="skip audio loudness normalization (on by default, ~-14 LUFS)")
    ap.add_argument("--no-cache", dest="cache", action="store_false",
                    help="force a fresh transcription instead of reusing the transcripts/ cache")
    ap.add_argument("--captions", dest="captions", action="store_true", default=True,
                    help="burn TikTok-style captions into each clip (default on)")
    ap.add_argument("--no-captions", dest="captions", action="store_false",
                    help="skip caption burn-in")
    ap.add_argument("--suggest", action="store_true",
                    help="write a title/description/hashtags .txt next to each clip (Claude)")
    ap.add_argument("--energy-weight", type=float, default=DEFAULT_ENERGY_WEIGHT,
                    help="audio-energy share when re-ranking, 0..1 "
                         f"(default {DEFAULT_ENERGY_WEIGHT:g}; 0 = pure LLM order)")
    ap.add_argument("--start", type=parse_hms, default=None,
                    help="only clip from this time of the source (seconds or MM:SS / HH:MM:SS); "
                         "transcribes just this window so a long video needn't be done in full")
    ap.add_argument("--end", type=parse_hms, default=None,
                    help="only clip up to this time of the source (seconds or MM:SS / HH:MM:SS)")
    args = ap.parse_args()

    if args.dumb and args.n is not None:
        print("ERROR: choose one of --dumb or --n, not both.", file=sys.stderr)
        sys.exit(2)
    if not args.dumb and args.n is None:
        print("ERROR: no mode selected. Pass --n N for the full pipeline, or --dumb.",
              file=sys.stderr)
        sys.exit(2)
    if not args.input and not args.transcript:
        print("ERROR: pass --input (video/URL/folder), or --transcript to reuse an existing one.",
              file=sys.stderr)
        sys.exit(2)
    if args.start is not None and args.end is not None and args.end <= args.start:
        print("ERROR: --end must be greater than --start.", file=sys.stderr)
        sys.exit(2)
    if (args.start is not None or args.end is not None) and args.transcript:
        print("Note: --start/--end are ignored when reusing --transcript "
              "(the transcript already fixes the range).", flush=True)

    try:
        inputs = _gather_inputs(args.input)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    batch = len(inputs) > 1
    if batch:
        print(f"== Batch: {len(inputs)} videos ==", flush=True)
        if args.transcript:
            print("Note: --transcript is ignored for a folder (each video is transcribed).",
                  flush=True)

    all_outs, failures = [], []
    for idx, inp in enumerate(inputs, 1):
        if batch:
            print(f"\n===== [{idx}/{len(inputs)}] {inp} =====", flush=True)
        try:
            if args.dumb:
                outs = run_dumb(inp, args.fit)
            else:
                # --transcript reuse only makes sense for a single input.
                transcript = None if batch else args.transcript
                outs = run_pipeline(inp, args.n, args.fit, args.model,
                                    args.min, args.max, args.overlap, transcript,
                                    args.captions, args.reframe, args.suggest,
                                    args.energy_weight, args.start, args.end,
                                    args.layout, args.facecam, args.facecam_frac,
                                    args.batch_size, args.loudnorm,
                                    args.rank_model, args.refine_top, args.cache)
            all_outs.extend(outs)
        except Exception as e:
            msg = f"{inp}: {type(e).__name__}: {e}"
            if not batch:
                print(f"ERROR: {msg}", file=sys.stderr)
                sys.exit(1)
            print(f"ERROR (skipping): {msg}", file=sys.stderr)
            failures.append(msg)

    print(f"\nDone. {len(all_outs)} clip(s) in ./{OUTPUT_DIR}:", flush=True)
    for o in all_outs:
        print(f"  {o}", flush=True)
    if failures:
        print(f"\n{len(failures)} video(s) failed:", flush=True)
        for m in failures:
            print(f"  {m}", flush=True)


if __name__ == "__main__":
    main()
