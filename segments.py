"""
Candidate-segment generator for Local Clipper.

Reads transcript.json (the word-timestamp output of transcribe.py) and proposes
candidate clip windows for the scorer to judge. Windows are 20-90s and their
edges are snapped to sentence/pause boundaries so cuts land cleanly instead of
slicing through the middle of a word or thought.

Each candidate is shaped for prompts/score.txt:
    {"id": <int>, "start": <sec>, "end": <sec>, "text": "<segment transcript>"}

Standalone:
    venv\\Scripts\\python.exe segments.py                       # reads transcript.json
    venv\\Scripts\\python.exe segments.py --transcript other.json --min 15 --max 60

As a library:
    from segments import generate_segments
    candidates = generate_segments(transcript_dict)
"""

import argparse
import json
import re
import sys

TRANSCRIPT = "transcript.json"

# Window length bounds (seconds) and how we sample within them.
MIN_LEN = 20.0
MAX_LEN = 90.0
# Roughly aim for windows near these lengths; for each start we keep the
# boundary-aligned window closest to each target that still fits [MIN, MAX].
TARGET_LENS = (25.0, 40.0, 60.0, 85.0)
# Advance the window start by at least this many seconds between start points so
# the candidate count scales with video length, not with sentence density.
START_STEP = 8.0
# A silence gap this long (seconds) between consecutive words is treated as a
# clean cut point even without sentence punctuation.
PAUSE_GAP = 0.6
# Safety cap so a very long video can't produce a huge (and costly) scoring call.
MAX_CANDIDATES = 60

_SENT_END = re.compile(r"[.!?…][\"')\]]?\s*$")


def _boundary_flags(words: list) -> list:
    """Return is_end[i]: True if word i can end a segment (sentence end / pause / last)."""
    n = len(words)
    is_end = [False] * n
    for i, w in enumerate(words):
        if i == n - 1:
            is_end[i] = True
            continue
        if _SENT_END.search(w["word"]):
            is_end[i] = True
            continue
        gap = words[i + 1]["start"] - w["end"]
        if gap >= PAUSE_GAP:
            is_end[i] = True
    return is_end


def generate_segments(transcript: dict, min_len: float = MIN_LEN,
                      max_len: float = MAX_LEN, start_step: float = START_STEP,
                      targets=TARGET_LENS, max_candidates: int = MAX_CANDIDATES) -> list:
    """Build sentence/pause-aligned candidate windows from a transcript dict."""
    words = transcript.get("words") or []
    if not words:
        return []

    is_end = _boundary_flags(words)
    n = len(words)
    # A clean start is the first word, or any word that follows a segment end.
    start_idxs = [i for i in range(n) if i == 0 or is_end[i - 1]]
    end_idxs = [i for i in range(n) if is_end[i]]

    seen = set()
    candidates = []
    last_start_t = None

    for si in start_idxs:
        s_t = words[si]["start"]
        # Thin start points so windows don't cluster on dense sentences.
        if last_start_t is not None and (s_t - last_start_t) < start_step:
            continue
        last_start_t = s_t

        for target in targets:
            want = s_t + target
            best_ei = None
            best_diff = None
            for ei in end_idxs:
                if ei < si:
                    continue
                dur = words[ei]["end"] - s_t
                if dur < min_len:
                    continue
                if dur > max_len:
                    break  # end_idxs is time-sorted; nothing longer will fit
                diff = abs(words[ei]["end"] - want)
                if best_diff is None or diff < best_diff:
                    best_diff, best_ei = diff, ei
            if best_ei is None:
                continue
            key = (si, best_ei)
            if key in seen:
                continue
            seen.add(key)
            text = "".join(w["word"] for w in words[si:best_ei + 1]).strip()
            candidates.append({
                "start": round(s_t, 3),
                "end": round(words[best_ei]["end"], 3),
                "text": text,
            })

    candidates.sort(key=lambda c: (c["start"], c["end"]))
    if len(candidates) > max_candidates:
        # Keep an even spread across the timeline rather than just the front.
        step = len(candidates) / max_candidates
        candidates = [candidates[int(i * step)] for i in range(max_candidates)]

    for i, c in enumerate(candidates):
        c["id"] = i
    # id first for readability in the JSON payload
    return [{"id": c["id"], "start": c["start"], "end": c["end"], "text": c["text"]}
            for c in candidates]


def main():
    ap = argparse.ArgumentParser(description="Generate candidate clip segments from a transcript")
    ap.add_argument("--transcript", default=TRANSCRIPT, help="path to transcript.json")
    ap.add_argument("--min", type=float, default=MIN_LEN, help="min window length (s)")
    ap.add_argument("--max", type=float, default=MAX_LEN, help="max window length (s)")
    ap.add_argument("--step", type=float, default=START_STEP, help="min seconds between window starts")
    ap.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES, help="cap on number of candidates")
    ap.add_argument("--output", default=None, help="write candidates JSON here (default: stdout summary)")
    args = ap.parse_args()

    try:
        with open(args.transcript, encoding="utf-8") as f:
            transcript = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: transcript not found: {args.transcript} (run transcribe.py first)",
              file=sys.stderr)
        sys.exit(1)

    cands = generate_segments(transcript, args.min, args.max, args.step,
                              max_candidates=args.max_candidates)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(cands, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(cands)} candidates -> {args.output}")
    else:
        print(f"{len(cands)} candidate segments:")
        for c in cands:
            preview = c["text"][:70] + ("…" if len(c["text"]) > 70 else "")
            print(f"  [{c['id']:>2}] {c['start']:>7.2f}-{c['end']:>7.2f}s  {preview}")


if __name__ == "__main__":
    main()
