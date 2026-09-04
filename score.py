"""
Moment scorer for Local Clipper.

Perception (transcription) runs locally; JUDGMENT runs on the Claude API. This
module sends candidate segments to Claude with the prompt in prompts/score.txt
and returns them ranked best-first.

The GPU is left untouched here on purpose (CLAUDE.md: the 8GB VRAM is reserved
for Whisper + video, never a local LLM).

Standalone (reads transcript.json, generates candidates, scores, prints ranking):
    venv\\Scripts\\python.exe score.py
    venv\\Scripts\\python.exe score.py --transcript transcript.json --model claude-sonnet-5

As a library:
    from score import score_segments
    ranked = score_segments(candidates)          # -> [{id,start,end,score,reason}, ...]
"""

import argparse
import json
import os
import re
import sys

from dotenv import load_dotenv

PROMPT_PATH = os.path.join("prompts", "score.txt")
# Default scorer per CLAUDE.md: cheap + fast. Sonnet is for final/hard cases.
DEFAULT_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 8192


def load_prompt(path: str = PROMPT_PATH) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def _extract_json_array(text: str) -> list:
    """Parse a JSON array out of the model's reply, tolerating stray wrapping.

    The prompt asks for a bare array, but models occasionally wrap it in prose
    or ```json fences. We strip fences, then fall back to slicing from the first
    '[' to the last ']' before giving up.
    """
    text = text.strip()
    # Strip a leading/trailing code fence if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON array found in model response")
        data = json.loads(text[start:end + 1])  # may still raise; caller handles

    if not isinstance(data, list):
        raise ValueError("model response was not a JSON array")
    return data


def score_segments(candidates: list, model: str = DEFAULT_MODEL,
                   prompt_path: str = PROMPT_PATH) -> list:
    """Score candidates with Claude and return them ranked best-first.

    Each returned item is {id, start, end, score, reason}. start/end are taken
    from OUR candidate (matched by id), not from the model's echo, so a clip is
    always cut at the boundary we chose even if the model rewrites the numbers.
    """
    if not candidates:
        return []

    import anthropic  # local import so `segments.py` alone needs no SDK

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    system = load_prompt(prompt_path)
    payload = json.dumps(
        [{"id": c["id"], "start": c["start"], "end": c["end"], "text": c["text"]}
         for c in candidates],
        ensure_ascii=False,
    )

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": payload}],
    )
    reply = "".join(b.text for b in resp.content if b.type == "text")

    try:
        raw = _extract_json_array(reply)
    except (ValueError, json.JSONDecodeError) as e:
        snippet = reply[:300].replace("\n", " ")
        raise RuntimeError(
            f"Could not parse scores as JSON ({e}). Model said: {snippet!r}"
        ) from e

    by_id = {c["id"]: c for c in candidates}
    scored = []
    for item in raw:
        if not isinstance(item, dict) or "id" not in item:
            continue
        cid = item["id"]
        base = by_id.get(cid)
        if base is None:
            continue  # ignore ids we didn't send
        try:
            sc = int(round(float(item.get("score", 0))))
        except (TypeError, ValueError):
            sc = 0
        sc = max(0, min(100, sc))
        scored.append({
            "id": cid,
            "start": base["start"],
            "end": base["end"],
            "score": sc,
            "reason": str(item.get("reason", "")).strip(),
        })

    if not scored:
        raise RuntimeError("Model returned no usable scores for the candidates.")

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _overlap_frac(a: dict, b: dict) -> float:
    """Fraction of the SHORTER segment covered by the other (0..1).

    Using intersection / min(length) instead of IoU so a short window fully
    nested inside a longer one counts as ~1.0 overlap (IoU would under-report
    it). That's exactly the near-duplicate case we want to catch.
    """
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    if inter <= 0:
        return 0.0
    shorter = min(a["end"] - a["start"], b["end"] - b["start"])
    return inter / shorter if shorter > 0 else 0.0


def select_top_distinct(ranked: list, n: int, overlap_thresh: float = 0.5) -> list:
    """Greedy non-max suppression: take the best clips that don't overlap.

    Walks the already-ranked list best-first and skips any candidate that
    overlaps an already-chosen clip by more than overlap_thresh, so the top N
    are N *distinct* moments rather than several cuts of the same hot moment.
    """
    chosen = []
    for seg in ranked:
        if all(_overlap_frac(seg, c) <= overlap_thresh for c in chosen):
            chosen.append(seg)
            if len(chosen) >= n:
                break
    return chosen


def main():
    ap = argparse.ArgumentParser(description="Score candidate segments with Claude (ranked best-first)")
    ap.add_argument("--transcript", default="transcript.json", help="path to transcript.json")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Claude model id for scoring")
    ap.add_argument("--min", type=float, default=None, help="min window length (s)")
    ap.add_argument("--max", type=float, default=None, help="max window length (s)")
    args = ap.parse_args()

    from segments import generate_segments, MIN_LEN, MAX_LEN

    try:
        with open(args.transcript, encoding="utf-8") as f:
            transcript = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: transcript not found: {args.transcript} (run transcribe.py first)",
              file=sys.stderr)
        sys.exit(1)

    cands = generate_segments(
        transcript,
        min_len=args.min if args.min is not None else MIN_LEN,
        max_len=args.max if args.max is not None else MAX_LEN,
    )
    if not cands:
        print("No candidate segments (transcript too short or empty).", file=sys.stderr)
        sys.exit(1)

    print(f"Scoring {len(cands)} candidates with {args.model} ...", flush=True)
    try:
        ranked = score_segments(cands, args.model)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nRanked segments (best first):")
    for r in ranked:
        print(f"  score {r['score']:>3}  {r['start']:>7.2f}-{r['end']:>7.2f}s  {r['reason']}")


if __name__ == "__main__":
    main()
