"""
Per-clip metadata suggestions for Local Clipper (Phase 8).

JUDGMENT on the Claude API again (never a local LLM): given ONE clip's
transcript, ask claude-haiku-4-5 for a title, description, and hashtags, and
save them as a .txt next to the clip. The prompt lives in prompts/suggest.txt
so it can be edited without touching code.

Standalone (prints a suggestion for a clip window; needs transcript.json):
    venv\\Scripts\\python.exe suggest.py --transcript transcript.json --start 53.24 --end 76.4
    venv\\Scripts\\python.exe suggest.py --text "the raw clip transcript ..."

As a library:
    from suggest import suggest_metadata, write_suggestion
    meta = suggest_metadata(clip_text)          # {title, description, hashtags}
    txt_path = write_suggestion("output/clip.mp4", meta)
"""

import argparse
import json
import os
import re
import sys

from dotenv import load_dotenv

PROMPT_PATH = os.path.join("prompts", "suggest.txt")
# Suggestions are cheap/short -> Haiku, per CLAUDE.md.
DEFAULT_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024


def load_prompt(path: str = PROMPT_PATH) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def _extract_json_object(text: str) -> dict:
    """Parse a JSON object out of the model's reply, tolerating fences/prose."""
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON object found in model response")
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("model response was not a JSON object")
    return data


def _norm_hashtags(raw) -> list:
    """Normalize hashtags to a de-duped list of '#lowercasetag'."""
    if isinstance(raw, str):
        raw = re.split(r"[\s,]+", raw)
    tags, seen = [], set()
    for t in raw or []:
        t = str(t).strip().lower()
        if not t:
            continue
        t = "#" + re.sub(r"[^0-9a-z_]", "", t.lstrip("#"))
        if t == "#" or t in seen:
            continue
        seen.add(t)
        tags.append(t)
    return tags


def clip_text_from_words(words: list, start: float, end: float) -> str:
    """Join the transcript words that fall inside [start, end]."""
    parts = [w["word"] for w in words
             if w["end"] > start and w["start"] < end]
    return "".join(parts).strip()


def suggest_metadata(clip_text: str, model: str = DEFAULT_MODEL,
                     prompt_path: str = PROMPT_PATH) -> dict:
    """Ask Claude for {title, description, hashtags} for one clip's transcript."""
    clip_text = (clip_text or "").strip()
    if not clip_text:
        raise ValueError("clip_text is empty")

    import anthropic  # local import: the rest of the pipeline needs no SDK

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    system = load_prompt(prompt_path)
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": clip_text}],
    )
    reply = "".join(b.text for b in resp.content if b.type == "text")

    try:
        data = _extract_json_object(reply)
    except (ValueError, json.JSONDecodeError) as e:
        snippet = reply[:300].replace("\n", " ")
        raise RuntimeError(
            f"Could not parse suggestion as JSON ({e}). Model said: {snippet!r}"
        ) from e

    return {
        "title": str(data.get("title", "")).strip(),
        "description": str(data.get("description", "")).strip(),
        "hashtags": _norm_hashtags(data.get("hashtags", [])),
    }


def format_suggestion(meta: dict) -> str:
    return (
        f"Title: {meta['title']}\n\n"
        f"Description:\n{meta['description']}\n\n"
        f"Hashtags:\n{' '.join(meta['hashtags'])}\n"
    )


def write_suggestion(clip_path: str, meta: dict) -> str:
    """Write the suggestion as a .txt next to the clip; return its path."""
    txt_path = os.path.splitext(clip_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(format_suggestion(meta))
    return txt_path


def main():
    # Social copy often includes emoji; keep the Windows console from choking.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Suggest title/description/hashtags for a clip (Claude)")
    ap.add_argument("--transcript", default="transcript.json",
                    help="transcript.json to pull the clip text from (with --start/--end)")
    ap.add_argument("--start", type=float, default=None, help="clip start (s)")
    ap.add_argument("--end", type=float, default=None, help="clip end (s)")
    ap.add_argument("--text", default=None, help="use this transcript text directly instead")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Claude model id")
    args = ap.parse_args()

    if args.text:
        text = args.text
    else:
        if args.start is None or args.end is None:
            print("ERROR: pass --text, or --start and --end to slice --transcript.",
                  file=sys.stderr)
            sys.exit(2)
        try:
            with open(args.transcript, encoding="utf-8") as f:
                words = json.load(f)["words"]
        except FileNotFoundError:
            print(f"ERROR: transcript not found: {args.transcript}", file=sys.stderr)
            sys.exit(1)
        text = clip_text_from_words(words, args.start, args.end)

    try:
        meta = suggest_metadata(text, args.model)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    print(format_suggestion(meta))


if __name__ == "__main__":
    main()
