"""
Caption burn-in stage for Local Clipper (Phase 7).

Turns the word-level timestamps in transcript.json into a TikTok / Reels-style
ASS subtitle file: one or two words on screen at a time, big and centered, with
the active (currently-spoken) word highlighted karaoke-style. cut.py burns the
.ass into the clip during the final NVENC encode.

Look & feel is entirely data-driven from caption_style.json (font, size,
colors, words-per-group, position...) so you can restyle without touching code.

Standalone (write an .ass for a clip window and inspect it):
    venv\\Scripts\\python.exe captions.py --transcript transcript.json \\
        --start 1278.6 --end 1302.36 --out sample.ass

As a library:
    from captions import load_style, write_ass
    style = load_style()
    ass_path = write_ass(words, clip_start, clip_end, style, "clip.ass")
"""

import argparse
import json
import os

STYLE_PATH = "caption_style.json"

# Fallback defaults -- used for any key missing from caption_style.json so the
# module still works if the config is trimmed down or absent.
DEFAULT_STYLE = {
    "font": "Arial Black",
    "font_size": 92,
    "bold": True,
    "uppercase": True,
    "words_per_group": 2,
    "max_gap": 0.6,          # start a new group if the pause before a word exceeds this (s)
    "text_color": "#FFFFFF",
    "highlight_color": "#FFE800",
    "outline_color": "#000000",
    "outline_width": 6,
    "shadow": 3,
    "highlight_scale": 112,  # percent; 100 = no pop, 112 = active word 12% bigger
    "alignment": 2,          # ASS numpad: 2 = bottom-center, 5 = middle-center
    "margin_v": 380,         # px from the aligned edge (lifts captions off the bottom)
    "margin_h": 90,
}

# 9:16 canvas the captions are composited onto -- must match cut.py's target
# frame, because the ass filter runs AFTER the crop/scale.
PLAY_W = 1080
PLAY_H = 1920


def load_style(path: str = STYLE_PATH) -> dict:
    """Load caption_style.json over the defaults. Missing file -> defaults."""
    style = dict(DEFAULT_STYLE)
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            style.update(json.load(f))
    return style


def _hex_to_ass(hex_color: str) -> str:
    """#RRGGBB -> ASS &HAABBGGRR (opaque). ASS stores colors byte-reversed."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"expected #RRGGBB, got {hex_color!r}")
    rr, gg, bb = h[0:2], h[2:4], h[4:6]
    return f"&H00{bb}{gg}{rr}".upper()


def _inline_color(hex_color: str) -> str:
    """#RRGGBB -> &Hbbggrr& for inline \\1c overrides (no alpha)."""
    h = hex_color.lstrip("#")
    rr, gg, bb = h[0:2], h[2:4], h[4:6]
    return f"&H{bb}{gg}{rr}&".upper()


def _fmt_ts(t: float) -> str:
    """Seconds -> ASS timestamp H:MM:SS.cc (centiseconds)."""
    if t < 0:
        t = 0.0
    cs = int(round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _clean(word: str, uppercase: bool) -> str:
    w = word.strip()
    # ASS override blocks are delimited by braces; neutralize any stray ones.
    w = w.replace("\\", "").replace("{", "(").replace("}", ")")
    return w.upper() if uppercase else w


def _window_words(words, clip_start, clip_end):
    """Words overlapping [clip_start, clip_end], with times shifted to be
    relative to clip_start and clamped to the clip length."""
    out = []
    dur = clip_end - clip_start
    for w in words:
        ws, we = w["start"], w["end"]
        if we <= clip_start or ws >= clip_end:
            continue
        rs = max(0.0, ws - clip_start)
        re = min(dur, we - clip_start)
        if re <= rs:
            continue
        out.append({"start": rs, "end": re, "word": w["word"]})
    return out


def _group_words(words, per_group, max_gap):
    """Chunk words into display groups of `per_group`, also breaking early when
    the pause before a word exceeds max_gap (so captions reset on real pauses)."""
    groups, cur = [], []
    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            if len(cur) >= per_group or gap > max_gap:
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def build_ass(words, clip_start: float, clip_end: float, style: dict) -> str:
    """Build the full ASS document (as a string) for one clip window."""
    per_group = max(1, int(style["words_per_group"]))
    uppercase = bool(style["uppercase"])
    hl_scale = int(style["highlight_scale"])
    hl_inline = _inline_color(style["highlight_color"])

    win = _window_words(words, clip_start, clip_end)
    groups = _group_words(win, per_group, float(style["max_gap"]))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PLAY_W}
PlayResY: {PLAY_H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Base,{style['font']},{style['font_size']},{_hex_to_ass(style['text_color'])},{_hex_to_ass(style['highlight_color'])},{_hex_to_ass(style['outline_color'])},&H64000000,{-1 if style['bold'] else 0},0,0,0,100,100,0,0,1,{style['outline_width']},{style['shadow']},{style['alignment']},{style['margin_h']},{style['margin_h']},{style['margin_v']},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = []
    for group in groups:
        g_start = group[0]["start"]
        g_end = group[-1]["end"]
        tokens = [_clean(w["word"], uppercase) for w in group]
        for i, w in enumerate(group):
            a_start = w["start"]
            # Hold the highlight on the current word until the next one begins,
            # so there are no un-highlighted flickers between words in a group.
            a_end = group[i + 1]["start"] if i + 1 < len(group) else g_end
            if a_end <= a_start:
                a_end = min(a_start + 0.06, g_end if g_end > a_start else a_start + 0.06)
            parts = []
            for j, tok in enumerate(tokens):
                if j == i:
                    parts.append(f"{{\\1c{hl_inline}\\fscx{hl_scale}\\fscy{hl_scale}}}{tok}{{\\r}}")
                else:
                    parts.append(tok)
            text = " ".join(parts)
            lines.append(
                f"Dialogue: 0,{_fmt_ts(a_start)},{_fmt_ts(a_end)},Base,,0,0,0,,{text}"
            )

    return header + "\n".join(lines) + "\n"


def write_ass(words, clip_start: float, clip_end: float, style: dict,
              out_path: str) -> str:
    """Write the ASS document for [clip_start, clip_end] to out_path."""
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(build_ass(words, clip_start, clip_end, style))
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Generate a TikTok-style ASS caption file for a clip window")
    ap.add_argument("--transcript", required=True, help="transcript.json with word timestamps")
    ap.add_argument("--start", type=float, required=True, help="clip start in seconds (source time)")
    ap.add_argument("--end", type=float, required=True, help="clip end in seconds (source time)")
    ap.add_argument("--out", default="captions.ass", help="output .ass path")
    ap.add_argument("--style", default=STYLE_PATH, help="caption style config json")
    args = ap.parse_args()

    with open(args.transcript, encoding="utf-8") as f:
        result = json.load(f)
    words = result["words"]
    style = load_style(args.style)

    path = write_ass(words, args.start, args.end, style, args.out)
    n = len(_window_words(words, args.start, args.end))
    print(f"Wrote {path} ({n} words in window {args.start:g}-{args.end:g}s)")


if __name__ == "__main__":
    main()
