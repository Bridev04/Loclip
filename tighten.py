"""
Silence / filler tightening for Local Clipper.

Speech has dead air: pauses between sentences, "uhh" gaps, breath. Cutting those
makes a clip punchier without changing what's said. Given the word timestamps we
already have, this computes the spans worth KEEPING (speech, with a little
breathing room) and collapses long gaps -- then cut.py drops the rest with an
ffmpeg select/aselect and re-times the result.

Because removing gaps shifts the timeline, captions have to move too. map_time()
converts an original timestamp to its position on the tightened timeline, and
remap_words() rebuilds the word list so burned captions still land on the word.

Pure functions, no ffmpeg/cv2 here -- easy to reason about and unit-test.
"""

# Defaults: collapse any pause longer than MAX_GAP down to KEEP_PAD of silence,
# and leave EDGE_PAD of air at the very start/end so it doesn't feel clipped.
MAX_GAP = 0.6
KEEP_PAD = 0.12
EDGE_PAD = 0.15


def keep_spans(words: list, clip_start: float, clip_end: float,
               max_gap: float = MAX_GAP, keep_pad: float = KEEP_PAD,
               edge_pad: float = EDGE_PAD) -> list:
    """Absolute-time spans to KEEP within [clip_start, clip_end].

    Walks the words inside the clip; whenever the gap between one word's end and
    the next word's start exceeds max_gap, the span is broken (leaving keep_pad
    of silence on each side of the cut). Leading/trailing silence is trimmed to
    edge_pad. Returns merged, non-overlapping (a, b) spans; a single span
    covering the whole clip means "nothing worth cutting".
    """
    ws = [w for w in words if w["end"] > clip_start and w["start"] < clip_end]
    if not ws:
        return [(clip_start, clip_end)]

    spans = []
    cur_a = max(clip_start, ws[0]["start"] - edge_pad)
    cur_b = min(clip_end, ws[0]["end"])
    for i in range(1, len(ws)):
        gap = ws[i]["start"] - ws[i - 1]["end"]
        if gap > max_gap:
            cur_b = min(clip_end, ws[i - 1]["end"] + keep_pad)
            spans.append((cur_a, cur_b))
            cur_a = max(clip_start, ws[i]["start"] - keep_pad)
        cur_b = min(clip_end, ws[i]["end"])
    spans.append((cur_a, min(clip_end, ws[-1]["end"] + edge_pad)))

    merged = []
    for a, b in spans:
        if b <= a:
            continue
        if merged and a <= merged[-1][1] + 1e-3:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged or [(clip_start, clip_end)]


def kept_duration(spans: list) -> float:
    return sum(b - a for a, b in spans)


def map_time(t: float, spans: list) -> float:
    """Map an absolute time onto the tightened timeline (clip-relative seconds).

    Time inside a kept span maps to its offset within the kept total; time in a
    removed gap clamps to the boundary of the surrounding kept material."""
    rel = 0.0
    for a, b in spans:
        if t < a:
            return rel
        if t <= b:
            return rel + (t - a)
        rel += (b - a)
    return rel


def remap_words(words: list, spans: list, clip_start: float) -> list:
    """Rebuild words on the tightened timeline for caption burn-in.

    Each word's start/end are mapped through map_time and shifted back by
    clip_start so captions.write_ass (which subtracts the clip start) produces
    tightened-relative times. Words fully inside a removed gap collapse to zero
    length and are dropped."""
    out = []
    for w in words:
        ns = clip_start + map_time(w["start"], spans)
        ne = clip_start + map_time(w["end"], spans)
        if ne - ns <= 1e-3:
            continue
        out.append({"word": w["word"], "start": round(ns, 3), "end": round(ne, 3)})
    return out


def select_expr(spans: list, clip_start: float) -> str:
    """ffmpeg select/aselect expression keeping the spans (clip-relative time).

    -ss resets input time to 0 at clip_start, so spans are shifted by -clip_start.
    The sum of between() terms is non-zero exactly on kept material. Wrap the
    result in single quotes in the filtergraph so its commas are literal."""
    terms = [f"between(t,{max(0.0, a - clip_start):.3f},{max(0.0, b - clip_start):.3f})"
             for a, b in spans]
    return "+".join(terms)


if __name__ == "__main__":
    # Tiny self-check.
    words = [{"word": "a", "start": 1.0, "end": 1.4},
             {"word": "b", "start": 1.5, "end": 1.9},
             {"word": "c", "start": 5.0, "end": 5.4}]  # 3.1s gap before c
    spans = keep_spans(words, 0.0, 7.0)
    print("spans:", spans, "kept:", round(kept_duration(spans), 3))
    print("map 1.0 ->", round(map_time(1.0, spans), 3),
          "| map 5.0 ->", round(map_time(5.0, spans), 3))
    print("select:", select_expr(spans, 0.0))
