"""
Audio-energy signal for Local Clipper (Phase 5).

Perception, so it runs LOCALLY on the CPU with librosa — the 8GB VRAM stays
reserved for Whisper + video (CLAUDE.md rule). For each candidate segment we
read the source audio and derive an energy score from how LOUD and how PUNCHY
the delivery is:
  - RMS loudness (mean + a high percentile) catches sustained volume and peaks.
  - Onset/peak strength catches sudden bursts -- laughs, applause, emphatic
    delivery -- that a flat loudness average would miss.
These are normalized to 0..1 ACROSS the video's own candidates (a moment is
"high energy" relative to the rest of THIS video, not some absolute dB).

The blend then mixes this with Claude's moment score so a genuinely exciting
beat can surface even if the transcript text alone read as unremarkable:
    blended = w * energy + (1 - w) * (llm_score / 100)      # w defaults to 0.3

This is purely additive re-ranking. It does NOT touch prompts/score.txt or the
scoring call -- it only reorders what the scorer already returned, before the
overlap suppression in score.select_top_distinct.

Standalone (reads transcript.json for candidates, prints per-segment energy):
    venv\\Scripts\\python.exe energy.py
    venv\\Scripts\\python.exe energy.py --input media\\j2rszuZ-9PY.mp4
    venv\\Scripts\\python.exe energy.py --transcript transcript.json --min 15 --max 60

As a library:
    from energy import compute_energies, blend_scores
    energies = compute_energies(video_path, candidates)   # -> {id: 0..1}
    reranked = blend_scores(ranked, energies, weight=0.3)  # re-sorted best-first
"""

import argparse
import json
import os
import sys
import tempfile

SR = 16000            # match transcribe.py's 16 kHz mono WAV
HOP = 512             # analysis hop for RMS / onset envelopes
# Default blend: energy is a supporting signal, judgment stays with Claude.
DEFAULT_ENERGY_WEIGHT = 0.3


def _extract_wav(video_path: str) -> str:
    """Extract a 16 kHz mono WAV, reusing transcribe.py's ffmpeg path.

    Decoding once here (rather than letting librosa/audioread open the mp4)
    keeps us on the same audio the transcript was built from and avoids a
    second, slower video decode.
    """
    from transcribe import extract_audio
    tmp_wav = os.path.join(tempfile.gettempdir(), "clipper_energy_audio.wav")
    return extract_audio(video_path, tmp_wav)


def _minmax(vals: list) -> list:
    """Min-max normalize to 0..1. All-equal (or single) -> 0.5 (neutral)."""
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0:
        return [0.5] * len(vals)
    return [(v - lo) / span for v in vals]


def _slice_stats(env, s_frame: int, e_frame: int):
    """(mean, 95th-percentile) of an envelope over [s_frame, e_frame).

    Returns (0.0, 0.0) for an empty/out-of-range slice so a candidate past the
    end of the audio contributes no energy instead of blowing up.
    """
    import numpy as np
    seg = env[max(0, s_frame):max(0, e_frame)]
    if seg.size == 0:
        return 0.0, 0.0
    return float(np.mean(seg)), float(np.percentile(seg, 95))


def compute_energies(video_path: str, candidates: list, sr: int = SR,
                     hop: int = HOP) -> dict:
    """Map each candidate id -> audio-energy score in 0..1.

    Energy blends loudness (RMS) and punchiness (onset strength); each raw
    feature is min-max normalized across THIS video's candidates, averaged, and
    the combined score is min-max normalized once more so it spans a full 0..1.
    """
    if not candidates:
        return {}

    import librosa
    import numpy as np

    wav = _extract_wav(video_path)
    try:
        y, sr = librosa.load(wav, sr=sr, mono=True)
    finally:
        if os.path.exists(wav):
            os.remove(wav)

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    ids, rms_mean, rms_peak, ons_mean, ons_peak = [], [], [], [], []
    for c in candidates:
        s_frame = int(c["start"] * sr / hop)
        e_frame = int(c["end"] * sr / hop)
        rm, rp = _slice_stats(rms, s_frame, e_frame)
        om, op = _slice_stats(onset, s_frame, e_frame)
        ids.append(c["id"])
        rms_mean.append(rm)
        rms_peak.append(rp)
        ons_mean.append(om)
        ons_peak.append(op)

    # Normalize each feature across candidates, then fold into one score. The
    # loudness pair and the onset pair each get half the weight so a loud-but-
    # flat passage and a punchy one are both reachable.
    nrm_mean = _minmax(rms_mean)
    nrm_peak = _minmax(rms_peak)
    nons_mean = _minmax(ons_mean)
    nons_peak = _minmax(ons_peak)

    combined = [
        0.5 * (0.5 * a + 0.5 * b) + 0.5 * (0.5 * c + 0.5 * d)
        for a, b, c, d in zip(nrm_mean, nrm_peak, nons_mean, nons_peak)
    ]
    energy = _minmax(combined)
    return {cid: round(e, 4) for cid, e in zip(ids, energy)}


def blend_scores(ranked: list, energies: dict,
                 weight: float = DEFAULT_ENERGY_WEIGHT) -> list:
    """Re-rank scored segments by a blend of energy and the LLM score.

    ``ranked`` is score.score_segments output ({id,start,end,score,reason}).
    Each item gains an ``energy`` (0..1) and ``blended`` (0..1) field, and the
    list is re-sorted best-first by ``blended``. weight is the energy share
    (0 = ignore energy / pure LLM order, 1 = pure energy).
    """
    weight = max(0.0, min(1.0, weight))
    out = []
    for r in ranked:
        e = energies.get(r["id"], 0.0)
        llm = max(0.0, min(1.0, r["score"] / 100.0))
        item = dict(r)
        item["energy"] = round(e, 4)
        item["blended"] = round(weight * e + (1.0 - weight) * llm, 4)
        out.append(item)
    out.sort(key=lambda x: x["blended"], reverse=True)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Compute per-segment audio-energy scores (0..1) with librosa")
    ap.add_argument("--transcript", default="transcript.json",
                    help="transcript.json to source candidate windows from")
    ap.add_argument("--input", default=None,
                    help="video to read audio from (default: the transcript's own input)")
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

    video = args.input or transcript.get("input")
    if not video:
        print("ERROR: no video path (pass --input or use a transcript with an 'input' field).",
              file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(video):
        print(f"ERROR: video not found: {video}", file=sys.stderr)
        sys.exit(1)

    cands = generate_segments(
        transcript,
        min_len=args.min if args.min is not None else MIN_LEN,
        max_len=args.max if args.max is not None else MAX_LEN,
    )
    if not cands:
        print("No candidate segments (transcript too short or empty).", file=sys.stderr)
        sys.exit(1)

    print(f"Computing audio energy for {len(cands)} candidates from {video} ...", flush=True)
    try:
        energies = compute_energies(video, cands)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    text_by_id = {c["id"]: c["text"] for c in cands}
    span_by_id = {c["id"]: (c["start"], c["end"]) for c in cands}
    print("\nPer-segment energy (highest first):")
    for cid, e in sorted(energies.items(), key=lambda kv: kv[1], reverse=True):
        start, end = span_by_id[cid]
        preview = text_by_id[cid][:60] + ("…" if len(text_by_id[cid]) > 60 else "")
        print(f"  energy {e:.3f}  [{cid:>2}] {start:>7.2f}-{end:>7.2f}s  {preview}")


if __name__ == "__main__":
    main()
