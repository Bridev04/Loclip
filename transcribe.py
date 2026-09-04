"""
Transcription stage for Local Clipper.

Takes a single --input (a local video path OR a video URL), resolves it to a
local file via ingest.resolve_input, extracts a 16 kHz mono WAV with ffmpeg,
then runs faster-whisper large-v3 (int8) on the GPU with word timestamps.

Writes transcript.json:
    {
      "input": "<resolved local path>",
      "language": "en",
      "duration": 123.45,           # seconds of audio
      "word_count": 512,
      "text": "the full transcript ...",
      "words": [ {"word": " Hello", "start": 0.12, "end": 0.38}, ... ]
    }

Standalone:
    venv\\Scripts\\python.exe transcribe.py --input C:\\path\\to\\video.mp4
    venv\\Scripts\\python.exe transcribe.py --input https://www.youtube.com/watch?v=...

As a library:
    from transcribe import transcribe
    result = transcribe(resolved_local_path)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

from ingest import resolve_input

MODEL = "large-v3"
OUTPUT = "transcript.json"


def extract_audio(video_path: str, out_wav: str,
                  start: float = None, end: float = None) -> str:
    """Extract mono 16 kHz PCM WAV from video_path using ffmpeg.

    16 kHz mono is what Whisper expects internally, so downmixing here keeps the
    decode cheap and avoids surprises from odd source sample rates/channels.

    When start/end (seconds) are given, only that window is extracted: -ss before
    -i is a fast input seek, and -t caps the duration. The WAV then begins at 0,
    so the caller offsets word timestamps by +start to restore absolute time.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH. On Windows: winget install Gyan.FFmpeg")

    cmd = [ffmpeg, "-y"]
    if start is not None and start > 0:
        cmd += ["-ss", f"{start}"]
    cmd += ["-i", video_path, "-vn", "-ac", "1", "-ar", "16000"]
    if end is not None:
        cmd += ["-t", f"{max(0.0, end - (start or 0.0))}"]
    cmd += ["-f", "wav", out_wav]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(out_wav):
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise RuntimeError("ffmpeg audio extraction failed:\n" + "\n".join(tail))
    return out_wav


def transcribe(input_path: str, model_name: str = MODEL,
               device: str = "cuda", compute_type: str = "int8",
               start: float = None, end: float = None) -> dict:
    """Resolve input, extract audio, and transcribe with word timestamps.

    When start/end (seconds) are given, only that window of the source is
    transcribed (so a long video needn't be processed in full), and all word
    timestamps are offset by +start so they stay in ABSOLUTE source time -- the
    scorer, cutter and captions all keep working against the full video.

    Returns the dict later written to transcript.json.
    """
    # Import torch FIRST: on Windows this registers venv\...\torch\lib on the
    # DLL search path, which is where the cu124 wheel ships cublas64_12.dll and
    # the cudnn DLLs that ctranslate2 (faster-whisper's GPU backend) loads.
    # Without this, standalone runs fail with "cublas64_12.dll ... cannot be loaded".
    import torch  # noqa: F401
    from faster_whisper import WhisperModel

    local_path = resolve_input(input_path)
    print(f"Input resolved -> {local_path}", flush=True)

    tmp_wav = os.path.join(tempfile.gettempdir(), "clipper_transcribe_audio.wav")
    try:
        if start is not None or end is not None:
            span = f"{_fmt_hms(start or 0.0)}–{_fmt_hms(end) if end is not None else 'end'}"
            print(f"Extracting audio (16 kHz mono WAV, range {span}) ...", flush=True)
        else:
            print("Extracting audio (16 kHz mono WAV) ...", flush=True)
        extract_audio(local_path, tmp_wav, start=start, end=end)

        print(f"Loading faster-whisper {model_name} on {device} ({compute_type}) ...", flush=True)
        wmodel = WhisperModel(model_name, device=device, compute_type=compute_type)

        print("Transcribing (word timestamps) ...", flush=True)
        # vad_filter drops silent regions before decoding. Besides being faster,
        # it prevents large-v3's habit of hallucinating boilerplate (e.g.
        # "Subtitles by the Amara.org community") over long silent/music tails.
        segments, info = wmodel.transcribe(tmp_wav, word_timestamps=True, vad_filter=True)

        # Offset word timestamps back into absolute source time when we only
        # transcribed a window (the extracted WAV started at `start`).
        offset = float(start) if start else 0.0
        words = []
        text_parts = []
        for seg in segments:
            text_parts.append(seg.text)
            for w in (seg.words or []):
                words.append({
                    "word": w.word,
                    "start": round(w.start + offset, 3),
                    "end": round(w.end + offset, 3),
                })

        result = {
            "input": local_path,
            "language": info.language,
            "duration": round(info.duration, 3),  # length of audio transcribed
            "word_count": len(words),
            "text": "".join(text_parts).strip(),
            "words": words,
        }
        if start is not None or end is not None:
            result["range"] = [offset, round(offset + info.duration, 3)]
        return result
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def _fmt_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def parse_hms(value) -> float:
    """Parse a time to seconds: plain seconds, or MM:SS / HH:MM:SS.

    Usable as an argparse `type=`. Raises ValueError on garbage so argparse
    reports it cleanly.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        raise ValueError("empty time")
    if ":" in s:
        total = 0.0
        for part in s.split(":"):
            total = total * 60 + float(part or 0)
        return total
    return float(s)


def main():
    ap = argparse.ArgumentParser(description="Transcribe a video/URL to transcript.json with word timestamps")
    ap.add_argument("--input", required=True, help="local video path or a video URL (e.g. YouTube)")
    ap.add_argument("--output", default=OUTPUT, help="where to write the transcript JSON")
    ap.add_argument("--model", default=MODEL, help="faster-whisper model name")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--compute-type", default="int8", help="e.g. int8, float16")
    ap.add_argument("--start", type=parse_hms, default=None,
                    help="only transcribe from this time (seconds or MM:SS / HH:MM:SS)")
    ap.add_argument("--end", type=parse_hms, default=None,
                    help="only transcribe up to this time (seconds or MM:SS / HH:MM:SS)")
    args = ap.parse_args()

    if args.start is not None and args.end is not None and args.end <= args.start:
        print("ERROR: --end must be greater than --start.", file=sys.stderr)
        sys.exit(2)

    try:
        result = transcribe(args.input, args.model, args.device, args.compute_type,
                            start=args.start, end=args.end)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {args.output}: "
        f"duration {_fmt_hms(result['duration'])} ({result['duration']:.1f}s), "
        f"{result['word_count']} words, lang={result['language']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
