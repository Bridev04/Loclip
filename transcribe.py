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


def extract_audio(video_path: str, out_wav: str) -> str:
    """Extract mono 16 kHz PCM WAV from video_path using ffmpeg.

    16 kHz mono is what Whisper expects internally, so downmixing here keeps the
    decode cheap and avoids surprises from odd source sample rates/channels.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH. On Windows: winget install Gyan.FFmpeg")

    proc = subprocess.run(
        [ffmpeg, "-y", "-i", video_path,
         "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", out_wav],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not os.path.exists(out_wav):
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise RuntimeError("ffmpeg audio extraction failed:\n" + "\n".join(tail))
    return out_wav


def transcribe(input_path: str, model_name: str = MODEL,
               device: str = "cuda", compute_type: str = "int8") -> dict:
    """Resolve input, extract audio, and transcribe with word timestamps.

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
        print("Extracting audio (16 kHz mono WAV) ...", flush=True)
        extract_audio(local_path, tmp_wav)

        print(f"Loading faster-whisper {model_name} on {device} ({compute_type}) ...", flush=True)
        wmodel = WhisperModel(model_name, device=device, compute_type=compute_type)

        print("Transcribing (word timestamps) ...", flush=True)
        # vad_filter drops silent regions before decoding. Besides being faster,
        # it prevents large-v3's habit of hallucinating boilerplate (e.g.
        # "Subtitles by the Amara.org community") over long silent/music tails.
        segments, info = wmodel.transcribe(tmp_wav, word_timestamps=True, vad_filter=True)

        words = []
        text_parts = []
        for seg in segments:
            text_parts.append(seg.text)
            for w in (seg.words or []):
                words.append({
                    "word": w.word,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                })

        return {
            "input": local_path,
            "language": info.language,
            "duration": round(info.duration, 3),
            "word_count": len(words),
            "text": "".join(text_parts).strip(),
            "words": words,
        }
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def _fmt_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser(description="Transcribe a video/URL to transcript.json with word timestamps")
    ap.add_argument("--input", required=True, help="local video path or a video URL (e.g. YouTube)")
    ap.add_argument("--output", default=OUTPUT, help="where to write the transcript JSON")
    ap.add_argument("--model", default=MODEL, help="faster-whisper model name")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--compute-type", default="int8", help="e.g. int8, float16")
    args = ap.parse_args()

    try:
        result = transcribe(args.input, args.model, args.device, args.compute_type)
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
