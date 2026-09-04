"""
Phase 1 smoke test for Local Clipper.

Verifies the four things the whole pipeline depends on:
  1. torch sees the GPU (+ CUDA version)
  2. faster-whisper large-v3 loads on GPU (int8) and transcribes a short sample
  3. system ffmpeg is present and exposes the h264_nvenc encoder
  4. a 1-token Claude API call succeeds using ANTHROPIC_API_KEY from .env

Run:
    venv\\Scripts\\python.exe scripts\\smoke_test.py --input path\\to\\sample.mp4

--input is optional. If omitted, check #2 generates a 3s tone with ffmpeg and
transcribes that instead (it will produce little/no text, which is fine — the
point is that the model loads and runs on the GPU).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

# make the repo root importable so we can use ingest.py from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- pretty result tracking -------------------------------------------------
RESULTS = {}


def record(name, ok, detail=""):
    RESULTS[name] = (ok, detail)
    mark = "PASS" if ok else "FAIL"
    line = f"[{mark}] {name}"
    if detail:
        line += f" -- {detail}"
    print(line, flush=True)


def section(title):
    print("\n" + "=" * 60, flush=True)
    print(title, flush=True)
    print("=" * 60, flush=True)


# --- check 1: torch + GPU ---------------------------------------------------
def check_torch():
    section("1. torch / GPU")
    try:
        import torch
    except Exception as e:
        record("torch import", False, f"{type(e).__name__}: {e}")
        return
    try:
        avail = torch.cuda.is_available()
        cuda_ver = torch.version.cuda
        if avail:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            detail = (
                f"torch {torch.__version__}, CUDA {cuda_ver}, GPU: {name}, "
                f"sm_{cap[0]}{cap[1]}, {total_gb:.1f} GB"
            )
            record("torch sees GPU", True, detail)
        else:
            record(
                "torch sees GPU",
                False,
                f"torch {torch.__version__} built for CUDA {cuda_ver} "
                f"but torch.cuda.is_available() is False",
            )
    except Exception as e:
        record("torch sees GPU", False, f"{type(e).__name__}: {e}")


# --- check 2: faster-whisper on GPU -----------------------------------------
def _make_sample_audio():
    """Return a path to a short audio file, generating a tone if needed."""
    tmp = os.path.join(tempfile.gettempdir(), "clipper_smoke_sample.wav")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-ar", "16000", tmp],
        capture_output=True,
    )
    return tmp if os.path.exists(tmp) else None


def check_whisper(input_path):
    section("2. faster-whisper large-v3 (GPU, int8)")
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        record("faster-whisper import", False, f"{type(e).__name__}: {e}")
        return

    sample = input_path
    generated = False
    if not sample:
        sample = _make_sample_audio()
        generated = True
        if not sample:
            record("whisper transcribe", False, "no --input and could not generate a tone (ffmpeg missing)")
            return
    else:
        # --input may be a local path OR a video URL (YouTube, etc.)
        try:
            from ingest import resolve_input
            sample = resolve_input(sample)
            print(f"    input resolved -> {sample}", flush=True)
        except Exception as e:
            record("whisper transcribe", False, f"could not resolve --input: {type(e).__name__}: {e}")
            return

    try:
        model = WhisperModel("large-v3", device="cuda", compute_type="int8")
    except Exception as e:
        record("whisper load on GPU", False, f"{type(e).__name__}: {e}")
        return

    try:
        segments, info = model.transcribe(sample, word_timestamps=True)
        words = 0
        preview = []
        for seg in segments:
            for w in (seg.words or []):
                words += 1
                if len(preview) < 12:
                    preview.append(w.word.strip())
        note = "generated tone (little/no speech expected)" if generated else "your sample"
        detail = f"ran on GPU, {words} words, lang={info.language}; {note}"
        if preview:
            detail += f'; sample: "{" ".join(preview)}"'
        record("whisper transcribe", True, detail)
    except Exception as e:
        record("whisper transcribe", False, f"{type(e).__name__}: {e}")


# --- check 3: ffmpeg + h264_nvenc -------------------------------------------
def check_ffmpeg():
    section("3. ffmpeg + h264_nvenc")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        record("ffmpeg present", False, "ffmpeg not found on PATH")
        return
    try:
        ver = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True)
        first = ver.stdout.splitlines()[0] if ver.stdout else "unknown"
        record("ffmpeg present", True, first)
    except Exception as e:
        record("ffmpeg present", False, f"{type(e).__name__}: {e}")
        return
    try:
        enc = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                             capture_output=True, text=True)
        if "h264_nvenc" in enc.stdout:
            record("h264_nvenc available", True, "NVENC H.264 encoder present")
        else:
            record("h264_nvenc available", False, "h264_nvenc not listed by this ffmpeg build")
    except Exception as e:
        record("h264_nvenc available", False, f"{type(e).__name__}: {e}")


# --- check 4: Claude API ----------------------------------------------------
def check_anthropic():
    section("4. Claude API (1-token call)")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception as e:
        record("anthropic call", False, f"python-dotenv error: {e}")
        return
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        record("anthropic call", False, "ANTHROPIC_API_KEY not set (create a .env file)")
        return
    try:
        import anthropic
    except Exception as e:
        record("anthropic call", False, f"import error: {type(e).__name__}: {e}")
        return
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        record("anthropic call", True, f"model responded (stop_reason={resp.stop_reason})")
    except Exception as e:
        record("anthropic call", False, f"{type(e).__name__}: {e}")


# --- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Local Clipper Phase 1 smoke test")
    ap.add_argument("--input", help="local video/audio path OR a video URL (e.g. YouTube) for the whisper check")
    args = ap.parse_args()

    check_torch()
    check_whisper(args.input)
    check_ffmpeg()
    check_anthropic()

    section("SUMMARY")
    all_ok = True
    for name, (ok, _) in RESULTS.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}", flush=True)
        all_ok = all_ok and ok
    print("", flush=True)
    if all_ok:
        print("All checks passed. Phase 1 complete.", flush=True)
        sys.exit(0)
    else:
        print("Some checks failed. See details above.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
