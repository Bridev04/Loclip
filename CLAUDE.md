# CLAUDE.md — Local Clipper

## What this is
A LOCAL, single-user tool. It is NOT a hosted product, has no web server,
no accounts, no auto-posting. Input: a long video file. Output: several
short captioned vertical (9:16) clips saved to /output that I upload to
social media manually.

## My machine
- OS: Windows 11
- GPU: NVIDIA RTX 5060 Laptop, 8GB GDDR7 VRAM, Blackwell architecture (sm_120)
- CUDA 12.8+ and recent PyTorch required (Blackwell needs current builds;
  old pinned versions cause "no kernel image available" errors)

## Division of labor (important)
- GPU does PERCEPTION: transcription, face detection, video encoding.
- Claude API does JUDGMENT: scoring which segments are compelling.
- Do NOT load a local LLM — the 8GB VRAM is reserved for Whisper + video.

## Stack
- Python 3.11+, single project venv
- faster-whisper (large-v3, int8 quantization) for transcription w/ word timestamps
- ffmpeg for all cutting/cropping/encoding; use NVENC (-c:v h264_nvenc)
- anthropic Python SDK for moment scoring
- mediapipe or opencv for face tracking (later phase)
- librosa for audio-energy signal (optional phase)

## Claude API models
- Default scorer: claude-haiku-4-5  (cheap, fast; check exact model ID in the Anthropic console)
- Final ranking / hard cases: claude-sonnet-5
- API key comes from an environment variable ANTHROPIC_API_KEY loaded via python-dotenv from a .env file. NEVER hardcode the key.

## Build principles
- Build a thin end-to-end slice FIRST (dumb cutter), then improve the moment
  detection, then add reframe and caption polish.
- Keep the moment-scoring PROMPT in its own file (prompts/score.txt) so I can
  edit it without touching code.
- Every module runnable standalone from the CLI with a --input flag.
- After each phase, add a short manual test step to README and commit.
