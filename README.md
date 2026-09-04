# Local Clipper

A local, single-user tool that turns a long video into short, captioned,
vertical (9:16) clips. Perception (transcription, cropping, encoding) runs on
the GPU; "is this a good moment?" judgment runs on the Claude API. See
[clipper-build-order.md](clipper-build-order.md) for the full phase plan and
[CLAUDE.md](CLAUDE.md) for machine/stack constraints.

## Setup

```bash
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
venv\Scripts\python.exe -m pip install faster-whisper ffmpeg-python anthropic python-dotenv librosa mediapipe
```

Also required:
- **ffmpeg** on PATH with NVENC (`h264_nvenc`). On Windows: `winget install Gyan.FFmpeg`.
- A **`.env`** file in the repo root (copy from `.env.example`):
  ```
  ANTHROPIC_API_KEY=sk-ant-...
  ```

## Manual tests

### Phase 1 — environment smoke test
Verifies torch+GPU, faster-whisper on GPU, ffmpeg+NVENC, and a live Claude API call.

```bash
venv\Scripts\python.exe scripts\smoke_test.py --input path\to\short_sample.mp4
# or a URL:
venv\Scripts\python.exe scripts\smoke_test.py --input https://www.youtube.com/watch?v=...
```

`--input` accepts a **local video/audio path or a video URL** (YouTube, etc.);
URLs are downloaded to `/media` first via yt-dlp. It's optional — without it the
whisper check transcribes a generated tone (just proving the model loads on the
GPU). Expect **all four checks to report PASS**. First run downloads the large-v3
Whisper weights (~3 GB), so give it a minute. Only clip content you have the
rights to.

### Resolving input on its own
```bash
venv\Scripts\python.exe ingest.py --input https://www.youtube.com/watch?v=...
```
Downloads (if a URL) and prints the local file path. Used by every pipeline stage.
