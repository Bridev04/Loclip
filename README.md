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

### Phase 2 — transcription
Resolves `--input` (local path or URL), extracts a 16 kHz mono WAV with ffmpeg,
runs faster-whisper large-v3 (int8) on the GPU with word timestamps, and writes
`transcript.json` (full text plus a `words` list of `{word, start, end}`).

```bash
venv\Scripts\python.exe transcribe.py --input path\to\video.mp4
# or a URL:
venv\Scripts\python.exe transcribe.py --input https://www.youtube.com/watch?v=...
```

On success it prints the total duration and word count, and `transcript.json`
appears in the repo root. Spot-check that `duration` and `word_count` look
sane and that a few `words` timestamps line up with what's said. Use
`--output` to write elsewhere. Only transcribe content you have the rights to.

### Phase 3 — cut + 9:16 reframe
Cuts a `[--start, --end]` segment (seconds), reframes it to a vertical
1080x1920 (9:16) frame, and encodes with the GPU (`h264_nvenc`) to `/output`.

```bash
venv\Scripts\python.exe cut.py --input path\to\video.mp4 --start 12 --end 45
# fit modes: cover (default) fills+crops; contain fits+letterboxes:
venv\Scripts\python.exe cut.py --input path\to\video.mp4 --start 12 --end 45 --fit contain
```

The clip lands in `/output` as `<stem>_<start>-<end>_<fit>.mp4`. Spot-check
with `ffprobe` that width=1080, height=1920, and duration ≈ `end - start`.
`cover` is the usual vertical-clip look (a landscape source loses its side
edges); a later phase replaces the static center-crop with face-tracking.

### Full pipeline (dumb end-to-end slice)
Transcribes the input, then blindly cuts the first 45 seconds into a 9:16 clip
— the thin end-to-end slice. No moment scoring yet; smarter moment detection
replaces the "first 45s" rule in a later phase.

```bash
venv\Scripts\python.exe main.py --input path\to\video.mp4 --dumb
```

Writes `transcript.json` and one clip to `/output`. If the source is shorter
than 45s, the clip is truncated to the source length. Pass `--fit contain` to
letterbox instead of crop.

### Phase 4 — moment detection (transcribe → score → cut top N)
The core step. `segments.py` slices `transcript.json` into 20–90s candidate
windows aligned to sentence/pause boundaries; `score.py` sends them to Claude
(`claude-haiku-4-5` by default) with the prompt in `prompts/score.txt` and
returns them ranked; `main.py --n N` cuts the top N into 9:16 clips.

```bash
venv\Scripts\python.exe main.py --input path\to\video.mp4 --n 5
```

Clips land in `/output` named `<stem>_rank01_score87_<start>-<end>.mp4` (rank +
score in the filename so the best sort first). The top N are chosen with greedy
overlap suppression, so they're N *distinct* moments rather than several cuts of
the same hot moment. Inspect the top picks; **tune by editing `prompts/score.txt`,
not the code**, then re-run. Useful flags:
- `--model claude-sonnet-5` — score with Sonnet instead of Haiku for hard cases.
- `--min` / `--max` — candidate window length bounds in seconds (default 20/90).
- `--overlap 0.2` — max overlap allowed between chosen clips (default 0.5; `1.0`
  disables dedup for pure top-N).
- `--transcript transcript.json` — reuse an existing transcript instead of
  re-transcribing, so you can iterate on scoring/selection fast (no `--input`
  needed). The video path is read from the transcript.
- `--fit contain` — letterbox instead of crop.

Score or segment on their own (both read `transcript.json`, no re-transcribe):
```bash
venv\Scripts\python.exe segments.py                 # print candidate windows
venv\Scripts\python.exe score.py                    # print the ranked segments
```

> Needs `ANTHROPIC_API_KEY` in `.env`. A run costs a fraction of a cent (Haiku).
> If the model returns malformed JSON, the scorer reports it instead of crashing.

### Resolving input on its own
```bash
venv\Scripts\python.exe ingest.py --input https://www.youtube.com/watch?v=...
```
Downloads (if a URL) and prints the local file path. Used by every pipeline stage.
