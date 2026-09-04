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
- The **YuNet face-detection model** for Phase 6 reframe (one-time, ~230 KB, into
  `models/`, which is gitignored):
  ```bash
  curl -L -o models/face_detection_yunet_2023mar.onnx https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
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

Transcription uses faster-whisper's **batched GPU inference** by default (same
`large-v3` model/accuracy, ~2–3× faster than one-window-at-a-time). If you hit
GPU out-of-memory, lower `--batch-size` (default 8; `1` = sequential). Batch
size is also a `main.py` flag.

To transcribe just a slice of a long video, add `--start` / `--end` (`SS`,
`MM:SS`, or `HH:MM:SS`); word timestamps are still written in absolute source
time, and a `range` field records the window:

```bash
venv\Scripts\python.exe transcribe.py --input path\to\video.mp4 --start 5:00 --end 12:30
```

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
- `--fit contain` — letterbox instead of crop (ignored when reframe is on).
- `--no-reframe` — static center crop instead of speaker face-tracking (tracking
  is on by default; see Phase 6).
- `--no-captions` — skip the caption burn-in (captions are on by default; see
  Phase 7).

Score or segment on their own (both read `transcript.json`, no re-transcribe):
```bash
venv\Scripts\python.exe segments.py                 # print candidate windows
venv\Scripts\python.exe score.py                    # print the ranked segments
```

> Needs `ANTHROPIC_API_KEY` in `.env`. A run costs a fraction of a cent (Haiku).
> If the model returns malformed JSON, the scorer reports it instead of crashing.

### Phase 6 — face-tracking auto-reframe
`reframe.py` keeps the speaker in frame when cropping 16:9 → 9:16. It samples
face detections across the clip (OpenCV's **YuNet**, run locally), picks the
main speaker (largest face, with hysteresis so it doesn't flip between people),
and builds a **smoothed** crop-center path — moving-average + a pan-speed limiter
+ path simplification — so the vertical frame glides to follow them with no
jitter. `cut.py` applies it as a dynamic ffmpeg `crop` before the scale (and
before captions). If no face (or too few) is detected it **falls back to the
static center-crop**. Reframe is **on by default** in the `--n` pipeline:

```bash
venv\Scripts\python.exe main.py --transcript transcript.json --n 5
# static center crop instead:
venv\Scripts\python.exe main.py --transcript transcript.json --n 5 --no-reframe
```

Verify tracking on one window (encodes a real clip; scrub it — the crop should
follow the speaker smoothly and never jump):

```bash
venv\Scripts\python.exe reframe.py --input media\2PxLYWjgLys.mp4 --start 53.24 --end 76.4 --output output\reframe_test.mp4
```

Inspect the detection track without encoding (how many samples saw a face, the
crop-x range, or whether it would fall back to a static crop):

```bash
venv\Scripts\python.exe reframe.py --input media\2PxLYWjgLys.mp4 --start 53.24 --end 76.4 --dump
```

Tuning flags (standalone `reframe.py`): `--sample-fps` (detection rate, default
6), `--smooth-seconds` (path smoothing window, default 0.8), `--max-pan-px-s`
(max crop pan speed, default 320). Needs the YuNet model in `models/` (see
Setup). Requires a landscape (wider-than-9:16) source; taller sources skip
tracking and use the static crop. **Known limitation:** on hard scene cuts the
crop slides to the new speaker over ~1s rather than jumping instantly — a
deliberate trade for smoothness; scene-cut-aware snapping is a future refinement.

### Facecam split layout (streamer: facecam on top, gameplay on bottom)
For streamer/gameplay clips, `--split` builds a vertical frame that stacks the
**facecam on top** and the **gameplay on the bottom**, instead of a single
cropped frame. The facecam region is found automatically (it's wherever faces
consistently appear — the detector reuses Phase 6's YuNet, picks the biggest
face cluster, and fits a 16:9 box around it), or you set it explicitly.

```bash
# whole pipeline, split layout, facecam auto-detected:
venv\Scripts\python.exe main.py --input path\to\stream.mp4 --n 5 --split

# pin the facecam (a multi-cam stream, or to keep it consistent across clips):
venv\Scripts\python.exe main.py --input path\to\stream.mp4 --n 5 --split --facecam bottom-left
venv\Scripts\python.exe main.py --input path\to\stream.mp4 --n 5 --split --facecam 0,380,570,320
```

- `--facecam` accepts a **corner** (`top-left` / `tr` / `bottom-left` / `br` / …),
  **pixels** `x,y,w,h`, or **fractions** of the frame `x,y,w,h` when all ≤ 1.
  Omit it to auto-detect. On a multi-cam stream auto-detect grabs the largest
  (closest) face, which can differ clip to clip — pin `--facecam` for a
  consistent cam.
- `--facecam-frac` sets the top share (default `0.4` = facecam 40% / gameplay 60%).
- Split overrides `--reframe`/`--fit`. If no facecam is found (or detection
  fails), it falls back to the static center crop. Captions still burn in over
  the finished split frame.

Eyeball one window straight from `cut.py` (no scoring, encodes a real clip):

```bash
venv\Scripts\python.exe cut.py --input path\to\stream.mp4 --start 1450 --end 1460 --layout split --output output\split_test.mp4
# force a specific cam:
venv\Scripts\python.exe cut.py --input path\to\stream.mp4 --start 1450 --end 1460 --layout split --facecam bottom-left --output output\split_test.mp4
```

In the web UI, tick **facecam split** and (optionally) type a corner or pixels
in the field beside it.

### Phase 7 — caption burn-in (TikTok/Reels style)
`captions.py` turns the word timestamps in `transcript.json` into an ASS
subtitle file — one or two words on screen at a time, big and centered, with
the active (currently-spoken) word highlighted karaoke-style — and `cut.py`
burns it into the clip during the final NVENC encode (the subtitle filter runs
*after* the 9:16 crop, so captions land inside the frame). Captions are **on by
default** in the `--n` pipeline, so the Phase 4 command already produces
captioned clips:

```bash
venv\Scripts\python.exe main.py --transcript transcript.json --n 5
# toggle captions off:
venv\Scripts\python.exe main.py --transcript transcript.json --n 5 --no-captions
```

To iterate on the *look* without re-running the whole pipeline, caption one
window straight from `cut.py`:

```bash
venv\Scripts\python.exe cut.py --input media\2PxLYWjgLys.mp4 --start 1278.6 --end 1302.36 --captions --transcript transcript.json --output output\captiontest.mp4
```

Open the clip and scrub: text should be centered in the lower third, uppercase,
white with a black outline, showing ~2 words at a time, and the spoken word
should light up yellow in time with the audio.

**Restyle without touching code** by editing `caption_style.json`:
- `font`, `font_size`, `bold`, `uppercase`
- `words_per_group` — 1 or 2 (words visible at once)
- `text_color`, `highlight_color`, `outline_color` — `#RRGGBB`
- `outline_width`, `shadow`, `highlight_scale` (active-word "pop", percent)
- `alignment` (ASS numpad: `2` = bottom-center, `5` = middle), `margin_v` (px
  lifted off the aligned edge), `margin_h`
- `max_gap` — start a fresh caption group when the pause before a word exceeds
  this (seconds), so captions reset on real pauses.

Inspect the generated subtitles directly (no encode) with:
```bash
venv\Scripts\python.exe captions.py --transcript transcript.json --start 1278.6 --end 1302.36 --out sample.ass
```

### Phase 8 — orchestration & polish (batch + suggestions)
The full pipeline in one command, over a single video **or a whole folder**, with
optional AI-generated posting metadata per clip.

```bash
# one video (transcribe -> score -> reframe -> caption -> export):
venv\Scripts\python.exe main.py --input path\to\video.mp4 --n 5

# a FOLDER of videos (each is transcribed and clipped; failures are skipped,
# not fatal, and reported at the end):
venv\Scripts\python.exe main.py --input path\to\videos_folder --n 5

# add per-clip title/description/hashtags saved next to each clip:
venv\Scripts\python.exe main.py --input path\to\video.mp4 --n 5 --suggest
```

`--suggest` asks `claude-haiku-4-5` (prompt in `prompts/suggest.txt`) for a
title, description, and hashtags from each clip's transcript and writes them to a
`.txt` beside the clip (e.g. `..._rank01_....mp4` → `..._rank01_....txt`). It's
best-effort: a suggestion failure warns and moves on, it never loses you a clip.
Test it on one window without cutting anything:

```bash
venv\Scripts\python.exe suggest.py --transcript transcript.json --start 53.24 --end 76.4
```

Batch notes: folder inputs are matched by extension (`.mp4 .mov .mkv .webm .avi
.m4v .flv .wmv`); each video's clips are named by its own stem so they don't
collide in `/output`. `--transcript` reuse is ignored for a folder (every video
is transcribed).

#### Every `main.py` flag
| Flag | Default | Meaning |
|---|---|---|
| `--input` | — | video file, video URL, **or a folder** of videos |
| `--n N` | — | full pipeline: cut the top N scored clips |
| `--dumb` | — | plumbing slice: transcribe then cut the first 45s (no scoring) |
| `--suggest` | off | write a title/description/hashtags `.txt` per clip (Claude) |
| `--reframe` / `--no-reframe` | on | face-track the speaker vs. static center crop |
| `--split` | off | streamer layout: facecam on top, gameplay on bottom (overrides reframe) |
| `--facecam` | auto | facecam region for `--split`: corner, `x,y,w,h` pixels, or fractions |
| `--facecam-frac` | 0.4 | top share of the frame for the facecam with `--split` |
| `--batch-size` | 8 | transcription batch size (lower on GPU OOM; `1` = sequential) |
| `--captions` / `--no-captions` | on | burn TikTok-style captions in vs. skip |
| `--no-loudnorm` | on | audio loudness normalization to ~-14 LUFS (on by default) |
| `--fit cover\|contain` | cover | static-crop mode (ignored when reframe is on) |
| `--model` | `claude-haiku-4-5` | first-pass scorer that shortlists every candidate |
| `--rank-model` | `claude-sonnet-5` | re-ranks the shortlist (two-stage); same as `--model` disables |
| `--refine-top` | 10 | how many top candidates the rank-model re-scores (`0` = single-stage) |
| `--min` / `--max` | 20 / 90 | candidate window length bounds (s) |
| `--overlap` | 0.5 | max overlap between chosen clips (`1.0` disables dedup) |
| `--energy-weight` | 0.3 | audio-energy share when re-ranking (`0` = pure LLM order); see Phase 5 |
| `--start` / `--end` | — | only clip this time window of the source (`SS`, `MM:SS`, or `HH:MM:SS`); transcribes just the window so a long video needn't be done in full |
| `--transcript` | — | reuse an existing `transcript.json` (single input only) |

### Phase 5 — audio-energy re-ranking
`energy.py` adds a **local, CPU** perception signal that Claude's text-only
scoring can't see: how *loud* and *punchy* each candidate actually sounds.
Using librosa it derives, per candidate, an energy score from RMS loudness
(mean + a high percentile) and onset/peak strength — so laughs, applause, and
emphatic delivery register — then normalizes those to 0–1 **across this video's
own candidates**. It blends that with the LLM score:

```
blended = w * energy + (1 - w) * (llm_score / 100)      # w = --energy-weight, default 0.3
```

The blend re-ranks candidates **before** the overlap suppression, so a genuinely
exciting beat can surface even if its transcript text read as unremarkable. This
is purely additive — it does **not** touch `prompts/score.txt` or the scoring
call. Energy runs on the CPU, leaving the 8GB VRAM free (CLAUDE.md rule). It's
on by default in the `--n` pipeline:

```bash
venv\Scripts\python.exe main.py --transcript transcript.json --n 5
# lean harder on energy, or turn it off for the old pure-LLM order:
venv\Scripts\python.exe main.py --transcript transcript.json --n 5 --energy-weight 0.5
venv\Scripts\python.exe main.py --transcript transcript.json --n 5 --energy-weight 0
```

The top-picks printout now shows `energy` and `blended` beside each `score`, so
you can see what the audio signal moved. The blend is best-effort: if the audio
can't be decoded, it warns and falls back to the pure LLM order rather than
losing you clips.

Inspect the per-segment energy on its own (no API call, no encode — reads
`transcript.json` for the candidate windows and the video path):

```bash
venv\Scripts\python.exe energy.py
# or point at a specific video:
venv\Scripts\python.exe energy.py --input media\j2rszuZ-9PY.mp4
```

It prints every candidate sorted by energy (highest first). Spot-check that
laugh/applause/emphatic moments score high and flat, quiet passages score low.

### Local web UI — clip and choose
`serve.py` is a small, **local** web app for driving the whole thing from a
browser: paste a **video URL or a local file path**, click **Clip**, watch the
pipeline run (live log), then **pick from the clips it produced**. Below that is
a gallery of everything already in `/output`. Stdlib only — no extra dependency.

```bash
venv\Scripts\python.exe serve.py
# opens http://127.0.0.1:8000/ in your browser (Ctrl+C to stop)
```

- **Clip** runs the same `main.py` pipeline (transcribe → score → energy blend →
  reframe → caption) as a subprocess and streams its output to the page. A URL
  is downloaded first (yt-dlp). Set how many clips with the **clips** field and
  tick **caption suggestions** for the `--suggest` metadata.
- **only clip from / to** (optional) restricts the run to a time window of the
  source — enter `5:00` / `12:30` (or plain seconds). Only that window is
  transcribed, so a long video (a 2-hour stream, say) is fast instead of having
  to transcribe the whole thing. Leave both blank for the entire video. The
  clips are still cut at their true positions in the full source.
- **facecam split** (optional) produces the streamer layout — the facecam on
  top, the gameplay on the bottom — instead of a single cropped frame. The
  facecam is auto-detected (the biggest/most-prominent face region); the field
  beside the checkbox overrides it with a corner (`bottom-left`, `top-right`, …)
  or exact pixels/fractions (`x,y,w,h`). See "Facecam split layout" below.
- When it finishes, the new clips appear under **Pick your clips** with a
  checkbox on each — tick the ones you want and **Download selected**, or grab
  them individually. Reload to fold them into the gallery below.
- Each clip (results and gallery) gets a 9:16 player with its rank/score/time-
  range parsed from the filename and the `--suggest` caption when the sibling
  `.txt` exists.

It's **local and single-user**, consistent with CLAUDE.md: binds `127.0.0.1`
(this machine only), no accounts, no uploads, and it never posts anywhere —
"choose" means review / select / download, then you upload manually. Only clip
content you have the rights to. HTTP Range is supported so the players scrub
properly; on Windows it pulls the real PATH from the registry so the spawned
pipeline finds ffmpeg / yt-dlp even if you launched the server from a shell with
a stale PATH.

Useful flags:
- `--output <dir>` — folder of clips to serve (default `output`).
- `--port <n>` — localhost port (default `8000`).
- `--no-open` — don't auto-open a browser tab.

### Resolving input on its own
```bash
venv\Scripts\python.exe ingest.py --input https://www.youtube.com/watch?v=...
```
Downloads (if a URL) and prints the local file path. Used by every pipeline stage.
