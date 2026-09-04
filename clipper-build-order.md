# Local Clipper — Build Order & Claude Code Playbook

A local, personal-use tool that takes a long video, finds the best moments, and outputs captioned vertical clips you upload by hand. Runs on an RTX 5060 laptop (8GB VRAM). Perception runs on your GPU; "is this a viral moment" judgment runs on the Claude API.

---

## How to use this doc

Work **one phase per Claude Code chat**. Each phase below gives you:
- **Goal** — what exists at the end
- **Paste this** — the kickoff message for that chat
- **Done when** — the test that tells you the phase is finished

Before anything else, do Step 0 (the project brief). Then go phase by phase, committing to git after each one so you can always roll back.

---

## Step 0 — The project brief (do this ONCE, first)

Claude Code automatically reads a file called `CLAUDE.md` in your repo root at the start of every session. This is how your constraints survive across chats. Create the repo folder, `cd` into it, start Claude Code, and paste this:

> Create a new file `CLAUDE.md` in the repo root with exactly the content I paste below. Then initialize a git repo, create a `.gitignore` for Python (venv, __pycache__, .env, and a `/media` and `/output` folder for large files), and make the first commit.
>
> ```markdown
> # CLAUDE.md — Local Clipper
>
> ## What this is
> A LOCAL, single-user tool. It is NOT a hosted product, has no web server,
> no accounts, no auto-posting. Input: a long video file. Output: several
> short captioned vertical (9:16) clips saved to /output that I upload to
> social media manually.
>
> ## My machine
> - OS: <FILL IN: Windows 11 / Linux / macOS>
> - GPU: NVIDIA RTX 5060 Laptop, 8GB GDDR7 VRAM, Blackwell architecture (sm_120)
> - CUDA 12.8+ and recent PyTorch required (Blackwell needs current builds;
>   old pinned versions cause "no kernel image available" errors)
>
> ## Division of labor (important)
> - GPU does PERCEPTION: transcription, face detection, video encoding.
> - Claude API does JUDGMENT: scoring which segments are compelling.
> - Do NOT load a local LLM — the 8GB VRAM is reserved for Whisper + video.
>
> ## Stack
> - Python 3.11+, single project venv
> - faster-whisper (large-v3, int8 quantization) for transcription w/ word timestamps
> - ffmpeg for all cutting/cropping/encoding; use NVENC (-c:v h264_nvenc)
> - anthropic Python SDK for moment scoring
> - mediapipe or opencv for face tracking (later phase)
> - librosa for audio-energy signal (optional phase)
>
> ## Claude API models
> - Default scorer: claude-haiku-4-5  (cheap, fast; check exact model ID in the Anthropic console)
> - Final ranking / hard cases: claude-sonnet-5
> - API key comes from an environment variable ANTHROPIC_API_KEY loaded via python-dotenv from a .env file. NEVER hardcode the key.
>
> ## Build principles
> - Build a thin end-to-end slice FIRST (dumb cutter), then improve the moment
>   detection, then add reframe and caption polish.
> - Keep the moment-scoring PROMPT in its own file (prompts/score.txt) so I can
>   edit it without touching code.
> - Every module runnable standalone from the CLI with a --input flag.
> - After each phase, add a short manual test step to README and commit.
> ```

Fill in your OS on the line marked `<FILL IN>` before you send it.

---

## Phase 1 — Environment smoke test (nail this before writing features)

**Goal:** Proof the GPU, CUDA, faster-whisper, ffmpeg, and the Anthropic SDK all actually work on your machine. This is the phase most likely to hit the Blackwell CUDA snag, so isolate it.

**Paste this:**
> Set up the Python environment for this project. Create a venv, install torch (CUDA 12.8+ build for Blackwell/sm_120), faster-whisper, ffmpeg-python, anthropic, python-dotenv, librosa, and mediapipe. Then write a script `scripts/smoke_test.py` that: (1) prints whether torch sees the GPU and the CUDA version, (2) loads faster-whisper large-v3 on GPU with int8 and transcribes a 10-second sample, (3) runs `ffmpeg -version` and confirms h264_nvenc is available, (4) makes a 1-token test call to the Claude API using ANTHROPIC_API_KEY from .env. Report clearly which of the four checks pass or fail.

**Done when:** all four checks pass. If the torch/GPU check fails, that's the Blackwell version issue — have Claude Code walk you through installing the correct CUDA-enabled torch build before moving on.

---

## Phase 2 — Transcription module

**Goal:** `transcribe.py --input video.mp4` produces `transcript.json` with word-level timestamps.

**Paste this:**
> Read CLAUDE.md and scripts/smoke_test.py. Build `transcribe.py`: it takes --input <video>, extracts audio with ffmpeg, runs faster-whisper large-v3 (int8, GPU) with word_timestamps=True, and writes transcript.json containing the full text plus a list of words each with {word, start, end}. Print total duration and word count when done. Add a manual test step to the README.

**Done when:** you run it on a real video and the JSON has sensible word timings (spot-check a few against the audio).

---

## Phase 3 — Thin end-to-end slice (the "dumb cutter")

**Goal:** Prove the whole pipeline runs video → clip, with NO smart detection yet. This de-risks all the plumbing before you touch the hard part.

**Paste this:**
> Read CLAUDE.md, transcribe.py, and transcript.json. Build `cut.py` that takes --input <video> and --start/--end seconds, cuts that segment with ffmpeg, crops/pads it to 9:16 vertical (1080x1920), encodes with h264_nvenc, and saves to /output. For now, also add a `--dumb` mode to main.py that transcribes then just cuts the first 45 seconds as a clip, so I can run the full pipeline start to finish in one command.

**Done when:** one command turns a source video into a playable vertical clip in /output. Don't judge the *content* yet — just that the plumbing works.

---

## Phase 4 — Moment detection (the core value — spend the most time here)

**Goal:** Claude scores candidate segments and you cut the top N. This is where the product lives, so iterate.

**Paste this:**
> Read CLAUDE.md and transcript.json. Build the moment-detection step:
> 1. `segments.py` generates candidate segments from the transcript — sliding windows of 20–90 seconds aligned to sentence/pause boundaries so cuts land cleanly.
> 2. Create `prompts/score.txt` containing a scoring prompt (keep it in this file, not in code). It should ask Claude to rate each candidate on hook strength, emotional intensity, quotability, and whether it's a self-contained thought, and to return STRICT JSON: a list of {start, end, score, reason}.
> 3. `score.py` sends the candidates to claude-haiku-4-5, parses the JSON safely, and returns the ranked list.
> 4. Wire main.py so `python main.py --input video.mp4 --n 5` transcribes, scores, and cuts the top 5 clips.
> Handle the case where the model returns malformed JSON.

**Tips for iterating here (over several messages in the SAME chat):**
- Watch which moments it picks. If they're weak, **edit `prompts/score.txt`, not the code** — ask Claude Code to tweak the wording and re-run.
- If Haiku's picks feel shallow, say: *"add a --model flag; use claude-sonnet-5 for a final re-rank of the top 10 candidates only."* (Two-pass: Haiku shortlists cheaply, Sonnet judges the finalists.)
- Give it a real, opinionated example: *"here's a clip it picked that's boring, and one it missed that's great — adjust the prompt so it prefers the second kind."*

**Done when:** on your own footage, the top clips are ones you'd actually consider posting. This is the bar that matters.

---

## Phase 5 — Audio-energy signal (optional; add only if Phase 4 picks feel flat)

**Goal:** Blend loudness/laughter peaks with the LLM score so high-energy moments rank up.

**Paste this:**
> Read CLAUDE.md and score.py. Add `energy.py` using librosa to compute a per-segment audio-energy score (RMS + onset peaks), normalize it 0–1, and blend it with the Claude score using a configurable weight (default 0.3 energy / 0.7 LLM). Expose the weight as a CLI flag so I can tune it.

**Done when:** high-energy moments (laughs, emphatic delivery) reliably surface, and you can dial the weight.

---

## Phase 6 — Auto-reframe / face tracking

**Goal:** Keep the speaker centered when cropping 16:9 → 9:16, so clips don't look broken. This is the trickiest CV part — isolate it in its own chat.

**Paste this:**
> Read CLAUDE.md and cut.py. Add `reframe.py`: use mediapipe face detection to find the main speaker per frame, compute a smoothed crop-center path (avoid jittery jumps — smooth over time), and apply a dynamic 9:16 crop that follows the speaker in the ffmpeg cut. Fall back to a centered static crop if no face is detected. Add a --reframe on/off flag.

**Done when:** in a test clip where the speaker moves, the crop tracks them smoothly with no jitter, and it falls back gracefully when there's no face.

---

## Phase 7 — Caption burn-in

**Goal:** Word-synced captions burned into the clip (the modern karaoke-highlight look).

**Paste this:**
> Read CLAUDE.md and transcript.json. Add `captions.py`: generate an ASS subtitle file from the word-level timestamps with active-word highlighting (karaoke style), configurable font/size/position/colors, and burn it into the clip with ffmpeg during the final encode. Put the style settings in a config file so I can restyle without code changes.

**Done when:** captions are readable, correctly timed, and styled the way you want — and you can change the style from the config.

---

## Phase 8 — Orchestration & polish

**Goal:** One clean command, batch support, and optional caption/title/hashtag suggestions.

**Paste this:**
> Read CLAUDE.md and main.py. Polish the CLI: `python main.py --input <file-or-folder> --n 5` should run the full pipeline (transcribe → score → reframe → caption → export) and support a folder of videos for batch processing. Add an optional --suggest flag that asks claude-haiku-4-5 to propose a title, description, and hashtags per clip, saved next to each clip as a .txt. Write a README documenting every flag.

**Done when:** one command turns a folder of raw videos into finished, captioned, titled clips in /output.

---

## General Claude Code habits (apply to every chat)

- **One phase per chat.** Fresh context = better focus and fewer regressions. Start each new chat by naming the phase and telling it which files to read (it re-reads `CLAUDE.md` automatically, but point it at the specific module files for that phase).
- **Paste real errors, not descriptions.** When something breaks, paste the exact command you ran and the full error text. "It didn't work" wastes a round trip.
- **Commit after every green phase.** `git commit` gives you a safe rollback point before the next phase can break things. Ask Claude Code to commit for you.
- **Keep prompts as data.** The scoring prompt lives in `prompts/score.txt`. You'll edit it far more than any code — never let it get buried in a Python string.
- **Point it at a sample.** Keep one short test video handy and always tell Claude Code its path so it can actually run and verify, not just write code.
- **Ask for a test step, every phase.** End each phase with "add a manual test step to the README." Future-you will thank you.
- **Let it read before it writes.** Beginning a phase with "read CLAUDE.md and <the relevant files>" keeps it from reinventing things you already built.

---

## Quick reference

| Phase | Output | Runs on |
|---|---|---|
| 1 Smoke test | Environment verified | GPU + API |
| 2 Transcribe | transcript.json w/ word timings | GPU (Whisper) |
| 3 Dumb cutter | video → vertical clip (plumbing) | GPU (ffmpeg/NVENC) |
| 4 Moment detection | ranked clips (the core) | Claude API |
| 5 Audio energy | energy-boosted ranking | CPU (librosa) |
| 6 Reframe | speaker-tracked crop | GPU (mediapipe) |
| 7 Captions | burned-in word-synced subs | GPU (ffmpeg) |
| 8 Orchestration | one-command batch pipeline | all |

Cost to run once built: cents per source hour (Claude API only); everything else is free and local.
