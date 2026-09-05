"""
Local Clipper — local web UI.

A tiny, LOCAL, single-user web app for driving the clipper from a browser:
paste a video URL or a local file path, click Clip, watch the pipeline run
(live log), then pick from the clips it produced. Below that is a gallery of
everything already in /output.

It also has a visual Trimmer at /editor: pick a downloaded video (or paste a
link to fetch one), scrub it on a timeline with a thumbnail filmstrip + audio
waveform, drag in/out handles, and cut that exact window to a 9:16 clip
(optionally face-track reframe or facecam split). ffmpeg builds the previews.

Consistent with CLAUDE.md: it binds 127.0.0.1 (this machine only), has no
accounts, no uploads, and never posts anywhere — "choose from those" means
review / select / download locally, then you upload manually. Clipping just runs
the same `main.py` you'd run on the command line, as a subprocess.

Stdlib only (no new dependency). HTTP Range is supported so the <video> players
scrub properly. On Windows it augments PATH from the registry so the spawned
pipeline finds ffmpeg / yt-dlp even if this server was launched from a shell with
a stale PATH.

Standalone:
    venv\\Scripts\\python.exe serve.py
    venv\\Scripts\\python.exe serve.py --output output --port 8000
    venv\\Scripts\\python.exe serve.py --no-open
"""

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
MEDIA_DIR = "media"  # source videos (downloads land here) for the trimmer
# Timeline previews (waveform + thumbnail filmstrip) are generated with ffmpeg
# and cached here, keyed by source file size+mtime.
PREVIEW_DIR = os.path.join(tempfile.gettempdir(), "loclip_previews")
FILMSTRIP_COUNT = 48
# Filenames from the --n pipeline: <stem>_rank01_score71_47.59-87.57.mp4
_RANK_RE = re.compile(r"_rank(\d+)_score(\d+)_([\d.]+)-([\d.]+)\.[^.]+$")
# Filenames from cut.py / --dumb: <stem>_47.59-87.57_cover.mp4 (no rank/score)
_SPAN_RE = re.compile(r"_([\d.]+)-([\d.]+)(?:_(cover|contain))?\.[^.]+$")
# Clip paths as main.py prints them, e.g. output/<stem>_rank01_....mp4
_OUT_RE = re.compile(r"output[\\/]([^\s\\/]+\.(?:mp4|mov|mkv|webm|m4v))")

HERE = os.path.dirname(os.path.abspath(__file__))

# --- Job state (single-user, so one job at a time) -------------------------
_JOB_LOCK = threading.Lock()
JOB = {"status": "idle", "input": "", "n": 0, "log": [], "clips": []}
# The running subprocess, so /cancel can stop a clip in progress.
_CURRENT = {"proc": None, "cancel": False}


def _kill_proc(proc):
    """Kill a subprocess and its children (ffmpeg/whisper/yt-dlp) if running."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            proc.terminate()
    except Exception:
        pass

# --- Trimmer: source videos, timeline previews, URL download ---------------
_INFO_CACHE = {}
_PREVIEWS = {}                     # name -> {"status": building|ready|error}
_PREVIEW_LOCK = threading.Lock()
_DL = {"status": "idle", "input": "", "name": "", "log": []}
_DL_LOCK = threading.Lock()


def _source_path(name: str):
    """Absolute path of a source video in MEDIA_DIR, or None (traversal-safe)."""
    name = os.path.basename(name)
    base = os.path.abspath(MEDIA_DIR)
    full = os.path.abspath(os.path.join(base, name))
    if os.path.commonpath([base, full]) != base or not os.path.isfile(full):
        return None
    return full


def list_sources() -> list:
    """Video files in MEDIA_DIR, newest first, as {name, size}."""
    try:
        names = os.listdir(MEDIA_DIR)
    except FileNotFoundError:
        return []
    out = []
    for n in names:
        p = os.path.join(MEDIA_DIR, n)
        if os.path.isfile(p) and os.path.splitext(n)[1].lower() in VIDEO_EXTS:
            out.append({"name": n, "size": os.path.getsize(p),
                        "mtime": os.path.getmtime(p)})
    out.sort(key=lambda c: c["mtime"], reverse=True)
    return [{"name": c["name"], "size": c["size"]} for c in out]


def source_info(name: str):
    """{width, height, duration} for a source, via ffprobe (cached)."""
    if name in _INFO_CACHE:
        return _INFO_CACHE[name]
    full = _source_path(name)
    probe = shutil.which("ffprobe")
    info = {"width": 0, "height": 0, "duration": 0.0}
    if full and probe:
        try:
            out = subprocess.run(
                [probe, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-show_entries", "format=duration", "-of", "json", full],
                capture_output=True, text=True)
            d = json.loads(out.stdout or "{}")
            st = (d.get("streams") or [{}])[0]
            info = {"width": int(st.get("width", 0) or 0),
                    "height": int(st.get("height", 0) or 0),
                    "duration": round(float((d.get("format") or {}).get("duration", 0) or 0), 3)}
        except (ValueError, OSError):
            pass
    if full:
        _INFO_CACHE[name] = info
    return info


def _preview_paths(name: str):
    full = _source_path(name)
    if not full:
        return None, None
    st = os.stat(full)
    key = hashlib.sha1(f"{name}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]
    return (os.path.join(PREVIEW_DIR, key + "_wave.png"),
            os.path.join(PREVIEW_DIR, key + "_strip.jpg"))


def _build_previews(name: str):
    """Generate the waveform PNG and thumbnail-strip JPG for a source (ffmpeg)."""
    full = _source_path(name)
    ffmpeg = shutil.which("ffmpeg")
    wave, strip = _preview_paths(name)
    if not full or not ffmpeg or not wave:
        with _PREVIEW_LOCK:
            _PREVIEWS[name] = {"status": "error"}
        return
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    dur = source_info(name)["duration"]
    try:
        if not os.path.exists(wave):
            subprocess.run(
                [ffmpeg, "-y", "-i", full, "-filter_complex",
                 "aformat=channel_layouts=mono,showwavespic=s=1600x80:colors=#7c9cff",
                 "-frames:v", "1", wave], capture_output=True, text=True)
        if not os.path.exists(strip) and dur > 0:
            fps = max(FILMSTRIP_COUNT / dur, 0.02)
            subprocess.run(
                [ffmpeg, "-y", "-i", full, "-vf",
                 f"fps={fps:.6f},scale=-1:80,tile={FILMSTRIP_COUNT}x1",
                 "-frames:v", "1", "-q:v", "4", strip], capture_output=True, text=True)
        status = "ready" if os.path.exists(wave) else "error"
    except OSError:
        status = "error"
    with _PREVIEW_LOCK:
        _PREVIEWS[name] = {"status": status}


def ensure_previews(name: str) -> str:
    """Kick off preview generation if needed; return building|ready|error."""
    wave, _ = _preview_paths(name)
    if wave and os.path.exists(wave):
        return "ready"
    with _PREVIEW_LOCK:
        cur = _PREVIEWS.get(name)
        if cur and cur["status"] in ("building", "ready"):
            return cur["status"]
        _PREVIEWS[name] = {"status": "building"}
    threading.Thread(target=_build_previews, args=(name,), daemon=True).start()
    return "building"


def _run_download(input_value: str):
    """Download a URL (or resolve a local path) into MEDIA_DIR via ingest.py."""
    with _DL_LOCK:
        _DL.update(status="running", input=input_value, name="", log=[])
    try:
        proc = subprocess.Popen(
            [sys.executable, "ingest.py", "--input", input_value], cwd=HERE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            _DL["log"].append(line.rstrip("\n"))
        proc.wait()
    except Exception as e:  # spawning ingest failed
        _DL["log"].append(f"ERROR: {type(e).__name__}: {e}")
        _DL.update(status="error")
        return
    # ingest.py prints the resolved local path; find a media/ file in the output.
    name = ""
    for line in reversed(_DL["log"]):
        for m in re.finditer(r"([^\s\"']+\.(?:mp4|mov|mkv|webm|m4v))", line):
            cand = os.path.basename(m.group(1))
            if _source_path(cand):
                name = cand
                break
        if name:
            break
    if proc.returncode == 0 and name:
        _DL.update(status="done", name=name)
    else:
        _DL.update(status="error")


def start_download(input_value: str) -> bool:
    with _DL_LOCK:
        if _DL["status"] == "running":
            return False
    threading.Thread(target=_run_download, args=(input_value,), daemon=True).start()
    return True


def _augment_path_windows():
    """Merge the machine + user PATH from the registry into this process.

    When this server is launched from a shell with a stale PATH, the pipeline
    subprocess would inherit it and fail to find ffmpeg / yt-dlp / nvenc. Pulling
    the real PATH from the registry fixes that regardless of how we were started.
    """
    if sys.platform != "win32":
        return
    import winreg
    extra = []
    for root, sub in (
        (winreg.HKEY_LOCAL_MACHINE,
         r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
        (winreg.HKEY_CURRENT_USER, "Environment"),
    ):
        try:
            with winreg.OpenKey(root, sub) as k:
                val, _ = winreg.QueryValueEx(k, "Path")
                extra.append(os.path.expandvars(val))
        except OSError:
            pass
    parts = os.environ.get("PATH", "").split(os.pathsep)
    seen = {p.lower() for p in parts if p}
    for chunk in extra:
        for p in chunk.split(os.pathsep):
            if p and p.lower() not in seen:
                parts.append(p)
                seen.add(p.lower())
    os.environ["PATH"] = os.pathsep.join(p for p in parts if p)


def _fmt_size(n: int) -> str:
    mb = n / (1024 * 1024)
    return f"{mb:.1f} MB" if mb >= 1 else f"{n / 1024:.0f} KB"


def _fmt_dur(seconds: float) -> str:
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    return f"{m}:{s:02d}"


def clip_info(clips_dir: str, name: str):
    """Metadata dict for one clip file, or None if it isn't a readable video."""
    if os.path.splitext(name)[1].lower() not in VIDEO_EXTS:
        return None
    path = os.path.join(clips_dir, name)
    if not os.path.isfile(path):
        return None

    info = {"name": name, "size": os.path.getsize(path),
            "rank": None, "score": None, "start": None, "end": None,
            "suggest": None, "reason": None, "energy": None, "blended": None}
    m = _RANK_RE.search(name)
    if m:
        info["rank"] = int(m.group(1))
        info["score"] = int(m.group(2))
        info["start"] = float(m.group(3))
        info["end"] = float(m.group(4))
    else:
        m = _SPAN_RE.search(name)
        if m:
            info["start"] = float(m.group(1))
            info["end"] = float(m.group(2))

    meta = os.path.splitext(path)[0] + ".meta.json"
    if os.path.isfile(meta):
        try:
            with open(meta, encoding="utf-8") as f:
                md = json.load(f)
            info["reason"] = md.get("reason") or None
            info["energy"] = md.get("energy")
            info["blended"] = md.get("blended")
        except (OSError, ValueError):
            pass

    txt = os.path.splitext(path)[0] + ".txt"
    if os.path.isfile(txt):
        try:
            with open(txt, encoding="utf-8") as f:
                info["suggest"] = f.read().strip()
        except OSError:
            pass
    return info


def scan_clips(clips_dir: str) -> list:
    """Every video in clips_dir, grouped by source stem then best-first."""
    try:
        names = sorted(os.listdir(clips_dir))
    except FileNotFoundError:
        return []
    clips = [c for c in (clip_info(clips_dir, n) for n in names) if c]

    def sort_key(c):
        stem = re.split(r"_rank\d+|_[\d.]+-[\d.]+", c["name"])[0]
        return (stem, c["rank"] if c["rank"] is not None else 9999, c["name"])

    clips.sort(key=sort_key)
    return clips


def _clips_from_log(log_lines: list, clips_dir: str) -> list:
    """Extract the clip basenames a run reported, in order, deduped & existing."""
    seen, out = set(), []
    for line in log_lines:
        for m in _OUT_RE.finditer(line):
            name = m.group(1)
            if name not in seen and os.path.isfile(os.path.join(clips_dir, name)):
                seen.add(name)
                out.append(name)
    return out


def _run_job(input_value: str, n: int, suggest: bool, clips_dir: str,
             start: str = "", end: str = "", split: bool = False,
             facecam: str = "", manual: bool = False, reframe: bool = False,
             vibe: str = ""):
    """Run the pipeline (or a plain exact cut) as a subprocess, streaming output.

    manual=True cuts exactly [start, end] via cut.py (no transcription/scoring)
    -- a fast "just grab this segment" path. Otherwise runs the full main.py."""
    if manual:
        args = [sys.executable, "cut.py", "--input", input_value,
                "--start", start, "--end", end]
        if split:
            args += ["--layout", "split"]
            if facecam:
                args += ["--facecam", facecam]
        elif reframe:
            args += ["--reframe"]
    else:
        args = [sys.executable, "main.py", "--input", input_value, "--n", str(n)]
        if suggest:
            args.append("--suggest")
        if start:
            args += ["--start", start]
        if end:
            args += ["--end", end]
        if split:
            args.append("--split")
            if facecam:
                args += ["--facecam", facecam]
        if vibe:
            args += ["--vibe", vibe]
    JOB["log"].append("$ " + " ".join(args[1:]))
    try:
        proc = subprocess.Popen(
            args, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
        )
        _CURRENT["proc"] = proc
        for line in proc.stdout:
            JOB["log"].append(line.rstrip("\n"))
        proc.wait()
        rc = proc.returncode
    except Exception as e:  # spawning itself failed
        JOB["log"].append(f"ERROR: {type(e).__name__}: {e}")
        JOB["status"] = "error"
        _CURRENT["proc"] = None
        return

    _CURRENT["proc"] = None
    JOB["clips"] = _clips_from_log(JOB["log"], clips_dir)
    if _CURRENT["cancel"]:
        JOB["log"].append("[cancelled]")
        JOB["status"] = "cancelled"
    elif rc == 0:
        JOB["status"] = "done"
    else:
        JOB["log"].append(f"[exited with code {rc}]")
        JOB["status"] = "error"


def start_job(input_value: str, n: int, suggest: bool, clips_dir: str,
              start: str = "", end: str = "", split: bool = False,
              facecam: str = "", manual: bool = False, reframe: bool = False,
              vibe: str = "") -> bool:
    """Start a clip job if none is running. Returns False if one already is."""
    with _JOB_LOCK:
        if JOB["status"] == "running":
            return False
        JOB.update(status="running", input=input_value, n=n, log=[], clips=[])
        _CURRENT["cancel"] = False
    threading.Thread(
        target=_run_job,
        args=(input_value, n, suggest, clips_dir, start, end, split, facecam,
              manual, reframe, vibe),
        daemon=True).start()
    return True


# --- HTML ------------------------------------------------------------------

def _card_html(c: dict) -> str:
    src = "/clips/" + urllib.parse.quote(c["name"])
    badges = []
    if c["rank"] is not None:
        badges.append(f'<span class="badge rank">#{c["rank"]}</span>')
    if c["score"] is not None:
        badges.append(f'<span class="badge score">score {c["score"]}</span>')
    if c.get("energy") is not None:
        badges.append(f'<span class="badge energy">energy {c["energy"]:.2f}</span>')
    meta = []
    if c["start"] is not None and c["end"] is not None:
        meta.append(f'{_fmt_dur(c["end"] - c["start"])} &middot; {c["start"]:g}–{c["end"]:g}s')
    meta.append(_fmt_size(c["size"]))
    reason = ""
    if c.get("reason"):
        reason = f'<div class="reason">{html.escape(c["reason"])}</div>'
    suggest = ""
    if c["suggest"]:
        suggest = ('<details class="suggest"><summary>suggested caption</summary>'
                   f'<pre>{html.escape(c["suggest"])}</pre></details>')
    return f"""<div class="card">
      <video controls preload="metadata" playsinline src="{src}"></video>
      <div class="info">
        <div class="badges">{''.join(badges)}</div>
        <div class="fname" title="{html.escape(c['name'])}">{html.escape(c['name'])}</div>
        <div class="meta">{' &middot; '.join(meta)}</div>
        {reason}
        {suggest}
        <a class="dl" href="{src}" download>download</a>
      </div>
    </div>"""


def render_page(clips_dir: str) -> bytes:
    clips = scan_clips(clips_dir)
    abs_dir = os.path.abspath(clips_dir)
    count = f"{len(clips)} clip{'s' if len(clips) != 1 else ''}"

    if clips:
        library = f'<div class="grid">{"".join(_card_html(c) for c in clips)}</div>'
    else:
        library = '<div class="empty">No clips yet — make some above.</div>'

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local Clipper</title>
<style>
  :root {{
    --bg:#0f1115; --panel:#151821; --card:#1a1d24; --line:#2a2e37; --text:#e8eaed;
    --dim:#9aa0aa; --accent:#7c9cff; --score:#3ecf8e; --err:#ff6b6b;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
    font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif; }}
  header {{ padding:16px 24px; border-bottom:1px solid var(--line);
    display:flex; align-items:baseline; gap:14px; }}
  header h1 {{ font-size:16px; margin:0; font-weight:600; }}
  header .path {{ color:var(--dim); font-size:12px; margin-left:auto;
    font-family:ui-monospace,Consolas,monospace; }}
  main {{ max-width:1200px; margin:0 auto; padding:24px; }}
  .create {{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
    padding:18px; margin-bottom:24px; }}
  .create .row {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
  .create input[type=text] {{ flex:1; min-width:280px; padding:10px 12px; font-size:14px;
    background:#0b0d11; border:1px solid var(--line); border-radius:8px; color:var(--text); }}
  .create input[type=number] {{ width:70px; padding:10px; background:#0b0d11;
    border:1px solid var(--line); border-radius:8px; color:var(--text); }}
  .create input.time {{ width:120px; padding:8px 10px; background:#0b0d11;
    border:1px solid var(--line); border-radius:8px; color:var(--text); }}
  .create label.opt {{ color:var(--dim); display:flex; align-items:center; gap:6px; }}
  button {{ padding:10px 18px; font-size:14px; font-weight:600; border:0; border-radius:8px;
    background:var(--accent); color:#0b0d11; cursor:pointer; }}
  button:disabled {{ opacity:.5; cursor:default; }}
  .cancelbtn {{ margin-left:auto; padding:5px 12px; font-size:12px; background:#2a1d22;
    color:var(--err); border:1px solid #4a2b31; }}
  .hint {{ color:var(--dim); font-size:12px; margin-top:8px; }}
  .job {{ margin-top:16px; }}
  .jobhead {{ display:flex; align-items:center; gap:10px; font-weight:600; }}
  .spinner {{ width:14px; height:14px; border:2px solid var(--line); border-top-color:var(--accent);
    border-radius:50%; animation:spin .8s linear infinite; }}
  @keyframes spin {{ to {{ transform:rotate(360deg); }} }}
  pre.log {{ background:#0b0d11; border:1px solid var(--line); border-radius:8px; padding:12px;
    max-height:240px; overflow:auto; font:12px/1.5 ui-monospace,Consolas,monospace;
    white-space:pre-wrap; word-break:break-word; margin:10px 0 0; }}
  h2 {{ font-size:14px; color:var(--dim); font-weight:600; margin:26px 0 12px;
    text-transform:uppercase; letter-spacing:.04em; }}
  .bar {{ display:flex; align-items:center; gap:12px; margin-bottom:12px; }}
  .grid {{ display:grid; gap:20px;
    grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    overflow:hidden; display:flex; flex-direction:column; position:relative; }}
  .card.sel {{ outline:2px solid var(--accent); }}
  .card .pick {{ position:absolute; top:10px; right:10px; width:20px; height:20px; z-index:1;
    accent-color:var(--accent); cursor:pointer; }}
  video {{ width:100%; aspect-ratio:9/16; background:#000; display:block; }}
  .info {{ padding:12px; display:flex; flex-direction:column; gap:7px; }}
  .badges {{ display:flex; gap:6px; }}
  .badge {{ font-size:12px; font-weight:600; padding:2px 8px; border-radius:999px; }}
  .badge.rank {{ background:rgba(124,156,255,.18); color:var(--accent); }}
  .badge.score {{ background:rgba(62,207,142,.16); color:var(--score); }}
  .badge.energy {{ background:rgba(255,176,32,.16); color:#ffb020; }}
  .reason {{ font-size:12px; color:var(--text); opacity:.85; font-style:italic; }}
  .fname {{ font-size:12px; color:var(--dim); font-family:ui-monospace,Consolas,monospace;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .meta {{ font-size:12px; color:var(--dim); }}
  .suggest {{ font-size:12px; }}
  .suggest summary {{ cursor:pointer; color:var(--accent); }}
  .suggest pre {{ white-space:pre-wrap; word-break:break-word; margin:6px 0 0;
    color:var(--text); font-family:inherit; background:#0b0d11; border:1px solid var(--line);
    border-radius:8px; padding:10px; }}
  .dl {{ font-size:12px; color:var(--accent); text-decoration:none; }}
  .dl:hover {{ text-decoration:underline; }}
  .empty {{ padding:40px; text-align:center; color:var(--dim); }}
</style>
</head>
<body>
<header>
  <h1>Local Clipper</h1>
  <a href="/editor" style="color:var(--accent);text-decoration:none;font-size:13px">✂ Trim a video</a>
  <span class="path">{html.escape(abs_dir)}</span>
</header>
<main>
  <div class="create">
    <div class="row">
      <input type="text" id="inp" placeholder="Video URL (YouTube…) or a local file path on this machine">
      <label class="opt">clips <input type="number" id="n" min="1" max="20" value="5"></label>
      <label class="opt"><input type="checkbox" id="suggest"> caption suggestions</label>
      <button id="go">Clip</button>
    </div>
    <div class="row" style="margin-top:10px">
      <label class="opt" style="flex:1">focus
        <input type="text" id="vibe" style="flex:1;min-width:200px;padding:9px 11px;background:#0b0d11;border:1px solid var(--line);border-radius:8px;color:var(--text)"
               placeholder="what to look for (optional): e.g. funny reactions, hot takes, practical tips"></label>
    </div>
    <div class="row" style="margin-top:10px">
      <label class="opt">only clip
        <input type="text" id="start" class="time" placeholder="from (e.g. 5:00)"></label>
      <label class="opt">→
        <input type="text" id="end" class="time" placeholder="to (e.g. 12:30)"></label>
      <label class="opt"><input type="checkbox" id="manual"> exact cut (no scoring)</label>
      <span class="hint" style="margin:0">optional — leave blank for the whole video. Tick “exact cut” to just grab from→to as-is (needs both).</span>
    </div>
    <div class="row" style="margin-top:10px">
      <label class="opt"><input type="checkbox" id="split"> facecam split (cam on top, gameplay on bottom)</label>
      <input type="text" id="facecam" class="time" style="width:220px" placeholder="facecam: auto (or x,y,w,h / corner)" disabled>
      <span class="hint" style="margin:0">for streamer clips; auto-detects the facecam, or set a corner (bottom-left) / pixels</span>
    </div>
    <div class="hint">Runs transcribe → score → reframe → caption locally, then lets you pick the best clips below. A URL is downloaded first (yt-dlp). Only clip content you have the rights to.</div>
    <div class="job" id="job" hidden>
      <div class="jobhead"><span id="spin" class="spinner"></span><span id="jobstatus"></span><button id="cancel" class="cancelbtn" hidden>■ Cancel</button></div>
      <pre class="log" id="log"></pre>
      <div id="results"></div>
    </div>
  </div>

  <h2>All clips in /output — {count}</h2>
  {library}
</main>
<script>
const $ = s => document.querySelector(s);
const go = $('#go'), job = $('#job'), logEl = $('#log'), statusEl = $('#jobstatus'), spin = $('#spin');
let timer = null;

function fmtDur(s) {{ s = Math.round(s); return Math.floor(s/60) + ':' + String(s%60).padStart(2,'0'); }}
function fmtSize(n) {{ const mb = n/1048576; return mb>=1 ? mb.toFixed(1)+' MB' : Math.round(n/1024)+' KB'; }}

function resultCard(c) {{
  const src = '/clips/' + encodeURIComponent(c.name);
  const badges = (c.rank!=null?`<span class="badge rank">#${{c.rank}}</span>`:'')
               + (c.score!=null?`<span class="badge score">score ${{c.score}}</span>`:'')
               + (c.energy!=null?`<span class="badge energy">energy ${{c.energy.toFixed(2)}}</span>`:'');
  let meta = [];
  if (c.start!=null && c.end!=null) meta.push(fmtDur(c.end-c.start)+' · '+c.start+'–'+c.end+'s');
  meta.push(fmtSize(c.size));
  const reason = c.reason ? `<div class="reason">${{c.reason.replace(/</g,'&lt;')}}</div>` : '';
  const suggest = c.suggest ? `<details class="suggest"><summary>suggested caption</summary><pre>${{c.suggest.replace(/</g,'&lt;')}}</pre></details>` : '';
  const div = document.createElement('div');
  div.className = 'card';
  div.innerHTML = `<input type="checkbox" class="pick" checked>
    <video controls preload="metadata" playsinline src="${{src}}"></video>
    <div class="info"><div class="badges">${{badges}}</div>
      <div class="fname" title="${{c.name}}">${{c.name}}</div>
      <div class="meta">${{meta.join(' · ')}}</div>${{reason}}${{suggest}}
      <a class="dl" href="${{src}}" download>download</a></div>`;
  const pick = div.querySelector('.pick');
  const sync = () => div.classList.toggle('sel', pick.checked);
  pick.addEventListener('change', sync); sync();
  return div;
}}

function renderResults(clips) {{
  const r = $('#results');
  if (!clips.length) {{ r.innerHTML = '<div class="empty">No clips were produced.</div>'; return; }}
  const bar = document.createElement('div');
  bar.className = 'bar';
  const dl = document.createElement('button');
  dl.textContent = 'Download selected';
  dl.onclick = () => {{
    r.querySelectorAll('.card').forEach((card,i) => {{
      const p = card.querySelector('.pick');
      if (p && p.checked) {{ const a = card.querySelector('a.dl');
        setTimeout(() => a.click(), i*400); }}
    }});
  }};
  bar.appendChild(dl);
  const note = document.createElement('span');
  note.style.color = 'var(--dim)'; note.style.fontSize = '12px';
  note.textContent = 'Tick the ones you want, then download. Refresh the page to add them to the library below.';
  bar.appendChild(note);
  const grid = document.createElement('div');
  grid.className = 'grid';
  clips.forEach(c => grid.appendChild(resultCard(c)));
  r.innerHTML = '<h2>Pick your clips</h2>';
  r.appendChild(bar); r.appendChild(grid);
}}

async function poll() {{
  const s = await fetch('/status').then(r => r.json());
  logEl.textContent = s.log;
  logEl.scrollTop = logEl.scrollHeight;
  if (s.status === 'running') {{ statusEl.textContent = 'Clipping “' + s.input + '” …'; $('#cancel').hidden = false; return; }}
  clearInterval(timer); timer = null;
  spin.style.display = 'none';
  go.disabled = false; $('#cancel').hidden = true;
  if (s.status === 'error') {{ statusEl.textContent = 'Failed — see log.'; statusEl.style.color = 'var(--err)'; }}
  else if (s.status === 'cancelled') {{ statusEl.textContent = 'Cancelled.'; statusEl.style.color = 'var(--err)'; renderResults(s.clips); }}
  else {{ statusEl.textContent = 'Done — ' + s.clips.length + ' clip(s).'; renderResults(s.clips); }}
}}
$('#cancel').onclick = async () => {{ $('#cancel').disabled = true; statusEl.textContent = 'Cancelling…';
  await fetch('/cancel', {{ method:'POST' }}).catch(()=>{{}}); $('#cancel').disabled = false; }};

go.onclick = async () => {{
  const input = $('#inp').value.trim();
  if (!input) {{ $('#inp').focus(); return; }}
  go.disabled = true; job.hidden = false; spin.style.display = '';
  statusEl.style.color = ''; statusEl.textContent = 'Starting…';
  logEl.textContent = ''; $('#results').innerHTML = '';
  const res = await fetch('/clip', {{ method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{ input, n: +$('#n').value || 5, suggest: $('#suggest').checked,
      start: $('#start').value.trim(), end: $('#end').value.trim(),
      split: $('#split').checked, facecam: $('#facecam').value.trim(),
      manual: $('#manual').checked, vibe: $('#vibe').value.trim() }}) }});
  if (res.status === 409) {{ statusEl.textContent = 'A job is already running.'; spin.style.display='none'; go.disabled=false; return; }}
  if (!res.ok) {{ const j = await res.json().catch(()=>({{}})); statusEl.textContent = j.error || 'Request failed.'; statusEl.style.color='var(--err)'; spin.style.display='none'; go.disabled=false; return; }}
  timer = setInterval(poll, 1000); poll();
}};
$('#inp').addEventListener('keydown', e => {{ if (e.key === 'Enter') go.click(); }});
$('#split').addEventListener('change', e => {{ $('#facecam').disabled = !e.target.checked; }});

// Restore an in-progress or finished job if the page is reloaded.
(async () => {{
  const s = await fetch('/status').then(r => r.json()).catch(() => null);
  if (!s || s.status === 'idle') return;
  job.hidden = false; go.disabled = s.status === 'running';
  spin.style.display = s.status === 'running' ? '' : 'none';
  logEl.textContent = s.log;
  if (s.status === 'running') {{ timer = setInterval(poll, 1000); poll(); }}
  else if (s.status === 'error') {{ statusEl.textContent = 'Failed — see log.'; statusEl.style.color = 'var(--err)'; }}
  else {{ statusEl.textContent = 'Done — ' + s.clips.length + ' clip(s).'; renderResults(s.clips); }}
}})();
</script>
</body>
</html>"""
    return page.encode("utf-8")


def render_editor() -> bytes:
    return _EDITOR_HTML.encode("utf-8")


_EDITOR_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trimmer — Local Clipper</title>
<style>
  :root { --bg:#0f1115; --panel:#151821; --card:#1a1d24; --line:#2a2e37; --text:#e8eaed;
    --dim:#9aa0aa; --accent:#7c9cff; --score:#3ecf8e; --err:#ff6b6b; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif; }
  a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
  header { padding:16px 24px; border-bottom:1px solid var(--line); display:flex; gap:14px; align-items:baseline; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  main { max-width:1100px; margin:0 auto; padding:24px; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; margin-bottom:18px; }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  input[type=text], select { padding:9px 11px; font-size:14px; background:#0b0d11;
    border:1px solid var(--line); border-radius:8px; color:var(--text); }
  input#url { flex:1; min-width:240px; } select#src { min-width:240px; flex:1; }
  input.time { width:110px; text-align:center; }
  label.opt { color:var(--dim); display:flex; align-items:center; gap:6px; }
  button { padding:9px 16px; font-size:14px; font-weight:600; border:0; border-radius:8px;
    background:var(--accent); color:#0b0d11; cursor:pointer; }
  button.ghost { background:#222736; color:var(--text); border:1px solid var(--line); }
  button:disabled { opacity:.5; cursor:default; }
  .hint { color:var(--dim); font-size:12px; }
  video { width:100%; max-height:52vh; background:#000; border-radius:10px; display:block; }
  /* timeline */
  .tl { position:relative; height:92px; margin-top:14px; border:1px solid var(--line);
    border-radius:8px; overflow:hidden; background:#0b0d11; user-select:none; touch-action:none; cursor:pointer; }
  .tl .film { position:absolute; inset:0 0 42px 0; background-size:100% 100%; background-repeat:no-repeat; background-color:#0b0d11; }
  .tl .wave { position:absolute; left:0; right:0; bottom:0; height:42px; width:100%; object-fit:fill; opacity:.85; background:#0b0d11; }
  .tl .sel { position:absolute; top:0; bottom:0; background:rgba(124,156,255,.22);
    border-left:2px solid var(--accent); border-right:2px solid var(--accent); }
  .tl .dim { position:absolute; top:0; bottom:0; background:rgba(11,13,17,.6); }
  .tl .handle { position:absolute; top:0; bottom:0; width:12px; margin-left:-6px; cursor:ew-resize; z-index:3; }
  .tl .handle::after { content:''; position:absolute; left:5px; top:0; bottom:0; width:2px; background:var(--accent); }
  .tl .play { position:absolute; top:0; bottom:0; width:2px; background:#fff; z-index:2; pointer-events:none; }
  .times { display:flex; gap:16px; align-items:center; margin-top:10px; flex-wrap:wrap; }
  .badge { font-size:12px; font-weight:600; padding:2px 8px; border-radius:999px; background:rgba(124,156,255,.18); color:var(--accent); }
  pre.log { background:#0b0d11; border:1px solid var(--line); border-radius:8px; padding:10px; max-height:150px;
    overflow:auto; font:12px/1.5 ui-monospace,Consolas,monospace; white-space:pre-wrap; margin:10px 0 0; }
  #result video { max-height:40vh; width:auto; }
</style>
</head>
<body>
<header>
  <h1>Local Clipper — Trimmer</h1>
  <a href="/clips">Auto-clip &amp; gallery ›</a>
</header>
<main>
  <div class="panel">
    <div class="row">
      <input type="text" id="url" placeholder="Paste a video URL (YouTube…) or a local file path — it loads here to trim">
      <button id="get">Load video</button>
    </div>
    <div class="row" style="margin-top:10px">
      <span class="hint">or open one you already downloaded:</span>
      <select id="src"><option value="">— downloaded videos —</option></select>
      <button id="load" class="ghost">Open</button>
    </div>
    <div class="hint" id="srchint" style="margin-top:8px">Paste a link and it downloads, then appears below to trim. Everything stays on your machine; nothing is uploaded.</div>
    <pre class="log" id="dllog" hidden></pre>
  </div>

  <div class="panel" id="editor" hidden>
    <video id="vid" preload="metadata" playsinline></video>
    <div class="tl" id="tl">
      <div class="film" id="film"></div>
      <img class="wave" id="wave" alt="">
      <div class="dim" id="dimL"></div>
      <div class="dim" id="dimR"></div>
      <div class="sel" id="sel"></div>
      <div class="handle" id="hIn"></div>
      <div class="handle" id="hOut"></div>
      <div class="play" id="ph"></div>
    </div>
    <div class="times">
      <button id="pp" class="ghost">▶ Play</button>
      <button id="setin" class="ghost">Set In [</button>
      <span class="badge">in <input class="time" id="tin" value="0:00"></span>
      <span class="badge">out <input class="time" id="tout" value="0:30"></span>
      <button id="setout" class="ghost">Set Out ]</button>
      <label class="opt"><input type="checkbox" id="loop" checked> loop selection</label>
      <span class="hint" id="dur"></span>
    </div>
    <div class="row" style="margin-top:12px">
      <label class="opt"><input type="checkbox" id="reframe"> face-track reframe</label>
      <label class="opt"><input type="checkbox" id="split"> facecam split</label>
      <input type="text" id="facecam" class="time" style="width:200px" placeholder="facecam: auto / corner" disabled>
      <button id="cut">Cut selection ▸</button>
      <button id="cancel" class="ghost" hidden>■ Cancel</button>
      <span class="hint" id="cutstatus"></span>
    </div>
    <div class="hint" style="margin-top:8px">Captions come from the auto-pipeline (they need the transcript); a hand-trimmed cut is exported without them. Loudness is normalized automatically.</div>
    <pre class="log" id="cutlog" hidden></pre>
    <div id="result" style="margin-top:12px"></div>
  </div>
</main>
<script>
const $ = s => document.querySelector(s);
const vid = $('#vid'), tl = $('#tl');
let dur = 0, inT = 0, outT = 30, drag = null, name = '', poller = null, dlPoller = null;

const fmt = t => { t = Math.max(0, t); const m = Math.floor(t/60), s = Math.floor(t%60);
  return m + ':' + String(s).padStart(2,'0'); };
const parseT = v => { v = String(v).trim(); if (v.includes(':')) {
  return v.split(':').reduce((a,p)=>a*60+(parseFloat(p)||0),0); } return parseFloat(v)||0; };

async function loadSources(sel) {
  const d = await fetch('/sources').then(r=>r.json()).catch(()=>({sources:[]}));
  const s = $('#src');
  s.innerHTML = '<option value="">— pick a downloaded video —</option>' +
    d.sources.map(o=>`<option value="${o.name}">${o.name}  (${(o.size/1048576).toFixed(0)} MB)</option>`).join('');
  if (sel) s.value = sel;
}

function layout() {
  const w = tl.clientWidth;
  const px = t => (dur>0 ? (t/dur)*w : 0);
  $('#sel').style.left = px(inT)+'px';  $('#sel').style.width = Math.max(0,px(outT)-px(inT))+'px';
  $('#hIn').style.left = px(inT)+'px';  $('#hOut').style.left = px(outT)+'px';
  $('#dimL').style.left = '0px'; $('#dimL').style.width = px(inT)+'px';
  $('#dimR').style.left = px(outT)+'px'; $('#dimR').style.width = Math.max(0,w-px(outT))+'px';
  $('#tin').value = fmt(inT); $('#tout').value = fmt(outT);
}
function playhead() { const w = tl.clientWidth; $('#ph').style.left = (dur>0?(vid.currentTime/dur)*w:0)+'px'; }

async function selectSource(n) {
  name = n; if (!n) return;
  $('#editor').hidden = false;
  vid.src = '/source/' + encodeURIComponent(n);
  const info = await fetch('/sourceinfo/'+encodeURIComponent(n)).then(r=>r.json()).catch(()=>({duration:0}));
  dur = info.duration || 0;
  inT = 0; outT = Math.min(30, dur||30);
  $('#dur').textContent = 'duration ' + fmt(dur);
  $('#film').style.backgroundImage = ''; $('#wave').removeAttribute('src');
  layout(); playhead();
  // build + load timeline previews (waveform + filmstrip)
  const tick = async () => {
    const p = await fetch('/previewstatus/'+encodeURIComponent(n)).then(r=>r.json()).catch(()=>({status:'error'}));
    if (p.status === 'ready') {
      $('#film').style.backgroundImage = `url(/filmstrip/${encodeURIComponent(n)})`;
      $('#wave').src = '/waveform/'+encodeURIComponent(n);
    } else if (p.status === 'building') { setTimeout(tick, 1200); }
  };
  tick();
}

// timeline pointer interaction
function timeAt(clientX) { const r = tl.getBoundingClientRect();
  return Math.max(0, Math.min(dur, ((clientX-r.left)/r.width)*dur)); }
$('#hIn').addEventListener('pointerdown', e=>{ drag='in'; e.preventDefault(); });
$('#hOut').addEventListener('pointerdown', e=>{ drag='out'; e.preventDefault(); });
tl.addEventListener('pointerdown', e=>{ if (drag) return;
  if (e.target.classList.contains('handle')) return;
  vid.currentTime = timeAt(e.clientX); playhead(); });
document.addEventListener('pointermove', e=>{ if (!drag) return;
  const t = timeAt(e.clientX);
  if (drag==='in') inT = Math.min(t, outT-0.2);
  else outT = Math.max(t, inT+0.2);
  layout(); });
document.addEventListener('pointerup', ()=>{ drag=null; });

vid.addEventListener('timeupdate', ()=>{ playhead();
  if ($('#loop').checked && vid.currentTime >= outT) { vid.currentTime = inT; }
});
vid.addEventListener('loadedmetadata', ()=>{ if (!dur) { dur = vid.duration||0; layout(); } });
window.addEventListener('resize', ()=>{ layout(); playhead(); });

$('#pp').onclick = ()=>{ if (vid.paused) { if (vid.currentTime<inT||vid.currentTime>outT) vid.currentTime=inT; vid.play(); $('#pp').textContent='❚❚ Pause'; } else { vid.pause(); $('#pp').textContent='▶ Play'; } };
$('#setin').onclick = ()=>{ inT = Math.min(vid.currentTime, outT-0.2); layout(); };
$('#setout').onclick = ()=>{ outT = Math.max(vid.currentTime, inT+0.2); layout(); };
$('#tin').addEventListener('change', ()=>{ inT = Math.max(0, Math.min(parseT($('#tin').value), outT-0.2)); layout(); });
$('#tout').addEventListener('change', ()=>{ outT = Math.min(dur||1e9, Math.max(parseT($('#tout').value), inT+0.2)); layout(); });
$('#split').addEventListener('change', e=>{ $('#facecam').disabled = !e.target.checked; });
document.addEventListener('keydown', e=>{ if (['INPUT','SELECT'].includes(e.target.tagName)) return;
  if (e.code==='Space'){ e.preventDefault(); $('#pp').click(); }
  else if (e.key==='i'||e.key==='I'){ $('#setin').click(); }
  else if (e.key==='o'||e.key==='O'){ $('#setout').click(); } });

$('#src').addEventListener('change', e=>{ if (e.target.value) selectSource(e.target.value); });
$('#load').onclick = ()=>{ const v=$('#src').value; if (v) selectSource(v); };
$('#url').addEventListener('keydown', e=>{ if (e.key==='Enter') $('#get').click(); });
$('#cancel').onclick = async ()=>{ $('#cancel').disabled=true; $('#cutstatus').textContent='Cancelling…';
  await fetch('/cancel',{method:'POST'}).catch(()=>{}); $('#cancel').disabled=false; };

$('#get').onclick = async ()=>{
  const input = $('#url').value.trim(); if (!input) return;
  $('#get').disabled = true; $('#dllog').hidden = false; $('#dllog').textContent = 'starting…';
  const res = await fetch('/download', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({input})});
  if (!res.ok) { $('#dllog').textContent = 'Download busy or failed.'; $('#get').disabled=false; return; }
  clearInterval(dlPoller);
  dlPoller = setInterval(async ()=>{
    const s = await fetch('/downloadstatus').then(r=>r.json());
    $('#dllog').textContent = s.log; $('#dllog').scrollTop = $('#dllog').scrollHeight;
    if (s.status !== 'running') { clearInterval(dlPoller); $('#get').disabled=false;
      if (s.status==='done' && s.name) { await loadSources(s.name); selectSource(s.name); }
    }
  }, 1000);
};

$('#cut').onclick = async ()=>{
  if (!name) return;
  $('#cut').disabled = true; $('#cancel').hidden = false;
  $('#cutstatus').textContent = 'Cutting…'; $('#cutstatus').style.color='';
  $('#cutlog').hidden = false; $('#cutlog').textContent=''; $('#result').innerHTML='';
  const body = { input: 'media/'+name, manual: true, start: String(inT.toFixed(2)), end: String(outT.toFixed(2)),
    split: $('#split').checked, facecam: $('#facecam').value.trim() };
  // reframe (non-split) uses the pipeline's reframe on a manual cut via cut.py? manual path is a plain cut;
  // for face-track we route through cut.py's --reframe by piggybacking on split=false + reframe flag:
  body.reframe = $('#reframe').checked;
  const res = await fetch('/clip', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  if (res.status===409){ $('#cutstatus').textContent='A job is already running.'; $('#cut').disabled=false; $('#cancel').hidden=true; return; }
  if (!res.ok){ const j=await res.json().catch(()=>({})); $('#cutstatus').textContent=j.error||'Failed.'; $('#cutstatus').style.color='var(--err)'; $('#cut').disabled=false; $('#cancel').hidden=true; return; }
  clearInterval(poller);
  poller = setInterval(async ()=>{
    const s = await fetch('/status').then(r=>r.json());
    $('#cutlog').textContent = s.log; $('#cutlog').scrollTop = $('#cutlog').scrollHeight;
    if (s.status!=='running'){ clearInterval(poller); $('#cut').disabled=false; $('#cancel').hidden=true;
      if (s.status==='error'){ $('#cutstatus').textContent='Failed — see log.'; $('#cutstatus').style.color='var(--err)'; }
      else if (s.status==='cancelled'){ $('#cutstatus').textContent='Cancelled.'; $('#cutstatus').style.color='var(--err)'; }
      else { const c = s.clips[s.clips.length-1];
        $('#cutstatus').textContent = 'Done!';
        if (c) $('#result').innerHTML = `<video controls src="/clips/${encodeURIComponent(c.name)}"></video>`
          + `<div class="hint" style="margin-top:6px">Saved as <b>${c.name}</b> — also in the <a href="/clips">gallery</a>. <a class="dl" href="/clips/${encodeURIComponent(c.name)}" download>download</a></div>`;
      }
    }
  }, 1000);
};

loadSources();
</script>
</body>
</html>"""


# --- HTTP ------------------------------------------------------------------

class ClipHandler(BaseHTTPRequestHandler):
    clips_dir = "output"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, ctype, body: bytes, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code=200):
        self._send(code, "application/json", json.dumps(obj).encode("utf-8"))

    def do_GET(self):
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        if path in ("/", "/index.html", "/editor"):
            self._send(200, "text/html; charset=utf-8", render_editor())
        elif path in ("/clips", "/gallery"):
            self._send(200, "text/html; charset=utf-8", render_page(self.server.clips_dir))
        elif path == "/status":
            snap = {"status": JOB["status"], "input": JOB["input"],
                    "log": "\n".join(JOB["log"]),
                    "clips": [c for c in (clip_info(self.server.clips_dir, n)
                                          for n in JOB["clips"]) if c]}
            self._send_json(snap)
        elif path == "/sources":
            self._send_json({"sources": list_sources()})
        elif path == "/downloadstatus":
            self._send_json({"status": _DL["status"], "name": _DL["name"],
                             "log": "\n".join(_DL["log"][-40:])})
        elif path.startswith("/sourceinfo/"):
            self._send_json(source_info(path[len("/sourceinfo/"):]))
        elif path.startswith("/previewstatus/"):
            name = path[len("/previewstatus/"):]
            self._send_json({"status": ensure_previews(name)})
        elif path.startswith("/waveform/"):
            wave, _ = _preview_paths(path[len("/waveform/"):])
            self._serve_image(wave, "image/png")
        elif path.startswith("/filmstrip/"):
            _, strip = _preview_paths(path[len("/filmstrip/"):])
            self._serve_image(strip, "image/jpeg")
        elif path.startswith("/clips/"):
            self._serve_file(self.server.clips_dir, path[len("/clips/"):])
        elif path.startswith("/source/"):
            self._serve_file(MEDIA_DIR, path[len("/source/"):])
        else:
            self.send_error(404, "Not found")

    def _serve_image(self, full, ctype):
        if not full or not os.path.isfile(full):
            self.send_error(404, "Not found")
            return
        try:
            with open(full, "rb") as f:
                data = f.read()
        except OSError:
            self.send_error(404, "Not found")
            return
        self._send(200, ctype, data)

    def do_POST(self):
        route = urllib.parse.urlparse(self.path).path
        if route == "/cancel":
            proc = _CURRENT.get("proc")
            if JOB["status"] == "running" and proc is not None:
                _CURRENT["cancel"] = True
                _kill_proc(proc)
                self._send_json({"status": "cancelling"})
            else:
                self._send_json({"status": "idle"})
            return
        if route not in ("/clip", "/download"):
            self.send_error(404, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send(400, "application/json", b'{"error":"bad request"}')
            return

        if route == "/download":
            inp = str(data.get("input", "")).strip()
            if not inp:
                self._send(400, "application/json", b'{"error":"no input"}')
            elif start_download(inp):
                self._send(200, "application/json", b'{"status":"started"}')
            else:
                self._send(409, "application/json", b'{"error":"a download is already running"}')
            return

        input_value = str(data.get("input", "")).strip()
        if not input_value:
            self._send(400, "application/json", b'{"error":"no input"}')
            return
        try:
            n = max(1, min(20, int(data.get("n", 5))))
        except (TypeError, ValueError):
            n = 5
        start = str(data.get("start", "")).strip()
        end = str(data.get("end", "")).strip()
        split = bool(data.get("split"))
        facecam = str(data.get("facecam", "")).strip()
        manual = bool(data.get("manual"))
        reframe = bool(data.get("reframe"))
        if manual and not (start and end):
            self._send(400, "application/json", b'{"error":"exact cut needs from and to"}')
            return
        vibe = str(data.get("vibe", "")).strip()[:200]
        ok = start_job(input_value, n, bool(data.get("suggest")),
                       self.server.clips_dir, start, end, split, facecam, manual,
                       reframe, vibe)
        if not ok:
            self._send(409, "application/json", b'{"error":"a job is already running"}')
        else:
            self._send(200, "application/json", b'{"status":"started"}')

    def _serve_file(self, base_dir: str, name: str):
        name = os.path.basename(name)
        base = os.path.abspath(base_dir)
        full = os.path.abspath(os.path.join(base, name))
        if os.path.commonpath([base, full]) != base or not os.path.isfile(full):
            self.send_error(404, "Not found")
            return

        ext = os.path.splitext(full)[1].lower()
        ctype = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
                 ".webm": "video/webm", ".mkv": "video/x-matroska"}.get(ext, "application/octet-stream")

        size = os.path.getsize(full)
        rng = self.headers.get("Range")
        start, end, partial = 0, size - 1, False
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if m:
                partial = True
                g1, g2 = m.group(1), m.group(2)
                if g1 == "":
                    start = max(0, size - int(g2))
                else:
                    start = int(g1)
                    end = int(g2) if g2 else size - 1
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return

        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with open(full, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # browser aborted the stream (seek / tab close) -- fine.


def serve(clips_dir: str, port: int, open_browser: bool):
    _augment_path_windows()
    if not os.path.isdir(clips_dir):
        print(f"NOTE: output folder '{clips_dir}' doesn't exist yet; "
              f"it will be created when you make clips.", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), ClipHandler)
    server.clips_dir = clips_dir
    url = f"http://127.0.0.1:{port}/"
    print(f"Serving clips from {os.path.abspath(clips_dir)}", flush=True)
    print(f"Open {url}  (local only; Ctrl+C to stop)", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        server.server_close()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Local web UI for reviewing and creating clips")
    ap.add_argument("--output", default="output", help="folder of clips to serve (default: output)")
    ap.add_argument("--port", type=int, default=8000, help="localhost port (default: 8000)")
    ap.add_argument("--no-open", dest="open_browser", action="store_false",
                    help="don't auto-open a browser tab")
    args = ap.parse_args()
    serve(args.output, args.port, args.open_browser)


if __name__ == "__main__":
    main()
