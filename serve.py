"""
Local Clipper — local web UI.

A tiny, LOCAL, single-user web app for driving the clipper from a browser:
paste a video URL or a local file path, click Clip, watch the pipeline run
(live log), then pick from the clips it produced. Below that is a gallery of
everything already in /output.

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
import html
import json
import os
import re
import subprocess
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
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
            "suggest": None}
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
             facecam: str = ""):
    """Run main.py as a subprocess, streaming its output into JOB['log']."""
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
    JOB["log"].append("$ " + " ".join(args[1:]))
    try:
        proc = subprocess.Popen(
            args, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
        )
        for line in proc.stdout:
            JOB["log"].append(line.rstrip("\n"))
        proc.wait()
        rc = proc.returncode
    except Exception as e:  # spawning itself failed
        JOB["log"].append(f"ERROR: {type(e).__name__}: {e}")
        JOB["status"] = "error"
        return

    JOB["clips"] = _clips_from_log(JOB["log"], clips_dir)
    if rc == 0:
        JOB["status"] = "done"
    else:
        JOB["log"].append(f"[main.py exited with code {rc}]")
        JOB["status"] = "error"


def start_job(input_value: str, n: int, suggest: bool, clips_dir: str,
              start: str = "", end: str = "", split: bool = False,
              facecam: str = "") -> bool:
    """Start a clip job if none is running. Returns False if one already is."""
    with _JOB_LOCK:
        if JOB["status"] == "running":
            return False
        JOB.update(status="running", input=input_value, n=n, log=[], clips=[])
    threading.Thread(
        target=_run_job,
        args=(input_value, n, suggest, clips_dir, start, end, split, facecam),
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
    meta = []
    if c["start"] is not None and c["end"] is not None:
        meta.append(f'{_fmt_dur(c["end"] - c["start"])} &middot; {c["start"]:g}–{c["end"]:g}s')
    meta.append(_fmt_size(c["size"]))
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
      <label class="opt">only clip
        <input type="text" id="start" class="time" placeholder="from (e.g. 5:00)"></label>
      <label class="opt">→
        <input type="text" id="end" class="time" placeholder="to (e.g. 12:30)"></label>
      <span class="hint" style="margin:0">optional — leave blank for the whole video; transcribes just this window so a long video is fast</span>
    </div>
    <div class="row" style="margin-top:10px">
      <label class="opt"><input type="checkbox" id="split"> facecam split (cam on top, gameplay on bottom)</label>
      <input type="text" id="facecam" class="time" style="width:220px" placeholder="facecam: auto (or x,y,w,h / corner)" disabled>
      <span class="hint" style="margin:0">for streamer clips; auto-detects the facecam, or set a corner (bottom-left) / pixels</span>
    </div>
    <div class="hint">Runs transcribe → score → reframe → caption locally, then lets you pick the best clips below. A URL is downloaded first (yt-dlp). Only clip content you have the rights to.</div>
    <div class="job" id="job" hidden>
      <div class="jobhead"><span id="spin" class="spinner"></span><span id="jobstatus"></span></div>
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
               + (c.score!=null?`<span class="badge score">score ${{c.score}}</span>`:'');
  let meta = [];
  if (c.start!=null && c.end!=null) meta.push(fmtDur(c.end-c.start)+' · '+c.start+'–'+c.end+'s');
  meta.push(fmtSize(c.size));
  const suggest = c.suggest ? `<details class="suggest"><summary>suggested caption</summary><pre>${{c.suggest.replace(/</g,'&lt;')}}</pre></details>` : '';
  const div = document.createElement('div');
  div.className = 'card';
  div.innerHTML = `<input type="checkbox" class="pick" checked>
    <video controls preload="metadata" playsinline src="${{src}}"></video>
    <div class="info"><div class="badges">${{badges}}</div>
      <div class="fname" title="${{c.name}}">${{c.name}}</div>
      <div class="meta">${{meta.join(' · ')}}</div>${{suggest}}
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
  if (s.status === 'running') {{ statusEl.textContent = 'Clipping “' + s.input + '” …'; return; }}
  clearInterval(timer); timer = null;
  spin.style.display = 'none';
  go.disabled = false;
  if (s.status === 'error') {{ statusEl.textContent = 'Failed — see log.'; statusEl.style.color = 'var(--err)'; }}
  else {{ statusEl.textContent = 'Done — ' + s.clips.length + ' clip(s).'; renderResults(s.clips); }}
}}

go.onclick = async () => {{
  const input = $('#inp').value.trim();
  if (!input) {{ $('#inp').focus(); return; }}
  go.disabled = true; job.hidden = false; spin.style.display = '';
  statusEl.style.color = ''; statusEl.textContent = 'Starting…';
  logEl.textContent = ''; $('#results').innerHTML = '';
  const res = await fetch('/clip', {{ method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{ input, n: +$('#n').value || 5, suggest: $('#suggest').checked,
      start: $('#start').value.trim(), end: $('#end').value.trim(),
      split: $('#split').checked, facecam: $('#facecam').value.trim() }}) }});
  if (res.status === 409) {{ statusEl.textContent = 'A job is already running.'; return; }}
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

    def do_GET(self):
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", render_page(self.server.clips_dir))
        elif path == "/status":
            snap = {"status": JOB["status"], "input": JOB["input"],
                    "log": "\n".join(JOB["log"]),
                    "clips": [c for c in (clip_info(self.server.clips_dir, n)
                                          for n in JOB["clips"]) if c]}
            self._send(200, "application/json", json.dumps(snap).encode("utf-8"))
        elif path.startswith("/clips/"):
            self._serve_clip(path[len("/clips/"):])
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/clip":
            self.send_error(404, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send(400, "application/json", b'{"error":"bad request"}')
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
        ok = start_job(input_value, n, bool(data.get("suggest")),
                       self.server.clips_dir, start, end, split, facecam)
        if not ok:
            self._send(409, "application/json", b'{"error":"a job is already running"}')
        else:
            self._send(200, "application/json", b'{"status":"started"}')

    def _serve_clip(self, name: str):
        name = os.path.basename(name)
        base = os.path.abspath(self.server.clips_dir)
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
