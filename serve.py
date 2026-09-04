"""
Local clip-review gallery for Local Clipper.

A tiny, LOCAL, read-only web page for eyeballing the clips in /output after a
run -- play each 9:16 clip in the browser, see its rank/score/time-range (parsed
from the filename), and read the --suggest title/description/hashtags beside it.
Consistent with CLAUDE.md: single-user, local-only (binds 127.0.0.1), no
accounts, no uploads, no auto-posting -- just a viewer for files you already
made and upload manually.

Stdlib only (no new dependency), with HTTP Range support so scrubbing/seeking
in the <video> player works. The output folder is re-read on every page load,
so re-running the pipeline and refreshing shows the new clips.

Standalone:
    venv\\Scripts\\python.exe serve.py
    venv\\Scripts\\python.exe serve.py --output output --port 8000
    venv\\Scripts\\python.exe serve.py --no-open      # don't auto-open a browser
"""

import argparse
import html
import os
import re
import sys
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
# Filenames from the --n pipeline: <stem>_rank01_score71_47.59-87.57.mp4
_RANK_RE = re.compile(r"_rank(\d+)_score(\d+)_([\d.]+)-([\d.]+)\.[^.]+$")
# Filenames from cut.py / --dumb: <stem>_47.59-87.57_cover.mp4 (no rank/score)
_SPAN_RE = re.compile(r"_([\d.]+)-([\d.]+)(?:_(cover|contain))?\.[^.]+$")


def _fmt_size(n: int) -> str:
    mb = n / (1024 * 1024)
    if mb >= 1:
        return f"{mb:.1f} MB"
    return f"{n / 1024:.0f} KB"


def _fmt_dur(seconds: float) -> str:
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    return f"{m}:{s:02d}"


def scan_clips(clips_dir: str) -> list:
    """Return metadata dicts for every video in clips_dir, sorted by name.

    Rank/score/time-range are parsed from the filename when present; a matching
    sibling .txt (from --suggest) is read in as free text.
    """
    clips = []
    try:
        names = sorted(os.listdir(clips_dir))
    except FileNotFoundError:
        return clips

    for name in names:
        path = os.path.join(clips_dir, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() not in VIDEO_EXTS:
            continue

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

        clips.append(info)

    # Group by source stem, then order best-first (rank asc) within a stem.
    def sort_key(c):
        stem = re.split(r"_rank\d+|_[\d.]+-[\d.]+", c["name"])[0]
        return (stem, c["rank"] if c["rank"] is not None else 9999, c["name"])

    clips.sort(key=sort_key)
    return clips


def render_page(clips_dir: str) -> bytes:
    clips = scan_clips(clips_dir)
    abs_dir = os.path.abspath(clips_dir)

    cards = []
    for c in clips:
        src = "/clips/" + urllib.parse.quote(c["name"])
        badges = []
        if c["rank"] is not None:
            badges.append(f'<span class="badge rank">#{c["rank"]}</span>')
        if c["score"] is not None:
            badges.append(f'<span class="badge score">score {c["score"]}</span>')
        badge_html = "".join(badges)

        meta_bits = []
        if c["start"] is not None and c["end"] is not None:
            dur = c["end"] - c["start"]
            meta_bits.append(f'{_fmt_dur(dur)} &middot; {c["start"]:g}–{c["end"]:g}s')
        meta_bits.append(_fmt_size(c["size"]))
        meta_html = " &middot; ".join(meta_bits)

        suggest_html = ""
        if c["suggest"]:
            suggest_html = (
                '<details class="suggest"><summary>suggested caption</summary>'
                f'<pre>{html.escape(c["suggest"])}</pre></details>'
            )

        cards.append(f"""
        <div class="card">
          <video controls preload="metadata" playsinline src="{src}"></video>
          <div class="info">
            <div class="badges">{badge_html}</div>
            <div class="fname" title="{html.escape(c['name'])}">{html.escape(c['name'])}</div>
            <div class="meta">{meta_html}</div>
            {suggest_html}
            <a class="dl" href="{src}" download>download</a>
          </div>
        </div>""")

    if clips:
        body = f'<div class="grid">{"".join(cards)}</div>'
        count = f"{len(clips)} clip{'s' if len(clips) != 1 else ''}"
    else:
        body = (
            '<div class="empty"><p>No clips in this folder yet.</p>'
            '<p>Generate some, then refresh:</p>'
            '<pre>venv\\Scripts\\python.exe main.py --input path\\to\\video.mp4 --n 5</pre>'
            '</div>'
        )
        count = "0 clips"

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local Clipper — {count}</title>
<style>
  :root {{
    --bg:#0f1115; --card:#1a1d24; --line:#2a2e37; --text:#e8eaed;
    --dim:#9aa0aa; --accent:#7c9cff; --score:#3ecf8e;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
    font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif; }}
  header {{ position:sticky; top:0; z-index:2; padding:16px 24px;
    background:rgba(15,17,21,.85); backdrop-filter:blur(8px);
    border-bottom:1px solid var(--line); display:flex; align-items:baseline; gap:14px; }}
  header h1 {{ font-size:16px; margin:0; font-weight:600; }}
  header .count {{ color:var(--dim); }}
  header .path {{ color:var(--dim); font-size:12px; margin-left:auto;
    font-family:ui-monospace,Consolas,monospace; }}
  .grid {{ display:grid; gap:20px; padding:24px;
    grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    overflow:hidden; display:flex; flex-direction:column; }}
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
    color:var(--text); font-family:inherit; }}
  .dl {{ font-size:12px; color:var(--accent); text-decoration:none; }}
  .dl:hover {{ text-decoration:underline; }}
  .empty {{ padding:64px 24px; text-align:center; color:var(--dim); }}
  .empty pre, .suggest pre {{ background:#0b0d11; border:1px solid var(--line);
    border-radius:8px; padding:10px; overflow:auto; }}
  .empty pre {{ display:inline-block; text-align:left; }}
</style>
</head>
<body>
<header>
  <h1>Local Clipper</h1>
  <span class="count">{count}</span>
  <span class="path">{html.escape(abs_dir)}</span>
</header>
{body}
</body>
</html>"""
    return page.encode("utf-8")


class ClipHandler(BaseHTTPRequestHandler):
    # Set on the server instance in serve().
    clips_dir = "output"

    def log_message(self, fmt, *args):  # quieter console
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)

        if path in ("/", "/index.html"):
            body = render_page(self.server.clips_dir)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        if path.startswith("/clips/"):
            self._serve_clip(path[len("/clips/"):])
            return

        self.send_error(404, "Not found")

    def _serve_clip(self, name: str):
        # Basename only + containment check: never serve outside the clips dir.
        name = os.path.basename(name)
        base = os.path.abspath(self.server.clips_dir)
        full = os.path.abspath(os.path.join(base, name))
        if os.path.commonpath([base, full]) != base or not os.path.isfile(full):
            self.send_error(404, "Not found")
            return

        ext = os.path.splitext(full)[1].lower()
        ctype = {
            ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
            ".webm": "video/webm", ".mkv": "video/x-matroska",
        }.get(ext, "application/octet-stream")

        size = os.path.getsize(full)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if m:
                partial = True
                g1, g2 = m.group(1), m.group(2)
                if g1 == "":  # suffix range: last N bytes
                    length = int(g2)
                    start = max(0, size - length)
                    end = size - 1
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
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser closed the connection (seek/skip) -- fine.


def serve(clips_dir: str, port: int, open_browser: bool):
    if not os.path.isdir(clips_dir):
        print(f"NOTE: output folder '{clips_dir}' doesn't exist yet; "
              f"the page will be empty until you generate clips.", flush=True)

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
    ap = argparse.ArgumentParser(description="Local read-only gallery for reviewing generated clips")
    ap.add_argument("--output", default="output", help="folder of clips to serve (default: output)")
    ap.add_argument("--port", type=int, default=8000, help="localhost port (default: 8000)")
    ap.add_argument("--no-open", dest="open_browser", action="store_false",
                    help="don't auto-open a browser tab")
    args = ap.parse_args()
    serve(args.output, args.port, args.open_browser)


if __name__ == "__main__":
    main()
