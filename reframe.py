"""
Face-tracking auto-reframe for Local Clipper (Phase 6).

When a 16:9 source is cropped to 9:16, a static center-crop lops the speaker off
if they sit left/right of center. This module finds the main speaker with a face
detector (OpenCV's YuNet), builds a SMOOTHED crop-center path over time, and
emits an ffmpeg `crop` whose x follows the speaker -- so the vertical frame pans
to keep them in shot. If no (or too few) faces are found it reports that, and
cut.py falls back to the static center-crop.

Perception (face detection) runs locally on the GPU/CPU via OpenCV; nothing here
touches the Claude API. The detector weights live in models/ (see README).

How the pan is applied:
    Detection is sampled at `sample_fps`; the per-sample crop-x is de-jittered
    (moving average + pan-speed limiter) and turned into a single piecewise-
    linear ffmpeg expression x(t). ffmpeg evaluates it per frame, so the crop
    glides between samples instead of stepping. No temp command files, no jitter.

Standalone (produces an actual reframed clip so you can eyeball the tracking):
    venv\\Scripts\\python.exe reframe.py --input media\\2PxLYWjgLys.mp4 \\
        --start 1278.6 --end 1302.36 --output output\\reframe_test.mp4
    # add --dump to print the detection track summary without encoding
"""

import argparse
import os
import sys

import cv2
import numpy as np

MODEL_PATH = os.path.join("models", "face_detection_yunet_2023mar.onnx")

# The crop x(t) is one piecewise-linear ffmpeg expression, one term per
# keyframe. ffmpeg rejects very long/complex option expressions, so we simplify
# the smoothed path (drop near-collinear points) and hard-cap the keyframe count
# well under that ceiling. A smooth pan needs only a handful of keyframes.
MAX_KEYFRAMES = 56
SIMPLIFY_EPS_PX = 2.5   # max crop-x error (px) tolerated when dropping keyframes

# Tunables. Defaults are conservative -- smooth over snappy. Standalone CLI flags
# below expose the ones worth experimenting with.
DEFAULTS = {
    "sample_fps": 6.0,       # how often to run face detection (per second of clip)
    "smooth_seconds": 0.8,   # moving-average window on the crop-center path
    "max_pan_px_s": 320.0,   # cap crop-center speed (source px/s) -> no whip-pans
    "score_threshold": 0.7,  # YuNet confidence to accept a detection
    "min_face_frac": 0.03,   # ignore faces narrower than this fraction of frame width
    "min_valid_frac": 0.12,  # if fewer than this share of samples saw a face -> static
}


def _cfg(overrides: dict = None) -> dict:
    c = dict(DEFAULTS)
    if overrides:
        c.update({k: v for k, v in overrides.items() if v is not None})
    return c


def _make_detector(w: int, h: int, score_threshold: float):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"YuNet model not found at {MODEL_PATH}. Download it once with:\n"
            "  curl -L -o models/face_detection_yunet_2023mar.onnx "
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        )
    det = cv2.FaceDetectorYN.create(MODEL_PATH, "", (w, h),
                                    score_threshold=score_threshold)
    det.setInputSize((w, h))
    return det


def _pick_main_face(faces, prev_cx, frame_w, cfg):
    """Choose the main speaker among detections: biggest face wins, ties broken
    by proximity to the previous center (so we don't flip between two people).
    Returns center-x in pixels, or None."""
    if faces is None or len(faces) == 0:
        return None
    min_w = cfg["min_face_frac"] * frame_w
    best, best_key = None, None
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        if w < min_w:
            continue
        cx = x + w / 2.0
        area = w * h
        # Rank by area, then (small) bonus for staying near the previous center.
        near = 0.0 if prev_cx is None else -abs(cx - prev_cx)
        key = (area, near)
        if best_key is None or key > best_key:
            best_key, best = key, cx
    return best


def detect_track(video_path, start, end, cfg):
    """Sample face detections across [start, end]. Returns
    (times_rel, centers_x, frame_w, frame_h) where centers_x[i] is the main
    face's center-x in source pixels at times_rel[i] (clip-relative), or NaN."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    det = _make_detector(W, H, cfg["score_threshold"])

    sample_dt = 1.0 / cfg["sample_fps"]
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    next_t = start
    prev_cx = None
    times, xs = [], []
    while True:
        if not cap.grab():           # advance without decoding skipped frames
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t > end + 1e-3:
            break
        if t + 1e-6 < next_t:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        _, faces = det.detect(frame)
        cx = _pick_main_face(faces, prev_cx, W, cfg)
        times.append(t - start)
        xs.append(cx if cx is not None else np.nan)
        if cx is not None:
            prev_cx = cx
        next_t += sample_dt
    cap.release()
    return np.asarray(times, dtype=float), np.asarray(xs, dtype=float), W, H


def _fill_nans(xs):
    """Linear-interpolate NaN gaps; leading/trailing clamp to nearest known.
    Returns None if every sample is NaN."""
    good = ~np.isnan(xs)
    if not good.any():
        return None
    idx = np.arange(len(xs))
    out = xs.copy()
    out[~good] = np.interp(idx[~good], idx[good], xs[good])
    return out


def _smooth(a, win):
    if win <= 1 or len(a) < 2:
        return a
    pad = win // 2
    ap = np.pad(a, (pad, pad), mode="edge")
    kernel = np.ones(win) / win
    return np.convolve(ap, kernel, mode="valid")[:len(a)]


def _limit_speed(x, times, max_px_s):
    """Forward pass clamping |dx/dt| so the crop never whip-pans."""
    out = x.copy()
    for i in range(1, len(out)):
        dt = max(1e-3, times[i] - times[i - 1])
        max_step = max_px_s * dt
        delta = out[i] - out[i - 1]
        if delta > max_step:
            out[i] = out[i - 1] + max_step
        elif delta < -max_step:
            out[i] = out[i - 1] - max_step
    return out


def _simplify(times, x, eps):
    """Ramer-Douglas-Peucker on the (t, x) path using vertical (x) error, so a
    long run of collinear samples collapses to its two endpoints. Iterative to
    avoid recursion limits. Returns index array to keep."""
    n = len(times)
    if n <= 2:
        return np.arange(n)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        t0, t1, x0, x1 = times[lo], times[hi], x[lo], x[hi]
        span = t1 - t0
        if span <= 1e-9:
            continue
        seg = np.arange(lo + 1, hi)
        line = x0 + (x1 - x0) * (times[seg] - t0) / span
        d = np.abs(x[seg] - line)
        k = int(np.argmax(d))
        if d[k] > eps:
            idx = lo + 1 + k
            keep[idx] = True
            stack.append((lo, idx))
            stack.append((idx, hi))
    return np.flatnonzero(keep)


def _keyframes(times, x, eps, cap):
    """Simplify the path to <= cap keyframes, raising eps if needed."""
    for _ in range(8):
        idx = _simplify(times, x, eps)
        if len(idx) <= cap:
            return times[idx], x[idx]
        eps *= 1.8
    # Backstop: still too many -> uniform subsample keeping first/last.
    sel = np.unique(np.linspace(0, len(times) - 1, cap).round().astype(int))
    return times[sel], x[sel]


def build_crop_path(times, xs, W, H, target_w, target_h, cfg):
    """Turn raw face centers into a smoothed crop-x path.

    Returns dict(cw, ch, maxx, times, x) or None to signal 'use static crop'
    (source not wider than target, or too few faces detected)."""
    tgt_aspect = target_w / target_h
    src_aspect = W / H
    if src_aspect <= tgt_aspect + 1e-6:
        return None  # already <= 9:16 wide; horizontal tracking is meaningless

    valid_frac = float(np.mean(~np.isnan(xs))) if len(xs) else 0.0
    if valid_frac < cfg["min_valid_frac"]:
        return None

    filled = _fill_nans(xs)
    if filled is None:
        return None

    cw = int(round(H * tgt_aspect))       # vertical-strip width, full height
    ch = H
    maxx = max(0, W - cw)

    cropx = np.clip(filled - cw / 2.0, 0, maxx)
    win = max(1, int(round(cfg["smooth_seconds"] * cfg["sample_fps"])))
    cropx = _smooth(cropx, win)
    cropx = _limit_speed(cropx, times, cfg["max_pan_px_s"])
    cropx = np.clip(cropx, 0, maxx)
    # Collapse the dense path to a handful of keyframes for a compact ffmpeg expr.
    kf_t, kf_x = _keyframes(times, cropx, SIMPLIFY_EPS_PX, MAX_KEYFRAMES)
    return {"cw": cw, "ch": ch, "maxx": maxx, "times": kf_t, "x": kf_x,
            "valid_frac": valid_frac}


def crop_x_expr(times, x, maxx):
    """Piecewise-linear x(t) as a cumulative-ramp ffmpeg expression:
        x0 + s0*clip(t-t0,0,d0) + s1*clip(t-t1,0,d1) + ...
    (flat before t0 and after the last sample). Wrapped in a final clip for
    safety. Contains commas, so cut.py single-quotes it in the filtergraph."""
    terms = [f"{x[0]:.2f}"]
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        if dt <= 1e-6:
            continue
        slope = (x[i + 1] - x[i]) / dt
        if abs(slope) < 1e-4:
            continue
        terms.append(f"{slope:.4f}*clip(t-{times[i]:.4f},0,{dt:.4f})")
    expr = "+".join(terms).replace("+-", "-")
    return f"clip({expr},0,{maxx})"


def build_reframe_vf(video_path, start, end, target_w, target_h,
                     overrides: dict = None):
    """Full base video-filter string for a face-tracking 9:16 reframe, or None
    if reframe isn't applicable (no/too-few faces, or non-widescreen source) --
    the caller then uses its static crop. Captions (if any) are appended by the
    caller AFTER this, so they land on the final scaled frame."""
    cfg = _cfg(overrides)
    times, xs, W, H = detect_track(video_path, start, end, cfg)
    plan = build_crop_path(times, xs, W, H, target_w, target_h, cfg)
    if plan is None:
        return None
    expr = crop_x_expr(plan["times"], plan["x"], plan["maxx"])
    vf = (f"crop=w={plan['cw']}:h={plan['ch']}:x='{expr}':y=0,"
          f"scale={target_w}:{target_h},setsar=1")
    return vf


def _summary(video_path, start, end, cfg):
    times, xs, W, H = detect_track(video_path, start, end, cfg)
    valid = ~np.isnan(xs)
    frac = float(np.mean(valid)) if len(xs) else 0.0
    print(f"source {W}x{H}, {len(times)} samples over {end-start:g}s "
          f"@ {cfg['sample_fps']:g}fps")
    print(f"faces found in {valid.sum()}/{len(xs)} samples ({frac*100:.0f}%)")
    if valid.any():
        fx = xs[valid]
        print(f"face center-x range: {fx.min():.0f}..{fx.max():.0f}px "
              f"(frame width {W})")
    plan = build_crop_path(times, xs, W, H, 1080, 1920, cfg)
    if plan is None:
        print("-> would FALL BACK to static center crop")
    else:
        print(f"-> tracking crop {plan['cw']}x{plan['ch']}, "
              f"x range {plan['x'].min():.0f}..{plan['x'].max():.0f}"
              f" (of 0..{plan['maxx']})")


def main():
    ap = argparse.ArgumentParser(description="Face-tracking 9:16 auto-reframe (Phase 6)")
    ap.add_argument("--input", required=True, help="local video path or a video URL")
    ap.add_argument("--start", type=float, required=True, help="clip start (s)")
    ap.add_argument("--end", type=float, required=True, help="clip end (s)")
    ap.add_argument("--output", default=None, help="output clip path (default under /output)")
    ap.add_argument("--dump", action="store_true",
                    help="print the detection-track summary and exit (no encode)")
    ap.add_argument("--sample-fps", type=float, default=None, help="detection samples/sec")
    ap.add_argument("--smooth-seconds", type=float, default=None, help="path smoothing window (s)")
    ap.add_argument("--max-pan-px-s", type=float, default=None, help="max crop pan speed (px/s)")
    args = ap.parse_args()

    overrides = {
        "sample_fps": args.sample_fps,
        "smooth_seconds": args.smooth_seconds,
        "max_pan_px_s": args.max_pan_px_s,
    }
    cfg = _cfg(overrides)

    from ingest import resolve_input
    local = resolve_input(args.input)

    if args.dump:
        _summary(local, args.start, args.end, cfg)
        return

    # Encode a real reframed clip via cut.py (imported lazily to avoid a cycle).
    from cut import cut_segment
    try:
        out = cut_segment(local, args.start, args.end, reframe=True,
                          reframe_cfg=overrides, out_path=args.output)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    size_mb = os.path.getsize(out) / 1024**2
    print(f"Wrote {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
