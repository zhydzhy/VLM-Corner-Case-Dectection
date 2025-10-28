#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi 5 multi-frame capture script (VLM preprocessing).

Default behavior:
- Capture 5 frames using rpicam-still, 1.5s apart
- Save to ./captures/<timestamp>/frames/frame_01.jpg ... frame_05.jpg
- Write manifest.json and update ./captures/latest symlink
- No raw_* images, only frame_XX.jpg
"""

import os
import sys
import json
import time
import shutil
import signal
import subprocess
import platform
from datetime import datetime
from pathlib import Path

# ============================================================
# === Default Configurations (easy to change for debugging) ===
# ============================================================

DEFAULT_ROOT = str(Path(__file__).resolve().parent / "captures")
DEFAULT_FRAMES = 5               # number of stills to capture
DEFAULT_INTERVAL = 1.5           # seconds between captures
DEFAULT_WIDTH = 1280             # image width
DEFAULT_HEIGHT = 720             # image height
DEFAULT_QUALITY = 95             # JPEG quality (1-100)
DEFAULT_TIMEOUT_PER_SHOT = 1.0   # per-shot timeout for rpicam-still (seconds)
DEFAULT_KEEP_VIDEO = False       # stitch frames into capture.mp4
DEFAULT_FPS_FOR_VIDEO = 8        # fps for ffmpeg video output

# ============================================================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def mk_latest_symlink(latest_target: str, link_path: str):
    """
    For Linux/macOS: create captures/latest -> captures/<timestamp>
    For Windows we skip.
    """
    if platform.system().lower().startswith("win"):
        return
    try:
        if os.path.islink(link_path) or os.path.exists(link_path):
            os.remove(link_path)
    except Exception:
        pass
    try:
        os.symlink(latest_target, link_path)
    except Exception as e:
        print(f"[WARN] Failed to create latest symlink: {e}")


def capture_single_frame(output_path: str,
                         width: int,
                         height: int,
                         quality: int,
                         timeout_ms: int = 1000):
    """
    Capture one still frame using rpicam-still.
    """
    cmd = [
        "rpicam-still",
        "--timeout", str(timeout_ms),
        "--output", output_path,
        "--width", str(width),
        "--height", str(height),
        "--quality", str(quality),
        "--nopreview",
        "--immediate",
    ]
    print(f"[INFO] Capturing: {output_path}")
    try:
        subprocess.run(cmd, check=True)
        if os.path.exists(output_path):
            return True
        else:
            print("[WARN] rpicam-still ran but no output found:", output_path)
            return False
    except FileNotFoundError:
        print("[ERR] rpicam-still not found. Make sure it is installed.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[ERR] rpicam-still failed: {e}")
        return False


def main():
    # Use default configuration (no CLI arguments needed)
    root = DEFAULT_ROOT
    frames = DEFAULT_FRAMES
    interval = DEFAULT_INTERVAL
    width = DEFAULT_WIDTH
    height = DEFAULT_HEIGHT
    quality = DEFAULT_QUALITY
    timeout_per_shot = DEFAULT_TIMEOUT_PER_SHOT
    keep_video = DEFAULT_KEEP_VIDEO
    fps_for_video = DEFAULT_FPS_FOR_VIDEO

    # Timestamp-based directory
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root, stamp)
    frames_dir = os.path.join(out_dir, "frames")
    ensure_dir(frames_dir)
    ensure_dir(root)

    print(f"\n[INFO] Capture Session Start")
    print(f"       Frames: {frames}")
    print(f"       Interval: {interval:.2f}s")
    print(f"       Resolution: {width}x{height}")
    print(f"       Output Directory: {frames_dir}\n")

    saved_frames = []

    for i in range(1, frames + 1):
        frame_path = os.path.join(frames_dir, f"frame_{i:02d}.jpg")
        ok = capture_single_frame(
            output_path=frame_path,
            width=width,
            height=height,
            quality=quality,
            timeout_ms=int(timeout_per_shot * 1000),
        )
        if not ok:
            print(f"[WARN] Failed to capture frame {i}, stopping early.")
            break

        saved_frames.append(frame_path)
        print(f"[INFO] Captured frame {i}/{frames}: {frame_path}")

        if i < frames:
            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\n[INFO] Interrupted by user. Stopping capture.")
                break

    # Optional: create timelapse video
    video_path = None
    if keep_video and len(saved_frames) > 1:
        video_path = os.path.join(out_dir, "capture.mp4")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-pattern_type", "glob",
            "-i", os.path.join(frames_dir, "frame_*.jpg"),
            "-r", str(fps_for_video),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            video_path
        ]
        print("[INFO] Generating timelapse video...")
        try:
            subprocess.run(ffmpeg_cmd, check=True)
        except FileNotFoundError:
            print("[WARN] ffmpeg not installed, skipping video.")
            video_path = None
        except subprocess.CalledProcessError as e:
            print(f"[WARN] ffmpeg failed: {e}")
            video_path = None

    # Save manifest.json
    manifest = {
        "timestamp_dir": stamp,
        "root": root,
        "backend": "rpicam-still-loop",
        "video_path": video_path,
        "resolution": {"width": width, "height": height},
        "frames_dir": frames_dir,
        "frames_saved": [
            {"index": idx + 1, "path": p}
            for idx, p in enumerate(saved_frames)
        ],
        "total_frames_captured": len(saved_frames),
        "interval_sec": interval,
        "timeout_per_shot_sec": timeout_per_shot,
    }

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Wrote manifest -> {os.path.join(out_dir, 'manifest.json')}")

    # Update latest symlink
    mk_latest_symlink(out_dir, os.path.join(root, "latest"))

    print(f"\n[OK] Capture complete: {out_dir}")
    if len(saved_frames) == 0:
        print("[WARN] No frames captured. Check camera connection.")
    else:
        print("[INFO] Frames ready for inference:")
        for p in saved_frames:
            print("   ", p)


def _sigint_passthrough(sig, frame):
    print("\n[INFO] Ctrl+C received. Finishing current step before exit...")


signal.signal(signal.SIGINT, _sigint_passthrough)

if __name__ == "__main__":
    main()
