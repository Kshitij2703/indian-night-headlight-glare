"""Rebuild the video frames of the Indian Night Headlight Glare dataset from the source YouTube videos.

The images are not redistributed (third-party footage). This script re-creates them from the videos listed in
sources.csv. It does not rely on a fixed timestamp: for every frame in manifest.csv it scans a window around the
recorded second and keeps the video frame whose perceptual hash (dHash) is closest to the hash of the original
image, so it still works if YouTube re-encodes a video or shifts it slightly.

Requirements: Python 3.9+, numpy, pillow, ffmpeg on PATH (and yt-dlp if you use --download).

Usage:
  python rebuild_frames.py --videos videos/ --out rebuilt/            # videos already downloaded as video-<n>.mp4
  python rebuild_frames.py --videos videos/ --out rebuilt/ --download  # download them first with yt-dlp
  python rebuild_frames.py --videos videos/ --out rebuilt/ --only v6   # one source only

Downloading is your responsibility: check that your use complies with YouTube's terms and with copyright law
in your country. The frames are for non-commercial research only and must not be redistributed.
"""
import argparse, csv, os, shutil, subprocess, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1280, 720            # size of the dataset images
GRID_FPS = 20               # search grid: 0.05 s
SEARCH = (-1.2, 0.2)        # seconds around the recorded second to search
MAX_HAMMING = 12            # above this the match is reported as uncertain


def dhash(img):
    """64-bit difference hash; the same function produced the hashes in manifest.csv"""
    g = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def download(video, url, folder):
    path = os.path.join(folder, f"{video}.mp4")
    if os.path.exists(path):
        return path
    if shutil.which("yt-dlp") is None:
        sys.exit("yt-dlp not found: install it (e.g. pip install yt-dlp) or download the videos yourself")
    subprocess.run(["yt-dlp", "-f", "bv*[height<=1080]+ba/b[height<=1080]/bv*[height<=1080]", "--merge-output-format", "mp4",
                    "-o", path, url], check=True)
    return path


def decode(video_path, w, h, last_k):
    """yield (grid index, RGB bytes) on a 0.05 s grid at w x h"""
    cmd = ["ffmpeg", "-v", "error", "-i", video_path, "-vf", f"fps={GRID_FPS},scale={w}:{h}:flags=area",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 3 * 4)
    k, nbytes = 0, w * h * 3
    try:
        while k <= last_k:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield k, buf
            k += 1
    finally:
        proc.stdout.close(); proc.kill(); proc.wait()


def rebuild(video_path, rows, out_dir):
    """Pass 1 (320x180): hash every candidate frame in each row's search window. Among candidates within 2 bits of
    the best match, take the one closest to the video's median offset (near-still scenes have many near-identical
    frames). Pass 2 (1280x720): save the chosen frames."""
    win = {}                                             # grid index -> row indices whose window covers it
    for i, r in enumerate(rows):
        n = int(r["second"])
        for k in range(int(round((n + SEARCH[0]) * GRID_FPS)), int(round((n + SEARCH[1]) * GRID_FPS)) + 1):
            if k >= 0:
                win.setdefault(k, []).append(i)
    targets = [int(r["dhash"], 16) for r in rows]
    cands = [[] for _ in rows]                           # per row: (hamming, grid index)
    for k, buf in decode(video_path, 320, 180, max(win)):
        if k in win:
            h = dhash(Image.frombuffer("RGB", (320, 180), buf, "raw", "RGB", 0, 1))
            for i in win[k]:
                cands[i].append((hamming(h, targets[i]), k))
    off = [min(c)[1] / GRID_FPS - int(r["second"]) for c, r in zip(cands, rows) if c]
    med = float(np.median(off)) if off else 0.0
    chosen = {}
    for i, (c, r) in enumerate(zip(cands, rows)):
        if not c:
            continue
        dmin = min(d for d, _ in c)
        near = [(abs(k / GRID_FPS - int(r["second"]) - med), d, k) for d, k in c if d <= dmin + 2]
        _, d, k = min(near)
        chosen.setdefault(k, []).append((i, d))
    report = {}
    for k, buf in decode(video_path, W, H, max(chosen) if chosen else -1):
        if k in chosen:
            img = Image.frombuffer("RGB", (W, H), buf, "raw", "RGB", 0, 1)
            for i, d in chosen[k]:
                r = rows[i]
                d_out = os.path.join(out_dir, "images", r["split"]); os.makedirs(d_out, exist_ok=True)
                img.save(os.path.join(d_out, r["image"]), quality=95)
                report[i] = {"image": r["image"], "status": "ok" if d <= MAX_HAMMING else "uncertain match",
                             "hamming": d, "offset_s": round(k / GRID_FPS - int(r["second"]), 2)}
    return [report.get(i, {"image": r["image"], "status": "not found (video too short?)", "hamming": "", "offset_s": ""})
            for i, r in enumerate(rows)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="folder with video-<n>.mp4 (created if --download)")
    ap.add_argument("--out", required=True, help="output dataset folder (images/, labels/, data.yaml)")
    ap.add_argument("--download", action="store_true", help="download missing videos with yt-dlp")
    ap.add_argument("--only", nargs="*", help="sources to rebuild, e.g. v1 v6 (default: all video sources)")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(HERE, "manifest.csv"), encoding="utf-8")))
    sources = {r["video"]: r for r in csv.DictReader(open(os.path.join(HERE, "sources.csv"), encoding="utf-8"))}
    os.makedirs(a.videos, exist_ok=True); os.makedirs(a.out, exist_ok=True)
    shutil.copytree(os.path.join(HERE, "labels"), os.path.join(a.out, "labels"), dirs_exist_ok=True)
    shutil.copy2(os.path.join(HERE, "data.yaml"), a.out)

    report = []
    for video in sorted({r["video"] for r in rows if r["rebuildable"] == "yes"}):
        vrows = [r for r in rows if r["video"] == video]
        if a.only and vrows[0]["source"] not in a.only:
            continue
        path = os.path.join(a.videos, f"{video}.mp4")
        if a.download:
            path = download(video, sources[video]["youtube_url"], a.videos)
        if not os.path.exists(path):
            print(f"{video}: {path} missing - skipped ({len(vrows)} frames). Use --download or place the file there.")
            continue
        print(f"{video}: rebuilding {len(vrows)} frames ...", flush=True)
        rep = rebuild(path, vrows, a.out)
        ok = sum(x["status"] == "ok" for x in rep)
        print(f"{video}: {ok}/{len(rep)} frames matched (hamming <= {MAX_HAMMING})", flush=True)
        report += rep
    with open(os.path.join(a.out, "rebuild_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "status", "hamming", "offset_s"]); w.writeheader(); w.writerows(report)
    skipped = sum(r["rebuildable"] != "yes" for r in rows)
    print(f"done: {len(report)} video frames written to {a.out}/images; {skipped} dashcam (ar_) images are not "
          f"rebuildable (labels only, see README). Report: rebuild_report.csv")


if __name__ == "__main__":
    main()
