"""Stage 2B: temporal context from the re-downloaded videos.

For every v1_/v5_/v6_ dataset frame N (video time N - 0.6 s, measured in align_videos.py) save the frames 0.1 s and 0.2 s
earlier, and cut twelve 10 s test clips at 10 fps from test-split time blocks with the most glare.
One ffmpeg decode pass per video (fps=10 grid, 960x540); output frame k is video time k / 10 s.
"""
import csv, glob, json, os, random, subprocess, sys, time
import numpy as np, cv2

ROOT = os.environ.get("GLARE_ROOT", os.path.expanduser("~/Desktop/vehicle-glare"))
DS = os.path.join(ROOT, "dataset", "glare_dataset_final")
OUT = os.path.join(ROOT, "dataset", "temporal_context")
W, H = 960, 540
# decode grid (fps) and offset per video: dataset frame N = video time N + offset. v6 sits between 0.1 s steps
# (first pass: 17/40 best at -0.6, 23/40 at -0.5), so it is decoded on a 0.05 s grid at -0.55 s.
GRID = {"v1": (10, -0.6), "v5": (10, -0.6), "v6": (20, -0.55)}
DT = 0.1               # spacing of the history frames and of the clip frames (s)
ONLY = set(sys.argv[1:]) or set(GRID)   # optional: re-run only some videos, e.g. "v6"
VIDEOS = {"v1": "video-1", "v5": "video-5", "v6": "video-6"}
CLIPS_PER_VIDEO = {"v6": 6, "v5": 4, "v1": 2}
CLIP_LEN = 10          # seconds
CHECK_PER_VIDEO = 40

def has_box(img):
    p = img.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    return os.path.exists(p) and any(l.strip() for l in open(p))

# dataset frames per video: second -> (split, path, glare)
frames = {p: {} for p in VIDEOS}
for split in ["train", "val", "test"]:
    for f in glob.glob(f"{DS}/images/{split}/v*_*.jpg"):
        pre, num = os.path.basename(f)[:-4].split("_")
        if pre in VIDEOS:
            frames[pre][int(num)] = (split, f, has_box(f))

def kidx(t, fps):
    return int(round(t * fps))

# ---- choose test clips: 10 s windows fully inside test blocks (no train/val frame within 2 s), most glare frames
clips = []
for pre, quota in CLIPS_PER_VIDEO.items():
    secs = frames[pre]
    other = sorted(s for s, v in secs.items() if v[0] != "test")
    cands = []
    for s in sorted(s for s, v in secs.items() if v[0] == "test"):
        win = [t for t in range(s, s + CLIP_LEN) if t in secs]
        if any(s - 2 <= o <= s + CLIP_LEN + 1 for o in other) or len(win) < 7:
            continue
        cands.append((sum(secs[t][2] for t in win), len(win), s))
    cands.sort(key=lambda c: (-c[0], -c[1], c[2]))
    chosen = []
    for g, n, s in cands:
        if all(abs(s - c) >= CLIP_LEN + 5 for c in chosen):
            chosen.append(s)
            clips.append({"clip": f"clip{len(clips):02d}_{pre}_s{s:05d}", "video": VIDEOS[pre], "prefix": pre, "start_s": s,
                          "dataset_frames_in_clip": n, "glare_frames_in_clip": g})
        if len(chosen) == quota:
            break

for c in clips: print("clip", c, flush=True)

# ---- targets per video: frame index k -> list of (kind, payload)
os.makedirs(OUT, exist_ok=True)
random.seed(0)
logp = os.path.join(OUT, "extraction_check.json")
log = json.load(open(logp)) if os.path.exists(logp) and ONLY != set(GRID) else {"videos": {}}
log.update({"history_dt_s": DT, "size": [W, H], "grid_fps_and_offset": {VIDEOS[p]: list(v) for p, v in GRID.items()}})
log.pop("offset_s", None); log.pop("fps_grid", None)
t_start = time.time()
for pre, vname in VIDEOS.items():
    if pre not in ONLY:
        continue
    FPS, OFFSET = GRID[pre]; STEP = int(round(DT * FPS))
    targets = {}
    for s, (split, path, _) in frames[pre].items():
        k0 = kidx(s + OFFSET, FPS)
        d = os.path.join(OUT, "history", split); os.makedirs(d, exist_ok=True)
        base = os.path.basename(path)[:-4]
        targets.setdefault(k0, []).append(("save", os.path.join(d, f"{base}_t0.jpg")))   # current frame, same source as history
        targets.setdefault(k0 - STEP, []).append(("save", os.path.join(d, f"{base}_m1.jpg")))
        targets.setdefault(k0 - 2 * STEP, []).append(("save", os.path.join(d, f"{base}_m2.jpg")))
    checks = random.sample(sorted(frames[pre]), min(CHECK_PER_VIDEO, len(frames[pre])))
    ref = {}
    for s in checks:
        im = cv2.resize(cv2.imread(frames[pre][s][1], cv2.IMREAD_GRAYSCALE), (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
        ref[s] = (im, {})
        for dk in range(-3, 4):
            targets.setdefault(kidx(s + OFFSET, FPS) + dk, []).append(("check", (s, dk)))
    for c in [c for c in clips if c["prefix"] == pre]:
        d = os.path.join(OUT, "clips", c["clip"]); os.makedirs(d, exist_ok=True)
        k0 = kidx(c["start_s"] + OFFSET, FPS)
        for j in range(int(CLIP_LEN / DT)):
            targets.setdefault(k0 + j * STEP, []).append(("save", os.path.join(d, f"f{j:03d}.jpg")))
    last = max(targets)

    cmd = ["ffmpeg", "-v", "error", "-i", os.path.join(ROOT, "videos", f"{vname}.mp4"),
           "-vf", f"fps={FPS},scale={W}:{H}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    k, saved, nbytes = 0, 0, W * H * 3
    while k <= last:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        if k in targets:
            img = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            for kind, payload in targets[k]:
                if kind == "save":
                    cv2.imwrite(payload, img, [cv2.IMWRITE_JPEG_QUALITY, 92]); saved += 1
                else:
                    s, dk = payload
                    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
                    ref[s][1][dk] = float(np.abs(g - ref[s][0]).mean())
        if k % 6000 == 0:
            print(f"{vname}: frame {k}/{last}  saved {saved}  ({time.time() - t_start:.0f} s)", flush=True)
        k += 1
    proc.stdout.close(); proc.kill(); proc.wait()

    best = [min(errs, key=errs.get) for _, errs in ref.values() if len(errs) == 7]
    log["videos"][vname] = {"grid_fps": FPS, "offset_s": OFFSET, "check_step_s": round(1 / FPS, 3), "dataset_frames": len(frames[pre]), "images_saved": saved, "decoded_frames": k,
                            "check_frames": len(best), "best_offset_steps_counts": {str(d): best.count(d) for d in sorted(set(best))},
                            "median_err_at_offset": round(float(np.median([e[0] for _, e in ref.values() if 0 in e])), 2)}
    print(f"{vname}: done, saved {saved} images, alignment check {log['videos'][vname]['best_offset_steps_counts']}", flush=True)

with open(os.path.join(OUT, "clips.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(clips[0].keys())); w.writeheader(); w.writerows(clips)
json.dump(log, open(logp, "w"), indent=1)
print(json.dumps(log, indent=1)); print(f"total {time.time() - t_start:.0f} s")
