"""Figures for the Phase 1 paper (IEEE two-column: one column = 3.39 in)."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = os.environ.get("GLARE_ROOT", os.path.expanduser("~/Desktop/vehicle-glare"))
DS = f"{ROOT}/dataset/glare_dataset_final"
AN = f"{ROOT}/paper/analysis"
FIG = f"{ROOT}/paper/figures"; os.makedirs(FIG, exist_ok=True)
MODELS = ["yolov8s", "yolo11n", "yolo11s", "yolo11m", "rtdetr_l"]
LABEL = {"yolov8s": "YOLOv8s", "yolo11n": "YOLO11n", "yolo11s": "YOLO11s", "yolo11m": "YOLO11m", "rtdetr_l": "RT-DETR-l"}
COLOR = {"yolov8s": "#2a78d6", "yolo11n": "#eb6834", "yolo11s": "#1baf7a", "yolo11m": "#eda100", "rtdetr_l": "#e87ba4"}
STYLE = {"yolov8s": (0, (4, 2)), "yolo11n": (0, (1, 1.5)), "yolo11s": "-", "yolo11m": (0, (6, 2, 1, 2)), "rtdetr_l": (0, (3, 1, 1, 1, 1, 1))}
MARK = {"yolov8s": "s", "yolo11n": "^", "yolo11s": "o", "yolo11m": "D", "rtdetr_l": "v"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
COL_W = 3.39
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 7,
                     "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.3, "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 400, "lines.linewidth": 1.3})
R = json.load(open(f"{AN}/results.json"))


def save(fig, name):
    fig.savefig(f"{FIG}/{name}", bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig); print("wrote", name)


# ---- Fig. 1: examples (test images, ground-truth boxes)
def labels(path):
    lab = path.replace("/images/", "/labels/")[:-4] + ".txt"
    return [list(map(float, l.split()))[1:] for l in open(lab) if l.strip()] if os.path.exists(lab) else []

rng = np.random.default_rng(4)
tiles = []
for src, cap in [("ar", "(a) dashcam clip"), ("v6", "(b) highway, windshield view"), ("v5", "(c) motorcycle view"), ("v1", "(d) driver-seat view")]:
    files = [f for f in sorted(glob.glob(f"{DS}/images/test/{src}_*.jpg")) if 2 <= len(labels(f)) <= 4]
    f = files[int(rng.integers(len(files)))]
    im = Image.open(f).convert("RGB").resize((800, 450), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for x, y, w, h in labels(f):
        d.rectangle([(x - w / 2) * 800, (y - h / 2) * 450, (x + w / 2) * 800, (y + h / 2) * 450], outline=(255, 214, 0), width=3)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    d.rectangle([0, 404, 800, 450], fill=(0, 0, 0))
    d.text((10, 410), cap, fill=(255, 255, 255), font=font)
    tiles.append(im); print("example:", os.path.basename(f), len(labels(f)), "boxes")
canvas = Image.new("RGB", (1608, 908), "white")
for i, t in enumerate(tiles):
    canvas.paste(t, ((i % 2) * 808, (i // 2) * 458))
canvas.save(f"{FIG}/fig1_examples.png"); print("wrote fig1_examples.png")

# ---- Fig. 2: leakage-free split on the video-6 timeline
pos = {}
for split in ["train", "val", "test"]:
    for f in glob.glob(f"{DS}/images/{split}/v6_*.jpg"):
        pos[int(os.path.basename(f)[3:8])] = split
SPC = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
fig, ax = plt.subplots(figsize=(COL_W, 0.85))
lo, hi = 0, 1500
for sp in ["train", "val", "test"]:
    xs = [(s, 1) for s, v in sorted(pos.items()) if v == sp and lo <= s < hi]
    ax.broken_barh(xs, (0, 1), facecolors=SPC[sp], label=sp)
ax.set_xlim(lo, hi); ax.set_ylim(0, 1); ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Video time (s), video-6, first 1,500 s")
ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.62), handlelength=1.2)
save(fig, "fig2_split_timeline.png")

# ---- Fig. 3: detection vs false-alarm curves (test)
C = json.load(open(f"{AN}/det_fa_curves.json"))
fig, ax = plt.subplots(figsize=(COL_W, 2.35))
for m in MODELS:
    fa = np.array(C[m]["fa"]) * 100; det = np.array(C[m]["det"]) * 100
    ax.step(fa, det, where="post", color=COLOR[m], linestyle=STYLE[m], label=LABEL[m])
    a = C[m]["at025"]
    ax.plot(a[0] * 100, a[1] * 100, MARK[m], color=COLOR[m], ms=4.2, mec="white", mew=0.6, zorder=5)
ax.set_xlim(0, 30); ax.set_ylim(78, 100); ax.set_yticks(range(80, 101, 5))
ax.set_xlabel("False alarms: no-glare test images with a box (%)")
ax.set_ylabel("Glare test images detected (%)")
ax.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right", ncol=1)
save(fig, "fig3_det_vs_fa.png")

# ---- Fig. 4: recall by glare size
S = R["size"]
fig, ax = plt.subplots(figsize=(COL_W, 1.9))
x = np.arange(len(S["bins"]))
for m in MODELS:
    ax.plot(x, np.array(S["recall_at_matched_threshold"][m]) * 100, marker=MARK[m], color=COLOR[m], linestyle=STYLE[m], ms=3.6, label=LABEL[m])
ax.set_xticks(x); ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(S["bins"], S["gt_boxes_per_bin"])])
ax.set_ylabel("Box recall (%)"); ax.set_xlabel("Glare box size, sqrt(area) at 1280x720")
ax.grid(True, axis="y", color=GRID, linewidth=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc="lower right", columnspacing=0.8, handlelength=1.6)
save(fig, "fig4_recall_by_size.png")

# ---- Fig. 5: neighbour similarity (leakage)
hn, hr = np.load(f"{AN}/hamming_neighbour.npy"), np.load(f"{AN}/hamming_random.npy")
fig, ax = plt.subplots(figsize=(COL_W, 1.55))
bins = np.arange(0, 41, 1)
ax.hist(hr, bins=bins, color="#8a8984", alpha=0.55, density=True, label="frames >= 60 s apart")
ax.hist(hn, bins=bins, color="#2a78d6", alpha=0.85, density=True, label="consecutive frames (1 s apart)")
ax.axvline(10.5, color=INK2, linewidth=0.7, linestyle=(0, (2, 2)))
ax.text(11.2, ax.get_ylim()[1] * 0.85, "near-duplicate\nthreshold", fontsize=6, color=INK2, va="top")
ax.set_xlabel("Perceptual-hash distance (bits of 64)"); ax.set_ylabel("Share of pairs")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.2)
save(fig, "fig5_neighbour_similarity.png")
