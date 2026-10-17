"""No-training analyses for the Phase 1 paper (uses cached predictions from run_inference.py).

1. Re-computed test metrics (Ultralytics-style matching and 101-point AP) with 95% bootstrap confidence intervals,
   and paired bootstrap differences against YOLO11s.
2. Validation-calibrated matched false-alarm operating points for all detectors, and test detection-vs-false-alarm curves.
3. Recall by glare size.
4. Brightness-threshold baseline (no learning), tuned on validation.
5. Leakage analysis: neighbour similarity, random-split simulation, YOLO11s train/val/test gap.
6. Per-source results and the video-only test subset (reproducible from the public release).
"""
import csv, glob, json, os
import numpy as np, cv2

ROOT = os.environ.get("GLARE_ROOT", os.path.expanduser("~/Desktop/vehicle-glare"))
DS = f"{ROOT}/dataset/glare_dataset_final"
AN = f"{ROOT}/paper/analysis"
PRED = f"{AN}/preds"
REL = f"{ROOT}/release/indian-night-headlight-glare-v1"
MODELS = ["yolov8s", "yolo11n", "yolo11s", "yolo11m", "rtdetr_l"]
LABEL = {"yolov8s": "YOLOv8s", "yolo11n": "YOLO11n", "yolo11s": "YOLO11s", "yolo11m": "YOLO11m", "rtdetr_l": "RT-DETR-l",
         "bright": "Brightness threshold"}
IOUS = np.linspace(0.5, 0.95, 10)
X101 = np.linspace(0, 1, 101)
B = 1000
CONF = 0.25
SIZE_BINS = [(0, 16), (16, 32), (32, 64), (64, 1e9)]
SIZE_NAMES = ["<16 px", "16-32 px", "32-64 px", ">=64 px"]
rng = np.random.default_rng(0)
R = {}                                                     # everything that goes into results.json


# ------------------------------------------------------------------ data
def load_gt(split):
    gt = {}
    for f in sorted(glob.glob(f"{DS}/images/{split}/*.jpg")):
        n = os.path.basename(f)
        lab = f"{DS}/labels/{split}/{n[:-4]}.txt"
        rows = [list(map(float, l.split())) for l in open(lab) if l.strip()] if os.path.exists(lab) else []
        gt[n] = np.array([[x - w / 2, y - h / 2, x + w / 2, y + h / 2] for _, x, y, w, h in rows], float).reshape(-1, 4)
    return gt


def box_iou(a, b):
    tl = np.maximum(a[:, None, :2], b[None, :, :2]); br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1); ab = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-12)


def match(gt, pb):
    """Ultralytics match_predictions: TP flag per prediction at each IoU threshold (IoU is scale invariant, so
    normalised coordinates are used directly)"""
    tp = np.zeros((len(pb), len(IOUS)), bool)
    if len(gt) == 0 or len(pb) == 0:
        return tp
    iou = box_iou(gt, pb)
    for i, t in enumerate(IOUS):
        m = np.argwhere(iou >= t)
        if len(m):
            if len(m) > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            tp[m[:, 1], i] = True
    return tp


def prepare(preds, gt, names):
    conf, tp, img, maxconf = [], [], [], np.zeros(len(names))
    for k, n in enumerate(names):
        pb = np.array(preds[n]["boxes"], float).reshape(-1, 4); pc = np.array(preds[n]["conf"], float)
        conf.append(pc); tp.append(match(gt[n], pb)); img.append(np.full(len(pc), k))
        maxconf[k] = pc.max() if len(pc) else 0.0
    conf, tp, img = np.concatenate(conf), np.concatenate(tp), np.concatenate(img)
    o = np.argsort(-conf, kind="stable")
    return {"conf": conf[o], "tp": tp[o], "img": img[o], "maxconf": maxconf}


def ap101(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0])); mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    return np.trapezoid(np.interp(X101, mrec, mpre), X101)


def map_w(d, w_img, ngt):
    """mAP50 and mAP50-95 for a weighted image set (bootstrap multiplicities as weights)"""
    w = w_img[d["img"]].astype(float); keep = w > 0
    t = d["tp"][keep]; w = w[keep, None]
    tpc = (t * w).cumsum(0); fpc = ((~t) * w).cumsum(0)
    n = float((w_img * ngt).sum())
    rec = tpc / (n + 1e-16); prec = tpc / np.maximum(tpc + fpc, 1e-16)
    aps = np.array([ap101(rec[:, j], prec[:, j]) for j in range(len(IOUS))])
    return aps[0], aps.mean()


def rates_w(maxconf, glare, w_img, thr=CONF):
    det = maxconf >= thr
    return ((w_img * (det & glare)).sum() / (w_img * glare).sum(), (w_img * (det & ~glare)).sum() / (w_img * ~glare).sum())


def metrics(d, ngt, glare, w=None, thr=CONF):
    w = np.ones(len(ngt)) if w is None else w
    m50, m5095 = map_w(d, w, ngt); hit, fa = rates_w(d["maxconf"], glare, w, thr)
    return np.array([m50, m5095, hit, fa])


def load_preds(m, split):
    return json.load(open(f"{PRED}/{m}_{split}.json"))


# ------------------------------------------------------------------ 1. test metrics + bootstrap
gt_test, gt_val = load_gt("test"), load_gt("val")
names_t, names_v = sorted(gt_test), sorted(gt_val)
ngt_t = np.array([len(gt_test[n]) for n in names_t]); glare_t = ngt_t > 0
ngt_v = np.array([len(gt_val[n]) for n in names_v]); glare_v = ngt_v > 0
D = {m: prepare(load_preds(m, "test"), gt_test, names_t) for m in MODELS}
DV = {m: prepare(load_preds(m, "val"), gt_val, names_v) for m in MODELS}

official = {m: json.load(open(f"{ROOT}/models/{m}.json")) for m in MODELS}
point = {m: metrics(D[m], ngt_t, glare_t) for m in MODELS}
print("re-computed vs official (test):")
for m in MODELS:
    o = official[m]
    print(f"  {LABEL[m]:10s} mAP50 {point[m][0]:.3f} (official {o['test_mAP50']:.3f})  mAP50-95 {point[m][1]:.3f} "
          f"({o['test_mAP50_95']:.3f})  detected {point[m][2]:.3f} ({o['img_hit_rate']:.3f})  FA {point[m][3]:.3f} ({o['img_false_pos_rate']:.3f})")

N = len(names_t)
boot = {m: np.zeros((B, 4)) for m in MODELS}
for b in range(B):
    w = np.bincount(rng.integers(0, N, N), minlength=N)
    for m in MODELS:
        boot[m][b] = metrics(D[m], ngt_t, glare_t, w)
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
KEYS = ["mAP50", "mAP50_95", "detected", "false_alarms"]
R["test"] = {m: {k: {"value": float(point[m][i]), "ci95": ci(boot[m][:, i])} for i, k in enumerate(KEYS)} for m in MODELS}
R["paired_diff_vs_yolo11s"] = {}
for m in MODELS:
    if m == "yolo11s":
        continue
    dif = boot[m] - boot["yolo11s"]
    R["paired_diff_vs_yolo11s"][m] = {k: {"diff": float(point[m][i] - point["yolo11s"][i]), "ci95": ci(dif[:, i]),
                                          "significant": bool(np.percentile(dif[:, i], 2.5) > 0 or np.percentile(dif[:, i], 97.5) < 0)}
                                      for i, k in enumerate(KEYS)}
for m in MODELS:
    R["test"][m]["official"] = {k: official[m][kk] for k, kk in zip(KEYS, ["test_mAP50", "test_mAP50_95", "img_hit_rate", "img_false_pos_rate"])}
    R["test"][m]["params_M"] = official[m]["params_M"]; R["test"][m]["GFLOPs"] = official[m]["GFLOPs"]
    R["test"][m]["ms_T4"] = {"yolo11m": 18.3, "rtdetr_l": 18.3}.get(m, official[m]["ms_per_img"])
print("bootstrap done")

# ------------------------------------------------------------------ 2. matched false alarms (all detectors)
def fa_threshold(maxconf, glare, target):
    """lowest threshold whose false-alarm rate on the given images is <= target"""
    neg = np.sort(maxconf[~glare])[::-1]
    k = int(np.floor(target * len(neg) + 1e-9))
    return float(neg[k] + 1e-6) if k < len(neg) else 0.0

target = rates_w(DV["yolo11s"]["maxconf"], glare_v, np.ones(len(names_v)))[1]
R["matched_fa"] = {"target_val_false_alarms": float(target), "models": {}}
for m in MODELS:
    t = fa_threshold(DV[m]["maxconf"], glare_v, target)
    hit, fa = rates_w(D[m]["maxconf"], glare_t, np.ones(N), t)
    fixed = {}
    for lvl in [0.05, 0.075, 0.10]:
        tt = fa_threshold(D[m]["maxconf"], glare_t, lvl)
        fixed[str(lvl)] = float(rates_w(D[m]["maxconf"], glare_t, np.ones(N), tt)[0])
    R["matched_fa"]["models"][m] = {"val_threshold": t, "test_detected": float(hit), "test_false_alarms": float(fa),
                                    "test_detected_at_fixed_fa": fixed}
curves = {}
for m in MODELS:
    mc = D[m]["maxconf"]; ts = np.unique(mc)[::-1]
    pos, neg = mc[glare_t], mc[~glare_t]
    curves[m] = {"fa": [float((neg >= t).mean()) for t in ts], "det": [float((pos >= t).mean()) for t in ts],
                 "at025": [float((neg >= CONF).mean()), float((pos >= CONF).mean())]}
json.dump(curves, open(f"{AN}/det_fa_curves.json", "w"))

# ------------------------------------------------------------------ 3. recall by glare size (test, conf >= 0.25, IoU >= 0.5)
def size_px(b):
    return np.sqrt((b[:, 2] - b[:, 0]) * 1280 * (b[:, 3] - b[:, 1]) * 720)

sizes_t = {n: size_px(gt_test[n]) for n in names_t}
allsz = np.concatenate(list(sizes_t.values()))
R["size"] = {"bins": SIZE_NAMES, "gt_boxes_per_bin": [int(((allsz >= lo) & (allsz < hi)).sum()) for lo, hi in SIZE_BINS],
             "median_box_px": float(np.median(allsz)), "recall": {}}
for m in MODELS:
    P = load_preds(m, "test"); hit = np.zeros(len(SIZE_BINS)); tot = np.zeros(len(SIZE_BINS))
    for n in names_t:
        g = gt_test[n]
        if not len(g):
            continue
        pc = np.array(P[n]["conf"]); pb = np.array(P[n]["boxes"]).reshape(-1, 4)[pc >= CONF]
        found = np.zeros(len(g), bool)
        if len(pb):
            iou = box_iou(g, pb); mm = np.argwhere(iou >= 0.5)
            if len(mm):
                mm = mm[iou[mm[:, 0], mm[:, 1]].argsort()[::-1]]
                mm = mm[np.unique(mm[:, 1], return_index=True)[1]]
                mm = mm[np.unique(mm[:, 0], return_index=True)[1]]
                found[mm[:, 0]] = True
        for j, (lo, hi) in enumerate(SIZE_BINS):
            sel = (sizes_t[n] >= lo) & (sizes_t[n] < hi)
            tot[j] += sel.sum(); hit[j] += (found & sel).sum()
    R["size"]["recall"][m] = [float(h / t) if t else None for h, t in zip(hit, tot)]

# ------------------------------------------------------------------ 4. brightness-threshold baseline
def grey640(split, names):
    out = {}
    for n in names:
        g = cv2.imread(f"{DS}/images/{split}/{n}", cv2.IMREAD_GRAYSCALE)
        h, w = g.shape
        out[n] = cv2.resize(g, (640, int(round(h * 640 / w))), interpolation=cv2.INTER_AREA) if w != 640 else g
    return out

def bright_preds(grey, T, amin, expand):
    P = {}
    for n, g in grey.items():
        H, W = g.shape
        k, lab, st, cen = cv2.connectedComponentsWithStats((g >= T).astype(np.uint8), connectivity=8)
        bx, sc = [], []
        for i in range(1, k):
            x, y, w, h, a = st[i]
            if a < amin:
                continue
            cx, cy, hw, hh = x + w / 2, y + h / 2, w * expand / 2, h * expand / 2
            bx.append([max(cx - hw, 0) / W, max(cy - hh, 0) / H, min(cx + hw, W) / W, min(cy + hh, H) / H]); sc.append(float(a))
        o = np.argsort(sc)[::-1][:300]
        P[n] = {"boxes": [bx[i] for i in o], "conf": [sc[i] / (sc[i] + 100.0) for i in o]}   # monotone score in (0, 1)
    return P

gv, gtst = grey640("val", names_v), grey640("test", names_t)
best = None
for T in [200, 220, 235, 250]:
    for amin in [2, 8, 32]:
        for ex in [1.0, 1.5, 2.5]:
            m50 = map_w(prepare(bright_preds(gv, T, amin, ex), gt_val, names_v), np.ones(len(names_v)), ngt_v)[0]
            if best is None or m50 > best[0]:
                best = (m50, T, amin, ex)
_, T, amin, ex = best
DB = prepare(bright_preds(gtst, T, amin, ex), gt_test, names_t)
DBv = prepare(bright_preds(gv, T, amin, ex), gt_val, names_v)
bt = fa_threshold(DBv["maxconf"], glare_v, target)
R["bright"] = {"params": {"threshold": T, "min_area_px_at_640w": amin, "box_expand": ex, "val_mAP50": float(best[0])},
               "test": dict(zip(KEYS, map(float, metrics(DB, ngt_t, glare_t, thr=1e-9)))),
               "test_at_matched_fa": dict(zip(["detected", "false_alarms"], map(float, rates_w(DB["maxconf"], glare_t, np.ones(N), bt))))}
print("baseline done", R["bright"])

# ------------------------------------------------------------------ 5. leakage analysis
pos = {}                                                    # source -> {position: split}; seconds for videos, index for ar_
for split in ["train", "val", "test"]:
    for f in glob.glob(f"{DS}/images/{split}/*.jpg"):
        pre, num = os.path.basename(f)[:-4].split("_")
        pos.setdefault(pre, {})[int(num)] = split

def nearest_train(src, p, assign):
    best_d = None
    for d in range(1, 4000):
        for q in (p - d, p + d):
            if assign.get(q) == "train":
                return d
        if best_d is None and d > 3000:
            break
    return None

leak = {}
for group, srcs in [("video", ["v1", "v2", "v5", "v6"]), ("dashcam", ["ar"])]:
    dists = [nearest_train(s, p, pos[s]) for s in srcs for p, sp in pos[s].items() if sp == "test"]
    dists = np.array([d for d in dists if d is not None])
    sims = {1: [], 2: []}
    for r in range(200):                                    # random frame-level split, same 70/15/15 proportions
        for s in srcs:
            ps = np.array(sorted(pos[s])); u = rng.random(len(ps))
            sp = np.where(u < 0.70, "train", np.where(u < 0.85, "val", "test"))
            a = dict(zip(ps.tolist(), sp.tolist()))
            for p in ps[sp == "test"]:
                for k in (1, 2):
                    sims[k].append(any(a.get(p + d) == "train" for d in range(-k, k + 1) if d))
    leak[group] = {"block_split_min_gap_to_train": int(dists.min()), "block_split_median_gap_to_train": float(np.median(dists)),
                   "block_split_share_within_2": float((dists <= 2).mean()),
                   "random_split_share_with_train_neighbour_within_1": float(np.mean(sims[1])),
                   "random_split_share_with_train_neighbour_within_2": float(np.mean(sims[2]))}

man = {r["image"]: r for r in csv.DictReader(open(f"{REL}/manifest.csv"))}
dh = {n: int(r["dhash"], 16) for n, r in man.items()}
ham = lambda a, b: bin(dh[a] ^ dh[b]).count("1")
def grey_mae(a, b):
    ga = cv2.resize(cv2.imread(f"{DS}/images/{man[a]['split']}/{a}", cv2.IMREAD_GRAYSCALE), (160, 90), interpolation=cv2.INTER_AREA)
    gb = cv2.resize(cv2.imread(f"{DS}/images/{man[b]['split']}/{b}", cv2.IMREAD_GRAYSCALE), (160, 90), interpolation=cv2.INTER_AREA)
    return float(np.abs(ga.astype(float) - gb.astype(float)).mean())
name_of = lambda s, p: f"{s}_{p:05d}.jpg"
nb, rp = [], []
for s in ["v1", "v5", "v6", "ar"]:
    ps = sorted(pos[s])
    nb += [(name_of(s, p), name_of(s, p + 1)) for p in ps if p + 1 in pos[s]]
    for _ in range(600):
        a, b = rng.choice(ps, 2, replace=False)
        if abs(int(a) - int(b)) >= 60:
            rp.append((name_of(s, int(a)), name_of(s, int(b))))
sel_nb = [nb[i] for i in rng.choice(len(nb), min(1500, len(nb)), replace=False)]
sel_rp = [rp[i] for i in rng.choice(len(rp), min(1500, len(rp)), replace=False)]
h_nb = np.array([ham(a, b) for a, b in nb]); h_rp = np.array([ham(a, b) for a, b in rp])
m_nb = np.array([grey_mae(a, b) for a, b in sel_nb]); m_rp = np.array([grey_mae(a, b) for a, b in sel_rp])
leak["similarity"] = {"n_neighbour_pairs": len(nb), "n_random_pairs": len(rp),
                      "neighbour_hamming_median": float(np.median(h_nb)), "random_hamming_median": float(np.median(h_rp)),
                      "neighbour_share_hamming_le_10": float((h_nb <= 10).mean()), "random_share_hamming_le_10": float((h_rp <= 10).mean()),
                      "neighbour_grey_mae_median": float(np.median(m_nb)), "random_grey_mae_median": float(np.median(m_rp))}
np.save(f"{AN}/hamming_neighbour.npy", h_nb); np.save(f"{AN}/hamming_random.npy", h_rp)

gt_train = load_gt("train"); names_tr = sorted(gt_train)
ngt_tr = np.array([len(gt_train[n]) for n in names_tr])
gap = {}
for split, g, nm, ng in [("train", gt_train, names_tr, ngt_tr), ("val", gt_val, names_v, ngt_v), ("test", gt_test, names_t, ngt_t)]:
    d = prepare(load_preds("yolo11s", split), g, nm)
    gap[split] = dict(zip(KEYS, map(float, metrics(d, ng, ng > 0))))
leak["yolo11s_train_val_test"] = gap
R["leakage"] = leak
print("leakage done", json.dumps(leak, indent=1))

# ------------------------------------------------------------------ 6. per source + video-only subset
R["per_source"] = {}
for src in ["ar", "v6", "v5", "v1"]:
    sel = np.array([n.startswith(src + "_") for n in names_t]).astype(float)
    R["per_source"][src] = {"images": int(sel.sum()), "boxes": int((ngt_t * sel).sum()),
                            "mAP50": {m: float(map_w(D[m], sel, ngt_t)[0]) for m in MODELS}}
vid = np.array([not n.startswith("ar_") for n in names_t]).astype(float)
R["video_only_test"] = {"images": int(vid.sum()), "glare_images": int((vid * glare_t).sum()),
                        "models": {m: dict(zip(KEYS, map(float, metrics(D[m], ngt_t, glare_t, vid)))) for m in MODELS}}

R["dataset"] = {"test_images": N, "test_glare_images": int(glare_t.sum()), "test_boxes": int(ngt_t.sum()),
                "val_images": len(names_v), "train_images": len(names_tr)}
json.dump(R, open(f"{AN}/results.json", "w"), indent=1)
print("wrote results.json")
