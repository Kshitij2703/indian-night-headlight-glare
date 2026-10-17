"""Latency of the five detectors under identical conditions on one machine (Apple M4 GPU via PyTorch MPS):
batch 1, 960 px, FP32, end-to-end Ultralytics predict (pre-processing + inference + post-processing),
median over 100 test images after 10 warm-up images."""
import glob, json, os, time
import numpy as np
from ultralytics import YOLO, RTDETR

ROOT = os.environ.get("GLARE_ROOT", os.path.expanduser("~/Desktop/vehicle-glare"))
files = sorted(glob.glob(f"{ROOT}/dataset/glare_dataset_final/images/test/*.jpg"))
rng = np.random.default_rng(0)
sel = [files[i] for i in rng.choice(len(files), 110, replace=False)]
out = {}
for m in ["yolov8s", "yolo11n", "yolo11s", "yolo11m", "rtdetr_l"]:
    model = (RTDETR if m.startswith("rtdetr") else YOLO)(f"{ROOT}/models/{m}_best.pt")
    tot, inf = [], []
    for k, f in enumerate(sel):
        t = time.perf_counter()
        r = model.predict(f, imgsz=960, conf=0.25, device="mps", verbose=False)[0]
        dt = (time.perf_counter() - t) * 1000
        if k >= 10:
            tot.append(dt); inf.append(r.speed["inference"])
    out[m] = {"end_to_end_ms_median": round(float(np.median(tot)), 1), "inference_ms_median": round(float(np.median(inf)), 1)}
    print(m, out[m], flush=True)
json.dump(out, open(f"{ROOT}/paper/analysis/speed_m4.json", "w"), indent=1)
