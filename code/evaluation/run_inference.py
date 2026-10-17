"""Run the five trained glare detectors (no training) and cache raw predictions per image.
Settings match Ultralytics validation: imgsz 960, conf 0.001, NMS IoU 0.7, max 300 detections."""
import glob, json, os, time
import numpy as np
from ultralytics import YOLO, RTDETR

ROOT = os.environ.get("GLARE_ROOT", os.path.expanduser("~/Desktop/vehicle-glare"))
DS = f"{ROOT}/dataset/glare_dataset_final"
OUT = f"{ROOT}/paper/analysis/preds"
MODELS = ["yolo11s", "yolo11n", "yolov8s", "yolo11m", "rtdetr_l"]
JOBS = [(m, s) for s in ["test", "val"] for m in MODELS] + [("yolo11s", "train")]

for m, split in JOBS:
    out = f"{OUT}/{m}_{split}.json"
    if os.path.exists(out):
        continue
    files = sorted(glob.glob(f"{DS}/images/{split}/*.jpg"))
    model = (RTDETR if m.startswith("rtdetr") else YOLO)(f"{ROOT}/models/{m}_best.pt")
    dev, res, t0 = "mps", {}, time.time()
    bs = 4 if m.startswith("rtdetr") else 16              # RT-DETR at 960 px runs out of unified memory at 16
    for i in range(0, len(files), bs):
        chunk = files[i:i + bs]
        try:
            rs = model.predict(chunk, imgsz=960, conf=0.001, iou=0.7, max_det=300, device=dev, verbose=False)
        except Exception as e:                       # some ops are not supported on Apple GPUs
            print(f"{m}: MPS failed ({str(e)[:80]}), switching to CPU", flush=True)
            dev = "cpu"
            rs = model.predict(chunk, imgsz=960, conf=0.001, iou=0.7, max_det=300, device=dev, verbose=False)
        for f, r in zip(chunk, rs):
            h, w = r.orig_shape
            b = r.boxes.xyxy.cpu().numpy() / np.array([w, h, w, h])
            res[os.path.basename(f)] = {"wh": [int(w), int(h)], "boxes": np.round(b, 5).tolist(),
                                        "conf": np.round(r.boxes.conf.cpu().numpy(), 5).tolist()}
    json.dump(res, open(out, "w"))
    print(f"{m} {split}: {len(files)} images in {time.time() - t0:.0f} s on {dev}", flush=True)
print("all done", flush=True)
