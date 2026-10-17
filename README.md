# Indian Night Headlight Glare

Headlight glare detection dataset (annotations only) for night-time driving on Indian roads.

- One class: headlight glare
- 5,278 images (train 3,681 / val 797 / test 800), 7,897 boxes
- YOLO format labels in `labels/`

## Files

- `labels/` - YOLO labels
- `manifest.csv` - list of frames and their split
- `sources.csv` - source videos
- `data.yaml` - dataset config
- `rebuild_frames.py` - rebuilds the video frames from the source videos
- `code/` - training and evaluation code

## Usage

```bash
python rebuild_frames.py --videos videos --out dataset --download
yolo detect train data=dataset/data.yaml model=yolo11s.pt imgsz=960
```

## License

Annotations: CC BY 4.0 (see `LICENSE-annotations.md`). Code: MIT (see `LICENSE-code.txt`).
