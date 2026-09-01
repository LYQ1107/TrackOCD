"""Run YOLOE prompt-free (PF) proposals over TrackOCD frames."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
FRAMES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/"
              "TAO-Amodal/frames")
YOLOE_REPO = ROOT / "third_party" / "research_refs_phase4n" / "YOLOE"

TAO = {
    "dev": ROOT / "outputs" / "iclr27_phase3a" / "smoke" /
    "tao_subset" / "validation_20.json",
    "heldout": ROOT / "outputs" / "iclr27_phase4n" / "audit" /
    "validation_heldout_tao_corrected.json",
}
PRE = {
    "dev": ROOT / "outputs" / "iclr27_phase3a" / "smoke" /
    "pre_assoc_detections",
    "heldout": ROOT / "outputs" / "iclr27_phase4l" / "heldout_export" /
    "pre_assoc_detections",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "heldout"], required=True)
    ap.add_argument("--ckpt", default=str(
        ROOT / "checkpoints" / "yoloe-v8l-seg-pf.pt"))
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(YOLOE_REPO))
    import torch
    torch.cuda.set_device(args.gpu)
    from ultralytics import YOLOE
    model = YOLOE(args.ckpt)
    d = json.loads(TAO[args.mode].read_text())
    # frame order per image from pre-assoc
    order = {}
    for p in PRE[args.mode].glob("*.jsonl"):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            order[int(r["image_id"])] = int(r["frame_order"])
    images = d["images"]
    # ensure deterministic order
    images = sorted(images, key=lambda im: (im["video_id"],
                                            order.get(im["id"], 0),
                                            im["file_name"]))
    paths = [FRAMES / im["file_name"] for im in images]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("missing frames: " + str(missing[:3]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "frame_id", "image_id", "bbox_xyxy",
                    "score", "raw_class", "source_detector"])
        for i in range(0, len(images), 16):
            batch = paths[i:i + 16]
            res = model.predict(batch, conf=args.conf, verbose=False)
            for im, r in zip(images[i:i + 16], res):
                if r.boxes is None:
                    continue
                xyxy = r.boxes.xyxy.cpu().tolist()
                conf = r.boxes.conf.cpu().tolist()
                cls = r.boxes.cls.cpu().tolist()
                for b, s, c in zip(xyxy, conf, cls):
                    w.writerow([im["video_id"],
                                order.get(im["id"], 0), im["id"],
                                json.dumps([round(v, 2) for v in b]),
                                round(float(s), 6),
                                str(r.names.get(int(c), int(c))),
                                "YOLOE_PF"])
            print("yoloe", args.mode, i + len(batch), "/", len(images),
                  flush=True)
    tmp.replace(args.out)
    print("YOLOE_DONE", args.mode, args.out)


if __name__ == "__main__":
    main()
