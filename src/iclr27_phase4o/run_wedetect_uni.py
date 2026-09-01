"""Run WeDetect-Uni universal proposals over TrackOCD frames."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
FRAMES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/"
              "TAO-Amodal/frames")
WD = ROOT / "third_party" / "research_refs_phase4n" / "WeDetect"

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
        ROOT / "checkpoints" / "wedetect_base_uni.pth"))
    ap.add_argument("--num-proposals", type=int, default=300)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(WD))
    import torch
    torch.cuda.set_device(args.gpu)
    import generate_proposal as gp
    model_size = "base" if "base" in args.ckpt else "large"
    model = gp.SimpleYOLOWorldDetector(
        backbone_size=model_size, prompt_dim=768, num_prompts=256,
        num_proposals=args.num_proposals)
    ck = torch.load(args.ckpt, map_location="cpu")
    keys = list(ck.keys())
    for key in keys:
        if "backbone" in key:
            ck[key.replace("backbone.image_model.model.", "backbone.")] = \
                ck.pop(key)
    keys = list(ck.keys())
    for key in keys:
        if "bbox_head" in key:
            nk = key.replace("bbox_head.head_module.", "bbox_head.")
            nk = nk.replace("0.2.", "0.6.").replace("1.2.", "1.6.")
            nk = nk.replace("2.2.", "2.6.").replace("1.bn", "4")
            nk = nk.replace("1.conv", "3").replace("0.bn", "1")
            nk = nk.replace("0.conv", "0")
            ck[nk] = ck.pop(key)
    msg = model.load_state_dict(ck, strict=False)
    print("load msg", msg, flush=True)
    model = model.cuda()
    model.eval()
    d = json.loads(TAO[args.mode].read_text())
    order = {}
    for p in PRE[args.mode].glob("*.jsonl"):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            order[int(r["image_id"])] = int(r["frame_order"])
    images = sorted(d["images"], key=lambda im: (
        im["video_id"], order.get(im["id"], 0), im["file_name"]))
    paths = [FRAMES / im["file_name"] for im in images]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "frame_id", "image_id", "bbox_xyxy",
                    "score", "raw_class", "source_detector"])
        with torch.no_grad():
            for i in range(0, len(images), 8):
                batch = [str(p) for p in paths[i:i + 8]]
                out = model(batch)
                for im, o in zip(images[i:i + 8], out):
                    bb = o["bboxes"].float().cpu()
                    sc = o["scores"].float().cpu()
                    for b, s in zip(bb, sc):
                        w.writerow([im["video_id"],
                                    order.get(im["id"], 0), im["id"],
                                    json.dumps([round(float(v), 2)
                                                for v in b]),
                                    round(float(s), 6), "",
                                    "WEDETECT_UNI"])
                print("wedetect", args.mode, i + len(batch), "/",
                      len(images), flush=True)
    tmp.replace(args.out)
    print("WEDETECT_DONE", args.mode, args.out)


if __name__ == "__main__":
    main()
