"""Phase 4I input manifest: hashes key frozen inputs and describes replay
packages.  Large binary trees are summarized by count/size, not hashed."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs" / "iclr27_phase4i" / "audit" / "input_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = [
        "checkpoints/simowt_weight.pth",
        "third_party/SimOWT/projects/IDOL/idol/idol.py",
        "third_party/SimOWT/projects/IDOL/idol/models/tracker.py",
        "third_party/SimOWT/detectron2/data/datasets/builtin.py",
        "runs/orbit_mdc/mdc_m2/model.pth",
        "data/trackocd_v1/protocols.json",
        "data/trackocd_v1/pure/public/train_known_tracks.jsonl",
        "data/trackocd_v1/pure/private/val_gt_track_labels.jsonl",
        "outputs/iclr27_phase3a/smoke/selected_20_videos.csv",
        "outputs/iclr27_phase3a/smoke/export_manifest.json",
        "outputs/iclr27_phase3a/smoke/export_schema.json",
        "outputs/iclr27_phase3b/full_export/export_manifest.json",
    ]
    hashes = {}
    for rel in files:
        p = ROOT / rel
        hashes[rel] = sha256(p) if p.exists() else "MISSING"

    def pkg_stats(base: Path):
        dirs = sorted([d for d in base.iterdir() if d.is_dir()])
        frames = 0
        npz_total = 0
        with_feats = 0
        for d in dirs:
            npzs = sorted(d.glob("frame_*.npz"))
            frames += len(npzs)
            npz_total += 1
            if npzs:
                import numpy as np
                with np.load(npzs[0]) as z:
                    if "track_feats" in z.files:
                        with_feats += 1
        return {"videos": len(dirs), "frames": frames,
                "videos_with_track_feats": with_feats,
                "npz_total": npz_total}

    manifest = {
        "file_hashes": hashes,
        "subset_replay_packages": pkg_stats(
            ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "replay_packages"),
        "full_replay_packages": pkg_stats(
            ROOT / "outputs" / "iclr27_phase3b" / "full_export"
                 / "replay_packages"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
