"""Phase 4J input manifest: hashes key Phase 4I artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs" / "iclr27_phase4j" / "audit" / "input_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = [
        "runs/orbit_mdc/mdc_m2/model.pth",
        "outputs/iclr27_phase4i/audit/detection_features/88/feats.npz",
        "outputs/iclr27_phase4i/audit/semantic_logs/B2_l0.1/88.jsonl",
        "outputs/iclr27_phase4i/audit/semantic_logs/B1/88.jsonl",
        "outputs/frame_online_trackocd/subset/B0/_/0000003423.json",
        "outputs/iclr27_phase3a/smoke/tao_subset/validation_20.json",
        "data/trackocd_v1/pure/splits/supported_known_ids.json",
    ]
    hashes = {}
    for rel in files:
        p = ROOT / rel
        hashes[rel] = sha256(p) if p.exists() else "MISSING"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(hashes, indent=2))
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
