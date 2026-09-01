"""Hash raw/scored prediction artifacts after the frozen-candidate boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    freeze = json.loads((root / "outputs/iclr27_phase19/manifests/prediction_freeze.json").read_text())
    marker = root / "outputs/iclr27_phase19/completion/public_predictions.frozen"
    if not marker.exists():
        raise SystemExit("prediction freeze marker is required")
    files = {}
    for path in sorted((root / "outputs/iclr27_phase19/predictions").glob("*_raw.json")):
        files[str(path.relative_to(root))] = sha(path)
    for path in sorted((root / "outputs/iclr27_phase19/predictions").glob("*_scored.json")):
        files[str(path.relative_to(root))] = sha(path)
    out = {"protocol": "trackocd_iclr27_phase19_prediction_hashes_after_freeze",
           "freeze_sha256": freeze["freeze_sha256"], "freeze_marker_sha256": sha(marker),
           "true_novel_labels_used_for_training_or_selection": False,
           "scored_files_are_post_freeze_measurements": True,
           "files": files}
    a.out.parent.mkdir(parents=True, exist_ok=True); tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n"); tmp.replace(a.out)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__": main()
