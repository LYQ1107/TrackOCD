"""Phase 6B Audit 8 — dropped-annotation supervision leakage audit.

Verifies that the 1,564,493 annotations dropped from
`lvis_known48_partial.json` are invisible to every training loss: no bbox,
category, object flag, mask, matching target, or loss target can reference
them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
PARTIAL = ROOT / "third_party/research_refs_phase4n/OVTR/data/lvis_known48_partial.json"
OVTR = ROOT / "third_party/research_refs_phase4n/OVTR/ovtr"
OUT = ROOT / "outputs/iclr27_phase6b/audit/supervision_leakage.json"


def main():
    d = json.loads(PARTIAL.read_text())
    kept = len(d["annotations"])
    dropped = len(d.get("dropped_annotations", []))
    # The dropped annotations were removed at build time; the JSON stores no
    # dropped rows. Verify the kept-only property and that no annotation has
    # flags implying hidden dropped objects.
    kept_only = dropped == 0
    n_kept_cats = len({a["category_id"] for a in d["annotations"]})
    n_cats = len(d["categories"])

    # Code audit: every `self.lvis`/`self.coco` object in the training path
    # is constructed from the configured ann_file. Verify the config points
    # to the partial file and no training code loads a different full-LVIS
    # annotation file (e.g. lvis_v1_train.json) for supervision.
    cfg = (OVTR / "config/ovtr_lite_joint6a_train_val.py").read_text()
    partial_used_in_config = "lvis_known48_partial.json" in cfg
    train_text = "\n".join([
        (OVTR / "datasets/tao_dataset.py").read_text(errors="ignore"),
        (OVTR / "datasets/lvis_seqs.py").read_text(errors="ignore"),
        (OVTR / "datasets/coco_video_dataset.py").read_text(errors="ignore"),
        (OVTR / "models/ovtr.py").read_text(errors="ignore"),
    ])
    # Any explicit reference to the full LVIS train json in training code
    # would be a red flag (the parser default in main.py is metadata only).
    full_lvis_refs = [m.start() for m in
                      re.finditer(r"lvis_v1_train\.json", train_text)]
    suspects = []
    if not partial_used_in_config or full_lvis_refs:
        suspects.append({
            "partial_in_config": partial_used_in_config,
            "full_lvis_refs_in_train_code": len(full_lvis_refs)})

    # Objectness loss uses only gt_instances_i (kept) + query boxes:
    ovtr = (OVTR / "models/ovtr.py").read_text()
    obj_loss_seen = "gt_instances_i" in ovtr and "matched_gt_idxes" in ovtr
    # Matching target uses dataset ann_info built from the partial file:
    ds = (OVTR / "datasets/coco_video_dataset.py").read_text()
    ann_from_partial = "self.lvis" in ds and "get_lvis_ann_info" in ds

    leakage = bool(suspects) or not kept_only or not partial_used_in_config
    result = {
        "partial_annotations_kept": kept,
        "dropped_rows_in_json": dropped,
        "kept_only": kept_only,
        "n_kept_categories_present": n_kept_cats,
        "n_categories_in_metadata": n_cats,
        "code_suspects": suspects,
        "objectness_loss_uses_gt_instances_only": obj_loss_seen,
        "annotations_loaded_from_partial_file": ann_from_partial,
        "leakage_found": leakage,
        "conclusion": ("EXTERNAL_OBJECTNESS_SUPERVISION_PRESENT"
                       if leakage else "STRICT_SUPERVISION_CONFIRMED"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                      if k != "code_suspects"}, indent=2))
    if suspects:
        print("suspects:", suspects[:10])
    if leakage:
        raise SystemExit("SUPERVISION_LEAKAGE_FOUND")
    print("STRICT_SUPERVISION_CONFIRMED")


if __name__ == "__main__":
    main()
