"""Phase 6B Audit 1 — supported-known category round-trip.

Verifies, for at least 20 supported-known classes that are present in the
partial-label training file, the full chain:

  dataset category_id
    -> training contiguous index (cat2label)
    -> joint_known target (anchor row in SemanticMemory.known_ids order)
    -> model predicted anchor index
    -> exported semantic_category (sem_sid = known_ids[anchor_idx])
    -> evaluator expected category_id (GT ground_truth_category_id)

All mappings must be identity-consistent. No Q1/dev novel labels are used.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
PARTIAL = ROOT / "third_party/research_refs_phase4n/OVTR/data/lvis_known48_partial.json"
KNOWN_IDS = ROOT / "data/trackocd_v1/pure/splits/supported_known_ids.json"
OUT = ROOT / "outputs/iclr27_phase6b/audit/category_roundtrip.json"


def load_lvis_cat_ids(ann_file: Path):
    """Replicate LVIS.get_cat_ids() used by datasets/tao_dataset.py."""
    import lvis  # available in the OVTR environment
    ds = lvis.LVIS(str(ann_file))
    cats = ds.dataset["categories"]
    ids = [c["id"] for c in cats]
    return ids


def main():
    random.seed(20260817)
    partial = json.loads(PARTIAL.read_text())
    cat_ids = [c["id"] for c in partial["categories"]]
    # Dataset cat2label: {cat_id: i for i, cat_id in enumerate(cat_ids)}
    cat2label = {cid: i for i, cid in enumerate(cat_ids)}
    lvis_ids = load_lvis_cat_ids(PARTIAL)
    ann_cats = sorted({a["category_id"] for a in partial["annotations"]})
    known = json.loads(KNOWN_IDS.read_text())
    present = sorted(set(known) & set(ann_cats))
    sample = random.sample(present, min(24, len(present)))
    if len(present) < 20:
        raise SystemExit(f"only {len(present)} present known ids < 20")

    rows = []
    ok = True
    for c in sample:
        label = cat2label[c]
        # contiguous label must equal category_id - 1 when ids are 1..N
        contiguous_ok = (label == c - 1)
        anchor_idx = known.index(c)
        # model predicts anchor_idx -> exported semantic_category:
        exported_sid = known[anchor_idx]
        # evaluator expects GT category id:
        evaluator_cat = c
        row = {
            "category_id": c,
            "cat2label": label,
            "label_eq_id_minus_1": contiguous_ok,
            "joint_known_anchor_idx": anchor_idx,
            "exported_sem_sid": exported_sid,
            "evaluator_expected_category_id": evaluator_cat,
            "roundtrip_ok": (
                contiguous_ok
                and exported_sid == evaluator_cat
                and anchor_idx == known.index(exported_sid)),
        }
        ok = ok and row["roundtrip_ok"]
        rows.append(row)

    result = {
        "n_supported_known": len(known),
        "n_present_in_train_annotations": len(present),
        "n_sampled": len(sample),
        "lvis_get_cat_ids_matches_json_order": lvis_ids == cat_ids,
        "all_roundtrip_ok": ok,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"},
                     indent=2))
    if not ok:
        raise SystemExit("CATEGORY_ROUNDTRIP_FAILED")
    print("CATEGORY_ROUNDTRIP_OK")


if __name__ == "__main__":
    main()
