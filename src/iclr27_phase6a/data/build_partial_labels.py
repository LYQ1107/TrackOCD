"""Build the Phase 6A partial-label training file.

Input: OVTR training annotation file (LVIS-style, 1203 categories over TAO
train image sequences).
Output: a file that keeps only the 48 TrackOCD supported-known categories
as labeled positives; every other annotation is dropped from the label set
so it becomes part of the unlabeled stream (PU setting). Official per-image
neg_category_ids / not_exhaustive_category_ids are preserved.

No benchmark novel GT and no external labeled OOD supervision are used.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.iclr27_phase4s.protocol import known_ids

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default="third_party/research_refs_phase4n/OVTR/data/lvis_clear_75_60.json")
    ap.add_argument(
        "--out",
        default="third_party/research_refs_phase4n/OVTR/data/lvis_known48_partial.json")
    args = ap.parse_args()

    known = known_ids()
    src = json.loads(Path(args.src).read_text())
    kept_anns = [a for a in src["annotations"] if a["category_id"] in known]
    dropped_anns = len(src["annotations"]) - len(kept_anns)
    out = {
        "info": {**src.get("info", {}),
                 "note": "Phase 6A partial labels: only 48 TrackOCD "
                         "supported-known categories are labeled; all other "
                         "annotations are unlabeled by design."},
        "licenses": src.get("licenses", []),
        "images": src["images"],
        "categories": src["categories"],
        "annotations": kept_anns,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    stats = {
        "known_ids": sorted(known),
        "known_ids_present": sorted(known & {a["category_id"] for a in kept_anns}),
        "known_ids_absent": sorted(known - {a["category_id"] for a in kept_anns}),
        "n_known_categories": len({a["category_id"] for a in kept_anns}),
        "n_known_annotations": len(kept_anns),
        "n_unlabeled_dropped_annotations": dropped_anns,
        "n_images": len(src["images"]),
        "n_annotated_images": len({a["image_id"] for a in kept_anns}),
        "neg_metadata_present": sum(
            1 for im in src["images"] if im.get("neg_category_ids")),
        "not_exhaustive_metadata_present": sum(
            1 for im in src["images"] if im.get("not_exhaustive_category_ids")),
    }
    out_json = Path(args.out).with_suffix(".stats.json")
    out_json.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
