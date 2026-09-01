#!/usr/bin/env python3
"""Freeze TrackOCD-v1.0: SHA256 manifests, benchmark/dataset/evaluation docs."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA = PROJECT_ROOT / "data" / "trackocd_v1"
DOCS = PROJECT_ROOT / "docs" / "trackocd_v1"
VERSION = "TrackOCD-v1.0"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_files(root):
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(PROJECT_ROOT))] = sha256(p)
    return out


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    for d in ("outputs/trackocd_v1/metrics", "outputs/trackocd_v1/tests",
              "outputs/trackocd_v1/baselines", "runs/trackocd_v1"):
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
    manifests = DATA / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    # image path integrity for val streams
    missing_paths = []
    seen = set()
    frame_root = PROJECT_ROOT / "data" / "raw" / "tao" / "frames"
    for stream in ("val_gt_track_stream.jsonl", "val_gt_track_stream_seed1027.jsonl",
                   "val_gt_track_stream_seed1028.jsonl", "val_gt_track_stream_seed1029.jsonl"):
        with open(PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public" / stream) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                for img in r.get("image_paths", []):
                    if img in seen:
                        continue
                    seen.add(img)
                    if not (frame_root / img).exists():
                        missing_paths.append(img)
    image_integrity = {
        "unique_image_paths_checked": len(seen),
        "missing": len(missing_paths),
        "missing_sample": missing_paths[:5],
    }

    protocol_files = {}
    for proto in ("pure", "ov_assisted"):
        protocol_files[proto] = collect_files(DATA / proto)
    code_files = {
        "src/trackocd_v1/build_protocols.py": sha256(PROJECT_ROOT / "src/trackocd_v1/build_protocols.py"),
        "src/trackocd_v1/evaluation/trackocd_evaluator.py": sha256(
            PROJECT_ROOT / "src/trackocd_v1/evaluation/trackocd_evaluator.py"),
        "tests/test_trackocd_evaluator.py": sha256(PROJECT_ROOT / "tests/test_trackocd_evaluator.py"),
    }
    stats = json.loads((DATA / "protocols.json").read_text())
    manifest = {
        "version": VERSION,
        "image_path_integrity": image_integrity,
        "protocols": {p: stats[p] for p in stats},
        "stream_order": [
            "val_gt_track_stream.jsonl",
            "val_gt_track_stream_seed1027.jsonl",
            "val_gt_track_stream_seed1028.jsonl",
            "val_gt_track_stream_seed1029.jsonl",
        ],
        "seeds": [1027, 1028, 1029],
        "files": protocol_files,
        "code": code_files,
        "evaluator_version": VERSION,
    }
    (manifests / "manifest_v1.0.json").write_text(json.dumps(manifest, indent=2))
    print("manifest files:", len(protocol_files["pure"]) + len(protocol_files["ov_assisted"]),
          "image integrity:", image_integrity)

    # ---- Docs ----
    p = stats["pure"]
    o = stats["ov_assisted"]
    (DOCS / "BENCHMARK_CARD.md").write_text(f"""# TrackOCD-v1.0 Benchmark Card

Version: {VERSION} (frozen 2026-08-05)

## Two official task protocols

### Protocol A: Pure TrackOCD

- known = official TAO-OW known categories with >=1 TAO-train annotation
  (train-supported known), computed dynamically: {p['supported_known_categories']} categories.
- novel = official TAO-OW unknown + official known categories with zero
  TAO-train visual samples: {p['novel_categories_total']} total,
  {p['novel_categories_appearing_in_val']} appearing in val.
- val tracks: {p['val_tracks']['supported_known']} supported-known +
  {p['val_tracks']['novel']} novel = {p['full_tracks']} non-distractor tracks.
- Repeated: {p['repeated_tracks']} tracks / {p['repeated_novel_categories']} novel cats.
- Balanced: {p['balanced_tracks']} tracks / {p['balanced_novel_categories']} novel cats.

### Protocol B: OV-assisted TrackOCD

- known = all {o['known_categories']} official TAO-OW known categories:
  {o['supported_known_categories']} supported-known + {o['zero_shot_known_categories']} zero-shot-known.
- novel = {o['novel_categories_total']} official unknown categories
  ({o['novel_categories_appearing_in_val']} appearing in val).
- val tracks: {o['val_tracks']['supported_known']} supported-known +
  {o['val_tracks']['zero_shot_known']} zero-shot-known +
  {o['val_tracks']['novel']} novel = {o['full_tracks']}.

## Freezing

- All public/private/splits/stats files and the evaluator are SHA256-pinned in
  `data/trackocd_v1/manifests/manifest_v1.0.json`.
- Any evaluator or protocol change requires a version bump (v1.1+); v1.0
  files are never silently overwritten.
- Known semantic ids never participate in the novel Hungarian matching
  (see EVALUATION_PROTOCOL.md).
""")

    for proto, ds in (("PURE", p), ("OV_ASSISTED", o)):
        (DOCS / f"DATASET_CARD_{proto}.md").write_text(f"""# Dataset Card: {proto} TrackOCD

Version: {VERSION}

| item | value |
|---|---:|
| supported-known categories | {ds['supported_known_categories']} |
| zero-shot-known categories | {ds['zero_shot_known_categories']} |
| novel categories (total) | {ds['novel_categories_total']} |
| novel categories appearing in val | {ds['novel_categories_appearing_in_val']} |
| val supported-known tracks | {ds['val_tracks'].get('supported_known', 0)} |
| val zero-shot-known tracks | {ds['val_tracks'].get('zero_shot_known', 0)} |
| val novel tracks | {ds['val_tracks'].get('novel', 0)} |
| full tracks | {ds['full_tracks']} |
| repeated tracks | {ds['repeated_tracks']} |
| balanced tracks | {ds['balanced_tracks']} |
| novel singleton categories | {ds['novel_singleton_categories']} |
| novel cross-video categories | {ds['novel_cross_video_categories']} |

Public streams contain no novel category ids/names, no novel category counts
and no future stream information. Private labels carry only
`ground_truth_category_id` and `protocol_role` and are readable only by the
evaluator. Image paths are validated against `data/raw/tao/frames`.
""")

    (DOCS / "EVALUATION_PROTOCOL.md").write_text("""# TrackOCD-v1.0 Evaluation Protocol

## Prediction label space

```json
{"prediction_type": "known", "semantic_category_id": 12}
{"prediction_type": "novel", "virtual_category_id": 37}
{"prediction_type": "unresolved"}
```

## Correctness rules

- Known GT track: correct iff `known` with the exact semantic category id.
  wrong-known, known->novel, and known->unresolved are all errors.
- Novel GT track: only `novel` predictions enter the novel contingency matrix.
  Hungarian matching is applied exclusively between predicted novel virtual
  ids and GT novel semantic categories. novel->known and novel->unresolved
  are routing errors and never enter Hungarian.
- Known semantic ids are never re-mapped by Hungarian.

## Reported metrics

Known: Supported-Known ACC, Zero-Shot-Known ACC (OV), Overall Known ACC,
Known-to-Novel Error, Known Misclassification Rate, Known Unresolved Rate.

Routing: Novel Routing Recall/Precision, False-Known Absorption Rate,
Unresolved Novel Rate.

Discovery: Route-aware Novel ACC (denominator all novel), Conditional Novel
ACC (denominator correctly routed novel), Novel-only NMI/ARI (routed subset),
Macro Novel Class ACC, predicted novel count / count error, mean
fragmentation, merge error, duplicate creation rate, mean assignment delay.

Overall: All Track ACC, macro known/novel harmonic mean, memory size,
inference time.

Predicted-track mode adds: GT/known/novel coverage, predicted-track
precision, End-to-End Correct Novel Track Rate.

## Isolation

- Training/online code never reads `data/trackocd_v1/*/private/`.
- No threshold is chosen with validation unknown labels or category counts.
- Router parameters are chosen only on the train-known proxy task.
""")
    print("docs written")


if __name__ == "__main__":
    main()
