"""Phase 4N open-source audit (detector / objectness / calibration)."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs" / "iclr27_phase4n" / "open_source"
DOC = ROOT / "docs" / "iclr27_phase4n"

INVENTORY = [
    {
        "method": "OWOBJ",
        "paper": "Open-World Objectness Modeling Unifies Novel Object Detection",
        "year": 2025, "venue": "CVPR",
        "repo": "AI4Math-ShanZhang/OWOBJ",
        "commit": "f5c583e39593168e2313c149ba69801d79619f42",
        "license": "Apache-2.0",
        "detector_arch": "OW-DETR-style transformer (objectness branch)",
        "open_world": "yes", "open_vocabulary": "no",
        "unknown_vs_background": "yes (objectness modeling)",
        "objectness": "yes", "calibration": "no", "track_integration": "no",
        "strict_online": "yes (inference)",
        "weights_available": "checkpoint in repo config (COCO OWOD)",
        "dataset_requirements": "COCO-OWOD for training; inference on images",
        "can_run_on_tao": "possible with conversion; not verified",
        "can_export_detections": "yes (bbox + objectness)",
        "preserve_association_interface": "n/a (detector-only)",
        "relevant_files": "configs/; models/; README",
        "relevant_functions": "objectness head; unknown-vs-background score",
        "direct_candidate": "yes (objectness-aware frontend)",
        "why": "2025 official implementation with objectness; needs weight download and TAO conversion",
    },
    {
        "method": "YOLO-UniOW",
        "paper": "YOLO-UniOW: Efficient Universal Open-World Object Detection",
        "year": 2024, "venue": "arXiv 2412.20645",
        "repo": "THU-MIG/YOLO-UniOW",
        "commit": "0061cdec3b50a60208dbe1b66268e886af92d2fa",
        "license": "GPL-3.0",
        "detector_arch": "YOLO-based universal open-world detector",
        "open_world": "yes", "open_vocabulary": "yes",
        "unknown_vs_background": "yes (unknown flagging)",
        "objectness": "yes", "calibration": "no", "track_integration": "no",
        "strict_online": "yes (inference)",
        "weights_available": "pretrained S/M/L on LVIS minival",
        "dataset_requirements": "LVIS/COCO-style; inference on images",
        "can_run_on_tao": "possible with conversion; not verified",
        "can_export_detections": "yes",
        "preserve_association_interface": "n/a",
        "relevant_files": "configs/; demo/; README",
        "relevant_functions": "unknown branch; universal detection head",
        "direct_candidate": "candidate",
        "why": "2024/2025 open-world detector; GPL-3 license limits code reuse but inference ok",
    },
    {
        "method": "OW-OVD",
        "paper": "OW-OVD: Unified Open World and Open Vocabulary Object Detection",
        "year": 2025, "venue": "CVPR",
        "repo": "xxyzll/OW_OVD",
        "commit": "2279742416c5a4b4b4e46d023d0cf652b59f0dce",
        "license": "none detected",
        "detector_arch": "YOLO-World-based unified OVD+OWOD",
        "open_world": "yes", "open_vocabulary": "yes",
        "unknown_vs_background": "yes",
        "objectness": "partial", "calibration": "no",
        "track_integration": "no", "strict_online": "yes (inference)",
        "weights_available": "not verified in repo",
        "dataset_requirements": "COCO-OWOD / OV-COCO style",
        "can_run_on_tao": "not verified",
        "can_export_detections": "yes",
        "preserve_association_interface": "n/a",
        "relevant_files": "configs/; yolo_world/",
        "relevant_functions": "unified known/unknown detection",
        "direct_candidate": "candidate",
        "why": "2025 unified open-world+open-vocab; no license file, weights unverified",
    },
    {
        "method": "YOLOE",
        "paper": "YOLOE: Real-Time Seeing Anything",
        "year": 2025, "venue": "CVPR",
        "repo": "THU-MIG/yoloe",
        "commit": "40cd606cabdbe2b566d6f14a6b162c89206e9a1b",
        "license": "AGPL-3.0",
        "detector_arch": "YOLO-based open-vocabulary detector (text/visual prompts)",
        "open_world": "partial (promptable)", "open_vocabulary": "yes",
        "unknown_vs_background": "no dedicated unknown branch",
        "objectness": "yes (promptable)", "calibration": "no",
        "track_integration": "no", "strict_online": "yes",
        "weights_available": "yes (repo/weights links)",
        "dataset_requirements": "COCO-style inference",
        "can_run_on_tao": "possible with image inference + class prompts",
        "can_export_detections": "yes",
        "preserve_association_interface": "n/a",
        "relevant_files": "yoloe/; configs/; app.py",
        "relevant_functions": "text-prompted detection; region-prompt mode",
        "direct_candidate": "candidate (OV frontend)",
        "why": "practical open-vocab detector with weights; AGPL limits code reuse; no unknown-vs-background branch",
    },
    {
        "method": "OmDet",
        "paper": "OmDet: Real-time and accurate open-vocabulary end-to-end object detection",
        "year": "2024-2026", "venue": "arXiv 2403.06892",
        "repo": "om-ai-lab/OmDet",
        "commit": "956e2f36a13c32bd1e30b14a790233268c2305fb",
        "license": "Apache-2.0",
        "detector_arch": "DETR-style open-vocabulary detector",
        "open_world": "partial", "open_vocabulary": "yes",
        "unknown_vs_background": "no",
        "objectness": "partial", "calibration": "no",
        "track_integration": "no", "strict_online": "yes",
        "weights_available": "yes (HuggingFace)",
        "dataset_requirements": "COCO/LVIS-style inference",
        "can_run_on_tao": "possible",
        "can_export_detections": "yes",
        "preserve_association_interface": "n/a",
        "relevant_files": "configs/; docs/",
        "relevant_functions": "open-vocab head; turbo variants",
        "direct_candidate": "candidate (OV frontend)",
        "why": "Apache-2.0 and weights available; no unknown-vs-background semantics",
    },
    {
        "method": "DetSeg",
        "paper": "DetSeg: Bounding the OoD Objects in Road Scenes",
        "year": 2025, "venue": "ICCV",
        "repo": "huachao0124/DetSeg-official",
        "commit": "37bc4ad459b81a8ef86c64f2661bec16ff6cd9bd",
        "license": "Apache-2.0 (OpenMMLab)",
        "detector_arch": "MMDetection-based detector+segmenter",
        "open_world": "partial (road OoD)", "open_vocabulary": "no",
        "unknown_vs_background": "yes (OoD object bounding)",
        "objectness": "yes", "calibration": "no", "track_integration": "no",
        "strict_online": "yes",
        "weights_available": "checkpoints dir present",
        "dataset_requirements": "road anomaly datasets",
        "can_run_on_tao": "unlikely without retraining",
        "can_export_detections": "yes",
        "preserve_association_interface": "n/a",
        "relevant_files": "ckpts/; configs/",
        "relevant_functions": "object-level OoD detection",
        "direct_candidate": "rejected for TrackOCD frontend",
        "why": "road-scene specific; not compatible with TAO open-world protocol",
    },
    {
        "method": "DualMem",
        "paper": "DualMem: Bypassing the Objectness Bottleneck for Calibrated Unknown-Stream Filtering",
        "year": 2026, "venue": "arXiv 2605.23634",
        "repo": "not found",
        "commit": "",
        "license": "unknown",
        "detector_arch": "post-hoc filter on frozen SigLIP features",
        "open_world": "yes", "open_vocabulary": "no",
        "unknown_vs_background": "yes (calibrated unknown-stream filter)",
        "objectness": "no", "calibration": "yes", "track_integration": "no",
        "strict_online": "partial (calibration split required)",
        "weights_available": "n/a",
        "dataset_requirements": "small image-disjoint annotated calibration split",
        "can_run_on_tao": "n/a",
        "can_export_detections": "n/a",
        "preserve_association_interface": "n/a",
        "relevant_files": "paper only",
        "relevant_functions": "positive/negative memory likelihood-ratio test",
        "direct_candidate": "principle only",
        "why": "highly relevant unknown-vs-background filter; official repo not located at audit time",
    },
    {
        "method": "OW-Rep",
        "paper": "OW-Rep: Open World Object Detection with Instance Representation Learning",
        "year": 2026, "venue": "WACV",
        "repo": "not found",
        "commit": "",
        "license": "unknown",
        "detector_arch": "OW-DETR + instance embedding modules",
        "open_world": "yes", "open_vocabulary": "no",
        "unknown_vs_background": "yes",
        "objectness": "yes", "calibration": "no", "track_integration": "no",
        "strict_online": "yes",
        "weights_available": "n/a",
        "dataset_requirements": "COCO-OWOD",
        "can_run_on_tao": "n/a",
        "can_export_detections": "n/a",
        "preserve_association_interface": "n/a",
        "relevant_files": "paper only",
        "relevant_functions": "unknown box refine; embedding transfer",
        "direct_candidate": "principle only",
        "why": "official repo not located at audit time",
    },
]

MATRIX = [
    ("OWOBJ", "yes", "no", "yes", "yes", "yes", "no", "objectness score",
     "direct candidate", "objectness-aware unknown-vs-background"),
    ("YOLO-UniOW", "yes", "yes", "yes", "yes", "yes", "no", "unknown branch",
     "candidate", "universal open-world detector"),
    ("OW-OVD", "yes", "yes", "yes", "partial", "yes", "no", "unified head",
     "candidate", "unified OVD+OWOD"),
    ("YOLOE", "yes", "yes", "no", "yes", "yes", "no", "prompt score",
     "candidate", "open-vocab frontend; no unknown branch"),
    ("OmDet", "yes", "yes", "no", "partial", "yes", "no", "open-vocab score",
     "candidate", "real-time OV detector"),
    ("DetSeg", "yes", "no", "yes", "yes", "yes", "no", "OoD objectness",
     "rejected", "road-scene specific"),
    ("DualMem", "yes", "no", "yes", "no", "no", "no", "likelihood-ratio",
     "principle only", "unknown-stream filter; repo not found"),
    ("OW-Rep", "yes", "no", "yes", "yes", "no", "no", "instance embedding",
     "principle only", "repo not found"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)
    fields = list(INVENTORY[0].keys())
    with open(OUT / "repository_inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(INVENTORY)
    with open(OUT / "mechanism_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["repo", "open_world", "open_vocabulary",
                    "unknown_vs_background", "objectness", "strict_online",
                    "track_integration", "signal", "verdict",
                    "phase4n_relevance"])
        w.writerows(MATRIX)
    (DOC / "OPEN_SOURCE_REPOSITORY_AUDIT.md").write_text(
        """# Phase 4N Open-Source Repository Audit

Scope: 2025-2026 open-world / open-vocabulary detection, objectness /
unknown-vs-background filtering, detection calibration, and open-world
tracking frontends.  Repositories were cloned and pinned in
`third_party/research_refs_phase4n/`; commits and licenses were read
from the clones.

## Verified repositories

| Repo | Year/Venue | License | Frontend relevance |
|---|---|---|---|
| OWOBJ | 2025 CVPR | Apache-2.0 | objectness-based unknown/background separation |
| YOLO-UniOW | 2024 arXiv | GPL-3.0 | universal open-world detector with unknown flagging |
| OW-OVD | 2025 CVPR | none detected | unified open-vocab + open-world detection |
| YOLOE | 2025 CVPR | AGPL-3.0 | promptable open-vocabulary detector |
| OmDet | 2024-2026 arXiv | Apache-2.0 | real-time open-vocabulary detector |
| DetSeg | 2025 ICCV | Apache-2.0 (OpenMMLab) | road OoD object bounding (rejected) |
| DualMem | 2026 arXiv | unknown | calibrated unknown-stream filter (repo not found) |
| OW-Rep | 2026 WACV | unknown | instance representation for OWOD (repo not found) |

## Conclusions

- Real 2025/2026 detector candidates with objectness / unknown
  separation exist (OWOBJ, YOLO-UniOW, OW-OVD).  All require weight
  downloads and TAO-format conversion before a detector-only benchmark.
- Open-vocabulary detectors (YOLOE, OmDet) have weights but no dedicated
  unknown-vs-background branch; they cannot replace TrackOCD's dynamic
  novel-identity discovery without an extra validity layer.
- DualMem (2026) is the closest *calibrated unknown-stream filter*
  principle; no official repo was located, so it is principle-only.
""")
    (DOC / "OPEN_SOURCE_IMPLEMENTATION_NOTES.md").write_text(
        """# Phase 4N Open-Source Implementation Notes

## OWOBJ (CVPR 2025)

Adds an open-world objectness modeling branch on top of an OW-DETR-style
detector to separate unknown objects from background.  This is the
closest architectural match for a validity-aware frontend.  If the
Phase 4N audit shows the frozen detector's FP stream cannot be fixed by
post-hoc validity evidence, OWOBJ is the first detector candidate to
benchmark detector-only (weights + TAO conversion required).

## YOLO-UniOW / OW-OVD

Both provide an explicit unknown detection branch (universal
open-world).  YOLO-UniOW is GPL-3.0 (inference-only use acceptable);
OW-OVD has no license file and unverified weights.  Either would need a
detector-only pass before TrackOCD integration.

## YOLOE / OmDet

Open-vocabulary frontends with usable weights; they output text-prompted
classes and cannot produce TrackOCD's dynamic novel IDs by themselves.
They could only serve as proposal generators plus a validity score, and
are therefore secondary candidates.

## DualMem (2026, principle)

A calibrated post-hoc filter using positive/negative memories and a
likelihood-ratio test in frozen feature space.  Its calibration-split
assumption conflicts with TrackOCD's strict no-held-out-labels rule; any
adaptation must use train/dev calibration only.

## No code copied

All repositories were reviewed for mechanisms only; no detector code or
weights were integrated without a pass gate.
""")
    print("OPEN_SOURCE_AUDIT_DONE", len(INVENTORY))


if __name__ == "__main__":
    main()
