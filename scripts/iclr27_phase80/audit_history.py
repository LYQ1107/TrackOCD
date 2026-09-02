#!/usr/bin/env python3
"""Materialize the Phase80 historical feature-source audit."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase80/audit/historical_feature_audit.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()


def summary_rows() -> list[dict[str, str]]:
    path = ROOT / "outputs/dinov3_bakeoff/metrics/backbone_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keep = {"V0", "V2", "O0", "O1"}
    return [r for r in rows if r.get("protocol") == "pure" and r.get("subset") == "full" and r.get("method") in keep]


def main() -> None:
    status = (ROOT / "runs/dinov3_bakeoff/status.txt").read_text(encoding="utf-8").strip()
    gate = json.loads((ROOT / "runs/dinov3_bakeoff/backbone_gate.json").read_text(encoding="utf-8"))
    weight = ROOT / "outputs/dinov3_bakeoff/tests/weight_integrity.json"
    test_report = json.loads((ROOT / "outputs/dinov3_bakeoff/tests/test_report.json").read_text(encoding="utf-8"))
    result = {
        "phase": "Phase80",
        "stage": "stage0_historical_feature_audit",
        "source_commit": git_head(),
        "historical_bakeoff": {
            "status": status,
            "gate": gate,
            "metrics_csv": str((ROOT / "outputs/dinov3_bakeoff/metrics/backbone_summary.csv").resolve()),
            "pure_full_rows": summary_rows(),
            "tests": test_report,
            "weight_integrity": json.loads(weight.read_text(encoding="utf-8")),
            "weight_integrity_sha256": sha(weight),
            "scripts": [
                str((ROOT / "scripts/compare_meta_timm_dinov3.py").resolve()),
                str((ROOT / "scripts/evaluate_dinov3_bakeoff.sh").resolve()),
                str((ROOT / "src/dinov3_bakeoff/extract.py").resolve()),
            ],
            "interpretation": {
                "old_output": "global CLS or timm pooled track embeddings; no dense patch-token cache was retained",
                "official_equivalence": False,
                "reason_not_continued": "NO_CLEAR_DINOV3_GAIN; route-aware novel accuracy and conditional accuracy did not meet the registered gate",
                "protocol_comparability": "historical bakeoff uses its own pure/ov-assisted track stream and is not Phase30 TRAIN-disjoint retrieval; it is a provenance audit, not a rerun",
            },
        },
        "phase80_primary_selection": {
            "method": "DINOv3 ViT-B/16 dense crop evidence",
            "weight_distribution": "TIMM_DISTRIBUTION",
            "model_id": "timm/vit_base_patch16_dinov3.lvd1689m",
            "feature_outputs": ["normalized CLS", "32 normalized patch tokens from fixed 4x8 spatial grid"],
            "causal_usage": "track rows are ordered by existing Phase30 event/frame order; prefix uses only rows at or before the requested prefix",
            "why_selected": "adds local/part-level evidence absent from the retained Phase15S global-only cache and is available in the existing environment",
            "not_claimed": ["Meta-official byte equivalence", "new backbone superiority", "controller or sealed success"],
        },
        "official_sources": [
            {
                "name": "DINOv3",
                "paper": "https://arxiv.org/abs/2508.10104",
                "repo": "https://github.com/facebookresearch/dinov3",
                "commit": "6876159a11b4df116f30f667f8c9888617df0751",
                "license": "DINOv3 License (LICENSE.md, updated 2025-08-19)",
                "year": 2025,
                "dense_tokens_available": True,
                "causal_compatible": True,
                "requires_text": False,
                "reuse": "visual backbone patch tokens only",
                "rejected": "text-aligned dino.txt heads and category inference are outside TrackOCD inference contract",
            },
            {
                "name": "Grounded Correspondence",
                "paper": "https://arxiv.org/abs/2605.03650",
                "repo": "https://github.com/LiZhYun/ICML2026-RethinkingOCL",
                "commit": "5d345268797425558b449337519af3ab24aeb6f1",
                "license": "repository license recorded in prior Phase75C audit; no code/weights imported",
                "year": 2026,
                "dense_tokens_available": False,
                "causal_compatible": "partial",
                "requires_text": False,
                "reuse": "correspondence/matching idea only",
                "rejected": "does not provide the new visual source required by this family; prior Phase75C result is retained",
            },
            {
                "name": "TRACT",
                "paper": "https://openaccess.thecvf.com/content/ICCV2025/html/Li_TRACT_ICCV_2025_paper.html",
                "repo": "https://github.com/Nathan-Li123/TRACT",
                "commit": "19f01d72f9f6c212c28fd9cb0171a5432cd41a6a",
                "license": "not imported; license left as repository-specific and unverified here",
                "year": 2025,
                "dense_tokens_available": False,
                "causal_compatible": "partial",
                "requires_text": False,
                "reuse": "trajectory-aware reference only",
                "rejected": "does not provide a drop-in dense crop source for the frozen Phase30 protocol",
            },
            {
                "name": "ObjectRelator",
                "paper": "https://openaccess.thecvf.com/content/ICCV2025/html/Qian_ObjectRelator_ICCV_2025_paper.html",
                "repo": "https://github.com/insait-institute/ObjectRelator",
                "commit": "25ecbc086cc812304de97764aa21f4bb8e0e6360",
                "license": "repository-specific; no code/weights imported",
                "year": 2025,
                "dense_tokens_available": False,
                "causal_compatible": "partial",
                "requires_text": "paper/repo-specific; not used",
                "year": 2025,
                "reuse": "static correspondence reference only",
                "rejected": "cross-view correspondence is not the visual source renewal itself and does not model TrackOCD online lifecycle",
            },
        ],
        "sealed_boundaries": {"devplus": False, "q1": False, "public_new": False, "held_labels": False, "future_rows": False},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gate": gate.get("status"), "primary": result["phase80_primary_selection"]["method"]}, indent=2))


if __name__ == "__main__":
    main()

