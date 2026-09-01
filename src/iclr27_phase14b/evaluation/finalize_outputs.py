"""Write compact Phase-14B protocol, causal, and operating-point records."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs/iclr27_phase14b"


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


opp_path = OUT / "eval/opportunity_audit.json"
opp = json.loads(opp_path.read_text())
opp["devplus_feature_coverage"] = {
    "available_tracks": 193,
    "missing_sample_ids": [],
    "total_tracks": 193,
    "source": "canonical causal raw DINOv2/CLIP/DINOv3 GT-box extraction",
}
opp["proposal_view"] = {
    "status": "unavailable",
    "proposal_aligned_tracks": 0,
    "proposal_aligned_cross_physical_pairs": 0,
    "proposal_aligned_cross_video_pairs": 0,
    "reason": "No compatible corrected TAO TRAIN detector/tracker proposal stream was completed for the locked 130-video DEV+ population. Existing OVTR/COVTrack public pipelines target TAO validation/test formats and require incompatible legacy environments; SimOWT is a detector/tracker pipeline requiring a separate class-agnostic proposal audit. GT boxes are retained only as diagnostic and no synthetic proposal noise or duplicate pairs were introduced.",
    "checked": [
        "third_party/research_refs_phase4n/OVTR (public checkpoint loaded; runtime integration smoke blocked before proposal output)",
        "third_party/research_refs_phase4n/COVTrack (public checkpoint environment blocked before model construction)",
        "third_party/SimOWT (public tracker requires separate TAO TRAIN proposal registration)",
    ],
}
atomic(opp_path, opp)

bench = json.loads((OUT / "eval/foundation_feature_benchmark.json").read_text())
rows = {}
for name, candidate in bench["candidates"].items():
    m = candidate["prefixes"]["16"]
    rows[name] = {
        "status": candidate["status"],
        "prefix": 16,
        "cross_video_r1": m["cross_video_recall_at_1"],
        "cross_video_r5": m["cross_video_recall_at_5"],
        "cross_video_map": m["cross_video_map"],
        "prototype": m["prototype_accuracy"],
        "distance_gap": m["distance_gap_different_minus_same"],
        "category_macro_cross_video_r1": m["category_macro"]["cross_video_r1"],
        "video_grouped_cross_video_r1": m["video_grouped_cross_video_r1"],
    }

summary = {
    "protocol": "docs/iclr27_phase14b/PROTOCOL.md",
    "q1_used": False,
    "view": "GT-box diagnostic only",
    "primary_proposal_view": {
        "status": "unavailable",
        "strict_evaluator_run": False,
        "reason": opp["proposal_view"]["reason"],
    },
    "frozen_phase8a_b_operating_point": {
        "status": "not_run",
        "known_occurrence_accuracy": None,
        "ct_reuse": None,
        "reason": "The exact B replay requires the primary corrected proposal stream; GT-box diagnostics cannot be presented as the system result.",
    },
    "train_only_score_normalization": {
        "status": "not_run",
        "reason": "No primary proposal stream; raw head-agnostic metrics are retained and no DEV+/Q1 operating-point tuning was performed.",
        "procedure_registered": "one shared TRAIN-only normalization across candidates, no DEV+/Q1 threshold search",
    },
    "legacy_gate": {
        "known_occurrence_accuracy": None,
        "ct_reuse": None,
        "pass": False,
        "reason": "not evaluated on primary view; no positive TrackOCD system claim",
    },
    "gt_box_foundation_summary_prefix16": rows,
    "learnability_controls": str(OUT / "eval/learnability_controls.json"),
    "proposal_opportunity": {
        "target_cross_physical_pairs": 100,
        "target_cross_video_pairs": 30,
        "measured_proposal_cross_physical_pairs": 0,
        "measured_proposal_cross_video_pairs": 0,
    },
}
atomic(OUT / "eval/trackocd_devplus_summary.json", summary)

contract = {
    "protocol": "docs/iclr27_phase14b/PROTOCOL.md",
    "q1_used": False,
    "future_frames_used": False,
    "physical_id_used_as_feature": False,
    "private_gt_used_for_feature_extraction": False,
    "devplus_category_labels_used_for_feature_extraction": False,
    "devplus_category_labels_used_for_candidate_selection": False,
    "feature_cache_label_free": True,
    "physical_id_semantic_id_separated": True,
    "causal_prefix_sampling": [1, 2, 4, 8, 16],
    "crop_context": "current box plus 10 percent context, clipped to current frame",
    "strict_evaluator_modified": False,
    "semantic_memory_modified": False,
    "primary_proposal_view": "unavailable; GT-box view is diagnostic only",
    "oracle_labels": "DEV+ offline labels used only in explicitly illegal oracle control",
    "supervised_diagnostic_labels": "public representation-training labels only; category_label_used=true",
    "artifacts": {
        "manifest": str(OUT / "manifests/devplus_tracks.jsonl"),
        "opportunity_audit": str(OUT / "eval/opportunity_audit.json"),
        "foundation_benchmark": str(OUT / "eval/foundation_feature_benchmark.json"),
        "learnability_controls": str(OUT / "eval/learnability_controls.json"),
    },
}
atomic(OUT / "eval/causal_contract.json", contract)
print("wrote opportunity, trackocd summary, and causal contract")
