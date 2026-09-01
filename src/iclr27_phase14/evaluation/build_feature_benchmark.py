"""Assemble the Phase 14 frozen-representation comparison table."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def read(path):
    return json.loads((ROOT / path).read_text())


def strict(path):
    x = read(path)
    return x.get("strict", x)


def geom(g, name):
    x = g["geometry"].get(name, {})
    return x.get("all_gt_novel", {}) if isinstance(x, dict) else {}


def main():
    g = read("outputs/iclr27_phase14/eval/q1_feature_geometry.json")
    rows = []

    def add(name, source, geometry_name=None, strict_path=None, status="measured"):
        item = {"representation": name, "source": source, "status": status}
        if geometry_name:
            item["geometry_full_q1_novel"] = geom(g, geometry_name)
        if strict_path:
            s = strict(strict_path)
            item["q1_strict"] = {k: s.get(k) for k in (
                "known_occurrence_acc", "first_novel_birth_acc",
                "novel_reuse_acc", "cross_physical_reuse_acc",
                "novel_nmi", "novel_ari", "n_born_novel_states")}
            item["acceptance"] = {
                "known_ge_0_60": bool(s.get("known_occurrence_acc", 0.0) >= 0.60),
                "ct_reuse_gt_0": bool(s.get("cross_physical_reuse_acc", 0.0) > 0.0),
                "joint_pass": bool(s.get("known_occurrence_acc", 0.0) >= 0.60 and
                                    s.get("cross_physical_reuse_acc", 0.0) > 0.0),
            }
        rows.append(item)

    add("DINOv2 crop mean", "local corrected Q1 DINOv2 cache", "dino_v2",
        "outputs/iclr27_phase14/eval/baseline_check/strict/summary.json")
    add("Frozen TSE (DINOv2 -> 128-D)", "Phase-6C TSE checkpoint", "tse",
        "outputs/iclr27_phase14/eval/baseline_check/strict/summary.json")
    add("Phase-8A B trajectory embedding", "Phase-8A B adapter checkpoint", "phase8a_b",
        "outputs/iclr27_phase14/eval/baseline_check/strict/summary.json")
    add("OpenAI CLIP ViT-B/32 crop + causal probe", "Phase-11 frozen CLIP probe",
        "clip_vit_b32", "outputs/iclr27_phase11/eval/clip_b_strict/summary.json")
    add("Phase-10 hybrid trajectory probe", "Phase-10 frozen B replay",
        None, "outputs/iclr27_phase10/eval/hybrid_small_q1/dev_strict/summary.json")
    add("Phase-13 real-TAO appearance+motion GRU", "Phase-13 frozen B replay",
        None, "outputs/iclr27_phase13/eval/full_strict/summary.json")
    add("DINOv3 CLS crop mean", "local DINOv3 W4 cache + frozen TSE/B",
        "dino_v3_cls", "outputs/iclr27_phase14/eval/dinov3_b_strict/summary.json")
    add("DINOv3 pooled crop mean", "local DINOv3 W4 cache",
        "dino_v3_pooled", status="geometry_only_no_b_replay")

    unavailable = [
        ("InternVideo2/2.5/3", "official repository/checkpoints not locally cached"),
        ("StreamFormer", "official code found; no local checkpoint used in audit"),
        ("MoSiC", "official code/checkpoints found online; no local Q1 feature cache"),
        ("TrackVerse object-centric", "dataset/code found online; no local trained encoder"),
        ("VESSA object-centric adaptation", "official code found online; no local checkpoint"),
        ("SRL video object-centric slots", "official code/checkpoints online; no local Q1 feature cache"),
        ("Trace Anything trajectory fields", "official code/weights online; no local Q1 feature cache"),
        ("TRACT tracking-aware TraCLIP", "repository present but no Q1 feature extraction run"),
        ("BYOV cross-view video", "official code online; training data/checkpoint not local"),
        ("OFCL open-world continual", "official code online; not a crop trajectory encoder"),
    ]
    for name, reason in unavailable:
        rows.append({"representation": name, "status": "not_run",
                     "reason": reason, "q1_strict": None})

    out = {
        "protocol": {
            "frozen_decision": "Phase-8A B CreateHead + TorchSemanticStateSet",
            "strict_evaluator": "src/iclr27_phase7a/evaluation/strict_eval_any.py",
            "acceptance": "Known >= 0.60 and CT-Reuse > 0",
            "q1_labels_used_for_features_or_actions": False,
            "private_gt_used_only_posthoc": True,
            "future_frames_used": False,
            "physical_id_used_as_semantic_feature": False,
        },
        "rows": rows,
    }
    path = ROOT / "outputs/iclr27_phase14/eval/feature_benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
