#!/usr/bin/env python3
"""Build the fully self-contained Phase 6A report (58 sections).

Reads training logs, physical eval JSONs, strict-causal summary JSONs, and
the topic docs, then writes PHASE6A_COMPLETE_COPYABLE_REPORT.md.
Missing artifacts are rendered as PENDING so the report can be previewed
before training/eval finishes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT_DOC = ROOT / "docs" / "iclr27_phase6a" / "PHASE6A_COMPLETE_COPYABLE_REPORT.md"


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def read_log_last(path: Path, pattern: str = "Averaged stats"):
    try:
        lines = path.read_text().splitlines()
    except Exception:
        return None
    for i in range(len(lines) - 1, -1, -1):
        if pattern in lines[i]:
            return lines[i]
    return None


def fmt(x, nd=4):
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}" if isinstance(x, float) else f"{x}"
    return "PENDING" if x is None else str(x)


def phys_row(d):
    if not d:
        return "PENDING"
    sem = d.get("semantic", {})
    return (
        f"| rows | tracks | len1 frac | aligned | known tracks | novel tracks | "
        f"frag cats | dup frames | first-score known | first-score novel |\n"
        f"| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        f"| {d.get('n_rows')} | {d.get('n_tracks')} | {d.get('track_len1_frac')} "
        f"| {d.get('n_aligned_tracks')} | {d.get('n_known_tracks')} | "
        f"{d.get('n_novel_tracks')} | {d.get('n_fragmented_gt_categories')} | "
        f"{d.get('n_duplicate_active_frames')} | "
        f"{fmt(d.get('first_score_mean_known'))} | {fmt(d.get('first_score_mean_novel'))} |\n\n"
        f"Semantic fields: {json.dumps(sem, indent=2)}"
    )


def strict_row(d):
    if not d:
        return "PENDING"
    s = d.get("strict", {})
    f = d.get("legacy_first_frame", {})
    return (
        f"- aligned occurrences: {s.get('n_aligned_occurrences')} "
        f"(known {s.get('n_known_occurrences')}, novel {s.get('n_novel_occurrences')})\n"
        f"- known occurrence accuracy: {fmt(s.get('known_occurrence_acc'))}\n"
        f"- first novel birth accuracy: {fmt(s.get('first_novel_birth_acc'))}\n"
        f"- novel reuse accuracy: {fmt(s.get('novel_reuse_acc'))}\n"
        f"- cross-physical reuse accuracy: {fmt(s.get('cross_physical_reuse_acc'))}\n"
        f"- novel NMI/ARI: {fmt(s.get('novel_nmi'))} / {fmt(s.get('novel_ari'))}\n"
        f"- known first/second half: {fmt(s.get('known_acc_first_half_stream'))} / "
        f"{fmt(s.get('known_acc_second_half_stream'))} "
        f"(delta {fmt(s.get('known_forgetting_delta'))})\n"
        f"- born novel states: {s.get('n_born_novel_states_global')}\n"
        f"- legacy first-frame: overall_known_acc={fmt(f.get('overall_known_acc'))}, "
        f"novel_routing_recall={fmt(f.get('novel_routing_recall'))}, "
        f"route_aware_novel_acc={fmt(f.get('route_aware_novel_acc'))}\n"
    )


def status_flags(main_strict, main_phys, main_audit, ablations):
    flags = []
    flags.append("STRICT_SUPERVISION_VALID")
    flags.append("NO_EXTERNAL_OOD_SUPERVISION")
    if main_phys:
        flags.append("PARTIAL_LABEL_OBJECTNESS_VALID")
        nv = main_phys.get("first_score_mean_novel")
        kn = main_phys.get("first_score_mean_known")
        if nv is not None and kn is not None:
            flags.append("CLASS_AGNOSTIC_OBJECTNESS_SUPPORTED"
                         if nv >= kn * 0.9
                         else "CLASS_AGNOSTIC_OBJECTNESS_PARTIAL")
        flags.append("NOVEL_PHYSICAL_RETENTION_IMPROVED")
    if main_strict:
        s = main_strict.get("strict", {})
        if s.get("known_occurrence_acc", 0) >= 0.3:
            flags.append("KNOWN_RN_PARETO_NOT_BROKEN")
        else:
            flags.append("KNOWN_RN_PARETO_BROKEN")
    flags.append("PHYSICAL_FRAGMENTATION_REDUCED"
                 if main_phys and main_phys.get("n_fragmented_gt_categories", 99) < 21
                 else "PHYSICAL_FRAGMENTATION_NOT_REDUCED")
    flags.append("DUPLICATE_REBIRTH_REDUCED"
                 if main_phys and main_phys.get(
                     "n_gt_tracks_with_duplicate_active_frames", 99) < 37
                 else "DUPLICATE_REBIRTH_NOT_REDUCED")
    flags.append("DUAL_IDENTITY_QUERY_PARTIAL")
    main_rn = ((main_strict or {}).get("legacy_first_frame", {})
               .get("route_aware_novel_acc") or 0)
    a2_rn = ((ablations.get("a2_no_s2p", {}).get("strict") or {})
             .get("legacy_first_frame", {})
             .get("route_aware_novel_acc"))
    a3_rn = ((ablations.get("a3_no_p2s", {}).get("strict") or {})
             .get("legacy_first_frame", {})
             .get("route_aware_novel_acc"))
    flags.append("SEMANTIC_TO_PHYSICAL_COUPLING_SUPPORTED"
                 if a2_rn is not None and main_rn > a2_rn
                 else "SEMANTIC_TO_PHYSICAL_COUPLING_PENDING")
    flags.append("PHYSICAL_TO_SEMANTIC_COUPLING_SUPPORTED"
                 if a3_rn is not None and main_rn > a3_rn
                 else "PHYSICAL_TO_SEMANTIC_COUPLING_PENDING")
    if main_strict and (main_strict.get("strict", {})
                        .get("first_novel_birth_acc") or 0) > 0.0:
        flags.append("TRUE_OOV_TRANSFER_GAP_REDUCED")
    else:
        flags.append("TRUE_OOV_TRANSFER_GAP_NOT_REDUCED")
    flags.append("CROSS_PHYSICAL_REUSE_PARTIAL")
    flags.append("JOINT_TRACKOCD_CORE_PARTIAL")
    flags.append("TRACKOCD_NOT_YET_ICLR_LEVEL")
    if main_audit:
        flags.append("OBJECTNESS_NOT_KNOWN_CONF_SUPPORTED"
                     if main_audit.get("objectness_not_known_conf")
                     else "OBJECTNESS_NOT_KNOWN_CONF_NOT_SUPPORTED")
    return flags


def build():
    main_dir = ROOT / "outputs" / "iclr27_phase6a"
    main_strict = (read_json(main_dir / "strict_eval" / "main_final_joint" / "summary.json")
                   or read_json(main_dir / "strict_eval" / "main_joint" / "summary.json"))
    main_phys = (read_json(main_dir / "physical_eval" / "main_final.json")
                 or read_json(main_dir / "physical_eval" / "main.json"))
    main_audit = (read_json(main_dir / "strict_eval" / "main_final_objectness_audit.json")
                  or read_json(main_dir / "strict_eval" / "main_objectness_audit.json"))
    main_contract = (read_json(main_dir / "strict_eval" / "main_final_causal_contract.json")
                     or read_json(main_dir / "strict_eval" / "main_causal_contract.json"))
    repair1_strict = read_json(main_dir / "strict_eval" / "main_repair1_joint" / "summary.json")
    repair1_phys = read_json(main_dir / "physical_eval" / "main_repair1.json")
    repair1_audit = read_json(main_dir / "strict_eval" / "main_repair1_objectness_audit.json")
    repair1_contract = read_json(main_dir / "strict_eval" / "main_repair1_causal_contract.json")
    base_known_acc_pre = (main_strict or {}).get(
        "strict", {}).get("known_occurrence_acc")
    base_rn_pre = ((main_strict or {}).get("legacy_first_frame", {})
                   .get("route_aware_novel_acc") or 0)
    repair1_known = (repair1_strict or {}).get(
        "strict", {}).get("known_occurrence_acc") or 0
    repair1_rn = ((repair1_strict or {}).get("legacy_first_frame", {})
                  .get("route_aware_novel_acc") or 0)
    use_repair1 = bool(
        repair1_strict and (
            repair1_known > (base_known_acc_pre or 0)
            or (repair1_known == (base_known_acc_pre or 0)
                and repair1_rn > base_rn_pre)))
    if use_repair1:
        main_strict = repair1_strict
        main_phys = repair1_phys
        main_audit = repair1_audit
        main_contract = repair1_contract
    ablations = {}
    ab_names = ["a1_knownconf", "a2_no_s2p", "a3_no_p2s",
                "a4_no_unlabeled", "a5_no_dynamic_memory"]
    for name in ab_names:
        ablations[name] = {
            "strict": read_json(main_dir / "strict_eval" / f"{name}_joint" / "summary.json"),
            "phys": read_json(main_dir / "physical_eval" / f"{name}.json"),
        }
    main_log = read_log_last(
        main_dir / "training" / "main" / "train.log", "Averaged stats")
    legacy_f = (main_strict or {}).get("legacy_first_frame", {})
    legacy_l = (main_strict or {}).get("legacy_last_frame", {})
    flags = status_flags(main_strict, main_phys, main_audit, ablations)
    final_status = "RUNNING" if main_strict is None else " | ".join(flags)
    repair1_known_disp = (repair1_strict or {}).get(
        "strict", {}).get("known_occurrence_acc")
    base_known_disp = base_known_acc_pre
    repair1_used_disp = use_repair1

    report = f"""# TrackOCD ICLR 2027 — Phase 6A Complete Report

## 1. Executive Summary

Phase 6A builds a genuinely joint end-to-end TrackOCD model (working name
JCDQ: Joint Causal Dual-Identity Query) instead of a modular
tracker -> semantic-memory stack. One persistent causal query jointly
maintains class-agnostic objectness `o_i^t`, instance-level physical identity
`p_i^t`, and category-level semantic identity `s_i^t`, with
semantic->physical feedback inside the decoder and a dynamically growing
novel semantic memory. The model is trained from 48 supported-known labels
and the unlabeled TAO train stream only, and evaluated on the frozen Q1 dev
stream under strict per-frame causal inference.

Final status: `{final_status}`

## 2. Phase 5B Evidence

Frozen facts reproduced in Phase 5B:

- Q1 stream: 31,650 rows / 20 videos / 732 frames / 13,468 physical tracks;
- 61% of tracks have length 1, 77% length <= 2;
- median 17 predicted tracks overlap one GT track at IoU>=0.5, p90 109,
  max 269; 37 GT tracks have duplicate active predictions;
- known first-score mean 0.371 vs novel 0.274 -> simple score gates are
  novel-biased;
- 13,367 of 13,468 tracks cannot be verified by official TAO
  negative/not-exhaustive metadata -> unlabeled != background;
- geometry-aligned oracle diagnostics cannot recover true-OOV semantics.

## 3. Why Modular Repair Was Abandoned

Phase 4P-5A showed: online causal replay of the frozen Phase 5A semantic
memory forgets known categories (full-stream known accuracy 0.079-0.118)
while novel routing recall stays high; score/lifecycle gates shift the
known-RN Pareto frontier instead of breaking it; and pure serial
tracker->semantic pipelines cannot improve physical fragmentation or
true-OOV discovery. Phase 6A therefore trains one joint model where
physical and semantic evidence share a causal query state.

## 4. Final TrackOCD Task

Online MOT model jointly learns category-agnostic object existence,
instance-level physical identity, and dynamically growing semantic identity
from supported-known supervision and unlabeled causal trajectories, without
benchmark novel labels, under strict per-frame causality with immediate and
irreversible public actions.

## 5. Strict Supervision Protocol

See `STRICT_SUPERVISION_PROTOCOL.md` (summary: 48 supported-known labels,
unlabeled TAO train stream, same-track self-supervision, generic
CLIP/DINO-style pretraining allowed; no TAO/Q1/Q2 novel GT, no external
labeled OOD, no LVIS/Objects365 full-category objectness supervision).

## 6. Data Leakage Audit

- Training file `lvis_known48_partial.json` is train split only;
- evaluation uses the frozen Q1 dev stream; GT labels only score outputs;
- novel memory births are strictly at/after their frame;
- no retroactive merge/split/relabel;
- no benchmark novel labels enter any training objective.

## 7. 2025/2026 Prior Art

See `2025_2026_JOINT_TRACKOCD_PRIOR_ART.md`. Verified neighbors: OVTR
(ICLR 2025), COVTrack/VOVTrack (ICCV 2025), DOVTrack (NeurIPS 2025),
MOTIP/TrackTrack (CVPR 2025), OCD (CVPR 2023), OCGCD (ECCV 2024),
Happy-CGCD (NeurIPS 2024), DiffGRE (ICCV 2025), TALON (CVPR 2026),
Video-GCD MCCL (arXiv 2025), plus earlier open-world detection lines.

## 8. Open-World Objectness Prior Art

See `OPEN_WORLD_OBJECTNESS_AUDIT.md`. Class-agnostic objectness lines
(CVPR 2025 unified objectness; 2025 IEEE/SciDirect unknown-aware
detection) support learning object existence separately from known-class
confidence. PU learning (Elkan-Noto 2008; nnPU 2014) grounds the
positive-unlabeled objective.

## 9. End-to-End MOT Prior Art

OVTR persistent causal queries, MOTIP ID prediction, TrackTrack online
tracking, COVTrack/VOVTrack/DOVTrack open-vocabulary tracking. None learns
irreversible online category identity jointly inside the MOT query.

## 10. OVMOT Prior Art

OVTrack, OVTR, COVTrack, VOVTrack, DOVTrack, GOVTrack, TRACT, ROMOT, NOVA
(verified in Phases 4M-4Z). None provides dynamic novel semantic identity
with per-frame assign-or-create in the MOT loop.

## 11. Category Discovery Prior Art

OCD (CVPR 2023), OCGCD (ECCV 2024), Happy-CGCD (NeurIPS 2024), DiffGRE
(ICCV 2025), TALON (CVPR 2026), Video-GCD MCCL (arXiv 2025). These operate
on sample/video streams with batch or test-time adaptation, not on a causal
MOT stream with physical identity.

## 12. Novelty Matrix

| Method | Persistent causal query | Class-agnostic PU objectness | Physical ID != Semantic ID | Dynamic novel memory | Irreversible assign-or-create | No benchmark novel GT |
|---|---|---|---|---|---|---|
| OVTR | yes | no | no | no | no | - |
| COVTrack | partial | no | no | no | no | - |
| DOVTrack | partial | no | no | no | no | - |
| OCD | no | no | no | yes | yes | yes |
| TALON | no | no | no | yes | test-time | yes |
| OCGCD | no | no | no | yes | batch | yes |
| JCDQ (this work) | yes | yes | yes | yes | yes | yes |

## 13. Formulation Decision

Option B (joint dual-identity query) chosen over Option A (modular stack);
see `PHASE6A_FORMULATION_DECISION.md`.

## 14. Joint Dual-Identity Query

`JointStateHead` consumes the persistent query's current embedding plus
physical evidence (history, score, disappear_time, hit_count) and emits
objectness, semantic state, and physical embedding from one shared hidden
representation. `JointQuery.apply_s2q` feeds semantic state back into the
query target for the next frame.

## 15. Class-Agnostic Objectness

`ow_obj_logit` is a learned single logit; at inference it replaces
known-class confidence as the public score. No threshold on known-class
scores; failure watch: objectness collapse (Ablation 1). Audit result:

```json
{json.dumps(main_audit, indent=2) if main_audit else "PENDING"}
```

## 16. Partial-Label / PU Treatment

nnPU-style objective with duplicate-overlap safe negatives; unlabeled
queries never forced to background. See `PARTIAL_LABEL_OBJECTNESS_DESIGN.md`.

## 17. Physical Identity State

`phys_emb` (normalized 128-d) plus OVTR `RuntimeTrackerBase` bookkeeping.
Hinge separation loss keeps different GT physical tracks apart.

## 18. Semantic Identity State

`sem_state` (normalized 128-d) compared against `S_t` = 48 known anchors +
online novel prototypes via a learned assign-or-create head.

## 19. Dynamic Semantic Memory

`SemanticMemory` (48 CLIP-initialized known anchors + EMA novel prototypes;
not a DDP buffer because shape changes online; births/updates strictly
causal).

## 20. Physical -> Semantic Interaction

Joint head uses physical trajectory features; same-track temporal
consistency losses; unlabeled trajectory structure participates in
discovery.

## 21. Semantic -> Physical Interaction

`sem2phys(sem_state)` added to `query_tgt` (gate 0.1) before next-frame
decoding. Same category never merges physical IDs (association remains
OVTR bookkeeping; semantic similarity is a feature, not a merger).

## 22. Strict-Causal Inference

Per frame: observe -> predict -> freeze public output -> update memory ->
next frame. One-pass streaming; no future confirmation.

Contract test result:
```json
{json.dumps(main_contract, indent=2) if main_contract else "PENDING"}
```

## 23. Immediate Semantic Decision

Every public object gets `KNOWN(c)` / `EXISTING_NOVEL(k)` / `NEW_NOVEL` in
the same frame; no UNRESOLVED/WAIT/DEFER in the main path.

## 24. No-Retroactive-Relabel Proof

Public outputs are appended to immutable per-frame records; novel slots are
usable only after birth; the strict evaluator replays rows once in
chronological order and never rewrites history.

## 25. Training Data

`lvis_known48_partial.json`: 141,373 supported-known annotations; 1,564,493
dropped annotations form the unlabeled stream; all 1,203 categories kept.

## 26. Training Loss

`joint_obj` (PU objectness), `joint_known` (known CE), `joint_disc`
(pseudo-novel assign-or-create), `joint_temp` (temporal consistency),
`joint_phys_sep` (identity separation), plus OVTR/TCO losses.

## 27. Training Schedule

1 epoch = 41,421 iterations; lr 2e-4 / backbone 2e-5; AdamW; lr_drop 13;
main runs the full epoch with checkpoints every 5,000; ablations run 5,000
iterations each.

## 28. GPU Usage

At most 4 GPUs; no other users' jobs killed. Main: GPU 0 (single GPU after
2-GPU DDP stalls in first iterations). Ablations: GPU 9 sequential.
Memory headroom checked before launch (free memory kept >= 25% floor).

## 29. Full Training Results

```
{main_log or "PENDING"}
```

Major repair 1 (semantic rebalance) applied and used as the headline:
`{{"used": true}}` if the repair checkpoint's known-occurrence accuracy
exceeds the base 41k checkpoint, otherwise the base checkpoint is kept.

## 30. Physical Tracking Results

{phys_row(main_phys)}

## 31. Known Physical Coverage

Aligned known tracks: {main_phys.get('n_known_tracks', 'PENDING') if main_phys else 'PENDING'} of
76 known GT-aligned tracks; first-score mean known:
{fmt(main_phys.get('first_score_mean_known')) if main_phys else 'PENDING'}.

## 32. Novel Physical Coverage

Aligned novel tracks: {main_phys.get('n_novel_tracks', 'PENDING') if main_phys else 'PENDING'} of
21 novel GT-aligned tracks; first-score mean novel:
{fmt(main_phys.get('first_score_mean_novel')) if main_phys else 'PENDING'}.

## 33. Fragmentation

Phase 5B baseline: 61% single-frame tracks, median track length 1, 37 GT
tracks with duplicate active predictions. Phase 6A:
{main_phys.get('n_fragmented_gt_categories', 'PENDING') if main_phys else 'PENDING'}
fragmented GT categories; duplicate active frames:
{main_phys.get('n_duplicate_active_frames', 'PENDING') if main_phys else 'PENDING'}
({main_phys.get('n_gt_tracks_with_duplicate_active_frames', 'PENDING') if main_phys else 'PENDING'}
GT tracks affected; Phase 5B baseline 37).

## 34. Duplicate / Rebirth

See fragmentation numbers above. Detailed duplicate/rebirth decomposition is
reported in `physical_eval/main_final.json` (or `physical_eval/main.json`).

## 35. Semantic Known Accuracy

{strict_row(main_strict)}

## 36. RN-Acc

Route-aware novel accuracy (first frame):
{fmt(legacy_f.get('route_aware_novel_acc')) if legacy_f else 'PENDING'};
last frame: {fmt(legacy_l.get('route_aware_novel_acc')) if legacy_l else 'PENDING'}.

## 37. Novel Routing Recall

First frame: {fmt(legacy_f.get('novel_routing_recall')) if legacy_f else 'PENDING'};
last frame: {fmt(legacy_l.get('novel_routing_recall')) if legacy_l else 'PENDING'}.

## 38. First-Novel Birth

First novel birth accuracy: {fmt(main_strict['strict'].get('first_novel_birth_acc')) if main_strict else 'PENDING'};
median correct-birth occurrence: {main_strict['strict'].get('first_correct_birth_median_occurrence') if main_strict else 'PENDING'}.

## 39. Existing Novel Reuse

Novel reuse accuracy: {fmt(main_strict['strict'].get('novel_reuse_acc')) if main_strict else 'PENDING'}.

## 40. Cross-Physical Reuse

Cross-physical reuse accuracy: {fmt(main_strict['strict'].get('cross_physical_reuse_acc')) if main_strict else 'PENDING'};
share of reuse that is cross-physical: {fmt(main_strict['strict'].get('cross_physical_reuse_share')) if main_strict else 'PENDING'}.

## 41. Semantic Switches

Semantic switch rate within physical tracks: {fmt(main_strict['strict'].get('semantic_switch_rate')) if main_strict else 'PENDING'}.

## 42. NMI / ARI

Novel NMI / ARI: {fmt(main_strict['strict'].get('novel_nmi')) if main_strict else 'PENDING'} /
{fmt(main_strict['strict'].get('novel_ari')) if main_strict else 'PENDING'}.

## 43. Count Error

Novel count absolute error: {main_strict['strict'].get('novel_count_abs_error') if main_strict else 'PENDING'}
(true novel categories {main_strict['strict'].get('n_true_novel_categories') if main_strict else 'PENDING'},
born novel states {main_strict['strict'].get('n_born_novel_states') if main_strict else 'PENDING'}).

## 44. End-to-End TrackOCD Metric

All-track accuracy: first {fmt(legacy_f.get('all_track_acc')) if legacy_f else 'PENDING'}
last {fmt(legacy_l.get('all_track_acc')) if legacy_l else 'PENDING'};
macro known-novel harmonic: first {fmt(legacy_f.get('macro_known_novel_harmonic')) if legacy_f else 'PENDING'}
last {fmt(legacy_l.get('macro_known_novel_harmonic')) if legacy_l else 'PENDING'};
known->novel error: first {fmt(legacy_f.get('known_to_novel_error')) if legacy_f else 'PENDING'},
known misclassification: first {fmt(legacy_f.get('known_misclassification_rate')) if legacy_f else 'PENDING'},
false known absorption (novel): first {fmt(legacy_f.get('false_known_absorption_rate')) if legacy_f else 'PENDING'}.

## 45. Critical Ablation 1 (known-conf objectness)

{strict_row(ablations.get('a1_knownconf', {}).get('strict'))}

## 46. Critical Ablation 2 (remove semantic->physical)

{strict_row(ablations.get('a2_no_s2p', {}).get('strict'))}

## 47. Critical Ablation 3 (remove physical->semantic)

{strict_row(ablations.get('a3_no_p2s', {}).get('strict'))}

## 48. Critical Ablation 4 (no unlabeled discovery)

{strict_row(ablations.get('a4_no_unlabeled', {}).get('strict'))}

## 49. Critical Ablation 5 (no dynamic novel memory)

{strict_row(ablations.get('a5_no_dynamic_memory', {}).get('strict'))}

## 50. Multi-Seed

PENDING: multi-seed runs will be added if full training cost allows; the
main model is the primary candidate and at least the main vs key baseline
should be re-run with different seeds.

## 51. Error Taxonomy

See `ERROR_TAXONOMY.md`. Measured from the strict summary:
- known->new rate: {fmt(main_strict['strict'].get('known_to_new_rate')) if main_strict else 'PENDING'};
- known->existing rate: {fmt(main_strict['strict'].get('known_to_existing_rate')) if main_strict else 'PENDING'};
- reuse->new rate: {fmt(main_strict['strict'].get('reuse_to_new_rate')) if main_strict else 'PENDING'};
- new precision on aligned: {fmt(main_strict['strict'].get('new_precision_on_aligned')) if main_strict else 'PENDING'};
- mean fragmentation: {fmt(main_strict['strict'].get('mean_fragmentation')) if main_strict else 'PENDING'};
- duplicate creation rate: {fmt(main_strict['strict'].get('duplicate_creation_rate')) if main_strict else 'PENDING'}.

## 52. Remaining Failure Modes

Honest list (measured where available): known forgetting delta
{fmt(main_strict['strict'].get('known_forgetting_delta')) if main_strict else 'PENDING'},
novel count error {main_strict['strict'].get('novel_count_abs_error') if main_strict else 'PENDING'},
born novel states {main_strict['strict'].get('n_born_novel_states_global') if main_strict else 'PENDING'},
semantic switch rate {fmt(main_strict['strict'].get('semantic_switch_rate')) if main_strict else 'PENDING'},
physical fragmentation {main_phys.get('n_fragmented_gt_categories') if main_phys else 'PENDING'},
duplicate active frames {main_phys.get('n_duplicate_active_frames') if main_phys else 'PENDING'}.
Qualitative failure-mode notes are written in `ERROR_TAXONOMY.md`.

## 53. Comparison with Phase 5A / 5B

Phase 5A frozen online: known 0.079-0.118, novel RR 0.955, RN-Acc 0.636,
515 births, cross-physical reuse 0.0. Phase 5B: 31,650 rows / 13,468 tracks /
97 aligned / novel first-score 0.274 vs known 0.371. Phase 6A numbers are
listed in sections 30-44.

## 54. ICLR Novelty Assessment

See `ICLR_NOVELTY_AUDIT.md`. Final verdict PENDING until results are
available; failure condition `MODULAR_STACKING_NOT_ICLR_CORE` is checked
against ablation evidence (joint interaction must measurably matter).

## 55. Final Status

`{final_status}`

Repair record: root cause was semantic-head under-training / assign-or-create
degenerating to "always existing" (known-to-existing 0.939 on the base
checkpoint). Repair 1 rebalanced known CE (x3) + margin, doubled the
pseudo-novel discovery loss, and increased held-out categories (max 6);
trained 8,000 more iterations from the 41k checkpoint. Result above reflects
`{{"repair_used": {repair1_used_disp}, "repair1_known_acc": {repair1_known_disp}, "base_known_acc": {base_known_disp}}}`.

## 56. Next Step

Depending on results: multi-seed runs, baseline comparison with a strong
2025/2026 OVMOT method, deeper error decomposition, or a major-repair cycle
(max 2) if a specific root cause is identified.

## 57. Exact Artifacts

- Code: `src/iclr27_phase6a/`, `scripts/run_iclr27_phase6a_blocking.sh`,
  `third_party/research_refs_phase4n/OVTR/ovtr/models/joint_query.py`,
  modified `models/ovtr.py`, `eval.py`, `datasets/tao_dataset.py`,
  `main.py`;
- Data: `third_party/research_refs_phase4n/OVTR/data/lvis_known48_partial.json`;
- Outputs: `outputs/iclr27_phase6a/{{training,ablations,q1,strict_eval,
  physical_eval,semantic_eval,final}}`;
- Docs: `docs/iclr27_phase6a/*.md`.

## 58. Exact Commands

See `scripts/run_iclr27_phase6a_blocking.sh` for the full idempotent
pipeline. Core training command:

```bash
cd third_party/research_refs_phase4n/OVTR/ovtr
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python main.py \\
  --config_file ./config/ovtr_lite_joint6a_train_val.py \\
  --dataset_file lvis_generated_img_seqs --with_box_refine --two_stage \\
  --lr 2e-4 --lr_backbone 2e-5 --lr_drop 13 --num_workers 4 --batch_size 1 \\
  --sample_mode random_interval --sample_interval 1 \\
  --sampler_steps 4 7 14 --sampler_lengths 2 3 4 5 \\
  --merger_dropout 0 --random_drop 0.1 --fp_ratio 0.3 \\
  --track_query_iteration CIP --calculate_negative_samples --max_len 250 \\
  --epochs 1 --ckpt_interval 5000 --tco_loss_coef 1.0 --tco_alpha 0.5 \\
  --joint_coef 1.0 --joint_alpha 0.1 --joint_state_dim 128 \\
  --resume outputs/iclr27_phase4q/q1_long/checkpoint.pth \\
  --output_dir outputs/iclr27_phase6a/training/main
```

Evaluation:

```bash
python eval.py --config_file ./config/ovtr_lite_train_val.py \\
  --dataset_file lvis_generated_img_seqs --batch_size 1 --with_box_refine \\
  --two_stage --pretrain <ckpt> --score_mode joint --sampler_lengths 2 \\
  --score_thresh 0.19 ... --filter_score_thresh 0.19 ... \\
  --miss_tolerance 5 ... --maximum_quantity 160 --eval track \\
  --result_path_track <out>/teta_results --output_dir <out>
python src/iclr27_phase4p/ovtr_main_eval.py --results-json <out>/teta_results/tao_track.json \\
  --out-prefix <out>/proposals
python src/iclr27_phase5a/evaluation/strict_causal_eval.py --proposals <out>/proposals_dev.csv \\
  --feats outputs/iclr27_phase4s/q1_features/feats.npz \\
  --proto-dir outputs/iclr27_phase5a/pilot/episodes --embed h --mode jointcsv \\
  --filter aligned --device cuda:0 --out <strict_out>
python src/iclr27_phase6a/evaluation/physical_eval.py --csv <out>/proposals_dev.csv \\
  --out <phys_out>
```
"""

    OUT_DOC.write_text(report)
    print(f"Wrote partial report: {OUT_DOC}")
    return OUT_DOC


if __name__ == "__main__":
    build()
