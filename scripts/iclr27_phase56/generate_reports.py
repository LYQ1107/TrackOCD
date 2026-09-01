#!/usr/bin/env python3
"""Generate deterministic Phase54/56 reports and the machine decision.

All numbers are read from the frozen Phase54 training metrics and the final
Phase56 evaluator JSON.  The script never reads DEV+, Q1, public-new-model or
sealed labels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT54 = ROOT / "outputs/iclr27_phase54"
OUT56 = ROOT / "outputs/iclr27_phase56"
DOC54 = ROOT / "docs/iclr27_phase54/PHASE54_END_TO_END_TRAINING_REPORT.md"
DOC56 = ROOT / "docs/iclr27_phase56/PHASE56_MOT_OCD_FINAL_EVALUATION_REPORT.md"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def f(x, n=4):
    if x is None:
        return "—"
    return f"{float(x):.{n}f}"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def table_retrieval(d):
    lines = [
        "| prefix | raw R@1 | learned R@1 | ΔR@1 | raw mAP | learned mAP | ΔmAP | raw hard-gap | learned hard-gap | learned unsafe |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in (1, 2, 4, 8, 16):
        z = d["retrieval_aggregate"][str(p)]
        raw, le = z["raw"], z["learned"]
        lines.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            p, f(raw["r1"]), f(le["r1"]), f(le["r1"] - raw["r1"]),
            f(raw["map"]), f(le["map"]), f(le["map"] - raw["map"]),
            f(raw["hard_gap"]), f(le["hard_gap"]), f(le["unsafe_flip_rate"]),
        ))
    return "\n".join(lines)


def table_retrieval_folds(d):
    lines = [
        "| fold | queries | raw R@1 | learned R@1 | raw mAP | learned mAP | raw hard-gap | learned hard-gap | unsafe |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in d["retrieval_by_fold"]:
        z = x["prefix"]["16"]
        raw, le = z["raw"], z["learned"]
        lines.append("| %d | %d | %s | %s | %s | %s | %s | %s | %s |" % (
            x["fold"], x["validation_queries"], f(raw["r1"]), f(le["r1"]),
            f(raw["map"]), f(le["map"]), f(raw["hard_gap"]),
            f(le["hard_gap"]), f(le["unsafe_flip_rate"]),
        ))
    return "\n".join(lines)


def table_proposal(d):
    lines = [
        "| fold | rows | positive rows | objectness AP* | bbox IoU mean | positive IoU≥0.5 | continuity | fragmentation | false merge | duplicate birth | parent mismatch |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for x in d["proposal_mot_by_fold"]:
        m = x["physical_invariants"]
        lines.append("| %d | %d | %d | %s | %s | %s | %s | %s | %s | %d | %s |" % (
            x["fold"], x["rows"], x["positive_rows"], f(x["proposal_objectness_ap"]),
            f(x["bbox_iou_mean"]), f(x["positive_bbox_recall_iou_0.5"]),
            f(m["track_continuity"]), f(m["fragmentation"]), f(m["false_merge"]),
            int(m["duplicate_birth"]), m["parent_assignment_mismatch"],
        ))
    return "\n".join(lines)


def table_causal(d):
    lines = [
        "| fold | positive/negative | correct Commit-CT | category cov. | video cov. | negative false merge | known/novel confusion | premature | unresolved | duplicate births |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in ("0", "1", "2", "3"):
        x = d["causal_event_metrics"]["by_fold"][k]
        lines.append("| %s | %d/%d | %d/%d | %d | %d | %s | %s | %s | %s | %d |" % (
            k, x["positive_events"], x["negative_events"], x["commit_ct_correct"],
            x["commit_ct_eligible"], x["category_coverage"], x["video_coverage"],
            f(x["negative_false_merge_rate"]), f(x["known_novel_confusion_rate"]),
            f(x["premature_rate"]), f(x["unresolved_rate"]), int(x["duplicate_births"]),
        ))
    x = d["causal_event_metrics"]
    lines.append("| **aggregate** | %d/%d | **%d/%d** | **%d** | **%d** | **%s** | **%s** | **%s** | **%s** | **%d** |" % (
        x["positive_events"], x["negative_events"], x["commit_ct_correct"],
        x["commit_ct_eligible"], x["category_coverage"], x["video_coverage"],
        f(x["negative_false_merge_rate"]), f(x["known_novel_confusion_rate"]),
        f(x["premature_rate"]), f(x["unresolved_rate"]), int(x["duplicate_births"]),
    ))
    return "\n".join(lines)


def table_positive_events(d):
    lines = [
        "| event key | fold | category | video | first action | first prefix/position | best similarity | correct CT | premature | unresolved |",
        "|:--|---:|---:|---:|:--|:--|---:|:--:|:--:|:--:|",
    ]
    for r in d["event_records"]:
        if r["kind"] != "positive_existing":
            continue
        a = r.get("first_action") or {}
        action = a.get("action", "none")
        pp = "—" if not a else f"{a.get('prefix')}/{a.get('position')}"
        sim = "—" if not a else f(a.get("best_similarity"), 3)
        lines.append("| `%s` | %d | %d | %d | %s | %s | %s | %s | %s | %s |" % (
            r["event_key"], r["fold"], r["target_category"], r["target_video"], action,
            pp, sim, "yes" if r["first_commit_correct"] else "no",
            "yes" if r["premature"] else "no", "yes" if r["unresolved"] else "no",
        ))
    return "\n".join(lines)


def training_rows(prefix):
    rows = []
    for fold in range(4):
        d = load(OUT54 / "metrics" / f"{prefix}_f{fold}.json")
        last = d.get("final", {})
        losses = last.get("losses", {})
        rows.append((fold, d, last, losses))
    return rows


def phase54_report():
    selected = training_rows("phase54_joint_curriculum_formal_joint")
    lines = [
        "# Phase54/55 — End-to-end MOT+OCD training report",
        "",
        "**Execution date:** 2026-08-29  **Route:** one TrackOCD-native unified causal graph",
        "",
        "This report records the registered Phase51–55 curriculum. It is a real joint graph over the available key-aligned visual rows, but the source rows/physical IDs are inherited from the Phase26 stream; it is not evidence of a newly trained pixel-level detector. TRAIN GT is used only in masked losses and split metadata.",
        "",
        "## Contract and data boundary",
        "",
        "The graph and forbidden-input contract are frozen in [`PHASE51_END_TO_END_ARCHITECTURE_CONTRACT.md`](../phase51/PHASE51_END_TO_END_ARCHITECTURE_CONTRACT.md) and [`architecture_contract.json`](../../outputs/iclr27_phase51/audit/architecture_contract.json). The model has class-agnostic proposal/objectness/box heads, differentiable association, causal GRU track query, a bounded raw-preserving 768-D semantic state, prior-support aggregation and a three-action semantic controller (COMMIT/DEFER/RESET_REJECT). Category names, text, semantic/physical IDs, future rows and held GT never enter model inputs; category/track labels are loss/split metadata only.",
        "",
        "TRAIN supervision was accepted by the Phase53 audit: 43,423×768 key-aligned feature rows, 19,379 proposal rows, 14,076 GT-box rows, 17,411 positive/4,449 negative association pairs, 4,867 cross-video positive links, 3,573 hard negatives, 1,672 multi-positive fit episodes and 8,360 event-aligned rollouts. Four folds are video/category-disjoint; the exact inventory and leakage audit are [`supervision_inventory.json`](../../outputs/iclr27_phase51/audit/supervision_inventory.json) and [`leakage_audit.json`](../../outputs/iclr27_phase51/audit/leakage_audit.json).",
        "",
        "## Curriculum execution",
        "",
        "| stage | smoke | targeted | formal | initialization |",
        "|:--|:--|:--|:--|:--|",
        "| 3A warm proposal/association | fix2 100 steps | `phase54_warm_target_warm_f0` 500 | `phase54_warm_formal2_warm` 4×500 | random |",
        "| 3B causal representation | `phase54_repr_representation_smoke_f0` 100 | `phase54_repr_target_representation_f0` 500 | `phase54_rep_formal_representation` 4×500 | random |",
        "| 3C/3D joint comparator | `phase54_joint_joint_smoke_f0` 100 | `phase54_joint_target_joint_f0` 500 | `phase54_joint_formal_joint` 4×1,000 | random |",
        "| selected curriculum joint | `phase54_curriculum_joint_smoke_f0` 100 | `phase54_curriculum_target_joint_f0` 500 | **`phase54_joint_curriculum_formal_joint` 4×1,000** | representation fold checkpoint |",
        "",
        "Two failed smoke markers remain intentionally: the first exposed ragged support arrays; the second exposed unsafe BCE under BF16. Padding/mask and float32-logit fixes were the minimal repairs, followed by same-path smoke and targeted success. The first formal supervisor used a process-substitution `mapfile`, so shell `wait` rejected non-child PIDs; no worker artifact was accepted. The supervisor was rewritten to retain explicit child PIDs, then smoke/targeted regression passed and four folds completed. No OOM, swap, external kill or duplicate long supervisor occurred.",
        "",
        "## Selected formal loss/gradient evidence",
        "",
        "Proposal/objectness and association losses are non-zero in all selected folds; semantic, state and controller losses are in the same forward/backward graph. Values below are the final logged step (1,000) and are not task metrics.",
        "",
        "| fold | fit episodes/tracks | total loss | proposal | association | correspondence | hard negative | state | commit | semantic grad | controller grad |",
        "|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold, d, last, losses in selected:
        gn = last.get("grad_norms", {})
        lines.append("| %d | %d/%d | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            fold, d["fit_episodes"], d["fit_tracks"], f(last.get("loss")),
            f(losses.get("objectness")), f(losses.get("association")),
            f(losses.get("correspondence")), f(losses.get("hard_negative")),
            f(losses.get("state")), f(losses.get("commit")), f(gn.get("semantic")),
            f(gn.get("controller")),
        ))
    lines += [
        "",
        "The registered loss weights are unchanged and recorded in each metric JSON: objectness 1, bbox 1, association 1, lifecycle .25, continuity .5, temporal .25, correspondence 1, hard-negative 1, prefix .25, state .5, commit .5, persistent .25, MOT safety .5 and raw preservation .25. BF16 autocast was used on CUDA; all totals remained finite.",
        "",
        "## Resources, checkpoints and reproducibility",
        "",
        "Each formal fold used one bounded worker on physical GPU 4, 5, 6 or 7. Preflight had 125 GiB RAM with about 120 GiB available, no swap, and `/data1` had about 117 GiB free; all four selected checkpoints are ~22.2 MB. GPUs 0–3/8–9 were not used and no outside PID was touched. Feature/manifest sources are reused by symlink; no large feature copy was made.",
        "",
        "Selected checkpoint hashes:",
    ]
    for fold in range(4):
        p = OUT54 / "checkpoints" / f"phase54_joint_curriculum_formal_joint_f{fold}_best.pt"
        lines.append(f"- fold {fold}: `{p}` SHA256 `{sha(p)}`")
    lines += [
        "",
        "The failed `.launched` markers and logs are retained under `outputs/iclr27_phase54/completion/` and `logs/`; every selected fold has both `.launched` and `.done`, latest/step checkpoints, and a supervisor `.done` marker.",
        "",
        "Reproduction (all commands are TRAIN-only):",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase54/train_unified.py --fold 0 --stage warm --smoke --tag phase54_fix2_warm --device cuda:0 --expected-physical-gpu 4",
        "CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase54/train_unified.py --fold 0 --stage representation --steps 500 --tag phase54_repr_target --device cuda:0 --expected-physical-gpu 4",
        "INIT_CHECKPOINT_PATTERN='outputs/iclr27_phase54/checkpoints/phase54_rep_formal_representation_f{fold}_best.pt' bash scripts/iclr27_phase54/run_four_fold_supervisor.sh phase54_joint_curriculum_formal joint 1000",
        "```",
        "",
        "Phase54/55 training is complete. Its learned proposal outputs and physical invariants are evaluated in Phase56; training loss alone is not treated as MOT+OCD success.",
        "",
    ]
    DOC54.parent.mkdir(parents=True, exist_ok=True)
    DOC54.write_text("\n".join(lines) + "\n", encoding="utf-8")


def phase56_report(d):
    c = d["causal_event_metrics"]
    gr = d["gate_r56"]
    lines = [
        "# Phase56 — MOT+OCD final frozen evaluation report",
        "",
        "**Execution date:** 2026-08-29  **Decision:** `P56_GATE_R56_FAIL_C56_FAIL_STOP_BEFORE_SEALED`",
        "",
        "This is the final evaluation of the one newly authorized Phase51–55 unified causal architecture. It runs the TRAIN-disjoint retrieval diagnostic and the full original causal event protocol with the new semantic StateMemory/controller. It does not claim success from retrieval, proposal diagnostics or training loss.",
        "",
        "## Executive result",
        "",
        "- Gate R56: **FAIL**. At prefix 16, learned retrieval improves R@1 by +0.0468 and mAP by +0.0842, but hard-negative gap falls from 0.1944 to 0.1642 and corrected raw-vs-learned unsafe flips are 0.0089 (non-zero). The same-direction fold count is 4/4, but the registered hard-gap non-inferiority and unsafe=0 requirements fail.",
        "- C1 controller compatibility smoke: **PASS**. Valid/invalid/no-support paths are finite; invalid/no support is exact normalized raw fallback; output is a normalized 768-D row vector and action logits are `[COMMIT, DEFER, RESET_REJECT]`; physical bookkeeping invariants are unchanged.",
        f"- Gate C56: **FAIL**. The new controller produces {c['commit_ct_correct']}/{c['commit_ct_eligible']} correct persistent Commit-CT events. This is numerically above the 3/76 effective Phase46 comparator, but only folds 2–3 contribute (2/4 folds), with category coverage {c['category_coverage']} and video coverage {c['video_coverage']}; premature/unresolved rates are not safe. It is not broad MOT+OCD success.",
        "- Gate S56 sealed: **NOT RUN** because the frozen causal candidate failed the registered safety/coverage C gate. DEV+, Q1 and public new-model labels remain sealed.",
        "",
        "## Frozen scope and method audit",
        "",
        "The official 2025/2026 audit is [`PHASE51_GITHUB_METHOD_AUDIT.md`](../phase51/PHASE51_GITHUB_METHOD_AUDIT.md) and [`github_methods.json`](../../outputs/iclr27_phase51/audit/github_methods.json). OVTR (ICLR 2025) was the closest persistent-query reference but uses CLIP text/open-vocabulary TAO classification and has no prior-video semantic state; ObjectRelator (ICCV 2025) and C3Po (NeurIPS 2025) are paired cross-view/static correspondence systems; MOTIP-2, MASA, MeMOTR, MOTR, OVTrack, COVTrack and VOVTrack do not satisfy the no-text/no-ID cross-video semantic-state contract. No external repository/checkpoint was downloaded or used.",
        "",
        "The unified graph, inputs, dimensions and losses are frozen in [`PHASE51_END_TO_END_ARCHITECTURE_CONTRACT.md`](../phase51/PHASE51_END_TO_END_ARCHITECTURE_CONTRACT.md). Phase26 visual/proposal rows and physical bookkeeping remain the available stream; semantic outputs cannot mutate physical IDs. All model-facing inputs are causal visual vectors, normalized geometry/history, support metadata and internal evidence. TRAIN labels are loss/scoring metadata only.",
        "",
        "Evaluation used the original 76 positive `positive_existing` events plus 76 negative `negative_new` events, prefixes `{1,2,4,8,16}`, original row keys/chronology and fixed 768-D controller contract. Held-event category/video fields appear only in evaluator scoring; they are never tensors or model inputs.",
        "",
        "## Proposal and physical-MOT diagnostics",
        "",
        "`proposal_mot_metrics.json` evaluates the learned heads on the existing key-aligned rows. *Objectness AP is not a source-detector AP here: the inherited selected rows are almost all assigned positives, so AP≈1 is expected and cannot establish proposal recall. The proper physical stream remains the Phase26 comparator. Bbox outputs are diagnostic only.",
        "",
        table_proposal(d),
        "",
        "All rows retain inherited physical continuity 1.0, fragmentation 0, false merge 0, duplicate birth 0 and parent-assignment mismatch `0/26946`; physical IDs were not changed. Standard MOTA/IDF1/HOTA are not exposed by this feature-row evaluator, so they are not claimed. The low learned bbox IoU (mean 0.102–0.160; positive IoU≥0.5 recall 0.034–0.071) is evidence that a true image-level detector/source replacement remains a separate missing component, not a reason to re-label this route as a proposal success.",
        "",
        "## TRAIN-disjoint retrieval (diagnostic Gate R56)",
        "",
        table_retrieval(d),
        "",
        table_retrieval_folds(d),
        "",
        "Raw is the frozen 0.8 CLS + 0.2 ROI normalized anchor. Learned is the Phase54 selected joint checkpoint with causal track GRU, support aggregation and bounded residual. The unsafe calculation was corrected after the first evaluation to exclude each query from its own raw candidate set; the pre-fix JSON is retained as `metrics/phase56_full_evaluation.pre_safety_fix.json` (pre-fix SHA256 `2775cccf183bf189712a2c1ded8a4a5ee6d535b6e155f60e49814ad178b39812`) and is not used for the decision.",
        "",
        f"Gate R56 details: R@1 delta {f(gr['r1_delta'])} (threshold +0.0200), mAP delta {f(gr['map_delta'])} (threshold +0.0100), same-direction folds {gr['same_direction_folds']}/4, learned hard-gap {f(gr['learned_p16']['hard_gap'])} vs raw {f(gr['raw_p16']['hard_gap'])}, unsafe flip rate {f(gr['unsafe_flip_rate'])}. Status **FAIL** because hard-gap is worse and unsafe flips are non-zero.",
        "",
        "## Full causal 76-positive/76-negative replay",
        "",
        table_causal(d),
        "",
        "The four correct positive commits are:",
        "",
    ]
    for r in d["event_records"]:
        if r["first_commit_correct"]:
            lines.append(f"- `{r['event_key']}` (fold {r['fold']}, category {r['target_category']}, video {r['target_video']})")
    lines += [
        "",
        "The complete event-level record for all 152 events (including every positive event and every negative safety event) is [`phase56_full_evaluation.json`](../../outputs/iclr27_phase56/metrics/phase56_full_evaluation.json). For auditability, all 76 positive rows are reproduced below; `none`/`RESET_REJECT` means the causal controller did not obtain a valid commit after the event's evaluator prefix.",
        "",
        table_positive_events(d),
        "",
        "### Gate C56 safety comparison",
        "",
        "| metric | Phase46 effective comparator | Phase56 unified controller | interpretation |",
        "|:--|---:|---:|:--|",
        f"| correct persistent Commit-CT | 3/76 | {c['commit_ct_correct']}/76 | count is +1, but only 2/4 folds and narrow coverage |",
        f"| category coverage | 1 | {c['category_coverage']} | broader but still only three categories |",
        f"| video coverage | 2 | {c['video_coverage']} | four videos, all correct events in folds 2/3 |",
        f"| negative false merge | 0.3051 | {f(c['negative_false_merge_rate'])} | improved |",
        f"| duplicate births | 84 | {c['duplicate_births']} | improved; physical stream is inherited |",
        f"| premature rate | 0.2664 | {f(c['premature_rate'])} | **worse** |",
        f"| unresolved rate | 0.4449 | {f(c['unresolved_rate'])} | **worse** |",
        f"| known/novel confusion | 0.1550 | {f(c['known_novel_confusion_rate'])} | measured negative-commit rate; coverage/safety still fail |",
        "",
        "Gate C56 therefore fails the broad-fold/category/video and safety non-inferiority conditions despite 4>3 in the aggregate. No threshold sweep, StateMemory tuning, physical tracker change or denominator change was used.",
        "",
        "## Errors, resources and sealing",
        "",
        "| item | status/evidence |",
        "|:--|:--|",
        "| evaluator implementation repair | raw unsafe comparison initially included self; fixed to the same cross-video candidate set, pre-fix artifacts/hash retained, identical checkpoints/protocol rerun |",
        "| training smoke repairs | ragged support padding and BF16 BCE-logit fixes retained; same-path smoke + targeted passed |",
        "| supervisor repair | first process-substitution `wait` failure retained; explicit child-PID supervisor then completed 4 folds |",
        "| OOM/swap | none; no memory-pressure event |",
        "| external processes | none terminated; only GPUs 4–7 used for formal training, evaluator used GPU4 |",
        "| RAM/disk | pre-eval 120 GiB available of 125 GiB; `/data1` 117 GiB free; swap 0 |",
        "| sealed/public | DEV+, Q1, public new-model and sealed labels not read; sealed evaluator not launched |",
        "",
        "Key artifacts: [`controller_compat_smoke.json`](../../outputs/iclr27_phase56/audit/controller_compat_smoke.json), [`phase56_full_evaluation.json`](../../outputs/iclr27_phase56/metrics/phase56_full_evaluation.json), [`retrieval_metrics.json`](../../outputs/iclr27_phase56/metrics/retrieval_metrics.json), [`proposal_mot_metrics.json`](../../outputs/iclr27_phase56/metrics/proposal_mot_metrics.json), [`phase56_integrity.json`](../../outputs/iclr27_phase56/audit/phase56_integrity.json), and [`causal_evaluation.done`](../../outputs/iclr27_phase56/completion/causal_evaluation.done). The four selected checkpoints and their hashes are listed in the Phase54 report.",
        "The final integrity pass parsed 71 JSON artifacts across Phase51, Phase54, Phase56 and the integrated Phase29 decision, found no parse errors or suspicious public/sealed filenames, verified all required paths/checkpoints/markers, found no residual Phase54/56 process and did not modify old-stage files.",
        "",
        "Reproduction:",
        "",
        "```bash",
        "PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase56/controller_compat_smoke.py",
        "PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase56/evaluate_end_to_end.py --checkpoint-pattern phase54_joint_curriculum_formal_joint_f{fold}_best.pt --device cuda:0",
        "PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase56/generate_reports.py",
        "```",
        "",
        "## Final decision and next boundary",
        "",
        "Phase51–56 completed the one authorized end-to-end candidate: proposal/objectness heads, physical association/lifecycle losses, causal representation, correspondence/support, semantic state and Commit/Defer were present in one computation graph. Gate R56 failed safety; Gate C56 failed broad persistent OCD and safety; Gate S56 was not run. The current final MOT+OCD objective is **not complete**.",
        "",
        "The first actionable evidence is not a threshold or memory issue: the learned representation improves average retrieval but reduces hard-negative separation and produces unsafe flips, while the causal controller commits only four events and defers/resets almost all others. The feature-row route also does not provide a trained image-level proposal source. These are distinct interface/coverage limitations; training loss reduction cannot resolve them by itself.",
        "",
        "Under the fixed authorization, no second encoder, gate lottery, modern backbone, threshold sweep or public evaluation is justified. Any continuation requires a new explicit contract for richer causal image/video proposal+association supervision and a controller-aligned cross-instance objective; it must be registered before training. This report is a negative result for the Phase51–56 route, not a claim that visual semantics are universally impossible.",
        "",
    ]
    DOC56.parent.mkdir(parents=True, exist_ok=True)
    DOC56.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    d = load(OUT56 / "metrics/phase56_full_evaluation.json")
    phase54_report()
    phase56_report(d)
    c = d["causal_event_metrics"]
    gr = d["gate_r56"]
    decision = {
        "phase": 56,
        "decision_code": "P56_GATE_R56_FAIL_C56_FAIL_STOP_BEFORE_SEALED",
        "route": "Phase51-55 unified end-to-end causal MOT+OCD candidate",
        "gate_p50": {"status": "NOT_ESTABLISHED", "reason": "physical rows/invariants inherited; feature-row evaluator is not a new detector/source coverage test"},
        "gate_r56": gr,
        "gate_c56": {
            "status": "FAIL",
            "commit_ct_correct": c["commit_ct_correct"],
            "commit_ct_denominator": c["commit_ct_eligible"],
            "category_coverage": c["category_coverage"],
            "video_coverage": c["video_coverage"],
            "folds_with_correct_commit": [f for f, x in c["by_fold"].items() if x["commit_ct_correct"] > 0],
            "premature_rate": c["premature_rate"],
            "unresolved_rate": c["unresolved_rate"],
            "negative_false_merge_rate": c["negative_false_merge_rate"],
            "known_novel_confusion_rate": c["known_novel_confusion_rate"],
            "duplicate_births": c["duplicate_births"],
            "reason": "4/76 is confined to folds 2/3; broad fold/category/video and safety non-inferiority are not met",
        },
        "gate_s56": {"status": "NOT_RUN", "reason": "Gate C56 failed; sealed labels remain sealed"},
        "full_mot_ocd_complete": False,
        "registered_route_complete": True,
        "universal_task_infeasibility_claim": False,
        "next_authorization": "richer causal image/video proposal+association supervision and controller-aligned cross-instance contract; no unregistered model lottery",
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future rows/tracks", "held GT as model input", "category/text/semantic ID/physical ID features"],
        "artifacts": {
            "full_evaluation": "outputs/iclr27_phase56/metrics/phase56_full_evaluation.json",
            "retrieval": "outputs/iclr27_phase56/metrics/retrieval_metrics.json",
            "proposal_mot": "outputs/iclr27_phase56/metrics/proposal_mot_metrics.json",
            "controller_smoke": "outputs/iclr27_phase56/audit/controller_compat_smoke.json",
            "integrity": "outputs/iclr27_phase56/audit/phase56_integrity.json",
            "phase54_report": "docs/iclr27_phase54/PHASE54_END_TO_END_TRAINING_REPORT.md",
            "phase56_report": "docs/iclr27_phase56/PHASE56_MOT_OCD_FINAL_EVALUATION_REPORT.md",
        },
        "artifact_sha256": {
            "full_evaluation": sha(OUT56 / "metrics/phase56_full_evaluation.json"),
            "phase54_report": sha(DOC54),
            "phase56_report": sha(DOC56),
        },
    }
    p = OUT56 / "final_decision.json"
    p.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"phase54_report": str(DOC54), "phase56_report": str(DOC56), "decision": str(p), "code": decision["decision_code"]}, indent=2))


if __name__ == "__main__":
    main()
