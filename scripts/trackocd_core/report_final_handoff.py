#!/usr/bin/env python3
"""Amendment-aware M0–M10 handoff from completed public results only.

No training, prediction, new geometry join, raw annotation, weight or image
access. Late diagnostic plots/manifest are not scientific preregistration.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import io
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, atomic_write_text, sha256_file
from scripts.trackocd_core.report_core_training import require_complete

OUT = ROOT / "outputs/trackocd_core"
DOCS = ROOT / "docs/trackocd_core"
BRANCH = "codex/trackocd-core-training-limited-masa"
PREFIXES = (1, 2, 4, 8, 16)
BASELINES = ("B0_frame_snapshot_vote", "B1_track_nearest", "B2_track_dpmeans")
REPRESENTATIONS = ("A1_SELECTED", "A2_EVIDENCE", "A2_CAPACITY_CONTROL")


def mean_std(values):
    if not values or any(not math.isfinite(float(v)) for v in values):
        raise ValueError("Finite actual observations required")
    m = math.fsum(values) / len(values)
    return m, math.sqrt(math.fsum((v - m) ** 2 for v in values) / len(values))


def grouped_diagnostics(val):
    groups = defaultdict(list)
    for c in val["cases"]:
        groups[c["method"], c["prefix"], c["coverage_target"]].append(c)
    rows = []
    for (method, prefix, target), cases in sorted(groups.items(), key=lambda x: str(x[0])):
        by_seed = defaultdict(list)
        for c in cases:
            by_seed[c["seed"]].append(c)
        if any(len(v) != 4 or len({c["order"] for c in v}) != 4 for v in by_seed.values()):
            raise ValueError("Every actual training seed needs all four orders")
        r = dict(scope=val["scope"], method=method, prefix=prefix,
                 coverage_target=target, training_seeds=len(by_seed), video_orders=4,
                 matched_known_denominator=1400, matched_novel_denominator=189,
                 full_known_denominator=4413, full_novel_denominator=819,
                 matched_only_not_primary=True, same_global_mapping_no_second_Hungarian=True)
        accessors = {
            "matched_old_acc": lambda c: c["errors"]["matched_only_diagnostic"]["old_acc"],
            "matched_new_acc": lambda c: c["errors"]["matched_only_diagnostic"]["new_acc"],
            "matched_h_score": lambda c: c["errors"]["matched_only_diagnostic"]["h_score"],
            "matched_ct": lambda c: c["persistent"]["matched_only_commit_ct_diagnostic"],
            "policy_seconds_per_track": lambda c: c["runtime"]["policy_seconds_per_track"],
            "policy_inference_seconds": lambda c: c["runtime"]["policy_inference_seconds"],
            "anonymous_memory_final": lambda c: c["runtime"]["anonymous_memory_final"],
            "committed_observation_latency": lambda c: c["runtime"]["mean_committed_observation_latency"],
            "predicted_ID_commitment_coverage": lambda c: c["runtime"]["commitment_coverage"],
            "reuse_coverage": lambda c: c["persistent"]["effective_commit_coverage"],
        }
        for key, get in accessors.items():
            values = [get(c) for c in cases]
            if any(v is None for v in values):
                raise ValueError("Do not manufacture latency or matched-only scores")
            seeds = [mean_std([get(c) for c in v])[0] for v in by_seed.values()]
            r[key + "_mean"], r[key + "_seed_std_ddof0"] = mean_std(seeds)
            _, r[key + "_order_and_seed_std_ddof0"] = mean_std(values)
        rows.append(r)
    return rows


def stage_audit():
    # PASS here is engineering/protocol completion, not a learned-method PASS.
    facts = [
        ("M0", "PASS", "PASS", "Exact 81/81 NAS v2 blobs recovered, not main Phase88/89/90; historical branches/stashes/assets preserved. 255 old COVTrack shards remain partial/unvalidated, not restored complete.", "M0_ASSET_AND_PROTOCOL_AUDIT.md"),
        ("M1", "ADDRESSED_BY_EXPLICIT_AMENDMENT", "BLOCKED", "No strong clean frontend qualified. User authorized frozen limited Native MASA replay; checkpoint training-stage binding incomplete; no Novel-exclusion guarantee or dense-frame claim.", "LIMITED_MASA_EVALUATION.md"),
        ("M2", "PASS", "PASS", "Shared frozen DINO 768-D, prefix 1/2/4/8/16 clipped for shorts, all 988 videos/304561 IDs, 1294110 observations, no future/GT/semantic input. CPU/CUDA exact replay proofs; no approximate search.", "TRAINING_PROTOCOL.md"),
        ("M3", "PASS", "PASS", "Full 4413/819 GT and order-specific 527/529/547/531 CT denominators; unknown/missing retained; known IDs never remapped, one global posthoc Hungarian, pure predicted memory CT with no GT repair.", "DATA_PROTOCOL_AND_LEAKAGE_AUDIT.md"),
        ("M4", "PASS_WITH_CONDITIONAL_INCOMPARABLE", "INCOMPARABLE_PHE_ONLY", "Raw frame vote / nearest / DP-Means same inputs and all prefixes/orders; PHE lacks compatible 78-class/common-feature supervision, not invented as reproduced.", "M4_FROZEN_BASELINE_TABLE.md"),
        ("M5", "PASS", "FAIL", "9 real adapter and 6 temporal/static fits; partial Known/H gains, not stable held-out Novel geometry or independent temporal advantage over capacity control.", "SCIENTIFIC_CONCLUSION.md"),
        ("M6", "PASS", "PASS", "Train intersect inherited Known only; distinct category/video folds; genuine Val Novel and other Train roles excluded; predicted-action memory rollouts.", "DATA_PROTOCOL_AND_LEAKAGE_AUDIT.md"),
        ("M7", "PASS", "FAIL", "18 actual 708-parameter matched D1/D2 fits with cost/credit differences, bounded costly WAIT and no oracle memory. Pure CT zero in learned controlled and Val policy matrices.", "CORE_TRAINING_FINAL_REPORT.md"),
        ("M8", "PASS", "FAIL", "Three independent training seeds, controlled held-out four-order/five-prefix matrices; old R0/R1 preserved, no rescue R2 or held-out redesign. Sparse four pseudo classes / 16 queries disclosed.", "SCIENTIFIC_CONCLUSION.md"),
        ("M9", "PASS_UNDER_AMENDED_LIMITED_SCOPE", "FAIL", "840 sealed unique all-ID executions / 2040 explicit aliases and metrics; primary full GT plus matched-only diagnostics, per-order/seed std, prefix and risk plots. Existing Native physical scores reused, no new tracking gain.", "FINAL_REPORT.md"),
        ("M10", "PASS", "FAIL", "All registered ablations, exact-capacity static/MLP, cross-video retrieval/CT, 780 fixed pairs, both coverage gaps and paired fix/new errors. 84 primary p16 conditional 500-video bootstraps; not independent causal intervals.", "FINAL_REPORT.md"),
    ]
    return [dict(stage=s, execution_status=e, scientific_status=g, finding=f, evidence_document=d)
            for s, e, g, f, d in facts]


def validate(core, rep, policy, val):
    require_complete(rep, policy, val)
    if core["status"] != "COMPLETE_REAL_TRAIN_AND_LIMITED_PREDICTED_EVALUATION_SCIENTIFIC_NEGATIVE":
        raise ValueError("Actual negative Core synthesis required")
    if (core["actual_fits"], len(core["retained_checkpoints"]), core["limited_MASA_unique_executions"]) != (33, 99, 840):
        raise ValueError("No narrowed or fabricated experiment completion")
    if any(core["scientific_conclusions"][k] for k in ("Representation_PASS", "Temporal_Evidence_PASS", "Persistent_Decision_PASS", "new_TrackOCD_method_paper_claim_supported")):
        raise ValueError("Actual negative gates cannot be promoted")
    for c in val["cases"]:
        d = c["errors"]["matched_only_diagnostic"]
        if (d["known_denominator"], d["novel_denominator"]) != (1400, 189) or not d["same_global_mapping_no_second_Hungarian"] or not d["not_primary_full_GT_result"]:
            raise ValueError("Matched-only is an auxiliary existing-map diagnostic")
    for name, sha in core["artifact_sha256"].items():
        if sha256_file(OUT / name) != sha:
            raise ValueError("Completed public input changed: " + name)
    for name, sha in core["final_document_sha256"].items():
        if sha256_file(DOCS / name) != sha:
            raise ValueError("Completed Core document changed: " + name)
    for name, sha in core["protected_sha256"].items():
        if sha256_file(ROOT / name) != sha:
            raise ValueError("Protected scientific source changed: " + name)


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def p16_primary(rows):
    return [r for r in rows if r["prefix"] == 16 and r["coverage_target"] in (None, 1.)]


def score_table(rows, matched=False):
    lines = ["| Method | Old % | New % | actual mean H % | pure CT % |", "|---|---:|---:|---:|---:|"]
    keys = ("matched_old_acc", "matched_new_acc", "matched_h_score", "matched_ct") if matched else ("old_acc", "new_acc", "h_score", "correct_commit_ct")
    for r in rows:
        values = [f"{100*r[k+'_mean']:.4f} ± {100*r[k+'_order_and_seed_std_ddof0']:.4f}" for k in keys]
        lines.append("| " + r["method"] + " | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def render_plots(aggregate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    selected = BASELINES + ("A1_SELECTED", "A2_EVIDENCE", "A2_CAPACITY_CONTROL", "D1_SIMPLE_MLP", "FULL")
    colors = dict(zip(selected, plt.get_cmap("tab10").colors))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, metric, label in zip(axes.flat, ("new_acc", "h_score", "correct_commit_ct", "false_merge_rate"), ("Novel ACC % (GT=819)", "Actual mean H % (not H of means)", "Pure cross-video Commit-CT %", "False Merge % (fixed GT opportunities)")):
        for method in selected:
            rr = sorted((r for r in aggregate if r["method"] == method and r["coverage_target"] in (None, 1.)), key=lambda r:r["prefix"])
            if [r["prefix"] for r in rr] != list(PREFIXES):
                raise ValueError("All frozen prefixes required")
            y = np.array([100*r[metric+"_mean"] for r in rr])
            sd = np.array([100*r[metric+"_order_and_seed_std_ddof0"] for r in rr])
            ax.plot(range(5), y, marker="o", label=method, color=colors[method], linewidth=1.5)
            ax.fill_between(range(5), np.maximum(0,y-sd), np.minimum(100,y+sd), color=colors[method], alpha=.07)
        ax.set_xticks(range(5), [str(p) for p in PREFIXES]); ax.set_xlabel("Independent causal prefix cap (shorts clipped)")
        ax.set_ylabel(label); ax.grid(alpha=.2)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle("Full limited Native MASA replay: all 988 videos / 304561 IDs\nShading = descriptive order+seed SD; aliases not independent; not dense-frame online")
    path = OUT / "FINAL_PREFIX_CURVES.png"; fig.savefig(path,dpi=160);plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    methods = ("D1_SIMPLE_MLP", "D2_RISK_AWARE", "FULL", "WITHOUT_WAIT", "WITHOUT_RISK")
    for method in methods:
        rr=sorted((r for r in aggregate if r["method"]==method and r["prefix"]==16),key=lambda r:r["coverage_target"])
        if [r["coverage_target"] for r in rr] != [.5,.75,1.]:raise ValueError("Every frozen Train-development point required")
        for ax,key in zip(axes,("effective_commit_coverage", "predicted_ID_commitment_coverage")):
            ax.scatter([100*r[key+"_mean"] for r in rr],[100*r["false_merge_rate_mean"] for r in rr],label=method,s=30)
        # No interpolation: coincident points are explicitly documented.
    for ax,label in zip(axes,("Actual reuse commitment coverage %", "Actual all-predicted-ID commitment coverage %")):
        ax.set_xlabel(label);ax.set_ylabel("False Merge % (fixed GT reuse denominator)");ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("p16 frozen development targets 0.50 / 0.75 / 1.00, not achieved Val targets\nActual points only; coincident points not dropped; no favorable interpolation")
    path=OUT/"FINAL_COVERAGE_RISK_POINTS.png";fig.savefig(path,dpi=160);plt.close(fig)


def main(commit):
    remote=subprocess.check_output(["git","-c","http.proxy=http://127.0.0.1:17890","ls-remote","origin","refs/heads/"+BRANCH],text=True,timeout=25).split()[0]
    source="scripts/trackocd_core/report_final_handoff.py"
    if remote != commit or subprocess.check_output(["git","show",f"{commit}:{source}"]) != Path(__file__).read_bytes():
        raise ValueError("Report-only source must be registered and exact remote before execution")
    names=["FINAL_REPORT.md","DATA_PROTOCOL_AND_LEAKAGE_AUDIT.md","TRAINING_AND_REPRODUCTION.md","M4_FROZEN_BASELINE_TABLE.md"]
    outputs=["FINAL_RESULT.json","FROZEN_PHYSICAL_FRONTEND.json","BASELINE_COMPARISON.csv","ABLATION_RESULTS.csv","PER_VIDEO_ORDER_RESULTS.csv","MATCHED_ONLY_AND_RUNTIME_DIAGNOSTICS.csv","FINAL_PREFIX_CURVES.png","FINAL_COVERAGE_RISK_POINTS.png"]
    if any((DOCS/n).exists() for n in names) or any((OUT/n).exists() for n in outputs):
        raise ValueError("Refuse to overwrite an existing canonical handoff")
    def read(n):return json.loads((OUT/(n+".json")).read_text())
    core=read("CORE_TRAINING_FINAL_RESULT"); rep=read("TRAIN_FIRST_REPRESENTATION_HELDOUT_RESULT");policy=read("TRAIN_FIRST_POLICY_HELDOUT_RESULT");val=read("LIMITED_MASA_EVALUATION_RESULT")
    validate(core,rep,policy,val);diag=grouped_diagnostics(val); audit=stage_audit()
    frontend=read("audit/masa_full_val_prediction_summary");physical=read("audit/masa_full_val_physical_result")
    amendment=json.loads((ROOT/"configs/trackocd_core/SCOPE_AMENDMENT_TRAIN_FIRST.json").read_text())
    if amendment["user_authorization_sha256"] != core["scope_authorization_sha256"] or amendment["frontend"]["strong_frontend_qualified"]:
        raise ValueError("Explicit limited scope authorization required")
    # These aliases add requested names, not experiments or better metrics.
    aliases={"BASELINE_COMPARISON.csv":"LIMITED_MASA_BASELINE_AND_POLICY_COMPARISON.csv", "PER_VIDEO_ORDER_RESULTS.csv":"PER_ORDER_RESULTS.csv"}
    for dest,src in aliases.items():atomic_write_text(OUT/dest,(OUT/src).read_text())
    atomic_write_text(OUT/"ABLATION_RESULTS.csv",csv_text([r for r in val["aggregate"] if r["method"] not in BASELINES]))
    atomic_write_text(OUT/"MATCHED_ONLY_AND_RUNTIME_DIAGNOSTICS.csv",csv_text(diag))
    frozen={"schema_version":"trackocd.core.limited-frontend-handoff.v1", "status":"FROZEN_LIMITED_FRONTEND_NOT_STRONG_M1_PASS",
        "name":"MASA_NATIVE", "historical_M1_status":"BLOCKED_FRONTEND_QUALITY", "scope_authorization_sha256":core["scope_authorization_sha256"],
        "late_handoff_manifest_not_new_scientific_freeze":True,"original_physical_preregistration_commit":frontend["preregistration_commit"],
        "checkpoint_sha256":frontend["frozen_model"]["checkpoint_sha256"], "native_model_config":frontend["frozen_model"]["native_model_config"],
        "native_source_files":frontend["frozen_model"]["source_files"], "config_sha256":frontend["config_sha256"],
        "physical_prediction_manifest_sha256":frontend["private_manifest_sha256"], "physical_result_sha256":sha256_file(OUT/"audit/masa_full_val_physical_result.json"),
        "common_feature_manifest_sha256":val["input_identity"]["feature_manifest_sha256"],
        "cadence":"ANNOTATED-CADENCE CAUSAL REPLAY", "same_physical_stream_all_methods":True,"detection_source":"Native frozen one-class SAM-MASA RPN/ROI, no vocabulary or public detection input",
        "training_supervision_provenance":physical["provenance"],"strong_frontend_qualified":False,"exact_checkpoint_supervision_stage_binding_verified":False,
        "all_novel_supervision_excluded_claim":False,"dense_full_frame_online_claim":False,"new_physical_inference":False,
        "videos":988,"images":36375,"predicted_IDs":304561,"reliable_coverage":{"Known":[1400,4413],"Novel":[189,819]},
        "fixed_full_GT":val["fixed_full_GT"],"physical_scores_identical_all_methods":val["physical_native_reference"]}
    atomic_json(OUT/"FROZEN_PHYSICAL_FRONTEND.json",frozen)
    render_plots(val["aggregate"])
    audit_table="| Stage | Execution / disposition | Scientific gate | Finding |\n|---|---|---|---|\n"+"\n".join(f"| {r['stage']} | {r['execution_status']} | {r['scientific_status']} | {r['finding']} |" for r in audit)+"\n"
    primary=p16_primary(val["aggregate"]);matched=p16_primary(diag)
    gt={r["method"]:r for r in core["train_p16_aggregate"]}
    gaps=[]
    for r in primary:
        if r["method"] not in gt:continue
        t=gt[r["method"]];gaps.append(dict(method=r["method"],Train_GT_old=t["old_acc_mean"],Val_pred_old=r["old_acc_mean"],Train_GT_new=t["new_acc_mean"],Val_pred_new=r["new_acc_mean"],Train_GT_H=t["h_score_mean"],Val_pred_H=r["h_score_mean"],Train_GT_CT=t["correct_commit_ct_mean"],Val_pred_CT=r["correct_commit_ct_mean"]))
    gap_table="| Method | Train GT H % | Val predicted H % | difference pp | Train GT CT % | Val predicted CT % |\n|---|---:|---:|---:|---:|---:|\n"+"\n".join(f"| {r['method']} | {100*r['Train_GT_H']:.4f} | {100*r['Val_pred_H']:.4f} | {100*(r['Val_pred_H']-r['Train_GT_H']):+.4f} | {100*r['Train_GT_CT']:.4f} | {100*r['Val_pred_CT']:.4f} |" for r in gaps)+"\n"
    final=f'''# TrackOCD M0–M10 final research handoff

## Outcome and explicit scope amendment

Complete **authorized research negative result**, not strong end-to-end
TrackOCD success. The original M1 strong-quality/provenance condition remains
BLOCKED_FRONTEND_QUALITY. The user's later Train-first amendment (SHA256
`{core['scope_authorization_sha256']}`) explicitly authorized limited Native
MASA and credible negative completion; it did not qualify M1. The original
FINAL_GOAL.json and frozen amendment remain historical preregistration records,
not live app-state controls. No missing work is hidden as a successful method.

33 actual fits / 33 selected and 99 retained checkpoints, 2220 controlled
cases; 840 unique all-ID sealed Val executions / 2040 explicitly aliased
logical references. All four video orders and all five prefixes retained.
No new training, inference, GT join or tuning for this report-only handoff.

## Every original M0–M10 requirement disposition

{audit_table}

M11 NOT_AUTHORIZED: independent representation/temporal/persistent gates
FAIL. PHE INCOMPARABLE, not silently omitted or reported as reproduced.
See [actual Core report](CORE_TRAINING_FINAL_REPORT.md), [architecture](METHOD_ARCHITECTURE.md),
[training protocol](TRAINING_PROTOCOL.md), [leakage audit](DATA_PROTOCOL_AND_LEAKAGE_AUDIT.md)
and [ten scientific answers](SCIENTIFIC_CONCLUSION.md).

## Primary full-GT predicted-track p16 results

Percentages: mean ± descriptive population SD across the frozen seed/order
cases (raw baselines one control seed). Not a confidence interval or 2040
independent runs; actual mean H is **not H(mean Old, mean New)**. Each seed's
four-order mean and both seed / order+seed SD are in BASELINE_COMPARISON.csv;
each order/seed/prefix and full scope are in PER_VIDEO_ORDER_RESULTS.csv.

{score_table(primary)}

All cases keep GT Known=4413, Novel=819. Pure CT fixed denominators by order
main/seed1027/seed1028/seed1029: 527/529/547/531. FULL and all 1800 learned
Val policy cases have pure CT zero; raw B1 has one positive case and B2 two,
so **not all 2040 cases have zero CT**. No unsupported TETA/OWTA/IDF1 is
invented for this Native semantic matrix. Identical inherited physical
HOTA={val['physical_native_reference']['HOTA']:.10f}, AssA={val['physical_native_reference']['AssA']:.10f},
DetA={val['physical_native_reference']['DetA']:.10f}, DetRe={val['physical_native_reference']['DetRe']:.10f}.
These are existing physical scores, not new semantic-feedback tracking gains.

## Matched-only auxiliary diagnostic, NOT the primary result

The same global posthoc mapping, no second Hungarian. Known=1400, Novel=189
after reliable geometry matching. These conditional accuracies must never
replace full-GT penalties or certify pure anonymous memory.

{score_table(matched,matched=True)}

## GT vs predicted descriptive gap

{gap_table}

Train controlled final labels=207 Known / 16 pseudo-Novel (four categories)
and Val=4413/819 are **different data, roles and supports**. The differences
above are descriptive, not a same-population causal estimate of tracker
damage. Some Val H exceeds controlled H, which does not establish recovery.
Native reliable Known1400/4413, Novel189/819; current observed reuse
118/527,123/529,121/547,126/531; both earlier/current observed support
87/527,93/529,92/547,94/531. The stricter both-length>=16 support is
69/527,68/529,70/547,74/531, not a hard ceiling for true clipped-short p16.

## Prefix, risk, runtime and memory diagnostics

![Five frozen causal prefix caps](../../outputs/trackocd_core/FINAL_PREFIX_CURVES.png)

![Actual coverage/risk points](../../outputs/trackocd_core/FINAL_COVERAGE_RISK_POINTS.png)

All Train-development target points 0.50/0.75/1.00 are shown, including
coincident points. The target is not achieved Val coverage; no favorable
interpolation or selection. At p16 target1 FULL and WithoutWAIT commit all
predicted IDs and have equal CT/FM: no safety gain established. Reuse and
all-predicted-ID coverage gaps both matter, not named target matching.
MATCHED_ONLY_AND_RUNTIME_DIAGNOSTICS.csv records actual inference seconds per
track, final anonymous states, commitment coverage and committed observation
latency for every method/prefix/point, with seed/order SD. Amortized replay
seconds/track is not dense-frame FPS. WAIT unresolved at cap is censored,
not zero latency; conditional committed latency cannot certify overall delay.
Aliased case runtime is not independently repeated timing.

Paired corrections/780 comparisons are in the actual Val JSON; 84 primary
p16 comparisons use fixed seed1027/500-video bootstrap **conditional on
frozen global mapping and replay state**, not independent causal/category
confidence. NO_MEMORY (clears anonymous history before every track, permitting
transient new state) and RESET_PER_VIDEO (no cross-video history) differ;
zero mergerate without reuse is not safe discovery. WITHOUT_RISK
is matched learned CE-only ablation; fixed threshold/DP-Means are the separate
B1/B2 baselines. Static evidence and D1/D2 parameter controls are actual fits.

## Original eight research questions

1. **Temporal vs mean/frame vote?** Partial Novel improvements, but independent
   temporal benefit FAIL against A1 and capacity-matched static. No stable CT gain.
2. **Same category across individuals/videos?** Overall Known retrieval improves,
   but unseen pseudo macroRank1 at p16 raw .3500 vs A1/A2 .3083 (all four
   categories have positives); all five learned pseudo Rank1 values lower.
3. **Risk vs threshold/DP-Means/MLP?** No: learned pureCT zero, FULL meanH zero,
   no fixed-pair primary CT/H advantage over B1/B2/D1 and no matched-risk success.
4. **WAIT preserves coverage while safer?** Not established; report both actual
   coverage gaps. Full p16 noWAIT identical, and Train rejection accompanies FM change.
5. **Reliable vocabulary-free anonymous reuse?** A few raw baseline cases work;
   learned reliable persistent discovery FAIL. Backend has no Novel vocabulary,
   but exact physical checkpoint all-stage Novel-supervision exclusion unverified.
6. **GT vs predicted?** Both fully measured separately; neither learned memory
   establishes pureCT. The descriptive differences above are not same-data causality.
7. **Independent innovation / not mere pipeline stacking?** Implemented category
   representation, temporal evidence, cost-trained predicted memory; controlled
   results do not support a new-method paper's advantage claim.
8. **Which bottleneck?** Coverage/shortness, prototype absence, unseen geometry
   and decision pollution all have evidence. Only24 Known GT tracks lack a
   prototype despite30 missing Known classes; avoid a misleading30/78 track
   fraction. Unknown302972IDs are not proven background/pollution; their memory
   prevents purity certification. No one causal bottleneck is uniquely proved.

## Artifacts, resources, reproduction and delivery

FINAL_RESULT.json binds canonical docs/tables/plots, source registration,
unchanged Core result, role/source/feature/protocol hashes and amendment.
CSV aliases are exact byte copies explicitly recorded there, not new results.
ABLATION_RESULTS.csv is a filtered actual Val aggregate with learned controls;
controlled Train ablations remain REPRESENTATION_ABLATION.csv and
PERSISTENT_POLICY_COMPARISON.csv, never pooled with Val scores.

Task-private measured files {core['resources']['conservative_task_private_file_bytes']} B
are below soft15/hard30 GiB; original entire project disk footprint is not
this incremental budget. Inference max case RSS {core['resources']['peak_inference_case_RSS_bytes']} B,
GPU reserved {core['resources']['peak_inference_case_GPU_reserved_bytes']} B.
Feature payload1859256557 B; 840 private ledgers627651777 B; all weights,
raw TAO data, full features, prediction NPZ and process witnesses stay local.

Final source registration `{commit}`; exact remote was checked before this
report-only execution. Final output commit must subsequently be tested,
pushed and compared to remote; see the later research log/delivery receipt,
not a fabricated self-referential final commit SHA. No TAO Test access,
foreign process intervention, Val fitting or new physical inference.
'''
    atomic_write_text(DOCS/"FINAL_REPORT.md",final)
    atomic_write_text(DOCS/"M4_FROZEN_BASELINE_TABLE.md","# Frozen M4 baseline handoff\n\nActual full limited Native MASA p16 baseline mean ± four-order population SD.\nRaw baselines have no training-seed repeat. Same input/features/evaluator and\nfixed full GT denominators; aliases not independent. All five prefix values\nin BASELINE_COMPARISON.csv; controlled Train table separately in\nTRAIN_BASELINE_COMPARISON.csv. PHE INCOMPARABLE (48/78 support and common-feature\ncheckpoint supervision compatibility unproven); no surrogate result.\n\n"+score_table([r for r in primary if r["method"] in BASELINES]))
    atomic_write_text(DOCS/"DATA_PROTOCOL_AND_LEAKAGE_AUDIT.md",'''# Data protocol and leakage audit — actual final scope

Only TAO Train intersect inherited Known IDs trains/chooses models and points.
78 inherited Known / 209 genuine Novel / 45 distractor roles preserved by
original recovered byte hashes. Train Known48 supported classes/2196 tracks/
43380 observations; genuine Novel7374 and unassigned2006 observations excluded
(distractor1879 also excluded). Legal model-fitting fold15 classes/1305 eligible
tracks; distinct full-budget adapter visits933/938/948, not all1305. Five
video-disjoint purposes with category-disjoint pseudo folds. Final heldout
207Known/16pseudo queries; four sparse pseudo classes, all with positives.
Prototype fold48classes; no Val prototype fill-in for30 unsupported Known classes.

Common frozen DINO ViT-B/14 checkpoint768-D, FP16 compact observation shards;
518resize/bilinear/context0.1/normalization pinned by feature config and source.
All 988 Val videos/304561IDs including114380singletons retained. Prefix caps
1/2/4/8/16 use only observed history and true clipped short length. Four
registered video orders, track IDs namespaced by video, no future videos,
labels, category names or posthoc mapping in live predicted memory. Loss
supervision is not memory initialization or GT repair. Exact CPU/CUDA proofs
and source/hash-bound adversarial tests are delivered.

Val evaluation starts after model/prototype/points freeze. GT roles/geometry
only in posthoc evaluators, missing GT included at4413/819 and fixed reuse
527/529/547/531. Same single posthoc Novel Hungarian for main and matched-only
diagnostic; Known IDs never remapped. Pure memory CT uses predicted members
and no semantic Hungarian. Unknown302972IDs retained, not labelled background
or automatically proven pollution. Bootstrap conditional on fixed mapping/state.

No TAO Test annotation/image read, no Val Novel training/tuning, no raw GT
Git upload. Limited Native physical runtime has one generic foreground class
and no Novel vocabulary/text input; exact all-stage physical checkpoint
supervision binding is UNVERIFIED. Therefore no blanket all-Novel-supervision
exclusion or strong M1 PASS. Historical COVTrack is vocabulary-assisted and
its255shards are partial/unvalidated; SimOWT provenance/score diagnosis is not
silently repaired. FROZEN_PHYSICAL_FRONTEND.json is a late hash-bound handoff
of the already frozen authorized limited stream, not a new scientific freeze.

See FINAL_RESULT.json (hash bindings and every requirement disposition),
TRAINING_PROTOCOL.md (complete fold/input contract), INPUT_SHIFT_DIAGNOSTIC.md
(posthoc descriptive, not intervention) and RESEARCH_LOG.md (stage history).
''')
    atomic_write_text(DOCS/"TRAINING_AND_REPRODUCTION.md",f'''# Actual training and reproduction handoff

Environment: existing trackocd-a100 Python3.10.21 / torch2.6.0+cu118 /
torchvision0.21.0. Source `/data3/liuyeqiang/trackocd_a100_env.sh`; do not
reuse historical GPU ordinals, PIDs or terminal IDs. No reinstall/download.

Architecture and exact registered optimizer/data budgets are in
METHOD_ARCHITECTURE.md, TRAINING_PROTOCOL.md and configs/trackocd_core.
Actual fits: nine768→512→256 adapters (525056parameters), six temporal/static
matched evidence fits (16833extra), eighteen matched17→32→4 D1/D2 policies
(708parameters). Three training seeds1027/1028/1029. Adapter/evidence1000steps,
policy20epochs×4orders. Development selection precedes all Val evaluation;
33selected and99retained checkpoints all actual local paths/bytes/SHA256
in CORE_TRAINING_FINAL_RESULT.json. Earlier selected checkpoints' exact
distinct-track visit footprint was not separately recorded; no invented count.

Reproduction entries (documentation only; do not blindly rerun completed jobs):
prepare_train_first.py; extract_train_first.py; train_core_representation.py
--phase representation / --phase evidence; train_persistent_policy.py;
evaluate_core_representation.py; evaluate_persistent_policy.py;
freeze_train_models.py; extract_limited_masa_features.py;
validate_full_limited_masa_features.py; run_limited_masa_inference.py;
evaluate_limited_masa.py; report_core_training.py; report_final_handoff.py.
Use each --help and the registered config/receipt before any fresh run.
Raw assets are deliberately not included in Git; model code alone is not an
asset-free reproduction claim. Do not reconstruct old COVTrack features as MASA.

All actual loss traces15360rows and native training PNGs delivered; different
D1/D2 objectives are not directly comparable losses. CPU/CUDA exact-replay
proofs check state/event equality; no historical Python3.7-equivalence claim.
Case ledgers304561rows/seal, full feature988payloads and52protected scientific
files were independently checked. Registration and every source/config/
feature/role hash are chained in the actual Core result and FINAL_RESULT.json.

Original frozen inference registrationeddbc50435e87b85b7141decfb2a5dc32290ef24;
scheduling-only recovery/collectorc662f76606e23ee2cd054b507db7c0da700dcece.
Eight adopted workers' orphan exit status is unavailable, not invented as0;
all group/seal completion proven. Ten newly owned inference child exits0;
763new evaluator exits0 plus77same-source sealed metrics reused. No live
worker/parent remains. Existing physical result reused without tracker rerun.

Report-only source preregistration `{commit}`, exact remote verified before
execution. Scoped tests: tests/trackocd_core, tests/trackocd_v2,
tests/baselines/pandas_bytetrack; not a blanket whole-repository test claim.
Final stage follows test → commit → push → exact remote verification.
Weights/raw images/labels/full features/prediction NPZ/private ledgers and
process witnesses remain local. No forced push, deletion, external kill or
TAO Test. Negative results do not authorize further fitting or M11 feedback.
''')
    files={"docs/trackocd_core/"+n:sha256_file(DOCS/n) for n in names}
    files.update({"outputs/trackocd_core/"+n:sha256_file(OUT/n) for n in outputs if n!="FINAL_RESULT.json"})
    evidence_names=("CORE_TRAINING_FINAL_RESULT.json","audit/source_recovery.json","audit/assets.json","audit/frontend.json","audit/masa_full_val_prediction_summary.json","audit/masa_full_val_physical_result.json","audit/physical_cross_video_support.json","TRAINING_CURVES_RESULT.json","TRAINING_LOSS_CURVES.png","EVIDENCE_TRAINING_LOSS_CURVES.png")
    evidence={"outputs/trackocd_core/"+n:sha256_file(OUT/n) for n in evidence_names}
    evidence.update({n:sha256_file(ROOT/n) for n in (source,"configs/trackocd_core/FINAL_GOAL.json","configs/trackocd_core/SCOPE_AMENDMENT_TRAIN_FIRST.json","configs/trackocd_core/roles.json")})
    result={"schema_version":"trackocd.core.m0-m10-final-handoff.v1","status":"COMPLETE_AMENDED_RESEARCH_NEGATIVE_NOT_STRONG_END_TO_END_SUCCESS",
        "original_M0_M10_all_requirements_audited":True,"strong_end_to_end_success":False,"historical_M1_status":"BLOCKED_FRONTEND_QUALITY",
        "explicit_user_scope_amendment_sha256":core["scope_authorization_sha256"],"scientific_conclusions":core["scientific_conclusions"],
        "required_stage_dispositions":audit,"M11":"NOT_AUTHORIZED_INDEPENDENT_SCIENTIFIC_GATES_FAILED","PHE":val["PHE"],
        "actual_fits":33,"retained_checkpoints":99,"controlled_cases":2220,"unique_Val_executions":840,"logical_Val_cases_including_aliases":2040,
        "source_registration_commit":commit,"source_exact_remote_verified_before_report":True,"report_is_posthoc_not_scientific_preregistration":True,
        "artifact_sha256":files,"prior_evidence_sha256":evidence,"explicit_byte_aliases":aliases,
        "GT_vs_predicted_p16_descriptive_not_same_data_causal_gap":gaps,"matched_only_runtime_aggregate":diag,
        "resources":core["resources"],"original_and_recovery_terminal_provenance":core["original_registration_and_operational_recovery_provenance"],
        "full_feature_validation":"All988payloads/IDs/prefixes verified; see LIMITED_MASA_FULL_FEATURE_VALIDATION.json",
        "protected_scientific_inputs_verified":len(core["protected_sha256"]),"new_training":False,"new_physical_inference":False,"new_GT_join":False,
        "Val_training_or_tuning":False,"TAO_Test_access":False,"foreign_process_interference":False,"new_asset_downloads":False,
        "completion_acceptance":"Negative research completion explicitly authorized by latest user amendment. Strong M1/end-to-end method advantage NOT achieved and not claimed.",
        "final_delivery_commit":"Recorded in subsequent delivery/log after actual test/push/remote verification; no self-referential fabricated SHA"}
    atomic_json(OUT/"FINAL_RESULT.json",result)
    print(json.dumps({k:result[k] for k in ("status","actual_fits","retained_checkpoints","unique_Val_executions","logical_Val_cases_including_aliases")}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-commit",required=True)
    main(parser.parse_args().preregistration_commit)
