"""Write the self-contained Phase18 final report from locked artifacts."""
from __future__ import annotations
import json, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/iclr27_phase18'; DOC=ROOT/'docs/iclr27_phase18'
def j(p): return json.loads((OUT/p).read_text())
def link(label,path): return f'[{label}]({path})'
def main():
    pub=j('eval/public_crossfit_result.json'); dec=j('eval/phase18_decision.json'); agg=pub['repair_R1_three_seed_aggregate']
    rows=[]
    for x in [pub['strongest_baseline'],pub['main_DSTM_seed1801'],*pub['repair_R1_seed_results'],pub['ablations']['B3'],pub['ablations']['without_merge'],pub['ablations']['without_history']]:
        rows.append(f"| {x['candidate']} | {x['commit_ct']['correct']}/{x['commit_ct']['eligible']} | {x['post_prefix_ct']['correct_rows']}/{x['post_prefix_ct']['rows']} | {x['existing_precision']:.3f} | {x['negative_false_merge_rate']:.3f} | {x['known_category_macro']:.3f} | {x['correct_categories']} / {x['correct_target_videos']} |")
    report=f'''# TrackOCD ICLR 2027 — Phase 18 complete report

## Executive verdict

Phase18 makes the revised causal cross-video CT task identifiable, but the trained Deferred Semantic Tracklet Memory (DSTM) and its one registered R1 repair do not pass the frozen public method gate. The terminal decision is **`{dec['decision_code']}`**; see {link('phase18_decision.json','../../outputs/iclr27_phase18/eval/phase18_decision.json')}. The public cross-fit population is development evidence, not a blind external test.

The legal oracle ceiling is nonzero: O1 and O5 each achieve **41/41 Commit-CT**, while O0 is also 41/41. B1 and B2 each achieve **9/41**, DSTM seed 1801 achieves **0/41**, and R1 seeds 1801/1802/1803 achieve **8/41, 12/41, 9/41**. The complete machine-readable result is {link('public_crossfit_result.json','../../outputs/iclr27_phase18/eval/public_crossfit_result.json')}.

R1 improves recovery on the 15 unreliable-prefix events (5, 6, and 5 correct commits per seed versus B3's zero), but its pooled existing precision is **0.457**, pooled false-merge rate **0.228**, mean Known category-macro **0.054**, and per-seed correct category/video coverage is only 1–3 / 3–6. It therefore fails both the method and DEV+ gates.

## 1. Phase17R diagnosis

The independent reproduction is {link('phase17r_terminal_diagnosis_reproduction.json','../../outputs/iclr27_phase18/eval/phase17r_terminal_diagnosis_reproduction.json')}. The historical audit categories are 267, 831, and 1014. Every registered order has 30 CT rows and zero current rows satisfying `assigned == 1 && row_iou >= 0.5`; 23 rows have exact zero IoU and 7 are positive but below 0.5. Category 267 has one reliable video, 831 has one, and 1014 has none, so no audit direction has both a reliable source and target. The independent calibration counts 67/53/45 are reproduced.

Code inspection confirms Phase17R `oracle_both_routing` oracleizes routing/observability but still uses the learned semantic pair scorer; it is not a perfect correspondence oracle. The replay also tests local novel continuity before global matching, preventing a provisional local state from later switching to a correct earlier global state. Phase17R files were not edited.

## 2. Frozen Phase18 protocol and identifiability

Phase18 retains the frozen physical tracker/objectness stream and separates physical track IDs, 48 supported-known category IDs, and opaque online novel semantic IDs beginning at 100000. Actions are `KNOWN`, `DEFER`, `NEW_NOVEL`, and `EXISTING_NOVEL`. `DEFER` updates only a local causal buffer; later reliable evidence may remap current/future mapping, while past actions and duplicate births remain immutable. The complete contract is {link('PROTOCOL.md','PROTOCOL.md')} and the machine contract is {link('task_contract.json','../../outputs/iclr27_phase18/manifests/task_contract.json')}.

The source population has **43,423 unique rows**. DINOv2/DINOv3 row-key sets match exactly; DINOv3 order is exact and DINOv2 is reindexed by immutable row key ({link('feature_alignment.json','../../outputs/iclr27_phase18/manifests/feature_alignment.json')}). The prediction-independent census has **11 eligible categories, 377 rows, 221 reliable rows, 41 positive events, 41 matched negatives, 28 unique target tracklets, 435 post-prefix rows, and 15 unreliable-prefix positives** ({link('eligible_category_census.json','../../outputs/iclr27_phase18/manifests/eligible_category_census.json')}, {link('identifiable_ct_denominators.json','../../outputs/iclr27_phase18/manifests/identifiable_ct_denominators.json')}).

F0–F3 held categories are {{20,965,1078}}, {{32,436,611}}, {{31,882,1108}}, and {{726,875}}. Nested calibration categories are excluded from fitting; every video containing a held category is excluded from that fold's fitting rows. Fold hash is `8601a12b…`, denominator hash is `a37f3663…`, and the fresh public lock records that no Phase18 learned prediction existed at freeze ({link('fold_manifest.json','../../outputs/iclr27_phase18/manifests/fold_manifest.json')}, {link('public_lock.json','../../outputs/iclr27_phase18/manifests/public_lock.json')}).

### Legal and unconstrained oracles

| Control | Commit-CT | Meaning |
|---|---:|---|
| O0 semantic-label oracle | 41/41 | unconstrained ceiling, not deployable |
| O1 legal reliability + semantic oracle | 41/41 | task-identifiability proof |
| O2 semantic oracle + learned readiness | DSTM 20/41; R1 29/41, 32/41, 35/41 | readiness alone is not sufficient |
| O3 learned semantics + exact readiness | DSTM 1/41; R1 11/41, 12/41, 10/41 | semantic-state learning remains weak |
| O4 old local-first perfect pair | 26/41 | premature local birth failure |
| O5 merge-capable perfect pair | 41/41 | legal defer/merge is sufficient |

All controls and learned O2/O3 values are in {link('oracle_contracts.json','../../outputs/iclr27_phase18/eval/oracle_contracts.json')}. O1's nonzero result rules out `P18-T4` still-unidentifiable as the terminal explanation.

## 3. Official method audit

The full audit is {link('OFFICIAL_METHOD_AUDIT.md','OFFICIAL_METHOD_AUDIT.md')}. Verified official repositories/commits/licenses were: [MOTIP](https://github.com/MCG-NJU/MOTIP) `ffc0e905…` (Apache-2.0, CVPR 2025), [OVTR](https://github.com/jinyanglii/OVTR) `500e72c…` (MIT, ICLR 2025), [HATReID-MOT](https://github.com/MCG-NJU/HATReID-MOT) `3eb440c…` (Apache-2.0, repository ECCV 2026), [V-JEPA 2](https://github.com/facebookresearch/vjepa2) `204698b…` (mostly MIT), [RADIO](https://github.com/NVlabs/RADIO) `c0f3701…` (NVIDIA terms), and [Track-On2](https://github.com/MaximilianToelle/track_on2_stream) `311d2cb…` (MIT). No external code lines or checkpoint were copied. DSTM reimplemented only compatible set-conditioned candidate decoding and bounded causal-memory ideas; no exact OVTR/V-JEPA/RADIO mapping was available.

## 4. Baselines and DSTM

Baseline architectures and results are in {link('baseline_results.json','../../outputs/iclr27_phase18/eval/baseline_results.json')}. B1 is the causal DINOv2 CLS/ROI tracklet prototype with calibrated readiness, DEFER, and explicit NEW/EXISTING thresholds. B2 trains a fold-local balanced logistic pair scorer on all legal cross-video novel pairs; only 3–8 positive and matched-negative pairs exist per fold, reflecting the actual legal population. B2 exactly matches B1 at 9/41. B3 is the same-capacity no-DEFER/no-merge control.

DSTM uses DINOv2 CLS/ROI plus 15 causal geometry fields, a single-layer GRU over the latest 32 rows, 48 known tokens, a variable permutation-invariant novel-state set, learned NEW/DEFER tokens, one transformer set layer, and four-head query-to-set attention. Global memory is capped at 32 states and 8 reliable anchors. Registered loss weights, optimizer, 20,000-update folds, deterministic stream, and balanced episodes are frozen in {link('dstm.json','../../configs/iclr27_phase18/dstm.json')} and described in {link('METHOD.md','METHOD.md')}.

DSTM seed 1801 completed 80,000 updates across four folds, all gradients finite, with best calibration steps 8000/1000/2000/6000 ({link('main_training_summary.json','../../outputs/iclr27_phase18/eval/main_training_summary.json')}). It collapsed toward DEFER: pre-prefix defer 1.0, unresolved 0.902, 8 duplicate births, and 0 Commit-CT.

## 5. Results and essential ablations

| Candidate | Commit-CT | Post-prefix CT | Existing precision | False merge | Known macro | Correct cats/videos |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

R1 was the only authorized repair, selected after the complete main sweep for DEFER/action collapse. It uses a two-stage readiness→identity path, masks DEFER after readiness, and trains identity on exact-reliable examples. The three full cross-fit seed artifacts are {link('seed1801','../../outputs/iclr27_phase18/eval/repair_r1_seed1801_crossfit.json')}, {link('seed1802','../../outputs/iclr27_phase18/eval/repair_r1_seed1802_crossfit.json')}, and {link('seed1803','../../outputs/iclr27_phase18/eval/repair_r1_seed1803_crossfit.json')}. The aggregate includes pooled counts, seed means, strata, coverage-risk, recovery, and clustered uncertainty in {link('public_crossfit_result.json','../../outputs/iclr27_phase18/eval/public_crossfit_result.json')}.

R1's category-clustered mean delta versus B1 is **{agg['uncertainty_vs_B1']['category_mean_delta_point']:.4f}**, with **{agg['uncertainty_vs_B1']['positive_category_count']}** positive category deltas; its descriptive 95% clustered bootstrap interval is [{agg['uncertainty_vs_B1']['category_clustered_bootstrap_95']['low']:.4f}, {agg['uncertainty_vs_B1']['category_clustered_bootstrap_95']['high']:.4f}]. The result is not treated as Gaussian external inference.

The first B3 attempt failed on an empty global-anchor shape path during calibration. The failure log and markers remain visible; the minimal repair writes the current observation as a deliberately contaminated B3 anchor. A 5-step smoke and full causality/alignment/transition regression passed ({link('smoke_b3_cycle1_summary.json','../../outputs/iclr27_phase18/eval/smoke_b3_cycle1_summary.json')}, {link('transition_and_causality_contract.json','../../outputs/iclr27_phase18/eval/transition_and_causality_contract.json')}).

## 6. Legacy stress, strata, and uncertainty

The unchanged Phase17R immediate-action stress replay is 0/30 for all three historical orders; it is explicitly diagnostic-only and not the Phase18 primary metric ({link('legacy_immediate_ct_stress.json','../../outputs/iclr27_phase18/eval/legacy_immediate_ct_stress.json')}). Per-category, target-video, object-size, tracklet-length, prefix-length, post-prefix-quality, and coverage-risk strata are retained in each cross-fit JSON and summarized in {link('public_crossfit_result.json','../../outputs/iclr27_phase18/eval/public_crossfit_result.json')}.

The primary population is small: 11 categories, 28 unique target tracklets, 41 directed appearances, and some categories with only two events. Category/video clustered intervals are descriptive. R1's seed variation (8, 12, 9) and category concentration (notably categories 875 and 1108) are therefore central negative evidence, not averaged away.

## 7. DEV+, Q1, resources, and incidents

The public candidate failed the preregistered method/DEV+ gate, so DEV+ and Q1 were not accessed. The final lock is {link('external_evaluation_lock.json','../../outputs/iclr27_phase18/manifests/external_evaluation_lock.json')}; no external labels, checkpoint, threshold, or post-prediction tuning entered the result.

Resource accounting is {link('resource_summary.json','../../outputs/iclr27_phase18/eval/resource_summary.json')}. Phase18 output remained about 950 MiB and official reference checkouts about 372 MiB; data/features are reused through symlinks. At most four Phase18 workers ran concurrently, RAM supervisors enforced the 25% MemAvailable floor, and no OOM, near-OOM, swap use, or other-user termination occurred. One terminal wait handle was reclaimed while `setsid` children completed; exact markers prevented duplicate launches. External GPU occupancy caused dynamic scheduling but no external process was touched.

## 8. Supported/unsupported claims and limitations

Supported: the revised Phase18 task is causally identifiable; legal DEFER-plus-merge semantics can attain its fixed denominator under perfect correspondence; Phase17R's audit was not a two-sided reliable CT opportunity; B1/B2 are stronger and more stable than this DSTM on the public development population; and R1 localizes a learned action/semantic-state failure rather than evaluator impossibility.

Unsupported: no Phase18 method improvement, no DEV+/Q1 generalization, no claim that any audited external method solves TrackOCD, and no reinterpretation of the legacy 0/30 score as Phase18 primary performance. The small eligible population and limited legal cross-instance pairs prevent a strong external statistical claim.

## 9. Next evidence-based ICLR plan

Stop Phase18 memory, lifecycle, decoder, and threshold tuning. Preserve the protocol, denominator, baselines, oracle contracts, and negative results. A next study should introduce independently justified cross-instance semantic supervision or a verified foundation representation, first testing cross-video retrieval/state discrimination on all 11 eligible categories under the same causal contract. If that representation cannot produce stable O3 correspondence, narrow or redefine the semantic-correspondence claim.

## Artifact index

- Protocol: {link('PROTOCOL.md','PROTOCOL.md')}, {link('METHOD.md','METHOD.md')}, {link('OFFICIAL_METHOD_AUDIT.md','OFFICIAL_METHOD_AUDIT.md')}
- Data/folds: {link('eligible_category_census.json','../../outputs/iclr27_phase18/manifests/eligible_category_census.json')}, {link('fold_manifest.json','../../outputs/iclr27_phase18/manifests/fold_manifest.json')}, {link('feature_alignment.json','../../outputs/iclr27_phase18/manifests/feature_alignment.json')}
- Results/decision: {link('baseline_results.json','../../outputs/iclr27_phase18/eval/baseline_results.json')}, {link('public_crossfit_result.json','../../outputs/iclr27_phase18/eval/public_crossfit_result.json')}, {link('phase18_decision.json','../../outputs/iclr27_phase18/eval/phase18_decision.json')}, {link('resource_summary.json','../../outputs/iclr27_phase18/eval/resource_summary.json')}
'''
    path=DOC/'PHASE18_DEFERRED_SEMANTIC_MEMORY_COMPLETE_REPORT.md'; tmp=path.with_suffix('.md.tmp'); tmp.write_text(report); os.replace(tmp,path)
    print(path)
if __name__=='__main__': main()
