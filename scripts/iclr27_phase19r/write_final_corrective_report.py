#!/usr/bin/env python
"""Generate the self-contained Phase19R corrective-stop report."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase19r"
REPORT = ROOT / "docs/iclr27_phase19r/PHASE19R_CORRECTNESS_REPAIR_COMPLETE_REPORT.md"


def read(p: Path):
    return json.loads(p.read_text())


def fmt(x, digits=3):
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def get_metric(source: str, fold: int) -> dict:
    if source == "mixed baseline":
        d = read(OUT / "metrics" / f"fold{fold}_training.json")
        e = next(x for x in d["logs"] if int(x["step"]) == 8000)
        return e["persistent_event_validation"]["metrics"]
    d = read(OUT / "metrics" / (f"event_aligned_f{fold}_4000.json" if source == "event-aligned" else f"event_repair_f{fold}_4000.json"))
    return d["logs"][-1]["validation"]["persistent_event_metrics"]


def row(source: str, fold: int) -> dict:
    m = get_metric(source, fold)
    return {"source": source, "fold": fold, "ct": f"{m['commit_ct']['correct']}/{m['commit_ct']['eligible']}",
            "cat": m.get("category_coverage", 0), "video": m.get("video_coverage", 0),
            "ep": m.get("existing_precision", 0.0), "er": m.get("existing_recall", 0.0),
            "fm": m.get("negative_false_merge_rate", m.get("false_merge_rate_macro", 0.0)),
            "kmicro": m.get("known_micro", 0.0), "kmacro": m.get("known_macro", 0.0),
            "unres": m.get("unresolved_rate", 0.0), "dup": m.get("duplicate_births", 0)}


def table(rows: list[dict]) -> str:
    s = "| run / fold | Commit-CT | category cov. | video cov. | existing P | existing R | false merge | known micro | known macro | unresolved | duplicate births |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in rows:
        s += f"| {r['source']} / F{r['fold']} | {r['ct']} | {r['cat']} | {r['video']} | {fmt(r['ep'])} | {fmt(r['er'])} | {fmt(r['fm'])} | {fmt(r['kmicro'])} | {fmt(r['kmacro'])} | {fmt(r['unres'])} | {r['dup']} |\n"
    return s


def main() -> None:
    decision = read(OUT / "audit/phase19r_corrective_decision.json")
    mismatch = read(OUT / "audit/event_mismatch.json")
    bench = read(OUT / "metrics/acceleration_benchmark.json")
    fast = read(OUT / "metrics/acceleration_benchmark_fast.json")
    eq = read(OUT / "audit/acceleration_equivalence.json")
    repair_rows = [row("event-repair", f) for f in range(4)]
    event_rows = [row("event-aligned", f) for f in range(4)]
    mixed_rows = [row("mixed baseline", f) for f in range(4)]
    all_rows = mixed_rows + event_rows + repair_rows
    cmp = decision["authoritative_comparison"]
    report = f"""# TrackOCD ICLR 2027 — Phase 19R Correctness Repair Complete Report

**Generated:** {datetime.now(timezone.utc).isoformat()}  
**Decision:** `{decision['decision_code']}`  
**Project:** `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT`  
**Scope:** this report covers the corrective continuation after Phase 19R; the Phase 19R namespace is independent of the historical Phase 19 implementation.

## 1. Final decision

The internal gate failed. Training stops at the completed 4,000-update four-fold event-aligned runs and their one evidence-based repair. No 12k/16k extension, third event-ratio/weight/threshold trial, final all-known 50k run, final freeze, or public evaluation was launched after the stop instruction. **Public new-model labels remain sealed.** The machine-readable decision and all per-fold fields are in [`phase19r_corrective_decision.json`](../../outputs/iclr27_phase19r/audit/phase19r_corrective_decision.json).

The required comparison is:

- authoritative mixed baseline: **2/76** persistent Commit-CT;
- first event-aligned (50% event + 50% mixed): **2/76**;
- second event-repair (over-defer commit-margin loss): **0/76**.

The repair is strictly worse than both the baseline and first round, removes the two fold-3 successes, has zero existing precision/recall in every fold, and satisfies none of the required improvement/safety conditions. Training loss is reported only as a diagnostic and is not evidence of task success.

## 2. What was preserved and what was not changed

The online MOT constraint, OCD constraint, causal evaluator, Physical-ID ≠ Semantic-ID rule, DSCT/current best physical tracking, class-agnostic objectness, corrected bounding-box/data pipeline, and frozen known-stage protocol were left intact. No RACC/TOSE/KPOC/known-gate/classifier/threshold/memory/lifecycle redesign was introduced. The controller remains the Phase19R RC-MS-OCD implementation with the shared persistent `StateMemory` transition core.

Phase 8–10 had already shown that memory/lifecycle/adapter tuning did not produce true cross-track CT-Reuse. Phase 11 foundation probes and Phase 12–14 correspondence audits likewise did not establish a learnable semantic alignment. Phase 15–19 repaired causal accounting and demonstrated that synthetic episode proxies can look useful while the persistent event evaluator remains near zero. This continuation therefore tested execution speed and rollout/evaluator alignment rather than adding another memory module.

## 3. Interrupted final process

The low-efficiency all-known final process was task-owned PID **20775** on GPU3. It had run for **09:32:34** wall time and had produced neither `final_rc_ms_latest.pt` nor `final_rc_ms_best.pt`; it had no children. It was terminated explicitly with SIGTERM, without a broad pattern and without touching any external process. Logs, configuration and incomplete output were retained; no progress was recoverable. The incident is recorded in [`research_log.md`](../../research_log.md). No replacement final run was started.

## 4. Acceleration implementation and equivalence

Implemented under `src/iclr27_phase19r` and `scripts/iclr27_phase19r`:

1. A fold/split/category/feature-source/prefix-rule keyed hard-pair cache computes ordered pair scores with batched matrix multiplication and atomically writes a small `.npz`; source feature arrays are not copied.
2. Feature-free JSONL episode-index shards store only track keys, causal prefix positions, role/category loss metadata, target overrides and masks. Features are resolved from the existing source at replay time; atomic metadata includes a SHA-256.
3. Training-only fast state banks keep causal sequential updates but defer trace/anchor host copies; candidate scoring uses already-normalized state tensors and BF16 autocast where available. Evaluator mode remains the full trace path.
4. Four folds use one GPU per fold rather than DDP. GPU1 was occupied by another task and was not touched; the corrective workers used GPUs 0,2,3,4. RAM remained above the 25% safety floor and no OOM/near-OOM/swap incident occurred.
5. Latest recovery checkpoints are written every 1,000 updates while formal validation remains at 4,000; optimizer/scheduler/RNG/global update/index cursor and prototype hash are serialized.

The fixed-episode equivalence artifact reports `passed=true`: hard-pair keys match, controller-logit max absolute difference is 0, loss difference is 0, and one optimizer-step parameter difference is 0 ([`acceleration_equivalence.json`](../../outputs/iclr27_phase19r/audit/acceleration_equivalence.json)).

## 5. Speed measurements

The old steady fold throughput was **0.877–0.907 updates/s**. The initial cache/index benchmark improved 500-update throughput from 1.222 to 1.242 updates/s (1.016×) because rollout/state-machine CPU work dominated. Three bounded hotspot repairs were profiled; the fastest state-bank path reached **1.946 updates/s (1.592×)**, still below the strict 2× requirement. The first event-aligned folds measured **1.742–1.811 updates/s**; the commit-margin repair measured **1.538–1.701 updates/s** because the extra loss adds work.

The raw benchmark records generation, rollout, backward, validation, RSS and GPU snapshots in [`acceleration_benchmark.json`](../../outputs/iclr27_phase19r/metrics/acceleration_benchmark.json) and [`acceleration_benchmark_fast.json`](../../outputs/iclr27_phase19r/metrics/acceleration_benchmark_fast.json). Thus the result is a substantial practical speedup, **not** a claim of stable 2× throughput on every fold/configuration.

## 6. Persistent-event mismatch audit

The per-event audit is [`event_mismatch.json`](../../outputs/iclr27_phase19r/audit/event_mismatch.json). It provides evidence, not a post-hoc explanation:

- synthetic episode validation had existing precision approximately 1.0 and existing recall approximately 0.35–0.42;
- the persistent evaluator had existing precision/recall 0 on the comparable mixed baseline;
- observed error labels (events may receive multiple labels) were unresolved/over-defer **58**, premature pre-prefix commit **48**, duplicate target births **84**, wrong existing state **24**, false merge **26**;
- the mismatch is consistent with one-prefix balanced episodes not reproducing continuous multi-prefix, cross-video persistent state evolution, capacity pressure and evaluator class imbalance.

It is **not proved** whether the remaining cause is insufficient stable cross-instance separability in frozen DINOv2, residual rollout/state-evolution non-equivalence, or both. Those are hypotheses for the next study, not claims established by this experiment.

## 7. Legal event-aligned experiment

Pseudo-held events were generated separately for each fold from supported TRAIN categories with at least two fit videos. The masked category IDs were used only as loss-side metadata and episode-local known masks; no held, DEV+, Q1 or public new-model labels entered model input, checkpoint selection or training. The event index shards contain no feature arrays and are independently hashed. Manifests and summaries are under [`outputs/iclr27_phase19r/manifests`](../../outputs/iclr27_phase19r/manifests) and [`outputs/iclr27_phase19r/audit`](../../outputs/iclr27_phase19r/audit).

## 8. Results by fold

The table uses the same 76 persistent held-known event denominators and the exact causal evaluator terms. The mixed baseline row is the preregistered comparable step-8,000 checkpoint (aggregate 2/76); the first and second rows are the completed 4,000-update corrective runs.

{table(all_rows)}

Aggregates from the machine-readable artifact:

| run | Commit-CT | category coverage sum | video coverage sum | mean existing P/R | mean false merge | mean known micro/macro | mean unresolved | duplicate births |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mixed baseline | {cmp['mixed_baseline']['commit_ct']['correct']}/{cmp['mixed_baseline']['commit_ct']['eligible']} | {cmp['mixed_baseline']['category_coverage_sum']} | {cmp['mixed_baseline']['video_coverage_sum']} | {fmt(cmp['mixed_baseline']['existing_precision_mean'])}/{fmt(cmp['mixed_baseline']['existing_recall_mean'])} | {fmt(cmp['mixed_baseline']['negative_false_merge_mean'])} | {fmt(cmp['mixed_baseline']['known_micro_mean'])}/{fmt(cmp['mixed_baseline']['known_macro_mean'])} | {fmt(cmp['mixed_baseline']['unresolved_mean'])} | {cmp['mixed_baseline']['duplicate_births']} |
| event-aligned | {cmp['event_aligned']['commit_ct']['correct']}/{cmp['event_aligned']['commit_ct']['eligible']} | {cmp['event_aligned']['category_coverage_sum']} | {cmp['event_aligned']['video_coverage_sum']} | {fmt(cmp['event_aligned']['existing_precision_mean'])}/{fmt(cmp['event_aligned']['existing_recall_mean'])} | {fmt(cmp['event_aligned']['negative_false_merge_mean'])} | {fmt(cmp['event_aligned']['known_micro_mean'])}/{fmt(cmp['event_aligned']['known_macro_mean'])} | {fmt(cmp['event_aligned']['unresolved_mean'])} | {cmp['event_aligned']['duplicate_births']} |
| event-repair | {cmp['event_repair']['commit_ct']['correct']}/{cmp['event_repair']['commit_ct']['eligible']} | {cmp['event_repair']['category_coverage_sum']} | {cmp['event_repair']['video_coverage_sum']} | {fmt(cmp['event_repair']['existing_precision_mean'])}/{fmt(cmp['event_repair']['existing_recall_mean'])} | {fmt(cmp['event_repair']['negative_false_merge_mean'])} | {fmt(cmp['event_repair']['known_micro_mean'])}/{fmt(cmp['event_repair']['known_macro_mean'])} | {fmt(cmp['event_repair']['unresolved_mean'])} | {cmp['event_repair']['duplicate_births']} |

The gate required aggregate Commit-CT to be clearly above 2/76, at least 3/4 folds to improve in the same direction, and no safety degradation. The first run tied the baseline and the repair regressed to 0/76; neither run passes.

## 9. Integrity and public boundary

- Four first-round `.done` markers and four repair `.done` markers exist; all summaries report 4,000 finite updates and best step 4,000.
- Each repair checkpoint has a recorded SHA-256 and unchanged before/after prototype hash in [`phase19r_corrective_decision.json`](../../outputs/iclr27_phase19r/audit/phase19r_corrective_decision.json).
- The final repair workers and supervisor exited cleanly; no Phase19R training process remained at the closing check.
- No `public_predictions.frozen`, `prediction_freeze.json`, `public_after_freeze.json` or `public_gate.json` artifact was created in this corrective branch. Public labels were not opened.
- Existing Phase 19/19R historical artifacts were not overwritten to disguise this negative result.

## 10. Literature and limitations

The verified official method/repository audit remains [`PHASE19R_OFFICIAL_METHOD_AUDIT.md`](PHASE19R_OFFICIAL_METHOD_AUDIT.md); this stop experiment did not introduce an unverified foundation method or claim an exact AGE/TALON reproduction. The measured controller is small and frozen-DINOv2 based; event-aligned training used pseudo-held supervision from TRAIN metadata, not Q1 labels. Therefore the negative result rules out this corrective path under the fixed evaluator, but does not prove that all possible cross-instance representations or the task formulation are impossible.

## 11. Next-stage recommendation

Stop threshold, lifecycle, memory, commit-weight and controller-architecture tuning on this branch. First establish a separately verifiable **cross-instance semantic correspondence / representation-learning baseline** (with held-out category/video protocols, direct retrieval and per-event causal checks). Only after that baseline demonstrates stable cross-track alignment should a new online state controller be designed around it.

## 12. Artifact index

- [`phase19r_corrective_decision.json`](../../outputs/iclr27_phase19r/audit/phase19r_corrective_decision.json)
- [`event_mismatch.json`](../../outputs/iclr27_phase19r/audit/event_mismatch.json)
- [`acceleration_equivalence.json`](../../outputs/iclr27_phase19r/audit/acceleration_equivalence.json)
- [`acceleration_benchmark.json`](../../outputs/iclr27_phase19r/metrics/acceleration_benchmark.json)
- [`acceleration_benchmark_fast.json`](../../outputs/iclr27_phase19r/metrics/acceleration_benchmark_fast.json)
- [`research_log.md`](../../research_log.md)
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT.with_name(REPORT.name + ".tmp"); tmp.write_text(report); tmp.replace(REPORT)
    print(str(REPORT))


if __name__ == "__main__": main()
