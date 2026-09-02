#!/usr/bin/env python3
"""Render the Phase76 autonomous route reports from immutable JSON metrics."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float): return f"{value:.{digits}f}"
    return str(value)


def fold_rows(obj: dict[str, Any]) -> list[dict[str, Any]]:
    return obj.get("fold_results", [])


def p16_table(obj: dict[str, Any], title: str) -> str:
    rows = fold_rows(obj)
    lines = [f"### {title}", "", "| fold | queries | raw R@1 | learned R@1 | ΔR@1 | raw mAP | learned mAP | ΔmAP | Δhard-gap | unsafe | HELP/use |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        p = row.get("p16", {})
        use = p.get("router_help_rate", p.get("teacher_use_rate", "—"))
        lines.append("| {fold} | {queries} | {raw_r1} | {r1} | {delta_r1} | {raw_map} | {map} | {delta_map} | {delta_hard_gap} | {unsafe} | {use} |".format(fold=row.get("fold"), queries=p.get("queries"), raw_r1=fmt(p.get("raw_r1")), r1=fmt(p.get("r1")), delta_r1=fmt(p.get("delta_r1")), raw_map=fmt(p.get("raw_map")), map=fmt(p.get("map")), delta_map=fmt(p.get("delta_map")), delta_hard_gap=fmt(p.get("delta_hard_gap")), unsafe=p.get("unsafe_flip_count"), use=fmt(use)))
    return "\n".join(lines) + "\n"


def route_report(name: str, body: str, path: Path) -> None:
    atomic_text(path, body.rstrip() + "\n")


def main() -> None:
    ar = load("outputs/iclr27_phase76ar/metrics/phase76ar_exact_retrieval.json")
    ss = load("outputs/iclr27_phase76s/metrics/phase76s_s1_formal_exact_retrieval.json")
    gg = load("outputs/iclr27_phase76g/metrics/phase76g_g1_formal_exact_retrieval.json")
    xx = load("outputs/iclr27_phase76x/metrics/phase76x_exact_retrieval.json")
    oo = load("outputs/iclr27_phase79o/metrics/phase79o_observability.json")
    examples = load("outputs/iclr27_phase76s/audit/example_summary.json")
    meta = load("outputs/iclr27_phase76g/audit/meta_manifest_summary.json")
    ar_leg, ar_mem = ar["aggregate"]["legal_fit"], ar["aggregate"]["memory_mimic"]
    ss_a, gg_a = ss["aggregate_p16"], gg["aggregate_p16"]
    xx_leg, xx_mem = xx["aggregate"]["legal_fit"], xx["aggregate"]["memory_mimic"]

    ar_body = f"""# Phase76AR — Selective Anchored Relation Report

**Decision:** `{ar['decision']}`  
**Protocol:** frozen Phase30 TRAIN-disjoint legal-fit and memory-mimic banks; prefixes 1/2/4/8/16; 984 queries per stream; no controller/StateMemory/held/public/sealed access.

Phase76A implementation defects were corrected in a separate namespace: true dual streams, per-match quality features, bank-aware abstention, bounded `0.10*tanh` residual, raw-first safety, prefix-union hard negatives and unbiased full validation. Smoke and targeted tests passed after one BCE-shape repair; four 10,000-step workers used GPUs 4–7 under one bounded supervisor.

{p16_table({'fold_results': ar['fold_results']['legal_fit']}, 'Legal-fit stream')}
{p16_table({'fold_results': ar['fold_results']['memory_mimic']}, 'Memory-mimic stream')}

Aggregate legal p16: R@1 `{ar_leg['raw_r1']:.6f}→{ar_leg['r1']:.6f}` (Δ `{ar_leg['delta_r1']:.6f}`), mAP `{ar_leg['raw_map']:.6f}→{ar_leg['map']:.6f}`, unsafe `{ar_leg['unsafe_flip_count']}`. Memory p16: R@1 `{ar_mem['raw_r1']:.6f}→{ar_mem['r1']:.6f}`, mAP `{ar_mem['raw_map']:.6f}→{ar_mem['map']:.6f}`, unsafe `{ar_mem['unsafe_flip_count']}`. Safety held, but the registered +0.02 R@1/+0.01 mAP and three-fold improvement requirements did not. This is a safe abstention/no-generalized-signal result, not OCD=0.

Artifacts: `outputs/iclr27_phase76ar/metrics/phase76ar_exact_retrieval.json`, `outputs/iclr27_phase76ar/audit/phase76ar_decision.json`, `outputs/iclr27_phase76ar/audit/repair_events.json`.
"""
    route_report("Phase76AR", ar_body, ROOT / "docs/iclr27_phase76ar/PHASE76AR_SELECTIVE_RELATION_REPORT.md")

    ex_lines = "\n".join(f"- f{s['fold']}: fit={s['fit_examples']} val={s['val_examples']} fit labels={s['fit_label_counts']} val labels={s['val_label_counts']}" for s in examples["folds"])
    s_body = f"""# Phase76S — Selective/Abstaining Relation Router Report

**Decision:** `{ss['decision']}`  
The route froze Phase76AR and trained a HELP/HARM/NEUTRAL router; HELP alone could select relation scores and all other actions returned exact raw scores. COVTrack ICCV 2025 (`https://github.com/zekunqian/COVTrack`) was consulted only for the confidence-routing idea; its category-aware semantic cue was excluded.

## Counterfactual data

{ex_lines}

Validation examples are all NEUTRAL, so the frozen relation produced no validation counterfactual improvement for this router to learn. Fit HELP examples total 28 and HARM is zero.

{p16_table({'fold_results': ss['fold_results']}, 'Exact TRAIN-disjoint validation')}

Aggregate p16: R@1 `{ss_a['raw_r1']:.6f}→{ss_a['r1']:.6f}`, mAP `{ss_a['raw_map']:.6f}→{ss_a['map']:.6f}`, Δhard-gap `{ss_a['delta_hard_gap']:.6f}`, unsafe `{ss_a['unsafe_flip_count']}`, router HELP rate `{ss_a['router_help_rate']:.6f}`, teacher agreement `{ss_a['teacher_agreement']:.6f}`. No registered gain gate passed. No controller, StateMemory or sealed/public data was accessed.

Artifacts: `outputs/iclr27_phase76s/metrics/phase76s_s1_formal_exact_retrieval.json`, `outputs/iclr27_phase76s/audit/example_summary.json`, `outputs/iclr27_phase76s/audit/resource_preflight.json`.
"""
    route_report("Phase76S", s_body, ROOT / "docs/iclr27_phase76s/PHASE76S_SELECTIVE_ROUTER_REPORT.md")

    group_lines = "\n".join(f"- f{s['fold']}: {s['fit_categories']}; group counts `{json.dumps(meta['folds'][s['fold']]['group_counts'], sort_keys=True)}`" for s in meta["folds"])
    g_body = f"""# Phase76G — Cross-Category Meta-Holdout Report

**Decision:** `{gg['decision']}`  
Each formal fold used only its own TRAIN fit categories. Sorted fit categories were split deterministically into four groups; every update trained on three groups and used a mean-plus-0.5-worst-group cross-entropy objective. No held category was used for fitting.

## Group manifests

{group_lines}

{p16_table({'fold_results': gg['fold_results']}, 'Exact TRAIN-disjoint validation')}

Aggregate p16: R@1 `{gg_a['raw_r1']:.6f}→{gg_a['r1']:.6f}`, mAP `{gg_a['raw_map']:.6f}→{gg_a['map']:.6f}`, Δhard-gap `{gg_a['delta_hard_gap']:.6f}`, unsafe `{gg_a['unsafe_flip_count']}`, HELP rate `{gg_a['router_help_rate']:.6f}`, worst-fold ΔR@1 `{gg_a['worst_fold_delta_r1']:.6f}`. The frozen relation had no validation counterfactual signal, so robust grouping could not create a gain; Gate R failed and no controller path was authorized.

Artifacts: `outputs/iclr27_phase76g/metrics/phase76g_g1_formal_exact_retrieval.json`, `outputs/iclr27_phase76g/manifests/`, `outputs/iclr27_phase76g/audit/phase76g_decision.json`.
"""
    route_report("Phase76G", g_body, ROOT / "docs/iclr27_phase76g/PHASE76G_CROSS_CATEGORY_REPORT.md")

    x_body = f"""# Phase76X — Soft Optimal-Transport Relation Primitive Report

**Decision:** `{xx['decision']}`  
This was the single alternative primitive permitted after AR/S/G: a symmetric uniform-marginal Sinkhorn match over causal frame-feature cosine matrices (temperature 0.07, 50 iterations), with a fixed 0.5 raw-score anchor. It was parameter-free and used no GPU/training.

Official references audited before implementation: ObjectRelator (`https://github.com/insait-institute/ObjectRelator`, ICCV 2025), C3Po (`https://github.com/c3po-correspondence/C3Po`, NeurIPS 2025), Grounded Correspondence (`https://github.com/LiZhYun/ICML2026-RethinkingOCL`, ICML 2026), and SlotContrast (`https://github.com/martius-lab/slotcontrast`, CVPR 2025). Only the generic correspondence primitive was borrowed; no external code, labels, text or IDs entered inference.

{p16_table({'fold_results': xx['fold_results']['legal_fit']}, 'Legal-fit stream')}
{p16_table({'fold_results': xx['fold_results']['memory_mimic']}, 'Memory-mimic stream')}

Legal p16 improved R@1 `{xx_leg['raw_r1']:.6f}→{xx_leg['r1']:.6f}` and mAP `{xx_leg['raw_map']:.6f}→{xx_leg['map']:.6f}`, but Δhard-gap `{xx_leg['delta_hard_gap']:.6f}` and unsafe `{xx_leg['unsafe_flip_count']}` failed safety; memory unsafe was `{xx_mem['unsafe_flip_count']}` and its R@1 gain was below threshold. This is not a safe Gate R result.
"""
    route_report("Phase76X", x_body, ROOT / "docs/iclr27_phase76x/PHASE76X_SOFT_OT_REPORT.md")

    o_body = f"""# Phase79O — Causal Physical Observability Report

**Decision:** `{oo['decision']}`  
After all representation routes failed, one physical O route was tested: retain every Q0 native candidate and add a strictly prior constant-velocity projection from the latest one or two boxes (maximum gap 2). This is causal trajectory aggregation; physical IDs are bookkeeping only and are not model features.

- Native rows: `{oo['projection']['native_rows']}`; event image keys: `{oo['projection']['event_image_keys']}`.
- Synthetic candidates: `{oo['projection']['synthetic_candidates']}` on `{oo['projection']['event_keys_with_projection']}` keys.
- Raw prefix16 positive both-reliable ceiling: `{oo['raw_prefix16_both_reliable']}/76`.
- Projection prefix16 ceiling: `{oo['projection_prefix16_both_reliable']}/76`; fold counts remain `{[oo['by_fold_prefix16'][str(f)]['both_reliable'] for f in range(4)]}`.

TRA​CT (`https://github.com/Nathan-Li123/TRACT`) and COVTrack (`https://github.com/zekunqian/COVTrack`) were audited as trajectory/confidence references; text/category cues were excluded. The fixed projection did not raise the observability ceiling, so no semantic/controller route was started.
"""
    route_report("Phase79O", o_body, ROOT / "docs/iclr27_phase79o/PHASE79O_OBSERVABILITY_REPORT.md")

    commits = subprocess.check_output(["git", "log", "-8", "--format=%h %s"], cwd=str(ROOT), text=True).strip()
    final = f"""# TrackOCD Autonomous ICLR Progress — Phase76AR → Phase79O

**Generated:** {dt.datetime.now(dt.timezone.utc).isoformat()}  
**Project:** `{ROOT}`  
**Git branch:** `main`  
**Current source commits (latest first):**
```text
{commits}
```

## Executive status

**Final status: `AUTONOMOUS_RESEARCH_EXHAUSTED_WITH_NEGATIVE_EVIDENCE`.**

The five substantive routes registered by the overnight plan were executed under the frozen feature/proposal protocol:

```text
Phase76AR corrected dual-stream relation ──FAIL──> Phase76S selective router
                                             └FAIL──> Phase76G category meta-holdout
                                                        └FAIL──> Phase76X soft-OT primitive
                                                                     └FAIL──> Phase79O causal observability
```

No route earned the right to enter Phase76B counterfactual memory, Phase76C frozen-controller compatibility, controller adaptation, or sealed evaluation. This is a reliable negative for these registered routes and frozen inputs—not a claim that DINOv2 contains no information and not `OCD=0`.

## Frozen task and boundaries

- Project identity: `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT`; public repository `https://github.com/LYQ1107/TrackOCD.git`; Luna session was not closed or replaced.
- Retrieval protocol: Phase30 TRAIN/validation video/category-disjoint manifests; legal-fit and memory-mimic streams; causal prefixes `{1,2,4,8,16}`; 984 queries per stream; raw normalized mean-frame cosine comparator.
- Labels/categories/track IDs are metadata for TRAIN split construction and scoring only. Inference tensors contain visual/geometry-derived features; no category text, semantic/physical ID feature, future row, held/DEV+/Q1/public-new/sealed label, StateMemory or controller action was used.
- All denominators, candidate order, row keys, seeds and raw comparator were retained. No threshold sweep, controller run, StateMemory mutation or public/sealed access occurred.

## Historical evidence and corrections

Phase75D showed a legal pairwise teacher signal but unsafe global behavior; Phase75E's adapter drifted raw geometry and failed strict gates; Phase76R found no safe adapter checkpoint window. Phase76A's first relation implementation had no true dual stream, scalar quality, biased validation, no bank confidence/bounded delta and incomplete hard negatives. Phase76AR corrected those defects, then preserved all failures/repairs in its own namespace.

## Route results

### Phase76AR

Legal p16 raw→learned R@1 `{ar_leg['raw_r1']:.6f}→{ar_leg['r1']:.6f}`, mAP `{ar_leg['raw_map']:.6f}→{ar_leg['map']:.6f}`, unsafe `{ar_leg['unsafe_flip_count']}`. Memory R@1 `{ar_mem['raw_r1']:.6f}→{ar_mem['r1']:.6f}`, mAP `{ar_mem['raw_map']:.6f}→{ar_mem['map']:.6f}`, unsafe `{ar_mem['unsafe_flip_count']}`. Safety passed, useful-gain gate failed.

### Phase76S

The frozen relation generated only 28 fit HELP examples, zero HARM, and all validation labels were NEUTRAL. Learned p16 remained raw (R@1 `{ss_a['r1']:.6f}`, mAP `{ss_a['map']:.6f}`), unsafe zero; Gate R failed because there was no validation gain.

### Phase76G

Rotating category-group meta-holdout/group-robust training also remained raw at p16 (R@1 `{gg_a['r1']:.6f}`, mAP `{gg_a['map']:.6f}`), unsafe zero, worst-fold deltas `{gg_a['worst_fold_delta_r1']:.6f}/{gg_a['worst_fold_delta_map']:.6f}`. No hidden-category generalization signal was present.

### Phase76X

Soft-OT raised legal R@1 by `{xx_leg['delta_r1']:.6f}` and mAP by `{xx_leg['delta_map']:.6f}`, but introduced `{xx_leg['unsafe_flip_count']}` unsafe flips and reduced hard-gap. Memory introduced `{xx_mem['unsafe_flip_count']}` unsafe flips. It failed the safety and three-fold gates.

### Phase79O

The causal projection generated `{oo['projection']['synthetic_candidates']}` extra candidates but did not change raw p16 observability: `{oo['raw_prefix16_both_reliable']}/76→{oo['projection_prefix16_both_reliable']}/76`; fold counts remained `[8,2,10,5]`. Thus the optional O route did not unlock further representation work.

## MOT/OCD and sealed status

No new physical MOT model was trained in these routes. The historical Q0/Phase75B physical stream remains the only valid MOT anchor; its proposal/observability evidence is retained separately. Because every representation route failed before a safe R gate, causal StateMemory/controller and persistent Commit-CT were deliberately **not run** for this branch. It would be invalid to report missing controller metrics as zero or to substitute retrieval R@1/mAP for persistent Commit-CT.

Consequently: controller compatibility = `NOT_AUTHORIZED`; persistent Commit-CT = `NOT_RUN`; sealed/public/Q1 = `SEALED_AND_NOT_ACCESSED`.

## Resources and process integrity

- Formal AR/S/G training used at most four workers, GPU mapping fold0→4, fold1→5, fold2→6, fold3→7; GPU0/1 external processes were left untouched. Phase76X/79O were CPU-only.
- Preflight recorded ~125 GiB RAM with >100 GiB available and `/data2` as the large checkpoint/cache target; `/data1` remained nearly full but was not used for large copies. No OOM, swap or external-process termination occurred. Every formal worker completed with atomic `.launched/.done` markers and checkpoints under `/data2/usr_for_deadline/` symlinked into project outputs.
- One bounded AR exact-evaluation call exceeded the 300 s tool boundary without writing artifacts; a fold/stream resumable evaluator was added and all eight partials were completed. One Phase76S exact call omitted `PYTHONPATH=.` and failed at import before writing artifacts; the same path was rerun with the minimal environment fix. These events are retained in route logs.
- Final process scan found no Phase76/79 worker or supervisor. JSON artifacts and report files are generated atomically; historical phase files were not overwritten.

## Reproduction commands

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
PYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase76ar/evaluate_exact.py --aggregate
PYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase76s/evaluate_router_exact.py --aggregate --tag s1_formal
PYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase76g/evaluate_group_router_exact.py --aggregate --tag g1_formal
PYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase76x/run_soft_ot_exact.py
PYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase79o/run_observability_projection.py
```

## Exact artifacts

- Phase76AR: `docs/iclr27_phase76ar/PHASE76AR_SELECTIVE_RELATION_REPORT.md`, `outputs/iclr27_phase76ar/metrics/phase76ar_exact_retrieval.json`.
- Phase76S: `docs/iclr27_phase76s/PHASE76S_SELECTIVE_ROUTER_REPORT.md`, `outputs/iclr27_phase76s/audit/phase76s_decision.json`.
- Phase76G: `docs/iclr27_phase76g/PHASE76G_CROSS_CATEGORY_REPORT.md`, `outputs/iclr27_phase76g/audit/phase76g_decision.json`.
- Phase76X: `docs/iclr27_phase76x/PHASE76X_SOFT_OT_REPORT.md`, `outputs/iclr27_phase76x/audit/phase76x_decision.json`.
- Phase79O: `docs/iclr27_phase79o/PHASE79O_OBSERVABILITY_REPORT.md`, `outputs/iclr27_phase79o/audit/phase79o_decision.json`.
- Route decisions, repair events, manifests, hashes, logs and checkpoints remain in their respective `outputs/` directories; large checkpoint targets are on `/data2/usr_for_deadline/` and exposed through symlinks.

## Evidence-based next action

The current registered routes are exhausted. A future task would need a newly authorized change to the legal supervision/feature contract or a genuinely new visual representation/source—not another threshold, router, memory, controller or backbone lottery under this frozen feature protocol. Any such task must first define how its evidence can be observed causally and how it will be measured in persistent MOT+OCD, then repeat the sealed-boundary audit.
"""
    progress = ROOT / "docs/AUTONOMOUS_TRACKOCD_ICLR_PROGRESS_REPORT.md"; atomic_text(progress, final)
    decision = {"status": "AUTONOMOUS_RESEARCH_EXHAUSTED_WITH_NEGATIVE_EVIDENCE", "phase": "Phase76AR-Phase79O", "routes": {"Phase76AR": ar["decision"], "Phase76S": ss["decision"], "Phase76G": gg["decision"], "Phase76X": xx["decision"], "Phase79O": oo["decision"]}, "controller_run": False, "state_memory_run": False, "persistent_commit_ct": "NOT_RUN_NO_R_GATE", "sealed_accessed": False, "public_or_dev_accessed": False, "held_labels_used_for_training_or_selection": False, "protocol_changed": False, "route_exhaustion": True, "next_action": "new authorization required for supervision/representation contract or causal proposal source; do not tune frozen controller/gates"}
    atomic_text(ROOT / "outputs/iclr27_phase79o/audit/autonomous_research_decision.json", json.dumps(decision, indent=2, sort_keys=True) + "\n")
    atomic_text(ROOT / "outputs/iclr27_phase79o/completion/autonomous_research.done", json.dumps({"status": decision["status"], "report": str(progress), "decision": str(ROOT / "outputs/iclr27_phase79o/audit/autonomous_research_decision.json")}, indent=2) + "\n")


if __name__ == "__main__": main()
