#!/usr/bin/env python3
"""Render Phase76A retrieval/contract evidence from machine artifacts."""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase76a"
DOC = ROOT / "docs/iclr27_phase76a/PHASE76A_RAW_ANCHORED_LOCAL_RELATION_REPORT.md"


def p(x: float) -> str: return f"{float(x):.6f}"


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as h:
            json.dump(value, h, indent=2, sort_keys=True, allow_nan=False); h.write("\n"); h.flush(); os.fsync(h.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def main() -> None:
    exact = json.loads((OUT / "metrics/exact_retrieval.json").read_text())
    prep = json.loads((OUT / "banks/prepare_summary.json").read_text())
    smoke = json.loads((OUT / "audit/contract_smoke.json").read_text())
    parity = json.loads((OUT / "audit/raw_anchor_parity.json").read_text())
    lit = json.loads((OUT / "audit/literature_audit.json").read_text())
    repair = json.loads((OUT / "audit/repair_events.json").read_text())
    decision = json.loads((OUT / "audit/phase76a_decision.json").read_text())
    pre = json.loads((OUT / "audit/formal_preflight.json").read_text()); post = json.loads((OUT / "audit/formal_postflight.json").read_text())
    lines = [
        "# TrackOCD Phase76A — Raw-Anchored Local Relation Reranker Report", "",
        f"Generated {dt.datetime.now(dt.timezone.utc).isoformat()} at git `{subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=False).stdout.strip()}`.", "",
        "## Decision and boundary", "",
        f"- Decision: **`{decision['decision']}`**. Gate R76A fails; no StateMemory, controller, Commit-CT, public/Q1, DEV+, held-label or sealed evaluation was run.",
        "- Phase75D/E historical reports and metrics are unchanged. This route is a bounded candidate-level relation experiment, not a replacement embedding and not an OCD result.",
        "- The global scorer is always the frozen raw cosine; the learned branch is used only on the explicit ≤15-candidate local bank.", "",
        "## Phase76R contract errata carried forward", "",
        "- Historical Phase75D teacher authorization was an erratum: legal-only guard returned true, corrected global guard is false for global folds 0 and 2 (ΔR1 < −0.02).",
        "- Phase75E actual optimizer was constant Adam 4e−5 (no scheduler), formal seeds were 750500+fold, and legal-first selection omitted global safety/drift constraints. See `outputs/iclr27_phase76r/audit/*erratum.json` and its report.",
        "- All 120 Phase75E checkpoints were audited; no safe feature-adapter window existed, motivating this structurally different route.", "",
        "## Contract and implementation", "",
        "The model uses normalized causal frame sets Q/C, detached CPU Hungarian indices, symmetric `concat(abs(q−c), q*c)` 1536-D match tokens, the registered 13-field summary, a 1536→256→LayerNorm→GELU→128 token MLP, 5→32→1 sigmoid quality MLP, and zero-initialized scalar delta/confidence heads. Final candidate score is `raw_cosine + confidence*delta`; at step zero delta=0 and confidence=.5, so the score is exactly raw. No 768-D object embedding is emitted and no category, text, semantic/physical ID, future or held label is an inference input.", "",
        "### Candidate-bank inventory", "",
        "| fold | fit banks | fit pair entries | val banks | val pair entries |",
        "|---:|---:|---:|---:|---:|",
    ]
    by = {(x["fold"], x["split"]): x for x in prep["summary"]}
    for f in range(4): lines.append(f"| {f} | {by[(f,'fit')]['banks']} | {by[(f,'fit')]['pairs']} | {by[(f,'val')]['banks']} | {by[(f,'val')]['pairs']} |")
    lines += [
        "", "Fit banks contain same-category/different-video positives (≤3) and deterministic raw top-12 different-category/different-video hard negatives (≤15 candidates). Validation banks use only Phase30 TRAIN-derived validation metadata and are never held/Q1/public data.", "",
        "## Contract smoke and raw parity", "",
        f"- Pair tokens `{smoke['pair_feature_dim']}`-D, summary `{smoke['summary_dim']}`-D, finite output, step-zero exact raw: `{smoke['step0_exact_raw']}`; standalone OVTR-environment smoke passed.",
        f"- Raw structural parity: `{parity['pass']}` for all 20 fold×prefix checks at tolerance 1e−7. The initial ≤1.79e−7 redundant-normalization failure is retained as `raw_anchor_parity_failed_r1.json`; the minimal operation-order fix passed.", "",
        "## Formal TRAIN-disjoint training", "",
        "Four workers ran 20,000 updates on GPUs 4/5/6/7 under one bounded supervisor. AdamW started at 1e−4 with 1000-step linear warmup and cosine to 1e−5, weight decay 1e−4, gradient clip 1.0; validation/checkpoint every 500. Deterministic hash-shuffled cycle visits were balanced (max−min=1). Best checkpoints were selected only from bounded Phase30 validation (f0 step1000, f1 step12500, f2 step8000, f3 step17500).", "",
        "| fold | best step | val queries | raw R1 | learned R1 | ΔR1 | raw mAP | learned mAP | ΔmAP | Δhard gap | unsafe |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in exact["fold_results"]:
        x=f["p16"]; lines.append(f"| {f['fold']} | {int(Path(f['best_checkpoint']).stem.split('step')[-1]) if 'step' in Path(f['best_checkpoint']).stem else '-'} | {x['queries']} | {p(x['raw_r1'])} | {p(x['r1'])} | {p(x['delta_r1'])} | {p(x['raw_map'])} | {p(x['map'])} | {p(x['delta_map'])} | {p(x['delta_hard_gap'])} | {x['unsafe_flip_count']} |")
    lines += [
        "", "## Exact validation aggregate and prefix safety", "",
        f"At p16 the query-micro aggregate is raw R@1 `{p(exact['aggregate_p16']['raw_r1'])}` → learned `{p(exact['aggregate_p16']['r1'])}` (Δ `{p(exact['aggregate_p16']['delta_r1'])}`), raw mAP `{p(exact['aggregate_p16']['raw_map'])}` → `{p(exact['aggregate_p16']['map'])}` (Δ `{p(exact['aggregate_p16']['delta_map'])}`), hard-gap Δ `{p(exact['aggregate_p16']['delta_hard_gap'])}`, unsafe `{exact['aggregate_p16']['unsafe_flip_count']}/{exact['aggregate_p16']['queries']}`.", "",
        "| prefix | mean ΔR1 | mean ΔmAP | unsafe flips |",
        "|---:|---:|---:|---:|",
    ]
    for x in exact["prefix_gate"]: lines.append(f"| {x['prefix']} | {p(x['delta_r1'])} | {p(x['delta_map'])} | {x['unsafe']} |")
    lines += [
        "", "All five prefixes have non-zero unsafe flips and negative aggregate deltas. Fold0 is the only positive validation fold; folds1–3 regress in R1/mAP and hard-gap. This is not a three-fold generalization result.", "",
        "## Memory-mimic diagnostic", "",
        "The bounded fit-bank memory-mimic is in-sample evidence only (candidate bank ≤15): each fold has zero unsafe and large positive ΔmAP/Δgap, but this does not override the independent validation failure or authorize StateMemory. It is retained to distinguish a usable relation operation from cross-fold generalization.", "",
        "| fold | ΔR1 | ΔmAP | Δhard gap | unsafe |",
        "|---:|---:|---:|---:|---:|",
    ]
    for x in exact["memory_mimic"]:
        y=x["p16"]; lines.append(f"| {x['fold']} | {p(y['delta_r1'])} | {p(y['delta_map'])} | {p(y['delta_hard_gap'])} | {y['unsafe_flip_count']} |")
    lines += [
        "", "## Gate R76A", "",
        "| check | result |",
        "|---|---|",
    ]
    for k,v in exact["checks"].items(): lines.append(f"| `{k}` | `{v}` |")
    lines += [
        "", "Gate R76A is **FAIL**: legal p16 does not meet +0.02 R@1/+0.01 mAP, unsafe is 94/984, hard-gap is worse in folds1–3, and p1/p2/p4/p8 are unsafe. Therefore Phase76B counterfactual memory, Phase76C frozen controller compatibility, Commit-CT, and sealed evaluation are not run.", "",
        "## Root-cause classification", "",
        "Primary: `LEGAL_OVERFIT`/`FOLD_IMBALANCE` — the only positive fold has 31 fit banks and a much larger validation bank, while folds1–3 expose cross-fold/domain shift; the local scorer learns fold-specific relations. Secondary: `LOSS_CONFLICT`/`CONFIDENCE_FAILURE` — the task residual changes candidate ordering enough to create unsafe raw-correct→learned-wrong flips (94 total), despite raw anchor remaining structurally intact. There is no evidence here for a geometry or frame-quality defect; the memory-mimic gains are in-sample and cannot be promoted to a causal state result.", "",
        "## Repairs and resources", "",
        f"- Repair ledger: `{len(repair['events'])}` events. Preparation r1 failed from a prematurely closed `mkstemp` fd; r2 completed all caches. Exact replay needed two minimal metric-field aggregation fixes; no data/protocol/checkpoint change.",
        f"- Formal preflight/postflight: `{pre['created_utc']}` / `{post['created_utc']}`. GPU map 4/5/6/7, one worker/fold; no OOM or external process termination. Process counts `{pre['process_count']}`→`{post['process_count']}`; `/data2` stores 97MB bank/cache metadata and ~5MB/checkpoint symlink targets.",
        "- All formal `.launched` and `.done` markers exist; no Phase76A process remains. Historical Phase75D/E artifacts were read-only.", "",
        "## Literature audit", "",
        "Official refs were revalidated by `git ls-remote` and pinned README/LICENSE fetches: RethinkingOCL commit `5d345268797425558b449337519af3ab24aeb6f1` (MIT), SlotContrast `55ec66dc02eeade630805789ef4a6c5df06f21ff` (MIT), TRACT `19f01d72f9f6c212c28fd9cb0171a5432cd41a6a` (repository Apache-2.0 metadata), and COVTrack `9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b` (repository Apache-2.0 metadata). They are references only; no external model weights/code entered inference.",
        "- Full machine audit: `outputs/iclr27_phase76a/audit/literature_audit.json`.", "",
        "## Reproduction", "",
        "```bash\nPYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase76a/smoke_contract.py\nPYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase76a/run_contract_audit.py\nPYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase76a/prepare_banks_and_cache.py\nCUDA_VISIBLE_DEVICES=4 PYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase76a/train_relation_fold.py --fold 0 --smoke --tag phase76a_smoke_r1 --device cuda:0 --expected-physical-gpu 4\nbash scripts/iclr27_phase76a/run_four_fold_supervisor.sh phase76a_formal1\nPYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase76a/evaluate_exact.py\n```", "",
        "## Final route status", "",
        "Phase76A is a reliable negative retrieval result for this local relation architecture under the frozen protocol. It does not claim TrackOCD task infeasibility and does not authorize a second encoder, threshold/memory tuning, controller or modern backbone. Phase76B/C reports are explicitly blocked by Gate R failure; the next legal action requires a separately approved evidence-backed route, not repetition of this reranker.", "",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f".{DOC.name}.",dir=str(DOC.parent))
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as h: h.write("\n".join(lines)+"\n"); h.flush(); os.fsync(h.fileno())
        os.replace(tmp,DOC)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    atomic(OUT/"audit/phase76a_decision.json", {**decision,"status":"GATE_R_FAIL","decision":"PHASE76A_GATE_R_FAIL_STOP_BEFORE_STATE_MEMORY","root_cause":["LEGAL_OVERFIT","FOLD_IMBALANCE","LOSS_CONFLICT","CONFIDENCE_FAILURE"],"report":str(DOC),"phase76b_run":False,"phase76c_run":False,"created_utc":dt.datetime.now(dt.timezone.utc).isoformat()})
    print(json.dumps({"phase":"Phase76A","decision":"PHASE76A_GATE_R_FAIL_STOP_BEFORE_STATE_MEMORY","report":str(DOC)},sort_keys=True))


if __name__ == "__main__": main()

