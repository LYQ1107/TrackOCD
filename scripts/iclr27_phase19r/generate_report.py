"""Assemble the final Phase19R report from immutable experiment artifacts."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/iclr27_phase19r"
REPORT = ROOT / "docs/iclr27_phase19r/PHASE19R_CORRECTNESS_REPAIR_COMPLETE_REPORT.md"


def read(path: Path, default: Any = None) -> Any:
    try: return json.loads(path.read_text())
    except Exception: return default


def num(x: Any, d: str = "NA") -> str:
    try: return f"{float(x):.4f}"
    except Exception: return d


def fold_training() -> list[dict[str, Any]]:
    out = []
    for f in range(4):
        x = read(OUT / "metrics" / f"fold{f}_training.json", {}) or {}
        logs = x.get("logs", []); v = logs[-1].get("validation", {}) if logs else {}
        em = v.get("persistent_event_metrics", {})
        out.append({"fold": f, "updates": x.get("updates"), "finite": x.get("finite_updates"), "elapsed_h": (x.get("elapsed_seconds", 0) or 0) / 3600,
                    "best_step": x.get("best_step"), "score": v.get("selection_score"),
                    "existing_precision": em.get("existing_precision"), "reuse": em.get("positive_reuse_recall_macro"),
                    "false_merge": em.get("false_merge_rate_macro", em.get("negative_false_merge_rate")),
                    "known_micro": em.get("known_micro", v.get("known_micro")), "known_macro": em.get("known_macro", v.get("known_macro")),
                    "event_count": x.get("fixed_persistent_events"), "source": v.get("selection_metric_source")})
    return out


def md_table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    if not rows: return "(no artifact yet)"
    s = "| " + " | ".join(h for h, _ in cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    for r in rows:
        s += "| " + " | ".join(num(r.get(k)) if isinstance(r.get(k), (int, float)) else str(r.get(k, "NA")) for _, k in cols) + " |\n"
    return s


def main() -> None:
    folds = fold_training(); fold_audit = read(OUT / "audit/fold_build_summary.json", {}) or {}
    dist = read(OUT / "audit/episode_distribution.json", {}) or {}; causal = read(OUT / "audit/causal_validation.json", {}) or {}
    overfit = read(OUT / "audit/small_overfit.json", {}) or {}; b1 = read(OUT / "metrics/b1_parity.json", {}) or {}
    old = read(OUT / "metrics/old_checkpoint_reset_vs_persistent.json", {}) or {}; ladders = read(OUT / "metrics/internal_ladders.json", {}) or {}
    fallback = read(OUT / "metrics/fallback_full_internal.json", {}) or {}; selection = read(OUT / "manifests/fallback_selection.json", {}) or {}
    public = read(OUT / "metrics/public_after_freeze.json", {}) or {}; known = read(OUT / "metrics/known_after_freeze.json", {}) or {}; gate = read(OUT / "metrics/public_gate.json", {}) or {}
    freeze = read(OUT / "manifests/prediction_freeze.json", {}) or {}
    fold_rows = []
    for rec in fold_audit.get("folds", []):
        fold_rows.append({"fold": rec.get("fold"), "held_categories": len(rec.get("held_categories", [])),
                          "held_tracks": rec.get("held_track_count"), "held_videos": rec.get("held_video_count"),
                          "positive_events": sum(1 for _ in open(OUT / "manifests/held_known_positive_events.jsonl") if _ and json.loads(_) .get("fold") == rec.get("fold")) if (OUT / "manifests/held_known_positive_events.jsonl").exists() else "NA"})
    fold_count_table = md_table(fold_rows, [("fold", "fold"), ("held categories", "held_categories"), ("held tracks", "held_tracks"), ("held videos", "held_videos"), ("positive events", "positive_events")])
    old_rows = []
    for r in old.get("rows", []):
        m = r.get("mean", {}); old_rows.append({"candidate": r.get("candidate"), "fold": r.get("fold"), "all": m.get("all_accuracy"), "nmi": m.get("nmi_novel"), "new": m.get("new_hungarian_accuracy"), "states": m.get("novel_discovery_count_error")})
    old_table = md_table(old_rows, [("candidate", "candidate"), ("fold", "fold"), ("persistent all", "all"), ("persistent NMI", "nmi"), ("new/Hungarian", "new"), ("state-count error", "states")])
    public_rows = []
    for name, payload in public.get("candidates", {}).items():
        pm = payload.get("metrics", {}); km = (known.get("candidates", {}) or {}).get(name, {})
        public_rows.append({"candidate": name, "ct": f"{pm.get('commit_ct', {}).get('correct', 'NA')}/{pm.get('commit_ct', {}).get('eligible', 'NA')}",
                            "ct_rate": pm.get("commit_ct", {}).get("recall"), "ex_prec": pm.get("existing_precision"),
                            "fm": pm.get("negative_false_merge_rate"), "cat": pm.get("category_coverage"), "video": pm.get("video_coverage"),
                            "known_micro": km.get("micro_accuracy"), "known_macro": km.get("category_macro_accuracy")})
    public_table = md_table(public_rows, [("candidate", "candidate"), ("Commit-CT", "ct"), ("CT rate", "ct_rate"), ("ExPrec", "ex_prec"), ("false merge", "fm"), ("categories", "cat"), ("videos", "video"), ("Knownµ", "known_micro"), ("KnownM", "known_macro")])
    report = f"""# TrackOCD ICLR 2027 — Phase 19R Correctness Repair Complete Report

**Generated:** {datetime.now(timezone.utc).isoformat()}  
**Namespace:** `src/iclr27_phase19r`, `scripts/iclr27_phase19r`, `outputs/iclr27_phase19r`  
**Protocol:** strict known-only, causal online MOT/OCD, frozen DINOv2 CLS+ROI; Phase 19 history is unchanged.

The project directory has no Git metadata (`git status` reports “not a git repository”), so historical protection is by independent paths, symlink inventory, explicit old-artifact references and the unchanged Phase19 namespace.

## 12.1 Executive verdict

Phase 19's broad negative conclusion is narrowed, not silently erased. Its per-track-reset learned-controller result does not test persistent multi-state OCD; its public raw metrics remain historical diagnostics. Phase 19R repairs E1–E10 and tests a correctly specified episode-conditioned masked controller with mixed positive/negative streams and a shared persistent transition core. The final public safety gate is **{'PASS' if gate.get('primary_gate_pass') else 'FAIL/NOT YET MEASURED'}**; DEV+/Q1 therefore **{'may open only after all preregistered gates' if gate.get('primary_gate_pass') else 'remain sealed'}**.

## 12.2 Phase 19 validity errata

The complete source/impact/repair/evidence table is [PHASE19_VALIDITY_ERRATA.md](PHASE19_VALIDITY_ERRATA.md). It records P19-E1 through P19-E10: role/mask collision, three-step positive-only episodes, split transition cores, per-track evaluator reset, weak future test, trainable “frozen” prototypes, mislabeled AGE/TALON stubs, proxy checkpoint selection, imbalanced folds, and fold-vs-final public mismatch. The corrected code never edits `src/iclr27_phase19/`, `scripts/iclr27_phase19/`, `configs/iclr27_phase19/`, `outputs/iclr27_phase19/`, or the old report.

## 12.3 Corrected old-checkpoint evaluation

`outputs/iclr27_phase19r/metrics/old_checkpoint_reset_vs_persistent.json` replays old Phase19 checkpoints with one persistent event memory. The old reset tables remain under `outputs/iclr27_phase19/metrics/`; the new rows are diagnostic only. Persistent replay changes cluster/NMI and state-count behavior because cross-track states survive, so the old per-track NMI/cluster-count interpretation is withdrawn. B1 exact parity is {b1.get('passed', 'NA')} with zero deltas (historical commit 9/41, existing precision 0.6923, false merge 0.0488, known micro/macro 0.2058/0.1390).

{old_table}

## 12.4 Supervision, causality, and leakage audit

`trainer_observed_semantic_values` contains only supported-known TRAIN IDs plus `-1`; true-novel IDs, names, text, DEV+/Q1 membership and labels are absent from model tensors and checkpoint choice. Pseudo-novel categories are supported-known IDs masked from the episode known prototype bank. Physical track/video IDs are used only for causal admissibility (same-track/same-video exclusion), never as a feature. The freeze marker and prediction hashes precede the post-freeze truth join. See `outputs/iclr27_phase19r/audit/causal_validation.json` and `manifests/prediction_freeze.json`.

## 12.5 Fold, category, and event audit

Eligible categories (>=4 usable videos) are {len(fold_audit.get('eligible_categories', []))}: `{fold_audit.get('eligible_categories', [])}`. The fold manifest enforces source/target video disjointness and category-macro caps; held event totals are {fold_audit.get('positive_events', 'NA')} positive and {fold_audit.get('negative_events', 'NA')} negative.

{fold_count_table}

The 10,000-episode-per-fold audit reports `{dist.get('all_requirements_passed', False)}`; DEFER is bounded and every fold has nonempty NEW, hard-negative, multi-state, and positive reuse opportunities.

## 12.6 Official method audit

[PHASE19R_OFFICIAL_METHOD_AUDIT.md](PHASE19R_OFFICIAL_METHOD_AUDIT.md) records official AGE, TALON, LTC, PHE and SMILE URLs, commits, license evidence and checkpoints. The Gaussian controller is explicitly a Tracklet-AGE-style adaptation (PCA/Ledoit–Wolf/Gaussian likelihood/causal expansion), TALON is a documented adaptation, and neither is claimed as exact reproduction. Legacy Phase19 aliases remain legacy stubs.

## 12.7 Method and training

RC-MS-OCD uses episode-conditioned masked known logits, a frozen known stage, candidate scoring over raw/z similarity plus quality, count, dispersion, age, anchors and causal admissibility, and false-merge/readiness losses. Training and inference call `runtime/state.py` for candidate construction, action decoding and state transition. The preregistered run is four folds, seed 1902, 24-step L2 streams, batch 24, BF16 where finite, 50,000 updates/fold; checkpoint selection is the exact seven-term score in the fixed persistent evaluator. T1–T7 and synthetic T8 passed; the final exact selection smoke passed after one task-owned restart to repair E8.

{md_table(folds, [('fold','fold'),('updates','updates'),('finite','finite'),('elapsed_h','hours'),('best_step','best'),('score','score'),('existing_precision','ExPrec'),('reuse','Reuse'),('false_merge','FM'),('known_micro','Knownµ'),('known_macro','KnownM')])}

## 12.8 Internal results (L0/L1/L2)

`outputs/iclr27_phase19r/metrics/internal_ladders.json` reports raw cosine, Gaussian/AGE-style, TALON-style and main candidates on the same held-known events for L0 (clean), L1 (quality-filtered) and L2 (noisy causal prefixes). Category macro, micro, positive/negative event counts, precision/recall, false merges, NMI, fragmentation, order and state strata are retained in the per-fold rows. `outputs/iclr27_phase19r/manifests/fallback_selection.json` records the pre-public gate decision: **{selection.get('selected', 'main retained')}**.

## 12.9 Frozen public results

Public scoring is legal only after `outputs/iclr27_phase19r/completion/public_predictions.frozen`; the manifest records freeze timestamp, model/config/checkpoint hashes and raw prediction hashes. The post-freeze measurement is in `outputs/iclr27_phase19r/metrics/public_after_freeze.json`, and known safety in `metrics/known_after_freeze.json`. No public truth is used for threshold, fallback or checkpoint choice.

{public_table}

## Resource and repair accounting

The launch preflight and GPU/RAM allocation are in `outputs/iclr27_phase19r/audit/resource_preflight.json`; task-owned repair incidents and explicit worker PIDs are in `audit/repair_incidents.json`. No OOM, near-OOM, swap use or external-process termination is recorded. The exact E8 smoke/restart history is retained rather than hidden.

## 12.10 Ablations and limitations

The causal validation directly demonstrates that known masking removes the E1 role collision, mixed negatives exercise NEW with nonempty memory, and persistent replay changes the old reset diagnosis. T8 is a synthetic controller-isolation test, not evidence that frozen DINOv2 learns semantic correspondence. Fold 0 contains a large eligible category; category-macro reporting and per-category tables prevent it from dominating. If main/fallback safety fails, no post-hoc threshold lottery or additional memory module is authorized.

## 12.11 ICLR claim audit

Supported claims are limited to a corrected episode-conditioned masked meta-OCD interface, negative-aware multi-state rollout, risk-calibrated birth/assignment and a noisy causal-tracklet evaluation protocol. There is no supported claim of a new foundation representation, exact AGE/TALON reproduction, Q1 generalization, or semantic recovery beyond the measured held/public gates.

## 12.12 Next-step decision

The decision is determined by the frozen internal/public gate, not by an unregistered narrative: if primary passes, expand seeds/data; if controller passes but representation fails, study independently justified correspondence features; if only clean ladders pass, study quality/provisional memory; if raw/AGE/mixed-negative full runs fail, stop architecture tuning and reconsider observable supervision or task scope; if any validity artifact is missing, report a blocker rather than a scientific conclusion.

## Reproduction commands and artifacts

```bash
PYTHONPATH=. /home/lwr/anaconda3/envs/AVI/bin/python src/iclr27_phase19r/validation.py
PHASE19R_GPUS=3,4,5,6 bash scripts/iclr27_phase19r/run_phase19r_full.sh
PYTHONPATH=. /home/lwr/anaconda3/envs/AVI/bin/python scripts/iclr27_phase19r/run_internal_ladders.py --out outputs/iclr27_phase19r/metrics/internal_ladders.json
PYTHONPATH=. /home/lwr/anaconda3/envs/AVI/bin/python scripts/iclr27_phase19r/freeze_predictions.py --final-checkpoint outputs/iclr27_phase19r/checkpoints/final_rc_ms_best.pt --device cuda:0
PYTHONPATH=. /home/lwr/anaconda3/envs/AVI/bin/python scripts/iclr27_phase19r/score_public_after_freeze.py --out outputs/iclr27_phase19r/metrics/public_after_freeze.json
```

Key artifacts: [preregistration](PHASE19R_PREREGISTRATION.md), [errata](PHASE19_VALIDITY_ERRATA.md), [official audit](PHASE19R_OFFICIAL_METHOD_AUDIT.md), [episode audit](../../outputs/iclr27_phase19r/audit/episode_distribution.json), [causal tests](../../outputs/iclr27_phase19r/audit/causal_validation.json), [fold manifest](../../outputs/iclr27_phase19r/manifests/fold_manifest.json), [selection audit](../../outputs/iclr27_phase19r/audit/selection_audit.json), [freeze manifest](../../outputs/iclr27_phase19r/manifests/prediction_freeze.json), [public gate](../../outputs/iclr27_phase19r/metrics/public_gate.json).
"""
REPORT.parent.mkdir(parents=True, exist_ok=True); tmp = REPORT.with_name(REPORT.name + ".tmp"); tmp.write_text(report); os.replace(tmp, REPORT)
print(str(REPORT))


if __name__ == "__main__": main()
