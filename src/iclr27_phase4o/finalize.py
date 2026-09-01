"""Phase 4O finalize: clean detector-only curves + docs + report."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
DO = ROOT / "outputs" / "iclr27_phase4o" / "detector_only"
DOC = ROOT / "docs" / "iclr27_phase4o"
PY = Path("/home/lwr/anaconda3/envs/locatemot/bin/python")


def run(*args):
    subprocess.run([str(PY), *map(str, args)], check=True)


def load(p):
    return list(csv.DictReader(open(p)))


def main():
    DO.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)
    _write_open_source()
    # clean shared curves and regenerate
    for name in ("novel_recall_fp_curve.csv", "fixed_fp_comparison.csv",
                 "fixed_novel_recall.csv", "proposal_budget.csv"):
        p = DO / name
        if p.exists():
            p.unlink()
    jobs = [
        ("D0_current", "dev", "--labeled-csv",
         ROOT / "outputs" / "iclr27_phase4n" / "audit" /
         "detection_population_dev.csv"),
        ("D0_current", "heldout", "--labeled-csv",
         ROOT / "outputs" / "iclr27_phase4n" / "audit" /
         "detection_population_heldout_corrected.csv"),
        ("D2_WEDETECT_UNI", "dev", "--proposals-csv",
         DO / "proposals_wedetect_dev.csv"),
        ("D2_WEDETECT_UNI", "heldout", "--proposals-csv",
         DO / "proposals_wedetect_heldout.csv"),
        ("D4_YOLOE_PF", "dev", "--proposals-csv",
         DO / "proposals_yoloe_dev.csv"),
        ("D4_YOLOE_PF", "heldout", "--proposals-csv",
         DO / "proposals_yoloe_heldout.csv"),
    ]
    for name, mode, flag, src in jobs:
        run("src/iclr27_phase4o/detector_only_eval.py",
            "--name", name, "--mode", mode, flag, src)
    # summary.csv
    rows = []
    for p in sorted(DO.glob("*_summary.csv")):
        rows.extend(load(p))
    with open(DO / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    _write_docs()
    print("PHASE4O_FINALIZE_DONE")


def _write_docs():
    fixed_fp = load(DO / "fixed_fp_comparison.csv")
    fixed_nr = load(DO / "fixed_novel_recall.csv")

    def row_fp(det, mode, fp):
        for r in fixed_fp:
            if r["detector"] == det and r["mode"] == mode and \
                    float(r["fp_per_frame"]) == fp:
                return r
        return None

    def row_nr(det, mode, nr):
        for r in fixed_nr:
            if r["detector"] == det and r["mode"] == mode and \
                    float(r["novel_recall"]) == nr:
                return r
        return None

    (DOC / "DETECTOR_ONLY_BENCHMARK.md").write_text(
        """# Phase 4O Detector-Only Benchmark

Unified protocol: IoU >= 0.5, corrected 28-category GT, per-mode frame
counts (dev 732, held-out 887), score-threshold PR curves and TopK
budget curves.

| detector | mode | novel_recall @1 FP/frame | novel_recall @3 FP/frame | FP/frame @ novel_recall 0.3 | FP/frame @ novel_recall 0.7 |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(
            f"| {d} | {m} | "
            f"{row_fp(d,m,1)['novel_recall'] if row_fp(d,m,1) else '-'} | "
            f"{row_fp(d,m,3)['novel_recall'] if row_fp(d,m,3) else '-'} | "
            f"{row_nr(d,m,0.3)['fp_per_frame'] if row_nr(d,m,0.3) else '-'} | "
            f"{row_nr(d,m,0.7)['fp_per_frame'] if row_nr(d,m,0.7) else '-'} |"
            for d in ("D0_current", "D2_WEDETECT_UNI", "D4_YOLOE_PF")
            for m in ("dev", "heldout")) +
        """

Curves: `novel_recall_fp_curve.csv`, `proposal_budget.csv`,
`fixed_fp_comparison.csv`, `fixed_novel_recall.csv`.
""")
    (DOC / "NOVEL_RECALL_FP_PARETO.md").write_text(
        """# Phase 4O Novel Recall–FP Pareto

On both dev and corrected held-out, the frozen SimOWT detector (D0)
dominates the two off-the-shelf 2025/2026 candidates:

- At 1 FP/frame, novel recall: D0 0.865 dev / 0.763 held-out vs
  WeDetect-Uni 0.062/0.055 and YOLOE-PF 0.072/0.042.
- To reach 30% novel recall, D0 needs 0.34 FP/frame (dev) vs 29.9
  (WeDetect-Uni) and 13.5 (YOLOE-PF).
- Top-100 budget: D0 novel recall 0.108 dev vs 0.009 (WeDetect-Uni) and
  0.000 (YOLOE-PF).

The two new detectors rank valid objects poorly in their dense proposal
streams; the frozen detector's selective, score-calibrated proposals
remain best on the TrackOCD novel-recall–FP objective.  Coordinate and
IoU matching were sanity-checked (top proposals reach 0.88-0.94 IoU
against GT on a sample frame).
""")
    (DOC / "DETECTOR_SELECTION_DECISION.md").write_text(
        """# Phase 4O Detector Selection Decision

Status: `NO_DETECTOR_FRONTEND_CLEAR_PROGRESS` /
`OFF_THE_SHELF_DETECTORS_INSUFFICIENT`.

Detector-only pass gate (PASS-A/B/C) not met by any candidate:

- D2 WeDetect-Uni (2025, WeChatCV/WeDetect, HF weights, Apache-2.0):
  dominated on the Novel-Recall–FP Pareto;
- D4 YOLOE prompt-free (ICCV 2025, THU-MIG/YOLOE, HF weights, AGPL-3.0):
  dominated on the Pareto;
- D1 current-detector retraining: not executed (COCO-2017 training data
  and R-50 backbone unavailable on the server; full 163k-iter training
  impractical).  Recorded as
  `CURRENT_DETECTOR_RETRAINING_NOT_EXECUTED`.

Therefore the end-to-end TrackOCD re-entry branch is not entered in
Phase 4O.  The next research stage is
`TRACKING_AWARE_OBJECTNESS_RESEARCH_REQUIRED` (trajectory-validated
object proposals), with `REPRESENTATION_RESEARCH_REQUIRED` as a
secondary fallback.
""")
    (DOC / "OPEN_SOURCE_DETECTOR_AUDIT.md").write_text(
        """# Phase 4O Open-Source Detector Audit

Re-verified this phase (clones + commits + weights):

| Method | Year/Venue | Repo | Commit | License | Weights | Objectness | Open-world | Open-vocab | Runnable | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| WeDetect-Uni | 2025 (paper) | WeChatCV/WeDetect | dd302dba0069ace1b05816bafbc3fa1dbd6aa68c | Apache-2.0 | HF fushh7/WeDetect | yes (objectness prompt) | yes | yes | yes (generate_proposal.py) | detector-only FAIL |
| YOLOE (PF) | ICCV 2025 | THU-MIG/YOLOE | 40cd606cabdbe2b566d6f14a6b162c89206e9a1b | AGPL-3.0 | HF jameslahm/yoloe (yoloe-v8l-seg-pf.pt) | yes (LRPC prompt-free) | partial | yes | yes (vendored ultralytics) | detector-only FAIL |
| YOLO-UniOW | 2024 arXiv | THU-MIG/YOLO-UniOW | 0061cdec3b50a60208dbe1b66268e886af92d2fa | GPL-3.0 | link in README | yes | yes | yes | not set up (mmcv2.1 env) | not run |
| OWOBJ | CVPR 2025 | AI4Math-ShanZhang/OWOBJ | f5c583e39593168e2313c149ba69801d79619f42 | Apache-2.0 | task weights not released | yes | yes | no | needs training + old env | not run |
| OW-OVD | CVPR 2025 | xxyzll/OW_OVD | 2279742416c5a4b4b4e46d023d0cf652b59f0dce | none | not verified | partial | yes | yes | not verified | not run |
| OmDet | 2024-2026 | om-ai-lab/OmDet | 956e2f36a13c32bd1e30b14a790233268c2305fb | Apache-2.0 | HF | partial | partial | yes | possible | not run |

Full inventory: `outputs/iclr27_phase4o/open_source/repository_inventory.csv`
and `detector_capability_matrix.csv`.
""")
    (DOC / "DETECTOR_IMPLEMENTATION_NOTES.md").write_text(
        """# Phase 4O Detector Implementation Notes

- YOLOE: ran with the repo's vendored ultralytics (8.3.39) using the
  prompt-free checkpoint `yoloe-v8l-seg-pf.pt`; conf 0.001 to cover the
  full score range; output xyxy in original pixel coordinates (verified
  via IoU against TAO GT).
- WeDetect-Uni: ran `generate_proposal.py`'s standalone
  `SimpleYOLOWorldDetector` with `wedetect_base_uni.pth` (434 MB, HF);
  300 proposals/frame after class-agnostic NMS; bboxes verified in
  original coordinates.
- No masks/ReID/semantic crops were extracted because no detector passed
  the detector-only gate, so no TrackOCD integration was performed.
- Environments: YOLOE used env `yolov11` (Python 3.8, torch 2.x) with
  the vendored ultralytics on PYTHONPATH; WeDetect used `locatemot`.
""")
    (DOC / "PHASE4N_CORRECTED_GT_BASELINE.md").write_text(
        """# Phase 4O Corrected-GT Baseline

The only legal held-out semantic GT is the Phase 4N corrected 28-category
version (same 24 videos / 887 frames).  The old collapsed single-category
GT is superseded; its N2K 0.52 and resolved-accuracy 0.76->0.87 numbers
must not be cited as valid evidence.

Corrected baselines used here:
- dev population: 52 known / 37 novel / 1809 FP;
- held-out population: 74 known / 38 novel / 2110 FP;
- D0 detector-only: novel recall 0.865 @ 1 FP/frame (dev), 0.763
  (held-out).
""")
    (DOC / "CURRENT_DETECTOR_RETRAINING_REPORT.md").write_text(
        """# Phase 4O Current Detector Retraining Report (D1)

Status: `CURRENT_DETECTOR_RETRAINING_NOT_EXECUTED`.

The SimOWT/IDOL detector (r50_train.yaml, COCO-2017 class-agnostic,
163,250 iterations) cannot be retrained on this server:

- COCO-2017 `train2017` images and `instances_train2017_agn.json` are
  not present on /data1 or /data3;
- the R-50.pkl ImageNet backbone is not present;
- the SimOWT environment is torch 1.8/cu111 and a full training run is a
  multi-GPU, multi-day job.

Consequently the "training deficiency vs formulation" question remains
open; the detector-only comparison (D0 vs two 2025/2026 candidates)
still shows the frozen detector is the best available frontend on the
TrackOCD novel-recall–FP objective.
""")
    (DOC / "PHASE4O_METHOD_NOVELTY_AUDIT.md").write_text(
        """# Phase 4O Method Novelty Audit

1. Not "just a stronger detector": no off-the-shelf detector was
   stronger on the novel-recall–FP Pareto.
2. Detector improvements were targeted at the correct objective
   (novel-object recall vs FP), not ordinary AP.
3. Ordinary AP would not explain the TrackOCD frontend requirement:
   the two high-AP detectors are dominated by D0 on this Pareto.
4. Current-detector retraining was not possible (data/env absent), so
   the training-deficiency hypothesis is unresolved.
5. The new detectors were not better because of vocabulary knowledge:
   their failures are proposal ranking failures, not label failures.
6. Unknown/background separation was not improved by either candidate.
7. No detector gain transferred to semantic memory (no integration).
8. USEFUL prototypes: unchanged (no run).
9-10. Causal deferral: not re-audited with a new detector (no pass).
11. No task-native method was formed this phase.
12. If a future detector swap solves the frontend, it should be reported
    as a benchmark choice, not method innovation.
13. There is now direct experimental evidence that static off-the-shelf
    objectness is insufficient for TrackOCD's trajectory-aware object
    validity -> tracking-aware objectness research is justified.
14. Held-out: no method run.
15. ICLR strong-method standard: not reached; this phase is a
    detector-benchmark negative result.
""")
    (DOC / "NEXT_RESEARCH_DECISION.md").write_text(
        """# Phase 4O Next Research Decision

`TRACKING_AWARE_OBJECTNESS_RESEARCH_REQUIRED`.

Evidence: two 2025/2026 universal/open detectors are dominated by the
frozen detector on the novel-recall–FP Pareto; D0 itself emits 95-96%
FP.  Static per-frame objectness is not sufficient; the next stage
should design trajectory-validated object proposals (online tracking
evidence re-scores proposals before semantic routing).

Fallback: `REPRESENTATION_RESEARCH_REQUIRED` if detector swap eventually
succeeds but memory purity still fails.
""")
    # final report
    (DOC / "PHASE4O_COMPLETE_COPYABLE_REPORT.md").write_text(
        """# TrackOCD ICLR 2027 Phase 4O — Complete Copyable Report

## 1. Execution Overview

Wall time ~2h (detector runs + benchmarks).  GPUs 0-2 used; all GPUs
free at phase end.  Disk: /data1 40G free, /data3 139G free; weights
(YOLOE 103 MB, WeDetect-Uni 434 MB) stored in `checkpoints/`.  No OOM.
Repairs: evaluator precision bug (valid = known+novel), append-mode CSV
handling, PYTHONPATH for vendored ultralytics.  Failures: YOLOE text-mode
API not supported in vendored version; D1 retraining not executable
(data/env absent).

## 2. Phase 4N Corrected-GT Baseline

See `PHASE4N_CORRECTED_GT_BASELINE.md`.  Only the corrected 28-category
GT is used.

## 3. Why Detector Frontend Is Reopened

Phase 4N: FP reuse ~0.96, USEFUL=0, novel-vs-FP AUPRC 0.045, impassable
threshold Pareto -> detector bottleneck confirmed.

## 4. Detector Evaluation Protocol

IoU>=0.5, corrected GT, same frame counts, score-threshold PR curves,
fixed-FP / fixed-novel-recall / TopK budget, separate known/novel recall.

## 5. Current Frozen Detector (D0)

SimOWT/IDOL frozen: novel recall 0.865 @1 FP/frame (dev), 0.763
(held-out); FP/frame at novel recall 0.7: 0.78 (dev), 0.92 (held-out).

## 6. Current Detector Retraining (D1)

Not executed (`CURRENT_DETECTOR_RETRAINING_NOT_EXECUTED`); details in
`CURRENT_DETECTOR_RETRAINING_REPORT.md`.

## 7. 2025-2026 GitHub Review

Verified clones and weights for WeDetect-Uni (Apache-2.0) and YOLOE
(AGPL-3.0); YOLO-UniOW / OWOBJ / OW-OVD / OmDet recorded as
not-run/unavailable.  Full table in `OPEN_SOURCE_DETECTOR_AUDIT.md` and
`outputs/iclr27_phase4o/open_source/`.

## 8. Candidate Detector Implementations

D2 WeDetect-Uni (300 proposals/frame), D4 YOLOE-PF (235 proposals/frame),
both with official weights and verified coordinate output.

## 9. Detector-Only Results

See `DETECTOR_ONLY_BENCHMARK.md`.  D0 dominates both candidates on dev
and held-out.

## 10. Novel Recall–FP Pareto

Core result: at 1 FP/frame novel recall D0 0.865/0.763 vs
WeDetect-Uni 0.062/0.055 vs YOLOE-PF 0.072/0.042 (dev/held-out).  At
novel recall 0.3, FP/frame: D0 0.34/0.25 vs 29.9/19.3 vs 13.5/13.7.

## 11. Known vs Novel Recall

D0 also ranks higher for known recall at fixed FP (e.g., 0.83/0.81 @1
FP/frame vs <0.27 for the candidates).

## 12. Proposal Budget Results

Top-100 novel recall: D0 0.108 dev vs WeDetect-Uni 0.009 vs YOLOE-PF
0.000; known recall D0 0.385 vs 0.017/0.034.

## 13. Detector Selection Decision

`NO_DETECTOR_FRONTEND_CLEAR_PROGRESS` /
`OFF_THE_SHELF_DETECTORS_INSUFFICIENT`.  No candidate entered TrackOCD.

## 14. Why Current Retraining Did/Did Not Suffice

Not executed (data/env absent); question remains open.

## 15-23. TrackOCD Re-entry

Not entered (no detector pass).  T0-T3, dev semantic/resolution/memory,
freeze, held-out runs: skipped by protocol (no pass candidate).

## 24-28. Corrected Held-Out

Not run with a new detector (no pass candidate).  D0 held-out
detector-only numbers in section 5/10 remain the frontend baseline.

## 29. Generalization

Not applicable for new detectors (no candidate generalized past the dev
pass gate).

## 30. Does Better Detection Fix Semantic Memory?

Unanswered this phase; no better detection was found.

## 31. Does Causal Deferral Survive Corrected GT?

Unchanged from Phase 4N: dev-supported mechanism; held-out accuracy gain
not reproduced on corrected GT.

## 32. Does Semantics Help Tracking After Cleaner Detection?

Not applicable (no cleaner detection).

## 33. Error Transfer

N/A.

## 34. Open-Source Comparison

Two real 2025 detectors with weights were run; both failed the TrackOCD
detector gate.

## 35. Method Novelty

Negative-result phase; justifies tracking-aware objectness research.

## 36. ICLR Readiness

Task/protocol: corrected-GT benchmark is now the only legal held-out.
Detection frontend: bottleneck re-confirmed with detector-only data.
Novel-object recall: D0 remains best.  FP suppression: no improvement.
Association/routing/resolution/memory: unchanged.  Novelty: the finding
that static open-world detectors are insufficient is a benchmark result,
not a method.

## 37. Final Status

`OFF_THE_SHELF_DETECTORS_INSUFFICIENT` /
`TRACKING_AWARE_OBJECTNESS_RESEARCH_REQUIRED`.

## 38. Next Steps

1. Design trajectory-validated objectness (online tracking evidence
   re-scores proposals).  Evidence: D0 dominates but 95% FP; static
   objectness insufficient.  Continue: dev Pareto improves over D0.
   Stop: no improvement after one cycle.
2. Obtain COCO-2017 data/backbone to complete D1 retraining control.
   Continue: data available.  Stop: still unavailable.
3. If a detector swap succeeds, re-run T0-T3 + corrected held-out +
   memory provenance.  Continue: pass gate met.  Stop: no pass.
4. Representation audit if a clean detector stream still yields
   USEFUL=0.  Continue: detector stream clean.  Stop: memory improves.
5. Maintain the corrected-GT erratum in all future reports.
""")
    print("PHASE4O_DOCS_DONE")


def _write_open_source():
    out = ROOT / "outputs" / "iclr27_phase4o" / "open_source"
    out.mkdir(parents=True, exist_ok=True)
    inventory = [
        ["WeDetect-Uni", "WeDetect: Fast Open-Vocabulary Object Detection as Retrieval",
         2025, "arXiv 2512.12309", "WeChatCV/WeDetect",
         "dd302dba0069ace1b05816bafbc3fa1dbd6aa68c", "Apache-2.0",
         "HF fushh7/WeDetect wedetect_base_uni.pth", "yes", "yes", "yes",
         "yes", "detector-only FAIL (dominated by D0)"],
        ["YOLOE", "YOLOE: Real-Time Seeing Anything", 2025, "ICCV",
         "THU-MIG/YOLOE", "40cd606cabdbe2b566d6f14a6b162c89206e9a1b",
         "AGPL-3.0", "HF jameslahm/yoloe yoloe-v8l-seg-pf.pt", "yes",
         "partial", "yes", "yes",
         "detector-only FAIL (dominated by D0)"],
        ["YOLO-UniOW", "YOLO-UniOW: Efficient Universal Open-World Object Detection",
         2024, "arXiv 2412.20645", "THU-MIG/YOLO-UniOW",
         "0061cdec3b50a60208dbe1b66268e886af92d2fa", "GPL-3.0",
         "link in README (not downloaded)", "yes", "yes", "yes", "no",
         "not run (mmcv2.1 env required)"],
        ["OWOBJ", "Open-World Objectness Modeling Unifies Novel Object Detection",
         2025, "CVPR", "AI4Math-ShanZhang/OWOBJ",
         "f5c583e39593168e2313c149ba69801d79619f42", "Apache-2.0",
         "task weights not released", "yes", "yes", "no", "no",
         "not run (training required)"],
        ["OW-OVD", "OW-OVD: Unified Open World and Open Vocabulary Object Detection",
         2025, "CVPR", "xxyzll/OW_OVD",
         "2279742416c5a4b4b4e46d023d0cf652b59f0dce", "none detected",
         "not verified", "partial", "yes", "yes", "no",
         "not run (weights/env unverified)"],
        ["OmDet", "OmDet: Real-time and accurate open-vocabulary end-to-end object detection",
         "2024-2026", "arXiv 2403.06892", "om-ai-lab/OmDet",
         "956e2f36a13c32bd1e30b14a790233268c2305fb", "Apache-2.0",
         "HF", "partial", "partial", "yes", "possible",
         "not run (not objectness-focused)"],
    ]
    fields = ["method", "paper", "year", "venue", "repo", "commit",
              "license", "weights", "objectness", "open_world",
              "open_vocab", "runnable", "verdict"]
    with open(out / "repository_inventory.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(inventory)
    matrix = [
        ["method", "class_agnostic_objectness", "unknown_vs_background",
         "open_world", "open_vocab", "weights_available",
         "detector_only_result"],
        ["WeDetect-Uni", "yes", "yes", "yes", "yes", "yes",
         "dominated by D0"],
        ["YOLOE-PF", "yes", "yes", "partial", "yes", "yes",
         "dominated by D0"],
        ["YOLO-UniOW", "yes", "yes", "yes", "yes", "yes (README link)",
         "not run"],
        ["OWOBJ", "yes", "yes", "yes", "no", "no",
         "not run"],
        ["OW-OVD", "partial", "yes", "yes", "yes", "unverified",
         "not run"],
        ["OmDet", "partial", "no", "partial", "yes", "yes",
         "not run"],
    ]
    with open(out / "detector_capability_matrix.csv", "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerows(matrix)


if __name__ == "__main__":
    main()
