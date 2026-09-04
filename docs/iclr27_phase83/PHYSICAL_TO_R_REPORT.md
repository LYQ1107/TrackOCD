# Phase83 Physical→R report

**Generated (UTC):** 2026-09-04T08:31:09.816073+00:00  
**Source commit:** `915121c77a8f1ccaf4851ae1df27dbd442829d50`  
**Route:** parameter-free causal temporal-appearance mean from the Phase82R native physical lineage, compared with the frozen Phase75D raw-DINO scorer. This is a TRAIN-disjoint retrieval diagnostic, not an OCD Commit-CT result.

## Contract and inputs

The candidate universe is the exact Phase30 validation universe (all validation tracks except the query itself and same-video tracks). Q0 row keys, candidate order, prefixes `(1, 2, 4, 8, 16)`, 984-query fold denominator, and scorer were not changed. Native rows are matched only on the same `(video_id,image_id)` and bbox IoU ≥ 0.5; temporal means use matched current/past rows only. Unmapped tracks are reported explicitly; the exact-universe view falls back to raw only for those tracks and is therefore a diagnostic, not a claim of a new full physical model.

Native lineage: `/data2/usr_for_deadline/trackocd_phase75b/event_full_sequence_repair2/native_lineage.jsonl` (SHA256 `d33e60f4603aaa8aa744d8d73553b42153be9f9b88a3a19aa6eb26884d31a2e1`); native features `/data2/usr_for_deadline/trackocd_phase82r/project_outputs/features/native_dinov2_corrected_r1.npz` (SHA256 `fecdfc3bf341fc28f81fed2fa19dba57063c49793a083f3d1c18835a9d722245`); corrected public CSV SHA256 `f082024da349ce700cd97574d88cc1babea7549d8453c47971daf55ed2576b09`. Public/DEV+/Q1/sealed labels were not accessed for model selection; event labels below are post-hoc mapping diagnostics only.

## Mapping coverage

5487/43423 public rows had a native bbox match at IoU≥0.5; 1046/6213 tracks (16.84%) were usable. The native stream covers only 91 event videos. In the 76-event post-hoc diagnostic, source mapped=76/76, target mapped=74/76, both=74/76; mean raw cosine=0.2718, temporal cosine=0.1976. This temporal event cosine is lower, not higher.

## Exact-universe and mapped-subset results

| view | prefix | queries | raw R@1 | temporal R@1 | ΔR@1 | raw mAP | temporal mAP | temporal gap | raw gap | unsafe |
|---|---|---|---|---|---|---|---|---|---|---|
| exact_mixed | 1 | 984 | 0.8861 | 0.8873 | 0.0012 | 0.8420 | 0.8358 | 0.1637 | 0.1652 | 3 |
| exact_mixed | 2 | 984 | 0.9203 | 0.9028 | -0.0176 | 0.8598 | 0.8509 | 0.1812 | 0.1816 | 7 |
| exact_mixed | 4 | 984 | 0.9114 | 0.8800 | -0.0314 | 0.8565 | 0.8478 | 0.1903 | 0.1943 | 9 |
| exact_mixed | 8 | 984 | 0.9064 | 0.8895 | -0.0169 | 0.8501 | 0.8464 | 0.1947 | 0.1908 | 6 |
| exact_mixed | 16 | 984 | 0.8932 | 0.8827 | -0.0105 | 0.8484 | 0.8473 | 0.1980 | 0.1896 | 5 |
| mapped_subset | 1 | 181 | 0.9233 | 0.8885 | -0.0348 | 0.8322 | 0.8159 | 0.1322 | 0.1427 | 10 |
| mapped_subset | 2 | 181 | 0.8694 | 0.8452 | -0.0243 | 0.8597 | 0.8283 | 0.1666 | 0.1777 | 7 |
| mapped_subset | 4 | 181 | 0.9424 | 0.9129 | -0.0294 | 0.8757 | 0.8455 | 0.1822 | 0.1977 | 5 |
| mapped_subset | 8 | 181 | 0.9476 | 0.8904 | -0.0572 | 0.8743 | 0.8460 | 0.1840 | 0.1842 | 4 |
| mapped_subset | 16 | 181 | 0.8990 | 0.8904 | -0.0086 | 0.8637 | 0.8484 | 0.1818 | 0.1803 | 4 |


Fold p16 exact-mixed comparison:

| fold | queries | raw R@1 | temporal R@1 | ΔmAP | Δhard-gap | unsafe | evaluable |
|---|---|---|---|---|---|---|---|
| 0 | 837 | 0.9737 | 0.9797 | 0.0019 | 0.0230 | 2 | 837 |
| 1 | 82 | 0.9756 | 0.9634 | -0.0083 | 0.0012 | 1 | 82 |
| 2 | 37 | 0.8378 | 0.8378 | -0.0062 | -0.0025 | 0 | 39 |
| 3 | 28 | 0.7857 | 0.7500 | 0.0081 | 0.0121 | 2 | 30 |


The full exact-mixed p16 aggregate is R@1 `0.8827` vs raw `0.8932`, mAP `0.8473` vs `0.8484`, hard-gap `0.1980` vs `0.1896`, with `5` unsafe flips. Only 1/4 folds were non-decreasing in both R@1 and mAP. The mapped-only p16 view has a changed denominator (181 queries) and also decreases R@1/mAP.

The first physical-R invocation was explicitly SIGTERM-ed at task-owned PID 17813 (and its wait shell 17963) after profiling showed repeated per-pair track averaging; it had produced no artifact. Caching raw vectors was the smallest repair, followed by the fold0 smoke/targeted run and the formal run. No external process was touched and no OOM occurred.

## R83 decision

`R83_DIAGNOSTIC_NO_SAFE_IMPROVEMENT`. The temporal physical lineage does not improve the frozen raw correspondence signal under the exact full validation contract and has non-zero unsafe flips. No representation training, controller, StateMemory, threshold sweep, DEV+/Q1/public-new or sealed evaluation was run. The registered next work remains the independent O-support route; any later C route is **NOT_RUN**.

Reproduce: `python scripts/iclr27_phase83/run_physical_r.py --run-id phase83-physical-r-temporal-20260904-full`.
