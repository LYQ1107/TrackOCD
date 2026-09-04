# TrackOCD Phase83 Resumed Final Report

**Window:** `2026-09-04T07:43:07Z` → `2026-09-04T17:43:07Z` (original clock; not restarted)  
**Resume:** `2026-09-04T09:03:35Z`  
**Premature-finalization correction:** prior HEAD `5db09123655368e643a29664ae77feed9ebb6ce6` was finalized before the registered window ended; this report is the corrected closeout.  
**True finalization time:** `2026-09-04T16:58:28Z`  
**Elapsed from original start:** `33321s`  
**Remaining at finalization:** `2678s`  
**Generation source HEAD:** `4f7ed5bc0ee8d6ec755d3330c00b42df11e03e25`  
**Finalization lock:** allowed=`True`, release reason=`deadline-minus-45-minute finalization window`

## Executive decision

`AUTONOMOUS_PHASE83_WINDOW_COMPLETE_WITH_NEGATIVE_EVIDENCE` is the status of this **single resumed Phase83 window**, not a claim that TrackOCD is universally infeasible. The registered R/O diagnostics were completed with negative or unsafe results. No safe R/O improvement authorized unchanged-controller compatibility; therefore C83 and sealed/public evaluation remain **NOT_RUN**. The immutable frozen comparator remains strict p16 observation `25/76` (source `49/76`, target `40/76`), while the post-hoc native pool ceiling is source `72/76`, target `64/76`, both `61/76`.

The conclusion is specific: the tested temporal-mean, prototype, listwise, native-set, and cross-video support formulations did not safely transfer to the exact causal R/O protocol in this window. They do not prove that a future, separately registered support/assignment contract or full physical-stream redesign cannot work.

## Frozen protocol and sealed boundary

- Exact Phase30/Phase75D R universe: 43,423 rows, 6,213 tracks, 984 queries; same candidate order, same-video exclusion, folds and prefixes `(1, 2, 4, 8, 16)`.
- O replay: 76 positive + 76 negative events, unchanged reliable rule and denominator. Event labels, IoU and assignment are post-hoc diagnostics or TRAIN target metadata only.
- Inference tensors contain only visual/geometry/causal history fields. No category/text, semantic or numeric physical ID, future row/track, DEV+, Q1, public-new or sealed label was accessed; no held result selected a checkpoint or threshold.
- Phase75B evaluator and physical stream were not rewritten; all Phase83 fields are versioned artifacts. No threshold, StateMemory, controller, backbone or sealed/public route was run.

## Route tree and decisions

| route | status | next / interpretation |
|---|---|---|
| R83 first physical temporal mean | FAIL | A2 full coverage |
| A2 full Q0 lineage + temporal mean | FAIL | A3 identity/prototype diagnostic |
| A3 identity diagnostic + M=3 prototypes | FAIL | B2/B3 support formulation |
| O83 binary row router | FAIL | B2 listwise + DEFER |
| B2 listwise candidate competition | FAIL | B3 contract audit/joint support |
| B3 joint support matcher | FAIL | B4 native runtime candidate set |
| B4 native candidate-set matcher | FAIL | B5 cross-video support diagnostic |
| B5 cross-video prior support diagnostic | FAIL | window closure; no C without safe R/O |
| C83 unchanged controller | NOT_RUN | requires safe R or O |
| sealed/public evaluation | NOT_RUN | requires C authorization |


### R83 first route (historical checkpoint)

The event-native temporal-appearance-mean stream was partial: 5,487/43,423 rows and 1,046/6,213 tracks were usable (16.84%), covering 74/76 event pairs. Exact mixed p16 was raw R@1 `0.893219` vs temporal `0.882735`, raw mAP `0.848374` vs `0.847251`, raw hard gap `0.189559` vs `0.198022`, with 5 unsafe flips. This was retained as a partial-coverage diagnostic, not a full-stream claim.

### A2 full-coverage Q0 physical lineage

Q0 native inference completed across 370 public TRAIN videos: 682,335 rows and 13,678 frame traces. The corrected DINOv2 cache is 682,335×768 (SHA256 `ed4405f7946f87579c086c332db743e59c79b9d046ab55ca756a1aea46723714`). Mapping used identical `(video_id,image_id)` and proposal-box IoU≥0.5; 6,213 tracks were mapped, but only 74/76 event pairs were native-mapped and 21.00% of public tracks had a complete row fraction. The full-coverage attempt therefore still required explicit mapping diagnostics; it did not justify a headline physical improvement.

| prefix | queries | raw R@1 | temporal R@1 | ΔR@1 | raw mAP | temporal mAP | temporal gap | raw gap | unsafe |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 984 | 0.8861 | 0.8873 | 0.0012 | 0.8420 | 0.8358 | 0.1637 | 0.1652 | 3 |
| 2 | 984 | 0.9203 | 0.9028 | -0.0176 | 0.8598 | 0.8509 | 0.1812 | 0.1816 | 7 |
| 4 | 984 | 0.9114 | 0.8800 | -0.0314 | 0.8565 | 0.8478 | 0.1903 | 0.1943 | 9 |
| 8 | 984 | 0.9064 | 0.8895 | -0.0169 | 0.8501 | 0.8464 | 0.1947 | 0.1908 | 6 |
| 16 | 984 | 0.8932 | 0.8827 | -0.0105 | 0.8484 | 0.8473 | 0.1980 | 0.1896 | 5 |


At p16, temporal R@1 is `0.8827` vs raw `0.8932`, mAP `0.8473` vs `0.8484`, hard gap `0.1980` vs `0.1896`, unsafe `5`. Decision: `A2_FAIL_PARTIAL_NATIVE_MAPPING_AND_NO_SAFE_R_GAIN`.

### A3 identity/prototype diagnostic

The identity audit mapped 1,194/1,298 native-R tracks (`0.9199`), with native appearance variance `0.143507` vs Q0 `0.206780`, native self-cosine `0.856493` vs Q0 `0.793219`, mean reconnected segments `3.0184`, adjacent segment cosine `0.558430`, and native query gap `0.027819` vs Q0 `0.025014`. Fixed M=3 contiguous causal prototypes with symmetric max cosine gave p16 R@1 `0.8905` vs raw `0.8932`, mAP `0.8505` vs `0.8484`, hard gap `0.1679` vs raw `0.1896`, unsafe `6`; only one of four folds was non-decreasing. This rejects the tested prototype formulation, not the general possibility of semantic representation learning.

### O83 and B2/B3/B4/B5 support routes

The original binary row router is retained as a negative comparator: it selected support on 46/76 positive and 52/76 negative events but yielded only 8/76 positive and 10/76 negative both-side reliable events at p16. It did not solve support assignment.

#### B2 listwise + explicit DEFER

Manifest: 8,841 groups, 33,684 candidates, 1,688 reliable TRAIN target groups and 7,153 DEFER groups. Formal training was four folds × 1,000 updates; the first scalar-loader smoke failure was preserved and repaired with an atomic fresh smoke/targeted run. Validation fold details:

| fold | steps | fit groups | val groups | val NLL | candidate/defer acc | defer recall | pred candidate | reliable target |
|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | 628 | 3803 | 2.2481 | 0.3276 | 0.3633 | 2380 | 814 |
| 1 | 1000 | 3917 | 855 | 0.4180 | 0.8561 | 0.9324 | 101 | 115 |
| 2 | 1000 | 4230 | 85 | 1.0099 | 0.6471 | 0.9167 | 21 | 37 |
| 3 | 1000 | 4541 | 68 | 1.7452 | 0.2647 | 0.3333 | 40 | 35 |


Frozen-event replay:

| prefix | pos | neg | frozen both | selected pos | reliable pos | selected neg | reliable neg |
|---|---|---|---|---|---|---|---|
| 1 | 76 | 76 | 17 | 26 | 2 | 25 | 3 |
| 2 | 76 | 76 | 22 | 42 | 4 | 43 | 7 |
| 4 | 76 | 76 | 22 | 43 | 5 | 44 | 10 |
| 8 | 76 | 76 | 23 | 44 | 6 | 45 | 12 |
| 16 | 76 | 76 | 25 | 48 | 6 | 48 | 12 |


Decision: `B2_FAIL_LISTWISE_NO_SUPPORT_GAIN`.

#### B3 joint support matcher and candidate-contract audit

Adding causal DINO history and candidate-set context did not solve the interface. At p16 it produced 9/76 positive and 4/76 negative both-side reliable events. The decisive contract audit found the runtime Q0 candidate count matched the B2/B3 public grouping in only 237/14,691 groups (`1.613%`); median absolute count difference was 19 and mean public-minus-native difference `-26.247`. This is a candidate-universe mismatch, not evidence that geometry or the listwise idea is impossible.

| fold | steps | fit groups | val groups | val NLL | candidate/defer acc | defer recall | pred candidate | reliable target |
|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | 628 | 3803 | 3.0081 | 0.2585 | 0.1820 | 3221 | 814 |
| 1 | 1000 | 3917 | 855 | 0.4209 | 0.8608 | 0.9041 | 147 | 115 |
| 2 | 1000 | 4230 | 85 | 0.8077 | 0.6706 | 0.9167 | 25 | 37 |
| 3 | 1000 | 4541 | 68 | 1.4001 | 0.5000 | 0.7576 | 21 | 35 |


| prefix | pos | neg | frozen both | selected pos | reliable pos | selected neg | reliable neg |
|---|---|---|---|---|---|---|---|
| 1 | 76 | 76 | 17 | 27 | 4 | 21 | 2 |
| 2 | 76 | 76 | 22 | 33 | 7 | 27 | 3 |
| 4 | 76 | 76 | 22 | 35 | 9 | 30 | 3 |
| 8 | 76 | 76 | 23 | 36 | 9 | 30 | 3 |
| 16 | 76 | 76 | 25 | 36 | 9 | 33 | 4 |


Decision: `B3_FAIL_RUNTIME_CANDIDATE_CONTRACT_MISMATCH_AND_NO_GAIN`.

#### B4 native runtime candidate set

The mismatch was repaired by constructing 13,631 native runtime groups with 464,146 bbox-bearing rows, 6,077 reliable TRAIN target groups and 7,554 DEFER groups. Formal four-fold 1,000-update matching on this exact set still yielded only 4/76 positive both-side reliable events at p16, compared with frozen 25/76; deterministic native base-score selection retained both-side candidates in 12/76 events while the post-hoc pool oracle was 61/76. Validation fold details:

| fold | steps | fit groups | val groups | val NLL | candidate/defer acc | defer recall | pred candidate | reliable target |
|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | 840 | 3122 | 3.3013 | 0.1358 | 0.6932 | 826 | 2871 |
| 1 | 1000 | 4296 | 597 | 3.2960 | 0.1675 | 0.9146 | 87 | 515 |
| 2 | 1000 | 4767 | 63 | 2.4142 | 0.1746 | 0.2857 | 41 | 56 |
| 3 | 1000 | 5121 | 43 | 4.0398 | 0.0698 | 1.0000 | 25 | 41 |


| prefix | pos | neg | frozen both | selected pos | reliable pos | selected neg | reliable neg |
|---|---|---|---|---|---|---|---|
| 1 | 76 | 76 | 17 | 19 | 2 | 15 | 0 |
| 2 | 76 | 76 | 22 | 26 | 2 | 21 | 0 |
| 4 | 76 | 76 | 22 | 28 | 2 | 27 | 0 |
| 8 | 76 | 76 | 23 | 32 | 4 | 30 | 0 |
| 16 | 76 | 76 | 25 | 32 | 4 | 30 | 0 |


Decision: `B4_FAIL_NATIVE_SET_MATCHER_NO_SUPPORT_GAIN`.

#### B5 cross-video prior support diagnostic

Using a completed source-track Q0 appearance against native target candidates (per-frame maximum cosine) was a diagnostic only. At p16 it provided target support for 20/76 and both-side support for 17/76, below frozen target 40/76 and source 49/76; negative target support was 8/76. It cannot authorize controller compatibility.

| prefix | pos | neg | frozen source | frozen target | cross target | cross both | negative target |
|---|---|---|---|---|---|---|---|
| 1 | 76 | 76 | 49 | 29 | 9 | 8 | 2 |
| 2 | 76 | 76 | 49 | 35 | 13 | 12 | 4 |
| 4 | 76 | 76 | 49 | 36 | 13 | 12 | 5 |
| 8 | 76 | 76 | 49 | 38 | 16 | 15 | 8 |
| 16 | 76 | 76 | 49 | 40 | 20 | 17 | 8 |


Decision: `B5_FAIL_CROSS_VIDEO_SUPPORT_DIAGNOSTIC_NO_SAFE_HEADLINE`.

## Complete p16 event failure index

The 76-event taxonomy is preserved verbatim in `outputs/iclr27_phase83/audit/failure_taxonomy_76.json` and CSV. The following table is generated directly from that artifact (`76` rows), so hard events were not removed or re-denominated:

| event | fold | category | overall | source | target | src n | tgt n | src max IoU | tgt max IoU | pool src | pool tgt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p19r-pos:f0:c347:s72:t1982:n0 | 0 | 347 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | G_OTHER | None | None | n/a | n/a | False | True |
| p19r-pos:f0:c347:s72:t1982:n1 | 0 | 347 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | G_OTHER | None | None | n/a | n/a | False | True |
| p19r-pos:f0:c347:s72:t1982:n2 | 0 | 347 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | False | True |
| p19r-pos:f0:c347:s72:t2023:n3 | 0 | 347 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | None | None | n/a | n/a | False | False |
| p19r-pos:f0:c579:s92:t349:n0 | 0 | 579 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | G_OTHER | None | None | n/a | n/a | True | True |
| p19r-pos:f0:c579:s92:t1524:n1 | 0 | 579 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | G_OTHER | None | None | n/a | n/a | True | True |
| p19r-pos:f0:c579:s92:t1524:n2 | 0 | 579 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f0:c579:s92:t1748:n3 | 0 | 579 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | G_OTHER | None | None | n/a | n/a | True | True |
| p19r-pos:f0:c805:s0:t1293:n0 | 0 | 805 | G_OTHER | G_OTHER | G_OTHER | None | None | n/a | n/a | True | True |
| p19r-pos:f0:c805:s0:t1293:n1 | 0 | 805 | G_OTHER | G_OTHER | G_OTHER | None | None | n/a | n/a | True | True |
| p19r-pos:f0:c805:s0:t1293:n2 | 0 | 805 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | G_OTHER | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f0:c805:s0:t1293:n3 | 0 | 805 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | G_OTHER | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f1:c211:s9:t959:n0 | 1 | 211 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f1:c211:s9:t965:n1 | 1 | 211 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f1:c211:s9:t978:n2 | 1 | 211 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f1:c211:s9:t978:n3 | 1 | 211 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | E_SUPPORT_SELECTION_WRONG | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | None | None | n/a | n/a | True | False |
| p19r-pos:f1:c229:s119:t1790:n0 | 1 | 229 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | G_OTHER | None | None | n/a | n/a | True | True |
| p19r-pos:f1:c229:s119:t1808:n1 | 1 | 229 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | G_OTHER | None | None | n/a | n/a | True | True |
| p19r-pos:f1:c229:s119:t2170:n2 | 1 | 229 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f1:c229:s119:t2314:n3 | 1 | 229 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f1:c235:s1126:t2677:n0 | 1 | 235 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | None | None | n/a | n/a | True | False |
| p19r-pos:f1:c235:s1126:t2758:n1 | 1 | 235 | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f1:c235:s1126:t2845:n2 | 1 | 235 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f1:c235:s1126:t2873:n3 | 1 | 235 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | G_OTHER | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c118:s275:t1831:n0 | 2 | 118 | G_OTHER | G_OTHER | G_OTHER | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c118:s275:t2340:n1 | 2 | 118 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | G_OTHER | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c118:s275:t1831:n2 | 2 | 118 | G_OTHER | G_OTHER | G_OTHER | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c118:s275:t2340:n3 | 2 | 118 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | G_OTHER | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c126:s220:t2148:n0 | 2 | 126 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c126:s220:t2251:n1 | 2 | 126 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c126:s220:t2851:n2 | 2 | 126 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c126:s1882:t2148:n3 | 2 | 126 | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c133:s1:t2128:n0 | 2 | 133 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c133:s1:t2194:n1 | 2 | 133 | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c133:s1:t2371:n2 | 2 | 133 | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c133:s1:t2874:n3 | 2 | 133 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | G_OTHER | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c237:s483:t2170:n0 | 2 | 237 | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c237:s483:t2251:n1 | 2 | 237 | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c237:s1453:t2170:n2 | 2 | 237 | E_SUPPORT_SELECTION_WRONG | G_OTHER | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c237:s1453:t2251:n3 | 2 | 237 | E_SUPPORT_SELECTION_WRONG | G_OTHER | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c382:s16:t1142:n0 | 2 | 382 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c382:s16:t1617:n1 | 2 | 382 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | G_OTHER | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c382:s16:t1632:n2 | 2 | 382 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c382:s16:t1687:n3 | 2 | 382 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c1144:s377:t827:n0 | 2 | 1144 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c1144:s377:t1927:n1 | 2 | 1144 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c1144:s377:t2565:n2 | 2 | 1144 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f2:c1144:s377:t2565:n3 | 2 | 1144 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c35:s545:t2059:n0 | 3 | 35 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | E_SUPPORT_SELECTION_WRONG | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | None | None | n/a | n/a | True | False |
| p19r-pos:f3:c35:s545:t2215:n1 | 3 | 35 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c35:s545:t2851:n2 | 3 | 35 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | G_OTHER | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c35:s653:t2059:n3 | 3 | 35 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | None | None | n/a | n/a | True | False |
| p19r-pos:f3:c41:s146:t1365:n0 | 3 | 41 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | None | None | n/a | n/a | True | False |
| p19r-pos:f3:c41:s146:t1643:n1 | 3 | 41 | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c41:s146:t1939:n2 | 3 | 41 | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c41:s146:t1939:n3 | 3 | 41 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | None | None | n/a | n/a | True | False |
| p19r-pos:f3:c81:s575:t1814:n0 | 3 | 81 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | G_OTHER | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c81:s575:t1814:n1 | 3 | 81 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c81:s575:t1814:n2 | 3 | 81 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c81:s575:t1955:n3 | 3 | 81 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c95:s0:t980:n0 | 3 | 95 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | G_OTHER | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c95:s0:t1933:n1 | 3 | 95 | G_OTHER | G_OTHER | G_OTHER | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c95:s0:t2051:n2 | 3 | 95 | G_OTHER | G_OTHER | G_OTHER | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c95:s0:t2522:n3 | 3 | 95 | G_OTHER | G_OTHER | G_OTHER | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c99:s569:t1691:n0 | 3 | 99 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | G_OTHER | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c99:s569:t2414:n1 | 3 | 99 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | None | None | n/a | n/a | True | False |
| p19r-pos:f3:c99:s569:t2414:n2 | 3 | 99 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | None | None | n/a | n/a | True | False |
| p19r-pos:f3:c99:s569:t2414:n3 | 3 | 99 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | None | None | n/a | n/a | True | False |
| p19r-pos:f3:c714:s212:t1927:n0 | 3 | 714 | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c714:s212:t2048:n1 | 3 | 714 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | None | None | n/a | n/a | True | False |
| p19r-pos:f3:c714:s212:t2048:n2 | 3 | 714 | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c714:s212:t1927:n3 | 3 | 714 | E_SUPPORT_SELECTION_WRONG | G_OTHER | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c980:s1681:t1905:n0 | 3 | 980 | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c980:s1681:t2460:n1 | 3 | 980 | E_SUPPORT_SELECTION_WRONG | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |
| p19r-pos:f3:c980:s1681:t2460:n2 | 3 | 980 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | None | None | n/a | n/a | True | False |
| p19r-pos:f3:c980:s1731:t1905:n3 | 3 | 980 | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | E_SUPPORT_SELECTION_WRONG | None | None | n/a | n/a | True | True |


Aggregated p16 classes are B proposal exists but max IoU<0.5 = 15, D assigned but transformed IoU<0.5 = 18, E support selection wrong = 36, G other = 7. The pool upper-bound and frozen reliability remain separate; oracle rows are never reported as learned O or OCD success.

## Checkpoint, marker, repair and resource audit

- B2 first smoke failed only because the replay loader treated scalar `bc/bd` as an array; the `.launched` and failed evidence were kept. The smallest loader fix was committed, then a fresh smoke and targeted run passed. No samples, seed, denominator or protocol changed.
- A3 first invocation had a missing `src` import path; the smallest path repair was committed and rerun. B4 first candidate build omitted `norm` (`NameError`); the helper repair was committed and rerun. B3's global normalization/sampler repair and B4's native candidate contract repair were each followed by smoke/targeted checks. These are implementation repairs, not hidden scientific retries.
- The first physical-R process was task-owned PID 17813 with wait shell 17963; it was SIGTERM-ed after profiling exposed repeated per-pair raw-vector recomputation and no artifact had been produced. No external process was touched and no OOM occurred.
- Completion markers: `38` `.done`, `29` `.launched`; unmatched launched markers are `[]`. Checkpoints are resumable `.npz` artifacts with hashes in the ledger. No Phase83 process remained at final audit.
- GPU 0–3 were occupied by external jobs during much of the run; the appearance extraction used idle GPUs 5–8. B2/B3/B4 routers were CPU-bound and did not need GPU placement. GPU4 was not touched after an external job appeared. RAM preflight had approximately 98 GiB available of 125 GiB total (≥25% headroom); `/data1` was near capacity, so large caches/checkpoints were stored under `/data2/usr_for_deadline/trackocd_phase83` and exposed by symlink. No OOM or near-OOM event occurred.

Checkpoint inventory (all paths and hashes are generated from the current filesystem; no checkpoint was silently replaced):

| checkpoint | size bytes | sha256 |
|---|---|---|
| /data2/usr_for_deadline/trackocd_phase83/project_outputs/checkpoints/support_router_formal_f0_step000500.npz | 12066 | c3bb1dd6916580efcd3acb691f48168a425b378883598a754baec9a4c637a577 |
| /data2/usr_for_deadline/trackocd_phase83/project_outputs/checkpoints/support_router_formal_f0_step001000.npz | 12066 | a5cf466b45301d0fa4ca49856e26a49871a34b73c2c39569a8f09141dabc21a9 |
| /data2/usr_for_deadline/trackocd_phase83/project_outputs/checkpoints/support_router_formal_f1_step000500.npz | 12066 | a581521d551a57fb05dbeb8715d4b39f1981977db88eca20f81b86e81be66c5f |
| /data2/usr_for_deadline/trackocd_phase83/project_outputs/checkpoints/support_router_formal_f1_step001000.npz | 12066 | 179ed08099e048ed4fc055333f6951fa12c55366bd8035d355a697239403ce3a |
| /data2/usr_for_deadline/trackocd_phase83/project_outputs/checkpoints/support_router_formal_f2_step000500.npz | 12066 | ac3489be949f6713edb9c0c8f6b53a7ef7d2cafc606f82b353d477b04e554216 |
| /data2/usr_for_deadline/trackocd_phase83/project_outputs/checkpoints/support_router_formal_f2_step001000.npz | 12066 | f27a2022ef3c175481686220bd6a1b57278515a94a8614b2a38a58564dae83f5 |
| /data2/usr_for_deadline/trackocd_phase83/project_outputs/checkpoints/support_router_formal_f3_step000500.npz | 12066 | fb61661f0fb73976c196a1b27adf3240a0cd588d618dc9163d37e031d8b73f26 |
| /data2/usr_for_deadline/trackocd_phase83/project_outputs/checkpoints/support_router_formal_f3_step001000.npz | 12066 | ca2fef9e25a059900ec24b776d13dbeaf09b49c4b776e0114f2aa6e0fef58ba8 |
| /data2/usr_for_deadline/trackocd_phase83/project_outputs/checkpoints/support_router_smoke_f0_step000100.npz | 12066 | 2dca614f6e223edabbc5b30d3998758bf30a9a94bd14a0be15b64f5c0f2fe0f0 |
| /data2/usr_for_deadline/trackocd_phase83/project_outputs/checkpoints/support_router_targeted_f0_step000500.npz | 12066 | c3bb1dd6916580efcd3acb691f48168a425b378883598a754baec9a4c637a577 |


Symlink/storage ledger:

| path | target | exists | sha256 |
|---|---|---|---|
| /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/outputs/iclr27_phase83 | /data2/usr_for_deadline/trackocd_phase83/project_outputs | True |  |
| /data2/usr_for_deadline/trackocd_phase83 |  | True | None |
| /data2/usr_for_deadline/trackocd_phase83/project_outputs |  | True | None |
| /data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl |  | True | 558e6739b47c50793ad8f2123eddd3eda77b53b59bdec608130314c481a32145 |
| /data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz |  | True | ed4405f7946f87579c086c332db743e59c79b9d046ab55ca756a1aea46723714 |


## Gates and what was not run

| gate | result | evidence |
|---|---|---|
| R83 physical temporal mean | FAIL | exact p16 R@1/mAP lower and 5 unsafe flips |
| A2 full-coverage physical | FAIL | mapping remained incomplete; no safe R headline |
| A3 multi-prototype | FAIL | p16 R@1 lower, 6 unsafe, 1/4 folds non-decreasing |
| O83 binary router | FAIL | positive/negative over-activation; both reliable 8/76 |
| B2 listwise | FAIL | both reliable 6/76 vs frozen 25/76 |
| B3 joint support | FAIL | runtime candidate-set mismatch; both reliable 9/76 |
| B4 native-set matcher | FAIL | both reliable 4/76 vs frozen 25/76 |
| B5 cross-video support | FAIL | both 17/76, target 20/76 vs frozen 40/76/49 source |
| C83 unchanged controller | NOT_RUN | no safe R/O result authorized it |
| sealed/public evaluation | NOT_RUN | sealed boundary remained closed |

Training loss, validation NLL, AUC, candidate oracle, or raw/top-K diagnostics are not persistent Commit-CT. This window therefore makes no claim of an OCD success or of a full MOT+OCD result.

## Reproduction and artifacts

The exact commands and input/output hashes are in `outputs/iclr27_phase83/audit/resumed_research_ledger.json`. The principal artifacts are:

- `outputs/iclr27_phase83/audit/a2_full_q0_lineage.json`, `a2_native_mapping.json`, `a3_identity_diagnostic.json`, `b3_candidate_contract_audit.json`, `b4_native_selection_audit.json`, `b5_cross_video_support.json`;
- `outputs/iclr27_phase83/metrics/physical_r_temporal.json`, `a2_temporal_r.json`, `a3_multiprototype_r.json`, `b2_listwise_replay_b2_formal.json`, `b3_joint_replay_b3_formal.json`, `b4_native_replay_b4_formal.json`, `o_support_replay_formal.json`;
- `outputs/iclr27_phase83/manifests/b2_candidate_sets_v1.json`, `b4_native_sets_v1.json`, `support_router_inventory_formal.json`;
- complete event evidence: `outputs/iclr27_phase83/audit/failure_taxonomy_76.json` and `.csv`.

## Final Phase83 scope

This is a corrected closeout of the original 10-hour Phase83 window. It records all registered routes that were actually run in the resumed window and preserves their negative evidence. It does **not** authorize a threshold/memory/controller/backbone lottery, and it does not close the wider TrackOCD research program. A future phase, if authorized, should repair the native support/assignment contract or construct a full-coverage physical stream first, then preregister one causal correspondence route; any later C must still use the unchanged 76+76 protocol and demonstrate safety before sealed evaluation.
