# Phase83 O-support report

**Generated (UTC):** 2026-09-04T08:30:25.601339+00:00  
**Source commit:** `9e6acb21bbebc41faa092621e7808eb675150021`  
**Route:** `support_quality_router_v1`, one hidden-128 class-agnostic MLP; no detector/proposal/physical tracker/controller changes.

## Frozen O contract and callgraph

The Phase75B evaluator, 76 positive + 76 negative events, prefixes `(1, 2, 4, 8, 16)`, row key and denominator were read-only. `assigned`, `row_iou`, and `track_temporal_iou` are upstream/evaluator metadata used only for TRAIN labels and post-hoc scoring; they are not router inputs. The callgraph and source hashes are in `outputs/iclr27_phase83/audit/support_assignment_callgraph.json`.

Frozen Phase75B p16 event reliability is source 49/76, target 40/76, both 25/76. The causal proposal-pool upper bound (native max IoU≥0.5, diagnostic only) is source 72/76, target 64/76, both 61/76. No proposal is missing at p16.

## 76-event failure taxonomy (p16)

| class | events |
|---|---|
| B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05 | 15 |
| D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05 | 18 |
| E_SUPPORT_SELECTION_WRONG | 36 |
| G_OTHER | 7 |


The 36 E/C cases have a good native pool but no reliable frozen assigned row, while B is a genuinely insufficient pool. Full evidence for every event/side (candidate counts, scores, frame IDs, IoUs and row details) is in `outputs/iclr27_phase83/audit/failure_taxonomy_76.json` and CSV. Prefix aggregation is in `observability_by_prefix.json`.

## TRAIN-only router data and training

Rows came only from non-event public TRAIN videos and roles `known_bank`/`novel_correspondence_train`; all 91 event videos were excluded. Inputs are score, normalized geometry, causal age/stability, history/gap, proposal density/ambiguity and current-vs-causal DINO cosine statistics. GT/assigned/IoU fields form the TRAIN target `assigned==1 AND row_iou>=0.5` only. No category, text, physical/semantic ID, future, event polarity or StateMemory field enters the tensor. Four folds use the frozen video/category-disjoint manifests; scaling is fit on each fold's TRAIN split and threshold is the preregistered p≥0.5.

| fold | steps | fit rows | fit pos rate | val ROC-AUC | val F1 | first loss | last loss |
|---|---|---|---|---|---|---|---|
| 0 | 1000 | 571 | 0.07355516637478109 | 0.7552 | 0.2717 | 1.9225 | 0.2335 |
| 1 | 1000 | 3638 | 0.16520065970313358 | 0.8264 | 0.5000 | 1.3658 | 0.8900 |
| 2 | 1000 | 4185 | 0.14074074074074075 | 0.8459 | 0.6486 | 1.2753 | 0.7636 |
| 3 | 1000 | 4439 | 0.1489074115791845 | 0.5552 | 0.2128 | 1.3483 | 0.6795 |


Smoke (100 updates) and fold0 targeted (500 updates) produced finite checkpoints and atomic markers. Formal folds (1000 updates) were CPU bounded because GPUs 0–3 were external jobs and the route does not require GPU; no OOM occurred. Checkpoints and hashes are recorded in `outputs/iclr27_phase83/metrics/support_router_aggregate_formal.json` and `manifests/support_router_inventory_formal.json`.

## Frozen 76+76 event replay

| prefix | pos | neg | frozen both | learned reliable both | learned selected both | negative selected both | negative reliable both |
|---|---|---|---|---|---|---|---|
| 1 | 76 | 76 | 17 | 8 | 37 | 42 | 11 |
| 2 | 76 | 76 | 22 | 9 | 39 | 44 | 11 |
| 4 | 76 | 76 | 22 | 9 | 40 | 47 | 11 |
| 8 | 76 | 76 | 23 | 9 | 43 | 48 | 11 |
| 16 | 76 | 76 | 25 | 8 | 46 | 52 | 10 |


At p16 the learned router selected support on 46/76 positive events but only 8/76 were both-side reliable (frozen=25/76); it selected support on 52/76 negative events (10/76 had both-side reliable rows). This is over-activation and does not improve O; it cannot be used to claim C or Commit-CT progress.

## O83 decision and reproduction

`O83_FAIL_NO_SUPPORT_GAIN`. The router does not approach the 61/76 pool upper bound, reduces strict both reliability from 25/76 to 8/76, and activates on many negative events. No threshold was tuned on held events, no controller/StateMemory/backbone/public/sealed path was run, and no event was removed. The next action is a contract-level support/proposal assignment investigation, not another router variant.

Reproduce: `python scripts/iclr27_phase83/train_support_router.py --folds 0 --steps 100 --tag smoke`; `python scripts/iclr27_phase83/train_support_router.py --folds 0 --steps 500 --tag targeted`; `python scripts/iclr27_phase83/train_support_router.py --folds 0,1,2,3 --steps 1000 --tag formal`.
