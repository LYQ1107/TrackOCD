# TrackOCD Phase85 — temporal-mean physical parity and support-conditioned correspondence

Status: registered TRAIN-only exploratory window (2026-09-05 UTC). This namespace is
independent of Phase84; all older outputs are read-only comparators.

## Frozen facts and hypotheses

Phase84 showed that the physical replay had a provenance/aggregation contract problem.
The first hypothesis is that a causal running application mean can be reconstructed
without changing the Q0 physical stream. A second hypothesis is that the legal
TRAIN prior-support pairs contain enough raw-anchored candidate coverage for a
bounded support-conditioned selector to improve cross-video correspondence. The
single-anchor physical adapter is explicitly a diagnostic and is not expected to
pass a retrieval gate.

Frozen physical comparators are the Phase83 native Q0 stream and Phase84 source
cache. The original 76 positive and 76 negative event protocol, prefix set
{1,2,4,8,16}, row keys, causal chronology, and evaluator denominators remain
unchanged. No DEV+, Q1, public-new-model or sealed labels are read.

## Registered order

1. Reconstruct causal temporal-mean physical lineage and verify an exact Q0 parity
   adapter. Missing image dimensions are an error; no 640x480 fallback is allowed.
2. Evaluate Q0 versus the temporal-mean and single-anchor physical representations
   on the TRAIN-disjoint Phase30 retrieval protocol. This does not select an online
   threshold or claim MOT/OCD success.
3. Build a legal prefix support manifest from public TRAIN episode records only.
   Query and support videos are disjoint, support is prior-only, and positive links
   are loss metadata/post-hoc labels, never model inputs. Candidate sets are sorted
   by raw cosine and capped at K=32. A source-track hash split (three folds) is
   materialized before training; categories and target videos are disjoint between
   each fit and validation partition.
4. Audit candidate rank recall at K=4/8/16/32. The default K is 16 when TRAIN
   validation recall at K=16 is at least 0.90, otherwise K=32. No K is selected
   from held events.
5. Train one small support reranker/defer model. It can add a bounded residual to
   raw cosine and predict a TRAIN teacher-policy defer flag. The fixed teacher is
   `support_quality >= 0.2 and bridge_margin >= raw_margin + 0.005`; there is no
   threshold sweep. Loss is balanced listwise/pairwise ranking plus safety-preserving
   defer BCE. Raw is the exact fallback for invalid/missing support.
6. Replay the frozen 76-event protocol only after the TRAIN model is frozen. Report
   raw, reranked, and reranked+defer diagnostics separately. No controller,
   StateMemory, threshold, backbone or sealed evaluation is authorized in this
   window unless an explicit later gate is met.

## Model and input contract

The selector receives only key-aligned DINO/Phase84 causal track vectors, candidate
geometry/quality metadata, raw cosine and causal age/history fields. Category names,
semantic or physical IDs, text, future rows/tracks, GT boxes and controller state are
not model inputs. TRAIN GT is used only to form positive/hard-negative labels and to
measure coverage. Support timestamps must precede the query event.

The residual is bounded (`0.05*tanh`) and cannot erase the raw score. The defer
head is a calibrated TRAIN policy; if support is missing or invalid, probability is
forced to zero and raw ranking is returned exactly. The controller interface is not
modified.

## Resources, recovery and gates

At most four bounded workers are allowed, with one worker per GPU and at least 25%
system RAM free. Each unit writes `.launched` before work and atomically writes
`.done` only after metrics/checkpoints are complete. Checkpoints are resumable and
large data remain on `/data2` through the Phase85 output symlink. Long jobs use one
blocking supervisor wait. A repair is smoke → targeted regression → resume, at most
three cycles for one root cause.

Physical parity must have max absolute error <=1e-5, zero bad rows, and denominator
984. The support diagnostic is retained even if the single-anchor route fails.
An eventual support route is considered promising only if TRAIN-disjoint p16 replay
improves raw without unsafe flips and with at least three fold directions; retrieval
does not by itself authorize controller or sealed evaluation.

## Reproduction

```bash
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase85/run_full_temporal_mean_physical.py
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase85/build_physical_r_adapter.py --mode q0 --tag q0_parity_v5
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase85/build_physical_r_adapter.py --mode improved --tag improved_single_anchor_v2
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase85/evaluate_physical_r.py --q0 outputs/iclr27_phase85/manifests/physical_r_q0_q0_parity_v5_vectors.npz --improved outputs/iclr27_phase85/manifests/physical_r_improved_improved_single_anchor_v2_vectors.npz
```

Public/Q1/sealed inputs remain sealed (`public_dev_q1_sealed_accessed=false`).
