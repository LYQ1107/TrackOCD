# TrackOCD Phase87 — Causal Persistent OCD Controller Redesign

**Status:** `PHASE87_REGISTERED_ROUTES_EXHAUSTED_FORMAL_GATE_FAIL`  
**Generated (UTC):** 2026-09-06T22:52:27.899273+00:00  
**Window:** see window_registration.json → 2026-09-07T03:53:10.043296+00:00  
**Code head at report generation:** `9db112460432772fcef04b88cdd3a84056dbd5db`

## Executive decision

Phase87 tested the one registered causal controller redesign on the frozen
Phase26/Phase86 physical feature stream. C0 learned a causal state/action policy
but failed safety; the one registered false-merge repair traded a small safety
improvement for lower precision; C1 added only legal prior-track support
features and still failed the pre-registered gate. The final C1 diagnostic was
18/76 Commit-CT, existing precision **0.1731**, and negative false-merge
**0.0789**. The precision and known-stream conditions therefore fail. No
unchanged-controller compatibility, public evaluation, or sealed evaluation
was run. These are controller/interface negative results, not a claim that the
entire TrackOCD task is impossible.

**Decision code:** `P87_CONTROLLER_ROUTES_EXHAUSTED_FORMAL_GATE_FAIL_NO_SEALED`

## 1. Frozen boundary and Phase86 carry-over

- Phase86 remained read-only. No DEV+, Q1, public-new or sealed label was used
  for training, checkpoint selection, or inference; the 76+76 event labels were
  used only for the registered diagnostic scoring after models were frozen.
- No future row/track, category text, semantic ID, or physical ID entered a
  model tensor. Physical IDs were used only for causal eligibility bookkeeping.
- The Phase86 RF correction was independently re-evaluated with observed-step
  ordering; its exact p16 metrics and fold rows are preserved in
  `audit/phase86_rf_causal_correction.json`, with decision
  `RF_PHASE86_OBSERVED_STEP_CORRECTION_NEGATIVE`. All six frozen U2 eval-mode
  checkpoints had rescue=0 and harm=0; decision
  `U2_COMPRESSED_FEATURE_COLLAPSE_CONFIRMED`.
- The frozen physical stream, row order, event denominator, prefixes
  {1,2,4,8,16}, and feature manifest hash were not changed.

## 2. Architecture and causal contract

`raw 768-D track rows + 15 geometry fields → one-layer GRU (256) → bounded
0.05*tanh residual (768-D semantic state) → StateMemoryV2 (16 states × 4
prototypes) → relation features → joint EXISTING/NEW/DEFER/RESET logits`.

Candidate eligibility is exactly `birth_video != current_video AND
birth_track != current_track`. Target evidence is an EMA (0.70 previous,
0.30 current), with a per-track session and a transactional global memory
update only at track end. EXISTING updates the selected state; NEW creates a
state; DEFER/RESET and unresolved tracks do not mutate global memory. Empty
prototype slots are masked at the joint action interface. There is no
threshold-chain or `risk_decode` path. The full contract is in
`docs/iclr27_phase87/PHASE87_PREREGISTRATION.md`.

The targeted contract artifact is `PASS`: joint action dim
19, NEW/DEFER/RESET present, source-before-target
and monotonic prefix checks passed, candidate video/track AND passed, and
non-zero training gradient passed.

## 3. TRAIN event materialization

The manifest excluded 91 videos from the historical event observability file,
kept strict source-before-target chronology, and made category/ID fields
metadata-only. Counts were:

| fold | fit/train events | train positive/negative | validation events | validation positive/negative |
|---:|---:|---:|---:|---:|
| 0 | 33 tracks / 51 | 21 / 30 | 1628 | 748 / 880 |
| 1 | 524 tracks / 950 | 432 / 518 | 191 | 91 / 100 |
| 2 | 505 tracks / 916 | 417 / 499 | 164 | 75 / 89 |
| 3 | 554 tracks / 1011 | 463 / 548 | 82 | 37 / 45 |

No held-event overlap or forbidden model input was found. The baseline artifact
`outputs/iclr27_phase87/metrics/phase19r_baseline_reference.json` records the
safe all-DEFER TRAIN comparator and the historical Phase19R lineage; the old
Phase19R event schema is not silently treated as equal to this new manifest.

## 4. Resource and execution audit

The preflight reserved fold0–3 on GPU5–8, with a four-worker bound and estimated
4 GB peak RSS per worker against a 125 GB host (at least 25% available-memory
floor). Large checkpoints and outputs use the symlink
`outputs/iclr27_phase87 → /data2/usr_for_deadline/trackocd_phase87/project_outputs`.
External GPU0–4 processes were not touched. The only implementation incident
was a desktop wrapper leaving duplicate contract-check processes; explicit
task-owned PIDs were terminated, the valid artifact was kept, and no training
worker or external task was killed. The held evaluator initially failed on a
manifest field alias (`target_tracklet_key` vs `target_track_key`); a normalizer
was added, smoke and targeted replay passed, and the original failure log was
preserved. There was no OOM, no broad kill, and no seed/denominator change.

## 5. Training results

### C0 causal persistent controller (20,000 updates/fold)

| fold | updates | loss first | loss last | mean grad norm | checkpoint SHA256 |
|---|---|---|---|---|---|
| 0 | 20000 | 3.6985 | 0.0000 | 2.7950 | 5db9bdc0a2961594e91cfbf82111eecc1eda50276413165bfd512990ddd6c67a |
| 1 | 20000 | 2.3356 | 0.8677 | 8.6876 | 7defbe653397227773a8504da72858367d5b524e7d543e1d16f93e2611c3cb41 |
| 2 | 20000 | 3.4624 | 2.8048 | 9.9365 | 3ea701861e72743848c4590a863f5dded15d9ecb62672046c657c137f50a5336 |
| 3 | 20000 | 627.7209 | 1.2578 | 8.7671 | 88a9e788dad3c932a19e4bf433a867a45502bace1fdffc8ad137bb03b32ee06f |

TRAIN validation improved over all-DEFER in all folds, but safety was not
stable. The fold validation summary is in
`outputs/iclr27_phase87/metrics/c0_val_f{0..3}.json`.

### Repair1 (false-merge weight 2 → 3)

| fold | updates | loss first | loss last | mean grad norm | checkpoint SHA256 |
|---|---|---|---|---|---|
| 0 | 20000 | 4.3541 | 0.0000 | 2.9980 | a4bac9e428f0c117805fc982a8a48d36e5874be7bd06caff94841e05d230af51 |
| 1 | 20000 | 2.3356 | 1.3771 | 8.9400 | d8ae952e7cd40cc65e82070cc7b5c1c6958f90fb432a717d2629eaf900bf4a6b |
| 2 | 20000 | 4.0767 | 2.7217 | 10.3243 | ed0de5b4432dbf7772e8ad636378418ec454ee5e6677d0cea5c49d4bdbdd1f70 |
| 3 | 20000 | 627.7209 | 1.2470 | 9.1980 | d11fc0af1c87bc0597a8560c68ca2678462aa2db40f14a2ac817d6540b8479b4 |

This was the only registered C0 repair. It was not selected from held events.

### C1 support integration (initialized from repair1)

| fold | updates | loss first | loss last | mean grad norm | checkpoint SHA256 |
|---|---|---|---|---|---|
| 0 | 20000 | 0.0570 | 0.0000 | 0.3975 | da2002681bd14fa02523fa7b5364e07f67b45d18abd2e35faf6036a0ee45e91c |
| 1 | 20000 | 0.6200 | 0.4648 | 18.5680 | 35625d3ac72cdc78f0872867fbddb9a1efe55cd1434086288ed9fbf1042ec755 |
| 2 | 20000 | 8.3393 | 0.7667 | 11.3868 | 059749732c60e49d448e956c0cf71878cf2b24ad3b170a9969ac6c1a8a194a33 |
| 3 | 20000 | 3.1957 | 0.0006 | 16.8145 | d11c2a7af532ce94a4aa01a85e0643e5aac3d3eceb4bbfef39ee8e8d710139ff |

C1 support inputs were eight causal values: source/query cosine, fixed
candidate-count normalization, source length, source variance, observation
quality, and prefix history consistency. They contain no category or identity
shortcut. C1 smoke and fold0 targeted tests completed before the four-fold
run.

## 6. TRAIN validation and 76+76 diagnostic replay

### C0 TRAIN validation

| fold | CT | eligible | existing precision | negative false merge | premature | unresolved |
|---|---|---|---|---|---|---|
| 0 | 117 | 748 | 0.2123 | 0.3216 | 0.2770 | 0.6468 |
| 1 | 18 | 91 | 0.1429 | 0.5300 | 0.5183 | 0.3351 |
| 2 | 15 | 75 | 0.5556 | 0.0562 | 0.0671 | 0.8354 |
| 3 | 11 | 37 | 0.5238 | 0.0444 | 0.1098 | 0.7439 |

### Frozen diagnostic event replay (positive=76, negative=76)

| route | fold | Commit-CT | existing precision | negative false merge | premature | unresolved |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0 | 3/12 | 0.3750 | 0.4167 | 0.0000 | 0.6667 |
| C0 | 1 | 2/12 | 0.2222 | 0.3333 | 0.2500 | 0.6250 |
| C0 | 2 | 3/24 | 0.7500 | 0.0417 | 0.0000 | 0.9167 |
| C0 | 3 | 11/28 | 0.5500 | 0.1071 | 0.1071 | 0.6429 |
| repair1 | 0 | 6/12 | 0.4615 | 0.5000 | 0.0833 | 0.4583 |
| repair1 | 1 | 2/12 | 0.2857 | 0.2500 | 0.2083 | 0.7083 |
| repair1 | 2 | 1/24 | 1.0000 | 0.0000 | 0.0000 | 0.9792 |
| repair1 | 3 | 8/28 | 0.4000 | 0.0714 | 0.2143 | 0.6429 |
| C1 support | 0 | 3/12 | 0.4286 | 0.3333 | 0.0417 | 0.7083 |
| C1 support | 1 | 2/12 | 1.0000 | 0.0000 | 0.1250 | 0.7917 |
| C1 support | 2 | 3/24 | 1.0000 | 0.0000 | 0.1667 | 0.6458 |
| C1 support | 3 | 10/28 | 0.5263 | 0.0714 | 0.2500 | 0.6071 |

Aggregate comparison:

| route | Commit-CT | existing precision | negative false merge | category coverage | video coverage |
|---|---:|---:|---:|---:|---:|
| C0 | 19/76 | 0.1450 | 0.1711 | 10 | 15 |
| repair1 | 17/76 | 0.1250 | 0.1447 | 9 | 13 |
| C1 support | 18/76 | 0.1731 | 0.0789 | 11 | 17 |

These replays are diagnostic OCD evidence, not a sealed score and not a
checkpoint-selection signal. Known micro/macro thresholds are marked not
applicable because this route intentionally evaluates novel events only; they
are therefore not silently reported as zero.

## 7. Gate decisions

| gate | result | evidence |
|---|---|---|
| Contract checks | PASS | joint action/causal/mask/gradient checks passed |
| C0 formal | FAIL | 19/76, precision .1450, false merge .1711 |
| repair1 formal | FAIL | 17/76, precision .1250, false merge .1447; recall/precision regressed |
| C1 formal | FAIL | 18/76, precision .1731, false merge .0789; known gate conditions unavailable |
| unchanged controller compatibility | NOT RUN | C1 Gate R/formal gate failed |
| public / sealed | NOT RUN | sealed boundary preserved |

The final formal decision is `C1_GATE_FAIL_STOP_PHASE87_CONTROLLER_ROUTES`.
The improvement in negative false merge under C1 did not solve the dominant
precision/unresolved trade-off. C1 therefore cannot legally be connected to
the frozen Phase19R controller, and no threshold, StateMemory, or controller
lottery is authorized under this phase.

## 8. Method audit and limitations

The read-only recent-method audit covers OVTR (`500e72c`, MIT), TRACT,
COVTrack, ObjectRelator (Apache-2.0), C3Po, AGE, and TALON with their official
URLs, revisions, and compatibility decisions in
`docs/iclr27_phase87/PHASE87_METHOD_AUDIT.md` and
`outputs/iclr27_phase87/audit/methods.json`. No external code or weights were
downloaded. None provides a verified text-free causal prior-video support set
with this 768-D transactional interface.

The main limitations are severe fold imbalance (fold0 has only 51 training
events), overfitting to the positive/negative action labels, unresolved
support quality, and a retrieval/controller interface whose precision gate is
not met. Training loss decreases sharply in some folds, but the replay safety
metrics show why loss cannot be treated as task success.

## 9. Reproduction and artifact ledger

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase87/contract_checks.py
/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase87/run_autonomous.py
scripts/iclr27_phase87/run_c0_four_fold_supervisor.sh
scripts/iclr27_phase87/run_replay_supervisor.sh
scripts/iclr27_phase87/run_repair1_four_fold_supervisor.sh
scripts/iclr27_phase87/run_repair1_replay_supervisor.sh
scripts/iclr27_phase87/run_c1_four_fold_supervisor.sh
scripts/iclr27_phase87/run_c1_replay_supervisor.sh
```

Key machine-readable artifacts:

- `outputs/iclr27_phase87/audit/window_registration.json`
- `outputs/iclr27_phase87/audit/phase86_rf_causal_correction.json`
- `outputs/iclr27_phase87/audit/phase86_u2_evalmode_correction.json`
- `outputs/iclr27_phase87/audit/causal_event_build.json`
- `outputs/iclr27_phase87/audit/contract_checks.json`
- `outputs/iclr27_phase87/audit/phase87_c0_decision.json`
- `outputs/iclr27_phase87/audit/phase87_repair1_decision.json`
- `outputs/iclr27_phase87/audit/phase87_c1_decision.json`
- `outputs/iclr27_phase87/metrics/c0_aggregate.json`, `repair1_aggregate.json`, `c1_aggregate.json`
- checkpoints under `/data2/usr_for_deadline/trackocd_phase87/project_outputs/checkpoints/`

The project output symlink target is `/data2/usr_for_deadline/trackocd_phase87/project_outputs`; no feature or checkpoint copy was made. At report time all
Phase87 workers and supervisors had exited, JSON artifacts parsed, and no
public/Q1/sealed artifact was created. Code was pushed to
`https://github.com/LYQ1107/TrackOCD` before report generation.

## 10. Next research action

Do not repeat C0/repair1/C1 weighting or threshold variants. The evidence
supports a new supervision/task-contract investigation: materialize balanced,
cross-fold causal support with a verifiable action utility target, or revisit
the physical-track/support observability interface. Any future route must be
registered separately and demonstrate safety/precision on TRAIN validation
before another controller compatibility attempt.
