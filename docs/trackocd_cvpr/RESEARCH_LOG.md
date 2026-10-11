# CVPR research log

## 2026-10-12 — CVPR-0 preregistration

Read the full new taskbook and prior core reports/source. Created the new
branch from verified remote `7546527`; preserved old branch, main and outputs.
Registered an uncompleted full Goal A/B/C and the 600 GPU-hour / four-GPU
initial training ceiling. First work is CPU-only historical forensics with
strict sealed-ledger verification. No new fit, Val inference or asset download.
This initial delivery is registration, not completion of CVPR-0 or the goal.

Fresh regression: 396 tests passed in 51.52 s across core, v2,
PANDAS/ByteTrack and the new forensic tests, with no skips. CUDA regression
used an actually idle UUID after process/resource checks; no training occurred.
The six new tests check absorbing contamination, unknown-member distinction,
invalid token lifecycle, truncated ledgers and future append. They are forensic
unit tests, not a claim that the new end-to-end causal protocol is validated.

Registration `20358ac` was pushed, fetched and remotely verified before work.
The first CPU audit verified all 60 ledgers and all evaluator count assertions,
but final JSON serialization failed on NumPy integer counters. This is an
engineering failure, not a scientific finding or a completed delivery. Its
partial CSVs and progress are preserved. Normalize input scalar types, add a
strict JSON test, register a separately named bounded retry; no training or
new inference is involved.

User authorized cleaning unused project data if storage becomes insufficient.
No deletion is necessary now. Inventory and exact references must precede any
cleanup; preserve unique assets, active jobs, valid shards, historical results
and other projects. Ambiguous assets require user confirmation.

Corrected registration `c3a1bbe` pushed/fetched/remotely verified. CPU audit
completed all60cases in75.66s, peak RSS847216640B. Historical evaluator count
assertions pass. Atp16 allfourorders seed1027 emits1NEW+304560EXISTING;
seeds1028/1029 emit304561KNOWN. Observed collapse/propagation, not a unique
causal explanation. No new training, Val inference or Test read. Next register
bounded Train-only frozen-checkpoint diagnostics.12new tests pass, including
real label/role mutation, future append, non-tied ID renumbering and read-only
gradient replay. Equal-time renumbering remains pending; these limited tests
do not prove a dense causal-online evaluation protocol.

Registration`6d6adbe` pushed/fetched/remotely verified before Train diagnostics.
930actual/zero-bias/oracle cases,100geometry cases and18checkpoint gradient
replays completed in243.03s; no optimizer, noGPU, noVal/Test/heldoutlabels.
Frozen zero-bias FULL waits everywhere in main/p16 development; fixed-2bias
causes seed-dependent action collapse. CE/expected-cost gradients are strongly
antagonistic at selected checkpoints; this is not historical gradient logging.
Action-only GT oracle can recover high controlled H/CT but at different
coverage and modified memory; it is not a method improvement. Candidate-only
oracle and first-bad-write suppression still have H0. See root-cause report.
Private14.24MiB traces remain local. No deletion/storage pressure.

Register54capacity/known-count conditions and90actual learned-policy plus
10fixed-baseline original-vs-ID-renumber pairs. FIFO and visual-hash Known
subsets are GT-free; original full label denominators are retained. Actual
Train time ties exist (p16:114tracks in each diagnostic partition). The old
numeric-ID tie ordering is a known protocol limitation, not silently fixed.

Registration`836d0ef` pushed, remotely verified and fetched (one transient TLS
fetch failure, bounded retry succeeded). Memory-control first attempt stopped
when a removed Known prototype also narrowed the sealed evaluator Known list;
the evaluator correctly refused to drop the original denominator. Preserve
the failed attempt, separate evaluator registry IDs from prototype availability,
add an actual full-denominator integration test and register a separately named
retry. No denominator filtering, training, Val feedback or history overwrite.

Corrected registration`bcb1261` pushed/fetched/remotely verified.54capacity/
Known-count cases,90learned renumber pairs and10fixed Raw-B1 pairs completed
in45.63s. Capacities are non-binding (no evictions), all54H/CT0. Actual old
equal-time ID dependence changes H in2/90learned and7/10baseline pairs.
Record old-protocol invariance FAIL; do not change historical outputs or claim
benign token renumbering explains metric changes. New causal protocol repair
is mandatory. Fresh broader regression before the latest denominator test:
405passed in50.83s; the16new CVPR tests include the denominator integration.

Preparing a metadata-only input inventory for the still-missing matched Train
physical counterfactual. New parser byte-skips annotation/category arrays and
decodes only images/videos. A truncated-object fixture caught a missing final
brace check before dataset use; fixed it. No new physical job is authorized by
this parser alone. Published MASA route was rechecked against its official
paper and pinned docs; this is publication-level support, not exact private
release training-stage binding or a qualified strong primary frontend.

Metadata helper post-fix tests caught an indentation regression; `bf98281`
therefore carries an unready helper (not used on any dataset) alongside the
completed controls. Preserve that commit, fix forward and require test completion
before further parser use or registration. The scientific control artifacts
were produced by the already tested and remote-registered`bcb1261` sources,
not this helper. No claim of20passing tests was made for`bf98281`.

After the forward fix, all20CVPR tests passed in2.66s. No dataset was parsed
before this result. `bf98281` delivery was fetched and exact-remote verified.

`8bd2473` pushed/fetched/exact-remote verified. Metadata-only read confirms
57frozen Train-development videos /2065annotation-cadence images, already
local248321596B; no annotations/categories decoded. FixedGT reference remains
258tracks/2949cached observations (prefix-capped, not full track truth).
Register no-GPU asset/plan preflight, then a separate native physical smoke.
Do not pretend this incomplete-release-provenance frontend is a qualified
primary, or that truncated GT makes a clean full end-to-end input ablation.
All22CVPR tests passed in2.62s before this preflight registration.

Registration`18e6c65` pushed/fetched/remotely verified. Asset preflight completed
in19.83s, noGPU/model/training.2065current Train images, DINO346378731B and
MASA558882875B existing checkpoints byte-hashed;14pinned upstream files verified.
Private image-plan SHA`4d12af78a8b321540338915a95e358df452704e33282041693c85cde9f78668f`
and590116B bind the57-video plan. Source-side labels/categories were not decoded.
Next register only first40-image video +4changed-future forwards (44ceiling),
one freshly idle GPU,600s/8GiBGPU/4GiBRSS, no new assets or training. This smoke
does not authorize full57-video execution or qualify a strong clean frontend.

Registration`82c06a939b3f077d3b743351050e7e0044cb0ccd` pushed/fetched/remotely
verified before smoke. A mistyped full SHA on the first launch command was
rejected by the exact-remote guard before any GPU/asset action; corrected from
the actual verified Git output, without weakening the guard. Smoke completed
44forwards/40canonical frames:262IDs/1422track rows/2000ROI detections. Prefix
outputs exactly unchanged after actual hashed future-pixel mutation. Frozen
tensors unchanged, noGT/text/Val/Test/optimizer.57.69s worker,60.11s supervisor,
GPU3443523584B/RSS1618227200B peaks; child terminal0.64,204B canonical private
shard is valid for reuse under a separately registered full57-video plan.
No full57job or DINO extraction launched. All26CVPR tests pass in2.22s.

Smoke delivery`d7f189b` pushed/fetched/exact-remote verified. Register posthoc
aggregation of175320retained Train logit rows from720actual frozen replays,
not a new model execution. Preserve raw/effective WAIT-bias logits, masked
action counts, exposed-top1controller failures and Top8-only candidate upper
bounds separately. Premature-commit causality and per-token histories cannot
be fabricated from independent capped-prefix traces; record those remaining
gaps. All29CVPR tests passed in2.34s before this registration.
User's conditional unused-project-data cleanup permission is recorded in the
new research scope; nothing has been deleted, no storage pressure is present.
### Retained-logit forensic delivery

- Exact registration43e38dee98a6f340c9e5649e144bed24043b1582 verified by
  push/fetch/ls-remote before the registered CPU aggregation.
- Complete175320rows,720actual cases,1440role groups;7.8199617s,
  peakRSS847216640B, zero model forwards/optimization/GPU-hours.
- Main/p16 A2+D2: seed1027 rejects199 correct Known-top1 queries;
  seeds1028/1029 make55/51 wrong Known assignments plus9/9 pseudo-Novel
  wrong-Known each. Healthy anonymous pools absent for9/9 pseudo queries.
- Raw traces remain private. Per-token trajectory was not retained;
  short-prefix error is not relabeled causal premature commitment.
- Overall CVPR research remains active; GoalsA/B/C and matched-input study
  pending. User conditional cleanup recorded; no deletion performed.
### Registered bounded per-state Train audit

- Separate unchanged actual D2 checkpoint replay:90main-order cases across
  Raw/A1/A2,three seeds,two permitted Train partitions,five caps;21915queries.
- Source observer accepts labels only after visual prediction and memory
  update; GT member histograms remain in a separate evaluator object.
- Require exact retained action/logit/order and evaluator parity per case.
  Private token/member/purity trajectories stay local; publish only state
  aggregates without category values. No dense-online/ID-invariance claim.
- CPU600s/RSS4GiB/output64MiB bounds; no optimizer/GPU/Val/Test/holdout.
- Actual module regression32passed; source CLI import checked before launch.
  Premature-commit causal counterfactual remains explicitly pending.
### Per-state Train audit actual delivery

- Registration8851dc0a98a5a612d059c8c5d877b6a772a898a7 verified before
  actual90case/21915query CPU run;32.6791439s,RSS847216640B.
- Every retained action/logit/physical-order/evaluator and mixed-count parity
  passed.38anonymous states across independent cases all end mixed.
- A2+D2/dev/main/p16/1027: firstbad Known anonymous binding144,
  firstcross-category conflict146,114members/17categories/31videos,
  finalmajority0.41228070;113laterwrites,111already-mixed reuses.
- Raw2,582,885B evaluator trace remains local. Overall root-cause work,
  matched physical input and new causal protocol remain pending.
