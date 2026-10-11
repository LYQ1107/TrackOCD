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
### Registered matched-Train physical diagnostic, not main qualification

- Separate source/config registration for57old Train-development videos,
  all2065metadata-cadence images; one frozen MASA worker, noGT/text/Val/Test.
- Explicitly validate and reference the completed40frame/64,204B video47
  smoke under its original config/result/shard hashes, without copying or
  re-extracting it. At most2025new forwards; no semantic training.
- Exact release-stage provenance remains incomplete. This route is a limited
  diagnostic, not primary-frontend qualification or full TAO prediction cache.
  Old258track/2949observation GT reference is capped16, not complete GT.
- Fresh idle UUID and process/RAM/disk checks at launch; oneGPU8GiBpeak,
  4GiBfreeGPU reserve, RSS4GiB,7200swall,64MiBphysicaloutput ceiling.
  Subsequent physical+feature planning capped1GiB; project incremental hard
  30GiB and free10GiB reserves maintained. Only directly owned child may stop
  on failure; all completed shards/logs preserved and resume is explicit.
- Actual34CVPR tests passed2.27s; strict plan/category/split/cadence mutation
  and no-copy/changed-source reuse tests included. CLI import checked.
  Actual private plan and source video integrity preflight also run before
  any GPU request. No training budget consumed or data deleted.
### Physical diagnostic launched; cross-video chronology registered

- Registered494615b654cbd7d2163276c318266c3c4efc1f24 pushed/fetched and exact
  remote SHA checked before owned supervisor started. Fresh idle UUID selected
  by code, not historical index. Worker244295 observed132.01s/410new forwards,
  11of57videos complete; current process is ours, foreign jobs untouched.
- Fresh core/v2/CVPR regression417passed59.48s. Available disk65240059904B;
  no data cleanup needed/performed. Current physical/source files stay frozen.
- Separate CPU/no-model registration: stream21915retained state rows from
  90cases, verify every prior/post member/purity snapshot and final38states;
  count Existing cross-video reuses from *prior* videos, not final video count.
  No new GT labels/model forwards/optimization;120s/RSS1GiB bounds.
- Actual36CVPR tests passed2.33s including chronological/purity tampering and
  cross-video prior-history fixtures; source CLI import checked.
### Retained cross-video chronology delivery and actual stage manifest

- Registration833dd77dc559b97c0426c706b8cfa5538c4bed63 verified then21915rows,
  90cases/38states streamed in0.4748713s, no new model/GT/optimization.
  Every prior/post member/purity and final state matches the retained audit.
- A2+D2/dev/main/p16/1027:106of113 Existing writes are true prior-history
  cross-video reuse;106after firstbad and already-mixed. Allindependent cases:
  8166cross-video reuses/8128after mixture, not one shared38state memory.
- Source-backed actual experiment_manifest added. Helper integration first
  refused an absent asset-preflight launch receipt; corrected binding records
  original receipt's missing commit and posthoc Git-byte verification against
  logged pre-execution18e6c65. No measured result altered or receipt invented.
-37CVPR tests passed3.13s; manifest assembled8completed substeps plus actual
  running physical diagnostic with no fabricated metric/complete-root status.
### Bounded Train physical actual delivery; all-ID DINO interface registration

- Registration494615b654cbd7d2163276c318266c3c4efc1f24: all57videos/
  2065registered Train images complete648.7777438s; supervisor653.0950387s,
  directly owned worker244295 terminal0.2025new forwards, valid40image source
  shard reused without copying/re-extracting.16933IDs/87130observations/
  103042detections/6311singletons; GPU3443523584B/RSS1628020736B peaks.
- Frozen state unchanged; no GT/text/Val/Test/optimizer/foreign interference.
  Valid atomic shards and manifest stay private. Limited release binding,
  annotation cadence,50ROI cap and selected/capped GT reference remain explicit.
- Separate interface registered: same common DINO source/checkpoint/crop/norm,
  all262video47IDs/1293first≤16observations including86singletons; no score or
  role filtering. Current-image batches8; one freshly idleGPU8GiBpeak,
  4GiBfree reserve/RSS4GiB/600s/output8MiB. No57video feature authorization.
- Real invisible-tail mutation checks cover all five caps with short tracks
  retained. Module/CLI/real-input preflight tested;38CVPR tests passed6.08s.
  No training or matched semantic score claimed; manifest updated with actual
  completed physical result, no stale running metric or data deletion.
### All-ID DINO worker failed final JSON; no duplicate encoding retry

- Registration43e4a283e0e2cb124e3d237eebbc1c485772fb40 remotely verified before
  owned worker249997. Actual180batch forwards reached image39,1293features
  written to1,860,310B private NPZ. Final result.json failed NumPyint64
  serialization in tail-check count; worker exited1, supervisor100.6577463s.
- Preserve original directory/log/progress/launch/NPZ. No normal PASS or
  invented GPU/RSS numeric peaks: these values were not persisted before
  JSON failure. Source path passed frozen/tail/compact-reader guards before
  the final serialization; hash value itself not retained.
- Correct helper boundary to native ints and add strictJSON regression.
  Separately register CPU120s/RSS1GiB recovery that verifies original archived
  source/log/payload bindings; all262ID/1293row metadata/geometry/quality,
  current pixels/L2/reader and actual prefix-tail mutations. No new encoder,
  GT labels/GPU/optimizer, payload copy/rewrite or favorable old-result edit.
- Fresh38CVPR tests and recovery CLI import passed before registration.
  Full57feature extraction remains unlaunched; scientific goals active.
### CPU-only feature receipt recovery actual delivery

- Exact remote682ad5a78cd564bfd86d0063a865cdec6620ae87 verified before CPU
  recovery;5.7846041s,reportedru_maxrss847216640B,GPUfalse/newforwardfalse.
-262IDs/1293rows/86singletons validated,1310prefix checks/477real tail
  mutations. Original1,860,310B NPZ SHA70c496ad... unchanged; no copying,
  rewriting, repeated encoding, new GT/Val/Test or optimizer.
- Preserve original exit1/100.6577463s, missing numerical GPU/RSS/frozen hash
  explicitly unretained. Recovery namespace is not a fabricated originalPASS.
- Full bounded physical inventory73509first≤16observations/16933IDs,
  112909824B uncompressed visual payload;72216new observations if source
  interface reused. No57feature job launched; matched integration next.
### Registered tiny semantic/capped-reference engineering integration

- CPU-only45cases: Raw/A1/A2 ×three old frozen D2 checkpoints ×fivecaps;
  all262valid predicted IDs sealed and Parquet roundtripped before geometry
  matching/evaluation. Identical fixed model/prototype/WAIT bias also runs
  inherited GT-controlled inputs; no new fit or operating-point selection.
- Firstvideo47 has7selected/capped Known GT references/94observations,
  zeroNovel/cross-video opportunities. Full7targets retained even unmatched;
  H/Novel/CT null/NOT_APPLICABLE, no discovery success from this small check.
- Class-free IoU≥.5 per-frame cardinality+IoU matching; canonical one-to-one
  assignment needs≥2shared observations,≥.5of capped GT support, no other
  matched GT identity. Labels/roles never used to select matches or enter
  prediction memory. Unmatched predictions retained and explicitly unknown,
  not background/pollution. Existing numeric-ID routing remains unrepaired.
- Source/config/model/cache/recovered-payload verification; CPU600s/RSS4GiB,
  private ledgers≤16MiB, no GPU/optimizer/Val/Test/final-holdout labels.
  Actual40CVPR tests and sourceCLI import passed before launch. Full57features
  still unlaunched; this is not the full matched-input scientific result.
