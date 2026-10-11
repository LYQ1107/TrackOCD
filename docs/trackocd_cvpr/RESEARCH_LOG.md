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
### Tiny frozen semantic integration actual delivery

- Exact remote6c7cca9afe7a583aa5acf3d36a0617c012ad087a verified before
  CPU45cases,36.8506527s,reportedRSS847216640B,no optimizer/GPU/Val/Test.
- All262predictedIDs/7GT-controlled references retained;90private ledgers
  seal/reader roundtrip exactly before geometric reference matching.
-85matched observations,3canonical targets/7selected capped Known GT,
  259unmatched predictedIDs remain unknown. All7targets in denominators,
  zeroNovel/cross-video opportunities → H/CT unavailable, not manufactured.
- Actual wiring pass only; no full matched-study, primary quality, dense
  protocol or uniquely input-only causal claim. Full57features not launched.
### Registered bounded Train feature continuation after real integration

- Full old-development diagnostic:57videos/16933IDs/73509first≤16observations,
  allshort/low-score/unmatched IDs retained. Explicitly validate/reference
  source recovered1293observations (originalfailure remains), no copy/redo;
  at most72216new observations. No GT/role/Val/Test encoder access.
- Uncompressed visual planning112909824B. Output≤256MiB, physical+features
  ≤1GiB via tighter per-run bounds; free10GiB and CVPRincremental hard30GiB
  reserves remain. No new images/weights/env downloads or deletion.
- Current-image crop batches8, identical commonDINO source/checkpoint/crop/
  normalization and FP32/FP16 protocol. Two workers alternate metadata-order
  videos, no label/score/feature-based assignment. Fresh idle UUID/process
  checks; perworker GPU8GiBpeak/free4GiB/RSS4GiB,25%systemRAM reserve,
 7200swall. Own supervisor stops only its direct children on safety failure.
- Video-atomic markers/strict JSON/reader validation; preserve attempts and
  explicitly resume valid complete shards. Worker file barrier denies GT,
  evaluator integration outputs/Val/Test; engineering status checked only by
  supervisor and never forwarded as model input. No new semantic training.
- Actual42CVPR tests passed, including disjoint metadata assignment, strict
  no-copy recovered-input reuse, changed protocol/physical source refusal.
  CLI import and actual57video/reused-input preflight passed. Limited
  frontend release binding, annotated cadence, truncated GT and old numeric
  routing remain limitations; no main scientific success claim.

### Running feature continuation and matched-reference preregistration

- Two owned frozen DINO workers are actually running, not training. Latest
  observed summaries: worker0 20complete videos/25567new observations and
  worker1 18complete videos/25827new observations; these are progress snapshots,
  not a completed57video result. Valid source1293observations remain referenced.
- Registered next45cases: Raw/A1/A2 x3frozen D2 seeds x5independent prefixes,
  both all16933predicted IDs and all258selected/capped GT references in the
  same actual metadata video order, same frozen Known proto/checkpoint/bias.
  All90private ledgers seal/roundtrip before category-free geometric joining.
  Keep249Known/9pseudo-Novel GT denominators and all unmatched prediction IDs.
  Matching cannot certify full GT completeness/global track purity. Old numeric
  ID routing and independent prefix semantics remain explicitly unrepaired.
- Completed feature result/manifest must actually exist and bind to exact
  a02455f registration/config. Their hashes will be measured at launch, not
  fabricated at preregistration. One-video-resident reader keeps every low-score
  singleton;8actual tests pass, including hidden-tail mutation/hash/count/path
  refusals. CPU1/RSS4GiB/wall5400s/output128MiB/no GPU/no optimizer/no Val/Test.
- Fresh broad regression425passed46.34s before the8new reader tests; no missing
  terminal output was assumed a PASS. Current free disk about34GiB; no cleanup
  needed or performed. User conditional cleanup limited to confirmed unused,
  recomputable project inputs; active inputs/valid shards/unique history and
  other projects preserved. No external process interference.
- Read-only TALON audit now includes full pinned model/loss/train/test/CUBsplit
  source and official CVF→repo binding. No LICENSE in complete pinned tree,
  so code reuse permission is not assumed; no source executed or vendored.
  Batch TTA/Hungarian Known remapping are not our causal/fixed-Known protocol.
  CVPR1 minimum twelve works/additional two/architecture are still pending;
  this preparation does not skip remaining CVPR0 diagnostics or select a method.

### Actual bounded Train feature delivery — 57/57 complete

- Exact a02455f074819dc002def1da6b3acc93073c4fa6 sources/config retained;
  actual two-worker supervisor returncodes0/0,error null,no foreign interference.
  Complete57videos/16933IDs/73509observations. New72216, referenced1293,
  valid source not copied/reencoded and its original failed exit still preserved.
- Supervisor1155.1163105s (~19.25min); worker elapsed1132.1724115/
  1148.8610429s, peakGPU reserved945815552B each, peakRSS1156579328/
  1157853184B. Training GPU-hours0, optimizer0, no GT/role/Val/Test encoding.
- Actual result SHA256 af12c39c6b8513a635cccccd3aa99f0e9c9c5d7fe1a871b5958e35be797954db;
  private complete manifest203517B SHA256
  39dcf2f750470c880876e4dc99d35a397d5e38ef10ba7dc3f02d150697d343a3.
  Only aggregate result is public; private manifest/NPZ/logs not Git-added.
- New measured input hashes now bind next preregistered matched-reference
  protocol before any semantic evaluation. Full reference roles independently
  rechecked development-only249Known/9pseudo; no heldout labels decoded.
- Broad actual regression433passed40.67s, including8new feature reader tests.
  CVPR retained tree165034185B (~157MiB), free disk~30GiB; no deletion needed.
  These are diagnostic feature results, not main frontend qualification or
  fullTAO cache/new training/GoalsA/B/C/scientific method success.
- LTC actual active model/generator/loss/train/evaluation/CUB code audit also
  recorded. Author arXiv→repo binding verified; Findings venue not independently
  verified and no LICENSE in pinned tree. Source/paper discrepancies retained,
  DSU numeric compaction and GT-column topK metrics not imported into TrackOCD.
  No upstream code executed, copied or trained; literature stage remains partial.

### Fixed actual-history future-input shadow preregistration

-72finite cases Raw/A1/A2 x3frozenD2seeds x2Trainpartitions(dev258/policy229)
  x4earlier prefixes1/2/4/8,17532queries. Each compares an external p16shadow
  at exactly the original prior visual memory, only when later evidence exists.
  No extra cache/inference/training, no heldout/Val/Test labels. Actual decision
  and shadow each predict visually before the GT-only nominal-cost observer.
- Both visual history and external observer are updated ONLY by original
  current-prefix predictions. Hash-check visual bank before/after each query;
  original sealed actions/logits/tokens/order/evaluation must exactly match
  retained unchanged state replay. Short tracks remain no-later-evidence cases.
- Count laterzero-nominal-cost, laterWAIT and laterstill-wrong outcomes against
  actual wrong commitments with laterinput, plus correct→wrong flips; these
  are future-information upper bounds, NOT observed waiting gain, unique
  early-commit causality, long-horizon regret, dense online, or model selection.
-4real tests passed: changedfuture changes onlyshadow; changedGTrole/category
  changes onlyexternalcost; deliberatelyillegal shadow write is refused;
  singleton gets no fabricated future. CPU1/RSS4GiB/wall600s/output32MiB.
  Actual execution waits for current matched-input CPU work, not yet complete.
- Documentation formatting correction: the preceding delivery had a diff-check
  warning for an extra EOF blank line; now corrected forward, no history rewritten.

### Actual matched inputs and fixed-history shadow delivery

- Registered ed699ab20448ccce30ac08ce59f49f350e814c84 matched run completed
  45cases in700.4741298s CPU-only, peakRSS1665089536B, training0. All90
  retained ledger hashes/bytes/rows rechecked,773595events. Full fixed
  denominators249Known/9pseudo/5reuse;107canonical matches out of258selected
  capped references,16826unknown unmatched Pred IDs. No GT in Pred inference.
- All CT0, but H is not universally0: Raw1029p2 Pred0.015503876/
  GT0.159021407 and Raw1029GTp16 .007936508. No selective denominator removal,
  main-quality claim, unique physical-input cause or numeric-ID repair.
- Registered356a38672e56d5b550b56d17c771c184e5e9b060 shadow completed72cases/
  17532queries in35.4547905s CPU-only, peakRSS847216640B. All actual actions,
  logits,tokens,order,metrics match original; shadow cannot update either
  visual history or external GT observer. Private trace976524B/hash/rows verified.
- Repeated-case sums (not unique tracks):7624Known wrong-with-later split
  877zero-nominal-cost/12WAIT/6735still-wrong;369correct→wrong.790pseudo wrong
  with later split18/0/772. A2dev1027p8 all85Known+7pseudo wrong stay wrong.
  Do not infer live wait gain, long-horizon regret or unique early-commit cause.
- Added aggregate table preserving the distinctions among input commitment,
  reference prediction/non-WAIT and reuse coverage; four actual input mutations
  are refused.59CVPR tests passed2.80s; broader442tests passed39.35s.
- User conditional cleanup policy remains in force; free disk~27GiB and own
  bounded outputs far below15GiB soft ceiling. Subsequent check shows23GiB
  free; the cause of workspace-wide growth is not attributed to any process.
  No files deleted; no external
  process interference, new weights, optimizer, Val/Test or heldout access.
  Both result receipts/CSV public; feature/prediction/GT traces stay private.

### Current-observation router interface preregistration (not dense acceptance)

- New isolated router ingests only the current frame, accumulates only seen
  observations, and keeps opaque physical routing IDs outside PrefixView.
  Same-frame groups are ordered by complete visible input bytes, never ID.
  Exactly indistinguishable complete prefixes form an atomic group rather
  than arbitrary ID-ordered writes; downstream policy integration must broadcast
  one decision and apply one group write. No old replay/controller changed.
-11actual tests passed0.13s: ID bijection plus event reorder, changed/added
  future input, exact ties and later separation, different prior histories,
  input alias mutation, nonfinite/negative/shape/duplicate refusal, chronology,
  empty frames and observed resource bounds. Interface field rejection is NOT
  a substitute for end-to-end category/event/role mutation tests.
- Registered nextCPU audit uses all57existing annotated-cadence first<=16
  feature inputs, no extra cache/labels/GT decode/GPU/training. Compare visible
  input and weighted-mean byte digests under opaque ID/row mutations; actual
  future-feature negation/extra futuretrack checks on firstvideo. CPU1/RSS2GiB/
  wall600s/output4MiB. No learned policy/actions/memory/semantic evaluator, so
  cannot report historical ID-invariance repair or dense-pixel acceptance.
- Grounded Correspondence author-linked project page successfully read via
  existing proxy after browser tool failed; official repo link now verified.
  Pinned base model/loss/train/evaluation/greedy initialization/Hungarian scan
  source reviewed with explicit partial ranges; MIT license read. LatestHEAD
  includes later extensions; exact conference-release/venue binding unverified.
  Instance matching/video-reset slot IDs are not persistent category discovery.
  No source copied/run; CVPR1 twelve-work/two-extra requirement still pending.
- Broad regression453tests passed40.00s before registered router execution.

### Actual current-input router mutation receipt

- Preregistered a51dfb8817e8103a3eff915fde7212a9cd1a527f was pushed/fetched/
  exact remote checked before execution.57videos/16933tracks/73509observations
  tested in35.3209724s CPU-only, peakRSS847216640B. All57opaque-ID+same-frame
  input-order mutations retain identical current-input/weighted-mean digests.
- Actual laterfeature negation and extra futuretrack/observation change later
  digests while preserving firstvideo prior digest through frame1200. Actual
  data had0exact-tie groups; identical-prefix atomic path tested synthetically.
- Result SHA2565d0a1f80f35a5e870f3a5bb9289e719199d37dbf5e63169d17a87f56c838039e.
  No labels/GT decoded, learned model forward, new GPU/features/cache, optimizer,
  Val/Test or new metric. No claim of repaired historic controller, end-to-end
  identity-invariance, category/event/role purity or dense video frontend.
-15completed engineering diagnostics are not15scientific mechanisms. CVPR0/
  GoalsA/B/C remain unfinished. Disk~20GiB free; only small aggregate receipt
  added, CVPR~162MiB; no deletion or interference with foreign assets/processes.

### Assignment-driven hash source audit — no upstream run claimed

- Actual pinned source22c94f4d66380256022b264a81383cc7edf6a2e7 inspected for
  model/loss/train/eval/CUBsplit, with full versus partial read ranges recorded.
  MIT root/Apache2.0 backbone header read. Supplied Clarence repo is a fork.
  Official CVF paper successfully streamed via existing proxy after browser403;
  Code-link destination extraction failed, so officialrepo binding still false.
- Complete source tree lacks PHE.py and several imported dataset modules;
  config placeholders, FPA local-name shadowing, unresolved total-loss names,
  NumPy-to-torch.cat collection and FPA view/target ordering discrepancies
  recorded as static findings, not an executed upstream failure or a paper
  performance rebuttal. No source patched/copied/run or weights downloaded.
- Only geometry hypothesis can be considered later against lawful controls;
  sign-hash clusters and GT-Hungarian Known remapping are not persistent
  category-memory acceptance. Twelve-work/two-extra audit and methods pending.
- PDF URI diagnostic consumed a capped10MB prefix and ended curl23; no partial
  digest misreported as full file hash. Subsequent pdftohtml link attempt failed
  silently; no new filesystem PDF or dataset, no asset cleanup.

### Storage permission refresh and VC-CGCD static protocol audit

- User conditional cleanup authorization retained: only verified unused,
  recomputable owned project targets if needed; never active/unique/valid
  assets, historical negative evidence, other projects or external processes.
  Snapshot2026-10-11T20:42:56Z: available18283311104B, project outputs
  allocated3501510656B, CVPR169144320B. No deletion, reclaimed0B; no approved
  cleanup targets. Large launches require fresh10GiB reserve checks.
- Previous stage716263a4fcef9ea5a687cf5c3acfd099f5b608c7 verified through
  GitHub branch API and git fetch/ls-remote exact local SHA. Old17899 proxy
  absent; existing7890 worked with per-command proxy, no relay restart,
  credential/global-config change or foreign-process interference.
- Author-linked VC-CGCD source pinned6a4e73d966d9e311ec0dfab8d31bf3c232926ef7:
  actual model/loss/train/eval/prototype/dataset/license audit, no code run.
  Readable VC module missing (bytecode not substituted). Unlabeled entry
  passes true retained labels to VC loss and true novel counts to head size;
  test accuracy selects next-session initialization. Static risks, not a
  measured runtime failure or imported TrackOCD mechanism.
- Twelve-work/two-extra literature stage, atomic policy/memory integration,
  dense causal protocol/new folds and GoalsA/B/C remain pending. Training0.

### SlotContrast pinned base active-path review

- OfficialCVF2025venue and author-project-Code link independently verified;
  actual source55ec66dc02eeade630805789ef4a6c5df06f21ff model/loss/train/eval,
  configs, frozenencoder/initializer/recurrent processor/decoder read, with
  partial ranges explicit. MIT read; no upstream code/weights execution.
- Perframe mapped encoder plus forward recurrent scan is structurally causal,
  not changed-future certification. Same-instance slot positives/batch-slot
  negatives and pervideo-reset slots do not implement category persistence.
  Wholevideo ARI/mBO are not fixedKnown/CT. No new scientific metric or training.
- Source has custom attention scaling; no silent repair or reproduction claim.
 12work/two-extra review and exact release/stream qualification still pending.

### TRACT source availability boundary

- OfficialCVF ICCV2025pp14390-14398 explicitly links Nathan-Li123/TRACT;
  pinned19f01d72f9f6c212c28fd9cb0171a5432cd41a6a complete tree lacks every
  models directory, including imported TraCLIP model/data and MASA association.
  Actual entry/TrainValscripts/utilities read, missing model/loss not guessed.
- Only source paths read; no TAO assets, Novel class-name files, Test data,
  weights or caches opened/downloaded; no entry executed. Public historical
  Test path strings do not authorize executing them. License/release gaps held.
- Full-trajectory named classification broadcast is not anonymous causal
  category persistence. This partial availability audit is not counted toward
  complete twelve-model/loss requirement; continue other source reviews.

### EAGC actual SimGCD source audit — one additional work

- Independently verified officialCVF CVPR2026 venue and author/project/code
  binding; pinned87af6ff66be58a0fef929c343639443d596d533d actual SimGCD
  backbone/head/loss/gradient hooks/reference/Train/eval/CUB protocol read.
  No upstream code/weights/dataset/split pickle executed or downloaded.
- Actual masked Known supervision excludes unlabelled GT, but GT All accuracy
  selects checkpoints and default head uses GT class count. Record separately
  from online leakage; no runtime failure or TrackOCD import claimed.
- Active gradient-hook normalization, EMA/clamp, separate contrast/classifier
  branches and ReLU wrapper recorded. Whole-project license and exact release
  binding unestablished. Requirements with machine-local references not installed.
- Training-gradient hypothesis is distinct from our selected-checkpoint CE/risk
  evidence; source review is not a new method or CVPR1 twelve-work/two-extra PASS.
  Storage freshcheck16590487552B above10GiB reserve; no deletion, training0.

### AGE source review and additional-work qualification

- OfficialICLR2026 proceedings/acceptedPDF explicitly bind Ashengl/AGE,
  letterl; pinned d54c3147e61caf82a0de808333cebdc552fcf049 MIT fullyread.
  Actual model/loss/Train/held-out-support calibration/Gaussian-update/eval
  reviewed. No source/data/checkpoint/bytecode run or downloaded.
- Label-free query-update structural evidence distinguished from evaluator
  labels and separate oracle-K transductive KMeans comparator. No end-to-end
  mutation or fixedKnown/CT acceptance claimed. Source numeric/prior/window/
  scheduling/seed differences recorded, not silently patched.
- Earlier raw4line source-prefix reader ended curl23; prefix proves literal1
  followed by comment (valid expression), not a wholefile hash or syntaxfailure.
- AGE additional MIT audit; EAGC no whole-project license, so second licensed
  extra not certified. Full twelve-work audit/method choice/new folds and
  dense causal atomic policy-memory integration pending; training0, no cleanup.

### DebGCD additional MIT actual-source review

- AcceptedICLR2025 paper/project/Code chain independently binds Visual-AI
  repo9ec30fe3d54dfccd9934a862cf5cb9ff05de6f05; MIT fullread,13actual
  source/config/script/license files reviewed. Unused osr_utils fetched but
  not listed as read; no source/weights/data/pickle/environment execution.
- Paper task-consistency gate absent in active loss; actual fullhead eval still
  computes auxiliary branches. Distinguish release/default/shell/paper paths
  without silently repairing or asserting reproduced performance.
- TestOldACC mixedGT-Hungarian checkpoint choice/default oracleK cannot transfer.
  AGE and DebGCD are2additional MIT reviews; full twelve-work requirement and
  causal policy/memory/newfolds/method choices still pending, training0.
- Latest targeted70tests PASS5.20s (before documentation-only DebGCD additions).
  Storage15661084672B above10GiB reserve; no approved cleanup targets/deletions.

### COVTrack native source / DYDM primary-paper qualification

- COVTrack officialICCV2025 PDF binds repo9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b.
  RootApache2 read; actual Train/config/runner/model/fusion/loss/current-past
  tracker/format/eval audited with explicit wholefile vs selectedrange scope.
  Native vocabulary/split-mask/dynamicthreshold dependencies and retrospective
  track-majority category rewrite preclude anonymous strictcausal acceptance.
- Source fusion/grads/coordinates and runner/paper/config differences recorded,
  not patched or executed. GT observer condition distinguished from actual
  fusedfeature dependencies; source-only is not endtoend purity proof. Training
  basefilename alone is not legal effective-supervision/checkpoint certificate.
- DYDM officialCVPR2026 main/supp method/experiments/algorithm/implementation
  read; no verified authorcore source in scoped searches. Paper-only record,
  not counted as complete codeaudit or third licensedextra. Whole-stage fitting
  and RLS accumulation are not prospective WAIT/reversible category lifecycle.
- No upstream source copied/executed, no data/vocabulary/weights/testassets
  opened or downloaded, no newtraining, no foreignprocess changes/deletion.
  Latest available storage14529511424B (~13.53GiB) above10GiB safetyreserve;
  only small public audit docs written. Larger stages require fresh space check.
- Twelve complete source/paper audits, fresh independentfolds and atomic
  candidate-policy/memory integration remain pending; no scientific completion.

### MASA source and conditional storage permission follow-up

- MASA official CVPR2024 PDF binds author main
  c5472b9c7615f35abdf1188cb1a0c5408fe50d66; rootApache2 and17 complete
  relevant model/config/train/loss/dataset/eval/license files read, plus3
  installed mmdet3.3 dependency sources hashed. Source review is not exact
  local checkpoint/old release qualification or paper-number reproduction.
- Training config is static-image SAM pseudo-instance supervision; no new
  training or actual SA1B annotation/weight access. Current/past physical
  association is not semantic category discovery. Stored with_cats unused in
  matching; empty frames bypass TTL update; default evaluator category vote
  is retrospective. Score/schema/config differences recorded, not repaired.
- Latest disk snapshot12749856768B (~11.87GiB), only~1.87GiB above10GiB
  reserve. User permits unused-data cleanup if necessary. TrackOCD outputs
  stable; shared UV cache and other project roots are not disposable by
  inference. No exact substantial unused task-owned target verified, no data
  removed, no space reclaimed or foreign process touched. No large launch.
- Small source audits continue with active goal; twelve-work acceptance,
  dense atomic policy/category memory and fresh independent folds pending.

### OVTR native source review

- OfficialICLR2025 acceptedPDF binds authorrepo
  500e72c19bf5f7f8717546911a5639fdc26bfee5; rootMIT and16actual pathscopes
  reviewed, selectedrange/wholefile scope explicit. Public paper main and
  textualappendicesA-D read, not allfigures or exactrelease weights.
- CIP current/past physical-query state != acrossvideo anonymous category
  memory. Text-ranked proposals/CTIcontent/semanticKL mean OFA is not
  automatically anonymous. Default fulltrack category voting is retrospective.
- Actual configprefusion=false/contentmaskcommented/resumeguardtruthy-eval/
  alignmentdimension-sum differences retained as static observations only.
  Processedannotation needed for known-only effective-supervision proof;
  native parser stores but does not apply rareignoreIDs. No data/weights/
  category-list assets opened, upstream code executed or newtraining.

### Conditional storage cleanup: verified duplicate COCO downloads only

- Subsequent metadata inventory identified two original downloaded imageZIPs
  in the old task-owned PANDAS workspace, distinct from retained inputs.
  Fresh314.48s singleCPU check verified originalarchiveSHA against assets.json
  and every extracted118287Train/5000Val image size/CRC; aggregate imageSHA
  receipts retained. Reported ru_maxrss827360KiB, not a claimed256MiB cap.
  No COCOannotation/TAOTest labels or newGT read. Eight synthetic verifier
  tests passed; verifier itself never deletes data.
- Canonical ownedregular single-link paths and unchanged device/inode/size/
  mtime verified; fuser found no holders after checker finished, and scoped
  runtime source search found no archive references. Only exact train2017.zip
  and val2017.zip were unlinked; no recursive or other-project cleanup.
- Payload20152447128B removed. Available bytes12201275392→32353411072,
  observed increase20152135680B (~18.77GiB), not exclusive attribution under
  concurrent filesystemwrites. Extractedimages/annotations/weights/results/
  validshards/cache/history/sharedenvironments retained. ZIPs not locally
  undoable, re-download+originalSHA check required to restore archives.
  Receipt: outputs/trackocd_cvpr/coco_download_cleanup_20261012.json.
- Fresh large-run output/RAM/GPU checks still required; this is storage
  maintenance, not newtraining or any GoalA/B/C scientific acceptance.
- Current targeted83tests PASS2.99s, no skips; audit metadata and synthetic
  verifier checks are engineering tests, not upstream or research replication.

### VB-CGCD and HypCD pinned active-path audit

- AcceptedICML2025 VB-CGCD explicitly binds pinned
  7fe9501b425b5775588d0a06d60c817ee1f04663. Nine whole actual source files,
  main method/experiments/discussion and selected implementation/count appendices
  read. No root license in complete tree; no source copied or executed.
- Released incremental GMM labels are Hungarian-aligned using TrainGT before
  fitting. Second pass keeps incoming labels after predicted-novel filtering.
  Empty SVI guide/deterministic parameters/observed likelihood do not establish
  paper latent Bayesian KL. Whole-session fitting/oracle count and static
  source-paper differences retained; no dataset/weight assets opened.
- AcceptedCVPR2025 HypCD binds author main
  ffc895b7ff38be8502bb7e3bd851f1d2a55dff73. MIT and14 complete relevant
  HypSimGCD model/loss/train/data/eval scopes read with main paper. Actual
  direct768D map, fixed-K offline losses, disjoint Test-selected weights and
  globalHungarian Known remapping are not our persistent causal protocol.
- Audits are public aggregate/source hash metadata, not scientific reproduction
  or authorization for upstream imports. Newtraining remains0GPUhours.
  Dense atomic policy/category-memory, fresh Train folds and GoalsA/B/C pending.

### Twelve scoped actual-core inspection checkpoint

- Completed TALON main experiments/ablations/conclusion and textualA-C
  limitations review; retained TTAlearning-rate/separation-sign paper/source
  distinctions. LTC main experiments and textualA-E read; actual official
  CVPR2026 FindingsPDF firstpage verifies venue and repository, replacing the
  prior unverified claim. Exact accepted/arxiv equivalence not certified.
- Coverageindex now counts12 distinct pinned actual active model/loss/train/eval
  scopes with relevant paper review, including source/config limits. Three
  MIT extras AGE/DebGCD/HypCD satisfy source-inspection minimum2. Excludes
  missing-core AssignmentHash/VC/TRACT, paper-onlyDYDM and README-onlyBIAtree.
  Not12replications, lawful frontend imports, CVPR1 stage or Goal completion.
- Full85 engineering tests PASS2.95s before new coveragechecks; no GPU training,
  GT datasets/Novel vocab/TAOTest opened, dependencies installed or processes
  intervened. Source-review output small; protected recovered assets unchanged.
- With paper/coverage metadata checks included, full87tests PASS3.00s.
  No skips or scientific acceptance claimed.

### Atomic legacy scheduling adapter: small interface diagnostic preregistration

- New current-input adapter makes content-sorted visible-prefix decisions and
  memory writes transactional across an entire frame; exact indistinguishable
  groups make one decision/write and broadcast one result. Physical IDs remain
  external routing handles, not policy features. Twelve synthetic tests passed,
  including actual input mutations and rollback after a later group fails.
- This is a Raw Top1/17-cue engineering control, not learned A1/A2 parity,
  candidate-aware scoring or reversible category memory. It deliberately retains
  legacy repeated-prefix aggregate writes, a scientific limitation. Callback
  side effects outside the owned router/memory cannot be rolled back.
- Predeclared tiny real-input check: first completed manifest record video47,
  earliest32 distinct cached frame indices/all retained observations therein;
  three fixed diagnostic policies NEW, available-EXISTING/NEW and WAIT. Known
  prototype bank empty; no GT/category assets or actual checkpoint opened.
  ID renumbering/frame permutation and future negation/append compare actual
  17-cue/action/routing correspondence/memory digests. Exact remote registration
  required before execution, no previous output may be overwritten.
- One CPU worker, RSS2GiB/wall120s/output128KiB guards, at least25%RAM and10GiB
  free disk. Existing feature SHA/result/manifest validated; no new extraction,
  GPU inference/training or scientific metric. Old controller ID-invariance FAIL
  remains; this preregistration is not a run result or Goal acceptance.
- Full103 engineering tests PASS7.92s, no skips; four additional synthetic
  runner checks compare actual route/cue/action/state/future mutation digests.
  The real-cache diagnostic has not yet been run at this registration point.

### Tiny atomic-interface attempt1 rejected invalid future fixture

- Actual registration379735c launch/exit1/failure receipt retained under
  atomic_legacy_smoke_v1. Appending the exact negative last observation to a
  single-observation physical prefix cancelled the quality-weighted aggregate.
  Adapter rejected zero evidence; no successful real-cache certification claimed.
- Bounded fixture correction only: repeat the last observed vector on its
  existing route and append a new future route. No policy/threshold selection,
  label access or old-run overwrite. New output_v2 and exact remote registration
  required before retry. Added explicit synthetic antipodal-prefix rejection,
  unchanged state and valid same-frame retry test. This is not a method repair.
- Failure wall41.61s, no PASS; full104 engineering tests PASS3.09s before
  retry registration. Small launch/failure metadata published, not input rows.

### Atomic-interface retry: real retained inputs, not trained policy proof

- Remote033081c exact registration, preserved separate_v1 failure. Retry_v2
  completed46.559s/847216640B RSS, singleCPU and0GPUtraining/optimization.
  First record video47 uses earliest32of40 cacheframes and all1101observations.
  Three fixed policies each pass actual ID/frame permutation and future
  negation/append on 17-cue/action/routing/current-memory digests. Baselines
  NEW1101; NEW1/EXISTING1100; WAIT1101 are intentionally fixed, not scores.
- No real identical-prefix ties, Known assets, GT labels, Val/Test, new feature
  extraction or learned checkpoint. Synthetic tie/rollback checks stay separate.
  Old ID-dependent controller FAIL, repeated-prefix aggregate weighting and
  dense/learned/versioned-memory gates unchanged. Not Goal acceptance.
- Result65fe010fc91eb7bbc79c27ce99893b6bcb5743a3e1e395f0d37ce19f104c3c89;
  small launch/result metadata public, retained feature inputs unchanged.
- Manifest now18registered stages/16completed substeps (not16methods); actual
  failure receipts no longer get misclassified as running without supervisor
  metadata. Chronological main report now explicitly records actual two-ZIP
  cleanup, not its stale historical no-cleanup snapshot. Full109tests PASS2.99s
  before adding the focused manifest-failure classification regression.
- With that regression included, full110tests PASS2.96s, no skips. Synthetic
  unit checks, published receipt checks and real input run are reported separately.
