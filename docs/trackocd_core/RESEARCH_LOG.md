# TrackOCD core research log

## 2026-10-10 — Exact v2 recovery (M0)

- Question: did the new server restore the actual later v2, including atomic
  formal shards / cross-track DINO batching / prefix-level resume, or only the
  older published source?
- Evidence: targeted fetch of NAS recovery `604d7eff`; all 81 NAS blob hashes
  equal Git tree and A100 bytes (642,256 bytes). NAS worktree HEAD `24907fe1`
  and index/source before-after digests unchanged. 50 files added, nine
  updated, 22 identical; main, original v2 and recovery stashes preserved.
- Migration failures: first synthetic run 43 passed / two failed on old-root
  assumptions. Keep recovered source bytes unchanged; isolate only the test
  environment in a separate conftest. Regression: 48 passed.
- Findings: actual batching/prefix resume and fixed GT denominator are back.
  Legacy Known Hungarian remapping and contaminated token scoring remain;
  M3 must implement a separate evaluator, not rewrite past results.
- Decision: retain exact source recovery. No algorithm hypothesis tested.
  Next: M1 lawful frontend quality/causality and source-metadata qualification.
  No new training, feature extraction, TAO Test access or foreign process action.

## 2026-10-10 — M1 frozen-reference quality and coverage

- Question: can a present frozen physical stream support Novel discovery?
- No model/inference/sweep: sequential read of all 988 existing BT-FG-0 NPZ
  track-ID columns. Counts match frozen runtime; canonical prediction SHA
  matches export. Median length 10, single-observation 10.19%, <=4 29.38%.
- Fixed category-free temporal-IoU >=.5 / per-video Hungarian coverage:
  Known 1442/4413 (32.68%), Novel 30/819 (3.66%). Full GT denominators, not
  matched-only accuracy. Counts agree with canonical export; complete NPZ
  box-byte equality was not established. Labels never enter method inputs.
- Source continuation: five Val-only NAS audit JSON files recovered exactly
  (260,702 bytes); historical run log loads [296,512] DetPro text features.
  Current selected vocabulary includes 203/209 TrackOCD Novel classes by
  synset mapping. Association toggle cannot remove detector dependence.
- Decision: retain PANDAS and COVTrack as references; no primary freeze.
  Native clean declarations are contradicted: FAIL_NO_NOVEL_VOCABULARY /
  INCOMPARABLE as core input. Cache byte validity is distinct from legality.
  Exact old config/prompt hashes and complete supervision remain unverified.
  M1 is BLOCKED_FRONTEND_QUALITY, not proof that every physical route fails.
- Next: small SimOWT source/config/fragmentation follow-up; meanwhile the
  expressly allowed GT-feasibility route can progress, not a predicted claim.

## 2026-10-10 — M2 permitted GT descriptor/interface smoke

- Question: can the existing checkpoint/runtime produce common 768-D visual
  inputs without future-tail/label leakage or an unnecessary dependency stack?
- Pin official DINOv2 source `7764ea0`; direct backbone construction and strict
  loading of the existing checkpoint, frozen/eval FP32. No duplicate weights,
  xFormers installation, text head or optimizer. Four Train Known GT tracks
  (Known 35/41), 64 observations; no Val/Novel/Test inputs to this smoke.
- Result: 10.78 s, 138,207 compact payload bytes; five file hashes and 20
  prefix views verify, stored FP16 means exactly agree. Singleton/batch feature
  max delta 5.76e-7. New PrefixView excludes labels/IDs/full mean/total length;
  future-tail perturbation tests pass. Regression: 57 passed.
- Decision: retain PASS_ENGINEERING_GT_ONLY. New source/preprocessing lineage
  is not proven identical to NAS features; no mixing, predicted main score or
  frame-online-memory claim. M1 qualified primary frontend remains blocked.
- Next: SimOWT provenance/fragmentation audit and legitimate frontend freeze;
  GT representation feasibility is allowed but not end-to-end completion.

## 2026-10-10 — M1 SimOWT/Q0 completed follow-up

- Question: is the historical ~650k stream a lawful, reliable primary asset?
- NAS source streaming scan: 649,378 tracklets / 1,853,369 observations,
  median length 1, single-observation 62.59%; current SHA matches normalized
  receipt. Saved-score recount covers all 5,232 GT keys: Known 1618/4413,
  Novel 179/819. Geometry was not rerun; persistent 83/527 is not Commit-CT.
- Actual static path: one-class detector, no Novel text vocabulary, current
  frame plus past memo; complete Val record is passed then discarded.
  Present weight matches official-asset manifest, but complete supervision
  history and dirty source/run-time binding remain unverified. This differs
  from COVTrack's positively evidenced vocabulary-assisted detection.
- Root-cause diagnostic: probability scores receive a second sigmoid;
  empty memo overwrites the shared score view with .7501. AST-pinned CPU
  check accepts 3/3 synthetic candidates instead of 1/3 at .2. Not a full
  stream score attribution, repair or threshold search. Regression: 62 passed.
- Decision: retain inspected candidate/reference; M1 still blocked, no
  primary freeze. Four private source files (36,132 B) match source SHA;
  only aggregate evidence/code/report are published. No large transfer,
  full inference/training, Test access or foreign-process interference.
- Next: independent hash-verified normalized-stream validation and lawful
  provenance, with GT-only evaluator/representation feasibility permitted.

## 2026-10-10 — M3 evaluator engineering preparation, not formal main stage

- Question: can scoring be independently correct before a GT-feasibility study?
- Full legacy audit found Known-state Hungarian relabeling and permissive
  contaminated token success. Preserve both files and every old result.
- New sealed ledger has no GT/Novel/match input. Posthoc full-GT join counts
  missing targets, exact Known IDs and one anonymous/Novel Hungarian only.
  Persistent scoring is chronological and separates pure reuse, wrong NEW,
  wrong Known, wrong merge, polluted, unknown, same-video and unresolved.
- All rates use fixed GT reuse opportunities; missing sources do not shrink
  eligibility. Unknown members cannot certify purity or prove a false merge.
  Token history is an absorbing linear-memory summary, not copied per step.
- Result: 110 tests passed, including 48 evaluator cases and 81-source byte
  identity. Twenty preconstructed toy order/prefix replays and missing-source
  CLI fixture pass (<76-MiB RSS); no real dataset, model or training opened.
- Decision: retain engineering preparation only. Actual primary geometry
  adapter/full-universe run and qualified M1 frontend remain pending. Toy
  perfect accuracy is not a learned algorithm or scientific result.
- NAS: user sent the read-only 294-MB native-row score-distribution request;
  its actual new turn completed. Source-side results are audited separately,
  not included as M3 synthetic or learned-method performance.

## 2026-10-10 — M1 actual SimOWT native-row score distribution

- Source-side read-only scan verifies the 294,735,737-byte native JSONL SHA
  and 1,853,369 observations / 988 videos, unchanged during read.
- Scores: 1,802,527 within [.5, sigmoid(1)] and 50,842 near .7501; zero
  below .5 or outside both tolerated regions. Degenerate boxes 16,290,
  all labels foreground 1. No new GT/annotation/Test needed.
- Result: artifact-level distribution matches the static operations;
  score quality is not raw calibrated foreground confidence. Does not prove
  complete historical source/weight lineage, lawful supervision, or that all
  fragmentation is caused by these operations. No silent repair of Q0.
- Resources: one bounded worker, 28.66 s, 18.45-MiB RSS, minimum available
  RAM 81.85%, enforced 256-MiB address-space budget. No data/weight/shard
  moved, external process changed, inference/training/cache started.
- Decision: preserve new aggregate evidence and original flow; 111 regression
  tests pass. M1 quality/provenance gate remains blocked, GT-only preparation
  remains authorized; not a new TAO detection/tracking/OCD experiment.

## 2026-10-10 — Preregistered bounded Train Known GT pilot

- Question: is cross-video category-level feasibility possible with actual
  legal observations rather than only synthetic evaluator fixtures?
- Train-only support inventory: 46 Known categories / 1,170 eligible >=16-ob
  tracks. Before reading descriptors/scores, fix 64 tracks / 1,024 observations:
  four adaptation classes, four policy pseudo-Novel, four heldout pseudo-Novel.
  Representation/policy/selection video sets are globally disjoint; Known
  probes intentionally share prototype classes. Original TAO roles unchanged.
- Decision: register bounded GT-only descriptor inference, four video orders,
  five prefixes and fixed comparable baseline gates; reuse compatible old
  smoke descriptors. No training, Val/Test or new physical frontend planned
  in this delivery. Four-Known support and long-track GT sampling are limited,
  not a formal M4 or end-to-end success. Regression: 115 tests pass.
- Next: commit/push the plan before running frozen DINO inference, then report
  the first actual small Train baseline results without tuning on heldout.

## 2026-10-10 — Bounded GT pilot descriptors completed

- Preregistration `8295d52` was pushed/remote-verified before frozen inference.
  Reuse 64 exact old smoke observations; infer only 960 new observations.
- Result: 64 GT tracks / 1,024 observations, 2,114,366 compact payload bytes,
  38.13 s, 543.62-MiB GPU / 1.24-GiB host peak; five payload hashes and 320
  prefix views checked. Same frozen encoder/crops, streaming batch four.
- Decision: retain small Train Known-only feasibility input. No learned
  model trained, no Val/Test or predicted-main score. Next: fixed simple
  baselines on the registered Train streams; do not change sampling/gates.

## 2026-10-10 — First actual small Train GT baseline table

- Question: do the fixed nearest/frame-vote/DP-Means baselines already provide
  reliable category-level cross-video reuse on the registered lawful pilot?
- Same frozen inputs / four Known prototype tracks; 120 real replays across
  two Train streams, four orders, five prefixes; 2,880 sealed decisions.
  Each full universe is Known 12 / pseudo-Novel 12 / GT reuse opportunities 8.
- Result: heldout p16 Correct Commit-CT = 0 for all methods; New ACC means
  39.58/39.58/43.75%, Old ACC 0/2.08/0%, H 0/3.47/0%. Error breakdown and
  all prefixes/orders retained. No stable benefit from increasing prefix.
- Decision: preserve weak results, do not retune on heldout or silently alter
  gates. Single-track prototypes/fixed operating point limit conclusions;
  this is not strongest-full-M4 or M9, nor evidence that all nearest routes
  fail. B3 remains INCOMPARABLE; no PHE/model score fabricated.
- Resources: one CPU worker, 1.445 s, 677.14-MiB peak, no learning/GPU,
  Val/Test or foreign process changes. Regression: 119 tests pass. Next:
  bounded A0/A1/A2 Train-only GT feasibility with category-held-out fitting.

## 2026-10-10 — Bounded A0/A1/A2 feasibility fit preregistration

- Only 12 representation-fit Known GT tracks / 192 observations may enter
  learning; prototype/policy/heldout tracks do not. Encoder/physical frontend
  remain frozen. Three seeds, 120 steps/model/seed, 600-s total bound.
- A1 is 525,056-param adapter/mean; A2 adds 16,833-param causal evidence and
  synthetic reliability supervision. Identical inputs/adapter initialization
  per seed, category-level cross-video positives, no genuine Novel/Val/Test.
- Keep original baseline gates; disclose post-adaptation calibration and
  auxiliary-loss/capacity confounds. Save all fixed-step results, no tuning or
  correction round. Small GT feasibility is not formal M5/M8/M9/M11 PASS.
- Baseline CSV formatting-only LF normalization leaves numeric cells/raw
  sealed decisions unchanged; original run-source hashes are preserved.

## 2026-10-10 — Actual A0/A1/A2 heldout Train GT negative finding

- Before training, push/remote-verify `b696190`. Six fixed 120-step fits use
  only 12 legal representation tracks, same-seed input corruption. 13.21 s,
  28.70-MiB GPU / 1.04-GiB host peak, 12,822,732 private checkpoint bytes.
- Frozen evaluation: 280 real replays / 6,720 sealed decisions, 9.46 s CPU;
  A0 exactly reproduces original B1 for all 40 cases. Regression: 123 pass.
- Loss falls ~2.4→.7, but heldout p16 A1/A2 Known 77.78% seed mean (7.86pp
  seed std), pseudo-Novel New/H/CT all zero; wrong Known 100% of reuse
  opportunities. A1/A2 decision aggregates equal on every seed/prefix.
- Decision: no independent temporal/discovery gain supported. Closed-world
  improvement/100% commit coverage is not discovery success. Keep every
  checkpoint, prefix/order result and negative finding; no hidden correction.
  Calibration/four-class fit are unisolated possible causes, not proven.
- Next: policy_train-only score separation diagnosis, no new fit or parameter
  search on heldout/Val Novel. One correction cap remains unused; no M11,
  qualified frontend, main M5/M8/M9 or end-to-end PASS. Goal remains active.

## 2026-10-10 — Policy-Train-only open-set geometry diagnosis

- Frozen model/prototype score distributions only, 35 cases, 3.80 s CPU;
  no heldout features, new fit, threshold search, Val/Test or memory repair.
- Policy p16 max-Known-score AUROC drops .9792 raw→.7153–.8681 adapted;
  rank-based deterioration cannot be fixed by a single monotone threshold.
  Unknown same/different cosine gap compresses .2708→.017–.046; report
  distributions, not an uncalibrated cross-space statistical superiority claim.
- Decision: investigate preservation of frozen visual open-set geometry,
  not hide failure with a gate change or larger network. Root cause is not
  uniquely proved; one correction remains unused and must be preregistered
  using fit-only teacher geometry, same data/seeds/steps/capacity.

## 2026-10-10 — Single controlled correction R1 preregistration

- Root evidence is policy_train-only score compression and lower rank-based
  open-set discrimination, not heldout/Val tuning. Test weight-5 fit-only
  frozen-DINO cosine-Gram preservation in both A1/A2 losses.
- No changes to samples, capacity, initialization seeds, 120-step budget,
  corruption, prototype supervision, .65/.55 gates, orders or prefixes.
  New R1 directory/checkpoints/results; preserve all original negative data.
- This consumes the one root-cause correction round when started. No second
  loss/architecture/threshold retry; no M11 or primary/frontend PASS implied.

## 2026-10-10 — R1 completed; no stable temporal gain, no further correction

- Push/remote-verify `63a3ad0` before six unchanged-budget R1 fits. 13.55 s,
  28.73-MiB GPU / ~1.06-GiB host; 12,822,732 new private checkpoint bytes.
  Frozen evaluation: 280 actual replays / 6,720 decisions, 9.41 s CPU;
  A0 exactly reproduces original B1. All original artifacts preserved.
- Heldout p16 seed/order means: A1/A2 Old 79.17%, New 9.03%, H 14.89%,
  CT 4.17%, wrong Known 85.42%. Partial recovery, not reliable discovery.
  Paired A2-A1 p16 all zero; one p1 seed improvement and p8 deterioration
  do not establish stable temporal contribution. Keep full prefix/seed table.
- Decision: retain A1 candidate, no A2/M11 or main scientific PASS. One
  correction round used; do not launch R2 or silently calibrate/resample.
  Primary frontend/provenance remains unresolved. GT decision learning is a
  separate scope from the authorized representation exception; clarify before
  treating it as a substitute for strict main-stage progression.

## 2026-10-10 — Official SimOWT training source trace, not release qualification

- Main priority returns to M1. Complete official tree, ten selected source/config
  files (169,655 B), all exact Git blob and SHA256 identities verified. No new
  training/inference/environment, annotation/GT/Test access or NAS job.
- COCO-named train mapper can route to TAO Train. Official Known-filter auxiliary
  set matches inherited 78, but its proposal/merge/release linkage is missing;
  separate extra-box generator saves original unfiltered annotations. Neither
  source possibility establishes actual release supervision or proves leakage.
- Actual NAS current idol/entrypoint/registry differ from pinned upstream despite
  equal HEAD; YAML bytes match. Preserve this gap, no pretend patch recovery.
- Decision: BLOCKED_PROVENANCE_NOT_PROVEN_LEAKAGE; no primary freeze, Q0 unchanged.
  Need stage/teacher/merged-file ledger and runtime binding before qualification.
  No extra GT correction or unapproved GT policy extension.
- Source-only scan 0.0189 s / 18.38-MiB peak; 131 tests pass. Initial new source
  equality expectation failed and was replaced with explicit observed mismatch,
  not a source/model alteration. Full details: M1_SIMOWT_TRAINING_CHAIN.md.

## 2026-10-10 — MASA and AED low-cost alternative preflight

- M1 permits existing/low-cost MASA/AED candidates. Fifteen pinned official
  source/config/docs files, 96,708 B, exact Git blob/SHA256 verified; one CPU
  worker, 0.0066 s, 17,864-KiB peak. No data/model execution or environment edit.
- MASA-SAM generic image/segment supervision motivates a candidate, not release
  certification. Default TAO config uses external Detic detections; disabling
  that flag alone does not supply the missing SamMasa detector prediction API.
  Native multi-level SAM -> released RPN/ROI -> anonymous past-only tracking
  requires an explicit new bounded wrapper and actual checkpoint key checks.
- Demo uses future-dependent centered boxes/full-segment scores/whole-track
  filtering; bypass all offline postprocessing. Formula witness changes an
  earlier coordinate 0->2 by changing only future coordinate index 2.
- Official LFS metadata pins optional weights totaling 933,925,642 B, not
  downloaded. Existing torch2.6 runtime lacks mmcv/mmdet/mmengine; official
  torch2.1 cu118 cp310 MMCV binary is listed, torch2.6 index returns 404;
  installed CUDA compiler is 13.2. Do not compile or alter base environment.
- Conditional plan: resolve exact minimal binary wheels and inspect safe weight
  tensor coverage first; <=8 Train images/one video, 600 seconds, <=8-GiB new
  total, fixed proposal settings, no training/text/GT/future input/full job.
  A smoke PASS would not qualify M1 or reproduce public-Detic benchmark scores.
- AED frequency-base filter is not an explicit 78-Known whitelist; actual
  release overlap/binding unverified, no prohibited-supervision claim or pack
  download. Original GT exception wording narrowed to representation only;
  GT policy authority still pending. Root correction remains consumed.
- Regression 137 pass; M1_MASA_AED_PREFLIGHT.md and small public source/asset
  receipts preserve limitations and next action. Primary gate remains blocked.

## 2026-10-10 — Actual frozen MASA checkpoint and binary dependency budget

- Previous preflight delivered and exact remote HEAD `21447b5` verified before
  asset recovery. Restore only the first pinned 558,882,875-B MASA-SAM-B weight;
  approximately 117-s transfer, exact LFS SHA256. No other model/data transfer.
- Safe CPU/mmap `weights_only` inspection: 419 tensors, all required component
  prefixes, one-class ROI dimensions and 12 backbone blocks. No unsafe fallback,
  extra globals, model construction, metadata/class-name input, numeric/GT/Test
  execution or complete supervision/runtime qualification. 3.23 s / 347.03 MiB.
- Second 375-MB pretrain asset deferred until strict frozen model matching;
  prefix presence alone is insufficient to certify every required parameter.
- Safe metadata follow-up: zero metadata keys, no saved training config;
  official generic-image training description is not a checkpoint-saved stage
  ledger. Keep this qualification limit; do not use COCO metadata fallback.
- Exact minimal binary-wheel lock resolves 48 packages. Complete ZIP-directory
  ranges total 4,407,901 B: unpacked allocation 5,378,584,576 B; resident budget
  with reserves/two optional weights 7,673,439,806 B <8 GiB. 61.11 s /27.62 MiB.
  No full large-wheel download, installation, compile or existing-env change.
- Initial instrument probes rejected small suffix ranges/direct archive lock
  entries; guarded handling added. Three PyTorch URLs gave Python-client 403;
  bounded official curl transport works. An incomplete budget was never treated
  as PASS. Initial incompatible uv switches were rejected before resolution.
- Measured budget does not establish installer transient peak or compatibility.
  Next: monitored isolated single-package/hardlink install, minimal image-module
  registration, strict key matching, then the fixed <=8 Train-image smoke.
  Do not call the default semantic metadata fallback or expand to a full job.
- Named new payload now 1,090,170,303 B plus small sources/receipts. No research
  training, inference, R2, GT policy extension, Test or primary frontend freeze.
- Regression **150 pass**; exact legacy v2 identity remains covered. Delivery
  contains only small code/lock/aggregate evidence, never checkpoint/wheel bytes.

## 2026-10-10 — Frozen MASA native runtime and eight-forward causal smoke

- Exact previous remote delivery `c04f51a` retained. Install 48 hash-locked binary
  wheels in a separate owned runtime; base distribution snapshot unchanged.
  RECORD checks: 20,423 files / 5,325,373,594 logical bytes, zero mismatch.
  Measured install peak 5,945,593,856 B, below the registered 8-GiB ceiling.
- Preserve initial verifier failure: Torch itself installed successfully with
  require-hashes, but uv omitted optional direct_url archive hash. Resume uses
  owned success/control-flow/URL/requirement proof plus streamed RECORD checks;
  no second Torch download, unsafe fallback, base downgrade or source compile.
- Recover 14 exact native image-module/config/license files, 125,292 B. Explicit
  frozen SAM multilevel -> adapter -> released one-class RPN/ROI -> past tracker;
  no MASA dataset/semantic initializer, metadata fallback or offline postprocessor.
  Strict 419 model keys/shapes match, all tensors finite/parameters frozen;
  no second pretrained weight required. Network/external children denied.
- Preserve pre-image mmap failure (torch2.1 requires string, not PathLike);
  safe string-filename correction, no extra globals or unsafe checkpoint loader.
  It consumed zero real-image forwards and no representation correction round.
- CPU binary ops, synthetic preprocessing and empty-proposal fixtures pass.
  Four existing Train images, two chronological replays = eight forwards total.
  Existing long-Known-GT selection bias and ordinal-time subsampling disclosed;
  no GT/semantic metadata passed to model. Future last-two pixels inverted;
  first-two detection/track arrays exactly identical. Bounded engineering PASS.
- Every ROI frame hits inherited 50-proposal cap. Original tracks 43/37/40/42;
  changed-future 43/37/40/40, finite nondegenerate anonymous outputs only.
  Not coverage/purity/physical metrics, not an official public-Detic reproduction,
  GT scientific temporal benefit or complete M1 qualification.
- Supervised smoke 16.88 s, worker 15.31 s; peak host 1,687,347,200 B,
  GPU reserved 3,439,329,280 B. Retained owned candidate allocated size
  5,945,954,304 B; prior other named logical payload 531,287,428 B.
  Mixed subtotal is not an exhaustive filesystem inventory. No foreign process
  touched, Train model optimization, Test/Val job, full cache, R2 or primary freeze.
- Regression **177 pass**. Small code/config/receipts/report delivered only.
  M1 remains BLOCKED_FRONTEND_QUALITY; complete supervision binding and a new
  explicit bounded physical quality/coverage/pollution preregistration come next.
  Completed smoke is preserved and cannot be automatically rerun for tuning.

## 2026-10-10 — Published training trace and physical-diagnostic preregistration

- Revalidated exact clean `7fb6fef` delivery. Previous goal turn is real progress:
  completed isolated install and actual frozen/causal eight-forward smoke.
- Six pinned small sources/docs, 32,848 B (four new, 25,116 B): SA-1B generic
  leaf, converter category1 and generic dataset metadata verified by AST only.
  Published model-zoo link/no-in-domain assertion verified. No SAM-B train
  configuration in complete tree; documented converter filename absent, supplied
  training model GroundingDINO. Exact SAM release-stage binding still missing;
  no forbidden-supervision assertion, upstream execution or data/weight download.
- Current Val GT and historical PANDAS canonical GT symlink have identical
  43,639,943 bytes/SHA256. Preserve canonical TAO_OW partial-annotation rules;
  don't substitute ordinary box-HOTA or remove unmatched outputs ourselves.
- Separate fixed M1 diagnostic: first ascending Val video IDs4/20/22/23, each
  first16 image metadata rows;64 forwards max, no GT-driven sample selection.
  Prediction/GT evaluator separated; compact outputs sealed first. Native
  model/weights unchanged, no threshold search or repeated completed smoke.
- Raw unknown observations are not proven false positives/background; track
  category/identity mixing is posthoc geometric evidence, not semantic memory
  input or universal purity. All selected GT targets and zero denominators kept.
- PANDAS same-image projection retains different dense-frame history/cadence;
  not a fair same-input association ablation or formal OCD comparison.
- Unchanged canonical adapter's fresh-process synthetic perfect fixture gives
  HOTA/AssA/DetA/DetRe=1; no real Val prediction/GT match has run at preregistration.
- Regression185 pass. Preregister and verify remote before bounded inference;
  no full-Val/cache, training, Test, primary freeze or M9 score from this pilot.

## 2026-10-10 — Actual fixed Val4 physical diagnostic (not main qualification)

- Remote-verified preregistration1d0f3a1 executed once;4videos x16images =64
  frozen native forwards, compact stream sealed before independent evaluation.
  No label-based resampling, threshold/cap change, training or completed-smoke rerun.
- Same328 GT rows and canonical TAO_OW/HOTA source for both routes. Combined
  native HOTA.169655/AssA.502096/DetA.059033/DetRe.531611; PANDAS frozen projection
  .122697/.372579/.041192/.583440. Actual per-video and alpha arrays retained.
  Not full-Val values; different native ordinal vs PANDAS dense-frame history
  prevents matched-input association/contribution claims.
- Coverage fixed clip universe: Known native8/26 vsPANDAS10/26; Novel0/2 forboth.
  Native everyROIframe cap50;495tracks/2843obs, median4,single119/495=24.04%.
  PANDAS annotated projection2807tracks/4543obs, median1,single71.71%.
- No observed different-category mix, but native2659/2843 rows unknown after
  geometry match,456tracks without GT support,477 with unknown observations.
  This cannot establish purity. Six tracks touch multiple individuals, not
  automatically multiple semantic categories. No unmatched outputs discarded;
  canonical preprocessing removeszero selectedrows. Small NovelN2 is not
  full-Val statistical support or an opportunity to select favorable new clips.
- Nativeworker27.74s/supervisor29.74s; host1669230592B/GPUreserved3441426432B.
  CPUeval5.60s/847216640B; fourprivateNPZ118457B, allboundedoutput2723840allocatedB;
  candidatewithretainedattempts5948731392allocatedB <8GiB. No foreigninterference,
  extraweight, semanticGTinput, training, Test/fullVal/cache or primaryfreeze.
- Result187tests pass. Publish counts/hash/resource/metric receipts only,
  retain rawboxes and clippedGT privately. M1BLOCKED_FRONTEND_QUALITY persists;
  no M9/semantic scientific PASS, no second correction or goal completion.
