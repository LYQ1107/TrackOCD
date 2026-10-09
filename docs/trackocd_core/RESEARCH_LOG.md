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
