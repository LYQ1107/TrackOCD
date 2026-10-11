# Root-supported mechanism agenda — hypotheses, not scientific results

Status: planning after twelve scoped paper/actual-core inspections and three
licensed-extra inspections. No new training, fresh fold acceptance, positive
semantic result or CVPR contribution claimed. Historical diagnostics remain
separate from future method experiments. Author-source inspection does not
authorize importing an unlicensed release or using its dataset/weights.

## Three distinct falsifiable mechanisms

| Direction | Actual root evidence, with limitations | Hypothesis and fair controls |
| --- | --- | --- |
| R: Raw-preserving temporal representation | Old dev Raw effective rank55.55 vs A1/A2 roughly6.8–9; pseudo Rank1 Raw0.25 vs adapted0.083–0.167 on only9queries. Future shadow sometimes corrects errors but also creates369new Known errors across repeated cases. Not a general Novel result or proof that longer input alone fixes failure. | Preserve frozen Raw geometry with an identity-start residual; learn cross-video semantic positives and causal prefix reliability. Compare Raw, fresh-fold A1/A2, same-capacity static, residual only, residual+open-set, residual+temporal and full. Static and temporal branches must match capacity and train data. |
| C: Candidate-conditioned calibrated decision | Seed1027 rejects199correct exposed Known Top1s;49more correct candidates rank2–8. No healthy pure cross-video anonymous candidate in the actual old histories; TopK cannot invent one. | Score each legal Known/anonymous candidate using its own Raw/adapt agreement, margin, support, dispersion, reliability and current-prefix uncertainty. Separately test candidate expansion vs scoring using Top1old-style MLP, TopKfixed cosine, calibrated cosine, same-capacity TopKMLP and conditioned scorer. |
| M: Versioned contradiction-aware category memory | Old dev seed1027 first bad anonymous Known binding144, proven mixture146;106true cross-video Existing reuses all after mixture. Single bad-write suppression still fails. All38states across independent old cases end mixed. | Separate tentative/stable/contradictory evidence, quarantine unsafe members and prospectively revise state from current/past visual contradictions. Compare Mean, EMA, TopKsimple, tentative only, quarantine only, contradiction only, revision without learned reliability, full and same-capacity static. Retain actual wrong histories and match both Novel-reuse and all-commit coverage. |

These are three different interventions (geometry, candidate decision, memory
lifecycle), not three parameter settings of one method. None is yet explored
scientifically; implementation/unit tests alone will not satisfy the user's
minimum-three-mechanism research requirement.

## What transfers from inspected work, and what does not

- TALON/LTC suggest current/past prototype adaptation and separation/density
  hypotheses. Native numeric rows, source/paper loss signs, batch scheduling,
  Known Hungarian remapping and absent rollback prohibit calling them a clean
  persistent-category controller. No unlicensed code copied.
- Grounded Correspondence/SlotContrast motivate causal temporal consistency.
  Their slots identify physical instances, not semantic categories across
  videos. Same-category different-instance negatives must not be inherited.
- EAGC motivates measuring actual new-training objective-gradient interaction,
  not relabeling the old selected-checkpoint CE/risk cosine as historical EAGC
  evidence. DebGCD motivates open-set separation; its fixed-K/test-selected
  native protocol is not copied. HypCD is a bounded geometry alternative only
  if Euclidean Raw-preserving mechanisms fail with a diagnosed geometric cause.
- AGE motivates past-only uncertainty/state summaries, not unbounded births or
  retroactive rollback. VB-CGCD motivates covariance/sample-size diagnostics,
  not GT-Hungarian incremental fitting or declaring an empty SVI guide Bayesian.
- COVTrack/MASA/OVTR are physical frontend sources, not semantic persistent
  memory. Vocabulary/release-stage/retrospective voting limits must be resolved
  separately. Completed MASA/DINO limited Train shards remain reusable diagnostic
  assets, not a qualified primary frontend or dense online certificate.

Evidence/source ranges and licences are in the existing CVPR1 audit files and
[literature/gap analysis](CVPR1_LITERATURE_AND_GAP_ANALYSIS.md); diagnostic
counts and their noncausal/oracle boundaries are in
[root-cause report](CVPR0_ROOT_CAUSE_REPORT.md). The transfer hypotheses above
are inferences from those sources, not claims made by their authors.

## Representation/decision implementation contracts

R must start from frozen legally bound Raw DINO, not the old fitted adapter.
Start identity-preserving; expose Raw alongside adapted evidence. Fit labels
only supervise loss outside model-facing inputs. Positive pairs cross videos
but share legal Train category; temporal same-track consistency is not a
substitute for category generalization. Keep singleton, short, fragmented,
poor-localization and unmatched predicted inputs; unknown is not background.

Finite candidate K is1/3/5/8. Candidate IDs/tokens are output bindings, never
score features. Test candidate-list permutation and category-ID renaming with
actual logits/output/state changes. Exact indistinguishable candidate ties
need an explicit equivariant ambiguity/WAIT contract, not numeric-ID sorting.
No absent-candidate labels, future length, oracle purity, semantic names,
evaluator role or physical-ID arithmetic enter scorer features.

Training must log actual CE/risk/delayed-credit gradients as it happens.
Observe conflicts before adding a finite justified gradient treatment; do not
attribute all failure to optimization or blindly prolong training epochs.

## Memory/evaluator contract before a learned repair can run

- Bounded past-only member provenance is outside scorer features. Re-observing
  a physical prefix replaces its current support rather than counting that
  whole prefix again; test exact-input group aliasing followed by divergence.
  The new atomic legacy control deliberately does NOT implement this correction.
- Minimum lifecycle Tentative/Confirmed/Quarantined; revision yields an append-
  only version/lineage record. Past sealed predictions never change. Stable
  category identity across videos is not a recycled physical-track identifier.
- No GT can trigger quarantine, revision, rollback or state reset. Learned
  reliability and contradiction operate only on current/past visible evidence.
  Teacher-forced GT history is a separately labelled training upper bound.
- Register strict lineage-history purity as well as current active-version
  purity. Never wash historical pollution by pruning members, resetting tokens
  or ignoring old wrong commits. Freeze how split/merge/version lineage affects
  CT/pureCT and fragmentation BEFORE model selection; this gate is still open.
- Both Novel-reuse coverage and all-commit coverage must match comparisons;
  keep full fixed denominators, latency/pending/no-commit counts and runtime.
  A better purity value obtained only by waiting more is not repair evidence.

## Fresh Train protocol and assets: observed feasibility, not a frozen split

Registered f832f4c inventory filters only four previously authorized Train
partitions; old final-heldout labels stay unread. Eligible old fit pool has
15classes/1867selected cappedGT tracks/21162observations/421videos,92singletons.
All15classes have at least5reference videos,13have at least7. These are NOT
full annotation counts or predicted-track denominators.

Existing57video predicted-cache reference intersection has only10classes with
at least2videos,7with3,3with5,2with7. This exposes insufficient *cached coverage*
for a balanced broad fresh protocol, NOT proof that corresponding pixels are
missing. Next safe step: metadata-only Train pixel availability inventory;
annotations/categories byte-skipped, no pixel decode or new inference. Reuse
valid shards if a subsequently registered fold needs their videos.

Old A1/A2 supervised all15eligible classes. Reusing them on a new holdout would
not be an unseen-class baseline. Retrain same-capacity A1/A2 within each fresh
fold, with no old fitted adapter or policy initialization. Raw pretrained
encoder lineage and legal supervision remain explicit.

Before any optimization: freeze category/video-disjoint fit, development and
independent pseudo-Novel sets; require cross-video positives for each evaluation
class; report support/queries, all exclusions and co-occurrence conflicts.
Class/video choice uses predeclared seed/metadata feasibility, never old model
scores or difficulty. Known probes need a separate legal Train prototype bank;
do not fill missing prototypes from Val GT. Existing selected-reference labels
alone cannot certify full video category absence; lawful census/exclusion
checks and pixel/cadence coverage remain prerequisites, not assumed solved.

## Bounded execution and rejection rules

Initial allocation ceilings: R120GPUh, C120GPUh, M120GPUh, frozen combined
validation180GPUh, justified contingency60GPUh; cumulative maximum600 and
concurrent maximum4GPUs. These are ceilings, not launches or spent budget.
First training smoke and one-seed/one-fold pilot need separate exact remote
registration, fresh GPU/RAM/disk checks and frozen protocol before execution.

At most two evidence-driven repairs per direction before retaining the negative
result and moving to the next distinct hypothesis. No infinite K/loss/threshold
search. Freeze finite loss weights/steps, seeds and orders before each execution;
at least three independent training seeds in final comparisons. Repeated video
orders on one model are not independent trained replicates.

R must improve heldout-class retrieval/open-set geometry without Known collapse
and beat matched static temporal controls. C must improve calibrated decisions
at matched candidate pool and coverage, not merely gain from K expansion. M
must reduce actual error propagation and improve genuine reuse/pureCT at matched
coverage/latency without historical purity laundering. Report failures and
missing opportunities, not only favorable denominators. GoalB uses the same
frozen physical frontend/full GT denominators after method lock; historical
repeatedly observed Val cannot select models. GoalA/B/C remain unachieved.
