# CVPR-0 root-cause diagnostics — in progress

## Completed substep: retrospective frozen FULL ledgers

The corrected run verified all 52 frozen protected files and all60 selected
sealed executions, including exact Known-correct, contamination-state,
contamination-write and unmatched-write counts against historical evaluation.
It streamed18,273,660events with one CPU worker, no training/inference and no
GPU. Actual wall time75.66s; peak RSS847,216,640bytes. Aggregate artifacts:
`outputs/trackocd_cvpr/cvpr0/historical_audit_integer_normalized/`.
The first serialization-failed attempt remains preserved locally.

## Observed p16 action collapse, all four video orders

| Seed | KNOWN | NEW | EXISTING | WAIT | Old correct /4413 | New correct /819 | H / CT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1027 | 0 | 1 | 304560 | 0 | 0 | 14 | 0 / 0 |
| 1028 | 304561 | 0 | 0 | 0 | 994 | 0 | 0 / 0 |
| 1029 | 304561 | 0 | 0 | 0 | 927 | 0 | 0 / 0 |

These are individual-seed outcomes, not H recomputed from averaged Old/New.
Seed1027 creates its first state from an unmatched/unknown predicted track,
then all subsequent predictions reuse it. Main-order first matched-Known
anonymous write:sequence92; first provable matched cross-category conflict:
sequence1425;303135later EXISTING decisions reuse an already provably mixed
state. Sequence positions are zero-based. Other orders first show proven
conflict at799,897 and153. Unknown membership is not itself proven pollution.

This confirms propagation in retained traces, not why the policy collapsed,
nor what a first-error intervention would recover. All seeds use the previously
frozen WAIT bias-2.0, not a new Val-selected threshold. Total p16 commitment
coverage is1.0; pure CT remains0. Other prefixes are reported in the artifacts;
the table does not generalize the exact p16 counts to all prefixes.

## Claims and missing evidence

Observed: seed-dependent all-Known vs one-state-Existing collapse, early bad
writes and absorbing mixed-history reuse. Not yet proven: unique gradient
cause, candidate-limited ceiling, bias-vs-model contribution, geometry cause,
or recovery from self-correcting memory. Old ledgers lack logits/candidate
lists/gradients; no reconstructed values are mislabeled as original logs.

## Next registered Train-only diagnostic

Frozen Raw/A1/A2 ×D1/D2 ×three seeds ×two permitted Train partitions ×four
orders ×five caps:720actual replays. Also120FULL zero-bias diagnostics,90
explicit oracle counterfactuals (main only),18checkpoint gradient replays and
100geometry cases including matched-capacity static evidence. No optimizer,
new Val inference or final-holdout labels. Top-K counts use the same actual
prior prediction-owned history. Oracle interventions change their own later
history and remain upper bounds, not deployed-method scores. Selected-
checkpoint gradients are not historical training gradients; D1 hypothetical
cost gradients are not contributions to its original CE-only optimization.

Outstanding: memory/known-count controls, real equal-time ID-renumbering audit,
lawful matched Train predicted-track input comparison, verified literature/code
review and bounded new mechanisms. CVPR-0 and Goals A/B/C remain incomplete.

## Completed substep: frozen Train checkpoint diagnostic

Registration`6d6adbe` pushed/fetched/remotely verified before execution. Actual
930replays,100geometry cases and18read-only gradient cases completed in243.03s
CPU-only, peak RSS847216640B. Only the four permitted old Train partitions
were label-read with Parquet predicate filtering; heldout_selection/Val/Test
labels were not read. Private logits/cues/GT-derived diagnostic targets/ranks
remain in a14,930,132-byte local Parquet, not Git or live policy state.

### p16 main Train-development, FULL=A2+D2

At zero WAIT bias all258queries WAIT for allthree seeds. At the frozen-2bias,
seed1027 emits1NEW,113EXISTING,144WAIT; seeds1028/1029 emit258KNOWN.
Actual H/CT are0for allthree. Known correct prototypes are available for all
249Known queries. Top1correct:199/194/198; Top8correct:248/245/248. Thus candidate
ranking misses occur, but absence of correct Known candidates cannot explain
all-Known/WAIT/Existing action collapse on this controlled Train population.
On actual histories no pure cross-video correct anonymous state exists for
the9pseudo-Novel queries; widening that empty legal pool cannot create a
correct state. Candidate-only Top8oracle retains H0. This is **not** evidence
that TopK is useless with a healthy memory.

Action-only oracle with the actual Top1candidate mechanism obtains
H0.88839/0.83039/0.83943 and CT1.0/0.8/0.8 at total commitment coverage
0.80620/0.78295/0.79845. These runs use GT for oracle actions and change future
history: **diagnostic upper bounds, not method scores or coverage-matched
improvements**. Suppressing only the first bad write still gives H0/CT0;
seed1027 eventually makes another bad birth. One-shot repair is insufficient
for this checkpoint/population, not proof that reversible memory cannot work.

### Read-only gradients and geometry

At A2+D2 selected checkpoints, policy_train/main/p16/zero-bias nominal-CE vs
expected-cost gradient cosine is-0.98133/-0.99466/-0.99372. Delayed-credit
events and gradients are0on these specific all-WAIT checkpoint rollouts.
This establishes antagonism **at selected checkpoints**, not the original
training gradient trajectory or a unique optimization cause. D1 cost gradients
are separately marked hypothetical because D1 originally used CE alone.

Train-development p16 Raw uncentered effective rank55.55169; A1 ranks
6.91547/9.07242/6.91841; A2 ranks6.92528/9.01292/6.79464. Known macro Rank1
improves from Raw0.37459, but pseudo-Novel macro Rank1 falls from Raw0.25 to
A1(0.16667/0.08333/0.08333), A2(0.16667each). Static same-capacity pseudo score
is also0.16667each. The new geometry analysis separates centered spectra from
uncentered anisotropy; all spectra and corrections/new-error counts are kept.
Only9pseudo-Novel queries are in this inherited development fold: no general
Novel, independent novelty, or temporal superiority claim is warranted.

Evidence-supported next mechanism directions remain tentative pending code
review and a new frozen independent Train protocol: Raw-preserving geometry,
candidate-conditioned calibrated actions, and repeated no-GT reversible
memory validation rather than a single first-write suppression.

## Completed substep: memory/known-count and ID controls

Corrected registration`bcb1261` was pushed/fetched/remotely verified. All54
capacity/known-count cases,90learned-policy ID-renumber pairs and10fixed Raw-B1
pairs completed in45.63s, CPU-only, peak RSS847216640B. The first attempted
Known-subset seal correctly failed the original-denominator contract; its
partial outputs are preserved. The retry keeps the full15Known evaluator ID
registry separate from availability of4/8/15visual prototypes. All249Known
and9pseudo-Novel development queries remain in denominators.

Capacity16/64/unbounded never triggers eviction: these collapsed policies
create at most one anonymous state. All54cases have H0/CT0. This is a
non-binding-capacity observation, **not proof that memory budget is generally
irrelevant**. Known-count shift changes seed1027 development actions from
1NEW+257EXISTING (four prototypes) to1NEW+113EXISTING+144WAIT (15prototypes).
Other seeds remain all-Known on this fold.

### Actual physical-ID invariance FAIL in the old protocol

Only physical IDs were bijectively renumbered; evidence, frame times, video
orders, model parameters and labels were unchanged. All90learned pairs have
changed equal-time event ordering. Raw decisions/action-or-token strings change
in12/90pairs (16Raw,6A1,6A2individual output differences summed across cases),
and H changes in2/90. Raw token string differences alone could reflect benign
token permutations; the H changes establish more than that. In fixed Raw-B1,
H changes in7/10pairs. Old numeric-ID tie ordering is therefore a real
protocol dependence, not an end-to-end invariance PASS.

Preserve all old code/results. A new namespace must define a current-observation
ordering or simultaneous-event contract without numeric identity shortcuts,
then actually rerun ID/category/event/purity/future and role isolation tests.
This protocol repair is required regardless of whether a new model scores well.
Remaining CVPR-0 evidence: complete action-logit/error/memory traces and a
matched Train predicted-vs-GT physical input diagnostic; no phase-complete claim.

## Matched Train input preparation, physical smoke not yet run

All57previously frozen development videos have2065local Train annotation-cadence
images totaling248321596B. Metadata-only preflight byte-skips annotations and
categories, verifies all current image hashes and existing frozen DINO/MASA
checkpoint bytes, and binds a private590116B image-only plan by SHA256.
No new images, weights or environments are needed. Planned new physical and
feature payload ceiling1GiB; first physical smoke is only one40-image video
plus4changed-future forwards, separately source/config registered.

The [official MASA paper](https://arxiv.org/html/2406.04221v1) and
[pinned training documentation](https://github.com/siyuanliii/masa/blob/c5472b9c7615f35abdf1188cb1a0c5408fe50d66/docs/train.md)
support generic unlabeled-image association learning. This supports a limited
publication-level diagnostic route, **not an exact checkpoint-stage data/config
binding or a qualified primary frontend**. The old GT reference has258tracks/
2949capped observations; it is not full-track truth. Annotated cadence and
truncated GT must stay explicit in any upcoming matched-input comparison.

Registered physical smoke now completed:40canonical Train frames plus4actual
changed-future forwards,262physicalIDs/1422observations,2000ROI detections.
Worker57.69s, GPU3443523584B/RSS1618227200B peaks; terminal0. Hash comparisons
confirm the firsttwo frames unchanged after inverting later pixels; all
parameters remain frozen. Valid64,204B video shard is retained for explicit
reuse. No semantic metrics, strong-frontend qualification, full57video job or
training follows automatically from this limited PASS. Next work requires its
own bounded registration and continued provenance/reference limitations.

## Completed substep: retained Train logits and controller errors

Registration `43e38dee98a6f340c9e5649e144bed24043b1582` was pushed, fetched
and checked against the exact remote before execution. All175320 retained
rows from720 actual frozen-checkpoint cases yielded1440 role-specific groups
in7.82s, oneCPU, peakRSS847216640B. No model forward, optimizer, new Val/Test
or independent final-holdout access. Raw logits remain private; public
`retained_logit_forensics_v1/` contains aggregate distributions and error counts.
Unavailable action masks are counted separately, not averaged as -1e9 logits;
every retained action is checked against the float32 frozen WAIT-bias argmax.

Main/development/p16 A2+D2 seed1027: all199 correct exposed Known-top1 queries
receive non-Known actions;49 additional correct Known candidates rank2–8 and
are not exposed by the old top1 interface. Its105 Known-to-EXISTING writes
are wrong,103 target an already mixed candidate;8 pseudo-Novel-to-EXISTING
writes also target mixed candidates.114 Known WAITs and1 pseudo-Novel WAIT
disagree with the nominal current-top1 action target. This nominal diagnostic
does not prove waiting is causally unnecessary under long-horizon risk.
Seeds1028/1029 have55/51 wrong Known assignments among249 Known queries,
plus all9 pseudo-Novel queries incorrectly assigned Known. Allthree actual
histories lack a correct pure cross-video anonymous candidate for all9
pseudo-Novel queries. These are separate retrieval-pool, rank and controller
observations, not a claim that one mechanism alone explains FULL=0.

The old retained Parquet has no selected-token column or member trajectory.
We do not fabricate token histories from aggregate counts. Per-error-state
trajectories require a separate bounded unchanged-checkpoint replay. Wrong
commits at prefix≤2 are reported as short-prefix errors, not automatically
as proven premature commits; histories differ across old prefix replays.

## Completed substep: actual per-state Train trajectories

Exact remote registration8851dc0a98a5a612d059c8c5d877b6a772a898a7;
90main-order D2 cases and21915queries completed in32.68s CPU-only. Every
case matches the retained logits, actions, physical order, evaluator metrics
and mixed-state count exactly. All38 anonymous states created across these
independent cases end mixed; cases producing onlyKnown/WAIT have no states.
This is a finite old-protocol diagnostic, not38states in one shared memory.
The2,582,885B member/purity/token trace remains private, SHA256
`1cde037dda55328902e3dc6a533ef29a2e87c713130a76f7a6bae30741d6c78f`.

Development/main/p16 A2+D2 seed1027: state birth and first wrong anonymous
Known write at sequence144; first matched cross-category conflict146;
113later Existing writes,111after the state is already mixed. Final114members,
17categories,31videos, majority fraction0.41228070. This separates first
bad Known binding from later cross-category contamination. Pure fraction and
membership transitions are directly observed, not guessed from one final
metric. Exact cross-video reuse counts can be streamed from the private
video/token chronology; aggregate video_count alone is not that count.
Other two p16 seeds create no anonymous states. GT-only observation occurs
after the frozen visual prediction/update; it never repairs live memory.
No reversible-memory improvement, dense online, inherited-ID-invariance or
premature-commit causal claim follows from this replay.

Retained cross-video chronology registration833dd77dc559b97c0426c706b8cfa5538c4bed63
then verified all21915 prior/post snapshots and final38states, CPU0.47s.
For the same p16 development state,106of113 Existing writes had another
video already in that state's history; all106 occurred after both its first
bad write and proven mixture. Across38states from independent cases,
8166cross-video reuses,8128already-mixed cross-video reuses. These are actual
retained-state statistics, not a deployed repair, prospective purity, or
future-video evidence in the model.

The phase experiment_manifest now binds completed results/CSV hashes and
source receipts to actual archived preregistration Git bytes. The old asset
preflight lacks a commit field; its manifest binding is explicitly posthoc
to the logged pre-execution18e6c65 registration, not an invented launch receipt.
The Train physical followup remains RUNNING with no completed semantic metric.

## Completed bounded Train physical input, semantic comparison pending

Registration494615b654cbd7d2163276c318266c3c4efc1f24 completed57videos/
2065registered images in648.78s worker time,653.10s supervisor; own worker
244295 exited0, no foreign interference.2025new forwards plus the valid
40image smoke shard referenced under its original registration, no copy or
re-extraction.16933physicalIDs/87130observations/103042detections,
6311singletonIDs. GPUpeak3443523584B/RSS1628020736B; no optimizer/GT/text/
Val/Test input. Raw57video outputs and47,175B manifest remain local,
manifest SHA256`0bdae57cec0e1c21fe8602b39101deaca178e938116e3c4aa59c4ca5a97193a5`.

These predictions are not a complete GT census: the258track GT reference is
an inherited selected/capped subset. Excess predicted IDs are not declared
background or proven fragmentation solely from this count. Annotated cadence,
native50ROI cap and incomplete exact release-stage binding remain limitations;
no clean-primary, HOTA or semantic-method result is claimed.

Next bounded interface registers all262IDs from the reused firstvideo,1293
first≤16observations including86singletons, frozen common DINO. Current-image
crop batches only; no score/GT/short-track filtering. Real invisible-tail
mutations test all five caps, with short tracks retained. This is not a
57video feature launch or dense-online/semantic/ID-protocol PASS.

### Actual interface failure and CPU-only recovery

Registered43e4a28 encoder completed180batches and wrote the1,860,310B
all-ID feature payload, but final JSON serialization failed on a NumPyint64
tail-check counter. Own worker249997 exited1; supervisor100.6577463s.
The failed directory/log/launch/progress/payload remain untouched. Missing
numerical GPU/RSS peaks and the frozen-state hash were not persisted; we do
not fabricate them or call this a normal successful worker exit.

Separately registered682ad5a78cd564bfd86d0063a865cdec6620ae87 CPU recovery
completed5.78s without a new encoder forward. It verified the original
archived launch-source hashes, failure code path, payload SHA/bytes, all262IDs/
1293rows against current registered pixels/geometry/quality/image/frame
metadata, L2 norms and compact reader. All1310prefix checks pass;477actually
mutate a nonempty invisible tail. This is a recovered valid input, not a new
model/evaluation success. Its unchanged private payload SHA256 is
`70c496ad97967502ee78e655357b5370c09e715c1e1844364cabed48e9fbcddc`.
Full57video feature inventory is73509first≤16observations/16933IDs, raw visual
storage112909824B before metadata/compression; any extraction must explicitly
reuse the1293valid observations and still needs separate bounded integration
and registration. No automatic cache or training launch, no deletion.

## Completed tiny semantic/seal/evaluator integration, not matched full study

Registration6c7cca9afe7a583aa5acf3d36a0617c012ad087a:45frozen Raw/A1/A2×
threeD2seeds×fivecaps completed36.85s CPU-only; all262predictedIDs and7
GT-controlled reference inputs retained.90private Parquet ledgers roundtrip
exactly; all predicted ledgers seal before geometry matching/evaluation.
Class-free matching finds85observation pairs and3canonical GT-reference
targets;259unmatched predictedIDs remain unknown, not presumed background.
All7selected capped targets stay in every denominator. No Novel queries,
no cross-video opportunities; H/Novel/CT null/NOT_APPLICABLE, not perfect.
This validates wiring with fixed parameters, not discovery gains, primary
frontend quality, a full57video matched-input study, or uniquely causal
physical-input attribution. Old numeric-ID stream ordering is not repaired.
