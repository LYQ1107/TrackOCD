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
