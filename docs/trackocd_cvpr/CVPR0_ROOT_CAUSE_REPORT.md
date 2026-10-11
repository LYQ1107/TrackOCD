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
