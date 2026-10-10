# Actual Train-only main heldout representation result

Status: actual completed GT-track controlled result, NOT predicted-track/Val
metrics or final scientific PASS.420causal replay cases,three learned seeds,
four orders,five prefixes.207Known/16pseudo-Novel tracks from the independent
T0 heldout_selection partition. Thresholds calibrated with equal25point budgets
on development only; no heldout threshold/model redesign. Exact registration
1a84c22 after the audited pre-metric partition-name correction.81.4758s CPU,
peak847216640B RSS. Features/checkpoints/private ledgers remain local.

## p16 first comparison

Percentages below are means of four order results per training seed,then means
across three seeds (raw has no training randomness). All prefixes/order/seed
results and actual seed std are in REPRESENTATION_ABLATION.csv and the JSON.

| Method | Old ACC | New ACC | H | Correct CT | False Merge |
|---|---:|---:|---:|---:|---:|
| Raw Frame+Vote | 9.18 | 76.56 | 16.38 | 59.13 | 11.90 |
| Raw Track Nearest | 0.97 | 53.13 | 1.90 | 39.68 | 0.00 |
| Raw Track DP-Means | 6.76 | 31.25 | 11.12 | 0.00 | 0.00 |
| Selected A1 + Nearest | 63.45 | 45.83 | 52.76 | 16.93 | 40.21 |
| A2 temporal + Nearest | 53.18 | 50.52 | 51.36 | 7.94 | 61.38 |
| Static capacity control + Nearest | 58.05 | 52.60 | 54.13 | 8.99 | 58.47 |

PHE remains INCOMPARABLE,not a missing score replaced by a fabricated baseline.
Raw nearest's calibrated high thresholds sacrifice Known recognition: its low
H is real,not evidence of raw visual features being uniformly inferior.

## What is and is not supported

Selected A1 improves overall category-macro cross-video Rank1 from49.335% to
64.463%,Known-versus-pseudo-Novel AUROC .807669→.852456,but spectral effective
rank falls54.176→8.586. This aggregate includes fitting-Known probe categories,
so it alone cannot prove retrieval improvement on unseen pseudo categories.
Role-separated diagnostics are required before the representation PASS verdict.

A1 all-pseudo-Novel wrongKnown rises0→28.125%;correct CT falls39.683→16.931%
relative to raw nearest and is lower than raw frame voting59.127%. Better H
mostly reflects substantially better Known recognition,not safe persistent
discovery. Loss convergence and engineering tests cannot establish success.

A2 p16 Rank1 .632726 is below A1 .644631;H .513555 below .527589,correct CT
.079365 below .169312 and false merge .613757 above .402116. The parameter-
matched static control has higher H .541325. These first results do not support
an independent temporal advantage. Retain TEMPORAL_EVIDENCE_NOT_SUPPORTED as
the provisional interpretation,with all-prefix/pair diagnostics still required.

Sixteen pseudo targets/four categories are sparse;three training seeds/four
orders do not manufacture independent categories. Conditional video bootstrap
must disclose dependence and hold mapping/stream state fixed. Do not change
the registered ongoing D1/D2 policy training in response to these results.

## Remaining work

Real predicted-memory D1/D2 training is running under the unchanged d270d37
registration. Policy heldout/coverage curves,full fair ablations,error correction
analysis,candidate freeze and MASA limited predicted evaluation remain pending.
Native MASA still has limited coverage/provenance and annotated-cadence limits;
M1 is not retroactively PASS. This stage is not the completed final goal.
