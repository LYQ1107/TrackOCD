# Actual small Train Known GT baselines — not M4/M9 main completion

Source preregistration `8295d52`, compact input delivery `1fd79b7`. This is
the explicitly permitted GT-feasibility branch while M1 remains blocked.
Actual frozen descriptors, not preconstructed synthetic correct decisions.
Original 78/209/45 roles stay unchanged; simulation unknowns are Train Known.

Three methods, two Train partitions, four fixed video orders, all five
prefixes: **120 completed replays / 2,880 track decisions**. Every replay has
12 Known GT tracks, 12 pseudo-Novel GT tracks, and eight fixed cross-video
reuse opportunities. Perfect GT physical coverage is not predicted coverage.
No learned model was trained, no threshold searched, no Val/Test read.

## Heldout-selection scores

Percentages are four-order means, not independent datasets. Full per-order
scores/counts and population std (`ddof=0`) are in the public JSON/CSV receipt;
the selection set is not an independent final test. Do not select one prefix
after seeing results. B0 is **snapshot frame-nearest voting**, not a verified
continuous frame-online controller.

| Method | Prefix | Old ACC | New ACC | H | Correct Commit-CT | False merge | False NEW/split |
|---|---|---:|---:|---:|---:|---:|---:|
| B0 frame snapshot vote | p1 | 0.00 | 41.67 | 0.00 | 12.50 | 0.00 | 78.13 |
| B0 frame snapshot vote | p2 | 0.00 | 41.67 | 0.00 | 12.50 | 3.13 | 78.13 |
| B0 frame snapshot vote | p4 | 8.33 | 41.67 | 13.89 | 12.50 | 9.38 | 78.13 |
| B0 frame snapshot vote | p8 | 2.08 | 41.67 | 3.47 | 3.13 | 21.88 | 75.00 |
| B0 frame snapshot vote | p16 | 0.00 | 39.58 | 0.00 | 0.00 | 25.00 | 75.00 |
| B1 track nearest | p1 | 0.00 | 41.67 | 0.00 | 12.50 | 0.00 | 78.13 |
| B1 track nearest | p2 | 0.00 | 41.67 | 0.00 | 12.50 | 3.13 | 78.13 |
| B1 track nearest | p4 | 8.33 | 41.67 | 13.89 | 12.50 | 3.13 | 71.88 |
| B1 track nearest | p8 | 8.33 | 41.67 | 13.89 | 3.13 | 18.75 | 62.50 |
| B1 track nearest | p16 | 2.08 | 39.58 | 3.47 | 0.00 | 31.25 | 56.25 |
| B2 track DP-Means | p1 | 0.00 | 41.67 | 0.00 | 12.50 | 6.25 | 78.13 |
| B2 track DP-Means | p2 | 0.00 | 37.50 | 0.00 | 6.25 | 15.63 | 78.13 |
| B2 track DP-Means | p4 | 0.00 | 39.58 | 0.00 | 3.13 | 21.88 | 68.75 |
| B2 track DP-Means | p8 | 0.00 | 43.75 | 0.00 | 3.13 | 21.88 | 62.50 |
| B2 track DP-Means | p16 | 0.00 | 43.75 | 0.00 | 0.00 | 28.13 | 59.38 |

WAIT/missed rates are zero and non-WAIT coverage is 100% for these forced
commitment baselines and complete GT streams. **That is not accuracy.** Wrong
Known rates and all pollution breakdown counts are also retained in JSON/CSV.
At p16, New ACC std is 3.61 percentage points for all three; False Merge std
is 8.84/6.25/10.36 points respectively. Correct Commit-CT is zero in every
p16 order. New ACC can be nonzero via posthoc global cluster mapping even
when online pure reuse fails; this difference is why separate metrics matter.

## Interpretation and next step

Observed result is weak; increasing prefix does not establish stable temporal
benefit. Four single-track Known prototypes and arbitrary fixed gates are a
limited operating point, not a fully tuned strongest baseline or evidence that
all possible nearest methods fail. Do not retroactively change the registered
thresholds (.65 Known cosine / .55 Existing cosine) to hide the outcome.

Keep the input/rows, protocols and raw sealed predictions unchanged. Next
permitted step is a bounded Train-only representation feasibility experiment:
A0 raw mean, A1 semantic adapter + mean, A2 learned temporal evidence. The
heldout pseudo categories cannot enter fitting. Any improved scores would
still require broader preregistered validation and a qualified predicted
frontend; this table alone supports no end-to-end scientific PASS.

B3 PHE is `INCOMPARABLE`: the compatible checkpoint/lineage is unavailable,
so no score is fabricated. No adapter/evidence/policy row is filled before
its actual experiment. B1/B2 recovered source remains byte-identical.

Resources: one CPU worker, 1.445 s baseline replay wall time, 677.14-MiB
host peak, no GPU job. Regression: 119 tests passed. Private prediction
Parquet is hashed but not uploaded; published artifacts contain only small
aggregate results, source/tests/config and this report.
