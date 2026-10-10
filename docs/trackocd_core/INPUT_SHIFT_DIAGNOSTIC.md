# Actual descriptive input/prototype coverage (2026-10-11)

Status: `COMPLETE_POSTHOC_DESCRIPTIVE_INPUT_SHIFT_NOT_CAUSAL_INTERVENTION`.
Source registration was exact remote
`b79d691bd678b22daca16303cd2ab74ad2a23b18`. These statistics were measured
after the predefined RawB0/main/prefix1 all-ID prediction seal and first Val
metrics, not registered beforehand as a causal hypothesis or intervention.
They cannot establish the unique cause of a model's performance change.

The checker verified all 988 common feature payload hashes, the 52 frozen
model/protocol hashes, actual legal Train metadata and the existing posthoc GT
geometry join. It did not load visual arrays, optimize a model, modify a point,
fill missing prototypes from Val, rerun physical inference or access Test.
Only scalar counts and hashes are published in `INPUT_SHIFT_DIAGNOSTIC_RESULT.json`.

## Available observations, including short and unmatched tracks

| Input/cohort | Tracks | First-at-most-16 observations | Mean available length | Singletons | At least 16 |
|---|---:|---:|---:|---:|---:|
| Selected Train features | 2,166 | 24,628 | 11.3703 | 102 | 1,147 |
| All native predicted IDs | 304,561 | 1,294,110 | 4.2491 | 114,380 | 24,767 |
| Reliable Known match | 1,400 | 17,497 | 12.4979 | 65 | 887 |
| Reliable Novel match | 189 | 2,731 | 14.4497 | 1 | 150 |
| No reliable GT match | 302,972 | 1,273,882 | 4.2046 | 114,314 | 23,730 |

There were zero other-role reliable matches in this existing join. The 302,972
unmatched IDs are **not** assumed background or proven memory pollution.
The conditional reliably matched cohorts are much longer than the all-ID
stream; they must not replace the full stream or full-GT denominators.
At caps 1/2/4/8/16, the all-ID observation counts are respectively
304,561 / 494,742 / 736,677 / 1,013,481 / 1,294,110. Short tracks are clipped
to available observations, never dropped because they cannot reach a cap.

## Legal Known prototypes crossed with actual physical coverage

| Known class cohort | Reliable physical match | No reliable match | All Known GT |
|---|---:|---:|---:|
| 15 optimization-eligible classes | 1,307 | 2,793 | 4,100 |
| 33 prototype-only classes | 87 | 202 | 289 |
| 30 classes without a Train prototype | 6 | 18 | 24 |
| Total | 1,400 | 3,013 | 4,413 |

Thus the missing 30/78 Known class prototypes concern 24 Known GT tracks here,
not 30/78 of the Known GT population. The 15-class optimization-eligible cohort
is not proof every eligible Train track was sampled, nor proof an earlier
selected checkpoint saw a full-budget sample footprint. Actual fit exposure is
reported separately. All 819 Novel GT targets remain in formal evaluation;
only 189 have a reliable physical match. No per-target semantic names or labels
are inference inputs or published by this checker.

## Runtime and limitations

One CPU worker, no GPU, 15.720 seconds. Maximum **sampled current VmRSS** was
153,104,384 bytes, within the 256 MiB current-RSS plan and joint 25% RAM reserve.
The separately reported kernel `ru_maxrss` was 847,216,640 bytes and can retain
a pre-exec peak; it is not silently replaced or mislabelled as sampled VmRSS.
This diagnostic describes actual inputs and coverage only. It does not support
Val recalibration, class exclusion, metric denominator shrinkage, a causal
ablation, or a claim that the full method matrix has finished.
