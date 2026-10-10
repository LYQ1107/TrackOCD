# Frozen limited-MASA evaluation

## Current execution status

The common predicted-track features are complete. Full semantic inference and
posthoc metrics have **not** yet run. This document records their prospective
protocol, not invented Val results.

- 988 existing TAO Validation videos / 36,375 annotated frames.
- All 304,561 frozen Native MASA physical IDs, including 114,380 singletons.
- All 1,294,110 actual first-at-most-16 observations; short tracks are clipped,
  never dropped or padded into fictitious evidence.
- Compressed common feature payload: 1,859,256,557 bytes (1.73 GiB).
- Actual extraction: 5,125.954 s; eight owned workers exited zero, no foreign
  process interference. Peak worker RSS 1,234,534,400 bytes; peak reserved GPU
  memory 1,474,297,856 bytes. Four completed integration shards and eight
  byte-compatible tiny-smoke observations reused, not re-extracted.
- Feature manifest SHA256:
  `899625d5b4135c6db450e5c86facbe98383a244b780f49475f436bcdd82fcc74`.
- Frozen Train model/protocol SHA256:
  `070f2182724449564ee8f6297f902b1c649c2d4a2f55fef5cac9222cbc991822`.

See `outputs/trackocd_core/LIMITED_MASA_FEATURE_FULL_RESULT.json` and the
all-shard validation receipt. Extraction used the already installed frozen
DINOv2 checkpoint, not a new detector/tracker, optimizer, GT or vocabulary.

## Registered matrix and causal evidence

Three actual training seeds, four video orders, five prefixes (1/2/4/8/16),
all raw baselines, learned representations/equal-capacity static control,
D1/D2 and all eight requested ablations. Train-development coverage targets
0.5/0.75/1.0 name frozen operating points, **not achieved Val coverage**.
The exact prospective matrix contains 2,040 logical cases / 840 distinct
fullstream executions. Exact-input aliases (including WithoutTemporal=D2/A1
and identical frozen WAIT biases) are disclosed and never independent trials.

Each replay sees only the actually observed prefix for a physical ID, makes
one irrevocable semantic decision, and mutates prediction-owned memory only
after that decision. WAIT is censored unresolved; it does not receive future
GT repair. Full-dimensional exhaustive search includes every live centroid;
no ANN, semantic pruning or matched-ID filtering. Conservative FP32 candidates
are refined with canonical NumPy dot. Both actual CPU and CUDA first-four
40-case proofs reproduce every previously sealed decision exactly. These are
within-current-environment equivalence proofs, not historical Python 3.7
numerical claims or measured fullscale speedup claims.

Frozen representations are computed by the unchanged CPU FP32 batch64
helper. At most one projected family bank is held per worker in RAM; no
projected feature cache is written to disk. Legal prototypes use 151 Train
tracks / 48 categories. All 78 inherited Known IDs remain permitted; 30
missing prototypes are not filled with Val GT. This is a disclosed shift
from the 15 fit-Known-prototype development episodes; no Val recalibration.

## Sealing, full-GT penalties and posthoc-only diagnostics

Separate evaluator workers open the existing geometry/role join only **after**
the corresponding immutable all-ID Parquet seal exists and its hash passes.
Inference workers cannot read annotations, GT joins, posthoc flags/metrics,
Test assets, network or subprocesses. The legal Train labels used for
prototypes are not represented as a false claim of 'no GT anywhere'.

Main denominators remain **4,413 Known / 819 Novel**, including missed GT.
Fixed cross-video reuse opportunities are 527 / 529 / 547 / 531 for the four
orders. Reliable physically matched 1,400 Known / 189 Novel are auxiliary
diagnostics only, never replacement main denominators. Every unmatched
prediction still mutates anonymous memory when committed; unknown members
cannot certify purity but are not automatically proven background/pollution.

Standard metrics use evaluator-only global Hungarian mapping once. Persistent
correct reuse has no such remapping and requires uncontaminated evidence of
the same Novel category from a previous different video. Paired diagnostics
count corrected errors **and introduced errors**. Fixed comparisons and
conditional video bootstrap (500 samples, seed1027) are registered before
full Val metrics; bootstrap does not re-run memory/mapping and is not a causal
CI or independent category experiment. Risk comparisons require both actual
fixed-Novel reuse coverage and overall predicted-ID commitment coverage gaps
at most 0.05. All-WAIT or unmatched coverage is not a persistence PASS.

## Resource-only amendment before full semantic metrics

The 840 execution identities and frozen science are unchanged. Raw B0/B1/B2
jobs have no projected bank to share, so the scheduler separates them into
three groups: 18 groups instead of 16. Learned groups retain one shared bank.
The bounded bank/source memory accounting gives a prospective 4 GiB worker
RSS ceiling instead of the earlier conservative 6 GiB plan; actual RSS guard
fails closed. Both supervisors reserve remaining peak headroom of their own
live children when launching, preserving 25% system RAM. Maximum eight
inference GPU workers and four CPU evaluators are dynamically throttled, not
guaranteed concurrent; GPU plan 4 GiB plus 8 GiB free reserve, stage storage
ceiling 10 GiB, total new-storage soft/hard 15/30 GiB. No foreign processes
are stopped. Valid exact-identity seals are resumable; partial attempts are
retained rather than overwritten.

## Scope and physical metrics

`LIMITED_COVERAGE_PROVENANCE_INCOMPLETE_FRONTEND`,
**ANNOTATED-CADENCE CAUSAL REPLAY**, not dense-frame online tracking or a
strong qualified frontend. Historical M1 stays `BLOCKED_FRONTEND_QUALITY`;
the new authorized Train-first method task is active independently.
Native/foundation supervision provenance remains incomplete. PHE is
`INCOMPARABLE` (48/78 coverage and feature/checkpoint supervision compatibility
unproven), not a fabricated matched-supervision baseline.

Every semantic method shares exactly the existing canonical full-Val Native
physical reference: HOTA 0.1450535920, AssA 0.4408791462, DetA 0.0484887546,
DetRe 0.6318088080. These are frozen physical results, not re-run tracking
evaluation or gains from semantic postprocessing. No TAO Test, new asset
downloads, Val training/tuning, new physical inference or safe feedback.
