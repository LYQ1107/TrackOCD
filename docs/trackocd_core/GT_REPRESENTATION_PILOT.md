# Bounded category-evidence GT pilot — preregistration

This is the expressly permitted Train Known GT representation-feasibility
exception while primary M1 remains `BLOCKED_FRONTEND_QUALITY`. It is not
formal M5/M8/M9 completion or an end-to-end success. The input plan from
`8295d52`, descriptors from `1fd79b7` and weak baseline results from `08b2dde`
remain unchanged. Baseline CSV line endings are normalized to LF only; the
original execution source hashes remain historical, not silently replaced.

## Fit and selection separation

Only the 12 `representation_fit` tracks (Known 41,126,133,211; three distinct
videos per class, 192 observations) enter learning. The four separate Known
prototype tracks, policy_train stream and heldout_selection stream never
enter the optimizer. Heldout pseudo classes 139/174/229/235 are unseen to
representation/policy fitting; Known probes intentionally share prototype
class IDs. All fit/stream partitions have disjoint videos. Genuine TAO Novel,
Val or Test labels are not involved.

Trainable components only: A1 semantic adapter 768→512→256 with GELU and L2
normalization, 525,056 parameters; A2 the same adapter plus a 261→64→1
causal reliability network, 541,889 parameters. A0 frozen raw DINO mean has
zero trainable parameters, optimizer or claimed training. Frozen DINO is
not loaded/retrained in this feature-only job, and the physical frontend
is not modified. Identical adapter initialization per A1/A2 seed.

Temporal network takes semantic descriptor, unit GT quality, cumulative
prefix consistency/dispersion, observed count and elapsed frames. Cumulative
statistics at each observation never inspect later observations. Predicted
positive weights yield a normalized category embedding, uncertainty and
effective observation maturity. Labels/roles/identities are absent from
forward arguments. Prefix/future-tail and identity-positive tests pass.

## Registered bounded fits, before training

Three seeds 1027/1028/1029, 120 AdamW steps per model/seed, batch all 12
legal fit tracks; equal cycling p1/p2/p4/p8/p16. LR .001, decay .0001,
cross-video supervised category contrastive temperature .1, gradient clip 1.
Positives are same category, different physical tracks and videos; negatives
are different categories. This is category learning, not same-individual ReID.

A1/A2 receive identical same-seed synthetic augmentations: small descriptor
noise and occasional one-observation replacement by another fit class.
Both optimize the same category loss. A2 additionally has weight-.1 synthetic
corruption reliability BCE; augmentation targets stay outside model inputs.
Consequently this comparison evaluates the **module plus its auxiliary
supervision**, not architecture alone; capacity/loss controls remain later
work before a contribution claim. No loss/selection-based early stopping,
checkpoint cherry-picking or hyperparameter retry.

One freshly idle GPU/worker, one CPU thread, 4-GiB RAM plan with >=25%
system headroom, maximum 600 seconds for six fits and 64-MiB checkpoint
payload. Save final fixed-step checkpoints privately with source/config/cache
SHA and seed. No DINO/detector/tracker checkpoint or raw crop duplication.

## Fixed evaluation and interpretation

After final checkpoints freeze, compare A0/A1/A2 on both previously fixed
Train streams, all four video orders and five prefixes with the same four
prototype tracks and byte-identical recovered B1 arbitration. Prototypes are
transformed by the corresponding representation, not given new supervision.
No GT/mapping enters policy state or repairs anonymous memory.

Preserve the .65/.55 cosine gates. These thresholds may not be equally
calibrated after adaptation; report the limitation rather than attribute any
gain purely to representation. Any future calibration must be a separately
registered Train-only experiment, not silently tuned on heldout/Val Novel.
Report all seeds/orders/prefixes, fixed denominators and pollution breakdown;
compare paired seeds rather than treating 12 correlated seed/order runs as
independent datasets. Loss decrease alone is not success. These tiny GT-only
scores cannot authorize M11, prove predicted robustness or main scientific PASS.

## Actual bounded result — negative discovery finding

Preregistration `b696190` was pushed and exact remote HEAD checked before
training. Six fits completed, 120 fixed steps each, 13.21 s total,
28.70-MiB GPU / 1.04-GiB host peak; private checkpoints total 12,822,732 B.
Identical per-seed synthetic corruption counts for A1/A2: 363/372/374.
Category loss declined from ~2.37–2.41 to ~.72–.76; that is **not** success.

Frozen evaluation completed 280 actual replays / 6,720 sealed decisions in
9.46 s on one CPU worker (~677.14-MiB peak). A0 exactly reproduces the
original frozen B1 metrics in every partition/order/prefix. All raw decisions
are retained in private hashed Parquet; public JSON/CSV keeps every seed,
order, prefix, count and error type. Regression: 123 tests pass.

Heldout selection, percentages (four-order mean per seed; all prefixes kept):

| Representation | Seed | p16 Old ACC | p16 New ACC | p16 H | p16 Correct CT | p16 Wrong Known |
|---|---:|---:|---:|---:|---:|---:|
| A0 raw mean | N/A | 2.08 | 39.58 | 3.47 | 0.00 | 12.50 |
| A1 adapter + mean | 1027 | 66.67 | 0.00 | 0.00 | 0.00 | 100.00 |
| A2 adapter + evidence | 1027 | 66.67 | 0.00 | 0.00 | 0.00 | 100.00 |
| A1 adapter + mean | 1028 | 83.33 | 0.00 | 0.00 | 0.00 | 100.00 |
| A2 adapter + evidence | 1028 | 83.33 | 0.00 | 0.00 | 0.00 | 100.00 |
| A1 adapter + mean | 1029 | 83.33 | 0.00 | 0.00 | 0.00 | 100.00 |
| A2 adapter + evidence | 1029 | 83.33 | 0.00 | 0.00 | 0.00 | 100.00 |

A1/A2 p16 Known mean is 77.78%, seed population std 7.86 points. At all
prefixes their New/H/CT are zero, and pseudo-Novel reuse opportunities are
all assigned Known. Their aggregate decision metrics match exactly across
paired seeds. This is **no independent temporal-discovery gain on this fixed
pilot**, not proof the module can never help elsewhere. Forced commitment
coverage is 100%; no anonymous merge is recorded when all unknowns go Known.

No early stop, hidden threshold change, dataset resampling or retraining was
performed after evaluation. Loss decline and Known-only improvement cannot
support an open-world-discovery claim. The limited four-class fit and
post-adaptation threshold calibration are possible explanations, not yet
isolated causes. Retain A1 as a candidate, not as scientific PASS; do not
promote A2 or M11 based on these results. Next: diagnose Known/pseudo-Novel
score separation on **policy_train only**, without refitting, expanding the
network or using Val Novel labels. Any correction must be explicitly
registered and respect the one-root-cause-correction cap.
