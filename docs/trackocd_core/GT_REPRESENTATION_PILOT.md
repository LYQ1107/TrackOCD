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
