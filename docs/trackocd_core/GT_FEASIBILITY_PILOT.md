# Small Train Known GT feasibility pilot — not the predicted main study

M1 remains `BLOCKED_FRONTEND_QUALITY`. This uses the user-authorized GT-track
exception; it does not freeze a frontend or complete formal M3/M4/M9. No Val,
genuine Novel or Test observations/labels are needed. No physical tracker is
trained, and this first delivery contains no representation/policy training.

## Registered data, before descriptors or baseline scores

The inherited 78/209/45 roles remain unchanged. Only the 48 Train-supported
Known categories are eligible. There are 46 categories with a >=16-observation
track, 1,170 such tracks. Deterministic support-based selection yields:

| Partition | Categories used | Tracks | Purpose |
|---|---|---:|---|
| Representation | Known 41,126,133,211 | 16 | Four separate prototype tracks; 12 future fit tracks, three videos/class |
| Policy Train | Same four Known plus simulated unknown 35,81,95,118 | 24 | Three video occurrences per category |
| Heldout selection | Same four Known plus simulated unknown 139,174,229,235 | 24 | Pseudo-Novel categories unseen to representation and policy fitting |

All three partitions have globally disjoint video sets (16/24/22 videos).
Known probe categories are deliberately shared with their prototypes; **only
pseudo-Novel** categories are category-held-out. Heldout selection is a model
selection set, not an independent final test. All simulated unknowns are
inherited Train Known classes, not real TAO Novel supervision. Preserve these
exact rows for all subsequent pilot methods; never resample based on scores.

Select the first physical track in each chosen video and its first 16
chronological observations. Exact grouping and unavailable-support failures
are in `src/trackocd_core/pilot.py`; the private single compact plan has a
public size/hash receipt. No per-track JSON or marker proliferation.

## Compact extraction boundary

At most 64 tracks / 1,024 observations, batch four, one worker and a freshly
idle GPU. The existing pinned DINOv2 ViT-B/14 checkpoint and crop/normalization
protocol are reused. Exact compatible earlier smoke observations are reused.
FP16 descriptors/prefix aggregates plus index/geometry/Train-label sidecar;
raw crops and weights are not duplicated or published. The model receives
only `PrefixView`, never labels, GT identities, full lengths or future tails.

Registered bounds: 4-GiB host RAM plan, at least 25% system headroom,
16-MiB new payload ceiling and 600-second extraction ceiling. This is a
small descriptor inference job, not a full-cache supervisor or long training.

## First baseline replay protocol (fixed before feature results)

- B0: nearest decisions for individual visible frame descriptors against the
  **same pre-track memory snapshot**, majority vote at the prefix boundary.
  Ties use first chronological vote; NEW is one provisional choice. Commit
  once per physical track and only then update memory with the prefix mean.
  This avoids hidden/unvoted frame tokens; report as snapshot frame-nearest
  voting, not a verified per-frame online controller.
- B1/B2: use byte-identical recovered `NearestPrototype` / `OnlineDPMeans`;
  only adapt their output into the new sealed evaluator. Both see the same
  prefix means, Known prototypes and anonymous-memory stream. Their existing
  arbitration differs and is not silently repaired.
- Fixed comparable gates: Known cosine .65 (DP distance .35), Existing cosine
  .55 (DP lambda .45). No heldout or Val Novel threshold search. These are
  preregistered pilot settings, not claims of optimal operating points.
- All methods use the same four legal Known prototype tracks, normalized
  p16 means prepared before evaluation. Four-Known support is **not** the
  full 78-Known main benchmark and must not compete with incompatible PHE.
- Four orders: sorted video IDs and Python seeded shuffles 1027/1028/1029.
  Five independent prefix replays p1/p2/p4/p8/p16, fresh anonymous memory
  each time. Within video, process prefix-completion frame then physical ID
  as deterministic routing/tie-break; neither becomes a model input.
- Only after sealing predictions, attach Train simulation labels to evaluate
  standard global anonymous Hungarian and pollution-sensitive fixed GT reuse
  opportunities. Hungarian/GT labels never repair or update policy memory.
- B3: `INCOMPARABLE` until a compatible PHE checkpoint and supervision/source
  lineage exist; no fabricated PHE score. Adapter/evidence/policy variants
  also remain pending, not filled with synthetic successes.

This support-biased long-track subset has perfect GT physical coverage and
unit quality, no missing detections or real tracker contamination. Its scores
can establish software/bounded representation feasibility, **not** success on
predicted TAO Val, robustness to frontend errors, or statistical generality.

## Reproduction

Run the registered plan once with `prepare_gt_pilot.py`; preserve the public
preregistration and private plan. After code/tests/config/preregistration are
committed and pushed, `extract_gt_pilot.py` creates its new compact cache once.
Both refuse to overwrite an existing completed artifact. Reuse/verify instead.
Outputs and raw features stay private; only small aggregate receipts/reports
and source/tests/config are published. Subsequent results append here without
rewriting the preregistered choices or conflating them with formal M4.

## Completed descriptor inference

Preregistration `8295d52` was pushed and independently remote-verified before
inference. The bounded run completed in 38.13 seconds: 64 tracks / 1,024
observations, with 64 exact earlier smoke observations reused and 960 new
frozen descriptors. Compact payload is 2,114,366 bytes (~2.02 MiB), GPU peak
543.62 MiB and host peak 1.24 GiB; one worker on a freshly idle GPU UUID.
All five payload hashes and 320 causal prefix views validate. No optimizer,
representation training, Val/Test access or foreign process action occurred.
Actual simple baseline scores remain the next separate step.
