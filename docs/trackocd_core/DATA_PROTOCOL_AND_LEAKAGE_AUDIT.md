# Data protocol and leakage audit — actual final scope

Only TAO Train intersect inherited Known IDs trains/chooses models and points.
78 inherited Known / 209 genuine Novel / 45 distractor roles preserved by
original recovered byte hashes. Train Known48 supported classes/2196 tracks/
43380 observations; genuine Novel7374 and unassigned2006 observations excluded
(distractor1879 also excluded). Legal model-fitting fold15 classes/1305 eligible
tracks; distinct full-budget adapter visits933/938/948, not all1305. Five
video-disjoint purposes with category-disjoint pseudo folds. Final heldout
207Known/16pseudo queries; four sparse pseudo classes, all with positives.
Prototype fold48classes; no Val prototype fill-in for30 unsupported Known classes.

Common frozen DINO ViT-B/14 checkpoint768-D, FP16 compact observation shards;
518resize/bilinear/context0.1/normalization pinned by feature config and source.
All 988 Val videos/304561IDs including114380singletons retained. Prefix caps
1/2/4/8/16 use only observed history and true clipped short length. Four
registered video orders, track IDs namespaced by video, no future videos,
labels, category names or posthoc mapping in live predicted memory. Loss
supervision is not memory initialization or GT repair. Exact CPU/CUDA proofs
and source/hash-bound adversarial tests are delivered.

Val evaluation starts after model/prototype/points freeze. GT roles/geometry
only in posthoc evaluators, missing GT included at4413/819 and fixed reuse
527/529/547/531. Same single posthoc Novel Hungarian for main and matched-only
diagnostic; Known IDs never remapped. Pure memory CT uses predicted members
and no semantic Hungarian. Unknown302972IDs retained, not labelled background
or automatically proven pollution. Bootstrap conditional on fixed mapping/state.

No TAO Test annotation/image read, no Val Novel training/tuning, no raw GT
Git upload. Limited Native physical runtime has one generic foreground class
and no Novel vocabulary/text input; exact all-stage physical checkpoint
supervision binding is UNVERIFIED. Therefore no blanket all-Novel-supervision
exclusion or strong M1 PASS. Historical COVTrack is vocabulary-assisted and
its255shards are partial/unvalidated; SimOWT provenance/score diagnosis is not
silently repaired. FROZEN_PHYSICAL_FRONTEND.json is a late hash-bound handoff
of the already frozen authorized limited stream, not a new scientific freeze.

See FINAL_RESULT.json (hash bindings and every requirement disposition),
TRAINING_PROTOCOL.md (complete fold/input contract), INPUT_SHIFT_DIAGNOSTIC.md
(posthoc descriptive, not intervention) and RESEARCH_LOG.md (stage history).
