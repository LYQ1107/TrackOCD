# M3 preparation — sealed, coverage-aware evaluator (GT feasibility only)

2026-10-10 (Asia/Shanghai). **PASS_ENGINEERING_SYNTHETIC_ONLY**, not formal M3
or predicted-track M9 completion. M1 still has no qualified physical frontend.
No TAO dataset/model/training experiment ran in this preparation. No original
v2 evaluator or historical score was modified.

## Legacy audit

Both inherited evaluator files were read in full:
`src/trackocd_v2/evaluation/standard_ocd.py` and `persistent.py`.

- Standard constructs its global assignment across **all states/categories**,
  including Known. A wrong Known ID can therefore be relabeled after the fact.
- Persistent checks whether **any** prior token member had the same category
  in another video. It does not reject already mixed-category history and
  does not distinguish wrong NEW / EXISTING / KNOWN adequately.
- The inherited GT-order/missing-opportunity improvement is retained as a
  principle, not mistaken for proof that contamination/decision typing works.

The 81 recovered source files remain byte-identical (642,256 bytes), freshly
verified against NAS recovery Git blobs. New code is isolated under
`src/trackocd_core/evaluation/`; old source/results remain references.

## Two separate stages, no evaluator feedback

1. `seal_decisions`: immutable typed physical IDs, registered order and
   prefix cap, **Known role IDs only**, prediction events. No GT target,
   Novel label/vocabulary or geometry argument. NEW must create a new token;
   EXISTING must reference an already-created token. WAIT may become a first
   commit; later revision/WAIT cannot erase that irrevocable commitment.
2. `join_evaluation`: after sealing, one-to-one geometry-only join to the
   complete fixed GT universe. Missing predictions remain GT targets.
   Unmatched predicted identities must explicitly map to None. Only then
   do independent Standard and Persistent scorers read GT roles/categories.

The registered identity is `(video_id, local_track_id)` for both namespaces.
Same local integer/string in another video is a different physical object.
Known role IDs and actual supported prototype coverage are distinct: full
Known evaluation denominators cannot be reduced to the 48 Train-supported
classes. Fair prototype availability remains a later baseline manifest gate.

Prefix count must be integer, nondecreasing per physical identity and <=p;
short tracks may provide fewer than p observations. Every order/prefix starts
fresh memory. This seals event timing, **not** a runtime frame-online proof
or proof that a method obtained its features causally; M2's reader and an
actual upstream/replay artifact must independently establish those properties.

## Registered metrics

Standard uses exact Known IDs. One global Hungarian assignment is restricted
to anonymous tokens versus Novel categories, only for scoring sealed outputs.
Old/New/All denominators contain all respective non-distractor GT tracks,
including missing/WAIT. H = 2*Old*New/(Old+New); when both are zero H is zero.
Cluster count includes every token creation, including unmatched or Known-
misassigned tokens. Only zero-Novel-support rows are omitted from the dense
assignment matrix for memory; they are not hidden from cluster count.

Persistent eligibility is defined by GT: a Novel target is eligible whenever
its category exists in an earlier registered video. A missing source or
target prediction cannot remove this opportunity. Report four populations:
all eligible Novel GT, fixed reuse opportunities, valid predicted opportunities,
and opportunities with a non-WAIT commitment.

| First commitment on a reuse opportunity | Outcome |
| --- | --- |
| No matched physical prediction | Missed; included in unresolved |
| Matched WAIT / no decision | WAIT; included in unresolved |
| KNOWN | Wrong Known, not an anonymous merge |
| NEW | False Split/New, even if its GT source was missed |
| EXISTING with pure same-class past other-video support | Correct Commit-CT |
| EXISTING into a past wrong class | Wrong-category merge |
| EXISTING reusing already mixed-class history | Contaminated reuse / false merge |
| EXISTING with unclassified past members only or alongside correct class | Unverified reuse, not certified correct or proven false merge |
| EXISTING with only same-video support | Unsupported Existing, not cross-video success |

All headline rates use **the same fixed GT opportunity denominator**.
False Merge = wrong-category merge + already contaminated reuse, separately
from False Split/New and Wrong Known. Unknown and same-video-only failures
have separate rates. The outcome counts partition that denominator exactly.
Effective Commit Coverage = valid non-WAIT commitments / fixed opportunities;
it includes incorrect commitments and is not accuracy. Matched-only Commit-CT
is an explicitly secondary diagnostic, never the primary result.

Purity uses an absorbing chronological summary: a known different GT category
poisons the token permanently. An unmatched/unknown member prevents certainty
of purity but is not by itself proof of a wrong class. Score before adding
the current member. Future contamination cannot retroactively erase a past
correct commit. The persistent scorer never uses Standard's Hungarian mapping.

Zero denominator yields **null / NOT_APPLICABLE**, never invented perfection.
The contract is registered in `configs/trackocd_core/evaluation_protocol.json`.
Actual primary-stream matching adapter and full fixed-universe hashes remain
pending; this document does not freeze a dataset/main experiment.

## Verification and resources

**110 tests passed**, including 48 new evaluator cases: all requested
anti-cheating cases; unknown/distractor pollution; missing source and target;
exact Known IDs; irreversible commits; duplicate GT join; future token/video/
prefix violations; four synthetic video orders × five prefix caps; strict
scalar types and a 2,000-member token without quadratic history copies.

The separate CLI executes 20 preconstructed pure-reuse fixtures plus a
missing-source case. These expected events are **not a learned model**, and
their toy perfect accuracy is **not a TAO result or scientific PASS**.
Its receipt includes code/config/test SHA256, measured <76-MiB peak RSS,
one CPU worker, no GPU, and ~0.004 s wall time. No real annotation, image,
feature, Test file, checkpoint, model inference or training is opened.

```bash
source /data3/liuyeqiang/trackocd_a100_env.sh
python scripts/trackocd_core/smoke_evaluation_protocol.py
python -m pytest -q tests/trackocd_core tests/trackocd_v2
```

Storage is small source/config/aggregate JSON only. No per-track output/crop
cache, duplicated frames, full predicted features or foreign-process action.
Next: source-side SimOWT score-distribution evidence and qualified frontend;
then the real fixed-universe adapter and appropriately scoped GT feasibility.
