# M1 — Official SimOWT source training-chain trace

2026-10-10 (Asia/Shanghai). **Static-source progress, not a qualified physical
frontend or complete checkpoint training certificate.** No inference, model
loading, training, new annotation/GT/Test access or NAS task was started.
The completed NAS native-row score scan need not be repeated.

## Actual source recovery and comparison

Read the complete Git tree of official SimOWT at
`753b2ac0082976753350d1b74a4d332088889e57`: 561 entries, not truncated,
no upstream AGENTS.md. Download only ten individually named UTF-8 source,
configuration and documentation files: **169,655 bytes**. Check each exact
size and Git blob SHA1; record its SHA256. No repository/binary/archive,
annotation pack, checkpoint, new environment or dataset was copied.
Full sources stay private; the public manifest and aggregate receipt preserve
identities and source URLs. They are official source, not the old dirty patch.

Compare against already completed NAS command
`exec-aee15cc8-b452-429a-905b-57e38ffdf01d`:

| File | Pinned upstream bytes | NAS current bytes | Byte/SHA256 match |
| --- | ---: | ---: | --- |
| r50_train.yaml | 1,174 | 1,174 | yes |
| idol.py | 54,461 | 58,283 | no |
| train_net.py | 6,866 | 6,977 | no |
| detectron2 dataset builtin.py | 10,559 | 11,076 | no |

Upstream idol SHA256 is
`6a66c3e2a0c56e53574a9546d248a50a604acaf9f9cd1efef6e8afb49007b318`;
NAS current is
`00970901e4501879f4722590a085817a4a49ec6f51683d798a8760e9c948c4bb`.
Previously read NAS snippets confirm the unchanged probability compression
and training branch, but are not a recovered full patch or historical run
snapshot. Sharing a Git HEAD does not imply identical worktree bytes.

## Training source findings

The [training YAML](https://github.com/22109095/SimOWT/blob/753b2ac0082976753350d1b74a4d332088889e57/projects/IDOL/configs/r50_train.yaml)
has no base-YAML inheritance, one foreground class, COCO_PRETRAIN=true,
initial R-50.pkl, and the coco_2017_train_agn dataset name. Its
[registry](https://github.com/22109095/SimOWT/blob/753b2ac0082976753350d1b74a4d332088889e57/detectron2/data/datasets/builtin.py)
points to instances_train2017_agn.json. This alone does **not** prove pure
COCO training: the [COCO clip mapper](https://github.com/22109095/SimOWT/blob/753b2ac0082976753350d1b74a4d332088889e57/projects/IDOL/idol/data/coco_clip.py#L121)
rewrites filenames without train2017/0 from coco/train2017 to tao/frames/train,
deduplicating train/train. Three synthetic filename examples verify this
string routing without opening any image or executing the mapper. These
are reachable paths, not observed rows of the actual training file. The two
mapper views are augmentations of one source image, not established real
adjacent video frames.

Two auxiliary generators have different supervision boundaries:

- [gen_faker_anno.py](https://github.com/22109095/SimOWT/blob/753b2ac0082976753350d1b74a4d332088889e57/dataset_extra/gen_faker_anno.py#L163)
  filters the stored TAO Train GT annotation list to the same inherited
  78 Known IDs before adding selected proposal boxes. It collapses classes
  to foreground and supplies rectangle masks. Proposal/teacher provenance
  remains unknown. Save statements are commented out; the default entrypoint
  calls a diagnostic, not gen(). Neither output generation nor incorporation
  into the released weight is proved.
- [gen_extra_anno.py](https://github.com/22109095/SimOWT/blob/753b2ac0082976753350d1b74a4d332088889e57/dataset_extra/gen_extra_anno.py#L26)
  filters only a local annos variable when generating extra Known boxes, then
  extends and saves the original data['annotations'] list. Original annotations
  would be retained if this script ran on an unfiltered input. Its use in the
  released checkpoint is UNVERIFIED; this source possibility is **not** positive
  evidence of forbidden supervision in that checkpoint.

The loader overwrites category integers with encoded longscore (default 1000).
IDOL extracts longscore/1000 then zeros class targets; the criterion uses the
square-root reliability to weight ReID/auxiliary loss, overriding COCO weights
to one. A one-class head therefore does not certify the source of box/instance
supervision. Source pointers and hashes are recorded in the receipt.

The pinned eval branch defaults to function_choice=2 (tracking), while pseudo
generation uses choices 1/3. The complete tree lacks the nested eval config
and truncode.py referenced by the model instructions. Thus those instructions
are not a complete reproducible self-training/merge pipeline. No historical
command from them was executed. The validation result filename test.json in
those instructions is not evidence that TAO Test was used; no Test asset was read.

## Decision and minimum missing evidence

**BLOCKED_PROVENANCE_NOT_PROVEN_LEAKAGE; primary selected = none.** Keep original
Q0 as an identified reference. Its observed score/fragmentation defects remain;
no frozen scores, boxes or IDs are altered. COVTrack's proven Novel-vocabulary
failure and SimOWT's unverified released-weight supervision remain distinct.

Before a corrected or original SimOWT candidate can qualify, obtain a small
existing provenance ledger, not a whole annotation pack:

1. Released checkpoint SHA -> all training/self-training stage source/config
   identities, initialization/teacher SHAs and final weight mapping.
2. Actual merged training JSON identities and source/split/role summaries;
   establish Known GT versus generated pseudo boxes and absence of genuine
   Novel/Val supervision. Merely collapsing IDs to foreground is insufficient.
3. Teacher proposal/pseudo-generation split, vocabulary and supervision lineage,
   plus actual merge/filter source. The dormant auxiliary scripts cannot certify it.
4. Historical run source/weight binding or a separately sealed future bounded
   frame-only replay. Current-source safety is not historical execution proof.

The complete paper remains unavailable through inspected official routes;
do not infer its detailed splits from the abstract or substitute an unverified
annotation-pack download. Existing requests for lawful evidence/scope direction
remain pending. Do not broaden GT representation feasibility into decision
learning or run another representation correction; the sole R1 round is used.

## Verification and resources

Source-only audit: one CPU worker, **0.0189 s / 18,820-KiB peak RSS**. Disk recheck
before download: 93,861,072,896 bytes available; RAM 107,248,529,408 bytes available.
No GPU needed. Regression: **131 passed in 3.86 s**, including exact identity
of all 81 recovered v2 files. Initial six new checks passed but the seventh
mistakenly expected NAS idol bytes to equal upstream; the observed mismatch
was retained and corrected into an explicit provenance comparison. No model,
experiment result or original source was repaired to make that check pass.

Reproduce locally with the private, hash-pinned ten-file source selection:

```bash
source /data3/liuyeqiang/trackocd_a100_env.sh
python scripts/trackocd_core/audit_simowt_training_source.py
python -m pytest -q tests/trackocd_core tests/trackocd_v2
```

Public receipt: outputs/trackocd_core/audit/simowt_official_training_source.json.
Goal is active, not complete; strict main M1–M10 remain unfinished.
