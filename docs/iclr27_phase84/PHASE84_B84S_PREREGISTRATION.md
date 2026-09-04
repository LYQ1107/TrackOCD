# Phase84 B84S preregistration

This is the single source-conditioned support matcher authorized after the
A84P physical reassociation diagnostic.  Phase83/Phase84 Q0, row keys, the
984-query R universe, 76 positive/76 negative O comparator, and the causal
prefix set `{1,2,4,8,16}` are frozen.

## Hypothesis

A completed TRAIN track can condition selection of a target candidate in the
same corrected native DINOv2 space more reliably than query-agnostic ranking.
The source is restricted to a previously completed track; the target is the
current candidate plus its causal history.  A single softmax contains all
candidate actions and an explicit DEFER action.

## Data and model contract

- Supervision is public TRAIN only, split video- and category-disjoint.  TRAIN
  category labels construct positive and hard-DEFER labels but never enter a
  tensor.  No DEV+, Q1, public-new or sealed label is read.
- Source features are corrected native Q0 DINOv2 (same extractor,
  preprocessing, and 768-D normalization) with a mean and up to three causal
  prefix-16 prototypes.  Target candidates are the full native Q0
  `(video_id,image_id)` candidate set; candidate descriptors include only
  native appearance, causal age/history, base score/rank and normalized box
  geometry.
- Each fit group contributes at most two positive and two hard-DEFER support
  candidates.  The action space remains candidate actions plus DEFER; no
  candidate is removed at inference.  The selector is a small permutation-
  invariant listwise scorer trained for 15 effective epochs.
- Inputs explicitly exclude category, text, semantic/physical ID, future
  rows, GT boxes/labels, StateMemory and controller actions.  Support timestamps
  are strictly prior in the episode manifest; same-track temporal prefixes are
  consistency metadata, not cross-instance positives.

## Gates

The B84S signal audit must show a measurable source-conditioned TRAIN gain
before training.  Learned selection is compared with frozen raw cosine on the
same validation groups.  A safe R signal requires p16 R@1 and mAP gain of at
least 0.01, no unsafe flip, and at least three folds non-decreasing; only then
can unchanged-controller compatibility be considered.  Retrieval alone never
counts as Commit-CT success.

