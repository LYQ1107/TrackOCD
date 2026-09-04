# Phase84 B84S Query-Conditioned Contract Repair (registered)

## Trigger and hypothesis

The completed `b84s_formal_r2` selector was trained on native candidate groups,
but `build_b84s_manifest.py` attached one deterministic first TRAIN source of a
matching category to each target image group.  It therefore did not train the
registered query/source pair contract.  Frozen event replay selected a reliable
candidate on 9/76 positive events at prefix 16, below the same-space raw
source-mean diagnostic (20/76), despite the TRAIN signal audit being strong.

The one authorized repair is `B84S-Q`: construct one listwise action group for
each legal TRAIN query/source pair from the existing Phase30 episode manifests.
Positive groups use a same-category, different-video completed source track;
`null_no_match_hard_negative` groups use the registered different-category hard
source and target **DEFER**.  Target candidates are the native Q0 per-image
candidate set, with TRAIN GT used only to construct the candidate/DEFER target.
The linear listwise matcher, 19-D descriptor layout, explicit DEFER action,
15 effective epochs, and checkpoint/evaluation rules remain unchanged.

## Fixed contract

- Native Q0 candidate stream and corrected DINOv2 cache are unchanged.
- Source and target use the same corrected native DINOv2 space and the source
  prefix-16 mean plus fixed three contiguous causal prototypes.
- Query source/target videos must differ; all 91 positive/negative Phase75B
  event videos are excluded from TRAIN fit/validation groups.
- No category, semantic/physical ID, GT box/IoU, event label, future row or
  text field is placed in the feature tensor.  Category/IoU are TRAIN target
  metadata only.
- Existing Phase30 fold assignments are retained for this repair so no event
  result can alter a split.  The manifest records fit/validation counts and
  any imbalance; no checkpoint is selected with held events.
- At most two positive and two hard-DEFER target groups are retained per source
  track, selected deterministically by episode id.  No candidate row is
  deleted from a native action set.

## Execution and gates

Run a 100-update smoke, a 500-update fold-0 targeted run, then one bounded
four-fold 15-epoch formal supervisor.  Checkpoints and `.launched`/`.done`
markers are atomic.  This is a retrieval/support diagnostic only; no controller,
StateMemory, threshold sweep, public/DEV+/Q1/sealed evaluation or backbone
change is permitted.  Event replay is run once after the formal checkpoints
are frozen.  The repair is retained only if it improves frozen B84S selection
without changing the event denominator or causal protocol.
