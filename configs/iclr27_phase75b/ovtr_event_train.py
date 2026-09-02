"""OVTR Q0 evaluation config for the frozen Phase75B TRAIN video replay.

Only the annotation/split is changed from the pinned validation config.  The
model, score mode, thresholds, and causal frame processing remain supplied by
the replay command.  This config is not used for training.
"""

_base_ = "../../third_party/research_refs_phase4n/OVTR/ovtr/config/ovtr_lite_train_val.py"

data = dict(
    val=dict(
        # The pinned OVTR TaoDataset selects its TAO loader when the
        # annotation path contains ``validation``.  This is a symlink made
        # by run_event_replay.py; it points byte-for-byte at TRAIN and does
        # not copy or alter the annotation file.
        ann_file="/data2/usr_for_deadline/trackocd_phase75b/train_validation.json",
        img_prefix="/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames",
    ),
    test=dict(
        ann_file="/data2/usr_for_deadline/trackocd_phase75b/train_validation.json",
        img_prefix="/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames",
    ),
)
