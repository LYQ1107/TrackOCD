"""Phase83 A2 read-only Q0 inference config over public TRAIN videos.

The annotation is exposed through the validation-named symlink created by the
A2 runner because pinned OVTR dispatches its TAO loader from that path token.
No training is enabled and no category/text branch is used by the runner.
"""

_base_ = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/third_party/research_refs_phase4n/OVTR/ovtr/config/ovtr_lite_train_val.py"

data = dict(
    val=dict(
        ann_file="/data2/usr_for_deadline/trackocd_phase83/phase83_train_validation.json",
        img_prefix="/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames",
    ),
    test=dict(
        ann_file="/data2/usr_for_deadline/trackocd_phase83/phase83_train_validation.json",
        img_prefix="/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames",
    ),
)
