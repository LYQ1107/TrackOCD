"""Phase69 OVTR adaptation config.

The upstream/local OVTR code is imported read-only.  The train annotation is
selected by PHASE69_FOLD so one immutable config can be used by the bounded
four-fold supervisor.  Text cross-attention is disabled; the legacy embedding
files are retained only because OVTR's constructor requires tensor shapes.
They are not consumed by the class-agnostic DSCT physical score/lifecycle
branch or exported as TrackOCD inputs.
"""
_base_ = '/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/third_party/research_refs_phase4n/OVTR/ovtr/config/ovtr_lite_train_val.py'
import os

_fold = int(os.environ.get('PHASE69_FOLD', '0'))
_manifest_root = '/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/outputs/iclr27_phase69/manifests'
# The policy manifests are consumed by train_fold.py; the OVTR LVIS loader
# still receives the immutable annotation source with images/annotations.
_train_ann = '/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/third_party/research_refs_phase4n/OVTR/data/lvis_clear_75_60.json'
_val_ann = '/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/third_party/research_refs_phase4n/OVTR/data/validation_ours_v1.json'

use_text_cross_attention = False
use_fusion_layer = False
data = dict(
    train=dict(ann_file=_train_ann),
    val=dict(ann_file=_val_ann, img_prefix='/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/third_party/research_refs_phase4n/OVTR/data/TAO/'),
    test=dict(ann_file=_val_ann, img_prefix='/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/third_party/research_refs_phase4n/OVTR/data/TAO/'),
)

# Keep the original causal image pipeline and persistent-query schedule.  The
# HDF5 image store is referenced in the inherited train pipeline; no data is
# copied into Phase69 outputs.
train_with_artificial_img_seqs = True
distribution_based_sampling = False
initial_grad = False
backbone_freeze_keywords = None
