"""Phase71 Q0-preserving adapter config.

The pinned OVTR code is imported read-only by the Phase71 wrapper.  The base
proposal/query/decoder/parent-assignment path is frozen initially; only the
trajectory-conditioned class-agnostic TCO quality/lifecycle adapter is
trainable.  Base and adapted scores remain separate at evaluation.
"""
_base_ = '/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/third_party/research_refs_phase4n/OVTR/ovtr/config/ovtr_lite_train_val.py'
import os

# Keep the Q0 decoder contract bit-identical during the physical warm-start.
# The legacy CLIP tensors are frozen constructor inputs for Q0 score
# reproduction; the new TCO adapter never receives text/category features.
use_text_cross_attention = True
use_fusion_layer = False
train_with_artificial_img_seqs = True
distribution_based_sampling = False
initial_grad = False
backbone_freeze_keywords = None
data = dict(
    train=dict(ann_file='/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/third_party/research_refs_phase4n/OVTR/data/lvis_clear_75_60.json'),
    val=dict(ann_file='/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/third_party/research_refs_phase4n/OVTR/data/validation_ours_v1.json', img_prefix='/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/third_party/research_refs_phase4n/OVTR/data/TAO/'),
    test=dict(ann_file='/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/third_party/research_refs_phase4n/OVTR/data/validation_ours_v1.json', img_prefix='/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/third_party/research_refs_phase4n/OVTR/data/TAO/'),
)
