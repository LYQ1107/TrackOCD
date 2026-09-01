"""Phase70 semantic/OCD integration config.

The Phase69 OVTR class-agnostic physical checkpoint is the immutable
initialization.  This config disables text cross-attention and keeps the
original causal LVIS sequence source; all Phase70 outputs live outside older
phase namespaces.
"""
_base_ = '/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/third_party/research_refs_phase4n/OVTR/ovtr/config/ovtr_lite_train_val.py'
use_text_cross_attention = False
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
