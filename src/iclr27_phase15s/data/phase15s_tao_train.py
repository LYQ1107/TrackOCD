"""Frozen Phase-6B inference config with only the public Phase15S annotation.

The base model/tracker configuration is imported read-only; no model or
checkpoint setting is changed.  This file lives outside the historical OVTR
tree so Phase15/15R remain immutable.
"""
_base_ = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/third_party/research_refs_phase4n/OVTR/ovtr/config/ovtr_lite_dsct6b_train_val.py"
_ann = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/iclr27_phase15s/sources/validation_public_roles.json"
_frames = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/iclr27_phase15s/sources/tao_train_frames"
data = dict(val=dict(ann_file=_ann, img_prefix=_frames),
            test=dict(ann_file=_ann, img_prefix=_frames))
modelname = "OVTR"
backbone = "resnet50"
