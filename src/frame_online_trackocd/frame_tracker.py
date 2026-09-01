"""FrameOnlineTracker: exact IDOL association with an optional soft
semantic-consistency term injected into the score matrix.

B0/B1 use sem_cost=None and must reproduce the original SimOWT tracker
byte-for-byte.  B2 passes a (N_raw, M_memo) consistency matrix; the
override filters it together with mask-nms-pre detections and adds
`lambda_s * sem_cost` to the bisoftmax scores before greedy assignment.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
SIMOWT_ROOT = ROOT / "third_party" / "SimOWT"
sys.path.insert(0, str(SIMOWT_ROOT))
sys.path.insert(0, str(SIMOWT_ROOT / "projects"))
sys.path.insert(0, str(SIMOWT_ROOT / "detectron2"))

# The tracker module imports detectron2.structures only for type hints;
# stub it so the frozen association code can run under the project's
# torch 2.x env without installing SimOWT's compiled detectron2.
_fake_d2 = types.ModuleType("detectron2")
_fake_struct = types.ModuleType("detectron2.structures")
for _cls in ("Boxes", "ImageList", "Instances", "BitMasks"):
    setattr(_fake_struct, _cls, type(_cls, (), {}))
setattr(_fake_d2, "structures", _fake_struct)
sys.modules["detectron2"] = _fake_d2
sys.modules["detectron2.structures"] = _fake_struct

_tracker_path = SIMOWT_ROOT / "projects" / "IDOL" / "idol" / "models" / "tracker.py"
_spec = importlib.util.spec_from_file_location("simowt_tracker_4i", _tracker_path)
_tracker_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tracker_mod)
IDOL_Tracker = _tracker_mod.IDOL_Tracker
mask_nms = _tracker_mod.mask_nms
mask_iou = _tracker_mod.mask_iou


class FrameOnlineTracker(IDOL_Tracker):
    """IDOL tracker with an optional semantic-cost matrix."""

    def __init__(self, *args, association_logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.association_logger = association_logger
        self.video_id = None

    def match(self, bboxes, labels, masks, track_feats, frame_id, indices,
              datasets_ori=[], ori_size=[], sem_cost=None, lambda_s=0.0):
        embeds = track_feats

        # mask nms
        valids = mask_nms(masks, bboxes[:, -1], None, self.nms_thr_pre)
        self.last_raw_positions = [i for i, v in enumerate(valids) if v]
        mask_new_indices = torch.tensor(indices)[valids].tolist()
        indices = mask_new_indices
        bboxes = bboxes[valids, :]
        labels = labels[valids]
        masks = masks[valids]
        embeds = embeds[valids, :]
        if sem_cost is not None:
            sem_cost = sem_cost[valids]
        ids = torch.full((bboxes.size(0),), -2, dtype=torch.long)

        if bboxes.size(0) > 0 and not self.empty:
            (memo_bboxes, memo_labels, memo_embeds, memo_ids,
             memo_vs, memo_long_embeds, memo_long_score,
             memo_exist_frame) = self.memo
            memo_exist_frame = memo_exist_frame.to(memo_embeds)
            memo_ids = memo_ids.to(memo_embeds)
            if self.match_metric == 'longrang':
                feats = torch.mm(embeds, memo_embeds.t())
            elif self.match_metric == 'bisoftmax':
                feats = torch.mm(embeds, memo_embeds.t())
                d2t_scores = feats.softmax(dim=1)
                t2d_scores = feats.softmax(dim=0)
                scores = (d2t_scores + t2d_scores) / 2
            elif self.match_metric == 'softmax':
                feats = torch.mm(embeds, memo_embeds.t())
                scores = feats.softmax(dim=1)
            elif self.match_metric == 'cosine':
                scores = torch.mm(
                    F.normalize(embeds, p=2, dim=1),
                    F.normalize(memo_embeds, p=2, dim=1).t())
            else:
                raise NotImplementedError

            appearance_scores = scores.clone()
            if sem_cost is not None and lambda_s > 0.0:
                scores = scores + lambda_s * sem_cost.to(scores)

            conf_list = []
            for i in range(bboxes.size(0)):
                if self.frame_weight:
                    non_backs = (memo_ids > -1) & (scores[i, :] > 0.5)
                    if (scores[i, non_backs] > 0.5).sum() > 1:
                        wighted_scores = scores.clone()
                        frame_weight = memo_exist_frame[
                            scores[i, :][memo_ids > -1] > 0.5]
                        wighted_scores[i, non_backs] = (
                            wighted_scores[i, non_backs] * frame_weight)
                        wighted_scores[i, ~non_backs] = (
                            wighted_scores[i, ~non_backs] *
                            frame_weight.mean())
                        conf, memo_ind = torch.max(wighted_scores[i, :], dim=0)
                    else:
                        conf, memo_ind = torch.max(scores[i, :], dim=0)
                else:
                    conf, memo_ind = torch.max(scores[i, :], dim=0)
                id = memo_ids[memo_ind]
                conf_list.append(conf)
                if conf > self.match_score_thr:
                    if id > -1:
                        ids[i] = id
                        scores[:i, memo_ind] = 0
                        scores[i + 1:, memo_ind] = 0
                if self.association_logger is not None:
                    ap_best = int(torch.argmax(appearance_scores[i, :]))
                    fn_best = int(torch.argmax(scores[i, :]))
                    raw_idx = self.last_raw_positions[i] \
                        if i < len(self.last_raw_positions) else -1
                    self.association_logger.log(
                        self.video_id, frame_id, raw_idx,
                        ap_best,
                        float(appearance_scores[i, ap_best]),
                        fn_best, float(scores[i, fn_best]),
                        float(scores[i, ap_best] -
                              appearance_scores[i, ap_best]),
                        float(scores[i, fn_best] -
                              appearance_scores[i, fn_best]),
                        int(memo_ind), int(id), int(ids[i]), float(conf),
                        int(memo_ids[ap_best]),
                        int(memo_ids[fn_best]),
                        float(scores[i, memo_ind] -
                              appearance_scores[i, memo_ind]))
            new_inds = (ids == -2) & (bboxes[:, 4] > self.addnew_score_thr).cpu()
            num_news = new_inds.sum()
            ids[new_inds] = torch.arange(
                self.num_tracklets, self.num_tracklets + num_news,
                dtype=torch.long)
            self.num_tracklets += num_news

            unselected_inds = torch.nonzero(ids == -2, as_tuple=False).squeeze(1)
            if len(unselected_inds) > 0:
                a = 1
            mask_ious = mask_iou(
                masks[unselected_inds].sigmoid() > 0.5,
                masks.permute(1, 0, 2, 3).sigmoid() > 0.5)
            for i, ind in enumerate(unselected_inds):
                if (mask_ious[i, :ind] < self.nms_thr_post).all():
                    ids[ind] = -1
            self.update_memo(ids, bboxes, embeds, labels, frame_id,
                             datasets_ori, ori_size, conf_list)

        elif self.empty:
            conf_list = bboxes[:, 4]
            for idx, item1 in enumerate(conf_list):
                conf_list[idx] = 0.7501
            init_inds = (ids == -2) & (bboxes[:, 4] > self.init_score_thr).cpu()
            num_news = init_inds.sum()
            ids[init_inds] = torch.arange(
                self.num_tracklets, self.num_tracklets + num_news,
                dtype=torch.long)
            self.num_tracklets += num_news
            unselected_inds = torch.nonzero(ids == -2, as_tuple=False).squeeze(1)
            mask_ious = mask_iou(
                masks[unselected_inds].sigmoid() > 0.5,
                masks.permute(1, 0, 2, 3).sigmoid() > 0.5)
            for i, ind in enumerate(unselected_inds):
                if (mask_ious[i, :ind] < self.nms_thr_post).all():
                    ids[ind] = -1
            self.update_memo(ids, bboxes, embeds, labels, frame_id,
                             datasets_ori, ori_size, conf_list)

        return bboxes, labels, ids, indices, masks
