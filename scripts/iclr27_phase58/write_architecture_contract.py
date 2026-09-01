#!/usr/bin/env python3
"""Write the Phase59 pixel-level causal architecture contract."""
import json, os, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/iclr27_phase58/audit/architecture_contract.json'
DOC=ROOT/'docs/iclr27_phase58/PHASE58_TRUE_PIXEL_END_TO_END_ARCHITECTURE.md'
contract={
 'phase':59,
 'name':'pixel_causal_trackocd',
 'input':'current RGB frame and strictly earlier RGB prefix only',
 'graph':['RGB','class_agnostic_objectness_bbox_head','persistent_physical_query','causal_association','track_lifecycle','causal_track_encoder','raw_preserving_768d_state','prior_only_support_bank','cross_video_correspondence','semantic_state_memory','commit_defer_reset','persistent_commit_ct_and_mot_safety'],
 'proposal':{'outputs':['grid_objectness','normalized_bbox_delta','quality'],'class_agnostic':True,'gt_used_only_for_loss':True,'frozen_passthrough':False},
 'physical':{'outputs':['birth','continue','terminate','association'],'physical_id':'bookkeeping_only','semantic_can_mutate_physical_id':False},
 'track_encoder':{'inputs':['visual_feature','bbox_geometry','motion','track_age','proposal_quality','association_confidence','causal_history'],'forbidden':['future_frame','future_track','category_text','category_id_feature','semantic_id','physical_id_feature','controller_action','held_gt']},
 'semantic':{'anchor_dim':768,'raw_fallback':'exact normalized current state when support invalid/missing','support_update':'bounded residual/evidence only','state_fields':['evidence','persistence','uncertainty','contradiction_history','support_reliability','last_causal_update']},
 'correspondence':{'inputs':['track_768d','geometry','history','support_quality','causal_timestamp'],'losses':['multi_positive','hard_negative','prefix_consistency']},
 'controller':{'actions':['COMMIT','DEFER','RESET_REJECT'],'causal':'evidence accumulation; no single-frame threshold','inputs':['correspondence_evidence','support_quality','persistence','contradiction_history','uncertainty','physical_mot_safety']},
 'training_losses':{'objectness':1.0,'bbox_giou':1.0,'association':1.0,'lifecycle':0.25,'continuity':0.5,'temporal_representation':0.25,'correspondence':1.0,'multi_positive':1.0,'hard_negative':1.0,'prefix_consistency':0.25,'state_persistence':0.5,'commit_defer':0.5,'persistent_commit_proxy':0.25,'mot_safety':0.5,'raw_preservation':0.25},
 'training_stages':['54A detector_physical_warm','54B causal_track_representation','54C correspondence_state','54D commit_defer','54E joint_finetune'],
 'dimensions':{'backbone_channels':128,'track_embedding':256,'semantic_state':768,'controller_logits':3},
 'protocol':{'folds':4,'prefixes':[1,2,4,8,16],'positive_events':76,'negative_events':76,'seed_base':575700,'sealed_read':False},
 'external_weights':{'used':False,'reason':'audited OVTR/MOTIP-2 weights require text/category or physical-ID semantics; no downloaded weight is needed for this bounded pixel baseline'},
}
OUT.parent.mkdir(parents=True,exist_ok=True); tmp=OUT.with_suffix('.tmp'); tmp.write_text(json.dumps(contract,indent=2)+'\n'); os.replace(tmp,OUT)
lines=['# Phase59 True Pixel End-to-End Architecture','',
'The Phase57 audit found no external checkpoint that simultaneously satisfies class-agnostic proposal, causal physical lifecycle, prior-only cross-video semantic correspondence and no-ID/no-text inference. This contract therefore uses a small local RGB model. It is a real pixel input graph, not the Phase26 feature-row stream; all old streams remain read-only comparators.', '',
'## Causal graph','', '`RGB current/history → class-agnostic objectness+box grid → persistent physical query/association → birth/continue/terminate lifecycle → causal track encoder → raw-preserving 768-D semantic state → prior-only support bank → cross-video correspondence → semantic StateMemory → Commit/Defer/Reset → persistent Commit-CT + MOT safety`','',
'## Contracts','',
'- The detector consumes RGB pixels and produces a dense class-agnostic candidate set (objectness, box deltas, quality). GT boxes are detached targets for training only; no selected-positive row or Phase26 proposal is required.',
'- Physical IDs exist only in internal bookkeeping. Semantic tensors cannot mutate physical IDs. Association/lifecycle losses are computed in the same forward graph as semantic/controller losses.',
'- Track features use visual activations, geometry, motion, age, quality and causal history. Category names/text, category/semantic/physical IDs, future frames/tracks, controller actions and held labels are excluded from tensors.',
'- The 768-D semantic state retains the visual anchor. A valid prior support can add a bounded residual/evidence update; missing or invalid support returns the normalized raw state exactly.',
'- The controller accumulates evidence and may COMMIT, DEFER or RESET_REJECT. It sees uncertainty, contradiction and physical safety state, not a single-frame category threshold.', '',
'## Loss and optimization audit','', 'All weights are fixed in `architecture_contract.json` before training: objectness 1, box/GIoU 1, association 1, lifecycle .25, continuity .5, temporal .25, correspondence/multi-positive/hard-negative 1, prefix .25, state .5, commit .5, persistent proxy .25, MOT safety .5 and raw preservation .25. The training report records each term, gradient norms and whether proposal/physical and semantic/controller parameters receive non-zero gradients.', '',
'## Resource and data boundary','', 'TRAIN-only four-fold video/category-disjoint manifests from Phase58 are used. No DEV+, Q1, public-new-model or sealed labels are read. At most four GPUs are used, one fold per GPU, with bounded workers and atomic checkpoints/markers.', '',
'## Reproduction','', '`python scripts/iclr27_phase58/write_architecture_contract.py` writes the JSON atomically. The model implementation is `src/iclr27_phase58/pixel_model.py`; the bounded worker and supervisor are under `scripts/iclr27_phase60/`.']
DOC.parent.mkdir(parents=True,exist_ok=True); DOC.write_text('\n'.join(lines)+'\n')
