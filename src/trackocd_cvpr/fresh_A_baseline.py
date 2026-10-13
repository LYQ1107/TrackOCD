"""Fresh fit5 A1/A2 numeric/sampling prerequisite, NOT trained evidence.

Reuse original project architectures/loss, never old adapters/checkpoints,
old15class tables or loaders. Caller MUST gate source/authority/complete current
fit cache before passing FitTracks. External Known slots/video correspondence
only form cross-video batches/loss; forward receives observed numeric tensors.
Different256D output is not a768D Raw residual: compare Gram geometry, not
undefined cross-space cosine. No optimizer, dev/final labels or asset reads here.
"""
from dataclasses import dataclass
import numpy as np
import torch
from src.trackocd_core.train_first_representation import make_model
from src.trackocd_core.representation import cross_video_category_loss

FAMILY_GEOMETRY = {'A1_ADAPTER':0.,'A1_GEOMETRY_1':1.,'A1_GEOMETRY_5':5.,
    'A2_EVIDENCE':0.,'A2_CAPACITY_CONTROL':0.}
FAMILIES = tuple(FAMILY_GEOMETRY)


def fresh_model(family, seed):
    if family not in FAMILIES or type(seed) is not int or seed not in (1027,1028,1029):
        raise ValueError('Fixed fresh baseline family and independent training seed')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = make_model(family)
    model._fresh_fit5_baseline_family = family  # external loss framing, not forward input
    return model


def _validated(prefix):
    visual, quality, elapsed = prefix.visual, prefix.quality, prefix.elapsed
    boxes = prefix.boxes
    n = len(quality)
    if (visual.ndim != 2 or visual.shape != (n,768) or not n
            or quality.shape != (n,) or elapsed.shape != (n,)
            or visual.dtype != torch.float32 or quality.dtype != torch.float32
            or visual.device.type != 'cpu' or quality.device.type != 'cpu' or elapsed.device.type != 'cpu'
            or elapsed.dtype not in (torch.int32,torch.int64)
            or boxes.shape != (n,4) or boxes.dtype != torch.float32 or boxes.device.type != 'cpu'
            or not torch.isfinite(boxes).all()
            or not torch.isfinite(visual).all() or not torch.isfinite(quality).all()
            or torch.any(visual.norm(dim=1) <= 1e-12) or torch.any((quality<0)|(quality>1))
            or elapsed[0] != 0 or torch.any(elapsed[1:] < elapsed[:-1])):
        raise ValueError('Observed nonempty CPU Raw/quality/integer elapsed only, no padding/future')
    return visual,quality,elapsed


def forward_ragged(model, prefixes):
    if not prefixes: raise ValueError('No dropped empty supervised batch')
    groups = {}; outputs = {}
    for i,prefix in enumerate(prefixes):
        value = _validated(prefix); groups.setdefault(len(value[1]),[]).append((i,value))
    for members in groups.values():
        tensors = [torch.stack([value[k] for _,value in members]) for k in range(3)]
        output = model(tensors[0],tensors[1],tensors[2].float())['embedding']
        if (output.shape != (len(members),256) or not torch.isfinite(output).all()
                or torch.any(output.norm(dim=1)<=1e-12)):
            raise ValueError('Original finite256D baseline output, not a Raw residual')
        for j,(index,_) in enumerate(members): outputs[index] = output[j]
    return torch.stack([outputs[i] for i in range(len(prefixes))])


def supervised_batch_loss(model, prefixes, fit_slots, videos, canonical_groups, *, temperature=.1, individual_reference=False):
    # Supervision outside model.forward, from caller-gated existing fit5 only.
    if (len(prefixes) != 10 or fit_slots.shape != (10,) or videos.shape != (10,)
            or canonical_groups.shape != (10,) or any(x.dtype != torch.int64 for x in (fit_slots,videos,canonical_groups))
            or any(x.device.type != 'cpu' for x in (fit_slots,videos,canonical_groups))
            or set(fit_slots.tolist()) != set(range(5))
            or any(int((fit_slots==c).sum()) != 2 for c in range(5))
            or temperature != .1):
        raise ValueError('Exactly five existing fit slots/two cross-video groups per class')
    family = getattr(model,'_fresh_fit5_baseline_family',None)
    if family not in FAMILIES: raise ValueError('Fresh registered baseline, no old checkpoint alias')
    embedding = (torch.stack([forward_ragged(model,[p])[0] for p in prefixes])
                 if individual_reference else forward_ragged(model,prefixes))
    loss = cross_video_category_loss(embedding,fit_slots,videos,canonical_groups,temperature)
    if FAMILY_GEOMETRY[family]:
        teacher = []
        for prefix in prefixes:
            visual,quality,_ = _validated(prefix)
            weights = quality if quality.sum()>1e-12 else torch.ones_like(quality)
            mean = (visual*weights[:,None]).sum(0)/weights.sum()
            if mean.norm() <= 1e-12: raise ValueError('Undefined Raw teacher, no silent row drop')
            teacher.append(mean/mean.norm())
        raw = torch.stack(teacher).detach()
        loss = loss + FAMILY_GEOMETRY[family]*torch.nn.functional.mse_loss(embedding@embedding.T,raw@raw.T)
    return loss,embedding


@dataclass(frozen=True)
class BatchPlan:
    # PRIVATE planning correspondence, never public row/GT/route exports.
    tracks: tuple
    fit_slots: tuple
    videos: tuple
    canonical_groups: tuple
    observed_counts: tuple


class Fit5BatchSampler:
    def __init__(self, tracks, seed):
        if seed not in (1027,1028,1029): raise ValueError('Fixed batch scheduling seed')
        aliases = {}; physical_known = 0; masked = 0
        for track in tracks:
            if track.loss_target is None: masked += 1; continue
            slot = track.loss_target
            if type(slot) is not int or slot not in range(5):
                raise ValueError('Only projected existing fit5 Known targets, never new labels')
            physical_known += 1; prefix = track.visible(track.count()); _validated(prefix)
            start,end = track.arrays['offsets'][track.position:track.position+2]
            frames = np.asarray(track.arrays['frame_index'][start:end])
            if frames.shape != (track.count(),) or frames.dtype.kind not in 'iu':
                raise ValueError('Exact observed absolute-frame alias provenance')
            # Complete immutable bytes, not digest-only equality or physical ID.
            arrays = (prefix.visual,prefix.boxes,prefix.quality,prefix.elapsed)
            key = (track.video,frames.dtype.str,frames.tobytes(),tuple(
                (tuple(a.shape),str(a.dtype),a.numpy().tobytes()) for a in arrays))
            if key in aliases and aliases[key].loss_target != slot:
                raise ValueError('Conflicting existing Known targets on exact Raw alias, no invented target')
            aliases.setdefault(key,track)
        self._groups = {c:{} for c in range(5)}
        for ordinal,(key,track) in enumerate(sorted(aliases.items(),key=lambda row:row[0])):
            self._groups[track.loss_target].setdefault(track.video,[]).append((ordinal,track))
        if any(len(groups)<2 for groups in self._groups.values()):
            raise ValueError('Every fit class needs actual cross-video positives; no fabricated batch')
        self._rng = np.random.default_rng(seed+11027)
        self._next_step = 0
        self.scope = {'physical_tracks':len(tracks),'eligible_known_physical_tracks':physical_known,
            'masked_no_fit5_target_tracks':masked,'canonical_known_groups':len(aliases),
            'exact_same_video_Raw_aliases_deduplicated':physical_known-len(aliases),
            'supervised_classes':5,'all_input_physical_tracks_accounted':True}

    def draw(self, step):
        if type(step) is not int or not 0 <= step < 200 or step != self._next_step:
            raise ValueError('Exact sequential Fixed200 calendar, no repeated/skipped batches')
        cap = (1,2,4,8,16)[step%5]; selected=[]; slots=[]; videos=[]; canonical=[]; counts=[]
        for slot,groups in self._groups.items():
            for video in self._rng.choice(sorted(groups),2,replace=False):
                candidates=groups[int(video)]; ordinal,track=candidates[int(self._rng.integers(len(candidates)))]
                selected.append(track); slots.append(slot); videos.append(int(video)); canonical.append(ordinal)
                counts.append(min(cap,track.count()))
        self._next_step += 1
        return BatchPlan(tuple(selected),tuple(slots),tuple(videos),tuple(canonical),tuple(counts))


def cross_space_Gram_drift(raw, adapted):
    if (raw.ndim != 2 or adapted.ndim != 2 or len(raw)!=len(adapted)
            or raw.shape[1]!=768 or adapted.shape[1]!=256 or not len(raw)
            or not np.isfinite(raw).all() or not np.isfinite(adapted).all()):
        raise ValueError('Same full physical observations, different legitimate feature spaces')
    r=np.asarray(raw,np.float64); a=np.asarray(adapted,np.float64)
    if np.any(np.linalg.norm(r,axis=1)<=1e-12) or np.any(np.linalg.norm(a,axis=1)<=1e-12):
        raise ValueError('Nonzero full physical embeddings, no dropping')
    r=r/np.linalg.norm(r,axis=1,keepdims=True); a=a/np.linalg.norm(a,axis=1,keepdims=True)
    delta=a@a.T-r@r.T
    return {'full_pairwise_Gram_MSE':float(np.square(delta).mean()),
        'full_pairwise_Gram_max_abs_difference':float(np.abs(delta).max()),
        'pointwise_Raw_cosine_not_defined_across_768D_and_256D':True}
