"""Two predeclared Raw score-fitting controls, no private input access here.

ONLY callers with fresh registered authority may supply lawful fit/policy
targets. Binary target means membership in the FIVE named offline Known
classes, not object foreground, background, arbitrary Novel truth or purity.
Class-macro weighting defines a conditional calibration task, not the real
deployment prior. Anonymous Existing gate remains fixed and uncalibrated.
"""
from collections import Counter, defaultdict
import hashlib
import numpy as np
import torch
from torch import nn
from .candidate_score_contract import build_candidates
from .candidate_training_loss import lawful_label
from .candidate_scientific_training import class_macro_loss
from .causal_observation_router import CurrentObservationRouter
from .policy_training_data import policy_current_frames

INITIAL = np.array([1., -.65], dtype=np.float64)
RIDGE = 1e-4


class NamedKnownLogit(nn.Module):
    """Float64 prevents changing the inherited float32-cosine/.65 boundary."""
    def __init__(self):
        super().__init__()
        self.affine = nn.Parameter(torch.tensor(INITIAL.copy(), dtype=torch.float64))

    def forward(self, max_known_cosine):
        x = torch.as_tensor(max_known_cosine, dtype=torch.float64)
        if (not torch.isfinite(x).all() or torch.any(x < -1) or torch.any(x > 1)
                or not torch.isfinite(self.affine).all() or self.affine[0] < 0):
            raise ValueError('Finite angular scalar and monotone membership map only')
        return self.affine[0]*x + self.affine[1]


def parameter_sha256(model):
    return hashlib.sha256(model.affine.detach().cpu().numpy().astype('<f8').tobytes()).hexdigest()


def calibrated_raw_scores(features, context, *, model, tau_existing=.55):
    """Original Existing>=Known arbitration, then fitted named-set gate.

    No label/ID/semantic/foreground inputs. Both initial threshold and tie
    behavior match the original Raw Top8 rule; fixed Existing gate is NOT fit.
    """
    if model.training or any(p.requires_grad for p in model.parameters()) or tau_existing != .55:
        raise ValueError('Frozen own score-fitting control and original fixed Existing gate')
    known = features[:,1] == 1; existing = ~known; cosine = features[:,0]
    if not known.any():
        raise ValueError('Actual offline NamedKnown proposals required')
    bk = float(cosine[known].max()); be = float(cosine[existing].max()) if existing.any() else -1.
    scores = np.full(len(features)+2, -2., dtype=np.float32)
    if existing.any() and be >= tau_existing and be >= bk:
        scores[np.flatnonzero(existing & (cosine == be))] = 1.
    else:
        with torch.inference_mode(): positive = float(model(bk)) >= 0
        if positive: scores[np.flatnonzero(known & (cosine == bk))] = 1.
        else: scores[-2] = 1.
    return scores


def group_label(projection, video, aliases):
    external = {(v,str(r)):label for (v,r),label in projection.targets.items()}
    values = [lawful_label(external.get((video,r),('no_retained_semantic_truth',None)))
              for r in aliases]
    return values[0] if values and all(v is not None for v in values) and len(set(values)) == 1 else None


def max_known_cosine(view, known):
    # Same actual float32 angular proposal kernel; anonymous history cannot
    # change NamedKnown features or the maximum over the fixed fit-only prior.
    features, _ = build_candidates(view,known,{},K=8).numeric_inputs()
    return float(features[features[:,1] == 1,0].max())


def collect_full_policy_samples(videos, projection, known, guard):
    """PRIVATE scalar/target/weight table, full clock incl masked observations.

    No samples are selected by dev, logits, pseudo opportunity or GT history.
    Query roles enter loss ONLY; even conflicting Known aliases stay masked.
    """
    if not videos or len({v for v,_ in videos}) != len(videos):
        raise ValueError('Complete unique policy videos, no repeated data-padding')
    grouped = defaultdict(list); counts = Counter()
    for video, arrays in videos:
        router = CurrentObservationRouter()
        for frame, rows in policy_current_frames(arrays):
            guard(); batch = router.ingest(frame,rows)
            counts['metadata_frames'] += 1; counts['empty_frames'] += int(not rows)
            counts['physical_observations'] += len(rows)
            for view, aliases in zip(batch.model_views,batch.routing_groups):
                label = group_label(projection,video,aliases); counts['canonical_groups'] += 1
                if label is None: counts['masked_groups'] += 1; continue
                grouped[label].append(max_known_cosine(view,known)); counts[label[0]+'_groups'] += 1
    if not grouped or not any(k[0]=='Known' for k in grouped) or not any(k[0]=='policy_pseudo' for k in grouped):
        raise ValueError('Both lawful Known and held-out Known-derived policy roles required')
    table = sorted((score,float(label[0]=='Known'),1./(len(grouped)*len(scores)))
                   for label,scores in grouped.items() for score in scores)
    private = np.asarray(table,dtype='<f8')
    return private, {**dict(counts), 'supervised_classes':len(grouped),
        'private_scalar_target_weight_table_sha256':hashlib.sha256(private.tobytes()).hexdigest(),
        'unknown_as_negative_BG_Novel_or_individual_anonymous_target':False,
        'class_macro_conditional_task_not_deployment_prior':True}


def objective_and_gradient(theta, private, ridge=RIDGE):
    from scipy.special import expit
    if ridge != RIDGE or private.ndim != 2 or private.shape[1] != 3 or not np.isfinite(private).all():
        raise ValueError('Fixed regularizer and lawful private scalar table')
    theta = np.asarray(theta,dtype=np.float64)
    if theta.shape != (2,) or not np.isfinite(theta).all() or theta[0] < 0:
        raise ValueError('Finite two-parameter monotone calibration control')
    x,y,w = private.T
    if (np.any(np.abs(x)>1) or not np.isin(y,[0.,1.]).all() or np.any(w<=0)
            or not np.isclose(w.sum(),1.,rtol=0,atol=1e-12)):
        raise ValueError('Masked unknowns cannot be turned into binary negatives')
    z = theta[0]*x+theta[1]; residual = expit(z)-y; delta = theta-INITIAL
    loss = np.sum(w*(np.logaddexp(0,z)-y*z))+.5*RIDGE*np.dot(delta,delta)
    gradient = np.array([np.sum(w*residual*x),np.sum(w*residual)])+RIDGE*delta
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise ValueError('Finite actual analytic objective/gradient')
    return float(loss),gradient


def membership_diagnostics(model, private):
    """Public conditional-lawful-role NLL/Brier/ECE, not full deployment truth.

    ECE uses fixed 10 equal-width probability bins; no diagnostic picks model,
    threshold, dataset, role, bin count or best checkpoint. Unknown groups are
    physically retained by the collector but absent from these binary targets.
    """
    objective_and_gradient(model.affine.detach().cpu().numpy(),private)
    from scipy.special import expit
    theta = model.affine.detach().cpu().numpy(); x,y,w = private.T
    logits = theta[0]*x+theta[1]; p = expit(logits)
    nll = np.logaddexp(0,logits)-y*logits; brier = (p-y)**2
    indices = np.minimum((p*10).astype(np.int64),9); bins = []
    ece = weighted_ece = 0.
    for i in range(10):
        take = indices == i; n = int(take.sum()); mass = float(w[take].sum())
        gap = abs(float(p[take].mean()-y[take].mean())) if n else 0.
        weighted_gap = abs(float(w[take]@(p[take]-y[take])/mass)) if mass else 0.
        ece += n/len(p)*gap; weighted_ece += mass*weighted_gap
        bins.append({'bin_index':i,'lawful_group_count':n,'class_macro_mass':mass})
    return {'lawful_groups':len(p),'Known_groups':int(y.sum()),'policy_pseudo_groups':int((1-y).sum()),
        'micro_NLL':float(nll.mean()),'class_macro_NLL':float(w@nll),
        'micro_Brier':float(brier.mean()),'class_macro_Brier':float(w@brier),
        'fixed_10_bin_micro_ECE':ece,'fixed_10_bin_class_macro_ECE':weighted_ece,'bins':bins,
        'declared_role_conditioning_and_class_weighting_not_full_deployment_prior':True,
        'calibration_quality_proved_or_model_selected_from_diagnostics':False}


def full_policy_fit(private, *, iteration_budget, function_budget, guard):
    """Single deterministic L-BFGS-B fit, no fake independent training seeds.

    Full policy exposure differs from C Known-anchor calendars. Solver success
    and KKT diagnostics are returned honestly, never implied from finite weights.
    """
    from scipy.optimize import minimize
    if (iteration_budget,function_budget) not in ((50,250),(200,1000)):
        raise ValueError('Only fixed pilot/formal bounded solver settings')
    calls = 0
    def objective(theta):
        nonlocal calls
        guard(); calls += 1
        if calls > function_budget: raise RuntimeError('Owned solver function budget')
        return objective_and_gradient(theta,private)
    result = minimize(objective,INITIAL.copy(),method='L-BFGS-B',jac=True,
        bounds=[(0.,None),(None,None)],callback=lambda x:guard(),
        options={'maxiter':iteration_budget,'maxfun':function_budget,'maxls':20,
                 'ftol':1e-12,'gtol':1e-8,'maxcor':10})
    guard(); model = NamedKnownLogit()
    with torch.no_grad(): model.affine.copy_(torch.tensor(result.x,dtype=torch.float64))
    loss,gradient = objective_and_gradient(result.x,private)
    projected = gradient.copy()
    if result.x[0] == 0 and gradient[0] > 0: projected[0] = 0
    return model.eval().requires_grad_(False),{
        'optimizer':'L-BFGS-B','iterations':int(result.nit),'function_evaluations':calls,
        'solver_success':bool(result.success),'solver_status':int(result.status),
        'final_regularized_class_macro_NLL':loss,
        'projected_gradient_infinity_norm':float(np.abs(projected).max()),
        'trainable_parameters':2,'parameter_sha256':parameter_sha256(model),
        'training_seeds_or_calendar_matched_to_C':False,
        'calibration_deployment_posterior_foreground_or_Goal_accepted':False}


def calendar_matched_fit(videos,projection,known,seed,*,frozen_calendar,episodes,updates_per_episode,
                         checkpoint_steps,guard,progress,checkpoint):
    """Same actual C frame calendar/budget, NOT identical target information."""
    if (type(seed) is not int or seed not in (1027,1028,1029) or episodes not in (1,4)
            or updates_per_episode != 50 or not frozen_calendar
            or frozen_calendar['budget_each'] != updates_per_episode
            or len(frozen_calendar['selected_frames']) != episodes
            or checkpoint_steps != ([50] if episodes==1 else [20,50,100,200])):
        raise ValueError('Explicit shared Known-rank plan and exact pilot/formal budgets')
    rng = np.random.default_rng(seed); model = NamedKnownLogit(); before = parameter_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    calendar = hashlib.sha256(); curves = []; summaries = []; private_predictions = []
    for episode in range(episodes):
        order = rng.permutation(len(videos)).tolist(); selected = set(frozen_calendar['selected_frames'][episode])
        if order != frozen_calendar['orders'][episode] or len(selected) != 50:
            raise ValueError('Same actual frozen frame plan, no substitute anchors')
        steps = 0; counts = Counter()
        for position,ordinal in enumerate(order):
            video,arrays = videos[ordinal]; router = CurrentObservationRouter()
            for frame,rows in policy_current_frames(arrays):
                guard(); batch = router.ingest(frame,rows); losses = defaultdict(list); families = Counter(); predictions = []
                counts['metadata_frames'] += 1; counts['empty_frames'] += int(not rows); counts['physical_observations'] += len(rows)
                for view,aliases in zip(batch.model_views,batch.routing_groups):
                    score = max_known_cosine(view,known); logit = model(score); predictions.append(float(logit.detach()))
                    label = group_label(projection,video,aliases)
                    if label is None: counts['masked_groups'] += 1; continue
                    target = torch.tensor(float(label[0]=='Known'),dtype=torch.float64)
                    losses[label].append(torch.nn.functional.binary_cross_entropy_with_logits(logit,target)); families[label[0]] += 1
                private_predictions.append((video,frame,tuple(predictions)))
                if (episode,position,frame) in selected:
                    if not families['Known']: raise ValueError('Frozen Known anchor unavailable')
                    loss = class_macro_loss(losses); optimizer.zero_grad(set_to_none=True); loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(loss) or not torch.isfinite(norm) or norm <= 0:
                        raise ValueError('Finite nonzero actual loss gradient required')
                    optimizer.step()
                    with torch.no_grad(): model.affine[0].clamp_(min=0.)
                    steps += 1; step = episode*50+steps
                    calendar.update(np.asarray([episode,position,frame],dtype='<i8').tobytes())
                    curves.append({'step':step,'episode':episode,'episode_optimizer_step':steps,
                        'class_macro_BCE':float(loss.detach()),'gradient_norm_before_clip':float(norm),
                        'lawful_group_counts':dict(families),'shared_calendar_sha256':calendar.hexdigest()})
                    if step in checkpoint_steps: checkpoint(model,optimizer,step,rng.bit_generator.state)
                    progress(curves[-1])
                guard()
        if steps != 50: raise ValueError('No partial calendar budget acceptance')
        summaries.append(dict(counts))
    if calendar.hexdigest() != frozen_calendar['selected_calendar_sha256'] or parameter_sha256(model)==before:
        raise ValueError('Actual entire frozen calendar and fresh real updates')
    optimizer.zero_grad(set_to_none=True)
    return model.eval().requires_grad_(False),optimizer,{
        'optimizer':'AdamW','optimizer_steps':len(curves),'trainable_parameters':2,
        'initialization_sha256':before,'endpoint_parameter_sha256':parameter_sha256(model),
        'actual_shared_calendar_sha256':calendar.hexdigest(),'episodes':summaries,'loss_curve':curves,
        'same_frames_not_same_target_information_capacity_or_optimizer_function_class':True,
        'anonymous_action_targets_GT_memory_edits_or_unknown_negative_added':False,
        'calibration_deployment_posterior_foreground_or_Goal_accepted':False},tuple(private_predictions)
