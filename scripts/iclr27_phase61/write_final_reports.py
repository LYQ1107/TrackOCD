#!/usr/bin/env python3
"""Render the Phase57--61 reports and machine decision from immutable metrics.

The script deliberately reads only TRAIN-contract artifacts and the frozen
post-inference event metrics.  It does not run an evaluator or access sealed
labels, and all JSON/Markdown writes use a temporary file followed by rename.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT60 = ROOT / "outputs/iclr27_phase60"
OUT61 = ROOT / "outputs/iclr27_phase61"
INV = ROOT / "outputs/iclr27_phase57/audit/supervision_inventory.json"
LEAK = ROOT / "outputs/iclr27_phase57/audit/leakage_audit.json"


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_json(path: Path, obj) -> None:
    atomic_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def sha(path: Path):
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_eval(tag: str):
    p = OUT61 / "metrics" / ("phase61_full_evaluation.json" if tag == "formal" else f"phase61_full_evaluation_{tag}.json")
    return json.loads(p.read_text())


def pct(x):
    return f"{100.0*float(x):.2f}%"


def f4(x):
    return f"{float(x):.4f}"


def detector_rows(evals):
    out = []
    for tag, d in evals.items():
        for i, m in enumerate(d["detector_by_fold"]):
            out.append((tag, i, m))
    return out


def render_training(inv, evals):
    rows = []
    for tag, d in evals.items():
        vals = d["detector_by_fold"]
        rows.append(
            f"| {tag} | " + " | ".join(f"{100*m['top20_recall_iou_0.5']:.2f}%" for m in vals) +
            f" | {100*sum(m['top20_recall_iou_0.5'] for m in vals)/4:.2f}% | " +
            " | ".join(f"{100*m['top20_recall_iou_0.3']:.2f}%" for m in vals) + " |"
        )
    fold_loss=[]
    for f in range(4):
        p=OUT60/f"metrics_phase60_repair3_f{f}.json"
        d=json.loads(p.read_text()); z=d["loss_log"][-1]
        fold_loss.append(f"| {f} | {d['steps']} | {f4(z['total'])} | {f4(z['detector']['objectness'])} | {f4(z['detector']['bbox'])} | {f4(z['detector']['quality'])} | {f4(z['grad_norms']['proposal'])} | {f4(z['grad_norms']['physical'])} | {f4(z['grad_norms']['semantic'])} | {f4(z['grad_norms']['controller'])} | {f4(d['elapsed_sec'])} |")
    folds=[]
    for x in inv["folds"]:
        p="/".join(str(x["prefix_track_coverage"][str(k)]) for k in (1,2,4,8,16))
        folds.append(f"| {x['fold']} | {x['fit_tracks']} | {x['validation_tracks']} | {x['fit_videos']} | {x['validation_videos']} | {x['fit_categories']} | {x['cross_video_positive_pairs']} | {x['hard_negative_pairs']} | {p} |")
    return """# Phase60 Pixel End-to-End Training Report

**Date:** 2026-08-29 (Asia/Shanghai)  
**Scope:** Phase57--61 raw-frame route; all prior phase artifacts are read-only.

## Outcome

The registered raw-RGB causal architecture was implemented and trained through
the smoke, targeted and four-fold formal curriculum.  The detector never
provided a useful image-level proposal source (repair3 prefix-independent
diagnostic top-20 IoU≥0.5 recall remained below 1.2%% per fold).  The causal
event evaluator therefore records a safety-invalid near-universal COMMIT
policy.  This is a negative route; it is not a claim that visual
correspondence is impossible.

## Data and contract

The immutable TRAIN inventory contains **%d images, %d boxes, %d videos, %d
categories and %d physical tracks**; missing frame paths: **%d**.  Four fixed
video/category-disjoint folds and the leakage audit are in
`outputs/iclr27_phase57/manifests/` and `outputs/iclr27_phase57/audit/`.
Pixels remain at the existing read-only frame symlink; no feature-row NPZ was
used by the model.  Category/video/track values are loss-only metadata and are
removed before forward passes.

## Curriculum and implementation

The graph is `RGB → class-agnostic dense objectness/box/quality → physical
association/lifecycle query → causal track representation → raw-preserving
768-D semantic state → prior-only support residual → semantic evidence and
COMMIT/DEFER/RESET controller`.  It has no text, category, semantic-ID,
physical-ID feature, future frame/track or held-label input.  Proposal and
physical heads are trainable (not passthrough); every formal log records
proposal, physical, semantic and controller gradient norms.

The fixed commands were:

```bash
/home/lwr/anaconda3/envs/MOTIP2/bin/python scripts/iclr27_phase60/train_pixel_e2e.py --fold 0 --device cuda:0 --steps 20 --batch-size 2 --workers 1 --seed 575700 --tag repair3_smoke
/home/lwr/anaconda3/envs/MOTIP2/bin/python scripts/iclr27_phase60/train_pixel_e2e.py --fold 0 --device cuda:0 --steps 100 --batch-size 4 --workers 2 --seed 575700 --tag repair3_target
./scripts/iclr27_phase60/run_four_fold_supervisor.sh repair3 1000
CUDA_VISIBLE_DEVICES=4 /home/lwr/anaconda3/envs/MOTIP2/bin/python scripts/iclr27_phase61/evaluate_pixel_e2e.py --device cuda:0 --tag repair3
```

Formal training used one bounded worker per fold on physical GPUs 4, 5, 6 and
7, batch size 4, two DataLoader workers, seed `575700+fold`, FP32 (BF16 was
not needed), atomic checkpoints every 100 steps and `.launched/.done` markers.

## Fixed fold inventory

| fold | fit tracks | val tracks | fit videos | val videos | fit categories | cross-video positives | hard negatives | prefix track coverage (1/2/4/8/16) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
%s

## Detector diagnostics (bounded 1,000-row/fold sample)

Values are real post-inference box IoU scoring, not feature-row AP.  The
`formal` run is the original unbalanced detector, `repair1` balances positive
and negative dense objectness/quality BCE, `repair2` adds normalized absolute
xy coordinates to the dense head, and `repair3` supplies every legal TRAIN
annotation in each image (the sampled track remains the semantic pair).

| tag | fold0 top20@.5 | fold1 top20@.5 | fold2 top20@.5 | fold3 top20@.5 | mean top20@.5 | fold0 top20@.3 | fold1 top20@.3 | fold2 top20@.3 | fold3 top20@.3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
%s

Repair3 did not restore proposal coverage: final recalls are still below the
Phase26 observable stream and far below the 41/76 proposal ceiling.  The
candidate is consequently not eligible for correspondence or sealed model
selection.

## Formal repair3 loss/gradient evidence

| fold | updates | total | objectness | bbox | quality | proposal grad | physical grad | semantic grad | controller grad | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
%s

## Root-cause repairs (preserved evidence)

1. **Initial formal:** one positive cell versus roughly 783 negatives and a
   fixed `pos_weight=8` converged to an all-negative objectness head.  It
   produced commit on all positive and all negative events.  The complete
   `formal` checkpoints and metrics remain intact.
2. **repair1:** balanced positive/negative means corrected the class-imbalance
   objective.  Objectness became non-constant, but a convolution-only dense
   head had no absolute coordinate channel; top-20 IoU recall stayed
   0.4--1.4%%.
3. **repair2:** normalized `(x,y)` channels were added to the same head.  The
   fold smoke/target and formal runs passed, but real IoU recall remained
   0--1.2%%; fit samples were also poor, showing that coordinates alone did not
   solve the weak from-scratch detector.
4. **repair3 (last allowed cycle):** all legal TRAIN boxes in an image became
   objectness positives, while held categories/videos were filtered per fold.
   Smoke, targeted and four-fold formal runs passed with non-zero gradients;
   real proposal recall still failed.  No fourth repair, backbone lottery,
   threshold or controller change was made.

No OOM, swap pressure, duplicate formal supervisor or external-process kill
occurred.  Each worker RSS was approximately 3.66--3.78 GiB; four workers
used roughly 15 GiB against >103 GiB available RAM.  Preflight snapshots are
`outputs/iclr27_phase60/audit/resource_*_preflight.txt`; `/data1` had
107--111 GiB free.  GPUs 0--3 and 8--9 were not used.

## Gate P50/R50/C50/S50 status

* **P50 FAIL:** learned image-level proposal coverage is not sufficient and
  cannot be compared favorably with the Phase26 physical/proposal stream.
* **R50 NOT PASS:** no valid learned track-candidate export exists; retrieval
  proxy numbers from earlier feature-row phases are not attributed to this
  raw detector.  No R gate is claimed.
* **C50 FAIL:** the frozen event evaluator produced 76/76 positive commits in
  repair3 but also 76/76 negative false commits, with premature rate 1.0 and
  known/novel confusion 1.0.  The count is therefore not a valid OCD gain.
* **S50 NOT RUN:** sealed evaluation is prohibited after P50/C50 failure.

The raw-frame route is complete as a registered negative candidate.  A future
route would need an audited, sufficiently pretrained class-agnostic detector
and full-sequence MOT exporter before semantic/controller claims are possible;
this report does not authorize another unregistered lottery.

## Artifacts and integrity

Formal checkpoints/metrics/markers are under `outputs/iclr27_phase60/`; frozen
event results are under `outputs/iclr27_phase61/metrics/`.  The Phase57 official
method audit and Phase58 contract are linked from the Phase61 report.  Public,
DEV+, Q1 and sealed labels were not read.
""" % (inv['images'], inv['annotations'], inv['videos'], inv['categories'], inv['tracks'], inv['missing_frame_count'], "\n".join(folds), "\n".join(rows), "\n".join(fold_loss))


def render_phase61(inv, evals, final):
    d=evals['repair3']; c=d['causal_event_metrics']
    tags=['formal','repair1','repair2','repair3']
    causal_rows=[]
    for t in tags:
        z=evals[t]['causal_event_metrics']; causal_rows.append(f"| {t} | {z['commit_ct']}/76 | {pct(z['negative_false_commit_rate'])} | {pct(z['premature_rate'])} | {pct(z['unresolved_rate'])} | {z['category_coverage']} | {z['video_coverage']} |")
    pref=[]
    for p in (1,2,4,8,16):
        rr=[r for r in d['event_records'] if r['kind']=='positive_existing']
        sr=sum(bool(r['prefix'][str(p)]['source_reliable']) for r in rr); tr=sum(bool(r['prefix'][str(p)]['target_reliable']) for r in rr); ce=sum(bool(r['prefix'][str(p)]['event_ceiling']) for r in rr)
        commits=sum(r['prefix'][str(p)]['action']=='COMMIT' for r in rr); defer=sum(r['prefix'][str(p)]['action']=='DEFER' for r in rr); reset=sum(r['prefix'][str(p)]['action']=='RESET_REJECT' for r in rr)
        pref.append(f"| {p} | {sr}/76 | {tr}/76 | {ce}/76 | {commits} | {defer} | {reset} |")
    evrows=[]
    for r in d['event_records']:
        p=r['prefix']['16']; cat='' if r.get('category') is None else str(r.get('category')); vid='' if r.get('target_video') is None else str(r.get('target_video'))
        evrows.append(f"| {r['event_key']} | {r['kind']} | {r['fold']} | {cat} | {vid} | {r['first_action'] or 'NONE'} | {'yes' if r['correct_commit_ct'] else 'no'} | {'yes' if r['negative_false_commit'] else 'no'} | {'yes' if p['source_reliable'] else 'no'} | {'yes' if p['target_reliable'] else 'no'} | {'yes' if p['event_ceiling'] else 'no'} | {f4(p['source_max_iou'])} | {f4(p['target_max_iou'])} |")
    hashes={}
    for p in [ROOT/'outputs/iclr27_phase57/audit/github_methods.json',ROOT/'outputs/iclr27_phase57/audit/supervision_inventory.json',ROOT/'outputs/iclr27_phase57/audit/leakage_audit.json',ROOT/'outputs/iclr27_phase58/audit/architecture_contract.json',OUT61/'metrics/phase61_full_evaluation_repair3.json']:
        hashes[str(p.relative_to(ROOT))]=sha(p)
    return """# Phase61 MOT+OCD Frozen Evaluation Report

**Date:** 2026-08-29  
**Candidate:** raw-RGB Phase57--60 graph, final repair3 checkpoints  
**Decision:** `P50_GATE_P_FAIL_RAW_PIXEL_DETECTOR_AND_CAUSAL_SAFETY_FAIL_STOP_BEFORE_SEALED`

## Executive result

The first genuine pixel-level proposal/physical/semantic/controller graph was
run end to end on raw TRAIN-derived frames.  Its image-level detector did not
recover the target boxes (final top-20 IoU≥0.5 recall: 0.1%%, 0.9%%, 0.6%%, 0.5%%
on folds 0--3).  The causal evaluator then committed almost every event,
including every negative event.  This fails the MOT+OCD objective and all
safety gates.  Sealed evaluation was correctly **not run**; DEV+, Q1 and
public-new-model labels remain sealed.

## Protocol and forbidden-input audit

The evaluation uses the original 76 positive and 76 negative manifests,
prefixes `{1,2,4,8,16}`, original row keys and causal chronology.  RGB pixels
are the only model input.  GT boxes/categories are read after inference for
scoring only.  No category text, semantic/physical ID feature, future frame or
track, held GT, DEV+, Q1 or public-new-model label was accessed.  Physical IDs
are bookkeeping only and the semantic branch cannot mutate parent assignment.

## Detector/proposal results

| tag | f0 top20@.5 | f1 top20@.5 | f2 top20@.5 | f3 top20@.5 | mean | f0 top20@.3 | f1 top20@.3 | f2 top20@.3 | f3 top20@.3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
%s

The raw Phase26 observable stream remains 25/76 (41/76 proposal ceiling), but
those numbers are not an input to this pixel model.  Repair3's real detector
does not supply a comparable source/target candidate pool.  The old
feature-row objectness AP≈1 is not used or presented as detector AP.

## Causal prefix event aggregates (repair3)

| prefix | source reliable | target reliable | perfect event ceiling | COMMIT | DEFER | RESET/REJECT |
|---:|---:|---:|---:|---:|---:|---:|
%s

## Positive/negative causal summary

| tag | positive Commit-CT | negative false-commit rate | premature rate | unresolved rate | category coverage | video coverage |
|---|---:|---:|---:|---:|---:|---:|
%s

Repair3's aggregate is 76/76 positive commits, **but negative false commit is
76/76**, premature rate is 1.0 and known/novel confusion is 1.0.  Therefore
the apparent positive count is a degenerate action policy, not persistent
semantic correspondence.  Repair2 was 75/76 with the same 1.0 negative rate;
the initial and repair1 runs were 76/76 with the same safety failure.

## Complete 152-event record

The table below is generated directly from
`outputs/iclr27_phase61/metrics/phase61_full_evaluation_repair3.json`; no event
was removed or re-denominated.

| event key | kind | fold | category | target video | first action | positive correct | negative false commit | source reliable p16 | target reliable p16 | ceiling p16 | source max IoU | target max IoU |
|---|---|---:|---:|---:|---|---|---|---|---|---|---:|---:|
%s

## Correspondence and MOT reporting boundary

The raw model has no valid full-sequence track exporter or registered
track-candidate retrieval evaluator.  Consequently no new R@1/mAP/hard-gap
number is claimed for this candidate; Phase56 feature-row retrieval
(`0.9000/0.8599` raw p16) remains a historical comparator only.  The existing
TrackEval TAO adapter requires full-sequence predictions, so standard HOTA,
DetA, AssA, MOTA and IDF1 are **not computable from this bounded event
artifact** and are explicitly null in the JSON.  It would be invalid to infer
MOT success from inherited continuity/duplicate counters.

Physical bookkeeping checks in the bounded evaluator: physical IDs mutated
`false`, parent assignment mutated `false`, duplicate births `0`; full
sequence continuity/fragmentation/ID-switch metrics are not claimed.

## Gate decisions

* **P50: FAIL.** A trainable image-level class-agnostic detector did not reach
  useful IoU coverage; proposal/source is the first actionable failure.
* **R50: NOT PASS.** No valid learned track-candidate retrieval was exported,
  so proxy retrieval is not substituted for a correspondence gate.
* **C50: FAIL.** The causal policy commits negatives at 100%%, violating false
  commit, premature and known/novel safety despite the positive count.
* **S50: NOT RUN.** Sealed evaluation is authorized only after frozen
  compatibility/safety passes; it remains sealed.

The registered raw-frame route (official audit → TRAIN contract → pixel graph →
smoke/targeted → four-fold formal → frozen 76+76 causal evaluation) is
complete as a negative evidence route.  The first actionable next direction is
an independently validated, pretrained class-agnostic detector and a full
sequence MOT exporter; no threshold, StateMemory, controller or backbone
lottery was run here.

## Resources, processes and artifacts

Formal workers ran only on GPUs 4--7 with >103 GiB available RAM at preflight
and ~3.7 GiB RSS per worker; no OOM, swap pressure or external PID termination
occurred.  Checkpoints are in `outputs/iclr27_phase60/checkpoints/`, metrics and
event records in `outputs/iclr27_phase61/metrics/`, and all four formal
`.launched/.done` pairs exist.  The raw frame root is a symlink to the existing
TAO frame store; no large feature or checkpoint copy was made.

Representative artifact SHA256:

| artifact | sha256 |
|---|---|
%s

Reproduction commands are listed in the Phase60 training report.  A final
read-only integrity check must confirm JSON parseability, marker/checkpoint
presence, valid symlinks, no forbidden-label files in Phase57--61 outputs and
no residual Phase57--61 process.
""" % ("\n".join([f"| {tag} | " + " | ".join(f"{100*m['top20_recall_iou_0.5']:.2f}%" for m in d['detector_by_fold']) + f" | {100*sum(m['top20_recall_iou_0.5'] for m in d['detector_by_fold'])/4:.2f}% | " + " | ".join(f"{100*m['top20_recall_iou_0.3']:.2f}%" for m in d['detector_by_fold']) + " |" for tag,d in evals.items()]), "\n".join(pref), "\n".join(causal_rows), "\n".join(evrows), "\n".join(f"| {k} | {v} |" for k,v in hashes.items()))


def main():
    inv=json.loads(INV.read_text()); leak=json.loads(LEAK.read_text())
    evals={t:load_eval(t) for t in ('formal','repair1','repair2','repair3')}
    # Keep the decision machine-readable and deliberately conservative: a high
    # positive count cannot pass when all negatives are false commits.
    final=evals['repair3']; c=final['causal_event_metrics']; det=final['detector_by_fold']
    aggregate_top20=sum(x['top20_recall_iou_0.5'] for x in det)/4
    decision={
      'phase':61,'route':'phase57_61_raw_pixel_end_to_end','timestamp':'2026-08-29',
      'decision_code':'P50_GATE_P_FAIL_RAW_PIXEL_DETECTOR_AND_CAUSAL_SAFETY_FAIL_STOP_BEFORE_SEALED',
      'gates':{
        'P50':{'status':'FAIL','reason':'image-level top20 true IoU>=0.5 recall is below 1.2% in every fold; no useful proposal/source coverage'},
        'R50':{'status':'NOT_PASS','reason':'no valid learned track-candidate retrieval exporter; proxy retrieval not used as a gate'},
        'C50':{'status':'FAIL','commit_ct':c['commit_ct'],'negative_false_commit_rate':c['negative_false_commit_rate'],'premature_rate':c['premature_rate'],'known_novel_confusion_rate':c['known_novel_confusion_rate']},
        'S50':{'status':'NOT_RUN','reason':'sealed evaluation requires frozen compatibility and safety gates; public/sealed labels remain sealed'}},
      'phase26_comparators':{'raw_observable_events':'25/76','proposal_ceiling':'41/76','historical_raw_retrieval_p16':{'r1':0.9,'map':0.8599},'phase56_controller_commit_ct':'4/76'},
      'raw_pixel_repair3':{'detector_by_fold':det,'mean_top20_iou_0.5':aggregate_top20,'causal':c,'standard_mot_metrics':final['standard_mot_metrics']},
      'protocol':{'positive_events':76,'negative_events':76,'prefixes':[1,2,4,8,16],'seed_base':575700,'denominator_changed':False,'row_key_changed':False,'evaluator_changed':False},
      'resources':{'gpus':[4,5,6,7],'max_workers':4,'ram_headroom_observed':'>103 GiB available; ~3.7 GiB RSS/worker','oom':False,'external_process_killed':False},
      'sealed_inputs_not_read':['DEV+','Q1','public new-model labels','future frames/tracks','held GT as model input'],
      'artifacts':{'phase60':'outputs/iclr27_phase60','phase61':'outputs/iclr27_phase61','event_metrics':'outputs/iclr27_phase61/metrics/phase61_full_evaluation_repair3.json'},
      'next_direction':'audit and validate a sufficiently pretrained class-agnostic detector/full-sequence MOT exporter before any semantic/controller claim; no unregistered lottery authorized'
    }
    atomic_json(OUT61/'final_decision.json',decision)
    atomic_text(ROOT/'docs/iclr27_phase60/PHASE60_PIXEL_END_TO_END_TRAINING_REPORT.md',render_training(inv,evals))
    atomic_text(ROOT/'docs/iclr27_phase61/PHASE61_MOT_OCD_SEALED_EVALUATION_REPORT.md',render_phase61(inv,evals,decision))
    # Lightweight completion/integrity artifact.  Process matching walks /proc
    # so the inspector shell itself is not mistaken for a live worker.
    json_errors=[]; json_count=0
    for d in (ROOT/'outputs/iclr27_phase57', ROOT/'outputs/iclr27_phase58', OUT60, OUT61):
        for p in d.rglob('*.json'):
            try: json.loads(p.read_text()); json_count += 1
            except Exception as e: json_errors.append({'path':str(p.relative_to(ROOT)),'error':str(e)})
    live=[]
    for q in Path('/proc').glob('[0-9]*'):
        try:
            pid=int(q.name); cmd=(q/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='ignore')
            if pid != os.getpid() and any(x in cmd for x in ('train_pixel_e2e.py','run_four_fold_supervisor.sh','evaluate_pixel_e2e.py')):
                live.append({'pid':pid,'cmd':cmd.strip()})
        except Exception:
            pass
    pairs=[]
    for f in range(4):
        pairs.append({'fold':f,'done':all((OUT60/'completion'/f'phase60_{t}_f{f}.done').exists() for t in ('formal','repair1','repair2','repair3')),
                      'launched':all((OUT60/'completion'/f'phase60_{t}_f{f}.launched').exists() for t in ('formal','repair1','repair2','repair3')),
                      'best_checkpoints':all((OUT60/'checkpoints'/f'phase60_repair3_f{f}_best.pt').exists() for _ in (0,))})
    symlinks=[]
    for p in (ROOT/'data/raw/tao/frames',):
        symlinks.append({'path':str(p.relative_to(ROOT)),'is_symlink':p.is_symlink(),'target':str(p.resolve()),'target_exists':p.resolve().exists()})
    forbidden=[]
    for d in (ROOT/'outputs/iclr27_phase57',ROOT/'outputs/iclr27_phase58',OUT60,OUT61):
        for p in d.rglob('*'):
            if p.is_file() and any(s in p.name.lower() for s in ('devplus','q1','public_new_model','sealed_label')):
                forbidden.append(str(p.relative_to(ROOT)))
    atomic_json(OUT61/'audit/phase61_integrity.json',{'phase':61,'json_count':json_count,'json_errors':json_errors,'formal_pairs':pairs,'symlinks':symlinks,'forbidden_named_files':forbidden,'live_phase57_61_processes':live,'integrity_pass':not json_errors and all(x['done'] and x['launched'] and x['best_checkpoints'] for x in pairs) and not forbidden and not live})
    print(json.dumps({'decision':decision['decision_code'],'report':'docs/iclr27_phase61/PHASE61_MOT_OCD_SEALED_EVALUATION_REPORT.md'},ensure_ascii=False))

if __name__=='__main__': main()
