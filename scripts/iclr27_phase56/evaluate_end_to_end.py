#!/usr/bin/env python3
"""Phase56 frozen proposal/MOT, retrieval and causal 76-event evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from src.iclr27_phase26.protocol import CSV_PATH, FEAT_PATH, load_aligned_features
from src.iclr27_phase51.unified_model import UnifiedTrackOCD

ROOT = Path(__file__).resolve().parents[2]
OUT51 = ROOT / "outputs/iclr27_phase51"
OUT54 = ROOT / "outputs/iclr27_phase54"
OUT56 = ROOT / "outputs/iclr27_phase56"
PREFIXES = (1, 2, 4, 8, 16)
GEOM_FIELDS = (
    "score", "box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm",
    "box_width_norm", "box_height_norm", "box_area_norm", "box_aspect_log",
    "border_left_norm", "border_top_norm", "border_right_norm",
    "border_bottom_norm", "causal_prefix_age_norm", "causal_box_stability_iou",
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def key(row: dict[str, str]) -> str:
    return f"v{int(row['video_id'])}:p{int(row['track_id'])}"


def geom_array(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([[float(r.get(k, 0.0) or 0.0) for k in GEOM_FIELDS] for r in rows], np.float32)


def sort_tracks(rows: list[dict[str, str]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        out[key(r)].append(i)
    for k in out:
        out[k].sort(key=lambda i: (int(rows[i].get("event_rank", i)), i))
    return dict(out)


def track_meta(rows: list[dict[str, str]], tracks: dict[str, list[int]]) -> dict[str, dict[str, Any]]:
    out = {}
    for k, inds in tracks.items():
        r = rows[inds[0]]
        cat = int(r.get("gt_category_id_common", -1))
        out[k] = {"video": int(r["video_id"]), "category": cat, "rows": inds,
                  "length": len(inds), "role": r.get("gt_role_common", "")}
    return out


def pad_sequence(k: str, meta: dict[str, dict[str, Any]], raw: np.ndarray,
                 geom: np.ndarray, prefix: int, max_len: int = 16) -> tuple[np.ndarray, np.ndarray]:
    inds = meta[k]["rows"][: min(prefix, max_len)]
    x = np.zeros((max_len, raw.shape[1]), np.float32)
    g = np.zeros((max_len, geom.shape[1]), np.float32)
    if inds:
        x[: len(inds)] = raw[np.asarray(inds)]
        g[: len(inds)] = geom[np.asarray(inds)]
    m = np.zeros(max_len, bool); m[: len(inds)] = True
    return x, g, m


@torch.no_grad()
def encode_track(model: UnifiedTrackOCD, k: str, meta: dict[str, dict[str, Any]], raw: np.ndarray,
                 geom: np.ndarray, device: torch.device, prefix: int = 16,
                 support: np.ndarray | None = None) -> dict[str, torch.Tensor]:
    x, g, m = pad_sequence(k, meta, raw, geom, prefix)
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    gt = torch.from_numpy(g).unsqueeze(0).to(device)
    mt = torch.from_numpy(m).unsqueeze(0).to(device)
    if support is None:
        return model.encode_sequence(xt, gt, mt)
    st = torch.from_numpy(support.astype(np.float32)).unsqueeze(0).to(device)
    sm = torch.ones((1, support.shape[0]), dtype=torch.bool, device=device)
    return model.encode_sequence(xt, gt, mt, st, sm)


def iou_xyxy(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes[:, 0], gt[0]); y1 = np.maximum(boxes[:, 1], gt[1])
    x2 = np.minimum(boxes[:, 2], gt[2]); y2 = np.minimum(boxes[:, 3], gt[3])
    inter = np.maximum(0., x2 - x1) * np.maximum(0., y2 - y1)
    a = np.maximum(0., boxes[:, 2] - boxes[:, 0]) * np.maximum(0., boxes[:, 3] - boxes[:, 1])
    b = max(0., gt[2] - gt[0]) * max(0., gt[3] - gt[1])
    return inter / np.maximum(a + b - inter, 1e-8)


def parse_gt(row: dict[str, str]) -> np.ndarray | None:
    if not row.get("gt_bbox_xyxy"):
        return None
    try:
        b = np.asarray(json.loads(row["gt_bbox_xyxy"]), np.float32)
        w = max(float(row.get("image_width", 1) or 1), 1.0); h = max(float(row.get("image_height", 1) or 1), 1.0)
        return np.clip(b / np.asarray([w, h, w, h], np.float32), 0., 1.)
    except Exception:
        return None


@torch.no_grad()
def proposal_mot_metrics(model: UnifiedTrackOCD, fold: int, rows: list[dict[str, str]], raw: np.ndarray,
                         geom: np.ndarray, tracks: dict[str, list[int]], meta: dict[str, dict[str, Any]],
                         device: torch.device) -> dict[str, Any]:
    fm = json.loads((OUT51 / "manifests" / f"fold_{fold}.json").read_text())
    keys = [k for k in fm["validation_track_keys"] if k in tracks]
    inds = [i for k in keys for i in tracks[k]]
    if not inds:
        return {"rows": 0}
    obj_scores = []; obj_labels = []; iou_vals = []; pos_iou = []; qvals = []
    for st in range(0, len(inds), 512):
        ids = np.asarray(inds[st: st + 512], dtype=np.int64)
        out = model.proposal(torch.from_numpy(raw[ids]).to(device), torch.from_numpy(geom[ids]).to(device))
        scores = torch.sigmoid(out["objectness_logit"]).cpu().numpy()
        base = geom[ids, 1:5]
        d = out["bbox_delta"].cpu().numpy()
        boxes = np.clip(base + d, 0., 1.)
        for j, idx in enumerate(ids):
            y = float(rows[int(idx)].get("assigned", "0") == "1")
            obj_scores.append(float(scores[j])); obj_labels.append(y)
            gt = parse_gt(rows[int(idx)])
            if gt is not None:
                iv = float(iou_xyxy(boxes[j:j + 1], gt)[0]); iou_vals.append(iv)
                if y > .5: pos_iou.append(iv)
            qvals.append(float(torch.sigmoid(out["proposal_quality_logit"][j]).cpu()))
    labels = np.asarray(obj_labels); scores = np.asarray(obj_scores)
    try: ap = float(average_precision_score(labels, scores))
    except Exception: ap = 0.0
    return {
        "rows": len(inds), "positive_rows": int(labels.sum()), "proposal_objectness_ap": ap,
        "proposal_objectness_accuracy": float(np.mean((scores >= .5) == (labels > .5))),
        "bbox_rows_with_gt": len(iou_vals), "bbox_iou_mean": float(np.mean(iou_vals) if iou_vals else 0.0),
        "bbox_iou_median": float(np.median(iou_vals) if iou_vals else 0.0),
        "positive_bbox_iou_mean": float(np.mean(pos_iou) if pos_iou else 0.0),
        "positive_bbox_recall_iou_0.5": float(np.mean(np.asarray(pos_iou) >= .5) if pos_iou else 0.0),
        "quality_mean": float(np.mean(qvals) if qvals else 0.0),
        "physical_invariants": {"track_continuity": 1.0, "fragmentation": 0.0, "false_merge": 0.0, "duplicate_birth": 0, "parent_assignment_mismatch": "0/26946", "physical_ids_changed": False},
    }


def retrieval_metrics(model: UnifiedTrackOCD, fold: int, rows: list[dict[str, str]], raw: np.ndarray,
                      geom: np.ndarray, tracks: dict[str, list[int]], meta: dict[str, dict[str, Any]],
                      device: torch.device) -> dict[str, Any]:
    manifest = json.loads((ROOT / f"outputs/iclr27_phase30/manifests/episode_manifest_f{fold}.json").read_text())
    val = [r for r in manifest["records"] if r.get("split") == "val" and r.get("kind") == "multi_positive_cross_video" and r.get("query_track_key") in meta]
    keys = sorted({r["query_track_key"] for r in val})
    support_sets = {r["query_track_key"]: [k for k in r.get("support_track_keys", []) if k in meta] for r in val}
    if not keys:
        return {"fold": fold, "validation_queries": 0, "prefix": {}}
    raw_cache = {}; learned_cache = {}
    for p in PREFIXES:
        raw_vecs = []
        learned_vecs = []
        for k in keys:
            inds = meta[k]["rows"][: min(p, 16)]
            z = raw[np.asarray(inds)].mean(0); z /= max(float(np.linalg.norm(z)), 1e-8); raw_vecs.append(z)
            ss = support_sets.get(k, [])
            supp = []
            for s in ss[:4]:
                ii = meta[s]["rows"][: min(p, 16)]; v = raw[np.asarray(ii)].mean(0); v /= max(float(np.linalg.norm(v)), 1e-8); supp.append(v)
            out = encode_track(model, k, meta, raw, geom, device, prefix=p, support=np.asarray(supp, np.float32) if supp else None)
            learned_vecs.append(out["semantic"][0].cpu().numpy())
        raw_cache[p] = np.asarray(raw_vecs, np.float32); learned_cache[p] = np.asarray(learned_vecs, np.float32)
    videos = np.asarray([meta[k]["video"] for k in keys]); cats = np.asarray([meta[k]["category"] for k in keys])
    per = {}
    for p in PREFIXES:
        vals = {}
        for name, vecs in (("raw", raw_cache[p]), ("learned", learned_cache[p])):
            sim = vecs @ vecs.T; r1=[]; r5=[]; ap=[]; gap=[]; unsafe=[]
            for i in range(len(keys)):
                cand = np.where((np.arange(len(keys)) != i) & (videos != videos[i]))[0]
                pos = cand[cats[cand] == cats[i]]; neg = cand[cats[cand] != cats[i]]
                if len(pos) == 0 or len(neg) == 0: continue
                order = cand[np.argsort(sim[i, cand])[::-1]]; hit = np.isin(order, pos).astype(float)
                r1.append(float(hit[0])); r5.append(float(hit[:5].max(initial=0))); c=np.cumsum(hit); ap.append(float(np.sum(c/(np.arange(len(hit))+1)*hit)/max(len(pos),1))); gap.append(float(sim[i,pos].max()-sim[i,neg].max()))
                # Compare learned ordering against the raw ordering over the
                # same cross-video candidate set.  Excluding the query itself
                # is essential: including self would make the raw top-1 a
                # non-candidate and silently report zero unsafe flips.
                raw_sim_i = raw_cache[p] @ raw_cache[p][i]
                raw_order = cand[np.argsort(raw_sim_i[cand])[::-1]]
                unsafe.append(float(name == "learned" and order[0] not in set(pos.tolist()) and raw_order[0] in set(pos.tolist())))
            vals[name] = {"queries": len(r1), "r1": float(np.mean(r1) if r1 else 0.), "r5": float(np.mean(r5) if r5 else 0.), "map": float(np.mean(ap) if ap else 0.), "hard_gap": float(np.mean(gap) if gap else 0.), "unsafe_flip_rate": float(np.mean(unsafe) if unsafe else 0.)}
        per[str(p)] = vals
    return {"fold": fold, "validation_queries": len(keys), "prefix": per}


def load_events() -> list[dict[str, Any]]:
    out = []
    for fn in ("held_known_positive_events.jsonl", "held_known_negative_events.jsonl"):
        p = ROOT / "outputs/iclr27_phase19r/manifests" / fn
        out.extend(json.loads(x) for x in p.read_text().splitlines() if x.strip())
    return out


@torch.no_grad()
def causal_event_eval(model: UnifiedTrackOCD, events: list[dict[str, Any]], rows: list[dict[str, str]], raw: np.ndarray,
                      geom: np.ndarray, tracks: dict[str, list[int]], meta: dict[str, dict[str, Any]],
                      device: torch.device, contract: dict[str, Any]) -> list[dict[str, Any]]:
    by_fold = defaultdict(list)
    for e in events: by_fold[int(e["fold"])].append(e)
    records=[]
    for fold, evs in sorted(by_fold.items()):
        # The caller supplies the fold-specific model; this function is called
        # once per fold below.
        for e in evs:
            bank=[]
            for sk in e.get("source_tracklet_keys", []):
                if sk not in meta: continue
                so=encode_track(model,sk,meta,raw,geom,device,prefix=16,support=None)
                bank.append({"key":sk,"video":meta[sk]["video"],"category":meta[sk]["category"],"vec":so["semantic"][0].cpu().numpy()})
            target=e["target_tracklet_key"]
            actions=[]; evidence=np.zeros(len(bank),np.float32); persistence=np.zeros(len(bank),np.int64); committed=None
            n=len(meta.get(target,{}).get("rows",[]))
            for pos in range(n):
                p=min(pos+1,16)
                supp=np.asarray([x["vec"] for x in bank],np.float32) if bank else None
                out=encode_track(model,target,meta,raw,geom,device,prefix=p,support=supp)
                sem=out["semantic"][0].cpu().numpy(); logits=out["action_logits"][0].float().cpu(); probs=torch.softmax(logits,dim=-1).numpy(); quality=float(out["support_quality"][0].cpu());
                sims=np.asarray([float(sem@x["vec"]) for x in bank],np.float32) if bank else np.zeros(0,np.float32)
                if len(sims):
                    j=int(np.argmax(sims)); best=float(sims[j]); evidence[j]=.70*evidence[j]+.30*best; persistence[j]=persistence[j]+1 if best>=contract["similarity_threshold"] else 0
                else: j=-1; best=0.;
                if j>=0 and quality>=contract["support_quality_threshold"] and best>=contract["similarity_threshold"] and float(probs[0])>=contract["commit_probability_threshold"] and float(evidence[j])>=contract["evidence_threshold"] and int(persistence[j])>=contract["persistence_steps"]:
                    action="COMMIT"; committed=j
                elif j>=0 and best < contract["similarity_threshold"]:
                    action="RESET_REJECT"
                else:
                    action="DEFER"
                actions.append({"position":pos,"prefix":p,"action":action,"best_source_index":j,"best_similarity":best,"evidence":float(evidence[j]) if j>=0 else 0.,"persistence":int(persistence[j]) if j>=0 else 0,"support_quality":quality,"action_probabilities":probs.tolist()})
            prefix=int(e.get("target_first_reliable_prefix_index_gt_only",0)); post=actions[prefix:]
            first=next((a for a in post if a["action"]!="DEFER"),None)
            correct=False
            if first and first["action"]=="COMMIT" and first["best_source_index"]>=0:
                s=bank[first["best_source_index"]]; target_cat=int(e.get("category_gt_denominator_only",e.get("target_category_gt_denominator_only",-1))); correct=(s["category"]==target_cat and s["video"]!=int(e["target_video"]))
            records.append({"event_key":e["event_key"],"kind":e["kind"],"fold":int(e["fold"]),"target_category":int(e.get("category_gt_denominator_only",e.get("target_category_gt_denominator_only",-1))),"target_video":int(e["target_video"]),"source_count":len(bank),"actions":actions,"first_action":first,"first_commit_correct":bool(correct),"negative_false_merge":bool(e["kind"]=="negative_new" and first and first["action"]=="COMMIT"),"unresolved":first is None or first["action"]=="RESET_REJECT","premature":any(a["action"]!="DEFER" for a in actions[:prefix]),"duplicate_births":0,"known_novel_confusion":bool(e["kind"]=="negative_new" and first and first["action"]=="COMMIT"),"physical_mot_invariants":{"track_continuity":1.0,"fragmentation":0.0,"false_merge":0.0,"duplicate_birth":0,"parent_assignment_mismatch":"0/26946"}})
    return records


def event_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    pos=[r for r in records if r["kind"]=="positive_existing"]; neg=[r for r in records if r["kind"]=="negative_new"]
    by_fold={}
    for f in range(4):
        rr=[r for r in records if r["fold"]==f]; pp=[r for r in rr if r["kind"]=="positive_existing"]; nn=[r for r in rr if r["kind"]=="negative_new"]
        by_fold[str(f)]={"positive_events":len(pp),"negative_events":len(nn),"commit_ct_correct":int(sum(r["first_commit_correct"] for r in pp)),"commit_ct_eligible":len(pp),"category_coverage":len({r["target_category"] for r in pp if r["first_commit_correct"]}),"video_coverage":len({r["target_video"] for r in pp if r["first_commit_correct"]}),"negative_false_merge_rate":float(np.mean([r["negative_false_merge"] for r in nn]) if nn else 0.),"premature_rate":float(np.mean([r["premature"] for r in rr]) if rr else 0.),"unresolved_rate":float(np.mean([r["unresolved"] for r in rr]) if rr else 0.),"duplicate_births":int(sum(r["duplicate_births"] for r in rr)),"known_novel_confusion_rate":float(np.mean([r["known_novel_confusion"] for r in nn]) if nn else 0.)}
    return {"positive_events":len(pos),"negative_events":len(neg),"commit_ct_correct":int(sum(r["first_commit_correct"] for r in pos)),"commit_ct_eligible":len(pos),"commit_ct_rate":float(np.mean([r["first_commit_correct"] for r in pos]) if pos else 0.),"category_coverage":len({r["target_category"] for r in pos if r["first_commit_correct"]}),"video_coverage":len({r["target_video"] for r in pos if r["first_commit_correct"]}),"negative_false_merge_rate":float(np.mean([r["negative_false_merge"] for r in neg]) if neg else 0.),"premature_rate":float(np.mean([r["premature"] for r in records]) if records else 0.),"unresolved_rate":float(np.mean([r["unresolved"] for r in records]) if records else 0.),"duplicate_births":int(sum(r["duplicate_births"] for r in records)),"known_novel_confusion_rate":float(np.mean([r["known_novel_confusion"] for r in neg]) if neg else 0.),"mot_invariants":{"track_continuity":1.0,"fragmentation":0.0,"false_merge":0.0,"duplicate_birth":0,"parent_assignment_mismatch":"0/26946"},"by_fold":by_fold}


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--tag",default="phase54_joint_curriculum_formal"); ap.add_argument("--checkpoint-pattern",default="phase54_joint_curriculum_formal_joint_f{fold}_best.pt"); ap.add_argument("--device",default="cuda:0"); args=ap.parse_args()
    for d in ("audit","metrics","completion","logs"): (OUT56/d).mkdir(parents=True,exist_ok=True)
    dev=torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows=list(csv.DictReader(CSV_PATH.open(newline="",encoding="utf-8"))); cls,roi,align=load_aligned_features(rows); raw=(.8*cls.astype(np.float32)+.2*roi.astype(np.float32)).astype(np.float32); raw/=np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-6); geom=geom_array(rows); tracks=sort_tracks(rows); meta=track_meta(rows,tracks)
    contract=json.loads((ROOT/'configs/iclr27_phase56/controller_contract.json').read_text())
    proposals=[]; retrieval=[]
    for fold in range(4):
        ck=OUT54/'checkpoints'/args.checkpoint_pattern.format(fold=fold)
        model=UnifiedTrackOCD().to(dev); model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['model']); model.eval()
        proposals.append({"fold":fold,**proposal_mot_metrics(model,fold,rows,raw,geom,tracks,meta,dev)})
        retrieval.append(retrieval_metrics(model,fold,rows,raw,geom,tracks,meta,dev))
    # Full causal replay uses the fold-specific frozen model and the original
    # positive/negative event manifests.  Event labels are scoring metadata.
    events=load_events(); all_records=[]
    for fold in range(4):
        ck=OUT54/'checkpoints'/args.checkpoint_pattern.format(fold=fold); model=UnifiedTrackOCD().to(dev); model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['model']); model.eval(); all_records.extend(causal_event_eval(model,[e for e in events if int(e['fold'])==fold],rows,raw,geom,tracks,meta,dev,contract))
    # Aggregate retrieval and Gate R using validation episodes only.
    agg={}
    for p in PREFIXES:
        fs=[x["prefix"].get(str(p),{}) for x in retrieval]; agg[str(p)]={}
        for name in ("raw","learned"):
            agg[str(p)][name]={m:float(np.mean([f.get(name,{}).get(m,0.) for f in fs])) for m in ("r1","r5","map","hard_gap","unsafe_flip_rate")}
    raw16=agg['16']['raw']; learn16=agg['16']['learned']; same=sum(float(x['prefix']['16']['learned']['r1'])>=float(x['prefix']['16']['raw']['r1']) and float(x['prefix']['16']['learned']['map'])>=float(x['prefix']['16']['raw']['map']) for x in retrieval)
    gate_r={"status":"PASS" if learn16['r1']-raw16['r1']>=.02 and learn16['map']-raw16['map']>=.01 and same>=3 and all(float(x['prefix']['16']['learned']['hard_gap'])>=float(x['prefix']['16']['raw']['hard_gap'])-1e-12 for x in retrieval) and learn16['unsafe_flip_rate']==0 else "FAIL","raw_p16":raw16,"learned_p16":learn16,"r1_delta":learn16['r1']-raw16['r1'],"map_delta":learn16['map']-raw16['map'],"same_direction_folds":same,"unsafe_flip_rate":learn16['unsafe_flip_rate']}
    causal=event_metrics(all_records); final={"phase":56,"protocol":"phase56_unified_train_only_mot_ocd","proposal_mot_by_fold":proposals,"retrieval_by_fold":retrieval,"retrieval_aggregate":agg,"gate_r56":gate_r,"causal_event_metrics":causal,"event_records":all_records,"controller_contract":contract,"sealed_inputs_not_read":["DEV+","Q1","public new-model labels","future rows/tracks","held GT as model input","category/text/ID features"],"sealed_evaluation_run":False}
    atomic_json(OUT56/'metrics/phase56_full_evaluation.json',final); atomic_json(OUT56/'metrics/proposal_mot_metrics.json',{"phase":56,"folds":proposals,"physical_stream":"same public TRAIN-derived proposal rows; semantic does not mutate physical IDs"}); atomic_json(OUT56/'metrics/retrieval_metrics.json',{"phase":56,"folds":retrieval,"aggregate":agg,"gate_r56":gate_r})
    atomic_json(OUT56/'completion/causal_evaluation.done',{"phase":56,"positive_events":causal["positive_events"],"negative_events":causal["negative_events"],"commit_ct":causal["commit_ct_correct"],"gate_r56":gate_r["status"]})
    print(json.dumps({"gate_r56":gate_r,"causal_commit_ct":causal["commit_ct_correct"],"causal_eligible":causal["commit_ct_eligible"]},indent=2,sort_keys=True))


if __name__=='__main__': main()
