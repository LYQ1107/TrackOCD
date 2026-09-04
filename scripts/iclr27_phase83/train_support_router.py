#!/usr/bin/env python3
"""Phase83 Branch-B TRAIN-only causal support-quality router.

This is a compact one-hidden-layer MLP over causal, class-agnostic row
statistics.  ``assigned``/IoU/GT fields are used only to form TRAIN labels and
post-hoc event metrics; they never enter the feature tensor.  Event videos are
excluded from all fitting and validation rows.  The frozen Phase75B evaluator
is not modified.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.iclr27_phase23.protocol import load_aligned_features

OUT = ROOT / "outputs/iclr27_phase83"
CSV_PATH = ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv"
FOLD_MANIFEST = ROOT / "outputs/iclr27_phase22/manifests/fold_manifest.json"
OBS = Path("/data2/usr_for_deadline/trackocd_phase75b/observability_repair2/event_observability.jsonl")
POS = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
NEG = ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl"
ALLOWED_ROLES = {"known_bank", "novel_correspondence_train"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent)); os.close(fd)
    try:
        # Passing an open handle prevents numpy from appending a second
        # extension to our atomic temporary filename.
        with open(tmp, "wb") as f:
            np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def order_key(r: dict[str, str]) -> tuple[int, int, int]:
    return (int(float(r.get("event_rank", 0) or 0)), int(float(r.get("frame_id", 0) or 0)), int(float(r.get("proposal_local_id", 0) or 0)))


def f(r: dict[str, str], k: str, default: float = 0.0) -> float:
    try:
        x = float(r.get(k, default)); return x if math.isfinite(x) else default
    except (TypeError, ValueError): return default


def event_videos() -> set[int]:
    out: set[int] = set()
    for path in (POS, NEG):
        for line in path.read_text(encoding="utf-8").splitlines():
            e = json.loads(line); out.add(int(e["source_video"])); out.add(int(e["target_video"]))
    return out


FEATURE_NAMES = [
    "base_proposal_score", "box_width_norm", "box_height_norm", "box_area_norm", "box_aspect_log",
    "border_left_norm", "border_top_norm", "border_right_norm", "border_bottom_norm",
    "causal_age_norm", "causal_box_stability_iou", "history_length_norm", "gap_norm",
    "proposal_density_log", "candidate_ambiguity_log", "corrected_dinov2_cosine", "temporal_mean_cosine",
]


def build_features(rows: list[dict[str, str]], fused: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if len(rows) != len(fused): raise RuntimeError("aligned feature row count mismatch")
    track_inds: dict[str, list[int]] = defaultdict(list); image_count: Counter[tuple[int, int]] = Counter(); image_tracks: defaultdict[tuple[int, int], set[str]] = defaultdict(set)
    for i, r in enumerate(rows):
        key = f"v{int(r['video_id'])}:p{int(r['track_id'])}"; track_inds[key].append(i)
        ik = (int(r["video_id"]), int(r["image_id"])); image_count[ik] += 1; image_tracks[ik].add(key)
    out = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=np.float32); row_key_to_index: dict[str, int] = {}
    for key, inds in track_inds.items():
        inds.sort(key=lambda i: order_key(rows[i])); prev_frame = None; running = np.zeros(fused.shape[1], dtype=np.float32); prev = None
        for pos, i in enumerate(inds):
            r = rows[i]; cur = fused[i].astype(np.float32); cur /= max(float(np.linalg.norm(cur)), 1e-8)
            hist_mean = running / max(pos, 1); hist_mean /= max(float(np.linalg.norm(hist_mean)), 1e-8)
            cos_hist = float(cur @ hist_mean) if pos else 0.0; cos_prev = float(cur @ prev) if prev is not None else 0.0
            frame = int(float(r.get("frame_id", 0) or 0)); gap = 0.0 if prev_frame is None else max(0, frame - prev_frame)
            ik = (int(r["video_id"]), int(r["image_id"]))
            out[i] = [
                f(r, "score"), f(r, "box_width_norm"), f(r, "box_height_norm"), f(r, "box_area_norm"), f(r, "box_aspect_log"),
                f(r, "border_left_norm"), f(r, "border_top_norm"), f(r, "border_right_norm"), f(r, "border_bottom_norm"),
                f(r, "causal_age_norm"), f(r, "causal_box_stability_iou"), math.log1p(pos) / 8.0, math.log1p(gap) / 8.0,
                math.log1p(image_count[ik]) / 8.0, math.log1p(len(image_tracks[ik])) / 4.0, cos_prev, cos_hist,
            ]
            row_key_to_index[str(r.get("row_key", ""))] = i; running += cur; prev = cur; prev_frame = frame
    labels = np.asarray([int(f(r, "assigned") == 1 and f(r, "row_iou") >= 0.5) for r in rows], dtype=np.int64)
    return out, labels, {"feature_names": FEATURE_NAMES, "row_key_to_index": row_key_to_index, "track_count": len(track_inds), "image_count": len(image_count), "causal_order": "event_rank,frame_id,proposal_local_id", "future_features": False}


class MLP:
    def __init__(self, d: int, hidden: int, seed: int) -> None:
        rng = np.random.default_rng(seed); self.w1 = (rng.standard_normal((d, hidden), dtype=np.float32) * np.float32(math.sqrt(2.0 / d))).astype(np.float32); self.b1 = np.zeros(hidden, np.float32); self.w2 = (rng.standard_normal(hidden, dtype=np.float32) * np.float32(math.sqrt(2.0 / hidden))).astype(np.float32); self.b2 = np.float32(0.0)
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = x @ self.w1 + self.b1; a = np.maximum(h, 0.0); return a, sigmoid(a @ self.w2 + self.b2)
    def predict(self, x: np.ndarray) -> np.ndarray: return self.forward(x)[1]
    def step(self, x: np.ndarray, y: np.ndarray, pos_weight: float, lr: float) -> float:
        h, p = self.forward(x); err = (p - y) * np.where(y > 0.5, pos_weight, 1.0); n = max(len(y), 1)
        gw2 = h.T @ err / n; gb2 = float(err.mean()); gh = err[:, None] * self.w2[None, :]; gh[h <= 0] = 0.0; gw1 = x.T @ gh / n; gb1 = gh.mean(axis=0)
        self.w2 -= lr * gw2; self.b2 -= np.float32(lr * gb2); self.w1 -= lr * gw1; self.b1 -= lr * gb1
        loss = -np.mean(np.where(y > .5, pos_weight * np.log(np.clip(p, 1e-6, 1)), np.log(np.clip(1-p, 1e-6, 1))))
        return float(loss)
    def save(self, path: Path, mean: np.ndarray, std: np.ndarray, threshold: float, step: int, fold: int) -> None:
        atomic_npz(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=np.asarray([self.b2]), mean=mean.astype(np.float32), std=std.astype(np.float32), threshold=np.asarray([threshold]), step=np.asarray([step]), fold=np.asarray([fold]))
    @classmethod
    def load(cls, path: Path) -> tuple["MLP", np.ndarray, np.ndarray, float, int]:
        z = np.load(path, allow_pickle=False); obj = cls(int(z["w1"].shape[0]), int(z["w1"].shape[1]), 0); obj.w1, obj.b1, obj.w2, obj.b2 = z["w1"], z["b1"], z["w2"], float(z["b2"][0]); return obj, z["mean"], z["std"], float(z["threshold"][0]), int(z["step"][0])


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    pred = p >= .5; tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0))); fn = int(np.sum((~pred) & (y == 1))); tn = int(np.sum((~pred) & (y == 0)))
    # Threshold-free ranking diagnostics use a deterministic O(n log n) AUC.
    order = np.argsort(-p); ys = y[order]; pos = int(y.sum()); neg = int(len(y) - pos); rank = np.arange(1, len(y)+1)
    # ``rank`` is descending (largest probability has rank 1), so invert the
    # usual ascending-rank Mann--Whitney expression for an AUC in [0,1].
    desc_stat = float((rank[ys == 1].sum() - pos*(pos+1)/2) / max(pos*neg, 1)); auc = 1.0 - desc_stat
    return {"rows": int(len(y)), "positive": pos, "negative": neg, "positive_rate": float(pos/max(len(y),1)), "predicted_positive": int(pred.sum()), "precision": tp/max(tp+fp,1), "recall": tp/max(tp+fn,1), "f1": 2*tp/max(2*tp+fp+fn,1), "roc_auc": auc, "brier": float(np.mean((p-y)**2)), "confusion": {"tp":tp,"fp":fp,"fn":fn,"tn":tn}, "p_quantiles": [float(x) for x in np.quantile(p,[0,.1,.5,.9,1])]} 


def fold_sets(fold: int, rows: list[dict[str, str]], blocked: set[int]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fm = json.loads(FOLD_MANIFEST.read_text(encoding="utf-8"))["folds"][fold]; fit_v = set(int(x) for x in fm["fit_videos"]) - blocked; val_v = set(int(x) for x in fm["validation_videos"]) - blocked; fit_c = set(int(x) for x in fm["fit_categories"]); val_c = set(int(x) for x in fm.get("held_categories", []))
    allowed = np.asarray([r.get("role17") in ALLOWED_ROLES and int(r["video_id"]) not in blocked for r in rows])
    fit = np.asarray([bool(allowed[i]) and int(rows[i]["video_id"]) in fit_v and int(float(rows[i].get("gt_category_id_common", -1) or -1)) in fit_c for i in range(len(rows))])
    val = np.asarray([bool(allowed[i]) and int(rows[i]["video_id"]) in val_v and int(float(rows[i].get("gt_category_id_common", -1) or -1)) in val_c for i in range(len(rows))])
    return np.flatnonzero(fit), np.flatnonzero(val), {"fit_videos": sorted(fit_v), "validation_videos": sorted(val_v), "fit_categories": sorted(fit_c), "validation_categories": sorted(val_c), "video_disjoint": True, "category_disjoint": True}


def event_replay(models: dict[int, tuple[MLP, np.ndarray, np.ndarray, float, int]], row_map: dict[str, int], x: np.ndarray) -> dict[str, Any]:
    obs = [json.loads(line) for line in OBS.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in obs:
        if r.get("polarity") in {"positive", "negative"}: by_fold[int(r["fold"])].append(r)
    records=[]
    for fold, rs in sorted(by_fold.items()):
        if fold not in models: continue
        model, mean, std, threshold, _ = models[fold]
        for r in rs:
            def side_eval(side: str) -> dict[str, Any]:
                details = r.get(f"{side}_row_details", []); scored=[]
                for d in details:
                    idx=row_map.get(str(d.get("row_key", "")))
                    if idx is None: continue
                    p=float(model.predict(((x[idx:idx+1]-mean)/std))[0]); scored.append({"row_key":d.get("row_key"),"p":p,"selected":bool(p>=threshold),"event_reliable":bool(d.get("event_reliable",False)),"q0_reliable":bool(d.get("q0_reliable",False))})
                chosen=max(scored,key=lambda z:z["p"],default=None); return {"candidate_count":len(details),"scored_count":len(scored),"support_selected":bool(chosen and chosen["p"]>=threshold),"selected_probability":None if chosen is None else chosen["p"],"selected_event_reliable":bool(chosen and chosen["event_reliable"] and chosen["p"]>=threshold),"selected_q0_reliable":bool(chosen and chosen["q0_reliable"] and chosen["p"]>=threshold),"teacher_or_raw_fallback": "support" if chosen and chosen["p"]>=threshold else "raw"}
            src, tgt = side_eval("source"), side_eval("target"); records.append({"event_key":r["event_key"],"model_event_uid":r["model_event_uid"],"fold":fold,"polarity":r["polarity"],"prefix":int(r["prefix"]),"source":src,"target":tgt,"both_support_selected":src["support_selected"] and tgt["support_selected"],"both_support_reliable":src["selected_event_reliable"] and tgt["selected_event_reliable"],"frozen_both_reliable":bool(r.get("both_reliable")),"frozen_source_reliable":bool(r.get("source_reliable")),"frozen_target_reliable":bool(r.get("target_reliable"))})
    out=[]
    for p in (1,2,4,8,16):
        rp=[z for z in records if z["prefix"]==p and z["polarity"]=="positive"]; rn=[z for z in records if z["prefix"]==p and z["polarity"]=="negative"]
        out.append({"prefix":p,"positive_events":len(rp),"negative_events":len(rn),"frozen_source_reliable":sum(z["frozen_source_reliable"] for z in rp),"frozen_target_reliable":sum(z["frozen_target_reliable"] for z in rp),"frozen_both_reliable":sum(z["frozen_both_reliable"] for z in rp),"learned_source_support_selected":sum(z["source"]["support_selected"] for z in rp),"learned_target_support_selected":sum(z["target"]["support_selected"] for z in rp),"learned_both_support_selected":sum(z["both_support_selected"] for z in rp),"learned_both_support_reliable":sum(z["both_support_reliable"] for z in rp),"negative_both_support_selected":sum(z["both_support_selected"] for z in rn),"negative_both_support_reliable":sum(z["both_support_reliable"] for z in rn),"teacher_threshold":0.5})
    return {"schema_version":"trackocd.phase83.o_support_replay.v1","records":records,"prefix_summary":out,"positive_denominator":76,"negative_denominator":76,"threshold_source":"pre-registered p>=0.5; no held tuning","gt_used_only_posthoc":True,"public_dev_q1_sealed_accessed":False}


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--folds",default="0,1,2,3"); ap.add_argument("--steps",type=int,default=1000); ap.add_argument("--tag",default="formal"); ap.add_argument("--batch-size",type=int,default=256); args=ap.parse_args()
    folds=tuple(int(x) for x in args.folds.split(",") if x.strip()); outcomp=OUT/"completion"; outck=OUT/"checkpoints"; outmet=OUT/"metrics"; outman=OUT/"manifests"; outcomp.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(CSV_PATH.open(newline="",encoding="utf-8"))); aligned,_roi,alignment=load_aligned_features(rows); fused=aligned.astype(np.float32); fused/=np.maximum(np.linalg.norm(fused,axis=1,keepdims=True),1e-8)
    x,y,meta=build_features(rows,fused); blocked=event_videos(); fm=[]; models={}; fold_metrics={}
    for fold in folds:
        marker=outcomp/f"support_router_{args.tag}_f{fold}.launched"; done=outcomp/f"support_router_{args.tag}_f{fold}.done"
        if done.exists(): continue
        if marker.exists(): raise RuntimeError(f"unit already launched without done: {marker}")
        atomic_json(marker,{"phase":"Phase83","tag":args.tag,"fold":fold,"pid":os.getpid(),"created_utc":dt.datetime.now(dt.timezone.utc).isoformat()})
        fit,val,split=fold_sets(fold,rows,blocked); 
        if len(fit)==0 or len(val)==0: raise RuntimeError(f"empty fold {fold}: fit={len(fit)} val={len(val)}")
        mean=x[fit].mean(0); std=x[fit].std(0); std=np.where(std<1e-5,1.0,std).astype(np.float32); xf=(x-mean)/std; pos=int(y[fit].sum()); neg=len(fit)-pos; pw=float(np.clip(neg/max(pos,1),1.0,30.0)); model=MLP(x.shape[1],128,8301+fold); rng=np.random.default_rng(8301+fold); losses=[]
        for step in range(1,args.steps+1):
            idx=rng.choice(fit,size=min(args.batch_size,len(fit)),replace=len(fit)<args.batch_size); losses.append(model.step(xf[idx],y[idx].astype(np.float32),pw,0.01))
            if step%500==0 or step==args.steps:
                model.save(outck/f"support_router_{args.tag}_f{fold}_step{step:06d}.npz",mean,std,.5,step,fold)
        train_p=model.predict(xf[fit]); val_p=model.predict(xf[val]); fm_obj={"fold":fold,"tag":args.tag,"steps":args.steps,"fit":split,"fit_metrics":metrics(y[fit],train_p),"validation_metrics":metrics(y[val],val_p),"pos_weight":pw,"loss_first":losses[0],"loss_last":losses[-1],"checkpoint":str((outck/f"support_router_{args.tag}_f{fold}_step{args.steps:06d}.npz").resolve()),"checkpoint_sha256":sha(outck/f"support_router_{args.tag}_f{fold}_step{args.steps:06d}.npz"),"event_videos_excluded":sorted(blocked)}; fold_metrics[str(fold)]=fm_obj; models[fold]=(model,mean,std,.5,args.steps); atomic_json(outmet/f"support_router_{args.tag}_f{fold}.json",fm_obj); atomic_json(done,{"status":"DONE","fold":fold,"tag":args.tag,"checkpoint":fm_obj["checkpoint"],"metrics":str((outmet/f"support_router_{args.tag}_f{fold}.json").resolve())})
    # Load any completed models in this tag for event replay; this keeps
    # recovery resumable and does not relaunch completed units.
    for fold in range(4):
        cp=outck/f"support_router_{args.tag}_f{fold}_step{args.steps:06d}.npz"
        if cp.exists(): models[fold]=MLP.load(cp)
    if models:
        replay=event_replay(models,meta["row_key_to_index"],x); atomic_json(outmet/f"o_support_replay_{args.tag}.json",replay)
    inventory={"schema_version":"trackocd.phase83.support_router_inventory.v1","rows":len(rows),"features":x.shape[1],"feature_names":FEATURE_NAMES,"labels":{ "reliable_rule":"assigned == 1 AND row_iou >= 0.5 (TRAIN target only)","positive":int(y.sum()),"negative":int(len(y)-y.sum())},"event_videos_excluded":sorted(blocked),"alignment":alignment,"csv_sha256":sha(CSV_PATH),"folds":fold_metrics,"public_dev_q1_sealed_accessed":False,"future_rows_or_tracks":False,"ids_as_model_input":False,"gt_fields_in_feature_tensor":False}
    atomic_json(outman/f"support_router_inventory_{args.tag}.json",inventory); atomic_json(OUT/"metrics"/f"support_router_aggregate_{args.tag}.json",{"phase":"Phase83","tag":args.tag,"folds":fold_metrics,"event_replay":str((outmet/f"o_support_replay_{args.tag}.json").resolve()) if (outmet/f"o_support_replay_{args.tag}.json").exists() else None,"inventory":str((outman/f"support_router_inventory_{args.tag}.json").resolve())})
    print(json.dumps({"tag":args.tag,"folds":sorted(fold_metrics),"steps":args.steps,"event_replay":str(outmet/f"o_support_replay_{args.tag}.json")},indent=2,sort_keys=True))


if __name__ == "__main__": main()
