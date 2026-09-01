"""Phase19 baseline/model evaluation under one causal event interface."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.iclr27_phase19.data.stream import Phase19Data
from src.iclr27_phase19.models.ra_ocd import RAOCD
from src.iclr27_phase19.runtime.state_machine import CausalStateMachine

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "data/iclr27_phase19/sources"
OUT = ROOT / "outputs/iclr27_phase19"


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp"); tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(tmp, path)


def sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def load_events() -> list[dict[str, Any]]:
    out = []
    for name in ["positive_events.jsonl", "negative_events.jsonl"]:
        out.extend(json.loads(x) for x in (SRC / name).read_text().splitlines() if x.strip())
    return sorted(out, key=lambda x: x["event_key"])


class RawController:
    def __init__(self, data: Phase19Data, variant: str = "raw", deferred: bool = True):
        self.data = data; self.variant = variant; self.deferred = deferred
        self.known = data.known_prototypes()
        self.states: list[dict[str, Any]] = []; self.next_sid = 100000
        self.tau_known = .72; self.tau_existing = .70; self.tau_ready = .45

    def reset(self) -> None:
        self.states = []; self.next_sid = 100000

    def _score(self, raw: np.ndarray, state: dict[str, Any]) -> float:
        sim = float(raw @ state["raw"])
        if self.variant == "age":
            # AGE-inspired shrinkage: down-weight diffuse/high-count states.
            return sim / (1.0 + .08 * float(state.get("dispersion", 0.0)))
        return sim

    def process_track(self, key: str, eval_cat: int | None = None, phase: str = "") -> list[dict[str, Any]]:
        self.reset() if phase == "reset" else None
        decisions = []
        for pos, idx in enumerate(self.data.track_rows[key]):
            raw, geom, quality, _ = self.data.prefix(key, pos)
            row = self.data.rows[idx]; video = int(row["video_id"])
            ready = (not self.deferred) or quality >= self.tau_ready
            action = "DEFER"; sid = None; confidence = quality
            if ready:
                sims = raw @ self.known.T
                known_i = int(np.argmax(sims)); known_s = float(sims[known_i])
                candidates = [(j, s) for j, s in enumerate(self.states)
                              if s["birth_track"] != key and s["birth_video"] != video]
                best = max(candidates, key=lambda x: self._score(raw, self.states[x[0]]), default=(None, -1.0))
                state_s = self._score(raw, self.states[best[0]]) if best[0] is not None else -1.0
                if known_s >= self.tau_known:
                    action, sid, confidence = "KNOWN", int(self.data.supported_ids[known_i]), known_s
                elif state_s >= self.tau_existing:
                    action, sid, confidence = "EXISTING", int(self.states[best[0]]["sid"]), state_s
                else:
                    action, sid, confidence = "NEW", self.next_sid, max(0.0, 1.0 - known_s)
                    self.next_sid += 1
                    self.states.append({"sid": sid, "raw": raw.copy(), "birth_video": video,
                                        "birth_track": key, "eval_category": eval_cat,
                                        "count": 1, "dispersion": 0.0})
                if action == "EXISTING":
                    st = next(s for s in self.states if s["sid"] == sid)
                    old = st["raw"].copy(); st["raw"] = .8 * old + .2 * raw
                    st["raw"] /= max(np.linalg.norm(st["raw"]), 1e-6)
                    st["dispersion"] = .8 * st["dispersion"] + .2 * float(1.0 - old @ raw)
                    st["count"] += 1
            decisions.append({"row_key": row["row_key"], "tracklet_position": pos, "phase": phase,
                              "action": action, "semantic_id": sid, "readiness": quality,
                              "confidence": confidence, "video": video})
        return decisions


class ModelController:
    def __init__(self, data: Phase19Data, checkpoint: Path, device: torch.device,
                 deferred: bool = True, raw_only: bool = False):
        self.data = data; self.device = device; self.deferred = deferred; self.raw_only = raw_only
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.model = RAOCD(torch.from_numpy(data.known_prototypes())).to(device)
        self.model.load_state_dict(ckpt["model_state"]); self.model.eval()
        self.tau_ready = .45
        self.track_cache: dict[str, list[tuple[np.ndarray, np.ndarray, float]]] = {}

    def _embeddings(self, key: str) -> list[tuple[np.ndarray, np.ndarray, float]]:
        if key in self.track_cache: return self.track_cache[key]
        out = []
        with torch.no_grad():
            for pos in range(len(self.data.track_rows[key])):
                raw, geom, quality, _ = self.data.prefix(key, pos)
                if self.raw_only:
                    out.append((raw, geom, quality)); continue
                rt = torch.from_numpy(raw).to(self.device)[None]
                gt = torch.from_numpy(geom).to(self.device)[None]
                e = self.model.embed(rt, gt)
                out.append((e["z_raw"][0].cpu().numpy(), e["z"][0].cpu().numpy(), float(e["quality"].item())))
        self.track_cache[key] = out
        return out

    def process_track(self, key: str, eval_cat: int | None = None, phase: str = "") -> tuple[list[dict[str, Any]], CausalStateMachine]:
        sm = CausalStateMachine(self.model, len(self.data.supported_ids), max_states=8,
                                allow_defer=self.deferred, tau_ready=self.tau_ready)
        outputs = []
        for pos, (raw, z, quality) in enumerate(self._embeddings(key)):
            # The shared runtime accepts raw plus causal geometry; model-side
            # embedding is recomputed from the same prefix, never future rows.
            idx = self.data.track_rows[key][pos]; row = self.data.rows[idx]
            geom = torch.from_numpy(self.data.prefix(key, pos)[1]).to(self.device)
            got = sm.predict(torch.from_numpy(raw).to(self.device), geom, int(row["video_id"]), key, quality_override=quality)
            if got["semantic_id"] is not None:
                for st in sm.states:
                    if st.sid == got["semantic_id"] and st.oracle_category is None:
                        st.oracle_category = eval_cat
            sid_out = (self.data.supported_ids[int(got["semantic_id"])]
                       if got["action"] == "KNOWN" and got["semantic_id"] is not None
                       else got["semantic_id"])
            outputs.append({"row_key": row["row_key"], "tracklet_position": pos, "phase": phase,
                            "action": got["action"], "semantic_id": sid_out,
                            "readiness": got["readiness"], "confidence": got["confidence"],
                            "video": int(row["video_id"])})
        return outputs, sm


def simulate_event(controller: Any, event: dict[str, Any]) -> dict[str, Any]:
    # Each event receives a fresh causal memory; source tracks are processed in
    # the registered order, then the target track.
    if hasattr(controller, "reset"): controller.reset()
    source = []
    source_cat = int(event.get("category_gt_denominator_only", event.get("distractor_category_gt_denominator_only")))
    if isinstance(controller, ModelController):
        # ModelController returns a state machine per track; combine states in
        # one event-level machine by using a local adapter below.
        machine = CausalStateMachine(controller.model, len(controller.data.supported_ids), max_states=8,
                                     allow_defer=controller.deferred, tau_ready=controller.tau_ready)
        def proc(key: str, cat: int, phase: str):
            out=[]
            for pos, (raw, z, quality) in enumerate(controller._embeddings(key)):
                idx=controller.data.track_rows[key][pos]; row=controller.data.rows[idx]
                geom=torch.from_numpy(controller.data.prefix(key,pos)[1]).to(controller.device)
                got=machine.predict(torch.from_numpy(raw).to(controller.device),geom,int(row['video_id']),key,quality_override=quality)
                if got['semantic_id'] is not None:
                    for st in machine.states:
                        if st.sid==got['semantic_id'] and st.oracle_category is None: st.oracle_category=cat
                sid_out = (controller.data.supported_ids[int(got['semantic_id'])]
                           if got['action'] == 'KNOWN' and got['semantic_id'] is not None
                           else got['semantic_id'])
                out.append({'row_key':row['row_key'],'tracklet_position':pos,'phase':phase,'action':got['action'],
                            'semantic_id':sid_out,'readiness':got['readiness'],'confidence':got['confidence'],
                            'video':int(row['video_id'])})
            return out
    else:
        machine = controller
        def proc(key: str, cat: int, phase: str): return machine.process_track(key, eval_cat=cat, phase=phase)
    for key in event["source_tracklet_keys"]:
        source.extend(proc(key, source_cat, "source"))
    target_cat = int(event.get("category_gt_denominator_only", event.get("target_category_gt_denominator_only")))
    target = proc(event["target_tracklet_key"], target_cat, "target")
    prefix = int(event["target_first_reliable_prefix_index_gt_only"])
    state_meta = {}
    if isinstance(controller, ModelController):
        state_meta = {s.sid: {"eval_category": s.oracle_category, "birth_video": s.video, "birth_track": s.track_key} for s in machine.states}
    else:
        state_meta = {s["sid"]: s for s in machine.states}
    def correct(d):
        if d["action"] != "EXISTING" or d["semantic_id"] is None: return False
        s = state_meta.get(int(d["semantic_id"]));
        return bool(s and s.get("eval_category") == target_cat and s.get("birth_video") != int(event["target_video"]) and s.get("birth_track") != event["target_tracklet_key"])
    post = target[prefix:]
    existing_post = [d for d in post if d["action"] == "EXISTING"]
    first = next((d for d in post if d["action"] != "DEFER"), None)
    premature = [d for d in target[:prefix] if d["action"] != "DEFER"]
    state_births = [{"semantic_id": int(sid), "birth_video": int(s.get("birth_video", s.get("video", -1))),
                     "birth_track": s.get("birth_track", s.get("track_key", ""))}
                    for sid, s in state_meta.items()]
    return {"event_key": event["event_key"], "kind": event["kind"], "fold": event.get("fold"),
            "target_category_evaluator_only": target_cat, "source_decisions": source,
            "target_decisions": target, "first_commit_after_prefix": first,
            "first_commit_correct": bool(first and correct(first)),
            "post_prefix_correct_rows": int(sum(correct(d) for d in post)), "post_prefix_rows": len(post),
            "existing_correct_rows": int(sum(correct(d) for d in existing_post)), "existing_rows": len(existing_post),
            "pre_prefix_rows": prefix, "pre_prefix_defer_rows": int(sum(d["action"] == "DEFER" for d in target[:prefix])),
            "premature_commit": bool(premature), "unresolved": first is None,
            "state_count": len(state_meta), "duplicate_target_births": int(sum(s.get("eval_category") == target_cat and s.get("birth_video") == int(event["target_video"]) for s in state_meta.values())),
            "correct_category": target_cat if any(correct(d) for d in post) else None,
            "target_video": int(event["target_video"]), "latency": next((i for i,d in enumerate(post) if correct(d)), None),
            "state_births": state_births}


def event_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    pos = [r for r in records if r["kind"] == "positive_existing"]
    neg = [r for r in records if r["kind"] != "positive_existing"]
    commits = [r for r in pos if r["first_commit_after_prefix"] is not None]
    return {"positive_events": len(pos), "negative_events": len(neg),
            "commit_ct": {"correct": int(sum(r["first_commit_correct"] for r in pos)), "eligible": len(pos),
                          "recall": float(np.mean([r["first_commit_correct"] for r in pos])) if pos else 0.0},
            "post_prefix_ct": {"correct_rows": int(sum(r["post_prefix_correct_rows"] for r in pos)), "rows": int(sum(r["post_prefix_rows"] for r in pos)),
                               "recall": float(sum(r["post_prefix_correct_rows"] for r in pos) / max(sum(r["post_prefix_rows"] for r in pos), 1))},
            "existing_precision": float(sum(r["existing_correct_rows"] for r in records) / max(sum(r["existing_rows"] for r in records), 1)),
            "existing_recall": float(sum(r["existing_correct_rows"] for r in pos) / max(sum(r["post_prefix_rows"] for r in pos), 1)),
            "negative_false_merge_rate": float(np.mean([r["existing_rows"] > 0 for r in neg])) if neg else 0.0,
            "premature_rate": float(np.mean([r["premature_commit"] for r in records])) if records else 0.0,
            "unresolved_rate": float(np.mean([r["unresolved"] for r in records])) if records else 0.0,
            "pre_prefix_defer_rate": float(sum(r["pre_prefix_defer_rows"] for r in records) / max(sum(r["pre_prefix_rows"] for r in records),1)),
            "mean_latency": float(np.mean([r["latency"] for r in pos if r["latency"] is not None])) if any(r["latency"] is not None for r in pos) else None,
            "duplicate_births": int(sum(r["duplicate_target_births"] for r in records)),
            "category_coverage": len({r["target_category_evaluator_only"] for r in pos if r["first_commit_correct"]}),
            "video_coverage": len({r["target_video"] for r in pos if r["first_commit_correct"]})}


def public_eval(controller: Any, name: str, raw_only: bool = False) -> dict[str, Any]:
    events = load_events(); records = [simulate_event(controller, e) for e in events]
    if raw_only:
        # Remove all evaluator category joins and derived correctness fields.
        keep = []
        for r in records:
            keep.append({k: v for k, v in r.items() if k not in {
                "target_category_evaluator_only", "first_commit_correct", "post_prefix_correct_rows",
                "existing_correct_rows", "correct_category", "duplicate_target_births", "latency",
            }})
        return {"protocol": "trackocd_iclr27_phase19_raw_prediction_freeze", "candidate": name,
                "event_count": len(keep), "records": keep}
    return {"protocol": "trackocd_iclr27_phase19_public_fixed_events", "candidate": name,
            "event_count": len(records), "metrics": event_metrics(records), "records": records}


def standard_internal(data: Phase19Data, variant: str, checkpoint: Path | None,
                      device: torch.device, ladder: str) -> dict[str, Any]:
    rng = np.random.default_rng(1901 + data.fold)
    def include(k: str) -> bool:
        if ladder == "L2":
            return True
        qualities = []
        for pos in range(len(data.track_rows[k])):
            qualities.append(data.prefix(k, pos)[2])
        if ladder == "L0":
            return any(bool(data.rows[i]["assigned"] == "1" and float(data.rows[i]["row_iou"]) >= .5)
                       for i in data.track_rows[k])
        return float(np.mean(qualities)) >= .35
    held_tracks = [k for k in data.track_rows if include(k) and data.track_cat_eval[k] in data.held_categories and data.track_video[k] in set(data.fold_record["validation_videos"])]
    visible_tracks = [k for k in data.track_rows if include(k) and data.track_cat_eval[k] in data.supported_set and data.track_cat_eval[k] not in data.held_categories and data.track_video[k] in set(data.fold_record["validation_videos"])]
    order_metrics=[]
    for order in range(3):
        tracks = held_tracks + visible_tracks
        rng.shuffle(tracks)
        if variant == "raw": c = RawController(data, "raw", deferred=False)
        elif variant == "age": c = RawController(data, "age", deferred=False)
        elif variant == "talon": c = RawController(data, "talon", deferred=False)
        else: c = ModelController(data, checkpoint, device, deferred=False)
        pred=[]; gt=[]; novel_pred=[]; novel_gt=[]
        for key in tracks:
            cat=data.track_cat_eval[key]
            if isinstance(c, ModelController): out, sm = c.process_track(key, cat, "internal")
            else: out=c.process_track(key, cat, "internal")
            commits=[x for x in out if x["action"] != "DEFER"]
            if not commits: p=-1
            else: p=commits[-1]["semantic_id"] if commits[-1]["action"] != "KNOWN" else commits[-1]["semantic_id"]
            pred.append(p); gt.append(cat)
            if cat in data.held_categories: novel_pred.append(p); novel_gt.append(cat)
        # Strict Hungarian accuracy over all track instances.
        pu=sorted(set(pred)); gu=sorted(set(gt)); mat=np.zeros((len(pu),len(gu)),int)
        for p,g in zip(pred,gt): mat[pu.index(p),gu.index(g)]+=1
        rr,cc=linear_sum_assignment(-mat) if mat.size else ([],[])
        mapping={pu[int(r)]:gu[int(c)] for r,c in zip(rr,cc)}
        all_acc=float(mat[rr,cc].sum()/max(len(gt),1)) if len(rr) else 0.0
        old=[int(p==g) for p,g in zip(pred,gt) if g not in data.held_categories]
        nmi=normalized_mutual_info_score(novel_gt,novel_pred) if len(set(novel_gt))>1 and novel_pred else 0.0
        ari=adjusted_rand_score(novel_gt,novel_pred) if len(novel_gt)>1 else 0.0
        new_acc=float(np.mean([mapping.get(p, object()) == g for p,g in zip(novel_pred,novel_gt)])) if novel_pred else 0.0
        order_metrics.append({"order":order,"all_accuracy":all_acc,"old_accuracy":float(np.mean(old)) if old else 0.0,
                              "new_hungarian_accuracy":new_acc,
                              "nmi_novel":float(nmi),"ari_novel":float(ari),"novel_discovery_count_error":abs(len(set(novel_pred))-len(set(novel_gt))),
                              "tracks":len(tracks),"held_track_count":len(held_tracks),"visible_track_count":len(visible_tracks)})
    return {"ladder":ladder,"fold":data.fold,"candidate":variant,"orders":order_metrics,
            "mean":{k:float(np.mean([x[k] for x in order_metrics])) for k in order_metrics[0] if k not in {"order","tracks"}},
            "order_sensitivity":float(np.std([x["all_accuracy"] for x in order_metrics]))}


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=["internal","public"],required=True)
    p.add_argument("--candidate",choices=["raw","age","talon","main","fallback_a"],default="raw")
    p.add_argument("--checkpoint",type=Path); p.add_argument("--fold",type=int,default=0)
    p.add_argument("--ladder",choices=["L0","L1","L2"],default="L0"); p.add_argument("--device",default="cpu")
    p.add_argument("--out",type=Path,required=True); p.add_argument("--raw-only",action="store_true"); args=p.parse_args()
    data=Phase19Data(args.fold, final=args.mode=="public")
    device=torch.device(args.device)
    if args.mode=="public":
        c=RawController(data,"raw",deferred=True) if args.candidate=="raw" else RawController(data,args.candidate,deferred=True) if args.candidate in {"age","talon"} else ModelController(data,args.checkpoint,device,deferred=True)
        result=public_eval(c,args.candidate,raw_only=args.raw_only)
    else:
        result=standard_internal(data,args.candidate,args.checkpoint,device,args.ladder)
    atomic(args.out,result); print(json.dumps({"complete":True,"mode":args.mode,"candidate":args.candidate,"out":str(args.out),"metrics":result.get("metrics",result.get("mean"))},indent=2))


if __name__=="__main__": main()
