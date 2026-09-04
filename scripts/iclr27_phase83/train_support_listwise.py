#!/usr/bin/env python3
"""Phase83 B2: TRAIN-only set-aware/listwise support assignment.

The model scores all proposals in one causal image candidate set and has an
explicit DEFER logit.  TRAIN ``assigned``/IoU values are converted to one
candidate-or-DEFER target; they are never part of the feature tensor.  Event
replay is post-hoc diagnostics on the immutable 76+76 observability manifest.
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
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.iclr27_phase23.protocol import load_aligned_features
from scripts.iclr27_phase83.build_support_candidate_sets import row_features

OUT = ROOT / "outputs/iclr27_phase83"
CSV_PATH = ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv"
DATA_PATH = Path("/data2/usr_for_deadline/trackocd_phase83/b2_candidate_sets/b2_candidate_sets_v1.npz")
MANIFEST = OUT / "manifests/b2_candidate_sets_v1.json"
OBS = Path("/data2/usr_for_deadline/trackocd_phase75b/observability_repair2/event_observability.jsonl")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class ListwiseSelector:
    """Small permutation-invariant candidate-set scorer with explicit DEFER."""

    def __init__(self, d: int, hidden: int = 96, seed: int = 8301) -> None:
        rng = np.random.default_rng(seed)
        self.d = d
        self.hidden = hidden
        self.w1 = (rng.standard_normal((d, hidden), dtype=np.float32) * np.float32(math.sqrt(2.0 / d))).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.wc = (rng.standard_normal((hidden * 3,), dtype=np.float32) * np.float32(math.sqrt(2.0 / (hidden * 3)))).astype(np.float32)
        self.bc = np.float32(0.0)
        self.wd = (rng.standard_normal((hidden * 2,), dtype=np.float32) * np.float32(math.sqrt(2.0 / (hidden * 2)))).astype(np.float32)
        self.bd = np.float32(0.0)
        self.m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.t = 0

    def params(self) -> dict[str, np.ndarray]:
        return {"w1": self.w1, "b1": self.b1, "wc": self.wc, "bc": np.asarray([self.bc], np.float32), "wd": self.wd, "bd": np.asarray([self.bd], np.float32)}

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if x.ndim != 2 or x.shape[1] != self.d or x.shape[0] < 1:
            raise ValueError(f"invalid candidate set shape {x.shape}")
        z = x @ self.w1 + self.b1
        h = np.tanh(z)
        cmean = h.mean(axis=0)
        cmax = h.max(axis=0)
        u = np.concatenate([h, np.broadcast_to(cmean, h.shape), np.broadcast_to(cmax, h.shape)], axis=1)
        cand = u @ self.wc + self.bc
        dfeat = np.concatenate([cmean, cmax])
        defer = np.asarray([dfeat @ self.wd + self.bd], np.float32)
        logits = np.concatenate([cand.astype(np.float32), defer])
        logits -= np.max(logits)
        probs = np.exp(np.clip(logits, -30.0, 30.0)).astype(np.float32)
        probs /= max(float(probs.sum()), 1e-8)
        cache = {"x": x, "h": h, "cmean": cmean, "cmax": cmax, "u": u, "dfeat": dfeat, "probs": probs}
        return probs, cache

    def loss_step(self, x: np.ndarray, target: int, weight: float, lr: float, beta1: float = .9, beta2: float = .999) -> float:
        probs, c = self.forward(x)
        target = int(target)
        if target < 0 or target >= len(probs):
            raise ValueError((target, len(probs)))
        g = probs.copy()
        g[target] -= np.float32(1.0)
        g *= np.float32(weight)
        n = x.shape[0]
        gwc = c["u"].T @ g[:n]
        gbc = np.asarray([g[:n].sum()], np.float32)
        gwd = c["dfeat"] * g[n]
        gbd = np.asarray([g[n]], np.float32)
        gu = g[:n, None] * self.wc[None, :]
        gh = gu[:, :self.hidden].copy()
        gmean = gu[:, self.hidden:2*self.hidden].sum(axis=0) + self.wd[:self.hidden] * g[n]
        gmax = gu[:, 2*self.hidden:].sum(axis=0) + self.wd[self.hidden:] * g[n]
        gh += gmean[None, :] / np.float32(n)
        argmax = np.argmax(c["h"], axis=0)
        for j, i in enumerate(argmax):
            gh[i, j] += gmax[j]
        gz = gh * (1.0 - c["h"] * c["h"])
        grads = {"w1": c["x"].T @ gz, "b1": gz.sum(axis=0), "wc": gwc, "bc": gbc, "wd": np.asarray(gwd, np.float32), "bd": gbd}
        self.t += 1
        for k, grad in grads.items():
            # Scalar parameters are kept as arrays in the optimizer state.
            param = np.asarray(self.params()[k])
            self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * grad
            self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * (grad * grad)
            mh = self.m[k] / (1.0 - beta1 ** self.t)
            vh = self.v[k] / (1.0 - beta2 ** self.t)
            update = np.float32(lr) * mh / (np.sqrt(vh) + np.float32(1e-8))
            if k == "w1": self.w1 -= update
            elif k == "b1": self.b1 -= update
            elif k == "wc": self.wc -= update
            elif k == "bc": self.bc = np.float32(self.bc - update[0])
            elif k == "wd": self.wd -= update
            elif k == "bd": self.bd = np.float32(self.bd - update[0])
        return float(-weight * math.log(max(float(probs[target]), 1e-8)))

    def choose(self, x: np.ndarray) -> tuple[int, np.ndarray]:
        p, _ = self.forward(x)
        return int(np.argmax(p)), p

    def save(self, path: Path, mean: np.ndarray, std: np.ndarray, step: int, fold: int) -> None:
        arrays = {k: v.astype(np.float32, copy=False) for k, v in self.params().items()}
        arrays.update({"mean": mean.astype(np.float32), "std": std.astype(np.float32), "step": np.asarray([step], np.int64), "fold": np.asarray([fold], np.int64), "t": np.asarray([self.t], np.int64)})
        for k in self.m:
            arrays[f"m_{k}"] = self.m[k].astype(np.float32)
            arrays[f"v_{k}"] = self.v[k].astype(np.float32)
        atomic_npz(path, arrays)

    @classmethod
    def load(cls, path: Path) -> tuple["ListwiseSelector", np.ndarray, np.ndarray, int, int]:
        z = np.load(path, allow_pickle=False)
        obj = cls(int(z["w1"].shape[0]), int(z["w1"].shape[1]), 0)
        obj.w1, obj.b1, obj.wc, obj.bc, obj.wd, obj.bd = z["w1"], z["b1"], z["wc"], float(z["bc"][0]), z["wd"], float(z["bd"][0])
        obj.t = int(z.get("t", np.asarray([0]))[0])
        for k in obj.m:
            if f"m_{k}" in z: obj.m[k] = z[f"m_{k}"]
            if f"v_{k}" in z: obj.v[k] = z[f"v_{k}"]
        return obj, z["mean"], z["std"], int(z["step"][0]), int(z["fold"][0])


def group_metrics(model: ListwiseSelector, x: np.ndarray, groups: list[int], offsets: np.ndarray, flat: np.ndarray, targets: np.ndarray, topk: tuple[int, ...] = (1, 5, 10, 20)) -> dict[str, Any]:
    total = len(groups); reliable = 0; pred_candidate = 0; correct = 0; defer_correct = 0; defer_pred = 0; losses = []; top = {k: 0 for k in topk}
    for gi in groups:
        inds = flat[offsets[gi]:offsets[gi + 1]]; target = int(targets[gi]); p, _ = model.forward(x[inds]); choice = int(np.argmax(p)); losses.append(-math.log(max(float(p[target]), 1e-8)))
        if target < len(inds):
            reliable += 1
            for k in topk:
                if target in np.argsort(-p[:len(inds)])[:k]: top[k] += 1
        if choice < len(inds): pred_candidate += 1
        else: defer_pred += 1
        if choice == target: correct += 1
        if target >= len(inds) and choice >= len(inds): defer_correct += 1
    return {"groups": total, "reliable_target_groups": reliable, "defer_target_groups": total - reliable, "predicted_candidate_groups": pred_candidate, "predicted_defer_groups": defer_pred, "candidate_or_defer_accuracy": correct / max(total, 1), "defer_accuracy": defer_correct / max(total - reliable, 1), "defer_recall": defer_correct / max(total - reliable, 1), "candidate_topk_recall": {str(k): top[k] / max(reliable, 1) for k in topk}, "mean_nll": float(np.mean(losses)) if losses else 0.0}


def load_rows_features() -> tuple[list[dict[str, str]], np.ndarray, dict[str, int]]:
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8")))
    aligned, _roi, alignment = load_aligned_features(rows)
    fused = aligned.astype(np.float32)
    fused /= np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-8)
    x = row_features(rows, fused)
    row_map = {str(r.get("row_key", "")): i for i, r in enumerate(rows)}
    return rows, x, row_map


def event_replay(models: dict[int, tuple[ListwiseSelector, np.ndarray, np.ndarray, int]], rows: list[dict[str, str]], x: np.ndarray, row_map: dict[str, int], groups: dict[tuple[int, int], list[int]]) -> dict[str, Any]:
    observations = [json.loads(line) for line in OBS.read_text(encoding="utf-8").splitlines() if line.strip()]
    records: list[dict[str, Any]] = []
    for event in observations:
        fold = int(event["fold"])
        if fold not in models: continue
        model, mean, std, step = models[fold]
        xf = (x - mean) / std
        def side_eval(side: str) -> dict[str, Any]:
            selected = []; reliable = []; details = event.get(f"{side}_row_details", [])
            for detail in details:
                video = int(detail.get("video_id", -1)); image = int(detail.get("image_id", -1)); inds = groups.get((video, image), [])
                if not inds: continue
                choice, p = model.choose(xf[np.asarray(inds, np.int64)])
                if choice >= len(inds): continue
                idx = inds[choice]; selected.append({"row_key": rows[idx].get("row_key"), "probability": float(p[choice]), "image_id": image})
                reliable.append(int(float(rows[idx].get("assigned", 0) or 0)) == 1 and float(rows[idx].get("row_iou", 0) or 0) >= .5)
            return {"candidate_count": len(details), "support_selected": bool(selected), "selected_reliable": bool(any(reliable)), "selected_count": len(selected), "selected": selected[:16], "step": step}
        src, tgt = side_eval("source"), side_eval("target")
        records.append({"event_key": event.get("event_key"), "model_event_uid": event.get("model_event_uid"), "fold": fold, "polarity": event.get("polarity"), "prefix": int(event.get("prefix", 0)), "source": src, "target": tgt, "both_support_selected": src["support_selected"] and tgt["support_selected"], "both_support_reliable": src["selected_reliable"] and tgt["selected_reliable"], "frozen_both_reliable": bool(event.get("both_reliable")), "frozen_source_reliable": bool(event.get("source_reliable")), "frozen_target_reliable": bool(event.get("target_reliable"))})
    summary = []
    for prefix in (1, 2, 4, 8, 16):
        pos = [r for r in records if r["prefix"] == prefix and r["polarity"] == "positive"]
        neg = [r for r in records if r["prefix"] == prefix and r["polarity"] == "negative"]
        summary.append({"prefix": prefix, "positive_events": len(pos), "negative_events": len(neg), "frozen_both_reliable": sum(r["frozen_both_reliable"] for r in pos), "learned_source_support_selected": sum(r["source"]["support_selected"] for r in pos), "learned_target_support_selected": sum(r["target"]["support_selected"] for r in pos), "learned_both_support_selected": sum(r["both_support_selected"] for r in pos), "learned_both_support_reliable": sum(r["both_support_reliable"] for r in pos), "negative_both_support_selected": sum(r["both_support_selected"] for r in neg), "negative_both_support_reliable": sum(r["both_support_reliable"] for r in neg)})
    return {"schema_version": "trackocd.phase83.b2.listwise_replay.v1", "records": records, "prefix_summary": summary, "positive_denominator": 76, "negative_denominator": 76, "posthoc_event_labels": True, "model_input_excludes_gt_ids_text_future": True, "public_dev_q1_sealed_accessed": False}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", default="0,1,2,3"); ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--tag", default="b2_formal"); ap.add_argument("--hidden", type=int, default=96); ap.add_argument("--batch-groups", type=int, default=1); args = ap.parse_args()
    if not DATA_PATH.exists() or not MANIFEST.exists(): raise FileNotFoundError("run build_support_candidate_sets.py first")
    z = np.load(DATA_PATH, allow_pickle=False); features = z["features"].astype(np.float32); flat = z["flat_indices"].astype(np.int64); offsets = z["offsets"].astype(np.int64); targets = z["targets"].astype(np.int64); manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    outcomp, outck, outmet = OUT / "completion", Path("/data2/usr_for_deadline/trackocd_phase83/b2_listwise_checkpoints"), OUT / "metrics"; outcomp.mkdir(parents=True, exist_ok=True); outck.mkdir(parents=True, exist_ok=True); outmet.mkdir(parents=True, exist_ok=True)
    requested = tuple(int(v) for v in args.folds.split(",") if v.strip()); models: dict[int, tuple[ListwiseSelector, np.ndarray, np.ndarray, int]] = {}; fold_metrics: dict[str, Any] = {}
    for fold in requested:
        marker = outcomp / f"b2_listwise_{args.tag}_f{fold}.launched"; done = outcomp / f"b2_listwise_{args.tag}_f{fold}.done"
        if done.exists(): continue
        if marker.exists(): raise RuntimeError(f"unit already launched without done: {marker}")
        atomic_json(marker, {"phase": "Phase83", "route": "B2_LISTWISE", "tag": args.tag, "fold": fold, "pid": os.getpid(), "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        fs = manifest["folds"][str(fold)]; fit = [int(v) for v in fs["fit_groups"]]; val = [int(v) for v in fs["validation_groups"]]
        if not fit or not val: raise RuntimeError(f"empty fold {fold}: fit={len(fit)} val={len(val)}")
        fit_flat = flat[np.concatenate([np.arange(offsets[g], offsets[g + 1]) for g in fit])]; mean = features[fit_flat].mean(axis=0); std = features[fit_flat].std(axis=0); std = np.where(std < 1e-5, 1.0, std).astype(np.float32); xf = (features - mean) / std
        rel = [g for g in fit if targets[g] < offsets[g + 1] - offsets[g]]; non = [g for g in fit if g not in set(rel)]; rng = np.random.default_rng(8301 + fold); model = ListwiseSelector(features.shape[1], args.hidden, 8301 + fold); losses = []; best_val = float("inf"); best_step = 0
        for step in range(1, args.steps + 1):
            pool = rel if (rel and non and rng.random() < .5) else (non if non else rel); gi = int(pool[rng.integers(0, len(pool))]); inds = flat[offsets[gi]:offsets[gi + 1]]; weight = 2.0 if gi in set(rel) else 1.0; losses.append(model.loss_step(xf[inds], int(targets[gi]), weight, .003))
            if step % 500 == 0 or step == args.steps:
                cp = outck / f"b2_listwise_{args.tag}_f{fold}_step{step:06d}.npz"; model.save(cp, mean, std, step, fold); vm = group_metrics(model, xf, val, offsets, flat, targets); 
                if vm["mean_nll"] < best_val: best_val = vm["mean_nll"]; best_step = step; model.save(outck / f"b2_listwise_{args.tag}_f{fold}_best.npz", mean, std, step, fold)
        train_m = group_metrics(model, xf, fit, offsets, flat, targets); val_m = group_metrics(model, xf, val, offsets, flat, targets); cp = outck / f"b2_listwise_{args.tag}_f{fold}_step{args.steps:06d}.npz"; obj = {"phase": "Phase83", "route": "B2_LISTWISE", "tag": args.tag, "fold": fold, "steps": args.steps, "fit_groups": len(fit), "validation_groups": len(val), "fit_metrics": train_m, "validation_metrics": val_m, "best_step_by_validation_nll": best_step, "best_validation_nll": best_val, "loss_first": losses[0], "loss_last": losses[-1], "checkpoint": str(cp.resolve()), "checkpoint_sha256": sha(cp), "candidate_manifest": str(MANIFEST.resolve()), "candidate_manifest_sha256": sha(MANIFEST), "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False, "gt_fields_in_feature_tensor": False}; atomic_json(outmet / f"b2_listwise_{args.tag}_f{fold}.json", obj); atomic_json(done, {"status": "DONE", "fold": fold, "tag": args.tag, "checkpoint": str(cp.resolve()), "metrics": str((outmet / f"b2_listwise_{args.tag}_f{fold}.json").resolve())})
        best = outck / f"b2_listwise_{args.tag}_f{fold}_best.npz"; models[fold] = ListwiseSelector.load(best)[:4]
        fold_metrics[str(fold)] = obj
    # Resumable loading for a subsequent invocation; replay uses TRAIN-val NLL best.
    for fold in range(4):
        best = outck / f"b2_listwise_{args.tag}_f{fold}_best.npz"
        if best.exists() and fold not in models: models[fold] = ListwiseSelector.load(best)[:4]
    if models:
        rows, all_x, row_map = load_rows_features(); all_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, r in enumerate(rows): all_groups[(int(r["video_id"]), int(r["image_id"]))].append(i)
        for k in all_groups: all_groups[k].sort(key=lambda i: (int(float(rows[i].get("proposal_local_id", 0) or 0)), int(float(rows[i].get("track_id", 0) or 0)), i))
        replay = event_replay(models, rows, all_x, row_map, all_groups); atomic_json(outmet / f"b2_listwise_replay_{args.tag}.json", replay)
    aggregate = {"phase": "Phase83", "route": "B2_LISTWISE", "tag": args.tag, "steps": args.steps, "folds": fold_metrics, "replay": str((outmet / f"b2_listwise_replay_{args.tag}.json").resolve()) if (outmet / f"b2_listwise_replay_{args.tag}.json").exists() else None, "candidate_manifest": str(MANIFEST.resolve()), "candidate_manifest_sha256": sha(MANIFEST), "labels_used_only_for_train_target_or_posthoc": True, "public_dev_q1_sealed_accessed": False, "controller_run": False, "sealed_run": False}; atomic_json(outmet / f"b2_listwise_aggregate_{args.tag}.json", aggregate); atomic_json(OUT / "status.json", {"phase": "Phase83", "route": "B2_LISTWISE", "status": "FORMAL_COMPLETE" if len(models) == 4 else "PARTIAL", "tag": args.tag, "aggregate": str((outmet / f"b2_listwise_aggregate_{args.tag}.json").resolve()), "public_dev_q1_sealed_accessed": False})
    print(json.dumps({"status": "COMPLETE", "route": "B2_LISTWISE", "tag": args.tag, "folds": sorted(models), "replay": str(outmet / f"b2_listwise_replay_{args.tag}.json")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
