#!/usr/bin/env python3
"""Train the fixed linear B84S-Q matcher on the balanced query manifest."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase84"
BASE = Path("/data2/usr_for_deadline/trackocd_phase84/project_outputs")
DATA = BASE / "manifests/b84sq_balanced_v3_features.npz"
MAN = OUT / "manifests/b84sq_balanced_v3_manifest.json"
CK = BASE / "checkpoints"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def atom_json(p: Path, v: object) -> None:
    p.parent.mkdir(parents=True, exist_ok=True); fd, n = tempfile.mkstemp(prefix="." + p.name + ".", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(v, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(n, p)
    finally:
        if os.path.exists(n): os.unlink(n)


def atom_npz(p: Path, **kw: object) -> None:
    p.parent.mkdir(parents=True, exist_ok=True); t = p.with_name("." + p.name + ".tmp.npz"); np.savez(t, **kw); os.replace(t, p)


def metrics(w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray, x: np.ndarray, offsets: np.ndarray, targets: np.ndarray, ids: list[int]) -> dict[str, float | int]:
    rows = []
    for g in ids:
        a, z = int(offsets[g]), int(offsets[g + 1]); n = z - a; xn = (x[a:z] - mean) / std; logits = np.concatenate([xn @ w, np.asarray([b], np.float32)]); choice = int(np.argmax(logits)); target = int(targets[g]); rows.append((choice, target, n))
    match = [r for r in rows if r[1] < r[2]]; defer = [r for r in rows if r[1] >= r[2]]; selected = [r for r in rows if r[0] < r[2]]
    correct = sum(int(c == t) for c, t, n in rows); correct_match = sum(int(c == t) for c, t, n in match); top5 = 0
    for g, (c, t, n) in zip(ids, rows):
        if t >= n: continue
        a, z = int(offsets[g]), int(offsets[g + 1]); order = np.argsort(((x[a:z] - mean) / std) @ w)[::-1]
        top5 += int(t in set(order[:5]))
    return {"groups": len(rows), "target_candidate_groups": len(match), "target_defer_groups": len(defer), "candidate_top1_recall": correct_match / max(1, len(match)), "candidate_top5_recall": top5 / max(1, len(match)), "candidate_or_defer_accuracy": correct / max(1, len(rows)), "defer_recall": sum(int(c >= n) for c, t, n in defer) / max(1, len(defer)), "support_precision": correct_match / max(1, sum(int(c < n) for c, t, n in match)), "support_recall": correct_match / max(1, len(match)), "false_support_rate": sum(int(c < n) for c, t, n in defer) / max(1, len(defer)), "predicted_defer_groups": sum(int(c >= n) for c, t, n in rows)}


def train(fold: int, tag: str, epochs: int, step_limit: int = 0) -> dict[str, object]:
    z = np.load(DATA, allow_pickle=False); x = z["features"].astype(np.float32); offsets = z["offsets"].astype(np.int64); targets = z["targets"].astype(np.int64); m = json.loads(MAN.read_text(encoding="utf-8")); fs = m["folds"][str(fold)]; fit = [int(v) for v in fs["fit_groups"]]; val = [int(v) for v in fs["validation_groups"]]
    comp = OUT / "completion"; met = OUT / "metrics"; comp.mkdir(parents=True, exist_ok=True); met.mkdir(parents=True, exist_ok=True); marker = comp / f"b84sq_{tag}_f{fold}.launched"; done = comp / f"b84sq_{tag}_f{fold}.done"
    if done.exists(): return json.loads((met / f"b84sq_{tag}_f{fold}.json").read_text(encoding="utf-8"))
    if marker.exists(): raise RuntimeError(f"unit already launched without done: {marker}")
    atom_json(marker, {"phase": "Phase84", "route": "B84S-Q", "tag": tag, "fold": fold, "pid": os.getpid(), "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "gpu": "cpu", "epochs": epochs})
    fit_match = [g for g in fit if int(targets[g]) < int(offsets[g + 1] - offsets[g])]; fit_defer = [g for g in fit if int(targets[g]) >= int(offsets[g + 1] - offsets[g])]
    if not fit_match or not fit_defer: raise RuntimeError(f"fold {fold} needs both TRAIN candidate and DEFER groups")
    rows = np.concatenate([np.asarray(fit_match, np.int64), np.asarray(fit_defer, np.int64)]); maxn = max(len(fit_match), len(fit_defer)); rng = np.random.default_rng(84100 + fold); mean = x[np.concatenate([np.arange(offsets[g], offsets[g + 1]) for g in fit])].mean(0); std = np.where(x[np.concatenate([np.arange(offsets[g], offsets[g + 1]) for g in fit])].std(0) < 1e-5, 1.0, x[np.concatenate([np.arange(offsets[g], offsets[g + 1]) for g in fit])].std(0)).astype(np.float32); w = np.zeros(x.shape[1], np.float32); b = 0.0; losses = []; steps = 0
    for epoch in range(epochs):
        balanced = np.concatenate([rng.choice(fit_match, size=maxn, replace=True), rng.choice(fit_defer, size=maxn, replace=True)]).astype(np.int64); rng.shuffle(balanced)
        for g in balanced:
            a, zz = int(offsets[g]), int(offsets[g + 1]); n = zz - a; xn = (x[a:zz] - mean) / std; logits = np.concatenate([xn @ w, np.asarray([b], np.float32)]); logits -= logits.max(); p = np.exp(logits); p /= max(float(p.sum()), 1e-12); target = min(int(targets[g]), n); loss = -np.log(max(float(p[target]), 1e-12)); grad = p[:-1].astype(np.float32); 
            if target < n: grad[target] -= 1.0
            w -= np.float32(0.04) * ((xn.T @ grad) / max(1, n)); b -= np.float32(0.04) * (float(p[-1]) - (1.0 if target == n else 0.0)); losses.append(float(loss)); steps += 1
            if steps % 1000 == 0: atom_npz(CK / f"b84sq_{tag}_f{fold}_step{steps:06d}.npz", w=w, b=np.asarray([b], np.float32), mean=mean, std=std, step=np.asarray([steps]), fold=np.asarray([fold]))
            if step_limit and steps >= step_limit: break
        if step_limit and steps >= step_limit: break
    cp = CK / f"b84sq_{tag}_f{fold}_step{steps:06d}.npz"; atom_npz(cp, w=w, b=np.asarray([b], np.float32), mean=mean, std=std, step=np.asarray([steps]), fold=np.asarray([fold])); fit_m = metrics(w, b, mean, std, x, offsets, targets, fit); val_m = metrics(w, b, mean, std, x, offsets, targets, val); obj = {"schema_version": "trackocd.phase84.b84sq.metrics.v1", "phase": "Phase84 B84S-Q", "route": "SOURCE_CONDITIONED_QUERY_LISTWISE_BALANCED", "tag": tag, "fold": fold, "epochs": epochs, "steps": steps, "feature_dim": int(x.shape[1]), "fit_groups": len(fit), "validation_groups": len(val), "fit_metrics": fit_m, "validation_metrics": val_m, "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None, "checkpoint": str(cp.resolve()), "checkpoint_sha256": sha(cp), "manifest": str(MAN.resolve()), "manifest_sha256": sha(MAN), "balanced_sampling": {"match_groups": len(fit_match), "defer_groups": len(fit_defer), "target_ratio": "50/50 via replacement"}, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False, "gt_fields_in_feature_tensor": False, "gpu": "cpu"}; atom_json(met / f"b84sq_{tag}_f{fold}.json", obj); atom_json(done, {"status": "DONE", "fold": fold, "tag": tag, "metrics": str((met / f"b84sq_{tag}_f{fold}.json").resolve()), "checkpoint": str(cp.resolve())}); return obj


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--tag", default="formal_v3"); ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--steps", type=int, default=0); a = ap.parse_args(); print(json.dumps(train(a.fold, a.tag, a.epochs, a.steps), indent=2, sort_keys=True))


if __name__ == "__main__": main()
