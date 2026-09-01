"""Phase 4C error-decomposition audits.

Produces the four required audit CSV sets:
  1. known_error_decomposition.csv / known_error_by_class.csv / known_error_by_domain.csv
  2. conditional_novel_error_decomposition.csv / conditional_error_by_occurrence.csv /
     conditional_error_by_support.csv
  3. meta_dev_validation_shift.csv
  4. shared_action_head_consistency.csv

Replays are performed once per method and reused for every table, so the
per-track totals remain consistent across tables.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.iclr27_phase4c.audit_common import (
    attach_gt,
    build_meta_proxy_rows,
    decorate_rows,
    emit_preds,
    frozen_known_protos,
    replay_all,
    assignment_from_preds,
)
from src.orbit.evaluate import load_model, build_known
from src.orbit.protocol import (
    load_frame_features,
    load_mean_features,
    load_train_labels,
    load_stream,
    load_gt,
)
from src.orbit.evaluate import embed_track

DEVICE = "cuda"
OUT = ROOT / "outputs" / "iclr27_phase4c" / "audit"
OUT.mkdir(parents=True, exist_ok=True)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_split(rows, feats, mean_feats, zs, rels, frozen_protos, adapted_protos,
              radii, model, labels, mode, birth_threshold, tag):
    decorate_rows(rows, feats, mean_feats, zs, frozen_protos, adapted_protos, labels)
    logs = replay_all(rows, feats, [r["sample_id"] for r in rows], zs, rels,
                      adapted_protos, radii, model, DEVICE, mode=mode,
                      birth_threshold=birth_threshold)
    return logs


def known_metrics(logs):
    known = [l for l in logs if l["true_role"] in ("supported_known", "zero_shot_known")]
    out = Counter()
    for l in known:
        if l["predicted_action"] == "KNOWN":
            if l["predicted_known_id"] is not None and int(l["predicted_known_id"]) == int(l["true_class"]):
                out["correct_known"] += 1
            else:
                out["wrong_known"] += 1
        elif l["predicted_action"] == "EXISTING_NOVEL":
            out["known_to_existing_novel"] += 1
        elif l["predicted_action"] == "NEW_NOVEL":
            out["known_to_new_novel"] += 1
        else:
            out["known_unresolved"] += 1
    return out, len(known)


def conditional_novel_metrics(logs, assignment):
    novel = [l for l in logs if l["true_role"] == "novel"]
    routed = [l for l in novel if l["predicted_virtual_novel_id"] is not None]
    out = Counter()
    detailed = []
    end_vid_gt = defaultdict(Counter)
    for l in routed:
        vid = int(l["predicted_virtual_novel_id"])
        end_vid_gt[vid][int(l["true_class"])] += 1
    for l in routed:
        vid = int(l["predicted_virtual_novel_id"])
        mapped = assignment.get(vid)
        correct = mapped is not None and mapped == int(l["true_class"])
        first = l["first_occurrence"] in (True, "True", "1")
        action = l["predicted_action"]
        if first and action == "EXISTING_NOVEL":
            cat = "first_occurrence_merged_to_existing"
        elif (not first) and action == "NEW_NOVEL":
            cat = "repeated_false_birth"
        elif (not first) and action == "EXISTING_NOVEL" and not correct:
            cat = "wrong_existing_assignment"
        elif first and action == "NEW_NOVEL" and not correct:
            cat = "first_birth_unmatched"
        elif first and action == "NEW_NOVEL" and correct:
            cat = "correct_first_birth"
        elif not first and action == "EXISTING_NOVEL" and correct:
            cat = "correct_existing_match"
        else:
            cat = "other"
        out[cat] += 1
        # pollution-chain secondary flag: virtual id mixed across >=2 GT classes
        mixed = len(end_vid_gt[vid]) >= 2
        is_minority = mixed and mapped is not None and int(l["true_class"]) != mapped
        detailed.append({
            "sample_id": l["sample_id"],
            "true_class": l["true_class"],
            "first_occurrence": l["first_occurrence"],
            "predicted_action": action,
            "virtual_id": vid,
            "mapped_class": mapped,
            "correct": correct,
            "error_category": cat,
            "polluted_cluster": mixed,
            "pollution_chain_error": is_minority,
            "best_novel_similarity": float(l["best_novel_similarity"]),
            "novel_margin": float(l["novel_margin"]),
            "prototype_support": int(l["prototype_support"]),
            "track_length": int(l["track_length"]),
            "domain": l["domain"],
            "arrival_index": int(l["arrival_index"]),
        })
    return out, len(novel), len(routed), detailed


def bucketize(v, edges):
    v = float(v)
    for i, e in enumerate(edges):
        if v < e:
            return f"<{e}"
    return f">={edges[-1]}"


def main():
    model, _ = load_model(ROOT / "runs/orbit/model_D1_b128_g0.3/model.pth", device=DEVICE)
    train_feats = load_frame_features("train_known_mean")
    train_labels = load_train_labels()
    all_classes = set(train_labels.values())
    adapted_protos, radii = build_known(model, train_feats, train_labels, all_classes, DEVICE)
    frozen_protos = frozen_known_protos(all_classes)

    # ---- official validation seed1027 ----
    val_rows = load_stream("pure", "main_seed1027")
    val_gt = load_gt("pure")
    val_gt_by_sid = {g["sample_id"]: g for g in val_gt}
    attach_gt(val_rows, val_gt_by_sid)
    val_feats = {sid: f[:8] for sid, f in load_frame_features("gt_tracks_mean").items()}
    val_mean = load_mean_features("gt_tracks_mean")
    val_sids = [r["sample_id"] for r in val_rows]
    val_mean_arr = np.stack([val_mean[s] for s in val_sids]).astype(np.float32)
    val_ones = np.ones(len(val_sids), dtype=np.float32)
    zs_val = np.zeros((len(val_sids), 768), dtype=np.float32)
    rels_val = np.zeros(len(val_sids), dtype=np.float32)
    for i, sid in enumerate(val_sids):
        zs_val[i], rels_val[i] = embed_track(model, val_feats[sid], DEVICE)

    logs_val = {}
    preds_val = {}
    assigns_val = {}
    decorate_rows(val_rows, val_feats, val_mean, zs_val, frozen_protos,
                  adapted_protos, train_labels)
    logs_ref_val = replay_all(val_rows, val_feats, val_sids, val_mean_arr,
                              val_ones, frozen_protos, {}, None, DEVICE,
                              mode="sequential")
    logs_d1_val = replay_all(val_rows, val_feats, val_sids, zs_val, rels_val,
                             adapted_protos, radii, model, DEVICE, mode="joint")
    logs_bc_val = replay_all(val_rows, val_feats, val_sids, zs_val, rels_val,
                             adapted_protos, radii, model, DEVICE, mode="bc",
                             birth_threshold=0.55)
    logs_val = {"ref": logs_ref_val, "d1": logs_d1_val, "bc": logs_bc_val}
    for tag, logs in logs_val.items():
        preds = emit_preds(logs)
        preds_val[tag] = preds
        res, _ = assignment_from_preds(preds, val_gt)
        assigns_val[tag] = res["hungarian_assignment"]
        print(tag, "val all/known/rn/cond:", round(res["all_track_acc"], 4),
              round(res["overall_known_acc"], 4), round(res["route_aware_novel_acc"], 4),
              round(res["conditional_novel_acc"], 4), flush=True)

    # ---- meta-dev proxy ----
    proxy_rows, proxy_gt_by_sid, proxy_feats_all, _ = build_meta_proxy_rows()
    proxy_feats = {sid: f[:8] for sid, f in proxy_feats_all.items()}
    attach_gt(proxy_rows, proxy_gt_by_sid)
    proxy_mean = load_mean_features("train_known_mean")
    proxy_sids = [r["sample_id"] for r in proxy_rows]
    proxy_mean_arr = np.stack([proxy_mean[s] for s in proxy_sids]).astype(np.float32)
    proxy_ones = np.ones(len(proxy_sids), dtype=np.float32)
    zs_proxy = np.zeros((len(proxy_sids), 768), dtype=np.float32)
    rels_proxy = np.zeros(len(proxy_sids), dtype=np.float32)
    for i, sid in enumerate(proxy_sids):
        zs_proxy[i], rels_proxy[i] = embed_track(model, proxy_feats[sid], DEVICE)
    proxy_gt = list(proxy_gt_by_sid.values())
    logs_proxy = {}
    assigns_proxy = {}
    decorate_rows(proxy_rows, proxy_feats, proxy_mean, zs_proxy, frozen_protos,
                  adapted_protos, train_labels)
    logs_ref_proxy = replay_all(proxy_rows, proxy_feats, proxy_sids,
                                proxy_mean_arr, proxy_ones, frozen_protos, {},
                                None, DEVICE, mode="sequential")
    logs_d1_proxy = replay_all(proxy_rows, proxy_feats, proxy_sids, zs_proxy,
                               rels_proxy, adapted_protos, radii, model,
                               DEVICE, mode="joint")
    logs_bc_proxy = replay_all(proxy_rows, proxy_feats, proxy_sids, zs_proxy,
                               rels_proxy, adapted_protos, radii, model,
                               DEVICE, mode="bc", birth_threshold=0.55)
    logs_proxy = {"ref": logs_ref_proxy, "d1": logs_d1_proxy, "bc": logs_bc_proxy}
    for tag, logs in logs_proxy.items():
        preds = emit_preds(logs)
        res, _ = assignment_from_preds(preds, proxy_gt)
        assigns_proxy[tag] = res["hungarian_assignment"]
        print(tag, "proxy all/known/rn/cond:", round(res["all_track_acc"], 4),
              round(res["overall_known_acc"], 4), round(res["route_aware_novel_acc"], 4),
              round(res["conditional_novel_acc"], 4), flush=True)

    # ============ 1. known error decomposition ============
    dec_rows = []
    for tag in ["ref", "d1", "bc"]:
        counts, n_known = known_metrics(logs_val[tag])
        row = {"method": tag.upper(), "num_known_tracks": n_known}
        for k in ["correct_known", "wrong_known", "known_to_existing_novel",
                  "known_to_new_novel", "known_unresolved"]:
            row[k] = counts[k]
            row[k + "_rate"] = round(counts[k] / n_known, 6) if n_known else 0.0
        dec_rows.append(row)
        # totals consistency guard
        assert sum(counts.values()) == n_known, (tag, sum(counts.values()), n_known)
    write_csv(OUT / "known_error_decomposition.csv", dec_rows)

    by_class_rows = []
    for tag in ["ref", "d1", "bc"]:
        agg = defaultdict(Counter)
        for l in logs_val[tag]:
            if l["true_role"] not in ("supported_known", "zero_shot_known"):
                continue
            if l["predicted_action"] == "KNOWN":
                ok = l["predicted_known_id"] is not None and int(l["predicted_known_id"]) == int(l["true_class"])
                agg[l["true_class"]]["correct" if ok else "wrong_known"] += 1
            else:
                agg[l["true_class"]][l["predicted_action"].lower()] += 1
        for c in sorted(agg, key=lambda x: int(x)):
            row = {"method": tag.upper(), "known_class": c,
                   "class_frequency": logs_val[tag][0]["class_frequency"]}
            a = agg[c]
            row.update({k: a[k] for k in ["correct", "wrong_known", "existing_novel", "new_novel", "unresolved"]})
            row["total"] = sum(a.values())
            by_class_rows.append(row)
    write_csv(OUT / "known_error_by_class.csv", by_class_rows)

    by_domain_rows = []
    for tag in ["ref", "d1", "bc"]:
        agg = defaultdict(Counter)
        for l in logs_val[tag]:
            if l["true_role"] not in ("supported_known", "zero_shot_known"):
                continue
            if l["predicted_action"] == "KNOWN":
                ok = l["predicted_known_id"] is not None and int(l["predicted_known_id"]) == int(l["true_class"])
                agg[l["domain"]]["correct" if ok else "wrong_known"] += 1
            else:
                agg[l["domain"]][l["predicted_action"].lower()] += 1
        for d in sorted(agg):
            a = agg[d]
            row = {"method": tag.upper(), "domain": d}
            row.update({k: a[k] for k in ["correct", "wrong_known", "existing_novel", "new_novel", "unresolved"]})
            row["total"] = sum(a.values())
            by_domain_rows.append(row)
    write_csv(OUT / "known_error_by_domain.csv", by_domain_rows)

    # ============ 2. conditional novel error decomposition ============
    cond_rows = []
    detailed_by_method = {}
    for tag in ["ref", "d1", "bc"]:
        counts, n_novel, n_routed, detailed = conditional_novel_metrics(
            logs_val[tag], assigns_val[tag])
        row = {"method": tag.upper(), "num_novel_tracks": n_novel, "num_routed": n_routed}
        for k in ["correct_first_birth", "correct_existing_match",
                  "repeated_false_birth", "wrong_existing_assignment",
                  "first_occurrence_merged_to_existing", "first_birth_unmatched", "other"]:
            row[k] = counts[k]
            row[k + "_rate_of_routed"] = round(counts[k] / n_routed, 6) if n_routed else 0.0
        row["pollution_chain_tracks"] = sum(1 for d in detailed if d["pollution_chain_error"])
        row["mixed_cluster_tracks"] = sum(1 for d in detailed if d["polluted_cluster"])
        cond_rows.append(row)
        detailed_by_method[tag] = detailed
        assert sum(counts.values()) == n_routed, (tag, sum(counts.values()), n_routed)
    write_csv(OUT / "conditional_novel_error_decomposition.csv", cond_rows)

    occ_rows = []
    for tag in ["ref", "d1", "bc"]:
        detailed = detailed_by_method[tag]
        for l in logs_val[tag]:
            if l["true_role"] != "novel" or l["predicted_virtual_novel_id"] is None:
                continue
            vid = int(l["predicted_virtual_novel_id"])
            occ = "first" if l["first_occurrence"] in (True, "True", "1") else ("second" if l["prototype_support"] == 1 else "third_and_later")
            by_cat = Counter()
            for d in detailed:
                if d["sample_id"] == l["sample_id"]:
                    by_cat[d["error_category"]] += 1
            cat = by_cat.most_common(1)[0][0] if by_cat else "other"
            occ_rows.append({
                "method": tag.upper(),
                "occurrence": occ,
                "error_category": cat,
                "track_length": int(l["track_length"]),
                "domain": l["domain"],
                "best_novel_similarity": float(l["best_novel_similarity"]),
                "novel_margin": float(l["novel_margin"]),
                "prototype_support": int(l["prototype_support"]),
            })
    # aggregate by occurrence x category
    occ_agg_rows = []
    agg = defaultdict(Counter)
    for r in occ_rows:
        agg[(r["method"], r["occurrence"])][r["error_category"]] += 1
    for (method, occ), c in sorted(agg.items()):
        row = {"method": method, "occurrence": occ, "total": sum(c.values())}
        row.update({k: c[k] for k in sorted(c)})
        occ_agg_rows.append(row)
    write_csv(OUT / "conditional_error_by_occurrence.csv", occ_agg_rows)

    support_agg_rows = []
    agg = defaultdict(Counter)
    for r in occ_rows:
        bucket = bucketize(r["prototype_support"], [1, 3, 5, 10])
        agg[(r["method"], bucket)][r["error_category"]] += 1
    for (method, bucket), c in sorted(agg.items()):
        row = {"method": method, "prototype_support_bucket": bucket, "total": sum(c.values())}
        row.update({k: c[k] for k in sorted(c)})
        support_agg_rows.append(row)
    write_csv(OUT / "conditional_error_by_support.csv", support_agg_rows)

    # ============ 3. meta-dev vs validation shift ============
    stats_fields = [
        "best_known_similarity", "known_margin", "best_novel_similarity",
        "novel_margin", "prob_known", "prob_existing", "prob_new",
        "track_length", "active_novel_prototypes", "prototype_support",
        "known_score_before", "known_score_after_frozen", "known_score_after_adapted",
    ]
    shift_rows = []
    for field in stats_fields:
        row = {"statistic": field}
        for tag in ["d1", "bc"]:
            for split, logs in [("meta_dev", logs_proxy[tag]), ("validation", logs_val[tag])]:
                vals = [float(l[field]) for l in logs if l[field] not in ("None", "?")
                        and not (isinstance(l[field], float) and np.isnan(l[field]))]
                if vals:
                    row[f"{split}_mean"] = round(float(np.mean(vals)), 6)
                    row[f"{split}_median"] = round(float(np.median(vals)), 6)
                    row[f"{split}_p25"] = round(float(np.percentile(vals, 25)), 6)
                    row[f"{split}_p75"] = round(float(np.percentile(vals, 75)), 6)
                else:
                    row[f"{split}_mean"] = row[f"{split}_median"] = ""
                    row[f"{split}_p25"] = row[f"{split}_p75"] = ""
        shift_rows.append(row)
    # structural composition rows
    for tag in ["d1", "bc"]:
        for split, logs in [("meta_dev", logs_proxy[tag]), ("validation", logs_val[tag])]:
            roles = Counter(l["true_role"] for l in logs)
            domains = Counter(l["domain"] for l in logs)
            row = {"statistic": f"{tag}_{split}_composition",
                   "known_fraction": round(roles["supported_known"] / len(logs), 4) if logs else "",
                   "novel_fraction": round(roles["novel"] / len(logs), 4) if logs else "",
                   "n_known": roles["supported_known"], "n_novel": roles["novel"],
                   "domain_HACS": domains["HACS"], "domain_LaSOT": domains["LaSOT"],
                   "domain_BDD": domains["BDD"], "domain_AVA": domains["AVA"],
                   "domain_YFCC100M": domains["YFCC100M"],
                   "domain_Charades": domains["Charades"],
                   "domain_ArgoVerse": domains["ArgoVerse"]}
            shift_rows.append(row)
    write_csv(OUT / "meta_dev_validation_shift.csv", shift_rows)

    # ============ 4. shared action head consistency ============
    head_rows = []
    logs = logs_val["d1"]
    known_logs = [l for l in logs if l["true_role"] in ("supported_known", "zero_shot_known")]
    sem_correct = [l for l in known_logs if l["predicted_known_id"] is not None
                   and int(l["predicted_known_id"]) == int(l["true_class"])]
    sem_wrong = [l for l in known_logs if l["predicted_known_id"] is not None
                 and int(l["predicted_known_id"]) != int(l["true_class"])]
    known_action_ok = sum(1 for l in known_logs if l["predicted_action"] == "KNOWN")
    sem_correct_action_not_known = sum(1 for l in sem_correct if l["predicted_action"] != "KNOWN")
    novel_logs = [l for l in logs if l["true_role"] == "novel"]
    existing_logs = [l for l in novel_logs if l["predicted_action"] == "EXISTING_NOVEL"]
    new_logs = [l for l in novel_logs if l["predicted_action"] == "NEW_NOVEL"]
    high_best_novel = [l for l in novel_logs if float(l["best_novel_similarity"]) >= 0.55]
    high_best_novel_new = [l for l in high_best_novel if l["predicted_action"] == "NEW_NOVEL"]
    low_best_novel = [l for l in novel_logs if float(l["best_novel_similarity"]) < 0.45]
    low_best_novel_existing = [l for l in low_best_novel if l["predicted_action"] == "EXISTING_NOVEL"]
    head_rows += [
        {"stat": "num_known_tracks", "value": len(known_logs)},
        {"stat": "known_semantic_argmax_correct", "value": len(sem_correct)},
        {"stat": "known_semantic_argmax_wrong", "value": len(sem_wrong)},
        {"stat": "known_action_known", "value": known_action_ok},
        {"stat": "sem_correct_but_action_not_known", "value": sem_correct_action_not_known,
         "rate": round(sem_correct_action_not_known / max(len(sem_correct), 1), 4)},
        {"stat": "novel_tracks", "value": len(novel_logs)},
        {"stat": "novel_existing_action", "value": len(existing_logs)},
        {"stat": "novel_new_action", "value": len(new_logs)},
        {"stat": "novel_best_sim_ge_0.55", "value": len(high_best_novel)},
        {"stat": "novel_best_sim_ge_0.55_but_new_action", "value": len(high_best_novel_new),
         "rate": round(len(high_best_novel_new) / max(len(high_best_novel), 1), 4)},
        {"stat": "novel_best_sim_lt_0.45", "value": len(low_best_novel)},
        {"stat": "novel_best_sim_lt_0.45_but_existing_action", "value": len(low_best_novel_existing),
         "rate": round(len(low_best_novel_existing) / max(len(low_best_novel), 1), 4)},
    ]
    # correlation of logits with evidence
    def pearson(xs, ys):
        xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float)
        if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
            return float("nan")
        return float(np.corrcoef(xs, ys)[0, 1])
    head_rows += [
        {"stat": "corr_known_sem_correct_vs_prob_known",
         "value": round(pearson([1 if l["predicted_known_id"] is not None and int(l["predicted_known_id"]) == int(l["true_class"]) else 0 for l in known_logs],
                                [l["prob_known"] for l in known_logs]), 4)},
        {"stat": "corr_best_novel_sim_vs_prob_existing",
         "value": round(pearson([l["best_novel_similarity"] for l in novel_logs],
                                [l["prob_existing"] for l in novel_logs]), 4)},
        {"stat": "corr_best_novel_sim_vs_prob_new",
         "value": round(pearson([l["best_novel_similarity"] for l in novel_logs],
                                [l["prob_new"] for l in novel_logs]), 4)},
        {"stat": "corr_best_known_sim_vs_prob_known",
         "value": round(pearson([l["best_known_similarity"] for l in known_logs],
                                [l["prob_known"] for l in known_logs]), 4)},
    ]
    # action marginal distribution at test
    action_marg = Counter(l["predicted_action"] for l in logs)
    head_rows += [{"stat": f"marginal_{a.lower()}", "value": action_marg[a]}
                  for a in ["KNOWN", "EXISTING_NOVEL", "NEW_NOVEL"]]
    # training target balance (per episode)
    num_known, support_per_class, query_per_class = 20, 4, 4
    n_pseudo = 38 - num_known
    head_rows += [
        {"stat": "train_targets_per_episode_known", "value": num_known * query_per_class},
        {"stat": "train_targets_per_episode_existing", "value": (n_pseudo - 1) * query_per_class},
        {"stat": "train_targets_per_episode_new", "value": query_per_class},
    ]
    write_csv(OUT / "shared_action_head_consistency.csv", head_rows)

    # save raw logs for later analysis / docs
    for tag in ["ref", "d1", "bc"]:
        write_csv(OUT / f"per_track_{tag}_val_seed1027.csv", logs_val[tag])
        write_csv(OUT / f"per_track_{tag}_meta_dev.csv", logs_proxy[tag])
    summary = {
        "num_val_tracks": len(logs_val["d1"]),
        "num_proxy_tracks": len(logs_proxy["d1"]),
        "known_error_total_consistency": all(sum(known_metrics(logs_val[t])[0].values()) == known_metrics(logs_val[t])[1] for t in ["ref", "d1", "bc"]),
        "conditional_total_consistency": all(
            sum(conditional_novel_metrics(logs_val[t], assigns_val[t])[0].values())
            == conditional_novel_metrics(logs_val[t], assigns_val[t])[2] for t in ["ref", "d1", "bc"]),
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
