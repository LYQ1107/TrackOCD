"""Detailed routing audit: decision ages, prefix-vs-single-frame at matched
subsets, complete-prefix aggregates, and taxonomy with ages."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.iclr27_phase4u.data import ROOT


def load(path: Path):
    records = json.loads((path / "records.json").read_text())
    tracks = json.loads((path / "tracks.json").read_text())
    states = np.load(path / "states.npz")["states"]
    by_key = defaultdict(list)
    for r in records:
        by_key[tuple(r["key"])].append(r)
    idx = 0
    data = {}
    for k, recs in by_key.items():
        n = len(recs)
        data[k] = {"recs": recs, "states": states[idx:idx + n],
                   "n": n, "role": recs[0].get("protocol_role"),
                   "sid": recs[0].get("sid"), "ts": tracks["_".join(map(str, k))]}
        idx += n
    return data


def scalar(rec, st):
    kl = np.asarray(rec["kl"], dtype=np.float32)
    return {
        "top1_p": rec["top1_p"], "margin": rec["margin"],
        "entropy": rec["entropy"], "energy": rec["energy"],
        "max_kl": float(kl.max()), "l1_known_p": rec["l1_probs"][0],
        "l1_novel_p": rec["l1_probs"][1], "q0": rec["q"][0], "r": rec["r"],
        "top1_idx": int(kl.argmax()),
    }


def prefix(td, a):
    recs, sts = td["recs"][:a], td["states"][:a]
    scs = [scalar(r, s) for r, s in zip(recs, sts)]
    top1s = [s["top1_idx"] for s in scs]
    counts = Counter(top1s)
    mode = counts.most_common(1)[0][0]
    switch = sum(1 for i in range(1, len(top1s)) if top1s[i] != top1s[i - 1])
    last = scs[-1]
    return {
        "last_top1_p": last["top1_p"],
        "last_entropy": last["entropy"],
        "last_energy": last["energy"],
        "last_margin": last["margin"],
        "last_l1_known_p": last["l1_known_p"],
        "last_q0": last["q0"],
        "last_r": last["r"],
        "mean_top1_p": float(np.mean([s["top1_p"] for s in scs])),
        "mean_entropy": float(np.mean([s["entropy"] for s in scs])),
        "mean_margin": float(np.mean([s["margin"] for s in scs])),
        "mean_energy": float(np.mean([s["energy"] for s in scs])),
        "max_top1_p": float(max(s["top1_p"] for s in scs)),
        "min_entropy": float(min(s["entropy"] for s in scs)),
        "switch_count": switch,
        "n_unique": len(counts),
        "mode_consistency": float(np.mean([t == mode for t in top1s])),
        "last2_agree": float(len(top1s) >= 2 and top1s[-1] == top1s[-2]),
        "state_cos_first": float(sts[-1] @ sts[0] /
                                 (np.linalg.norm(sts[-1]) * np.linalg.norm(sts[0]) + 1e-12)),
        "state_cos_prev": float(len(sts) >= 2 and
                                (sts[-1] @ sts[-2] /
                                 (np.linalg.norm(sts[-1]) * np.linalg.norm(sts[-2]) + 1e-12))),
    }


def au(y, s):
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def main():
    dump = ROOT / "outputs/iclr27_phase4z/routing_audit/dev_dump_full"
    out = ROOT / "outputs/iclr27_phase4z/routing_audit/audit"
    out.mkdir(parents=True, exist_ok=True)
    data = load(dump)
    aligned = {k: td for k, td in data.items() if td["role"] is not None}

    # decision-age distribution
    ages = defaultdict(list)
    for k, td in aligned.items():
        first = td["ts"].get("first")
        if first is None:
            ages["no_decision"].append((k, td["role"]))
            continue
        ages[td["role"]].append((first["t"] + 1, first["l1"], k, td))
    print("decision ages:")
    for role, items in ages.items():
        if role == "no_decision":
            print(" ", role, len(items))
            continue
        d = Counter(t for t, l1, k, td in items)
        print(" ", role, len(items), dict(sorted(d.items())))

    # matched single-frame vs prefix at each age (same track subset)
    rows = []
    for a in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]:
        sub = {k: td for k, td in aligned.items() if td["n"] >= a}
        if len(sub) < 8:
            continue
        y = [1 if td["role"].startswith(("supported", "zero")) else 0 for td in sub.values()]
        sf = defaultdict(list)
        pf = defaultdict(list)
        for td in sub.values():
            sc = scalar(td["recs"][a - 1], td["states"][a - 1])
            p = prefix(td, a)
            for name, v in sc.items():
                if name != "top1_idx":
                    sf[name].append(v)
            for name, v in p.items():
                pf[name].append(v)
        row = {"age": a, "n": len(sub),
               "single": {k: au(y, v) for k, v in sf.items()},
               "prefix": {k: au(y, v) for k, v in pf.items()}}
        rows.append(row)
        print(json.dumps({a: row["single"]["top1_p"], "pf_last": row["prefix"]["last_top1_p"],
                          "pf_cons": row["prefix"]["mode_consistency"],
                          "pf_switch": row["prefix"]["switch_count"],
                          "pf_mean_ent": row["prefix"]["mean_entropy"],
                          "pf_cos": row["prefix"]["state_cos_prev"]}, sort_keys=False))

    # complete-prefix diagnostics
    y = [1 if td["role"].startswith(("supported", "zero")) else 0 for td in aligned.values()]
    sf_all, pf_all = defaultdict(list), defaultdict(list)
    for td in aligned.values():
        sc = scalar(td["recs"][-1], td["states"][-1])
        p = prefix(td, td["n"])
        for name, v in sc.items():
            if name != "top1_idx":
                sf_all[name].append(v)
        for name, v in p.items():
            pf_all[name].append(v)
    complete = {"n": len(aligned),
                "single": {k: au(y, v) for k, v in sf_all.items()},
                "prefix": {k: au(y, v) for k, v in pf_all.items()}}
    print("complete:", json.dumps(complete, indent=1))

    # taxonomy with ages
    tax = []
    for k, td in aligned.items():
        first = td["ts"].get("first")
        role = td["role"]
        is_known = role.startswith(("supported", "zero"))
        if first is None:
            tax.append({"key": k, "role": role, "label": "UNRESOLVED"})
            continue
        recs, sts = td["recs"], td["states"]
        fi = next((i for i, r in enumerate(recs) if r["t"] == first["t"]), len(recs) - 1)
        top1s = [scalar(r, s)["top1_idx"] for r, s in zip(recs, sts)]
        p_max = max(scalar(r, s)["top1_p"] for r, s in zip(recs, sts))
        switch = sum(1 for i in range(1, len(top1s)) if top1s[i] != top1s[i - 1])
        if first["l1"] == 0:
            label = "KNOWN_OK" if is_known else ("STABLE_FALSE_KNOWN" if switch == 0
                                                  else "UNSTABLE_FALSE_KNOWN")
            if not is_known:
                if p_max >= 0.8:
                    label += "_HIGH_CONF"
                elif p_max < 0.5:
                    label += "_LOW_CONF"
        elif first["l1"] == 1:
            label = "NOVEL_OK" if not is_known else "KNOWN_TO_NOVEL"
        else:
            label = "UNRESOLVED"
        tax.append({"key": list(k), "role": role, "label": label,
                    "decision_age": first["t"] + 1, "n_steps": td["n"],
                    "max_top1_p": p_max, "switches": switch,
                    "final_consistency": prefix(td, td["n"])["mode_consistency"]})
    counts = Counter(t["label"] for t in tax)
    print("taxonomy:", dict(counts))
    (out / "taxonomy.json").write_text(json.dumps(tax, indent=2))
    (out / "audit_detail.json").write_text(json.dumps({
        "matched_prefix_vs_single": rows,
        "complete": complete,
        "taxonomy_counts": dict(counts),
        "decision_ages": {str(k): [{"age": t, "l1": l1, "key": list(kk)}
                                   for t, l1, kk, _ in v]
                          for k, v in ages.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
