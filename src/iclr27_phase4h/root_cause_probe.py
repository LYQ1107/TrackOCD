"""Root-cause logistic probe across permutation replays."""
from __future__ import annotations

import csv
import json

import numpy as np

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"


def main():
    logs = json.load(open(f"{ROOT}/outputs/iclr27_phase4h/audit/permutation_logs.json"))
    hardness = {int(r["class"]): float(r["mean_adapted_best_known"])
                for r in csv.DictReader(open(
                    f"{ROOT}/outputs/iclr27_phase4h/audit/novel_class_hardness.csv"))}
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score

    rows = [l for l in logs if l["true_role"] == "novel"]
    X = np.array([[hardness[int(l["true_class"])], l["arrival_index"] / 5232.0,
                   l["memory_size"] / 700.0] for l in rows], dtype=np.float32)
    y = np.array([int(l["predicted_action"] == "KNOWN") for l in rows])
    feats = ["class_hardness", "arrival_position", "memory_size"]
    results = []
    for name, cols in [("hardness_only", [0]),
                       ("hardness+position", [0, 1]),
                       ("hardness+position+memory", [0, 1, 2]),
                       ("memory_only", [2])]:
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[:, cols], y)
        p = clf.predict_proba(X[:, cols])[:, 1]
        results.append({
            "model": name,
            "auc": round(roc_auc_score(y, p), 4),
            "log_loss": round(log_loss(y, p), 4),
            "coef": "; ".join(f"{feats[c]}={clf.coef_[0][i]:.3f}"
                              for i, c in enumerate(cols)),
        })
    with open(f"{ROOT}/outputs/iclr27_phase4h/audit/root_cause_probe.csv",
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    for r in results:
        print(r)

    # per-class N2K stability across P0/P3/P4
    by_class = {}
    for l in rows:
        if l["permutation"] not in ("P0", "P3", "P4"):
            continue
        by_class.setdefault(int(l["true_class"]), {})[l["permutation"]] = (
            by_class.get(int(l["true_class"]), {}).get(l["permutation"], [])
            + [int(l["predicted_action"] == "KNOWN")])
    stable = sum(1 for c, d in by_class.items()
                 if len(d.get("P0", [])) and len(d.get("P3", []))
                 and abs(np.mean(d["P0"]) - np.mean(d["P3"])) < 0.2)
    print("classes with |N2K(P0)-N2K(P3)|<0.2:", stable, "/", len(by_class))


if __name__ == "__main__":
    main()
