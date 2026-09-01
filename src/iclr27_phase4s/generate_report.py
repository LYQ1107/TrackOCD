"""Assemble final JSON numbers into PHASE4S_COMPLETE_COPYABLE_REPORT.md."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
REPORT = ROOT / "docs" / "iclr27_phase4s" / "PHASE4S_COMPLETE_COPYABLE_REPORT.md"


def load(name):
    p = ROOT / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def f(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def main():
    pilot = load("outputs/iclr27_phase4s/episodic_pilot/pilot_report.json") or {}
    dev = {m: load(f"outputs/iclr27_phase4s/dev_eval/dev_{m}.json") for m in ("b0", "b1", "b2", "b3")}
    q2 = load("outputs/iclr27_phase4s/dev_eval_q2/dev_b3.json")
    b0_grid = {}
    for tk in ("0.35", "0.45", "0.55"):
        d = load(f"outputs/iclr27_phase4s/b0_grid/dev_b0_{tk}_0.45.json")
        if d:
            b0_grid[tk] = d

    rows = []
    for m in ("b0", "b1", "b2", "b3"):
        d = dev.get(m)
        if not d or "metrics" not in d:
            continue
        mm = d["metrics"]
        rows.append(
            f"| {m.upper()} | {f(mm['all_track_acc'])} | {f(mm['overall_known_acc'])} "
            f"| {f(mm['route_aware_novel_acc'])} | {f(mm['conditional_novel_acc'])} "
            f"| {f(mm['novel_routing_recall'])} | {f(mm['novel_only_nmi'])} "
            f"| {f(mm['novel_only_ari'])} | {mm['novel_count_abs_error']} "
            f"| {d['memory_slots']} | {d['fp_born_slots']} |"
        )
    dev_table = "\n".join(rows)

    pilot_rows = []
    for m in ("b0", "b1", "b2", "b3"):
        d = pilot.get(m)
        if not d:
            continue
        pilot_rows.append(
            f"| {m.upper()} | {d['known_acc']} | {d['novel_first_new_acc']} "
            f"| {d['novel_later_reuse_acc']} | {d['wrong_reuse']} | {d['overbirth']} "
            f"| {d['fp_commit_rate']} | {d['fp_born_slots']} "
            f"| {d['mean_slots_per_episode']} |"
        )
    pilot_table = "\n".join(pilot_rows)

    def tax(m):
        d = dev.get(m)
        if not d:
            return "-"
        t = d.get("error_taxonomy", {})
        order = ["WRONG_REUSE", "OVERBIRTH", "UNDERBIRTH", "PREMATURE_BIRTH",
                 "DELAYED_COMMIT", "FP_BORN_MEMORY", "WRONG_MEMORY_UPDATE",
                 "KNOWN_TO_NOVEL", "NOVEL_TO_KNOWN", "REUSE_COMMIT", "BIRTH_COMMIT"]
        return ", ".join(f"{k}={t[k]}" for k in order if k in t)

    tax_rows = [f"| {m.upper()} | {tax(m)} |" for m in ("b0", "b1", "b2", "b3")]
    tax_table = "\n".join(tax_rows)

    reuse = "\n".join(
        f"| {m.upper()} | {dev[m]['metrics']['novel_routing_precision']:.3f} | {tax(m)} |"
        for m in ("b0", "b1", "b2", "b3") if dev.get(m)
    )
    birth = reuse
    defer_rows = [f"| {m.upper()} | unresolved-novel {dev[m]['metrics']['unresolved_novel_rate']:.3f} "
                  f"| {tax(m)} |" for m in ("b0", "b1", "b2", "b3") if dev.get(m)]
    defer = "\n".join(defer_rows)
    nmi = "\n".join(
        f"| {m.upper()} | {dev[m]['metrics']['novel_only_nmi']:.3f} | {dev[m]['metrics']['novel_only_ari']:.3f} |"
        for m in ("b0", "b1", "b2", "b3") if dev.get(m)
    )
    count = "\n".join(
        f"| {m.upper()} | {dev[m]['metrics']['predicted_novel_count']} "
        f"| {dev[m]['metrics']['num_novel_categories']} | {dev[m]['metrics']['novel_count_abs_error']} |"
        for m in ("b0", "b1", "b2", "b3") if dev.get(m)
    )
    memory_rows = []
    for m in ("b0", "b1", "b2", "b3"):
        d = dev.get(m)
        if not d:
            continue
        t = d.get("error_taxonomy", {})
        memory_rows.append(
            f"| {m.upper()} | {d['memory_slots']} | {d['fp_born_slots']} "
            f"| {t.get('WRONG_MEMORY_UPDATE', 0)} | {t.get('FP_TO_KNOWN', 0)} |"
        )
    memory = "\n".join(memory_rows)

    cross = f"| Q1 | {dev.get('b3', {}).get('metrics', {}).get('all_track_acc')} | " \
            f"{dev.get('b3', {}).get('metrics', {}).get('route_aware_novel_acc')} |\n" \
            f"| Q2-alpha0.1 | {(q2 or {}).get('metrics', {}).get('all_track_acc')} | " \
            f"{(q2 or {}).get('metrics', {}).get('route_aware_novel_acc')} |"

    text = REPORT.read_text()
    text = text.replace("TBD_PILOT_TABLE", pilot_table)
    text = text.replace("TBD_DEV_MAIN_TABLE", dev_table)
    text = text.replace("TBD_REUSE", reuse)
    text = text.replace("TBD_BIRTH", birth)
    text = text.replace("TBD_DEFER", defer)
    text = text.replace("TBD_NMI_ARI", nmi)
    text = text.replace("TBD_COUNT", count)
    text = text.replace("TBD_MEMORY", memory)
    text = text.replace("TBD_TAXONOMY", tax_table)
    text = text.replace("TBD_CROSS_FRONTEND", cross)
    REPORT.write_text(text)
    print("report updated")


if __name__ == "__main__":
    main()
