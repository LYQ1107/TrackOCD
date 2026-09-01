"""Byte-exact J0 equivalence vs Phase 4I B2 lambda=0.1 predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
REF = ROOT / "outputs" / "frame_online_trackocd" / "subset" / "B2" / "l0.1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--got-dir", type=Path,
                    default=ROOT / "outputs" / "iclr27_phase4j" /
                    "subset" / "J0")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "outputs" / "iclr27_phase4j" /
                    "audit" / "j0_equivalence.json")
    args = ap.parse_args()
    ref_names = {p.name for p in REF.glob("*.json") if p.name != "trackeval.json"}
    got_names = {p.name for p in args.got_dir.glob("*.json")
                 if p.name != "trackeval.json"}
    missing = sorted(ref_names - got_names)
    extra = sorted(got_names - ref_names)
    bad = []
    for name in sorted(ref_names & got_names):
        a = json.loads((REF / name).read_text())
        b = json.loads((args.got_dir / name).read_text())
        if a != b:
            bad.append(name)
    result = {
        "ref_images": len(ref_names),
        "got_images": len(got_names),
        "missing": missing,
        "extra": extra,
        "diff_images": bad[:20],
        "n_diff": len(bad),
        "byte_exact": len(ref_names) == len(got_names) and not missing \
            and not extra and not bad,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
