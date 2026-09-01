"""Report-convention TrackEval flat metrics for Phase 4J.

The Phase 4I report convention reads the TrackEval COMBINED table values:
HOTA/AssA/DetA are means over the per-alpha arrays (13.893 -> 0.1389),
LocA is LocA(0), and CLEAR/Identity are scalar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def flat(path):
    d = json.loads(Path(path).read_text())
    out = {}
    for tracker, c in d.items():
        out[tracker] = {
            "HOTA": float(np.mean(c["HOTA"]["HOTA"])),
            "DetA": float(np.mean(c["HOTA"]["DetA"])),
            "AssA": float(np.mean(c["HOTA"]["AssA"])),
            "LocA": float(np.mean(c["HOTA"]["LocA"])),
            "IDF1": float(c["Identity"]["IDF1"]),
            "MOTA": float(c["CLEAR"]["MOTA"]),
            "MOTP": float(c["CLEAR"]["MOTP"]),
            "IDSW": float(c["CLEAR"]["IDSW"]),
            "Frag": float(c["CLEAR"]["Frag"]),
            "MT": float(c["CLEAR"]["MT"]),
            "PT": float(c["CLEAR"]["PT"]),
            "ML": float(c["CLEAR"]["ML"]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackeval-json", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    res = flat(args.trackeval_json)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
