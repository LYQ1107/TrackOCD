#!/usr/bin/env python3
"""Parse OVTR/COVTrack TETA summary lines from an eval log into JSON."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lines = Path(args.log).read_text().splitlines()
    combined = None
    base = None
    novel = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*COMBINED\s+[\d.]+", line):
            parts = line.split()
            combined = {
                "TETA": float(parts[1]),
                "LocA": float(parts[2]),
                "AssocA": float(parts[3]),
                "ClsA": float(parts[4]),
            }
        if re.match(r"^\s*Base\s+[\d.]+", line):
            parts = line.split()
            if len(parts) >= 10:
                base = {
                    "TETA": float(parts[1]),
                    "LocA": float(parts[2]),
                    "AssocA": float(parts[3]),
                    "ClsA": float(parts[4]),
                }
        if re.match(r"^\s*Novel\s+[\d.]+", line):
            parts = line.split()
            if len(parts) >= 10:
                novel = {
                    "TETA": float(parts[1]),
                    "LocA": float(parts[2]),
                    "AssocA": float(parts[3]),
                    "ClsA": float(parts[4]),
                }
    out = {"combined": combined, "base": base, "novel": novel}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
