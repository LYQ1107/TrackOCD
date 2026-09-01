"""Phase15S namespace wrapper for the immutable DSCT-to-CSV alignment.

The frame-local IoU matching implementation is reused read-only from the
Phase15R helper; this wrapper changes only paths and provenance outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.iclr27_phase15r.data.build_dsct_proposals import convert

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--annotation", default="data/iclr27_phase15s/sources/validation_public_roles.json"); ap.add_argument("--track-output", default="outputs/iclr27_phase15s/dsct_bank/public_roles/teta_results/tao_track.json"); ap.add_argument("--out", default="outputs/iclr27_phase15s/dsct_bank/public_roles/proposals.csv"); args = ap.parse_args()
    print(json.dumps(convert(ROOT / args.annotation, ROOT / args.track_output, ROOT / args.out), indent=2, sort_keys=True))


if __name__ == "__main__": main()
