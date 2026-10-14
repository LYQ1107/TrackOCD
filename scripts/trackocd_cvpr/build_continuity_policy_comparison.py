"""Materialize the public all-five continuity-policy aggregate."""
import json
from pathlib import Path

from src.trackocd_cvpr.continuity_policy_receipts import complete_policy_comparison


ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/trackocd_cvpr/cvpr3/continuity_reuse_policy_all5_public_comparison_v1.json'


def main():
    OUT.write_text(json.dumps(complete_policy_comparison(ROOT),indent=2,sort_keys=False)+'\n')
    print(OUT)


if __name__=='__main__':
    main()
