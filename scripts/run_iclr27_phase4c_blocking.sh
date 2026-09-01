#!/usr/bin/env bash
# TrackOCD ICLR 2027 Phase 4C blocking runner / artifact validator.
# The long jobs (audits, F1/F2 training, official seed1027) were executed as
# single blocking commands during development; this script validates that all
# required artifacts exist and are internally consistent.
set -u
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
cd "$ROOT" || exit 1
fail=0
check() {
  if [ -s "$1" ]; then echo "OK   $1"; else echo "MISS $1"; fail=1; fi
}

echo "== open source =="
check outputs/iclr27_phase4c/open_source/repository_inventory.csv
check outputs/iclr27_phase4c/open_source/mechanism_matrix.csv
check docs/iclr27_phase4c/OPEN_SOURCE_REPOSITORY_AUDIT.md
check docs/iclr27_phase4c/OPEN_SOURCE_IMPLEMENTATION_NOTES.md
for d in SimGCD official_gcd OCGCD VB-CGCD pdc-dp-means; do
  check "third_party/research_refs/$d/.git/HEAD"
done

echo "== audits =="
for f in known_error_decomposition.csv known_error_by_class.csv known_error_by_domain.csv \
         conditional_novel_error_decomposition.csv conditional_error_by_occurrence.csv \
         conditional_error_by_support.csv meta_dev_validation_shift.csv \
         shared_action_head_consistency.csv; do
  check "outputs/iclr27_phase4c/audit/$f"
done
for f in KNOWN_ERROR_DECOMPOSITION.md CONDITIONAL_NOVEL_ERROR_DECOMPOSITION.md \
         META_DEV_VALIDATION_SHIFT.md SHARED_ACTION_HEAD_AUDIT.md; do
  check "docs/iclr27_phase4c/$f"
done

echo "== design / implementation =="
check docs/orbit_fc/OPEN_SOURCE_DESIGN_TRANSFER.md
check docs/orbit_fc/ORBIT_FC_METHOD_SPECIFICATION.md
check docs/orbit_fc/CAUSAL_CONTRACT.md
check outputs/orbit_fc/design/design_decision.json
for f in model known_gate novel_reuse_birth causal_memory losses train evaluate protocol; do
  check "src/orbit_fc/$f.py"
done

echo "== meta-dev =="
check outputs/orbit_fc/meta_dev/config_comparison.csv
check outputs/orbit_fc/meta_dev/error_tradeoff.csv
check docs/orbit_fc/META_DEV_MODEL_SELECTION.md
check runs/orbit_fc/fc_F1/model.pth
check runs/orbit_fc/fc_F2/model.pth

echo "== official single seed =="
check outputs/orbit_fc/results/orbit_fc_seed1027.csv
check runs/orbit_fc/orbit_fc_seed1027.json
check docs/orbit_fc/ORBIT_FC_FINAL_REPORT.md
check docs/orbit_fc/ORBIT_FC_DECISION.md
check runs/orbit_fc/status.txt

echo "== tests / final =="
check outputs/iclr27_phase4c/tests/test_report.json
check outputs/orbit_fc/tests/test_report.json
check docs/iclr27_phase4c/INTEGRATED_FINAL_REPORT.md
check docs/iclr27_phase4c/ICLR_READINESS_DECISION.md

if grep -q ORBIT_FC_SINGLE_SEED_FAILED runs/orbit_fc/status.txt; then
  echo "OK   single-seed failure status recorded"
else
  echo "MISS ORBIT_FC_SINGLE_SEED_FAILED status"; fail=1
fi
if [ -e outputs/orbit_fc/results/orbit_fc_three_seed_summary.csv ]; then
  echo "ERR  three-seed artifacts must not exist after failure"; fail=1
fi
if [ -e outputs/orbit_fc/results/orbit_fc_ablation_seed1027.csv ]; then
  echo "ERR  ablation artifacts must not exist after failure"; fail=1
fi
exit $fail
