#!/usr/bin/env bash
# TrackOCD ICLR 2027 Phase 4D blocking runner / artifact validator.
set -u
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
cd "$ROOT" || exit 1
fail=0
check() {
  if [ -s "$1" ]; then echo "OK   $1"; else echo "MISS $1"; fail=1; fi
}

echo "== open source =="
check outputs/iclr27_phase4d/open_source/additional_repository_inventory.csv
check docs/iclr27_phase4d/ADDITIONAL_OPEN_SOURCE_REVIEW.md
for d in OpenMax energy_ood Happy_CGCD; do check "third_party/research_refs/$d/.git/HEAD"; done

echo "== long-stream protocol =="
check outputs/iclr27_phase4d/long_stream/stream_manifest.json
check outputs/iclr27_phase4d/long_stream/scale_bucket_statistics.csv
check outputs/iclr27_phase4d/long_stream/proxy_method_comparison.csv
check docs/iclr27_phase4d/LONG_STREAM_META_PROTOCOL.md
check docs/iclr27_phase4d/LONG_STREAM_PROXY_VALIDATION.md

echo "== mechanism studies =="
check outputs/orbit_msr/meta_dev/known_gate_comparison.csv
check outputs/orbit_msr/meta_dev/novel_reuse_comparison.csv
check outputs/orbit_msr/meta_dev/calibration_comparison.csv
check outputs/orbit_msr/meta_dev/integrated_candidates.csv
check docs/iclr27_phase4d/KNOWN_GATE_STUDY.md
check docs/iclr27_phase4d/NOVEL_REUSE_STUDY.md
check docs/iclr27_phase4d/TRAINING_CALIBRATION_STUDY.md
for d in msr_kg1 msr_kg2 msr_nr1 msr_nr2 msr_t2 msr_c2; do check "runs/orbit_msr/$d/model.pth"; done

echo "== ORBIT-MSR =="
check docs/orbit_msr/ORBIT_MSR_METHOD_SPECIFICATION.md
check docs/orbit_msr/DESIGN_DECISION.md
check docs/orbit_msr/CAUSAL_CONTRACT.md
check outputs/orbit_msr/design/decision.json
check outputs/orbit_msr/results/candidate_1_seed1027.csv
check outputs/orbit_msr/results/candidate_2_seed1027.csv
check docs/orbit_msr/ORBIT_MSR_FINAL_REPORT.md
check docs/orbit_msr/ORBIT_MSR_DECISION.md

echo "== external baseline =="
check outputs/iclr27_phase4d/external_baseline/pdc_dp_means_meta_dev.csv
check outputs/iclr27_phase4d/external_baseline/pdc_dp_means_seed1027.csv
check docs/iclr27_phase4d/PDC_DP_MEANS_ADAPTATION.md

echo "== tests / final =="
check outputs/iclr27_phase4d/tests/test_report.json
check outputs/orbit_msr/tests/test_report.json
check docs/iclr27_phase4d/INTEGRATED_FINAL_REPORT.md
check docs/iclr27_phase4d/NEXT_METHOD_RESEARCH_DECISION.md

if [ -e outputs/orbit_msr/results/exploratory_three_seed.csv ]; then
  echo "ERR  three-seed must not exist without gate support"; fail=1
fi
exit $fail
