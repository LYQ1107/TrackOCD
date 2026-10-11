# Actual training and reproduction handoff

Environment: existing trackocd-a100 Python3.10.21 / torch2.6.0+cu118 /
torchvision0.21.0. Source `/data3/liuyeqiang/trackocd_a100_env.sh`; do not
reuse historical GPU ordinals, PIDs or terminal IDs. No reinstall/download.

Architecture and exact registered optimizer/data budgets are in
METHOD_ARCHITECTURE.md, TRAINING_PROTOCOL.md and configs/trackocd_core.
Actual fits: nine768→512→256 adapters (525056parameters), six temporal/static
matched evidence fits (16833extra), eighteen matched17→32→4 D1/D2 policies
(708parameters). Three training seeds1027/1028/1029. Adapter/evidence1000steps,
policy20epochs×4orders. Development selection precedes all Val evaluation;
33selected and99retained checkpoints all actual local paths/bytes/SHA256
in CORE_TRAINING_FINAL_RESULT.json. Earlier selected checkpoints' exact
distinct-track visit footprint was not separately recorded; no invented count.

Reproduction entries (documentation only; do not blindly rerun completed jobs):
prepare_train_first.py; extract_train_first.py; train_core_representation.py
--phase representation / --phase evidence; train_persistent_policy.py;
evaluate_core_representation.py; evaluate_persistent_policy.py;
freeze_train_models.py; extract_limited_masa_features.py;
validate_full_limited_masa_features.py; run_limited_masa_inference.py;
evaluate_limited_masa.py; report_core_training.py; report_final_handoff.py.
enrich_final_handoff.py adds the explicit per-seed four-order reporting CSV,
not new evaluations; use the recorded separate source registration.
Use each --help and the registered config/receipt before any fresh run.
Raw assets are deliberately not included in Git; model code alone is not an
asset-free reproduction claim. Do not reconstruct old COVTrack features as MASA.

All actual loss traces15360rows and native training PNGs delivered; different
D1/D2 objectives are not directly comparable losses. CPU/CUDA exact-replay
proofs check state/event equality; no historical Python3.7-equivalence claim.
Case ledgers304561rows/seal, full feature988payloads and52protected scientific
files were independently checked. Registration and every source/config/
feature/role hash are chained in the actual Core result and FINAL_RESULT.json.

Original frozen inference registrationeddbc50435e87b85b7141decfb2a5dc32290ef24;
scheduling-only recovery/collectorc662f76606e23ee2cd054b507db7c0da700dcece.
Eight adopted workers' orphan exit status is unavailable, not invented as0;
all group/seal completion proven. Ten newly owned inference child exits0;
763new evaluator exits0 plus77same-source sealed metrics reused. No live
worker/parent remains. Existing physical result reused without tracker rerun.

Report-only source preregistration `8a593e56601231b38ec365e5f649d8006f057f7a`, exact remote verified before
execution. Scoped tests: tests/trackocd_core, tests/trackocd_v2,
tests/baselines/pandas_bytetrack; not a blanket whole-repository test claim.
Final stage follows test → commit → push → exact remote verification.
Weights/raw images/labels/full features/prediction NPZ/private ledgers and
process witnesses remain local. No forced push, deletion, external kill or
TAO Test. Negative results do not authorize further fitting or M11 feedback.
