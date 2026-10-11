# TrackOCD Train-first final report

Completed real33fits / 99retained checkpoints / 33selected checkpoints;
420representation +1800policy controlled cases and840full limited-MASA
executions /2040explicit logical aliases. Outcome: **credible scientific
negative**, not failed implementation, strongfrontend PASS or new-method
paper proof. This is the user-authorized independent main, not R2. Historical
M1 stays BLOCKED_FRONTEND_QUALITY; no historical outcomes/assets overwritten.

## Legal data and training

TAO Train inherited Known only,500videos/18274images inventory2196tracks /
48supportedcategories. Eligible fittingpool15classes1305tracks14714available
observations, not 'all48 optimized' orall1305visited ineachfit. Full1000-step
class-balanced runs visit933/938/948distincttracks for the three seeds,
identical acrosspairedmodels. These arefullfit counts,not the separately
unrecorded footprint ofearlier250/500selectedcheckpoints. All5partitions
video-disjoint; development/policy/final
pseudo category reservations disjoint from fitting and each other. 30tracks
without a lawful selected role excluded explicitly,1026short inventorytracks
retained eligible. Commonselected2166tracks/24628observations;151prototype
tracks/48categories,258dev/229policy/223heldout(207Known+16pseudo). Features,
weights,rawGT/predNPZ/privateflags/ledgers never uploaded toGitHub.

FrozenDINOv2 ViTB14/518bilinear/context.1,commonFP16,cpuFP32coreforward.
T1nine1000-step fits weights0/1/5,globaldevselectplainA1,selected500/250/500;
T2sixmatched1000-step temporal/staticcapacity fits withsameA1selectedsteps;
T3eighteen20epoch×4order actualpredicted-memoryfits,708params D1/D2,
sameinit/data/budget,frozenrepresentation,noGTrepair. Full reallosses and
pairedinit/batch/checkpointhashes are delivered. No Val Novel fitting/tuning,
Novel text input,Test access,newassets or physicalrerun.

## Controlled Train p16 results

Percentages: four-ordermean per seed then three-seed mean,rawonlyonecontrol.

| Method | Old % | New % | mean H % | pure CT % | False Merge % | reuse coverage % |
|---|---:|---:|---:|---:|---:|---:|
| B0_frame_snapshot_vote | 9.18 | 76.56 | 16.38 | 59.13 | 11.90 | 100.00 |
| B1_track_nearest | 0.97 | 53.12 | 1.90 | 39.68 | 0.00 | 100.00 |
| B2_track_dpmeans | 6.76 | 31.25 | 11.12 | 0.00 | 0.00 | 100.00 |
| A1_SELECTED | 63.45 | 45.83 | 52.76 | 16.93 | 40.21 | 100.00 |
| A2_EVIDENCE | 53.18 | 50.52 | 51.36 | 7.94 | 61.38 | 100.00 |
| A2_CAPACITY_CONTROL | 58.05 | 52.60 | 54.13 | 8.99 | 58.47 | 100.00 |
| D1_SIMPLE_MLP | 50.89 | 7.81 | 11.82 | 0.00 | 0.00 | 100.00 |
| D2_RISK_AWARE | 51.53 | 20.83 | 0.00 | 0.00 | 30.95 | 97.62 |
| WITHOUT_RISK | 62.00 | 5.21 | 6.92 | 0.00 | 0.00 | 100.00 |
| FULL | 51.21 | 20.83 | 0.00 | 0.00 | 30.03 | 96.69 |
| WITHOUT_WAIT | 51.21 | 20.83 | 0.00 | 0.00 | 33.33 | 100.00 |
| WITHOUT_MEMORY | 51.21 | 2.08 | 0.00 | 0.00 | 0.00 | 72.75 |
| RESET_PER_VIDEO | 51.21 | 12.50 | 0.00 | 0.00 | 0.00 | 80.16 |


Representation raw geometry and role-separated retrieval are in
TRAIN_FIRST_REPRESENTATION_ROLE_ANALYSIS.json. 1020existing sealedcase metrics
exactly rechecked,480pairederror comparisons; fixes AND introducederrors
reported. All2220 controlled rows are retained alongsideT5 in canonicalCSVs.

## Frozen full limited-MASA p16 results

All988videos/36375annotatedframes/304561IDs including114380singletons,
1294110commonfirst-at-most16observations. MainGT4413Known/819Novel and
fixed527/529/547/531reuse,notmatched-only1400/189denominators. Short/unmatched
IDs mutate memory whencommitted. Target1.0 is a Train-dev operatingpoint,
not Valcoveragepromise; actualcoverage in the tables/pairedresults.
GT unresolved/WAIT-rate metrics include physical misses; distinguish explicit
predicted-ID WAIT action rate (runtime), which has a different denominator.

| Method | Old % | New % | mean H % | pure CT % | False Merge % | reuse coverage % |
|---|---:|---:|---:|---:|---:|---:|
| A1_SELECTED | 7.81 | 11.33 | 8.93 | 0.00 | 11.07 | 22.87 |
| A2_CAPACITY_CONTROL | 6.58 | 11.78 | 8.21 | 0.00 | 11.19 | 22.87 |
| A2_EVIDENCE | 5.08 | 12.34 | 7.15 | 0.00 | 10.38 | 22.87 |
| B0_frame_snapshot_vote | 0.66 | 14.62 | 1.26 | 0.00 | 6.56 | 22.87 |
| B1_track_nearest | 0.02 | 13.43 | 0.05 | 0.00 | 1.17 | 22.87 |
| B2_track_dpmeans | 1.76 | 11.29 | 3.04 | 0.05 | 0.19 | 22.87 |
| D1_SIMPLE_MLP | 11.90 | 3.14 | 3.80 | 0.00 | 0.00 | 22.87 |
| D2_RISK_AWARE | 14.38 | 0.57 | 0.00 | 0.00 | 7.62 | 22.87 |
| FULL | 14.51 | 0.57 | 0.00 | 0.00 | 7.62 | 22.87 |
| RESET_PER_VIDEO | 14.51 | 4.23 | 0.00 | 0.00 | 3.03 | 22.87 |
| WITHOUT_ADAPTER | 7.11 | 1.14 | 0.00 | 0.00 | 15.25 | 22.87 |
| WITHOUT_MEMORY | 14.51 | 0.61 | 0.00 | 0.00 | 0.00 | 15.95 |
| WITHOUT_RELIABILITY | 14.46 | 0.57 | 0.00 | 0.00 | 7.62 | 22.87 |
| WITHOUT_RISK | 16.50 | 2.44 | 2.35 | 0.00 | 0.00 | 22.87 |
| WITHOUT_TEMPORAL | 14.38 | 0.57 | 0.00 | 0.00 | 7.62 | 22.87 |
| WITHOUT_WAIT | 14.51 | 0.57 | 0.00 | 0.00 | 7.62 | 22.87 |


HOTA 0.1450535920, AssA 0.4408791462,
DetA 0.0484887546, DetRe 0.6318088080
are the **same existing canonical Native physicalreference** for everymethod,
not new tracking runs orsemanticphysicalgains. ANNOTATED-CADENCE CAUSAL REPLAY,
not dense-frameonline. PHE INCOMPARABLE; incomplete supervision provenance
precludes strict all-Novel-supervision-exclusion claims.

| Order | fixed GT reuse | current reliable | missed current | both any observed | both>=16 |
|---|---:|---:|---:|---:|---:|
| main | 527 | 118 | 409 | 87 | 69 |
| seed1027 | 529 | 123 | 406 | 93 | 68 |
| seed1028 | 547 | 121 | 426 | 92 | 70 |
| seed1029 | 531 | 126 | 405 | 94 | 74 |


Posthoc descriptive Known prototype/physical-coverage counts (not new GT input
to frozen inference and not a preregistered causal intervention):

```json
[
  {
    "GT_tracks": 1307,
    "prototype_group": "fit15",
    "reliable_physical_match": true
  },
  {
    "GT_tracks": 2793,
    "prototype_group": "fit15",
    "reliable_physical_match": false
  },
  {
    "GT_tracks": 87,
    "prototype_group": "prototype_only33",
    "reliable_physical_match": true
  },
  {
    "GT_tracks": 202,
    "prototype_group": "prototype_only33",
    "reliable_physical_match": false
  },
  {
    "GT_tracks": 6,
    "prototype_group": "no_prototype30",
    "reliable_physical_match": true
  },
  {
    "GT_tracks": 18,
    "prototype_group": "no_prototype30",
    "reliable_physical_match": false
  }
]
```

Strict>=16support13.17% isnotclippedshort-trackabsoluteceiling. See
SCIENTIFIC_CONCLUSION.md for allten requiredquestions, stablepair signs,
failureinterpretations and limitations. No post-result model/bias/cost tuning.

## Reproduction, checkpoints and provenance

Branch `codex/trackocd-core-training-limited-masa`; report source registered
and exactremoteverified at `2f0fe3bce683b0bdd77748ff42e5ae91cd17f486`. Original frozen inference registration:
`eddbc50435e87b85b7141decfb2a5dc32290ef24`;
scheduling-only recovery registration:
`c662f76606e23ee2cd054b507db7c0da700dcece`;
actual evaluator case registrations:
`c662f76606e23ee2cd054b507db7c0da700dcece, eddbc50435e87b85b7141decfb2a5dc32290ef24`;
full evaluator collector registration `c662f76606e23ee2cd054b507db7c0da700dcece`.
Recovery did not create new models, points or a new method after Val metrics.
Both actual successful parent terminal receipts are required; adopted orphan
exit codes are unavailable and never fabricated as zero. Their atomic group
completion and every immutable case seal constitute explicit terminal proof.
Model freeze `070f2182724449564ee8f6297f902b1c649c2d4a2f55fef5cac9222cbc991822`
binds52protectedsource/config/receipt/checkpointhashes. Finalreportcommit and
exactremote are verified **after** committing these reports; no impossible
self-referential commit SHA is fabricated inside its own artifact.

Use the existing `/data3/liuyeqiang/.venvs/trackocd-a100` entry environment;
configs, script CLI and actual receipts specify input/source/protocol hashes.
Train: `train_core_representation.py` with `--phase representation` or
`--phase evidence`, and `train_persistent_policy.py`; controlled:
`evaluate_core_representation.py` and `evaluate_persistent_policy.py`.
Predicted: `extract_limited_masa_features.py`,
`validate_full_limited_masa_features.py`, `run_limited_masa_inference.py`,
`evaluate_limited_masa.py`; require exact prereg remote SHA and actualcompleted
inputs, not historical PID/session reuse. Existing completed outputs refuse
overwrite. For an independent reproduction use a separately preregistered
empty destination/config identity and lawful matching localassets, never
delete/reset originaloutputs. Exhaustive CPU/CUDA actual40case proofs and
syntheticregressions accompany the executor; no historicalPython3.7 equality
or tiny-proof CUDA speedup is inferred.

| Phase / representation / model / seed | selected step or epoch | actual local checkpoint | SHA256 |
|---|---:|---|---|
| representation / - / A1_ADAPTER / 1027 | 500 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/representation/A1_ADAPTER_seed1027_step500.pt` | `9111d5aec9dc5a90daa7c38a3ee0397c6a2a58aab0abb54cdd476eca12835f9c` |
| representation / - / A1_GEOMETRY_1 / 1027 | 500 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/representation/A1_GEOMETRY_1_seed1027_step500.pt` | `5786f9d689d1ab656d5662408040a13cdf693591276a4c27feedc5463d5ae7fe` |
| representation / - / A1_GEOMETRY_5 / 1027 | 250 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/representation/A1_GEOMETRY_5_seed1027_step250.pt` | `8675bc5254a5519f5632769ba26349a9a96e9c18f31bd53ee5a477d6bb34f75d` |
| representation / - / A1_ADAPTER / 1028 | 250 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/representation/A1_ADAPTER_seed1028_step250.pt` | `9dba57810ba046511b2d88d0a8d72fe29bdaf342a36c7f7d559dd36a5dbd5302` |
| representation / - / A1_GEOMETRY_1 / 1028 | 250 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/representation/A1_GEOMETRY_1_seed1028_step250.pt` | `fb4958da0e93387fcafa774890d7f7314324dea76d3b80e58c4ebcf541d5061e` |
| representation / - / A1_GEOMETRY_5 / 1028 | 250 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/representation/A1_GEOMETRY_5_seed1028_step250.pt` | `d6da14ae0af4c2ef55a5f9034a1c82e3463c808e25adb6d3357d6d72120d310a` |
| representation / - / A1_ADAPTER / 1029 | 500 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/representation/A1_ADAPTER_seed1029_step500.pt` | `fdece5be977ae917e619a5c661de0eb5ce34ed40e013916f1a425601cfcd0892` |
| representation / - / A1_GEOMETRY_1 / 1029 | 500 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/representation/A1_GEOMETRY_1_seed1029_step500.pt` | `94f9a02181ae70cefec978fd03216b5c4ac366aa6ac6017b05dac74c4c100112` |
| representation / - / A1_GEOMETRY_5 / 1029 | 500 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/representation/A1_GEOMETRY_5_seed1029_step500.pt` | `db869e25476155d0582a6cfb75a86b0a387c6f020c88da18ee225253288d47ba` |
| evidence / - / A2_EVIDENCE / 1027 | 500 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/evidence/A2_EVIDENCE_seed1027_step500.pt` | `e334eb3b3aad40d592ba0541160146e6f2f089206fb2d43407e03e8eee7a957e` |
| evidence / - / A2_CAPACITY_CONTROL / 1027 | 500 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/evidence/A2_CAPACITY_CONTROL_seed1027_step500.pt` | `9161b5b1c508d06bf0add5d39a8c0ae71193408943eed72386f6be0e2b74af59` |
| evidence / - / A2_EVIDENCE / 1028 | 250 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/evidence/A2_EVIDENCE_seed1028_step250.pt` | `ad69c059218d178f9f52f21648de4db4e35a3905cc171acef4fea28335f5c45d` |
| evidence / - / A2_CAPACITY_CONTROL / 1028 | 250 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/evidence/A2_CAPACITY_CONTROL_seed1028_step250.pt` | `63a73e4a5b40f3b61012a63810e25d3d73e1af28c8178a2eb738802e7745b2bd` |
| evidence / - / A2_EVIDENCE / 1029 | 500 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/evidence/A2_EVIDENCE_seed1029_step500.pt` | `0454c5884409725be037c4c03087864dcad9f8d1b216bf63e0322608a13bc888` |
| evidence / - / A2_CAPACITY_CONTROL / 1029 | 500 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/evidence/A2_CAPACITY_CONTROL_seed1029_step500.pt` | `83b43b1e326fa93e4a27973187fb143978913edf8c38a425f0c70d823c1571b0` |
| policy / A0_RAW / D1_SIMPLE_MLP / 1027 | 5 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/A0_RAW_D1_SIMPLE_MLP_seed1027_epoch5.pt` | `5f6f5560cef90d52db032adbee53c1536a1c1d5f306e86f8d874d018b870c192` |
| policy / A0_RAW / D2_RISK_AWARE / 1027 | 5 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/A0_RAW_D2_RISK_AWARE_seed1027_epoch5.pt` | `1c257d6b8178b0e72f80a4d0b0b0f6d7a457de87cecb770b03cee66222540eda` |
| policy / A0_RAW / D1_SIMPLE_MLP / 1028 | 20 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/A0_RAW_D1_SIMPLE_MLP_seed1028_epoch20.pt` | `52df80b34e78f5eac9939ffa840d5eb9a2bb917ce30b2819e42a7f3aac0f8909` |
| policy / A0_RAW / D2_RISK_AWARE / 1028 | 5 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/A0_RAW_D2_RISK_AWARE_seed1028_epoch5.pt` | `bc7433d03066efa244bfc63f24a65eac62be3270bf1745c317be456d74e2c03a` |
| policy / A0_RAW / D1_SIMPLE_MLP / 1029 | 5 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/A0_RAW_D1_SIMPLE_MLP_seed1029_epoch5.pt` | `7a59f2d64c40573df4b17dc6a44de66a16792f83c8139c25e4ade174533728d9` |
| policy / A0_RAW / D2_RISK_AWARE / 1029 | 5 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/A0_RAW_D2_RISK_AWARE_seed1029_epoch5.pt` | `775946dc63aa5a29201ad19cc2b34ec8b13a41f72e409f8ea111d8b6d72a90e9` |
| policy / A1_SELECTED / D1_SIMPLE_MLP / 1027 | 20 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/A1_SELECTED_D1_SIMPLE_MLP_seed1027_epoch20.pt` | `4ef56857944aef3d7cb172812ff2e5dd80be76b5f1fae6a8410a35a4e4aba1ac` |
| policy / A1_SELECTED / D2_RISK_AWARE / 1027 | 5 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/A1_SELECTED_D2_RISK_AWARE_seed1027_epoch5.pt` | `b108a826d34de90e86e7fe77fadc7e9b502afcf7dbeb2c2a9d90922247eb1b70` |
| policy / A1_SELECTED / D1_SIMPLE_MLP / 1028 | 20 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/A1_SELECTED_D1_SIMPLE_MLP_seed1028_epoch20.pt` | `cb90e383ccb5da1f2233d96af4a4b487d58a2747c14014e5c614f225c0944d1e` |
| policy / A1_SELECTED / D2_RISK_AWARE / 1028 | 5 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/A1_SELECTED_D2_RISK_AWARE_seed1028_epoch5.pt` | `56be3530e70d38449b6342ab66e439ac26d2e8a97c104cea47a31c5d5b692fa5` |
| policy / A1_SELECTED / D1_SIMPLE_MLP / 1029 | 5 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/A1_SELECTED_D1_SIMPLE_MLP_seed1029_epoch5.pt` | `0200285abfc33a584da2396096287d898708594a794a2faf61c6338be0acf2dc` |
| policy / A1_SELECTED / D2_RISK_AWARE / 1029 | 5 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/A1_SELECTED_D2_RISK_AWARE_seed1029_epoch5.pt` | `131a081ebbac783f5beefef5667c2eba38d4558e53e850a40d134ed48d1f6807` |
| policy / A2_EVIDENCE / D1_SIMPLE_MLP / 1027 | 5 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/A2_EVIDENCE_D1_SIMPLE_MLP_seed1027_epoch5.pt` | `f185504a9cf1ab0c2dcd36ac1b4c501b793187859d2e58db351abba111408ec5` |
| policy / A2_EVIDENCE / D2_RISK_AWARE / 1027 | 5 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/A2_EVIDENCE_D2_RISK_AWARE_seed1027_epoch5.pt` | `7c3bd00eca086685fdcc991e806e97cfa28af1ef0eff9092129af19c36bbf9e2` |
| policy / A2_EVIDENCE / D1_SIMPLE_MLP / 1028 | 20 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/A2_EVIDENCE_D1_SIMPLE_MLP_seed1028_epoch20.pt` | `d659845717997eec4cf45bae39efb7f0290b323021a24e95b4c2750b1570deb7` |
| policy / A2_EVIDENCE / D2_RISK_AWARE / 1028 | 5 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/A2_EVIDENCE_D2_RISK_AWARE_seed1028_epoch5.pt` | `f677ab154a20bee06af7df75ba0ec611aaa61a8732c519b8099c6addb6ca188e` |
| policy / A2_EVIDENCE / D1_SIMPLE_MLP / 1029 | 5 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/A2_EVIDENCE_D1_SIMPLE_MLP_seed1029_epoch5.pt` | `f9bf83b829871991eab753848cec16375a85968609de3f97628a8b27c9454169` |
| policy / A2_EVIDENCE / D2_RISK_AWARE / 1029 | 5 | `/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/A2_EVIDENCE_D2_RISK_AWARE_seed1029_epoch5.pt` | `2f57535c2329f0385db31e76f5daee146dfe02ab182b8033398fb66cbaa24d8a` |


## Actual resource receipts

```json
{
  "actual_training": {
    "representation": {
      "receipt_path": "/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/representation/training_receipt.json",
      "receipt_sha256": "e1ec85c4baf0ea74612d780e581fc14124575ad7444e50599186dbb886792334",
      "preregistration_commit": "ec61fc64d6b183e916b1d2eb25197bd27765a777",
      "actual_fits": 9,
      "resources": {
        "gpu_peak_allocated_bytes": 81032192,
        "gpu_peak_reserved_bytes": 96468992,
        "gpu_uuid": "GPU-f931e51b-55b2-c2c2-6506-b6f986864d54",
        "peak_rss_bytes": 1192591360,
        "wall_seconds": 276.56891459497274
      },
      "checkpoint_bytes": 56775888,
      "full_fit_sample_coverage_not_selected_early_checkpoint_footprint": [
        {
          "model": "A1_ADAPTER",
          "seed": 1027,
          "fit_tracks_available": 1305,
          "distinct_fit_tracks_actually_seen_over_full_budget": 933
        },
        {
          "model": "A1_GEOMETRY_1",
          "seed": 1027,
          "fit_tracks_available": 1305,
          "distinct_fit_tracks_actually_seen_over_full_budget": 933
        },
        {
          "model": "A1_GEOMETRY_5",
          "seed": 1027,
          "fit_tracks_available": 1305,
          "distinct_fit_tracks_actually_seen_over_full_budget": 933
        },
        {
          "model": "A1_ADAPTER",
          "seed": 1028,
          "fit_tracks_available": 1305,
          "distinct_fit_tracks_actually_seen_over_full_budget": 938
        },
        {
          "model": "A1_GEOMETRY_1",
          "seed": 1028,
          "fit_tracks_available": 1305,
          "distinct_fit_tracks_actually_seen_over_full_budget": 938
        },
        {
          "model": "A1_GEOMETRY_5",
          "seed": 1028,
          "fit_tracks_available": 1305,
          "distinct_fit_tracks_actually_seen_over_full_budget": 938
        },
        {
          "model": "A1_ADAPTER",
          "seed": 1029,
          "fit_tracks_available": 1305,
          "distinct_fit_tracks_actually_seen_over_full_budget": 948
        },
        {
          "model": "A1_GEOMETRY_1",
          "seed": 1029,
          "fit_tracks_available": 1305,
          "distinct_fit_tracks_actually_seen_over_full_budget": 948
        },
        {
          "model": "A1_GEOMETRY_5",
          "seed": 1029,
          "fit_tracks_available": 1305,
          "distinct_fit_tracks_actually_seen_over_full_budget": 948
        }
      ]
    },
    "evidence": {
      "receipt_path": "/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/evidence/training_receipt.json",
      "receipt_sha256": "27cb162a64582278b34c7e30a3ccda3bb62403b122a53ef9ec035d6628585466",
      "preregistration_commit": "9dba258f1547c14d803597c742a13659081d21d4",
      "actual_fits": 6,
      "resources": {
        "gpu_peak_allocated_bytes": 81915392,
        "gpu_peak_reserved_bytes": 98566144,
        "gpu_uuid": "GPU-f931e51b-55b2-c2c2-6506-b6f986864d54",
        "peak_rss_bytes": 1230462976,
        "wall_seconds": 251.33419468201464
      },
      "checkpoint_bytes": 39086823,
      "full_fit_sample_coverage_not_selected_early_checkpoint_footprint": [
        {
          "model": "A2_EVIDENCE",
          "seed": 1027,
          "fit_tracks_available": 1305,
          "distinct_fit_tracks_actually_seen_over_full_budget": 933
        },
        {
          "model": "A2_CAPACITY_CONTROL",
          "seed": 1027,
          "fit_tracks_available": 1305,
          "distinct_fit_tracks_actually_seen_over_full_budget": 933
        },
        {
          "model": "A2_EVIDENCE",
          "seed": 1028,
          "fit_tracks_available": 1305,
          "distinct_fit_tracks_actually_seen_over_full_budget": 938
        },
        {
          "model": "A2_CAPACITY_CONTROL",
          "seed": 1028,
          "fit_tracks_available": 1305,
          "distinct_fit_tracks_actually_seen_over_full_budget": 938
        },
        {
          "model": "A2_EVIDENCE",
          "seed": 1029,
          "fit_tracks_available": 1305,
          "distinct_fit_tracks_actually_seen_over_full_budget": 948
        },
        {
          "model": "A2_CAPACITY_CONTROL",
          "seed": 1029,
          "fit_tracks_available": 1305,
          "distinct_fit_tracks_actually_seen_over_full_budget": 948
        }
      ]
    },
    "policy": {
      "receipt_path": "/data3/liuyeqiang/TrackOCD/outputs/trackocd_core/core_training/train_first_v1/policy/training_receipt.json",
      "receipt_sha256": "0fdd8c3b4e9e32726cfb53ef64226b16d70fdf7c50314260daf0bb01b477d66e",
      "preregistration_commit": "d270d376173aeb6438f739140a0bdb9cc948e297",
      "actual_fits": 18,
      "resources": {
        "checkpoint_bytes": 313488,
        "cpu_workers": 1,
        "gpu_used": false,
        "peak_rss_bytes": 847216640,
        "wall_seconds": 446.9463669839897
      },
      "checkpoint_bytes": 313488,
      "full_fit_sample_coverage_not_selected_early_checkpoint_footprint": [
        {
          "model": "D1_SIMPLE_MLP",
          "seed": 1027,
          "fit_tracks_available": null,
          "distinct_fit_tracks_actually_seen_over_full_budget": null
        },
        {
          "model": "D2_RISK_AWARE",
          "seed": 1027,
          "fit_tracks_available": null,
          "distinct_fit_tracks_actually_seen_over_full_budget": null
        },
        {
          "model": "D1_SIMPLE_MLP",
          "seed": 1028,
          "fit_tracks_available": null,
          "distinct_fit_tracks_actually_seen_over_full_budget": null
        },
        {
          "model": "D2_RISK_AWARE",
          "seed": 1028,
          "fit_tracks_available": null,
          "distinct_fit_tracks_actually_seen_over_full_budget": null
        },
        {
          "model": "D1_SIMPLE_MLP",
          "seed": 1029,
          "fit_tracks_available": null,
          "distinct_fit_tracks_actually_seen_over_full_budget": null
        },
        {
          "model": "D2_RISK_AWARE",
          "seed": 1029,
          "fit_tracks_available": null,
          "distinct_fit_tracks_actually_seen_over_full_budget": null
        },
        {
          "model": "D1_SIMPLE_MLP",
          "seed": 1027,
          "fit_tracks_available": null,
          "distinct_fit_tracks_actually_seen_over_full_budget": null
        },
        {
          "model": "D2_RISK_AWARE",
          "seed": 1027,
          "fit_tracks_available": null,
          "distinct_fit_tracks_actually_seen_over_full_budget": null
        },
        {
          "model": "D1_SIMPLE_MLP",
          "seed": 1028,
          "fit_tracks_available": null,
          "distinct_fit_tracks_actually_seen_over_full_budget": null
        },
        {
          "model": "D2_RISK_AWARE",
          "seed": 1028,
          "fit_tracks_available": null,
          "distinct_fit_tracks_actually_seen_over_full_budget": null
        },
        {
          "model": "D1_SIMPLE_MLP",
          "seed": 1029,
          "fit_tracks_available": null,
          "distinct_fit_tracks_actually_seen_over_full_budget": null
        },
        {
          "model": "D2_RISK_AWARE",
          "seed": 1029,
          "fit_tracks_available": null,
          "distinct_fit_tracks_actually_seen_over_full_budget": null
        },
        {
          "model": "D1_SIMPLE_MLP",
          "seed": 1027,
          "fit_tracks_available": null,
          "distinct_fit_tracks_actually_seen_over_full_budget": null
        },
        {
          "model": "D2_RISK_AWARE",
          "seed": 1027,
          "fit_tracks_available": null,
          "distinct_fit_tracks_actually_seen_over_full_budget": null
        },
        {
          "model": "D1_SIMPLE_MLP",
          "seed": 1028,
          "fit_tracks_available": null,
          "distinct_fit_tracks_actually_seen_over_full_budget": null
        },
        {
          "model": "D2_RISK_AWARE",
          "seed": 1028,
          "fit_tracks_available": null,
          "distinct_fit_tracks_actually_seen_over_full_budget": null
        },
        {
          "model": "D1_SIMPLE_MLP",
          "seed": 1029,
          "fit_tracks_available": null,
          "distinct_fit_tracks_actually_seen_over_full_budget": null
        },
        {
          "model": "D2_RISK_AWARE",
          "seed": 1029,
          "fit_tracks_available": null,
          "distinct_fit_tracks_actually_seen_over_full_budget": null
        }
      ]
    }
  },
  "full_features": {
    "GT_or_role_input": false,
    "TAO_Test_access": false,
    "annotated_cadence_not_dense_frame_online": true,
    "config_sha256": "1b8442ce64e5b9b50683a22728449873e3e6e92f1bd35a8473fad6459c9eb964",
    "detector_tracker_inference": false,
    "limited_not_strong_frontend": true,
    "mode": "full",
    "model_freeze_sha256": "070f2182724449564ee8f6297f902b1c649c2d4a2f55fef5cac9222cbc991822",
    "observations": 1294110,
    "optimizer_used": false,
    "payload_bytes": 1859256557,
    "preregistration_commit": "1778489f9ea03a666a8352f8f5f9ac2b0d2288c3",
    "private_manifest_sha256": "899625d5b4135c6db450e5c86facbe98383a244b780f49475f436bcdd82fcc74",
    "reused_smoke_observations": 8,
    "source_sha256": {
      "configs/trackocd_core/limited_masa_features.json": "1b8442ce64e5b9b50683a22728449873e3e6e92f1bd35a8473fad6459c9eb964",
      "scripts/trackocd_core/extract_limited_masa_features.py": "86cf79899364503e3897e53c2954f8d7bb8936b8abe8aec5f1d957eeac0854c2",
      "src/trackocd_core/limited_masa_features.py": "1334566a1b3fb413309aea741eaea6ccf4e9d6e8f00e0552644c3ac2410d9218"
    },
    "status": "COMPLETE_FULL_LIMITED_MASA_ALL_ID_PREFIX_FEATURES",
    "tracks": 304561,
    "unmatched_tracks_removed": false,
    "videos": 988,
    "wall_seconds": 5125.954169404024,
    "workers": [
      {
        "GPU_peak_allocated_bytes": 1148281856,
        "GPU_peak_reserved_bytes": 1474297856,
        "GPU_uuid": "GPU-f931e51b-55b2-c2c2-6506-b6f986864d54",
        "GT_or_roles_input": false,
        "observations": 161739,
        "optimizer_used": false,
        "peak_rss_bytes": 1201348608,
        "reused_smoke_observations": 0,
        "videos": 124,
        "wall_seconds": 5004.67420200599
      },
      {
        "GPU_peak_allocated_bytes": 1148281856,
        "GPU_peak_reserved_bytes": 1474297856,
        "GPU_uuid": "GPU-f07a0798-7ddc-ec7c-a27c-d47124bbfe81",
        "GT_or_roles_input": false,
        "observations": 160039,
        "optimizer_used": false,
        "peak_rss_bytes": 1197928448,
        "reused_smoke_observations": 0,
        "videos": 124,
        "wall_seconds": 5040.481307003996
      },
      {
        "GPU_peak_allocated_bytes": 1148281856,
        "GPU_peak_reserved_bytes": 1474297856,
        "GPU_uuid": "GPU-e947d8da-fc7a-4c0b-7d91-e9e67d34f4a6",
        "GT_or_roles_input": false,
        "observations": 161359,
        "optimizer_used": false,
        "peak_rss_bytes": 1227378688,
        "reused_smoke_observations": 0,
        "videos": 124,
        "wall_seconds": 4965.608107378008
      },
      {
        "GPU_peak_allocated_bytes": 1148281856,
        "GPU_peak_reserved_bytes": 1474297856,
        "GPU_uuid": "GPU-cd127bcd-08c4-fe6a-78ca-fc41123d7119",
        "GT_or_roles_input": false,
        "observations": 166168,
        "optimizer_used": false,
        "peak_rss_bytes": 1228546048,
        "reused_smoke_observations": 0,
        "videos": 124,
        "wall_seconds": 4772.167058035004
      },
      {
        "GPU_peak_allocated_bytes": 1148281856,
        "GPU_peak_reserved_bytes": 1474297856,
        "GPU_uuid": "GPU-c2902ddf-62f4-cc2f-5d9f-d33c74555847",
        "GT_or_roles_input": false,
        "observations": 165523,
        "optimizer_used": false,
        "peak_rss_bytes": 1234534400,
        "reused_smoke_observations": 0,
        "videos": 123,
        "wall_seconds": 5104.144283067988
      },
      {
        "GPU_peak_allocated_bytes": 1148281856,
        "GPU_peak_reserved_bytes": 1474297856,
        "GPU_uuid": "GPU-a4095968-9191-50eb-d56c-ed633b31e2c0",
        "GT_or_roles_input": false,
        "observations": 164289,
        "optimizer_used": false,
        "peak_rss_bytes": 1231216640,
        "reused_smoke_observations": 0,
        "videos": 123,
        "wall_seconds": 4887.522606948012
      },
      {
        "GPU_peak_allocated_bytes": 1148281856,
        "GPU_peak_reserved_bytes": 1474297856,
        "GPU_uuid": "GPU-995c9557-d84e-1a76-f02f-b1ddea34e56b",
        "GT_or_roles_input": false,
        "observations": 160004,
        "optimizer_used": false,
        "peak_rss_bytes": 1234001920,
        "reused_smoke_observations": 0,
        "videos": 123,
        "wall_seconds": 4797.822633833013
      },
      {
        "GPU_peak_allocated_bytes": 1148281856,
        "GPU_peak_reserved_bytes": 1474297856,
        "GPU_uuid": "GPU-daa9b388-4540-242c-83d7-3261bb232a7c",
        "GT_or_roles_input": false,
        "observations": 154989,
        "optimizer_used": false,
        "peak_rss_bytes": 1230962688,
        "reused_smoke_observations": 0,
        "videos": 123,
        "wall_seconds": 4936.400075401994
      }
    ]
  },
  "full_inference_wall_seconds": 54466.0684670546,
  "peak_inference_case_RSS_bytes": 2724122624,
  "peak_inference_case_GPU_reserved_bytes": 960495616,
  "posthoc_evaluation": {
    "all_private_ledgers_bytes": 627651777,
    "evaluator_supervisor_wall_seconds": 47572.056599737,
    "max_case_evaluator_RSS_bytes": 431505408
  },
  "conservative_task_private_file_bytes": 2958066883,
  "storage_soft_bytes": 16106127360,
  "storage_hard_bytes": 32212254720,
  "system_RAM_reserve_fraction": 0.25,
  "foreign_process_interference": false
}
```

Full resource/source/artifact SHA256 maps and99localcheckpoint identities are
in CORE_TRAINING_FINAL_RESULT.json. Preserve25%systemRAM,soft15/hard30GiB new
storage,4GiB inferenceworker peakguard,4GiB GPUplan+8GiBfree reserve. Owned
workers only; no externalprocess termination/deletion/newdownload. Rawprivate
assets staylocal. Trainingloss completion and349+unitregressions are engineering
evidence,not scientific PASS. No further safe-feedback/training experiment is
automatically authorized by this result.
