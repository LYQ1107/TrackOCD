# TrackOCD Phase85 — Autonomous Research Report

Status: **AUTONOMOUS_PHASE85_COMPLETE_WITH_VALIDATED_INTERFACE_NEGATIVE_EVIDENCE**  
Window: `2026-09-05T08:50:01Z` → `2026-09-05T18:50:01Z`  
Finalization lock opened: `2026-09-05T18:05:01Z`  
Report generated: `2026-09-05T18:05:05.050829+00:00`  
SCIENCE_HEAD: `e4e2f070d17adbdce290a2b418990143452e9522`  
REPORT_GENERATION_HEAD: `b7da9a72e5f5300d6d8a4a7ac6a4416687cd5bce`  

## Executive decision

Phase85 repaired the Phase84 implementation and evaluation contracts, then completed every registered physical/support route. The real Q0 adapter parity passed, but the corrected temporal-mean and selective physical streams both degraded the frozen R metrics. The raw-anchored support reranker produced a small positive selection increase while increasing negative activation; its separate DEFER head removed most positive selections. The TRAIN-only selective-source combination was worse. Therefore no alignment, controller/StateMemory, threshold sweep, modern backbone, or sealed/public evaluation was authorized. This is validated interface/selection negative evidence, not a claim that TrackOCD is universally infeasible.

## Protocol, data boundary, and storage

- Fixed denominators are 76 positive and 76 negative causal events at prefixes `(1, 2, 4, 8, 16)`; the frozen R universe has 984 validation queries with identical candidate order and same-video exclusion.
- Inference inputs remain visual features, geometry, motion, causal history and internal bookkeeping only. Category names/text, semantic or physical IDs as features, future rows/tracks, held GT, DEV+, Q1, public-new and sealed labels were not accessed. TRAIN labels appear only as post-hoc supervision/audit metadata.
- The explicit leakage contract audit is `/data2/usr_for_deadline/trackocd_phase85/project_outputs/audit/leakage_contract.json` (status **PASS**): three TRAIN-derived manifests declare no public/DEV+/Q1/sealed access, no future rows/tracks, and no ID-as-model-input flags; source-token mentions are declarations/audit fields rather than inference paths.
- Large artifacts are stored on `/data2/usr_for_deadline/trackocd_phase85/project_outputs` and exposed via the project symlink `outputs/iclr27_phase85`; source/target hashes and provenance are in `/data2/usr_for_deadline/trackocd_phase85/project_outputs/audit/report_provenance.json`. Phase84 artifacts were read-only.

## Phase84 issue repairs and audit

The issue audit is `/data2/usr_for_deadline/trackocd_phase85/project_outputs/audit/phase84_issue_audit.json` and the repair ledger is `/data2/usr_for_deadline/trackocd_phase85/project_outputs/audit/repair_events.json`. The eight recorded issues were: last-observation appearance mislabeled as temporal mean; fake `raw_vectors-raw_vectors` parity; multi-root anchor membership; opaque IoU joins; severe multi-category union contamination; missing TrackEval; B84S-Q artifact mix-up; and early idle finalization. Phase85 uses running `app_sum/app_count/app_mean`, causal union inheritance, a true Q0 reconstruction, one last-mapped anchor root, explicit exact-key/IoU fallback counts, provenance assertions, and a lock-aware finalizer.

Post-hoc TRAIN contamination remains a safety signal: the repaired audit records `11,816` same-category and `1,105` cross-category labeled unions (roots with multiple categories are retained as evidence; labels never enter inference).

On the fixed 91 event-video subset, the event-root audit is `/data2/usr_for_deadline/trackocd_phase85/project_outputs/audit/event_physical_contamination.json`. Q0 has `0.050072046109510084` multi-category roots, temporal mean `0.07758620689655173`, and selective gate `0.06952887537993921`. Relative to Q0, temporal mean changes `2520` public event rows and selective changes `1545`; this supports the conclusion that physical membership authority remains a safety risk even though TrackEval changes are modest.

## P1 physical implementation and TrackEval

The temporal-mean full native replay is `/data2/usr_for_deadline/trackocd_phase85/project_outputs/metrics/temporal_mean_full.json`: `682335` rows across `370` videos, with `34495` reconnect and `90884` keep decisions. It preserves Q0 rows and uses causal observed-step timing, dormant-only candidates, fixed accept score 0.5 and max gap 16. Lineage SHA256 is `7bf5f00363b69d6e992f42e5ea165c5b870927338f2dd242847c081072e99bf0` and union-event SHA256 is `9dd7eb048268859e29202138f15fbf4bf3b2ea849bd39434b4c8c1a5ae02fe4b`.

The class-agnostic event-video TrackEval comparison is:

| stream | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Frag |
|---|---|---|---|---|---|---|---|
| Q0 | 14.619000 | 6.814900 | 32.246000 | -822.320000 | 7.748500 | 2647.000000 | 748.000000 |
| temporal_mean | 14.743000 | 6.807000 | 32.811000 | -821.980000 | 7.990500 | 2614.000000 | 748.000000 |
| selective_gate | 14.658000 | 6.815000 | 32.406000 | -822.140000 | 7.849200 | 2629.000000 | 749.000000 |

These are diagnostics on the same 91 event videos; they do not replace full sealed MOT or persistent Commit-CT.

The corresponding full 370-video class-agnostic TrackEval summaries (Q0 versus temporal mean) are:

| stream | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Frag |
|---|---|---|---|---|---|---|---|
| Q0 full | 13.484000 | 5.957300 | 31.725000 | -960.450000 | 6.628300 | 10911.000000 | 3082.000000 |
| temporal full | 13.625000 | 5.948700 | 32.397000 | -960.200000 | 6.843100 | 10813.000000 | 3081.000000 |

Selective physical TrackEval is intentionally restricted to the event91 diagnostic subset; no additional full-stream learned model score is treated as a gate.

## P3/P4 real Q0 adapter parity and physical→R

The Q0 adapter `/data2/usr_for_deadline/trackocd_phase85/project_outputs/audit/physical_r_q0_q0_parity_v5_adapter.json` reconstructs `FrozenTrackTable.raw_vector` from native lineage and passes the registered gate: max absolute vector error `5.960464477539063e-08`, bad queries `0`, denominator `984`. The join has `43319` exact rows, `79` explicit IoU fallbacks and `25` unmatched rows out of `43423`.

### Frozen-R prefix comparison: corrected temporal mean

| prefix | queries | R@1 | raw R@1 | mAP | raw mAP | hard-gap | raw gap | unsafe |
|---|---|---|---|---|---|---|---|---|
| 1 | 984 | 0.903385 | 0.886107 | 0.848525 | 0.849847 | 0.157081 | 0.167362 | 5 |
| 2 | 984 | 0.863466 | 0.920327 | 0.841706 | 0.865919 | 0.158387 | 0.182402 | 18 |
| 4 | 984 | 0.830241 | 0.911356 | 0.825233 | 0.863895 | 0.155079 | 0.197118 | 26 |
| 8 | 984 | 0.844988 | 0.913191 | 0.821146 | 0.860234 | 0.138170 | 0.194414 | 22 |
| 16 | 984 | 0.818582 | 0.899976 | 0.811837 | 0.859949 | 0.130423 | 0.194436 | 22 |

At prefix16 the one-anchor temporal stream is R@1 `0.8185816337861971` versus raw `0.8999761394529293`, mAP `0.8118369769335796` versus `0.8599492940330089`, hard-gap `0.13042264435504686` versus `0.19443571733250695`, with `22` unsafe flips and zero folds non-decreasing in both metrics. This is **PHYSICAL_TO_R_FAIL**.

The TRAIN-only selective union gate is a separate physical route. Its lineage/union hashes and TrackEval output are in `/data2/usr_for_deadline/trackocd_phase85/project_outputs/physical/selective_formal_r1/full_temporal_summary.json` and `/data2/usr_for_deadline/trackocd_phase85/project_outputs/metrics/trackeval/selective_event91/selective_event91/cls_comb_cls_av_summary.txt`. Selective p16 R@1/mAP/hard-gap are `0.8148736575172452`/`0.8064953489823633`/`0.12892581462462385` versus raw, with `22` unsafe flips and zero folds non-decreasing. This is **PHYSICAL_SELECTIVE_TO_R_FAIL**.

## B85S raw-anchored set-aware support

The legal TRAIN support manifest and top-K audit are `/data2/usr_for_deadline/trackocd_phase85/project_outputs/manifests/phase85_support_prefix_manifest.json` and `/data2/usr_for_deadline/trackocd_phase85/project_outputs/audit/support_topk_audit.json`. It contains `2095` groups, `46649` candidate rows and feature dimension `19`. The preregistered TRAIN audit fixed K=32 because top16 oracle recall was below 90%; all prefixes share the same causal `stable_raw_topk` implementation. The TRAIN/event distribution comparison is `/data2/usr_for_deadline/trackocd_phase85/project_outputs/audit/support_train_event_shift.json`.

Formal bounded reranker training used 15 effective epochs per fold and atomic checkpoints:

| fold | steps | epochs | fit groups | val groups | raw top1 | rerank top1 | harm | net rescue | bridge use | defer acc | checkpoint | sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 9000 | 15 | 375 | 1080 | 0.347826 | 0.231884 | 12 | -8 | 0.125000 | 0.848148 | /data2/usr_for_deadline/trackocd_phase85/project_outputs/checkpoints/support_reranker_formal_r1_f0_step009000.pt | 3d152a04cc2bc0cfa85646344bdee588330808e591fd5297aeb2cc31206a2ce4 |
| 1 | 44460 | 15 | 1630 | 310 | 0.330579 | 0.338843 | 3 | 1 | 0.293548 | 0.632258 | /data2/usr_for_deadline/trackocd_phase85/project_outputs/checkpoints/support_reranker_formal_r1_f1_step044460.pt | e8c923d3ea2b7e7210ded3a6cee2516654f7e451d4a215d2fd672dad1e80cec3 |
| 2 | 19080 | 15 | 730 | 705 | 0.189873 | 0.189873 | 3 | 0 | 0.156028 | 0.760284 | /data2/usr_for_deadline/trackocd_phase85/project_outputs/checkpoints/support_reranker_formal_r1_f2_step019080.pt | 8f8a83294b5222eb8f2a4aab77363a4a3b90935898bee35b9971a3e161573bdb |

The fixed event replay reports raw, bounded-rerank-only and final rerank+DEFER separately:

| prefix | polarity | raw reliable | reranked reliable | final reliable | deferred | frozen source | frozen target | frozen both |
|---|---|---|---|---|---|---|---|---|
| 1 | positive | 10 | 11 | 3 | 48 | 49 | 29 | 17 |
| 1 | negative | 2 | 5 | 3 | 41 | 49 | 29 | 18 |
| 2 | positive | 14 | 16 | 5 | 54 | 49 | 35 | 22 |
| 2 | negative | 4 | 6 | 3 | 43 | 49 | 35 | 20 |
| 4 | positive | 15 | 19 | 7 | 55 | 49 | 36 | 22 |
| 4 | negative | 5 | 9 | 5 | 45 | 49 | 36 | 21 |
| 8 | positive | 17 | 22 | 7 | 57 | 49 | 38 | 23 |
| 8 | negative | 8 | 10 | 5 | 46 | 49 | 38 | 22 |
| 16 | positive | 20 | 23 | 8 | 58 | 49 | 40 | 25 |
| 16 | negative | 8 | 15 | 8 | 47 | 49 | 40 | 24 |

At prefix16 this is positive `20/76 → 23/76 → 8/76` and negative `8/76 → 15/76 → 8/76`. The reranker-only increase is not safe because negative activation rises by seven events; the DEFER head abstains on 58 positive events and therefore cannot recover the raw anchor. The final learned route is **B85S_FAIL_SAFETY**.

The registered follow-up “raw ranking + learned DEFER” policy (ignoring the reranker score while retaining the TRAIN-only DEFER decision) was also evaluated. At prefix16 it retained `5/76` positive and `2/76` negative reliable events after deferring 58/47 events, versus the raw 20/8 reference. This policy also fails to preserve the raw anchor and does not authorize alignment.

### Event-level p16 evidence (all 76 positive and 76 negative events)

The following table is generated from the frozen replay and joined only to the post-hoc pool/selection taxonomy; it is not used to choose any model or threshold.

| event | uid | fold | polarity | source | target | raw | rerank | final | defer | taxonomy |
|---|---|---|---|---|---|---|---|---|---|---|
| p19r-neg:f0:c347:s72:t1982:n0 | evt_000076 | 0 | negative | v2522:p6147 | v1982:p4998 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f0:c347:s72:t1982:n1 | evt_000077 | 0 | negative | v170:p723 | v1982:p5006 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f0:c347:s72:t1982:n2 | evt_000078 | 0 | negative | v170:p723 | v1982:p5007 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f0:c347:s72:t2023:n3 | evt_000079 | 0 | negative | v2340:p5707 | v2023:p5057 | false | false | false | true | pool_limited |
| p19r-neg:f0:c579:s92:t1524:n1 | evt_000081 | 0 | negative | v1365:p3836 | v1524:p4025 | false | true | false | true | defer_harm_or_safe_candidate |
| p19r-neg:f0:c579:s92:t1524:n2 | evt_000082 | 0 | negative | v1459:p3916 | v1524:p4054 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f0:c579:s92:t1748:n3 | evt_000083 | 0 | negative | v2580:p6177 | v1748:p4621 | false | false | false | true | pool_limited |
| p19r-neg:f0:c579:s92:t349:n0 | evt_000080 | 0 | negative | v1675:p4485 | v349:p1451 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f0:c805:s0:t1293:n0 | evt_000084 | 0 | negative | v2362:p5741 | v1293:p3721 | false | true | true | false | rerank_rescue |
| p19r-neg:f0:c805:s0:t1293:n1 | evt_000085 | 0 | negative | v2362:p5741 | v1293:p3729 | false | true | true | false | rerank_rescue |
| p19r-neg:f0:c805:s0:t1293:n2 | evt_000086 | 0 | negative | v2132:p5443 | v1293:p3749 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f0:c805:s0:t1293:n3 | evt_000087 | 0 | negative | v2132:p5443 | v1293:p3751 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f1:c211:s9:t959:n0 | evt_000088 | 1 | negative | v1173:p3543 | v959:p3028 | false | false | false | true | pool_limited |
| p19r-neg:f1:c211:s9:t965:n1 | evt_000089 | 1 | negative | v1160:p3449 | v965:p3031 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f1:c211:s9:t978:n2 | evt_000090 | 1 | negative | v1126:p3398 | v978:p3034 | false | false | false | true | pool_limited |
| p19r-neg:f1:c211:s9:t978:n3 | evt_000091 | 1 | negative | v1126:p3398 | v978:p3038 | false | false | false | true | pool_limited |
| p19r-neg:f1:c229:s119:t1790:n0 | evt_000092 | 1 | negative | v1231:p3661 | v1790:p4665 | true | true | true | false | raw_and_rank_reliable |
| p19r-neg:f1:c229:s119:t1808:n1 | evt_000093 | 1 | negative | v1793:p4677 | v1808:p4698 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f1:c229:s119:t2170:n2 | evt_000094 | 1 | negative | v1173:p3543 | v2170:p5494 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f1:c229:s119:t2314:n3 | evt_000095 | 1 | negative | v777:p2982 | v2314:p5617 | false | true | true | false | rerank_rescue |
| p19r-neg:f1:c235:s1126:t2677:n0 | evt_000096 | 1 | negative | v849:p3000 | v2677:p6248 | false | false | false | true | pool_limited |
| p19r-neg:f1:c235:s1126:t2758:n1 | evt_000097 | 1 | negative | v1231:p3661 | v2758:p6332 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f1:c235:s1126:t2845:n2 | evt_000098 | 1 | negative | v730:p2961 | v2845:p6393 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f1:c235:s1126:t2873:n3 | evt_000099 | 1 | negative | v212:p1057 | v2873:p6423 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f2:c1144:s377:t1927:n1 | evt_000121 | 2 | negative | v2851:p6413 | v1927:p4823 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f2:c1144:s377:t2565:n2 | evt_000122 | 2 | negative | v1882:p4783 | v2565:p6172 | false | false | false | true | pool_limited |
| p19r-neg:f2:c1144:s377:t2565:n3 | evt_000123 | 2 | negative | v1882:p4783 | v2565:p6173 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f2:c1144:s377:t827:n0 | evt_000120 | 2 | negative | v220:p1075 | v827:p2996 | false | false | false | true | pool_limited |
| p19r-neg:f2:c118:s275:t1831:n0 | evt_000100 | 2 | negative | v16:p181 | v1831:p4762 | true | true | false | true | defer_harm_or_safe_candidate |
| p19r-neg:f2:c118:s275:t1831:n2 | evt_000102 | 2 | negative | v16:p181 | v1831:p4762 | true | true | false | true | defer_harm_or_safe_candidate |
| p19r-neg:f2:c118:s275:t2340:n1 | evt_000101 | 2 | negative | v2251:p5604 | v2340:p5696 | false | true | true | false | rerank_rescue |
| p19r-neg:f2:c118:s275:t2340:n3 | evt_000103 | 2 | negative | v2251:p5604 | v2340:p5696 | false | true | true | false | rerank_rescue |
| p19r-neg:f2:c126:s1882:t2148:n3 | evt_000107 | 2 | negative | v2936:p6474 | v2148:p5462 | true | true | false | true | defer_harm_or_safe_candidate |
| p19r-neg:f2:c126:s220:t2148:n0 | evt_000104 | 2 | negative | v2936:p6474 | v2148:p5462 | true | true | false | true | defer_harm_or_safe_candidate |
| p19r-neg:f2:c126:s220:t2251:n1 | evt_000105 | 2 | negative | v275:p1237 | v2251:p5605 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f2:c126:s220:t2851:n2 | evt_000106 | 2 | negative | v1927:p4823 | v2851:p6413 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f2:c133:s1:t2128:n0 | evt_000108 | 2 | negative | v220:p1075 | v2128:p5419 | true | true | false | true | defer_harm_or_safe_candidate |
| p19r-neg:f2:c133:s1:t2194:n1 | evt_000109 | 2 | negative | v2197:p5532 | v2194:p5516 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f2:c133:s1:t2371:n2 | evt_000110 | 2 | negative | v1927:p4823 | v2371:p5744 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f2:c133:s1:t2874:n3 | evt_000111 | 2 | negative | v2851:p6413 | v2874:p6439 | false | false | false | true | pool_limited |
| p19r-neg:f2:c237:s1453:t2170:n2 | evt_000114 | 2 | negative | v1914:p4793 | v2170:p5508 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f2:c237:s1453:t2251:n3 | evt_000115 | 2 | negative | v2340:p5696 | v2251:p5604 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f2:c237:s483:t2170:n0 | evt_000112 | 2 | negative | v1914:p4793 | v2170:p5508 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f2:c237:s483:t2251:n1 | evt_000113 | 2 | negative | v2340:p5696 | v2251:p5604 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f2:c382:s16:t1142:n0 | evt_000116 | 2 | negative | v1882:p4775 | v1142:p3405 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f2:c382:s16:t1617:n1 | evt_000117 | 2 | negative | v1831:p4762 | v1617:p4287 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f2:c382:s16:t1632:n2 | evt_000118 | 2 | negative | v1929:p4829 | v1632:p4336 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f2:c382:s16:t1687:n3 | evt_000119 | 2 | negative | v1882:p4783 | v1687:p4511 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f3:c35:s545:t2059:n0 | evt_000124 | 3 | negative | v1742:p4612 | v2059:p5203 | false | false | false | true | pool_limited |
| p19r-neg:f3:c35:s545:t2215:n1 | evt_000125 | 3 | negative | v1365:p3835 | v2215:p5575 | true | true | false | true | defer_harm_or_safe_candidate |
| p19r-neg:f3:c35:s545:t2851:n2 | evt_000126 | 3 | negative | v0:p15 | v2851:p6417 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f3:c35:s653:t2059:n3 | evt_000127 | 3 | negative | v1742:p4612 | v2059:p5203 | false | false | false | true | pool_limited |
| p19r-neg:f3:c41:s146:t1365:n0 | evt_000128 | 3 | negative | v2522:p6140 | v1365:p3835 | false | false | false | false | pool_limited |
| p19r-neg:f3:c41:s146:t1643:n1 | evt_000129 | 3 | negative | v1955:p4983 | v1643:p4384 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f3:c41:s146:t1939:n2 | evt_000130 | 3 | negative | v212:p1053 | v1939:p4891 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f3:c41:s146:t1939:n3 | evt_000131 | 3 | negative | v212:p1053 | v1939:p4894 | false | false | false | true | pool_limited |
| p19r-neg:f3:c714:s212:t1927:n0 | evt_000144 | 3 | negative | v3:p63 | v1927:p4827 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f3:c714:s212:t1927:n3 | evt_000147 | 3 | negative | v3:p63 | v1927:p4827 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f3:c714:s212:t2048:n1 | evt_000145 | 3 | negative | v1681:p4500 | v2048:p5112 | false | false | false | true | pool_limited |
| p19r-neg:f3:c714:s212:t2048:n2 | evt_000146 | 3 | negative | v1681:p4500 | v2048:p5119 | false | false | false | true | pool_limited |
| p19r-neg:f3:c81:s575:t1814:n0 | evt_000132 | 3 | negative | v3:p63 | v1814:p4731 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f3:c81:s575:t1814:n1 | evt_000133 | 3 | negative | v978:p3032 | v1814:p4736 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f3:c81:s575:t1814:n2 | evt_000134 | 3 | negative | v978:p3032 | v1814:p4752 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f3:c81:s575:t1955:n3 | evt_000135 | 3 | negative | v1643:p4384 | v1955:p4983 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f3:c95:s0:t1933:n1 | evt_000137 | 3 | negative | v545:p2471 | v1933:p4835 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f3:c95:s0:t2051:n2 | evt_000138 | 3 | negative | v1955:p4983 | v2051:p5158 | false | false | false | false | pool_present_selection_gap |
| p19r-neg:f3:c95:s0:t2522:n3 | evt_000139 | 3 | negative | v1365:p3835 | v2522:p6140 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f3:c95:s0:t980:n0 | evt_000136 | 3 | negative | v1731:p4583 | v980:p3041 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f3:c980:s1681:t1905:n0 | evt_000148 | 3 | negative | v1742:p4612 | v1905:p4792 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f3:c980:s1681:t2460:n1 | evt_000149 | 3 | negative | v575:p2640 | v2460:p5925 | false | true | true | false | rerank_rescue |
| p19r-neg:f3:c980:s1681:t2460:n2 | evt_000150 | 3 | negative | v212:p1053 | v2460:p5943 | false | false | false | true | pool_limited |
| p19r-neg:f3:c980:s1731:t1905:n3 | evt_000151 | 3 | negative | v1742:p4612 | v1905:p4792 | false | false | false | true | pool_present_selection_gap |
| p19r-neg:f3:c99:s569:t1691:n0 | evt_000140 | 3 | negative | v1041:p3135 | v1691:p4526 | true | true | true | false | raw_and_rank_reliable |
| p19r-neg:f3:c99:s569:t2414:n1 | evt_000141 | 3 | negative | v3:p63 | v2414:p5867 | false | false | false | false | pool_limited |
| p19r-neg:f3:c99:s569:t2414:n2 | evt_000142 | 3 | negative | v2215:p5575 | v2414:p5869 | false | false | false | true | pool_limited |
| p19r-neg:f3:c99:s569:t2414:n3 | evt_000143 | 3 | negative | v146:p636 | v2414:p5871 | false | false | false | true | pool_limited |
| p19r-pos:f0:c347:s72:t1982:n0 | evt_000000 | 0 | positive | v72:p296 | v1982:p4998 | false | false | false | false | pool_present_selection_gap |
| p19r-pos:f0:c347:s72:t1982:n1 | evt_000001 | 0 | positive | v72:p296 | v1982:p5006 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f0:c347:s72:t1982:n2 | evt_000002 | 0 | positive | v72:p296 | v1982:p5007 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f0:c347:s72:t2023:n3 | evt_000003 | 0 | positive | v72:p296 | v2023:p5057 | false | false | false | true | pool_limited |
| p19r-pos:f0:c579:s92:t1524:n1 | evt_000005 | 0 | positive | v92:p368 | v1524:p4025 | true | true | true | false | raw_and_rank_reliable |
| p19r-pos:f0:c579:s92:t1524:n2 | evt_000006 | 0 | positive | v92:p368 | v1524:p4054 | false | true | true | true | defer_harm_or_safe_candidate |
| p19r-pos:f0:c579:s92:t1748:n3 | evt_000007 | 0 | positive | v92:p368 | v1748:p4621 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f0:c579:s92:t349:n0 | evt_000004 | 0 | positive | v92:p368 | v349:p1451 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f0:c805:s0:t1293:n0 | evt_000008 | 0 | positive | v0:p24 | v1293:p3721 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f0:c805:s0:t1293:n1 | evt_000009 | 0 | positive | v0:p24 | v1293:p3729 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f0:c805:s0:t1293:n2 | evt_000010 | 0 | positive | v0:p24 | v1293:p3749 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f0:c805:s0:t1293:n3 | evt_000011 | 0 | positive | v0:p24 | v1293:p3751 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f1:c211:s9:t959:n0 | evt_000012 | 1 | positive | v9:p144 | v959:p3028 | false | false | false | true | pool_limited |
| p19r-pos:f1:c211:s9:t965:n1 | evt_000013 | 1 | positive | v9:p144 | v965:p3031 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f1:c211:s9:t978:n2 | evt_000014 | 1 | positive | v9:p144 | v978:p3034 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f1:c211:s9:t978:n3 | evt_000015 | 1 | positive | v9:p144 | v978:p3038 | false | false | false | true | pool_limited |
| p19r-pos:f1:c229:s119:t1790:n0 | evt_000016 | 1 | positive | v119:p470 | v1790:p4665 | false | true | true | false | pool_limited |
| p19r-pos:f1:c229:s119:t1808:n1 | evt_000017 | 1 | positive | v119:p470 | v1808:p4698 | false | false | false | false | pool_present_selection_gap |
| p19r-pos:f1:c229:s119:t2170:n2 | evt_000018 | 1 | positive | v119:p470 | v2170:p5494 | false | false | false | false | pool_present_selection_gap |
| p19r-pos:f1:c229:s119:t2314:n3 | evt_000019 | 1 | positive | v119:p470 | v2314:p5617 | true | true | true | false | raw_and_rank_reliable |
| p19r-pos:f1:c235:s1126:t2677:n0 | evt_000020 | 1 | positive | v1126:p3398 | v2677:p6248 | false | false | false | true | pool_limited |
| p19r-pos:f1:c235:s1126:t2758:n1 | evt_000021 | 1 | positive | v1126:p3398 | v2758:p6332 | true | true | false | true | defer_harm_or_safe_candidate |
| p19r-pos:f1:c235:s1126:t2845:n2 | evt_000022 | 1 | positive | v1126:p3398 | v2845:p6393 | true | false | false | true | rerank_harm |
| p19r-pos:f1:c235:s1126:t2873:n3 | evt_000023 | 1 | positive | v1126:p3398 | v2873:p6423 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f2:c1144:s377:t1927:n1 | evt_000045 | 2 | positive | v377:p1615 | v1927:p4823 | false | true | false | true | defer_harm_or_safe_candidate |
| p19r-pos:f2:c1144:s377:t2565:n2 | evt_000046 | 2 | positive | v377:p1615 | v2565:p6172 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f2:c1144:s377:t2565:n3 | evt_000047 | 2 | positive | v377:p1615 | v2565:p6173 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f2:c1144:s377:t827:n0 | evt_000044 | 2 | positive | v377:p1615 | v827:p2996 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f2:c118:s275:t1831:n0 | evt_000024 | 2 | positive | v275:p1237 | v1831:p4762 | true | true | true | true | defer_harm_or_safe_candidate |
| p19r-pos:f2:c118:s275:t1831:n2 | evt_000026 | 2 | positive | v275:p1242 | v1831:p4762 | true | true | false | true | defer_harm_or_safe_candidate |
| p19r-pos:f2:c118:s275:t2340:n1 | evt_000025 | 2 | positive | v275:p1237 | v2340:p5696 | true | true | true | true | defer_harm_or_safe_candidate |
| p19r-pos:f2:c118:s275:t2340:n3 | evt_000027 | 2 | positive | v275:p1242 | v2340:p5696 | true | true | false | true | defer_harm_or_safe_candidate |
| p19r-pos:f2:c126:s1882:t2148:n3 | evt_000031 | 2 | positive | v1882:p4775 | v2148:p5462 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f2:c126:s220:t2148:n0 | evt_000028 | 2 | positive | v220:p1075 | v2148:p5462 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f2:c126:s220:t2251:n1 | evt_000029 | 2 | positive | v220:p1075 | v2251:p5605 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f2:c126:s220:t2851:n2 | evt_000030 | 2 | positive | v220:p1075 | v2851:p6413 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f2:c133:s1:t2128:n0 | evt_000032 | 2 | positive | v1:p42 | v2128:p5419 | true | true | false | true | defer_harm_or_safe_candidate |
| p19r-pos:f2:c133:s1:t2194:n1 | evt_000033 | 2 | positive | v1:p42 | v2194:p5516 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f2:c133:s1:t2371:n2 | evt_000034 | 2 | positive | v1:p42 | v2371:p5744 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f2:c133:s1:t2874:n3 | evt_000035 | 2 | positive | v1:p42 | v2874:p6439 | false | false | false | true | pool_limited |
| p19r-pos:f2:c237:s1453:t2170:n2 | evt_000038 | 2 | positive | v1453:p3869 | v2170:p5508 | false | true | false | true | defer_harm_or_safe_candidate |
| p19r-pos:f2:c237:s1453:t2251:n3 | evt_000039 | 2 | positive | v1453:p3869 | v2251:p5604 | true | false | false | true | rerank_harm |
| p19r-pos:f2:c237:s483:t2170:n0 | evt_000036 | 2 | positive | v483:p2246 | v2170:p5508 | false | false | false | false | pool_present_selection_gap |
| p19r-pos:f2:c237:s483:t2251:n1 | evt_000037 | 2 | positive | v483:p2246 | v2251:p5604 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f2:c382:s16:t1142:n0 | evt_000040 | 2 | positive | v16:p181 | v1142:p3405 | false | true | false | true | defer_harm_or_safe_candidate |
| p19r-pos:f2:c382:s16:t1617:n1 | evt_000041 | 2 | positive | v16:p181 | v1617:p4287 | true | false | false | true | rerank_harm |
| p19r-pos:f2:c382:s16:t1632:n2 | evt_000042 | 2 | positive | v16:p181 | v1632:p4336 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f2:c382:s16:t1687:n3 | evt_000043 | 2 | positive | v16:p181 | v1687:p4511 | true | true | false | true | defer_harm_or_safe_candidate |
| p19r-pos:f3:c35:s545:t2059:n0 | evt_000048 | 3 | positive | v545:p2471 | v2059:p5203 | false | false | false | false | pool_limited |
| p19r-pos:f3:c35:s545:t2215:n1 | evt_000049 | 3 | positive | v545:p2471 | v2215:p5575 | true | true | true | false | raw_and_rank_reliable |
| p19r-pos:f3:c35:s545:t2851:n2 | evt_000050 | 3 | positive | v545:p2471 | v2851:p6417 | false | false | false | false | pool_present_selection_gap |
| p19r-pos:f3:c35:s653:t2059:n3 | evt_000051 | 3 | positive | v653:p2864 | v2059:p5203 | false | false | false | true | pool_limited |
| p19r-pos:f3:c41:s146:t1365:n0 | evt_000052 | 3 | positive | v146:p612 | v1365:p3835 | false | false | false | true | pool_limited |
| p19r-pos:f3:c41:s146:t1643:n1 | evt_000053 | 3 | positive | v146:p612 | v1643:p4384 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f3:c41:s146:t1939:n2 | evt_000054 | 3 | positive | v146:p612 | v1939:p4891 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f3:c41:s146:t1939:n3 | evt_000055 | 3 | positive | v146:p612 | v1939:p4894 | false | false | false | true | pool_limited |
| p19r-pos:f3:c714:s212:t1927:n0 | evt_000068 | 3 | positive | v212:p1053 | v1927:p4827 | true | true | false | true | defer_harm_or_safe_candidate |
| p19r-pos:f3:c714:s212:t1927:n3 | evt_000071 | 3 | positive | v212:p1064 | v1927:p4827 | true | true | false | true | defer_harm_or_safe_candidate |
| p19r-pos:f3:c714:s212:t2048:n1 | evt_000069 | 3 | positive | v212:p1053 | v2048:p5112 | false | false | false | true | pool_limited |
| p19r-pos:f3:c714:s212:t2048:n2 | evt_000070 | 3 | positive | v212:p1053 | v2048:p5119 | true | true | false | true | defer_harm_or_safe_candidate |
| p19r-pos:f3:c81:s575:t1814:n0 | evt_000056 | 3 | positive | v575:p2639 | v1814:p4731 | false | false | false | false | pool_present_selection_gap |
| p19r-pos:f3:c81:s575:t1814:n1 | evt_000057 | 3 | positive | v575:p2639 | v1814:p4736 | false | false | false | false | pool_present_selection_gap |
| p19r-pos:f3:c81:s575:t1814:n2 | evt_000058 | 3 | positive | v575:p2639 | v1814:p4752 | false | false | false | false | pool_limited |
| p19r-pos:f3:c81:s575:t1955:n3 | evt_000059 | 3 | positive | v575:p2639 | v1955:p4983 | false | false | false | false | pool_present_selection_gap |
| p19r-pos:f3:c95:s0:t1933:n1 | evt_000061 | 3 | positive | v0:p15 | v1933:p4835 | true | true | false | true | defer_harm_or_safe_candidate |
| p19r-pos:f3:c95:s0:t2051:n2 | evt_000062 | 3 | positive | v0:p15 | v2051:p5158 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f3:c95:s0:t2522:n3 | evt_000063 | 3 | positive | v0:p15 | v2522:p6140 | true | true | false | true | defer_harm_or_safe_candidate |
| p19r-pos:f3:c95:s0:t980:n0 | evt_000060 | 3 | positive | v0:p15 | v980:p3041 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f3:c980:s1681:t1905:n0 | evt_000072 | 3 | positive | v1681:p4500 | v1905:p4792 | false | false | false | true | pool_present_selection_gap |
| p19r-pos:f3:c980:s1681:t2460:n1 | evt_000073 | 3 | positive | v1681:p4500 | v2460:p5925 | true | true | false | true | defer_harm_or_safe_candidate |
| p19r-pos:f3:c980:s1681:t2460:n2 | evt_000074 | 3 | positive | v1681:p4500 | v2460:p5943 | false | false | false | true | pool_limited |
| p19r-pos:f3:c980:s1731:t1905:n3 | evt_000075 | 3 | positive | v1731:p4583 | v1905:p4792 | false | true | false | true | defer_harm_or_safe_candidate |
| p19r-pos:f3:c99:s569:t1691:n0 | evt_000064 | 3 | positive | v569:p2628 | v1691:p4526 | true | true | true | false | raw_and_rank_reliable |
| p19r-pos:f3:c99:s569:t2414:n1 | evt_000065 | 3 | positive | v569:p2628 | v2414:p5867 | false | false | false | false | pool_limited |
| p19r-pos:f3:c99:s569:t2414:n2 | evt_000066 | 3 | positive | v569:p2628 | v2414:p5869 | false | false | false | false | pool_limited |
| p19r-pos:f3:c99:s569:t2414:n3 | evt_000067 | 3 | positive | v569:p2628 | v2414:p5871 | false | false | false | false | pool_limited |

The machine-readable full taxonomy is `/data2/usr_for_deadline/trackocd_phase85/project_outputs/audit/support_alignment_feasibility.json`. At p16, 60/76 positive and 57/76 negative events have a top32 pool candidate with IoU≥0.5 under the fixed post-hoc audit, but 35 positive and 42 negative events remain pool-present selection gaps. This separates candidate availability from learned ranking/defer behavior; it does not authorize alignment because the registered positive/safety routing criterion (positive≥26 and negative≤9 selected reliable events) is not met.

### P1+S0 selective-source combination

The causal selective-lineage source cache covers `1298` tracks at p16 with no fallback. Its independent replay is `/data2/usr_for_deadline/trackocd_phase85/project_outputs/metrics/support_event_replay_selective_source_v1.json`: p16 raw/rerank/final reliable events are positive `8/76 → 12/76 → 6/76` and negative `6/76 → 8/76 → 5/76`. This closes the physical-source transfer hypothesis for this window.

## Route gates and what was not run

| route | decision | evidence |
|---|---|---|
| P0/P1 contract repair | PASS | temporal state, joins, parity/provenance repaired |
| P3 Q0 parity | PASS | 984 queries; max error <=1e-6 |
| P5 temporal physical→R | FAIL | R@1/mAP below raw; 22 unsafe; 0/4 folds safe |
| P5 selective physical→R | FAIL | R@1/mAP below raw; 22 unsafe; 0/4 folds safe |
| B85S reranker | FAIL | +3 positive but +7 negative at p16 |
| B85S DEFER | FAIL | final positive 8/76; excessive abstention |
| B85S raw+DEFER | FAIL | positive 5/76, negative 2/76; raw anchor not preserved |
| B85S selective-source | FAIL | raw p16 positive 8/76 |
| alignment | NOT_AUTHORIZED | routing criterion not met |
| controller/StateMemory/Commit-CT | NOT_RUN | no safe P/R route |
| sealed/public evaluation | NOT_RUN | sealed boundary remained closed |

No persistent Commit-CT number is reported for Phase85: the controller was not authorized after the physical/support gates failed. Retrieval and TrackEval values above are diagnostics, never a substitute for causal OCD.

## Resources, repairs, and integrity

- Resource/space snapshots and symlink ledger are in `/data2/usr_for_deadline/trackocd_phase85/project_outputs/audit/research_ledger.json` and the registration. Large files remain on `/data2`; nothing was copied into `/data1` beyond tracked small code/docs.
- The latest low-frequency resource snapshot includes `free -h`, `df -h /data1 /data2`, `nvidia-smi`, and process count in the research ledger (`60` nvidia-smi lines captured); no Phase85 worker remained at the integrity check.
- One initial selective replay implementation was terminated only for task-owned PIDs `32861,32862` after profiling an avoidable per-row Torch bottleneck; the NumPy frozen-forward replacement passed smoke/targeted tests. No OOM and no external process termination occurred. The initial `physical_gate_smoke_r1` marker is retained without `.done` as failed evidence: `physical_gate_smoke_r1_f0.launched`.
- A system-Python missing-torch invocation and a one-time audit import-path failure were repaired with the audited environment/project-root path; no scientific output was overwritten. All outputs use atomic writes. JSON and provenance checks passed before this report.
- The final integrity audit `/data2/usr_for_deadline/trackocd_phase85/project_outputs/audit/integrity_check.json` parsed `79` JSON artifacts with `0` parse failures, found `0` missing key artifacts, `133` checkpoints and no forbidden named files or residual Phase85 process. The failed smoke marker is intentionally preserved as evidence.
- Historical Phase84/Phase83 files were read-only; public DEV+/Q1/new-model/sealed labels were not accessed.

## Reproduction

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
python scripts/iclr27_phase85/audit_phase84_issues.py
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase85/run_full_temporal_mean_physical.py --tag temporal_mean_full
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase85/build_physical_r_adapter.py --mode q0 --tag q0_parity_v5
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase85/build_physical_r_adapter.py --mode selective --tag selective_gate_v1
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase85/evaluate_physical_r.py --candidate outputs/iclr27_phase85/manifests/physical_r_improved_improved_single_anchor_v2_vectors.npz --candidate-name improved_single_anchor_v2 --output-tag physical_r_temporal_comparison_v2
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase85/evaluate_physical_r.py --candidate outputs/iclr27_phase85/manifests/physical_r_selective_selective_gate_v1_vectors.npz --candidate-name selective_gate_v1 --output-tag physical_r_selective_comparison
python scripts/iclr27_phase85/audit_support_selection.py
python scripts/iclr27_phase85/audit_support_alignment_feasibility.py
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase85/evaluate_support_phase85.py --policy raw_defer --output-tag support_event_replay_raw_defer_v1
python scripts/iclr27_phase85/generate_final_report.py --check-only
```

The final report is generated only after the registered lock. The machine decision is `/data2/usr_for_deadline/trackocd_phase85/project_outputs/audit/phase85_decision.json` and provenance is `/data2/usr_for_deadline/trackocd_phase85/project_outputs/audit/report_provenance.json`.

## Limitations and next direction

Phase85 establishes valid contract-level negative evidence: temporal appearance state and adapter parity are no longer confounds, yet physical reassociation does not transfer to the frozen R space and raw-anchored learned support is not safe on the fixed event distribution. The dominant remaining evidence is a candidate-pool selection/generalization gap plus physical canonical-root contamination; support alignment and controller behavior remain unmeasured under this window by design. A future route must be separately registered around causal source/query coverage and representation/interface supervision, preserving raw fallback and physical MOT safety. Threshold, StateMemory, controller, and backbone lottery are not justified by these artifacts.

## Artifact index

All source paths, route/tag/schema assertions and SHA256 values are in `/data2/usr_for_deadline/trackocd_phase85/project_outputs/audit/report_provenance.json`; repair and resource events are in `/data2/usr_for_deadline/trackocd_phase85/project_outputs/audit/repair_events.json` and `/data2/usr_for_deadline/trackocd_phase85/project_outputs/audit/research_ledger.json`. `FINAL_REPOSITORY_HEAD` is recorded by the finalization commit in the accompanying machine decision; this report's generation head is explicitly separated above.
