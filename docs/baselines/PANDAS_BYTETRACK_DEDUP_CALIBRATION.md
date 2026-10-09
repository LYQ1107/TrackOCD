# PANDAS tracking-only NMS calibration

All four variants evaluated the same deterministic 20 videos, 22,322 full-stream
frames, and 792 canonical annotated images (2,510 GT boxes). Old tracking
outputs were reused. ByteTrack parameters were fixed throughout.

| Variant | Candidate mean / median | Retained | Physical tracks | Median track length | HOTA | DetA | AssA | LocA | IDF1 | MOTA | IDSW | Frag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original FG-0 | 167.87 / 162 | 100% | 51,125 | 10 | 0.101396 | 0.041197 | 0.253483 | 0.804879 | 0.045418 | -12.6817 | 821 | 153 |
| Dedup-A (0.5/50) | 36.61 / 41 | 21.81% | 4,218 | 11 | 0.273374 | 0.204234 | 0.373223 | 0.814622 | 0.221288 | -1.2307 | 256 | 158 |
| Dedup-B (0.7/50) | 42.17 / 50 | 25.12% | 5,186 | 12 | 0.232334 | 0.174916 | 0.315554 | 0.810703 | 0.171869 | -1.7386 | 394 | 160 |
| Dedup-C (0.7/100) | 59.84 / 60 | 35.64% | 5,183 | 12 | 0.231965 | 0.174498 | 0.315322 | 0.810694 | 0.171464 | -1.7486 | 394 | 159 |

| Variant | All proposal recall | Base proposal recall | Novel proposal recall | Lost previously covered GT |
|---|---:|---:|---:|---:|
| Original FG-0 | 0.797211 | 0.882820 | 0.558559 | 0 |
| Dedup-A | 0.773705 | 0.867925 | 0.481982 | 59 |
| Dedup-B | 0.772908 | 0.869414 | 0.436937 | 61 |
| Dedup-C | 0.790438 | 0.879841 | 0.522523 | 17 |

The metrics demonstrate a major tracking-interface problem in this calibration
set, but suppression also removes GT-covering candidates. The preset All/Base/
Novel recall-loss limit is 0.02 absolute. None passes: C loses 0.036036 Novel
recall, A loses 0.076577, and B loses 0.121622. C is locked for diagnostic
confirmation because it has the least All recall loss; it is not selected as
a validated full-Val replacement.

The complete recall, spatial one-to-one matching, unmatched-prediction counts,
GT-loss examples, per-file hashes and exact metrics are in
`outputs/baselines/pandas_dedup/calibration_results.json`. Independent
confirmation is complete and is recorded in the final report. No further
variants were run.
