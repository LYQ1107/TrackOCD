# CVPR-2 dense frontend RAM pause — 2026-10-21

This is a resource-safety record, not a scientific result.

Three owned, inference-only, category-free COVTrack Train namespaces were
gracefully interrupted after the frozen 25% RAM guard was crossed:

| shard | GPU | last progress | pause receipt |
| ---: | ---: | ---: | --- |
| 1 | 0 | 38,583 / 120,066 | `outputs/trackocd_cvpr/cvpr2/covtrack_train_dense_shards_v1_shard_1_tccoff3_20261021/resource_pause_20261021.json` |
| 2 | 1 | 36,420 / 108,606 | `outputs/trackocd_cvpr/cvpr2/covtrack_train_dense_shards_v1_shard_2_tccoff3_20261021/resource_pause_20261021.json` |
| 3 | 3 | 37,142 / 91,153 | `outputs/trackocd_cvpr/cvpr2/covtrack_train_dense_shards_v1_shard_3_tccoff3_20261021/resource_pause_20261021.json` |

At the pause decision, `MemTotal=131633092 kB`,
`MemAvailable=30828388 kB`, the reserve floor was `32908273 kB`, and
`SwapFree=356 kB`. Only the three task-owned PTYs received Ctrl-C. A
post-pause process/GPU check found no remaining process matching these run
IDs; external GPU processes were not touched. The native logs remain local and
their byte counts/SHA256 values are recorded in the compact receipts.

No `run.json`, `done.json`, or `normalization.json` exists for these three
namespaces. They are incomplete, no partial output is promoted, and no
training, OCD/MOT metric, model selection, TAO Val/Test label access, or
Goal A/B/C acceptance is claimed. Resumption requires a fresh namespace after
the RAM guard genuinely passes.
