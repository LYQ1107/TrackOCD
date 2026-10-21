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

## Fourth resource-protected pause of shard 1 retry (2026-10-21 UTC)

- A fresh single-worker retry was launched only for shard 1, with the same
  frozen COVTrack-native checkpoint/config and track-category consensus
  disabled. It reached `44615/120066` reported tasks before the machine
  reserve became unsafe.
- `MemAvailable` fell to about `11 GiB`, below the registered `32908273 kB`
  (25%) floor, while the owned native worker alone had about `22 GiB` RSS.
  The launcher has a start-time resource gate but no runtime checkpoint or
  offset-resume mechanism. The verified owned process group `2621979`
  therefore received `SIGTERM`; no external project process was targeted.
  After exit, `MemAvailable` recovered to `35791028 kB` and the owned GPU
  processes were absent.
- The compact evidence is
  `outputs/trackocd_cvpr/cvpr2/covtrack_train_dense_shards_v1_shard_1_tccoff4_20261021/resource_pause_20261021.json`.
  The native log is retained with SHA256
  `eac0321325c913e95da08fd0ded89822d016899fb3c2f20b51925e94fb1a8c54`.
  No `run.json`, `done.json`, or `normalization.json` exists; no partial
  observations were promoted and this namespace is not a valid completed
  shard or a resumable cache.
- This remains a resource-safety pause, not a scientific result: no training,
  formal OCD/MOT metric, model selection, TAO Val/Test label access, or Goal
  A/B/C acceptance is claimed. Shard 1 must be registered in a fresh
  namespace only after materially more RAM and storage headroom are available.
