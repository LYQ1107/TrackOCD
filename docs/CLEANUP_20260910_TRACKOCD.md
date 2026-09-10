# TrackOCD project storage cleanup — 2026-09-10

The Phase90 run uses only the read-only Phase88 shared feature/checkpoint
lineage and the Phase90/Phase89 output namespaces (which are on `/data2`).
The following frozen historical output directories were not open by any
process and were moved, rather than deleted, to the project archive:

`/data2/usr_for_deadline/trackocd_archive/project_cleanup_20260910/outputs/`

The original project paths are symlinks to the archived directories, so old
reports and scripts remain addressable while their blocks no longer consume
`/data1`.

| Original path | Bytes moved | Files | Archive target |
|---|---:|---:|---|
| `outputs/iclr27_phase15s` | 417,828,273 | 41 | `outputs/iclr27_phase15s` |
| `outputs/iclr27_phase17r` | 864,519,630 | 77 | `outputs/iclr27_phase17r` |
| `outputs/iclr27_phase18` | 995,595,977 | 229 | `outputs/iclr27_phase18` |
| `outputs/iclr27_phase19r` | 838,777,158 | 200 | `outputs/iclr27_phase19r` |
| `outputs/iclr27_phase29` | 737,396,965 | 95 | `outputs/iclr27_phase29` |
| `outputs/iclr27_phase50` | 472,826,631 | 54 | `outputs/iclr27_phase50` |
| `outputs/iclr27_phase62` | 489,108,576 | 1 | `outputs/iclr27_phase62` |
| `outputs/iclr27_phase68` | 644,149,597 | 1,224 | `outputs/iclr27_phase68` |

Two superseded SimOWT prediction snapshots were also moved to the same
archive and replaced by symlinks:

- `outputs/simowt/val_predictions.json` (1,220,382,281 bytes)
- `runs/simowt_partial_val_predictions.json` (285,379,056 bytes)

Their checksums, sizes, timestamps, and archive paths are recorded in
`/data2/usr_for_deadline/trackocd_archive/project_cleanup_20260910/`.
No current Phase90 checkpoint, marker, manifest, report, code, or active
training input was deleted. The cleanup did not terminate any process.
