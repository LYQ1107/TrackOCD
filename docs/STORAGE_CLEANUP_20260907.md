# Storage cleanup — 2026-09-07

`/data1` reached 100% usage (about 27 GB free). The following completed,
inactive historical experiment directories are being moved to the recoverable
archive below; their original paths will be replaced by symlinks so existing
reports and scripts continue to resolve:

Archive root: `/data2/usr_for_deadline/trackocd_archive/20260907/outputs/`

| Directory | Approx. size before move | Reason |
|---|---:|---|
| `outputs/iclr27_phase71` | 15 GB | completed historical route; no active worker |
| `outputs/iclr27_phase6b` | 12 GB | completed historical route; no active worker |
| `outputs/iclr27_phase4q` | 12 GB | completed OVTR ablations; preserved in archive |
| `outputs/iclr27_phase6a` | 6.0 GB | completed historical route; no active worker |
| `outputs/iclr27_phase4b` | 5.8 GB | completed historical route; no active worker |
| `outputs/iclr27_phase60` | 4.2 GB | completed historical route; no active worker |
| `outputs/iclr27_phase4m` | 2.7 GB | completed historical route; no active worker |
| `outputs/iclr27_phase3b` | 2.6 GB | completed historical route; no active worker |
| `outputs/iclr27_phase4r` | 2.1 GB | completed historical route; no active worker |
| `outputs/iclr27_phase4p` | 1.1 GB | completed historical route; no active worker |
| `outputs/iclr27_phase4l` | 1.2 GB | completed historical route; no active worker |

The move is recoverable: files remain under `/data2`; no checkpoint, report,
metric, or source artifact is intentionally deleted. The final table with
actual sizes and symlink targets is appended after the move.

## Completed move

All eleven directories were moved successfully and replaced with symlinks:

| Directory | Archived size | Symlink target |
|---|---:|---|
| `outputs/iclr27_phase71` | 15 GB | `/data2/usr_for_deadline/trackocd_archive/20260907/outputs/iclr27_phase71` |
| `outputs/iclr27_phase6b` | 12 GB | `/data2/usr_for_deadline/trackocd_archive/20260907/outputs/iclr27_phase6b` |
| `outputs/iclr27_phase4q` | 12 GB | `/data2/usr_for_deadline/trackocd_archive/20260907/outputs/iclr27_phase4q` |
| `outputs/iclr27_phase6a` | 6.0 GB | `/data2/usr_for_deadline/trackocd_archive/20260907/outputs/iclr27_phase6a` |
| `outputs/iclr27_phase4b` | 5.8 GB | `/data2/usr_for_deadline/trackocd_archive/20260907/outputs/iclr27_phase4b` |
| `outputs/iclr27_phase60` | 4.2 GB | `/data2/usr_for_deadline/trackocd_archive/20260907/outputs/iclr27_phase60` |
| `outputs/iclr27_phase4m` | 2.7 GB | `/data2/usr_for_deadline/trackocd_archive/20260907/outputs/iclr27_phase4m` |
| `outputs/iclr27_phase3b` | 2.6 GB | `/data2/usr_for_deadline/trackocd_archive/20260907/outputs/iclr27_phase3b` |
| `outputs/iclr27_phase4r` | 2.1 GB | `/data2/usr_for_deadline/trackocd_archive/20260907/outputs/iclr27_phase4r` |
| `outputs/iclr27_phase4p` | 1.1 GB | `/data2/usr_for_deadline/trackocd_archive/20260907/outputs/iclr27_phase4p` |
| `outputs/iclr27_phase4l` | 1.2 GB | `/data2/usr_for_deadline/trackocd_archive/20260907/outputs/iclr27_phase4l` |

`/data1` free space increased from approximately 27 GB to 90 GB; `/data2`
still has approximately 1008 GB free. The original paths resolve successfully
through their symlinks.
