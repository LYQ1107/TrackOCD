# CVPR research log

## 2026-10-12 — CVPR-0 preregistration

Read the full new taskbook and prior core reports/source. Created the new
branch from verified remote `7546527`; preserved old branch, main and outputs.
Registered an uncompleted full Goal A/B/C and the 600 GPU-hour / four-GPU
initial training ceiling. First work is CPU-only historical forensics with
strict sealed-ledger verification. No new fit, Val inference or asset download.
This initial delivery is registration, not completion of CVPR-0 or the goal.

Fresh regression: 396 tests passed in 51.52 s across core, v2,
PANDAS/ByteTrack and the new forensic tests, with no skips. CUDA regression
used an actually idle UUID after process/resource checks; no training occurred.
The six new tests check absorbing contamination, unknown-member distinction,
invalid token lifecycle, truncated ledgers and future append. They are forensic
unit tests, not a claim that the new end-to-end causal protocol is validated.

Registration `20358ac` was pushed, fetched and remotely verified before work.
The first CPU audit verified all 60 ledgers and all evaluator count assertions,
but final JSON serialization failed on NumPy integer counters. This is an
engineering failure, not a scientific finding or a completed delivery. Its
partial CSVs and progress are preserved. Normalize input scalar types, add a
strict JSON test, register a separately named bounded retry; no training or
new inference is involved.

User authorized cleaning unused project data if storage becomes insufficient.
No deletion is necessary now. Inventory and exact references must precede any
cleanup; preserve unique assets, active jobs, valid shards, historical results
and other projects. Ambiguous assets require user confirmation.

Corrected registration `c3a1bbe` pushed/fetched/remotely verified. CPU audit
completed all60cases in75.66s, peak RSS847216640B. Historical evaluator count
assertions pass. Atp16 allfourorders seed1027 emits1NEW+304560EXISTING;
seeds1028/1029 emit304561KNOWN. Observed collapse/propagation, not a unique
causal explanation. No new training, Val inference or Test read. Next register
bounded Train-only frozen-checkpoint diagnostics.12new tests pass, including
real label/role mutation, future append, non-tied ID renumbering and read-only
gradient replay. Equal-time renumbering remains pending; these limited tests
do not prove a dense causal-online evaluation protocol.
