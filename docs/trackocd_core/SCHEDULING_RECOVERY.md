# Frozen evaluation scheduling recovery (2026-10-11, Asia/Shanghai)

This is operational recovery of the already registered T5 experiment, not a
new experiment, representation, training run, checkpoint choice or Val sweep.

Read-only checks found the original inference/evaluator parents absent and
their progress files stale, while eight exact current-host inference workers
remained alive under PID 1 and continued to seal new cases. No terminal parent
receipt exists; the reason for parent disappearance is not yet established.
Historical terminal handles are not evidence of a live process and are not
used to relaunch or terminate anything.

The former loopback proxy 17890 was absent. Direct GitHub, the existing shared
7890 upstream, and the app connector failed. Additional port checks found an
existing functional loopback proxy at 17899. A separate bounded, loopback-only
TCP relay restores 17890 -> 17899 without changing/stopping the shared proxy or
SSH services, without a new subscription, and without logging proxy traffic.
The two pending source/document commits were then pushed and exact remote HEAD
was verified as `4d80520b618e101ca2e6e88b3a65169cdb3e0d72`.

Before execution, the recovery source must pass tests, be committed/pushed, and
match exact remote HEAD. It then verifies the unchanged original source hashes,
complete common features, original prospective plan and input identities.
It adopts only same-user live workers with exact script/arguments, assignment,
process start ticks and actual GPU UUID. Old PIDs alone cannot confer ownership.
It does not restart these workers, recompute already sealed cases or modify
model checkpoints, operating points, the 52-file model freeze or 840-execution /
2,040-logical-case matrix. Entire completed groups need not rebuild a bank.

An adopted orphan is not a child of the new parent. Its exit code is unavailable,
not fabricated as zero: completion requires its atomic group terminal receipt
and every input/hash-bound complete case. New children retain real return codes.
On failure, safety cleanup applies only to children created by the new parent;
adopted workers and foreign processes remain untouched. Original elapsed stage
time is retained rather than granting a fresh 48-hour allowance.

The scheduler retains the registered eight-worker maximum, 4 GiB per-worker
planning, joint evaluator headroom, 25% system RAM reserve, fresh UUID selection,
GPU/free-disk limits and stage storage ceiling. A detached original evaluator
can continue exact sealed-case posthoc metrics after its registered source is
verified. No new Val annotations, GT in inference, Test access, training, physical
MASA rerun, downloads, or method retuning are introduced.

The relay and parents must be launched independently of an interactive terminal
lifetime (new session, no stdin, bounded logs). This prevents terminal teardown
from itself sending a hangup; it does not guarantee survival of host failure,
external termination or loss of the upstream SSH proxy. Actual startup/terminal
receipts, not this prospective document, establish execution and completion.

## Actual startup verification

The source was pushed and exact remote HEAD verified as
`c662f76606e23ee2cd054b507db7c0da700dcece` after 362 scoped tests passed.
The independent recovery parent then verified all common feature markers and
the exact original plan SHA256
`eb0636415fe753823b24a248c590170917f660656e3357cdd22e4efa67079b54`.
Its private atomic startup receipt is
`recovery_ee443631621240218c40d0a1fdf85633_started.json`.
It adopted eight original workers; a subsequent start-tick/command identity
check confirmed all eight were still the same live processes, not replacements.
The original evaluator source was also launched as a separate detached parent.
Both parents were observed live, with progress age four seconds and no error.
At this observation, 113 cases were sealed and 77 had actual posthoc metrics.
The evaluator was waiting for joint planned RAM headroom, not declared failed
or complete. These are a point-in-time operational snapshot, not final scores
or the full 840-execution result. Completion still requires actual terminal
receipts, all case metrics, fixed paired comparisons and the final report.
