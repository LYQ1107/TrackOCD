# TrackOCD core research log

## 2026-10-10 — Exact v2 recovery (M0)

- Question: did the new server restore the actual later v2, including atomic
  formal shards / cross-track DINO batching / prefix-level resume, or only the
  older published source?
- Evidence: targeted fetch of NAS recovery `604d7eff`; all 81 NAS blob hashes
  equal Git tree and A100 bytes (642,256 bytes). NAS worktree HEAD `24907fe1`
  and index/source before-after digests unchanged. 50 files added, nine
  updated, 22 identical; main, original v2 and recovery stashes preserved.
- Migration failures: first synthetic run 43 passed / two failed on old-root
  assumptions. Keep recovered source bytes unchanged; isolate only the test
  environment in a separate conftest. Regression: 48 passed.
- Findings: actual batching/prefix resume and fixed GT denominator are back.
  Legacy Known Hungarian remapping and contaminated token scoring remain;
  M3 must implement a separate evaluator, not rewrite past results.
- Decision: retain exact source recovery. No algorithm hypothesis tested.
  Next: M1 lawful frontend quality/causality and source-metadata qualification.
  No new training, feature extraction, TAO Test access or foreign process action.
