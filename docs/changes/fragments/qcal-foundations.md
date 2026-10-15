### qcal foundations, wave-2 follow-ups, PR-F pieces (`docs/changes/qcal-foundations.md`)

Coverage and test tooling:
- Coverage measures function bodies whose signatures contain `...` (the exclusion is anchored).
- `scripts/mutation_check.py` breaks each defence in `tests/mutation/defences.toml` in a copy
  of the repository and reports KILLED only when a named test fails.
- The prediction cache treats unreadable cached bytes as a miss, and never replaces them.

Phase 1 lab (PR-F pieces that need no decision):
- `python -m qcal_lab check-eval-loop` runs the EvalLoop contract (now
  `qcal_lab.eval_contract`) on Ian's loop, or on the fixture stand-in with `--fixture`.
- `python -m qcal_lab protocol-cases` writes seven synthetic, protocol-separating cases to
  `tests/parity/cases/` for Ian to run the oracle on; `--check` reports drift.
- A detector sanity-check scaffold: `[sanity.<detector>]` in `tooling.toml` (Ian's values,
  D2), reported by `status`; no AP is measured until D8.
