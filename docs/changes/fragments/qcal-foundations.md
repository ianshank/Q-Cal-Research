### qcal foundations, wave-2 follow-ups, PR-F pieces (`docs/changes/qcal-foundations.md`)

Coverage and test tooling:
- Coverage measures function bodies whose signatures contain `...` (the exclusion is anchored).
- `scripts/mutation_check.py` breaks each defence in `tests/mutation/defences.toml` in a copy
  of the repository and reports KILLED only when a named test fails.
- The prediction cache treats unreadable cached bytes as a miss, and never replaces them.
