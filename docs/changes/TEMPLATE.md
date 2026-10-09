# change: <slug>

## Gate served

## Why

## What changes

## Out of scope

## Acceptance
- [ ] `make check` green (lint, types, tests with coverage gate)
- [ ] `qcal leakage` PASS (all four splits) once manifests exist
- [ ] parity tests green where a reference exists
- [ ] no enforcement-surface edits without Ian's signed commit
- [ ] RESEARCH_LOG.md entry; CLAIMS.md updated if numbers changed
- [ ] cross-review file with reviewed_sha equal to the head and verdict approve

## Tasks
Scaffold (Claude Code) → hand-write (Ian) → parity tests → runs (`qcal registry run-batch`)
→ tables (`qcal registry tables`) → review (adversarial-reviewer + Antigravity)
