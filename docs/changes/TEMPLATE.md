# change: <slug>

## Status
Proposed | Accepted | Implemented | Superseded by `<slug>`, with the date.
Say whether it edits the enforcement surface (then it needs Ian's signed commit).

## Gate served
G0 | G1 | … and the date. One line on why this gate needs it.

## Why
The problem, with file and line references verified against a named commit.

## What changes
Enforcement surface (signed), then agent-owned, file by file.

## Decisions
For each decision: what was chosen, the alternatives considered, and the consequences
(what becomes easier, what becomes harder, what it rules out).

## Compatibility
- Formats: hashed files, envelopes, predictions headers, record fields (schema 1 only gains
  fields).
- Config keys: added, removed (with the replacement message), retyped.
- CLI: new or changed commands, flags, exit codes and output.
- Records: what existing records mean after the change; whether any must be re-run.

## Out of scope

## Ian decisions requested
Numbered, each with the date it is needed by.

## Ian hours
Reading, deciding and signing, in hours.

## Acceptance
- [ ] `make check` green (lint, types, tests with coverage gate)
- [ ] `qcal agent-layer` PASS
- [ ] `qcal leakage` PASS (all four splits) once manifests exist
- [ ] parity tests green where a reference exists
- [ ] no enforcement-surface edits without Ian's signed commit
- [ ] advisory adversarial review saved under `review/claude/<slug>.md` and its blocking
      findings fixed or answered here
- [ ] RESEARCH_LOG.md entry; CLAIMS.md updated if numbers changed
- [ ] the other model's review at review/<reviewer>/<branch-slug>.md passes `qcal ci review-check`
      (approve; reviewed_sha a commit of the PR; only review/ changed since)

## Tests
Unit, integration, security and regression files, with what each proves.

## Tasks
Scaffold (Claude Code) → hand-write (Ian) → parity tests → runs (`qcal registry run-batch`)
→ tables (`qcal registry tables`) → review (adversarial-reviewer + Antigravity)
