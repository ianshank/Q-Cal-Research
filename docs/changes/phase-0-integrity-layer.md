# change: phase-0-integrity-layer

## Gate served
G0 (Oct 23, 2026): the tooling Phase 0 of docs/SDLC_IMPLEMENTATION_PLAN.md requires before
any science code is written.

## Why
Plan v2 §3.8 and §9: a six-hour minimal viable agent layer where signed commits and CI are the
control and hooks are feedback, plus the deterministic `qcal` commands that replace four v1 agents.

## What changes
- `qcal` package: layered config, logging, glob policy, hooks, run registry (records, store,
  experiments, design expansion, environment, executor, runner, index, audit, tables), claims,
  leakage, licenses, agent-layer validation, CI signature and immutability checks, CLI.
- Agent layer: CLAUDE.md, AGENTS.md, two subagents, the weekly-review skill, hooks, settings.
- CI: `ci.yml` (lint, types, tests, integrity checks on head) and `integrity.yml`
  (`pull_request_target`, base-branch code and policy judging the head's commits).
- Templates for Ian's documents, materialized by `qcal init`.

## Out of scope
Detection-calibration science code (calibrators, LaECE metrics, quantization, model wrappers):
Phase 1, after G0 counsel clearance. Ian-only files are templates only; Ian creates them.

## Acceptance
- [x] lint, strict types, full test suite with coverage gate
- [x] hook fixtures from plan §10.3 pass through the real wrapper
- [x] signature check verified with real SSH-signed commits in tests
- [ ] Ian adds a key to allowed_signers and switches signing.mode to enforce (signed)
- [ ] branch ruleset requires the `integrity` and `ci` checks

## Tasks
Implemented by Claude Code; reviewed by independent subagents (red team on guards and CI,
code review, security review).
