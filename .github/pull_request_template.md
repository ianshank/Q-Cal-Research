## What and why

## Gate and change proposal
Gate: G?
Proposal: docs/changes/<slug>.md

## Checklist
- [ ] `make check` is green (lint, strict types, tests with the coverage gate)
- [ ] `qcal leakage` PASS if data or splits changed
- [ ] No enforcement-surface edits, or they are in Ian's signed commits
- [ ] RESEARCH_LOG.md entry for this work
- [ ] CLAIMS.md updated if any number changed
- [ ] Other model's review at review/<reviewer>/<branch-slug>.md (`qcal ci review-check`): approve, reviewed_sha in this PR, only review/ changed since

## Provenance
Authored-By-Human: <files Ian wrote by hand, or "none">
Assisted-By: <agents or models involved, or "none">
