---
name: pre-pr
description: Run every deterministic pre-PR validation (lint, strict types, all test suites with the coverage gate, integrity checks, secrets, container) and report the results without fixing anything.
argument-hint: "[--quick]"
disable-model-invocation: true
allowed-tools: Bash(make pre-pr), Bash(make validate), Bash(git status:*), Bash(git diff:*), Read
---
Arguments: $ARGUMENTS.

Run the checks; do not interpret around them. With `--quick`, run `make validate` (lint, types
and integrity checks, no tests). Otherwise run `make pre-pr`, which runs `make check` (lint,
`mypy --strict`, the full suite with the coverage gate, and `qcal agent-layer`, `qcal claims`,
`qcal registry index --check`, `qcal registry tables --check`, `qcal leakage --if-present`,
`qcal licenses`), then gitleaks and the container suite when those tools are installed.

Report, in this order:
1. One line per check: PASS, FAIL or SKIPPED, with the exact command.
2. For each FAIL, the first error verbatim (file:line where given) and the command that
   reproduces it alone, for example `make test-security` or `qcal agent-layer`.
3. `git status --short`: files that are modified but not committed.
4. Whether the diff touches the enforcement surface (`qcal policy check <paths>`), which
   needs a change proposal in docs/changes/ and Ian's signed commit.

Never edit files, change configuration, skip, disable or mark tests as expected failures to
get a green run. A SKIPPED check is reported as skipped, never as passed.
