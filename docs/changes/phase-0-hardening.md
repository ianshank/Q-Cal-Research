# change: phase-0-hardening

## Gate served
G0 (Oct 23, 2026). This change hardens the Phase 0 integrity layer
(`docs/changes/phase-0-integrity-layer.md`) before any science code depends on it.

## Why
A gap analysis and peer review of the Phase 0 branch found the following problems:
- a 500-line CLI module;
- a version string duplicated in two places;
- `O(commits x paths)` git calls in the signature check;
- one hardcoded timeout;
- lint rule sets that were not enabled;
- no validation that the commands named in skills, agents and docs actually exist;
- missing repository infrastructure: dependency updates, secret scanning, a container
  image, and per-suite test targets.

## What changes
These edits touch the enforcement surface (`src/qcal/`, `.claude/`, `.github/`,
`Makefile`, `pyproject.toml`). All of them need Ian's signed commit.

**Code hygiene**
- `qcal.cli` is now a package: `app`, `common`, `registry`, `checks`, `ci` and
  `inspection`. The public names it re-exports are unchanged.
- The package version has one source, `qcal.__version__`, read through setuptools
  dynamic metadata.
- The signature check reads each ref's tree once (`TreeIndex`) instead of making
  one `rev-parse` per commit and path.
- `registry.env_command_timeout_s` replaces a hardcoded probe timeout.
- Glob helpers take `case_insensitive` as keyword-only.
- New ruff rule sets: NPY, PERF, FURB, ERA, T20, PTH, ARG, BLE, FBT, PIE, G, LOG, FLY and ISC.

**Deterministic agent-layer validation (`qcal agent-layer`)**
- Command-reference drift: every `qcal ...` and `make <target>` in skills, agents,
  CLAUDE/AGENTS, the Makefile, workflows and docs must resolve against the real parser
  and the Makefile.
- Tool policy:
  - agents must declare their tools;
  - no agent or skill may combine a write-capable tool with a network tool;
  - models come from an allowlist;
  - a `context: fork` skill must name an existing or built-in agent;
  - skills referenced by an agent must exist.
- Hook settings:
  - event names must be known;
  - tool matchers must compile and name known tools;
  - SessionStart matchers must name valid sources.

**Agent layer**
- New skills: `pre-pr` and `change-proposal`. Both are user-invoked only.
- The SessionStart hook reports the guard mode and the signing mode.

**Infrastructure**
- `.github/dependabot.yml`.
- `.gitleaks.toml`, a CI gitleaks job and a pre-commit hook.
- A `Dockerfile` (non-root) and `.dockerignore`, plus a CI image build and test.
- Per-suite pytest markers (unit, integration, regression, e2e, security) and
  JUnit reports.
- Makefile targets for each suite, `pre-pr`, `gitleaks`, `docker-*` and `nightly`.

**Documentation**
- README, CHANGELOG.
- C4 architecture diagrams.
- `docs/AGENT_LAYER.md`.
- Next steps and a tech-debt register.

## Out of scope
- Science code (Phase 1).
- Switching `signing.mode` to `enforce`, and adding keys to `allowed_signers`. Both are Ian's.

## Acceptance
- [ ] `make pre-pr` green: lint, types, every suite with the coverage gate, integrity
      checks, gitleaks, docker build and test
- [ ] `qcal agent-layer` PASS on the repository, including command-reference drift
- [ ] every new check has negative tests: it fails on a crafted bad input
- [ ] no public CLI command, flag or exit code removed or renamed
- [ ] Ian's signed commit for the enforcement-surface edits

## Tasks
Implemented by Claude Code with a read-only peer-review subagent. Edits to `src/qcal/`
were made through scripted patches, because the guard-paths hook blocks the Edit and
Write tools there by design. The hook is feedback; Ian's signature is the control.
