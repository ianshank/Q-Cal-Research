# Q-Cal-Research

Research on detector calibration under INT8 edge quantization and domain shift, built on an
integrity layer that makes every reported number traceable to a pre-registered, recorded run.

> **Status: Phase 0 (integrity tooling).** No science code or results exist yet. Phase 1
> (calibrators, LaECE metrics, quantization) starts after gate G0. See
> [docs/SDLC_IMPLEMENTATION_PLAN.md](docs/SDLC_IMPLEMENTATION_PLAN.md) and
> [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md).

## What is here

- **`qcal`**: a Python package and CLI.
  - A run registry: pre-registered cells, immutable records, a derived index, and tables
    generated from that index.
  - A claims checker: every number in the paper and in `CLAIMS.md` traces to a run.
  - Leakage, license and agent-layer audits.
  - Claude Code hook guards.
  - Git-object-only CI checks: signed commits, an append-only registry, cross-review.
- **Agent layer**:
  - `CLAUDE.md` and `AGENTS.md`;
  - two subagents and three skills;
  - fail-closed hooks.

  See [docs/AGENT_LAYER.md](docs/AGENT_LAYER.md).
- **Architecture**: C4 diagrams in [docs/architecture/](docs/architecture/README.md).
- **Change proposals**: [docs/changes/](docs/changes/). A proposal comes before
  non-trivial work.

## Quick start

```bash
make venv install          # .venv with the dev extras
make check                 # lint, mypy --strict, every test suite with the coverage gate, integrity
make help                  # every target
```

The CLI (exit codes: 0 pass, 1 a check or run failed, 2 usage or configuration error):

```bash
qcal config                              # merged configuration and where each layer came from
qcal registry cells --emit               # expand the design in EXPERIMENTS.yaml
qcal registry run-batch 'C-*' --dry-run  # what would run; only pre-registered cells and seeds
qcal registry index                      # runs/index.csv from the immutable records
qcal registry tables                     # LaTeX tables from the index
qcal registry audit                      # pre-registered coverage, duplicates, bad supersedes
qcal claims                              # every number in paper/ and CLAIMS.md traces to a run
qcal agent-layer                         # agents, skills, hooks, MCP, documented commands
qcal ci review-check --base origin/main --head HEAD --branch claude/my-change
```

`--debug` (or `QCAL_DEBUG=1`) prints tracebacks; `--log-format json` gives structured logs.

## Configuration

Every tunable value lives in configuration, never in code. The layers are:

1. packaged defaults in `src/qcal/resources/defaults.toml`, which document every key;
2. the repository's `qcal.toml` on top of them;
3. environment overrides on top of both: `QCAL__SECTION__KEY=<toml value>`, for example
   `QCAL__REGISTRY__MAX_RUNS_PER_BATCH=10`.

`qcal config` prints the merged result.

## Tests

| Suite | Marker / target | What it covers |
|---|---|---|
| unit | `make test-unit` | each module in isolation, including property-based tests (Hypothesis) |
| integration | `make test-integration` | real git, SSH-signed commits, subprocess executors, the hook wrapper |
| regression | `make test-regression` | one test per fixed review, red-team or peer-review finding |
| security | `make test-security` | bypass attempts on the guards and the signed-commit control |
| e2e | `make test-e2e` | the documented workflow and `scripts/nightly.sh` through the real CLI |

The branch-coverage gate is set in `pyproject.toml` (`fail_under`).

On CI (`CI=true`), a test that needs `ssh-keygen` or `gpg` fails instead of skipping. Before
opening a pull request, run `make pre-pr`; it adds gitleaks and the container suite when
those tools are installed.

## How changes merge

1. Branch `claude/<slug>` (or `gemini/<slug>`), with a change proposal in `docs/changes/`.
2. `ci.yml` runs on the head: lint, types, all suites, integrity checks, secret scan,
   container build.
3. `integrity.yml` runs the **base** branch's code on the head's git objects:
   - protected paths changed only in Ian's SSH-signed commits;
   - run records only added;
   - the other model's review at `review/<reviewer>/<branch-slug>.md` is approving and
     current.
4. Ian merges. Protected paths need his signed commit.

Signing and review checks are report-only (`bootstrap`) until Ian switches `signing.mode`
and `review.mode` to `enforce` in a signed commit.

## License

Apache-2.0 (see [LICENSE](LICENSE)). Data is public and license-checked only (clean-room).
