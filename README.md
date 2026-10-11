# Q-Cal-Research

Research on detector calibration under INT8 edge quantization and domain shift, built on an
integrity layer that makes every reported number traceable to a pre-registered, recorded run.

> **Status: Phase 1 scaffolding.** No results exist. The integrity layer (Phase 0) is in
> place. The clean-room FP32 reproduction pipeline ([docs/LAB.md](docs/LAB.md)) runs end to
> end on a synthetic fixture. It still waits on Ian's hand-written evaluation loop, the
> environment spike, the split decision and the oracle outputs; `make lab-status` lists
> them. The plan gates Phase 1 on G0. See
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
- **`qcal_lab`**: the Phase 1 science package ([docs/LAB.md](docs/LAB.md)).
  - COCO-format data, deterministic split manifests, a synthetic fixture.
  - Platt scaling, isotonic regression and the two-threshold class-wise procedure of
    Kuzucu et al. (arXiv:2405.20459).
  - Detectors behind `qcal.protocols.Detector`: the fixture, and an MMDetection adapter.
  - The experiment program that `qcal registry run` executes, `make smoke`, and an
    oracle-parity harness.
- **Agent layer**:
  - `CLAUDE.md` and `AGENTS.md`;
  - four subagents and five skills;
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
make smoke                 # Phase 1 loop on a synthetic fixture through the real registry
make lab-status            # what still blocks a registered Phase 1 run
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

`qcal config` prints the merged result, its sources and the environment variables that
shaped it. The set of keys is closed: `qcal config --check` fails on an unknown, removed or
mistyped key, so a misspelled policy key cannot fall back to its default unnoticed.

Environment overrides are a debugging aid. A registered run (`qcal registry run|run-batch`)
refuses them, except `QCAL__LOGGING__*`, and refuses a `QCAL_CONFIG` or `QCAL_ROOT` that
points away from the repository, a `qcal.toml` that differs from `HEAD`, and an
`--experiments` file other than the configured one. The experiment program never sees
`QCAL__*` variables, and it reports the digest of every configuration file it read; the run
fails if one changed while it ran. Git itself runs without redirecting `GIT_*` variables,
with replacement objects ignored and `core.fsmonitor` off.

## Tests

| Suite | Marker / target | What it covers |
|---|---|---|
| unit | `make test-unit` | each module in isolation, including property-based tests (Hypothesis) |
| integration | `make test-integration` | real git, SSH-signed commits, subprocess executors, the hook wrapper |
| regression | `make test-regression` | one test per fixed review, red-team or peer-review finding |
| security | `make test-security` | bypass attempts on the guards and the signed-commit control |
| e2e | `make test-e2e` | the documented workflow, `scripts/nightly.sh` and `make smoke` through the real CLI |
| parity | `make parity` | `qcal_lab` and Ian's metrics against `fiveai/detection_calibration` outputs; skips, saying why, until those exist |

The branch-coverage gate is set in `pyproject.toml` (`fail_under`).

On CI (`CI=true`), a test that needs `ssh-keygen` or `gpg` fails instead of skipping. Before
opening a pull request, run `make pre-pr`; it adds gitleaks and the container suite when
those tools are installed.

## How changes merge

1. Branch `claude/<slug>` (or `gemini/<slug>`), with a change proposal in `docs/changes/`.
2. `ci.yml` runs on the head: lint, types, all suites, integrity checks, `make smoke`,
   secret scan, container build.
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
