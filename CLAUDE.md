# CLAUDE.md: Q-Cal

Shared rules for every agent are in AGENTS.md (Claude Code reads it natively; it
is not imported here so it does not load twice). Plan: docs/SDLC_IMPLEMENTATION_PLAN.md.

## Integrity (non-negotiable)
0. Hooks are feedback, not permission. Ian's signed commits plus CI are the control.
   Never edit the enforcement surface (`.claude/`, `.github/`, `Makefile`, `pyproject.toml`,
   `qcal.toml`, `src/qcal/` infrastructure) except through a change proposal in `docs/changes/`.
1. Never fabricate or estimate results. Every number in `paper/`, `README.md` and `CLAIMS.md`
   comes from the run index through `qcal registry tables`, as `\qcalval{run:<id>:<metric>}{value}`
   in LaTeX or `value (run:<id>:<metric>)` in Markdown. `qcal claims` verifies them.
2. `EXPERIMENTS.yaml` is pre-registered: never edit it. Propose changes in `AMENDMENTS.md`.
3. Never tune on test splits. Calibrators fit on `calibrator_fit_split` only; selection on val only.
4. Never write `*/handwritten/*`. You may read and test it and report deviations.
5. Never author claims, abstract, interpretation, limitations or conclusions.
6. Clean-room: public, license-checked data only; nothing from the employer. Stop and flag anything
   that resembles employer material.
7. Runs are launched only through `qcal registry run|run-batch`, which records seed, git SHA,
   config hash and environment. Never write `runs/registry/` or `runs/index.csv` by hand.

## Conventions
- No hardcoded values: configuration lives in `qcal.toml` over packaged defaults
  (`src/qcal/resources/defaults.toml`); override with `QCAL__SECTION__KEY=value`.
- Protocol-based DI (`src/qcal/protocols.py`, `qcal.components.ComponentRegistry`).
- Deterministic scripts over agents: launching, auditing, licensing and tables are `qcal` commands.
- Tests for every module; ruff + mypy --strict clean; coverage gate in `pyproject.toml`.
- Logging via `qcal.log.get_logger`; `--debug` or `QCAL_DEBUG=1` for tracebacks.
- Change proposals in `docs/changes/` before non-trivial work; branch `claude/<slug>`.

## Commands
```bash
make check          # lint + types + tests (what CI runs)
make test           # pytest with coverage gate
qcal agent-layer    # validate agents, skills, hooks, .mcp.json
qcal registry audit # pre-registered coverage
qcal claims         # every number traces to a run
```

## Compute
GPU0 = RTX 5060 Ti 16 GB (training/QAT); GPU1 = RTX 5060 8 GB (FP32/FP16 passes, x86 PTQ).
Pin with CUDA_VISIBLE_DEVICES; never DDP across the two. P40 runs the reference oracle in its
own CUDA 11.8 environment. Jetson Orin Nano Super produces headline INT8 numbers (no DLA).

## Workflow
Branch `claude/<slug>` from the default branch; one worktree per agent. A PR merges with green
CI, a signed-commit check for protected paths, a RESEARCH_LOG.md entry, CLAIMS.md updated when
numbers change, and the other model's review in `review/<branch>.md`.
