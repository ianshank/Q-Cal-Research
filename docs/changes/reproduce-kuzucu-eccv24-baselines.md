# change: reproduce-kuzucu-eccv24-baselines

## Gate served
G1 / K1 (Nov 8, 2026): clean-room FP32 reproduction of Kuzucu et al., *On Calibration of
Object Detectors: Pitfalls, Evaluation and Baselines* (ECCV 2024, arXiv:2405.20459), for at
least one detector (ATSS R50) within LaECE0 ±1.0 and AP ±0.5 of the published numbers.

**Gate status when this change was written (Oct 10, 2026).** G0 (Oct 23) is not recorded
in `DECISIONS.md`. The plan says Phase 1 science code waits for G0 and that Q-Cal code stays
private until counsel clears it (plan §1.6). This repository is public. This change therefore
holds only generic, published-method infrastructure. It contains:
- a COCO-format loader;
- deterministic split manifests;
- the Platt and isotonic calibrators and the class-wise two-threshold procedure from the
  paper;
- an experiment program that honours the registry's executor contract;
- a synthetic smoke test;
- an oracle-parity harness.

It contains no quantization code, nothing specific to the Q-Cal hypotheses, and no
results. Ian decides whether it merges before G0.

## Why
Phase 1 needs a pipeline that the pre-registered cells can run through `qcal registry run`:
- detector predictions on the four splits;
- a calibrator fit on `calibrator_fit_split` only;
- thresholds selected on `val` only;
- metrics on `test` from Ian's hand-written evaluation loop.

The pipeline also needs a `make smoke` that proves the chain end to end in under five
minutes on a CPU runner (plan §5, Appendix D.1).

## What changes
Agent-owned (`src/qcal_lab/`, `tests/`, `configs/`, `docs/`):
- `qcal_lab.config`: packaged defaults < `configs/lab.toml`. Science settings never come
  from the environment. The experiment program refuses a lab configuration that
  `registry.config_hash_inputs` would not hash into the run record.
- `qcal_lab.data`:
  - `coco`: a COCO-format loader using only the standard library;
  - `splits`: deterministic, order-independent split manifests whose digest equals
    `qcal leakage`'s, and which refuse to overwrite a different split;
  - `fixture`: a synthetic dataset that only the fixture detector and the fixture
    evaluation loop accept.
- `qcal_lab.predictions`: JSON Lines prediction files with deterministic bytes, plus a
  content-addressed cache.
- `qcal_lab.calib`, following arXiv:2405.20459:
  - identity;
  - Platt scaling (Eq. 8–9, soft IoU targets, a ≥ 0);
  - isotonic regression (PAVA, scikit-learn's documented conventions);
  - the two-threshold class-wise procedure of Alg. A.1/A.2, with identity for classes
    left empty (App. C.3);
  - threshold selection.
- `qcal_lab.models`:
  - a fixture detector;
  - an MMDetection adapter (lazy import, injected API). It is unverified against a live
    MMDetection until the environment spike.
- `qcal_lab.evaluation`: the `EvalLoop` protocol that Ian's
  `src/qcal_lab/handwritten/eval_loop.py` implements. It covers matching targets, the LRP
  threshold objective and the final metrics. The interface is a proposal; Ian may change it.
- `qcal_lab.experiment`, run as `python -m qcal_lab run`: the registry's experiment program.
  `qcal_lab.smoke`, run as `python -m qcal_lab smoke`: the whole loop in a temporary project
  through the real registry. `qcal_lab.parity` with `tests/parity/`: the oracle harness.
- The `paper-reproducer` and `prior-art-scout` agents and the `/reproduce-check` and
  `/prior-art` skills (plan §3.2, §3.3).

Enforcement surface (needs Ian's signed commit):
- `pyproject.toml`: `qcal_lab` in mypy and coverage, package data, the `parity` marker.
- `Makefile`: a real `smoke` target and a new `parity` target.
- `qcal.toml`: `executor.command` runs `python3 -I -m qcal_lab run ...`.
- `.github/workflows/ci.yml`: the `smoke` job (plan Appendix D.1).
- `.mcp.json`: the read-only Hugging Face server for `prior-art-scout`.
- `.claude/agents/{paper-reproducer,prior-art-scout}.md` and
  `.claude/skills/{reproduce-check,prior-art}/SKILL.md`.
- `CLAUDE.md`: `make smoke` in the command list.

Nothing under `src/qcal/` changes.

Deviations from the plan text:
- Hydra configs became TOML. One configuration system, and the registry already refuses
  free-form overrides, which are Hydra's main feature.
- `prior-art-scout` has no Write tool: the tool policy forbids write and network together.
  The caller saves its report.

## Adversarial review (adversarial-reviewer, verdict: block)

Seven blocking findings, each fixed with tests (`tests/unit/test_lab_*.py`,
`tests/security/test_lab_boundaries.py`):

- **B1. Leakage was checked by split name only.** Before any detector work, a run now
  requires `qcal leakage` PASS and refuses fit, select and evaluate splits that share an
  image id. `splits verify` fails on overlaps, missing manifests and duplicated ids.
- **B2. u_c was selected on the fit split.** Both thresholds are now selected on val only,
  and fit, select and evaluate must be three different splits (CLAUDE.md rule 3). The
  threshold-role settings are removed.
- **B3. Any module could act as the evaluation loop.** The loop's source file must now be
  `ian_only`, or be the fixture stand-in. The fixture is recognised by content digest, not
  by its description. The record stores the module, file and sha256.
- **B4. The prediction cache was incomplete and trusted.**
  - The key now also covers the image directory and bytes, and the package source.
  - Hits are checked against the header and image ids.
  - Entries record their producer, and the run record names it.
  - `runs/cache/**` is `registry_only`.
- **B5. paper-reproducer could write its own oracle.** The oracle outputs and the reference
  values are `ian_data`. A case's tolerance may only tighten the default. Its Bash is an
  `allow-only` list.
- **B6. Split shopping was one command away.** `--replace` is removed from the CLI, and the
  manifests are `ian_data`.
- **B7. The packaged defaults were not hashed.** `src/qcal_lab/resources/*.toml` is in
  `registry.config_hash_inputs`, and every run records the effective configuration's
  sha256.

Non-blocking fixes:
- real detectors no longer see ground-truth boxes;
- `fit_draw.seeded` is recorded;
- TF32 flags are recorded;
- retained detections per class are recorded;
- the threshold splits are recorded;
- a Platt line-search failure no longer counts as convergence;
- the Armijo and minimum-step constants, and the isotonic duplicate resolution, are now
  configuration;
- a per-class calibration with no calibrator for a class is an error;
- smoke refuses a non-empty project and re-runs uncached;
- `/reproduce-check` may call the leakage subagent.

Deferred (NEXT_STEPS): an Alg. A.1 parity case kind, a temperature-scaling baseline, and
verifying that the Hugging Face MCP tools are reachable under `prior-art-scout`'s tool list.

Further enforcement-surface edits:
- `qcal.toml`:
  - the `ian_data` category;
  - `runs/cache/**` added to `registry_only`;
  - `hooks.deny_categories` and `signing.signed_categories`;
  - `registry.config_hash_inputs`.
- `.claude/agents/paper-reproducer.md`: the Bash allow list.
- `.claude/skills/reproduce-check/SKILL.md`: the `Agent` tool.
- `AGENTS.md`: the `ian_data` rule.

## Out of scope
- Ian's hand-written files: `src/qcal_lab/handwritten/eval_loop.py` and `laece.py`
  (CLAUDE.md rule 4). Agents also do not write AP, LRP, D-ECE, LaECE0 or LaACE0.
- `EXPERIMENTS.yaml`, the split decision (the paper's published minival/minitest
  membership, or a fresh seeded partition), dataset allowlist entries in `DECISIONS.md`.
- Oracle outputs from `fiveai/detection_calibration` (CC BY-NC-SA 4.0). These are run
  unmodified on the P40, and the outputs go to `tests/parity/fixtures/`.
- The detector checkpoint, its hash and the MMDetection build (environment spike).
- Published reference numbers for `/reproduce-check` gate 2.
- Quantization (Phase 2).

## Acceptance
- [x] `make smoke` < 5 min (synthetic fixture through `qcal registry run-batch`)
- [ ] `qcal leakage` PASS (all four splits) once Ian's manifests exist
- [ ] parity tests green against oracle outputs (harness ready; outputs pending)
- [ ] protected-paths-signed green; Ian signs the enforcement-surface edits above
- [ ] RESEARCH_LOG.md entry (Ian)
- [ ] the other model's review at `review/gemini/<branch-slug>.md` passes `qcal ci review-check`

## Tasks
Scaffold (Claude Code, this change) → environment spike and detector checkpoint (Ian) →
hand-write `eval_loop.py` and `laece.py` (Ian) → oracle outputs on the P40 (Ian) →
parity tests (paper-reproducer) → `EXPERIMENTS.yaml` cells → runs (`qcal registry
run-batch`) → tables (`qcal registry tables`) → `/reproduce-check` → review
(adversarial-reviewer + Antigravity).
