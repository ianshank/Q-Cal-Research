# Architecture (C4)

The C4 model describes Q-Cal at four levels:

1. context: who and what the system serves;
2. containers: the deployable and storable parts;
3. components: inside the `qcal` package;
4. a dynamic view: how a pull request moves through the controls.

All diagrams are Mermaid, render on GitHub, and were checked with a Mermaid validator.

The design rule behind every box: **hooks are feedback, signed commits plus CI are the
control** (see `CLAUDE.md` and [`docs/AGENT_LAYER.md`](../AGENT_LAYER.md)).

## Level 1: system context

```mermaid
C4Context
  title Q-Cal research integrity system: context
  Person(ian, "Ian", "Researcher. Owns Ian-only documents and signs protected changes with an SSH key")
  Person_Ext(claude, "Claude Code", "Agent: writes Phase 1 science code, tests, scripts on claude/* branches")
  Person_Ext(gemini, "Antigravity (Gemini)", "Agent: notebooks, figures, cross-review on gemini/* branches")
  System(qcal, "Q-Cal repository + qcal tooling", "Run registry, claims checker, agent guards, CI integrity checks")
  System_Ext(github, "GitHub", "Pull requests, Actions (ci.yml, integrity.yml), branch rulesets, Dependabot")
  System_Ext(gpus, "Compute", "RTX 5060 Ti / 5060 (training, PTQ), P40 reference oracle, Jetson Orin Nano (INT8)")
  System_Ext(data, "Public datasets", "License-checked data only (clean-room); DVC-tracked, never committed")

  Rel(ian, qcal, "Edits Ian-only docs, signs commits, runs qcal init")
  Rel(claude, qcal, "Edits through hooks; launches runs via qcal registry")
  Rel(gemini, qcal, "Reviews; writes review/gemini/")
  Rel(qcal, github, "Pushes branches; PRs judged by CI")
  Rel(github, qcal, "integrity.yml runs base-branch qcal on PR head data")
  Rel(qcal, gpus, "Executor runs pre-registered cells")
  Rel(qcal, data, "Reads split manifests; leakage and license checks")
```

## Level 2: containers

```mermaid
C4Container
  title Q-Cal: containers
  Person(ian, "Ian", "Signs protected changes")
  Person_Ext(agent, "Coding agent", "Claude Code or Antigravity")

  System_Boundary(repo, "Q-Cal repository") {
    Container(agentlayer, "Agent layer", ".claude/, CLAUDE.md, AGENTS.md", "Subagents, skills, settings, hook wrapper")
    Container(hooks, "Hook guards", "python -m qcal.hooks (stdlib only)", "guard-paths, guard-bash, scope-write, deny-read, allow-only, claims")
    Container(cli, "qcal CLI", "Python package qcal.cli", "registry, claims, leakage, licenses, agent-layer, ci, policy, config, init")
    ContainerDb(registry, "Run registry", "runs/registry/*.json, runs/index.csv", "Immutable run records and the derived index")
    ContainerDb(docs, "Ian-only documents", "EXPERIMENTS.yaml, DECISIONS.md, CLAIMS.md, RESEARCH_LOG.md", "Pre-registration and decisions")
    Container(paper, "Paper and tables", "paper/, configs/tables/", "LaTeX with run-tagged numbers")
    Container(policy, "Policy", "qcal.toml over defaults.toml", "Categories, modes, every tunable value")
    Container(lab, "qcal_lab", "Python package, agent-owned", "Phase 1 science: data, splits, calibrators, detectors, experiment program, smoke")
    Container(handwritten, "Ian's evaluation loop", "src/qcal_lab/handwritten/ (Ian-only)", "Matching targets, LRP objective, reported metrics")
  }
  System_Boundary(gh, "GitHub Actions") {
    Container(ci, "ci.yml", "head code", "lint, mypy, suites, integrity, gitleaks, container")
    Container(integrity, "integrity.yml", "pull_request_target, base code", "signatures, registry-immutable, review-check")
  }

  Rel(agent, agentlayer, "Loads")
  Rel(agentlayer, hooks, "PreToolUse / Stop / SessionStart")
  Rel(hooks, policy, "Reads")
  Rel(agent, cli, "Runs")
  Rel(cli, registry, "Writes records, index")
  Rel(cli, paper, "Renders tables, verifies claims")
  Rel(cli, docs, "Reads pre-registration")
  Rel(ian, docs, "Writes, signs")
  Rel(cli, lab, "executor.command runs python -m qcal_lab run")
  Rel(lab, handwritten, "EvalLoop: targets, threshold_objective, metrics")
  Rel(ian, handwritten, "Writes by hand")
  Rel(ci, cli, "Runs on head")
  Rel(integrity, cli, "Runs base version on head objects")
```

| Container | Trust | Notes |
|---|---|---|
| Hook guards | feedback | Fail closed: the wrapper turns any crash into exit 2. Stdlib-only, so they run before the venv exists. |
| qcal CLI | tool | Exit codes: 0 pass, 1 a check or run failed, 2 usage or configuration error. |
| Run registry | append-only | One immutable JSON per run; `runs/index.csv` is derived and checked for staleness. |
| qcal_lab | agent-owned | Runs only behind the executor contract; the registry, not `qcal_lab`, writes the record. See [LAB.md](../LAB.md). |
| ci.yml | head code | Untrusted by design; it can only prove the head is self-consistent. |
| integrity.yml | base code | `pull_request_target`; the head is fetched as git data and never executed. |

## Level 3: components of the `qcal` package

```mermaid
C4Component
  title qcal package: components
  Container_Boundary(qcal, "qcal") {
    Component(config, "config + log", "qcal.config, qcal.log", "Layered TOML config, env overrides, text/JSON logging")
    Component(policy, "policy + globs", "qcal.policy, qcal.globs", "Path categories, clean-room substrings")
    Component(clipkg, "cli", "qcal.cli.app, common, registry, checks, ci, inspection", "COMMAND_GROUPS; uniform exit codes and report rendering")
    Component(hooks, "hooks", "qcal.hooks.guards, bash, cli", "Pure guard functions; push analysis; fail-closed dispatcher")
    Component(reg, "registry", "qcal.registry.*", "experiments, cells, runner, executor, store, index, audit, tables")
    Component(integ, "integrity", "qcal.integrity.*", "claims, aggregates, leakage, licenses, agent_layer, agent_settings, command_refs")
    Component(cichecks, "ci", "qcal.ci.signatures, immutability, review", "Git-object-only checks for pull_request_target")
    Component(reports, "reports + components", "qcal.reports, qcal.components", "CheckReport protocol; ComponentRegistry for DI")
  }
  Rel(clipkg, reg, "registry commands")
  Rel(clipkg, integ, "claims, leakage, licenses, agent-layer")
  Rel(clipkg, cichecks, "ci commands")
  Rel(hooks, policy, "evaluate paths")
  Rel(cichecks, policy, "policy from base ref")
  Rel(integ, reg, "index rows")
  Rel(integ, clipkg, "command drift parses the real parser")
  Rel(reg, config, "every tunable value")
  Rel(clipkg, reports, "emit_report")
```

Extension points, all without editing the dispatchers:

| To add | Do this |
|---|---|
| A CLI command group | write `add_commands(sub)` in a new `qcal/cli/<group>.py` and append it to `COMMAND_GROUPS` |
| An executor | `@EXECUTORS.register("name")`; select with `--executor name` or `executor.kind` |
| An environment collector | `@COLLECTORS.register("name")`; list it in `registry.env_collectors` |
| An aggregate | register it in `qcal.integrity.aggregates`; table specs and claims refs pick it up |
| A check report | implement `passed`, `to_dict()` and `render_text()` (`qcal.reports.CheckReport`) and return `emit_report(...)` |
| A policy category | add `[policy.categories] name = [...]` in `qcal.toml` (signed) |

## Level 3: components of the `qcal_lab` package

```mermaid
C4Component
  title qcal_lab package: components
  Container_Boundary(lab, "qcal_lab") {
    Component(labcfg, "config", "qcal_lab.config", "defaults < configs/lab.toml; must be hashed by the registry")
    Component(data, "data", "qcal_lab.data.coco, splits, fixture", "COCO ground truth, seeded manifests, synthetic fixture")
    Component(preds, "predictions", "qcal_lab.predictions", "Deterministic JSON Lines; content-addressed cache")
    Component(models, "models", "qcal_lab.models (DETECTORS)", "fixture detector, MMDetection adapter")
    Component(calib, "calib", "qcal_lab.calib (CALIBRATORS), two_threshold, thresholds", "identity, Platt, isotonic; Alg. A.1/A.2")
    Component(evalseam, "evaluation", "qcal_lab.evaluation", "EvalLoop protocol; loads Ian's module")
    Component(exp, "experiment", "qcal_lab.experiment, cli", "Executor-contract program; plan and split-role checks")
    Component(smoke, "smoke, status, parity", "qcal_lab.smoke, status, parity", "Fixture loop through the registry; readiness; oracle harness")
  }
  Rel(exp, data, "manifests, ground truth")
  Rel(exp, models, "raw predictions")
  Rel(exp, preds, "write, cache")
  Rel(exp, calib, "train on fit/select, apply on evaluate")
  Rel(calib, evalseam, "targets, threshold objective")
  Rel(exp, evalseam, "metrics on evaluate only")
  Rel(smoke, exp, "via qcal registry run-batch")
  Rel(exp, labcfg, "every science setting")
```

## Dynamic view: a pull request

```mermaid
sequenceDiagram
  autonumber
  actor Agent as Claude Code
  participant Hooks as Hook guards
  participant CLI as qcal CLI
  participant GH as GitHub
  participant CI as ci.yml (head)
  participant INT as integrity.yml (base code)
  actor Rev as Other model
  actor Ian
  Agent->>Hooks: Edit / Write / Bash
  Hooks-->>Agent: allow, or block with exit 2
  Agent->>CLI: qcal registry run-batch (pre-registered cells only)
  CLI-->>Agent: immutable run records, index, tables
  Agent->>CLI: make pre-pr
  Agent->>GH: push claude/slug, open PR
  GH->>CI: lint, types, suites, integrity, gitleaks, container
  GH->>INT: verify-signatures, registry-immutable, review-check
  Rev->>GH: review/gemini/claude-slug.md (approve, reviewed_sha)
  Ian->>GH: signed commit for protected paths, merge
```

## Decisions recorded here

- **Deterministic scripts over agents.** Launching runs, auditing coverage, building tables,
  checking licenses and validating the agent layer are `qcal` commands with tests, not prompts.
- **Policy from the base ref.** Every CI check that judges a pull request reads `qcal.toml`
  and `allowed_signers` from the base commit, so a pull request cannot relax its own rules.
- **Net content from the fork point.** The signature check diffs `merge-base(base, head)..head`, so a
  branch that is merely behind base passes. A merge that drops or rolls back protected
  content still fails.
- **Science code behind the executor contract.** `qcal_lab` never writes the registry. A
  result exists only as a record that `qcal registry run` wrote from the program's result
  file. The fixture stand-ins refuse real datasets, so the smoke test cannot fabricate a
  number.
- **One configuration system.** The plan's Hydra configs became TOML layered the same way as
  `qcal.toml`. The registry refuses free-form overrides, which are Hydra's main feature.
- **Report-only bootstrap.** `signing.mode` and `review.mode` start as `bootstrap`; Ian
  switches each to `enforce` in a signed commit once keys and reviewers are in place.
