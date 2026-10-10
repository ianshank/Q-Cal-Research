# Agent layer: agents, skills, hooks, harness and how each is validated

## The rule

Hooks are **feedback**: they stop an agent before a mistake.

The **control** is what CI checks on every pull request:

- **Ian's SSH-signed commits** for protected paths (`qcal ci verify-signatures`);
- **an append-only run registry** (`qcal ci registry-immutable`);
- **the other model's current review** (`qcal ci review-check`).

Whatever an agent does in its shell, these checks run on the base branch's code and decide
what merges. Everything below is designed so that a hook failure blocks rather than allows,
and so that every piece is validated deterministically by `qcal agent-layer`, not by a model.

## Inventory

### Subagents (`.claude/agents/`)

| Agent | Tools | Model | Purpose |
|---|---|---|---|
| `adversarial-reviewer` | Read, Grep, Glob | opus | Hostile, read-only review in the `review/TEMPLATE.md` schema. The caller saves it to `review/claude/<branch-slug>.md`. |
| `data-leakage-checker` | Read, Grep, Glob, Bash | sonnet | Runs `qcal leakage --json`, then greps for test-keyed selection. A frontmatter `PreToolUse` hook (`allow-only`) restricts Bash to that one command. |
| `paper-reproducer` | Read, Grep, Glob, Edit, Write, Bash | sonnet | Phase 1. Implements published baselines in `src/qcal_lab`, citing equations, and writes oracle parity tests. It runs in its own worktree (`maxTurns: 40`). A `scope-write` hook limits writes to `src/qcal_lab`, `tests` and `configs`; the session guard still refuses `*/handwritten/*` and `ian_data` (oracle outputs, manifests, reference values). An `allow-only` hook limits Bash to make targets and read-only git. No network tool. |
| `prior-art-scout` | Read, Grep, Glob, WebSearch; MCP `huggingface` | sonnet | Phase 1. Novelty re-check. It has no write tool; the caller saves its report to `review/prior-art/<date>.md`. A `deny-read ian_only` hook keeps Ian's private documents out of any prompt-injection path. The Hugging Face server is anonymous. |

### Skills (`.claude/skills/`)

| Skill | Invoked by | Purpose |
|---|---|---|
| `/weekly-review <week>` | user or model | Gates, hours, WIP and amendments from Ian's documents. Signatures come from `qcal ci verify-signatures`, so the answer does not depend on host git config. |
| `/pre-pr [--quick]` | user only (`disable-model-invocation`) | Runs `make pre-pr` (or `make validate`) and reports PASS, FAIL or SKIPPED per check. It never edits anything to get green. |
| `/change-proposal <slug> <goal>` | user only | Drafts `docs/changes/<slug>.md` from the template, with enforcement-surface paths found by `qcal policy check`. |
| `/reproduce-check` | user or model | Gate G1/K1 report: `make smoke`, `make lab-status`, the leakage checker, `make parity`, the audit. A skipped parity case is NOT ASSESSED. It never edits anything. |
| `/prior-art <claim>` | user or model | Runs in a fork as `prior-art-scout`; returns a report for `review/prior-art/`. |

### Hooks (`.claude/settings.json`)

| Event | Matcher | Command | Failure mode |
|---|---|---|---|
| PreToolUse | `Edit\|Write\|MultiEdit\|NotebookEdit` | `run_hook.sh guard-paths` | closed (exit 2) |
| PreToolUse | `Bash` | `run_hook.sh guard-bash` | closed |
| Stop | (all) | `run_hook.sh --fail-open claims` | open: CI runs the same check as the gate |
| SessionStart | `startup\|resume` | `session_start.sh` | never fails; prints the guard, signing and review modes |

Implemented but not yet wired by default:

- `scope-write <dirs...>`: allow writes only under the given directories;
- `deny-read <globs or categories...>`: keep files private from a subagent.

Attach them in a subagent's frontmatter `hooks:` when an agent needs that scope.

### Permissions

The allow list covers the deterministic commands (`make check`, `qcal`, `ruff`, `mypy`,
`pytest`, read-only git).

The deny list covers:

- force pushes;
- `rm -rf runs`;
- `dvc remove`;
- the init command with `--force` (Ian runs that himself).

`guard-bash` also denies the last one, through the configurable `hooks.extra_bash_deny`.

## The harness

`run_hook.sh` is the only thing Claude Code runs:

- **Fail closed.** Claude Code treats only exit 2 as "block". The wrapper turns every
  other non-zero exit, a missing interpreter, or a crash into 2. `--fail-open` turns
  them into 0 instead, for the Stop hook only.
- **No shadowing.** It runs `python -P -s -m qcal.hooks` with `PYTHONPATH=<project>/src`
  first, so neither the working directory nor user site-packages can supply a fake
  `qcal` package.
- **Argument errors follow the fail mode.** The wrapper exports
  `QCAL_HOOK_FAIL_OPEN=1|0`; a hook that cannot parse its arguments exits accordingly.
- **Modes.** `QCAL_GUARD_MODE=enforce|warn|off` comes from the environment Claude Code
  was launched with. `QCAL_HOOK_LOG=<file>` writes a JSON-lines decision log.

What the guards decide, and their known limits:

- `guard-paths` judges a path against the verified git work tree that holds it: the
  project, or a linked worktree whose `.git` file and `.git/worktrees/<name>/gitdir`
  point at each other. A hand-made `.git` marker is ignored. Edit tools may not write git
  internals (`.git/**`, any `.git` file). A symlink is also judged at its target's tree.
- `guard-bash` is push-only by design (plan §3.4). It refuses:
  - force flags (including unambiguous abbreviations);
  - pushes to protected branches in any spelling: `HEAD`, `@`, `heads/`, `refs/heads/`;
  - implicit pushes whose branch it cannot resolve: `cd` or `-C` targets that use
    variables, or `--git-dir`;
  - command substitution in a push command;
  - variable or wildcard refspecs;
  - pushes fed by `xargs`;
  - `-c remote|push|branch|url|alias.*`;
  - shell aliases. Persistent aliases are expanded through `git config`.

  It does not try to stop file writes from the shell; the signed-commit check covers
  those. It recognizes known spellings, not every possible one. The control is the branch
  ruleset, which forbids direct pushes to the default branch.

## Deterministic validation: `qcal agent-layer`

It runs in `make check`, in CI (`ci.yml` integrity job), in pre-commit, and in
`scripts/nightly.sh`. All the lists it checks against live in `[agent_layer]` configuration.

| Check | Fails on |
|---|---|
| Frontmatter | unknown keys (Claude Code ignores them silently), missing required keys, name not matching the file or directory |
| Tool policy | an agent without explicit `tools`; any agent or skill holding a write-capable tool (Edit, Write, MultiEdit, NotebookEdit, Bash) together with a network tool (WebFetch, WebSearch, `mcp__*`) |
| Models | a model outside `allowed_models` or `model_id_pattern` |
| References | `context: fork` naming a missing agent; an agent listing a missing skill; `agent:` without `context: fork` |
| Typed keys | `disable-model-invocation`, `user-invocable`, `background` must be booleans |
| Hooks (settings and agent frontmatter) | unknown events; matchers that do not compile or name unknown tools or SessionStart sources; scripts not located via `$CLAUDE_PROJECT_DIR`, missing or not executable; wrapper commands naming a hook that does not exist; a `guard-paths` matcher missing an edit tool, or a `guard-bash` matcher without `Bash` |
| MCP | remote servers without `type`; literal bearer tokens |
| Command drift | any `qcal ...` or `make <target>` in code in CLAUDE.md, AGENTS.md, README, skills, agents, hooks, workflows, docs and scripts that does not resolve against the real argument parser (subcommands and option names) or the Makefile |

Tests:

- `tests/unit/test_agent_layer.py`, `tests/unit/test_agent_policy.py` and
  `tests/unit/test_command_refs.py` cover each rule with passing and failing inputs;
- `tests/integration/test_repository_layout.py` runs the checks on this repository;
- `tests/integration/test_hook_wrapper.py` and `tests/security/` drive the real wrapper with
  crafted payloads.

## Roadmap: opportunities identified in the gap analysis

These are proposals, not implemented. Each needs a change proposal and, where it touches
`.claude/`, Ian's signed commit.

**Skills from reusable actions**

- `/run-cell <pattern>`: the deterministic chain `qcal registry run-batch`,
  `qcal registry index`, `qcal registry tables`, `qcal registry audit --json`,
  `qcal claims`, reported as one table. It runs only pre-registered cells.
- `/review-pr`: invoke `adversarial-reviewer` with `context: fork`, then save its output
  to `review/claude/<branch-slug>.md` and run `qcal ci review-check --mode bootstrap` to
  show what CI will say.
- `/amendment <cell or hypothesis>`: draft an `AMENDMENTS.md` entry (never
  `EXPERIMENTS.yaml`) with the evidence that motivates it.

**Agents**

- `parity-tester` and `literature-scout`: done in Phase 1 as `paper-reproducer` (scoped
  writes) and `prior-art-scout` (read-only).
- `report-writer` and `hf-publisher`: Phase 5 (plan §3.2).

**Hooks and loops**

- PostToolUse on `Edit|Write` under `paper/**`, `configs/tables/**` and `README.md`: run
  `qcal claims` for immediate feedback. This needs a per-file option on `claims`.
- Stop: also run `qcal registry index --check` and `qcal registry tables --check` (fail
  open; CI is the gate).
- SubagentStop for `adversarial-reviewer`: validate that the output parses as the review
  frontmatter schema before the caller saves it.
- Loops:
  - schedule `scripts/nightly.sh` (cron, or a scheduled workflow that uploads the JSON-lines
    report);
  - schedule `/weekly-review` as a Routine every Monday.
