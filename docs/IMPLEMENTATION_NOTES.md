# qcal: Phase 0 implementation notes

This package implements Phase 0 of [the SDLC implementation plan](SDLC_IMPLEMENTATION_PLAN.md):
the integrity layer that must exist before any Q-Cal science code is written. It is
domain-agnostic research tooling. Detection-calibration code (calibrators, LaECE metrics,
quantization, model wrappers) is Phase 1 and is gated on G0 (Oct 23, 2026), see below.

## What exists

| Area | Module | Replaces / implements |
|---|---|---|
| Layered configuration | `qcal.config`, `resources/defaults.toml`, repo `qcal.toml` | "no hardcoded values" |
| Logging | `qcal.log` | text or JSON logs, `--debug`, `QCAL_DEBUG` |
| Path policy | `qcal.globs`, `qcal.policy` | ownership map (plan §2) |
| Agent guards | `qcal.hooks`, `.claude/hooks/run_hook.sh` | `guard_paths.sh`, push-only `guard_bash.sh`, scoped hooks |
| Run registry | `qcal.registry.*` | v1 agents `experiment-runner`, `ablation-auditor`, `results-table-builder` |
| Claims check | `qcal.integrity.claims` | `check_claims.py` (plan §1.5) |
| Leakage check | `qcal.integrity.leakage` | deterministic half of `data-leakage-checker` |
| License audit | `qcal.integrity.licenses` | v1 agent `license-auditor` |
| Agent-layer validation | `qcal.integrity.agent_layer` | frontmatter test (plan §3.7) |
| CI integrity | `qcal.ci.signatures`, `qcal.ci.immutability` | signed-commit check, append-only registry |
| Templates | `qcal init`, `resources/templates/` | Ian's documents, never authored by agents |
| Interfaces | `qcal.protocols` | Detector, Calibrator, Metric, Quantizer shapes for Phase 1 |

Agent layer: `CLAUDE.md`, `AGENTS.md`, two subagents (`adversarial-reviewer`,
`data-leakage-checker`), one skill (`/weekly-review`), four hooks, `.mcp.json` with GitHub in
read-only mode. CI: `.github/workflows/ci.yml` and `.github/workflows/integrity.yml`.

## Enforcement model

Hooks are feedback; signed commits plus CI are the control (plan v2 §1.2).

1. **Signed commits.** `qcal ci verify-signatures` lists every commit in `base..head`, finds the
   ones touching `ian_only` or `enforcement_surface` paths, and requires a good SSH signature
   from a key in `allowed_signers`. Merge commits are judged only on files that differ from
   every parent, so merging the base branch does not demand a new signature.
2. **Base-branch judge.** `integrity.yml` runs on `pull_request_target`: the workflow, the
   `qcal` code, `qcal.toml` and `allowed_signers` all come from the base commit. The PR head
   is fetched as git data and never executed, so a PR cannot relax the rules that judge it.
3. **Immutable records.** `qcal ci registry-immutable` allows only added files under
   `runs/registry/`, each a valid record named after its run id.
4. **Head-side consistency.** `ci.yml` regenerates the index and tables and fails if the
   committed copies differ, and runs the claims, leakage, license and agent-layer checks.
5. **Hooks.** `guard-paths` blocks edit tools on protected and clean-room paths (after
   resolving symlinks); `guard-bash` blocks force-pushes and pushes to protected branches;
   the Stop hook runs the claims check. The wrapper converts every crash into exit 2, because
   Claude Code treats any other non-zero exit as "allow". The Stop hook fails open and does
   not block twice in a row, so it cannot trap a session.
6. **Human switch.** `QCAL_GUARD_MODE=enforce|warn|off` comes from the environment Claude Code
   was launched with, which an agent cannot change.

## Refinements to the plan, and why

| Plan text | Implemented as | Reason |
|---|---|---|
| `\num{run:ID}{value}` | `\qcalval{run:<id>:<metric>}{value}` and `agg:<fn>:<metric>:<id>+<id>` refs | `\num` collides with siunitx; a run has many metrics; table cells aggregate several seeds |
| `index.csv` has no deleted lines | records are immutable; the index is checked by regeneration | adding a metric column rewrites every CSV row; immutability belongs on the source records |
| Signature CI job in `ci.yml` | separate `integrity.yml` on `pull_request_target` | a `pull_request` workflow runs the head's copy of itself, which the PR could weaken |
| `qcal-registry` subcommands | `qcal registry ...`, with `qcal-registry` kept as an alias | one entry point for all checks |
| Bash hooks with `jq` | Python hooks (stdlib only) behind a bash wrapper | no `jq` dependency, unit-testable, and a crash cannot fail open |
| `CODEOWNERS` | not added | GitHub forbids self-approval, so it cannot gate a solo repository (plan §1.2) |
| `X-MCP-Readonly` header | used, unverified | verify with a label write before relying on it; otherwise run the local server with `--read-only` |

## Commands

```bash
make install              # editable install with dev extras
make check                # lint, strict types, full test suite, integrity checks
qcal init                 # create Ian's document templates if missing (never overwrites)
qcal registry cells       # expand EXPERIMENTS.yaml design -> explicit cell list (--emit)
qcal registry run C-x --seed 0
qcal registry run-batch 'C-*' --dry-run
qcal registry index       # runs/registry/*.json -> runs/index.csv (--check in CI)
qcal registry audit       # pre-registered coverage (--strict, --json)
qcal registry tables      # configs/tables/*.toml -> paper/tables/*.tex (--check in CI)
qcal claims               # every number traces to a run
qcal leakage              # split manifests are disjoint
qcal licenses             # dependency, import and dataset-card audit
qcal agent-layer          # validate agents, skills, hooks, .mcp.json
qcal policy check PATH    # which protection category a path falls in
qcal config               # merged configuration and where each layer came from
```

## Configuration and debugging

All values come from `src/qcal/resources/defaults.toml`, overridden by the repository's
`qcal.toml`, overridden by `QCAL__<SECTION>__<KEY>=<toml value>` environment variables.
`qcal config` prints the merged result and its sources.

- `--debug` or `QCAL_DEBUG=1`: DEBUG logs and full tracebacks.
- `--log-format json` or `QCAL_LOG_FORMAT=json`: one JSON object per log line.
- `QCAL_HOOK_LOG=<path>`: every hook decision appended as JSONL (tool, rule, reason, mode).
- Exit codes: 0 success, 1 a check or run failed, 2 usage or configuration error.

## Not implemented here, deliberately

- **Science code.** The plan gates Phase 1 on G0 counsel clearance, and the Review's IP rule
  keeps Q-Cal code private until then. **This repository is currently public.** Making it
  private before any calibration code lands is Ian's call and is recommended.
- **Ian-only documents.** `EXPERIMENTS.yaml`, `DECISIONS.md`, `CLAIMS.md` and `RESEARCH_LOG.md`
  are templates only. Ian creates them with `qcal init` and fills them in by hand.
- **Phase 1+ agents and skills.** `paper-reproducer`, `prior-art-scout`, `report-writer`,
  `hf-publisher` and their skills enter with the gates they serve (plan §3.2, §3.3).
- **Executor command.** `executor.command` is empty until Phase 1 provides an experiment
  program; `qcal registry run` explains this instead of guessing.

## Ian's next steps

1. Decide repository visibility before G0.
2. Generate an SSH signing key, add it to `allowed_signers`, and in a signed commit set
   `signing.mode = "enforce"` in `qcal.toml`.
3. Rename the default branch from `civ` to `main` if wanted (`git.protected_branches` covers both).
4. Add a branch ruleset on the default branch: require the `integrity` and `ci` checks,
   linear history, no force-push.
5. Run `qcal init`, then write `EXPERIMENTS.yaml` and `DECISIONS.md` by hand.
