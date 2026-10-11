# Contributing to Q-Cal

Three parties write to this repository: Ian (the researcher, who signs), Claude Code on
`claude/*` branches, and Antigravity on `gemini/*` branches. The rules every contributor
follows are in [AGENTS.md](AGENTS.md); the Claude Code rules are in [CLAUDE.md](CLAUDE.md).
This file explains the workflow for humans, including the steps only Ian can take.

## The rules in brief

- Every number in `paper/`, `README.md` and `CLAIMS.md` comes from a registered run, cited as
  `value (run:<id>:<metric>)` in Markdown. `qcal claims` checks this.
- `EXPERIMENTS.yaml` is pre-registered: propose changes in `AMENDMENTS.md`.
- Calibrators fit on the calibrator-fit split only; thresholds are selected on val only; test
  is never used to tune anything. `qcal leakage` checks the split manifests.
- `*/handwritten/*`, `ian_data` (split manifests, oracle outputs, published reference values)
  and the paper's claims, abstract, limitations and conclusions are Ian's alone.
- Clean-room: public, license-checked data only. `qcal licenses` checks the configured data
  and imports.
- Runs are launched only through the registry CLI (see the Runs section below).
- Configuration lives in `qcal.toml` over the packaged defaults; no hardcoded values.

## Workflow

1. Write a change proposal from [docs/changes/TEMPLATE.md](docs/changes/TEMPLATE.md) and add
   it to the [index](docs/changes/README.md).
2. Branch `claude/<slug>` or `gemini/<slug>` from the default branch (`civ` today; `main`
   after the planned rename), one worktree per agent.
3. Run the local checks before opening a pull request:

   ```bash
   make check      # lint, mypy --strict, the full suite with the coverage gate, integrity
   make smoke      # the Phase 1 loop on a synthetic fixture through the real registry
   make pre-pr     # check, plus gitleaks and the container suite when installed
   ```

4. Open a pull request with the template filled in. `ci.yml` runs on the head;
   `integrity.yml` runs the base branch's code on the head's git objects.
5. The other model reviews the branch and its review is committed at
   `review/<reviewer>/<branch-slug>.md` (slug: the branch with `/` replaced by `-`).
   `qcal ci review-check` requires an approving verdict whose `reviewed_sha` is a commit of
   the pull request, with nothing but `review/` changed since.
6. Ian adds a RESEARCH_LOG.md entry, signs what needs signing, and merges.

## Signing an agent branch (Ian)

A protected path is any path in a category listed in `signing.signed_categories`;
`qcal policy check <path>` names a path's category. Claude Code edits the enforcement
surface only on a branch, through a change proposal (CLAUDE.md rule 0); AGENTS.md forbids it
for other agents. No agent ever writes Ian-only files, `ian_data` or the registry's files.
Agent commits are unsigned, so `qcal ci verify-signatures` rejects them once
`signing.mode = "enforce"`. Two rules apply, and both must hold:

- every non-merge commit in the range that changes a protected path carries a good SSH
  signature from a key in `allowed_signers` whose principal equals the committer email;
- every protected path that differs from the fork point holds content a signed commit in the
  range introduced.

Commits that touch only agent-owned paths need no signature, and merging the base branch into
a pull request needs none either.

**Prerequisites.** Your key is already in the base branch's `allowed_signers`, git signs with
SSH (`gpg.format = ssh`, `user.signingkey` set), and `user.email` equals your principal there.

**1. Let the checker list what you would sign.** Never work from a hand-written path list:
the checker names every unsigned commit and each protected file it changes.

```bash
git fetch origin
qcal ci verify-signatures --base origin/civ --head origin/claude/<slug> --mode enforce
git diff --name-only origin/civ...origin/claude/<slug> | xargs qcal policy check
```

**2. Refuse, do not sign,** a branch whose changes include an `ian_only`, `ian_data` or
`registry_only` path. Agents never write those, so such a change is a defect to report, not
content to vouch for. Review files (`cross_review`) are expected only under `review/claude/`.

**3. Read the full diff of every protected file the checker listed:**

```bash
git diff origin/civ...origin/claude/<slug> -- <each protected file listed above>
```

**4. Sign.** Both ways below sign every commit or the whole branch, so your signature vouches
for its agent-owned changes too; read those diffs as well.

*Keep the agent's commits.* Rebase re-creates each commit with you as committer and signs it.
Rebase drops merge commits and replays their conflict resolutions, so read the result again.

```bash
git switch --no-track -c claude/<slug>-signed origin/claude/<slug>
git rebase -S origin/civ
qcal ci verify-signatures --base origin/civ --head HEAD --mode enforce
```

*One signed commit.* Simpler history; the agent's commit messages go into its body.

```bash
git switch --no-track -c claude/<slug>-signed origin/civ
git merge --squash origin/claude/<slug>
git commit -S
qcal ci verify-signatures --base origin/civ --head HEAD --mode enforce
```

**5. Push under a `claude/` name.** The reviewer is chosen by branch prefix
(`review.reviewer_by_branch_prefix`). A branch named outside `claude/` falls back to "any
reviewer", so a Claude-authored change could pass on a Claude review. Either push the signed
branch as `claude/<slug>-signed` and open its pull request, or replace the agent's branch with
it yourself (agents' hooks deny force pushes). Signing changes every commit id, so the
cross-review must name a commit of the signed branch as its `reviewed_sha`: sign first, then
ask for the review, then commit the review file (also a signed path).

## What the hooks tell agents

The hooks in `.claude/settings.json` are feedback for agents, not the control; signed commits
and CI are. When a hook blocks an agent, its message says why:

- **guard-paths** blocks an edit to a path in `hooks.deny_categories`: Ian-only files,
  `ian_data`, the registry's own files, the enforcement surface and git internals. The
  message for each category is in `policy.messages`.
- **guard-bash** blocks force pushes, pushes to protected branches, push commands it cannot
  read literally (command substitution, deep nesting), the configured extra patterns (such as
  the init command with `--force`) and clean-room substrings.
- **claims** runs when an agent stops. It checks the repository's files (`paper/`,
  `README.md`, `CLAIMS.md`) for numbers without a run reference, not what an agent says in
  chat; CI runs the same check as the gate.
- **session start** prints the guard, signing and review modes.

`QCAL_GUARD_MODE=warn` or `off` relaxes the hooks for a session Ian launches; CI is unchanged.

## Runs

Launch runs only through `qcal registry run` or `qcal registry run-batch`. A registered run
refuses `QCAL__*` overrides outside `[logging]`, an uncommitted `qcal.toml` or
`EXPERIMENTS.yaml`, and an `--experiments` file other than the configured one. Never edit
`runs/registry/` or `runs/index.csv` by hand; `qcal registry index` regenerates the index.

## Tests

| Suite | How to run | What it proves |
|---|---|---|
| unit | `make test-unit` | each module, with property-based tests (Hypothesis) |
| integration | `make test-integration` | real git, SSH-signed commits, subprocesses, the hook wrapper |
| regression | `make test-regression` | one test per fixed review or red-team finding |
| security | `make test-security` | bypass attempts on the guards, the signed-commit control and the run gates |
| e2e | `make test-e2e` | the documented workflow through the real CLI |
| contract | `pytest tests/contract` | the `EvalLoop` MUST checks, on the fixture loop and on Ian's loop once it exists |
| oracle | `pytest tests/oracle` | the calibrators against scikit-learn and SciPy (`pip install -e .[oracle]`) |
| parity | `make parity` | `qcal_lab` against the fiveai oracle outputs; skips until they exist |

Hypothesis runs the `ci` profile by default (fixed examples, no database). Set
`HYPOTHESIS_PROFILE=explore` for a longer randomized search. Set `ORACLE_REQUIRED=1` to turn
the oracle suite's skips into failures. Mark a test with `@pytest.mark.rule("C3")` when it
enforces a rule from CLAUDE.md or AGENTS.md; `scripts/traceability.py --check` lists rules
that no test covers.

## Debugging

`qcal --debug <command>` or `QCAL_DEBUG=1` prints tracebacks. `--log-format json` gives
structured logs. `qcal config` shows the merged configuration and every source that shaped
it; `qcal config --check` rejects unknown, removed and mistyped keys.
