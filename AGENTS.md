# AGENTS.md: shared by Claude Code and Antigravity

- Truth = the default branch + EXPERIMENTS.yaml + runs/registry/*.json (rendered to
  runs/index.csv) + CLAIMS.md. Never fabricate; every number traces to a run id.
- Never edit EXPERIMENTS.yaml (use AMENDMENTS.md); never touch */handwritten/; never tune on
  test; never edit the enforcement surface listed in qcal.toml / src/qcal/resources/defaults.toml.
- Ownership: Claude Code owns src/qcal_lab/ (Phase 1 science code), configs/, tests/, scripts/,
  the paper build and review/claude/. src/qcal/ is the integrity layer and is Ian-signed.
  Antigravity owns notebooks/, analysis/figures/ and review/gemini/. Ian owns the Ian-only
  paths; his commits are signed and CI verifies them.
- Branches claude/* and gemini/*, one git worktree per agent. Merge only through a PR with green
  CI and the other model's review at review/<reviewer>/<branch-slug>.md (slug: the branch with
  `/` replaced by `-`): verdict approve, reviewed_sha a commit of the PR, and nothing but
  review/ changed since. `qcal ci review-check` verifies this.
- Clean-room: public datasets only; no employer material, devices, networks or accounts.
