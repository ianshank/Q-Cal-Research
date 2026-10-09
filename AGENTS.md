# AGENTS.md: shared by Claude Code and Antigravity

- Truth = the default branch + EXPERIMENTS.yaml + runs/registry/*.json (rendered to
  runs/index.csv) + CLAIMS.md. Never fabricate; every number traces to a run id.
- Never edit EXPERIMENTS.yaml (use AMENDMENTS.md); never touch */handwritten/; never tune on
  test; never edit the enforcement surface listed in qcal.toml / src/qcal/resources/defaults.toml.
- Ownership: Claude Code owns src/ (outside the enforcement surface), configs/, tests/, scripts/,
  the paper build and review/claude-*.md. Antigravity owns notebooks/, analysis/figures/ and
  review/gemini-*.md. Ian owns the Ian-only paths; his commits are signed and CI verifies them.
- Branches claude/* and gemini/*, one git worktree per agent. Merge only through a PR with green
  CI and a cross-review file whose reviewed_sha equals the PR head.
- Clean-room: public datasets only; no employer material, devices, networks or accounts.
