# Change proposals

Every non-trivial change starts here, before the code (CLAUDE.md, Conventions). A proposal is
the decision record for its pull request: why, what changes, the decisions and their
alternatives, compatibility, and what Ian is asked to decide. Copy [TEMPLATE.md](TEMPLATE.md).

A proposal that edits the enforcement surface (`.claude/`, `.github/`, `Makefile`,
`pyproject.toml`, `qcal.toml`, `src/qcal/`) merges only with Ian's signed commit.

| Proposal | Gate | Surface | Status |
|---|---|---|---|
| [phase-0-integrity-layer](phase-0-integrity-layer.md) | G0 | enforcement | implemented (merged in PR #2) |
| [phase-0-hardening](phase-0-hardening.md) | G0 | enforcement | implemented (merged in PR #2) |
| [reproduce-kuzucu-eccv24-baselines](reproduce-kuzucu-eccv24-baselines.md) | G1/K1 | `src/qcal_lab`, wiring | implemented (merged in PR #2); science inputs are Ian's |
| [cycle-2026-10-g1-readiness](cycle-2026-10-g1-readiness.md) | G0, G1 | plan only | proposed (PR #4) |
| [registry-run-gates](registry-run-gates.md) | G0 | enforcement | proposed (PR #4); needs Ian's signature |
| [registry-run-contract](registry-run-contract.md) | G1 | enforcement | proposed (PR #4); needs Ian's signature |
| [guard-bash-redirection-segments](guard-bash-redirection-segments.md) | G0 | enforcement | proposed (PR #4); needs Ian's signature |
| [run-identity-and-formats](run-identity-and-formats.md) | G1 | `src/qcal_lab`, `configs/` | proposed (PR #4) |
| [evalloop-v2](evalloop-v2.md) | G1 | `src/qcal_lab`, docs; one signed `qcal.toml` line | proposed (PR #4); interface for Ian to accept |

Status values, as in the template: proposed, accepted (Ian), implemented, superseded by another proposal.
