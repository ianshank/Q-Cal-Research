# Next steps and tech-debt register

## Next steps

### Ian (by hand; agents cannot do these)

1. Add your SSH public key to `allowed_signers` (`<email> namespaces="git" <key>`) in a
   signed commit. The email must equal your committer email.
2. Re-sign the Phase 0 branch's enforcement-surface changes, for example with a signed merge
   or a signed squash. Then set `signing.mode = "enforce"` in `qcal.toml`, in a signed
   commit.
3. Branch ruleset on the default branch:
   - require the `ci` and `integrity` checks;
   - require signed commits;
   - block force pushes.
4. Decide the cross-review policy and set `review.mode = "enforce"` when Antigravity
   reviews land at `review/gemini/<branch-slug>.md`.
5. Run `qcal init`, then write `EXPERIMENTS.yaml`, `DECISIONS.md` (gates, kill criteria, WIP
   limit), `CLAIMS.md` and `RESEARCH_LOG.md`.
6. Rename the default branch `civ` to `main` (the plan's recommendation); update
   `git.protected_branches` and the ruleset.
7. Resolve the README overlap with PR #1. Both PRs edit `README.md`; keep both sections
   (this PR's overview and PR #1's "Research review protocol").

### Phase 1 (after G0)

- Science code in `src/qcal_lab/`: detector wrappers, calibrators, LaECE metrics, and the
  quantization pipeline, implementing the `qcal.protocols` shapes.
- An experiment program that honours the executor's result-file contract, and a real
  `make smoke`.
- A container image with the CUDA and TensorRT stack (a separate Dockerfile target).
- The roadmap skills, agents and hooks in [AGENT_LAYER.md](AGENT_LAYER.md#roadmap-opportunities-identified-in-the-gap-analysis).

## Tech-debt register

| ID | Item | Impact | Proposed resolution | Size |
|---|---|---|---|---|
| TD-1 | GitHub Actions are pinned to major tags (`@v4`), not commit SHAs | A compromised tag could run in `integrity.yml` (base code, read-only token) | Pin to SHAs. Dependabot keeps them current. | S |
| TD-2 | `guard-bash` is push-only; shell writes to protected files are not blocked | An agent can write a protected file from Bash. The signed-commit check catches it at the PR, not in the session. | By design (plan §3.4). Optionally add a PostToolUse hook on Bash that runs `qcal policy check` on `git status` paths and warns. | M |
| TD-3 | Command-drift validation checks command paths and option names, not argument values or positional counts | A documented command with a wrong argument still passes | Extend it to `parse_known_args` with placeholder substitution | M |
| TD-4 | The hook-event and known-tool lists in `[agent_layer]` must track Claude Code releases | A new event or tool is reported as unknown until the list is updated | Keep lists in config (done); review them quarterly | S |
| TD-5 | `_TEGRA_RELEASE` (`/etc/nv_tegra_release`) is a platform constant in the Jetson collector | None today; a non-standard L4T image would not be detected | Make it a collector option if a second Jetson image appears | S |
| TD-6 | Five report dataclasses share only a protocol, not a base class | Small duplication in `to_dict` | Acceptable: the fields differ. Revisit if a sixth report appears. | S |
| TD-7 | Local `docker build` in the cloud sandbox cannot reach `deb.debian.org` | The slim + apt path is verified only on CI | None needed in the repository. Allow the host in the environment's network settings to verify locally. | S |
| TD-8 | Review files committed in the PR satisfy `review-check`. A reviewer could be impersonated by a commit authored as the other model. | The cross-review is only as strong as `review/gemini/**` protection | `review/gemini/**` is on the enforcement surface, so such a commit needs Ian's signature in enforce mode. Consider requiring the reviewer's own signing key. | M |
| TD-9 | Parquet output is tested only in the dedicated CI job | The main matrix covers CSV only | Acceptable; the parquet job runs on every PR | S |

Closed in this change: every finding of the Phase 0 peer review. Tests named
`test_<finding>_*` in `tests/regression/test_peer_review_findings.py` and
`tests/security/test_guard_bypasses.py` cover them.
