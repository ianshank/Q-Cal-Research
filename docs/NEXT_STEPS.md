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

### Phase 1: what remains (`make lab-status` lists the open items)

The scaffold is in place ([LAB.md](LAB.md)). Still open:

**Ian (by hand)**
1. **G0 and repository visibility.** Record G0 in `DECISIONS.md`. The scaffold landed before
   it, on a public repository; decide whether to make the repository private (plan §1.6).
2. **Environment spike.** Set `detectors.atss_r50.config`, `checkpoint` and
   `checkpoint_sha256` in `configs/lab.toml`, and record the mmcv build in `DECISIONS.md`.
3. **Evaluation loop.** Write `src/qcal_lab/handwritten/eval_loop.py` and `laece.py`.
   `build(options)` returns an object with `targets`, `threshold_objective` and `metrics`;
   [LAB.md](LAB.md#ians-evaluation-loop) describes the proposed interface. The metrics are
   AP, LRP, D-ECE, LaECE0 and LaACE0.
4. **Splits.** Choose the paper's published minival/minitest membership (from a CC BY-NC-SA
   repository) or a fresh seeded partition. Write the four manifests with
   `python -m qcal_lab splits`, and confirm `qcal leakage` reports PASS.
5. **Oracle outputs.** Run `fiveai/detection_calibration` unmodified on the P40 and commit
   its outputs as `tests/parity/fixtures/*.json` (schema in `tests/parity/README.md`).
6. **Pre-registration and reference values.** Add the Phase 1 cells to `EXPERIMENTS.yaml`.
   Record the published reference values in `docs/reference/kuzucu_eccv24.md` for
   `/reproduce-check` gate 2.
7. **Signatures.** Sign the enforcement-surface edits of this change (listed in
   `docs/changes/reproduce-kuzucu-eccv24-baselines.md`).

**Review items deferred from the adversarial review**
- An Alg. A.1 parity case kind (u_c/v_c selection on identical inputs); today only
  calibrator fit/transform and metrics have case kinds.
- A temperature-scaling baseline (the paper's Table 8 comparison), if Ian pre-registers it.
- Check that `prior-art-scout` can call the Hugging Face MCP tools with its explicit
  `tools` list, and widen the list if needed.
- `executor.command` runs `python3` from PATH: launch runs from the project venv so the
  interpreter recorded by the registry is the one that ran (`program_python` in each
  record shows it).

**Agents, once those exist**
- `paper-reproducer`: make every calibrator parity case pass; report each deviation.
- `/reproduce-check` before K1 (Nov 8); `/prior-art` monthly.
- Phase 2: a CUDA/TensorRT container target; COCO-C and Foggy shifts; ONNX and TensorRT.

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
| TD-8 | Review files are trusted once Ian signs them | Ian's signature vouches for the review, not the reviewer's identity | Every `review/*/**` file is signed (`cross_review`). Optionally require the reviewer's own signing key. | M |
| TD-10 | Run records that were never committed can be deleted locally, so rerun-until-favourable is invisible to CI | Supersedes need a recorded reason, but uncommitted runs leave no trace | Commit records promptly (signed). Optionally upload every record to an append-only store from the executor. | M |
| TD-11 | `guard-bash` recognizes known push spellings only | An unusual spelling can still push from an agent's shell | The ruleset forbidding direct pushes is the control; add spellings as they are found, each with a security test | S |
| TD-9 | Parquet output is tested only in the dedicated CI job | The main matrix covers CSV only | Acceptable; the parquet job runs on every PR | S |
| TD-12 | The two-threshold grid (`calibration.grid_*`) is not stated in the paper | Thresholds may differ from the oracle's | Confirm against the oracle outputs (paper-reproducer); record the grid in `configs/lab.toml` | S |
| TD-13 | Isotonic duplicate-score merging follows scikit-learn's convention as remembered | Tiny differences on tied scores | The calibrator parity cases decide | S |
| TD-14 | The MMDetection adapter is tested only against fakes | API drift would surface at the first real run | The environment spike runs one image end to end; add an integration test on the GPU runner | M |
| TD-15 | Calibrators are pure Python | Slower than NumPy at about 10^5 detections | Acceptable now (seconds per fit); move to NumPy if profiling says so | S |
| TD-16 | Platt scaling uses damped Newton; the paper uses L-BFGS | Same minimiser for a convex objective; iteration traces differ | Parity cases decide; the saved calibrator records iterations and convergence | S |

Closed in this change: every finding of the Phase 0 peer review. Tests named
`test_<finding>_*` in `tests/regression/test_peer_review_findings.py` and
`tests/security/test_guard_bypasses.py` cover them.
