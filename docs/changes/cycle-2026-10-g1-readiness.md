# change: cycle-2026-10-g1-readiness

## Status
Proposed, Oct 11, 2026 (plan v2 after the expert panel). Agent-owned document; it changes no
enforcement surface by itself. Each PR it lists carries its own proposal in `docs/changes/`.

Progress on the agent branch (draft PR #4, base `civ`):

| PR | State |
|---|---|
| PR-0 | this document, the template, the index, `SECURITY.md`, `CONTRIBUTING.md` |
| PR-A1 | implemented; `docs/changes/registry-run-gates.md`; waits on Ian's signature (session 1) |
| PR-E | implemented (tests only, plus the one-line `oracle` extra, signed surface) |
| PR-D2 | interface proposal `docs/changes/evalloop-v2.md`; code after Ian accepts it |
| PR-A2 | implemented; `docs/changes/registry-run-contract.md`; waits on Ian's signature (session 1) |
| PR-D1 | implemented (agent-owned); `docs/changes/run-identity-and-formats.md`; D5 and F10 are Ian's |
| others | not started: they wait on session 1 and on D1–D10 |

These six share draft PR #4 instead of one PR each, as the delivery plan below intends. They
were built in one session before the branch names were settled. The signed-surface changes
sit in their own commits, so Ian can sign them alone or split the PR: PR-A1 in `d707031` and
`88563df`; PR-A2 in `5b9c36d`, `d125717`, `5a91857`, `a9a0bc9` and `06aba65`; the guard-bash
fix in `cc41488`. `qcal ci verify-signatures --base origin/civ --head HEAD` lists these, plus
PR-E's one-line `pyproject.toml` change (`88eeea4`) and the commits that add advisory reviews
under `review/` (`8c76044`, and the wave-2 review). The PR-D1 commits touch no protected
path.

Open question for Ian from wave 2: `.gitleaksignore` (added in `3134df2` for one false
positive, a golden sha256 digest) is not on the protected surface, so a line there could
silence the secret scan without a signature. Adding it to the enforcement surface is a
one-line change to `qcal.toml`, which is his.
Any later commit on the PR needs a fresh cross-review, because `qcal ci review-check` accepts
only `review/` changes after the reviewed commit.

**Advisory review of the PR-0 documents and the EvalLoop v2 proposal** (`adversarial-reviewer`
on `76ab43d`): *revise*, 5 blocking and 24 non-blocking findings. The review is saved at
`review/claude/claude-sdlc-agents-implementation-plan-gmb29t.md`.

- `CONTRIBUTING.md`, B1: the signing recipe started from a hand-written path list, so Ian could
  sign protected files he never read. It now starts from what `qcal ci verify-signatures`
  lists, and refuses branches that touch Ian-only, `ian_data` or registry paths.
- `CONTRIBUTING.md`, B3: a signed branch named `sign/<slug>` fell back to "any reviewer". It is
  now `claude/<slug>-signed`.
- B2, B4 and B5 changed `docs/changes/evalloop-v2.md`, whose own section records them.
- Non-blocking findings applied:
  - N15: signing prerequisites, `--no-track`, `rebase -S`, dropped merges.
  - N16: the claims hook checks files.
  - N17: Antigravity never edits the enforcement surface.
  - N18: reporting until private reporting is on.
  - N19: this paragraph.
  - N20: unverified paper values.
  - N21: D7's rationale.
  - N22: D1 needs rule 3 and an amendment; PR-F depends on D8.
  - N23: status words.
  - N24: the note on line references below.

## Gate served
G0 (Oct 23) and G1/K1 (Nov 8).

## Plan
Commands and make targets called *planned* arrive with the PR that names them. File and line
references in the plan are to `d84b9e0`, the commit the audits read.

### Context

PR #2 merged into `civ` on Oct 10 (merge commit `ce0cbd4`): the SDLC plan, the Phase 0
integrity layer and the Phase 1 scaffold (`src/qcal_lab`); ~31k lines, 2,142 tests, 99.4%
coverage. PR #1 (Copilot README) is open; Dependabot PR #3 (three Actions bumps) is open.
Gates: G0 Oct 23 (not recorded), G1/K1 Nov 8. Ian's budget is 9 h/week (`SDLC:298`).

**How this plan was produced.** Three read-only audits of HEAD `d84b9e0` (plan-vs-code,
conventions, tests/agent layer), a design pass, then a first draft. The project's
`adversarial-reviewer` returned *revise* (6 blocking). The draft was then reviewed by a
four-lens expert panel, each read-only against the code and (for the ML lens) arXiv:2405.20459
v1: **ML research scientist** (18 findings), **MLOps/reproducibility engineer** (15),
**SQE/test architect** (14), **software architect/SDLC lead** (21). Every finding and its
disposition is recorded in the last section. Four premises of the first draft were wrong and
are corrected here: there is no `calibrator_fit_draw` factor (the seed draws the fit subset
via `calibrator_fit_split_size`, `experiment.py:375`); `qcal_lab.__version__` *is* used (it is
in the prediction-cache key, `models/base.py:89`); the budget is 9 h/week, not 15; agent PRs
are opened under Ian's account, so CODEOWNERS can never gate them.

#### Verified state

**Control plane off.** `signing.mode`/`review.mode` are `bootstrap`; `allowed_signers` has no
keys; `civ` has no ruleset; none of Ian's five documents exist; `review/` holds only the
template; G0 is not recorded. `CLAUDE.md`: "hooks are feedback; Ian's signed commits plus CI
are the control". Until this lands, nothing agents build is under that control.

**Three classes of integrity gap in the registry, all closable before the first registered run.**
- *Configuration (A1/A2, from the first review).* `config_hash` hashes `configs/**` and the
  lab defaults, never `qcal.toml` (`qcal.toml:34-39`); the launcher applies `QCAL__*` env
  overrides (`cli/app.py:82`, `config.py:172-191`), passes the whole environment to the child
  (`executor.py:147-150`), and the child reloads config the same way (`experiment.py:357`).
  `--experiments X` records factors from any YAML while the child runs the configured file's
  (`cli/common.py:42-49`, `runner.py:111-113,192`; `experiments_sha256` is written at
  `runner.py:157` and never checked). `require_clean_tree = false` (`defaults.toml:159`). Git
  itself can be redirected: `gitutil.git()` passes the caller's environment (`gitutil.py:48`),
  so `GIT_DIR`, `GIT_CONFIG_COUNT`, `GIT_INDEX_FILE` and `git replace` change what "HEAD" means.
  Unknown config keys are silently accepted (`config.py:42-51`): a misspelled
  `signing.signed_categories` in `qcal.toml:16` would silently drop `ian_data` from signing.
- *Execution (M1–M9).* The environment is collected in the launcher after the child exits
  (`runner.py:169-176`), from the launcher's interpreter, not the program's; `nvidia` joins
  every GPU on the host into one string (`environment.py:114-123`); `CUDA_VISIBLE_DEVICES` is
  never recorded. The record never names the GPU that ran. `executor.timeout_s = 0`; a timeout
  kills the child, not its process group; Ctrl-C writes no record (`runner.py:163-167`); a
  failed run drops its artifacts and the child's environment (`runner.py:193-195`). Only
  `duration_s` is recorded, no CPU/GPU time, device or batch id. A cache hit trusts a gitignored
  sidecar's `produced_by` (`predictions.py:196-210`); `put` overwrites (`:212-229`); the detector
  is built before the cache is consulted (`experiment.py:387-391`), so a fully cached run still
  needs a GPU, contradicting `SDLC:64`. No durable artifact store exists; DVC is named in five
  places and configured nowhere. The science stack (torch/mmcv/mmdet) is unpinned.
- *Numerics and identity (M6/M7, S1/S2, F7, F12).* `precision = "fp32"` keeps torch defaults,
  i.e. TF32 convolutions on sm_120 while the P40 oracle has no TF32; cuDNN benchmark/
  deterministic flags, `float32_matmul_precision`, `CUBLAS_WORKSPACE_CONFIG` and
  `NVIDIA_TF32_OVERRIDE` are neither set nor recorded. The cache key hashes every `.py`/`.toml`
  in `qcal_lab` plus `__version__` (`experiment.py:244-255,310`), so a calibrator docstring edit
  forces new GPU inference. Three pre-registered factors — `target`, `fit_precision`,
  `threshold_regime` — are validated and then ignored (`experiment.py:134-139,163`): extending
  `target` to `trt_jetson` would silently reuse torch FP32 predictions under that label. The
  predictions header cannot distinguish FP32 from INT8 files (`predictions.py:39-51`).

**The science cannot yet express the paper's protocol (F1–F4; ML lens, from v1 of the paper).**
- The paper fits and selects both thresholds on one 2.5K `minival` and evaluates on a 2.5K
  `minitest` (Alg. A.1; App. C.2). The scaffold requires fit ≠ select ≠ evaluate as four
  disjoint subsets of val2017 (`experiment.py:183-187`; qcal's `defaults.toml:259`). Alg. A.1
  cannot run as published, and each split is a fraction of the paper's.
- The panel reads the paper's AP as COCO top-100 on calibrated, unthresholded detections
  (Table 5: isotonic moves AP, Platt does not) [unverified; Ian confirms under D4].
  `metrics()` receives only the post-`v_c` set (`experiment.py:408-409`;
  `evaluation.py:68-71`), so that AP cannot be computed.
- Table 9's ATSS-R50 row names no mmdet config, and the panel recalls the model zoo's ATSS
  R50 1× AP as several points below the published AP [unverified; the spike checks, and the
  published values belong in Ian's `docs/reference/`]. A 2.5K re-split cannot close a gap of
  that size.
- In Phase 1 the seed draws nothing unless `calibrator_fit_split_size` is set; N seeds are N
  identical runs and `tables.py:218` would report a false zero variance. The actual variance
  sources (test-image sampling, split assignment, grid, inference stack) are unmeasured. The
  ±1.0 LaECE0 tolerance exceeds half the Platt–isotonic gap, so G1 could pass with the paper's
  ordering reversed. `SDLC:57` already says G1 rests first on oracle parity on identical
  detection files; the plan must make that operational.
- Matching rule, τ per stage, tie-breaks, crowd handling and bin edges are not in the paper;
  `LAB.md:98-100` leaves them to Ian with no written protocol. There is no independent oracle of
  any kind until the P40 outputs exist.

**The scaffold's convention debt is real but secondary** (logging through the Config, exit
codes, ~25 hardcoded sites, duplicated helpers, unversioned formats, import-side-effect
registries, no mutation/benchmark tests). It stays in the plan, after G1, except where it
touches a hashed format or a record (which is free now and expensive later).

### The case

1. **Ian's time is the constraint, not ours.** 9 h/week, two gates in four weeks, and the
   science inputs (spike, loop, splits, oracle) are his alone. Tooling is capped at ~2 h/week of
   his time, delivered as dated signing packets with a cut line (`SDLC` risk 10).
2. **Make G1 interpretable before making it green.** Four decisions (split design, AP set,
   checkpoint identity, variance protocol) decide G1 regardless of code. Agents prepare both
   options for each; Ian decides this week.
3. **Fix the record before the first run writes one.** Schema 1 only gains fields; the
   reproducibility contract (device, environment from the child, cache producer, artifact
   store, lock file, numerics state, cost) lands now or never.
4. **Lock the Phase 2 identity decisions now.** Narrow cache key; every pre-registered axis a
   role; precision/target in the predictions header; per-role prediction source. These are free
   today and cost GPU/Jetson reruns later. No Phase 2 code.
5. **Verify the calibrators independently now.** Invariant suites, scikit-learn/scipy/pycocotools
   as *mathematical* oracles (not the fiveai code), and `EvalLoop` contract tests settle TD-13
   and TD-16 before the P40 outputs exist, and give Ian protocol feedback the day his loop lands.
6. **Keep the control honest.** The adversarial pass is advisory and saved to
   `review/claude/`; the cross-model review stays Ian's rule and CI's check.

### Decisions Ian must take this week (agents prepare both options where there are two)

| # | Decision | Recommendation | Evidence / where it lands |
|---|---|---|---|
| D1 | Split design for the reproduction cells: (a) `paper` — fit = select = minival (2.5K), evaluate = minitest (2.5K), per Alg. A.1; (b) keep four disjoint splits and make the published-number comparison non-gating | **(a)**, pre-registered as a `split_design` factor; the evaluate split stays untouched (rule 3's intent); `trt_calib_images` from train2017; drop the 5000 fit-size level | CLAUDE.md rule 3 wording (signed); `split_roles`/`check_disjoint` (agent); manifests (Ian) |
| D2 | Which ATSS-R50 checkpoint the paper used; the model-zoo AP it should reproduce | Identify before the spike; a pre-declared pass/fail sanity check (raw top-100 AP vs zoo value) that is never used to *choose* a checkpoint | `docs/models/atss_r50.md` (Ian); check (agent) |
| D3 | Matching/τ protocol spec (τ per stage, ≥ vs >, greedy order, ties, crowd, bin edges, classes with GT but no detections) | Write `docs/reference/kuzucu_eccv24.md` protocol section *before* the loop; agents generate synthetic inputs that separate conventions, Ian runs the oracle on them | Ian (ian_data); synthetic cases (agent) |
| D4 | Which detection set feeds which metric: raw / calibrated-unthresholded (top-100 AP) / operating (LaECE0, LRP) | Accept the `EvalLoop` v2 interface (three sets, match-once layer, bootstrap by resampled datasets) | PR-D2 (agent drafts; Ian accepts before writing his loop) |
| D5 | Numeric regime for G1 cells | `fp32_tf32_off`; deterministic flags on; `NVIDIA_TF32_OVERRIDE` recorded | PR-D1 |
| D6 | G1 acceptance | Hard gate: oracle parity on identical detection files. Secondary: paired image-bootstrap 95% CI vs published; ordering checks (uncal > Platt > isotonic LaECE0; LRP unchanged; AP unchanged by Platt, except classes where Platt's a = 0 or scores clip). One seed where the seed draws nothing; tables refuse CIs over identical runs | `EXPERIMENTS.yaml`/`DECISIONS.md` (Ian); harness (PR-F); tables refusal (PR-A2) |
| D7 | Control plane: signing key, `enforce`, ruleset, `civ`→`main`, visibility, PRs #1/#3 | Do first (S0); do **not** enable GitHub's signed-commit rule (it rejects the agents' unsigned commits on every branch it covers [its effect on merge methods is unverified]; `qcal ci verify-signatures` is the control) | Ian, Oct 13 |
| D8 | Who owns bootstrap/Holm code and system metrics (latency, thermal) given `AGENTS.md:17-18` ("only Ian's loop reports metrics") | Statistics over Ian's per-image outputs are agent code under test; system metrics are `environment`, not `metrics` | AGENTS.md wording (signed) |
| D9 | Storage: retention, off-machine copy, cross-GPU cache-hit policy, `compute_budget` | Store under `runs/artifacts` (PR-G); one GPU0-vs-GPU1 measurement decides whether the GPU SKU enters the cache key | `DECISIONS.md` (Ian) |
| D10 | Jetson execution model (Python 3.10 on JetPack 6.2 vs `requires-python >= 3.11`) | Decide in the spike: registry on-device vs remote worker | `DECISIONS.md` (Ian) |

Also for pre-registration (Ian): the second reproduction detector is the paper's D-DETR R50, not
DETR-R50 (`SDLC:60` chose DETR for ONNX portability — conflict to resolve); Q-Cal's ≤4-corruption
COCO-C is not comparable to the paper's 15×3; the grid, `score_thr`, `max_per_img = 100` and
the K1 fallback order are frozen before the first registered run (F10).

### Architecture decisions locked now (so Phase 2 is additive)

1. Cache key = detector fingerprint + digest of the prediction-producing modules only
   (`models/`, `data/coco.py`, `predictions.py`, a code-constant list); no `__version__`, no
   whole-package digest; includes `target`, `quant_path`, compute capability, CUDA/cuDNN,
   effective numerics state, opencv/pillow/mmcv-build.
2. Every pre-registered axis is a role (`target`, `quant_path`, `fit_precision`,
   `threshold_regime`, `split_design` join the six); `factors.supported` and `factors.defaults`
   keys ⊆ roles; `RunPlan` maps each role (fit/select/evaluate) to (dataset, detector,
   precision/target, source).
3. Predictions header carries `precision`, `target`, `quant_path`, `shift`,
   `score_definition`, model artifact sha256 and the effective `test_cfg`; rows reserve
   `source_index`, `class_logit`, `aux_logit`. A replay detector kind consumes a hashed file.
4. Record schema 1 only gains fields: `provenance.{policy_sha256, config_sources,
   config_environment, config_inputs_sha256, batch_id, launcher, retry_of}`, `resources.*`,
   `environment.{container_image, lock_sha256, cost}` reserved, `inputs[]` reserved,
   `ArtifactRef.produced_by`.
5. Run identity covers science settings only: smoke/parity/fixture tables move out of the
   hashed defaults; artifact names and format versions are code constants.
6. Config is closed-world: keys come from the defaults; open tables, per-kind schemas and removed
   keys are declared in code; `qcal config --check` and `python -m qcal_lab config --check`.
7. `EvalLoop` v2: Ian's `match(...) → MatchTable` layer, then metrics over the table with
   per-image weights and three detection sets; matched once per split, thresholded in one pass.
8. Datasets carry kind, image-id namespace, split namespace and label map; `manifest_pattern`
   gains `{dataset}`.
9. Science protocols move to `qcal_lab` in the Phase 2 proposal (`Backend`; `Quantizer` split
   into Exporter/CalibCacheBuilder/EngineBuilder; `Loss` beside `EvalLoop`). Not now.
10. One `executor.command` with `{python}` = launcher interpreter; one version; one-way imports
    with an allowlisted `qcal` API (boundary test).

### Delivery plan

Branches from `main` (after D7's rename; `civ` until then). One proposal per PR under
`docs/changes/`, each with the new Decisions/Compatibility/Ian-hours sections. The plan itself
is committed as `docs/changes/cycle-2026-10-g1-readiness.md` so `/weekly-review` sees it.

| PR | Branch | Gate | Surface | Depends | Ian h | Sign/review by |
|---|---|---|---|---|---|---|
| S0 | — | G0 | GitHub settings, `allowed_signers`, `qcal.toml` | – | 1.5 | Oct 13 |
| PR-0 | `claude/cycle-plan-and-process` | G0 | `docs/`, `SECURITY.md`, `CONTRIBUTING.md` | – | 0.25 | Oct 14 |
| WP0 | `/prior-art` → `review/prior-art/<run-date>.md` | G0 | `review/` | – | 0.25 | Oct 14 |
| PR-A1 | `claude/registry-run-gates` | G0 | `src/qcal` (signed), `experiment.py` | S0 | 1.0 | **Session 1, Oct 17** |
| PR-A2 | `claude/registry-run-contract` | G1 | `src/qcal` (signed), `experiment.py` | A1 | 1.5 | **Session 1, Oct 17** |
| PR-E | `claude/calibrator-verification` | G1 | `tests/`, `pyproject` dev extras (signed, 1 line) | – | 0.5 | Oct 17 (extras line in Session 1) |
| PR-D1 | `claude/run-identity-and-formats` | G1 | `src/qcal_lab`, `configs/` | A1 | 0.5 | Oct 24 |
| PR-D2 | `claude/evalloop-v2-and-replay` | G1 | `src/qcal_lab`, `docs/`, `qcal.toml` (signed, 1 line) | D4 | 1.0 (interface review) | **Oct 17 proposal; code Oct 24** |
| PR-B | `claude/qcal-foundations` | G1 | `src/qcal`, `.claude/` (signed) | A1 | 1.0 | **Session 2, Oct 24** |
| PR-C | `claude/build-ci-alignment` | G1 | Makefile, workflows, pyproject, `uv.lock`, `envs/` (signed) | #3 merged | 0.5 | **Session 2, Oct 24** |
| PR-G | `claude/artifact-store-and-cards` | G1 | `src/qcal` (signed), `docs/models/` | A2 | 0.5 | **Session 2, Oct 24** |
| PR-F | `claude/g1-parity-and-bootstrap` | G1 | `src/qcal_lab`, `tests/parity`, `scripts/` | D2, E, D6, D8 | 0.5 | Oct 31 (freeze) |
| post-G1 | WP3b conventions, WP4b reports, WP5 `/review-pr`, S17 protocols | – | – | – | – | after Nov 8 |

**Parallelism (no shared files):** PR-0 ∥ WP0 ∥ PR-A1 ∥ PR-E (E is tests-only; its regression
file is its own). After A1: PR-A2 ∥ PR-D1 ∥ PR-D2-proposal. After A2: PR-B ∥ PR-C ∥ PR-G.
Then PR-F. Each PR owns `tests/regression/test_<slug>.py`; CHANGELOG gets one fragment per
proposal under `docs/changes/fragments/` assembled at tags (`v0.2.0` at G0, `v0.3.0` at the
G1 freeze); `IMPLEMENTATION_NOTES.md` is edited only by PR-0 and the post-G1 batch.

**Ian's tooling time:** S0 1.5 + Session 1 ≈ 3 + Session 2 ≈ 2 + reviews ≈ 1.5 → ~8 h over
four weeks (<2 h/week). Science time (spike, loop, splits, oracle, cards) is separate and larger.

**Cut line, in drop order if Ian's time runs short:** post-G1 batch (already deferred) →
PR-C's `qcal-lab` script and ruleset JSON (Ian can click the ruleset) → PR-G (interim: a
documented manual copy of `runs/{cache,results,logs}` after each session) → PR-B's agent-layer
checks → PR-A2's `resources` block (reserve the key only). **Never cut:** S0, PR-A1, PR-D1,
PR-D2's interface, PR-E's oracles, D1–D6.

**Agents per PR:** `adversarial-reviewer` (advisory) on every PR, verdict saved to
`review/claude/<branch-slug>.md` and committed once per signing session; `data-leakage-checker` on
A1, D1, D2, F; `paper-reproducer` (worktree, Bash allow-list) for `calib/` and parity work in
D1, E, F; `prior-art-scout` for WP0; Explore/Plan agents for the `EvalLoop` v2 draft. No agent
has Write plus unconstrained network (`SDLC` risk 18).

#### S0 — control plane (Ian, Oct 13)

Signing key in `allowed_signers` (signed commit; committer email must match);
`signing.mode = "enforce"`; `registry.require_clean_tree = true`; rename `civ` → `main` and
update `git.protected_branches`; ruleset on `main` requiring the *job names* "lint and types",
"tests (python 3.11/3.12/3.13)", "integrity checks (head)", "smoke (synthetic fixture through
the registry)", "secret scan (gitleaks)", "container build and suite", "signed protected paths,
immutable registry"; block force-push; **no** GitHub signed-commit rule (D7); merge or close
PR #1 and merge #3; decide visibility; `qcal init`. Verification: the first PR after the
ruleset shows the integrity check as required and passing.

#### PR-0 — cycle plan and process (agent-owned, G0)

`docs/changes/cycle-2026-10-g1-readiness.md` (this plan, in the template's sections);
`docs/changes/TEMPLATE.md` gains Status, Decisions (alternatives, consequences), Compatibility
(formats, config keys, CLI, record), Ian decisions requested (with dates), Ian hours;
`docs/changes/README.md` as the ADR index; `SECURITY.md` (Ian adds the contact and turns on
private vulnerability reporting); `CONTRIBUTING.md` (how Ian re-signs an agent branch; what the
hooks tell agents, for humans); `docs/AGENT_LAYER.md:47-52` reworded; `evaluation.py:9-11`
docstring fix lands with D2.

#### PR-A1 — registered-run gates (signed, G0)

**Goal.** A registered run's configuration and git state come from committed files; the
environment, `--experiments`, a dirty `qcal.toml` and git redirection cannot change them.
Simplified per S13: the rule is "never", not a switch.

- `src/qcal/gitutil.py`: run git with a scrubbed, allowlisted environment; set
  `GIT_NO_REPLACE_OBJECTS=1`, `GIT_CONFIG_NOSYSTEM=1`, `LC_ALL=C` (Q2).
- `src/qcal/config.py`: `Config.environment_inputs` (every `QCAL__*` consumed, plus
  `QCAL_CONFIG`/`QCAL_ROOT` when used); keep `policy_config(repo, ref)` where it is but make the
  runner use it; closed-world key registry derived from the defaults files, typed by the
  default's type; open tables declared in code (`policy.categories`, `policy.messages`,
  `review.reviewer_by_branch_prefix`); element types for empty-list defaults; a removed-keys map
  with messages; `qcal config --check` (S5). Stays stdlib-only; added to `test_stdlib_only.py`.
- `src/qcal/registry/runner.py`: refuse (exit 2, nothing written) when `environment_inputs`
  has any non-`logging.*` key, when `QCAL_CONFIG` names anything but `root/qcal.toml` or
  `QCAL_ROOT` disagrees with `--root`, when `qcal.toml` differs from `HEAD:qcal.toml`, when the
  tree is dirty, or when `--experiments` does not resolve to `paths.experiments` (`--dry-run`
  exempt). Provenance gains `policy_sha256`, `config_sources`, `config_environment`,
  `config_inputs_sha256` (files only; audit flags a cell whose seeds differ, `--strict` fails).
- `src/qcal/registry/executor.py`: strip `QCAL__*`, `QCAL_ROOT`, `QCAL_CONFIG` from the child
  env (code constant); keep `QCAL_DEBUG`/`QCAL_LOG_*`.
- `src/qcal/registry/records.py`: `SCHEMA_VERSION: Final = 1`; delete the
  `registry.schema_version` config key (removed-keys map explains); reject `> 1`/`< 1`.
- `src/qcal/cli/app.py`: `main(argv, *, environ=None)` so smoke's in-process calls pass a
  clean environment (N1c).
- `src/qcal_lab/experiment.py:357`: the same gate from the child (`PlanError`); the child
  records the sha256 of `EXPERIMENTS.yaml`, `qcal.toml` and `configs/lab.toml` it read, and the
  runner **fails the run** on mismatch (closes the gate→child race; `merge_environment` only
  warns today, `runner.py:291-302`).
- CLAUDE.md: "`QCAL__…` overrides: never for registered runs" (signed, one line).

**Tests.** unit: config (Hypothesis: captured names == generated names; key registry rejects
unknown/mistyped/removed keys), runner (each refusal → no record; policy from HEAD not tree),
executor (strip list), records (schema bounds), lab experiment (child gate, recorded shas).
security `tests/security/test_run_input_bypasses.py`: manifests-dir redirect; narrowed
`config_hash_inputs`; `executor.command` swap; `QCAL_CONFIG`/`QCAL_ROOT` redirects;
`QCAL__POLICY__CATEGORIES__IAN_ONLY` promotion of an agent module; dirty `qcal.toml`;
`--experiments` with other factors; env injected only into the child; `GIT_DIR` to a sibling
repo with a permissive `qcal.toml`; `GIT_CONFIG_COUNT` injecting `core.worktree`;
`git replace` of the `qcal.toml` blob; `GIT_INDEX_FILE` hiding a dirty tree. e2e: refusals exit 2
with the registry unchanged; `QCAL__X__Y=1 make smoke` passes.

#### PR-A2 — registered-run reproducibility contract (signed, G1)

**Goal.** A record says which program, interpreter, device and environment produced it, how it
failed if it failed, and what it cost. Additive under schema 1.

- *Interpreter and environment* (M1): `{python}` placeholder = launcher `sys.executable`
  (`executor.py:35-37`, `qcal.toml:44`; no absolute path in config, which is hashed); launcher
  collectors move under `launcher.*`; `registry.recorded_env_vars` allowlist recorded verbatim
  (`CUDA_VISIBLE_DEVICES`, `NVIDIA_TF32_OVERRIDE`, `CUBLAS_WORKSPACE_CONFIG`,
  `PYTORCH_CUDA_ALLOC_CONF`, `CUDA_MODULE_LOADING`, `LD_LIBRARY_PATH`, `OMP_NUM_THREADS`);
  `nvidia` collector per GPU (index, uuid, name, compute capability, memory, driver);
  `merge_environment` no longer hides child keys that collide.
- *Failure handling* (M8): `executor.timeout_s` non-zero in `qcal.toml` (Ian sets; 4 h
  suggested); child in its own session, SIGTERM → grace → SIGKILL of the group; the child always
  writes a result envelope with `status`, `failure_kind` (timeout | cuda_oom | host_oom |
  cuda_error | plan | config | interrupted | unknown), environment and partial artifacts, and the
  registry records them for failed runs; a record is written on `BaseException` then re-raised;
  `runs/inflight/<run_id>.json` marker (audit lists stale ones); a lock per (cell, seed);
  `provenance.retry_of` set automatically; no automatic retries.
- *Resources* (M9): `resources.{wall_s, cpu_user_s, cpu_sys_s, peak_rss, device_kind,
  device_name, device_uuid, gpu_busy_s, peak_gpu_mem, cache_hits, cache_misses, cloud_usd}`
  from `os.wait4` plus the child's report; flattened as `res.*`; `provenance.batch_id`,
  `launcher`; `audit --budget` later (report-only).
- *Identity* (M2): the launcher refuses a registered run unless `qcal.__file__` is under
  `<root>/src`; tables refuse mixed `config_inputs_sha256` within a cell and refuse CIs over
  identical runs (D6); `environment.container_image = "none"` reserved; `inputs[]` and
  `environment.cost` reserved (S7).
- *Manifests* (S16): `data.manifest_pattern` gains `{dataset}` with `id` as the default.

**Tests.** unit per field; integration: real `SubprocessExecutor` with a script that dumps
`os.environ`, sleeps past the timeout (group killed, `failure_kind = timeout`), raises
`KeyboardInterrupt` in the launcher (record exists); regression `test_registry_run_contract.py`.

#### PR-E — calibrator verification and test depth (agent-owned, G1; one signed line for extras)

**Goal.** Independent evidence that the calibrators and the evaluation seam are right, before
the fiveai outputs exist. Highest value per Ian-hour in the plan.

- `tests/unit/test_calib_properties.py` + slow references under `tests/reference/` (Q1):
  isotonic — non-decreasing on any grid, idempotent on increasing targets, knots invariant to
  permutation and k-fold duplication and to any strictly increasing score remap, block
  characterisation (weighted mean; strictly increasing blocks; prefix/suffix mean bounds), exact
  at knots / linear between / clip outside, shift-by-c, duplicate-resolution boundary examples;
  Platt — optimality conditions with an independently computed gradient (interior ‖∇L‖ ≤ k·tol;
  boundary ∂L/∂b ≈ 0, ∂L/∂a ≥ −tol), loss ≤ min(L(1,0), L(0, logit ȳ)), permutation/duplication
  invariance, (1−p, 1−y) ⇒ (a, −b), recovers known (a*, b*), anti-correlated ⇒ a = 0,
  determinism, separable/all-zero/0-and-1 examples; two-threshold — thresholds on the grid,
  `select_threshold` ≡ reference argmin, `keep_class` nesting and order, `apply` ≡
  filter∘transform∘filter, identity for empty classes, image-order invariance,
  **evaluate-split invariance** (perturb evaluate predictions and GT → byte-identical
  `calibration.json`), `metrics` called exactly once. Hypothesis strategies generate ties.
- `tests/oracle/` (Q7, F6; `importorskip`; `ORACLE_REQUIRED=1` turns skips into failures; extra
  `oracle = ["scikit-learn", "scipy", "pycocotools"]`, pinned in PR-C): isotonic vs
  `IsotonicRegression(increasing=True, y_min=0, y_max=1, out_of_bounds="clip")` on tie-heavy
  float32/float64 (settles TD-13); Platt vs `scipy.optimize.minimize(L-BFGS-B, bounds a ≥ 0)`
  comparing **loss** to 1e-10 and transforms to 1e-6, not parameters (settles TD-16); in-sample
  identities (interior Platt optimum ⇒ mean calibrated = mean target; Platt CE ≤ identity's;
  isotonic SE ≤ Platt's ≤ identity's); AP of Ian's loop vs pycocotools once the loop exists.
  Oracle code only under `tests/oracle/`, never imported by `src/`; the import scan is extended
  to `tests/**` in PR-C with sklearn/scipy/numpy allowed only there and in `benchmarks/`.
- `tests/contract/` (Q4): `EvalLoop` MUST checks parametrised over `fixture` (always) and
  `configured` (skips NOT ASSESSED while `handwritten/eval_loop.py` is absent; fails if present
  and broken): factory returns an `EvalLoop` and does not mutate options; targets per image,
  one per detection, finite in [0, 1]; objective is a float for every label × stage incl. an
  empty class; metrics pass `check_metrics`; two calls and two builds give bitwise-equal
  canonical JSON; inputs not mutated; identical bytes under `PYTHONHASHSEED` 0 and 1; no hidden
  I/O (`sys.addaudithook` sees no manifest/annotation reads, no network). Checks live in
  `tests/contract/eval_loop_checks.py` until PR-F moves them into `qcal_lab.evaluation` for the
  `check-eval-loop` command (one implementation, two consumers).
- `tests/e2e/test_lab_registry_loop.py` (Q5): a git-initialised smoke project through
  subprocess `python -I -m qcal`: `run-batch` → `index` → `audit --strict` → `tables` →
  `CLAIMS.md` citing `value (run:<id>:<metric>)` → `claims` PASS → tamper table (FAIL) → edit a
  record metric (`index --check` FAIL) → `--rerun --reason` (claims on the superseded run FAIL)
  → flip an artifact byte (`xfail(strict=True)` until PR-G's verify command) → PR-A1 refusals.
- Traceability (Q8): `rule(id)` marker via `pytest_configure`; `scripts/traceability.py --check`
  (stdlib; AST scan; rule ids from CLAUDE.md "## Integrity" and AGENTS.md; fails on a rule with
  zero tests outside an owned `KNOWN_GAPS` table; run from a unit test). Known gaps today: rule 0
  (no change-proposal check), rule 2 (EXPERIMENTS edit undetected — PR-A1 closes), rule 5
  ("interpretation" unprotected — Ian's policy), rule 6 (import scan skips `tests/` — PR-C).
- Determinism (Q9): Hypothesis profiles `ci` (200, derandomize, no database, no deadline,
  print_blob), `dev` (50), `explore` (500, randomized; `make pre-pr` and a weekly non-gating
  unit-only job in PR-C); `_clean_env` strips `GIT_*` and sets `LC_ALL=C`; hash-seed and TZ
  independence *tested* on smoke artifacts, not pinned; `pytest-randomly` installed, off by
  default, the planned `test-shuffle` target; no retries.
- Also: behavioural round-trips (`transform` equality and canonical-JSON stability, not bytes,
  so PR-D1's envelopes only change strategies); corrupt-cache test (`experiment.py:284-286`,
  hidden by the coverage bug fixed in PR-C); `runpy` tests for the three `__main__` modules;
  `tests/unit/test_agent_settings.py`, `test_reports.py`; `mmdet.{set_tf32, tf32_flags}`,
  `select_class_thresholds`, `source_digest`; `scripts/unreferenced_public_names.py`;
  import-boundary test (`qcal` never imports `qcal_lab`; `qcal_lab` imports an allowlisted
  `qcal` API). Peer-review findings 5/11/14/18: Ian supplies the numbering.

#### PR-D1 — run identity and formats (agent-owned, G1; after A1)

Lock-now decisions 1–3, 5, 8 (lab side) and the strict formats.

- *Cache key* (S1, M7, F16): `source_digest` over a code-constant module list; drop
  `__version__`; add `target`, `quant_path`, compute capability, CUDA runtime, cuDNN, effective
  numerics state, opencv, pillow, mmcv build variant, resolved mmdet config digest (incl.
  `_base_`), effective `test_cfg`; test: editing `calib/` leaves the key unchanged, editing
  `models/` changes it.
- *Roles* (S2): `target`, `quant_path`, `fit_precision`, `threshold_regime`, `split_design`
  become roles with single Phase 1 values; closed-world check that `factors.supported` and
  `factors.defaults` keys ⊆ roles; `_FACTOR_ROLES` stays in code; `RunPlan` maps each role to
  (dataset, detector, precision/target, source) (F11).
- *Split design* (D1): only after Ian's signed rewording of CLAUDE.md rule 3 and an
  `AMENDMENTS.md` entry adding the `split_design` factor, since this relaxes the fit ≠ select
  guard (`experiment.py:184-188`). `split_design = paper | disjoint`; `paper` allows
  fit = select and keeps evaluate disjoint; `check_disjoint` enforces per design; leakage PASS
  still required.
- *Cache producer* (M3): a hit is accepted only if `produced_by` names a record in
  `runs/registry/` listing the same path and sha (else miss + warning); `put` is
  create-exclusive and an existing key with different bytes is a hard error; the sidecar stores
  key inputs and the producer's device/environment digest, copied into the consuming record;
  the detector is built on the first miss only; `**parts` cannot override `format`/`version`.
- *Numerics* (M6, F7): default `fp32_tf32_off`; registered runs set `cudnn.benchmark=False`,
  `cudnn.deterministic=True`, `use_deterministic_algorithms(True)`; record all flags,
  `float32_matmul_precision`, `CUBLAS_WORKSPACE_CONFIG`, `NVIDIA_TF32_OVERRIDE`; spike checks
  torch ≥ 2.9's `fp32_precision` API [verify].
- *Child environment* (M1b, M5): GPU used (name, uuid, capability), `torch.version.cuda`,
  cuDNN, sha of `torch.__config__.show()`, OpenMMLab `collect_env()` as an artifact, full
  `importlib.metadata` list as an artifact with digest, `lock_sha256`, `freeze_matches_lock`
  (+ diff); refuse a registered run unless `qcal_lab.__file__` is under `<root>/src`;
  `eval_loop.handwritten_sha256` over every file under `*/handwritten/` (M2).
- *Predictions header* (S6, F12): `precision`, `target`, `quant_path`, `shift`,
  `score_definition`, model sha256, effective `test_cfg`; rows reserve `source_index`,
  `class_logit`, `aux_logit`; `logit` semantics documented per detector (ATSS score = class
  sigmoid × centerness).
- *Formats*, all strict (no legacy branch; no artifacts exist): envelopes `qcal_lab.calibrator`
  (Platt `to_dict` carries solver settings), `qcal_lab.calibration` (atomic write; `from_dict`
  raises `CalibrationError`), `qcal_lab.prediction_cache_meta`, `qcal.executor_result`,
  `qcal_lab.parity_case` (+ `oracle.environment`); `VERSION = 1` constants in code; records of
  Platt boundary hits (a = 0) per class (F14).
- *Identity hygiene* (S8, B4): `[smoke]`, `[parity]`, `[detectors.fixture]` and `[fixture_eval]`
  (incl. `hit_iou`) move to an unhashed resource; artifact and cache file names become code
  constants; `[eval_loop.options]` stays empty in packaged defaults.
- *Pins* (F13): `configs/lab.toml` sets `score_threshold` and `max_per_image = 100` explicitly;
  a label-map check that the checkpoint's class list equals `GroundTruth.category_names` (F6).
- *Datasets* (S16): entries carry kind, id namespace, split namespace, label map.
- *Dev path* (M11): `python -m qcal_lab dev` — `DEV-` ids under `runs/dev/`, read-only cache,
  fit/select splits only, no metrics, no evaluate access; registered runs refuse `DEV-*`
  producers; `smoke` child runs with `-I` (Q10b).
- Tests: unit per change; security (`test_lab_boundaries.py`): dev path cannot read evaluate;
  a producer outside `runs/registry` is a miss; `PYTHONPATH` cannot shadow `qcal_lab` in the
  smoke child; golden digests for `config_hash`, `cell_id`, `split_digest`, `rank_key`/`draw`,
  predictions header and rows, `fixture_bytes`, `lab_config_effective_sha256` pinned *before*
  the key change and re-pinned with the change explained in the proposal.

#### PR-D2 — `EvalLoop` v2 and the replay detector (agent-owned, G1; interface by Oct 17)

**Goal.** The interface Ian writes against can compute the paper's metrics on the paper's
detection sets, support paired bootstrap, and stay stable through Phase 2.

- `docs/changes/evalloop-v2.md` **first** (by Oct 17, before Ian writes his loop): two layers.
  Ian's `match(predictions, ground_truth, *, stage) → MatchTable` (per detection: image, label,
  score, matched object or none, IoU; per class: object counts; crowd ignores) runs once per
  split and stage under three stated rules. `threshold_objectives` returns a whole curve from
  one table, and `metrics` receives three detection sets (raw; calibrated before v_c; operating).
  Test: "match once, then threshold" ≡ re-matching, including synthetic cases. The fixture loop
  implements v2; Ian's module declares the version, checked at load time.
- Replay detector kind (F11): consumes a raw-predictions file with a provenance sidecar from
  `data/replay/` (`ian_data`, signed); lets G1 decompose the gap into published − oracle,
  oracle − our pipeline on the oracle's detections, and our pipeline on the oracle's
  detections − our full run; later carries TensorRT/Jetson outputs with device and engine
  recorded.
- `evaluation.py:9-11` and `two_threshold.py:14-15` docstring fixes (F18).
- Tests: contract suite (PR-E) runs against v2; replay fingerprint covers file and sidecar;
  an evaluate-invariance test and a spy show no evaluate image reaches matching or selection.

#### PR-B — qcal foundations and agent guards (signed, G1; after A1)

Two commits in one PR, each reviewable alone.
- *Errors/serialize/paths*: `src/qcal/errors.py` (`QcalError.exit_code`, `QcalUsageError`;
  existing classes keep stdlib bases), `cli/common.py::exit_code_for`, `src/qcal/serialize.py`
  (canonical JSON, `sha256_file` moved here, atomic writes, strict `check_envelope`,
  `is_finite_number`), `src/qcal/paths.py::display_path`, public `leakage.split_digest`;
  `ci/__init__.py::MODES`; bad mode → `ConfigError`; `--github` + `--json` → usage error;
  `cells --emit --json`; `tables.py:239` names a real command; `immutability.py` logs each
  violation; `reports.verdict` reused; dead `_log` in `hooks/bash.py` removed. **No
  `--reviewer` flag** (B5). Closed-world exception test scoped to `qcal` (extended to `qcal_lab`
  post-G1).
- *Guards and agent layer*: `allow_only` stays exact-token (B3); the reproducer's list gains
  exact `git diff src/qcal_lab`, `git diff tests`, `git diff configs`, `git diff --stat`,
  `git log --oneline -20`; hard-link denial when `st_nlink > 1` and the inode matches a
  protected file; NFKC + format-character stripping before clean-room substring matching
  (Q10c/e); `prior-art-scout` `deny-read ian_only ian_data`; agent `mcpServers` must name
  `.mcp.json` servers and count as network; `isolation`/`maxTurns` validated; config-driven
  command-reference parsers covering `python -m qcal_lab …` (lazy dotted-path import; never run
  in `integrity.yml`); config-key references in docs drift-checked (S19; functions written with
  `()`, files with extensions).
- Tests: unit per change; security through the real wrapper (`git diff src/qcal_lab` allowed;
  newline-chained, `--output <path>`, `;`, `--no-index`, `--ext-diff` denied; hard link to
  `EXPERIMENTS.yaml` denied; a confusable spelling of a clean-room substring denied); regression `test_qcal_foundations.py`.

#### PR-C — build, CI and dependency alignment (signed, G1; after Ian merges #3)

- `Makefile`: `QCAL ?= $(PYTHON) -I -m qcal`; `check` adds `smoke`; the planned `deps-lock` target,
  the planned `test-shuffle` target, the planned `gpu-check` target (script from PR-F); CLAUDE.md's "what CI runs" line
  updated. `.pre-commit-config.yaml`: local `mypy`.
- `pyproject.toml`: `pytest-timeout` (`timeout = 120`, `timeout_method = "signal"`),
  `pytest-randomly`, `oracle` extra (pinned), `contract`/`oracle`/`benchmark` markers;
  **coverage `exclude_also` anchored** (`^\s*\.\.\.\s*$`, `:\s*\.\.\.\s*$`) — the unanchored
  pattern hides 248 statements and 96 branches today (Q3) — plus a layout test that every
  exclude pattern is anchored.
- Lock policy (S9, M5): Tier A `uv.lock` (universal, 3.11–3.13, dev + parquet + oracle);
  CI and Docker `uv sync --locked`; `integrity.yml` installs hash-pinned; image pinned by
  digest; Dependabot `uv` ecosystem, monthly. Tier B `envs/<platform>/requirements.lock`
  (`uv pip compile --generate-hashes`) for x86-cu128, jetson-jp62, p40-oracle — produced by
  Ian's spike, with `scripts/build_mmcv.sh` and the mmcv wheel sha. `envs/**`, `docker/**`,
  `Dockerfile`, `scripts/build_*.sh` join the enforcement surface.
- Workflows: every `uses:` SHA-pinned (TD-1); an `oracle` job (3.12); a `weekly` non-gating
  job: unit tests under `HYPOTHESIS_PROFILE=explore` and a CPU mmdet API-drift check (CPU torch,
  mmcv CPU wheel, mmdet 3.3.0, ATSS config with random weights on a synthetic image; TD-14,
  M12 stage 0; no registry access); `.github/rulesets/main.json` with a layout test that its
  check names equal the job names; layout tests: only a future `gpu.yml` may say `self-hosted`
  and only on `workflow_dispatch`; no `pull_request_target` workflow references the PR head.
  **No nightly `run-batch` workflow** (Phase 2; public repo).
- `qcal-lab` console script (S18); a `lab` subcommand of qcal is **not** mounted (it would import agent-owned
  code into `integrity.yml`'s trusted job). Import scan extended to `tests/**` and
  `benchmarks/**` with sklearn/scipy/numpy allowed only under `tests/oracle/**` and
  `benchmarks/**`; `qcal licenses` run inside the science interpreter in the planned `gpu-check` target.

#### PR-G — artifact store and model/dataset cards (signed, G1; after A2)

- `[artifacts] store_dir = "runs/artifacts"`, content-addressed `sha256/<2>/<64>`, read-only;
  on every record write (ok or failed) the registry copies or hard-links each artifact and the
  run log (log sha recorded); `ArtifactRef.produced_by`; `qcal registry verify-artifacts
  [--strict]` re-hashes stored files for every current record and checks cache-producer links;
  `DECISIONS.md` template gains "Artifact storage" (retention G4 + 12 months; one off-machine
  copy; nothing derived from Cityscapes leaves the machine) and "Cross-GPU cache policy";
  DVC wording changed to "planned" in `.gitignore`, `configs/lab.toml`, `docs/data/coco.md`.
- `qcal licenses`: requires `docs/models/<detector>.md` for every configured checkpoint
  (source URL, zoo config at mmdet v3.3.0, sha256, weights licence, training and pretraining
  datasets) and checks its datasets against the allowlist; dataset cards gain `sha256`;
  `qcal_lab status` checks card sha == `checkpoint_sha256` and dataset-card sha == annotations
  sha (M10). `.gitignore` gains `*.pth`, `*.pt`, `*.ckpt`, `checkpoints/`.
- Tests: unit; e2e flips an artifact byte → `verify-artifacts` FAIL (removes PR-E's xfail).

#### PR-F — G1 parity, bootstrap and GPU checks (agent-owned, G1 freeze Oct 31; needs D3/D6)

- Parity kind `"calibration"` (Alg. A.1) with **ε-optimality**, not grid indices (F9): the
  objective at our threshold ≤ the objective at the oracle's + ε, then final metrics; run with
  fit = select = the oracle's minival (`train_calibration` allows it) so the case tests the
  algorithm independently of D1; grid pre-registered from the oracle's thresholds; parity
  tolerance for Platt compares loss, not parameters (`parity.py:60-69` only tightens — relax
  for loss comparisons); `KINDS += "calibration"`; `status.py` counts it; NOT ASSESSED until
  both Ian's loop and his fixtures (ian_data) exist, and a mismatch is not attributed to either
  side by itself (N8).
- Synthetic protocol-separating parity inputs (D3/F5): small generated cases where ≥ vs >,
  greedy order, crowd handling and bin edges each give different answers; Ian runs the oracle on
  them and commits the outputs.
- Bootstrap harness (D6/F4, ownership per D8): paired image bootstrap over per-image outputs of
  Ian's loop (resampled datasets under fresh ids, one `metrics` call per replicate), 95% CIs
  vs published; ordering checks; per-class LaECE0 with CIs and
  per-class object counts per role (F15); refuses CIs over identical runs.
- `python -m qcal_lab check-eval-loop [--json]` + status item, sharing PR-E's contract checks
  (MUST asserted, SHOULD reported; metric names only).
- Detector sanity check (D2/F3): raw top-100 AP via pycocotools vs the zoo value, pass/fail,
  pre-declared, never a selection device.
- `scripts/gpu_check.sh` → the planned `gpu-check` target (M12 stage 1; Ian runs on GPU1): smoke with the
  GPU pinned; one-image test with the real checkpoint; the same GPU twice must be byte-identical;
  GPU0-vs-GPU1 agreement report; TF32 on/off delta; `qcal licenses` in the science interpreter.
  Fixes `scripts/nightly.sh` (`-I`, `CUDA_VISIBLE_DEVICES`, explicit interpreter) in passing.
- Tests: unit with the fixture loop and deliberately broken loops; parity README schema;
  `data-leakage-checker` on the harness.

#### After G1 (deferred; not cut, just later)

- **WP3b conventions**: `qcal_lab` logging through the Config (`--log-level`, `--log-format`,
  `--version`, DEBUG at decision points, smoke forwards the level), lab exceptions on
  `QcalError`, remaining hardcoded sites, duplicates onto `serialize`/`paths`, config-driven
  component module lists with the `qcal_lab.` prefix as a code constant, Platt/isotonic required
  kwargs, `splits … --json`, lab `config --check`.
- **WP4b reports**: `benchmarks/` outside `testpaths` (budgets in `benchmarks/config.toml`,
  report-only; measures `select_class_thresholds` at 1e5 × 80 × G ∈ {20, 100} — the SQE
  estimate is ≈1 min/run at 1e5, so TD-15 is not a G1 blocker, but G ≥ 100 under TD-12 would be
  fixed in pure Python first); the planned `mutate` target report-only on `calib/` then `integrity/claims.py`,
  with a committed triage baseline `tests/mutation/calib_baseline.toml` and a `mutmut` spike
  (src layout, `also_copy` for package data).
- **WP5 `/review-pr`**: advisory; `<base-ref>` argument; writes `review/claude/<branch-slug>.md`; runs
  `qcal ci review-check` unmodified and states that `claude/*` branches need the gemini review.
- **S17 protocols** to `qcal_lab` with the Phase 2 proposal; `docker/reference.Dockerfile` and
  the planned `reproduce-full` target at G4 (M13).

### Ian's list (ordered, dated)

1. **Oct 13 — S0** (1.5 h): key, `enforce`, `require_clean_tree`, `main`, ruleset by job name,
   PRs #1/#3, visibility, `qcal init`. Record G0 inputs as they land.
2. **Oct 14–17 — decisions D1–D6, D8** (1 h, written into `DECISIONS.md`/`EXPERIMENTS.yaml`
   drafts); identify the ATSS checkpoint (D2); start `docs/reference/kuzucu_eccv24.md` with the
   protocol section (D3) *before* writing the loop; accept or amend the `EvalLoop` v2 interface
   (PR-D2 proposal, 1 h).
3. **Oct 17 — Session 1** (3 h): read and sign PR-A1, PR-A2, PR-E's extras line; commit the
   session's `review/claude/*.md` files and one RESEARCH_LOG entry (CLAUDE.md wording change
   to "one entry per signing session" is in PR-A1 if Ian agrees).
4. **Oct 18–23 — environment spike** (science time): ATSS on GPU1 (mmcv build or
   `MMCV_WITH_OPS=0`), Tier B lock files + `scripts/build_mmcv.sh`, Jetson Python/TensorRT
   check (D10), P40 oracle environment freeze, `docs/models/atss_r50.md`. **G0 Oct 23.**
5. **Oct 24 — Session 2** (2 h): sign PR-B, PR-C, PR-G; review files; RESEARCH_LOG entry.
6. **Oct 24–31 — science inputs**: `handwritten/eval_loop.py` (v2) and `laece.py`; split
   decision and manifests (`{dataset}` namespace); fiveai outputs on the P40 incl. the synthetic
   protocol cases; `tests/parity/fixtures/`; Phase 1 cells. `make lab-status`, the contract
   suite and `check-eval-loop` report readiness. **Freeze Oct 31.**
7. **Nov 1–8 — G1**: the planned `gpu-check` target; registered runs; parity; `/reproduce-check`. Supply the
   peer-review finding numbering when convenient (PR-E's four tests).
8. Environment network policy: allow `huggingface.co` so `prior-art-scout` reaches the HF MCP.

### RACI (per PR)

Agent (Claude Code): drafts code, tests, proposal, review packet — **R**. `adversarial-reviewer`,
`data-leakage-checker`, `paper-reproducer`: **C** (consulted; their outputs are in the packet).
Ian: **A** (decides D1–D10, signs, writes RESEARCH_LOG and all ian_only/ian_data). Gemini
cross-review: **C** per CI's rule. Nobody else is **I**; the repository is the record.

### Top-5 delivery risks

| # | Risk | L/I | Mitigation stated in this plan |
|---|---|---|---|
| 1 | Tooling eats the hours G0/G1 need (loop, LaECE0, splits, oracle, spike) | Likely / High | 2 h/week cap; dated packets; cut line; post-G1 batch; `/weekly-review` tracks tooling vs science hours |
| 2 | A hashed format, key or record changes after the first registered run | Likely / High | Lock-now list in PR-D1 by Oct 24; golden tests; Compatibility section in every proposal; schema 1 only gains fields |
| 3 | G1 passes or fails for reasons unrelated to correctness (split design, AP set, checkpoint, false zero variance) | Likely / High | D1–D6 this week; replay detector decomposes the gap; oracle parity is the hard gate; bootstrap CIs and ordering checks |
| 4 | Control plane stays report-only or the ruleset is wrong | Likely / High | S0 first with exact job names; ruleset as JSON + layout test; no GitHub signed-commit rule; PR-A1 merges only after `enforce` |
| 5 | Environment surprises land late (Blackwell mmcv, Jetson Python 3.10, P40 CUDA 11.8, TF32) | Likely / High | Spike by Oct 23 covers all four; Tier B locks; `fp32_tf32_off` default; the planned `gpu-check` target before any registered run |

### Verification (end-to-end)

- Every PR: `make check` (lint, `mypy --strict`, full suite ≥ 95% with the anchored exclusion,
  integrity, smoke) green; `python -I -m qcal agent-layer` PASS; CI green; advisory review
  saved; traceability `--check` passes.
- PR-A1: `QCAL__PATHS__MANIFESTS_DIR=/tmp/x qcal registry run <cell> --seed 0` → exit 2, nothing
  written; same for a dirty `qcal.toml`, `--experiments other.yaml`, `GIT_DIR=…`,
  `git replace`; `qcal config --check` rejects a misspelled `signed_categories`; `QCAL__X__Y=1
  make smoke` passes; a record carries `policy_sha256` and `config_inputs_sha256`.
- PR-A2: a child that sleeps past `timeout_s` yields `failure_kind = timeout` with its
  environment recorded; Ctrl-C leaves a record; `res.*` columns appear in the index; the record
  names the GPU uuid under `CUDA_VISIBLE_DEVICES=1`.
- PR-E: `pytest tests/unit/test_calib_properties.py tests/oracle tests/contract` green with
  `ORACLE_REQUIRED=1` in the oracle job; TD-13/TD-16 closed in NEXT_STEPS with the oracle
  evidence; the contract suite reports NOT ASSESSED for the configured loop on the repo and
  passes on the fixture loop; the e2e lab→claims loop passes end to end.
- PR-D1: editing `calib/platt.py` leaves every cache key unchanged; `python -m qcal_lab dev`
  cannot read the evaluate manifest; a calibrator dict without an envelope is rejected; a
  `trt_jetson` cell is refused until a detector kind handles it; `fp32_tf32_off` is the default
  and the record shows every numerics flag.
- PR-D2: the fixture loop passes the v2 contract; "match once then threshold" ≡ re-matching;
  the replay detector reproduces a run's raw predictions byte-for-byte (native format).
- PR-B/C/G: `allow-only` denies the five chained/file-writing forms through the real wrapper;
  `uv sync --locked` in CI; every `uses:` SHA-pinned; `verify-artifacts` fails on a flipped
  byte; `qcal licenses` fails without a model card.
- PR-F: parity case kind runs with the fixture loop (ε-optimal) and reports NOT ASSESSED on the
  repo; the planned `gpu-check` target on GPU1 shows two-pass byte identity and the TF32 delta.

### Peer review record

**Round 1 — `adversarial-reviewer` on draft 1: revise (B1–B6, N1–N9).** All applied: policy at
HEAD (now with git-env scrubbing per Q2), `--experiments` lock, `allow-only` stays exact,
`hit_iou` fixture-only, advisory review language and no `--reviewer`, gates per WP and WP2
split, strict envelopes, benchmark config out of lab defaults, roles in code, `schema_version`
key deleted, serial A1 → rest, `timeout_method = signal`, `runpy` instead of omit, base-ref
argument, go/no-go on new Phase 1 work, run-date file name, nightly dropped.

**Round 2 — four-lens panel on draft 1 (dispositions).**

*ML research scientist (F1–F18).* Accepted: F1 (D1 + `split_design`), F2 (D4, `EvalLoop` v2),
F3 (D2, sanity check), F4 (D6, bootstrap, ordering, tables refusal), F5 (D3, synthetic cases),
F6 (oracle suite, in-sample identities, label-map check; loss-based parity tolerance), F7
(`fp32_tf32_off`, flags, gpu-check), F8 (v2 two-layer design), F9 (ε-optimality), F10 (freeze
list; mixed-hash refusal), F11 (per-role source, replay detector), F12 (header fields, reserved
row fields), F13–F16, F18. Noted for Ian's pre-registration: F17. Not adopted as stated: the
first-draft WP6 grid-index comparison (replaced by F9).

*MLOps/reproducibility (M1–M15).* Accepted: M1 (A2 + D1), M2 (A2/D1 + `require_clean_tree` in
S0), M3 (D1), M4 (PR-G), M5 (PR-C Tier B + spike), M6 (D1), M7 (D1 + D9 measurement), M8 (A2),
M9 (A2), M10 (PR-G), M11 (D1 dev path; `--dry-run` exempt), M12 (PR-C stage 0; PR-F stage 1;
stage 2 only after the visibility decision), M14, M15. Modified: M13 — reference Dockerfile
deferred to G4; `container_image` reserved; `oracle.environment` added to parity cases.

*SQE/test architect (Q1–Q14).* Accepted: Q1 (PR-E), Q2 (A1), Q3 (PR-C), Q4 (PR-E; shared
implementation with PR-F's command), Q5 (PR-E), Q6 (benchmark deferred; TD-15 not a blocker;
threshold-search fix only if G ≥ 100), Q7 (PR-E + PR-C), Q8 (PR-E), Q9 (PR-E/PR-C; hash seed
and TZ tested not pinned), Q10 (a→A1, b→D1, c/e→PR-B, d→PR-B, f→xfail boundary tests,
g→no change), Q11 (post-G1 with spike and committed baseline), Q12–Q14 (behavioural
round-trips; numerics-focused additions).

*Architect/SDLC lead (S1–S21).* Accepted: S1, S2, S6, S8 (D1), S3 (dated sessions, hours, cut
line, RACI, review files per session; CLAUDE.md RESEARCH_LOG wording offered to Ian), S4 (S0:
job names, ruleset JSON, no GitHub signing rule, rename before branching), S5 (A1 qcal side;
lab side post-G1 except the roles check in D1), S7 (A1/A2), S9 (PR-C), S10 (D10), S11 (boundary
test, `sha256_file` moved, fragments + tags, one version post-S1), S12 (reordered as above),
S13 (A1 simplified), S14 (PR-0), S15 (PR-G), S16 (A2 + D1), S18 (PR-C; no mount), S19 (PR-B),
S20 (PR-0; CODEOWNERS documentation-only), S21 (D8). Deferred with reason: S17 (Phase 2
proposal; speculative without the Phase 2 code). Premise corrections adopted in Context.

**Rejected outright:** none. **Still open for Ian:** D1–D10; the second detector (D-DETR vs
DETR); COCO-C comparability; rule 5's "interpretation" path; the peer-review numbering.
