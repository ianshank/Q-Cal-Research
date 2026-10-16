# change: qcal-foundations

## Status
Proposed, Oct 14, 2026. Edits the enforcement surface (`src/qcal/`, `.claude/agents/`,
`pyproject.toml`, `qcal.toml`, the packaged defaults), so it needs Ian's signed commits, in
signing session 2 (Oct 24). Cycle plan: `docs/changes/cycle-2026-10-g1-readiness.md` (PR-B,
the wave-2 follow-ups, and the parts of PR-F that need no decision).

Rev. 2 of the plan, after a four-seat review (ML research, MLOps/reproducibility, SQE/test
architecture, architecture/SDLC/security). Every seat blocked rev. 1; the findings that changed
the plan are named in **Decisions**.

The work was developed on `claude/qcal-foundations-wip` (draft PR #5, "CI only"). On Oct 16
PR #5 was merged into PR #4's branch, so wave 3 now rides PR #4 and is signed with it: PR-B1
(S1, S4, S5 and their follow-ups) and PR-B2 (S2, S3 and the guard follow-ups) are signing
ranges within PR #4, not separate pull requests. Rev. 3 follow-ups, including the signed S6,
are on `claude/qcal-foundations-r3`, a CI-only draft against PR #4's branch.

## Gate served
G1 (Nov 8; freeze Oct 31). The first registered run (Nov 1) must not:
- lose an interrupted run's artifacts (N1);
- refuse a launcher because an audit is running (N8);
- average seeds whose numbers came from different code (N4);
- reuse stale cached predictions after an mmcv rebuild (N15).

Ian also needs `check-eval-loop` and the synthetic protocol cases between Oct 24 and 31.

## Why
Verified against `50efe97`:

- **Coverage.** The exclusion `"\\.\\.\\."` (`pyproject.toml:69`) is unanchored. A `def` whose
  signature says `tuple[str, ...]` drops its whole body from coverage: 30 functions, among
  them `Policy.categories_for` and `clean_room_hits`, `splits.draw` and the prediction-cache
  path. Anchored, the suite measures 7048 statements at 99.44%.
- **CLI.**
  - A misspelt `signing.mode` or `review.mode` raises `ValueError`, which exits 1, not 2
    (`signatures.py:197-199`, `review.py:119-121`).
  - `--github --json` prints annotations and JSON on one stdout, so the JSON is invalid
    (`cli/ci.py:72-80`).
  - `registry cells --emit --json` ignores `--json` (`cli/registry.py:135-140`).
  - `tables.py:241-245` tells users to rerun with a `--rerun` flag that `registry run` lacks (only
    `run-batch` has it).
- **Guards.**
  - `allow_only` tokenises `git diff\nsrc/qcal_lab` like `git diff src/qcal_lab`
    (`guards.py:261-273`).
  - There is no hard-link check.
  - Clean-room matching is a plain lowercase substring test (`policy.py:83-85`), so a
    full-width or zero-width spelling passes.
  - `prior-art-scout` may read `ian_data`.
- **Agent layer.** `mcpServers` is not checked against `.mcp.json` and does not count as
  network. `isolation` and `maxTurns` values are not validated. Nothing requires a network
  agent to deny-read Ian's files.
- **Interrupts (N1).** `_wait` terminates the program and re-raises (`executor.py:296-299`), so
  the envelope is never read. The runner then records an empty result (`runner.py:442-447`).
- **Locks (N8).** The audit probes each pair lock with `LOCK_EX|LOCK_NB` (`audit.py:198-207`).
  A launcher taking that lock at the same moment is refused (`runner.py:407-413`), and
  `run_batch` aborts even under `--keep-going`.
- **Table identity (N4).** Tables compare only `config_inputs_sha256`
  (`aggregates.py:48-54`). Seeds run under two versions of Ian's loop, or two versions of the
  calibrators, are averaged without complaint.
- **Hashed inputs (N12).** `require_hashed` (`qcal_lab/config.py:109-132`) and `config_inputs`
  (`gates.py:186-199`) decide "hashed" differently: case sensitivity, symlinks, table specs.
- **Cache (N2, N15).**
  - Bytes no record vouches for block their key for ever.
  - The cache key misses `formats.py`, the header that `experiment.py` builds, and the mmcv
    build: a source rebuild keeps the version string.
- **Label map (N14).** MMDetection's `init_detector` substitutes COCO class names when the
  checkpoint has none, so the label-map check compares COCO with COCO.

## What changes

### Enforcement surface (signed)
- **S1: anchored coverage, minimal CLI fixes.**
  - Coverage:
    - `exclude_also` uses `^\s*\.\.\.\s*$` and `:\s*\.\.\.\s*(#.*)?$`;
    - a layout test keeps every pattern from matching a function's signature lines.
  - CLI:
    - `ci.MODES` and `ci.check_mode`, which raise `ConfigError` (exit 2);
    - `--github` and `--json` are mutually exclusive;
    - `cells --emit --json` emits JSON;
    - the tables message names real commands;
    - `registry_immutable` logs each violation.
  - Public `leakage.split_digest`, which refuses ids containing a newline.
- **S2: guards.**
  - `allow_only` refuses control characters.
  - The reproducer's exact `git diff`/`git log` forms.
  - `prior-art-scout` denies reads of `ian_data`.
  - Clean-room matching folds compatibility forms: strip the configured categories (`Cf`,
    `Mn`), NFKC, map dashes to `-`, casefold when `case_insensitive` is set, NFKC again.
    Confusables are not folded (a Cyrillic `с` for a Latin `c` still passes; review N8).
  - `guard_paths` and `deny_read` refuse a file with more than one hard link.
- **S3: agent layer.**
  - `mcpServers` must name `.mcp.json` servers, and they count as network.
  - `isolation` and `maxTurns` are validated.
  - Network agents must wrap `deny-read` around Read, Grep and Glob with the configured
    categories.
- **S4: launcher (N1, N8).**
  - `_wait` never raises. After an interrupt, further signals escalate to SIGKILL and are
    absorbed. The envelope is read with a non-zero code: partial artifacts and the
    environment are kept, metrics are dropped.
  - Each launcher holds an exclusive lock on its own in-flight marker, and the audit probes
    that lock, never the pair lock. A launcher retries a busy pair lock briefly.
  - `--keep-going` skips a busy pair.
  - `provenance.log_sha256`.
- **S5: identity (N4, N12).**
  - `provenance.definition_sha256`: the run's definition without its seed.
  - `registry.identity_columns`: packaged default `["config_inputs_sha256"]`, today's
    behaviour. `qcal.toml` proposes a science-scoped set for Ian.
  - Fail-closed checks. Audit, tables and claims share the comparison.
  - `gates.hashed_input`: membership in `config_inputs`.

### Agent-owned
- **Coverage baseline and mutation harness.** Done in the first commit: the tests for what the
  anchored exclusion revealed, `scripts/mutation_check.py` and `tests/mutation/defences.toml`.
- **F3, the lab's side of the follow-ups.**
  - N1: a pessimistic envelope after each artifact.
  - N12: `require_hashed` uses `gates.hashed_input`.
  - N4: `raw_predictions_sha256` per role.
  - N2: quarantine of unvouched bytes, only when their producer is provably not running.
  - N14: classes read from the checkpoint before `init_detector`, with an attestation key for
    checkpoints that carry none.
  - N15: `formats.py`, the header and `mmcv._ext` join the cache key.
- **PR-F-now.**
  - Synthetic protocol cases under `tests/parity/cases/`, written by
    `python -m qcal_lab protocol-cases` (`--check` reports drift; the suite runs it). Seven
    cases, each with the alternatives it separates and what each predicts (match outcomes,
    bins, classes in the mean, the kept threshold). The README says how Ian records the
    oracle's answer under `tests/parity/fixtures/protocol/`.
  - `python -m qcal_lab check-eval-loop [--json] [--fixture] [--strict]`, with PR-E's contract
    checks moved into `src/qcal_lab/eval_contract.py` (contract version 1). `status` gains an
    "evaluation loop contract" item once Ian's loop loads.
  - A detector sanity-check scaffold (`src/qcal_lab/sanity.py`): `zoo_ap`, `tolerance` and
    `source` come from Ian's declarations file (Decision 11; D2); the AP measurement stays off
    until D8. `status` reports "detector sanity check <name> declared", TODO until declared
    and NOT ASSESSED in its detail until D8.

## Decisions
1. **Branch flow.** Develop on a work-in-progress branch, then rebase onto `civ` under a new
   name once PR #4 lands.
   - *Alternative:* stack on PR #4 and merge `civ` in later. Signing re-creates PR #4's commits
     (`CONTRIBUTING.md:86-110`), so the unsigned originals would stay in `civ..head`, fail
     `verify-signatures` under enforce, and need a force-push that agents may not make.
2. **A minimal signed PR-B.**
   - *Alternative:* the full `serialize`/`paths`/`errors` sweep, about 25 signed files. That is
     2.6-3.4 h of Ian's reading in a 2 h session, and it rewrites the record-write and
     `config_hash` paths a week before the first registered run.
   - *Consequence:* the sweep, `exit_code_for`, the command-parser families and the
     config-key drift check (S19) move to WP3b, after G1. When `errors.py` lands, its exit
     codes live in a leaf module: importing them from `qcal.cli` would create an import cycle
     that fails every hook.
3. **Hard links: deny `st_nlink > 1`.**
   - *Alternative:* enumerate the work tree with git and match inodes, which is what the cycle
     plan spelled out. That is slower, needs git inside the hook, and fails closed on most real
     cases anyway.
   - *Consequence:* hard-linked files (a uv link-mode `.venv`, PR-G's future store links)
     cannot be edited through tools. `cp` breaks the link.
4. **Unicode folding applies only to clean-room matching.** Path protection is unchanged: a
   full-width `EXPERIMENTS.yaml` is another file. The normalization form is validated against
   a code-constant set, so a typo is a configuration error, not a crash in every hook.
5. **Interrupts.** The audit no longer probes pair locks.
   - *Alternative:* probe with `LOCK_SH`. That does not help: a held shared lock still refuses
     an exclusive one.
6. **Table identity is Ian's decision.**
   - The packaged default keeps today's behaviour.
   - `policy_sha256` (raw `qcal.toml`) and `experiments_sha256` (the whole file) stay audit
     findings and never refuse a table. Refusing on them would freeze G1's tables after any
     comment edit or new cell, while leaving what changes numbers (Ian's loop, the lab code)
     unchecked.
   - The proposed set in `qcal.toml` covers:
     - the run's definition without its seed;
     - Ian's loop digest;
     - the lab source digest;
     - the dataset digest;
     - the raw-prediction digests.

     A test runs several seeds of one cell and requires each column to agree, so no
     seed-dependent value can be chosen.
7. **Cache quarantine (N2) only when the producer is provably not running.** That means it has
   no live in-flight marker, under a per-key lock. *Alternative:* "no record yet" counts as
   unvouched. That would quarantine a running producer's bytes, and the producer's own record
   would then fail when its artifacts are re-hashed.
8. **Checkpoints without class names (N14)** are refused unless Ian's declarations file
   attests their label order: `label_order_attested_sha256` is
   `qcal_lab.models.mmdet.label_order_digest(checkpoint_sha256, category names)`, so it holds
   for one checkpoint and one order, and `checkpoint_sha256` must be set. The refusal names
   the formula, never the value: Ian computes it after checking the order by hand (D2).
9. **PR-F-now.** Only pieces that need no D1-D8 decision and no EvalLoop v2 code. Expected
   outcomes are match predicates, not metric values. The bootstrap, the parity calibration
   kind and `gpu_check.sh` stay in PR-F.
   - The case settings live in `tooling.toml`, not the hashed lab defaults: they change no
     run, so `_effective_digest` and `config_hash` do not move for them.
   - The sanity status item reads "declared": what must precede a registered run is the
     declaration, so the value cannot be chosen afterwards. The check itself stays NOT
     ASSESSED until D8 and is never reported as passed.
10. **Mutation evidence** comes from `scripts/mutation_check.py`, committed and reproducible.
    It replaces session-local scripts. A kill counts only when a named test fails.
11. **Ian's declarations live where agents cannot write** (advisory review B1). The sanity
    values and the label-order attestation are read from `[declarations] file`
    (`docs/reference/declarations.toml`), which `qcal_lab.declarations` accepts only if
    `qcal.toml` both denies it to edit tools and requires a signed commit for it (`ian_data`
    today). *Alternative:* `tooling.toml` or `configs/lab.toml`, which agents maintain, so an
    agent could paste the value a refusal printed and the check would pass unseen. Ian
    creates the file:

    ```toml
    [detectors.atss_r50]
    zoo_ap = ""             # the published value, in `units`
    tolerance = ""          # in `units`
    units = ""              # "fraction" (pycocotools) or "percent"
    source = ""             # where the zoo value is published
    checkpoint_sha256 = ""  # must equal detectors.atss_r50.checkpoint_sha256
    image_set = ""          # the images the zoo value was measured on
    test_cfg = {}           # must equal the configured score_threshold and max_per_image
    on_fail = ""            # "stop": stop and debug; never swap the checkpoint
    label_order_attested_sha256 = ""   # only for a checkpoint without class names
    ```
12. **The sanity declaration is enforced and bound** (panel round 2, rev. 3). A non-fixture
    detector does not run until its declaration is complete, names the configured
    checkpoint's sha256 and test-time settings, and states `on_fail = "stop"`; the refusal
    comes before any detector work. Values are compared as written (decimal), in the declared
    units, so 0.8 is within 0.1 of 0.7 and a percent value cannot pass as a fraction; a
    tolerance above `[sanity] max_tolerance_fraction` (`tooling.toml`, a plausibility cap) is
    refused. The declarations file's digest is recorded in `inputs_read`. *Alternative:*
    enforcement only in `lab-status`, which a launch never consults.
13. **Contract additions** (panel round 2, rev. 3).
    - **MUST:** the loop's results do not depend on the order the images are listed in.
    - **SHOULD** (reported, never a failure; the objective is Ian's):
      - an objective under which keeping no detections beats keeping them all, which drifts
        a threshold search to the top of its grid;
      - all-zero targets;
      - results that change when images without detections are left out.

      The panel proposed the last one as a MUST. It is a SHOULD here because the pipeline
      always passes every image (`keep_class`), so a loop that relies on that is correct
      under the contract. Requiring it would test a call the pipeline never makes.
    - The fixture loop now counts missed objects and images from the ground truth, so it
      is clean under all three; its values on the smoke project are unchanged.

## Compatibility
- **Exit codes.**
  - A misspelt `signing.mode` or `review.mode` exits 2 (was 1).
  - `--github --json` together exits 2 (was invalid JSON).
  - No other exit code changes.
- **Records.** Schema 1 gains optional fields only: `provenance.log_sha256`,
  `provenance.definition_sha256`, and marker `host`/`boot_id`. Record bytes are pinned by a
  new test.
- **Index.** New core columns are added for the identity set; `qcal registry index`
  regenerates the index. No index is committed yet.
- **Goldens.**
  - The prediction-key composition golden moves: the header and `mmcv_ext_sha256` join the key.
  - `_effective_digest` moves: the N14 key.
  - `config_hash`, `cell_id`, `split_digest`, the predictions bytes and the record bytes do not
    move.
- **Cache.** Cached predictions miss once (N15). No registered run exists yet.
- **Lab commands.** `python -m qcal_lab` gains `check-eval-loop` and `protocol-cases`.
  `status` gains the contract, sanity-check and prediction-cache items.
- **Configuration.** New keys, each with a packaged default:
  - `policy.unicode_normalization`, `policy.strip_unicode_categories`;
  - `agent_layer.isolation_values`, `agent_layer.max_turns_limit`,
    `agent_layer.network_agent_deny_read`;
  - `registry.lock_wait_s`, `registry.lock_poll_s`, `registry.identity_columns`;
  - in `tooling.toml`: `[protocol_cases]` and `[declarations] file`.
- **Prediction cache sidecar.** Gains `producers` (every run that put the bytes); old
  sidecars read as their one `produced_by`.
- **Agents.** The reproducer gains five exact commands. The scout loses read access to
  `ian_data`.
- **Operations.**
  - `inflight_dir` must be on a host-local filesystem: `flock` over NFS without lockd, or over
    sshfs, is local only.
  - A container's stop timeout must exceed `executor.kill_grace_s` plus the collectors, or the
    launcher dies before it writes the record. Docker's default of 10 s is too short.

## Out of scope
- PR-C (build alignment, waits on #3).
- PR-G (artifact store and cards). In the meantime:
  - `provenance.log_sha256` hashes each run's log;
  - Ian copies `runs/{cache,results,logs}` off the machine after each session, checking the
    sha256s against the records;
  - PR-G gains a `--ingest` backfill for runs made before it lands.
- WP3b (`serialize`, `paths`, `errors`, command-parser families, S19). Amended: the lab's
  prediction-path formats stay where they are, because moving them would re-key every cached
  prediction.

## Known limits (advisory review, not fixed here)
- **N3.** A Grep over a directory that holds a hard link to an `ian_data` file returns its
  content: the hard-link refusal judges the named path only. Hard links into the repository
  are themselves refused for edits.
- **N6.** A marker is created before it is locked, so an audit in that instant can report a
  live run as stale. The audit only reports; creating the marker as temp, lock, then link
  would close the window (WP3b).

## Ian decisions requested
1. **By Oct 24:** the identity set in `qcal.toml` (`registry.identity_columns`). Keep the
   proposed science set, narrow it, or add `policy_sha256`. The advisory review raised two
   points for this decision (N1, N2):
   - it leaves out result-changing inputs: the third-party packages Ian's loop uses
     (pycocotools, numpy, scipy), the Python version, and a loop configured outside
     `src/qcal_lab/handwritten/`; `env.eval_loop` plus a packages digest would cover them;
   - `env.qcal_lab_source_sha256` hashes every file of the package, tooling and status code
     included, so declaring a sanity value or a docstring edit stops seeds from pooling, and
     `definition_sha256` holds the cell id, so `tables.allow_pooled_cells` cannot pool under
     this set.
2. **By the environment spike (Oct 18-23):** whether the ATSS checkpoint carries class names.
   If it does not, attest its label order in `docs/reference/declarations.toml` (Decision 11;
   D2). Declare the sanity check there too: `zoo_ap`, `tolerance` and `source`.
   Before running the oracle on the protocol cases, confirm `[protocol_cases]`
   `iou_threshold`, `max_detections` and `bin_count` match the oracle's.
3. **Oct 17:** merge PR #4 with "Create a merge commit". GitHub's squash or rebase merge
   re-creates the commits with GitHub as committer, which drops the signatures. Disable those
   two buttons in the ruleset.
4. **Oct 24:** file this branch's advisory review under `review/claude/` at the signing session.
   It is a `cross_review` path, so it is a signed commit.
5. **After each session, until PR-G:** the interim artifact copy above.

## Ian hours
About 1.8: S1 0.2, S2 0.3, S3 0.2, S4 0.5, S5 0.2, agent-owned commits 0.25, signing 0.15.
Measured after the review fixes: 874+/144- lines on the signed surface, about 1.6 h at
~650 lines/h (PR-B1 about 1.1 h, PR-B2 about 0.5 h), plus S6.

## Acceptance
- [ ] `make check` green with the anchored exclusion (coverage ≥ 95)
- [ ] `qcal agent-layer` PASS
- [ ] `python3 scripts/mutation_check.py`: every entry KILLED
- [ ] `qcal ci verify-signatures --base <PR #4 head> --head HEAD` lists exactly S1-S5
- [ ] advisory adversarial review: blocking findings fixed or answered here
- [ ] the other model's review at `review/gemini/claude-qcal-foundations.md` passes
      `qcal ci review-check` once the real pull request is open
- [ ] RESEARCH_LOG.md entry (Ian, session 2)

## Tests
- **`tests/unit`:** one test per rule, plus property tests for Unicode folding and
  `split_digest`.
- **`tests/security`:** through the real hook wrapper:
  - control characters;
  - the reproducer's commands;
  - hard links and symlinks to them;
  - folded clean-room spellings;
  - the scout's reads.
- **`tests/regression`:**
  - `test_registry_run_contract.py` for N1 and N8;
  - `test_registry_run_gates.py` for N4 and N12;
  - `test_run_identity_and_formats.py` for the lab follow-ups.
- **`tests/integration/test_repository_layout.py`:** the anchored exclusion; the workflows'
  `qcal` commands parse.
- **`tests/unit/test_lab_eval_contract.py`, `test_lab_protocol_cases.py`,
  `test_lab_sanity.py`:** the PR-F pieces; each protocol case's geometry is checked with exact
  fractions.
- **`tests/mutation/defences.toml`:** one mutant per defence.

## Tasks
Proposal → coverage baseline and harness → S1-S5 (signed by Ian, session 2) → F3 and
PR-F-now (agent) → advisory review → rebase onto `civ` after PR #4 → the real PR.
