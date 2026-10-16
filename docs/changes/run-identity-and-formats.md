# change: run-identity-and-formats

## Status
Proposed, Oct 11, 2026. Agent-owned (`src/qcal_lab/`, `configs/`, `tests/`); no
enforcement-surface edit. Cycle plan: `docs/changes/cycle-2026-10-g1-readiness.md` (PR-D1).
It builds on the run contract (`docs/changes/registry-run-contract.md`, PR-A2) and lands in
the same draft PR as separate commits that need no signature.

## Gate served
G1/K1 (Nov 8). What a cached prediction, a calibration file or a run record means must be
fixed before the first registered run writes one. After that, every change to a key or format
costs GPU or Jetson reruns.

## Why
Verified against `8c76044`:

- **The cache key covers too much and too little.**
  - Too much: `source_digest` hashes every `.py` and `.toml` file in `qcal_lab`
    (`experiment.py:245-256`), and `detector_fingerprint` adds `__version__`
    (`models/base.py:89`). A calibrator docstring edit therefore forces new GPU inference.
  - Too little: it records no target, GPU, CUDA, cuDNN, numerics or image-decoding library.
  - It includes the absolute `images_dir` path, so the same images on another machine miss.
- **The cache trusts its sidecar.**
  - A hit accepts any `produced_by` (`predictions.py:196-210`), and `put` overwrites
    (`:212-230`).
  - The detector is built before the cache is consulted (`experiment.py:391-395`), so a fully
    cached run still needs a GPU.
- **Three pre-registered factors are validated and then ignored.** `target`, `fit_precision`
  and `threshold_regime` (`experiment.py:120-164`) are checked but never used. A `trt_jetson`
  cell would silently reuse torch FP32 predictions under that label.
- **Formats have no declared version.**
  - The predictions header cannot tell FP32 from INT8 (`predictions.py:39-51`).
  - Calibrator and calibration files have no envelope, and `calibration.json` is written
    non-atomically (`experiment.py:481-486`).
- **"fp32" is not one regime.** Plain `fp32` keeps torch's defaults, which on recent GPUs
  means TF32 convolutions. cuDNN benchmark and deterministic flags,
  `float32_matmul_precision`, `CUBLAS_WORKSPACE_CONFIG` and `NVIDIA_TF32_OVERRIDE` are neither
  set nor recorded (`models/mmdet.py:56-66,149-161`).
- **Identity hygiene.**
  - The `[smoke]` and `[detectors.fixture]` tables are part of every run's `config_hash`,
    because the whole defaults file is hashed.
  - Artifact file names are scattered literals.
  - No test pins a digest, so an identity change can land unnoticed.

## What changes
In order, each step a separate commit:

1. **Goldens** (`tests/regression/test_run_identity_goldens.py`). Today's `cell_id`,
   `split_digest`, `rank_key`, `draw`, `config_hash`, fixture bytes, predictions bytes,
   `cache_key`, effective lab configuration digest and fixture-detector output are pinned.
   Every later change to one of them re-pins it in the same commit and is listed under
   Compatibility below.
2. **Roles and identity hygiene.**
   - `target`, `quant_path`, `fit_precision`, `threshold_regime` and `split_design` become
     roles. `split_design` supports only `disjoint` until Ian decides D1.
   - The keys of `factors.supported` and `factors.defaults` must be roles.
   - A detector kind declares the targets it can produce, so a `trt_jetson` cell is refused.
   - `RunPlan.roles()` names what each role (fit, select, evaluate) reads: dataset,
     detector, precision, target and source. Phase 1 gives every role the same.
   - `[smoke]` and the fixture loop's hit IoU move to an unhashed `resources/tooling.toml`.
     The fixture loop reads its hit IoU when it is built, never while it evaluates.
   - Two tables stay hashed. `[parity]`, because its tolerance is a G1 gate value.
     `[detectors.fixture]`, because it decides the fixture runs' predictions, so it is an
     input of those runs; it is looked up like every other detector.
   - Artifact file names become code constants.
3. **Strict formats.** One module (`qcal_lab/formats.py`) writes and checks every envelope.
   - Predictions version 2. The header gains a `source`: precision, target, quant path,
     shift, score definition, model sha256 and the `test_cfg` overrides the detector applies.
     The overrides are recorded, not the merged MMDetection values: those follow from the
     overrides plus the config file, whose sha256 the detector fingerprint already records,
     and resolving them would need MMDetection on every cache hit. A source is computed from
     configuration alone, so a cache hit never builds the detector (step 5).
   - The header lists the row fields. Rows reserve `source_index`, `class_logit` and
     `aux_logit`, written as null; a reader refuses a value there rather than drop it.
   - `cache_key` refuses parts named `format` or `version`, and covers the source.
   - Envelopes with a format name and version: calibrator, calibration (written atomically),
     cache sidecar and parity case. A file without its envelope is refused; a cache sidecar
     without one is a miss.
   - The calibrator envelope also records Platt's solver settings and whether the fit ended
     on the a = 0 bound, per class. A parity case records `oracle.environment`.
4. **Numerics and the program's environment.**
   - `[numerics.regimes.<precision>]` names each regime's torch and cuDNN settings.
     - `fp32` keeps torch's defaults, as today.
     - `fp32_tf32_off` also turns off cuDNN benchmark, turns on deterministic algorithms and
       requires `CUBLAS_WORKSPACE_CONFIG`.
   - The default stays `fp32`. Switching to `fp32_tf32_off` is Ian's decision D5 and one line
     of configuration.
   - A precision without a regime table is refused before any detector work, and a regime
     whose `required_env` is unset refuses before anything is switched.
   - The program records:
     - every effective switch, `NVIDIA_TF32_OVERRIDE` and `CUBLAS_WORKSPACE_CONFIG`;
     - the device it used (name, uuid, compute capability), and the CUDA and cuDNN versions;
     - a digest of torch's build configuration, the package list (a `qcal_lab.packages`
       artifact), and the handwritten files' digest;
     - where `qcal` and `qcal_lab` were imported from. With `registry.require_code_in_root`
       both must come from `paths.source_dir`, as the launcher requires of `qcal`.
     A lock-file digest waits for PR-C, which adds the lock file.
   - An MMDetection detector must set both test-time settings, so the recorded `test_cfg` is
     all that applied. `configs/lab.toml` pins `max_per_image = 100`. `score_threshold` is
     Ian's to set (F10), so an `atss_r50` run refuses to start while it is unset, and
     `python -m qcal_lab status` lists it.
   - The model's class names must equal the dataset's, in label order; a model without class
     names is refused.
5. **Cache key and producer.**
   - The key covers only the modules that produce predictions: `models/`, `data/coco.py`,
     `data/fixture.py`, `predictions.py` and `numerics.py` (`PREDICTION_MODULES`).
   - It drops `__version__` and the absolute images path: the images' bytes count, not
     where they live.
   - It adds the predictions source (target, quant path, `test_cfg`), the requested numerics,
     and opencv, pillow and mmcv versions. Each detector kind registers what beneath its
     settings decides its predictions (`KEY_PARTS`): for MMDetection the compute capability,
     CUDA, cuDNN, torch's build digest and the resolved config, whose `_base_` files the
     config file's own digest misses. The fixture registers nothing.
   - The detector is built only on the first cache miss; the run records whether it was.
   - A hit counts only if the producer's record lists the same path and sha256. Science code
     may not import `qcal.registry.store` (it writes records), so the check reads that one
     record as `<registry_dir>/<run_id>.json`; a test pins this to the store's layout.
   - `put` creates files exclusively and never replaces other bytes under a key: records list
     cached files as artifacts, so replacing one would break their hashes. Identical bytes
     are kept and their sidecar names the new producer. On other bytes the run keeps its own
     predictions in its artifact directory and records the conflict.
6. **Smoke isolation.** Smoke's experiment program runs as `{python} -I`, like registered
   runs, and a test keeps smoke's command identical to the repository's `executor.command`.

## Decisions
- **Goldens before changes.** Alternative: rely on the existing behavioural tests. Rejected:
  they compare a digest with itself, so they cannot see a digest change.
- **Narrow key over a whole-package digest.** A whole-package key forces inference after any
  edit. Alternative: hash the whole package minus `calib/`. Rejected: the next new module
  would silently enter the key. The key's module list is a code constant, with a test that
  editing `calib/` leaves the key unchanged and editing `models/` changes it.
- **Strict formats, no legacy readers.** No registered run or committed artifact exists yet, so
  there is nothing to stay compatible with. After the first registered run, formats only gain
  fields.
- **`fp32` stays the default.** D5 is Ian's call; the regime table makes it one line.
- **Producer check against the registry.** Alternative: trust the sidecar. Rejected: the cache
  is gitignored and agent-writable, and a forged sidecar would make a cached file look as if a
  registered run produced it.

## Compatibility
- Goldens re-pinned by this change, each named in the commit that changes it:
  - effective lab configuration (tables moved out, regimes added);
  - predictions bytes (version 2 header);
  - `cache_key` (format version, then parts and module list);
  - the fixture detector's output, once: it is now pinned over the detections themselves,
    so later format changes leave it alone. Re-encoded in the version 1 layout, the same
    detections still hash to the earlier golden.

  `cell_id`, `split_digest`, `rank_key`, `draw`, `config_hash` and the fixture bytes do not
  change.
- Config keys: new roles under `[factors]`, `[numerics.regimes.*]`, and a packaged
  `resources/tooling.toml` (`[smoke]`, `[fixture_eval]`). `[smoke]` leaves the hashed
  defaults; a `[smoke]` table in `configs/lab.toml` is refused with a pointer to the new file.
  Tests override smoke settings through `run_smoke(..., settings=...)`.
- Records: the program's environment gains keys; schema 1 only gains fields.
- Cache: every existing cache entry misses once. No registered run exists.
- Parity fixtures: none committed yet; the parity-case envelope applies from the first one.

## Adversarial review (advisory) and what changed
`adversarial-reviewer` on `f57816f`: **block** (full report in
`review/claude/claude-sdlc-agents-implementation-plan-gmb29t.md`). The program's blocking
findings, each fixed with a test that fails without its fix:
- **B1 (program side).** Every artifact the program lists carries the sha256 of the bytes it
  wrote or read; the launcher compares it. A cache hit reads the file once: those bytes are
  hashed, checked against the producer's record and parsed (`loads_predictions`), so a file
  swapped after the sidecar check is never used.
- **B2.** The supported-value check looked roles up by factor name, so under a rename
  (`[factors] split_design = "split"`) it checked nothing. It now checks each role's
  effective value, and a cell may use factor names only.
- **B3.** `NVIDIA_TF32_OVERRIDE` and `CUBLAS_WORKSPACE_CONFIG` change results beneath torch
  and were recorded but not keyed; the cache key now includes them.
- **B4.** `seed_effective` was true whenever a fit size was set. It is now true only when the
  draw is a proper subset of the fit split and the calibrator reads fit data
  (`CalibratorKind.uses_fit_data`, false for the identity).

Non-blocking findings applied:
- N3: exclusive writes work without hard links, and temporary names are unique across
  containers.
- N9: smoke's byte-for-byte check leaves out the package list, which describes the
  interpreter.
- N10: switches are recorded after the model is built.
- N11: the resolved config is hashed as canonical JSON, independent of yapf.
- N13: a golden pins the cache key's composition.
- N14: a test assertion that compared a value with itself.
- N15: a stale comment.

Follow-ups:
- N2: cache bytes no record lists still block their key. They could be quarantined and
  replaced.
- N11: a cache hit still needs torch, mmengine and the same compute capability, by design:
  they are in the key.
- N14: real `init_detector` substitutes COCO class names when a checkpoint has none. The
  label-map check then compares those names with the dataset's.
- N16: `_RESOLVABLE_ULPS` is recorded in the solver settings but is not configuration. The
  fixture loop's hit IoU is unhashed tooling that decides the smoke runs' fixture metrics.

## Out of scope
- The split design (`paper`) waits for D1 and its rule 3 rewording.
- Dataset kind, split namespace and label-map fields under `[datasets.*]`. The label-map
  check reads the categories in the annotations file, so no field is needed yet; Phase 2's
  shifted datasets add them with their first use.
- The EvalLoop v2 code waits for Ian's acceptance.
- A development-run command (`DEV-` ids) is deferred to the next lab change.

## Ian decisions requested
1. D5: keep `fp32` as the default or switch to `fp32_tf32_off` (one line, before G1 runs).
2. F10: the value of `score_threshold` for `atss_r50`, frozen before the first registered run.

## Ian hours
About 0.5 hours to read; no signature is needed.

## Acceptance
- [ ] `make check` green (lint, types, tests with coverage gate)
- [ ] `qcal agent-layer` PASS
- [ ] editing `calib/platt.py` leaves every cache key unchanged; editing `models/` changes it
- [ ] a cache entry whose producer is not in the registry is a miss
- [ ] a `trt_jetson` cell is refused until a detector kind produces that target
- [ ] goldens re-pinned only where listed above
- [ ] advisory adversarial review saved under `review/claude/<branch-slug>.md`
- [ ] the other model's review at `review/gemini/<branch-slug>.md`

## Tests
- regression: `tests/regression/test_run_identity_goldens.py`
- unit: `test_lab_experiment.py`, `test_lab_predictions.py`, `test_lab_models.py`,
  `test_lab_calibrators.py`, `test_lab_two_threshold.py`, `test_lab_config.py`
- security: `tests/security/test_lab_boundaries.py` (a forged producer is a miss)

## Tasks
Proposal and goldens → roles → formats → numerics → cache → smoke, each a commit with its
tests; advisory review; then Ian's decisions D5 and F10.
