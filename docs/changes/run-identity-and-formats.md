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
3. **Strict formats.**
   - Predictions version 2. The header gains precision, target, quant path, shift, score
     definition, model sha256 and the effective `test_cfg`. Rows reserve `source_index`,
     `class_logit` and `aux_logit`.
   - `cache_key` refuses parts named `format` or `version`.
   - Envelopes with a format name and version: calibrator, calibration (written atomically),
     cache sidecar and parity case. A file without its envelope is refused.
   - The calibrator envelope also records Platt's solver settings and any class where the
     slope stopped at the a = 0 bound.
4. **Numerics and the program's environment.**
   - `[numerics.regimes.<precision>]` names each regime's torch and cuDNN settings.
     - `fp32` keeps torch's defaults, as today.
     - `fp32_tf32_off` also turns off cuDNN benchmark, turns on deterministic algorithms and
       requires `CUBLAS_WORKSPACE_CONFIG`.
   - The default stays `fp32`. Switching to `fp32_tf32_off` is Ian's decision D5 and one line
     of configuration.
   - The program records:
     - every effective flag and `NVIDIA_TF32_OVERRIDE`;
     - the device it used, and the CUDA and cuDNN versions;
     - a digest of torch's build configuration, the package list (as an artifact), and the
       handwritten files' digest.
   - `configs/lab.toml` pins `max_per_image = 100`. `score_threshold` is Ian's to set (F10), so
     an `atss_r50` run refuses to start while it is unset.
   - The checkpoint's class names must equal the dataset's.
5. **Cache key and producer.**
   - The key covers only the modules that produce predictions: `models/`, `data/coco.py`,
     `data/fixture.py` and `predictions.py`.
   - It drops `__version__` and the absolute images path.
   - It adds target, quant path, compute capability, CUDA, cuDNN, numerics, opencv, pillow,
     mmcv, the resolved detector configuration and `test_cfg`.
   - The detector is built only on the first cache miss.
   - A hit counts only if a record in `runs/registry/` lists the same path and sha256.
   - `put` refuses to replace different bytes under an existing key.
6. **Smoke isolation.** Smoke's experiment program runs as `{python} -I`, like registered
   runs.

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
  - `cache_key` (parts and module list);
  - the fixture detector's output, through its header.

  `cell_id`, `split_digest`, `rank_key`, `draw`, `config_hash` and the fixture bytes do not
  change.
- Config keys: new roles under `[factors]`, `[numerics.regimes.*]`, and a packaged
  `resources/tooling.toml` (`[smoke]`, `[fixture_eval]`). `[smoke]` leaves the hashed
  defaults; a `[smoke]` table in `configs/lab.toml` is refused with a pointer to the new file.
  Tests override smoke settings through `run_smoke(..., settings=...)`.
- Records: the program's environment gains keys; schema 1 only gains fields.
- Cache: every existing cache entry misses once. No registered run exists.
- Parity fixtures: none committed yet; the parity-case envelope applies from the first one.

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
