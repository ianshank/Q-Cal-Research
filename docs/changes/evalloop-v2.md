# change: evalloop-v2

## Status
Proposed, Oct 11, 2026. Interface for Ian to accept or amend by **Oct 17**, before he writes
`src/qcal_lab/handwritten/eval_loop.py`. Agent-owned surface (`src/qcal_lab/`, `tests/`,
`docs/`); no enforcement-surface edit. Code follows only after Ian accepts the interface
(cycle plan PR-D2, code by Oct 24). The one code change in this proposal's commit is the
docstring fix in `qcal_lab.evaluation` (panel finding F18).

## Gate served
G1/K1 (Nov 8). The interface Ian writes against must be able to compute the paper's metrics
on the paper's detection sets, support a paired image bootstrap, and stay stable through
Phase 2. Changing it after his loop exists costs his time twice.

## Why
Verified against `88563df`:

- **The paper's AP cannot be computed (F2).** `metrics()` receives only the output of
  Alg. A.2, after the operating threshold v_c (`experiment.py:412-413`). The panel reads the
  paper's AP as COCO top-100 AP on calibrated detections before v_c, since isotonic
  regression changes AP and Platt scaling does not [unverified; Ian confirms against the
  oracle code under D4].
- **Matching runs once per candidate (F8).** `select_class_thresholds`
  (`calib/two_threshold.py:157-171`) calls `threshold_objective` once per class, candidate
  and stage. With the default grid that is 80 classes × 20 candidates × 2 stages, and every
  call re-matches its subset inside Ian's loop. Nothing lets a per-candidate objective curve
  be recorded, which ε-optimality parity needs (F9, PR-F).
- **A bootstrap cannot be expressed (F4, F8).** Predictions and ground truth are keyed by
  image id (`qcal/protocols.py:28`, `data/coco.py:51-52`). A resample with duplicates
  collapses to the distinct images.
- **One prediction source feeds every role (F11).** There is no way to evaluate the
  oracle's own detections through our pipeline, so a G1 gap cannot be attributed to the
  detector, the calibrators or the metrics.
- **The module docstring is wrong (F18).** `evaluation.py:9-11` says thresholds are
  selected on the fit and select splits; `experiment.py:405-406` passes the select split
  for both stages.

## What changes
Interface (this proposal; Ian's acceptance makes it binding):

```python
INTERFACE_VERSION: Final = 2
STAGES: Final = ("targets", "calibration", "operating")  # Ian maps each stage to its tau


@dataclass(frozen=True)
class MatchedDetection:
    image_id: str
    index: int  # position in ImageDetections.detections
    label: int
    score: float  # the score matching ordered by
    object_index: int | None  # into ground_truth.boxes[image_id]; None: unmatched
    iou: float  # with the matched object; 0.0 when unmatched
    ignored: bool = False  # matched a crowd region: neither a hit nor a false positive


@dataclass(frozen=True)
class MatchTable:
    stage: str
    tau: float
    detections: tuple[MatchedDetection, ...]  # every input detection, once
    objects: Mapping[int, Mapping[str, int]]  # label -> image -> non-crowd objects
    convention: Mapping[str, str]  # Ian's protocol choices (D3), verbatim


@dataclass(frozen=True)
class DetectionSets:
    raw: tuple[ImageDetections, ...]  # detector output, raw scores
    calibrated: tuple[ImageDetections, ...]  # calibrated scores, before v_c (D4: u_c or not)
    operating: tuple[ImageDetections, ...]  # Alg. A.2 output: u_c, calibrate, v_c


@runtime_checkable
class EvalLoop(Protocol):  # version 2
    interface_version: int

    def match(
        self, predictions: Sequence[ImageDetections], ground_truth: GroundTruth, *, stage: str
    ) -> MatchTable: ...

    def targets(self, table: MatchTable) -> Mapping[str, Sequence[float]]: ...

    def threshold_objectives(
        self, table: MatchTable, *, label: int, thresholds: Sequence[float]
    ) -> Sequence[float]: ...

    def metrics(
        self,
        sets: DetectionSets,
        ground_truth: GroundTruth,
        *,
        weights: Mapping[str, int] | None = None,
    ) -> Mapping[str, float]: ...
```

How the pipeline uses it (agent code, after acceptance):

1. **Fit split, targets.** Drop raw scores below u_c, `match(..., stage="targets")` once, then
   `targets(table)`. The calibrator fits on those targets, as today.
2. **Select split, u_c.** `match(raw select, stage="calibration")` once; per class,
   `threshold_objectives(table, label=c, thresholds=grid)` returns the whole curve in one
   call. The pipeline picks the lowest finite value (ties keep the lowest threshold, as now)
   and records every curve in `calibration.json`.
3. **Select split, v_c.** Calibrate the select split, `match(..., stage="operating")` once,
   and select v_c the same way.
4. **Evaluate split.** Build the three detection sets and call `metrics` once. The pipeline
   never computes a reported metric or an objective itself (CLAUDE.md rule 4).

"Match once, then threshold" equals re-matching each thresholded subset exactly when
matching is greedy in descending score order and per class: a detection's match then depends
only on detections scored at least as high, and a score threshold keeps a prefix of that
order. Both calibrators are monotone non-decreasing per class (Platt with a ≥ 0, isotonic),
so a v_c threshold on calibrated scores also keeps a prefix. Isotonic ties keep or drop a
whole tie group together, so any deterministic tie-break preserves the property. The contract
suite checks it on Ian's loop. A matcher that is not greedy, such as a Hungarian assignment,
fails the check; such a loop sets the optional attribute `prefix_matching = False`, and the
pipeline then re-matches each thresholded subset, as version 1 does today.

**Weights.** `weights[image_id] = w` means the image counts as `w` independent copies; a
missing image or `None` means 1, and 0 means the image is absent. A paired image bootstrap
(PR-F) passes integer resampling counts instead of a collapsed id list. Ian's loop applies
them to every count it accumulates (hits, false positives, objects, bin counts).

**Replay detector (F11).** A detector kind `replay` reads a raw-predictions file whose
sha256 is pinned in `configs/lab.toml`. Its fingerprint is that sha plus the format version.
An import command converts a COCO results file (`image_id`, `category_id`, `bbox`, `score`)
into the predictions format. Running a cell with the oracle's detections splits a G1 gap
into three parts: published minus oracle, oracle minus our pipeline on the oracle's
detections, and our pipeline on the oracle's detections minus our full run.

Files (after acceptance):
- `qcal_lab.evaluation`: the types above, `INTERFACE_VERSION`, `check_match_table`,
  `check_objectives`, the version check in `load_eval_loop`, and a version 1 adapter.
- `qcal_lab.calib.two_threshold`: selection from match tables and objective curves; curves
  in the calibration provenance.
- `qcal_lab.experiment`: the three detection sets; `environment.eval_loop.interface_version`.
- `qcal_lab.fixture_eval`: version 2, greedy one-to-one matching on the fixture only.
- `qcal_lab.models`: the `replay` kind; `qcal_lab.predictions`: the COCO results import.
- `qcal_lab.parity`: a metric case names the set its detections represent (`set`, default
  `operating`).
- `tests/contract`: the version 2 checks below.

## Decisions
- **Two layers, matching then metrics** (F8). Alternative: keep one `threshold_objective`
  per candidate and cache inside Ian's loop. Rejected: the cache would be invisible to the
  pipeline, curves could not be recorded, and the equivalence could not be tested from
  outside. Consequence: Ian writes four methods instead of three, and matching becomes a
  value the pipeline can record and compare.
- **Ian owns matching, objectives and metrics; the pipeline owns orchestration.** Matching
  embodies the protocol choices of D3, so it belongs with the metrics. Alternative: the
  pipeline matches and Ian only scores. Rejected: it would move protocol decisions into
  agent code.
- **`metrics` receives detections, not match tables.** COCO AP matches at ten IoU thresholds
  and LaECE0 at one; a single table cannot serve both. Ian may call his own `match` inside.
- **Integer weights, not resampled id lists.** Duplicated ids cannot be represented in
  mappings keyed by image id. Weights keep every structure unchanged and make "weight 0 ≡
  absent" testable.
- **A version 1 adapter stays for smoke and development runs.** It wraps a version 1 loop:
  `match` records the inputs, `threshold_objectives` calls the old method once per
  threshold, and `metrics` passes only the operating set and refuses weights. Registered
  cells refuse it, because it cannot compute the paper's AP set. Alternative: remove
  version 1. Rejected for now: the adapter is small and keeps existing tests meaningful.

## Compatibility
- Formats: `calibration.json` gains `objective_curves` (per stage, class and candidate).
  PR-D1's strict envelopes land first, so the change ships under a declared version.
- Config keys: `[detectors.<name>]` accepts `kind = "replay"` with `file` and `sha256`. No
  key is removed.
- CLI: one new import command under the lab CLI; no change to `qcal`.
- Records: `environment.eval_loop` gains `interface_version`; schema 1 only gains fields. No
  registered run exists, so nothing needs re-running.
- Ian's code: none exists yet. A module must declare `interface_version = 2`; a module
  without it is treated as version 1 and runs only through the adapter, with a message
  naming the version it lacks.
- Cache: the prediction cache key does not cover `evaluation.py` once PR-D1 narrows it, so
  this change invalidates no cached predictions.

## Out of scope
- The bootstrap harness, ordering checks and per-class CIs: PR-F consumes `weights`.
- Cross-precision pairing for Phase 2 (`agreement_rate_vs_fp32`): PR-D1 reserves
  `source_index` in prediction rows; the pairing method arrives with the Phase 2 proposal.
- Any metric implementation. LaECE0, LRP and AP stay in Ian's hand-written loop.

## Ian decisions requested
1. **By Oct 17:** accept or amend the interface above.
2. **By Oct 17 (D4):** which set each reported metric uses, and whether the `calibrated`
   set drops scores below u_c first. Recommended: AP on `calibrated` without u_c, LaECE0
   and LRP on `operating` [unverified against the paper; confirm in the oracle code].
3. **Before writing the loop (D3):** the protocol the `convention` field records: tau per
   stage, IoU `>=` or `>` at tau = 0, greedy order and tie-break, crowd handling, which score
   orders evaluation-time matching (raw or calibrated), classes with objects but no
   detections, and LaECE0 bin edges. They go in `docs/reference/kuzucu_eccv24.md` (Ian's
   data).
4. Weights mean "independent copies" (recommended) rather than a normalised density.
5. Keep the version 1 adapter for smoke and development runs (recommended) or drop it.

## Ian hours
About 1 hour to read and decide items 1, 2, 4 and 5. Item 3 and the loop itself are science
time.

## Acceptance
- [ ] Ian accepts or amends the interface (by Oct 17)
- [ ] `make check` green (lint, types, tests with coverage gate)
- [ ] `qcal agent-layer` PASS
- [ ] the fixture loop passes every version 2 MUST check
- [ ] "match once, then threshold" equals re-matching on the fixture loop, for every class,
      stage and candidate
- [ ] a replay run reproduces a fixture run's raw predictions byte for byte
- [ ] `data-leakage-checker` PASS: the evaluate manifest is never read during fit or select
- [ ] advisory adversarial review saved under `review/claude/evalloop-v2.md`
- [ ] RESEARCH_LOG.md entry (Ian)
- [ ] the other model's review at `review/gemini/<branch-slug>.md`

## Tests
Contract MUST checks, run on the fixture loop always and on Ian's loop once it exists (NOT
ASSESSED until then):
- `interface_version == 2`;
- `match` lists every input detection exactly once with its label and score unchanged;
  `object_index` is `None` or a box of the same label; no non-crowd object is matched twice;
  `iou` is in [0, 1] and 0 when unmatched;
- `threshold_objectives` returns one number per threshold, each finite or NaN;
- the curve from one table equals, at every threshold, the value from re-matching the
  thresholded subset (exact equality);
- `targets(table)` gives one target in [0, 1] per detection, as today;
- `metrics` with `weights=None` equals all-ones weights exactly; weight 0 equals removing the
  image; weight `w` equals `w` copies of the image under fresh ids (fixture);
- the existing checks: no input mutation, identical output under two hash seeds, no hidden
  file or network reads, metric names usable in claim references.

Unit: the adapter; `check_match_table` and `check_objectives` on hand-built bad tables;
curves recorded and reloaded; the replay detector's fingerprint and its refusal of a file
whose sha256 differs; the COCO results import (box conversion, unknown category, duplicate
rows). Regression: `tests/regression/test_evalloop_v2.py`, one test per finding (F2, F8,
F11, F18).
