# change: evalloop-v2

## Status
Proposed, Oct 11, 2026. Interface for Ian to accept or amend by **Oct 17**, before he writes
`src/qcal_lab/handwritten/eval_loop.py`. Code follows only after Ian accepts it (cycle plan
PR-D2, code by Oct 24). Mostly agent-owned (`src/qcal_lab/`, `tests/`, `docs/`). One line
edits the enforcement surface and needs Ian's signed commit: `qcal.toml` adds the replay
directory to `ian_data`. The only code change in this proposal's commits is two docstrings
(panel finding F18).

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
- **A bootstrap cannot be expressed (F4, F8).** Ground truth is keyed by image id and a
  subset collapses repeated ids (`data/coco.py:75`); the predictions format refuses repeated
  images (`predictions.py:58-59`).
- **One prediction source feeds every role (F11).** There is no way to evaluate the
  oracle's own detections through our pipeline, so a G1 gap cannot be attributed to the
  detector, the calibrators or the metrics.
- **The docstrings were wrong (F18).** `evaluation.py:9-11` and `two_threshold.py:14-15` said
  thresholds come from the fit and select splits; `experiment.py:405-406` passes the select
  split for both stages. Both docstrings are fixed with this proposal.

## What changes
Interface (this proposal; Ian's acceptance makes it binding):

```python
# In Ian's module, checked before the factory is called:
INTERFACE_VERSION = 2

# In qcal_lab.evaluation:
STAGES: Final = ("targets", "calibration", "operating")  # Ian maps each stage to its tau


@dataclass(frozen=True)
class MatchedDetection:
    image_id: str
    index: int  # position in ImageDetections.detections of the input
    label: int
    score: float  # the input score, unchanged; matching orders by it
    object_index: int | None  # into ground_truth.boxes[image_id]; None: unmatched
    iou: float  # with the matched object; 0.0 when unmatched
    ignored: bool = False  # matched a crowd region: neither a hit nor a false positive


@dataclass(frozen=True)
class MatchTable:
    stage: str
    tau: float
    detections: tuple[MatchedDetection, ...]  # every input detection, once; none cut
    objects: Mapping[int, Mapping[str, int]]  # label -> image -> non-crowd objects
    convention: Mapping[str, str]  # Ian's protocol choices (D3), verbatim


@dataclass(frozen=True)
class DetectionSets:
    raw: tuple[ImageDetections, ...]  # detector output, raw scores
    calibrated: tuple[ImageDetections, ...]  # calibrated scores, before v_c (see D4)
    operating: tuple[ImageDetections, ...]  # Alg. A.2 output: u_c, calibrate, v_c


@runtime_checkable
class EvalLoop(Protocol):  # version 2
    def match(
        self, predictions: Sequence[ImageDetections], ground_truth: GroundTruth, *, stage: str
    ) -> MatchTable: ...

    def targets(self, table: MatchTable) -> Mapping[str, Sequence[float | None]]: ...

    def threshold_objectives(
        self, table: MatchTable, *, label: int, thresholds: Sequence[float]
    ) -> Sequence[float]: ...

    def metrics(self, sets: DetectionSets, ground_truth: GroundTruth) -> Mapping[str, float]: ...
```

`targets` may return `None` for a detection the protocol excludes from fitting (for example
one matched to a crowd region, if D3 says so); the calibrator skips it. Calibrated detections
keep their input order, and PR-D1's reserved `source_index` links each one to its raw
detection, so D3 may choose raw-score order for evaluation-time matching.

How the pipeline uses it (agent code, after acceptance):

1. **Select split, u_c.** `match(raw select, stage="calibration")` once; per class,
   `threshold_objectives(table, label=c, thresholds=grid)` returns the whole curve in one
   call. The pipeline picks the lowest finite value (ties keep the lowest threshold, as now).
2. **Fit split, targets.** Drop raw scores below u_c, `match(..., stage="targets")` once, then
   `targets(table)`. The calibrator fits on those targets, as today.
3. **Select split, v_c.** Calibrate the select split, `match(..., stage="operating")` once,
   and select v_c the same way.
4. **Evaluate split.** Build the three detection sets and call `metrics` once.

The pipeline never computes a reported metric or an objective, and never derives counts from
a `MatchTable`: it validates a table's structure and hands it back to Ian's methods. Only
Ian's hand-written loop reports metrics (AGENTS.md); agents never write it (CLAUDE.md
rule 4).

**Match once, then threshold.** One table per split and stage gives the same objective at
every threshold as re-matching each thresholded subset, provided Ian's `match` follows three
rules. The contract suite checks each one (see Tests).

1. Matching is per class and greedy in descending order of the score it is given.
2. Ties are broken by a rule that depends only on the tied detections themselves and their
   relative input order (a stable sort qualifies; a seeded shuffle of the input does not).
3. `match` cuts nothing. The per-image top-k is the detector's `max_per_image`, pinned in
   PR-D1, and is applied before any of this.

Under these rules a detection's match depends only on detections ranked above it, and a
score threshold keeps a prefix of that ranking. A tie group is kept or dropped whole. With
Platt at a = 0 a whole class is one tie group, which the rules still cover. If D3 chooses
raw-score order for evaluation-time matching, calibrator monotonicity matters: both
calibrators are monotone non-decreasing per class, so a threshold on calibrated scores still
keeps a prefix of the raw order. A loop that cannot follow the rules (for example a Hungarian
assignment) sets `prefix_matching = False` in its module, and the pipeline re-matches each
thresholded subset, as version 1 does today.

**Bootstrap without weights.** `metrics` takes no weights. PR-F builds each bootstrap
replicate as a dataset in its own right: every drawn image is copied under a fresh id
(`<id>#<k>`), copies sit next to each other in ascending id order, and `metrics` is called
once per replicate. That is exact for every metric, including COCO AP and LaACE0, whose
values depend on how tied scores are ordered. Isotonic flat blocks make such ties common.

**Replay detector (F11).** A detector kind `replay` reads a predictions file and its
provenance sidecar from `data/replay/`, which joins `ian_data`. Ian writes these files and
signs them; agents never do. The sidecar names:

- producer, repository, commit and command;
- stage, which must be `raw`: detector output before any calibration or thresholding beyond
  the detector's own;
- device, precision, target, and the model or engine sha256;
- every image id the producer processed.

An import command converts a COCO results file (`image_id`, `category_id`, `bbox`, `score`)
into the predictions format. It refuses a sidecar with a missing field or a stage other than
`raw`. An image listed as processed but absent from the results gets no detections; an image
not listed is refused. The fingerprint covers the predictions file, the sidecar and the format
version. A run refuses a replay file whose precision or target differs from its cell's.
Replay cells enter the pre-registration through an `AMENDMENTS.md` entry (Ian).

Running a cell with the oracle's detections splits a G1 gap into three parts: published minus
oracle, oracle minus our pipeline on the oracle's detections, and our pipeline on the oracle's
detections minus our full run. The `raw` rule keeps the oracle's own calibrator out: it was
fit on minival, which under D1 option (b) overlaps our test split.

Files (after acceptance):
- `qcal_lab.evaluation`: the types above, `check_match_table`, `check_objectives`, and the
  module version check in `load_eval_loop`.
- `qcal_lab.calib.two_threshold`: selection from match tables and objective curves.
- `qcal_lab.experiment`: the three detection sets; `environment.eval_loop.interface_version`.
- `qcal_lab.fixture_eval`: version 2, greedy one-to-one matching on the fixture only.
- `qcal_lab.models`: the `replay` kind; `qcal_lab.predictions`: the COCO results import.
- `qcal_lab.parity`: a metric case names the set its detections represent (`set`, default
  `operating`).
- `qcal.toml` (signed): `data/replay/**` joins `ian_data`.
- `tests/contract`: the version 2 checks and synthetic cases below.

## Decisions
- **Two layers, matching then metrics** (F8). Alternative: keep one `threshold_objective`
  per candidate and cache inside Ian's loop. Rejected: the cache would be invisible to the
  pipeline, curves could not be recorded, and the equivalence could not be tested from
  outside. Consequence: Ian writes four methods instead of three, and matching becomes a
  value the pipeline can check.
- **Ian owns matching, objectives and metrics; the pipeline owns orchestration.** Matching
  embodies the protocol choices of D3, so it belongs with the metrics. Alternative: the
  pipeline matches and Ian only scores. Rejected: it would move protocol decisions into
  agent code.
- **`metrics` receives detections, not match tables.** COCO AP matches at ten IoU thresholds
  and LaECE0 at one; a single table cannot serve both. Ian may call his own `match` inside.
- **Resampled datasets, not weights.** Weights ("image i counts w times") are not well
  defined for metrics that depend on the order of tied scores, such as COCO AP and LaACE0:
  weighted counts differ from physical copies when a tie group mixes hits and misses.
  Alternative: weights meaning "w adjacent copies in ascending id order", with exact checks
  only for order-free metrics (LaECE0, LRP). Rejected for now: it puts a subtle convention in
  Ian's loop. Consequence: a bootstrap costs one `metrics` call per replicate; PR-F measures
  that cost before choosing the replicate count.
- **No version 1 adapter.** The fixture loop moves to version 2 and no other version 1 loop
  exists. A module without `INTERFACE_VERSION = 2` is refused with a message naming the
  version.
- **Replay files are Ian's data.** Alternative: pin their sha256 in `configs/lab.toml`.
  Rejected: agents own that file, so an agent could point a registered run at detections of
  its choosing.

## Compatibility
- Formats: `calibration.json` gains `objective_curves` (per stage, class and candidate), with
  NaN written as `null`. The curves stay in that artifact; the record keeps the selected
  thresholds and the curves' digest, so the index does not grow by thousands of columns.
  PR-D1's strict envelopes land first, so the change ships under a declared version.
- Config keys: `[detectors.<name>]` accepts `kind = "replay"` with `file`. No key is removed.
- Policy: `ian_data` gains `data/replay/**` (signed).
- CLI: one new import command under the lab CLI; no change to `qcal`.
- Records: `environment.eval_loop` gains `interface_version`; schema 1 only gains fields. No
  registered run exists, so nothing needs re-running.
- Ian's code: none exists yet.
- Cache: this change adds files under `models/` and edits `predictions.py`, which PR-D1's
  narrowed cache key covers. The key therefore changes once, before any registered run.

## Out of scope
- The bootstrap harness, ordering checks and per-class CIs: PR-F.
- Cross-precision pairing for Phase 2 (`agreement_rate_vs_fp32`): PR-D1 reserves
  `source_index` in prediction rows; the pairing method arrives with the Phase 2 proposal.
- Any metric implementation. LaECE0, LRP and AP stay in Ian's hand-written loop.

## Ian decisions requested
1. **By Oct 17:** accept or amend the interface above.
2. **By Oct 17 (D4):** which set each reported metric uses, and whether the `calibrated` set
   drops scores below u_c first. Recommended: AP on `calibrated` without u_c, LaECE0 and LRP
   on `operating` [unverified against the paper; confirm in the oracle code]. Three caveats:
   - "Platt does not change AP" fails at a = 0, where a class becomes one tie group, and for
     scores clipped at Platt's epsilon. D6's check that Platt leaves AP unchanged must
     exclude such classes.
   - Isotonic's `out_of_bounds = "clip"` gives every score below the fitted minimum the same
     value, so "without u_c" creates a large tie group at the bottom of each class.
   - `calibrate()` always applies u_c today (`two_threshold.py:79-90`); "without u_c" needs
     a new code path.
3. **Before writing the loop (D3):** the protocol the `convention` field records:
   - tau per stage, and IoU `>=` or `>` at tau = 0;
   - greedy order and the tie-break (within the three rules above);
   - crowd handling, including what target a crowd-matched fit detection gets;
   - which score orders evaluation-time matching (raw or calibrated);
   - classes with objects but no detections;
   - LaECE0 bin edges, and the AP tie order (pycocotools sorts ties stably by input order).

   They go in `docs/reference/kuzucu_eccv24.md` (Ian's data).
4. Bootstrap by resampled datasets rather than weights (recommended).
5. Replay files in `data/replay/` as `ian_data`, with the sidecar above; sign the
   `qcal.toml` line; add replay cells through `AMENDMENTS.md` when wanted.

## Ian hours
About 1.5 hours to read and decide items 1, 2, 4 and 5. Item 3 and the loop itself are
science time.

## Acceptance
- [ ] Ian accepts or amends the interface (by Oct 17)
- [ ] `make check` green (lint, types, tests with coverage gate)
- [ ] `qcal agent-layer` PASS
- [ ] the fixture loop passes every version 2 MUST check, including the synthetic cases
- [ ] evaluate invariance: perturbing the evaluate split's predictions and ground truth leaves
      every match table, curve and threshold byte-identical
- [ ] a spy shows `match` and `threshold_objectives` never receive an evaluate image id
- [ ] a replay run reproduces a fixture run's raw predictions byte for byte (native format)
- [ ] advisory adversarial review saved under `review/claude/<branch-slug>.md`
- [ ] RESEARCH_LOG.md entry (Ian)
- [ ] the other model's review at `review/gemini/<branch-slug>.md`

## Tests
Contract MUST checks, run on the fixture loop always and on Ian's loop once it exists (NOT
ASSESSED until then):
- the module declares `INTERFACE_VERSION = 2`;
- `match` lists every input detection exactly once with its label and score unchanged;
  `object_index` is `None` or a box of the same label; no non-crowd object is matched twice;
  `iou` is in [0, 1] and 0 when unmatched;
- `threshold_objectives` returns one number per threshold, each finite or NaN;
- the curve from one table equals, at every threshold, the value from re-matching the
  thresholded subset, to 1e-12 (exactly, if the loop sums with `math.fsum`);
- `targets(table)` gives one target in [0, 1], or `None`, per detection;
- the existing checks: no input mutation, identical output under two hash seeds, no hidden
  file or network reads, metric names usable in claim references.

The smoke fixture never has more than 100 detections in an image or a matching conflict, so
the equivalence check also runs on three synthetic cases:
- more than 100 detections in one image across several classes;
- a greedy-versus-optimal conflict, where a Hungarian assignment differs from greedy;
- tied scores in one class that mix hits and misses.

Unit: `check_match_table` and `check_objectives` on hand-built bad tables; curves written with
NaN as `null` and reloaded; the replay detector's fingerprint, and its refusal of a changed
file, a missing sidecar field, a non-raw stage and an unlisted image; the COCO results import
(box conversion, unknown category, duplicate rows). The COCO round trip is not byte-exact in
float64 (x1 + (x2 − x1) can differ from x2), so byte equality is tested on the native format
only. Regression: `tests/regression/test_evalloop_v2.py`, one test per finding (F2, F8, F11,
F18).

## Tasks
Interface accepted (Ian) → agent code and contract tests (PR-D2) → Ian's loop against
version 2 → the contract suite on his loop → parity cases (PR-F).

## Adversarial review (advisory) and what changed
`adversarial-reviewer` on the first draft (committed in `76ab43d`): **revise**, 5 blocking
and 24 non-blocking findings across this proposal and the PR-0 documents. The review is saved
at `review/claude/claude-sdlc-agents-implementation-plan-gmb29t.md`. Changes here:

- **B2, replay provenance.** Replay files move to `ian_data` with a provenance sidecar; the
  stage must be `raw`; precision and target are checked; replay cells need an amendment.
  This adds the signed `qcal.toml` line in Status.
- **B4, weights.** Not well defined for order-dependent metrics. Replaced by resampled
  datasets.
- **B5, prefix equivalence.** Three explicit rules for `match` (per class and greedy, a
  presence-independent tie-break, no cut) and three synthetic contract cases.
- **Non-blocking.**
  - N1: raw-order matching is possible through `source_index`.
  - N2: pipeline steps reordered.
  - N3: the monotonicity argument is reworded.
  - N4: `None` targets.
  - N5: D4 caveats.
  - N6: cache claim corrected.
  - N7: curves only in the artifact, NaN as `null`.
  - N8: the leakage-checker acceptance item is replaced by an invariance test and a spy.
  - N9: no adapter.
  - N10: the version lives on the module.
  - N11: byte equality on the native format only.
  - N12: review path.
  - N13: AGENTS.md cited.
  - N14: the `two_threshold.py` docstring is fixed.
  - N23: Tasks section added.

B1 and B3 concern `CONTRIBUTING.md`, and the remaining non-blocking findings concern the other
PR-0 documents; the cycle plan's status section records them.
