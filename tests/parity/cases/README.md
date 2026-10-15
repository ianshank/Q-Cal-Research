# Synthetic protocol cases

Small generated inputs on which two evaluation protocols that differ in one choice give
different answers (cycle plan PR-F, D3/F5). They are written by
`python -m qcal_lab protocol-cases` from `[protocol_cases]` in
`src/qcal_lab/resources/tooling.toml` and the threshold grid in `[calibration]`. Never edit
them by hand: `python -m qcal_lab protocol-cases --check` (run by the test suite) fails when a
file differs from the generator.

This directory is agent-owned. The oracle's answers are Ian's and go under
`tests/parity/fixtures/` (`ian_data`), never here.

## What a case holds

| Key | Meaning |
|---|---|
| `ground_truth` | A COCO annotation document (one image). |
| `detections` | A COCO results list. A detection's index in it is its id in `predicts`. |
| `separates` | The protocol choice the case tells apart. |
| `alternatives` | One entry per way of making that choice: `rule` and what it `predicts`. |
| `settings` | The generator settings the case was built with. |

`predicts` holds outcomes, never metric values:

- `outcomes`: per detection, `TP:<annotation id>`, `FP`, `ignored` (absorbed by a crowd
  region) or `dropped` (cut by `max_detections`);
- `bins`: per detection, the bin index its score falls in;
- `classes_in_mean`: the category ids a per-class mean averages over;
- `threshold`: the threshold a selection keeps when several tie.

| Case | Separates |
|---|---|
| `iou_at_threshold` | a match needs IoU ≥ t, or IoU > t (integer boxes, IoU exactly t) |
| `equal_score_order` | greedy matching among equal scores follows input order, or the better IoU |
| `crowd_only_overlap` | a detection on a crowd region is ignored, or a false positive |
| `scores_on_bin_edges` | bins are left-closed (1.0 in the last bin), or right-closed |
| `class_without_detections` | a class with objects but no detections is averaged in, or skipped |
| `ties_at_max_detections` | the cut at `max_detections` keeps input order on a tie, or not |
| `threshold_selection_ties` | a tie between grid thresholds keeps the lowest, or the highest |

## Before the oracle runs

Set `iou_threshold`, `max_detections` and `bin_count` in `[protocol_cases]` to the values the
oracle uses, then regenerate (`python -m qcal_lab protocol-cases`) and commit the result.

## How Ian records the oracle's answer

1. Run the oracle, unmodified, in its own environment (`tests/parity/README.md`), on each
   case's `ground_truth` and `detections`.
2. Read off what it decided: which detections it matched, ignored or cut, the bin counts, the
   classes in its per-class mean, the threshold it kept.
3. Commit, signed, one file per case under `tests/parity/fixtures/protocol/`:

   ```json
   {"kind": "protocol-outcome", "case": "iou_at_threshold",
    "case_sha256": "<sha256 of the case file the oracle ran on>",
    "oracle": {"repository": "fiveai/detection_calibration", "commit": "<sha>",
               "command": "<what was run>"},
    "rule": "<the alternative's rule, verbatim>",
    "notes": "<anything the alternatives did not foresee>"}
   ```

   If the oracle matches none of the alternatives, `rule` is `"none"` and `notes` says what it
   did. Our code is then compared with the oracle's choice; a difference is reported as a
   deviation (`paper-reproducer` for calibrators; Ian for his evaluation loop), not fixed by
   editing a case.
