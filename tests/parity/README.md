# Oracle parity

These tests compare our clean-room code with `fiveai/detection_calibration`, the code
release of Kuzucu et al., arXiv:2405.20459, run on identical inputs. Run them with
`make parity`. Until oracle outputs exist, every test skips and names what is missing.

## Licence and clean room

The oracle is licensed CC BY-NC-SA 4.0. It is a test oracle only:

- It runs unmodified, in its own environment, on the P40 (CUDA 11.8, torch 2.1; plan §5
  Phase 0 item 4b).
- Only its numeric outputs are committed here, never its code.
- No module under `src/` imports it. `qcal licenses` allows `detection_calibration`
  imports only under `tests/parity/`.

## Case files (`fixtures/*.json`)

Each case is one JSON file. The `oracle` object must name the repository and the commit
that produced the numbers. Recording the exact command as well is recommended.

Calibrator case. This tests `qcal_lab.calib`:

```json
{"kind": "calibrator", "calibrator": "isotonic",
 "oracle": {"repository": "fiveai/detection_calibration", "commit": "<sha>",
            "command": "<what was run>"},
 "fit": {"scores": [0.1, 0.4], "targets": [0.0, 0.7]},
 "transform": {"scores": [0.2], "expected": [0.35]},
 "tolerance": 1e-6}
```

Metric case. This tests Ian's hand-written evaluation loop:

```json
{"kind": "metric", "metric": "LaECE0",
 "oracle": {"repository": "fiveai/detection_calibration", "commit": "<sha>"},
 "predictions": "atss_minitest.jsonl", "ground_truth": "minitest.json",
 "expected": 0.0, "tolerance": 1e-6}
```

The numbers in these examples are placeholders that show the shape. They are not results.

`predictions` is a `qcal_lab.predictions` file, and `ground_truth` is a COCO-format file.
Both paths are relative to the case file. The default tolerance is `parity.abs_tolerance`
in `configs/lab.toml`.

## When a case fails

- **Calibrator case.** `paper-reproducer` reports the deviation and fixes our
  implementation, citing the paper equation.
- **Metric case.** Nobody edits Ian's file. The failing test and an explanation of the
  difference go to Ian (CLAUDE.md rule 4).
