---
name: reproduce-check
description: Report the gap to Gate G1/K1 for the clean-room FP32 reproduction (smoke, leakage, oracle parity and Phase 1 readiness) without editing anything.
allowed-tools: Bash(make smoke), Bash(make parity), Bash(make lab-status), Bash(qcal leakage:*), Bash(qcal registry audit:*), Read, Agent
---
Run these steps in order and report each result verbatim. Never edit files, configuration or
tests to change an outcome.

1. `make smoke`: the whole loop on the synthetic fixture, through the real registry, in
   under 5 minutes.
2. `make lab-status`: what still blocks a registered run. That covers Ian's evaluation loop,
   the detector checkpoint, the dataset, the split manifests and the oracle outputs.
3. The data-leakage-checker subagent: PASS or FAIL with evidence. It runs
   `qcal leakage --json`.
4. `make parity`: oracle parity. A skipped case is NOT ASSESSED, never passed.
5. If EXPERIMENTS.yaml exists, run `qcal registry audit --json` for the Phase 1 cells.

Gate G1 (plan §5, Phase 1), per detector:
- **Gate 1, parity.** Every oracle case passes on identical inputs.
- **Gate 2, published numbers.** LaECE0 within ±1.0 and AP within ±0.5 of Kuzucu et al.
  (arXiv:2405.20459) Table 9. Compare run IDs from the index against the reference values
  Ian records in `docs/reference/kuzucu_eccv24.md`. If that file does not exist, report
  gate 2 as NOT ASSESSED. Never type a published or measured number from memory.

Output a table with three columns:
- the check;
- its status: PASS, FAIL or NOT ASSESSED;
- the evidence: the command, file:line or run ID.

Then name the one item that blocks K1 (Nov 8).
