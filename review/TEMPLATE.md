---
reviewer: gemini-3.1-pro | claude-adversarial-reviewer
reviewed_sha: <full PR head SHA>
verdict: approve | block
blocking:
  - id: B1
    file: src/qcal/example.py:42
    finding: <one sentence, with evidence>
    resolved_in: <commit SHA, or empty while open>
non_blocking: []
---
Answer each question with evidence or "none found":

1. Leakage: is there any path from test images or test metrics into fitting or selection?
2. Untraceable numbers: is there any number in the diff not backed by a run record?
3. Spec drift: is there any behaviour not in EXPERIMENTS.yaml or an AMENDMENTS.md entry?
