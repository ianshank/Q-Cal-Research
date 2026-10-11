---
reviewer: claude-adversarial-reviewer
reviewed_sha: 76ab43de763b2aaf24855cfd508cf035279c96da
verdict: block
blocking:
  - id: B1
    file: CONTRIBUTING.md:59-81
    finding: The signing recipe has Ian read a diff of 8 hand-picked paths, then sign every commit or the whole squash, which signs unread changes to run records, oracle fixtures, manifests, allowed_signers, conftest.py and review files.
    resolved_in: 8ac9d6a58788544b48b39613272b0cdee54add92
  - id: B2
    file: docs/changes/evalloop-v2.md:124-129
    finding: The replay detector has no provenance; its sha is pinned in agent-owned configs, its producer, stage and target are unrecorded, and replay cells are not pre-registered.
    resolved_in: 8ac9d6a58788544b48b39613272b0cdee54add92
  - id: B3
    file: CONTRIBUTING.md:83-86
    finding: A pull request from sign/<slug> matches no reviewer prefix, so review-check falls back to "claude or gemini" and a Claude-authored change can merge on a Claude review.
    resolved_in: 8ac9d6a58788544b48b39613272b0cdee54add92
  - id: B4
    file: docs/changes/evalloop-v2.md:119-122
    finding: '"Weight w = w copies" is not well defined for COCO AP or LaACE0 when scores tie, so the MUST check cannot be met by a pycocotools-faithful AP on isotonic outputs.'
    resolved_in: 8ac9d6a58788544b48b39613272b0cdee54add92
  - id: B5
    file: docs/changes/evalloop-v2.md:109-117
    finding: '"Match once, then threshold" omits preconditions (no top-k cut inside match; a presence-independent tie-break), and the smoke fixture cannot catch a violation.'
    resolved_in: 8ac9d6a58788544b48b39613272b0cdee54add92
non_blocking: [N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16, N17, N18, N19, N20, N21, N22, N23, N24]
---
Advisory review by the project's read-only `adversarial-reviewer` subagent, saved by the
caller as its definition requires. It is not the cross-model review that
`qcal ci review-check` requires for `claude/*` branches; that is `review/gemini/`.

The reviewer read the working tree whose documents were committed as `76ab43d`. Its verdict
was **revise**; the template allows only approve or block, so the field says block. Every
finding concerned documents. No code was wrong. How each finding was handled is recorded in
`docs/changes/evalloop-v2.md` (its last section) and in the status section of
`docs/changes/cycle-2026-10-g1-readiness.md`.

## Blocking

- **B1. The signing recipe signed content Ian never read.**
  - The diff in `CONTRIBUTING.md` was limited to eight paths.
  - `signing.signed_categories` also covers `EXPERIMENTS.yaml`, `CLAIMS.md`, handwritten
    files, manifests, oracle fixtures, reference values, run records, the cache, reviews,
    `allowed_signers`, `.mcp.json`, the root `conftest.py`, `*.pth` and `uv.lock`.
  - Shell writes are not stopped by the hooks. So a run record or manifest written through
    Bash could pass under Ian's key.
  - *Fix:* start from `qcal ci verify-signatures` on the agent branch, read every listed file,
    and refuse branches that touch Ian-only, `ian_data` or registry paths.
- **B2. Replay provenance.**
  - The sha256 was pinned in agent-owned `configs/lab.toml`.
  - A COCO results file carries no stage, producer, precision or target. It omits images with
    no detections, so "not run" looks the same as "nothing detected".
  - A post-calibration oracle file would carry a calibrator fit on minival, which under D1(b)
    overlaps our test split.
  - Replay cells need an `AMENDMENTS.md` entry.
  - *Fix:* replay files are `ian_data` with a provenance sidecar; the stage must be `raw`;
    precision and target are checked; the fingerprint covers the sidecar.
- **B3. A `sign/<slug>` branch swaps the reviewer.**
  - `required_reviewers` falls back to any configured reviewer when no prefix matches.
  - *Fix:* push signed history as `claude/<slug>-signed`, or back to `claude/<slug>`.
- **B4. Weights.**
  - COCO AP's precision envelope depends on the order of tied detections. Weighted counts
    differ from physical copies when a tie group mixes hits and misses.
  - Isotonic flat blocks make such ties common, and LaACE0 has the same problem.
  - *Fix:* drop weights and build resampled datasets under fresh ids, or define weights as
    adjacent copies and limit exact checks to order-free metrics.
- **B5. Prefix equivalence.**
  - A top-k cut inside `match` is not representable.
  - A tie-break that depends on which detections are present breaks the equivalence.
  - The smoke fixture has no more than 100 detections per image and no greedy/optimal
    conflict, so the contract suite could not catch either violation.
  - *Fix:* MUST rules for per-class greedy matching, a presence-independent tie-break and no
    cut, plus synthetic contract cases.

## Non-blocking (summary)

| Finding | What it said |
|---|---|
| N1 | The "score unchanged" check and the missing raw-to-calibrated link ruled out raw-order matching. Platt at a = 0 makes a class one tie group. |
| N2 | Pipeline steps were out of order. |
| N3 | The calibrator monotonicity argument needed rewording. |
| N4 | Crowd-matched fit detections need a target rule. |
| N5 | "Platt does not change AP" fails at a = 0 and for clipped scores. Isotonic clipping collapses low scores, and "without u_c" needs a new code path. |
| N6 | The cache claim was false. |
| N7 | Curves in records would bloat the index, and NaN would be written as invalid JSON. |
| N8 | The leakage-checker acceptance item could not pass. An invariance test and a spy are needed instead. |
| N9 | The adapter was unnecessary. |
| N10 | Where `interface_version` lives was ambiguous. |
| N11 | The COCO round trip is not byte-exact. |
| N12 | The review path should be `review/claude/<branch-slug>.md`. |
| N13 | Cite AGENTS.md for "only Ian's loop reports metrics". |
| N14 | The `two_threshold.py` docstring was still wrong. |
| N15 | Signing prerequisites, `--no-track`, dropped merges, and what the signature vouches for. |
| N16 | The claims hook checks files, not chat. |
| N17 | AGENTS.md forbids other agents from editing the enforcement surface. |
| N18 | Private reporting is not on yet, and the protected list was incomplete. |
| N19 | Four PRs share PR #4, and a later commit invalidates the cross-review. |
| N20 | Paper values in the cycle plan were not marked unverified. |
| N21 | The D7 rationale was unverified. |
| N22 | D1 needs a rule 3 rewording and an amendment; PR-F depends on D8. |
| N23 | Status words differed, and the proposal had no Tasks section. |
| N24 | Line references drift from HEAD. |

## Template questions

1. **Leakage.** The interface itself is clean. The risks were replay files (B2), no guard
   that evaluate ids never reach `match` (N8), and manifests signed unread (B1).
2. **Untraceable numbers.** The diff contains no results. The cycle plan's paper values were
   unreferenced (N20). B1 would have let fabricated run records in under Ian's signature.
3. **Spec drift.** Replay cells and `split_design` both need `AMENDMENTS.md` entries (B2,
   N22).
