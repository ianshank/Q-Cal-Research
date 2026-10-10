---
name: prior-art
description: Monthly novelty re-check of the Q-Cal claim, run by the read-only prior-art-scout agent; the caller saves the returned report to review/prior-art/<date>.md.
argument-hint: "<claim>"
context: fork
agent: prior-art-scout
---
Claim: $ARGUMENTS

You have no conversation history; everything you need is here.

1. Read the newest file under `review/prior-art/`, if there is one. Pass that path
   explicitly to Glob or Read.
2. Search for 2019–2026 papers that measure detector calibration after INT8 PTQ or QAT, on
   edge accelerators, or under domain shift. The metrics are ECE, D-ECE, LaECE, LaECE0,
   LaACE0, LRP and OCE. Start from the two papers the plan positions against: arXiv:2609.16085
   (INT8 portability on Jetson, no calibration metrics) and arXiv:2412.01782 (DETR
   calibration, FP32 only).
3. Return the report as Markdown with these parts:
   - a ranked list: title, venue and year, arXiv ID or DOI, overlap, difference, and
     `[unverified]` where an ID could not be resolved;
   - a verdict: done, incremental or open;
   - the most dangerous related paper;
   - three ways to narrow the claim;
   - a "new since last check" section.

You write nothing; the caller saves the report.
