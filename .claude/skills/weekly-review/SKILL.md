---
name: weekly-review
description: Week N review against the gates and kill criteria in DECISIONS.md, hours from RESEARCH_LOG.md, the WIP limit, open amendments and registry coverage.
argument-hint: "<week-number>"
---
Week: $ARGUMENTS.

Gather evidence first, then judge. Read:
- RESEARCH_LOG.md for planned versus actual hours this week and last week;
- DECISIONS.md for gates, kill criteria and the WIP limit;
- AMENDMENTS.md for open amendments;
- `git log --since='7 days ago' --format='%G? %h %s'` for what changed and whether protected
  changes were signed (G = good signature);
- `qcal registry audit --json` for pre-registered coverage and placeholders (run it; if
  EXPERIMENTS.yaml does not exist yet, say so).

Report, in this order:
1. Hours planned versus actual. If two consecutive weeks are under 6 hours, recommend dropping H3.
2. Scope creep: work outside the current change proposal in docs/changes/.
3. Each upcoming gate as green, amber or red, with the evidence for the colour.
4. WIP-limit violations.
5. Recommendation: continue, amend, kill or switch, citing DECISIONS.md.
6. The one task Ian must do by hand this week.

Never invent hours, gate states or results. If an input is missing, say exactly what cannot be
assessed.
