---
name: change-proposal
description: Draft a change proposal in docs/changes/ from the template before non-trivial work, naming the gate it serves and every enforcement-surface path it touches.
argument-hint: "<slug> <one-line goal>"
disable-model-invocation: true
allowed-tools: Read, Glob, Grep, Write(docs/changes/*), Bash(qcal policy check:*), Bash(git log:*)
---
Arguments: $ARGUMENTS. The first word is the slug (lowercase, hyphens); the rest is the goal.

1. Read docs/changes/TEMPLATE.md and two recent proposals in docs/changes/ for tone and depth.
2. Read DECISIONS.md (if present) and docs/SDLC_IMPLEMENTATION_PLAN.md for the gate this work
   serves. If no gate fits, say so; do not invent one.
3. List the paths the work will touch and run `qcal policy check <paths>`. Every path in
   `enforcement_surface` or `ian_only` goes under "What changes" with the note that it needs
   Ian's signed commit; Ian-only content (claims, abstract, conclusions, EXPERIMENTS.yaml) is
   out of scope for agents and goes under "Out of scope".
4. Write docs/changes/<slug>.md with every template heading. Acceptance items must be
   commands that can pass or fail (`make check`, `qcal leakage`, a named test), not
   intentions.
5. Branch name: claude/<slug>. The reviewer is the other model; its review lands at
   review/<reviewer>/claude-<slug>.md.

Never write numbers, results or claims into a proposal. Propose; do not start the work.
