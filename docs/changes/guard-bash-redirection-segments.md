# change: guard-bash-redirection-segments

## Status
Proposed, Oct 11, 2026. Edits the enforcement surface (`src/qcal/hooks/bash.py`), so it needs
Ian's signed commit; meant for signing session 1 (Oct 17) with the other registry changes.

## Gate served
G0 (Oct 23). The guard must not block legitimate commands at random, or agents learn to work
around it.

## Why
`guard-bash` crashed with `IndexError: list index out of range` and blocked the command (it
fails closed) when a shell segment consisted only of a redirection. A here-document line such
as `assert max(abs(ga), abs(gb)) < 1e-9` produces the segment `< 1e-9` after the parentheses
split it. The cause is in `tokenize` (`hooks/bash.py:89-105`): it dropped empty segments
before removing redirections, so a segment emptied by that removal reached
`segment[0]` in `analyze`. Found while writing PR-A2's tests; verified against `d125717`.

Failing closed means this was never a bypass, only a false block.

## What changes
`tokenize` drops segments that are empty after redirections are removed. One line.

## Decisions
- **Fix the tokenizer, not the caller.** Every consumer of `tokenize` assumes non-empty
  segments; the guarantee belongs where segments are made.

## Compatibility
No behaviour change for any command the guard already handled; commands that crashed it are
now analysed.

## Out of scope
The guard's wider parsing limits (here-documents are analysed as if they were commands) stay
as documented in `docs/AGENT_LAYER.md`.

## Ian decisions requested
1. Sign this change in session 1 (Oct 17).

## Ian hours
About 5 minutes.

## Acceptance
- [ ] `make check` green (lint, types, tests with coverage gate)
- [ ] the crashing command is analysed, and a push to a protected branch inside a
      here-document command list is still found
- [ ] Ian's signed commit

## Tests
- unit: `tests/unit/test_hooks_bash.py` (a redirection-only segment; the exact command that
  crashed)
