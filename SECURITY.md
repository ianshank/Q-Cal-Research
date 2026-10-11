# Security policy

Q-Cal is a research repository. Its security matters because the integrity layer decides
which numbers can enter the paper: a bypass is a way to publish a result that no registered
run produced.

## Reporting a vulnerability

Report privately through GitHub's private vulnerability reporting on this repository
(Security → Report a vulnerability). Ian turns it on when the repository is public; until
then, open an issue that asks for a private contact and gives no details. Do not describe a
bypass in a public issue.

Include the commit you tested, the command or file that shows the problem, and what it lets
someone do. You will get an acknowledgement within a week; this is a one-person project, so
fixes land with the next signed change.

## In scope

- **The signed-commit control.** A way to change any path in `signing.signed_categories`
  (the enforcement surface, Ian-only and `ian_data` files, run records, committed reviews)
  that `qcal ci verify-signatures` accepts without Ian's signature.
- **The CI judge.** A pull request that changes the code, policy or keys that judge it, or
  that gets its head executed by `integrity.yml`.
- **The registry.** A record that `qcal ci registry-immutable` accepts but that rewrites or
  deletes an earlier one; a registered run whose record does not describe what ran
  (configuration, git state, pre-registration, inputs).
- **Claims.** A number in `paper/`, `README.md` or `CLAIMS.md` that `qcal claims` accepts
  without a run behind it.
- **Splits.** A path by which calibrator fitting or threshold selection reads the evaluate split
  that `qcal leakage` or the run gates do not catch.
- **Secrets.** Anything committed that `make gitleaks` should have caught.

## Out of scope

- The local hooks under `.claude/hooks/`. They are feedback for agents, not a security
  boundary; signed commits plus CI are the control. A hook bypass is welcome as a bug report
  but is not a vulnerability unless CI also accepts the result.
- Anything that needs write access to the default branch or Ian's signing key.
- Third-party datasets, model weights and the oracle code; report those upstream.

## Clean-room

Never send employer material, credentials or data from any private source, even to show a
problem. Use public data and synthetic examples only.
