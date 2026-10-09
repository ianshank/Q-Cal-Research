# Q-Cal: SDLC Implementation Plan for the Agentic Tool Execution Layer

Status: proposal, v1 (Oct 9, 2026). Source: *Adversarial Review of Plans A/B and a Corrected 12-Month CV Research Plan* (the "Review"), sections C (Corrected Plan), D (Tool Execution Layer), and F (risks).

This document is the SDLC team's reading of the Review and a concrete, phased plan to implement it with Claude Code (agents, subagents, skills, hooks, MCP servers, headless runs), Claude chat (research advisor Project), and Google Antigravity (cross-model review). It does not restate the career or legal advice in the Review; it implements the engineering system that makes the Jan 31, 2027 artifact possible.

Claim tags follow the Review's convention: [Certain] / [Likely] / [Guessing].

---

## 0. Executive summary

The Review's diagnosis is that the gap is *finishing*, not skill: ~19 projects, zero completed public, benchmarked CV training artifacts [Likely]. The engineering consequence is that the tool layer must be built to **force one experiment through the loop run → table → write → publish** and to make it mechanically hard to (a) fabricate or mistype a number, (b) leak test data, (c) drift from the pre-registration, or (d) start a second project.

The team's plan, in one paragraph: turn this repository into the `qcal/` repo from Review §D2, with a hook-enforced ownership boundary (agents own `src/`, `configs/`, `tests/`, `scripts/`, paper build; Ian owns `EXPERIMENTS.yaml`, `*/handwritten/`, `CLAIMS.md`, abstract/claims/limitations), ten narrowly-scoped subagents, seven slash-command skills that encode the weekly rituals and gates, a project `.mcp.json` limited to read-only research and docs servers, a headless nightly runner with a hard dollar budget, a CI pipeline whose smoke test is the merge gate, and an Antigravity/Gemini cross-review lane. Delivery is phased against the Review's 16-week schedule (G0 Oct 23 → G4 Jan 31) so that the tool layer never gets ahead of the science it serves.

What this plan deliberately does **not** automate (Review §D2 "What Ian must do by hand"): the FP32 eval loop and LaECE0 implementation, the QAT loss derivation through the straight-through estimator, hypotheses and kill criteria, the ≥20-case failure analysis, the abstract/claims/limitations, `DECISIONS.md` rationales, and one manual end-to-end Jetson TensorRT INT8 build. The agent layer is built to *block* itself from doing those.

---

## 1. SDLC team analysis of the Review

Six roles read the Review; each reports what it constrains and what it decided.

### 1.1 Product owner / principal investigator (Ian, with Claude chat as advisor)

- **Scope is fixed by pre-registration.** The Review gives a research question, three hypotheses (H1–H3), a falsifier, five baselines, six ablation axes, three seeds, metrics (AP, LRP, D-ECE, LaECE0, LaACE0, Jetson latency/throughput), and four kill criteria with dates (K1 Nov 8, K2 Dec 6, K3 Jan 10, K4 counsel). These become `EXPERIMENTS.yaml` and `DECISIONS.md` **before** any code beyond scaffolding is merged.
- **WIP limit is a product rule, not advice:** one primary project plus one batch-run slot (≤2 h/week attention) [Likely]. The plan encodes this in the weekly-review skill and the `DECISIONS.md` template.
- **Definition of Done (Review §C4)** is adopted verbatim as the release checklist (§7 below).
- **Decision:** the backup lane D-Cal gets a stub `configs/experiment/dcal/` and a `DECISIONS.md` entry only. No D-Cal code before G0 resolves. If G0 switches lanes, the same agent layer applies unchanged; only datasets, metrics modules, and `EXPERIMENTS.yaml` differ.

### 1.2 Architect / tech lead

- The Review's repo tree (§D2) is sound. The team adds three things the tree implies but does not name: a `registry/` CLI that is the *only* write path to `runs/index.csv`; a `protocols.py` dependency-injection seam so handwritten modules and agent-written modules can be swapped under parity tests; and a `review/` directory where cross-model reviews land as files (so reviews are versioned, diffable artifacts, not chat).
- **Hard ownership boundary** is enforced three ways, in order of strength: a `PreToolUse` hook that denies edits (exit 2) to Ian-only paths; a CI job that fails any PR touching those paths unless the PR carries the `ian-authored` label; and `CODEOWNERS`. Hooks catch the agent locally, CI catches anything that slips, CODEOWNERS catches humans.
- **Hydra configs, no hardcoded values** (Review §D2 Conventions). Every run is `experiment=<id>` where `<id>` must exist in `EXPERIMENTS.yaml`; the runner refuses otherwise.
- **Decision:** default branch is renamed from `civ` to `main` in Phase 0 (the Review's `AGENTS.md` and all tooling assume `main`). Branch namespaces `claude/*`, `gemini/*`, `ian/*`.

### 1.3 ML engineer

- Reproduction first (G1/K1): the FP32 Kuzucu baselines (Platt, isotonic, class-wise isotonic, two-threshold calibrators; D-ECE, LaECE0, LaACE0, LRP) must match published numbers within ±1.0 LaECE0 / ±0.5 AP for ≥1 detector by Nov 8, or fall back to repo checkpoints. The `paper-reproducer` subagent writes *parity tests against the public `fiveai/detection_calibration` outputs*; Ian hand-writes the eval loop and LaECE0; the agent's job is to prove Ian's implementation matches the reference, not to write it.
- **The fiveai reference is CC BY-NC-SA 4.0, not Apache** [Certain, from the research check in Appendix E]. It is therefore used only as an unmodified test oracle (installed dependency in `tests/parity/`), never vendored or ported line-by-line. Ian's `handwritten/laece.py` and the agent-written calibrators are clean-room implementations from the paper; `paper-reproducer` writes the parity tests against the oracle's outputs. This matches the Review's clean-room rule 4 and removes a licensing problem for the public release.
- Detector stack: MMDetection 3.x (Apache-2.0; last release v3.3.0, May 2024, so mmcv/mmengine are pinned) with ATSS, **plain DETR-R50** as the DETR variant, and YOLOX-s. Deformable-DETR/DINO need the TensorRT multiscale-deformable-attention plugin and are absent from the Hailo zoo, so plain DETR keeps one ONNX graph portable to both targets [Certain per mmdeploy and Hailo zoo listings]. **Ultralytics is excluded** (AGPL-3.0) and the license-auditor subagent greps for it.
- Quantization. PTQ: TensorRT on Jetson. JetPack 6.2 ships TensorRT 10.3, which still supports implicit INT8 calibration (`trtexec --int8 --calib`, entropy/minmax/percentile calibrators); TensorRT 11 removes implicit calibration and accepts only explicit Q/DQ ONNX (NVIDIA ModelOpt) [Certain]. The PTQ axis is therefore {entropy, minmax, percentile} × {implicit TRT10, explicit Q/DQ}, and engines are rebuilt and re-timed per TensorRT version. QAT: `torch.ao.quantization` is deprecated; use torchao PT2E (`prepare_qat_pt2e` / `convert_pt2e`) or NVIDIA ModelOpt when the target is a TensorRT Q/DQ graph [Certain]. Export the backbone+neck+head without post-processing; keep Hungarian matching and NMS outside the exported graph. Calibrator fitting on the calib split only; selection on val only; test touched once per registered run.
- **Decision:** ONNX export is a first-class artifact with its own parity test (FP32 PyTorch vs FP32 ONNX Runtime within tolerance) before any INT8 number is trusted. Export raw logits before sigmoid/softmax and before NMS so calibration metrics use raw scores; fix input shapes (dynamic-shape calibration corrupts ranges, per arXiv:2609.16085); DETR runs single-batch.
- **Decision:** Hailo-8 is a separate arm, not a second measurement of the same INT8 model: the Dataflow Compiler quantizes natively and does not honour bring-your-own Q/DQ graphs, and the zoo has YOLOX-s and DETR-R18 but not ATSS [Likely]. Stays in the 15 h/week variant as the Review specifies.
- **Flag for Ian (pre-registration, not decided here):** arXiv:2412.01782 shows D-ECE and LaECE0 can be gamed by score thresholding. The prior-art check recommends pre-registering a threshold-robust secondary metric (OCE or similar). This belongs in `EXPERIMENTS.yaml`, which only Ian edits.

### 1.4 MLOps / infrastructure engineer

- Compute topology (Review §D5): GPU0 trains/QAT, GPU1 evals/sweeps, **no DDP across mismatched cards**; P40 is FP32-only in a separate environment (older CUDA arch; verify `sm_61` support); Jetson does TensorRT builds and latency only; Hailo-8 only in the 15 h/week variant; cloud only for DETR-class QAT, capped at $200/artifact.
- Run registry: MLflow local (gitignored) for artifacts; `runs/index.csv` committed as the single source of truth for tables. Every row logs seed, git SHA, config hash, GPU, driver, CUDA, package versions. DVC tracks dataset manifests and checkpoints.
- Nightly: a cron on ian-pc launches `claude -p` with `--allowedTools "Read,Bash,Grep,Glob"`, `--max-budget-usd 3`, `--output-format json`, driving the `experiment-runner` subagent; output lands in `runs/nightly/`. The headless agent can launch and register, never edit.
- CI (GitHub Actions, CPU): ruff, mypy, pytest unit, 20-image smoke (`make smoke` < 5 min). GPU parity tests run on a self-hosted runner label `gpu` on ian-pc, non-blocking until Phase 2.

### 1.5 QA / research-integrity engineer

- Four read-only auditor subagents form the integrity chain that runs before any table is built or any PR merges: `data-leakage-checker` (hash image IDs per split; grep for test-based selection), `ablation-auditor` (EXPERIMENTS.yaml vs runs/index.csv; never proposes dropping cells), `results-table-builder` (only via `make tables`; every cell carries `run:<id>`), `adversarial-reviewer` (hostile review of a diff, table, or section; opus).
- `check_claims.sh` on `Stop` warns on any decimal number in `paper/sections/*.tex` without a `run:` tag. CI runs the same check as a hard failure on `paper/`.
- Pre-registration drift is handled only through `AMENDMENTS.md` (date, reason, what changed, which runs are affected). The hook denies edits to `EXPERIMENTS.yaml`; the weekly-review skill lists open amendments.
- **Decision:** the claims-audit output format from the open PR #1 README (Sentence / Claimed result / Evidence-run ID / Assessment / Suggested fix; `supported` / `overstated` / `unsupported` / `untraceable`) is adopted as the `/claims-audit` skill's schema so that chat (Claude Project) and code (Claude Code) use the same vocabulary.

### 1.6 Security / IP-compliance engineer

- Clean-room rule (Review §A9) is encoded, not just documented: `DECISIONS.md` carries the dataset allowlist; `license-auditor` subagent checks every dependency and dataset card; the `guard_paths.sh` hook also denies any path containing `nbcu`, `edge-dit`, `edge_dit`, or `edgedit` (case-insensitive) [team addition]; and `RESEARCH_LOG.md` entries are the personal-time log the Review asks for.
- **Q-Cal code stays private until G0 clears** (Review §F2 risk 1). The repo visibility flip is a Phase 2 task gated on `DECISIONS.md` recording counsel's view.
- Antigravity operating rules (Review §D3): terminal auto-execute off, no secrets in the workspace, only local read-only MCP servers, review-only role (`.agents/rules/research-integrity.md` with `trigger: always_on`). The documented Antigravity vulnerabilities (prompt-injection → RCE, credential exfiltration) are the reason Antigravity never gets write access to `src/` and runs in a separate worktree.
- MCP servers for Claude Code are limited to read-only research/doc servers plus GitHub; no MCP server that can write to Drive/Notion is enabled for subagents (§3.5).
- No secrets in `.mcp.json`; tokens via environment variables only; `.env` gitignored; `run_secret_scanning` on the repo at each phase gate.

### 1.7 Release / documentation engineer

- Release artifacts (Review §C4 and §E): public repo with license, dataset cards, clean-room provenance note; `make reproduce`; ≥3 seeds with 95% CIs traced via `CLAIMS.md`; 6–8 page LaTeX report; HF model cards for FP32/INT8/QAT checkpoints with calibration metrics and a Space showing FP32 vs INT8 reliability diagrams; arXiv submission with endorser; venue (TMLR or CVPR 2027 workshop; ReScience C for the reproduction half).
- `report-writer` subagent does LaTeX build and bibliography hygiene (every `\cite` resolves to arXiv ID or DOI), with *marked* copy-edits only. It never authors claims.
- `hf-publisher` subagent (team addition) pushes model cards and the Space from `make release` using the Hugging Face MCP/CLI; it reads metrics from `runs/index.csv` only.

---

## 2. Target repository architecture

Adopted from Review §D2 with the additions from §1.2.

```
qcal/  (this repository, root)
├── AGENTS.md                 shared rules: Claude Code + Antigravity
├── CLAUDE.md                 @AGENTS.md + integrity rules + conventions + compute
├── RESEARCH_LOG.md           personal-time log (Ian-only)
├── EXPERIMENTS.yaml          pre-registration (Ian-only; hook-denied)
├── AMENDMENTS.md             dated amendments to EXPERIMENTS.yaml
├── DECISIONS.md              gates, kill criteria, dataset allowlist, lane choice
├── CLAIMS.md                 every number in paper/README → run IDs (Ian-only)
├── CODEOWNERS
├── LICENSE                   Apache-2.0 (code); dataset licenses in data cards
├── configs/                  Hydra: model/ data/ quant/ calib/ experiment/
├── src/qcal/
│   ├── protocols.py          Protocol classes: Detector, Calibrator, Metric, Quantizer, Runner
│   ├── registry/             registry CLI: the only writer of runs/index.csv
│   ├── data/                 COCO (+Kuzucu ID/OOD splits), COCO-C, Cityscapes/Foggy loaders; split hashing
│   ├── models/               MMDetection wrappers (ATSS, DETR-variant, YOLOX-s); ONNX export
│   ├── quant/                PTQ (TensorRT builders), QAT (PyTorch), observers
│   ├── calib/                Platt, isotonic, class-wise isotonic, two-threshold (parity-tested)
│   ├── metrics/              AP, LRP, D-ECE, LaECE0, LaACE0 (parity-tested vs fiveai reference)
│   └── handwritten/          Ian-only: eval_loop.py, laece.py, qat_loss.py (hook-denied)
├── analysis/
│   ├── figures/              matplotlib → vector PDF; colorblind-safe; 95% CI error bars
│   └── handwritten/          Ian-only: error_analysis.md (≥20 cases/condition)
├── notebooks/                Antigravity-owned, exploratory only; never a source of numbers
├── review/                   cross-model reviews as files: review/<branch>.md
├── runs/                     MLflow local (gitignored); runs/index.csv committed; runs/nightly/
├── scripts/                  GPU-pinned launchers; jetson_*.sh; hailo_*.sh; nightly.sh
├── tests/                    unit; parity vs fiveai reference outputs; split-leakage tests
├── paper/                    LaTeX (CVPR style); sections/; figures/; refs.bib
├── docs/                     this plan; data cards; provenance note; change proposals
├── .claude/
│   ├── agents/               subagents (§3.2)
│   ├── skills/               slash-command skills (§3.3)
│   ├── hooks/                guard_paths.sh, guard_bash.sh, check_claims.sh, session_start.sh, log_subagent.sh
│   └── settings.json         hooks + permissions
├── .mcp.json                 project MCP servers (§3.5)
├── .agents/                  Antigravity: rules/, skills/cross-review/, mcp_config.json
├── .github/workflows/        ci.yml, paper.yml, gpu-parity.yml (self-hosted, non-blocking)
├── dvc.yaml                  dataset manifests, checkpoints
└── Makefile                  smoke, reproduce, tables, paper, release, leakage, nightly
```

**Ownership map (enforced by hook + CI + CODEOWNERS):**

| Path | Owner | Agents may |
|---|---|---|
| `EXPERIMENTS.yaml`, `CLAIMS.md`, `*/handwritten/*`, `paper/sections/{abstract,claims,limitations,conclusion}.tex`, `RESEARCH_LOG.md`, `DECISIONS.md` rationale text | Ian | read, test, propose in `AMENDMENTS.md` or `review/` |
| `src/`, `configs/`, `tests/`, `scripts/`, `paper/` (build + non-claim sections), `Makefile`, `.github/`, `.claude/` | Claude Code | edit via `claude/*` branches + PR |
| `notebooks/`, `analysis/figures/`, `review/gemini-*.md` | Antigravity | edit via `gemini/*` branches + PR |
| `runs/index.csv` | registry CLI only | never hand-edited by anyone |

---

## 3. Agent layer design (Claude Code)

Facts about Claude Code features below were re-verified against the official docs on Oct 9, 2026 by the `claude-code-guide` agent; see §3.7 for the verification notes and anything that could not be confirmed.

### 3.1 Principle: least privilege per role

Every subagent gets the minimum tool set, the cheapest adequate model, and (where it writes code) an isolated worktree. Read-only auditors get `disallowedTools: Edit, Write` *and* omit those tools from `tools`, so the restriction holds even if defaults change. No subagent gets `permissionMode: bypassPermissions`.

### 3.2 Subagents (`.claude/agents/`)

Seven from the Review §D2, plus three team additions (marked +).

| Agent | Model | Tools | Isolation | Purpose / hard rule |
|---|---|---|---|---|
| `paper-reproducer` | sonnet | Read, Grep, Glob, Edit, Write, Bash | worktree, maxTurns 40 | Implement a published baseline exactly; cite reference file/line per function; write parity tests vs reference outputs; report every deviation. |
| `experiment-runner` | haiku | Read, Bash, Grep, Glob (no Edit/Write) | — | Launch and register only run IDs present in `EXPERIMENTS.yaml`, on pinned GPUs, via the registry CLI. On failure: collect logs, stop. Never change configs. |
| `ablation-auditor` | sonnet | Read, Grep, Glob, Bash (no Edit/Write) | — | Diff `EXPERIMENTS.yaml` vs `runs/index.csv`; flag missing cells, unregistered runs, seed gaps. Never propose dropping cells. |
| `data-leakage-checker` | sonnet | Read, Grep, Glob, Bash (no Edit/Write) | — | Hash image IDs per split; grep configs/logs for test-based selection; output PASS/FAIL with evidence. Runs before any table. |
| `results-table-builder` | haiku | Read, Bash, Write | — | Build tables only via `make tables` from `runs/index.csv`; every cell carries `run:<id>`. Never types a number. |
| `report-writer` | sonnet | Read, Edit, Bash | — | LaTeX build, bibliography hygiene, *marked* copy-edits of Ian's prose. Never authors claims. Verifies every `\cite` has arXiv ID or DOI. |
| `adversarial-reviewer` | opus | Read, Grep, Glob | — | Read-only hostile review of a diff, table, or section. Leads with the most damaging weakness: leakage, missing baselines, overclaiming, untraceable numbers, spec drift. Also reviews every `gemini/*` branch. |
| + `prior-art-scout` | sonnet | Read, Write (to `review/prior-art/` only), WebSearch, WebFetch; MCP: alphaXiv, Hugging Face | — | Monthly novelty re-check (Review §F2 risk 4). Writes `review/prior-art/<date>.md` with title/venue/arXiv ID/overlap/difference; marks unverified. Never edits paper/. |
| + `license-auditor` | haiku | Read, Grep, Glob, Bash (no Edit/Write) | — | Checks every dependency, dataset card and checkpoint license against `DECISIONS.md` allowlist; fails on AGPL (Ultralytics), BDD100K unless cleared, any non-commercial-only asset used for a public release. |
| + `hf-publisher` | sonnet | Read, Bash; MCP: Hugging Face | — | `make release` only: pushes model cards (metrics read from `runs/index.csv`), dataset cards, and the Space. Refuses if `data-leakage-checker` has not PASSed on the current SHA. |

Full frontmatter and bodies are in Appendix A.

### 3.3 Skills (`.claude/skills/<name>/SKILL.md`)

Skills encode the rituals so they are invoked by name and run the same way every time.

| Skill | Invocation | What it does | Delegates to |
|---|---|---|---|
| `change-proposal` | `/change-proposal <slug>` | Creates `docs/changes/<slug>.md` from the Review's template (Why / What changes / Out of scope / Acceptance / Tasks), opens a `claude/<slug>` branch. | — |
| `reproduce-check` | `/reproduce-check` | Runs `make smoke`, `make leakage`, parity tests; reports gaps vs G1/K1 thresholds (±1.0 LaECE0, ±0.5 AP). | `data-leakage-checker`, `paper-reproducer` |
| `nightly` | `/nightly <experiment-glob>` | Validates IDs against `EXPERIMENTS.yaml`, pins GPU, launches via `experiment-runner`, writes `runs/nightly/<date>.json`. Used by the cron headless run. | `experiment-runner` |
| `tables` | `/tables` | Leakage check → `make tables` → ablation audit → diff of `paper/tables/*.tex`. | `data-leakage-checker`, `results-table-builder`, `ablation-auditor` |
| `claims-audit` | `/claims-audit <tex-or-md>` | Produces the Sentence / Claimed result / Evidence-run ID / Assessment / Suggested fix table (schema from PR #1). | `adversarial-reviewer` |
| `weekly-review` | `/weekly-review <N>` | Hours planned vs actual (from `RESEARCH_LOG.md`), scope creep, each upcoming gate green/amber/red with evidence, WIP-limit check, open amendments, the one hand-task for the week. Never invents hours or gate states. | `ablation-auditor` |
| `prior-art` | `/prior-art "<claim>"` | Monthly novelty check via alphaXiv/HF MCP. | `prior-art-scout` |
| `release-check` | `/release-check` | Walks the Definition of Done (§7); blocks `make release` on any unchecked item. | `license-auditor`, `data-leakage-checker`, `hf-publisher` |

Full `SKILL.md` texts in Appendix B.

### 3.4 Hooks (`.claude/settings.json` + `.claude/hooks/`)

| Event | Matcher | Script | Behaviour |
|---|---|---|---|
| `PreToolUse` | `Edit\|Write\|MultiEdit\|NotebookEdit` | `guard_paths.sh` | Exit 2 (deny) on Ian-only paths and on any path matching `nbcu`/`edge[-_]?dit` (case-insensitive). Message: "BLOCKED: <path> is Ian-only. Propose in AMENDMENTS.md." |
| `PreToolUse` | `Bash` | `guard_bash.sh` | Deny: `git push --force*`, `git push origin main`, in-place edits of protected files via `sed -i`/`tee`/`>` redirection, `rm -rf runs`, `dvc remove`, any command touching `EXPERIMENTS.yaml` or `runs/index.csv` except through `qcal-registry`. |
| `PostToolUse` | `Edit\|Write` | `format.sh` | `ruff format` + `ruff check --fix` on changed `.py`; no-op otherwise. |
| `Stop` | — | `check_claims.sh` | Warn on decimal numbers without `run:` tags in `paper/sections/*.tex`. |
| `SubagentStop` | — | `log_subagent.sh` | Append agent name, SHA, timestamp, and summary line to `runs/agent_log.jsonl` (gitignored) for the weekly review. |
| `SessionStart` | — | `session_start.sh` | Cloud sessions: create venv, `pip install -e .[dev]`, pull the 20-image smoke fixture via DVC, print GPU/driver facts. Local: verify `CUDA_VISIBLE_DEVICES` is set. |

Scripts in Appendix C.

### 3.5 MCP servers (`.mcp.json`, project scope)

Read-only research and documentation servers only. GitHub is the one write-capable server, scoped to this repository for PRs, issues, and CI status.

| Server | Used by | Why |
|---|---|---|
| alphaXiv | `prior-art-scout`, `/prior-art`, Claude chat Project | Verified paper search, arXiv ID resolution, PDF Q&A. Replaces "invent a citation". |
| Hugging Face | `prior-art-scout`, `hf-publisher` | Model/dataset card reads; checkpoint and Space publishing at release. |
| Context7 | `paper-reproducer`, main session | Current MMDetection, TensorRT, torch.ao / torchao quantization docs instead of training-data memory. |
| GitHub | main session only | PRs, CI status, issue tracking, secret scanning. Not exposed to subagents. |
| Mermaid Chart | main session | Pipeline diagrams for docs/README (not paper figures). |

Explicitly **not** enabled for agents: Google Drive, Notion, Replit, Vercel, any server with write access to documents outside the repo. The Review's Antigravity security notes (§D3) apply to Claude Code MCP hygiene as well: no credentials in `.mcp.json`; tokens via `${ENV_VAR}`.

### 3.6 Headless nightly runner

`scripts/nightly.sh` (cron, ian-pc, 01:00 local):

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
claude -p "Use the nightly skill to launch $NIGHTLY_GLOB from EXPERIMENTS.yaml. Register via qcal-registry. On any failure collect logs and stop." \
  --allowedTools "Read,Bash,Grep,Glob" \
  --max-budget-usd 3 \
  --max-turns 30 \
  --output-format json > "runs/nightly/$(date +%F).json"
```

Guardrails: hooks still apply in `-p` mode; `--allowedTools` excludes Edit/Write; dollar cap; the registry CLI validates every run ID against `EXPERIMENTS.yaml` before launch.

### 3.7 Verification notes on Claude Code features

The fact sheet in §A.0 was produced from the official docs on Oct 9, 2026. Design consequences already applied in the appendices:
- Unknown frontmatter keys are silently ignored, so Phase 0 includes a `tests/test_agent_frontmatter.py` that parses every `.claude/agents/*.md` and `.claude/skills/*/SKILL.md` and rejects keys outside the documented set.
- Hook matchers such as `Edit|Write|MultiEdit|NotebookEdit` are exact-name lists, not regexes [Certain]. The `MultiEdit` stdin schema is unverified, so `guard_paths.sh` reads `file_path`, `path`, and `notebook_path` and denies if none is present but the tool is an edit tool.
- Subagent `mcpServers` entries reference server names from `.mcp.json` (`alphaxiv`, `huggingface`), so those names must match exactly.
- Skills may run in a forked subagent via `context: fork` + `agent: <name>`; the plan uses this for `/prior-art` (agent `prior-art-scout`) and `/tables` steps, with self-contained skill bodies because a fork does not see conversation history.
- `--max-budget-usd` is a client-side estimate that includes subagent spend; the registry CLI's own run-count cap is the hard limit for nightly launches.
- `CLAUDE_CODE_REMOTE` is `"true"` in cloud sessions; `session_start.sh` keys on it.
- Plugin-packaged subagents lose `hooks`, `mcpServers`, and `permissionMode`, so the agent layer stays in-repo (`.claude/`), not in a plugin.

---

## 4. Claude chat (Project "Q-Cal Research") and Antigravity lanes

### 4.1 Claude Project
Upload `RESEARCH_LOG.md`, `EXPERIMENTS.yaml`, `DECISIONS.md`, `CLAIMS.md`, the Kuzucu PDF, 10 job postings, and weekly `runs/index.csv` exports. Project instructions and prompts (a)–(e) are the ones in Review §D1; the README in PR #1 is the expanded version and should be merged first so chat and code share it. The Project is the **owner** of prior art, pre-registration critique, and reviewer-2 passes; Claude Code's auditors are the second opinion, Gemini the third.

### 4.2 Antigravity (Gemini 3.1 Pro)
Role: interactive run debugging, notebooks, figures, HF Space browser check, and independent review of every `claude/*` PR via `.agents/skills/cross-review/SKILL.md` writing `review/<branch>.md`. Rules in `.agents/rules/research-integrity.md` (`trigger: always_on`) and `figures.md` (`trigger: glob` on notebooks and `analysis/figures/*.py`). Separate worktree `../qcal-gemini`. Terminal auto-execute off; only local read-only MCP servers. Migrate from IDE workflows to skills before Oct 19, 2026 (Review §D3) [Certain per Review].

### 4.3 Merge rule
A PR merges only with: smoke green, leakage PASS, no protected-path edits, `RESEARCH_LOG.md` entry, `CLAIMS.md` updated when numbers change, and the *other* model's review attached (`review/<branch>.md` for `claude/*`; `adversarial-reviewer` output for `gemini/*`).

---

## 5. Phased delivery plan (mapped to Review §C5)

Hours assume the 9 h/week baseline; 15 h/week extras in brackets. Each phase lists the agent-layer deliverables, the science deliverables they serve, and the gate.

### Phase 0: Freeze, scaffold, pre-register (Wk 1–2, Oct 12–25) → Gate G0 Oct 23

Agent-layer deliverables (Claude Code, ~6 h):
1. Merge PR #1 (review protocol README). Rename default branch `civ` → `main`; protect `main` (PR required, CI required, no force-push).
2. Commit `CLAUDE.md`, `AGENTS.md`, `CODEOWNERS`, `LICENSE`, `.gitignore`, `.claude/{agents,skills,hooks,settings.json}`, `.mcp.json`, `.agents/`, `Makefile` (targets stubbed to fail loudly), `.github/workflows/ci.yml` (lint/types/unit only; smoke added in Phase 1).
3. Templates: `EXPERIMENTS.yaml` (schema only, Ian fills), `AMENDMENTS.md`, `DECISIONS.md` (G0–G4, K1–K4, dataset allowlist), `CLAIMS.md`, `RESEARCH_LOG.md`, `docs/changes/TEMPLATE.md`.
4. Test the hooks: a scripted check that an Edit to `EXPERIMENTS.yaml` is denied and an Edit to `src/qcal/calib/platt.py` is allowed (`tests/test_hooks.sh`).
5. `/prior-art` first run → `review/prior-art/2026-10-14.md`. A first pass was done for this plan (Appendix E): the exact Q-Cal question appears open [Likely]; the two papers to position against are arXiv:2609.16085 (INT8 portability on Jetson, no calibration metrics) and arXiv:2412.01782 (DETR calibration, FP32 only). The week-1 run should confirm this and feed prompt (a) in the Claude Project.
6. Freeze the other repos: pin README notes only; no code moves.

Ian (by hand): read NBCU agreement; book attorney; email scientist; hand-draft `EXPERIMENTS.yaml`; red-team it with prompt (b); job-posting table; **G0 decision logged in `DECISIONS.md`**.

Exit criteria: hooks tested; CI green on scaffold; `EXPERIMENTS.yaml` committed with the `ian-authored` label; G0 recorded (Q-Cal or D-Cal).

### Phase 1: Clean-room FP32 reproduction (Wk 3–4, Oct 26–Nov 8) → Gate G1 / K1 Nov 8

Change proposal: `reproduce-kuzucu-eccv24-baselines` (Review §D2 template).
- Claude Code scaffolds `protocols.py`, `registry/`, `data/` (COCO + Kuzucu ID/OOD splits, split hashing), `models/` wrapper for detector #1 (ATSS), Hydra configs, `make smoke` (20-image fixture, < 5 min), `tests/test_splits.py`.
- Ian hand-writes `handwritten/eval_loop.py` and `handwritten/laece.py`.
- `paper-reproducer` (worktree) writes parity tests comparing Ian's LaECE0/D-ECE/LaACE0/LRP and the four calibrators to `fiveai/detection_calibration` reference outputs on the same predictions; reports deviations.
- `data-leakage-checker` PASS required before the first registered run.
- `/reproduce-check` reports the gap vs published numbers.
- Acceptance (from the template): `make smoke` < 5 min; LaECE0 ±1.0 / AP ±0.5 vs published for ≥1 detector with run IDs; leakage PASS; Ian's handwritten files present and parity-tested.

K1 (Nov 8): not within tolerance → one more week, then fall back to repo checkpoints and record the amendment. If G1 fails twice, the reproduction itself becomes artifact 1 (ReScience-style) per Review §F3.

### Phase 2: Detectors #2–3, ONNX, TensorRT PTQ, H1 (Wk 5–6, Nov 9–22)

- `models/`: DETR variant and YOLOX-s; ONNX export + ONNX-vs-PyTorch parity test.
- `quant/ptq/`: TensorRT builders (minmax, entropy) via `scripts/jetson_*.sh`. **Ian does one manual end-to-end Jetson INT8 build before the script exists** (Review §D2 hand-list item 7).
- `data/`: COCO-C severity 1/3/5 generation with `imagecorruptions` (Apache-2.0; pin numpy/scikit-image, the package is unmaintained) and an explicit, pre-registered corruption subset; Cityscapes → Foggy loaders from the official download only (non-commercial research license, registration, no redistribution; all HF mirrors are unlicensed re-uploads and are blocked by `license-auditor`). Licenses recorded in `DECISIONS.md` and `docs/data/`.
- `/nightly` cron goes live for INT8 eval sweeps on GPU1; Jetson latency/throughput logged per run.
- `/tables` produces the H1 table with paired 95% bootstrap CIs; `adversarial-reviewer` reviews it; Antigravity cross-review on the PR.
- [15 h/wk: Hailo DFC compile on x86, `scripts/hailo_*.sh`.]
- Repo visibility: still private unless `DECISIONS.md` records counsel clearance.

### Phase 3: H2 and calibration-set ablations; error analysis (Wk 7–8, Nov 23–Dec 6) → Gate G2 / K2 Dec 6

- `calib/`: calibrator fitting on INT8 outputs; ablation axes (calibrator-fit precision FP32/INT8; calibration-set size 500/2k/5k).
- `ablation-auditor` reports cell coverage weekly; `/weekly-review 7`, `/weekly-review 8`.
- Ian writes `analysis/handwritten/error_analysis.md` (≥20 cases per condition); Antigravity builds the figure notebooks.
- K2: H1 falsified → negative-result note path (`paper/` switches to the short-note template); skip H3.

### Phase 4: QAT with calibration-aware loss, H3 (Wk 9–13, Dec 7–Jan 10) → Gate G3 / K3 Jan 10

- Ian hand-derives the QAT loss and its gradient through the STE → `handwritten/qat_loss.py`; Claude Code wraps it under `protocols.Loss` and writes gradient-check tests (finite differences vs autograd through the fake-quant STE).
- `quant/qat/`: plain QAT and QAT+cal-loss; 3 seeds × 2 variants overnight on GPU0 (~36–90 GPU-h [Guessing per Review]); cloud only for DETR-class, ≤$200.
- `/nightly` drives the seed matrix; `experiment-runner` registers; `ablation-auditor` confirms completeness; `/tables`.
- Wk 13: `/claims-audit` on every result-bearing sentence; `adversarial-reviewer` on the full draft tables.
- K3: QAT unstable or no gain → report H3 as negative.

### Phase 5: Report, cross-review, release (Wk 14–16, Jan 11–31) → Gate G4 Jan 31

- Ian writes abstract, claims, interpretation, limitations. `report-writer` builds LaTeX, checks bibliography, marks copy-edits. `check_claims.sh` and CI's claims check must be clean.
- Prompt (d) Reviewer-2 pass in the Claude Project; Gemini review of the paper PR; `adversarial-reviewer` final pass.
- `/release-check` walks §7; `license-auditor` PASS; `hf-publisher` pushes model cards and Space. ONNX, TensorRT engine and Hailo HEF files go in HF *model* repos (Spaces have a small storage cap); each card records TensorRT version, JetPack, GPU SM and the git SHA, since engines are not portable across TensorRT versions. [15 h/wk: video demo on CC-licensed footage].
- `make reproduce` regenerates every table from a clean clone (CI job on a tagged release).
- arXiv submission (endorser lined up in Phase 0), public repo, resume v2.

### Post-G4 (Feb–Oct 2027), for completeness
Applications from Feb; workshop CFP verification; MS gate Feb 14; Stratégos M5 batch slot; TMLR/workshop submission Mar; AlphaGalerkin run after counsel (25 h cap); D-Cal to arXiv by end of June; optional agent-acceptance paper only if N is adequate. The same agent layer serves D-Cal with new `data/`, `metrics/` modules and a new `EXPERIMENTS.yaml`.

---

## 6. Weekly operating cadence (Review §D4, tool-mapped)

| Day | Hours | Activity | Tool / skill |
|---|---|---|---|
| Mon | 1.5 | Reading, design, `DECISIONS.md` | Claude Project |
| Tue–Thu | 4.5 [+3] | Implementation; `/change-proposal`; overnight runs via `/nightly` | Claude Code + Ian on `handwritten/` |
| Fri | 1 [+1.5] | Figures, `/cross-review`, HF Space check | Antigravity |
| Sat | 1.5 [+1.5] | Hand-written analysis | Ian |
| Sun | 0.5 | `/weekly-review N`; log hours in `RESEARCH_LOG.md`; prompt (e) in Project | Claude Code + Claude Project |

Hours-collapse rule (Review §F2 risk 5): the weekly-review skill flags two consecutive weeks under 6 h and recommends dropping H3.

---

## 7. Definition of Done (release checklist; Review §C4)

- [ ] Public repo with license, dataset cards, clean-room provenance note (`docs/PROVENANCE.md`)
- [ ] `make reproduce` regenerates every table from a clean clone (CI-verified)
- [ ] ≥3 seeds and 95% CIs; every number traced via `CLAIMS.md` (`/claims-audit` clean)
- [ ] Pre-registered `EXPERIMENTS.yaml` plus `AMENDMENTS.md` log
- [ ] All pre-registered ablations reported, negatives included (`ablation-auditor` clean)
- [ ] Failure analysis covering ≥20 cases per condition, written by Ian
- [ ] 6–8 page LaTeX report; `report-writer` bibliography check clean
- [ ] HF model cards and Space (`hf-publisher`)
- [ ] arXiv submission with endorser; named venue and deadline in `DECISIONS.md`
- [ ] `license-auditor` PASS; `data-leakage-checker` PASS on the release SHA; secret scan clean
- [ ] Other-model review attached to the release PR

---

## 8. Risk register (engineering view of Review §F2–F3)

| # | Risk | Likelihood | Mitigation in this plan | Owner |
|---|---|---|---|---|
| 1 | IP clearance stalls; Q-Cal must stay private | Likely | Repo private until `DECISIONS.md` records clearance; agent layer is lane-agnostic so D-Cal switch costs only `data/`, `metrics/`, `EXPERIMENTS.yaml` | Ian + security |
| 2 | Reproduction drift on Blackwell/MMDetection | Likely | Containerized reference env (`docker/reference.Dockerfile` pinned to fiveai's versions); parity tests; K1 timebox; repo-checkpoint fallback | ML eng |
| 3 | H1 null | Likely | Negative-result note template pre-built in `paper/`; still a finished artifact | PI |
| 4 | Scooped | Likely | `/prior-art` monthly via alphaXiv; narrowing options kept in `DECISIONS.md` | prior-art-scout |
| 5 | Hours collapse | Likely | `/weekly-review` two-week rule → drop H3 | PI |
| 6 | Agent fabricates or mistypes a number | Certain without controls | Only `results-table-builder` via `make tables`; `check_claims.sh`; CI claims check; `/claims-audit` | QA |
| 7 | Agent edits pre-registration or handwritten code | Certain without controls | `guard_paths.sh` deny; `guard_bash.sh` deny; CI protected-path job; CODEOWNERS | Architect |
| 8 | Test-split leakage | Likely without controls | `data-leakage-checker` before any table; `tests/test_splits.py`; calibrators fit on calib split only (config-enforced) | QA |
| 9 | Antigravity prompt-injection / exfiltration | Documented | Review-only role, separate worktree, auto-execute off, no secrets, local read-only MCP only | Security |
| 10 | Tool layer becomes the project (meta-work) | Likely given portfolio history | Phase 0 capped at ~6 h; every later tool change needs a change proposal tied to a gate | PI |
| 11 | DDP across mismatched GPUs, P40 arch drop | Likely | No DDP; P40 separate env; `session_start.sh` prints `torch.cuda.get_arch_list()` | MLOps |
| 12 | Cloud spend | Guessing | `--max-budget-usd` on headless; $200 cap per artifact in `DECISIONS.md`; cloud only for DETR-class QAT | MLOps |
| 13 | Reference code license (CC BY-NC-SA) contaminates the public release | Certain if vendored | Oracle-only use in `tests/parity/`; clean-room metrics and calibrators; `license-auditor` fails on any import of the reference outside tests | Security |
| 14 | TensorRT version drift (10.x implicit vs 11.x explicit INT8) invalidates engines mid-project | Likely | Pin JetPack in `DECISIONS.md`; registry logs TRT version per run; both PTQ paths pre-registered | MLOps |

---

## 9. Phase 0 backlog (proposed GitHub issues)

1. Merge PR #1 (review protocol README); rename default branch to `main`; branch protection.
2. Add `CLAUDE.md`, `AGENTS.md`, `CODEOWNERS`, `LICENSE`, `.gitignore`, `pyproject.toml` (ruff, mypy, pytest).
3. Add `.claude/agents/*` (10 subagents) and `.claude/skills/*` (8 skills).
4. Add `.claude/hooks/*` and `.claude/settings.json`; add `tests/test_hooks.sh`.
5. Add `.mcp.json` (alphaXiv, Hugging Face, Context7, GitHub, Mermaid) with env-var tokens.
6. Add `.agents/rules/*.md`, `.agents/skills/cross-review/SKILL.md`, `.agents/mcp_config.json`.
7. Add `Makefile`, `dvc.yaml`, `.github/workflows/ci.yml` (lint, types, unit; protected-path check; claims check).
8. Add templates: `EXPERIMENTS.yaml` schema, `AMENDMENTS.md`, `DECISIONS.md` (gates, kill criteria, dataset allowlist, compute caps), `CLAIMS.md`, `RESEARCH_LOG.md`, `docs/changes/TEMPLATE.md`, `docs/PROVENANCE.md`.
9. `docker/reference.Dockerfile` pinned to the fiveai reference environment.
10. First `/prior-art` run → `review/prior-art/2026-10-14.md`.
11. Scaffold `src/qcal/protocols.py` and `registry/` CLI with unit tests (no science yet).
12. `docs/changes/reproduce-kuzucu-eccv24-baselines.md` (Phase 1 change proposal).

---

## Appendix A: Agent layer files

### A.0 Feature verification fact sheet (docs fetched Oct 9, 2026)

Sources: code.claude.com/docs/en/{sub-agents, hooks, hooks-guide, skills, cli-reference, headless, mcp, memory, cloud-environments, plugins/components, plugins/manifest-reference}.md.

**Subagents.** Required: `name` (≤256 chars, no colon), `description`. Confirmed optional: `tools`, `disallowedTools` (comma string or YAML list), `model` (sonnet / opus / haiku / fable / full model ID / inherit), `permissionMode`, `maxTurns`, `skills`, `mcpServers` (server name string, or inline definition scoped to that subagent), `hooks` (per-subagent; project-level needs workspace trust), `memory` (user / project / local), `background`, `effort` (low..max), `isolation: worktree`. Also: `color`, `initialPrompt`, `omitClaudeMd`. Unknown keys are silently ignored. Plugin subagents ignore `hooks`, `mcpServers`, `permissionMode`.

**Hooks.** Events include SessionStart, UserPromptSubmit, PreToolUse, PermissionRequest, PostToolUse, PostToolUseFailure, SubagentStart, SubagentStop, Stop, PreCompact, SessionEnd, and others. Matcher: letters/digits/`|`/commas only means exact list; anything else is an unanchored regex; optional `if` takes permission-rule syntax such as `Edit(*.py)`. Stdin JSON carries `session_id`, `cwd`, `hook_event_name`, `tool_name`, `tool_input` (absolute `file_path` for Read/Edit/Write; `command` for Bash). Deny: exit 2 with stderr fed back, or exit 0 with `{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"..."}}`. A subagent-frontmatter Stop hook becomes SubagentStop. Default command-hook timeout 600 s. `MultiEdit` stdin schema: unverified.

**Skills.** Frontmatter: `name`, `description`, `when_to_use`, `argument-hint`, `disable-model-invocation`, `user-invocable`, `allowed-tools` (auto-approved for the invoking turn only), `disallowed-tools`, `model`, `effort`, `context: fork`, `agent`, `background`, `hooks`, `paths`. `/name args` → `$ARGUMENTS`. Forked skills do not see history.

**Headless.** Confirmed: `--output-format text|json|stream-json`, `--allowedTools`, `--tools` (restrict existence), `--max-budget-usd` (client-side estimate incl. subagents), `--bare` (skips hooks, skills, MCP, CLAUDE.md), `--permission-mode plan|default|acceptEdits|auto|dontAsk|bypassPermissions`, `--agents <json|file>`, `--max-turns`, `--mcp-config`, `--strict-mcp-config`.

**MCP.** Project scope `.mcp.json` → `mcpServers`; `type` required for remote servers (`http`, with `headers` supporting `${ENV}`); an entry with `url` and no `type` is treated as stdio and fails. `claude mcp reset-project-choices` resets approvals.

**CLAUDE.md imports.** `@AGENTS.md` works; relative to the importing file; max four hops; skipped inside code spans; keep files under ~200 lines. AGENTS.md is also read directly from v2.1.277.

**Cloud.** `CLAUDE_CODE_REMOTE="true"` in cloud sessions; SessionStart hooks in repo `.claude/settings.json` run there (user-level hooks do not); setup scripts are configured per environment.

### A.1 `CLAUDE.md`

```markdown
# CLAUDE.md — Q-Cal
@AGENTS.md

## Integrity (non-negotiable)
1. Never fabricate or estimate results; every number in paper/, README, CLAIMS.md comes from runs/index.csv via `make tables` and carries a `run:<id>` tag.
2. EXPERIMENTS.yaml is pre-registered: never edit; propose changes in AMENDMENTS.md (date, reason, what changed, affected runs).
3. Never tune anything on test splits; calibrators fit on the calib split only; model/threshold selection on val only; test is evaluated once per registered run.
4. Never write or modify src/qcal/handwritten/ or analysis/handwritten/; you may read and test them and report deviations.
5. Never author claims, abstract, interpretation, limitations, or conclusions; LaTeX fixes, log-built tables, and marked copy-edits only.
6. Clean-room: datasets listed in DECISIONS.md only; no NBCU/Edge-DIT material; stop and flag anything that resembles it.
7. Log seed, git SHA, config hash, GPU, driver, CUDA, package versions for every run (the registry CLI does this; never bypass it).

## Conventions
No hardcoded values (Hydra). Protocol-based DI via src/qcal/protocols.py + registry/. Tests for every module; parity tests against the fiveai/detection_calibration reference for calibrators and metrics. ruff + mypy clean. Change proposals in docs/changes/ before non-trivial work.

## Compute
GPU0 = RTX 5060 Ti 16 GB (train/QAT); GPU1 = RTX 5060 8 GB (eval/sweeps); pin via CUDA_VISIBLE_DEVICES; never DDP across the two. P40 is FP32-only in its own env. Jetson: TensorRT builds and latency only. Cloud only for DETR-class QAT, ≤$200 per artifact.

## Workflow
Branch claude/<slug> from main; one worktree per agent; PR requires smoke green, leakage PASS, no protected-path edits, RESEARCH_LOG.md entry, CLAIMS.md updated if numbers changed, and the other model's review in review/<branch>.md.
```

### A.2 `AGENTS.md`

```markdown
# AGENTS.md — shared by Claude Code and Antigravity
- Truth = git main + EXPERIMENTS.yaml + runs/index.csv + CLAIMS.md. Never fabricate; every number traces to a run ID.
- Never edit EXPERIMENTS.yaml (use AMENDMENTS.md); never touch */handwritten/; never tune on test.
- Ownership: Claude Code → src/, configs/, tests/, scripts/, paper build. Antigravity → notebooks/, analysis/figures/, review/. Ian → everything listed as Ian-only in CODEOWNERS.
- Branches claude/* and gemini/*, one git worktree per agent; merge only via PR reviewed by the other model with green CI.
- Clean-room: public datasets only; no employer material.
```

### A.3 Subagent files (`.claude/agents/`)

```markdown
---
name: paper-reproducer
description: Implement a published baseline exactly as specified by a paper/repo, with parity tests against reference outputs. Use for Kuzucu et al. calibrators and metrics.
tools: Read, Grep, Glob, Edit, Write, Bash
model: sonnet
isolation: worktree
maxTurns: 40
---
Cite the reference file and line for every function you port. Write parity tests that compare outputs to the reference implementation on identical inputs (tolerance stated per test). Report every deviation from the paper or repo in your final message. Never touch src/qcal/handwritten/; when Ian's implementation differs from the reference, write a failing parity test and explain the difference instead of editing his file.
```

```markdown
---
name: experiment-runner
description: Launch and register only run IDs present in EXPERIMENTS.yaml on pinned GPUs via the registry CLI. Use for overnight sweeps and nightly runs.
tools: Read, Bash, Grep, Glob
disallowedTools: Edit, Write
model: haiku
---
Validate every run ID against EXPERIMENTS.yaml before launching. Launch only via `qcal-registry run <id> --seed <s>`. On failure collect logs under runs/logs/<id>/ and stop; never change configs, never retry with modified parameters, never launch an ID that is not pre-registered.
```

```markdown
---
name: ablation-auditor
description: Compare EXPERIMENTS.yaml to runs/index.csv; flag missing cells, unregistered runs, and seed gaps. Use before tables and in weekly reviews.
tools: Read, Grep, Glob, Bash
disallowedTools: Edit, Write
model: sonnet
---
Enumerate every pre-registered cell (precision × calibrator × fit-precision × shift × calib-set-size × seed). Report: missing cells, runs in index.csv with no matching cell, seeds < 3, and amendments in AMENDMENTS.md not reflected. Never propose dropping cells.
```

```markdown
---
name: data-leakage-checker
description: Before any table is built, verify zero calib/val/test overlap and no test-metric-based selection. Outputs PASS/FAIL with evidence.
tools: Read, Grep, Glob, Bash
disallowedTools: Edit, Write
model: sonnet
---
Hash image IDs per split from the data manifests and assert pairwise disjointness. Grep configs, scripts and run logs for any selection, early stopping, or threshold choice keyed on test metrics. Confirm calibrators were fit on the calib split only. Output PASS or FAIL with file:line evidence for every finding.
```

```markdown
---
name: results-table-builder
description: Build tables only via `make tables` from runs/index.csv; every cell carries its run ID. Never type a number manually.
tools: Read, Bash, Write
model: haiku
---
Run `make tables`. Write only the generated files under paper/tables/. Every cell must carry `run:<id>`; if the generator cannot attach an ID, fail and report. Never edit a number by hand, never compute a number outside the generator.
```

```markdown
---
name: report-writer
description: LaTeX build, bibliography hygiene, and marked copy-edits of Ian's prose. Never authors claims, abstract, interpretation, or conclusions.
tools: Read, Edit, Bash
model: sonnet
---
Build paper/ with latexmk and fix build errors. Verify every \cite resolves to an entry in refs.bib with an arXiv ID or DOI; flag unverifiable references in review/bib-check.md. Copy-edits must be marked with %% EDIT: comments and must not change any number, claim, or hedge. Never edit abstract, claims, limitations, or conclusion sections.
```

```markdown
---
name: adversarial-reviewer
description: Read-only hostile review of a diff, table, or paper section. Use before merging any PR and before any gate.
tools: Read, Grep, Glob
model: opus
---
Lead with the most damaging weakness: leakage, missing baselines, overclaiming, untraceable numbers, spec drift from EXPERIMENTS.yaml, hardcoded values, Protocol violations. Rank findings; mark each blocking or non-blocking; cite file:line. Write the review to review/<branch>.md if asked; never edit code.
```

```markdown
---
name: prior-art-scout
description: Monthly novelty re-check for the Q-Cal contribution using alphaXiv and Hugging Face. Writes review/prior-art/<date>.md. Never edits paper/.
tools: Read, Write, WebSearch, WebFetch
mcpServers: alphaxiv, huggingface
model: sonnet
---
For the stated claim, list the closest papers 2019–2026 with title, venue/year, arXiv ID, overlap, and difference. Mark anything you could not resolve to an arXiv ID or DOI as [unverified]. Give a verdict: done / incremental / open, the most dangerous related paper, and three narrowing options. Write only under review/prior-art/.
```

```markdown
---
name: license-auditor
description: Check dependencies, dataset cards, and checkpoints against the DECISIONS.md allowlist. Fails on AGPL, non-commercial-only assets used for public release, or datasets not in the allowlist.
tools: Read, Grep, Glob, Bash
disallowedTools: Edit, Write
model: haiku
---
Read DECISIONS.md allowlist. Run pip-licenses; grep for ultralytics, bdd100k, and any non-allowlisted dataset. Check every data card in docs/data/ has license, source URL, and registration note. Output PASS/FAIL with evidence.
```

```markdown
---
name: hf-publisher
description: Publish model cards, dataset cards, and the Space at release via `make release`. Refuses unless leakage PASS and release-check are recorded for the current SHA.
tools: Read, Bash
mcpServers: huggingface
model: sonnet
---
Read metrics only from runs/index.csv via `make tables`. Confirm runs/release_check.json records PASS for data-leakage-checker and license-auditor on the current git SHA; otherwise stop. Push cards and the Space with the exact SHA in the card metadata.
```

### A.4 `.claude/settings.json`

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Edit|Write|MultiEdit|NotebookEdit",
        "hooks": [{ "type": "command", "command": ".claude/hooks/guard_paths.sh" }] },
      { "matcher": "Bash",
        "hooks": [{ "type": "command", "command": ".claude/hooks/guard_bash.sh" }] }
    ],
    "PostToolUse": [
      { "matcher": "Edit|Write",
        "hooks": [{ "type": "command", "command": ".claude/hooks/format.sh" }] }
    ],
    "Stop": [
      { "hooks": [{ "type": "command", "command": ".claude/hooks/check_claims.sh" }] }
    ],
    "SubagentStop": [
      { "hooks": [{ "type": "command", "command": ".claude/hooks/log_subagent.sh" }] }
    ],
    "SessionStart": [
      { "hooks": [{ "type": "command", "command": ".claude/hooks/session_start.sh" }] }
    ]
  },
  "permissions": {
    "allow": [
      "Bash(make smoke)", "Bash(make tables)", "Bash(make leakage)", "Bash(pytest *)",
      "Bash(ruff *)", "Bash(mypy *)", "Bash(qcal-registry *)", "Bash(git status)",
      "Bash(git diff *)", "Bash(git log *)"
    ],
    "deny": [
      "Bash(git push --force*)", "Bash(rm -rf runs*)", "Bash(dvc remove*)"
    ]
  }
}
```

### A.5 `.mcp.json`

```json
{
  "mcpServers": {
    "alphaxiv":   { "type": "http", "url": "https://mcp.alphaxiv.org/mcp" },
    "huggingface": { "type": "http", "url": "https://huggingface.co/mcp",
                     "headers": { "Authorization": "Bearer ${HF_TOKEN}" } },
    "context7":   { "type": "http", "url": "https://mcp.context7.com/mcp" },
    "github":     { "type": "http", "url": "https://api.githubcopilot.com/mcp/",
                     "headers": { "Authorization": "Bearer ${GITHUB_TOKEN}" } }
  }
}
```

Server URLs are the ones in use for this session's connectors; confirm each against the provider's current docs when adding (they change). Tokens come from the environment only.

---

## Appendix B: Skills (`.claude/skills/<name>/SKILL.md`)

Each file begins with frontmatter `name` and `description`; bodies are condensed here.

**change-proposal**
```markdown
---
name: change-proposal
description: Create a change proposal in docs/changes/<slug>.md and a claude/<slug> branch before non-trivial work.
---
Arguments: <slug>. Create docs/changes/<slug>.md from docs/changes/TEMPLATE.md with sections Why / What changes / Out of scope / Acceptance (checkboxes) / Tasks (who: Claude Code, Ian, paper-reproducer, Antigravity). Name the gate it serves. Create branch claude/<slug> from main. Do not start implementation in this invocation.
```

**reproduce-check**
```markdown
---
name: reproduce-check
description: Run smoke, leakage, and parity tests and report the gap to G1/K1 thresholds.
---
Run `make smoke`, then the data-leakage-checker subagent, then `pytest tests/parity -q`. Read runs/index.csv for the latest FP32 runs and compare LaECE0 and AP to the published values in docs/reference/kuzucu_eccv24.md. Report per detector: within ±1.0 LaECE0 and ±0.5 AP? Never edit anything.
```

**nightly**
```markdown
---
name: nightly
description: Launch pre-registered runs overnight on a pinned GPU via experiment-runner and record the launch manifest.
---
Arguments: <experiment-glob>. Expand against EXPERIMENTS.yaml only; refuse unknown IDs. Confirm CUDA_VISIBLE_DEVICES is set. Delegate launches to the experiment-runner subagent. Write runs/nightly/<date>.json with launched IDs, SHA, GPU. Stop on first failure.
```

**tables**
```markdown
---
name: tables
description: Build result tables with the full integrity chain: leakage check, make tables, ablation audit.
---
1. data-leakage-checker must PASS; else stop. 2. results-table-builder runs `make tables`. 3. ablation-auditor reports coverage. 4. Show `git diff paper/tables/`. Never type a number.
```

**claims-audit**
```markdown
---
name: claims-audit
description: Audit every result-bearing sentence in a draft against runs/index.csv and EXPERIMENTS.yaml.
---
Arguments: <file>. For each result-bearing sentence output a table: Sentence | Claimed result | Evidence/run ID | Assessment (supported / overstated / unsupported / untraceable) | Suggested fix. Check CI/seed-spread justification and consistency with pre-registration. Do not supply missing results or author claims. If the CSV or pre-registration is missing, report blocked.
```

**weekly-review**
```markdown
---
name: weekly-review
description: Week N review against gates and kill criteria in DECISIONS.md; hours from RESEARCH_LOG.md; WIP limit; open amendments.
---
Arguments: <N>. Read RESEARCH_LOG.md (planned vs actual hours), DECISIONS.md (gates, kill criteria), AMENDMENTS.md, runs/agent_log.jsonl, and the ablation-auditor report. Output: hours; scope creep (any work outside the current change proposal); each upcoming gate green/amber/red with evidence; WIP-limit violations; two-week <6 h rule; recommendation continue / amend / kill / switch; the one task Ian must do by hand this week. Never invent hours or gate states; if inputs are missing, say what cannot be assessed.
```

**prior-art**
```markdown
---
name: prior-art
description: Monthly novelty re-check via alphaXiv and Hugging Face; writes review/prior-art/<date>.md.
argument-hint: "<claim>"
context: fork
agent: prior-art-scout
---
Claim: $ARGUMENTS. Read the newest file in review/prior-art/ first. Search for 2019–2026 papers measuring detector calibration (ECE, D-ECE, LaECE, LaACE, LRP) after INT8 PTQ or QAT, on edge accelerators, or under domain shift. Write review/prior-art/<today>.md with the schema in Appendix A.3 and a "new since last check" section.
```

**release-check**
```markdown
---
name: release-check
description: Walk the Definition of Done before `make release`.
---
Check each item in docs/SDLC_IMPLEMENTATION_PLAN.md §7 with evidence (file paths, run IDs, CI URLs). Run license-auditor and data-leakage-checker on the current SHA and write runs/release_check.json. Any unchecked item blocks release; say which.
```

---

## Appendix C: Hook scripts (`.claude/hooks/`)

```bash
#!/usr/bin/env bash
# guard_paths.sh — deny agent edits to Ian-only and clean-room-forbidden paths (exit 2 = deny)
set -euo pipefail
path="$(jq -r '.tool_input.file_path // .tool_input.path // .tool_input.notebook_path // empty')"
[ -z "$path" ] && exit 0
lower="$(printf '%s' "$path" | tr '[:upper:]' '[:lower:]')"
case "$lower" in
  *experiments.yaml|*/handwritten/*|*claims.md|*research_log.md|\
  *paper/sections/abstract*|*paper/sections/claims*|*paper/sections/limitations*|*paper/sections/conclusion*|\
  *runs/index.csv)
    echo "BLOCKED: $path is Ian-only or registry-only. Propose in AMENDMENTS.md or review/." >&2; exit 2 ;;
  *nbcu*|*edge-dit*|*edge_dit*|*edgedit*)
    echo "BLOCKED: $path violates the clean-room rule (DECISIONS.md)." >&2; exit 2 ;;
esac
exit 0
```

```bash
#!/usr/bin/env bash
# guard_bash.sh — deny shell commands that bypass the path guard or the registry
set -euo pipefail
cmd="$(jq -r '.tool_input.command // empty')"
[ -z "$cmd" ] && exit 0
deny() { echo "BLOCKED: $1" >&2; exit 2; }
echo "$cmd" | grep -Eq 'git push .*(--force|-f\b|\+)' && deny "force push"
echo "$cmd" | grep -Eq 'git push (origin )?main\b' && deny "direct push to main; open a PR"
echo "$cmd" | grep -Eq '(sed -i|tee|>>?|mv|cp|rm).*(EXPERIMENTS\.yaml|CLAIMS\.md|handwritten/|runs/index\.csv)' && deny "protected file written outside the registry"
echo "$cmd" | grep -Eq 'rm -rf? +runs|dvc remove' && deny "destructive run-store command"
echo "$cmd" | grep -Eiq 'nbcu|edge[-_]?dit' && deny "clean-room violation"
exit 0
```

```bash
#!/usr/bin/env bash
# check_claims.sh — warn on untagged numbers in paper sections (Stop hook)
grep -RInE '[0-9]+\.[0-9]+' paper/sections/*.tex 2>/dev/null | grep -v 'run:' | grep -q . && \
  echo "WARNING: numbers without run: tags in paper/sections; use results-table-builder." >&2
exit 0
```

```bash
#!/usr/bin/env bash
# log_subagent.sh — append a line per finished subagent for the weekly review
set -euo pipefail
mkdir -p runs
jq -c --arg sha "$(git rev-parse --short HEAD 2>/dev/null || echo none)" --arg ts "$(date -Is)" \
  '{ts:$ts, sha:$sha, agent:(.agent_name // .subagent_type // "unknown"), summary:(.last_assistant_message // "" | .[0:200])}' \
  >> runs/agent_log.jsonl
exit 0
```

```bash
#!/usr/bin/env bash
# session_start.sh — environment facts and cloud bootstrap
set -euo pipefail
if [ -n "${CLAUDE_CODE_REMOTE:-}" ] || [ ! -d .venv ]; then
  python3 -m venv .venv && . .venv/bin/activate && pip install -q -e ".[dev]" || true
  command -v dvc >/dev/null && dvc pull data/smoke 2>/dev/null || true
fi
python3 - <<'PY' 2>/dev/null || true
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "archs", torch.cuda.get_arch_list())
PY
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exit 0
```

```bash
#!/usr/bin/env bash
# format.sh — ruff on edited Python files (PostToolUse)
path="$(jq -r '.tool_input.file_path // empty')"
case "$path" in *.py) ruff format "$path" >/dev/null 2>&1; ruff check --fix "$path" >/dev/null 2>&1;; esac
exit 0
```

---

## Appendix D: CI, Makefile, registry, templates

### D.1 `.github/workflows/ci.yml` (jobs)
- `lint`: ruff check + ruff format --check.
- `types`: mypy src/.
- `unit`: pytest tests/unit (CPU).
- `smoke`: `make smoke` on the 20-image fixture (CPU, < 5 min), required for merge from Phase 1.
- `protected-paths`: fails if the PR diff touches Ian-only paths and the PR lacks the `ian-authored` label.
- `claims`: fails if any decimal number in `paper/sections/*.tex` lacks a `run:` tag.
- `licenses`: pip-licenses + grep for AGPL packages.
- `gpu-parity` (`gpu-parity.yml`, self-hosted runner `gpu` on ian-pc, non-blocking): `pytest tests/parity`.
- `reproduce` (`paper.yml`, on tag `v*`): clean clone → `make reproduce` → assert `git diff --exit-code paper/tables/`.

### D.2 Makefile targets
`smoke` (20-image FP32 eval, asserts metrics are finite), `leakage` (runs the split-hash test), `tables` (runs/index.csv → paper/tables/*.tex with run tags), `reproduce` (all registered runs' tables from logged artifacts), `paper` (latexmk), `release` (release-check → hf-publisher), `nightly`.

### D.3 Registry CLI (`qcal-registry`)
`run <experiment-id> --seed <s>`: validates ID against `EXPERIMENTS.yaml`, resolves the Hydra config, records seed, git SHA, config hash, GPU name, driver, CUDA, package versions, launches, and on completion appends one row to `runs/index.csv` (columns: run_id, experiment_id, seed, sha, config_hash, gpu, driver, cuda, precision, detector, calibrator, fit_precision, shift, calib_set_size, AP, LRP, DECE, LaECE0, LaACE0, latency_ms, throughput, started_at, finished_at, status). `index.csv` is append-only; corrections are new rows with `status=superseded_by:<run_id>`.

### D.4 `EXPERIMENTS.yaml` schema (Ian fills)
```yaml
version: 1
question: "Do INT8 PTQ and QAT on edge accelerators change detectors' localization-aware calibration, in-domain and under shift, and can it be preserved cheaply?"
hypotheses:
  H1: {statement: "...", metric: LaECE0, min_effect: 1.0, test: paired_bootstrap_95ci_excludes_0, detectors_min: 3}
  H2: {statement: "...", recovers_fraction_min: 0.5}
  H3: {statement: "...", ap_cost_max: 1.0}
falsifier: "H1 gap < 0.5 and CI includes 0 → negative-result note"
axes:
  precision: [fp32, fp16, ptq_minmax, ptq_entropy, qat, qat_cal]
  calibrator: [none, platt, isotonic, classwise_isotonic, two_threshold]
  fit_precision: [fp32, int8]
  shift: [id, cocoC_s1, cocoC_s3, cocoC_s5, foggy]
  calib_set_size: [500, 2000, 5000]
  seeds: [0, 1, 2]
detectors: [atss_r50, detr_variant, yolox_s]
metrics: [AP, LRP, DECE, LaECE0, LaACE0, jetson_latency_ms, jetson_throughput]
splits: {calib: "...", val: "...", test: "..."}
kill_criteria: {K1: "2026-11-08 ...", K2: "2026-12-06 ...", K3: "2027-01-10 ...", K4: "counsel"}
```

### D.5 `docs/changes/TEMPLATE.md`
```markdown
# change: <slug>
## Gate served
## Why
## What changes
## Out of scope
## Acceptance
- [ ] make smoke < 5 min
- [ ] data-leakage-checker PASS
- [ ] parity tests green
- [ ] no protected-path edits
- [ ] RESEARCH_LOG.md entry; CLAIMS.md updated if numbers changed
- [ ] other model's review in review/<branch>.md
## Tasks
Scaffold (Claude Code) → hand-write (Ian) → parity tests (paper-reproducer) → runs (experiment-runner) → tables (/tables) → review (adversarial-reviewer + Antigravity)
```

### D.6 `DECISIONS.md` skeleton
Sections: Lane (Q-Cal / D-Cal, G0 date, counsel view, written NBCU acknowledgment status); Gates G1–G4 with dates and evidence required; Kill criteria K1–K4; Dataset allowlist with licenses (COCO 2017, COCO-C generated, Cityscapes + Foggy; BDD100K excluded until cleared); Detector allowlist (MMDetection ATSS, DETR variant, YOLOX-s; Ultralytics excluded); Compute caps (cloud ≤$200/artifact; nightly ≤$3/run); Venue and deadline; WIP limit; Repo visibility decision.

---

## Appendix E: Prior-art and tooling verification notes (Oct 9, 2026)

Produced via the alphaXiv, Hugging Face and Context7 MCP servers plus web search; this is the first `/prior-art` pass and should be re-run in week 1 and monthly thereafter. Tags per the Review's convention.

### E.1 The three IDs named in the Review
- **arXiv:2508.19600** [Certain]. Karimov, Imani, Kazakov, *Quantization Robustness to Input Degradations for Object Detection* (v3, May 2026). YOLO12 under FP32/FP16/UINT8/static-INT8 TensorRT on COCO val plus seven synthetic degradations (Albumentations, not COCO-C). Static INT8 costs ~3–7 mAP; degradation-aware calibration data gives no consistent gain. Reports mAP only, no ECE/LaECE/D-ECE.
- **arXiv:2604.26857** [Certain]. Karjol and Hanna, *Edge AI for Automotive VRU Safety: Deployable Detection via KD* (Apr 2026). YOLOv8 KD on BDD100K, TensorRT INT8 PTQ on an RTX 5070 (not Jetson). "Calibration" there means precision/false-alarm rate, not ECE. Note: it mis-cites 2508.19600; do not propagate that citation.
- **arXiv:2412.01782** [Certain]. Park, Sobolewski, Azizan, *Quantifying the Reliability of Predictions in Detection Transformers*. Shows D-ECE/LaECE0 can be gamed by thresholding; proposes OCE; evaluates DETR variants on COCO → Cityscapes → Foggy. FP32 only.

### E.2 Closest 2024–2026 papers
1. arXiv:2609.16085, *Is INT8 Portable? Cross-Platform Measurement Study* (2026) [Certain]. Jetson AGX Orin GPU/DLA, Hexagon, DEEPX; DETR-R50 INT8 PTQ collapses 0.42→0.24 mAP on Jetson, driven by the activation calibrator; INT8 outputs are non-deterministic across kernels (~4% prediction flips). No calibration metrics. Methodological prior: fix scales, fix input shapes, re-validate per target.
2. arXiv:2607.29040, *Rethinking Detection Calibration (ReDC)* (2026) [Certain]. D-ECE/LaECE/LaECE0/LaACE0 on COCO→COCO-C and Cityscapes→Foggy with Deformable-DETR, DINO, VFNet, Cascade R-CNN. Same protocol as Q-Cal; no quantization. Used only three corruption types.
3. arXiv:2610.01409, *Localisation-Aware Uncertainty for Pretrained Object Detection* (Oct 2026) [Likely; abstract only]. Post-hoc, shift; no quantization.
4. arXiv:2607.18540, *Recti-Q* (2026) [Certain]. W4 PTQ keeps ID accuracy, loses OOD accuracy; classification.
5. arXiv:2609.31155, *Teacher-Anchored Selection of PTQ Models under Domain Shift* (2026) [Certain]. ECE of PTQ families under shift; "overconfident-collapse regime"; classification.
6. arXiv:2606.31456, zero-shot quantization for detectors (2026) [Likely]. Accuracy-focused.
7. arXiv:2603.05964, *QATMA* (2026) [Likely]. Detector QAT; no calibration metrics in abstract.
8. arXiv:2609.03604, compression × test-time adaptation (2026) [Certain]. Pruning drives predictions toward uniform under ImageNet-C; classification.

**Verdict [Likely]:** the exact Q-Cal question (LaECE0/LaACE0/D-ECE of ATSS/DETR/YOLOX after INT8 PTQ and QAT, on Jetson Orin / Hailo-8, in-domain and under COCO-C / Foggy) does not appear answered. Each neighbour covers two of the four axes. Residual risk: a late-2026 workshop paper not yet indexed. Position against 2609.16085 and 2412.01782.

### E.3 Tooling facts
- MMDetection 3.x: Apache-2.0; v3.3.0 (May 2024) is the last release; ATSS, DETR, Deformable DETR, DINO, YOLOX with COCO checkpoints [Certain]. mmdeploy supports ATSS, DETR, YOLOX, Deformable DETR, DINO on ONNXRuntime and TensorRT; DETR-like models single-batch; TensorRT ≥ 8.4 recommended; Deformable-DETR/DINO need the TensorRT multiscale-deformable-attention plugin [Certain].
- PyTorch QAT: `torch.ao.quantization` deprecated with deletion planned around 2.10 (wording persists in 2.11/2.12 docs, so removal may have slipped) [Likely]. Recommended: torchao PT2E (`torch.export` → `prepare_qat_pt2e` → train → `convert_pt2e`); for TensorRT Q/DQ graphs, NVIDIA ModelOpt [Certain]. `torch.export` of MMDetection heads with dynamic NMS/Hungarian matching is fragile.
- TensorRT on Jetson: JetPack 6.2 = TensorRT 10.3, CUDA 12.6 [Certain]. TRT 10.x supports implicit calibration (`IInt8EntropyCalibrator2`, `IInt8MinMaxCalibrator`, legacy percentile; `trtexec --int8 --calib`); TRT 11.x removes `IInt8Calibrator` and `--calib`, explicit Q/DQ only [Certain]. Use DLA only for CNNs; DETR fragments into hundreds of GPU-fallback layers.
- Hailo-8: Dataflow Compiler gated behind Developer Zone registration and EULA [Likely]. Model zoo (MIT) has `yolox_s_leaky`, `yolox_tiny`, `detr_resnet_v1_18_bn`; no ATSS, Deformable-DETR, DINO. Native quantization; BYO Q/DQ not honoured.
- Hugging Face: no per-repo cap; large binaries in model repos, not Spaces [Certain].

### E.4 Datasets and licenses
- COCO 2017: annotations CC BY 4.0; images Flickr-sourced under Flickr terms [Likely; official page unreachable from the sandbox]. HF mirror `detection-datasets/coco` has no license tag: acceptable for CI smoke fixtures, use the official download for paper numbers.
- COCO-C: generate with `imagecorruptions` (bethgelab, Apache-2.0; 15 + 4 corruptions, severities 1–5) per arXiv:1907.07484; MMDetection ships `tools/analysis_tools/test_robustness.py` [Certain]. Package unmaintained; pin numpy/scikit-image.
- Cityscapes / Foggy Cityscapes: non-commercial research only, registration with institutional email, no redistribution; Foggy is distributed only via the Cityscapes download page [Likely; official page unreachable]. Avoid all HF mirrors (unlicensed re-uploads).
- fiveai/detection_calibration: **CC BY-NC-SA 4.0** plus bundled MMDetection code [Certain]. Use as unmodified test oracle only; clean-room reimplementation for anything released.

### E.5 Unverified
Exact wording of the COCO and Cityscapes terms pages and the Hailo EULA (hosts unreachable from the sandbox); whether `torch.ao.quantization` has actually been deleted in a shipped release; the JetPack 6.2.x TensorRT point version.
