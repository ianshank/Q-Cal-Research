# Q-Cal developer entry points. Every value is overridable: make test PYTHON=python3.12
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
QCAL   ?= $(PYTHON) -m qcal
PYTEST_ARGS ?=

.DEFAULT_GOAL := help
.PHONY: help venv install lint format typecheck test test-fast test-hooks check integrity \
        agent-layer index tables audit claims leakage licenses reproduce smoke init lock unlock clean

help: ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

venv: ## create .venv
	python3 -m venv .venv

install: ## install the package with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## ruff lint and format check
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format: ## apply ruff formatting and fixes
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

typecheck: ## mypy --strict
	$(PYTHON) -m mypy

test: ## full suite with the coverage gate
	$(PYTHON) -m pytest --cov $(PYTEST_ARGS)

test-fast: ## unit tests only, no coverage
	$(PYTHON) -m pytest tests/unit -q $(PYTEST_ARGS)

test-hooks: ## hook guards through the real wrapper
	$(PYTHON) -m pytest tests/integration/test_hook_wrapper.py -q $(PYTEST_ARGS)

check: lint typecheck test integrity ## everything CI runs on the head

integrity: agent-layer ## head-side integrity checks
	$(QCAL) registry index --check
	$(QCAL) registry tables --check
	$(QCAL) claims
	$(QCAL) leakage --if-present
	$(QCAL) licenses

agent-layer: ## validate agents, skills, hooks and .mcp.json
	$(QCAL) agent-layer

index: ## regenerate runs/index.csv from run records
	$(QCAL) registry index

tables: ## regenerate paper tables from the index
	$(QCAL) registry tables

audit: ## pre-registered coverage
	$(QCAL) registry audit

claims: ## every number traces to a run
	$(QCAL) claims

leakage: ## split manifests are disjoint
	$(QCAL) leakage

licenses: ## license and forbidden-import audit
	$(QCAL) licenses

reproduce: ## index and tables are reproducible from committed records
	$(QCAL) registry index --check
	$(QCAL) registry tables --check

smoke: ## end-to-end FP32 smoke run (arrives with Phase 1)
	@echo "smoke: no experiment program exists yet; Phase 1 (after G0) adds it." >&2; exit 1

init: ## create Ian's document templates if missing
	$(QCAL) init

lock: ## make Ian-only files read-only on this machine
	$(QCAL) policy list --category ian_only | xargs -r chmod a-w

unlock: ## make Ian-only files writable again (Ian only)
	$(QCAL) policy list --category ian_only | xargs -r chmod u+w

clean: ## remove caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov build dist
	find . -name __pycache__ -type d -prune -not -path './.venv/*' -exec rm -rf {} +
