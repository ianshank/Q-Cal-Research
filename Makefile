# Q-Cal developer entry points. Every value is overridable: make test PYTHON=python3.12
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
QCAL   ?= $(PYTHON) -m qcal
PYTEST_ARGS ?=
REPORTS ?= reports
GITLEAKS ?= gitleaks
DOCKER ?= docker
IMAGE ?= qcal:dev
DOCKER_BUILD_ARGS ?=

.DEFAULT_GOAL := help
.PHONY: help venv install lint format typecheck test test-fast test-hooks test-unit \
        test-integration test-regression test-security test-e2e test-reports check validate \
        pre-pr integrity agent-layer index tables audit claims leakage licenses reproduce smoke \
        parity lab-status \
        init lock unlock gitleaks docker-build docker-test nightly clean

help: ## list targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-17s %s\n", $$1, $$2}'

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
	$(PYTHON) -m pytest -m unit -q $(PYTEST_ARGS)

test-unit: ## unit suite (tests/unit)
	$(PYTHON) -m pytest -m unit -q $(PYTEST_ARGS)

test-integration: ## integration suite: real git, subprocesses, the hook wrapper
	$(PYTHON) -m pytest -m integration -q $(PYTEST_ARGS)

test-regression: ## one test per fixed review or red-team finding
	$(PYTHON) -m pytest -m regression -q $(PYTEST_ARGS)

test-security: ## attacks on the guards and the signed-commit control
	$(PYTHON) -m pytest -m security -q $(PYTEST_ARGS)

test-e2e: ## the documented workflow through the real CLI
	$(PYTHON) -m pytest -m e2e -q $(PYTEST_ARGS)

test-hooks: ## hook guards through the real wrapper
	$(PYTHON) -m pytest tests/integration/test_hook_wrapper.py -q $(PYTEST_ARGS)

test-reports: ## full suite with JUnit and coverage XML in $(REPORTS)/
	$(PYTHON) -m pytest --cov --cov-report=xml:$(REPORTS)/coverage.xml \
		--junitxml=$(REPORTS)/junit.xml $(PYTEST_ARGS)

check: lint typecheck test integrity ## everything CI runs on the head

validate: lint typecheck integrity ## fast checks without the test suite

pre-pr: check ## check, plus secrets and the container when the tools are present
	@if command -v $(GITLEAKS) >/dev/null 2>&1; then $(MAKE) --no-print-directory gitleaks; \
	else echo "pre-pr: $(GITLEAKS) not found; CI runs the secret scan" >&2; fi
	@if command -v $(DOCKER) >/dev/null 2>&1 && $(DOCKER) info >/dev/null 2>&1; then \
		$(MAKE) --no-print-directory docker-test; \
	else echo "pre-pr: $(DOCKER) unavailable; CI builds and tests the container" >&2; fi

integrity: agent-layer ## head-side integrity checks
	$(QCAL) config --check
	$(QCAL) registry index --check
	$(QCAL) registry tables --check
	$(QCAL) claims
	$(QCAL) leakage --if-present
	$(QCAL) licenses

agent-layer: ## validate agents, skills, hooks, .mcp.json and documented commands
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

smoke: ## the Phase 1 loop on a synthetic fixture through the real registry (< 5 min)
	$(PYTHON) -m qcal_lab smoke

parity: ## parity with the fiveai oracle outputs (tests/parity; skips until they exist)
	$(PYTHON) -m pytest tests/parity -q $(PYTEST_ARGS)

lab-status: ## what still blocks a registered Phase 1 run
	$(PYTHON) -m qcal_lab status

init: ## create Ian's document templates if missing
	$(QCAL) init

lock: ## make Ian-only files read-only on this machine
	$(QCAL) policy list --category ian_only | xargs -r chmod a-w

unlock: ## make Ian-only files writable again (Ian only)
	$(QCAL) policy list --category ian_only | xargs -r chmod u+w

gitleaks: ## scan git history and the working tree for secrets
	$(GITLEAKS) git --config .gitleaks.toml --redact --no-banner .
	$(GITLEAKS) git --pre-commit --config .gitleaks.toml --redact --no-banner .

docker-build: ## build the CPU image ($(IMAGE))
	$(DOCKER) build $(DOCKER_BUILD_ARGS) -t $(IMAGE) .

docker-test: docker-build ## run the full suite inside the image
	$(DOCKER) run --rm $(IMAGE) make test PYTHON=python

nightly: ## scheduled self-checks; see scripts/nightly.sh
	scripts/nightly.sh

clean: ## remove caches and reports
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis .coverage coverage.xml htmlcov \
		build dist $(REPORTS)
	find . -name __pycache__ -type d -prune -not -path './.venv/*' -exec rm -rf {} +
