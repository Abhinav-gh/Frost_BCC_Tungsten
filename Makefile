# Frost developer tasks. `make help` lists everything.
VENV ?= mace
PY   := $(CURDIR)/.venvs/frost-$(VENV)/bin/python
CU   := $(CURDIR)/.venvs/frost-$(VENV)/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME := $(CU)
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD := 1
export TORCH_CUDA_ARCH_LIST := 8.6

.DEFAULT_GOAL := help
.PHONY: help verify status test test-gpu lint fmt typecheck probes systems data clean cuda-fix install

help:  ## show this help
	@grep -hE '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## editable install into both environments
	@for v in mace nequip; do \
	  VIRTUAL_ENV=$(CURDIR)/.venvs/frost-$$v uv pip install -e ".[dev]" --no-deps -q; \
	done; echo "installed"

cuda-fix:  ## re-create the CUDA symlinks both venvs need in order to link
	@for v in mace nequip; do \
	  C=$(CURDIR)/.venvs/frost-$$v/lib/python3.12/site-packages/nvidia/cu13; \
	  ln -sfn lib $$C/lib64; \
	  (cd $$C/lib && for f in *.so.*; do ln -sf $$f $${f%%.so.*}.so; done); \
	done; echo "cuda symlinks ok"

verify:  ## full setup verification (36 checks)
	@./scripts/verify.sh

status:  ## what is installed, downloaded and running
	@./scripts/status.sh

test:  ## unit and integration tests
	@$(PY) -m pytest -q -m "not gpu"

test-gpu:  ## tests that need a CUDA device
	@$(PY) -m pytest -q -m gpu

probes:  ## re-run the week-0 theorem probes
	@$(PY) experiments/probes/probe_zero_closure.py
	@$(PY) experiments/probes/probe_force_error.py

systems:  ## build and print the simulation cells
	@$(PY) -m frost.systems

data:  ## fetch tier A datasets
	@$(PY) scripts/fetch_data.py --tier A

lint:  ## ruff check + format check
	@$(PY) -m ruff check . && $(PY) -m ruff format --check .

fmt:  ## apply ruff fixes and formatting
	@$(PY) -m ruff check --fix . && $(PY) -m ruff format .

typecheck:  ## mypy on the package
	@$(PY) -m mypy src/frost

clean:  ## remove caches and build artefacts
	@rm -rf build dist .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	@find . -name __pycache__ -type d -not -path "./.venvs/*" -exec rm -rf {} + 2>/dev/null || true
	@echo "cleaned"
