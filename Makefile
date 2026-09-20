# ─────────────────────────────────────────────────────────────────────────────
# ML-Based NIDS — Mode 1  |  Makefile
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: help install lint test clean train evaluate replay dashboard

PYTHON  ?= python
PIP     ?= pip

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ─── Environment ─────────────────────────────────────────────────────────────
install:  ## Install package + dependencies in editable mode
	$(PIP) install -e ".[dev]"
	$(PIP) install -r requirements.txt

# ─── Quality ─────────────────────────────────────────────────────────────────
lint:  ## Run ruff linter
	ruff check src/ tests/

format:  ## Auto-format with ruff
	ruff format src/ tests/

test:  ## Run pytest with coverage
	pytest --cov=nids --cov-report=term-missing tests/

# ─── Pipeline ────────────────────────────────────────────────────────────────
data-prep:  ## TODO: download & clean UNSW-NB15 dataset
	$(PYTHON) -m nids.dataset

preprocess:  ## TODO: feature engineering + train/test split
	$(PYTHON) -m nids.preprocessing

train:  ## TODO: train binary + multiclass models
	$(PYTHON) -m nids.train

evaluate:  ## TODO: evaluate models and save reports
	$(PYTHON) -m nids.evaluate

explain:  ## TODO: generate SHAP explanations
	$(PYTHON) -m nids.explain

replay:  ## TODO: replay a scenario CSV through the pipeline
	$(PYTHON) -m nids.replay --scenario data/scenarios/example.csv

# ─── Dashboard ───────────────────────────────────────────────────────────────
dashboard:  ## Launch the Streamlit dashboard
	streamlit run dashboard/app.py

# ─── Housekeeping ────────────────────────────────────────────────────────────
clean:  ## Remove build artefacts and __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/ htmlcov/ .coverage
