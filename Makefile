# =============================================================================
# Makefile - Task Automation
# =============================================================================
# Federal University of Bahia - MSc in Industrial Engineering
# Author: Ghabriel Anton Gomes de Sá
#
# Usage:
#   make help          - Show available commands
#   make install       - Install dependencies
#   make format        - Format code
#   make lint          - Check code
#   make test          - Run tests
#   make run           - Run default experiment
# =============================================================================

.PHONY: help install install-dev install-gpu install-hooks format lint check \
        test test-cov run run-all run-cv figures dashboard list-datasets \
        clean clean-results clean-figures

# Variables
PYTHON := python
PIP := pip
DATASET ?= computer_hardware
METHOD ?= wilcar_unconstrained

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m  # No Color

# =============================================================================
# Help
# =============================================================================
help:
	@echo ""
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  MSc Thesis - Neural Networks with Gain Sign Constraints$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(GREEN)Installation:$(NC)"
	@echo "  make install        Install basic dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make install-gpu    Install GPU dependencies (PyTorch)"
	@echo "  make install-hooks  Install pre-commit hooks"
	@echo ""
	@echo "$(GREEN)Code Quality:$(NC)"
	@echo "  make format         Format code (ruff + isort)"
	@echo "  make lint           Check code (ruff)"
	@echo "  make check          Check everything (format + lint)"
	@echo ""
	@echo "$(GREEN)Tests:$(NC)"
	@echo "  make test           Run tests"
	@echo "  make test-cov       Run tests with coverage"
	@echo ""
	@echo "$(GREEN)Experiments:$(NC)"
	@echo "  make run            Run single method"
	@echo "                      DATASET=$(DATASET) METHOD=$(METHOD)"
	@echo "  make run-all        Run all methods"
	@echo "                      DATASET=$(DATASET)"
	@echo "  make run-cv         Run cross-validation"
	@echo "                      DATASET=$(DATASET)"
	@echo "  make list-datasets  List available datasets"
	@echo ""
	@echo "$(GREEN)Visualization:$(NC)"
	@echo "  make figures        Generate publication figures"
	@echo "  make dashboard      Regenerate comparative dashboard"
	@echo ""
	@echo "$(GREEN)Cleanup:$(NC)"
	@echo "  make clean          Remove temporary files"
	@echo "  make clean-results  Remove experiment results"
	@echo "  make clean-figures  Remove generated figures"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make run DATASET=qsar_fish_toxicity METHOD=elm"
	@echo "  make run-all DATASET=computer_hardware"
	@echo "  make run-cv DATASET=qsar_aquatic_toxicity"
	@echo ""

# =============================================================================
# Installation
# =============================================================================
install:
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-dev:
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

install-gpu:
	@echo "$(BLUE)Installing GPU dependencies (PyTorch)...$(NC)"
	$(PIP) install torch
	@echo "$(GREEN)✓ GPU dependencies installed$(NC)"

install-hooks:
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	pre-commit install
	@echo "$(GREEN)✓ Pre-commit hooks installed$(NC)"

# =============================================================================
# Code Quality
# =============================================================================
format:
	@echo "$(BLUE)Formatting code...$(NC)"
	ruff format .
	isort .
	@echo "$(GREEN)✓ Code formatted$(NC)"

lint:
	@echo "$(BLUE)Checking code...$(NC)"
	ruff check .
	@echo "$(GREEN)✓ Check completed$(NC)"

check: format lint
	@echo "$(GREEN)✓ All checks passed$(NC)"

# =============================================================================
# Tests
# =============================================================================
test:
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v
	@echo "$(GREEN)✓ Tests completed$(NC)"

test-cov:
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report in htmlcov/$(NC)"

# =============================================================================
# Experiments
# =============================================================================
run:
	@echo "$(BLUE)Running experiment...$(NC)"
	@echo "  Dataset: $(DATASET)"
	@echo "  Method:  $(METHOD)"
	$(PYTHON) run_experiment.py --dataset $(DATASET) --method $(METHOD)

run-all:
	@echo "$(BLUE)Running all methods...$(NC)"
	@echo "  Dataset: $(DATASET)"
	$(PYTHON) run_all_methods.py --dataset $(DATASET)

run-cv:
	@echo "$(BLUE)Running cross-validation...$(NC)"
	@echo "  Dataset: $(DATASET)"
	$(PYTHON) run_cross_validation.py --dataset $(DATASET)

list-datasets:
	@echo "$(BLUE)Available datasets:$(NC)"
	$(PYTHON) run_experiment.py --list

# =============================================================================
# Visualization
# =============================================================================
figures:
	@echo "$(BLUE)Generating publication figures...$(NC)"
	@echo "  Dataset: $(DATASET)"
	$(PYTHON) generate_figures.py --dataset $(DATASET)
	@echo "$(GREEN)✓ Figures generated$(NC)"

dashboard:
	@echo "$(BLUE)Regenerating comparative dashboard...$(NC)"
	@echo "  Dataset: $(DATASET)"
	$(PYTHON) regenerate_dashboard.py --dataset $(DATASET)
	@echo "$(GREEN)✓ Dashboard regenerated$(NC)"

# =============================================================================
# Cleanup
# =============================================================================
clean:
	@echo "$(BLUE)Removing temporary files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true
	@echo "$(GREEN)✓ Temporary files removed$(NC)"

clean-results:
	@echo "$(YELLOW)WARNING: This will remove all results in results/$(NC)"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] && rm -rf results/* || echo "Cancelled"

clean-figures:
	@echo "$(YELLOW)WARNING: This will remove all figures in figures/$(NC)"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] && rm -rf figures/* || echo "Cancelled"