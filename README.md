# UNIVERSIDADE FEDERAL DA BAHIA (FEDERAL UNIVERSITY OF BAHIA)
## ESCOLA POLITÉCNICA (POLYTECHNIC SCHOOL)
## PROGRAMA DE PÓS-GRADUAÇÃO EM ENGENHARIA INDUSTRIAL (GRADUATE PROGRAM IN INDUSTRIAL ENGINEERING)

**Student:** Ghabriel Anton Gomes de Sá  
**Advisors:** Marcelo Embiruçu & Cristiano Fontes  

<h2 align="center">Constructive Neural Networks for Regression with Gain Sign Constraints</h2>

---

## Description

This project implements and compares **6 neural network methods** with a single hidden layer for MISO (Multiple Input, Single Output) regression problems, with the possibility of **gain sign constraints**.

The proposed method (WILCAR) combines:
- Weight initialization via **linearization** (Taylor series + Least Squares)
- **Constructive** approach with weight reuse
- Possibility of **gain sign constraints** via SLSQP optimization

---

## Implemented Methods

| ID | Method | Initialization | Training | Constraints |
|----|--------|----------------|----------|-------------|
| 1 | **WILCAR Unconstrained** | Linearization | Backpropagation | ❌ |
| 2 | **WILCAR Constrained** | Linearization | SLSQP | ✅ |
| 3 | **RIXM** | Xavier (Random) | Backpropagation | ❌ |
| 4 | **RIXM Constrained** | Xavier (Random) | SLSQP | ✅ |
| 5 | **ELM** | Xavier (Random, fixed) | Pseudo-inverse | ❌ |
| 6 | **ELM Constrained** | Xavier (Random, fixed) | SLSQP | ✅ |

### Method Descriptions

- **WILCAR (Weight Initialization via Linearization with Constructive Algorithm and Reuse):** Proposed method that uses linearization for intelligent weight initialization.
- **RIXM (Random Initialization Xavier Method):** Baseline with random Xavier initialization.
- **ELM (Extreme Learning Machine):** Hidden layer weights are fixed (random), only output weights are trained via pseudo-inverse.

### 100% SCR Guarantee (Constrained Methods)

Constrained methods (2, 4, and 6) implement a **reinitialization strategy** that guarantees 100% Signal Conformity Rate (SCR):
- Up to 100 reinitialization attempts per configuration
- Verification on the **test** set (not training)
- Configurations that don't achieve 100% SCR are skipped

---

## Proposed Method (WILCAR)

### Weight Initialization

1. **Linearization** via Taylor series of the nonlinear neural model (network with 1 hidden neuron), using the mean of input and output variables as the equilibrium point.

2. **Multiple Linear Regression** via Least Squares Method applied to the dataset.

3. **Mathematical comparison** of results to obtain initial weight estimates (input → hidden layer weights).

### Constructive Expansion

The neural network expands progressively:
- Adds hidden neurons one by one up to the defined limit
- **Reuses weights** from previous training as starting point
- New neurons are initialized with **Xavier**
- **Biases** initialized as zero
- **Early stopping** based on constructive patience

---

## Project Structure

```
thesis/
├── data/
│   ├── raw/                    # Original datasets (.csv)
│   └── processed/              # Preprocessed data
├── src/
│   ├── __init__.py
│   ├── base.py                 # Base classes and configurations
│   ├── wilcar/
│   │   ├── __init__.py
│   │   ├── unconstrained.py    # Method 1
│   │   └── constrained.py      # Method 2
│   ├── rixm/
│   │   ├── __init__.py
│   │   ├── rixm.py             # Method 3
│   │   └── rixm_constrained.py # Method 4
│   ├── elm/
│   │   ├── __init__.py
│   │   ├── elm.py              # Method 5
│   │   └── elm_constrained.py  # Method 6
│   └── utils/
│       ├── __init__.py
│       ├── data.py             # Data loading and preprocessing
│       ├── metrics.py          # MSE, RMSE, R², MAE, MAPE
│       └── visualization.py    # Dashboards and plots
├── results/                    # Experiment results
├── figures/                    # Generated figures
├── notebooks/                  # Jupyter notebooks
├── tests/                      # Unit tests
│
│   # Execution Scripts
├── run_experiment.py           # Single method execution
├── run_all_methods.py          # Batch execution (6 methods)
├── run_cross_validation.py     # 5-fold cross-validation
├── generate_figures.py         # Publication-quality figures
├── regenerate_dashboard.py     # Comparative dashboards
│
│   # Environment Configuration
├── environment.yml             # Micromamba/Conda environment
├── requirements.txt            # pip dependencies
├── pyproject.toml              # Python project configuration
│
│   # Git and Development Configuration
├── .gitignore                  # Files ignored by Git
├── .gitattributes              # Git attributes (line endings)
├── .editorconfig               # Editor settings
├── .pre-commit-config.yaml     # Pre-commit hooks
├── Makefile                    # Task automation
│
├── LICENSE                     # Proprietary License
└── README.md
```

---

## Installation

### Via Micromamba (Recommended)

```bash
# Create environment
micromamba create -f environment.yml

# Activate environment
micromamba activate mestrado_ghabriel_pei

# Verify installation
python -c "import numpy; import scipy; print('OK')"
```

### Via Conda

```bash
conda env create -f environment.yml
conda activate mestrado_ghabriel_pei
```

### Via pip

```bash
pip install -r requirements.txt

# For GPU support (optional)
pip install torch
```

### Main Dependencies

| Package | Minimum Version |
|---------|-----------------|
| Python | >= 3.10 |
| NumPy | >= 1.24 |
| SciPy | >= 1.10 |
| Pandas | >= 2.0 |
| scikit-learn | >= 1.3 |
| Matplotlib | >= 3.7 |
| Seaborn | >= 0.12 |
| statsmodels | >= 0.14 |
| PyTorch | >= 2.0 (optional, GPU) |

---

## Usage

### Single Method Execution

```bash
# Method 1 (WILCAR) - default
python run_experiment.py --dataset computer_hardware

# Method 2 (WILCAR Constrained)
python run_experiment.py --dataset computer_hardware --method wilcar_constrained

# Method 3 (RIXM)
python run_experiment.py --dataset computer_hardware --method rixm

# Method 4 (RIXM Constrained)
python run_experiment.py --dataset computer_hardware --method rixm_constrained

# Method 5 (ELM)
python run_experiment.py --dataset computer_hardware --method elm

# Method 6 (ELM Constrained)
python run_experiment.py --dataset computer_hardware --method elm_constrained

# With custom parameters
python run_experiment.py --dataset qsar_fish_toxicity --method wilcar_unconstrained \
    --max-neurons 100 --patience 50 --seed 42

# List available datasets
python run_experiment.py --list
```

### Batch Execution (All Methods)

```bash
# All methods with default parameters
python run_all_methods.py --dataset computer_hardware

# Global parameters
python run_all_methods.py --dataset qsar_fish_toxicity --max-neurons 100 --patience 50

# Per-method parameters
python run_all_methods.py --dataset computer_hardware \
    --max-neurons-m1 750 --patience-m1 150 \
    --max-neurons-m2 300 --patience-m2 100 \
    --max-neurons-m3 750 --patience-m3 150 \
    --max-neurons-m4 100 --patience-m4 50 \
    --max-neurons-m5 125 \
    --max-neurons-m6 150 --patience-m6 50

# Only specific methods
python run_all_methods.py --dataset computer_hardware --methods 1 2 5 6

# Disable GPU
python run_all_methods.py --dataset computer_hardware --no-gpu
```

### Cross-Validation Execution

```bash
# 5-fold cross-validation with all methods
python run_cross_validation.py --dataset computer_hardware

# With custom parameters
python run_cross_validation.py --dataset qsar_fish_toxicity --folds 10 --seed 42
```

### Figure Generation

```bash
# Generate publication-quality figures
python generate_figures.py --dataset computer_hardware

# Regenerate comparative dashboard
python regenerate_dashboard.py --dataset computer_hardware
```

---

## Supported Datasets

| Dataset | Variables | Samples | Expected Signs |
|---------|-----------|---------|----------------|
| `computer_hardware` | 6 | 209 | `[-1, +1, +1, +1, +1, +1]` |
| `qsar_fish_toxicity` | 6 | 908 | `[-1, +1, -1, +1, +1, +1]` |
| `qsar_aquatic_toxicity` | 8 | 546 | `[+1, -1, +1, +1, +1, -1, +1, +1]` |

### Expected Signs Justification

#### Computer Hardware
| Variable | Sign | Justification |
|----------|------|---------------|
| MYCT (cycle time) | - | Higher cycle time → slower CPU → lower performance |
| MMIN (min memory) | + | More memory → better performance |
| MMAX (max memory) | + | More memory → better performance |
| CACH (cache) | + | More cache → better performance |
| CHMIN (min channels) | + | More I/O channels → better performance |
| CHMAX (max channels) | + | More I/O channels → better performance |

#### QSAR Fish Toxicity
| Variable | Sign | Justification |
|----------|------|---------------|
| CIC0 | - | Lower values → more heteroatoms → higher toxicity |
| SM1_Dz(Z) | + | Higher values → more heteroatoms (F, Cl, Br) → higher toxicity |
| GATS1i | - | Lower values → more aromatic/lipophilic → higher narcotic toxicity |
| NdsCH | + | Electrophilic carbons (=CH-) → reactive with biological nucleophiles |
| NdssC | + | Electrophilic carbons (=C<) → reactive with biological nucleophiles |
| MLOGP | + | Higher lipophilicity → driving force of narcosis → higher toxicity |

#### QSAR Aquatic Toxicity
| Variable | Sign | Justification |
|----------|------|---------------|
| TPSA | + | Polar surface area facilitates interaction with biological targets |
| SAacc | - | H-bond acceptors increase hydrophilicity → reduced bioconcentration |
| H-050 | + | H-bond donors enable specific interactions with biomolecules |
| MLOGP | + | Lipophilicity is the driving force of narcosis |
| RDCHI | + | Molecular size correlates with lipophilicity |
| GATS1p | - | Low values → more polarizable bonds → more lipophilic → higher toxicity |
| nN | + | Nitrogen nucleophilicity → covalent interactions → specific toxicity |
| C-040 | + | Electrophilic carbons → reactivity with biological nucleophiles |

### Adding a New Dataset

1. Place the `.csv` file in `data/raw/`
2. Add the expected signs in `src/utils/data.py`:
```python
EXPECTED_SIGNALS_THEORY = {
    # ... existing datasets ...
    'new_dataset': [+1, -1, +1, ...]  # theoretical signs
}
```

---

## Outputs

### Single Method (`run_experiment.py`)

```
results/<dataset>/<method>/run_<timestamp>/
├── metrics/
│   ├── detailed_metrics.csv    # Metrics per number of neurons
│   ├── best_model_summary.json # Best model summary
│   └── gains_analysis.csv      # Gains analysis
├── models/
│   └── best_model_n<X>.pkl     # Serialized model
├── figures/
│   ├── performance_dashboard.png/pdf
│   ├── gains_comparison.png
│   └── learning_curves.png
└── logs/
    └── execution_config.json
```

### Batch Execution (`run_all_methods.py`)

```
results/<dataset>/batch_<timestamp>/
├── method_1_wilcar_unconstrained/
├── method_2_wilcar_constrained/
├── method_3_rixm/
├── method_4_rixm_constrained/
├── method_5_elm/
├── method_6_elm_constrained/
├── comparative/
│   ├── all_methods_dashboard.png/pdf     # 6 methods comparison
│   ├── paired_wilcar_1_2.png/pdf         # WILCAR vs WILCAR+R
│   ├── paired_rixm_3_4.png/pdf           # RIXM vs RIXM+R
│   ├── paired_elm_5_6.png/pdf            # ELM vs ELM+R
│   ├── unconstrained_1_3_5.png/pdf       # Unconstrained methods
│   ├── constrained_2_4_6.png/pdf         # Constrained methods
│   ├── summary_table.csv                 # Summary table
│   └── results_table.tex                 # LaTeX table
└── batch_config.json
```

### Cross-Validation (`run_cross_validation.py`)

```
results/<dataset>/cross_validation_<timestamp>/
├── fold_1/
│   ├── method_1_wilcar_unconstrained/
│   ├── method_2_wilcar_constrained/
│   └── ...
├── fold_2/
├── ...
├── fold_5/
├── cv_summary.csv              # Summary statistics
├── cv_results.json             # Detailed results
└── cv_config.json              # Configuration used
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error |
| **RMSE** | Root Mean Squared Error |
| **R²** | Coefficient of Determination |
| **SCR** | Signal Conformity Rate |

### Signal Conformity Rate (SCR)

Measures the percentage of variables whose calculated gains have the correct sign (according to theory):

$$SCR = \frac{\text{Variables with correct sign}}{\text{Variables with known sign}} \times 100\%$$

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_neurons` | 750 | Maximum number of hidden neurons |
| `patience_constructive` | 150 | Constructive patience (neurons without improvement) |
| `patience_early_stopping` | 250 | Training patience (epochs without improvement) |
| `learning_rate` | 0.1 | Initial learning rate |
| `lr_decay` | 0.999 | Learning rate decay factor |
| `min_lr` | 1e-3 | Minimum learning rate |
| `iterations` | 2501 | Maximum iterations per training |
| `delta_perturbation` | 0.1 | Delta for gain calculation |
| `tolerance` | 1e-6 | Convergence tolerance |
| `seed` | 0 | Seed for reproducibility |
| `test_size` | 0.2 | Test set fraction |
| `use_gpu` | False | Enable GPU acceleration (requires PyTorch) |
| `verbose` | 1 | Verbosity level (0=silent, 1=summary, 2=detailed) |

---

## References

- **Xavier Initialization:** Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.
- **ELM:** Huang, G. B., Zhu, Q. Y., & Siew, C. K. (2006). Extreme learning machine: Theory and applications.
- **Constructive Algorithms:** Kwok, T. Y., & Yeung, D. Y. (1997). Constructive algorithms for structure learning in feedforward neural networks for regression problems.
- **WILCAR (original):** Gomes de Sá, G.A., Fontes, C.H., & Embiruçu, M. (2022). A new method for building single feedforward neural network models for multivariate static regression problems: a combined weight initialization and constructive algorithm. *Evolutionary Intelligence*. https://doi.org/10.1007/s12065-022-00813-z

---

## License

Copyright (c) 2025 Ghabriel Anton Gomes de Sá. All Rights Reserved.

This software is proprietary and may be subject to patent protection. See [LICENSE](LICENSE) for details.

---

## Development Environment Setup

### Git Initialization

```bash
# Clone repository (if applicable)
git clone https://github.com/ghabriel-ags/WILCAR
cd mestrado-neural-networks

# Or initialize new repository
git init
git add .
git commit -m "feat: initial commit"
```

### Pre-commit Installation

```bash
# Install pre-commit
pip install pre-commit

# Install hooks in the repository
pre-commit install

# Run on all files (first time)
pre-commit run --all-files
```

### Available Make Commands

```bash
make help          # Show all available commands
make install       # Install dependencies
make install-dev   # Install development dependencies
make install-gpu   # Install GPU dependencies (PyTorch)
make format        # Format code (ruff + isort)
make lint          # Check code
make test          # Run tests
make run           # Run experiment (DATASET=... METHOD=...)
make run-all       # Run all methods
make run-cv        # Run cross-validation
make figures       # Generate publication figures
make dashboard     # Regenerate comparative dashboard
make clean         # Remove temporary files
```

### Configuration Files

| File | Description |
|------|-------------|
| `environment.yml` | Micromamba/Conda environment |
| `requirements.txt` | pip dependencies |
| `pyproject.toml` | Project configuration (ruff, isort, pytest) |
| `.gitignore` | Files ignored by Git |
| `.gitattributes` | Git attributes (line endings WSL/Windows) |
| `.editorconfig` | Editor settings |
| `.pre-commit-config.yaml` | Pre-commit hooks |
| `Makefile` | Task automation |

---

## Contact

- **Author:** Ghabriel Anton Gomes de Sá
- **Program:** Graduate in Industrial Engineering - Universidade Federal da Bahia
- **Advisors:** Prof. Marcelo Embiruçu & Prof. Cristiano Fontes
