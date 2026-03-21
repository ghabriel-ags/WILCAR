# UNIVERSIDADE FEDERAL DA BAHIA (FEDERAL UNIVERSITY OF BAHIA)
## ESCOLA POLITÉCNICA (POLYTECHNIC SCHOOL)
## PROGRAMA DE PÓS-GRADUAÇÃO EM ENGENHARIA INDUSTRIAL (GRADUATE PROGRAM IN INDUSTRIAL ENGINEERING)

**Student:** Ghabriel Anton Gomes de Sá  
**Advisors:** Marcelo Embiruçu & Cristiano Fontes  

<h2 align="center">Constructive Neural Networks for Regression and Classification with Gain Sign Constraints</h2>

---

## Description

This repository contains the complete implementation of the methods developed and evaluated in the MSc dissertation. It implements and compares **6 constructive neural network methods** with a single hidden layer for both **regression** and **binary classification** problems, with the possibility of **gain sign constraints** enforced via SLSQP optimization.

The proposed method (WILCAR) combines:
- Weight initialization via **linearization** (Taylor series + Least Squares)
- **Constructive** approach with weight reuse
- Possibility of **gain sign constraints** via SLSQP optimization
- Support for **optional (partial) constraints**, where only a subset of variables is constrained
- **Dynamic Cross-Validation (DCV)** for architecture selection (WILCAR only)

This is the **final version** of the code, incorporating all functionalities described in Chapters 2, 4, 5, and 6 of the dissertation. Due to iterative improvements in hyperparameters, data preprocessing, and validation strategies, the results produced by this code may differ from those originally reported in the published article (Sá, Fontes, and Embiruçu, 2022).

---

## Implemented Methods

| Method | Initialization | Training | Loss (Reg.) | Loss (Cls.) | Constraints |
|--------|----------------|----------|-------------|-------------|-------------|
| **WILCAR** | Linearization | Backpropagation | MSE | BCE | ❌ |
| **WILCAR-C** | Linearization | SLSQP | MSE | BCE | ✅ |
| **RIXM** | Xavier (Random) | Backpropagation | MSE | BCE | ❌ |
| **RIXM-C** | Xavier (Random) | SLSQP | MSE | BCE | ✅ |
| **ELM** | Random (fixed) | Pseudo-inverse | MSE | MSE* | ❌ |
| **ELM-C** | Random (fixed) | SLSQP | MSE | BCE | ✅ |

\* ELM retains MSE with pseudo-inverse for classification, following Huang et al. (2006).

### Method Descriptions

- **WILCAR (Weight Initialization via Linearization with Constructive Algorithm and Reuse):** Proposed method that uses linearization for intelligent weight initialization and reuses weights from previous architectures during constructive expansion.
- **RIXM (Random Initialization Xavier Method):** Baseline with random Xavier initialization. Each architecture is trained from scratch (no weight reuse).
- **ELM (Extreme Learning Machine):** Hidden layer weights are fixed (random); only output weights are computed via Moore–Penrose pseudo-inverse. Each architecture is initialized from scratch.

### Constrained Variants (-C)

Constrained methods enforce gain sign constraints through inequality constraints in the SLSQP optimization. A **reinitialization strategy** guarantees 100% Signal Conformity Rate (SCR):
- Up to 100 reinitialization attempts per configuration
- Verification on the **test** set
- Configurations that do not achieve 100% SCR are skipped

### Optional (Partial) Constraints

The signal vector supports three states per variable:
- `+1`: positive gain expected (output increases with input)
- `-1`: negative gain expected (output decreases with input)
- `0`: unconstrained (no monotonicity imposed)

This enables **partial constraint** scenarios where only variables with strong theoretical support are constrained.

---

## Tasks Supported

### Regression (`--task regression`, default)
- Loss: Mean Squared Error (MSE)
- Metrics: R², RMSE, MSE, SCR
- Model selection: best R² on test set
- Target normalization: MinMaxScaler to [0, 1]

### Binary Classification (`--task classification`)
- Loss: Binary Cross-Entropy (BCE) for backprop/SLSQP methods; MSE for ELM pseudo-inverse
- Metrics: Accuracy, MCC, F1-score, SCR
- Model selection: best MCC on test set
- Target: binary {0, 1}, **not normalized**
- Prediction threshold: 0.5

---

## Validation Strategies

### Standard K-Fold Cross-Validation
Applied to all six methods. The constructive process runs independently within each fold. Results are reported as mean ± standard deviation across K folds.

### Dynamic Cross-Validation (DCV)
Applied exclusively to WILCAR and WILCAR-C (methods with weight reuse). Embeds K-fold cross-validation within the constructive loop, using the aggregated error across all folds as the stopping criterion. Produces a single, globally optimal architecture n*.

---

## Project Structure

```
thesis/
├── data/
│   ├── raw/                    # Original datasets (.csv)
│   └── processed/              # Preprocessed data
├── src/
│   ├── __init__.py
│   ├── base.py                 # Base classes, TrainingConfig, TrainingResults
│   ├── wilcar/
│   │   ├── __init__.py
│   │   ├── unconstrained.py    # WILCAR (backprop, weight reuse, linearization)
│   │   └── constrained.py      # WILCAR-C (SLSQP + reinitialization)
│   ├── rixm/
│   │   ├── __init__.py
│   │   ├── rixm.py             # RIXM (backprop, no reuse)
│   │   └── rixm_constrained.py # RIXM-C (SLSQP + reinitialization)
│   ├── elm/
│   │   ├── __init__.py
│   │   ├── elm.py              # ELM (pseudo-inverse)
│   │   └── elm_constrained.py  # ELM-C (SLSQP + reinitialization)
│   └── utils/
│       ├── __init__.py
│       ├── data.py             # Data loading, preprocessing, signal vectors
│       ├── metrics.py          # Regression and classification metrics
│       └── visualization.py    # Dashboards and plots
├── results/                    # Experiment results
│
│   # Execution Scripts
├── run_experiment.py           # Single method execution
├── run_all_methods.py          # Batch execution (6 methods)
├── run_cross_validation.py     # K-fold cross-validation
├── run_dynamic_cv.py           # Dynamic Cross-Validation (WILCAR only)
├── generate_figures.py         # Publication-quality figures
├── regenerate_dashboard.py     # Comparative dashboards
│
│   # Environment Configuration
├── environment.yml             # Micromamba/Conda environment
├── requirements.txt            # pip dependencies
├── pyproject.toml              # Python project configuration
├── .gitignore
├── .gitattributes
├── .editorconfig
├── .pre-commit-config.yaml
├── LICENSE
└── README.md
```

---

## Installation

### Via Micromamba (Recommended)

```bash
micromamba create -f environment.yml
micromamba activate wilcar
python -c "import numpy; import scipy; print('OK')"
```

### Via pip

```bash
pip install -r requirements.txt
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
| PyTorch | >= 2.0 (optional, GPU) |

---

## Usage

### Regression (default)

```bash
# Single method
python run_experiment.py --dataset computer_hardware --method wilcar_unconstrained

# All methods
python run_all_methods.py --dataset computer_hardware

# 5-fold cross-validation
python run_cross_validation.py --dataset computer_hardware --methods 1 2 3 4 5 6

# Dynamic Cross-Validation (WILCAR only)
python run_dynamic_cv.py --dataset computer_hardware --methods 1 2
```

### Binary Classification

```bash
# Single method
python run_experiment.py --task classification --dataset breast__binary_ --method wilcar_unconstrained

# All methods
python run_all_methods.py --task classification --dataset breast__binary_

# 5-fold cross-validation
python run_cross_validation.py --task classification --dataset breast__binary_ --methods 1 2 3 4 5 6

# Loop over all classification datasets
for ds in Algerian_forest_fires__binary_ breast__binary_ Diagnostic_Breast_Cancer__binary_ heart_failure_clinical_records_dataset__binary_ pima__binary_; do
    python run_cross_validation.py --task classification --methods 1 2 3 4 5 6 --dataset "$ds"
done
```

---

## Supported Datasets

### Regression (11 datasets, Chapter 5)

| Dataset | p | n | CR | Signal Vector |
|---------|---|---|-----|---------------|
| `airfoil_self_noise` | 5 | 1503 | 80% | `[-1, 0, -1, +1, -1]` |
| `computer_hardware` | 6 | 209 | 100% | `[-1, +1, +1, +1, +1, +1]` |
| `ENB2012_Y1` (Heating Load) | 8 | 768 | 75% | `[-1, +1, +1, +1, -1, 0, +1, 0]` |
| `ENB2012_Y2` (Cooling Load) | 8 | 768 | 75% | `[-1, +1, +1, +1, -1, 0, +1, 0]` |
| `lavender_friction_DFC` | 3 | 625 | 67% | `[+1, -1, 0]` |
| `optical_interconnection_network` | 3 | 630 | 67% | `[-1, 0, +1]` |
| `qsar_aquatic_toxicity` | 8 | 546 | 100% | `[+1, -1, +1, +1, +1, -1, +1, +1]` |
| `qsar_fish_toxicity` | 6 | 908 | 100% | `[-1, +1, -1, +1, +1, +1]` |
| `Real_estate_valuation` | 3 | 414 | 100% | `[-1, -1, +1]` |
| `synchronous_machine` | 4 | 557 | 100% | `[+1, +1, -1, +1]` |
| `yacht_hydrodynamics` | 6 | 308 | 17% | `[0, 0, 0, 0, +1, 0]` |

### Binary Classification (5 datasets, Chapter 6)

| Dataset | p | n | CR | Signal Vector |
|---------|---|---|-----|---------------|
| `Algerian_forest_fires__binary_` | 10 | 244 | 90% | `[+1, -1, 0, -1, +1, +1, +1, +1, +1, +1]` |
| `breast__binary_` | 9 | 683 | 100% | `[+1, +1, +1, +1, +1, +1, +1, +1, +1]` |
| `Diagnostic_Breast_Cancer__binary_` | 30 | 569 | 90% | `[+1]×9, [0], [+1]×9, [0], [+1]×9, [0]` |
| `heart_failure_clinical_records_dataset__binary_` | 12 | 299 | 58% | `[+1, +1, 0, 0, -1, +1, 0, +1, -1, 0, 0, -1]` |
| `pima__binary_` | 8 | 768 | 75% | `[+1, +1, 0, +1, 0, +1, +1, +1]` |

CR = Constraint Ratio (proportion of variables with definite gain sign).

---

## Evaluation Metrics

### Regression

| Metric | Description |
|--------|-------------|
| **R²** | Coefficient of Determination (primary selection metric) |
| **RMSE** | Root Mean Squared Error |
| **MSE** | Mean Squared Error |
| **SCR** | Signal Conformity Rate |

### Classification

| Metric | Description |
|--------|-------------|
| **MCC** | Matthews Correlation Coefficient (primary selection metric) |
| **Accuracy** | Proportion of correct predictions |
| **F1-score** | Harmonic mean of precision and recall |
| **SCR** | Signal Conformity Rate |

### Signal Conformity Rate (SCR)

Measures the proportion of constrained variables whose computed gains match the expected signs:

$$SCR = \frac{\text{Variables with correct sign}}{\text{Variables with known sign}} \times 100\%$$

Computed via numerical perturbation (δ = 0.1) on the test set.

---

## Training Parameters

| Parameter | Unconstrained | Constrained | Description |
|-----------|---------------|-------------|-------------|
| `max_neurons` | 750 | 50 | Maximum hidden neurons |
| `patience_constructive` | 150 | 15 | Constructive patience |
| `patience_early_stopping` | 250 | 250 | Epoch-level patience |
| `learning_rate` | 0.1 | 0.1 | Initial learning rate |
| `lr_decay` | 0.999 | 0.999 | LR decay factor per epoch |
| `iterations` | 2501 | 2501 | Max epochs per step |
| `delta_perturbation` | 0.1 | 0.1 | Delta for gain computation |
| `seed` | 0 | 0 | Random seed |
| `K` | 5 | 5 | Number of CV folds |

These hyperparameters are uniform across all datasets and both tasks (regression and classification).

---

## References

- **WILCAR:** Sá, G.A.G., Fontes, C.H., & Embiruçu, M. (2022). A new method for building single feedforward neural network models for multivariate static regression problems. *Evolutionary Intelligence*, 15, 1221–1231.
- **Gain Sign Constraints:** Sá, G.A.G., Fontes, C.H., & Embiruçu, M. (2026). Physics-informed neural networks without a phenomenological model available. *Neural Computing and Applications*. Under review.
- **Physical Consistency:** Fontes, C.H., Sá, G.A.G., & Embiruçu, M. (2024). Do properly validated networks ensure minimum physical consistency with reality? *Journal of Artificial Intelligence and Systems*, 6, 1–18.
- **ELM:** Huang, G.B., Zhu, Q.Y., & Siew, C.K. (2006). Extreme learning machine: Theory and applications. *Neurocomputing*, 70, 489–501.
- **Xavier Initialization:** Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.

---

## License

Copyright (c) 2025–2026 Ghabriel Anton Gomes de Sá. All Rights Reserved.

This software is proprietary and may be subject to patent protection. See [LICENSE](LICENSE) for details.

---

## Contact

- **Author:** Ghabriel Anton Gomes de Sá
- **Program:** MSc in Industrial Engineering — Universidade Federal da Bahia
- **Advisors:** Prof. Marcelo Embiruçu & Prof. Cristiano Fontes
- **Repository:** [github.com/ghabriel-ags/WILCAR](https://github.com/ghabriel-ags/WILCAR)