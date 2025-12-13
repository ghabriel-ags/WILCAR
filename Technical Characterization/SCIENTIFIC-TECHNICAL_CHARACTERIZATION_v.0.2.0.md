# Technical Characterization: Neural Network Framework for Regression with Gain Sign Constraints

**Federal University of Bahia — Graduate Program in Industrial Engineering**  
**Author:** Ghabriel Anton Gomes de Sá  
**Advisors:** Prof. Dr. Marcelo Embiruçu, Prof. Dr. Cristiano Fontes  
**Version:** 0.3.0  
**Date:** December 2025

---

## Abstract

This document provides a rigorous technical characterization of a computational framework for training single-hidden-layer feedforward neural networks (SLFNs) under gain sign constraints. The framework implements six distinct methods derived from three base algorithms—WILCAR, RIXM, and ELM—each offered in unconstrained and constrained variants. The constrained methods guarantee 100% Signal Conformity Rate (SCR) through a novel reinitialization strategy, ensuring that the trained models respect theoretical expectations regarding the directional influence of input variables on the output. This characterization serves as the foundational reference for the methodology section of an academic publication.

---

## 1. Introduction and Motivation

### 1.1 The Problem of Physical Consistency in Neural Networks

Neural networks, despite their remarkable approximation capabilities, are often criticized for producing models that violate fundamental physical or domain-specific principles. In regression problems arising from scientific and engineering domains, the relationship between input variables and the target output frequently follows known monotonicity patterns. For instance:

- In thermodynamic systems, temperature increases typically lead to increased reaction rates
- In pharmacokinetic models, drug concentration is expected to correlate positively with dosage
- In economic models, certain price elasticities have well-established directional behaviors

When a neural network produces a model where the partial derivative ∂ŷ/∂xᵢ (the "gain" with respect to input i) has a sign contrary to theoretical expectations, the model—regardless of its statistical accuracy—becomes suspect from a domain perspective and potentially unsuitable for extrapolation or interpretation.

### 1.2 The Gain Sign Constraint Paradigm

We define the **gain** of a neural network model f with respect to input variable xᵢ as:

$$g_i = \frac{\partial f(\mathbf{x})}{\partial x_i}$$

For practical computation, this is approximated via finite differences:

$$\hat{g}_i = \frac{f(\mathbf{x} + \delta \mathbf{e}_i) - f(\mathbf{x})}{\delta}$$

where **e**ᵢ is the i-th canonical basis vector and δ is a small perturbation (typically δ = 0.1).

The **expected signal** for variable i, denoted σᵢ ∈ {-1, 0, +1}, encodes prior knowledge:
- σᵢ = +1: positive influence expected (increasing xᵢ should increase ŷ)
- σᵢ = -1: negative influence expected (increasing xᵢ should decrease ŷ)
- σᵢ = 0: no prior expectation (unconstrained)

The **Signal Conformity Rate (SCR)** measures the proportion of constrained variables whose calculated gain signs match expectations:

$$\text{SCR} = \frac{|\{i : \sigma_i \neq 0 \land \text{sign}(\hat{g}_i) = \sigma_i\}|}{|\{i : \sigma_i \neq 0\}|}$$

### 1.3 Research Objectives

This framework addresses the following objectives:

1. **Develop constrained training methods** that guarantee SCR = 100% while maintaining competitive predictive performance
2. **Provide comparative analysis** between unconstrained and constrained variants of three distinct base algorithms
3. **Establish reproducible benchmarks** using publicly available datasets from the UCI Machine Learning Repository

---

## 2. Network Architecture

### 2.1 Single-Hidden-Layer Feedforward Network (SLFN)

All methods in this framework employ the SLFN architecture, which provides a principled balance between approximation capability and interpretability. The network structure is:

```
Input Layer          Hidden Layer           Output Layer
[n features]    →    [k neurons]       →    [1 output]
    xᵢ         →    hⱼ = φ(Σ wᵢⱼxᵢ + bⱼ)  →    ŷ = ψ(Σ vⱼhⱼ + c)
```

Formally, for input vector **x** ∈ ℝⁿ:

$$\hat{y} = \psi\left(\sum_{j=1}^{k} v_j \cdot \phi\left(\sum_{i=1}^{n} w_{ij} x_i + b_j\right) + c\right)$$

where:
- **W** ∈ ℝᵏˣⁿ: input-to-hidden weight matrix
- **b** ∈ ℝᵏ: hidden layer biases
- **v** ∈ ℝᵏ: hidden-to-output weights
- c ∈ ℝ: output bias
- φ: hidden layer activation function
- ψ: output layer activation function

### 2.2 Activation Functions

The framework employs two activation functions depending on the method:

**Sigmoid Function (WILCAR, RIXM):**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

with numerically stable implementation:
$$\sigma(z) = \begin{cases} \frac{1}{1 + e^{-z}} & \text{if } z \geq 0 \\ \frac{e^z}{1 + e^z} & \text{if } z < 0 \end{cases}$$

**Rectified Linear Unit (ELM):**
$$\text{ReLU}(z) = \max(0, z)$$

### 2.3 Constructive Network Growth

All methods employ a **constructive approach** where the network grows incrementally:

1. Start with k = 1 hidden neuron
2. Train the network to convergence
3. Evaluate performance metrics
4. Add one neuron (k ← k + 1)
5. Repeat until stopping criterion is met

**Weight Reuse Strategy** (varies by method):

| Method | Weight Reuse |
|--------|--------------|
| WILCAR (M1, M2) | Yes — preserves weights from k-1 neurons, only initializes new neuron |
| RIXM (M3, M4) | No — each model trained from scratch |
| ELM (M5, M6) | No — each model trained independently |

The weight reuse in WILCAR offers advantages:
- Warm-start initialization accelerates convergence
- Gradual complexity increase preserves learned representations
- Natural model selection via early stopping

---

## 3. Base Algorithms

### 3.1 WILCAR (Methods 1 and 2)

**WILCAR** (Weight Initialization via Linearization for Constructive ARchitectures) is a constructive neural network training algorithm characterized by a principled initialization strategy for the first hidden neuron.

#### 3.1.1 Linearization of the Nonlinear Model

The initialization process begins with the linearization of the nonlinear neural network model using **Taylor Series expansion**. The equilibrium point chosen for linearization is the **mean of the input variables** (**x**_M), which ensures that the linearization occurs around a central and representative point in the data space.

This strategic choice provides:
- A more precise approximation for weight initialization
- Alignment with the central tendency of the training data
- Improved convergence during subsequent training

#### 3.1.2 Multiple Linear Regression

After linearization, **Multiple Linear Regression** is applied to the dataset using the **Ordinary Least Squares (OLS)** method. This statistical approach estimates the coefficients that best describe the relationship between input and output variables:

$$\hat{y}_{\text{linear}} = b + \sum_{i=1}^{n} k_i x_i$$

where:
- b is the intercept term
- kᵢ are the regression coefficients for each input variable
- The premise is that near the linearization point, variables exhibit a linear or quasi-linear relationship

#### 3.1.3 Mathematical Comparison and Weight Derivation

The final phase involves comparing the mathematical expressions derived from model linearization with the coefficients obtained from multiple linear regression. We seek weights (w₁, ..., wₙ, β) such that the sigmoid neuron approximates the linear behavior near the equilibrium point.

The system of equations derived from Taylor expansion:
$$\sigma(z_M) - \sigma'(z_M) \cdot \mathbf{w}^\top \mathbf{x}_M = b$$
$$\sigma'(z_M) \cdot w_i = k_i, \quad i = 1, \ldots, n$$

where z_M = **w**ᵀ**x**_M + β and σ is the sigmoid activation function.

This nonlinear system is solved using `scipy.optimize.fsolve` with multiple initial guesses to ensure convergence. The identified weights serve as initial values for the connections between input variables and the first hidden neuron.

#### 3.1.4 Subsequent Neurons: Xavier Initialization

For neurons j > 1, weights are initialized using Xavier (Glorot) initialization:
$$w_{ij} \sim \mathcal{N}\left(0, \sqrt{\frac{1}{n}}\right)$$

This initialization is crucial to prevent gradients from becoming too small (vanishing) or too large (exploding) during training.

#### 3.1.5 Preservation of Learned Weights

A key feature of WILCAR is the **preservation of previously learned weights**. When expanding the network from k-1 to k neurons:

1. All weights from the k-1 neuron model are preserved
2. Only the new k-th neuron is initialized (via Xavier)
3. Training continues from this warm-start configuration

This iterative training process allows the network to grow in complexity without discarding previously learned information, using the best previous model as the starting point for additional training.

#### 3.1.6 Training via Backpropagation

The network is trained using gradient descent with sigmoid activation and the following update rules:

$$\mathbf{W}^{(t+1)} = \mathbf{W}^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}}$$

where η is the learning rate and ℒ is the Mean Squared Error loss:
$$\mathcal{L} = \frac{1}{m} \sum_{p=1}^{m} (\hat{y}_p - y_p)^2$$

**Training Configuration:**
- Initial learning rate: η₀ = 0.1
- Learning rate decay: η_t = max(η_{t-1} × 0.999, 10⁻³)
- Maximum iterations per model: 2,501
- Early stopping patience: 250 iterations without improvement
- Convergence tolerance: 10⁻⁶

### 3.2 RIXM (Methods 3 and 4)

**RIXM** (Random Initialization Xavier Method) serves as a **baseline** for comparison with WILCAR. Unlike WILCAR, RIXM uses a simpler random initialization strategy:

#### 3.2.1 Key Differences from WILCAR

| Aspect | WILCAR | RIXM |
|--------|--------|------|
| First neuron initialization | Linearization (Taylor expansion) | Xavier random |
| Weight reuse between models | Yes (warm start) | No (each model from scratch) |
| Purpose | Optimized training | Baseline comparison |

#### 3.2.2 Xavier Initialization

All neurons (including the first) are initialized using Xavier (Glorot) initialization:
$$w_{ij} \sim \mathcal{N}\left(0, \sqrt{\frac{1}{n}}\right)$$

Biases are initialized to zero: $b_j = 0$

#### 3.2.3 Training Without Weight Reuse

Each model with k neurons is trained **from scratch**:
1. Initialize all k neurons with Xavier random weights
2. Train via backpropagation to convergence
3. Evaluate performance
4. Increment k and repeat (no weight preservation)

This approach isolates the effect of WILCAR's initialization strategy, allowing direct comparison of:
- Linearization initialization vs. random initialization
- Warm-start weight reuse vs. independent training

#### 3.2.4 Training Configuration

Same as WILCAR:
- Initial learning rate: η₀ = 0.1
- Learning rate decay: η_t = max(η_{t-1} × 0.999, 10⁻³)
- Maximum iterations per model: 2,501
- Early stopping patience: 250 iterations
- Convergence tolerance: 10⁻⁶

### 3.3 ELM — Extreme Learning Machine (Methods 5 and 6)

**ELM** represents a fundamentally different paradigm where hidden layer weights are randomly assigned and never updated.

#### 3.3.1 Random Feature Mapping

Hidden layer weights are drawn from a uniform distribution:
$$w_{ij} \sim \mathcal{U}(-1, 1)$$

These weights remain **fixed** throughout training.

#### 3.3.2 Analytical Output Weight Calculation

Given hidden layer activations **H** ∈ ℝᵐˣᵏ (m samples, k neurons), output weights are calculated via the Moore-Penrose pseudo-inverse:

$$\boldsymbol{\beta} = \mathbf{H}^\dagger \mathbf{y} = (\mathbf{H}^\top \mathbf{H})^{-1} \mathbf{H}^\top \mathbf{y}$$

This provides a **single-pass, closed-form solution** with no iterative optimization, resulting in extremely fast training times.

#### 3.3.3 ELM Training Procedure

Unlike WILCAR/RIXM, ELM does **not** reuse weights between models. Each model with k neurons is trained independently:

1. Generate random input weights **W** ∈ ℝⁿˣᵏ
2. Compute hidden layer output: **H** = ReLU(**XW** + **b**)
3. Calculate output weights: **β** = **H**†**y**
4. Evaluate performance

---

## 4. Gain Sign Constraint Enforcement

### 4.1 Unconstrained Methods (M1, M3, M5)

The unconstrained variants (WILCAR, RIXM, ELM) perform standard training without any constraint on gain signs. The SCR is computed post-hoc for analysis but does not influence the training process.

### 4.2 Constrained Methods (M2, M4, M6)

The constrained variants employ a **reinitialization strategy** to guarantee SCR = 100%. While all three methods share the same goal, their implementations differ based on their base algorithms.

#### 4.2.1 Method-Specific Initialization Strategies

| Method | First Neuron | Subsequent Neurons | Weight Reuse |
|--------|-------------|-------------------|--------------|
| M2 (WILCAR+R) | Constrained linearization (SLSQP) | Xavier + reinitialization | Yes |
| M4 (RIXM+R) | Xavier random | Xavier + reinitialization | No |
| M6 (ELM+R) | Random uniform | Random uniform | No |

#### 4.2.2 WILCAR+R (M2): Constrained Linearization

For the first hidden neuron, the linearization-based initialization is reformulated as a **constrained optimization problem** with two key components:

**Objective Function**: The `objective_function` calculates the discrepancy between the outputs of the linearized model and the values obtained from multiple linear regression:

$$\min_{\mathbf{w}, \beta} \quad (b - f_0)^2 + \sum_{i=1}^{n}(k_i - g_i)^2$$

where:
- b is the intercept from linear regression
- f₀ is the network output at the equilibrium point
- kᵢ are the regression coefficients
- gᵢ are the linearized gain coefficients

The objective aims to minimize this discrepancy, adjusting weights and bias to align the linearized model with regression results.

**Gain Constraint Function**: The `gain_constraint` function applies restrictions based on expected gain signs during optimization:

$$\sigma_i \cdot \hat{g}_i \geq \epsilon, \quad \forall i : \sigma_i \neq 0$$

where:
- σᵢ is the expected signal (+1 or -1)
- ĝᵢ is the calculated gain
- ε = 0.05 is a small margin ensuring strict inequality

These constraints ensure the adjusted model respects the expected gain signs for each input variable.

**Optimization Process**: Using `scipy.optimize.minimize` with SLSQP method, weights and bias are optimized to minimize the objective function while satisfying gain constraints. The optimization starts from a random initial point (x0).

**Result Validation**: After optimization, the `calculate_and_compare_gains` function computes the gains with adjusted weights and compares them with expected gain signs, validating that the optimization met the specified constraints.

#### 4.2.3 Constrained Training with Reinitialization

For subsequent neurons (and all neurons in RIXM+R), a **reinitialization strategy** ensures 100% SCR:

```
Algorithm: Constrained Training with Reinitialization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Training data, previous model parameters (if any), target neuron count k
Output: Model with SCR = 100%

1. best_scr ← 0
2. best_params ← None
3. for attempt = 1 to MAX_ATTEMPTS do
4.     if WILCAR+R and attempt = 1 then
5.         params ← expand_network(previous_params, k)  // Weight reuse
6.     else if WILCAR+R then
7.         params ← expand_network(previous_params, k)  // New random neuron only
8.     else  // RIXM+R
9.         params ← initialize_from_scratch(k)  // All neurons random
10.    end if
11.    
12.    params ← optimize_with_constraints(params)
13.    scr ← evaluate_scr_on_test_set(params)
14.    
15.    if scr > best_scr then
16.        best_scr ← scr
17.        best_params ← params
18.    end if
19.    
20.    if scr = 1.0 then
21.        return best_params  // Success: 100% SCR achieved
22.    end if
23. end for
24. return best_params  // Best effort if 100% not achieved
```

**Key Design Decisions:**
- **Maximum attempts**: MAX_ATTEMPTS = 100 reinitializations per model configuration
- **Reinitialization scope (WILCAR+R)**: Only the newest neuron is reinitialized; previous neurons retain weights
- **Reinitialization scope (RIXM+R)**: All neurons reinitialized (no weight preservation)
- **Verification domain**: SCR is evaluated on the **test set** to ensure generalization
- **Constraint formulation**: Each gain constraint is formulated as an inequality constraint for SLSQP

#### 4.2.4 ELM Constrained (M6) Specifics

Since ELM uses fixed random hidden weights, the constraint enforcement operates differently:

1. Generate random hidden weights W ~ U(-1, 1)
2. Optimize output weights via SLSQP (not pseudo-inverse) with gain constraints
3. Evaluate SCR on test set
4. If SCR < 100%, regenerate hidden weights and repeat
5. Continue until SCR = 100% or MAX_ATTEMPTS reached

Unlike traditional ELM which uses pseudo-inverse for output weights, ELM+R uses SLSQP optimization to enforce gain sign constraints while minimizing MSE.

---

## 5. Evaluation Metrics

### 5.1 Regression Performance Metrics

**Mean Squared Error (MSE):**
$$\text{MSE} = \frac{1}{m} \sum_{p=1}^{m} (\hat{y}_p - y_p)^2$$

**Root Mean Squared Error (RMSE):**
$$\text{RMSE} = \sqrt{\text{MSE}}$$

**Coefficient of Determination (R²):**
$$R^2 = 1 - \frac{\sum_{p=1}^{m} (\hat{y}_p - y_p)^2}{\sum_{p=1}^{m} (y_p - \bar{y})^2}$$

where ȳ is the mean of observed values.

### 5.2 Constraint Compliance Metrics

**Signal Conformity Rate (SCR):**
$$\text{SCR} = \frac{\text{Number of conforming variables}}{\text{Number of constrained variables}} \times 100\%$$

A variable is **conforming** if sign(ĝᵢ) = σᵢ.

**Full SCR Adherence**: A method achieves "full adherence" if SCR = 100% for all trained models (n = 1, 2, ..., k_max).

### 5.3 Computational Efficiency

**Training Time**: Total wall-clock time for complete training procedure.

**Time per Model**: Average time to train a single model configuration.

### 5.4 Overfitting Analysis

**Overfitting Gap:**
$$\Delta R^2 = R^2_{\text{train}} - R^2_{\text{test}}$$

Values significantly greater than zero indicate overfitting.

---

## 6. Experimental Configuration

### 6.1 Datasets

Three datasets from the UCI Machine Learning Repository are employed:

#### 6.1.1 Computer Hardware (computer_hardware.csv)

| Attribute | Description |
|-----------|-------------|
| **Source** | UCI Machine Learning Repository |
| **Samples** | 209 |
| **Input Variables** | 6 |
| **Output Variable** | Published Relative Performance (PRP) |
| **Expected Signals** | [-1, +1, +1, +1, +1, +1] |

**Variable Descriptions and Expected Signs Justification:**

| Variable | Sign | Description | Justification |
|----------|------|-------------|---------------|
| MYCT | -1 | Machine Cycle Time (ns) | Higher cycle time → slower CPU → lower performance |
| MMIN | +1 | Minimum Main Memory (KB) | More memory → better performance |
| MMAX | +1 | Maximum Main Memory (KB) | More memory → better performance |
| CACH | +1 | Cache Memory (KB) | More cache → better performance |
| CHMIN | +1 | Minimum Channels | More I/O channels → better performance |
| CHMAX | +1 | Maximum Channels | More I/O channels → better performance |

#### 6.1.2 QSAR Fish Toxicity (fish_toxicity.csv)

| Attribute | Description |
|-----------|-------------|
| **Source** | UCI Machine Learning Repository |
| **Samples** | 908 |
| **Input Variables** | 6 |
| **Output Variable** | LC50 -log(mol/L) — Lethal Concentration 50% (96h, *Pimephales promelas*) |
| **Expected Signals** | [-1, +1, -1, +1, +1, +1] |

**Variable Descriptions and Expected Signs Justification:**

| Variable | Sign | Description | Justification |
|----------|------|-------------|---------------|
| CIC0 | -1 | Complementary Information Content index | Lower values → more heteroatom diversity → higher specific toxicity. From original article: "higher toxicity is possessed by molecules with lower CIC0" |
| SM1_Dz(Z) | +1 | Spectral moment (Barysz matrix, atomic number) | Higher values correlate with heteroatom presence (ρ = 0.86 with heteroatom count). Halogenated compounds (F, Cl, Br) show higher toxicity. From article: "higher toxicity... with larger SM1_Dz(Z) values" |
| GATS1i | -1 | Geary autocorrelation (ionization potential) | Lower values → higher carbon content and aromatic character → higher lipophilicity → increased narcotic toxicity. From article: "toxicity increases with decreasing values of the descriptor" |
| NdsCH | +1 | Count of =CH- groups (sp² carbons) | Electrophilic carbon centers react with biological nucleophiles → increased specific toxicity |
| NdssC | +1 | Count of =C< groups (sp² carbons) | Similar to NdsCH: electrophilic functional groups (ketones, aldehydes, esters) increase toxicity |
| MLOGP | +1 | Moriguchi octanol-water partition coefficient | Lipophilicity is the driving force of narcosis. From article: "molecules with larger MLOGP... tend to have greater toxicity" |

**Reference:** Cassotti, M., Ballabio, D., Todeschini, R., Consonni, V. (2015). A similarity-based QSAR model for predicting acute toxicity towards the fathead minnow. *SAR and QSAR in Environmental Research*, 26, 217-243.

#### 6.1.3 QSAR Aquatic Toxicity (aquatic_toxicity.csv)

| Attribute | Description |
|-----------|-------------|
| **Source** | UCI Machine Learning Repository |
| **Samples** | 546 |
| **Input Variables** | 8 |
| **Output Variable** | LC50 -log(mol/L) — Lethal Concentration 50% (48h, *Daphnia magna*) |
| **Expected Signals** | [+1, -1, +1, +1, +1, -1, +1, +1] |

**Variable Descriptions and Expected Signs Justification:**

| Variable | Sign | Description | Justification |
|----------|------|-------------|---------------|
| TPSA(Tot) | +1 | Topological Polar Surface Area | Polar surface area facilitates interaction with biological targets in *D. magna*. Unlike fish (where narcosis dominates), crustaceans may have receptors where polar interactions increase toxicity |
| SAacc | -1 | Surface Area of H-bond acceptors | H-bond acceptors increase overall hydrophilicity → reduced bioconcentration → decreased toxicity. Net effect is protective despite potential receptor interactions |
| H-050 | +1 | Hydrogens bonded to heteroatoms | H-bond **donors** (-OH, -NH, -SH) enable specific interactions with biological nucleophiles (protein carbonyls). Distinct from SAacc (acceptors) |
| MLOGP | +1 | Moriguchi octanol-water partition coefficient | Lipophilicity drives narcotic toxicity (consistent with fish toxicity mechanism) |
| RDCHI | +1 | Reciprocal distance Randic-like index | Encodes molecular size; larger molecules → higher lipophilicity → increased bioconcentration → higher toxicity |
| GATS1p | -1 | Geary autocorrelation (polarizability) | Low values indicate highly polarizable bonds (I, Br, aromatic systems) → lipophilic character → increased toxicity. Inverse relationship: GATS1p↓ → toxicity↑ |
| nN | +1 | Number of nitrogen atoms | Nitrogen nucleophilicity (amines) → covalent interactions with electrophilic biological targets → specific toxicity mechanism |
| C-040 | +1 | Electrophilic carbon atoms (R-C(=X)-X, R-C≡X) | Electron-poor carbons (esters, acids, nitriles) react with biological nucleophiles → increased specific toxicity |

**Reference:** Cassotti, M., Ballabio, D., Consonni, V., Mauri, A., Tetko, I. V., Todeschini, R. (2014). Prediction of acute aquatic toxicity towards daphnia magna using GA-kNN method. *Alternatives to Laboratory Animals (ATLA)*, 42, 31-41.

### 6.2 Data Preprocessing

The preprocessing pipeline ensures data consistency and adequacy for the modeling process:

1. **Data Import**: Data is imported from CSV files following specific format requirements (semicolon-separated, no header, last column as output variable)

2. **Expected Gain Signals Collection**: Expected signs for each input variable's gain are obtained via the `get_expected_signals` function, which stores these expectations for constraint enforcement

3. **Missing Value Treatment**: Initial inspection identifies and removes rows with missing or inconsistent values. This includes analysis of zero values to determine if they represent missing data or valid attribute values

4. **Normalization**: All variables (inputs and output) are normalized to the [0, 1] range using **MinMaxScaler**. This facilitates the learning process and improves training efficiency by ensuring all values are on the same scale

5. **Input/Output Delimitation**: The dataset is divided into input variables (all columns except the last) and output variable (last column)

6. **Train/Test Split**: The dataset is split into training (80%) and testing (20%) subsets using `train_test_split` with a fixed random seed (seed=0) for reproducibility

### 6.3 Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| max_neurons | 750 | Maximum number of hidden neurons |
| patience_constructive | 150 | Models without improvement before stopping |
| patience_early_stopping | 250 | Iterations without improvement per model |
| learning_rate | 0.1 | Initial learning rate |
| lr_decay | 0.999 | Learning rate decay factor per iteration |
| min_lr | 10⁻³ | Minimum learning rate |
| iterations | 2,501 | Maximum iterations per model |
| delta_perturbation | 0.1 | Perturbation for gain calculation |
| tolerance | 10⁻⁶ | Convergence tolerance |
| seed | 0 | Random seed for reproducibility |
| test_size | 0.2 | Fraction of data reserved for testing |
| MAX_REINIT_ATTEMPTS | 100 | Maximum reinitialization attempts (constrained methods) |

---

## 7. Method Summary

| ID | Name | Base Algorithm | Constraint | Key Characteristics |
|----|------|----------------|------------|---------------------|
| M1 | WILCAR | WILCAR | None | Linearization init, backprop, weight reuse |
| M2 | WILCAR+R | WILCAR | Gain Sign | SLSQP optimization, weight reuse, reinitialization |
| M3 | RIXM | RIXM | None | Xavier random init, backprop, no weight reuse (baseline) |
| M4 | RIXM+R | RIXM | Gain Sign | Xavier random init, SLSQP, reinitialization (baseline) |
| M5 | ELM | ELM | None | Random fixed hidden weights, pseudo-inverse |
| M6 | ELM+R | ELM | Gain Sign | Random hidden weights, SLSQP output weights, reinitialization |

**Legend:**
- "+R" suffix indicates constrained variant with **R**einitialization strategy
- All constrained methods (M2, M4, M6) guarantee SCR = 100%
- RIXM methods serve as baselines to isolate the effect of WILCAR's linearization strategy

---

## 8. Software Implementation

### 8.1 Technology Stack

| Component | Technology | Minimum Version |
|-----------|------------|-----------------|
| **Language** | Python | 3.10 |
| **Numerical Computing** | NumPy | 1.24 |
| **Optimization** | SciPy | 1.10 |
| **Statistics** | Statsmodels | 0.14 |
| **Data Processing** | Pandas | 2.0 |
| **Machine Learning** | Scikit-learn | 1.3 |
| **Visualization** | Matplotlib | 3.7 |
| **Visualization** | Seaborn | 0.12 |
| **GPU Acceleration** | PyTorch (optional) | 2.0 |

### 8.2 Code Architecture

```
thesis/
├── src/
│   ├── __init__.py              # Main package exports
│   ├── base.py                  # Base classes, configurations, dataclasses
│   ├── wilcar/
│   │   ├── __init__.py
│   │   ├── unconstrained.py     # M1: WILCAR unconstrained
│   │   └── constrained.py       # M2: WILCAR+R constrained
│   ├── rixm/
│   │   ├── __init__.py
│   │   ├── rixm.py              # M3: RIXM unconstrained (baseline)
│   │   └── rixm_constrained.py  # M4: RIXM+R constrained (baseline)
│   ├── elm/
│   │   ├── __init__.py
│   │   ├── elm.py               # M5: ELM unconstrained
│   │   └── elm_constrained.py   # M6: ELM+R constrained
│   └── utils/
│       ├── __init__.py
│       ├── data.py              # Data loading, preprocessing, expected signals
│       ├── metrics.py           # MSE, RMSE, R², SCR calculations
│       └── visualization.py     # Plotting and dashboard generation
├── data/
│   └── raw/                     # UCI dataset CSV files
├── results/                     # Experiment outputs (gitignored)
├── figures/                     # Generated visualizations (gitignored)
├── notebooks/                   # Jupyter notebooks (gitignored)
├── tests/                       # Unit tests
│
│   # Execution Scripts
├── run_experiment.py            # Single method execution
├── run_all_methods.py           # Batch execution with comparative analysis
├── run_cross_validation.py      # K-fold cross-validation
├── generate_figures.py          # Publication-quality figure generation
├── regenerate_dashboard.py      # Dashboard regeneration with custom settings
│
│   # Configuration Files
├── pyproject.toml               # Project metadata and tool configuration
├── requirements.txt             # pip dependencies
├── environment.yml              # Conda/Micromamba environment
├── Makefile                     # Task automation
├── .pre-commit-config.yaml      # Code quality hooks
├── .gitignore                   # Git ignore patterns
└── .gitattributes               # Git attributes (line endings)
```

### 8.3 Execution Scripts

| Script | Purpose | Example Usage |
|--------|---------|---------------|
| `run_experiment.py` | Train single method on single dataset | `python run_experiment.py --dataset computer_hardware --method wilcar_constrained` |
| `run_all_methods.py` | Train all 6 methods with comparative analysis | `python run_all_methods.py --dataset computer_hardware` |
| `run_cross_validation.py` | K-fold cross-validation | `python run_cross_validation.py --dataset computer_hardware --folds 5` |
| `generate_figures.py` | Publication-quality figures | `python generate_figures.py --dataset computer_hardware` |
| `regenerate_dashboard.py` | Regenerate dashboards with custom zoom | `python regenerate_dashboard.py --results-dir results/... --r2-ylim 0.75 0.92` |

### 8.4 Reproducibility

All experiments are fully reproducible through:
- Fixed random seeds for all stochastic operations
- Versioned dependencies via `requirements.txt` and `environment.yml`
- Saved model configurations in JSON format
- Serialized results in pickle format for exact reproduction
- Pre-commit hooks ensuring code quality consistency

---

## 9. Visualization Framework

### 9.1 Individual Method Dashboard (8 Panels)

Each method generates a comprehensive dashboard with the following panels:

1. **Mean Squared Error**: Train/Test MSE vs. number of neurons (y-axis clamped to [0,1] if extrapolating)
2. **Root Mean Squared Error**: Train/Test RMSE vs. number of neurons (y-axis clamped to [0,1] if extrapolating)
3. **Coefficient of Determination**: Train/Test R² vs. number of neurons (y-axis clamped to [0,1] if extrapolating)
4. **Signal Conformity Rate Evolution**: SCR progression during training
5. **Overfitting Analysis**: Gap between train and test R² (y-axis clamped to [-1,1] if extrapolating)
6. **Training Time per Model**: Computational cost analysis
7. **Gain Comparison**: Bar chart of calculated gains vs. expected signals with conformity indicators
8. **Executive Summary**: Best model statistics in tabular format

**Dashboard Title Format**: `{Method Name} - Complete Dashboard` with `Dataset: {dataset_name}` subtitle.

### 9.2 Comparative Dashboard (4 Panels)

1. **Test R² vs Network Size**: All methods on single plot with method-specific colors and legend
2. **SCR Adherence (Whole Training)**: Binary indicator (✓/✗) showing if ALL models achieved 100% SCR
3. **Best Test R² by Method**: Bar chart with dynamic y-axis scaling based on actual values
4. **Results Summary Table**: Neurons, R², RMSE, SCR, and Time for best models

### 9.3 Paired Comparison Dashboard

Side-by-side analysis of unconstrained vs. constrained variants within each algorithm family:
- WILCAR (M1) vs WILCAR+R (M2)
- RIXM (M3) vs RIXM+R (M4)
- ELM (M5) vs ELM+R (M6)

### 9.4 Dashboard Regeneration

The `regenerate_dashboard.py` script allows regeneration of all dashboards with custom settings:

```bash
# Regenerate with R² zoom (applied to dashboard + separate file)
python regenerate_dashboard.py --results-dir results/dataset/batch_XXXXXX \
    --r2-ylim 0.75 0.92

# Regenerate with neurons zoom (only in separate zoomed file)
python regenerate_dashboard.py --results-dir results/dataset/batch_XXXXXX \
    --neurons-xlim 1 50
```

This script also regenerates all individual method dashboards with:
- Updated dataset names in titles
- Corrected y-axis limits when metrics extrapolate valid ranges

---

## 10. Concluding Remarks

This technical characterization establishes the complete theoretical and implementation foundation for the neural network framework with gain sign constraints. The six methods provide a systematic exploration of the trade-offs between:

- **Predictive accuracy** (R², MSE, RMSE)
- **Physical consistency** (SCR)
- **Computational efficiency** (training time)
- **Model complexity** (number of neurons)

The constrained methods (M2, M4, M6) achieve the primary objective of guaranteeing 100% SCR while maintaining competitive predictive performance, thereby producing models that are simultaneously accurate and physically interpretable.

---

## References

1. Huang, G.-B., Zhu, Q.-Y., & Siew, C.-K. (2006). Extreme learning machine: Theory and applications. *Neurocomputing*, 70(1-3), 489-501.

2. Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *Proceedings of AISTATS*.

3. Sá, G. A. G., Fontes, C. H., & Embiruçu, M. (2022). A new method for building single feedforward neural network models for multivariate static regression problems: a combined weight initialization and constructive algorithm. *Evolutionary Intelligence*. https://doi.org/10.1007/s12065-022-00813-z

4. Cassotti, M., Ballabio, D., Todeschini, R., & Consonni, V. (2015). A similarity-based QSAR model for predicting acute toxicity towards the fathead minnow (Pimephales promelas). *SAR and QSAR in Environmental Research*, 26, 217-243.

5. Cassotti, M., Ballabio, D., Consonni, V., Mauri, A., Tetko, I. V., & Todeschini, R. (2014). Prediction of acute aquatic toxicity towards daphnia magna using GA-kNN method. *Alternatives to Laboratory Animals (ATLA)*, 42, 31-41.

6. Dua, D., & Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine.