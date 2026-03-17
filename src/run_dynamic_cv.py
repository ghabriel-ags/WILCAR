#!/usr/bin/env python3
"""
Dynamic Cross-Validation (VCD) for Constructive Neural Networks
===============================================================
Federal University of Bahia — MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá

Dynamic Cross-Validation integrates k-fold CV into the constructive process:
for each number of neurons n, performance is evaluated across K folds and the
aggregated error J_T(n) = Σ J_k(n) determines the optimal n*.

After finding n*, a final model is retrained using ALL data.

This approach removes the dependency on a single train/test split for
determining the optimal network size.

Usage:
    # Run M1 (WILCAR Unconstrained) on all datasets
    python run_dynamic_cv.py --methods 1

    # Run M1 and M2 on a specific dataset
    python run_dynamic_cv.py --methods 1 2 --dataset computer_hardware

    # Custom folds and patience
    python run_dynamic_cv.py --methods 1 --n-folds 10 --patience 50

    # Estimate time only
    python run_dynamic_cv.py --methods 1 --estimate-only

Output:
    results/dynamic_cv/<timestamp>/
    ├── <dataset_name>/
    │   ├── method_<id>/
    │   │   ├── vcd_history.csv
    │   │   ├── fold_details.csv
    │   │   ├── final_model_results.json
    │   │   └── vcd_summary.json
    │   └── ...
    ├── consolidated_vcd_results.csv
    └── vcd_config.json
"""

import sys
import argparse
import numpy as np
import pandas as pd
import json
import time
import copy
import os
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# PATH SETUP
# =============================================================================

def setup_paths():
    """Setup Python path to find project modules."""
    script_dir = Path(__file__).parent.resolve()

    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    src_dir = script_dir / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    parent_src = script_dir.parent / "src"
    if parent_src.exists() and str(parent_src) not in sys.path:
        sys.path.insert(0, str(parent_src))

    print(f"📂 Script directory: {script_dir}")

setup_paths()


# =============================================================================
# CONFIGURATION
# =============================================================================

# All datasets (order matches Appendix A)
ALL_DATASETS = [
    'airfoil_self_noise',       # A.1
    'computer_hardware',        # A.2
    'energy_heating',           # A.3a
    'energy_cooling',           # A.3b
    'lavender_friction',        # A.4
    'optical_network',          # A.5
    'aquatic_toxicity',         # A.6
    'fish_toxicity',            # A.7
    'real_estate',              # A.8
    'synchronous_machine',      # A.9
    'yacht_hydrodynamics',      # A.10
]


@dataclass
class VCDMethodConfig:
    """Configuration for a specific method in VCD."""
    max_neurons: int = 750
    patience: int = 150


@dataclass
class VCDConfig:
    """Dynamic Cross-Validation configuration."""
    n_folds: int = 5
    seed: int = 0
    shuffle: bool = True

    # Default constructive parameters
    default_max_neurons: int = 750
    default_patience: int = 150

    # Training parameters (same for all methods)
    patience_early_stopping: int = 250
    learning_rate: float = 0.1
    lr_decay: float = 0.999
    min_lr: float = 1e-3
    iterations: int = 2501
    delta_perturbation: float = 0.1
    tolerance: float = 1e-6

    # Aggregation metric for J_T
    aggregation_metric: str = 'mse_test'

    # Methods to run
    methods: List[int] = field(default_factory=lambda: [1, 2])

    # Datasets to run
    datasets: List[str] = field(default_factory=lambda: ALL_DATASETS.copy())


# =============================================================================
# DATASET-SPECIFIC CONFIGURATIONS
# =============================================================================
# Reuse same configs as static CV for consistency

# Standardized hyperparameters (same for all datasets):
#   M1/M3 (unconstrained, backprop): max=750, patience=150
#   M2/M4 (constrained, SLSQP):     max=50,  patience=15
#   M5 (ELM unconstrained):          max=750, patience=150
#   M6 (ELM constrained):            max=50,  patience=15
def _default_vcd_configs():
    return {
        1: VCDMethodConfig(max_neurons=750, patience=150),   # WILCAR
        2: VCDMethodConfig(max_neurons=50, patience=15),     # WILCAR+R
        3: VCDMethodConfig(max_neurons=750, patience=150),   # RIXM
        4: VCDMethodConfig(max_neurons=50, patience=15),     # RIXM+R
        5: VCDMethodConfig(max_neurons=750, patience=150),   # ELM
        6: VCDMethodConfig(max_neurons=50, patience=15),     # ELM+R
    }

VCD_METHOD_CONFIGS = {ds: _default_vcd_configs() for ds in ALL_DATASETS}


def get_vcd_method_config(dataset: str, method_id: int) -> VCDMethodConfig:
    """Get method configuration for a dataset."""
    if dataset in VCD_METHOD_CONFIGS and method_id in VCD_METHOD_CONFIGS[dataset]:
        return VCD_METHOD_CONFIGS[dataset][method_id]
    return VCDMethodConfig()


# Dataset files: key → CSV filename (without .csv extension)
DATASET_FILES = {
    'computer_hardware': 'computer_hardware',
    'fish_toxicity': 'qsar_fish_toxicity',
    'aquatic_toxicity': 'qsar_aquatic_toxicity',
    'airfoil_self_noise': 'airfoil_self_noise',
    'energy_heating': 'ENB2012_Y1',
    'energy_cooling': 'ENB2012_Y2',
    'lavender_friction': 'lavender_friction_DFC',
    'optical_network': 'optical_interconnection_network',
    'real_estate': 'Real_estate_valuation',
    'synchronous_machine': 'synchronous machine',
    'yacht_hydrodynamics': 'yacht hydrodynamics',
}

EXPECTED_SIGNALS = {
    'computer_hardware': [-1, 1, 1, 1, 1, 1],
    'fish_toxicity': [-1, 1, -1, 1, 1, 1],
    'aquatic_toxicity': [1, -1, 1, 1, 1, -1, 1, 1],
    'airfoil_self_noise': [-1, 0, -1, 1, -1],
    'energy_heating': [1, -1, 1, -1, 1, 0, 1, 0],
    'energy_cooling': [1, -1, 1, -1, 1, 0, 1, 0],
    'lavender_friction': [0, 1, 1],
    'optical_network': [-1, 0, 1],
    'real_estate': [-1, -1, 1],
    'synchronous_machine': [1, -1, 1, 1],
    'yacht_hydrodynamics': [0, 0, 0, 0, 0, 1],
}

METHOD_NAMES = {
    1: 'M1: WILCAR',
    2: 'M2: WILCAR+R',
    3: 'M3: RIXM',
    4: 'M4: RIXM+R',
    5: 'M5: ELM',
    6: 'M6: ELM+R'
}


# =============================================================================
# DATA LOADING AND NORMALIZATION
# =============================================================================

def load_raw_data(dataset_name: str, base_dir: str = ".") -> Tuple[np.ndarray, np.ndarray]:
    """Load raw data from CSV file."""
    filename = DATASET_FILES[dataset_name]
    data_path = Path(base_dir) / "data" / "raw" / f"{filename}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path, header=None, sep=';')
    data = data.dropna(axis=1, how='all')  # Remove empty columns from trailing semicolons
    data = data.dropna()

    inputs = data.iloc[:, :-1].values
    targets = data.iloc[:, -1].values
    return inputs, targets


def normalize_data(X_train: np.ndarray, X_test: np.ndarray,
                   y_train: np.ndarray, y_test: np.ndarray) -> Tuple:
    """Normalize data using MinMaxScaler fitted on training data only."""
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_norm = scaler_X.fit_transform(X_train)
    X_test_norm = scaler_X.transform(X_test)

    y_train_norm = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_norm = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    return X_train_norm, X_test_norm, y_train_norm, y_test_norm


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_training_config(vcd_config: VCDConfig,
                           method_config: VCDMethodConfig) -> 'TrainingConfig':
    """Create TrainingConfig from VCD configuration."""
    from base import TrainingConfig

    return TrainingConfig(
        max_neurons=method_config.max_neurons,
        patience_constructive=method_config.patience,
        patience_early_stopping=vcd_config.patience_early_stopping,
        learning_rate=vcd_config.learning_rate,
        lr_decay=vcd_config.lr_decay,
        min_lr=vcd_config.min_lr,
        iterations=vcd_config.iterations,
        delta_perturbation=vcd_config.delta_perturbation,
        tolerance=vcd_config.tolerance,
        seed=vcd_config.seed,
        use_gpu=False,
        verbose=0
    )


def create_model(method_id: int, **kwargs):
    """Create a model instance for the given method."""
    import io
    import contextlib

    # Import factories
    try:
        from unconstrained import WILCARUnconstrainedNumpy
        from constrained import WILCARConstrainedNumpy
    except ImportError:
        from wilcar.unconstrained import WILCARUnconstrainedNumpy
        from wilcar.constrained import WILCARConstrainedNumpy

    from rixm import RIXMNumpy, RIXMConstrainedNumpy
    from elm import ELMNumpy, ELMConstrainedNumpy

    classes = {
        1: WILCARUnconstrainedNumpy,
        2: WILCARConstrainedNumpy,
        3: RIXMNumpy,
        4: RIXMConstrainedNumpy,
        5: ELMNumpy,
        6: ELMConstrainedNumpy,
    }

    if method_id not in classes:
        raise ValueError(f"Method {method_id} not supported. Available: 1-6")

    # Suppress initialization output
    with contextlib.redirect_stdout(io.StringIO()):
        model = classes[method_id](**kwargs)

    return model


# =============================================================================
# PARALLEL FOLD WORKER
# =============================================================================

def _vcd_fold_worker(fold_idx: int, fold_info: Dict, fold_state: Optional[Dict],
                     n: int, method_id: int, expected_signals: List[int],
                     training_config_dict: Dict, dataset_name: str,
                     src_dir: str) -> Tuple[Dict, Optional[Dict]]:
    """
    Worker function for parallel fold execution.

    Recreates the model, restores state from the previous constructive step,
    executes constructive_step(n), and returns updated parameters.

    Returns:
        Tuple: (result_dict, new_params_dict)
    """
    import warnings
    warnings.filterwarnings('ignore')
    import sys
    import copy
    import io
    import contextlib

    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from base import TrainingConfig

    config = TrainingConfig(**training_config_dict)
    config.seed = training_config_dict['seed'] + fold_idx * 100
    config.verbose = 0

    with contextlib.redirect_stdout(io.StringIO()):
        model = create_model(
            method_id=method_id,
            train_inputs=fold_info['train_inputs'],
            train_targets=fold_info['train_targets'],
            test_inputs=fold_info['test_inputs'],
            test_targets=fold_info['test_targets'],
            expected_signals=expected_signals,
            config=config,
            dataset_name=dataset_name
        )

    if fold_state is not None:
        model.set_params(fold_state)
        # For M2 (constrained): restore _vcd_previous_param for SLSQP weight reuse
        if hasattr(model, '_vcd_previous_param'):
            model._vcd_previous_param = copy.deepcopy(fold_state)

    result = model.constructive_step(
        n=n,
        x_train=fold_info['x_train'],
        y_train=fold_info['y_train'],
        x_test=fold_info['x_test'],
        y_test=fold_info['y_test']
    )

    new_params = model.get_params()
    return result, new_params


# =============================================================================
# VCD RESULTS DATACLASS
# =============================================================================

@dataclass
class VCDResults:
    """Complete results from Dynamic Cross-Validation."""
    method_id: int
    method_name: str
    dataset_name: str
    optimal_neurons: int
    best_J_T: float

    # History per number of neurons
    history: List[Dict] = field(default_factory=list)

    # Per-fold details at optimal n*
    fold_metrics_at_optimal: List[Dict] = field(default_factory=list)
    fold_conformity_details: List[List[Dict]] = field(default_factory=list)

    # Final model (retrained with all data)
    final_n_neurons: int = 0
    final_train_r2: float = 0.0
    final_train_mse: float = 0.0
    final_scr: float = 0.0

    # CV aggregate metrics at n*
    cv_mean_r2_test: float = 0.0
    cv_std_r2_test: float = 0.0
    cv_mean_scr: float = 0.0

    # Timing
    vcd_time: float = 0.0
    retrain_time: float = 0.0
    total_time: float = 0.0

    # Config
    n_folds: int = 5
    max_neurons_evaluated: int = 0


# =============================================================================
# CORE VCD ALGORITHM
# =============================================================================

def run_vcd_for_method(method_id: int, inputs: np.ndarray, targets: np.ndarray,
                       expected_signals: List[int], vcd_config: VCDConfig,
                       method_config: VCDMethodConfig,
                       dataset_name: str, output_dir: Path,
                       parallel: bool = True, skip_retrain: bool = True,
                       base_dir: str = ".") -> VCDResults:
    """
    Execute Dynamic Cross-Validation for a single method on a single dataset.

    Algorithm:
        1. Create K folds and initialize K models
        2. For each n (neurons):
           a. For each fold k: execute constructive_step(n)
           b. Aggregate J_T(n) = Σ mse_test_k(n)
           c. Check patience
        3. Identify n*
        4. Retrain final model with all data using n* neurons
    """
    method_name = METHOD_NAMES[method_id]

    print(f"\n  {'─'*60}")
    print(f"  VCD: {method_name} on {dataset_name}")
    print(f"  max_neurons={method_config.max_neurons}, patience={method_config.patience}")
    print(f"  {'─'*60}")

    # =========================================================
    # STEP 1: Setup K folds and create K models
    # =========================================================
    kfold = KFold(n_splits=vcd_config.n_folds, shuffle=vcd_config.shuffle,
                  random_state=vcd_config.seed)

    fold_models = []
    fold_data = []

    print(f"\n  🔧 Initializing {vcd_config.n_folds} fold models...")

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(inputs)):
        X_train, X_test = inputs[train_idx], inputs[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]

        # Normalize (fit on fold training data only)
        X_train_n, X_test_n, y_train_n, y_test_n = normalize_data(
            X_train, X_test, y_train, y_test
        )

        # Create training config for this fold
        config = create_training_config(vcd_config, method_config)
        config.seed = vcd_config.seed + fold_idx * 100

        # Create model instance
        model = create_model(
            method_id=method_id,
            train_inputs=X_train_n,
            train_targets=y_train_n,
            test_inputs=X_test_n,
            test_targets=y_test_n,
            expected_signals=expected_signals,
            config=config,
            dataset_name=dataset_name
        )

        fold_models.append(model)
        fold_data.append({
            'x_train': X_train_n.T,             # (n_features, n_samples)
            'y_train': y_train_n.reshape(1, -1), # (1, n_samples)
            'x_test': X_test_n.T,
            'y_test': y_test_n.reshape(1, -1),
            'train_inputs': X_train_n,           # (n_samples, n_features) for worker model creation
            'train_targets': y_train_n,          # (n_samples,)
            'test_inputs': X_test_n,             # (n_samples, n_features)
            'test_targets': y_test_n,            # (n_samples,)
            'n_train': len(train_idx),
            'n_test': len(test_idx),
        })

    print(f"  ✅ {vcd_config.n_folds} fold models initialized")

    # =========================================================
    # PARALLEL SETUP
    # =========================================================
    if parallel:
        from joblib import Parallel, delayed
        n_jobs = min(vcd_config.n_folds, os.cpu_count() or 1)
        fold_states = [None] * vcd_config.n_folds
        src_dir = str(Path(__file__).parent.resolve())
        config_template = create_training_config(vcd_config, method_config)
        config_dict = {
            'max_neurons': config_template.max_neurons,
            'patience_constructive': config_template.patience_constructive,
            'patience_early_stopping': config_template.patience_early_stopping,
            'learning_rate': config_template.learning_rate,
            'lr_decay': config_template.lr_decay,
            'min_lr': config_template.min_lr,
            'iterations': config_template.iterations,
            'delta_perturbation': config_template.delta_perturbation,
            'tolerance': config_template.tolerance,
            'seed': vcd_config.seed,
            'use_gpu': False,
            'verbose': 0,
        }
        print(f"  ⚡ Parallel mode: {n_jobs} workers")

    # =========================================================
    # STEP 2: Main VCD loop
    # =========================================================
    best_J_T = float('inf')
    best_n = 0
    patience_counter = 0
    history = []

    vcd_start = time.time()

    print(f"\n  {'n':>5s} | {'J_T':>12s} | {'R² (mean±std)':>18s} | "
          f"{'SCR':>6s} | {'Time':>7s}")
    print(f"  {'─'*5}─┼─{'─'*12}─┼─{'─'*18}─┼─{'─'*6}─┼─{'─'*7}")

    for n in range(1, method_config.max_neurons + 1):
        step_start = time.time()

        J_T = 0.0
        step_metrics = {
            'n': n,
            'fold_mse_test': [],
            'fold_rmse_test': [],
            'fold_r2_test': [],
            'fold_r2_train': [],
            'fold_scr': [],
            'fold_times': [],
            'fold_conformity_details': [],
        }

        all_folds_ok = True

        if parallel:
            # --- Parallel path: run all K folds concurrently ---
            parallel_results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_vcd_fold_worker)(
                    fold_idx=k,
                    fold_info=fold_data[k],
                    fold_state=fold_states[k],
                    n=n,
                    method_id=method_id,
                    expected_signals=expected_signals,
                    training_config_dict=config_dict,
                    dataset_name=dataset_name,
                    src_dir=src_dir,
                )
                for k in range(vcd_config.n_folds)
            )
            for k, (result, new_params) in enumerate(parallel_results):
                fold_states[k] = new_params
                if 'success' in result and not result['success']:
                    all_folds_ok = False
                    break
                J_T += result['mse_test']
                step_metrics['fold_mse_test'].append(result['mse_test'])
                step_metrics['fold_rmse_test'].append(result['rmse_test'])
                step_metrics['fold_r2_test'].append(result['r2_test'])
                step_metrics['fold_r2_train'].append(result['r2_train'])
                step_metrics['fold_scr'].append(result['scr'])
                step_metrics['fold_conformity_details'].append(result.get('conformity_details', []))
                step_metrics['fold_times'].append(result['training_time'])

        else:
            # --- Sequential path (fallback / debug) ---
            for k in range(vcd_config.n_folds):
                result = fold_models[k].constructive_step(
                    n=n,
                    x_train=fold_data[k]['x_train'],
                    y_train=fold_data[k]['y_train'],
                    x_test=fold_data[k]['x_test'],
                    y_test=fold_data[k]['y_test']
                )
                # For M2 (constrained): check if optimization succeeded
                if 'success' in result and not result['success']:
                    all_folds_ok = False
                    break
                J_T += result['mse_test']
                step_metrics['fold_mse_test'].append(result['mse_test'])
                step_metrics['fold_rmse_test'].append(result['rmse_test'])
                step_metrics['fold_r2_test'].append(result['r2_test'])
                step_metrics['fold_r2_train'].append(result['r2_train'])
                step_metrics['fold_scr'].append(result['scr'])
                step_metrics['fold_conformity_details'].append(result.get('conformity_details', []))
                step_metrics['fold_times'].append(result['training_time'])

        step_time = time.time() - step_start

        if not all_folds_ok:
            # Skip this n (for M2 constrained)
            print(f"  {n:5d} | {'SKIPPED':>12s} | {'fold failed':>18s} | "
                  f"{'─':>6s} | {step_time:6.1f}s")
            continue

        # Aggregate metrics
        step_metrics['J_T'] = J_T
        step_metrics['mean_r2_test'] = float(np.mean(step_metrics['fold_r2_test']))
        step_metrics['std_r2_test'] = float(np.std(step_metrics['fold_r2_test']))
        step_metrics['mean_r2_train'] = float(np.mean(step_metrics['fold_r2_train']))
        step_metrics['mean_scr'] = float(np.mean(step_metrics['fold_scr']))
        step_metrics['mean_mse_test'] = float(np.mean(step_metrics['fold_mse_test']))
        step_metrics['time'] = step_time

        history.append(step_metrics)

        # Log
        print(f"  {n:5d} | {J_T:12.6f} | "
              f"{step_metrics['mean_r2_test']:.4f}±{step_metrics['std_r2_test']:.4f}"
              f"{'':>4s} | {step_metrics['mean_scr']:5.1%} | {step_time:6.1f}s"
              f"{'  ★' if J_T < best_J_T else ''}")

        # Stopping criterion
        if J_T < best_J_T:
            best_J_T = J_T
            best_n = n
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= method_config.patience:
            print(f"\n  ⏹️  Early stopping after {method_config.patience} steps "
                  f"without improvement")
            break

    vcd_time = time.time() - vcd_start

    if best_n == 0:
        print(f"\n  ⚠️  No valid configuration found!")
        return VCDResults(
            method_id=method_id, method_name=method_name,
            dataset_name=dataset_name, optimal_neurons=0, best_J_T=float('inf'),
            vcd_time=vcd_time, total_time=vcd_time, n_folds=vcd_config.n_folds
        )

    print(f"\n  🏆 Optimal: n* = {best_n} | J_T* = {best_J_T:.6f}")

    # Get CV metrics at optimal n*
    optimal_step = next(h for h in history if h['n'] == best_n)

    # =========================================================
    # STEP 3: Retrain final model with ALL data (optional)
    # =========================================================
    retrain_time = 0.0
    final_n_neurons = 0
    final_train_r2 = 0.0
    final_train_mse = 0.0
    final_scr = 0.0

    if not skip_retrain:
        print(f"\n  🔄 Retraining final model with n*={best_n} using all data...")

        retrain_start = time.time()

        # Normalize all data
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_all_norm = scaler_X.fit_transform(inputs)
        y_all_norm = scaler_y.fit_transform(targets.reshape(-1, 1)).ravel()

        # Create final model
        final_config = create_training_config(vcd_config, method_config)
        final_config.max_neurons = best_n
        final_config.patience_constructive = best_n + 10  # Don't stop before n*
        final_config.verbose = 0
        final_config.seed = vcd_config.seed

        import io
        import contextlib

        with contextlib.redirect_stdout(io.StringIO()):
            final_model = create_model(
                method_id=method_id,
                train_inputs=X_all_norm,
                train_targets=y_all_norm,
                test_inputs=X_all_norm,     # Same data (no separate test)
                test_targets=y_all_norm,
                expected_signals=expected_signals,
                config=final_config,
                dataset_name=dataset_name
            )
            final_results = final_model.train()

        retrain_time = time.time() - retrain_start
        final_n_neurons = final_results.best_neurons
        final_train_r2 = final_results.best_train_r2
        final_train_mse = final_results.best_train_mse
        final_scr = final_results.best_conformity

        print(f"  ✅ Final model: n={final_n_neurons} | "
              f"R²(train)={final_train_r2:.4f} | "
              f"SCR={final_scr:.1%} | "
              f"Time={retrain_time:.1f}s")
    else:
        print(f"\n  ⏭️  Skipping retrain (use --retrain to enable)")

    # =========================================================
    # STEP 4: Compile results
    # =========================================================
    total_time = vcd_time + retrain_time

    vcd_results = VCDResults(
        method_id=method_id,
        method_name=method_name,
        dataset_name=dataset_name,
        optimal_neurons=best_n,
        best_J_T=best_J_T,
        history=history,
        fold_metrics_at_optimal=[{
            'fold': k,
            'mse_test': optimal_step['fold_mse_test'][k],
            'r2_test': optimal_step['fold_r2_test'][k],
            'r2_train': optimal_step['fold_r2_train'][k],
            'scr': optimal_step['fold_scr'][k],
        } for k in range(vcd_config.n_folds)],
        fold_conformity_details=[
            optimal_step['fold_conformity_details'][k]
            for k in range(vcd_config.n_folds)
        ],
        final_n_neurons=final_n_neurons,
        final_train_r2=final_train_r2,
        final_train_mse=final_train_mse,
        final_scr=final_scr,
        cv_mean_r2_test=optimal_step['mean_r2_test'],
        cv_std_r2_test=optimal_step['std_r2_test'],
        cv_mean_scr=optimal_step['mean_scr'],
        vcd_time=vcd_time,
        retrain_time=retrain_time,
        total_time=total_time,
        n_folds=vcd_config.n_folds,
        max_neurons_evaluated=history[-1]['n'] if history else 0,
    )

    # Save results
    _save_vcd_results(vcd_results, output_dir, dataset_name, method_id)

    return vcd_results


# =============================================================================
# RESULT SAVING
# =============================================================================

def _save_vcd_results(results: VCDResults, output_dir: Path,
                      dataset_name: str, method_id: int):
    """Save VCD results to disk."""
    method_dir = output_dir / dataset_name / f"method_{method_id}"
    method_dir.mkdir(parents=True, exist_ok=True)

    # VCD History (J_T curve)
    history_rows = []
    for h in results.history:
        history_rows.append({
            'n_neurons': h['n'],
            'J_T': h['J_T'],
            'mean_mse_test': h['mean_mse_test'],
            'mean_r2_test': h['mean_r2_test'],
            'std_r2_test': h['std_r2_test'],
            'mean_r2_train': h['mean_r2_train'],
            'mean_scr': h['mean_scr'],
            'time': h['time'],
        })

    pd.DataFrame(history_rows).to_csv(
        method_dir / "vcd_history.csv", index=False
    )

    # Fold details at optimal
    pd.DataFrame(results.fold_metrics_at_optimal).to_csv(
        method_dir / "fold_details_at_optimal.csv", index=False
    )

    # Conformity details at optimal (per variable, per fold)
    if results.fold_conformity_details:
        conf_rows = []
        for k, fold_details in enumerate(results.fold_conformity_details):
            for var_detail in fold_details:
                conf_rows.append({
                    'fold': k,
                    'variable': var_detail['variable'],
                    'calculated_gain': var_detail['calculated_gain'],
                    'calculated_signal': var_detail['calculated_signal'],
                    'expected_signal': var_detail['expected_signal'],
                    'conformity': var_detail['conformity'],
                })
        if conf_rows:
            pd.DataFrame(conf_rows).to_csv(
                method_dir / "conformity_details_at_optimal.csv", index=False
            )

    # Summary JSON
    summary = {
        'method_id': results.method_id,
        'method_name': results.method_name,
        'dataset_name': results.dataset_name,
        'optimal_neurons': results.optimal_neurons,
        'best_J_T': results.best_J_T,
        'cv_mean_r2_test': results.cv_mean_r2_test,
        'cv_std_r2_test': results.cv_std_r2_test,
        'cv_mean_scr': results.cv_mean_scr,
        'final_n_neurons': results.final_n_neurons,
        'final_train_r2': results.final_train_r2,
        'final_train_mse': results.final_train_mse,
        'final_scr': results.final_scr,
        'vcd_time': results.vcd_time,
        'retrain_time': results.retrain_time,
        'total_time': results.total_time,
        'n_folds': results.n_folds,
        'max_neurons_evaluated': results.max_neurons_evaluated,
    }

    with open(method_dir / "vcd_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  📁 Results saved to: {method_dir}")


# =============================================================================
# CONSOLIDATED RESULTS
# =============================================================================

def generate_consolidated_results(all_results: Dict[str, Dict[int, VCDResults]],
                                  output_dir: Path):
    """Generate consolidated results table across all datasets and methods."""
    rows = []
    for dataset_name, method_results in all_results.items():
        for method_id, results in method_results.items():
            rows.append({
                'dataset': dataset_name,
                'method_id': results.method_id,
                'method_name': results.method_name,
                'n*': results.optimal_neurons,
                'J_T*': results.best_J_T,
                'cv_R2_mean': results.cv_mean_r2_test,
                'cv_R2_std': results.cv_std_r2_test,
                'cv_SCR': results.cv_mean_scr,
                'final_n': results.final_n_neurons,
                'final_R2_train': results.final_train_r2,
                'final_SCR': results.final_scr,
                'vcd_time_min': results.vcd_time / 60,
                'total_time_min': results.total_time / 60,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "consolidated_vcd_results.csv", index=False)

    print(f"\n  ✅ Consolidated results: {output_dir / 'consolidated_vcd_results.csv'}")
    return df


# =============================================================================
# TIME ESTIMATION
# =============================================================================

# Rough estimates: seconds per constructive step (1 neuron) per fold
# Based on existing CV timings divided by avg neurons found
# Calibrated from actual VCD runs (seconds per step, wall-clock with parallel folds)
# M1/M3: backprop ~1-10s/step depending on dataset size
# M2/M4/M6: SLSQP 10-300s/step on large datasets, 2-30s on small
# M5: pseudo-inverse, near-instant
ESTIMATED_TIME_PER_STEP = {
    'airfoil_self_noise':   {1: 10.0, 2: 120.0, 3: 14.0, 4: 150.0, 5: 0.05, 6: 100.0},
    'fish_toxicity':        {1: 8.0,  2: 80.0,  3: 10.0, 4: 100.0, 5: 0.03, 6: 60.0},
    'energy_heating':       {1: 6.0,  2: 60.0,  3: 8.0,  4: 80.0,  5: 0.03, 6: 50.0},
    'energy_cooling':       {1: 6.0,  2: 60.0,  3: 8.0,  4: 80.0,  5: 0.03, 6: 50.0},
    'aquatic_toxicity':     {1: 4.0,  2: 40.0,  3: 5.0,  4: 50.0,  5: 0.02, 6: 30.0},
    'synchronous_machine':  {1: 3.0,  2: 20.0,  3: 4.0,  4: 25.0,  5: 0.02, 6: 15.0},
    'lavender_friction':    {1: 2.0,  2: 10.0,  3: 3.0,  4: 15.0,  5: 0.01, 6: 8.0},
    'optical_network':      {1: 2.0,  2: 10.0,  3: 3.0,  4: 15.0,  5: 0.01, 6: 8.0},
    'computer_hardware':    {1: 1.0,  2: 5.0,   3: 1.5,  4: 8.0,   5: 0.01, 6: 3.0},
    'real_estate':          {1: 1.0,  2: 5.0,   3: 1.5,  4: 8.0,   5: 0.01, 6: 3.0},
    'yacht_hydrodynamics':  {1: 0.8,  2: 5.0,   3: 1.0,  4: 8.0,   5: 0.01, 6: 3.0},
}


def estimate_vcd_time(vcd_config: VCDConfig) -> float:
    """Estimate total VCD time in seconds."""
    total = 0
    for dataset in vcd_config.datasets:
        for method_id in vcd_config.methods:
            mc = get_vcd_method_config(dataset, method_id)
            time_per_step = ESTIMATED_TIME_PER_STEP.get(dataset, {}).get(method_id, 2.0)
            # Each step runs K folds
            dataset_method_time = mc.max_neurons * time_per_step
            total += dataset_method_time
    return total


def print_time_estimate(vcd_config: VCDConfig):
    """Print formatted time estimate."""
    print("\n" + "=" * 70)
    print("TIME ESTIMATION (VCD)")
    print("=" * 70)

    total = 0
    for dataset in vcd_config.datasets:
        print(f"\n  {dataset.upper()}:")
        for method_id in vcd_config.methods:
            mc = get_vcd_method_config(dataset, method_id)
            time_per_step = ESTIMATED_TIME_PER_STEP.get(dataset, {}).get(method_id, 2.0)
            est_time = mc.max_neurons * time_per_step
            total += est_time
            print(f"    {METHOD_NAMES[method_id]:15s} "
                  f"(n≤{mc.max_neurons}, p={mc.patience}) "
                  f"~{est_time/60:.0f} min")

    print(f"\n  {'─'*60}")
    print(f"  TOTAL: ~{total/3600:.1f} hours ({total/60:.0f} minutes)")
    print(f"  {'─'*60}")

    return total


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dynamic Cross-Validation (VCD) for Constructive Neural Networks"
    )
    parser.add_argument('--dataset', type=str, default=None,
                        choices=ALL_DATASETS,
                        help="Run specific dataset (default: all)")
    parser.add_argument('--methods', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                        help="Methods to run (1=WILCAR, 2=WILCAR+R, 3=RIXM, 4=RIXM+R, 5=ELM, 6=ELM+R)")
    parser.add_argument('--n-folds', type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('--base-dir', type=str, default=".",
                        help="Base directory for data and results")
    parser.add_argument('--estimate-only', action='store_true',
                        help="Only estimate time, don't run")
    parser.add_argument('--no-parallel', action='store_true', default=False,
                        help="Run folds sequentially instead of in parallel (for debugging)")
    parser.add_argument('--retrain', action='store_true', default=False,
                        help="Retrain final model with all data using n* (slower)")
    parser.add_argument('-y', '--yes', action='store_true', default=False,
                        help="Skip confirmation prompt for long runs")

    args = parser.parse_args()

    # Validate methods
    invalid = [m for m in args.methods if m not in [1, 2, 3, 4, 5, 6]]
    for m in invalid:
        print(f"⚠️  Method {m} not valid. Available: 1-6")
    args.methods = [m for m in args.methods if m in [1, 2, 3, 4, 5, 6]]

    if not args.methods:
        print("❌ No valid methods selected. Exiting.")
        return

    # Setup configuration
    vcd_config = VCDConfig(
        n_folds=args.n_folds,
        seed=args.seed,
        methods=args.methods,
    )

    if args.dataset:
        vcd_config.datasets = [args.dataset]

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.base_dir) / "results" / "dynamic_cv" / f"vcd_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Header
    print("\n" + "=" * 70)
    print("DYNAMIC CROSS-VALIDATION (VCD)")
    print("=" * 70)
    print(f"  Datasets: {vcd_config.datasets}")
    print(f"  Methods:  {[METHOD_NAMES[m] for m in vcd_config.methods]}")
    print(f"  Folds:    {vcd_config.n_folds}")
    print(f"  Seed:     {vcd_config.seed}")
    print(f"  Parallel: {'yes (' + str(min(vcd_config.n_folds, os.cpu_count() or 1)) + ' workers)' if not args.no_parallel else 'no (sequential)'}")
    print(f"  Output:   {output_dir}")
    print("=" * 70)

    # Time estimation
    total_est = print_time_estimate(vcd_config)

    if args.estimate_only:
        print("\n✅ Estimate only mode. Exiting.")
        return

    # Confirm for long runs
    if total_est > 3600 and not args.yes:
        print(f"\n⚠️  Estimated time: {total_est/3600:.1f} hours.")
        try:
            response = input("Continue? [y/N]: ").strip().lower()
        except EOFError:
            response = 'y'  # Auto-continue when stdin is not available (piped)
        if response != 'y':
            print("Aborted.")
            return

    # Save config
    with open(output_dir / "vcd_config.json", 'w') as f:
        json.dump(asdict(vcd_config), f, indent=2)

    # Run VCD
    all_results = {}
    total_start = time.time()

    for dataset_name in vcd_config.datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*70}")

        # Load data
        print(f"\n  📊 Loading data...")
        inputs, targets = load_raw_data(dataset_name, args.base_dir)
        expected_signals = EXPECTED_SIGNALS[dataset_name]
        print(f"  Samples: {inputs.shape[0]} | Features: {inputs.shape[1]}")
        print(f"  Expected signals: {expected_signals}")

        dataset_results = {}

        for method_id in vcd_config.methods:
            try:
                method_config = get_vcd_method_config(dataset_name, method_id)

                vcd_results = run_vcd_for_method(
                    method_id=method_id,
                    inputs=inputs,
                    targets=targets,
                    expected_signals=expected_signals,
                    vcd_config=vcd_config,
                    method_config=method_config,
                    dataset_name=dataset_name,
                    output_dir=output_dir,
                    parallel=not args.no_parallel,
                    skip_retrain=not args.retrain,
                    base_dir=args.base_dir
                )

                dataset_results[method_id] = vcd_results

            except Exception as e:
                print(f"\n  ❌ Error: {method_id} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()

        all_results[dataset_name] = dataset_results

    total_time = time.time() - total_start

    # Generate consolidated results
    print("\n" + "=" * 70)
    print("GENERATING OUTPUTS")
    print("=" * 70)

    if any(all_results.values()):
        consolidated = generate_consolidated_results(all_results, output_dir)

    # Final summary
    print("\n" + "=" * 70)
    print("VCD COMPLETE")
    print("=" * 70)
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Results: {output_dir}")

    # Summary table
    print(f"\n  {'─'*60}")
    print(f"  {'Dataset':<20s} | {'Method':<15s} | {'n*':>4s} | "
          f"{'R² (CV)':>12s} | {'SCR':>5s}")
    print(f"  {'─'*20}─┼─{'─'*15}─┼─{'─'*4}─┼─{'─'*12}─┼─{'─'*5}")

    for dataset_name, method_results in all_results.items():
        for method_id, res in method_results.items():
            print(f"  {dataset_name:<20s} | {res.method_name:<15s} | "
                  f"{res.optimal_neurons:4d} | "
                  f"{res.cv_mean_r2_test:.4f}±{res.cv_std_r2_test:.4f} | "
                  f"{res.cv_mean_scr:4.0%}")

    print(f"  {'─'*60}")


if __name__ == "__main__":
    main()
