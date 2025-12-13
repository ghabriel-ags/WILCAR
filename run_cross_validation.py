#!/usr/bin/env python3
"""
5-Fold Cross-Validation for All Methods
========================================
Federal University of Bahia – MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá

This script performs 5-fold cross-validation for all 6 neural network methods
across all 3 datasets, generating comprehensive results with mean ± std.

Usage:
    # Run all datasets
    python run_cross_validation.py
    
    # Run specific dataset
    python run_cross_validation.py --dataset computer_hardware
    
    # Run specific methods
    python run_cross_validation.py --methods 1 2 5 6
    
    # Custom number of folds
    python run_cross_validation.py --n-folds 10

Output:
    results/cross_validation/<timestamp>/
    ├── computer_hardware/
    │   ├── fold_results.csv          # Results per fold
    │   ├── summary_stats.csv         # Mean ± std per method
    │   └── detailed_results.json     # Complete results
    ├── fish_toxicity/
    │   └── ...
    ├── aquatic_toxicity/
    │   └── ...
    ├── consolidated_results.csv      # All datasets summary
    ├── latex_tables.tex              # Ready for manuscript
    └── cv_config.json
"""

import sys
import argparse
import numpy as np
import pandas as pd
import json
import time
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
    """
    Setup Python path to find project modules.
    Handles both development (running from src/) and installed scenarios.
    """
    script_dir = Path(__file__).parent.resolve()
    
    # Add script directory
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    # Add src directory if it exists
    src_dir = script_dir / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    # Add parent/src if running from scripts/
    parent_src = script_dir.parent / "src"
    if parent_src.exists() and str(parent_src) not in sys.path:
        sys.path.insert(0, str(parent_src))
    
    print(f"📁 Script directory: {script_dir}")
    print(f"📁 Python path includes: {sys.path[:3]}...")

setup_paths()

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MethodConfig:
    """Configuration for a specific method."""
    max_neurons: int = 750
    patience: int = 150


@dataclass 
class CVConfig:
    """Cross-validation configuration."""
    n_folds: int = 5
    seed: int = 0
    shuffle: bool = True
    
    # Default parameters (used when not specified per method)
    default_max_neurons: int = 750
    default_patience: int = 150
    
    # Other training parameters (same for all methods)
    patience_early_stopping: int = 250
    learning_rate: float = 0.1
    lr_decay: float = 0.999
    min_lr: float = 1e-3
    iterations: int = 2501
    delta_perturbation: float = 0.1
    tolerance: float = 1e-6
    
    # Methods to run (1-6)
    methods: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    
    # Datasets to run
    datasets: List[str] = field(default_factory=lambda: [
        'computer_hardware', 
        'fish_toxicity', 
        'aquatic_toxicity'
    ])


# =============================================================================
# DATASET-SPECIFIC METHOD CONFIGURATIONS
# =============================================================================
# These match the parameters used in the single-run experiments

METHOD_CONFIGS = {
    'computer_hardware': {
        1: MethodConfig(max_neurons=750, patience=150),   # WILCAR - default
        2: MethodConfig(max_neurons=200, patience=20),    # WILCAR+R
        3: MethodConfig(max_neurons=750, patience=150),   # RIXM - default
        4: MethodConfig(max_neurons=200, patience=20),    # RIXM+R
        5: MethodConfig(max_neurons=750, patience=150),   # ELM - default
        6: MethodConfig(max_neurons=150, patience=15),    # ELM+R
    },
    'fish_toxicity': {
        1: MethodConfig(max_neurons=750, patience=150),   # WILCAR - default
        2: MethodConfig(max_neurons=200, patience=20),    # WILCAR+R
        3: MethodConfig(max_neurons=750, patience=150),   # RIXM - default
        4: MethodConfig(max_neurons=200, patience=20),    # RIXM+R
        5: MethodConfig(max_neurons=750, patience=150),   # ELM - default
        6: MethodConfig(max_neurons=150, patience=15),    # ELM+R - REDUCED (was 750/150)
    },
    'aquatic_toxicity': {
        1: MethodConfig(max_neurons=750, patience=150),   # WILCAR - default
        2: MethodConfig(max_neurons=200, patience=20),    # WILCAR+R
        3: MethodConfig(max_neurons=750, patience=150),   # RIXM - default
        4: MethodConfig(max_neurons=200, patience=20),    # RIXM+R
        5: MethodConfig(max_neurons=750, patience=150),   # ELM - default
        6: MethodConfig(max_neurons=150, patience=15),    # ELM+R
    }
}


def get_method_config(dataset: str, method_id: int) -> MethodConfig:
    """Get the configuration for a specific dataset/method combination."""
    if dataset in METHOD_CONFIGS and method_id in METHOD_CONFIGS[dataset]:
        return METHOD_CONFIGS[dataset][method_id]
    # Fallback to default
    return MethodConfig()
    
    # Datasets to run
    datasets: List[str] = field(default_factory=lambda: [
        'computer_hardware', 
        'fish_toxicity', 
        'aquatic_toxicity'
    ])


# Dataset configurations
DATASET_FILES = {
    'computer_hardware': 'computer_hardware',
    'fish_toxicity': 'qsar_fish_toxicity',
    'aquatic_toxicity': 'qsar_aquatic_toxicity'
}

EXPECTED_SIGNALS = {
    'computer_hardware': [-1, 1, 1, 1, 1, 1],
    'fish_toxicity': [-1, 1, -1, 1, 1, 1],
    'aquatic_toxicity': [1, -1, 1, 1, 1, -1, 1, 1]
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
# DATA LOADING
# =============================================================================

def load_raw_data(dataset_name: str, base_dir: str = ".") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load raw data from CSV file.
    
    Returns:
        Tuple: (inputs, targets) as numpy arrays
    """
    filename = DATASET_FILES[dataset_name]
    data_path = Path(base_dir) / "data" / "raw" / f"{filename}.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    data = pd.read_csv(data_path, header=None, sep=';')
    data = data.dropna()
    
    inputs = data.iloc[:, :-1].values
    targets = data.iloc[:, -1].values
    
    return inputs, targets


def normalize_data(X_train: np.ndarray, X_test: np.ndarray, 
                   y_train: np.ndarray, y_test: np.ndarray) -> Tuple:
    """
    Normalize data using MinMaxScaler fitted on training data only.
    """
    # Combine for scaling
    n_features = X_train.shape[1]
    
    # Fit scaler on training data only
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_norm = scaler_X.fit_transform(X_train)
    X_test_norm = scaler_X.transform(X_test)
    
    y_train_norm = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_norm = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    return X_train_norm, X_test_norm, y_train_norm, y_test_norm


# =============================================================================
# METHOD FACTORY
# =============================================================================

def create_training_config(cv_config: CVConfig, method_config: MethodConfig) -> 'TrainingConfig':
    """Create TrainingConfig from CVConfig and MethodConfig."""
    from base import TrainingConfig
    
    return TrainingConfig(
        max_neurons=method_config.max_neurons,
        patience_constructive=method_config.patience,
        patience_early_stopping=cv_config.patience_early_stopping,
        learning_rate=cv_config.learning_rate,
        lr_decay=cv_config.lr_decay,
        min_lr=cv_config.min_lr,
        iterations=cv_config.iterations,
        delta_perturbation=cv_config.delta_perturbation,
        tolerance=cv_config.tolerance,
        seed=cv_config.seed,
        use_gpu=False,
        verbose=0  # Minimal output during CV
    )


def get_method_factories():
    """
    Import method factory functions.
    Returns a dict mapping method_id to factory function.
    """
    try:
        # Import factory functions (preferred approach)
        from wilcar.unconstrained import create_wilcar_unconstrained
        from wilcar.constrained import create_wilcar_constrained
        from rixm import create_rixm, create_rixm_constrained
        from elm import create_elm, create_elm_constrained
        
        return {
            1: create_wilcar_unconstrained,
            2: create_wilcar_constrained,
            3: create_rixm,
            4: create_rixm_constrained,
            5: create_elm,
            6: create_elm_constrained
        }
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
        print("   Trying alternative import pattern...")
        
        try:
            # Alternative: direct class imports
            from unconstrained import WILCARUnconstrainedNumpy
            from constrained import WILCARConstrainedNumpy
            from rixm import RIXMNumpy
            from rixm_constrained import RIXMConstrainedNumpy
            from elm import ELMNumpy
            from elm_constrained import ELMConstrainedNumpy
            
            # Wrap classes as factory functions
            def make_factory(cls):
                def factory(**kwargs):
                    return cls(**kwargs)
                return factory
            
            return {
                1: make_factory(WILCARUnconstrainedNumpy),
                2: make_factory(WILCARConstrainedNumpy),
                3: make_factory(RIXMNumpy),
                4: make_factory(RIXMConstrainedNumpy),
                5: make_factory(ELMNumpy),
                6: make_factory(ELMConstrainedNumpy)
            }
        except ImportError as e2:
            raise ImportError(
                f"Could not import method modules.\n"
                f"Primary error: {e}\n"
                f"Secondary error: {e2}\n"
                f"Make sure the script is in the src/ directory with all method files."
            )


def run_method(method_id: int, 
               train_inputs: np.ndarray, train_targets: np.ndarray,
               test_inputs: np.ndarray, test_targets: np.ndarray,
               expected_signals: List[int],
               config: 'TrainingConfig',
               dataset_name: str) -> Dict[str, Any]:
    """
    Run a single method and return results.
    """
    # Get method factories (cached after first call)
    if not hasattr(run_method, '_factories'):
        run_method._factories = get_method_factories()
    
    factory = run_method._factories[method_id]
    
    # Redirect stdout to suppress verbose output
    import io
    import contextlib
    
    with contextlib.redirect_stdout(io.StringIO()):
        model = factory(
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            expected_signals=expected_signals,
            config=config,
            dataset_name=dataset_name
        )
        
        start_time = time.time()
        results = model.train()
        training_time = time.time() - start_time
    
    # Extract metrics
    return {
        'method_id': method_id,
        'method_name': METHOD_NAMES[method_id],
        'best_neurons': results.best_neurons,
        'test_r2': results.best_test_r2,
        'test_mse': results.best_test_mse,
        'test_rmse': results.best_test_rmse,
        'train_r2': results.best_train_r2,
        'train_mse': results.best_train_mse,
        'scr': results.best_conformity,
        'training_time': training_time,
        'overfitting': results.best_train_r2 - results.best_test_r2
    }


# =============================================================================
# CROSS-VALIDATION EXECUTION
# =============================================================================

def run_cv_for_dataset(dataset_name: str, cv_config: CVConfig, 
                       output_dir: Path, base_dir: str = ".") -> pd.DataFrame:
    """
    Run complete cross-validation for a single dataset.
    """
    print(f"\n{'='*70}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Load data
    print(f"\n📊 Loading data...")
    inputs, targets = load_raw_data(dataset_name, base_dir)
    expected_signals = EXPECTED_SIGNALS[dataset_name]
    
    print(f"   Samples: {inputs.shape[0]} | Features: {inputs.shape[1]}")
    print(f"   Expected signals: {expected_signals}")
    
    # Setup K-Fold
    kfold = KFold(n_splits=cv_config.n_folds, shuffle=cv_config.shuffle, 
                  random_state=cv_config.seed)
    
    # Results storage
    all_results = []
    
    # Create output directory for this dataset
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Print method configurations for this dataset
    print(f"\n   Method configurations:")
    for method_id in cv_config.methods:
        mc = get_method_config(dataset_name, method_id)
        print(f"   {METHOD_NAMES[method_id]}: max_neurons={mc.max_neurons}, patience={mc.patience}")
    
    # Run CV
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(inputs)):
        print(f"\n{'─'*50}")
        print(f"FOLD {fold_idx + 1}/{cv_config.n_folds}")
        print(f"{'─'*50}")
        print(f"   Train: {len(train_idx)} samples | Test: {len(test_idx)} samples")
        
        # Split data
        X_train, X_test = inputs[train_idx], inputs[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]
        
        # Normalize (fit on train only)
        X_train_norm, X_test_norm, y_train_norm, y_test_norm = normalize_data(
            X_train, X_test, y_train, y_test
        )
        
        # Run each method
        for method_id in cv_config.methods:
            # Get method-specific configuration
            method_config = get_method_config(dataset_name, method_id)
            
            print(f"\n   🔄 Running {METHOD_NAMES[method_id]} "
                  f"(n≤{method_config.max_neurons}, p={method_config.patience})...", 
                  end=" ", flush=True)
            
            try:
                # Create training config with method-specific parameters
                training_config = create_training_config(cv_config, method_config)
                
                # Update seed for reproducibility within fold
                training_config.seed = cv_config.seed + fold_idx * 100 + method_id
                
                result = run_method(
                    method_id=method_id,
                    train_inputs=X_train_norm,
                    train_targets=y_train_norm,
                    test_inputs=X_test_norm,
                    test_targets=y_test_norm,
                    expected_signals=expected_signals,
                    config=training_config,
                    dataset_name=dataset_name
                )
                
                # Add fold info and config used
                result['fold'] = fold_idx + 1
                result['dataset'] = dataset_name
                result['max_neurons_config'] = method_config.max_neurons
                result['patience_config'] = method_config.patience
                
                all_results.append(result)
                
                print(f"✓ R²={result['test_r2']:.4f} | SCR={result['scr']:.0%} | "
                      f"n={result['best_neurons']} | t={result['training_time']:.1f}s")
                
                # Incremental save after each method (in case of crash)
                df_partial = pd.DataFrame(all_results)
                df_partial.to_csv(dataset_dir / "fold_results_partial.csv", index=False)
                
            except Exception as e:
                print(f"✗ Error: {str(e)[:50]}")
                import traceback
                traceback.print_exc()
                
                # Add error result
                all_results.append({
                    'method_id': method_id,
                    'method_name': METHOD_NAMES[method_id],
                    'fold': fold_idx + 1,
                    'dataset': dataset_name,
                    'error': str(e),
                    'test_r2': np.nan,
                    'test_mse': np.nan,
                    'test_rmse': np.nan,
                    'train_r2': np.nan,
                    'scr': np.nan,
                    'best_neurons': np.nan,
                    'training_time': np.nan,
                    'overfitting': np.nan
                })
    
    # Create DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Save fold results
    df_results.to_csv(dataset_dir / "fold_results.csv", index=False)
    print(f"\n💾 Fold results saved: {dataset_dir / 'fold_results.csv'}")
    
    # Calculate summary statistics
    summary = calculate_summary_stats(df_results)
    summary.to_csv(dataset_dir / "summary_stats.csv", index=False)
    print(f"💾 Summary saved: {dataset_dir / 'summary_stats.csv'}")
    
    # Save detailed JSON
    with open(dataset_dir / "detailed_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    return df_results


def calculate_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean ± std for each method.
    """
    metrics = ['test_r2', 'test_rmse', 'test_mse', 'train_r2', 'scr', 
               'best_neurons', 'training_time', 'overfitting']
    
    summary_rows = []
    
    for method_id in df['method_id'].unique():
        method_df = df[df['method_id'] == method_id]
        method_name = METHOD_NAMES[int(method_id)]
        
        row = {
            'method_id': int(method_id),
            'method_name': method_name
        }
        
        for metric in metrics:
            if metric in method_df.columns:
                values = method_df[metric].dropna()
                if len(values) > 0:
                    row[f'{metric}_mean'] = values.mean()
                    row[f'{metric}_std'] = values.std()
                    row[f'{metric}_formatted'] = f"{values.mean():.4f} ± {values.std():.4f}"
                else:
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_std'] = np.nan
                    row[f'{metric}_formatted'] = "N/A"
        
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def generate_latex_tables(all_summaries: Dict[str, pd.DataFrame], 
                          output_dir: Path) -> str:
    """
    Generate LaTeX tables ready for manuscript.
    """
    latex = []
    latex.append("% Auto-generated LaTeX tables for cross-validation results")
    latex.append(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    latex.append("")
    
    for dataset_name, summary in all_summaries.items():
        latex.append(f"% Dataset: {dataset_name}")
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(f"\\caption{{5-Fold Cross-Validation Results for {dataset_name.replace('_', ' ').title()}}}")
        latex.append(f"\\label{{tab:cv_{dataset_name}}}")
        latex.append(r"\begin{tabular}{lcccccc}")
        latex.append(r"\toprule")
        latex.append(r"Method & Neurons & Test R² & Test RMSE & SCR & Time (s) & Overfitting \\")
        latex.append(r"\midrule")
        
        for _, row in summary.iterrows():
            neurons = f"{row['best_neurons_mean']:.1f} ± {row['best_neurons_std']:.1f}"
            r2 = f"{row['test_r2_mean']:.4f} ± {row['test_r2_std']:.4f}"
            rmse = f"{row['test_rmse_mean']:.4f} ± {row['test_rmse_std']:.4f}"
            scr = f"{row['scr_mean']*100:.1f}\\% ± {row['scr_std']*100:.1f}\\%"
            time_str = f"{row['training_time_mean']:.1f} ± {row['training_time_std']:.1f}"
            overfit = f"{row['overfitting_mean']:.4f} ± {row['overfitting_std']:.4f}"
            
            latex.append(f"{row['method_name']} & {neurons} & {r2} & {rmse} & {scr} & {time_str} & {overfit} \\\\")
        
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        latex.append("")
    
    latex_content = "\n".join(latex)
    
    with open(output_dir / "latex_tables.tex", 'w') as f:
        f.write(latex_content)
    
    return latex_content


def generate_consolidated_table(all_summaries: Dict[str, pd.DataFrame],
                                 output_dir: Path) -> pd.DataFrame:
    """
    Generate consolidated results table across all datasets.
    """
    rows = []
    
    for dataset_name, summary in all_summaries.items():
        for _, row in summary.iterrows():
            rows.append({
                'Dataset': dataset_name,
                'Method': row['method_name'],
                'Neurons': f"{row['best_neurons_mean']:.1f} ± {row['best_neurons_std']:.1f}",
                'Test R²': f"{row['test_r2_mean']:.4f} ± {row['test_r2_std']:.4f}",
                'Test RMSE': f"{row['test_rmse_mean']:.4f} ± {row['test_rmse_std']:.4f}",
                'SCR': f"{row['scr_mean']*100:.1f}% ± {row['scr_std']*100:.1f}%",
                'Time (s)': f"{row['training_time_mean']:.1f} ± {row['training_time_std']:.1f}",
                'Overfitting': f"{row['overfitting_mean']:.4f} ± {row['overfitting_std']:.4f}",
                # Raw values for sorting
                'test_r2_mean': row['test_r2_mean'],
                'scr_mean': row['scr_mean']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "consolidated_results.csv", index=False)
    
    return df


# =============================================================================
# TIME ESTIMATION
# =============================================================================

# Approximate training times (seconds) based on previous runs WITH SPECIFIC CONFIGS
# These times already account for the reduced max_neurons/patience for constrained methods
ESTIMATED_TIMES = {
    'computer_hardware': {
        1: 260,    # WILCAR (750, 150)
        2: 8,      # WILCAR+R (200, 20) - fast due to reduced params
        3: 1530,   # RIXM (750, 150)
        4: 115,    # RIXM+R (200, 20)
        5: 20,     # ELM (750, 150)
        6: 3       # ELM+R (150, 15) - very fast
    },
    'fish_toxicity': {
        1: 142,    # WILCAR (750, 150)
        2: 360,    # WILCAR+R (200, 20)
        3: 5670,   # RIXM (750, 150) - very slow
        4: 491,    # RIXM+R (200, 20)
        5: 31,     # ELM (750, 150)
        6: 50      # ELM+R (150, 15) - REDUCED params, much faster now
    },
    'aquatic_toxicity': {
        1: 78,     # WILCAR (750, 150)
        2: 426,    # WILCAR+R (200, 20)
        3: 1809,   # RIXM (750, 150)
        4: 1408,   # RIXM+R (200, 20)
        5: 2,      # ELM (750, 150)
        6: 30      # ELM+R (150, 15) - fast due to reduced params
    }
}


def estimate_total_time(cv_config: CVConfig) -> Tuple[float, Dict]:
    """
    Estimate total cross-validation time.
    
    Returns:
        Tuple: (total_seconds, breakdown_dict)
    """
    breakdown = {}
    total = 0
    
    for dataset in cv_config.datasets:
        dataset_time = 0
        dataset_breakdown = {}
        
        for method_id in cv_config.methods:
            method_time = ESTIMATED_TIMES.get(dataset, {}).get(method_id, 300)
            fold_time = method_time * cv_config.n_folds
            dataset_breakdown[method_id] = fold_time
            dataset_time += fold_time
        
        breakdown[dataset] = {
            'total': dataset_time,
            'methods': dataset_breakdown
        }
        total += dataset_time
    
    return total, breakdown


def print_time_estimate(cv_config: CVConfig):
    """Print formatted time estimate."""
    total_seconds, breakdown = estimate_total_time(cv_config)
    
    print("\n" + "=" * 70)
    print("TIME ESTIMATION")
    print("=" * 70)
    
    for dataset, info in breakdown.items():
        hours = info['total'] / 3600
        print(f"\n{dataset.upper()}: ~{hours:.1f} hours")
        for method_id in cv_config.methods:
            if method_id in info['methods']:
                mc = get_method_config(dataset, method_id)
                seconds = info['methods'][method_id]
                mins = seconds / 60
                config_str = f"(n≤{mc.max_neurons}, p={mc.patience})"
                print(f"   {METHOD_NAMES[method_id]:15s} {config_str:18s} ~{mins:.1f} min")
    
    total_hours = total_seconds / 3600
    print(f"\n{'─'*70}")
    print(f"TOTAL ESTIMATED TIME: ~{total_hours:.1f} hours ({total_seconds/60:.0f} minutes)")
    print(f"{'─'*70}")
    
    # Warnings for slow methods
    slow_methods = []
    for dataset in cv_config.datasets:
        for method_id in cv_config.methods:
            time_per_fold = ESTIMATED_TIMES.get(dataset, {}).get(method_id, 0)
            if time_per_fold > 1000:  # > 16 minutes per fold
                mc = get_method_config(dataset, method_id)
                slow_methods.append((dataset, METHOD_NAMES[method_id], time_per_fold, mc))
    
    if slow_methods:
        print("\n⚠️  SLOW METHODS WARNING:")
        for dataset, method, time_s, mc in slow_methods:
            print(f"   {method} on {dataset}: ~{time_s/60:.0f} min/fold "
                  f"(n≤{mc.max_neurons}, p={mc.patience})")
        print("   Consider running these separately or reducing parameters")
    
    return total_seconds


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="5-Fold Cross-Validation for Neural Network Methods"
    )
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['computer_hardware', 'fish_toxicity', 'aquatic_toxicity'],
                        help="Run specific dataset (default: all)")
    parser.add_argument('--methods', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                        help="Methods to run (1-6)")
    parser.add_argument('--n-folds', type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('--base-dir', type=str, default=".",
                        help="Base directory for data and results")
    parser.add_argument('--estimate-only', action='store_true',
                        help="Only estimate time, don't run")
    parser.add_argument('--resume', type=str, default=None,
                        help="Resume from previous run directory")
    parser.add_argument('--skip-existing', action='store_true',
                        help="Skip methods/folds that already have results")
    
    args = parser.parse_args()
    
    # Setup configuration
    cv_config = CVConfig(
        n_folds=args.n_folds,
        seed=args.seed,
        methods=args.methods
    )
    
    if args.dataset:
        cv_config.datasets = [args.dataset]
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.base_dir) / "results" / "cross_validation" / f"cv_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print header
    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 70)
    print(f"Datasets: {cv_config.datasets}")
    print(f"Methods: {[METHOD_NAMES[m] for m in cv_config.methods]}")
    print(f"Folds: {cv_config.n_folds}")
    print(f"Seed: {cv_config.seed}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # Time estimation
    print_time_estimate(cv_config)
    
    if args.estimate_only:
        print("\n✅ Estimate only mode. Exiting without running.")
        return
    
    # Confirm before running long jobs
    total_seconds, _ = estimate_total_time(cv_config)
    if total_seconds > 3600:  # > 1 hour
        print(f"\n⚠️  This will take approximately {total_seconds/3600:.1f} hours.")
        response = input("Continue? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return
    
    # Save config
    with open(output_dir / "cv_config.json", 'w') as f:
        json.dump(asdict(cv_config), f, indent=2)
    
    # Run CV for each dataset
    all_results = {}
    all_summaries = {}
    total_start = time.time()
    
    for dataset_name in cv_config.datasets:
        try:
            df_results = run_cv_for_dataset(
                dataset_name=dataset_name,
                cv_config=cv_config,
                output_dir=output_dir,
                base_dir=args.base_dir
            )
            all_results[dataset_name] = df_results
            
            # Load summary
            summary_path = output_dir / dataset_name / "summary_stats.csv"
            all_summaries[dataset_name] = pd.read_csv(summary_path)
            
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start
    
    # Generate outputs
    print("\n" + "=" * 70)
    print("GENERATING OUTPUTS")
    print("=" * 70)
    
    if all_summaries:
        # Consolidated table
        consolidated = generate_consolidated_table(all_summaries, output_dir)
        print(f"✅ Consolidated results: {output_dir / 'consolidated_results.csv'}")
        
        # LaTeX tables
        generate_latex_tables(all_summaries, output_dir)
        print(f"✅ LaTeX tables: {output_dir / 'latex_tables.tex'}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")
    
    # Print summary table
    if all_summaries:
        print("\n" + "-" * 70)
        print("SUMMARY (Test R² mean ± std)")
        print("-" * 70)
        for dataset_name, summary in all_summaries.items():
            print(f"\n{dataset_name.upper()}:")
            for _, row in summary.iterrows():
                scr_str = f"{row['scr_mean']*100:.0f}%" if row['scr_mean'] == 1.0 else f"{row['scr_mean']*100:.1f}%"
                print(f"  {row['method_name']:15s} R²={row['test_r2_mean']:.4f}±{row['test_r2_std']:.4f}  "
                      f"SCR={scr_str:>6s}  n={row['best_neurons_mean']:5.1f}")


if __name__ == "__main__":
    main()