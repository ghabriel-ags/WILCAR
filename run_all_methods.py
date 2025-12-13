#!/usr/bin/env python3
"""
Run All Methods - Batch Execution of 6 Neural Network Methods
==============================================================
Federal University of Bahia – MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá

This script runs all 6 methods in sequence and generates comparative dashboards.

Usage:
    # Run all methods with default parameters
    python run_all_methods.py --dataset computer_hardware
    
    # Run with custom parameters for all methods
    python run_all_methods.py --dataset qsar_fish_toxicity --max-neurons 100 --patience 50
    
    # Run with custom parameters per method
    python run_all_methods.py --dataset computer_hardware \\
        --max-neurons-m1 200 --patience-m1 100 \\
        --max-neurons-m2 50 --patience-m2 30 \\
        --max-neurons-m5 500 --patience-m5 150
    
    # Run only specific methods
    python run_all_methods.py --dataset computer_hardware --methods 1 2 5 6

Methods:
    1. WILCAR Unconstrained - Linearization init, backprop, no constraints
    2. WILCAR Constrained - Linearization init, SLSQP, gain constraints
    3. RIXM - Random Xavier init, backprop, no constraints
    4. RIXM Constrained - Random Xavier init, SLSQP, gain constraints
    5. ELM - Random fixed hidden, pseudo-inverse, no constraints
    6. ELM Constrained - Random fixed hidden, SLSQP, gain constraints

Output:
    results/<dataset>/<timestamp>/
    ├── method_1_wilcar_unconstrained/   # Individual results
    ├── method_2_wilcar_constrained/
    ├── method_3_rixm/
    ├── method_4_rixm_constrained/
    ├── method_5_elm/
    ├── method_6_elm_constrained/
    ├── comparative/                      # Comparative dashboards
    │   ├── all_methods_dashboard.png
    │   ├── paired_wilcar_1_2.png
    │   ├── paired_rixm_3_4.png
    │   ├── paired_elm_5_6.png
    │   ├── unconstrained_1_3_5.png
    │   ├── constrained_2_4_6.png
    │   └── summary_table.csv
    └── batch_config.json
"""

import sys
import argparse
import numpy as np
import json
import pickle
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Project imports
from base import TrainingConfig, TrainingResults, check_gpu_availability
from wilcar.unconstrained import create_wilcar_unconstrained
from wilcar.constrained import create_wilcar_constrained
from rixm import create_rixm, create_rixm_constrained
from elm import create_elm, create_elm_constrained
from utils.data import (
    load_raw_data, 
    preprocess_data, 
    get_expected_signals,
    save_dataset_config,
)
from utils.visualization import (
    plot_performance_dashboard,
    plot_gains_comparison,
    plot_learning_curves,
    plot_comparative_dashboard,
    plot_paired_comparison,
    create_summary_table,
    export_latex_table,
    METHOD_INFO
)

# For visualization
import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# METHOD FACTORIES
# =============================================================================

METHOD_FACTORIES = {
    'wilcar_unconstrained': create_wilcar_unconstrained,
    'wilcar_constrained': create_wilcar_constrained,
    'rixm': create_rixm,
    'rixm_constrained': create_rixm_constrained,
    'elm': create_elm,
    'elm_constrained': create_elm_constrained
}


# =============================================================================
# BATCH CONFIGURATION
# =============================================================================

@dataclass
class MethodConfig:
    """Configuration for a single method."""
    max_neurons: int = 750
    patience: int = 150
    
@dataclass
class BatchConfig:
    """Configuration for batch execution."""
    dataset: str = "computer_hardware"
    methods: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    seed: int = 0  # Same default as run_experiment.py
    test_size: float = 0.2
    verbose: int = 1
    use_gpu: bool = True
    
    # Default parameters (used if method-specific not provided)
    default_max_neurons: int = 750
    default_patience: int = 150
    
    # Method-specific parameters (None means use default)
    method_configs: Dict[int, MethodConfig] = field(default_factory=dict)
    
    def get_config_for_method(self, method_id: int) -> MethodConfig:
        """Get configuration for specific method."""
        if method_id in self.method_configs:
            return self.method_configs[method_id]
        return MethodConfig(
            max_neurons=self.default_max_neurons,
            patience=self.default_patience
        )


# =============================================================================
# INDIVIDUAL METHOD EXECUTION
# =============================================================================

def run_single_method(
    method_id: int,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    test_inputs: np.ndarray,
    test_targets: np.ndarray,
    expected_signals: List[int],
    method_config: MethodConfig,
    batch_config: BatchConfig,
    dataset_name: str,
    output_dir: Path
) -> Tuple[Optional[TrainingResults], Path]:
    """
    Run a single method and save results.
    
    Returns:
        Tuple of (results, output_directory)
    """
    info = METHOD_INFO[method_id]
    method_name = info['name']
    
    print("\n" + "=" * 70)
    print(f"METHOD {method_id}: {info['display']}")
    print(f"Max neurons: {method_config.max_neurons} | Patience: {method_config.patience}")
    print("=" * 70)
    
    # Create output directory
    method_dir = output_dir / f"method_{method_id}_{method_name}"
    method_dir.mkdir(parents=True, exist_ok=True)
    (method_dir / "figures").mkdir(exist_ok=True)
    (method_dir / "logs").mkdir(exist_ok=True)
    
    # Create training config (same parameters as run_experiment.py)
    config = TrainingConfig(
        max_neurons=method_config.max_neurons,
        patience_constructive=method_config.patience,
        patience_early_stopping=250,
        learning_rate=0.1,
        lr_decay=0.999,
        iterations=2501,
        delta_perturbation=0.1,
        seed=batch_config.seed,
        verbose=batch_config.verbose,
        use_gpu=batch_config.use_gpu
    )
    
    # Create model
    factory = METHOD_FACTORIES[method_name]
    model = factory(
        train_inputs=train_inputs,
        train_targets=train_targets,
        test_inputs=test_inputs,
        test_targets=test_targets,
        expected_signals=expected_signals,
        config=config,
        dataset_name=dataset_name
    )
    
    # Train
    start_time = time.time()
    try:
        results = model.train()
    except Exception as e:
        print(f"\n❌ Error training method {method_id}: {e}")
        return None, method_dir
    
    training_time = time.time() - start_time
    
    # Save results
    print(f"\n💾 Saving results for Method {method_id}...")
    
    # Pickle
    with open(method_dir / "logs" / "results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    # JSON summary
    summary = {
        'method_id': method_id,
        'method_name': info['display'],
        'dataset': dataset_name,
        'best_neurons': results.best_neurons,
        'best_test_r2': float(results.best_test_r2),
        'best_test_mse': float(results.best_test_mse),
        'best_test_rmse': float(results.best_test_rmse),
        'best_train_r2': float(results.best_train_r2),
        'best_conformity': float(results.best_conformity),
        'total_time': float(results.total_time),
        'config': {
            'max_neurons': method_config.max_neurons,
            'patience': method_config.patience
        }
    }
    with open(method_dir / "logs" / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Individual plots
    fig = plot_performance_dashboard(
        results, 
        expected_signals,
        method_name=f"{info['display']} - {dataset_name}",
        save_path=method_dir / "figures" / "performance_dashboard.png",
        show=False
    )
    plt.close(fig)
    
    if results.conformity_details:
        plot_gains_comparison(
            results.conformity_details,
            expected_signals,
            results.best_neurons,
            results.best_conformity,
            method_name=f"{info['display']} - {dataset_name}",
            save_path=method_dir / "figures" / "gains_comparison.png",
            show=False
        )
    
    plot_learning_curves(
        results,
        method_name=f"{info['display']} - {dataset_name}",
        save_path=method_dir / "figures" / "learning_curves.png",
        show=False
    )
    
    print(f"✅ Method {method_id} completed in {training_time:.1f}s")
    print(f"   Best: n={results.best_neurons} | R²={results.best_test_r2:.4f} | SCR={results.best_conformity:.1%}")
    
    return results, method_dir

# =============================================================================
# MAIN BATCH EXECUTION
# =============================================================================

def run_batch(batch_config: BatchConfig, base_dir: Path):
    """
    Run all methods in batch and generate comparative dashboards.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = base_dir / "results" / batch_config.dataset / f"batch_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparative_dir = output_dir / "comparative"
    comparative_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("BATCH EXECUTION - ALL METHODS")
    print("=" * 70)
    print(f"Dataset: {batch_config.dataset}")
    print(f"Methods: {batch_config.methods}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # Save batch config
    config_dict = {
        'dataset': batch_config.dataset,
        'methods': batch_config.methods,
        'seed': batch_config.seed,
        'test_size': batch_config.test_size,
        'timestamp': timestamp,
        'default_max_neurons': batch_config.default_max_neurons,
        'default_patience': batch_config.default_patience,
        'method_configs': {
            str(k): asdict(v) for k, v in batch_config.method_configs.items()
        }
    }
    with open(output_dir / "batch_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Load data
    print("\n📊 Loading data...")
    try:
        data = load_raw_data(batch_config.dataset, base_dir=str(base_dir))
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return None
    
    n_inputs = data.shape[1] - 1
    print(f"   Samples: {data.shape[0]} | Variables: {n_inputs}")
    
    # Get expected signals
    print("\n🔍 Identifying expected signals...")
    expected_signals, detected_name = get_expected_signals(n_inputs, batch_config.dataset)
    dataset_name = detected_name or batch_config.dataset
    
    # Preprocess
    print("\n⚙️ Preprocessing data...")
    train_inputs, train_targets, test_inputs, test_targets, scaler = preprocess_data(
        data, test_size=batch_config.test_size, seed=batch_config.seed
    )
    
    print(f"   Train: {train_inputs.shape[0]} | Test: {test_inputs.shape[0]}")
    
    # Save processed data
    save_dataset_config(
        dataset_name=dataset_name,
        filename=batch_config.dataset,
        expected_signals=expected_signals,
        data_shape=data.shape,
        seed=batch_config.seed,
        base_dir=str(base_dir)
    )
    
    # Run all methods
    all_results: Dict[int, TrainingResults] = {}
    total_start = time.time()
    
    for method_id in batch_config.methods:
        method_config = batch_config.get_config_for_method(method_id)
        
        results, method_dir = run_single_method(
            method_id=method_id,
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            expected_signals=expected_signals,
            method_config=method_config,
            batch_config=batch_config,
            dataset_name=dataset_name,
            output_dir=output_dir
        )
        
        all_results[method_id] = results
    
    total_time = time.time() - total_start
    
    # Generate comparative dashboards
    print("\n" + "=" * 70)
    print("📊 GENERATING COMPARATIVE DASHBOARDS")
    print("=" * 70)
    
    # 1. All methods
    print("\n1. All methods dashboard...")
    plot_comparative_dashboard(
        all_results,
        batch_config.methods,
        dataset_name,
        comparative_dir / "all_methods_dashboard.png",
        title="All Methods Comparison"
    )
    
    # 2. Paired comparisons (1-2, 3-4, 5-6)
    print("\n2. Paired comparisons (by family)...")
    pairs = [(1, 2, 'WILCAR'), (3, 4, 'RIXM'), (5, 6, 'ELM')]
    for m_a, m_b, family in pairs:
        if m_a in batch_config.methods and m_b in batch_config.methods:
            plot_paired_comparison(
                all_results, m_a, m_b, dataset_name,
                comparative_dir / f"paired_{family.lower()}_{m_a}_{m_b}.png",
                title=f"{family}: Unconstrained vs Constrained"
            )
    
    # 3. Triplet comparisons (1-3-5, 2-4-6)
    print("\n3. Triplet comparisons (by constraint type)...")
    
    # Unconstrained methods
    unconstrained = [m for m in [1, 3, 5] if m in batch_config.methods]
    if len(unconstrained) > 1:
        plot_comparative_dashboard(
            all_results,
            unconstrained,
            dataset_name,
            comparative_dir / "unconstrained_1_3_5.png",
            title="Unconstrained Methods (No Gain Constraints)"
        )
    
    # Constrained methods
    constrained = [m for m in [2, 4, 6] if m in batch_config.methods]
    if len(constrained) > 1:
        plot_comparative_dashboard(
            all_results,
            constrained,
            dataset_name,
            comparative_dir / "constrained_2_4_6.png",
            title="Constrained Methods (With Gain Constraints)"
        )
    
    # 4. Summary table
    print("\n4. Summary table...")
    df = create_summary_table(
        all_results,
        batch_config.methods,
        dataset_name,
        comparative_dir / "summary_table.csv"
    )
    
    # 5. LaTeX table for papers
    print("\n5. LaTeX table...")
    export_latex_table(
        all_results,
        batch_config.methods,
        dataset_name,
        comparative_dir / "results_table.tex"
    )
    
    # Print final summary
    print("\n" + "=" * 70)
    print("✅ BATCH EXECUTION COMPLETED")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nResults saved to: {output_dir}")
    print("\n📊 Summary:")
    print(df.to_string(index=False))
    
    # Identify best method
    if not df.empty:
        best_idx = df['Best_Test_R2'].idxmax()
        best_method = df.loc[best_idx]
        print(f"\n🏆 Best method: {best_method['Method_Name']}")
        print(f"   R² = {best_method['Best_Test_R2']:.4f} | SCR = {best_method['Best_SCR']:.1%}")
    
    return output_dir, all_results


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run all 6 neural network methods in batch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods with defaults
  python run_all_methods.py --dataset computer_hardware
  
  # Custom global parameters
  python run_all_methods.py --dataset qsar_fish_toxicity --max-neurons 100 --patience 50
  
  # Per-method parameters
  python run_all_methods.py --dataset computer_hardware \\
      --max-neurons-m1 200 --max-neurons-m2 50 --max-neurons-m5 500
  
  # Run specific methods only
  python run_all_methods.py --dataset computer_hardware --methods 1 2 5 6
        """
    )
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (without .csv)')
    
    # Methods to run
    parser.add_argument('--methods', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                        choices=[1, 2, 3, 4, 5, 6],
                        help='Methods to run (default: all)')
    
    # Global defaults
    parser.add_argument('--max-neurons', type=int, default=750,
                        help='Default max neurons (default: 750)')
    parser.add_argument('--patience', type=int, default=150,
                        help='Default patience (default: 150)')
    
    # Per-method max-neurons
    for i in range(1, 7):
        parser.add_argument(f'--max-neurons-m{i}', type=int, default=None,
                            help=f'Max neurons for method {i}')
    
    # Per-method patience
    for i in range(1, 7):
        parser.add_argument(f'--patience-m{i}', type=int, default=None,
                            help=f'Patience for method {i}')
    
    # Other options
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set fraction (default: 0.2)')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (default: 1)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU')
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Base directory (default: script directory)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine base directory
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = Path(__file__).parent
    
    # Build batch config
    batch_config = BatchConfig(
        dataset=args.dataset,
        methods=args.methods,
        seed=args.seed,
        test_size=args.test_size,
        verbose=args.verbose,
        use_gpu=not args.no_gpu,
        default_max_neurons=args.max_neurons,
        default_patience=args.patience
    )
    
    # Set per-method configs
    for i in range(1, 7):
        max_n = getattr(args, f'max_neurons_m{i}')
        pat = getattr(args, f'patience_m{i}')
        
        if max_n is not None or pat is not None:
            batch_config.method_configs[i] = MethodConfig(
                max_neurons=max_n if max_n is not None else args.max_neurons,
                patience=pat if pat is not None else args.patience
            )
    
    # Check GPU
    if batch_config.use_gpu:
        check_gpu_availability()
    
    # Run batch
    run_batch(batch_config, base_dir)


if __name__ == "__main__":
    main()