#!/usr/bin/env python3
"""
Experiment with Real Dataset - WILCAR, RIXM and ELM (Methods 1-6)
=================================================================
Federal University of Bahia – MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá

This script runs WILCAR, RIXM and ELM methods with real datasets.

Usage:
    python run_experiment.py                                        # Method 1 (default)
    python run_experiment.py --method wilcar_unconstrained          # Method 1
    python run_experiment.py --method wilcar_constrained            # Method 2
    python run_experiment.py --method rixm                          # Method 3
    python run_experiment.py --method rixm_constrained              # Method 4
    python run_experiment.py --method elm                           # Method 5
    python run_experiment.py --method elm_constrained               # Method 6
    python run_experiment.py --dataset qsar_fish_toxicity           # Other dataset
    python run_experiment.py --list                                 # List datasets

Methods:
    - wilcar_unconstrained (Method 1): WILCAR without gain sign constraints
    - wilcar_constrained (Method 2): WILCAR with gain sign constraints
    - rixm (Method 3): Baseline - random initialization, no weight reuse
    - rixm_constrained (Method 4): Baseline with gain sign constraints
    - elm (Method 5): Extreme Learning Machine (pseudo-inverse)
    - elm_constrained (Method 6): ELM with gain sign constraints

Supported Datasets:
    - computer_hardware (Computer Hardware)
    - qsar_fish_toxicity (Fish Toxicity)
    - qsar_aquatic_toxicity (Aquatic Toxicity)
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Adicionar src ao path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Imports do projeto
from base import TrainingConfig, check_gpu_availability
from wilcar.unconstrained import create_wilcar_unconstrained
from wilcar.constrained import create_wilcar_constrained
from rixm import create_rixm, create_rixm_constrained
from elm import create_elm, create_elm_constrained
from utils.data import (
    load_raw_data, 
    preprocess_data, 
    get_expected_signals,
    save_dataset_config,
    save_processed_data,
    EXPECTED_SIGNALS_THEORY
)
from utils.visualization import (
    plot_performance_dashboard,
    plot_gains_comparison,
    plot_learning_curves,
    print_summary,
    print_conformity_analysis
)

# Para salvar resultados
import json
import pickle
import pandas as pd


def list_available_datasets(data_dir: Path):
    """List available datasets in data/raw/"""
    print("\n📂 Available datasets in data/raw/:")
    print("-" * 50)
    
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("   ⚠️ No .csv files found!")
        print(f"   Place your datasets in: {data_dir}")
        return []
    
    datasets = []
    for f in sorted(csv_files):
        name = f.stem
        datasets.append(name)
        
        # Check if recognized
        recognized = "✅" if name in EXPECTED_SIGNALS_THEORY or \
                           any(key in name.lower() for key in ['machine', 'fish', 'aquatic']) else "⚠️"
        
        print(f"   {recognized} {name}.csv")
    
    print("-" * 50)
    print("\n📋 Datasets with known theoretical signals:")
    for name, signals in EXPECTED_SIGNALS_THEORY.items():
        print(f"   • {name}: {signals}")
    
    return datasets


def save_results(results, model, dataset_name: str, expected_signals: list,
                output_dir: Path, config: TrainingConfig):
    """Save all experiment results."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directories
    run_dir = output_dir / dataset_name / results.method_name / f"run_{timestamp}"
    metrics_dir = run_dir / "metrics"
    models_dir = run_dir / "models"
    figures_dir = run_dir / "figures"
    logs_dir = run_dir / "logs"
    
    for d in [metrics_dir, models_dir, figures_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Saving results to: {run_dir}")
    
    # 1. Detailed metrics (CSV)
    neurons = list(range(1, len(results.r2_test) + 1))
    gap_r2 = [r2t - r2e for r2t, r2e in zip(results.r2_train, results.r2_test)]
    
    metrics_df = pd.DataFrame({
        'n_neurons': neurons,
        'mse_train': results.mse_train,
        'rmse_train': results.rmse_train,
        'r2_train': results.r2_train,
        'mse_test': results.mse_test,
        'rmse_test': results.rmse_test,
        'r2_test': results.r2_test,
        'conformity_rate': results.conformity_rates,
        'iteration_time': results.iteration_times,
        'gap_r2': gap_r2
    })
    metrics_df.to_csv(metrics_dir / "detailed_metrics.csv", index=False)
    print(f"  ✓ Detailed metrics saved")
    
    # 2. Best model summary (JSON)
    best_summary = {
        'dataset': dataset_name,
        'method': results.method_name,
        'best_neurons': results.best_neurons,
        'train_mse': results.best_train_mse,
        'train_rmse': results.best_train_rmse,
        'train_r2': results.best_train_r2,
        'test_mse': results.best_test_mse,
        'test_rmse': results.best_test_rmse,
        'test_r2': results.best_test_r2,
        'conformity_rate': results.best_conformity,
        'total_time': results.total_time,
        'avg_time_per_model': results.avg_time_per_model,
        'models_trained': len(results.r2_test),
        'timestamp': timestamp
    }
    
    with open(metrics_dir / "best_model_summary.json", 'w') as f:
        json.dump(best_summary, f, indent=2)
    print(f"  ✓ Best model summary saved")
    
    # 3. Gain analysis (CSV)
    if results.conformity_details:
        gains_df = pd.DataFrame(results.conformity_details)
        gains_df['dataset'] = dataset_name
        gains_df['method'] = results.method_name
        gains_df['n_neurons'] = results.best_neurons
        gains_df.to_csv(metrics_dir / "gains_analysis.csv", index=False)
        print(f"  ✓ Gain analysis saved")
    
    # 4. Model (pickle)
    if hasattr(model, 'param') and model.param is not None:
        with open(models_dir / f"best_model_n{results.best_neurons}.pkl", 'wb') as f:
            pickle.dump(model.param, f)
        print(f"  ✓ Model saved")
    
    # 5. Execution config (JSON)
    exec_config = {
        'execution_info': {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'method': results.method_name
        },
        'hyperparameters': {
            'max_neurons': config.max_neurons,
            'patience_constructive': config.patience_constructive,
            'patience_early_stopping': config.patience_early_stopping,
            'learning_rate': config.learning_rate,
            'lr_decay': config.lr_decay,
            'iterations': config.iterations,
            'delta_perturbation': config.delta_perturbation
        },
        'data_info': {
            'expected_signals': expected_signals
        },
        'environment': {
            'seed': config.seed,
            'use_gpu': config.use_gpu
        }
    }
    
    with open(logs_dir / "execution_config.json", 'w') as f:
        json.dump(exec_config, f, indent=2)
    print(f"  ✓ Execution config saved")
    
    # 6. Plots
    # Dashboard
    fig = plot_performance_dashboard(
        results, 
        expected_signals,
        method_name=f"{results.method_name} - {dataset_name}",
        save_path=figures_dir / "performance_dashboard.png",
        show=False
    )
    fig.savefig(figures_dir / "performance_dashboard.pdf", dpi=600, bbox_inches='tight')
    print(f"  ✓ Dashboard saved (PNG + PDF)")
    
    # Gains plot
    if results.conformity_details:
        plot_gains_comparison(
            results.conformity_details,
            expected_signals,
            results.best_neurons,
            results.best_conformity,
            method_name=f"{results.method_name} - {dataset_name}",
            save_path=figures_dir / "gains_comparison.png",
            show=False
        )
        print(f"  ✓ Gains plot saved")
    
    # Learning curves
    plot_learning_curves(
        results,
        method_name=f"{results.method_name} - {dataset_name}",
        save_path=figures_dir / "learning_curves.png",
        show=False
    )
    print(f"  ✓ Learning curves saved")
    
    print(f"\n✅ All results saved to: {run_dir}")
    return run_dir


def run_experiment(dataset_name: str, base_dir: Path, config: TrainingConfig,
                   method: str = "wilcar_unconstrained", show_plots: bool = True):
    """
    Run complete experiment with a real dataset.
    
    Args:
        dataset_name: File name (without .csv)
        base_dir: Project base directory
        config: Training configuration
        method: "wilcar_unconstrained" (Method 1) or "wilcar_constrained" (Method 2)
        show_plots: If True, display plots at the end
    """
    
    # Determine method display name
    method_display = {
        "wilcar_unconstrained": "WILCAR UNCONSTRAINED (Method 1)",
        "wilcar_constrained": "WILCAR CONSTRAINED (Method 2)",
        "rixm": "RIXM - Random Initialization Xavier Method (Method 3)",
        "rixm_constrained": "RIXM CONSTRAINED (Method 4)",
        "elm": "ELM - Extreme Learning Machine (Method 5)",
        "elm_constrained": "ELM CONSTRAINED (Method 6)"
    }
    
    print("=" * 70)
    print(f"EXPERIMENT - {method_display.get(method, method.upper())}")
    print(f"Dataset: {dataset_name}")
    print("=" * 70)
    
    # 1. Load data
    print("\n📊 Loading data...")
    try:
        data = load_raw_data(dataset_name, base_dir=str(base_dir))
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\n💡 Tip: Check if the file exists in {base_dir}/data/raw/")
        list_available_datasets(base_dir / "data" / "raw")
        return None, None
    
    n_inputs = data.shape[1] - 1
    n_samples = data.shape[0]
    
    print(f"   Samples: {n_samples}")
    print(f"   Input variables: {n_inputs}")
    
    # 2. Get expected signals
    print("\n🔍 Identifying expected signals...")
    try:
        expected_signals, detected_name = get_expected_signals(n_inputs, dataset_name)
    except ValueError as e:
        print(f"\n❌ {e}")
        return None, None
    
    # 3. Preprocess
    print("\n⚙️ Preprocessing data...")
    train_inputs, train_targets, test_inputs, test_targets, scaler = preprocess_data(
        data, test_size=0.2, seed=config.seed
    )
    
    # 4. Save processed data (optional)
    print("\n💾 Saving processed data...")
    save_dataset_config(
        dataset_name=detected_name or dataset_name,
        filename=dataset_name,
        expected_signals=expected_signals,
        data_shape=data.shape,
        seed=config.seed,
        base_dir=str(base_dir)
    )
    save_processed_data(
        dataset_name=detected_name or dataset_name,
        train_inputs=train_inputs,
        train_targets=train_targets,
        test_inputs=test_inputs,
        test_targets=test_targets,
        scaler=scaler,
        base_dir=str(base_dir)
    )
    
    # 5. Create model based on method
    print("\n🔧 Creating model...")
    
    if method == "wilcar_unconstrained":
        model = create_wilcar_unconstrained(
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            expected_signals=expected_signals,
            config=config,
            dataset_name=detected_name or dataset_name
        )
    elif method == "wilcar_constrained":
        model = create_wilcar_constrained(
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            expected_signals=expected_signals,
            config=config,
            dataset_name=detected_name or dataset_name
        )
    elif method == "rixm":
        model = create_rixm(
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            expected_signals=expected_signals,
            config=config,
            dataset_name=detected_name or dataset_name
        )
    elif method == "rixm_constrained":
        model = create_rixm_constrained(
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            expected_signals=expected_signals,
            config=config,
            dataset_name=detected_name or dataset_name
        )
    elif method == "elm":
        model = create_elm(
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            expected_signals=expected_signals,
            config=config,
            dataset_name=detected_name or dataset_name
        )
    elif method == "elm_constrained":
        model = create_elm_constrained(
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            expected_signals=expected_signals,
            config=config,
            dataset_name=detected_name or dataset_name
        )
    else:
        print(f"\n❌ Unknown method: {method}")
        print("   Available methods: wilcar_unconstrained, wilcar_constrained, rixm, rixm_constrained, elm, elm_constrained")
        return None, None
    
    # 6. Train
    print("\n🏋️ Starting training...")
    results = model.train()
    
    # 7. Display summary
    print_summary(results, method_name=results.method_name)
    
    if results.conformity_details:
        print_conformity_analysis(results.conformity_details, expected_signals)
    
    # 8. Save results
    output_dir = base_dir / "results"
    run_dir = save_results(
        results, model, 
        detected_name or dataset_name,
        expected_signals,
        output_dir, config
    )
    
    # 9. Display plots
    if show_plots:
        print("\n📈 Displaying plots...")
        plot_performance_dashboard(
            results,
            expected_signals,
            method_name=f"{results.method_name} - {detected_name or dataset_name}",
            show=True
        )
    
    return results, model


def main():
    parser = argparse.ArgumentParser(
        description='Run WILCAR experiment with real dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_experiment.py --dataset computer_hardware
    python run_experiment.py --dataset computer_hardware --method wilcar_constrained
    python run_experiment.py --dataset computer_hardware --method rixm
    python run_experiment.py --dataset computer_hardware --method rixm_constrained
    python run_experiment.py --dataset computer_hardware --method elm
    python run_experiment.py --dataset computer_hardware --method elm_constrained
    python run_experiment.py --dataset qsar_fish_toxicity --max-neurons 100
    python run_experiment.py --list
    
Methods:
    wilcar_unconstrained  - Method 1: WILCAR without gain sign constraints
    wilcar_constrained    - Method 2: WILCAR with gain sign constraints (SLSQP)
    rixm                  - Method 3: Baseline (Random Xavier Init, no weight reuse)
    rixm_constrained      - Method 4: Baseline with gain sign constraints (SLSQP)
    elm                   - Method 5: Extreme Learning Machine (pseudo-inverse)
    elm_constrained       - Method 6: ELM with gain sign constraints (SLSQP)
        """
    )
    
    parser.add_argument('--dataset', '-d', type=str, default='computer_hardware',
                       help='Dataset name (without .csv)')
    parser.add_argument('--method', '-m', type=str, default='wilcar_unconstrained',
                       choices=['wilcar_unconstrained', 'wilcar_constrained', 'rixm', 'rixm_constrained', 'elm', 'elm_constrained'],
                       help='Method to use (default: wilcar_unconstrained)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available datasets')
    parser.add_argument('--max-neurons', '-n', type=int, default=750,
                       help='Maximum number of neurons (default: 750)')
    parser.add_argument('--patience', '-p', type=int, default=150,
                       help='Constructive patience (default: 150)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Do not display plots')
    parser.add_argument('--seed', '-s', type=int, default=0,
                       help='Seed for reproducibility (default: 0)')
    
    args = parser.parse_args()
    
    # Base directory
    base_dir = Path(__file__).parent
    
    # List datasets
    if args.list:
        list_available_datasets(base_dir / "data" / "raw")
        return
    
    # Check GPU
    print("\n🔍 Checking environment...")
    gpu_info = check_gpu_availability()
    print(f"   PyTorch: {'✅' if gpu_info['torch_available'] else '❌'}")
    print(f"   CUDA: {'✅' if gpu_info['cuda_available'] else '❌'}")
    
    # Configuration
    config = TrainingConfig(
        max_neurons=args.max_neurons,
        patience_constructive=args.patience,
        patience_early_stopping=250,
        learning_rate=0.1,
        lr_decay=0.999,
        iterations=2501,
        delta_perturbation=0.1,
        seed=args.seed,
        use_gpu=gpu_info['cuda_available'],
        verbose=1
    )
    
    print(f"\n⚙️ Configuration:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Method: {args.method}")
    print(f"   Max neurons: {config.max_neurons}")
    print(f"   Patience: {config.patience_constructive}")
    print(f"   Seed: {config.seed}")
    
    # Run
    results, model = run_experiment(
        dataset_name=args.dataset,
        base_dir=base_dir,
        config=config,
        method=args.method,
        show_plots=not args.no_plots
    )
    
    if results:
        print("\n" + "=" * 70)
        print("✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 70)


if __name__ == "__main__":
    main()