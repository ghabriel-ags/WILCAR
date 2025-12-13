#!/usr/bin/env python3
"""
Generate Publication Figures from Experimental Results
=======================================================
Federal University of Bahia – MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá

This script generates two figures for the manuscript:
1. Individual Value Plot (strip plot) of Test R² from 5-fold CV results
2. Learning curves (Test R² vs Neurons) from a single-run (illustrative)

Usage:
    python generate_figures.py --dataset computer_hardware
    python generate_figures.py --dataset fish_toxicity
    python generate_figures.py --dataset aquatic_toxicity
"""

import sys
from pathlib import Path

# Add src to path for pickle to find base module
script_dir = Path(__file__).parent.resolve()
src_dir = script_dir / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))
else:
    sys.path.insert(0, str(script_dir))

import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

# =============================================================================
# CONFIGURATION
# =============================================================================

METHOD_NAMES = {
    1: 'WILCAR (Un)',
    2: 'WILCAR (C)',
    3: 'RIXM (Un)',
    4: 'RIXM (C)',
    5: 'ELM (Un)',
    6: 'ELM (C)'
}

METHOD_FOLDERS = {
    1: 'method_1_wilcar_unconstrained',
    2: 'method_2_wilcar_constrained',
    3: 'method_3_rixm',
    4: 'method_4_rixm_constrained',
    5: 'method_5_elm',
    6: 'method_6_elm_constrained'
}

METHOD_COLORS = {
    1: '#1f77b4',  # Blue - WILCAR (Un)
    2: '#0d3d6e',  # Dark Blue - WILCAR (C)
    3: '#2ca02c',  # Green - RIXM (Un)
    4: '#145214',  # Dark Green - RIXM (C)
    5: '#ff7f0e',  # Orange - ELM (Un)
    6: '#d62728',  # Red - ELM (C)
}

METHOD_STYLES = {
    1: {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'markersize': 3},
    2: {'color': '#0d3d6e', 'linestyle': '--', 'marker': 's', 'markersize': 4, 'linewidth': 2},
    3: {'color': '#2ca02c', 'linestyle': '-', 'marker': '^', 'markersize': 3},
    4: {'color': '#145214', 'linestyle': '--', 'marker': 'v', 'markersize': 4, 'linewidth': 2},
    5: {'color': '#ff7f0e', 'linestyle': '-', 'marker': 'd', 'markersize': 3},
    6: {'color': '#d62728', 'linestyle': '--', 'marker': 'p', 'markersize': 4, 'linewidth': 2},
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_single_run_results(batch_dir: Path) -> Dict[int, object]:
    """Load TrainingResults from a single-run batch directory."""
    try:
        import base
    except ImportError:
        print("⚠️ Warning: Could not import base module. Pickle may fail.")
    
    results = {}
    
    for method_id, folder_name in METHOD_FOLDERS.items():
        pkl_path = batch_dir / folder_name / "logs" / "results.pkl"
        
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                results[method_id] = pickle.load(f)
            print(f"✓ Loaded {METHOD_NAMES[method_id]}: {len(results[method_id].r2_test)} neurons")
        else:
            print(f"✗ Not found: {pkl_path}")
    
    return results


def load_cv_results(cv_fold_results_path: Path) -> pd.DataFrame:
    """Load cross-validation fold results."""
    df = pd.read_csv(cv_fold_results_path)
    print(f"✓ Loaded CV results: {len(df)} rows ({df['fold'].nunique()} folds × {df['method_id'].nunique()} methods)")
    return df


# =============================================================================
# FIGURE 1: INDIVIDUAL VALUE PLOT (STRIP PLOT)
# =============================================================================

def create_stripplot_figure(df: pd.DataFrame, dataset_name: str, output_path: Path):
    """
    Create individual value plot (strip plot) of Test R² from cross-validation results.
    Each point represents one fold's result.
    """
    print(f"  Creating strip plot with {len(df)} rows")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    methods = sorted(df['method_id'].unique())
    labels = [METHOD_NAMES[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    
    x_positions = np.arange(len(methods))
    
    # Plot individual points for each method
    for i, method_id in enumerate(methods):
        method_data = df[df['method_id'] == method_id]['test_r2'].values
        
        # Add small jitter to x position
        np.random.seed(42 + i)  # Reproducible jitter
        jitter = np.random.uniform(-0.12, 0.12, size=len(method_data))
        x = np.full(len(method_data), i) + jitter
        
        # Plot individual points
        ax.scatter(x, method_data, color=colors[i], s=80, alpha=0.7, 
                  edgecolor='black', linewidth=0.5, zorder=3)
        
        # Plot mean as horizontal line
        mean_val = method_data.mean()
        ax.hlines(mean_val, i - 0.25, i + 0.25, color='black', linewidth=2, zorder=4)
    
    # Style
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=11)
    ax.set_ylabel('Test R²', fontsize=12)
    
    # No title (removed for publication)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Fixed y-axis scale: 0 to 1
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: {output_path}")


# =============================================================================
# FIGURE 2: LEARNING CURVES (3 PANELS)
# =============================================================================

def create_learning_curves_figure(results: Dict[int, object], dataset_name: str, 
                                   output_path: Path):
    """
    Create learning curves (Test R² vs Hidden Units) from single-run results.
    3 panels side by side: WILCAR, RIXM, ELM (each with Un/C variants).
    Max neurons per panel is determined by the maximum reached by either variant.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Define method groups
    method_groups = [
        {'name': 'WILCAR', 'methods': [1, 2]},
        {'name': 'RIXM', 'methods': [3, 4]},
        {'name': 'ELM', 'methods': [5, 6]},
    ]
    
    for ax_idx, group in enumerate(method_groups):
        ax = axes[ax_idx]
        
        # Determine max neurons from data (max of both variants in this group)
        max_n = 0
        for method_id in group['methods']:
            if method_id in results:
                max_n = max(max_n, len(results[method_id].r2_test))
        
        if max_n == 0:
            max_n = 100  # Fallback
        
        for method_id in group['methods']:
            if method_id not in results:
                continue
                
            r = results[method_id]
            neurons = np.arange(1, len(r.r2_test) + 1)
            r2_plot = np.array(r.r2_test)
            
            style = METHOD_STYLES[method_id].copy()
            linewidth = style.pop('linewidth', 1.5)
            
            ax.plot(neurons, r2_plot, 
                    label=METHOD_NAMES[method_id],
                    linewidth=linewidth,
                    **style)
            
            # Mark best model
            best_idx = r.best_neurons - 1
            if best_idx < len(r.r2_test):
                ax.scatter(r.best_neurons, r.r2_test[best_idx], 
                          s=100, color=style['color'], edgecolor='black', 
                          linewidth=1.5, zorder=5)
        
        # Style for each panel
        ax.set_xlabel('Hidden Units', fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel('Test R²', fontsize=11)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(1, max_n)
        ax.set_ylim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: {output_path}")


# =============================================================================
# COMBINED FIGURE (2 PANELS)
# =============================================================================

def create_combined_figure(df_cv: pd.DataFrame, results_single: Dict[int, object],
                           dataset_name: str, output_path: Path, max_neurons: int = 150):
    """Create combined figure with strip plot and learning curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # =========================================================================
    # Panel A: Individual value plot (strip plot)
    # =========================================================================
    ax = axes[0]
    methods = sorted(df_cv['method_id'].unique())
    labels = [METHOD_NAMES[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    
    x_positions = np.arange(len(methods))
    
    for i, method_id in enumerate(methods):
        method_data = df_cv[df_cv['method_id'] == method_id]['test_r2'].values
        
        np.random.seed(42 + i)
        jitter = np.random.uniform(-0.12, 0.12, size=len(method_data))
        x = np.full(len(method_data), i) + jitter
        
        ax.scatter(x, method_data, color=colors[i], s=60, alpha=0.7,
                  edgecolor='black', linewidth=0.5, zorder=3)
        
        mean_val = method_data.mean()
        ax.hlines(mean_val, i - 0.2, i + 0.2, color='black', linewidth=2, zorder=4)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=10)
    ax.set_ylabel('Test R²', fontsize=11)
    # No title for publication
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    # Fixed y-axis scale: 0 to 1
    ax.set_ylim(0, 1)
    
    # =========================================================================
    # Panel B: Learning curves
    # =========================================================================
    ax = axes[1]
    
    for method_id in sorted(results_single.keys()):
        r = results_single[method_id]
        neurons = np.arange(1, len(r.r2_test) + 1)
        
        mask = neurons <= max_neurons
        neurons_plot = neurons[mask]
        r2_plot = np.array(r.r2_test)[mask]
        
        if len(neurons_plot) == 0:
            continue
            
        style = METHOD_STYLES[method_id].copy()
        linewidth = style.pop('linewidth', 1.5)
        
        ax.plot(neurons_plot, r2_plot, 
                label=METHOD_NAMES[method_id],
                linewidth=linewidth,
                **style)
        
        if r.best_neurons <= max_neurons:
            best_idx = r.best_neurons - 1
            if best_idx < len(r.r2_test):
                ax.scatter(r.best_neurons, r.r2_test[best_idx], 
                          s=80, color=style['color'], edgecolor='black', 
                          linewidth=1.5, zorder=5)
    
    ax.set_xlabel('Number of Neurons', fontsize=11)
    ax.set_ylabel('Test R²', fontsize=11)
    ax.set_title('Test R² vs Network Size (Illustrative)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(1, max_neurons)
    
    # Dynamic y-axis
    all_r2 = []
    for r in results_single.values():
        r2_values = r.r2_test[:max_neurons]
        all_r2.extend([v for v in r2_values if v > -0.5])
    if all_r2:
        y_max = min(1.05, max(all_r2) + 0.05)
        ax.set_ylim(0, y_max)
    
    # Suptitle
    fig.suptitle(f'Dataset: {dataset_name.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['computer_hardware', 'fish_toxicity', 'aquatic_toxicity'])
    parser.add_argument('--batch-dir', type=str, required=True,
                        help="Path to single-run batch directory")
    parser.add_argument('--cv-results', type=str, required=True,
                        help="Path to CV fold_results.csv")
    parser.add_argument('--output-dir', type=str, default="figures",
                        help="Output directory for figures")
    parser.add_argument('--src-dir', type=str, default=None,
                        help="Path to src directory (for pickle to find base module)")
    
    args = parser.parse_args()
    
    # Add src-dir to path if specified
    if args.src_dir:
        sys.path.insert(0, args.src_dir)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "="*60)
    print(f"Loading data for {args.dataset}")
    print("="*60)
    
    batch_dir = Path(args.batch_dir)
    results_single = load_single_run_results(batch_dir)
    
    cv_path = Path(args.cv_results)
    df_cv = load_cv_results(cv_path)
    
    # Filter CV results for this dataset
    print(f"  CV DataFrame has {len(df_cv)} rows")
    if 'dataset' in df_cv.columns:
        print(f"  Unique datasets: {df_cv['dataset'].unique()}")
        print(f"  Filtering for: {args.dataset}")
        df_cv = df_cv[df_cv['dataset'] == args.dataset]
        print(f"  After filtering: {len(df_cv)} rows")
    
    if len(df_cv) == 0:
        print("⚠️ ERROR: No data after filtering! Check dataset name.")
        return
    
    # Generate figures
    print("\n" + "="*60)
    print("Generating figures")
    print("="*60)
    
    # Figure 1: Strip plot (individual value plot)
    create_stripplot_figure(
        df_cv, args.dataset,
        output_dir / f"{args.dataset}_stripplot_cv.png"
    )
    
    # Figure 2: Learning curves (3 panels)
    create_learning_curves_figure(
        results_single, args.dataset,
        output_dir / f"{args.dataset}_learning_curves.png"
    )
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()