#!/usr/bin/env python3
"""
Generate Publication Figures from Experimental Results
=======================================================
Federal University of Bahia – MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá

This script generates figures for the manuscript:
1. Individual Value Plot (strip plot) of Test R² from 5-fold CV results
2. Learning curves by family (WILCAR | RIXM | ELM panels)
3. Learning curves by constraint (Unconstrained | Constrained panels)

Layout optimized for Neurocomputing / Neural Networks journals.

Usage:
    python generate_figures.py --dataset computer_hardware --batch-dir results/computer_hardware/batch_... --cv-results results/cv/fold_results.csv
    python generate_figures.py --dataset fish_toxicity --batch-dir ... --cv-results ...
    python generate_figures.py --dataset aquatic_toxicity --batch-dir ... --cv-results ...
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
import matplotlib as mpl
from typing import Dict

# =============================================================================
# PLOT STYLE CONFIGURATION (Journal-ready)
# =============================================================================

def setup_journal_style():
    """Configure matplotlib for journal-quality figures."""
    plt.rcParams.update({
        # Font
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        
        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        
        # Axes
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,
        
        # Figure
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # Legend
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
    })


# =============================================================================
# METHOD CONFIGURATION
# =============================================================================

# Display names
METHOD_NAMES = {
    1: 'WILCAR',
    2: 'WILCAR',
    3: 'RIXM',
    4: 'RIXM',
    5: 'ELM',
    6: 'ELM'
}

# Names with constraint indicator
METHOD_NAMES_FULL = {
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

# Colors by FAMILY (same color for Un and C variants)
# Colorblind-friendly palette
FAMILY_COLORS = {
    'WILCAR': '#0072B2',  # Blue
    'RIXM': '#009E73',    # Green
    'ELM': '#D55E00',     # Orange/Red
}

# Method to family mapping
METHOD_FAMILY = {
    1: 'WILCAR', 2: 'WILCAR',
    3: 'RIXM', 4: 'RIXM',
    5: 'ELM', 6: 'ELM'
}

# Markers for best point only
METHOD_MARKERS = {
    1: 'o', 2: 's',
    3: '^', 4: 'v', 
    5: 'D', 6: 'p'
}

# Colors for strip plot (by method)
METHOD_COLORS = {
    1: '#0072B2',  # Blue - WILCAR (Un)
    2: '#0072B2',  # Blue - WILCAR (C)
    3: '#009E73',  # Green - RIXM (Un)
    4: '#009E73',  # Green - RIXM (C)
    5: '#D55E00',  # Orange - ELM (Un)
    6: '#D55E00',  # Orange - ELM (C)
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_single_run_results(batch_dir: Path) -> Dict[int, object]:
    """Load TrainingResults from a single-run batch directory."""
    try:
        import base
    except ImportError:
        print("  ⚠️ Warning: Could not import base module. Pickle may fail.")
    
    results = {}
    
    for method_id, folder_name in METHOD_FOLDERS.items():
        pkl_path = batch_dir / folder_name / "logs" / "results.pkl"
        
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                results[method_id] = pickle.load(f)
            print(f"  ✓ Loaded {METHOD_NAMES_FULL[method_id]}: {len(results[method_id].r2_test)} neurons")
        else:
            print(f"  ✗ Not found: {pkl_path}")
    
    return results


def load_cv_results(cv_fold_results_path: Path) -> pd.DataFrame:
    """Load cross-validation fold results."""
    df = pd.read_csv(cv_fold_results_path)
    print(f"  ✓ Loaded CV results: {len(df)} rows ({df['fold'].nunique()} folds × {df['method_id'].nunique()} methods)")
    return df


# =============================================================================
# FIGURE 1: INDIVIDUAL VALUE PLOT (STRIP PLOT)
# =============================================================================

def create_stripplot_figure(df: pd.DataFrame, dataset_name: str, output_path: Path):
    """
    Create individual value plot (strip plot) of Test R² from cross-validation results.
    Each point represents one fold's result.
    """
    setup_journal_style()
    print(f"    Creating strip plot with {len(df)} rows")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Prepare data
    methods = sorted(df['method_id'].unique())
    labels = [METHOD_NAMES_FULL[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    
    x_positions = np.arange(len(methods))
    
    # Plot individual points for each method
    for i, method_id in enumerate(methods):
        method_data = df[df['method_id'] == method_id]['test_r2'].values
        
        # Add small jitter to x position
        np.random.seed(42 + i)  # Reproducible jitter
        jitter = np.random.uniform(-0.15, 0.15, size=len(method_data))
        x = np.full(len(method_data), i) + jitter
        
        # Plot individual points
        ax.scatter(x, method_data, color=colors[i], s=50, alpha=0.7, 
                  edgecolor='white', linewidth=0.5, zorder=3)
        
        # Plot mean as horizontal line
        mean_val = method_data.mean()
        ax.hlines(mean_val, i - 0.25, i + 0.25, color='black', linewidth=2, zorder=4)
    
    # Style
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.set_ylabel('Test R²')
    ax.set_ylim(0, 1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='-')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"    💾 Saved: {output_path}")
    print(f"    💾 Saved: {output_path.with_suffix('.pdf')}")


# =============================================================================
# FIGURE 3: STRIP PLOT - NUMBER OF HIDDEN UNITS (2 PANELS: Un | C)
# =============================================================================

def create_neurons_stripplot_figure(df: pd.DataFrame, dataset_name: str, output_path: Path):
    """
    Create strip plot of number of hidden units from cross-validation results.
    
    2 panels side by side:
    - Left: Unconstrained methods (M1, M3, M5)
    - Right: Constrained methods (M2, M4, M6)
    
    Each point represents the best number of neurons for one fold.
    This replaces the "Number of hidden units" column in the results table.
    """
    setup_journal_style()
    print(f"    Creating neurons strip plot with {len(df)} rows")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    unconstrained_ids = [1, 3, 5]
    constrained_ids = [2, 4, 6]
    
    # =========================================================================
    # LEFT PANEL: Unconstrained Methods
    # =========================================================================
    methods_un = [m for m in unconstrained_ids if m in df['method_id'].unique()]
    labels_un = [METHOD_NAMES_FULL[m] for m in methods_un]
    colors_un = [METHOD_COLORS[m] for m in methods_un]
    
    x_positions_un = np.arange(len(methods_un))
    
    for i, method_id in enumerate(methods_un):
        method_data = df[df['method_id'] == method_id]['best_neurons'].values
        
        # Add small jitter
        np.random.seed(42 + method_id)
        jitter = np.random.uniform(-0.15, 0.15, size=len(method_data))
        x = np.full(len(method_data), i) + jitter
        
        ax1.scatter(x, method_data, color=colors_un[i], s=50, alpha=0.7,
                   edgecolor='white', linewidth=0.5, zorder=3)
    
    ax1.set_xticks(x_positions_un)
    ax1.set_xticklabels(labels_un, rotation=0, ha='center')
    ax1.set_ylabel('Number of Hidden Units')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linestyle='-')
    
    # Dynamic y-axis for unconstrained (can be large)
    all_neurons_un = df[df['method_id'].isin(unconstrained_ids)]['best_neurons'].values
    if len(all_neurons_un) > 0:
        y_max_un = max(all_neurons_un) * 1.1
        ax1.set_ylim(0, y_max_un)
    
    # =========================================================================
    # RIGHT PANEL: Constrained Methods
    # =========================================================================
    methods_c = [m for m in constrained_ids if m in df['method_id'].unique()]
    labels_c = [METHOD_NAMES_FULL[m] for m in methods_c]
    colors_c = [METHOD_COLORS[m] for m in methods_c]
    
    x_positions_c = np.arange(len(methods_c))
    
    for i, method_id in enumerate(methods_c):
        method_data = df[df['method_id'] == method_id]['best_neurons'].values
        
        # Add small jitter
        np.random.seed(42 + method_id)
        jitter = np.random.uniform(-0.15, 0.15, size=len(method_data))
        x = np.full(len(method_data), i) + jitter
        
        ax2.scatter(x, method_data, color=colors_c[i], s=50, alpha=0.7,
                   edgecolor='white', linewidth=0.5, zorder=3)
    
    ax2.set_xticks(x_positions_c)
    ax2.set_xticklabels(labels_c, rotation=0, ha='center')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='-')
    
    # Dynamic y-axis for constrained (typically smaller)
    all_neurons_c = df[df['method_id'].isin(constrained_ids)]['best_neurons'].values
    if len(all_neurons_c) > 0:
        y_max_c = max(all_neurons_c) * 1.1
        ax2.set_ylim(0, y_max_c)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"    💾 Saved: {output_path}")
    print(f"    💾 Saved: {output_path.with_suffix('.pdf')}")


# =============================================================================
# FIGURE 2A: LEARNING CURVES BY FAMILY (3 PANELS)
# =============================================================================

def create_learning_curves_figure(results: Dict[int, object], dataset_name: str, 
                                   output_path: Path):
    """
    Create learning curves (Test R² vs Hidden Units) from single-run results.
    3 panels side by side: WILCAR, RIXM, ELM (each with Un/C variants).
    """
    setup_journal_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Define method groups
    method_groups = [
        {'name': 'WILCAR', 'methods': [1, 2]},
        {'name': 'RIXM', 'methods': [3, 4]},
        {'name': 'ELM', 'methods': [5, 6]},
    ]
    
    for ax_idx, group in enumerate(method_groups):
        ax = axes[ax_idx]
        family_color = FAMILY_COLORS[group['name']]
        
        # Determine max neurons from data
        max_n = 0
        for method_id in group['methods']:
            if method_id in results:
                max_n = max(max_n, len(results[method_id].r2_test))
        
        if max_n == 0:
            max_n = 100
        
        for method_id in group['methods']:
            if method_id not in results:
                continue
                
            r = results[method_id]
            neurons = np.arange(1, len(r.r2_test) + 1)
            
            # Determine line style (solid for Un, dashed for C)
            is_constrained = method_id in [2, 4, 6]
            linestyle = '--' if is_constrained else '-'
            linewidth = 1.5
            
            ax.plot(neurons, r.r2_test, 
                    label=METHOD_NAMES_FULL[method_id],
                    color=family_color,
                    linestyle=linestyle,
                    linewidth=linewidth)
            
            # Mark best model
            best_idx = r.best_neurons - 1
            if best_idx < len(r.r2_test):
                ax.scatter(r.best_neurons, r.r2_test[best_idx], 
                          s=80, color=family_color, 
                          marker=METHOD_MARKERS[method_id],
                          edgecolor='black', linewidth=1, zorder=5)
        
        # Style
        ax.set_xlabel('Hidden Units')
        if ax_idx == 0:
            ax.set_ylabel('Test R²')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.set_xlim(1, max_n)
        ax.set_ylim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"    💾 Saved: {output_path}")
    print(f"    💾 Saved: {output_path.with_suffix('.pdf')}")


# =============================================================================
# FIGURE 2B: LEARNING CURVES BY CONSTRAINT TYPE (2 PANELS) - JOURNAL QUALITY
# =============================================================================

def create_learning_curves_by_constraint(results: Dict[int, object], dataset_name: str,
                                          output_path: Path):
    """
    Create learning curves (Test R² vs Hidden Units) grouped by constraint type.
    
    2 panels side by side:
    - Left: Unconstrained methods (M1, M3, M5)
    - Right: Constrained methods (M2, M4, M6)
    
    Layout optimized for Neurocomputing / Neural Networks journals:
    - Consistent colors by family (WILCAR=blue, RIXM=green, ELM=orange)
    - Clean solid lines (no markers along curve)
    - Larger markers only at best configuration
    - Serif fonts, appropriate sizes
    - Light grid
    """
    setup_journal_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    
    unconstrained_ids = [1, 3, 5]
    constrained_ids = [2, 4, 6]
    
    # =========================================================================
    # LEFT PANEL: Unconstrained Methods
    # =========================================================================
    max_neurons_un = 0
    
    for method_id in unconstrained_ids:
        if method_id not in results:
            continue
            
        r = results[method_id]
        neurons = np.arange(1, len(r.r2_test) + 1)
        max_neurons_un = max(max_neurons_un, len(neurons))
        
        family = METHOD_FAMILY[method_id]
        color = FAMILY_COLORS[family]
        
        # Plot line (no markers along curve)
        ax1.plot(neurons, r.r2_test,
                label=METHOD_NAMES_FULL[method_id],
                color=color,
                linestyle='-',
                linewidth=1.8)
        
        # Mark best model with larger marker
        best_idx = r.best_neurons - 1
        if best_idx < len(r.r2_test):
            ax1.scatter(r.best_neurons, r.r2_test[best_idx],
                       s=100, 
                       color=color,
                       marker=METHOD_MARKERS[method_id],
                       edgecolor='black',
                       linewidth=1.2,
                       zorder=5)
    
    ax1.set_xlabel('Hidden Units')
    ax1.set_ylabel('Test R²')
    ax1.legend(loc='best', framealpha=0.95)
    ax1.set_xlim(0, max_neurons_un * 1.02)
    ax1.set_ylim(0, 1.0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # =========================================================================
    # RIGHT PANEL: Constrained Methods
    # =========================================================================
    max_neurons_c = 0
    
    for method_id in constrained_ids:
        if method_id not in results:
            continue
            
        r = results[method_id]
        neurons = np.arange(1, len(r.r2_test) + 1)
        max_neurons_c = max(max_neurons_c, len(neurons))
        
        family = METHOD_FAMILY[method_id]
        color = FAMILY_COLORS[family]
        
        # Plot line (no markers along curve)
        ax2.plot(neurons, r.r2_test,
                label=METHOD_NAMES_FULL[method_id],
                color=color,
                linestyle='-',
                linewidth=1.8)
        
        # Mark best model with larger marker
        best_idx = r.best_neurons - 1
        if best_idx < len(r.r2_test):
            ax2.scatter(r.best_neurons, r.r2_test[best_idx],
                       s=100,
                       color=color,
                       marker=METHOD_MARKERS[method_id],
                       edgecolor='black',
                       linewidth=1.2,
                       zorder=5)
    
    ax2.set_xlabel('Hidden Units')
    ax2.legend(loc='best', framealpha=0.95)
    ax2.set_xlim(0, max_neurons_c * 1.1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"    💾 Saved: {output_path}")
    print(f"    💾 Saved: {output_path.with_suffix('.pdf')}")


# =============================================================================
# COMBINED FIGURE (LEGACY)
# =============================================================================

def create_combined_figure(df_cv: pd.DataFrame, results_single: Dict[int, object],
                           dataset_name: str, output_path: Path, max_neurons: int = 150):
    """Create combined figure with strip plot and learning curves."""
    setup_journal_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Strip plot
    ax = axes[0]
    methods = sorted(df_cv['method_id'].unique())
    labels = [METHOD_NAMES_FULL[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    
    x_positions = np.arange(len(methods))
    
    for i, method_id in enumerate(methods):
        method_data = df_cv[df_cv['method_id'] == method_id]['test_r2'].values
        
        np.random.seed(42 + i)
        jitter = np.random.uniform(-0.12, 0.12, size=len(method_data))
        x = np.full(len(method_data), i) + jitter
        
        ax.scatter(x, method_data, color=colors[i], s=50, alpha=0.7,
                  edgecolor='white', linewidth=0.5, zorder=3)
        
        mean_val = method_data.mean()
        ax.hlines(mean_val, i - 0.2, i + 0.2, color='black', linewidth=2, zorder=4)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.set_ylabel('Test R²')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Panel B: Learning curves
    ax = axes[1]
    
    for method_id in sorted(results_single.keys()):
        r = results_single[method_id]
        neurons = np.arange(1, len(r.r2_test) + 1)
        
        mask = neurons <= max_neurons
        neurons_plot = neurons[mask]
        r2_plot = np.array(r.r2_test)[mask]
        
        if len(neurons_plot) == 0:
            continue
        
        family = METHOD_FAMILY[method_id]
        color = FAMILY_COLORS[family]
        is_constrained = method_id in [2, 4, 6]
        linestyle = '--' if is_constrained else '-'
        
        ax.plot(neurons_plot, r2_plot, 
                label=METHOD_NAMES_FULL[method_id],
                color=color,
                linestyle=linestyle,
                linewidth=1.5)
        
        if r.best_neurons <= max_neurons:
            best_idx = r.best_neurons - 1
            if best_idx < len(r.r2_test):
                ax.scatter(r.best_neurons, r.r2_test[best_idx], 
                          s=70, color=color, 
                          marker=METHOD_MARKERS[method_id],
                          edgecolor='black', linewidth=1, zorder=5)
    
    ax.set_xlabel('Hidden Units')
    ax.set_ylabel('Test R²')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(1, max_neurons)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"    💾 Saved: {output_path}")
    print(f"    💾 Saved: {output_path.with_suffix('.pdf')}")


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
    print(f"📊 Loading data for: {args.dataset}")
    print("="*60)
    
    batch_dir = Path(args.batch_dir)
    results_single = load_single_run_results(batch_dir)
    
    cv_path = Path(args.cv_results)
    df_cv = load_cv_results(cv_path)
    
    # Filter CV results for this dataset
    if 'dataset' in df_cv.columns:
        df_cv = df_cv[df_cv['dataset'] == args.dataset]
        print(f"  Filtered CV data: {len(df_cv)} rows")
    
    if len(df_cv) == 0:
        print("⚠️ ERROR: No CV data after filtering! Check dataset name.")
        return
    
    # Generate figures
    print("\n" + "="*60)
    print("📈 Generating figures")
    print("="*60)
    
    # Figure 1: Strip plot R² (CV results)
    print("\n  📊 Figure 1: Strip plot R² (CV results)")
    create_stripplot_figure(
        df_cv, args.dataset,
        output_dir / f"{args.dataset}_stripplot_cv.png"
    )
    
    # Figure 2: Learning curves by constraint type (2 panels) - MAIN FIGURE
    print("\n  📊 Figure 2: Learning curves by constraint (JOURNAL QUALITY)")
    create_learning_curves_by_constraint(
        results_single, args.dataset,
        output_dir / f"{args.dataset}_learning_curves_by_constraint.png"
    )
    
    # Figure 3: Strip plot number of neurons (2 panels: Un | C)
    print("\n  📊 Figure 3: Strip plot neurons (Un | C)")
    create_neurons_stripplot_figure(
        df_cv, args.dataset,
        output_dir / f"{args.dataset}_neurons_stripplot.png"
    )
    
    # Figure 2A (optional): Learning curves by family (3 panels)
    print("\n  📊 Figure (optional): Learning curves by family")
    create_learning_curves_figure(
        results_single, args.dataset,
        output_dir / f"{args.dataset}_learning_curves_by_family.png"
    )
    
    print("\n" + "="*60)
    print("✅ Done!")
    print("="*60)


if __name__ == "__main__":
    main()