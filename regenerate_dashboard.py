#!/usr/bin/env python3
"""
Regenerate Dashboards - Recreate comparative plots from saved results
=====================================================================
Federal University of Bahia – MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá

This script regenerates comparative dashboards from existing batch results.
Dashboards always show full range (R²: 0-1, all neurons).
Zoom options (--r2-ylim, --neurons-xlim) generate a separate r2_zoomed.png/pdf file.

Usage:
    # Basic usage (full range dashboards)
    python regenerate_dashboard.py --results-dir results/computer_hardware/batch_20251201_143022
    
    # With R² zoom (generates additional r2_zoomed.png/pdf)
    python regenerate_dashboard.py --results-dir results/computer_hardware/batch_20251201_143022 \\
        --r2-ylim 0.75 0.92
    
    # With neurons zoom (generates additional r2_zoomed.png/pdf)
    python regenerate_dashboard.py --results-dir results/computer_hardware/batch_20251201_143022 \\
        --neurons-xlim 1 50
"""

import sys
import argparse
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from base import TrainingResults
from utils.visualization import (
    METHOD_INFO,
    plot_comparative_dashboard,
    plot_paired_comparison,
    plot_r2_zoomed,
    plot_performance_dashboard,
    create_summary_table,
    export_latex_table,
)

import matplotlib.pyplot as plt


def check_extrapolation(results: TrainingResults) -> bool:
    """
    Check if any metric values extrapolate beyond their valid range.
    
    Ranges:
    - MSE, RMSE, R²: [0, 1]
    - Overfitting gap (R² train - R² test): [-1, 1]
    
    Returns:
        True if any value extrapolates its valid range
    """
    # Check MSE [0, 1]
    mse_values = np.concatenate([results.mse_train, results.mse_test])
    if np.any(mse_values < 0) or np.any(mse_values > 1):
        return True
    
    # Check RMSE [0, 1]
    rmse_values = np.concatenate([results.rmse_train, results.rmse_test])
    if np.any(rmse_values < 0) or np.any(rmse_values > 1):
        return True
    
    # Check R² [0, 1]
    r2_values = np.concatenate([results.r2_train, results.r2_test])
    if np.any(r2_values < 0) or np.any(r2_values > 1):
        return True
    
    # Check Overfitting gap [-1, 1]
    gap = np.array(results.r2_train) - np.array(results.r2_test)
    if np.any(gap < -1) or np.any(gap > 1):
        return True
    
    return False


def load_results_from_batch(batch_dir: Path) -> Tuple[Dict[int, TrainingResults], List[int], str, List[int]]:
    """
    Load all results from a batch directory.
    
    Args:
        batch_dir: Path to batch results directory
        
    Returns:
        Tuple of (results_dict, method_ids, dataset_name, expected_signals)
    """
    print(f"\n📂 Loading results from: {batch_dir}")
    
    # Load batch config
    config_file = batch_dir / "batch_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"batch_config.json not found in {batch_dir}")
    
    with open(config_file, 'r') as f:
        batch_config = json.load(f)
    
    dataset_name = batch_config.get('dataset', 'Unknown')
    methods = batch_config.get('methods', [1, 2, 3, 4, 5, 6])
    expected_signals = batch_config.get('expected_signals', [])
    
    print(f"   Dataset: {dataset_name}")
    print(f"   Methods: {methods}")
    
    # Load results for each method
    all_results = {}
    
    for method_id in methods:
        info = METHOD_INFO.get(method_id, {})
        method_name = info.get('name', f'method_{method_id}')
        method_dir = batch_dir / f"method_{method_id}_{method_name}"
        
        # Try to load results.pkl
        pkl_path = method_dir / "logs" / "results.pkl"
        
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                results = pickle.load(f)
            all_results[method_id] = results
            print(f"   ✓ Method {method_id}: Loaded ({results.best_neurons} neurons, R²={results.best_test_r2:.4f})")
        else:
            print(f"   ⚠ Method {method_id}: results.pkl not found")
    
    return all_results, methods, dataset_name, expected_signals


def regenerate_dashboards(
    batch_dir: Path,
    output_dir: Optional[Path] = None,
    r2_ylim: Optional[Tuple[float, float]] = None,
    neurons_xlim: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 150
):
    """
    Regenerate all comparative dashboards with custom settings.
    
    Zoom behavior:
    - r2_ylim: Applied to dashboard R² plot AND generates separate zoomed file
    - neurons_xlim: Only applied to separate zoomed file (dashboard uses full range)
    
    Individual dashboards are regenerated only when metrics extrapolate [0, 1].
    
    Args:
        batch_dir: Path to batch results directory
        output_dir: Output directory (default: batch_dir/comparative_custom)
        r2_ylim: Y-axis limits for R² plot (min, max) - applied to dashboard + separate file
        neurons_xlim: X-axis limits for zoomed plot (min, max) - separate file only
        figsize: Figure size
        dpi: DPI for saved figures
    """
    batch_dir = Path(batch_dir)
    
    # Load results
    all_results, methods, dataset_name, expected_signals = load_results_from_batch(batch_dir)
    
    if not all_results:
        print("\n❌ No results found!")
        return
    
    # Set output directory for comparative dashboards
    if output_dir is None:
        output_dir = batch_dir / "comparative_custom"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Regenerating dashboards...")
    print(f"   Output: {output_dir}")
    if r2_ylim:
        print(f"   R² zoom: {r2_ylim[0]:.2f} - {r2_ylim[1]:.2f} (applied to dashboard + separate file)")
    if neurons_xlim:
        print(f"   Neurons zoom: {neurons_xlim[0]} - {neurons_xlim[1]} (separate file only)")
    
    # 1. All methods comparison
    # R² zoom is applied to dashboard; neurons zoom only to separate file
    print("\n1. All methods dashboard...")
    fig = plot_comparative_dashboard(
        all_results,
        methods,
        dataset_name,
        save_path=None,
        title="All Methods Comparison",
        r2_ylim=r2_ylim,  # Apply R² zoom to dashboard
        figsize=figsize
    )
    if fig:
        fig.savefig(output_dir / "all_methods_dashboard.png", dpi=dpi, bbox_inches='tight')
        fig.savefig(output_dir / "all_methods_dashboard.pdf", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("   ✓ Saved all_methods_dashboard.png/pdf")
    
    # 1b. R² zoomed plot (separate file, only if neurons_xlim specified)
    # This allows seeing the neurons zoom which is not applied to dashboard
    if neurons_xlim:
        print("\n1b. R² zoomed plot (separate, with neurons zoom)...")
        # Build title based on what's zoomed
        zoom_parts = []
        if r2_ylim:
            zoom_parts.append(f"R²: {r2_ylim[0]:.2f}-{r2_ylim[1]:.2f}")
        zoom_parts.append(f"Neurons: {neurons_xlim[0]}-{neurons_xlim[1]}")
        zoom_title = f"Test R² vs Network Size (Zoomed: {', '.join(zoom_parts)})"
        
        fig = plot_r2_zoomed(
            all_results,
            methods,
            dataset_name,
            r2_ylim=r2_ylim if r2_ylim else (0, 1),
            save_path=None,
            neurons_xlim=neurons_xlim,
            title=zoom_title
        )
        if fig:
            fig.savefig(output_dir / "r2_zoomed.png", dpi=dpi, bbox_inches='tight')
            fig.savefig(output_dir / "r2_zoomed.pdf", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("   ✓ Saved r2_zoomed.png/pdf")
    
    # 2. Paired comparisons (1-2, 3-4, 5-6) - without zoom
    print("\n2. Paired comparisons...")
    pairs = [(1, 2, 'WILCAR'), (3, 4, 'RIXM'), (5, 6, 'ELM')]
    for m_a, m_b, family in pairs:
        if m_a in all_results and m_b in all_results:
            fig = plot_paired_comparison(
                all_results, m_a, m_b, dataset_name,
                save_path=None,
                title=f"{family}: Unconstrained vs Constrained",
                figsize=figsize
            )
            if fig:
                filename = f"paired_{family.lower()}_{m_a}_{m_b}"
                fig.savefig(output_dir / f"{filename}.png", dpi=dpi, bbox_inches='tight')
                fig.savefig(output_dir / f"{filename}.pdf", dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"   ✓ Saved {filename}.png/pdf")
    
    # 3. Unconstrained methods
    print("\n3. Unconstrained methods...")
    unconstrained = [m for m in [1, 3, 5] if m in all_results]
    if len(unconstrained) > 1:
        fig = plot_comparative_dashboard(
            all_results,
            unconstrained,
            dataset_name,
            save_path=None,
            title="Unconstrained Methods",
            r2_ylim=r2_ylim,
            figsize=figsize
        )
        if fig:
            fig.savefig(output_dir / "unconstrained_1_3_5.png", dpi=dpi, bbox_inches='tight')
            fig.savefig(output_dir / "unconstrained_1_3_5.pdf", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("   ✓ Saved unconstrained_1_3_5.png/pdf")
    
    # 4. Constrained methods
    print("\n4. Constrained methods...")
    constrained = [m for m in [2, 4, 6] if m in all_results]
    if len(constrained) > 1:
        fig = plot_comparative_dashboard(
            all_results,
            constrained,
            dataset_name,
            save_path=None,
            title="Constrained Methods (100% SCR)",
            r2_ylim=r2_ylim,
            figsize=figsize
        )
        if fig:
            fig.savefig(output_dir / "constrained_2_4_6.png", dpi=dpi, bbox_inches='tight')
            fig.savefig(output_dir / "constrained_2_4_6.pdf", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("   ✓ Saved constrained_2_4_6.png/pdf")
    
    # 5. Summary table
    print("\n5. Summary table...")
    df = create_summary_table(
        all_results,
        methods,
        dataset_name,
        output_dir / "summary_table.csv"
    )
    print("   ✓ Saved summary_table.csv")
    
    # 6. LaTeX table
    print("\n6. LaTeX table...")
    export_latex_table(
        all_results,
        methods,
        dataset_name,
        output_dir / "results_table.tex"
    )
    print("   ✓ Saved results_table.tex")
    
    # 7. Individual dashboards (regenerate all with updated format)
    print("\n7. Regenerating individual dashboards...")
    regenerated_count = 0
    
    for method_id in methods:
        if method_id not in all_results:
            continue
            
        results = all_results[method_id]
        info = METHOD_INFO.get(method_id, {})
        method_name = info.get('name', f'method_{method_id}')
        method_dir = batch_dir / f"method_{method_id}_{method_name}"
        
        # Get expected signals from conformity_details if not provided
        method_expected_signals = expected_signals
        if not method_expected_signals and results.conformity_details:
            method_expected_signals = [d['expected_signal'] for d in results.conformity_details]
        
        # Skip if we still don't have expected signals
        if not method_expected_signals:
            print(f"   ⚠ M{method_id}: Cannot regenerate (no expected_signals available)")
            continue
        
        # Check if extrapolation occurs (for info only)
        has_extrapolation = check_extrapolation(results)
        extrapolation_note = " (with axis clamping)" if has_extrapolation else ""
        
        # Regenerate performance dashboard
        fig = plot_performance_dashboard(
            results,
            method_expected_signals,
            method_name=info.get('short', f"M{method_id}"),
            dataset_name=dataset_name,
            save_path=method_dir / "figures" / "performance_dashboard.png",
            show=False
        )
        if fig:
            plt.close(fig)
        
        print(f"   ✓ M{method_id}: Regenerated{extrapolation_note}")
        regenerated_count += 1
    
    print(f"\n✅ Dashboards regenerated successfully!")
    print(f"   Comparative: {output_dir}")
    print(f"   Individual: {regenerated_count} method(s) regenerated in original folders")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate comparative dashboards with custom settings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic regeneration (all dashboards with full range)
    python regenerate_dashboard.py --results-dir results/computer_hardware/batch_20251201_143022
    
    # With R² zoom (applied to dashboard + separate r2_zoomed file)
    python regenerate_dashboard.py --results-dir results/computer_hardware/batch_20251201_143022 \\
        --r2-ylim 0.75 0.92
    
    # With neurons zoom (separate r2_zoomed file only)
    python regenerate_dashboard.py --results-dir results/computer_hardware/batch_20251201_143022 \\
        --neurons-xlim 1 100
    
    # Combined zoom (R² on dashboard + both on separate file)
    python regenerate_dashboard.py --results-dir results/computer_hardware/batch_20251201_143022 \\
        --r2-ylim 0.8 0.9 --neurons-xlim 1 50
        """
    )
    
    parser.add_argument('--results-dir', '-r', type=str, required=True,
                        help='Path to batch results directory')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory (default: results-dir/comparative_custom)')
    parser.add_argument('--r2-ylim', type=float, nargs=2, default=None,
                        metavar=('MIN', 'MAX'),
                        help='R² y-axis limits (e.g., 0.75 0.92) - applied to dashboard + separate file')
    parser.add_argument('--neurons-xlim', type=int, nargs=2, default=None,
                        metavar=('MIN', 'MAX'),
                        help='Neurons x-axis limits (e.g., 1 100) - separate file only')
    parser.add_argument('--figsize', type=int, nargs=2, default=[14, 10],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Figure size in inches (default: 14 10)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved figures (default: 150)')
    
    args = parser.parse_args()
    
    # Convert arguments
    r2_ylim = tuple(args.r2_ylim) if args.r2_ylim else None
    neurons_xlim = tuple(args.neurons_xlim) if args.neurons_xlim else None
    figsize = tuple(args.figsize)
    
    # Run
    regenerate_dashboards(
        batch_dir=args.results_dir,
        output_dir=args.output_dir,
        r2_ylim=r2_ylim,
        neurons_xlim=neurons_xlim,
        figsize=figsize,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()