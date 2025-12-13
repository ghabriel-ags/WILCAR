"""
Visualization Module - Standardized plots for results analysis
===============================================================
Federal University of Bahia – MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá

All labels and texts in English for academic papers.
Includes both individual method dashboards and comparative visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import pandas as pd


# =============================================================================
# METHOD CONFIGURATION (for comparative plots)
# =============================================================================

METHOD_INFO = {
    1: {
        'name': 'wilcar_unconstrained',
        'display': 'WILCAR Unconstrained',
        'short': 'M1: WILCAR',
        'family': 'WILCAR',
        'constrained': False,
        'color': '#2196F3',  # Blue
        'marker': 'o'
    },
    2: {
        'name': 'wilcar_constrained',
        'display': 'WILCAR Constrained',
        'short': 'M2: WILCAR+R',
        'family': 'WILCAR',
        'constrained': True,
        'color': '#1565C0',  # Dark Blue
        'marker': 's'
    },
    3: {
        'name': 'rixm',
        'display': 'RIXM',
        'short': 'M3: RIXM',
        'family': 'RIXM',
        'constrained': False,
        'color': '#4CAF50',  # Green
        'marker': '^'
    },
    4: {
        'name': 'rixm_constrained',
        'display': 'RIXM Constrained',
        'short': 'M4: RIXM+R',
        'family': 'RIXM',
        'constrained': True,
        'color': '#2E7D32',  # Dark Green
        'marker': 'v'
    },
    5: {
        'name': 'elm',
        'display': 'ELM',
        'short': 'M5: ELM',
        'family': 'ELM',
        'constrained': False,
        'color': '#FF9800',  # Orange
        'marker': 'D'
    },
    6: {
        'name': 'elm_constrained',
        'display': 'ELM Constrained',
        'short': 'M6: ELM+R',
        'family': 'ELM',
        'constrained': True,
        'color': '#E65100',  # Dark Orange
        'marker': 'p'
    }
}


# =============================================================================
# PLOT STYLE SETUP
# =============================================================================

def setup_plot_style():
    """Configure default plot style."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11


def _set_neuron_xlim(ax, n_neurons: int):
    """
    Set X-axis limits with proper margins for neuron plots.
    Ensures first point (n=1) is visible with small left margin,
    ticks start at 1 (not 0), and last value is shown without overlap.
    """
    margin = max(1, n_neurons * 0.02)  # 2% margin, minimum 1
    ax.set_xlim([1 - margin, n_neurons + margin])
    
    # Define appropriate tick positions starting from 1
    if n_neurons <= 10:
        ticks = list(range(1, n_neurons + 1))
    elif n_neurons <= 50:
        step = 5
        ticks = list(range(0, n_neurons + 1, step))
        ticks = [t if t > 0 else 1 for t in ticks]
    elif n_neurons <= 200:
        step = 25
        ticks = list(range(0, n_neurons + 1, step))
        ticks = [t if t > 0 else 1 for t in ticks]
    else:
        step = 100 if n_neurons <= 500 else 200
        ticks = list(range(0, n_neurons + 1, step))
        ticks = [t for t in ticks if t > 0]
    
    # Ensure 1 is first tick
    if not ticks or ticks[0] != 1:
        ticks = [1] + [t for t in ticks if t != 1]
    
    # Add last value (n_neurons) if not already present
    if ticks[-1] != n_neurons:
        # Check if last tick is too close to n_neurons (less than 75% of step)
        min_distance = step * 0.75 if n_neurons > 50 else 2
        if n_neurons - ticks[-1] <= min_distance:
            # Remove penultimate tick to avoid overlap, keep last value
            ticks[-1] = n_neurons
        else:
            ticks.append(n_neurons)
    
    ax.set_xticks(ticks)


def _clamp_ylim_if_extrapolating(ax, *data_arrays, ymin_bound: float = 0, ymax_bound: float = 1):
    """
    Clamp y-axis limits to bounds if data extrapolates beyond them.
    
    If any data point is outside [ymin_bound, ymax_bound], sets the axis
    limits to the full range [ymin_bound, ymax_bound].
    If all data is within bounds, matplotlib auto-scales.
    
    Args:
        ax: matplotlib axis
        *data_arrays: arrays of data values to check
        ymin_bound: lower bound (default 0)
        ymax_bound: upper bound (default 1)
    """
    all_data = np.concatenate([np.asarray(d) for d in data_arrays])
    data_min = np.nanmin(all_data)
    data_max = np.nanmax(all_data)
    
    # Check for extrapolation in either direction
    extrapolates = (data_min < ymin_bound) or (data_max > ymax_bound)
    
    # If any extrapolation, clamp to full bounds
    if extrapolates:
        ax.set_ylim([ymin_bound, ymax_bound])


# =============================================================================
# INDIVIDUAL METHOD DASHBOARD
# =============================================================================

def plot_performance_dashboard(results, expected_signals: List[int],
                               method_name: str = "Method",
                               dataset_name: str = "",
                               save_path: Optional[Path] = None,
                               show: bool = True) -> plt.Figure:
    """
    Generate complete performance dashboard for a single method.

    Args:
        results: TrainingResults with metrics
        expected_signals: List of expected gain signals
        method_name: Method name for title
        dataset_name: Dataset name for title
        save_path: Path to save figure (optional)
        show: If True, display figure

    Returns:
        matplotlib Figure
    """
    setup_plot_style()

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Data
    n_models = len(results.r2_test)
    neurons = np.arange(1, n_models + 1)
    best_n = results.best_neurons

    # Overfitting gap (Train R² - Test R²)
    gap_r2 = np.array(results.r2_train) - np.array(results.r2_test)

    # === ROW 1: MSE, RMSE and R² =================================================

    # MSE
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(neurons, results.mse_train, 'o-', label='Train', alpha=0.7, markersize=4)
    ax1.plot(neurons, results.mse_test, 's-', label='Test', alpha=0.7, markersize=4)
    _set_neuron_xlim(ax1, n_models)
    ax1.axvline(best_n, color='red', linestyle='--', alpha=0.5, label=f'Best (n={best_n})')
    ax1.set_xlabel('Hidden Neurons')
    ax1.set_ylabel('MSE')
    ax1.set_title('Mean Squared Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    _clamp_ylim_if_extrapolating(ax1, results.mse_train, results.mse_test)

    # RMSE
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(neurons, results.rmse_train, 'o-', label='Train', alpha=0.7, markersize=4)
    ax2.plot(neurons, results.rmse_test, 's-', label='Test', alpha=0.7, markersize=4)
    _set_neuron_xlim(ax2, n_models)
    ax2.axvline(best_n, color='red', linestyle='--', alpha=0.5, label=f'Best (n={best_n})')
    ax2.set_xlabel('Hidden Neurons')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Root Mean Squared Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    _clamp_ylim_if_extrapolating(ax2, results.rmse_train, results.rmse_test)

    # R²
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(neurons, results.r2_train, 'o-', label='Train', alpha=0.7, markersize=4)
    ax3.plot(neurons, results.r2_test, 's-', label='Test', alpha=0.7, markersize=4)
    _set_neuron_xlim(ax3, n_models)
    ax3.axvline(best_n, color='red', linestyle='--', alpha=0.5, label=f'Best (n={best_n})')
    ax3.set_xlabel('Hidden Neurons')
    ax3.set_ylabel('R²')
    ax3.set_title('Coefficient of Determination R²')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    _clamp_ylim_if_extrapolating(ax3, results.r2_train, results.r2_test)

    # === ROW 2: Conformity, Overfitting and Time ================================

    # Signal Conformity Rate
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(neurons, results.conformity_rates, 'o-', color='green', markersize=4)
    _set_neuron_xlim(ax4, n_models)
    ax4.axvline(best_n, color='red', linestyle='--', alpha=0.5, label=f'Best (n={best_n})')
    ax4.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='100%')
    ax4.set_xlabel('Hidden Neurons')
    ax4.set_ylabel('Conformity Rate')
    ax4.set_title('Signal Conformity Rate Evolution')
    ax4.set_ylim([0, 1.05])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Train-Test Gap (Overfitting)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(neurons, gap_r2, 'o-', color='orange', markersize=4)
    _set_neuron_xlim(ax5, n_models)
    ax5.axvline(best_n, color='red', linestyle='--', alpha=0.5, label=f'Best (n={best_n})')
    ax5.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax5.set_xlabel('Hidden Neurons')
    ax5.set_ylabel('R² Gap (Train - Test)')
    ax5.set_title('Overfitting Analysis')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    _clamp_ylim_if_extrapolating(ax5, gap_r2, ymin_bound=-1, ymax_bound=1)

    # Training Time per Model
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(neurons, results.iteration_times, 'o-', color='purple', markersize=4)
    _set_neuron_xlim(ax6, n_models)
    ax6.axvline(best_n, color='red', linestyle='--', alpha=0.5, label=f'Best (n={best_n})')
    ax6.set_xlabel('Hidden Neurons')
    ax6.set_ylabel('Time (s)')
    ax6.set_title('Training Time per Model')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # === ROW 3: Gain Analysis ===================================================

    # Calculated vs Expected Gains
    ax7 = fig.add_subplot(gs[2, :2])
    n_inputs = len(expected_signals)
    var_names = [f'Var {i+1}' for i in range(n_inputs)]
    x_pos = np.arange(n_inputs)

    if results.conformity_details:
        # Colors based on conformity
        colors = [
            'green' if (detail['calculated_signal'] == detail['expected_signal'] or
                        detail['expected_signal'] == 0)
            else 'red'
            for detail in results.conformity_details
        ]

        gains = [d['calculated_gain'] for d in results.conformity_details]
        ax7.bar(x_pos, gains, color=colors, alpha=0.7,
                edgecolor='black', linewidth=1)

        # Expected signal line
        expected_for_plot = [s if s != 0 else np.nan for s in expected_signals]
        ax7.plot(x_pos, expected_for_plot, 'ko-', markersize=8, linewidth=2,
                 label='Expected Signal', zorder=5)

        # Annotations
        for i, detail in enumerate(results.conformity_details):
            if detail['expected_signal'] != 0:
                symbol = '✓' if detail['calculated_signal'] == detail['expected_signal'] else '✗'
                y_pos = detail['calculated_gain']
                va = 'bottom' if y_pos >= 0 else 'top'
                ax7.text(i, y_pos, symbol, ha='center', va=va,
                         fontsize=12, fontweight='bold')

    ax7.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax7.set_xlabel('Input Variable')
    ax7.set_ylabel('Gain')
    ax7.set_title(
        f'Gain Comparison - Best Model (n={best_n}, SCR={results.best_conformity:.1%})'
    )
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(var_names)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

    # Summary in last subplot
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    # Count conforming and diverging variables
    if results.conformity_details:
        n_conform = sum(
            1 for d in results.conformity_details
            if d['expected_signal'] != 0 and d['calculated_signal'] == d['expected_signal']
        )
        n_evaluated = sum(1 for d in results.conformity_details if d['expected_signal'] != 0)
        n_divergent = n_evaluated - n_conform
    else:
        n_conform = n_divergent = n_evaluated = 0

    overfitting_best = gap_r2[best_n - 1] if best_n <= len(gap_r2) else 0.0

    summary_text = f"""
EXECUTIVE SUMMARY
{method_name.upper()}

Best Model:
  * Neurons:   {best_n}
  * Test MSE:  {results.best_test_mse:.4f}
  * Test RMSE: {results.best_test_rmse:.4f}
  * Test R2:   {results.best_test_r2:.4f}

Conformity:
  * SCR: {results.best_conformity:.1%}
  * Conforming: {n_conform}/{n_evaluated}
  * Divergent: {n_divergent}/{n_evaluated}

Performance:
  * Time: {results.total_time:.1f}s
  * Models: {n_models}
  * Overfitting: {overfitting_best:.4f}
"""

    ax8.text(
        0.1, 0.30, summary_text, fontsize=11, family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
    )

    plt.suptitle(
        f'{method_name} - Complete Dashboard\nDataset: {dataset_name}' if dataset_name else f'{method_name} - Complete Dashboard',
        fontsize=16,
        fontweight='bold',
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_gains_comparison(conformity_details: List[Dict],
                          expected_signals: List[int],
                          n_neurons: int,
                          tcs: float,
                          method_name: str = "Method",
                          save_path: Optional[Path] = None,
                          show: bool = True) -> plt.Figure:
    """
    Generate specific gain comparison plot (for paper).
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    n_inputs = len(expected_signals)
    var_names = [f'Var {i+1}' for i in range(n_inputs)]
    x_pos = np.arange(n_inputs)

    colors = [
        'green' if (detail['calculated_signal'] == detail['expected_signal'] or
                    detail['expected_signal'] == 0)
        else 'red'
        for detail in conformity_details
    ]

    gains = [d['calculated_gain'] for d in conformity_details]
    ax.bar(x_pos, gains, color=colors, alpha=0.7,
           edgecolor='black', linewidth=1.5)

    expected_for_plot = [s if s != 0 else np.nan for s in expected_signals]
    ax.plot(x_pos, expected_for_plot, 'ko-', markersize=10, linewidth=2.5,
            label='Expected Signal', zorder=5)

    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Input Variable', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gain (∂ŷ/∂x)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Gain Conformity - {method_name} (n={n_neurons}, SCR={tcs:.1%})',
        fontsize=14,
        fontweight='bold',
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(var_names)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gains plot saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_learning_curves(results, method_name: str = "Method",
                         save_path: Optional[Path] = None,
                         show: bool = True) -> plt.Figure:
    """
    Generate learning and error curves (R² and RMSE vs number of neurons).
    """
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_models = len(results.r2_test)
    neurons = np.arange(1, n_models + 1)
    best_n = results.best_neurons

    # R²
    ax1 = axes[0]
    ax1.plot(neurons, results.r2_train, 'b-o', label='Train R²', markersize=4)
    ax1.plot(neurons, results.r2_test, 'r-s', label='Test R²', markersize=4)
    _set_neuron_xlim(ax1, n_models)
    ax1.axvline(best_n, color='green', linestyle='--', alpha=0.7,
                label=f'Best (n={best_n})')
    ax1.set_xlabel('Number of Hidden Neurons')
    ax1.set_ylabel('R² Score')
    ax1.set_title(f'{method_name} - Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)

    # RMSE
    ax2 = axes[1]
    ax2.plot(neurons, results.rmse_train, 'b-o', label='Train RMSE', markersize=4)
    ax2.plot(neurons, results.rmse_test, 'r-s', label='Test RMSE', markersize=4)
    _set_neuron_xlim(ax2, n_models)
    ax2.axvline(best_n, color='green', linestyle='--', alpha=0.7,
                label=f'Best (n={best_n})')
    ax2.set_xlabel('Number of Hidden Neurons')
    ax2.set_ylabel('RMSE')
    ax2.set_title(f'{method_name} - Error Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Learning curves saved: {save_path}")

    if show:
        plt.show()

    return fig


def print_summary(results, method_name: str = "Method"):
    """
    Print formatted results summary.
    """
    print("\n" + "=" * 80)
    print(f"SUMMARY - {method_name}".center(80))
    print("=" * 80)

    print(f"\n⏱️  EXECUTION TIME:")
    print(f"    Total time: {results.total_time:.2f}s ({results.total_time/60:.2f}min)")
    print(f"    Avg time/model: {results.avg_time_per_model:.2f}s")
    print(f"    Models trained: {len(results.r2_test)}")

    print(f"\n🏆 BEST MODEL:")
    print(f"    Hidden neurons: {results.best_neurons}")
    print(f"    " + "─" * 40)
    print(f"    TRAIN:")
    print(f"      MSE:  {results.best_train_mse:.6f}")
    print(f"      RMSE: {results.best_train_rmse:.6f}")
    print(f"      R²:   {results.best_train_r2:.6f}")
    print(f"    " + "─" * 40)
    print(f"    TEST:")
    print(f"      MSE:  {results.best_test_mse:.6f}")
    print(f"      RMSE: {results.best_test_rmse:.6f}")
    print(f"      R²:   {results.best_test_r2:.6f}")
    print(f"    " + "─" * 40)
    print(f"    SCR: {results.best_conformity:.2%}")

    print("=" * 80)


def print_conformity_analysis(conformity_details: List[Dict],
                              expected_signals: List[int]):
    """
    Print detailed conformity analysis.
    """
    print("\n📊 GAIN ANALYSIS:")
    print("=" * 80)
    print(f"\n{'Variable':<12} {'Calc. Gain':<14} {'Exp. Signal':<14} {'Status':<12}")
    print("-" * 80)

    n_conform = 0
    n_divergent = 0
    n_unknown = 0

    for i, detail in enumerate(conformity_details):
        calc_gain = detail['calculated_gain']
        calc_signal = detail['calculated_signal']
        exp_signal = detail['expected_signal']

        if exp_signal == 0:
            status = "Unknown"
            symbol = "?"
            n_unknown += 1
        elif calc_signal == exp_signal:
            status = "✓ Conforming"
            symbol = "✓"
            n_conform += 1
        else:
            status = "✗ Divergent"
            symbol = "✗"
            n_divergent += 1

        exp_str = f"{exp_signal:+d}" if exp_signal != 0 else "?"
        print(
            f"Var {i+1:<8} {calc_signal:>+2.0f} ({calc_gain:>+8.4f})  "
            f"{exp_str:>2}              "
            f"{symbol} {status:<12}"
        )

    n_evaluated = n_conform + n_divergent
    scr = n_conform / n_evaluated if n_evaluated > 0 else 0

    print("-" * 80)
    print(f"\n📈 CONFORMITY SUMMARY:")
    print(f"    Conforming:  {n_conform}/{n_evaluated} ({scr:.1%})")
    print(f"    Divergent:   {n_divergent}/{n_evaluated}")
    if n_unknown > 0:
        print(f"    Unknown:     {n_unknown}")
    print(f"    " + "─" * 40)
    print(f"    SCR (Signal Conformity Rate): {scr:.2%}")
    print("=" * 80)


# =============================================================================
# COMPARATIVE DASHBOARD (ALL METHODS)
# =============================================================================

def plot_comparative_dashboard(
    all_results: Dict[int, Any],
    method_ids: List[int],
    dataset_name: str,
    save_path: Optional[Path] = None,
    title: str = "Comparative Dashboard",
    show: bool = False,
    r2_ylim: Optional[tuple] = None,
    figsize: tuple = (16, 12)
) -> plt.Figure:
    """
    Create comparative dashboard for multiple methods.
    
    Args:
        all_results: Dict mapping method_id -> TrainingResults
        method_ids: List of method IDs to include
        dataset_name: Dataset name for title
        save_path: Path to save figure (optional)
        title: Dashboard title
        show: Whether to display figure
        r2_ylim: Y-axis limits for R² plot (min, max). If None, uses (0, 1)
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    if not all_results:
        print("⚠️ No results to plot")
        return None
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Find max neurons for consistent x-axis
    max_neurons = max(
        len(all_results[mid].r2_test) 
        for mid in method_ids 
        if mid in all_results and all_results[mid] is not None
    )
    
    valid_ids = [mid for mid in method_ids if mid in all_results and all_results[mid] is not None]
    
    # 1. R² vs Neurons (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    for mid in valid_ids:
        results = all_results[mid]
        info = METHOD_INFO[mid]
        neurons = list(range(1, len(results.r2_test) + 1))
        ax1.plot(neurons, results.r2_test, 
                label=info['short'], 
                color=info['color'],
                marker=info['marker'],
                markersize=4,
                markevery=max(1, len(neurons)//20),
                linewidth=1.5)
    
    ax1.set_xlabel('Number of Neurons', fontsize=11)
    ax1.set_ylabel('Test R²', fontsize=11)
    ax1.set_title('Test R² vs Network Size', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)  # Optimized legend placement
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(r2_ylim if r2_ylim else [0, 1])
    _set_neuron_xlim(ax1, max_neurons)
    
    # 2. Full SCR Adherence Indicator (top right)
    # Shows whether ALL models achieved 100% SCR for each method (binary: Yes/No)
    ax2 = fig.add_subplot(gs[0, 1])
    
    x_pos = np.arange(len(valid_ids))
    labels = [f"M{mid}" for mid in valid_ids]
    
    # Check if ALL models had 100% SCR for each method
    full_adherence = []
    for mid in valid_ids:
        results = all_results[mid]
        scr_values = results.conformity_rates
        # True if ALL models achieved 100% SCR
        all_100 = all(s >= 0.9999 for s in scr_values)
        full_adherence.append(all_100)
    
    # Colors: green if full adherence, red if not
    colors = ['#4CAF50' if adh else '#F44336' for adh in full_adherence]
    
    # Create bars (height = 1 for visual consistency)
    bars = ax2.bar(x_pos, [1] * len(valid_ids), color=colors, edgecolor='black', linewidth=0.5)
    
    # Add checkmark or X on each bar
    for i, (bar, adh) in enumerate(zip(bars, full_adherence)):
        symbol = '✓' if adh else '✗'
        ax2.text(bar.get_x() + bar.get_width()/2, 0.5,
                symbol, ha='center', va='center', fontsize=24, fontweight='bold',
                color='white')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_title('SCR Adherence (Whole Training)', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1.2])
    ax2.set_xlim([-0.5, len(valid_ids) - 0.5])
    
    # Remove y-axis for clean look
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', edgecolor='black', label='100% Adherence'),
        Patch(facecolor='#F44336', edgecolor='black', label='< 100% Adherence')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # 3. Best Results Bar Chart (bottom left) - Clean design
    ax3 = fig.add_subplot(gs[1, 0])
    
    x_pos = np.arange(len(valid_ids))
    r2_values = [all_results[mid].best_test_r2 for mid in valid_ids]
    colors = [METHOD_INFO[mid]['color'] for mid in valid_ids]
    labels = [f"M{mid}" for mid in valid_ids]
    
    bars = ax3.bar(x_pos, r2_values, color=colors, edgecolor='black', linewidth=0.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, fontsize=11)
    
    # Dynamic ylim: start at 0, end slightly above max value + label space
    max_r2 = max(r2_values)
    ax3.set_ylim([0, max_r2 + 0.12])  # 0.12 = space for labels + margin
    ax3.set_xlim([-0.5, len(valid_ids) - 0.5])
    
    # Remove grid, spines, and ylabel for clean look
    ax3.grid(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.tick_params(left=False)
    ax3.set_yticks([])
    
    # Add value labels on bars
    for bar, val in zip(bars, r2_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Title with proper padding
    ax3.set_title('Best Test R² by Method', fontsize=12, fontweight='bold', pad=10)
    
    # 4. Summary Table (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Build table data
    table_data = []
    for mid in valid_ids:
        results = all_results[mid]
        table_data.append([
            f"M{mid}",
            f"{results.best_neurons}",
            f"{results.best_test_r2:.4f}",
            f"{results.best_test_rmse:.4f}",
            f"{results.best_conformity:.1%}",
            f"{results.total_time:.1f}s"
        ])
    
    columns = ['Method', 'Neurons', 'R²', 'RMSE', 'SCR', 'Time']
    
    table = ax4.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='upper center',
        colColours=['#E3F2FD'] * len(columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color code rows by method
    for i, mid in enumerate(valid_ids):
        for j in range(len(columns)):
            cell = table[(i + 1, j)]
            cell.set_facecolor(METHOD_INFO[mid]['color'] + '30')
    
    # Title above table
    ax4.set_title('Results Summary (Best Model for each Method)', fontsize=12, fontweight='bold', pad=10)
    
    # Main title
    fig.suptitle(f'{title}\nDataset: {dataset_name}', fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_r2_zoomed(
    all_results: Dict[int, Any],
    method_ids: List[int],
    dataset_name: str,
    r2_ylim: tuple,
    save_path: Optional[Path] = None,
    neurons_xlim: Optional[tuple] = None,
    title: str = "Test R² vs Network Size (Zoomed)",
    show: bool = False,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Create a standalone zoomed R² plot for detailed comparison.
    
    Args:
        all_results: Dict mapping method_id -> TrainingResults
        method_ids: List of method IDs to include
        dataset_name: Dataset name for title
        r2_ylim: Y-axis limits for R² plot (min, max) - REQUIRED
        save_path: Path to save figure (optional)
        neurons_xlim: X-axis limits for neuron plots (min, max)
        title: Plot title
        show: Whether to display figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    if not all_results:
        print("⚠️ No results to plot")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Find max neurons for consistent x-axis
    max_neurons = max(
        len(all_results[mid].r2_test) 
        for mid in method_ids 
        if mid in all_results and all_results[mid] is not None
    )
    
    valid_ids = [mid for mid in method_ids if mid in all_results and all_results[mid] is not None]
    
    for mid in valid_ids:
        results = all_results[mid]
        info = METHOD_INFO[mid]
        neurons = list(range(1, len(results.r2_test) + 1))
        ax.plot(neurons, results.r2_test, 
                label=info['short'], 
                color=info['color'],
                marker=info['marker'],
                markersize=5,
                markevery=max(1, len(neurons)//20),
                linewidth=2)
    
    ax.set_xlabel('Number of Neurons', fontsize=12)
    ax.set_ylabel('Test R²', fontsize=12)
    ax.set_title(f'{title}\nDataset: {dataset_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(r2_ylim)
    
    if neurons_xlim:
        ax.set_xlim(neurons_xlim)
    else:
        _set_neuron_xlim(ax, max_neurons)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# PAIRED COMPARISON (2 METHODS)
# =============================================================================

def plot_paired_comparison(
    all_results: Dict[int, Any],
    method_a: int,
    method_b: int,
    dataset_name: str,
    save_path: Optional[Path] = None,
    title: str = "Paired Comparison",
    show: bool = False,
    figsize: tuple = (14, 10)
) -> plt.Figure:
    """
    Create detailed comparison between two methods.
    
    Args:
        all_results: Dict mapping method_id -> TrainingResults
        method_a: First method ID
        method_b: Second method ID
        dataset_name: Dataset name for title
        save_path: Path to save figure (optional)
        title: Figure title
        show: Whether to display figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    if method_a not in all_results or method_b not in all_results:
        print(f"⚠️ Missing results for methods {method_a} and/or {method_b}")
        return None
    
    results_a = all_results[method_a]
    results_b = all_results[method_b]
    info_a = METHOD_INFO[method_a]
    info_b = METHOD_INFO[method_b]
    
    if results_a is None or results_b is None:
        print(f"⚠️ Null results for methods {method_a} and/or {method_b}")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Find max neurons for consistent x-axis
    max_neurons = max(len(results_a.r2_test), len(results_b.r2_test))
    
    # 1. R² comparison
    ax = axes[0, 0]
    neurons_a = list(range(1, len(results_a.r2_test) + 1))
    neurons_b = list(range(1, len(results_b.r2_test) + 1))
    
    ax.plot(neurons_a, results_a.r2_test, label=info_a['short'], 
            color=info_a['color'], marker=info_a['marker'], markersize=4,
            markevery=max(1, len(neurons_a)//15), linewidth=2)
    ax.plot(neurons_b, results_b.r2_test, label=info_b['short'],
            color=info_b['color'], marker=info_b['marker'], markersize=4,
            markevery=max(1, len(neurons_b)//15), linewidth=2)
    
    # Mark best points
    ax.scatter([results_a.best_neurons], [results_a.best_test_r2], 
               color=info_a['color'], s=150, zorder=5, edgecolor='black', linewidth=2)
    ax.scatter([results_b.best_neurons], [results_b.best_test_r2],
               color=info_b['color'], s=150, zorder=5, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Number of Neurons')
    ax.set_ylabel('Test R²')
    ax.set_title('Test R² Evolution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    _set_neuron_xlim(ax, max_neurons)
    
    # 2. SCR comparison (line chart - original format)
    ax = axes[0, 1]
    scr_a = [r * 100 for r in results_a.conformity_rates]
    scr_b = [r * 100 for r in results_b.conformity_rates]
    
    ax.plot(neurons_a, scr_a, label=info_a['short'],
            color=info_a['color'], marker=info_a['marker'], markersize=4,
            markevery=max(1, len(neurons_a)//15), linewidth=2)
    ax.plot(neurons_b, scr_b, label=info_b['short'],
            color=info_b['color'], marker=info_b['marker'], markersize=4,
            markevery=max(1, len(neurons_b)//15), linewidth=2)
    
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Number of Neurons')
    ax.set_ylabel('SCR (%)')
    ax.set_title('Signal Conformity Rate Evolution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    _set_neuron_xlim(ax, max_neurons)
    
    # 3. Training MSE comparison
    ax = axes[1, 0]
    ax.plot(neurons_a, results_a.mse_train, label=f'{info_a["short"]} (Train)',
            color=info_a['color'], linestyle='-', linewidth=2)
    ax.plot(neurons_a, results_a.mse_test, label=f'{info_a["short"]} (Test)',
            color=info_a['color'], linestyle='--', linewidth=2)
    ax.plot(neurons_b, results_b.mse_train, label=f'{info_b["short"]} (Train)',
            color=info_b['color'], linestyle='-', linewidth=2)
    ax.plot(neurons_b, results_b.mse_test, label=f'{info_b["short"]} (Test)',
            color=info_b['color'], linestyle='--', linewidth=2)
    
    ax.set_xlabel('Number of Neurons')
    ax.set_ylabel('MSE')
    ax.set_title('MSE Evolution (Train vs Test)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    _set_neuron_xlim(ax, max_neurons)
    
    # 4. Summary comparison table
    ax = axes[1, 1]
    ax.axis('off')
    
    metrics = ['Best Neurons', 'Best R²', 'Best RMSE', 'Best SCR', 'Total Time']
    values_a = [
        str(results_a.best_neurons),
        f"{results_a.best_test_r2:.4f}",
        f"{results_a.best_test_rmse:.4f}",
        f"{results_a.best_conformity:.1%}",
        f"{results_a.total_time:.1f}s"
    ]
    values_b = [
        str(results_b.best_neurons),
        f"{results_b.best_test_r2:.4f}",
        f"{results_b.best_test_rmse:.4f}",
        f"{results_b.best_conformity:.1%}",
        f"{results_b.total_time:.1f}s"
    ]
    
    # Determine winner for each metric
    winners = []
    for i, metric in enumerate(metrics):
        if metric == 'Best Neurons':
            winners.append('')  # No winner for neurons
        elif metric == 'Total Time':
            # Lower is better
            if results_a.total_time < results_b.total_time:
                winners.append('◀')
            elif results_b.total_time < results_a.total_time:
                winners.append('▶')
            else:
                winners.append('')
        elif metric == 'Best RMSE':
            # Lower is better
            if results_a.best_test_rmse < results_b.best_test_rmse:
                winners.append('◀')
            elif results_b.best_test_rmse < results_a.best_test_rmse:
                winners.append('▶')
            else:
                winners.append('')
        else:
            # Higher is better (R², SCR)
            val_a = results_a.best_test_r2 if 'R²' in metric else results_a.best_conformity
            val_b = results_b.best_test_r2 if 'R²' in metric else results_b.best_conformity
            if val_a > val_b:
                winners.append('◀')
            elif val_b > val_a:
                winners.append('▶')
            else:
                winners.append('')
    
    table_data = [[m, va, w, vb] for m, va, w, vb in zip(metrics, values_a, winners, values_b)]
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Metric', f'M{method_a}', '', f'M{method_b}'],
        cellLoc='center',
        loc='upper center',
        colColours=['#E3F2FD', info_a['color'] + '50', '#FFFFFF', info_b['color'] + '50'],
        colWidths=[0.3, 0.25, 0.1, 0.25]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 1.8)
    
    ax.set_title('Results Comparison', fontsize=12, fontweight='bold', pad=10)
    
    fig.suptitle(f'{title}\nDataset: {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# SUMMARY TABLE (CSV)
# =============================================================================

def create_summary_table(
    all_results: Dict[int, Any],
    method_ids: List[int],
    dataset_name: str,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Create CSV summary table of all results.
    
    Args:
        all_results: Dict mapping method_id -> TrainingResults
        method_ids: List of method IDs to include
        dataset_name: Dataset name
        save_path: Path to save CSV (optional)
    
    Returns:
        pandas DataFrame with summary
    """
    rows = []
    for mid in method_ids:
        if mid in all_results and all_results[mid] is not None:
            results = all_results[mid]
            info = METHOD_INFO[mid]
            rows.append({
                'Method_ID': mid,
                'Method_Name': info['display'],
                'Short_Name': info['short'],
                'Family': info['family'],
                'Constrained': info['constrained'],
                'Best_Neurons': results.best_neurons,
                'Best_Test_R2': results.best_test_r2,
                'Best_Test_MSE': results.best_test_mse,
                'Best_Test_RMSE': results.best_test_rmse,
                'Best_Train_R2': results.best_train_r2,
                'Best_SCR': results.best_conformity,
                'Total_Time_s': results.total_time,
                'Avg_Time_per_Model_s': results.avg_time_per_model,
                'Models_Trained': len(results.r2_test),
                'Dataset': dataset_name
            })
    
    df = pd.DataFrame(rows)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"  ✓ Saved: {save_path.name}")
    
    return df


# =============================================================================
# LATEX TABLE EXPORT
# =============================================================================

def export_latex_table(
    all_results: Dict[int, Any],
    method_ids: List[int],
    dataset_name: str,
    save_path: Optional[Path] = None
) -> str:
    """
    Export results as LaTeX table for academic papers.
    
    Args:
        all_results: Dict mapping method_id -> TrainingResults
        method_ids: List of method IDs
        dataset_name: Dataset name
        save_path: Path to save .tex file (optional)
    
    Returns:
        LaTeX table string
    """
    # Header
    latex = r"""\begin{table}[htbp]
\centering
\caption{Comparative Results - """ + dataset_name + r"""}
\label{tab:results_""" + dataset_name.lower().replace(' ', '_') + r"""}
\begin{tabular}{lcccccc}
\toprule
Method & Neurons & R² & RMSE & SCR (\%) & Time (s) \\
\midrule
"""
    
    # Find best values for highlighting
    valid_results = {mid: all_results[mid] for mid in method_ids 
                    if mid in all_results and all_results[mid] is not None}
    
    if valid_results:
        best_r2 = max(r.best_test_r2 for r in valid_results.values())
        best_rmse = min(r.best_test_rmse for r in valid_results.values())
        best_scr = max(r.best_conformity for r in valid_results.values())
    
    # Data rows
    for mid in method_ids:
        if mid in valid_results:
            results = valid_results[mid]
            info = METHOD_INFO[mid]
            
            # Format with bold for best values
            r2_str = f"\\textbf{{{results.best_test_r2:.4f}}}" if results.best_test_r2 == best_r2 else f"{results.best_test_r2:.4f}"
            rmse_str = f"\\textbf{{{results.best_test_rmse:.4f}}}" if results.best_test_rmse == best_rmse else f"{results.best_test_rmse:.4f}"
            scr_str = f"\\textbf{{{results.best_conformity*100:.1f}}}" if results.best_conformity == best_scr else f"{results.best_conformity*100:.1f}"
            
            latex += f"{info['short']} & {results.best_neurons} & {r2_str} & {rmse_str} & {scr_str} & {results.total_time:.1f} \\\\\n"
    
    # Footer
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex)
        print(f"  ✓ Saved: {save_path.name}")
    
    return latex