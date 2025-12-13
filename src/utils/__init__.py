"""
Utils Module - Helper functions for the project
=============================================
Federal University of Bahia – MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá
"""

from .metrics import (
    calculate_mse,
    calculate_rmse,
    calculate_r2,
    calculate_all_metrics,
    calculate_mae,
    calculate_mape
)

from .data import (
    EXPECTED_SIGNALS_THEORY,
    DATASET_MAPPING,
    detect_dataset_name,
    get_expected_signals,
    load_raw_data,
    preprocess_data,
    load_dataset,
    save_dataset_config,
    save_processed_data,
    load_processed_data,
    load_dataset_config
)

from .visualization import (
    # Individual
    setup_plot_style,
    plot_performance_dashboard,
    plot_gains_comparison,
    plot_learning_curves,
    print_summary,
    print_conformity_analysis,
    # Comparative
    METHOD_INFO,
    plot_comparative_dashboard,
    plot_paired_comparison,
    create_summary_table,
    export_latex_table
)

__all__ = [
    # Metrics
    'calculate_mse', 'calculate_rmse', 'calculate_r2', 
    'calculate_all_metrics', 'calculate_mae', 'calculate_mape',
    
    # Data
    'EXPECTED_SIGNALS_THEORY', 'DATASET_MAPPING',
    'detect_dataset_name', 'get_expected_signals',
    'load_raw_data', 'preprocess_data', 'load_dataset',
    'save_dataset_config', 'save_processed_data', 
    'load_processed_data', 'load_dataset_config',
    
    # Visualization - Individual
    'setup_plot_style', 'plot_performance_dashboard',
    'plot_gains_comparison', 'plot_learning_curves',
    'print_summary', 'print_conformity_analysis',
    
    # Visualization - Comparative
    'METHOD_INFO', 'plot_comparative_dashboard',
    'plot_paired_comparison', 'create_summary_table',
    'export_latex_table'
]