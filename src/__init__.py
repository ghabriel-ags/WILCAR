"""
Neural Network Methods for Regression with Gain Sign Constraints
================================================================
Federal University of Bahia – MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá
Advisors: Marcelo Embiruçu and Cristiano Fontes

Modules:
- base: Base classes and configurations
- wilcar: WILCAR methods (1–2)
- rixm: RIXM methods (3–4)
- elm: ELM methods (5–6)
- utils: Utilities (data, metrics, visualization)
"""

from .base import (
    TrainingConfig,
    TrainingResults,
    DatasetConfig,
    BaseNeuralNetwork,
    check_gpu_availability,
    TORCH_AVAILABLE
)

from .wilcar import (
    WILCARUnconstrained,
    WILCARConstrained,
    create_wilcar_unconstrained,
    create_wilcar_constrained
)

from .rixm import (
    RIXM,
    RIXMConstrained,
    create_rixm,
    create_rixm_constrained
)

from .elm import (
    ELM,
    ELMConstrained,
    create_elm,
    create_elm_constrained
)

from .utils import (
    # Data
    load_dataset,
    load_raw_data,
    preprocess_data,
    get_expected_signals,
    save_dataset_config,
    save_processed_data,
    load_processed_data,
    EXPECTED_SIGNALS_THEORY,
    
    # Metrics
    calculate_mse,
    calculate_rmse,
    calculate_r2,
    calculate_all_metrics,
    
    # Visualization
    plot_performance_dashboard,
    plot_gains_comparison,
    plot_learning_curves,
    print_summary,
    print_conformity_analysis,
    
    # Visualization - Comparative
    METHOD_INFO,
    plot_comparative_dashboard,
    plot_paired_comparison,
    create_summary_table,
    export_latex_table
)

__version__ = "0.2.0"
__author__ = "Ghabriel Anton Gomes de Sá"

__all__ = [
    # Base
    'TrainingConfig', 'TrainingResults', 'DatasetConfig',
    'BaseNeuralNetwork', 'check_gpu_availability', 'TORCH_AVAILABLE',
    
    # WILCAR (Methods 1-2)
    'WILCARUnconstrained', 'create_wilcar_unconstrained',
    'WILCARConstrained', 'create_wilcar_constrained',
    
    # RIXM (Methods 3-4)
    'RIXM', 'create_rixm',
    'RIXMConstrained', 'create_rixm_constrained',
    
    # ELM (Methods 5-6)
    'ELM', 'create_elm',
    'ELMConstrained', 'create_elm_constrained',
    
    # Data
    'load_dataset', 'load_raw_data', 'preprocess_data', 'get_expected_signals',
    'save_dataset_config', 'save_processed_data', 'load_processed_data',
    'EXPECTED_SIGNALS_THEORY',
    
    # Metrics
    'calculate_mse', 'calculate_rmse', 'calculate_r2', 'calculate_all_metrics',
    
    # Visualization
    'plot_performance_dashboard', 'plot_gains_comparison',
    'plot_learning_curves', 'print_summary', 'print_conformity_analysis'
    
    # Visualization - Comparative
    'METHOD_INFO', 'plot_comparative_dashboard',
    'plot_paired_comparison', 'create_summary_table',
    'export_latex_table'
]