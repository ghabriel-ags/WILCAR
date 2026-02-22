"""
Data Module - Functions for data loading and saving
====================================================
Federal University of Bahia – MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá

Expected Signals Convention:
- +1: Positive gain expected (output increases when input increases)
- -1: Negative gain expected (output decreases when input increases)
-  0: Unknown/unconstrained (no restriction applied)
- None: Same as 0, converted internally

Signal Loading Hierarchy:
1. custom_signals parameter (if provided) - always takes priority
2. JSON config file (data/raw/{filename}_config.json)
3. Interactive mode (prompts user for input)

When using partial constraints, only variables with known signs (+1 or -1)
will have constraints applied. Variables with 0 or None are free.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List, Union


# =============================================================================
# EXPECTED SIGNALS VALIDATION AND UTILITIES
# =============================================================================

def validate_expected_signals(signals: List[Union[int, None]], 
                              n_inputs: int) -> List[int]:
    """
    Validate and normalize expected signals list.
    
    Converts None to 0 and validates all values are in {-1, 0, 1, None}.
    
    Args:
        signals: List of expected gain signs (+1, -1, 0, or None)
        n_inputs: Expected number of input variables
    
    Returns:
        Normalized list with only -1, 0, +1 values
    
    Raises:
        ValueError: If signals list is invalid
    """
    if len(signals) != n_inputs:
        raise ValueError(
            f"Expected {n_inputs} signals, got {len(signals)}"
        )
    
    normalized = []
    for i, sig in enumerate(signals):
        if sig is None:
            normalized.append(0)
        elif sig in [-1, 0, 1]:
            normalized.append(int(sig))
        else:
            raise ValueError(
                f"Invalid signal value at index {i}: {sig}. "
                f"Expected -1, 0, 1, or None."
            )
    
    return normalized


def get_constraint_summary(signals: List[int]) -> Dict[str, Any]:
    """
    Get summary of constraint configuration.
    
    Args:
        signals: List of expected gain signs
    
    Returns:
        Dictionary with constraint statistics
    """
    n_total = len(signals)
    n_positive = sum(1 for s in signals if s == 1)
    n_negative = sum(1 for s in signals if s == -1)
    n_constrained = n_positive + n_negative
    n_unconstrained = sum(1 for s in signals if s == 0)
    
    constrained_vars = [i+1 for i, s in enumerate(signals) if s != 0]
    unconstrained_vars = [i+1 for i, s in enumerate(signals) if s == 0]
    
    return {
        'n_total': n_total,
        'n_constrained': n_constrained,
        'n_unconstrained': n_unconstrained,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'constrained_vars': constrained_vars,
        'unconstrained_vars': unconstrained_vars,
        'constraint_ratio': n_constrained / n_total if n_total > 0 else 0,
        'is_fully_constrained': n_unconstrained == 0,
        'is_partially_constrained': 0 < n_constrained < n_total,
        'is_unconstrained': n_constrained == 0
    }


def print_constraint_summary(signals: List[int], dataset_name: str = None):
    """
    Print a formatted summary of constraint configuration.
    
    Args:
        signals: List of expected gain signs
        dataset_name: Optional dataset name for display
    """
    summary = get_constraint_summary(signals)
    
    header = f"Constraint Configuration"
    if dataset_name:
        header += f" ({dataset_name})"
    
    print(f"\n{'='*50}")
    print(f"📋 {header}")
    print(f"{'='*50}")
    print(f"   Total variables: {summary['n_total']}")
    print(f"   Constrained: {summary['n_constrained']} "
          f"({summary['n_positive']} positive, {summary['n_negative']} negative)")
    print(f"   Unconstrained: {summary['n_unconstrained']}")
    
    if summary['is_fully_constrained']:
        print(f"   Mode: FULLY CONSTRAINED")
    elif summary['is_partially_constrained']:
        print(f"   Mode: PARTIALLY CONSTRAINED (hybrid)")
        print(f"   Constrained variables: {summary['constrained_vars']}")
        print(f"   Free variables: {summary['unconstrained_vars']}")
    else:
        print(f"   Mode: UNCONSTRAINED")
    
    print(f"   Signals: {signals}")
    print(f"{'='*50}\n")


# =============================================================================
# JSON CONFIGURATION FILE FUNCTIONS
# =============================================================================

def get_config_path(filename: str, base_dir: str = "..") -> Path:
    """
    Get the path to the dataset configuration JSON file.
    
    Args:
        filename: Dataset filename (without .csv)
        base_dir: Base directory
    
    Returns:
        Path to config JSON file
    """
    filename_clean = filename.lower().replace('.csv', '')
    return Path(base_dir) / "data" / "raw" / f"{filename_clean}_config.json"


def load_signals_from_json(filename: str, base_dir: str = "..") -> Optional[Dict[str, Any]]:
    """
    Load expected signals from JSON configuration file.
    
    Args:
        filename: Dataset filename
        base_dir: Base directory
    
    Returns:
        Configuration dict or None if file doesn't exist
    """
    config_path = get_config_path(filename, base_dir)
    
    if not config_path.exists():
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"✅ Configuration loaded from: {config_path}")
    return config


def save_signals_to_json(filename: str, signals: List[int], 
                         base_dir: str = "..",
                         variable_names: List[str] = None,
                         notes: str = None) -> Path:
    """
    Save expected signals to JSON configuration file.
    
    Args:
        filename: Dataset filename
        signals: List of expected gain signs
        base_dir: Base directory
        variable_names: Optional list of variable names
        notes: Optional notes about the configuration
    
    Returns:
        Path to saved config file
    """
    config_path = get_config_path(filename, base_dir)
    
    config = {
        'expected_signals': signals,
    }
    
    if variable_names:
        config['variable_names'] = variable_names
    
    if notes:
        config['notes'] = notes
    
    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Configuration saved to: {config_path}")
    return config_path


# =============================================================================
# INTERACTIVE MODE FUNCTIONS
# =============================================================================

def interactive_signal_input(n_inputs: int, filename: str = None) -> List[int]:
    """
    Interactively collect expected gain signs from user.
    
    Args:
        n_inputs: Number of input variables
        filename: Optional filename for display
    
    Returns:
        List of expected gain signs
    """
    print("\n" + "=" * 60)
    print("📝 INTERACTIVE SIGNAL CONFIGURATION")
    print("=" * 60)
    
    if filename:
        print(f"   Dataset: {filename}")
    print(f"   Number of input variables: {n_inputs}")
    print()
    print("   For each variable, enter the expected gain sign:")
    print("     +1  → Positive gain (output ↑ when input ↑)")
    print("     -1  → Negative gain (output ↓ when input ↓)")
    print("      0  → Unknown / No constraint")
    print()
    print("   Tip: Press Enter for 0 (unknown)")
    print("-" * 60)
    
    signals = []
    
    for i in range(n_inputs):
        while True:
            try:
                user_input = input(f"   Variable {i+1}: ").strip()
                
                # Default to 0 if empty
                if user_input == "":
                    signal = 0
                else:
                    signal = int(user_input)
                
                if signal not in [-1, 0, 1]:
                    print(f"      ⚠️ Invalid value. Please enter -1, 0, or +1.")
                    continue
                
                signals.append(signal)
                break
                
            except ValueError:
                print(f"      ⚠️ Invalid input. Please enter -1, 0, or +1.")
    
    print("-" * 60)
    print(f"\n   ✅ Configuration: {signals}")
    
    # Show summary
    summary = get_constraint_summary(signals)
    print(f"   Constrained: {summary['n_constrained']} variables")
    print(f"   Unconstrained: {summary['n_unconstrained']} variables")
    
    return signals


def interactive_signal_input_with_save(n_inputs: int, filename: str, 
                                        base_dir: str = "..") -> List[int]:
    """
    Interactively collect signals and offer to save to JSON.
    
    Args:
        n_inputs: Number of input variables
        filename: Dataset filename
        base_dir: Base directory
    
    Returns:
        List of expected gain signs
    """
    signals = interactive_signal_input(n_inputs, filename)
    
    # Ask if user wants to save
    print()
    save_input = input("   Save configuration for future use? [Y/n]: ").strip().lower()
    
    if save_input != 'n':
        # Ask for optional notes
        notes = input("   Notes (optional, press Enter to skip): ").strip()
        notes = notes if notes else None
        
        save_signals_to_json(filename, signals, base_dir, notes=notes)
        print(f"   Next time, this configuration will be loaded automatically.")
    
    print("=" * 60 + "\n")
    
    return signals


# =============================================================================
# SIGNAL LOADING FUNCTIONS
# =============================================================================

def get_expected_signals(n_inputs: int, filename: str, 
                        custom_signals: List[Union[int, None]] = None,
                        base_dir: str = "..",
                        interactive: bool = True) -> Tuple[List[int], str]:
    """
    Obtain expected gain signals for a dataset.
    
    Signal loading hierarchy:
        1. custom_signals parameter (if provided)
        2. JSON config file (data/raw/{filename}_config.json)
        3. Interactive mode (if interactive=True)
    
    Args:
        n_inputs: Number of input variables
        filename: Dataset filename
        custom_signals: Optional custom signal configuration (overrides all)
        base_dir: Base directory for finding config files
        interactive: If True, prompt user when signals not found; if False, raise error
    
    Returns:
        Tuple: (expected_signals, dataset_name)
    
    Signal Convention:
        +1: Positive gain expected
        -1: Negative gain expected
         0: Unknown/unconstrained (no restriction)
        None: Same as 0
    
    Examples:
        # Full constraints (all signs known)
        signals = [-1, 1, 1, 1, 1, 1]
        
        # Partial constraints (some signs unknown)
        signals = [-1, 1, 0, 0, 1, 1]  # Variables 3,4 are free
        
        # No constraints
        signals = [0, 0, 0, 0, 0, 0]
    """
    dataset_name = filename.lower().replace('.csv', '')
    
    # -----------------------------------------------------------------
    # Priority 1: Custom signals provided as parameter
    # -----------------------------------------------------------------
    if custom_signals is not None:
        signals = validate_expected_signals(custom_signals, n_inputs)
        print(f"✅ Using custom signal configuration")
        print_constraint_summary(signals, dataset_name)
        return signals, dataset_name
    
    # -----------------------------------------------------------------
    # Priority 2: JSON configuration file
    # -----------------------------------------------------------------
    json_config = load_signals_from_json(filename, base_dir)
    if json_config is not None:
        signals_from_json = json_config.get('expected_signals', [])
        
        if len(signals_from_json) == n_inputs:
            signals = validate_expected_signals(signals_from_json, n_inputs)
            print_constraint_summary(signals, dataset_name)
            return signals, dataset_name
        else:
            print(f"⚠️ JSON config found but has {len(signals_from_json)} signals, "
                  f"expected {n_inputs}. Ignoring.")
    
    # -----------------------------------------------------------------
    # Priority 3: Interactive mode
    # -----------------------------------------------------------------
    if interactive:
        print(f"\n⚠️ No configuration found for dataset '{filename}'.")
        print(f"   Expected config file: {get_config_path(filename, base_dir)}")
        print(f"   Entering interactive mode...")
        
        signals = interactive_signal_input_with_save(n_inputs, filename, base_dir)
        signals = validate_expected_signals(signals, n_inputs)
        return signals, dataset_name
    
    # -----------------------------------------------------------------
    # No signals found and interactive mode disabled
    # -----------------------------------------------------------------
    raise ValueError(
        f"❌ No configuration found for dataset '{filename}'.\n"
        f"   Expected config file: {get_config_path(filename, base_dir)}\n"
        f"   Options:\n"
        f"     1. Create config file: data/raw/{dataset_name}_config.json\n"
        f"     2. Provide custom_signals parameter with {n_inputs} values\n"
        f"     3. Set interactive=True to enter signals manually"
    )


def create_partial_signals(base_signals: List[int], 
                           unconstrained_vars: List[int]) -> List[int]:
    """
    Create partial constraint configuration from full configuration.
    
    Args:
        base_signals: Full signal configuration (all +1/-1)
        unconstrained_vars: List of variable indices (1-based) to leave unconstrained
    
    Returns:
        Modified signal list with specified variables set to 0
    
    Example:
        >>> base = [-1, 1, 1, 1, 1, 1]
        >>> partial = create_partial_signals(base, [3, 4])
        >>> print(partial)
        [-1, 1, 0, 0, 1, 1]  # Variables 3,4 now unconstrained
    """
    signals = base_signals.copy()
    for var_idx in unconstrained_vars:
        if 1 <= var_idx <= len(signals):
            signals[var_idx - 1] = 0
        else:
            raise ValueError(f"Variable index {var_idx} out of range [1, {len(signals)}]")
    return signals


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_raw_data(filename: str, base_dir: str = "..") -> pd.DataFrame:
    """
    Load raw data from a CSV file.
    
    Args:
        filename: File name (without .csv extension)
        base_dir: Base directory path
    
    Returns:
        DataFrame with raw data
    """
    filename_clean = filename.lower().replace('.csv', '')
    data_path = Path(base_dir) / "data" / "raw" / f"{filename_clean}.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"❌ File not found: {data_path}")
    
    data = pd.read_csv(data_path, header=None, sep=';')
    
    if data.shape[1] < 2:
        raise ValueError(f"Dataset must have at least 2 columns. Found: {data.shape[1]}")
    
    print(f"✅ Data loaded: {data.shape[0]} samples, {data.shape[1]-1} inputs, 1 output")
    
    return data


def preprocess_data(data: pd.DataFrame, 
                   test_size: float = 0.2, 
                   seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 'MinMaxScaler']:
    """
    Preprocess data: normalization and train/test split.
    
    Args:
        data: Raw DataFrame
        test_size: Test set proportion
        seed: Random seed for reproducibility
    
    Returns:
        Tuple: (train_inputs, train_targets, test_inputs, test_targets, scaler)
    """
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    
    # Remove missing data
    n_before = len(data)
    data = data.dropna()
    n_after = len(data)
    if n_before > n_after:
        print(f"⚠️ {n_before - n_after} rows removed (missing values)")
    
    # Normalize
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data.values)
    
    # Separate inputs and output
    inputs = data_normalized[:, :-1]
    targets = data_normalized[:, -1]
    
    # Train/test split
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(
        inputs, targets, test_size=test_size, random_state=seed
    )
    
    print(f"✅ Data processed:")
    print(f"   Train: {train_inputs.shape[0]} samples")
    print(f"   Test: {test_inputs.shape[0]} samples")
    
    return train_inputs, train_targets, test_inputs, test_targets, scaler


def load_dataset(filename: str, base_dir: str = "..", 
                test_size: float = 0.2, seed: int = 0,
                custom_signals: List[Union[int, None]] = None,
                interactive: bool = True) -> Dict[str, Any]:
    """
    Load and process a complete dataset.
    
    Signal loading hierarchy:
        1. custom_signals parameter (if provided)
        2. JSON config file (data/raw/{filename}_config.json)
        3. Interactive mode (if interactive=True)
    
    Args:
        filename: File name
        base_dir: Base directory
        test_size: Test proportion
        seed: Seed for reproducibility
        custom_signals: Optional custom signal configuration for partial constraints
        interactive: If True, prompt user for signals when not found; 
                    if False, raise error when signals not found
    
    Returns:
        Dictionary with all data and configurations
    
    Example with partial constraints:
        # Only constrain variables 1, 2, 5, 6 (leave 3, 4 free)
        data = load_dataset('computer_hardware', 
                           custom_signals=[-1, 1, 0, 0, 1, 1])
    
    Example with new dataset (interactive):
        # Will prompt user for signals if config file not found
        data = load_dataset('my_new_dataset')
    
    Example with new dataset (non-interactive):
        # Will raise error if config file not found
        data = load_dataset('my_new_dataset', interactive=False)
    """
    # Load raw data
    data = load_raw_data(filename, base_dir)
    
    # Get expected signals (using hierarchy)
    n_inputs = data.shape[1] - 1
    expected_signals, dataset_name = get_expected_signals(
        n_inputs, filename, 
        custom_signals=custom_signals,
        base_dir=base_dir,
        interactive=interactive
    )
    
    # Preprocess
    train_inputs, train_targets, test_inputs, test_targets, scaler = preprocess_data(
        data, test_size, seed
    )
    
    # Get constraint summary
    constraint_info = get_constraint_summary(expected_signals)
    
    return {
        'train_inputs': train_inputs,
        'train_targets': train_targets,
        'test_inputs': test_inputs,
        'test_targets': test_targets,
        'scaler': scaler,
        'expected_signals': expected_signals,
        'dataset_name': dataset_name,
        'n_inputs': n_inputs,
        'n_samples': data.shape[0],
        'constraint_info': constraint_info
    }


# =============================================================================
# SAVING FUNCTIONS
# =============================================================================

def save_dataset_config(dataset_name: str, filename: str, expected_signals: List[int],
                       data_shape: Tuple[int, int], scaler_type: str = 'MinMaxScaler',
                       seed: int = 0, base_dir: str = "..") -> Path:
    """
    Save dataset configuration.
    
    Args:
        dataset_name: Standardized dataset name
        filename: Original filename
        expected_signals: Expected gain signals
        data_shape: Data shape tuple (n_samples, n_features)
        scaler_type: Scaler type used
        seed: Random seed
        base_dir: Base directory
    
    Returns:
        Path to saved config file
    """
    config_dir = Path(base_dir) / "data" / "processed" / dataset_name
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = config_dir / "dataset_config.json"
    
    constraint_info = get_constraint_summary(expected_signals)
    
    config = {
        'dataset_info': {
            'name': dataset_name,
            'original_filename': filename,
            'n_samples': int(data_shape[0]),
            'n_features': int(data_shape[1]),
            'n_inputs': int(data_shape[1] - 1),
            'n_outputs': 1
        },
        'expected_signals': {
            'values': expected_signals,
            'constraint_summary': {
                'n_constrained': constraint_info['n_constrained'],
                'n_unconstrained': constraint_info['n_unconstrained'],
                'constrained_vars': constraint_info['constrained_vars'],
                'unconstrained_vars': constraint_info['unconstrained_vars'],
                'mode': 'fully_constrained' if constraint_info['is_fully_constrained'] 
                        else ('partially_constrained' if constraint_info['is_partially_constrained']
                              else 'unconstrained')
            }
        },
        'preprocessing': {
            'scaler': scaler_type,
            'range': [0, 1] if scaler_type == 'MinMaxScaler' else None,
            'random_state': seed,
            'test_size': 0.2
        },
        'metadata': {
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source': 'UCI Machine Learning Repository'
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Dataset config saved: {config_path}")
    return config_path


def save_processed_data(dataset_name: str, 
                       train_inputs: np.ndarray, train_targets: np.ndarray,
                       test_inputs: np.ndarray, test_targets: np.ndarray,
                       scaler, base_dir: str = "..") -> Tuple[Path, Path]:
    """
    Save processed data (split + scaler).
    
    Args:
        dataset_name: Dataset name
        train_inputs: Training inputs
        train_targets: Training targets
        test_inputs: Test inputs
        test_targets: Test targets
        scaler: Fitted scaler
        base_dir: Base directory
    
    Returns:
        Tuple: (split_path, scaler_path)
    """
    data_dir = Path(base_dir) / "data" / "processed" / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save split
    split_path = data_dir / "train_test_split.npz"
    np.savez_compressed(
        split_path,
        train_inputs=train_inputs,
        train_targets=train_targets,
        test_inputs=test_inputs,
        test_targets=test_targets
    )
    print(f"✅ Train/test split saved: {split_path}")
    
    # Save scaler
    scaler_path = data_dir / "scaler_params.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved: {scaler_path}")
    
    return split_path, scaler_path


def load_processed_data(dataset_name: str, base_dir: str = "..") -> Tuple:
    """
    Load processed dataset data.
    
    Args:
        dataset_name: Dataset name
        base_dir: Base directory
    
    Returns:
        Tuple: (train_inputs, train_targets, test_inputs, test_targets, scaler)
    """
    data_dir = Path(base_dir) / "data" / "processed" / dataset_name
    
    split_path = data_dir / "train_test_split.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Split not found: {split_path}")
    
    data = np.load(split_path)
    train_inputs = data['train_inputs']
    train_targets = data['train_targets']
    test_inputs = data['test_inputs']
    test_targets = data['test_targets']
    
    scaler_path = data_dir / "scaler_params.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"✅ Processed data loaded: {dataset_name}")
    return train_inputs, train_targets, test_inputs, test_targets, scaler


def load_dataset_config(dataset_name: str, base_dir: str = "..") -> Optional[Dict]:
    """
    Load dataset configuration.
    
    Args:
        dataset_name: Dataset name
        base_dir: Base directory
    
    Returns:
        Configuration dictionary or None if not found
    """
    config_path = Path(base_dir) / "data" / "processed" / dataset_name / "dataset_config.json"
    
    if not config_path.exists():
        print(f"⚠️ Config not found: {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✅ Dataset config loaded: {dataset_name}")
    return config