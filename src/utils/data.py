"""
Data Module - Functions for data loading and saving
====================================================
Federal University of Bahia – MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# =============================================================================
# KNOWN DATASET CONFIGURATIONS
# =============================================================================

EXPECTED_SIGNALS_THEORY = {
    'computer_hardware': [-1, 1, 1, 1, 1, 1],
    'fish_toxicity': [-1, 1, -1, 1, 1, 1],
    'aquatic_toxicity': [1, -1, 1, 1, 1, -1, 1, 1]
}

DATASET_MAPPING = {
    'machine': 'computer_hardware',
    'machine.csv': 'computer_hardware',
    'computer_hardware': 'computer_hardware',
    'qsar_fish_toxicity': 'fish_toxicity',
    'qsar_fish_toxicity.csv': 'fish_toxicity',
    'fish_toxicity': 'fish_toxicity',
    'fish': 'fish_toxicity',
    'qsar_aquatic_toxicity': 'aquatic_toxicity',
    'qsar_aquatic_toxicity.csv': 'aquatic_toxicity',
    'aquatic_toxicity': 'aquatic_toxicity',
    'aquatic': 'aquatic_toxicity'
}


# =============================================================================
# DATASET DETECTION FUNCTIONS
# =============================================================================

def detect_dataset_name(filename: str) -> Optional[str]:
    """
    Automatically detect dataset name from filename.
    
    Args:
        filename: File name (with or without extension)
    
    Returns:
        Standardized dataset_name or None if not recognized
    """
    filename_normalized = filename.lower().replace('.csv', '')
    
    if filename_normalized in DATASET_MAPPING:
        return DATASET_MAPPING[filename_normalized]
    
    for key, value in DATASET_MAPPING.items():
        if key in filename_normalized or filename_normalized in key:
            return value
    
    return None


def get_expected_signals(n_inputs: int, filename: str, 
                        auto_load: bool = True) -> Tuple[List[int], Optional[str]]:
    """
    Automatically obtain expected gain signals.
    
    Args:
        n_inputs: Number of input variables
        filename: Dataset filename
        auto_load: Whether to auto-detect dataset
    
    Returns:
        Tuple: (expected_signals, dataset_name)
    """
    
    dataset_name = None
    
    if auto_load:
        dataset_name = detect_dataset_name(filename)
        
        if dataset_name and dataset_name in EXPECTED_SIGNALS_THEORY:
            signals = EXPECTED_SIGNALS_THEORY[dataset_name]
            
            if len(signals) == n_inputs:
                print(f"✅ Dataset identified: '{dataset_name}'")
                print(f"✅ Signals loaded: {signals}")
                return signals, dataset_name
            else:
                print(f"⚠️ Dataset '{dataset_name}' recognized, but number of inputs does not match")
                print(f"   Expected: {len(signals)} inputs, Found: {n_inputs}")
    
    raise ValueError(
        f"❌ Dataset '{filename}' was not recognized automatically.\n"
        f"   Valid datasets: {list(EXPECTED_SIGNALS_THEORY.keys())}"
    )


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
    
    data_path = Path(base_dir) / "data" / "raw" / f"{filename}.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"❌ File not found: {data_path}")
    
    data = pd.read_csv(data_path, header=None, sep=';')
    
    if data.shape[1] < 2:
        raise ValueError(f"Dataset must have at least 2 columns. Found: {data.shape[1]}")
    
    print(f"✅ Data loaded: {data.shape[0]} samples, {data.shape[1]-1} inputs, 1 output")
    
    return data


def preprocess_data(data: pd.DataFrame, 
                   test_size: float = 0.2, 
                   seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Preprocess data: normalization and train/test split.
    
    Args:
        data: Raw DataFrame
        test_size: Test set proportion
        seed: Random seed for reproducibility
    
    Returns:
        Tuple: (train_inputs, train_targets, test_inputs, test_targets, scaler)
    """
    
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
                test_size: float = 0.2, seed: int = 0) -> Dict[str, Any]:
    """
    Load and process a complete dataset.
    
    Args:
        filename: File name
        base_dir: Base directory
        test_size: Test proportion
        seed: Seed for reproducibility
    
    Returns:
        Dictionary with all data and configurations
    """
    # Load raw data
    data = load_raw_data(filename, base_dir)
    
    # Get expected signals
    n_inputs = data.shape[1] - 1
    expected_signals, dataset_name = get_expected_signals(n_inputs, filename)
    
    if dataset_name is None:
        dataset_name = filename.lower().replace('.csv', '')
    
    # Preprocess
    train_inputs, train_targets, test_inputs, test_targets, scaler = preprocess_data(
        data, test_size, seed
    )
    
    return {
        'train_inputs': train_inputs,
        'train_targets': train_targets,
        'test_inputs': test_inputs,
        'test_targets': test_targets,
        'scaler': scaler,
        'expected_signals': expected_signals,
        'dataset_name': dataset_name,
        'n_inputs': n_inputs,
        'n_samples': data.shape[0]
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
            'description': {
                '1': 'positive gain expected',
                '-1': 'negative gain expected',
                '0': 'unknown gain'
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