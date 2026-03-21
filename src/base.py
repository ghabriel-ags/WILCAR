"""
Base Module - Base classes and configurations for all methods
=============================================================
Federal University of Bahia – MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá
"""

import numpy as np
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

# Try to import PyTorch (optional)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# CONFIGURATIONS AND DATACLASSES
# =============================================================================

@dataclass
class TrainingConfig:
    """Centralized training configuration."""
    max_neurons: int = 750
    patience_constructive: int = 150
    patience_early_stopping: int = 250
    learning_rate: float = 0.1
    lr_decay: float = 0.999
    min_lr: float = 1e-3
    iterations: int = 2501
    delta_perturbation: float = 0.1
    tolerance: float = 1e-6
    seed: int = 0
    use_gpu: bool = False
    batch_size: Optional[int] = None
    verbose: int = 1  # 0=silent, 1=summary, 2=detailed
    task: str = 'regression'  # 'regression' or 'classification'


@dataclass
class TrainingResults:
    """Results of a single training run."""
    # Metrics per number of neurons
    mse_train: List[float] = field(default_factory=list)
    rmse_train: List[float] = field(default_factory=list)
    r2_train: List[float] = field(default_factory=list)
    mse_test: List[float] = field(default_factory=list)
    rmse_test: List[float] = field(default_factory=list)
    r2_test: List[float] = field(default_factory=list)
    conformity_rates: List[float] = field(default_factory=list)
    iteration_times: List[float] = field(default_factory=list)
    
    # Classification metrics per neuron count
    accuracy_train: List[float] = field(default_factory=list)
    accuracy_test: List[float] = field(default_factory=list)
    f1_test: List[float] = field(default_factory=list)
    mcc_test: List[float] = field(default_factory=list)
    bce_train: List[float] = field(default_factory=list)
    bce_test: List[float] = field(default_factory=list)

    # Best model
    best_neurons: int = 0
    best_train_mse: float = float('inf')
    best_train_rmse: float = float('inf')
    best_train_r2: float = -float('inf')
    best_test_mse: float = float('inf')
    best_test_rmse: float = float('inf')
    best_test_r2: float = -float('inf')
    best_conformity: float = 0.0

    # Best classification metrics
    best_test_accuracy: float = 0.0
    best_test_f1: float = 0.0
    best_test_mcc: float = -1.0
    
    # Conformity details for best model
    conformity_details: List[Dict] = field(default_factory=list)
    
    # Time
    total_time: float = 0.0
    avg_time_per_model: float = 0.0
    
    # Metadata
    method_name: str = ""
    dataset_name: str = ""
    timestamp: str = ""


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    filename: str
    n_samples: int
    n_inputs: int
    expected_signals: List[int]
    scaler_type: str = 'MinMaxScaler'


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class BaseNeuralNetwork(ABC):
    """
    Abstract base class for all methods (WILCAR, RIXM, ELM).
    
    Subclasses must implement:
    - _initialize_first_hidden_neuron(): Method-specific initialization
    - train(): Method-specific training logic
    - predict(): Prediction for new data
    """
    
    def __init__(self, 
                 train_inputs: np.ndarray,
                 train_targets: np.ndarray,
                 test_inputs: np.ndarray,
                 test_targets: np.ndarray,
                 expected_signals: List[int],
                 config: TrainingConfig,
                 dataset_name: str = "unknown"):
        """
        Initialize base neural network.
        
        Args:
            train_inputs: Training input data (n_samples, n_features)
            train_targets: Training targets (n_samples,)
            test_inputs: Test input data
            test_targets: Test targets
            expected_signals: Expected gain signs
            config: Training configuration
            dataset_name: Dataset name
        """
        # Store data
        self.train_inputs = train_inputs
        self.train_targets = train_targets
        self.test_inputs = test_inputs
        self.test_targets = test_targets
        self.expected_signals = expected_signals
        self.config = config
        self.dataset_name = dataset_name
        
        # Dimensions
        self.n_samples = train_inputs.shape[0]
        self.n_inputs = train_inputs.shape[1]
        
        # Network parameters
        self.param = None
        self.n_neurons = 0
        
        # Setup device (CPU/GPU)
        self.device = self._setup_device()
        
        # Seed for reproducibility
        np.random.seed(config.seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(config.seed)
    
    def _setup_device(self) -> str:
        """Setup computation device."""
        if self.config.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU enabled: {gpu_name}")
        else:
            device = 'cpu'
            if self.config.use_gpu:
                if not TORCH_AVAILABLE:
                    print("⚠️ GPU requested but PyTorch not available. Using CPU.")
                elif not torch.cuda.is_available():
                    print("⚠️ GPU requested but CUDA not available. Using CPU.")
        return device
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid function with overflow protection."""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)), 
                       np.exp(x) / (1 + np.exp(x)))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Sigmoid derivative: σ'(x) = σ(x) * (1 - σ(x))"""
        s = BaseNeuralNetwork.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def bce_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
        """Binary Cross-Entropy loss (numerically stable)."""
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
    
    def calculate_conformity(self, model_predict_fn) -> Tuple[float, List[Dict]]:
        """
        Calculate Signal Conformity Rate (SCR).
        
        Args:
            model_predict_fn: Model prediction function
            
        Returns:
            Tuple: (conformity_rate, details_per_variable)
        """
        x_base = self.test_inputs.T
        y_base = model_predict_fn(x_base)
        
        delta = self.config.delta_perturbation
        conformity_details = []
        n_conforming = 0
        n_evaluated = 0
        
        for i in range(self.n_inputs):
            x_perturbed = np.copy(x_base)
            x_perturbed[i, :] += delta
            y_perturbed = model_predict_fn(x_perturbed)
            
            gain = np.mean((y_perturbed - y_base) / delta)
            calculated_signal = int(np.sign(gain))
            expected_signal = self.expected_signals[i]
            
            if expected_signal != 0:
                n_evaluated += 1
                conforming = (calculated_signal == expected_signal)
                if conforming:
                    n_conforming += 1
            else:
                conforming = None
            
            conformity_details.append({
                'variable': i + 1,
                'calculated_gain': float(gain),
                'calculated_signal': calculated_signal,
                'expected_signal': expected_signal,
                'conformity': conforming
            })
        
        scr = n_conforming / n_evaluated if n_evaluated > 0 else 0
        return scr, conformity_details
    
    @abstractmethod
    def _initialize_first_hidden_neuron(self) -> np.ndarray:
        """
        Initialize weights for the first hidden neuron.
        Must be implemented by each specific method.
        """
        pass
    
    @abstractmethod
    def train(self) -> TrainingResults:
        """
        Execute full training.
        Must be implemented by each specific method.
        """
        pass
    
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions for new data.
        Must be implemented by each specific method.
        """
        pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability."""
    info = {
        'torch_available': TORCH_AVAILABLE,
        'cuda_available': False,
        'gpu_name': None,
        'gpu_memory': None
    }
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        info['cuda_available'] = True
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    
    return info