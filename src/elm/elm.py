"""
ELM (Method 5) - Extreme Learning Machine
==========================================
Federal University of Bahia - MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá

Features:
- Random initialization for hidden layer weights (fixed, not trained)
- Output weights calculated via Moore-Penrose pseudo-inverse
- ReLU activation function
- NO iterative training (single-pass learning)
- NO weight reuse between models
- Each model with n neurons is trained from scratch

This is the traditional ELM approach - extremely fast but without
gain sign guarantees.
"""

import numpy as np
import time
import copy
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from numpy.linalg import pinv

# Project imports
import sys
from pathlib import Path

# Add src to path if needed
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from base import BaseNeuralNetwork, TrainingConfig, TrainingResults
from utils.metrics import calculate_all_metrics


# =============================================================================
# NUMPY IMPLEMENTATION (CPU)
# =============================================================================

class ELMNumpy(BaseNeuralNetwork):
    """
    ELM - Extreme Learning Machine.
    
    Traditional ELM: random hidden weights (fixed), output weights via pseudo-inverse.
    Very fast training but no control over gain signs.
    """
    
    METHOD_NAME = "ELM"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("\n🔧 ELM: Using random fixed hidden weights + pseudo-inverse")
    
    # =========================================================================
    # ACTIVATION FUNCTIONS
    # =========================================================================
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(x, 0)
    
    @staticmethod
    def sigmoid_safe(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid (alternative activation)."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-np.clip(x, -500, 500))),
            np.exp(np.clip(x, -500, 500)) / (1 + np.exp(np.clip(x, -500, 500)))
        )
    
    # =========================================================================
    # PARAMETER INITIALIZATION
    # =========================================================================
    
    def _initialize_first_hidden_neuron(self) -> Optional[np.ndarray]:
        """
        ELM does not use special initialization for first neuron.
        Returns None.
        """
        return None
    
    def _initialize_parameters(self, n_neurons: int) -> Dict[str, np.ndarray]:
        """
        Initialize ELM parameters with n hidden neurons.
        
        Hidden weights are random uniform [-1, 1] and FIXED (not trained).
        Output weights will be calculated via pseudo-inverse.
        """
        param = {
            'input_weights': np.random.uniform(-1, 1, (self.n_inputs, n_neurons)),
            'biases': np.zeros(n_neurons),
            'output_weights': None  # Will be calculated during training
        }
        return param
    
    # =========================================================================
    # FORWARD AND PREDICT
    # =========================================================================
    
    def _hidden_nodes(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate hidden layer output.
        
        Args:
            X: Input data (n_samples, n_features)
        
        Returns:
            Hidden layer activations (n_samples, n_neurons)
        """
        G = np.dot(X, self.param['input_weights']) + self.param['biases']
        H = self.relu(G)
        return H
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input data (n_features, n_samples) - transposed format
        
        Returns:
            Output predictions
        """
        # ELM expects (n_samples, n_features)
        X = x.T
        H = self._hidden_nodes(X)
        output = np.dot(H, self.param['output_weights'])
        return output.T  # Return in (1, n_samples) format
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions for new data."""
        return self.forward(x)
    
    # =========================================================================
    # SINGLE MODEL TRAINING (PSEUDO-INVERSE)
    # =========================================================================
    
    def _train_single_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[float, float, float]:
        """
        Train ELM model via Moore-Penrose pseudo-inverse.
        
        This is a single-pass calculation, not iterative optimization.
        
        Args:
            X_train: Training inputs (n_samples, n_features)
            y_train: Training targets (n_samples,)
        
        Returns:
            Tuple: (mse, rmse, r2)
        """
        # Calculate hidden layer output
        H = self._hidden_nodes(X_train)
        
        # Calculate output weights via pseudo-inverse: β = H⁺ * y
        self.param['output_weights'] = np.dot(pinv(H), y_train)
        
        # Calculate predictions
        y_pred = np.dot(H, self.param['output_weights'])
        
        # Calculate metrics
        mse, rmse, r2 = calculate_all_metrics(y_train, y_pred)
        
        return mse, rmse, r2
    
    # =========================================================================
    # FULL TRAINING (CONSTRUCTIVE)
    # =========================================================================
    
    def train(self) -> TrainingResults:
        """
        Execute constructive training for ELM.
        
        Each model with n neurons is trained from scratch (no weight reuse).
        Training is very fast due to single-pass pseudo-inverse calculation.
        """
        results = TrainingResults()
        results.method_name = self.METHOD_NAME
        results.dataset_name = self.dataset_name
        results.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data (ELM uses row-based format)
        X_train = self.train_inputs  # (n_samples, n_features)
        y_train = self.train_targets  # (n_samples,)
        X_test = self.test_inputs
        y_test = self.test_targets
        
        # Tracking
        best_r2_so_far = -float('inf')
        no_improve_count = 0
        best_model_params = None
        
        start_time = time.time()
        
        print("\n" + "=" * 70)
        print(f"TRAINING - {self.METHOD_NAME}")
        print("=" * 70)
        
        for n in range(1, self.config.max_neurons + 1):
            iter_start = time.time()
            
            # Initialize new model from scratch (no weight reuse)
            self.param = self._initialize_parameters(n)
            self.n_neurons = n
            
            # Train (single-pass pseudo-inverse)
            mse_train, rmse_train, r2_train = self._train_single_model(X_train, y_train)
            
            # Test predictions
            H_test = self._hidden_nodes(X_test)
            y_pred_test = np.dot(H_test, self.param['output_weights'])
            mse_test, rmse_test, r2_test = calculate_all_metrics(y_test, y_pred_test)
            
            # Conformity (using transposed format for compatibility)
            tcs, conf_details = self.calculate_conformity(self.predict)
            
            # Store metrics
            results.mse_train.append(mse_train)
            results.rmse_train.append(rmse_train)
            results.r2_train.append(r2_train)
            results.mse_test.append(mse_test)
            results.rmse_test.append(rmse_test)
            results.r2_test.append(r2_test)
            results.conformity_rates.append(tcs)
            
            iter_time = time.time() - iter_start
            results.iteration_times.append(iter_time)
            
            # Update best model
            if r2_test > results.best_test_r2:
                results.best_test_r2 = r2_test
                results.best_test_mse = mse_test
                results.best_test_rmse = rmse_test
                results.best_train_r2 = r2_train
                results.best_train_mse = mse_train
                results.best_train_rmse = rmse_train
                results.best_neurons = n
                results.best_conformity = tcs
                results.conformity_details = conf_details
                best_model_params = copy.deepcopy(self.param)
            
            # Log
            if self.config.verbose >= 1:
                print(f"n={n:3d} | Train R²={r2_train:.4f} | Test R²={r2_test:.4f} | "
                      f"SCR={tcs:5.1%} | Time={iter_time:5.3f}s")
            
            # Constructive early stopping
            if r2_test > best_r2_so_far:
                best_r2_so_far = r2_test
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= self.config.patience_constructive:
                print(f"\n⏹️ Early stopping: {self.config.patience_constructive} models without improvement")
                break
        
        results.total_time = time.time() - start_time
        results.avg_time_per_model = np.mean(results.iteration_times)
        
        # Restore best model
        if best_model_params is not None:
            self.param = best_model_params
            self.n_neurons = results.best_neurons
        
        # Summary
        print("\n" + "=" * 70)
        print(f"✅ Training completed in {results.total_time:.1f}s")
        print(f"🏆 Best: n={results.best_neurons} | R²={results.best_test_r2:.4f} | SCR={results.best_conformity:.1%}")
        print("=" * 70)
        
        return results


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_elm(train_inputs, train_targets, test_inputs, test_targets,
               expected_signals, config: TrainingConfig,
               dataset_name: str = "unknown"):
    """
    Factory to create Method 5 (ELM) instance.
    
    Args:
        train_inputs, train_targets: Training data
        test_inputs, test_targets: Test data
        expected_signals: Expected gain signs
        config: TrainingConfig
        dataset_name: Dataset name
    
    Returns:
        ELM instance
    """
    print("💻 Using NumPy implementation (ELM - Extreme Learning Machine)")
    return ELMNumpy(
        train_inputs, train_targets, test_inputs, test_targets,
        expected_signals, config, dataset_name
    )


# Alias for compatibility
ELM = ELMNumpy