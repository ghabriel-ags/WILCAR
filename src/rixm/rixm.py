"""
RIXM (Method 3) - Random Initialization Xavier Method
======================================================
Federal University of Bahia - MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá

Features:
- Fully random initialization (Xavier) for all neurons
- Biases initialized to zero
- Training via backpropagation (same as Method 1)
- NO weight reuse between models
- Each model with n neurons is trained from scratch

This method serves as a BASELINE for comparison with WILCAR methods.
"""

import numpy as np
import time
import copy
from datetime import datetime
from typing import Dict, Tuple, Optional

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

class RIXMNumpy(BaseNeuralNetwork):
    """
    RIXM - Random Initialization Xavier Method.
    
    Simple baseline: random initialization (Xavier) for all weights,
    no reuse between models. Each model is trained from scratch.
    """
    
    METHOD_NAME = "RIXM"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # No special initialization - everything will be Xavier random
        print("\n🔧 RIXM: Using random Xavier initialization (no WILCAR)")
    
    # =========================================================================
    # PARAMETER INITIALIZATION (100% RANDOM)
    # =========================================================================
    
    def _initialize_first_hidden_neuron(self) -> Optional[np.ndarray]:
        """
        RIXM does not use special initialization.
        Returns None - all weights will be Xavier random.
        """
        return None
    
    def _initialize_parameters(self, n_neurons: int) -> Dict[str, np.ndarray]:
        """
        Initialize network parameters with n hidden neurons.
        Uses Xavier initialization for all weights, bias = 0.
        """
        param = {
            'W1': np.random.randn(n_neurons, self.n_inputs) * np.sqrt(1. / self.n_inputs),
            'b1': np.zeros((n_neurons, 1)),
            'W2': np.random.randn(1, n_neurons) * np.sqrt(1. / n_neurons),
            'b2': np.zeros((1, 1))
        }
        return param
    
    # =========================================================================
    # FORWARD AND PREDICT
    # =========================================================================
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass."""
        Z1 = self.param['W1'].dot(x) + self.param['b1']
        A1 = self.sigmoid(Z1)
        Z2 = self.param['W2'].dot(A1) + self.param['b2']
        A2 = self.sigmoid(Z2)
        return A2, A1, Z1
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions for new data."""
        output, _, _ = self.forward(x)
        return output
    
    # =========================================================================
    # SINGLE MODEL TRAINING (BACKPROPAGATION)
    # =========================================================================
    
    def _train_single_model(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """
        Train current model via backpropagation.
        Identical to Method 1.
        """
        lr = self.config.learning_rate
        m = x.shape[1]
        
        best_loss = float('inf')
        no_improve = 0
        
        for epoch in range(self.config.iterations):
            # Forward
            A2, A1, Z1 = self.forward(x)
            
            # Loss
            current_loss = np.mean((A2 - y) ** 2)
            
            # Early stopping per epoch
            if current_loss < best_loss - self.config.tolerance:
                best_loss = current_loss
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= self.config.patience_early_stopping:
                break
            
            # Backward
            dZ2 = A2 - y
            dW2 = dZ2.dot(A1.T) / m
            db2 = np.sum(dZ2, axis=1, keepdims=True) / m
            
            dA1 = self.param['W2'].T.dot(dZ2)
            dZ1 = dA1 * A1 * (1 - A1)
            dW1 = dZ1.dot(x.T) / m
            db1 = np.sum(dZ1, axis=1, keepdims=True) / m
            
            # Update
            self.param['W1'] -= lr * dW1
            self.param['b1'] -= lr * db1
            self.param['W2'] -= lr * dW2
            self.param['b2'] -= lr * db2
            
            # Learning rate decay
            lr = max(lr * self.config.lr_decay, self.config.min_lr)
        
        # Final metrics
        y_pred = self.predict(x)
        mse, rmse, r2 = calculate_all_metrics(y.flatten(), y_pred.flatten())
        
        return mse, rmse, r2
    
    # =========================================================================
    # FULL TRAINING (CONSTRUCTIVE WITHOUT WEIGHT REUSE)
    # =========================================================================
    
    def train(self) -> TrainingResults:
        """
        Execute constructive training.
        
        Difference from WILCAR: each model with n neurons is trained FROM SCRATCH,
        without reusing weights from the previous model.
        """
        results = TrainingResults()
        results.method_name = self.METHOD_NAME
        results.dataset_name = self.dataset_name
        results.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data
        x_train = self.train_inputs.T
        y_train = self.train_targets.reshape(1, -1)
        x_test = self.test_inputs.T
        y_test = self.test_targets.reshape(1, -1)
        
        # Tracking
        best_r2_so_far = -float('inf')
        no_improve_count = 0
        best_model_params = None
        
        start_time = time.time()
        
        print("\n" + "=" * 70)
        print(f"TRAINING - {self.METHOD_NAME} (Baseline)")
        print("=" * 70)
        
        for n in range(1, self.config.max_neurons + 1):
            iter_start = time.time()
            
            # =====================================================
            # KEY DIFFERENCE: Always initialize from scratch!
            # Does not reuse weights from previous model
            # =====================================================
            self.param = self._initialize_parameters(n)
            self.n_neurons = n
            
            # Train
            mse_train, rmse_train, r2_train = self._train_single_model(x_train, y_train)
            
            # Test metrics
            y_pred_test = self.predict(x_test)
            mse_test, rmse_test, r2_test = calculate_all_metrics(
                y_test.flatten(), y_pred_test.flatten()
            )
            
            # Conformity
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
                      f"SCR={tcs:5.1%} | Time={iter_time:5.1f}s")
            
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

def create_rixm(train_inputs, train_targets, test_inputs, test_targets,
                expected_signals, config: TrainingConfig,
                dataset_name: str = "unknown"):
    """
    Factory to create Method 3 (RIXM) instance.
    
    Args:
        train_inputs, train_targets: Training data
        test_inputs, test_targets: Test data
        expected_signals: Expected gain signs
        config: TrainingConfig
        dataset_name: Dataset name
    
    Returns:
        RIXM instance
    """
    print("💻 Using NumPy implementation (RIXM - Baseline)")
    return RIXMNumpy(
        train_inputs, train_targets, test_inputs, test_targets,
        expected_signals, config, dataset_name
    )


# Alias for compatibility
RIXM = RIXMNumpy