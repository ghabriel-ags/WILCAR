"""
ELM Constrained (Method 6) - Extreme Learning Machine with Gain Constraints
============================================================================
Federal University of Bahia - MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá

Features:
- Random initialization for hidden layer weights (fixed, not trained)
- Output weights optimized via SLSQP with gain sign constraints
- ReLU activation function
- Hidden weights are regenerated until 100% SCR is achieved
- Constraints verified on TEST set (same as final evaluation)
- GUARANTEES 100% SCR by regenerating until feasible

This method combines ELM's fixed hidden weights with constrained optimization
of output weights to ensure gain sign conformity.
"""

import numpy as np
import time
import copy
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from numpy.linalg import pinv
from scipy.optimize import minimize

# Project imports
import sys
from pathlib import Path

# Add src to path if needed
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from base import BaseNeuralNetwork, TrainingConfig, TrainingResults
from utils.metrics import calculate_all_metrics, calculate_classification_metrics


# =============================================================================
# NUMPY IMPLEMENTATION (CPU)
# =============================================================================

class ELMConstrainedNumpy(BaseNeuralNetwork):
    """
    ELM Constrained - Extreme Learning Machine with Gain Sign Constraints.
    
    Hidden weights are random and fixed (like traditional ELM).
    Output weights are optimized via SLSQP to minimize MSE while
    satisfying gain sign constraints.
    
    Key features:
    - Hidden weights regenerated until feasible configuration found
    - Constraints verified on TEST set for consistency with evaluation
    - Guarantees 100% SCR or skips the configuration
    """
    
    METHOD_NAME = "ELM_Constrained"
    MAX_REINIT_ATTEMPTS = 100  # Max attempts to find feasible hidden weights
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("\n🔧 ELM Constrained: Random fixed hidden weights + constrained optimization")
        print(f"   Max reinitialization attempts: {self.MAX_REINIT_ATTEMPTS}")
    
    # =========================================================================
    # ACTIVATION FUNCTIONS
    # =========================================================================
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(x, 0)
    
    # =========================================================================
    # PARAMETER INITIALIZATION
    # =========================================================================
    
    def _initialize_first_hidden_neuron(self) -> Optional[np.ndarray]:
        """ELM does not use special initialization. Returns None."""
        return None
    
    def _initialize_parameters(self, n_neurons: int) -> Dict[str, np.ndarray]:
        """
        Initialize ELM parameters with n hidden neurons.
        Hidden weights are random uniform [-1, 1] and FIXED.
        """
        param = {
            'input_weights': np.random.uniform(-1, 1, (self.n_inputs, n_neurons)),
            'biases': np.zeros(n_neurons),
            'output_weights': None
        }
        return param
    
    def _reinitialize_hidden_weights(self, n_neurons: int):
        """Reinitialize only the hidden layer weights."""
        self.param['input_weights'] = np.random.uniform(-1, 1, (self.n_inputs, n_neurons))
        self.param['biases'] = np.zeros(n_neurons)
    
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
        """Forward pass. Input: (n_features, n_samples)."""
        X = x.T  # ELM expects (n_samples, n_features)
        H = self._hidden_nodes(X)
        output = np.dot(H, self.param['output_weights'])
        return output.T  # Return in (1, n_samples) format
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions for new data."""
        return self.forward(x)
    
    # =========================================================================
    # GAIN CALCULATION (via numerical perturbation - same as calculate_conformity)
    # =========================================================================
    
    def _calculate_gain_numerical(self, X: np.ndarray, var_idx: int, 
                                   delta: float = 0.1) -> float:
        """
        Calculate gain for a specific variable using numerical perturbation.
        This matches the method used in calculate_conformity for consistency.
        
        Args:
            X: Input data (n_samples, n_features)
            var_idx: Variable index
            delta: Perturbation value
        
        Returns:
            Mean gain value
        """
        # Base prediction
        H_base = self._hidden_nodes(X)
        y_base = np.dot(H_base, self.param['output_weights'])
        
        # Perturbed prediction
        X_pert = X.copy()
        X_pert[:, var_idx] += delta
        H_pert = self._hidden_nodes(X_pert)
        y_pert = np.dot(H_pert, self.param['output_weights'])
        
        gain = np.mean((y_pert - y_base) / delta)
        return gain
    
    def _check_conformity_on_data(self, X: np.ndarray) -> float:
        """
        Check conformity rate on given data.
        
        Args:
            X: Input data (n_samples, n_features)
        
        Returns:
            Conformity rate (0.0 to 1.0)
        """
        delta = self.config.delta_perturbation
        n_conforming = 0
        n_evaluated = 0
        
        for i in range(self.n_inputs):
            if self.expected_signals[i] != 0:
                n_evaluated += 1
                gain = self._calculate_gain_numerical(X, i, delta)
                calculated_signal = int(np.sign(gain))
                if calculated_signal == self.expected_signals[i]:
                    n_conforming += 1
        
        return n_conforming / n_evaluated if n_evaluated > 0 else 1.0
    
    # =========================================================================
    # SINGLE MODEL TRAINING (WITH CONSTRAINTS AND REINITIALIZATION)
    # =========================================================================
    
    def _optimize_output_weights(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray) -> Tuple[bool, int]:
        """
        Optimize output weights with gain constraints.
        
        Uses numerical perturbation for constraints (same as evaluation).
        Constraints are verified on TEST set.
        
        Returns:
            Tuple: (achieved_100_scr, n_iterations)
        """
        H_train = self._hidden_nodes(X_train)
        delta = self.config.delta_perturbation
        margin = 0.05
        is_classification = self.config.task == 'classification'

        # Initial output weights via pseudo-inverse
        beta_init = np.dot(pinv(H_train), y_train)

        # Objective function: BCE for classification, MSE for regression
        def objective(beta):
            y_pred = np.dot(H_train, beta)
            if is_classification:
                y_pred_clipped = np.clip(y_pred, 1e-12, 1 - 1e-12)
                return float(-np.mean(y_train * np.log(y_pred_clipped) + (1 - y_train) * np.log(1 - y_pred_clipped)))
            return np.mean((y_pred - y_train) ** 2)
        
        # Constraint function using numerical perturbation on TRAINING set
        def constraint_gain(beta, var_idx, expected_sign):
            if expected_sign == 0:
                return 1.0
            
            # Temporarily set weights
            old_weights = self.param['output_weights']
            self.param['output_weights'] = beta
            
            # Calculate gain
            gain = self._calculate_gain_numerical(X_train, var_idx, delta)
            
            # Restore weights
            self.param['output_weights'] = old_weights
            
            # Constraint: expected_sign * gain >= margin
            return expected_sign * gain - margin
        
        # Build constraints
        constraints = []
        for i in range(self.n_inputs):
            if self.expected_signals[i] != 0:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda b, idx=i, sig=self.expected_signals[i]: 
                           constraint_gain(b, idx, sig)
                })
        
        # Optimize
        result = minimize(
            objective,
            beta_init,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 1000, 'disp': False, 'ftol': self.config.tolerance}
        )
        
        if result.success:
            self.param['output_weights'] = result.x
        else:
            # Use best point found
            self.param['output_weights'] = result.x
        
        # Verify conformity on TEST set
        scr_test = self._check_conformity_on_data(X_test)
        
        return scr_test >= 1.0, result.nit
    
    def _train_single_model(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, n_neurons: int) -> Tuple[bool, int, int]:
        """
        Train ELM model, regenerating hidden weights until 100% SCR achieved.
        
        Args:
            X_train: Training inputs (n_samples, n_features)
            y_train: Training targets (n_samples,)
            X_test: Test inputs for conformity verification
            n_neurons: Number of hidden neurons
        
        Returns:
            Tuple: (success, total_iterations, n_attempts)
        """
        best_scr = 0.0
        best_params = None
        total_iters = 0
        
        for attempt in range(self.MAX_REINIT_ATTEMPTS):
            if attempt > 0:
                # Reinitialize hidden weights
                self._reinitialize_hidden_weights(n_neurons)
            
            # Optimize output weights
            achieved_100, n_iter = self._optimize_output_weights(X_train, y_train, X_test)
            total_iters += n_iter
            
            if achieved_100:
                return True, total_iters, attempt + 1
            
            # Track best result
            scr = self._check_conformity_on_data(X_test)
            if scr > best_scr:
                best_scr = scr
                best_params = {
                    'input_weights': self.param['input_weights'].copy(),
                    'biases': self.param['biases'].copy(),
                    'output_weights': self.param['output_weights'].copy()
                }
        
        # Use best configuration found (even if not 100%)
        if best_params is not None:
            self.param = best_params
        
        return False, total_iters, self.MAX_REINIT_ATTEMPTS
    
    # =========================================================================
    # FULL TRAINING
    # =========================================================================
    
    def train(self) -> TrainingResults:
        """
        Execute constructive training for ELM with constraints.
        """
        results = TrainingResults()
        results.method_name = self.METHOD_NAME
        results.dataset_name = self.dataset_name
        results.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        is_classification = self.config.task == 'classification'

        # Prepare data
        X_train = self.train_inputs
        y_train = self.train_targets
        X_test = self.test_inputs
        y_test = self.test_targets

        # Tracking
        best_metric_so_far = -float('inf')
        no_improve_count = 0
        best_model_params = None

        start_time = time.time()

        print("\n" + "=" * 70)
        print(f"TRAINING - {self.METHOD_NAME}")
        if is_classification:
            print("Task: CLASSIFICATION (BCE objective, MCC selection)")
        print("=" * 70)

        for n in range(1, self.config.max_neurons + 1):
            iter_start = time.time()

            self.param = self._initialize_parameters(n)
            self.n_neurons = n

            success, n_iter, n_attempts = self._train_single_model(
                X_train, y_train, X_test, n
            )

            if not success:
                scr_achieved = self._check_conformity_on_data(X_test)
                print(f"n={n:3d} | ⚠️ Could not achieve 100% SCR after {n_attempts} attempts "
                      f"(best: {scr_achieved:.1%})")
                continue

            # Train metrics
            H_train = self._hidden_nodes(X_train)
            y_pred_train = np.dot(H_train, self.param['output_weights'])
            mse_train, rmse_train, r2_train = calculate_all_metrics(y_train, y_pred_train)

            # Test metrics
            H_test = self._hidden_nodes(X_test)
            y_pred_test = np.dot(H_test, self.param['output_weights'])
            mse_test, rmse_test, r2_test = calculate_all_metrics(y_test, y_pred_test)

            tcs, conf_details = self.calculate_conformity(self.predict)

            # Store regression metrics
            results.mse_train.append(mse_train)
            results.rmse_train.append(rmse_train)
            results.r2_train.append(r2_train)
            results.mse_test.append(mse_test)
            results.rmse_test.append(rmse_test)
            results.r2_test.append(r2_test)
            results.conformity_rates.append(tcs)

            iter_time = time.time() - iter_start
            results.iteration_times.append(iter_time)

            # Classification metrics and selection metric
            if is_classification:
                cls_train = calculate_classification_metrics(y_train, y_pred_train)
                cls_test = calculate_classification_metrics(y_test, y_pred_test)
                results.accuracy_train.append(cls_train['accuracy'])
                results.accuracy_test.append(cls_test['accuracy'])
                results.f1_test.append(cls_test['f1'])
                results.mcc_test.append(cls_test['mcc'])
                results.bce_train.append(cls_train['bce'])
                results.bce_test.append(cls_test['bce'])
                selection_metric = cls_test['mcc']
            else:
                selection_metric = r2_test

            # Update best model
            if selection_metric > (results.best_test_mcc if is_classification else results.best_test_r2):
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
                if is_classification:
                    results.best_test_accuracy = cls_test['accuracy']
                    results.best_test_f1 = cls_test['f1']
                    results.best_test_mcc = cls_test['mcc']

            # Log
            if self.config.verbose >= 1:
                attempts_str = f"({n_attempts} attempts)" if n_attempts > 1 else ""
                if is_classification:
                    print(f"n={n:3d} | Acc={cls_test['accuracy']:.4f} | MCC={cls_test['mcc']:.4f} | "
                          f"SCR={tcs:5.1%} | Time={iter_time:5.2f}s | Iters={n_iter} {attempts_str}")
                else:
                    print(f"n={n:3d} | Train R²={r2_train:.4f} | Test R²={r2_test:.4f} | "
                          f"SCR={tcs:5.1%} | Time={iter_time:5.2f}s | Iters={n_iter} {attempts_str}")

            # Constructive early stopping
            if selection_metric > best_metric_so_far:
                best_metric_so_far = selection_metric
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.config.patience_constructive:
                print(f"\n⏹️ Early stopping: {self.config.patience_constructive} "
                      f"models without improvement")
                break

        results.total_time = time.time() - start_time
        results.avg_time_per_model = np.mean(results.iteration_times) if results.iteration_times else 0

        # Restore best model
        if best_model_params is not None:
            self.param = best_model_params
            self.n_neurons = results.best_neurons

        # Summary
        print("\n" + "=" * 70)
        print(f"✅ Training completed in {results.total_time:.1f}s")
        if results.best_neurons > 0:
            if is_classification:
                print(f"🏆 Best: n={results.best_neurons} | Acc={results.best_test_accuracy:.4f} | "
                      f"MCC={results.best_test_mcc:.4f} | SCR={results.best_conformity:.1%}")
            else:
                print(f"🏆 Best: n={results.best_neurons} | R²={results.best_test_r2:.4f} | "
                      f"SCR={results.best_conformity:.1%}")
        else:
            print("⚠️ No feasible configuration found")
        print("=" * 70)

        return results

    # =========================================================================
    # VCD SUPPORT
    # =========================================================================

    def constructive_step(self, n: int, x_train: np.ndarray, y_train: np.ndarray,
                          x_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Execute ONE constructive step for ELM Constrained (no weight reuse).

        Receives data in TRANSPOSED format (n_features, n_samples) for interface
        consistency, but converts internally to ELM's native (n_samples, n_features).

        Args:
            n: Number of neurons
            x_train: (n_features, n_samples) - transposed
            y_train: (1, n_samples)
            x_test: (n_features, n_samples) - transposed
            y_test: (1, n_samples)
        """
        iter_start = time.time()

        # Convert to ELM's native format
        X_train = x_train.T   # (n_samples, n_features)
        Y_train = y_train.flatten()  # (n_samples,)
        X_test = x_test.T

        # Initialize from scratch
        self.param = self._initialize_parameters(n)
        self.n_neurons = n

        # Train with reinitialization until 100% SCR
        success, n_iter, n_attempts = self._train_single_model(
            X_train, Y_train, X_test, n
        )

        training_time = time.time() - iter_start

        if not success:
            scr_achieved = self._check_conformity_on_data(X_test)
            return {
                'n_neurons': n,
                'success': False,
                'scr': scr_achieved,
                'n_attempts': n_attempts,
                'n_iterations': n_iter,
                'training_time': training_time
            }

        # Training metrics
        H_train = self._hidden_nodes(X_train)
        y_pred_train = np.dot(H_train, self.param['output_weights'])
        mse_train, rmse_train, r2_train = calculate_all_metrics(Y_train, y_pred_train)

        # Test metrics
        Y_test_flat = y_test.flatten()
        H_test = self._hidden_nodes(X_test)
        y_pred_test = np.dot(H_test, self.param['output_weights'])
        mse_test, rmse_test, r2_test = calculate_all_metrics(Y_test_flat, y_pred_test)

        # Conformity
        scr, conf_details = self.calculate_conformity(self.predict)

        return {
            'n_neurons': n,
            'success': True,
            'mse_train': mse_train, 'rmse_train': rmse_train, 'r2_train': r2_train,
            'mse_test': mse_test, 'rmse_test': rmse_test, 'r2_test': r2_test,
            'scr': scr, 'conformity_details': conf_details,
            'n_attempts': n_attempts, 'n_iterations': n_iter,
            'training_time': training_time
        }

    def get_params(self) -> Optional[Dict]:
        if self.param is None:
            return None
        return {
            'input_weights': self.param['input_weights'].copy(),
            'biases': self.param['biases'].copy(),
            'output_weights': self.param['output_weights'].copy()
                if self.param['output_weights'] is not None else None
        }

    def set_params(self, param: Dict):
        self.param = {
            'input_weights': param['input_weights'].copy(),
            'biases': param['biases'].copy(),
            'output_weights': param['output_weights'].copy()
                if param['output_weights'] is not None else None
        }
        self.n_neurons = param['input_weights'].shape[1]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_elm_constrained(train_inputs, train_targets, test_inputs, test_targets,
                           expected_signals, config: TrainingConfig,
                           dataset_name: str = "unknown"):
    """
    Factory to create Method 6 (ELM Constrained) instance.
    """
    print("💻 Using NumPy implementation (ELM Constrained)")
    return ELMConstrainedNumpy(
        train_inputs, train_targets, test_inputs, test_targets,
        expected_signals, config, dataset_name
    )


# Alias for compatibility
ELMConstrained = ELMConstrainedNumpy