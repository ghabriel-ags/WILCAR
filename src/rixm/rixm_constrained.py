"""
RIXM Constrained (Method 4) - Random Initialization Xavier Method with Constraints
===================================================================================
Federal University of Bahia - MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá

Features:
- Fully random initialization (Xavier) for all neurons
- Biases initialized to zero
- Training via scipy.optimize.minimize (SLSQP) with gain constraints
- NO weight reuse between models
- Each model with n neurons is trained from scratch
- Reinitializes weights until 100% SCR is achieved or max attempts reached

This method serves as a CONSTRAINED BASELINE for comparison with WILCAR methods.
It shows the effect of gain constraints without the WILCAR initialization strategy.
"""

import numpy as np
import time
import copy
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from scipy.optimize import minimize

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

class RIXMConstrainedNumpy(BaseNeuralNetwork):
    """
    RIXM Constrained - Random Initialization with Gain Constraints.
    
    Baseline with constraints: random initialization (Xavier) for all weights,
    no reuse between models, but with gain sign constraints enforced via SLSQP.
    
    Key feature: Reinitializes weights until 100% SCR is achieved on test set.
    """
    
    METHOD_NAME = "RIXM_Constrained"
    MAX_REINIT_ATTEMPTS = 100  # Max attempts to find feasible configuration
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("\n🔧 RIXM Constrained: Random Xavier initialization + constrained optimization")
        print(f"   Max reinitialization attempts: {self.MAX_REINIT_ATTEMPTS}")
    
    # =========================================================================
    # AUXILIARY FUNCTIONS
    # =========================================================================
    
    @staticmethod
    def sigmoid_safe(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-np.clip(x, -500, 500))),
            np.exp(np.clip(x, -500, 500)) / (1 + np.exp(np.clip(x, -500, 500)))
        )
    
    # =========================================================================
    # PARAMETER INITIALIZATION (100% RANDOM)
    # =========================================================================
    
    def _initialize_first_hidden_neuron(self) -> Optional[np.ndarray]:
        """RIXM does not use special initialization. Returns None."""
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
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        Z1 = self.param['W1'].dot(x) + self.param['b1']
        A1 = self.sigmoid_safe(Z1)
        Z2 = self.param['W2'].dot(A1) + self.param['b2']
        A2 = self.sigmoid_safe(Z2)
        return A2
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions for new data."""
        return self.forward(x)
    
    # =========================================================================
    # PARAMETER <-> VECTOR CONVERSION (for scipy.optimize)
    # =========================================================================
    
    def _params_to_vector(self, param: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert parameters dict → vector."""
        return np.concatenate([
            param['W1'].flatten(),
            param['b1'].flatten(),
            param['W2'].flatten(),
            param['b2'].flatten()
        ])
    
    def _vector_to_params(self, vector: np.ndarray, n_neurons: int) -> Dict[str, np.ndarray]:
        """Convert vector → parameters dict."""
        n_W1 = n_neurons * self.n_inputs
        n_b1 = n_neurons
        n_W2 = n_neurons
        
        W1 = vector[:n_W1].reshape(n_neurons, self.n_inputs)
        b1 = vector[n_W1:n_W1+n_b1].reshape(n_neurons, 1)
        W2 = vector[n_W1+n_b1:n_W1+n_b1+n_W2].reshape(1, n_neurons)
        b2 = vector[-1:].reshape(1, 1)
        
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    
    # =========================================================================
    # CONFORMITY CHECK ON TEST SET
    # =========================================================================
    
    def _check_conformity_on_data(self, x_data: np.ndarray) -> float:
        """
        Check conformity rate on given data.
        
        Args:
            x_data: Input data (n_features, n_samples) - transposed format
        
        Returns:
            Conformity rate (0.0 to 1.0)
        """
        delta = self.config.delta_perturbation
        y_base = self.predict(x_data)
        
        n_conforming = 0
        n_evaluated = 0
        
        for i in range(self.n_inputs):
            if self.expected_signals[i] != 0:
                n_evaluated += 1
                
                x_pert = x_data.copy()
                x_pert[i, :] += delta
                y_pert = self.predict(x_pert)
                
                gain = np.mean((y_pert - y_base) / delta)
                calculated_signal = int(np.sign(gain))
                
                if calculated_signal == self.expected_signals[i]:
                    n_conforming += 1
        
        return n_conforming / n_evaluated if n_evaluated > 0 else 1.0
    
    # =========================================================================
    # SINGLE OPTIMIZATION ATTEMPT
    # =========================================================================
    
    def _optimize_single_attempt(self, x: np.ndarray, y: np.ndarray, 
                                  n_neurons: int) -> Tuple[bool, int]:
        """
        Single optimization attempt via SLSQP with constraints.
        
        Returns:
            Tuple: (success, n_iterations)
        """
        # Objective function: MSE
        def objective(vec):
            param = self._vector_to_params(vec, n_neurons)
            Z1 = param['W1'].dot(x) + param['b1']
            A1 = self.sigmoid_safe(Z1)
            Z2 = param['W2'].dot(A1) + param['b2']
            y_pred = self.sigmoid_safe(Z2)
            return np.mean((y_pred - y) ** 2)
        
        # Gain constraints
        def constraint_gain(vec, var_idx, expected_sign):
            if expected_sign == 0:
                return 1.0
            
            param = self._vector_to_params(vec, n_neurons)
            
            # Calculate gain on training set
            Z1 = param['W1'].dot(x) + param['b1']
            A1 = self.sigmoid_safe(Z1)
            Z2 = param['W2'].dot(A1) + param['b2']
            y_base = self.sigmoid_safe(Z2)
            
            x_pert = x.copy()
            x_pert[var_idx, :] += self.config.delta_perturbation
            
            Z1_p = param['W1'].dot(x_pert) + param['b1']
            A1_p = self.sigmoid_safe(Z1_p)
            Z2_p = param['W2'].dot(A1_p) + param['b2']
            y_pert = self.sigmoid_safe(Z2_p)
            
            gain = np.mean((y_pert - y_base) / self.config.delta_perturbation)
            
            margin = 0.05
            return expected_sign * gain - margin
        
        # Build constraints
        constraints = []
        for i in range(self.n_inputs):
            if self.expected_signals[i] != 0:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda v, idx=i, sig=self.expected_signals[i]: 
                           constraint_gain(v, idx, sig)
                })
        
        # Initial vector
        x0 = self._params_to_vector(self.param)
        
        # Optimization
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 1000, 'disp': False, 'ftol': self.config.tolerance},
            tol=self.config.tolerance
        )
        
        # Always update params with result (even if not fully converged)
        self.param = self._vector_to_params(result.x, n_neurons)
        
        return result.success, result.nit
    
    # =========================================================================
    # SINGLE MODEL TRAINING (WITH REINITIALIZATION UNTIL 100% SCR)
    # =========================================================================
    
    def _train_single_model(self, x_train: np.ndarray, y_train: np.ndarray,
                            x_test: np.ndarray, n_neurons: int) -> Tuple[bool, int, int]:
        """
        Train model, reinitializing until 100% SCR achieved on test set.
        
        Args:
            x_train: Training inputs (n_features, n_samples)
            y_train: Training targets (1, n_samples)
            x_test: Test inputs for conformity verification
            n_neurons: Number of hidden neurons
        
        Returns:
            Tuple: (achieved_100_scr, total_iterations, n_attempts)
        """
        best_scr = 0.0
        best_params = None
        total_iters = 0
        
        for attempt in range(self.MAX_REINIT_ATTEMPTS):
            if attempt > 0:
                # Reinitialize all weights
                self.param = self._initialize_parameters(n_neurons)
            
            # Optimize
            success, n_iter = self._optimize_single_attempt(x_train, y_train, n_neurons)
            total_iters += n_iter
            
            # Check conformity on TEST set
            scr = self._check_conformity_on_data(x_test)
            
            # Track best result
            if scr > best_scr:
                best_scr = scr
                best_params = copy.deepcopy(self.param)
            
            # If 100% SCR achieved, we're done
            if scr >= 1.0:
                return True, total_iters, attempt + 1
        
        # Use best configuration found (even if not 100%)
        if best_params is not None:
            self.param = best_params
        
        return False, total_iters, self.MAX_REINIT_ATTEMPTS
    
    # =========================================================================
    # FULL TRAINING (CONSTRUCTIVE WITHOUT WEIGHT REUSE)
    # =========================================================================
    
    def train(self) -> TrainingResults:
        """
        Execute constructive training with gain constraints.
        
        Key features:
        - Each model with n neurons is trained FROM SCRATCH
        - Reinitializes weights until 100% SCR achieved on test set
        - Skips configurations where no feasible solution found
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
        print(f"TRAINING - {self.METHOD_NAME} (Baseline with Constraints)")
        print("=" * 70)
        
        for n in range(1, self.config.max_neurons + 1):
            iter_start = time.time()
            
            # Initialize from scratch (no weight reuse)
            self.param = self._initialize_parameters(n)
            self.n_neurons = n
            
            # Train with reinitialization until 100% SCR
            success, n_iter, n_attempts = self._train_single_model(
                x_train, y_train, x_test, n
            )
            
            if not success:
                # Could not achieve 100% SCR - skip this configuration
                scr_achieved = self._check_conformity_on_data(x_test)
                print(f"n={n:3d} | ⚠️ Could not achieve 100% SCR after {n_attempts} attempts "
                      f"(best: {scr_achieved:.1%})")
                continue
            
            # Train metrics
            y_pred_train = self.predict(x_train)
            mse_train, rmse_train, r2_train = calculate_all_metrics(
                y_train.flatten(), y_pred_train.flatten()
            )
            
            # Test metrics
            y_pred_test = self.predict(x_test)
            mse_test, rmse_test, r2_test = calculate_all_metrics(
                y_test.flatten(), y_pred_test.flatten()
            )
            
            # Conformity (should be 100% by construction)
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
                attempts_str = f"({n_attempts} attempts)" if n_attempts > 1 else ""
                print(f"n={n:3d} | Train R²={r2_train:.4f} | Test R²={r2_test:.4f} | "
                      f"SCR={tcs:5.1%} | Time={iter_time:5.1f}s | Iters={n_iter} {attempts_str}")
            
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
        results.avg_time_per_model = np.mean(results.iteration_times) if results.iteration_times else 0
        
        # Restore best model
        if best_model_params is not None:
            self.param = best_model_params
            self.n_neurons = results.best_neurons
        
        # Summary
        print("\n" + "=" * 70)
        print(f"✅ Training completed in {results.total_time:.1f}s")
        if results.best_neurons > 0:
            print(f"🏆 Best: n={results.best_neurons} | R²={results.best_test_r2:.4f} | SCR={results.best_conformity:.1%}")
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
        Execute ONE constructive step with constraints (no weight reuse).

        Args:
            n: Number of neurons
            x_train: (n_features, n_samples) - transposed
            y_train: (1, n_samples)
            x_test: (n_features, n_samples) - transposed
            y_test: (1, n_samples)
        """
        iter_start = time.time()

        # Always from scratch (RIXM has NO weight reuse)
        self.param = self._initialize_parameters(n)
        self.n_neurons = n

        # Train with reinitialization until 100% SCR
        # NOTE: M4's _train_single_model does NOT take previous_param
        success, n_iter, n_attempts = self._train_single_model(
            x_train, y_train, x_test, n
        )

        training_time = time.time() - iter_start

        if not success:
            scr_achieved = self._check_conformity_on_data(x_test)
            return {
                'n_neurons': n,
                'success': False,
                'scr': scr_achieved,
                'n_attempts': n_attempts,
                'n_iterations': n_iter,
                'training_time': training_time
            }

        # Training metrics
        y_pred_train = self.predict(x_train)
        mse_train, rmse_train, r2_train = calculate_all_metrics(
            y_train.flatten(), y_pred_train.flatten()
        )

        # Test metrics
        y_pred_test = self.predict(x_test)
        mse_test, rmse_test, r2_test = calculate_all_metrics(
            y_test.flatten(), y_pred_test.flatten()
        )

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
        return copy.deepcopy(self.param)

    def set_params(self, param: Dict):
        self.param = copy.deepcopy(param)
        self.n_neurons = param['W1'].shape[0]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_rixm_constrained(train_inputs, train_targets, test_inputs, test_targets,
                            expected_signals, config: TrainingConfig,
                            dataset_name: str = "unknown"):
    """
    Factory to create Method 4 (RIXM Constrained) instance.
    """
    print("💻 Using NumPy implementation (RIXM Constrained - Baseline with Constraints)")
    return RIXMConstrainedNumpy(
        train_inputs, train_targets, test_inputs, test_targets,
        expected_signals, config, dataset_name
    )


# Alias for compatibility
RIXMConstrained = RIXMConstrainedNumpy