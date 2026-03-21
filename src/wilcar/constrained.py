"""
WILCAR Constrained (Method 2) - With Gain Sign Constraints
==========================================================
Federal University of Bahia - MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá

Features:
- Initial weights for 1st neuron via constrained optimization (SLSQP)
- Other neurons via Xavier initialization
- Training via scipy.optimize.minimize with gain constraints
- Constructive approach (adds neurons incrementally)
- Weight reuse from model n-1 to n
- Reinitializes new neurons until 100% SCR achieved or max attempts reached
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
from utils.metrics import calculate_all_metrics, calculate_classification_metrics


# =============================================================================
# NUMPY IMPLEMENTATION (CPU)
# =============================================================================

class WILCARConstrainedNumpy(BaseNeuralNetwork):
    """
    WILCAR with gain sign constraints - NumPy Implementation (CPU).
    
    Uses scipy.optimize.minimize (SLSQP) to ensure gain signs
    respect theoretical expectations throughout training.
    
    Key feature: Reinitializes new neurons until 100% SCR achieved on test set.
    """
    
    METHOD_NAME = "WILCAR_Constrained"
    MAX_REINIT_ATTEMPTS = 100  # Max attempts to find feasible configuration
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Get initial weights via constrained optimization
        print("\n🔧 Computing initial weights with gain constraints...")
        self.initial_weights = self._initialize_first_hidden_neuron()
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
    # FIRST NEURON INITIALIZATION (WITH CONSTRAINTS)
    # =========================================================================
    
    def _initialize_first_hidden_neuron(self) -> np.ndarray:
        """
        Get initial weights via constrained optimization on gain signs.
        
        Uses SLSQP to minimize discrepancy with linear regression
        while ensuring gains have correct signs.
        """
        import statsmodels.api as sm
        
        # Baseline linear regression (same as Method 1)
        inputs_with_const = sm.add_constant(self.train_inputs, prepend=True)
        ols_result = sm.OLS(self.train_targets, inputs_with_const).fit()
        
        # Regression coefficients
        k = np.array(ols_result.params[1:])  # Variable coefficients
        b = ols_result.params[0]              # Intercept
        
        # Equilibrium point (input mean)
        x_M = np.mean(self.train_inputs, axis=0)
        
        # Base point for gain calculation (training mean)
        x_base = np.mean(self.train_inputs, axis=0)
        delta = self.config.delta_perturbation
        
        # -----------------------------------------------------------------
        # Objective function: minimize discrepancy with linear regression
        # -----------------------------------------------------------------
        def objective(x):
            w, beta = x[:-1], x[-1]
            z = np.dot(w, x_M) + beta
            sig = self.sigmoid_safe(z)
            sig_prime = sig * (1 - sig)
            
            # Intercept equation
            eq_intercept = sig - sig_prime * np.dot(w, x_M) - b
            
            # Coefficient equations
            eq_coefs = sig_prime * w - k
            
            return eq_intercept**2 + np.sum(eq_coefs**2)
        
        # -----------------------------------------------------------------
        # Gain constraints
        # -----------------------------------------------------------------
        def gain_constraint(x, idx, expected_signal):
            """Constraint for gain: expected_signal * gain >= margin"""
            if expected_signal == 0:
                return 1.0  # Always satisfied
            
            w, beta = x[:-1], x[-1]
            
            # Calculate gain
            x_pert = x_base.copy()
            x_pert[idx] += delta
            
            y_base = self.sigmoid_safe(np.dot(w, x_base) + beta)
            y_pert = self.sigmoid_safe(np.dot(w, x_pert) + beta)
            
            gain = (y_pert - y_base) / delta
            
            margin = 0.05
            return expected_signal * gain - margin
        
        # Build constraint list
        constraints = []
        for i in range(self.n_inputs):
            if self.expected_signals[i] != 0:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=i, sig=self.expected_signals[i]: 
                           gain_constraint(x, idx, sig)
                })
        
        # -----------------------------------------------------------------
        # Optimization with multiple attempts
        # -----------------------------------------------------------------
        best_result = None
        best_residual = float('inf')
        
        print(f"🔄 Attempting constrained optimization (up to 10 attempts)...")
        
        for trial in range(10):
            # Initial guess
            if trial == 0:
                x0 = np.append(k, b)
            elif trial == 1:
                x0 = np.append(k * (1 + 0.1 * np.random.randn(self.n_inputs)),
                               b * (1 + 0.1 * np.random.randn()))
            else:
                x0 = np.random.randn(self.n_inputs + 1) * 0.5
            
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 5000, 'ftol': 1e-9},
                tol=1e-9
            )
            
            if result.success:
                residual = objective(result.x)
                
                # Verify gains were satisfied
                gains_ok = True
                for i in range(self.n_inputs):
                    if self.expected_signals[i] != 0:
                        constraint_val = gain_constraint(result.x, i, self.expected_signals[i])
                        if constraint_val < -1e-6:
                            gains_ok = False
                            break
                
                if gains_ok and residual < best_residual:
                    best_result = result.x
                    best_residual = residual
                    print(f"   ✓ Attempt {trial+1}: success (residual={residual:.2e})")
                else:
                    if not gains_ok:
                        print(f"   ✗ Attempt {trial+1}: gains not satisfied")
                    else:
                        print(f"   - Attempt {trial+1}: worse residual ({residual:.2e})")
            else:
                print(f"   ✗ Attempt {trial+1}: failed ({result.message})")
            
            # If very good solution found, stop
            if best_residual < 1e-8:
                break
        
        if best_result is not None:
            print(f"✅ Constrained optimization completed!")
            print(f"   Best residual: {best_residual:.2e}")
            return best_result
        else:
            print(f"⚠️ Constrained optimization did not converge.")
            print(f"   Fallback: linear regression weights")
            return np.append(k, b)
    
    # =========================================================================
    # NETWORK INITIALIZATION AND EXPANSION
    # =========================================================================
    
    def _initialize_parameters(self, n_neurons: int) -> Dict[str, np.ndarray]:
        """Initialize network parameters with n hidden neurons."""
        param = {
            'W1': np.random.randn(n_neurons, self.n_inputs) * np.sqrt(1. / self.n_inputs),
            'b1': np.zeros((n_neurons, 1)),
            'W2': np.random.randn(1, n_neurons) * np.sqrt(1. / n_neurons),
            'b2': np.zeros((1, 1))
        }
        
        # Apply initial weights (with constraints) to first neuron
        if self.initial_weights is not None:
            param['W1'][0, :] = self.initial_weights[:-1]
            param['b1'][0, 0] = self.initial_weights[-1]
        
        return param
    
    def _expand_network(self, previous_param: Dict[str, np.ndarray], 
                        n_neurons: int) -> Dict[str, np.ndarray]:
        """
        Expand network by adding a hidden neuron.
        Reuses weights from previous model (n-1).
        """
        new_param = copy.deepcopy(previous_param)
        
        # Add neuron to hidden layer
        new_w1 = np.random.randn(1, self.n_inputs) * np.sqrt(1. / self.n_inputs)
        new_param['W1'] = np.vstack([new_param['W1'], new_w1])
        new_param['b1'] = np.vstack([new_param['b1'], np.zeros((1, 1))])
        
        # Expand output layer connections
        new_w2 = np.random.randn(1, 1) * np.sqrt(1. / n_neurons)
        new_param['W2'] = np.hstack([new_param['W2'], new_w2])
        
        return new_param
    
    def _reinitialize_new_neuron(self, n_neurons: int):
        """Reinitialize only the newest neuron (keep others unchanged)."""
        # Reinitialize last neuron in hidden layer
        self.param['W1'][-1, :] = np.random.randn(self.n_inputs) * np.sqrt(1. / self.n_inputs)
        self.param['b1'][-1, 0] = 0.0
        
        # Reinitialize its output connection
        self.param['W2'][0, -1] = np.random.randn() * np.sqrt(1. / n_neurons)
    
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
        is_classification = self.config.task == 'classification'

        # Objective function: BCE for classification, MSE for regression
        def objective(vec):
            param = self._vector_to_params(vec, n_neurons)
            Z1 = param['W1'].dot(x) + param['b1']
            A1 = self.sigmoid_safe(Z1)
            Z2 = param['W2'].dot(A1) + param['b2']
            y_pred = self.sigmoid_safe(Z2)
            if is_classification:
                y_pred_clipped = np.clip(y_pred, 1e-12, 1 - 1e-12)
                return float(-np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped)))
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
        
        # Always update params with result
        self.param = self._vector_to_params(result.x, n_neurons)
        
        return result.success, result.nit
    
    # =========================================================================
    # SINGLE MODEL TRAINING (WITH REINITIALIZATION UNTIL 100% SCR)
    # =========================================================================
    
    def _train_single_model(self, x_train: np.ndarray, y_train: np.ndarray,
                            x_test: np.ndarray, n_neurons: int,
                            previous_param: Optional[Dict]) -> Tuple[bool, int, int]:
        """
        Train model, reinitializing new neuron until 100% SCR achieved on test set.
        
        Args:
            x_train: Training inputs (n_features, n_samples)
            y_train: Training targets (1, n_samples)
            x_test: Test inputs for conformity verification
            n_neurons: Number of hidden neurons
            previous_param: Parameters from previous model (for weight reuse)
        
        Returns:
            Tuple: (achieved_100_scr, total_iterations, n_attempts)
        """
        best_scr = 0.0
        best_params = None
        total_iters = 0
        
        for attempt in range(self.MAX_REINIT_ATTEMPTS):
            if attempt == 0:
                # First attempt: use expanded network (or initialize)
                if n_neurons == 1 or previous_param is None:
                    # Initialize from scratch if first neuron OR if no valid previous model
                    # Note: _initialize_parameters preserves initial_weights for first neuron
                    self.param = self._initialize_parameters(n_neurons)
                else:
                    self.param = self._expand_network(previous_param, n_neurons)
            else:
                # Subsequent attempts: reinitialize
                if n_neurons == 1 or previous_param is None:
                    # Full reinitialization (keeping first neuron's constrained weights)
                    self.param = self._initialize_parameters(n_neurons)
                else:
                    # Restore previous params and expand with new random neuron
                    self.param = self._expand_network(previous_param, n_neurons)
            
            self.n_neurons = n_neurons
            
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
    # FULL TRAINING (CONSTRUCTIVE)
    # =========================================================================
    
    def train(self) -> TrainingResults:
        """
        Execute full constructive training with gain constraints.

        Key features:
        - Weight reuse from model n-1 to n
        - Reinitializes new neuron until 100% SCR achieved on test set
        - Skips configurations where no feasible solution found
        """
        results = TrainingResults()
        results.method_name = self.METHOD_NAME
        results.dataset_name = self.dataset_name
        results.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        is_classification = self.config.task == 'classification'

        # Prepare data
        x_train = self.train_inputs.T
        y_train = self.train_targets.reshape(1, -1)
        x_test = self.test_inputs.T
        y_test = self.test_targets.reshape(1, -1)

        # Tracking
        best_metric_so_far = -float('inf')
        no_improve_count = 0
        best_model_params = None
        previous_param = None

        start_time = time.time()

        print("\n" + "=" * 70)
        print(f"TRAINING - {self.METHOD_NAME}")
        if is_classification:
            print("Task: CLASSIFICATION (BCE objective, MCC selection)")
        print("=" * 70)

        for n in range(1, self.config.max_neurons + 1):
            iter_start = time.time()

            # Train with reinitialization until 100% SCR
            success, n_iter, n_attempts = self._train_single_model(
                x_train, y_train, x_test, n, previous_param
            )

            if not success:
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
                cls_train = calculate_classification_metrics(y_train.flatten(), y_pred_train.flatten())
                cls_test = calculate_classification_metrics(y_test.flatten(), y_pred_test.flatten())
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
                          f"SCR={tcs:5.1%} | Time={iter_time:5.1f}s | Iters={n_iter} {attempts_str}")
                else:
                    print(f"n={n:3d} | Train R²={r2_train:.4f} | Test R²={r2_test:.4f} | "
                          f"SCR={tcs:5.1%} | Time={iter_time:5.1f}s | Iters={n_iter} {attempts_str}")

            # Constructive early stopping
            if selection_metric > best_metric_so_far:
                best_metric_so_far = selection_metric
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.config.patience_constructive:
                print(f"\n⏹️ Early stopping: {self.config.patience_constructive} models without improvement")
                break

            # Save params for next iteration (weight reuse)
            previous_param = copy.deepcopy(self.param)

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
                print(f"🏆 Best: n={results.best_neurons} | R²={results.best_test_r2:.4f} | SCR={results.best_conformity:.1%}")
        else:
            print("⚠️ No feasible configuration found")
        print("=" * 70)

        return results

    # =========================================================================
    # VCD SUPPORT - STEP-BY-STEP CONSTRUCTIVE INTERFACE
    # =========================================================================

    def constructive_step(self, n: int, x_train: np.ndarray, y_train: np.ndarray,
                          x_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Execute ONE constructive step with gain constraints.

        Includes reinitialization logic to achieve 100% SCR.
        Maintains self._vcd_previous_param for weight reuse between calls.
        Must be called sequentially: n=1, n=2, n=3, ...

        Used by Dynamic Cross-Validation (VCD) to control the constructive
        process from outside the model.

        Args:
            n: Number of neurons for this configuration
            x_train: Training inputs (n_features, n_samples) - transposed format
            y_train: Training targets (1, n_samples)
            x_test: Test inputs (n_features, n_samples)
            y_test: Test targets (1, n_samples)

        Returns:
            Dict with metrics. Includes 'success' key (bool) indicating
            whether 100% SCR was achieved.
        """
        iter_start = time.time()

        # Initialize previous_param tracker if needed
        if not hasattr(self, '_vcd_previous_param'):
            self._vcd_previous_param = None

        # Train with reinitialization until 100% SCR (existing logic)
        success, n_iter, n_attempts = self._train_single_model(
            x_train, y_train, x_test, n, self._vcd_previous_param
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

        # Save params for weight reuse in next step
        self._vcd_previous_param = copy.deepcopy(self.param)

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

        # Signal conformity
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
        """Return a deep copy of current network parameters."""
        if self.param is None:
            return None
        return copy.deepcopy(self.param)

    def set_params(self, param: Dict):
        """Restore network parameters from a saved copy."""
        self.param = copy.deepcopy(param)
        self.n_neurons = param['W1'].shape[0]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_wilcar_constrained(train_inputs, train_targets, test_inputs, test_targets,
                              expected_signals, config: TrainingConfig,
                              dataset_name: str = "unknown"):
    """
    Factory to create Method 2 (WILCAR Constrained) instance.
    """
    print("💻 Using NumPy implementation (WILCAR Constrained)")
    return WILCARConstrainedNumpy(
        train_inputs, train_targets, test_inputs, test_targets,
        expected_signals, config, dataset_name
    )


# Alias for compatibility
WILCARConstrained = WILCARConstrainedNumpy