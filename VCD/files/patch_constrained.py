"""
PATCH: constrained.py
======================
Adicionar os 3 métodos abaixo à classe WILCARConstrainedNumpy,
ANTES da seção "FACTORY FUNCTION".

Inserir logo após o método train() (após a linha 611 no arquivo atual).
"""

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
