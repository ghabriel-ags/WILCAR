"""
PATCH: unconstrained.py
========================
Adicionar os 3 métodos abaixo à classe WILCARUnconstrainedNumpy,
ANTES da seção "IMPLEMENTAÇÃO PYTORCH (GPU)".

Inserir logo após o método train() (após a linha 307 no arquivo atual).
"""

    # =========================================================================
    # VCD SUPPORT - STEP-BY-STEP CONSTRUCTIVE INTERFACE
    # =========================================================================

    def constructive_step(self, n: int, x_train: np.ndarray, y_train: np.ndarray,
                          x_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Execute ONE constructive step: add the n-th neuron and train.
        
        Maintains internal state (self.param) for weight reuse between calls.
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
            Dict with metrics: {
                'n_neurons', 'mse_train', 'rmse_train', 'r2_train',
                'mse_test', 'rmse_test', 'r2_test',
                'scr', 'conformity_details', 'training_time'
            }
        """
        iter_start = time.time()
        
        # Initialize or expand network
        if n == 1:
            self.param = self._initialize_parameters(n)
            self.n_neurons = n
        else:
            self._expand_network(n)
        
        # Train via backpropagation
        mse_train, rmse_train, r2_train = self._train_single_model(x_train, y_train)
        
        # Test metrics
        y_pred_test = self.predict(x_test)
        mse_test, rmse_test, r2_test = calculate_all_metrics(
            y_test.flatten(), y_pred_test.flatten()
        )
        
        # Signal conformity
        scr, conf_details = self.calculate_conformity(self.predict)
        
        return {
            'n_neurons': n,
            'mse_train': mse_train, 'rmse_train': rmse_train, 'r2_train': r2_train,
            'mse_test': mse_test, 'rmse_test': rmse_test, 'r2_test': r2_test,
            'scr': scr, 'conformity_details': conf_details,
            'training_time': time.time() - iter_start
        }

    def get_params(self) -> Dict:
        """Return a deep copy of current network parameters."""
        if self.param is None:
            return None
        return {k: v.copy() for k, v in self.param.items()}

    def set_params(self, param: Dict):
        """Restore network parameters from a saved copy."""
        self.param = {k: v.copy() for k, v in param.items()}
        self.n_neurons = param['W1'].shape[0]
