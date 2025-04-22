# src/method1_constrained.py
import statsmodels.api as sm
from sklearn.metrics import r2_score
from scipy.optimize import minimize
import numpy as np
from utils import mse, predict_network, gain_constraint, objective_function, calculate_and_compare_gains

class Method1Constrained:
    """Método 1 (C-WILCAR) com restrições nos sinais dos ganhos."""

    def __init__(self, x_train, y_train, x_test, y_test, expected_signals):
        self.x_train = x_train.T if x_train.shape[0] < x_train.shape[1] else x_train
        self.y_train = y_train.reshape(-1, 1)
        self.x_test = x_test.T if x_test.shape[0] < x_test.shape[1] else x_test
        self.y_test = y_test.reshape(-1, 1)

        self.input_size = self.x_train.shape[0]
        self.expected_signals = np.array(expected_signals)

        print(f"Debugging sizes:")
        print(f"  x_train.shape = {self.x_train.shape}")
        print(f"  y_train.shape = {self.y_train.shape}")
        print(f"  expected_signals.shape = {self.expected_signals.shape}")
        print(f"  input_size = {self.input_size}")

        if len(self.expected_signals) != self.input_size:
            raise ValueError(f"Erro: expected_signals tem tamanho {len(self.expected_signals)}, mas deveria ter {self.input_size}.")

        self.best_params = None
        self.best_metrics = {
            'Train MSE': float('inf'),
            'Train RMSE': float('inf'),
            'Train R2': -float('inf'),
            'Test MSE': float('inf'),
            'Test RMSE': float('inf'),
            'Test R2': -float('inf'),
            'Best Neurons': None
        }

    def initialize_weights_with_constraints(self, n):
        """Inicializa os pesos garantindo que respeitem as restrições dos sinais dos ganhos."""
        print(f"\nInicializando pesos com restrições para {n} neurônios...")

        if self.x_train.shape[0] > self.x_train.shape[1]:  
            self.x_train = self.x_train.T  

        self.y_train = self.y_train.reshape(-1, 1)

        inputs_all_cte = sm.add_constant(self.x_train, prepend=True)  

        print(f"x_train.shape: {self.x_train.shape}")
        print(f"y_train.shape: {self.y_train.shape}")
        print(f"inputs_all_cte.shape: {inputs_all_cte.shape}")

        model = sm.OLS(self.y_train, inputs_all_cte).fit()
        print(model.summary())

        b = model.params[0]  
        k = model.params[1:]  

        x0 = np.random.rand(self.input_size + 1)

        constraints = [
            {'type': 'ineq', 'fun': lambda W, i=i: gain_constraint(W, self.x_train, n, self.input_size, i, self.expected_signals[i])}
            for i in range(self.input_size) if self.expected_signals[i] != 0
        ]

        result = minimize(
            objective_function,
            x0,
            args=(np.mean(self.x_train, axis=0), k, b, self.input_size),
            constraints=constraints,
            method='SLSQP',
            options={'maxiter': 10000, 'ftol': 1e-06}
        )

        if not result.success:
            raise ValueError(f"Erro na otimização dos pesos iniciais: {result.message}")

        print(f"Pesos ajustados: {result.x}")

        extended_params = self.extend_initial_params(result.x, n)
        return self.reshape_initial_weights(extended_params, n)

    def extend_initial_params(self, x, n_neurons):
        """Estende os parâmetros iniciais para incluir pesos e bias adicionais para a camada de saída."""
        W1_b1 = x

        variance = 2 / (n_neurons + 1)
        W2 = np.random.normal(0, np.sqrt(variance), (1, n_neurons))
        b2 = np.zeros(1)

        return np.concatenate([W1_b1, W2.flatten(), b2])

    def reshape_initial_weights(self, x, n_neurons):
        """Redimensiona os pesos para formar as matrizes necessárias para a rede neural."""
        n_W1 = n_neurons * self.input_size
        n_b1 = n_neurons
        n_W2 = n_neurons
        n_b2 = 1

        W1 = x[:n_W1].reshape(n_neurons, self.input_size)
        b1 = x[n_W1:n_W1 + n_b1].reshape(n_neurons, 1)
        W2 = x[n_W1 + n_b1:n_W1 + n_b1 + n_W2].reshape(1, n_neurons)
        b2 = x[-n_b2:].reshape(1, 1)

        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def train(self, max_neurons=50, loss_function="mse"):
        """Treina a rede neural garantindo restrições de sinais nos ganhos."""
        mse_train, mse_test, rmse_train, rmse_test, r2_train, r2_test = [], [], [], [], [], []

        for n in range(1, max_neurons + 1):
            print(f"\n📌 Treinando com {n} neurônios...")

            params = self.initialize_weights_with_constraints(n)

            constraints = [
                {'type': 'ineq', 'fun': lambda W, i=i: gain_constraint(W, self.x_train, n, self.input_size, i, self.expected_signals[i])}
                for i in range(self.input_size) if self.expected_signals[i] != 0
            ]

            W0 = np.concatenate([params['W1'].flatten(), params['b1'].flatten(),
                                 params['W2'].flatten(), params['b2'].flatten()])

            result = minimize(
                objective_function,
                W0,
                args=(self.x_train, self.y_train, self.input_size),
                constraints=constraints,
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-05}
            )

            if not result.success:
                print(f"⚠️ Treinamento interrompido: {result.message}")
                break

            optimized_params = self.reshape_initial_weights(result.x, n)

            y_train_pred = predict_network(optimized_params, self.x_train).flatten()
            y_test_pred = predict_network(optimized_params, self.x_test).flatten()

            mse_train.append(mse(self.y_train, y_train_pred))
            mse_test.append(mse(self.y_test, y_test_pred))
            rmse_train.append(np.sqrt(mse_train[-1]))
            rmse_test.append(np.sqrt(mse_test[-1]))
            r2_train.append(r2_score(self.y_train.flatten(), y_train_pred))
            r2_test.append(r2_score(self.y_test.flatten(), y_test_pred))

            calculate_and_compare_gains(optimized_params, self.x_test, self.expected_signals, self.input_size)

            if mse_test[-1] < self.best_metrics['Test MSE']:
                self.best_metrics.update({
                    'Train MSE': mse_train[-1],
                    'Train RMSE': rmse_train[-1],
                    'Train R2': r2_train[-1],
                    'Test MSE': mse_test[-1],
                    'Test RMSE': rmse_test[-1],
                    'Test R2': r2_test[-1],
                    'Best Neurons': n
                })

            self.best_params = optimized_params

        return mse_train, mse_test, rmse_train, rmse_test, r2_train, r2_test, self.best_metrics