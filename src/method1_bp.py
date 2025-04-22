import numpy as np
import statsmodels.api as sm
from scipy.optimize import fsolve
from sklearn.metrics import r2_score
from utils import mse, calculate_and_compare_gains, sigmoid

class Method1:
    def __init__(self, x_train, y_train, x_test, y_test, expected_signals, n, previous_model=None, initial_weights=None):
        """ Inicializa a rede neural sem restrições. """

        self.x_train = x_train.T if x_train.shape[0] > x_train.shape[1] else x_train
        self.y_train = y_train.reshape(-1, 1)
        self.x_test = x_test.T if x_test.shape[0] > x_test.shape[1] else x_test
        self.y_test = y_test.reshape(-1, 1)
        self.expected_signals = expected_signals
        self.n = n
        self.lr = 0.1  # Taxa de aprendizado

        if previous_model is None:
            self.param = self.initialize_first_neuron_weights()
        else:
            self.param = self.expand_network(previous_model, n)

    def initialize_first_neuron_weights(self):
        """ Inicializa os pesos do primeiro neurônio oculto utilizando Linearização + Taylor. """

        # Obtendo número de variáveis de entrada corretamente
        n_inputs = self.x_train.shape[0]  # Número de variáveis de entrada

        # 🔹 Ajuste da regressão linear múltipla
        inputs_all_cte = sm.add_constant(self.x_train.T, prepend=True)
        res = sm.OLS(self.y_train, inputs_all_cte).fit()
        print(res.summary())  # Exibe resultados da regressão

        # 🔹 Extração dos coeficientes da regressão
        x_M = np.mean(self.x_train, axis=1)
        k = res.params[1:]  # Coeficientes angulares (pesos)
        b = res.params[0]  # Intercepto da regressão

        # 🔹 Chute inicial para os pesos e bias
        x0 = np.random.rand(n_inputs + 1, 1)

        def fsolve_equations(x):
            """ Define o sistema de equações baseado na aproximação de Taylor. """
            w, beta = x[:-1], x[-1]
            f_eq = [sigmoid(np.dot(w, x_M) + beta) -
                    (sigmoid(np.dot(w, x_M) + beta) - (sigmoid(np.dot(w, x_M) + beta)) ** 2) * np.dot(w, x_M) - b]
            for i in range(n_inputs):
                f_eq.append((sigmoid(np.dot(w, x_M) + beta) - (sigmoid(np.dot(w, x_M) + beta)) ** 2) * w[i] - k[i])
            return f_eq

        # 🔹 Resolvendo o sistema para encontrar os pesos iniciais
        result = fsolve(fsolve_equations, x0).flatten()
        print(f"Pesos e bias iniciais ajustados: {result}")

        # 🔹 Convertendo `result` para um array NumPy seguro
        result = np.array(result, dtype=np.float64)

        # 🔹 Obtendo um exemplo correto do conjunto de teste (com as variáveis de entrada)
        x_base = self.x_test[:, 0].reshape(-1, 1)  # CORRIGIDO! Pegamos a primeira amostra corretamente

        # 🔹 Checagem de dimensionalidade corrigida
        expected_input_size = n_inputs  # Agora usamos o número correto de variáveis de entrada

        if len(result[:-1]) != expected_input_size:
            raise ValueError(
                f"Erro de dimensionalidade: os pesos têm {len(result[:-1])}, "
                f"mas deveriam ter {expected_input_size} (número de variáveis de entrada)."
            )

        # 🔹 Avaliação dos sinais dos ganhos
        calculate_and_compare_gains(result, x_base, self.expected_signals, n_inputs)

        # 🔹 Criando um dicionário para os parâmetros da rede neural
        param = {
            'W1': np.zeros((self.n, n_inputs), dtype=np.float64),
            'b1': np.zeros((self.n, 1), dtype=np.float64),
            'W2': np.random.randn(1, self.n) * np.sqrt(1. / self.n),
            'b2': np.zeros((1, 1), dtype=np.float64)
        }

        # 🔹 Atribuindo os pesos corretamente
        param['W1'][0, :] = np.array(result[:-1], dtype=np.float64).reshape(1, -1)
        param['b1'][0, 0] = float(result[-1])  # Garante que seja um escalar

        return param

    def expand_network(self, previous_model, n):
        """ Expande a rede neural preservando os pesos dos neurônios aprendidos. """
        new_param = previous_model.param.copy()

        if n > previous_model.n:
            additional_weights = np.random.randn(1, self.x_train.shape[0]) * np.sqrt(1. / self.x_train.shape[0])
            new_param['W1'] = np.vstack([new_param['W1'], additional_weights])
            new_param['b1'] = np.vstack([new_param['b1'], np.zeros((1, 1))])
            new_param['W2'] = np.hstack([new_param['W2'], np.random.randn(1, 1) * np.sqrt(1. / n)])

        return new_param

    def train_test(self, iter=1500):
        """Treina a rede e avalia o desempenho nos dados de treino e teste."""
        for _ in range(iter):
            Yh = self.forward().reshape(self.y_train.shape)
            self.backward(Yh)

        # Cálculo das métricas para os dados de treinamento
        Yh_train = self.forward().reshape(1, -1)
        mse_train = mse(self.y_train.flatten(), Yh_train.flatten())
        rmse_train = np.sqrt(mse_train)
        r2_train = r2_score(self.y_train.flatten(), Yh_train.flatten())

        # Cálculo das métricas para os dados de teste
        Yh_test = self.forward(self.x_test).reshape(1, -1)
        mse_test = mse(self.y_test.flatten(), Yh_test.flatten())
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(self.y_test.flatten(), Yh_test.flatten())

        return mse_train, rmse_train, r2_train, mse_test, rmse_test, r2_test
    
    def forward(self, X=None):
        """
        Executa a propagação para frente na rede neural.

        Parâmetros:
            X (array, opcional): Dados de entrada. Se None, usa self.x_train.
        
        Retorna:
            A2 (array): Saída da rede neural.
        """
        if X is None:
            X = self.x_train  # Usa os dados de treinamento se X não for fornecido

        Z1 = np.dot(self.param['W1'], X) + self.param['b1']
        A1 = sigmoid(Z1)
        Z2 = np.dot(self.param['W2'], A1) + self.param['b2']
        A2 = sigmoid(Z2)

        return A2

    def backward(self, Yh):
        """
        Retropropagação do erro e atualização dos pesos.

        Parâmetros:
        - Yh: Saída da rede neural (previsões).
        """
        m = self.y_train.shape[1]  # Número de amostras

        # 🔹 Cálculo do erro na saída (garantir que seja (1, n_amostras))
        dZ2 = (Yh - self.y_train).reshape(1, -1)  # Garantindo (1, n_amostras)

        # 🔹 Recalcular `A1` manualmente sem alterar `forward`
        Z1 = np.dot(self.param['W1'], self.x_train) + self.param['b1']  # (n_ocultos, n_amostras)
        A1 = sigmoid(Z1)  # (n_ocultos, n_amostras)

        # 🔹 Multiplicação correta: (1, n_amostras) x (n_amostras, n_ocultos) → (1, n_ocultos)
        dW2 = np.dot(dZ2, A1.T) / m  # (1, n_ocultos)
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m  # (1,1)

        # 🔹 Propagação do erro para a camada oculta
        dZ1 = np.dot(self.param['W2'].T, dZ2) * (A1 * (1 - A1))  # (n_ocultos, n_amostras)

        # 🔹 Multiplicação correta: (n_ocultos, n_amostras) x (n_amostras, n_inputs) → (n_ocultos, n_inputs)
        dW1 = np.dot(dZ1, self.x_train.T) / m  # (n_ocultos, n_inputs)
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m  # (n_ocultos,1)

        # 🔹 Atualização dos pesos
        self.param['W1'] -= self.lr * dW1
        self.param['b1'] -= self.lr * db1
        self.param['W2'] -= self.lr * dW2
        self.param['b2'] -= self.lr * db2