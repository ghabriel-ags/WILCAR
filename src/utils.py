# src/utils.py
import os
import numpy as np
import pandas as pd

def sigmoid(x):
    """Função sigmoide numericamente estável."""
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    """Derivada da função sigmoide."""
    z = sigmoid(x)
    return z * (1 - z)

def predict_network(params, x):
    """
    Faz previsões usando a rede neural com os parâmetros fornecidos.

    Parâmetros:
        params (dict): Parâmetros da rede neural.
        x (array): Dados de entrada.

    Retorna:
        array: Previsões da rede neural.
    """
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']

    if x.shape[0] != W1.shape[1]:  # Corrige se a entrada tiver dimensões erradas
        x = x.T

    Z1 = np.dot(W1, x) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    return sigmoid(Z2)

def calculate_gain(params, x, var_index, delta=0.1):
    """
    Calcula o ganho de uma variável de entrada específica.

    Parâmetros:
        params (dict): Parâmetros da rede neural.
        x (array): Dados de entrada.
        var_index (int): Índice da variável de entrada.
        delta (float, opcional): Pequena perturbação aplicada para calcular o ganho.

    Retorna:
        float: Média do ganho calculado.
    """
    x_perturbed = np.copy(x)
    x_perturbed[var_index, :] += delta

    y_base = predict_network(params, x)
    y_perturbed = predict_network(params, x_perturbed)

    gain = (y_perturbed - y_base) / delta
    return np.mean(gain, axis=1)[0]  # Retorna a média do ganho para estabilidade

def calculate_and_compare_gains(adjusted_weights, x_base, expected_signals, n_inputs):
    """
    Calcula os sinais dos ganhos com base nos pesos ajustados e os compara com os sinais esperados.

    Parâmetros:
    - adjusted_weights: array-like
        Pesos ajustados, incluindo os pesos dos neurônios de entrada e o bias.
    - x_base: array-like
        Vetor de entrada base utilizado para calcular os ganhos.
    - expected_signals: list
        Lista contendo os sinais esperados para os ganhos de cada variável de entrada.
    - n_inputs: int
        Número de variáveis de entrada.

    Retorna:
    - None
        A função imprime os resultados da comparação entre os ganhos calculados e os sinais esperados.
    """
    # Separando os pesos e o bias
    w, beta = adjusted_weights[:-1], adjusted_weights[-1]

    # Calculando a saída base da rede
    y_base = sigmoid(np.dot(w, x_base) + beta)

    calculated_signals = []
    for i in range(n_inputs):
        x_var = x_base.copy()
        x_var[i] += 0.1  # Pequena perturbação

        # Calculando nova saída com perturbação
        new_y = sigmoid(np.dot(w, x_var) + beta)

        # Calculando o sinal do ganho
        gain_signal = np.sign((new_y - y_base) / 0.1)
        calculated_signals.append(gain_signal)

    # Comparando os sinais calculados com os sinais esperados
    for i, (calculated, expected) in enumerate(zip(calculated_signals, expected_signals)):
        print(f"Variável {i + 1}: Ganho calculado = {calculated}, Ganho esperado = {expected}")
        if expected != 0 and calculated != expected:
            print(f"  -> ❌ Divergência detectada na variável {i + 1}.")
        elif expected == 0:
            print(f"  -> ℹ️ Ganho desconhecido esperado; ganho calculado foi {calculated}.")
        else:
            print(f"  -> ✅ Correspondência encontrada.")

    return calculated_signals

def mse(y_true, y_pred):
    """Calcula o erro quadrático médio (MSE)."""
    return np.mean((y_true - y_pred) ** 2)

def save_metrics(metrics, experiment_name, filename):
    """Salva métricas em um CSV."""

    results_dir = f"results/{filename}_{experiment_name}"
    os.makedirs(results_dir, exist_ok=True)

    metrics = {k: [float(v) for v in values] for k, values in metrics.items()}

    df = pd.DataFrame(metrics)
    df.to_csv(f"{results_dir}/{filename}_{experiment_name}_metrics.csv", index=False)

def get_expected_signals(n_inputs):
    """
    Solicita ao usuário os sinais esperados para o ganho de cada variável de entrada.
    
    Retorna:
        list: Lista contendo os sinais esperados.
    """
    expected_signals = []
    print("Informe os sinais esperados dos ganhos variáveis de entrada.")
    print("Digite 1 para um ganho positivo, -1 para um ganho negativo e 0 caso o ganho seja desconhecido:")

    for i in range(n_inputs):
        while True:
            try:
                signal = int(input(f"Sinal esperado para o ganho da variável {i+1}: ").strip())
                if signal in [1, -1, 0]:
                    expected_signals.append(signal)
                    break
                else:
                    print("Entrada inválida. Digite 1, -1, ou 0.")
            except ValueError:
                print("Entrada inválida. Digite um número inteiro (1, -1 ou 0).")

    return expected_signals

def gain_constraint(W, x, n, input_size, var_index, expected_gain):
    """
    Define uma restrição de ganho para a otimização.

    Parâmetros:
        W (array): Vetor de pesos da rede.
        x (array): Dados de entrada.
        n (int): Número de neurônios na camada oculta.
        input_size (int): Número de variáveis de entrada.
        var_index (int): Índice da variável de entrada.
        expected_gain (int): Sinal do ganho esperado.

    Retorna:
        float: Resultado da restrição.
    """
    params = param_vector_to_dict(W, n, input_size)
    gain = calculate_gain(params, x, var_index)

    margin = 0.01  # Pequena margem para tolerância
    if expected_gain == 1:
        return gain - margin
    elif expected_gain == -1:
        return -gain - margin
    return 0

def param_vector_to_dict(vector, n, input_size):
    """Converte um vetor plano de volta para um dicionário de parâmetros com os formatos corretos."""
    n_W1 = n * input_size
    n_b1 = n
    n_W2 = n
    n_b2 = 1

    W1 = vector[:n_W1].reshape(n, input_size)
    b1 = vector[n_W1:n_W1 + n_b1].reshape(n, 1)
    W2 = vector[n_W1 + n_b1:n_W1 + n_b1 + n_W2].reshape(1, n)
    b2 = vector[-n_b2:].reshape(1, 1)

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def param_dict_to_vector(params):
    """Converte um dicionário de parâmetros em um vetor plano."""
    W1_flat = params['W1'].flatten()
    b1_flat = params['b1'].flatten()
    W2_flat = params['W2'].flatten()
    b2_flat = params['b2'].flatten()
    return np.concatenate([W1_flat, b1_flat, W2_flat, b2_flat])

def objective_function(W, x, y, input_size):
    """
    Função objetivo usada na otimização dos pesos, baseada no erro quadrático médio (MSE).

    Parâmetros:
        W (array): Vetor de pesos da rede.
        x (array): Dados de entrada.
        y (array): Saídas verdadeiras.
        input_size (int): Número de variáveis de entrada.

    Retorna:
        float: MSE entre os valores reais e as previsões do modelo.
    """
    n = (len(W) - 1) // (input_size + 1)
    params = param_vector_to_dict(W, n, input_size)
    predictions = predict_network(params, x)
    return mse(y, predictions.flatten())