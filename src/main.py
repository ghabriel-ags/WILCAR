import time
import numpy as np
import os
from data_preparation import load_data, preprocess_data
from method1_bp import Method1
from method1_constrained import Method1Constrained
from utils import get_expected_signals, save_metrics
from plot_utils import generate_plots

# Caminho do diretório onde os arquivos CSV estão armazenados
RAW_DATA_PATH = "data/raw"

def show_project_info():
    """Exibe informações do projeto no console."""
    print("=" * 80)
    print("SISTEMA DE TREINAMENTO DE REDES NEURAIS - MÉTODO 1 (WILCAR)")
    print("=" * 80)
    print("UNIVERSIDADE FEDERAL DA BAHIA - ESCOLA POLITÉCNICA")
    print("PROGRAMA DE PÓS-GRADUAÇÃO EM ENGENHARIA INDUSTRIAL")
    print("Discente: Ghabriel Anton Gomes de Sá")
    print("Orientadores: Marcelo Embiruçu e Cristiano Fontes")
    print("=" * 80)

def main():
    # Exibir informações do projeto
    show_project_info()

    # Passo 1: Importação do conjunto de dados
    while True:
        filename = input("\nDigite o nome do arquivo CSV (sem extensão) ou 'sair' para encerrar: ").strip()

        if filename.lower() == "sair":
            print("Encerrando o programa.")
            return

        file_path = os.path.join(RAW_DATA_PATH, filename + ".csv")

        if os.path.exists(file_path):
            break
        else:
            print(f"❌ Erro: O arquivo '{file_path}' não foi encontrado. Tente novamente.")

    data = load_data(filename)

    if data is None:
        print("❌ Erro: Problema na leitura dos dados. Encerrando o programa.")
        return

    # Passo 2: Pré-processamento dos dados
    train_inputs, test_inputs, train_targets, test_targets = preprocess_data(data, filename)

    if train_inputs is None:
        print("❌ Erro no pré-processamento dos dados. Encerrando o programa.")
        return

    # Passo 3: Obtenção dos sinais esperados
    n_inputs = train_inputs.shape[1]
    expected_signals = get_expected_signals(n_inputs)

    # Passo 4: Escolha do experimento
    experiment_map = {
        "1": ("WILCAR", False, "mse"),
        "2": ("C-WILCAR (MSE)", True, "mse"),
        "3": ("C-WILCAR (RMSE)", True, "rmse"),
        "0": ("Encerrar", None, None)
    }

    while True:
        print("\nEscolha o experimento:")
        print("1 - Método 1 (WILCAR) Sem Restrição")
        print("2 - Método 1 (WILCAR) Com Restrição (MSE)")
        print("3 - Método 1 (WILCAR) Com Restrição (RMSE)")
        print("0 - Encerrar o programa")

        choice = input("Digite o número do experimento desejado: ").strip()

        if choice in experiment_map:
            experiment_name, use_constraints, loss_function = experiment_map[choice]
            if choice == "0":
                print("Encerrando o programa.")
                return
            break
        print("❌ Opção inválida. Digite 1, 2, 3 ou 0.")

    print(f"\n🚀 Iniciando treinamento do {experiment_name}...")

    # Passo 5: Instanciar o modelo
    try:
        if use_constraints:
            model = Method1Constrained(train_inputs, train_targets, test_inputs, test_targets, expected_signals)
        else:
            model = Method1(train_inputs, train_targets, test_inputs, test_targets, expected_signals, n=1)  # Iniciando com um neurônio
    except Exception as e:
        print(f"❌ Erro ao instanciar o modelo: {e}")
        return

    # Inicialização das métricas antes do treinamento
    best_metrics = {
        'Train MSE': None,
        'Train RMSE': None,
        'Train R2': None,
        'Test MSE': None,
        'Test RMSE': None,
        'Test R2': None,
        'Best Neurons': None
    }

    # Medir tempo de treinamento
    start_time = time.time()
    try:
        mse_train, rmse_train, r2_train, mse_test, rmse_test, r2_test = model.train_test()

        # Atualizar `best_metrics` apenas se o treinamento foi bem-sucedido
        best_metrics.update({
            'Train MSE': mse_train,
            'Train RMSE': rmse_train,
            'Train R2': r2_train,
            'Test MSE': mse_test,
            'Test RMSE': rmse_test,
            'Test R2': r2_test,
            'Best Neurons': model.n  # Número de neurônios usados
        })
    except Exception as e:
        print(f"❌ Erro durante o treinamento: {e}")
        return
    end_time = time.time()

    print("\n⏳ Tempo total de treinamento:", round(end_time - start_time, 2), "segundos")

    # Exibição das métricas do melhor modelo encontrado
    if all(value is not None for value in best_metrics.values()):
        best_metrics = {key: float(value) if isinstance(value, np.generic) else value for key, value in best_metrics.items()}
        print("\n📊 Melhor modelo encontrado:")
        for metric, value in best_metrics.items():
            print(f"  ✅ {metric}: {value:.6f}")
    else:
        print("⚠️ Nenhum modelo válido foi encontrado. Verifique os dados e tente novamente.")
        return

    # Salvando apenas as métricas do melhor modelo encontrado
    save_metrics(best_metrics, experiment_name, filename)

    # Geração dos gráficos
    if all(value is not None for value in [mse_train, mse_test, rmse_train, rmse_test, r2_train, r2_test]):
        generate_plots(mse_train, mse_test, rmse_train, rmse_test, r2_train, r2_test, experiment_name, filename)
    else:
        print("⚠️ Aviso: Dados insuficientes para gerar gráficos.")

if __name__ == "__main__":
    main()