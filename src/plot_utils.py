import matplotlib.pyplot as plt
import os

def generate_plots(mse_train, mse_test, rmse_train, rmse_test, r2_train, r2_test, experiment_name, filename):
    """
    Gera e salva gráficos de métricas do experimento.

    Parâmetros:
    - mse_train, mse_test: Erro Quadrático Médio (MSE) no treino e teste.
    - rmse_train, rmse_test: Raiz do Erro Quadrático Médio (RMSE) no treino e teste.
    - r2_train, r2_test: R2 no treino e teste.
    - experiment_name: Nome do experimento (ex: "WILCAR", "C-WILCAR (MSE)").
    - filename: Nome base do arquivo CSV utilizado.
    """
    save_path = f"results/{filename}_{experiment_name}"
    os.makedirs(save_path, exist_ok=True)  # Garante que o diretório existe

    metrics = {
        "Train MSE": mse_train,
        "Test MSE": mse_test,
        "Train RMSE": rmse_train,
        "Test RMSE": rmse_test,
        "Train R2": r2_train,
        "Test R2": r2_test
    }

    for metric_name, values in metrics.items():
        if values and len(values) > 0:  # Evita erro ao tentar plotar listas vazias
            plt.figure(figsize=(10, 5))
            plt.plot(values, marker='o')
            plt.xlabel("Hidden Nodes")
            plt.ylabel(metric_name)
            plt.title(f"{experiment_name} - {metric_name}")
            plt.grid(True)

            file_path = f"{save_path}/{metric_name.replace(' ', '_').lower()}.png"
            plt.savefig(file_path)
            plt.show()
        else:
            print(f"⚠️ Aviso: Não há dados suficientes para gerar o gráfico de {metric_name}.")

    print(f"📊 Gráficos salvos em: {save_path}")