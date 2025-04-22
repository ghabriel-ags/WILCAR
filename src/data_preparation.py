import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Definição dos diretórios de entrada e saída dos dados
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

# Garante que o diretório de dados processados exista
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

def load_data(filename):
    """
    Carrega os dados a partir de um arquivo CSV localizado em 'data/raw/'.
    
    Parâmetros:
    - filename: Nome do arquivo (sem extensão)

    Retorna:
    - DataFrame contendo os dados brutos ou None se o arquivo não for encontrado.
    """
    file_path = os.path.join(RAW_DATA_PATH, f"{filename}.csv")

    if not os.path.exists(file_path):
        print(f"❌ Erro: O arquivo '{file_path}' não foi encontrado.")
        return None

    try:
        data = pd.read_csv(file_path, header=None, sep=';')

        if data.empty:
            print(f"❌ Erro: O arquivo '{file_path}' está vazio.")
            return None

        print(f"📂 Dados carregados com sucesso de '{file_path}'.")
        return data
    except Exception as e:
        print(f"❌ Erro ao carregar os dados: {e}")
        return None

def preprocess_data(df, filename):
    """
    Normaliza os dados, trata valores ausentes e divide em conjuntos de treino/teste.
    
    Parâmetros:
    - df: DataFrame contendo os dados brutos.
    - filename: Nome do arquivo de entrada (usado para salvar os dados processados).

    Retorna:
    - train_inputs: Entradas para o treinamento.
    - test_inputs: Entradas para o teste.
    - train_targets: Saídas para o treinamento.
    - test_targets: Saídas para o teste.
    """
    if df is None or df.empty:
        print("❌ Erro: O conjunto de dados está vazio.")
        return None, None, None, None

    print("\n### INÍCIO DO PRÉ-PROCESSAMENTO ###\n")

    # Remoção de valores ausentes
    initial_size = df.shape[0]
    df.dropna(inplace=True)
    final_size = df.shape[0]

    if final_size == 0:
        print("❌ Erro: Todos os dados foram removidos por conterem valores ausentes.")
        return None, None, None, None

    print(f"✅ Dados faltantes removidos. {initial_size - final_size} linhas eliminadas.")

    # Normalização dos dados
    print("\n🔄 Normalizando os dados (escala entre 0 e 1)...")
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(df)
    print("✅ Dados normalizados com sucesso.")

    # Separação entre entrada e saída
    inputs_all = data_normalized[:, :-1]
    targets_all = data_normalized[:, -1]

    print("\n🔹 O conjunto de dados será dividido em:")
    print("  - 80% para TREINAMENTO 🏋️‍♂️")
    print("  - 20% para TESTE 🧪")

    # Divisão entre treino e teste
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(
        inputs_all, targets_all, test_size=0.2, random_state=42
    )

    print(f"\n📊 Conjuntos de dados gerados:")
    print(f"  🔹 Treino: {train_inputs.shape[0]} instâncias")
    print(f"  🔹 Teste: {test_inputs.shape[0]} instâncias")

    # Salvando os dados de treino e teste separadamente
    train_file_path = os.path.join(PROCESSED_DATA_PATH, f"{filename}_train.csv")
    test_file_path = os.path.join(PROCESSED_DATA_PATH, f"{filename}_test.csv")

    train_df = pd.DataFrame(np.column_stack((train_inputs, train_targets)))
    test_df = pd.DataFrame(np.column_stack((test_inputs, test_targets)))

    train_df.to_csv(train_file_path, index=False, header=False, sep=";")
    test_df.to_csv(test_file_path, index=False, header=False, sep=";")

    print(f"\n📂 Dados processados salvos:")
    print(f"  ✅ Treino: {train_file_path} ({train_inputs.shape[0]} instâncias)")
    print(f"  ✅ Teste: {test_file_path} ({test_inputs.shape[0]} instâncias)")

    print("\n### PRÉ-PROCESSAMENTO FINALIZADO ###\n")

    return train_inputs, test_inputs, train_targets, test_targets