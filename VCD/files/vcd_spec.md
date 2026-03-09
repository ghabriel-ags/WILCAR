# Especificação Técnica: Validação Cruzada Dinâmica (VCD)

## 1. Conceito

A VCD integra o k-fold cross-validation ao processo construtivo da rede neural.
Em vez de usar um único split para decidir quando parar de adicionar neurônios,
a VCD avalia cada configuração (n neurônios) em K folds e usa o erro agregado
como critério de parada.

```
ALGORITMO VCD:

Para n = 1, 2, ..., max_neurons:
    Para cada fold k = 1, ..., K:
        - Se n == 1: inicializar modelo_k com dados do fold k
        - Senão: expandir modelo_k de n-1 para n neurônios (weight reuse)
        - Treinar modelo_k no treino do fold k
        - Avaliar no teste do fold k → J_k(n)
    
    J_T(n) = Σ J_k(n)  (soma dos erros de teste dos K folds)
    
    Se J_T(n) < melhor_J_T:
        melhor_J_T = J_T(n)
        n* = n
        patience_counter = 0
    Senão:
        patience_counter += 1
    
    Se patience_counter >= patience:
        PARAR
    
RETREINO FINAL:
    Treinar modelo com n* neurônios usando TODOS os dados
```

## 2. Estratégia de Implementação

### Princípio: NÃO modificar os arquivos existentes

Os métodos atuais (unconstrained.py, constrained.py) funcionam e serão usados
no k-fold estático. A VCD será implementada em um novo arquivo que reutiliza
os componentes internos dos métodos existentes.

### Arquivos a criar:
- `run_dynamic_cv.py` — Script principal da VCD (novo)

### Arquivos a modificar:
- `unconstrained.py` — Adicionar método `constructive_step()` (aditivo, não quebra nada)
- `constrained.py` — Adicionar método `constructive_step()` (aditivo, não quebra nada)

## 3. Modificação nos Métodos WILCAR

### 3.1 M1: WILCARUnconstrainedNumpy — adicionar `constructive_step()`

```python
def constructive_step(self, n: int, x_train: np.ndarray, y_train: np.ndarray,
                      x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Executa UM passo construtivo: adiciona o n-ésimo neurônio e treina.
    
    Mantém o estado interno (self.param) para weight reuse entre chamadas.
    Deve ser chamado sequencialmente: n=1, n=2, n=3, ...
    
    Args:
        n: Número de neurônios para esta configuração
        x_train: Dados de treino (n_features, n_samples) - formato transposto
        y_train: Targets de treino (1, n_samples)
        x_test: Dados de teste (n_features, n_samples)
        y_test: Targets de teste (1, n_samples)
    
    Returns:
        Dict com métricas: {
            'n_neurons': int,
            'mse_train': float, 'rmse_train': float, 'r2_train': float,
            'mse_test': float, 'rmse_test': float, 'r2_test': float,
            'scr': float, 'conformity_details': list,
            'training_time': float
        }
    """
    iter_start = time.time()
    
    # Inicializar ou expandir
    if n == 1:
        self.param = self._initialize_parameters(n)
        self.n_neurons = n
    else:
        self._expand_network(n)
    
    # Treinar (backprop)
    mse_train, rmse_train, r2_train = self._train_single_model(x_train, y_train)
    
    # Métricas de teste
    y_pred_test = self.predict(x_test)
    mse_test, rmse_test, r2_test = calculate_all_metrics(
        y_test.flatten(), y_pred_test.flatten()
    )
    
    # Conformidade
    scr, conf_details = self.calculate_conformity(self.predict)
    
    return {
        'n_neurons': n,
        'mse_train': mse_train, 'rmse_train': rmse_train, 'r2_train': r2_train,
        'mse_test': mse_test, 'rmse_test': rmse_test, 'r2_test': r2_test,
        'scr': scr, 'conformity_details': conf_details,
        'training_time': time.time() - iter_start
    }
```

### 3.2 M2: WILCARConstrainedNumpy — adicionar `constructive_step()`

```python
def constructive_step(self, n: int, x_train: np.ndarray, y_train: np.ndarray,
                      x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Executa UM passo construtivo COM constraints.
    
    Inclui a lógica de reinicialização até 100% SCR.
    Mantém self._previous_param para weight reuse entre chamadas.
    """
    iter_start = time.time()
    
    # Inicializar previous_param tracker se necessário
    if not hasattr(self, '_previous_param'):
        self._previous_param = None
    
    # Treinar com reinicialização (lógica existente)
    success, n_iter, n_attempts = self._train_single_model(
        x_train, y_train, x_test, n, self._previous_param
    )
    
    if not success:
        scr_achieved = self._check_conformity_on_data(x_test)
        return {
            'n_neurons': n,
            'success': False,
            'scr': scr_achieved,
            'training_time': time.time() - iter_start
        }
    
    # Salvar params para weight reuse no próximo passo
    self._previous_param = copy.deepcopy(self.param)
    
    # Métricas de treino
    y_pred_train = self.predict(x_train)
    mse_train, rmse_train, r2_train = calculate_all_metrics(
        y_train.flatten(), y_pred_train.flatten()
    )
    
    # Métricas de teste
    y_pred_test = self.predict(x_test)
    mse_test, rmse_test, r2_test = calculate_all_metrics(
        y_test.flatten(), y_pred_test.flatten()
    )
    
    # Conformidade
    scr, conf_details = self.calculate_conformity(self.predict)
    
    return {
        'n_neurons': n,
        'success': True,
        'mse_train': mse_train, 'rmse_train': rmse_train, 'r2_train': r2_train,
        'mse_test': mse_test, 'rmse_test': rmse_test, 'r2_test': r2_test,
        'scr': scr, 'conformity_details': conf_details,
        'n_attempts': n_attempts, 'n_iterations': n_iter,
        'training_time': time.time() - iter_start
    }
```

### 3.3 Métodos auxiliares adicionais

Ambas as classes precisam de:

```python
def get_params(self) -> Dict[str, np.ndarray]:
    """Retorna cópia dos parâmetros atuais."""
    return copy.deepcopy(self.param)

def set_params(self, param: Dict[str, np.ndarray]):
    """Restaura parâmetros salvos."""
    self.param = copy.deepcopy(param)
    self.n_neurons = param['W1'].shape[0]
```

## 4. Script run_dynamic_cv.py

### 4.1 Estrutura principal

```python
"""
Dynamic Cross-Validation (VCD) for Constructive Neural Networks
===============================================================
Federal University of Bahia — MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá

A VCD integra o k-fold CV ao processo construtivo: para cada número de
neurônios n, avalia-se o desempenho em K folds e usa-se o erro agregado
J_T(n) = Σ J_k(n) para determinar o n* ótimo.

Usage:
    python run_dynamic_cv.py --dataset computer_hardware --method wilcar_unconstrained
    python run_dynamic_cv.py --dataset fish_toxicity --method wilcar_constrained
    python run_dynamic_cv.py --dataset computer_hardware --methods 1 2
    python run_dynamic_cv.py --n-folds 10 --dataset aquatic_toxicity
"""
```

### 4.2 Configuração

```python
@dataclass
class VCDConfig:
    """Configuração da Validação Cruzada Dinâmica."""
    n_folds: int = 5
    seed: int = 0
    shuffle: bool = True
    
    # Parâmetros construtivos
    max_neurons: int = 750
    patience_constructive: int = 150
    
    # Parâmetros de treinamento
    patience_early_stopping: int = 250
    learning_rate: float = 0.1
    lr_decay: float = 0.999
    min_lr: float = 1e-3
    iterations: int = 2501
    delta_perturbation: float = 0.1
    tolerance: float = 1e-6
    
    # Métrica de agregação
    aggregation_metric: str = 'mse_test'  # Qual métrica somar entre folds
    
    # Métodos
    methods: List[int] = field(default_factory=lambda: [1, 2])
    
    # Datasets
    datasets: List[str] = field(default_factory=lambda: [
        'computer_hardware',
        'fish_toxicity', 
        'aquatic_toxicity'
    ])
```

### 4.3 Fluxo principal — `run_vcd_for_method()`

```python
def run_vcd_for_method(method_id: int, inputs: np.ndarray, targets: np.ndarray,
                       expected_signals: List[int], vcd_config: VCDConfig,
                       dataset_name: str) -> VCDResults:
    """
    Executa VCD completa para um método em um dataset.
    
    Fluxo:
    1. Cria K folds
    2. Para cada n (neurônios):
       a. Para cada fold k: executa constructive_step(n)
       b. Agrega J_T(n) = Σ mse_test_k(n)
       c. Verifica patience
    3. Identifica n*
    4. Retreina com todos os dados usando n*
    """
    
    # Setup K-Fold
    kfold = KFold(n_splits=vcd_config.n_folds, shuffle=vcd_config.shuffle,
                  random_state=vcd_config.seed)
    
    # Preparar folds: criar modelo para cada fold
    fold_models = []
    fold_data = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(inputs)):
        X_train, X_test = inputs[train_idx], inputs[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]
        
        # Normalizar (fit no treino do fold)
        X_train_n, X_test_n, y_train_n, y_test_n = normalize_data(
            X_train, X_test, y_train, y_test
        )
        
        # Criar instância do modelo para este fold
        config = create_training_config(vcd_config)
        config.seed = vcd_config.seed + fold_idx * 100
        config.verbose = 0
        
        model = create_model(
            method_id=method_id,
            train_inputs=X_train_n,
            train_targets=y_train_n,
            test_inputs=X_test_n,
            test_targets=y_test_n,
            expected_signals=expected_signals,
            config=config,
            dataset_name=dataset_name
        )
        
        fold_models.append(model)
        fold_data.append({
            'x_train': X_train_n.T,       # Transpor para formato (features, samples)
            'y_train': y_train_n.reshape(1, -1),
            'x_test': X_test_n.T,
            'y_test': y_test_n.reshape(1, -1)
        })
    
    # =========================================================
    # LOOP PRINCIPAL DA VCD
    # =========================================================
    best_J_T = float('inf')
    best_n = 0
    patience_counter = 0
    
    # Armazenamento por neurônio
    history = []          # Lista de dicts, um por n
    fold_histories = [[] for _ in range(vcd_config.n_folds)]
    
    for n in range(1, vcd_config.max_neurons + 1):
        step_start = time.time()
        
        J_T = 0.0
        step_metrics = {
            'n': n,
            'fold_mse_test': [],
            'fold_r2_test': [],
            'fold_scr': [],
            'fold_r2_train': [],
        }
        
        all_folds_ok = True
        
        for k in range(vcd_config.n_folds):
            result = fold_models[k].constructive_step(
                n=n,
                x_train=fold_data[k]['x_train'],
                y_train=fold_data[k]['y_train'],
                x_test=fold_data[k]['x_test'],
                y_test=fold_data[k]['y_test']
            )
            
            fold_histories[k].append(result)
            
            # Para M2 (constrained): verificar se convergiu
            if 'success' in result and not result['success']:
                all_folds_ok = False
                break
            
            J_T += result['mse_test']
            step_metrics['fold_mse_test'].append(result['mse_test'])
            step_metrics['fold_r2_test'].append(result['r2_test'])
            step_metrics['fold_scr'].append(result['scr'])
            step_metrics['fold_r2_train'].append(result['r2_train'])
        
        if not all_folds_ok:
            # Skip esta configuração (para M2)
            print(f"  n={n:3d} | ⚠️ Fold falhou, pulando")
            continue
        
        step_metrics['J_T'] = J_T
        step_metrics['mean_r2_test'] = np.mean(step_metrics['fold_r2_test'])
        step_metrics['std_r2_test'] = np.std(step_metrics['fold_r2_test'])
        step_metrics['mean_scr'] = np.mean(step_metrics['fold_scr'])
        step_metrics['time'] = time.time() - step_start
        
        history.append(step_metrics)
        
        # Log
        print(f"  n={n:3d} | J_T={J_T:.6f} | R²={step_metrics['mean_r2_test']:.4f}"
              f"±{step_metrics['std_r2_test']:.4f} | SCR={step_metrics['mean_scr']:.1%}"
              f" | {step_metrics['time']:.1f}s")
        
        # Critério de parada
        if J_T < best_J_T:
            best_J_T = J_T
            best_n = n
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= vcd_config.patience_constructive:
            print(f"\n  ⏹️ VCD early stopping: {vcd_config.patience_constructive} "
                  f"passos sem melhora")
            print(f"  n* = {best_n} (J_T* = {best_J_T:.6f})")
            break
    
    # =========================================================
    # RETREINO FINAL COM TODOS OS DADOS
    # =========================================================
    print(f"\n  🔄 Retreinando modelo final com n*={best_n} usando todos os dados...")
    
    # Normalizar todos os dados
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_all_norm = scaler_X.fit_transform(inputs)
    y_all_norm = scaler_y.fit_transform(targets.reshape(-1, 1)).ravel()
    
    # Criar modelo final (usa todos os dados tanto para treino quanto para "teste")
    final_config = create_training_config(vcd_config)
    final_config.max_neurons = best_n
    final_config.patience_constructive = best_n + 10  # Não parar antes de n*
    final_config.verbose = 1
    
    final_model = create_model(
        method_id=method_id,
        train_inputs=X_all_norm,
        train_targets=y_all_norm,
        test_inputs=X_all_norm,     # Mesmo conjunto (sem teste separado)
        test_targets=y_all_norm,
        expected_signals=expected_signals,
        config=final_config,
        dataset_name=dataset_name
    )
    
    # Treinar modelo final construtivamente até n*
    # Opção: usar train() normal com max_neurons=n*
    final_results = final_model.train()
    
    # =========================================================
    # COMPILAR RESULTADOS
    # =========================================================
    return VCDResults(
        method_id=method_id,
        dataset_name=dataset_name,
        optimal_neurons=best_n,
        best_J_T=best_J_T,
        history=history,
        fold_histories=fold_histories,
        final_model_results=final_results,
        vcd_config=vcd_config
    )
```

### 4.4 Dataclass de Resultados

```python
@dataclass
class VCDResults:
    """Resultados completos da VCD."""
    method_id: int
    dataset_name: str
    optimal_neurons: int
    best_J_T: float
    
    # Histórico por número de neurônios
    history: List[Dict]           # [{n, J_T, fold_mse_test, mean_r2_test, ...}]
    fold_histories: List[List]    # K listas, cada uma com resultados por n
    
    # Modelo final (retreinado com todos os dados)
    final_model_results: TrainingResults
    
    # Configuração
    vcd_config: VCDConfig = None
    
    # Métricas do k-fold no n*
    @property
    def cv_metrics_at_optimal(self) -> Dict:
        """Métricas do k-fold agregado no n* ótimo."""
        for h in self.history:
            if h['n'] == self.optimal_neurons:
                return {
                    'mean_r2_test': h['mean_r2_test'],
                    'std_r2_test': h['std_r2_test'],
                    'mean_scr': h['mean_scr'],
                    'J_T': h['J_T'],
                    'fold_r2_test': h['fold_r2_test'],
                    'fold_mse_test': h['fold_mse_test'],
                }
        return {}
```

## 5. Outputs esperados

```
results/dynamic_cv/<timestamp>/
├── <dataset_name>/
│   ├── method_<id>/
│   │   ├── vcd_history.csv         # J_T(n) para cada n
│   │   ├── fold_results.csv        # Métricas por fold por n
│   │   ├── final_model_results.json
│   │   └── vcd_curve.png           # Gráfico J_T(n) vs n
│   └── ...
├── consolidated_vcd_results.csv
├── comparison_static_vs_dynamic.csv  # Se disponível
└── vcd_config.json
```

## 6. Pontos de atenção na implementação

### 6.1 Weight reuse entre passos construtivos
Cada fold mantém seu próprio modelo com estado persistente.
A chamada `constructive_step(n)` deve avançar o modelo do fold de n-1 → n.
NÃO recriar o modelo a cada n.

### 6.2 Seed por fold
Cada fold deve ter seed diferente para a inicialização Xavier dos novos neurônios:
`config.seed = base_seed + fold_idx * 100`

### 6.3 M2 (constrained): falha em algum fold
Se um fold não consegue 100% SCR para determinado n, há duas opções:
- **Opção A**: Pular aquele n inteiro (implementação inicial)
- **Opção B**: Usar o melhor resultado parcial daquele fold

Começar com Opção A (mais simples e conservadora).

### 6.4 Retreino final
O modelo final é treinado com train() normal, com max_neurons=n*.
Como usa todos os dados (sem split), as métricas de "teste" serão iguais
às de treino. As métricas de generalização são as do k-fold (J_T(n*)).

### 6.5 Custo computacional
VCD com K=5 e max_neurons=750 = até 3750 treinamentos por método/dataset.
Para M1 (backprop, ~1s/step), isso é ~1h por dataset.
Para M2 (SLSQP + reinit, ~10-100s/step), pode ser muito lento.
Considerar usar max_neurons menores para M2 inicialmente.

## 7. Ordem de implementação sugerida

1. Adicionar `constructive_step()` + `get_params()` + `set_params()` a M1
2. Criar `run_dynamic_cv.py` com suporte apenas a M1
3. Testar com computer_hardware (dataset mais rápido)
4. Adicionar `constructive_step()` a M2
5. Estender `run_dynamic_cv.py` para M2
6. Testar com os 3 datasets
7. Adicionar geração de gráficos e tabelas comparativas
