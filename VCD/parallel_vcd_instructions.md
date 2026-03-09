# Paralelização da VCD: Folds em Paralelo

## Contexto

O `run_dynamic_cv.py` atual executa os K folds sequencialmente dentro de cada
passo construtivo (cada n). Como os folds são independentes dentro de um mesmo n,
podemos paralelizá-los para usar todos os cores da CPU.

## O que precisa mudar

Apenas o arquivo `src/run_dynamic_cv.py`, na função `run_vcd_for_method()`.

### Problema principal

Cada fold mantém um **modelo com estado persistente** (weight reuse de n-1 → n).
Com multiprocessing, os objetos precisam ser serializados (pickle) entre processos.
A solução é usar `get_params()` / `set_params()` para salvar e restaurar o estado
entre passos paralelos.

### Estratégia: joblib com modelo recriado a cada batch

Em vez de manter K processos vivos com estado, a cada passo n:
1. Salvar os parâmetros de cada fold: `fold_states[k] = model_k.get_params()`
2. Executar K folds em paralelo via joblib
3. Cada worker recebe: (fold_idx, fold_data, fold_state, n, config)
4. O worker recria o modelo, restaura o estado, executa constructive_step(n)
5. Retorna: (result_dict, new_params)
6. Atualizar fold_states com os novos parâmetros

### Implementação

Adicionar uma função worker no módulo:

```python
def _vcd_fold_worker(fold_idx: int, fold_info: Dict, fold_state: Optional[Dict],
                     n: int, method_id: int, expected_signals: List[int],
                     training_config_dict: Dict, dataset_name: str) -> Tuple[Dict, Dict]:
    """
    Worker function for parallel fold execution.
    
    Runs one constructive step for one fold. Designed to be called via joblib.
    
    Args:
        fold_idx: Fold index (0 to K-1)
        fold_info: Dict with 'x_train', 'y_train', 'x_test', 'y_test' 
                   (arrays, NOT transposed - transpose inside worker)
        fold_state: Saved model parameters from previous step (None for n=1)
        n: Current number of neurons
        method_id: 1 or 2
        expected_signals: List of expected gain signs
        training_config_dict: Dict of TrainingConfig parameters
        dataset_name: Dataset name
    
    Returns:
        Tuple: (result_dict, new_params_dict)
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    from base import TrainingConfig
    
    # Reconstruct config
    config = TrainingConfig(**training_config_dict)
    config.seed = config.seed + fold_idx * 100
    config.verbose = 0
    
    # Create model (suppress output)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        model = create_model(
            method_id=method_id,
            train_inputs=fold_info['train_inputs'],
            train_targets=fold_info['train_targets'],
            test_inputs=fold_info['test_inputs'],
            test_targets=fold_info['test_targets'],
            expected_signals=expected_signals,
            config=config,
            dataset_name=dataset_name
        )
    
    # Restore state from previous step
    if fold_state is not None:
        model.set_params(fold_state)
        model.n_neurons = n - 1  # Will be expanded to n in constructive_step
    
    # Execute one constructive step
    result = model.constructive_step(
        n=n,
        x_train=fold_info['x_train'],
        y_train=fold_info['y_train'],
        x_test=fold_info['x_test'],
        y_test=fold_info['y_test']
    )
    
    # Get updated params for next step
    new_params = model.get_params()
    
    return result, new_params
```

Modificar o loop principal em `run_vcd_for_method()`:

```python
from joblib import Parallel, delayed
import os

# Detect number of cores (use all available, or K, whichever is smaller)
n_jobs = min(vcd_config.n_folds, os.cpu_count() or 1)

# Prepare fold data (store as regular arrays, not transposed)
# Keep transposed versions for the worker
fold_infos = []
for k in range(vcd_config.n_folds):
    fold_infos.append({
        'train_inputs': fold_data[k]['train_inputs'],   # (n_samples, n_features)
        'train_targets': fold_data[k]['train_targets'],  # (n_samples,)
        'test_inputs': fold_data[k]['test_inputs'],
        'test_targets': fold_data[k]['test_targets'],
        'x_train': fold_data[k]['x_train'],    # (n_features, n_samples) transposed
        'y_train': fold_data[k]['y_train'],    # (1, n_samples)
        'x_test': fold_data[k]['x_test'],
        'y_test': fold_data[k]['y_test'],
    })

# State tracking
fold_states = [None] * vcd_config.n_folds  # Parameters for each fold

# Convert training config to dict for serialization
config_template = create_training_config(vcd_config, method_config)
config_dict = {
    'max_neurons': config_template.max_neurons,
    'patience_constructive': config_template.patience_constructive,
    'patience_early_stopping': config_template.patience_early_stopping,
    'learning_rate': config_template.learning_rate,
    'lr_decay': config_template.lr_decay,
    'min_lr': config_template.min_lr,
    'iterations': config_template.iterations,
    'delta_perturbation': config_template.delta_perturbation,
    'tolerance': config_template.tolerance,
    'seed': vcd_config.seed,
    'use_gpu': False,
    'verbose': 0,
}

# Main VCD loop
for n in range(1, method_config.max_neurons + 1):
    step_start = time.time()
    
    # Run K folds in parallel
    parallel_results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_vcd_fold_worker)(
            fold_idx=k,
            fold_info=fold_infos[k],
            fold_state=fold_states[k],
            n=n,
            method_id=method_id,
            expected_signals=expected_signals,
            training_config_dict=config_dict,
            dataset_name=dataset_name
        )
        for k in range(vcd_config.n_folds)
    )
    
    # Unpack results
    J_T = 0.0
    all_folds_ok = True
    step_metrics = { ... }  # same as before
    
    for k, (result, new_params) in enumerate(parallel_results):
        fold_states[k] = new_params  # Save state for next step
        
        if 'success' in result and not result['success']:
            all_folds_ok = False
            break
        
        J_T += result['mse_test']
        # ... aggregate metrics same as before
    
    # ... rest of the loop (patience, logging) stays the same
```

## Pontos de atenção

1. **fold_data precisa guardar os dados NÃO transpostos** (para criar o modelo no
   worker) E os dados transpostos (para o constructive_step). Ajustar a preparação
   dos folds no início da função.

2. **O worker precisa recriar o modelo a cada step** porque joblib serializa tudo.
   Isso adiciona overhead de ~0.1s por fold por step (criação do modelo + 
   inicialização de pesos via linearização). Para M1 com steps de ~1s, o overhead 
   é ~10%. Para steps mais longos (M2, datasets maiores), é negligível.

3. **Fallback sequencial**: manter a opção de rodar sequencial via flag 
   `--parallel / --no-parallel` para debugging.

4. **joblib já está no ambiente** (dependência do scikit-learn).

5. **Manter `fold_models` para a versão sequencial** e usar `fold_states` + 
   `_vcd_fold_worker` para a versão paralela. Usar um flag para decidir qual rodar.

## Como implementar

Modifique APENAS `src/run_dynamic_cv.py`:

1. Adicionar a função `_vcd_fold_worker()` no módulo
2. Adicionar `--parallel` / `--no-parallel` flags ao argparse
3. Na função `run_vcd_for_method()`:
   - Manter o loop sequencial atual como fallback
   - Adicionar branch paralelo que usa joblib quando `--parallel`
   - Default: `--parallel` ativado
4. Ajustar a preparação de fold_data para incluir dados não-transpostos
   (necessários para criar o modelo no worker)

## Teste

```bash
# Paralelo (default)
python src/run_dynamic_cv.py --methods 1 --dataset computer_hardware --base-dir .

# Sequencial (fallback/debug)
python src/run_dynamic_cv.py --methods 1 --dataset computer_hardware --base-dir . --no-parallel

# Verificar que resultados são idênticos entre paralelo e sequencial (mesma seed)
```

## Ganho esperado

- computer_hardware (209 amostras, 6 features): ~3-4x speedup
- fish_toxicity (908 amostras, 6 features): ~4-5x speedup  
- Com K=5 folds e 5+ cores disponíveis: tempo reduz de ~30min para ~7min por dataset/método
