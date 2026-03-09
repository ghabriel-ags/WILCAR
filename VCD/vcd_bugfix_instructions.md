# Correções Urgentes na VCD — 3 Issues

## ISSUE 1 (CRÍTICO): ELM (M5) produzindo R² negativo

### Evidência
M5/airfoil_self_noise com VCD: R² = -0.078 no n*=242
O Artigo 1 reportou R² = 0.508 para o mesmo dataset/método.
Todos os R² por fold são negativos (-0.04 a -0.17), indicando que o modelo
está PIOR que predizer a média. Algo está fundamentalmente errado.

### Como debugar

1. Primeiro, rode um teste RÁPIDO fora da VCD para confirmar que o ELM funciona:

```python
import numpy as np
import sys; sys.path.insert(0, 'src')
from elm import ELMNumpy
from base import TrainingConfig
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Carregar airfoil
data = pd.read_csv('data/raw/airfoil_self_noise.csv', header=None, sep=';')
data = data.dropna(axis=1, how='all')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalizar
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sx = MinMaxScaler(); sy = MinMaxScaler()
X_train_n = sx.fit_transform(X_train)
X_test_n = sx.transform(X_test)
y_train_n = sy.fit_transform(y_train.reshape(-1,1)).ravel()
y_test_n = sy.transform(y_test.reshape(-1,1)).ravel()

config = TrainingConfig(max_neurons=50, patience_constructive=20, seed=0, verbose=0)
model = ELMNumpy(X_train_n, y_train_n, X_test_n, y_test_n, [-1,0,-1,1,-1], config, 'airfoil')

# Teste 1: train() normal — deve dar R² positivo
results = model.train()
print(f"train() normal: R²={results.best_test_r2:.4f}, n={results.best_neurons}")

# Teste 2: constructive_step — comparar com train()
model2 = ELMNumpy(X_train_n, y_train_n, X_test_n, y_test_n, [-1,0,-1,1,-1], config, 'airfoil')
x_train_T = X_train_n.T
y_train_T = y_train_n.reshape(1, -1)
x_test_T = X_test_n.T
y_test_T = y_test_n.reshape(1, -1)

for n in range(1, 11):
    r = model2.constructive_step(n, x_train_T, y_train_T, x_test_T, y_test_T)
    print(f"constructive_step n={n}: R²={r['r2_test']:.4f}, MSE={r['mse_test']:.6f}")
```

Se train() dá R² positivo mas constructive_step dá negativo, o bug está no
constructive_step do ELM. Provável causa: formato de dados (a conversão
transposed → row-format pode estar errada ou o cálculo de métricas de teste
pode estar usando dados no formato errado).

Se train() também dá R² negativo, o bug é mais profundo (possivelmente no
_train_single_model do ELM ou na normalização).

2. Pontos a verificar no constructive_step do ELM (src/elm/elm.py):

- A conversão de formato: `X_train = x_train.T` deve produzir (n_samples, n_features)
- `Y_train = y_train.flatten()` deve produzir (n_samples,)
- `_train_single_model(X_train, Y_train)` espera (n_samples, n_features) e (n_samples,)
- Para métricas de teste: `H_test = self._hidden_nodes(X_test)` onde X_test = x_test.T
- `y_pred_test = np.dot(H_test, self.param['output_weights'])` deve dar (n_samples,)
- `calculate_all_metrics(Y_test, y_pred_test)` — verificar que shapes batem

3. Verificar TAMBÉM o _vcd_fold_worker em run_dynamic_cv.py:

- Para M5, o worker recria o modelo com `train_inputs` (n_samples, n_features)
- Depois chama constructive_step com `x_train` (n_features, n_samples)
- Verificar se `fold_info['train_inputs']` e `fold_info['x_train']` são realmente
  transpostos um do outro
- Verificar se a seed está sendo setada corretamente no worker

4. Verificar M6 (ELM+R) também — provavelmente tem o mesmo bug.

### Diagnóstico: duas hipóteses

**Hipótese A (bug no código):** O constructive_step do ELM tem um problema de
formato de dados ou cálculo de métricas. EVIDÊNCIA A FAVOR: se o teste 1
(train() normal) dá R²~0.5 mas o teste 2 (constructive_step) dá R² negativo
para os mesmos dados e mesmos n's.

**Hipótese B (comportamento legítimo da VCD com ELM):** O ELM sem weight reuse
é fundamentalmente instável — cada n reinicializa tudo do zero. No k-fold
estático, cada fold encontra seu próprio n* (que pode ser diferente entre folds).
Na VCD, todos os folds são avaliados no MESMO n. Se o ELM é muito instável,
fixar o mesmo n para todos os folds produz resultados ruins em média, mesmo que
individualmente alguns folds tenham R² alto naquele n. EVIDÊNCIA A FAVOR: se o
constructive_step individual reproduz resultados semelhantes ao train(), mas
quando rodado em 5 folds com VCD a média de R² fica negativa.

Para distinguir, adicionar um teste 3 ao script de debug:

```python
# Teste 3: simular VCD manualmente com 2 folds
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
for n in [10, 20, 50]:
    r2s = []
    for train_idx, test_idx in kfold.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        sx2 = MinMaxScaler(); sy2 = MinMaxScaler()
        Xtr_n = sx2.fit_transform(Xtr); Xte_n = sx2.transform(Xte)
        ytr_n = sy2.fit_transform(ytr.reshape(-1,1)).ravel()
        yte_n = sy2.transform(yte.reshape(-1,1)).ravel()
        m = ELMNumpy(Xtr_n, ytr_n, Xte_n, yte_n, [-1,0,-1,1,-1],
                     TrainingConfig(max_neurons=n, patience_constructive=n+5, seed=0, verbose=0), 'test')
        res = m.train()
        r2s.append(res.best_test_r2)
    print(f"n={n}: fold R²s = {[f'{r:.4f}' for r in r2s]}, mean = {np.mean(r2s):.4f}")
```

Se a média de R² em n fixo é consistentemente baixa/negativa mesmo com train()
normal, é Hipótese B e o problema é conceitual, não um bug.

### Ação para cada cenário

**Se Hipótese A (bug):** Corrigir o constructive_step e/ou o worker.

**Se Hipótese B (comportamento legítimo):**
O ELM (M5) e ELM+R (M6) são inadequados para VCD porque não têm weight reuse.
Nesse caso, para M5/M6, o approach correto é o k-fold estático (cada fold
encontra seu próprio n*), NÃO a VCD.

Opções de implementação:
(a) Manter VCD para M5/M6 mas reportar os resultados como estão — evidenciando 
    que a VCD não é adequada para métodos sem weight reuse. Isso é um resultado
    válido para a dissertação.
(b) Para M5/M6, usar o k-fold estático no lugar da VCD. Implementar isso como
    um fallback automático no run_dynamic_cv.py quando o método não tem weight
    reuse.
(c) Para M5/M6 na VCD, em vez de reinicializar cada fold do zero a cada n,
    manter os hidden weights fixos do n anterior e apenas adicionar neurônios
    incrementalmente (como o ELM incremental, I-ELM). Isso daria "weight reuse"
    ao ELM. MAS isso mudaria a natureza do método M5, que é treinar do zero.

Recomendação: implementar (a) por agora — rodar e reportar. Se R² é
consistentemente negativo para M5/M6 com VCD, isso será discutido no Cap. 5
como uma limitação da VCD para métodos sem weight reuse. É um resultado legítimo
e interessante. NÃO tentar "consertar" o comportamento se não for bug.

### Resumo: o que fazer
1. Rodar os 3 testes de debug
2. Se Hipótese A: corrigir o bug
3. Se Hipótese B: documentar e seguir em frente (resultados são válidos)
4. Em ambos os casos, implementar Issues 2 e 3 (conformity_details e r2_train)


## ISSUE 2: Salvar conformity_details por variável

### Problema
O fold_details_at_optimal.csv salva apenas o SCR agregado (ex: 0.5),
mas NÃO salva quais variáveis estão conformes. Precisamos disso para a
Tabela 5.12 do Capítulo 5.

### O que precisa mudar

No run_dynamic_cv.py, na função run_vcd_for_method(), quando compilamos
fold_metrics_at_optimal, o constructive_step já retorna 'conformity_details'
(uma lista de dicts, um por variável, com 'variable', 'calculated_gain',
'calculated_signal', 'expected_signal', 'conformity').

Salvar essa informação num arquivo adicional. Na função _save_vcd_results():

```python
# Conformity details at optimal (per variable, per fold)
if results.fold_conformity_details:
    conf_rows = []
    for k, fold_details in enumerate(results.fold_conformity_details):
        for var_detail in fold_details:
            conf_rows.append({
                'fold': k,
                'variable': var_detail['variable'],
                'calculated_gain': var_detail['calculated_gain'],
                'calculated_signal': var_detail['calculated_signal'],
                'expected_signal': var_detail['expected_signal'],
                'conformity': var_detail['conformity'],
            })
    pd.DataFrame(conf_rows).to_csv(
        method_dir / "conformity_details_at_optimal.csv", index=False
    )
```

Para alimentar isso, no loop principal, quando guardamos os resultados no n*,
precisamos salvar os conformity_details de cada fold. Adicionar ao VCDResults:
```python
fold_conformity_details: List[List[Dict]] = field(default_factory=list)
```

E no loop principal, quando step_metrics é registrado, guardar os
conformity_details de cada fold. Isso requer que o loop agregue essa
informação. O constructive_step já retorna 'conformity_details' no dict
de resultado — basta coletá-los.

No loop VCD, dentro do `for k in range(n_folds)`:
```python
step_metrics['fold_conformity_details'].append(result.get('conformity_details', []))
```

E ao compilar fold_metrics_at_optimal:
```python
fold_conformity_details=[
    optimal_step['fold_conformity_details'][k]
    for k in range(vcd_config.n_folds)
]
```

### Formato esperado do conformity_details_at_optimal.csv:
```
fold,variable,calculated_gain,calculated_signal,expected_signal,conformity
0,1,-0.0023,-1,-1,True
0,2,0.0015,1,0,
0,3,-0.0041,-1,-1,True
0,4,0.0082,1,1,True
0,5,0.0011,1,-1,False
1,1,...
```


## ISSUE 3: Adicionar R² de treino por fold

### Problema
O fold_details_at_optimal.csv tem mse_test, r2_test e scr,
mas NÃO tem r2_train. Precisamos para análise de overfitting.

### Correção
No loop VCD, já coletamos fold_r2_train em step_metrics.
Ao salvar fold_details_at_optimal, adicionar:

```python
fold_metrics_at_optimal=[{
    'fold': k,
    'mse_test': optimal_step['fold_mse_test'][k],
    'r2_test': optimal_step['fold_r2_test'][k],
    'r2_train': optimal_step['fold_r2_train'][k],
    'scr': optimal_step['fold_scr'][k],
} for k in range(vcd_config.n_folds)]
```

Isso já funciona porque fold_r2_train é coletado no loop. Basta
incluí-lo no dict.


## Ordem de execução

1. DEBUG Issue 1: rodar os 3 testes de diagnóstico (teste 1, 2, 3)
2. Se Hipótese A (bug): corrigir e re-testar
   Se Hipótese B (legítimo): documentar conclusão e seguir
3. Implementar Issue 2 (conformity_details no output)
4. Implementar Issue 3 (r2_train por fold no output)
5. Matar TODOS os batches em execução:
   kill $(pgrep -f run_dynamic_cv)
   (mesmo se Issue 1 não era bug, Issues 2 e 3 mudam o formato dos outputs)
6. Re-testar um caso rápido para confirmar novos outputs:
   python src/run_dynamic_cv.py --methods 1 --dataset real_estate --base-dir . 2>&1 | head -30
   Verificar que conformity_details_at_optimal.csv e r2_train aparecem nos outputs.
7. Commit:
   git add -A
   git commit -m "fix: debug ELM VCD behavior, add conformity_details and r2_train to outputs"
