# Extensão da VCD para Métodos 3-6 (RIXM, RIXM+R, ELM, ELM+R)

## Contexto

A VCD já funciona para M1 (WILCAR) e M2 (WILCAR+R). Precisamos estender o
suporte para os 4 métodos restantes. Isso envolve:

1. Adicionar `constructive_step()`, `get_params()`, `set_params()` a cada classe
2. Atualizar `run_dynamic_cv.py` para suportar métodos 3-6
3. Atualizar o worker paralelo para lidar com as diferenças

## Diferenças críticas entre métodos

### M3 (RIXMNumpy) em src/rixm.py
- **SEM weight reuse**: cada n reinicializa do zero (`self.param = self._initialize_parameters(n)`)
- Backprop idêntico ao M1
- Estrutura de params: `{'W1', 'b1', 'W2', 'b2'}` (igual M1)
- `constructive_step` é simples: inicializa, treina, retorna métricas

### M4 (RIXMConstrainedNumpy) em src/rixm_constrained.py
- **SEM weight reuse**: cada n reinicializa do zero
- SLSQP + reinicialização até 100% SCR
- Estrutura de params: `{'W1', 'b1', 'W2', 'b2'}` (igual M2)
- `_train_single_model` NÃO recebe `previous_param` (diferente do M2!)
  - Assinatura: `_train_single_model(x_train, y_train, x_test, n_neurons)`
  - Sem o argumento `previous_param` que M2 tem

### M5 (ELMNumpy) em src/elm.py
- **SEM weight reuse**: cada n reinicializa do zero
- Treino via pseudo-inverse (single-pass, muito rápido)
- **Estrutura de params DIFERENTE**: `{'input_weights', 'biases', 'output_weights'}`
- **Formato de dados DIFERENTE internamente**: usa `(n_samples, n_features)`, NÃO transposto
  - `_train_single_model(X_train, y_train)` recebe `(n_samples, n_features)` e `(n_samples,)`
  - `_hidden_nodes(X)` recebe `(n_samples, n_features)`
  - Mas `predict(x)` e `forward(x)` recebem transposto `(n_features, n_samples)` (compatível com BaseNeuralNetwork)
- `calculate_conformity()` do base usa `predict()` com formato transposto → funciona

### M6 (ELMConstrainedNumpy) em src/elm_constrained.py
- **SEM weight reuse**: cada n reinicializa do zero
- SLSQP para output weights + regenera hidden weights até 100% SCR
- Estrutura de params: `{'input_weights', 'biases', 'output_weights'}` (igual M5)
- `_train_single_model(X_train, y_train, X_test, n_neurons)` — formato `(n_samples, n_features)`
- `_check_conformity_on_data(X)` recebe formato `(n_samples, n_features)` (DIFERENTE de M2/M4!)

## Implementação: constructive_step para cada método

### M3 (RIXMNumpy) — adicionar em src/rixm.py após train()

```python
    # =========================================================================
    # VCD SUPPORT
    # =========================================================================

    def constructive_step(self, n: int, x_train: np.ndarray, y_train: np.ndarray,
                          x_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Execute ONE constructive step (no weight reuse - from scratch).
        
        Args:
            n: Number of neurons
            x_train: (n_features, n_samples) - transposed
            y_train: (1, n_samples)
            x_test: (n_features, n_samples) - transposed
            y_test: (1, n_samples)
        """
        iter_start = time.time()
        
        # Always from scratch (RIXM has NO weight reuse)
        self.param = self._initialize_parameters(n)
        self.n_neurons = n
        
        # Train via backpropagation
        mse_train, rmse_train, r2_train = self._train_single_model(x_train, y_train)
        
        # Test metrics
        y_pred_test = self.predict(x_test)
        mse_test, rmse_test, r2_test = calculate_all_metrics(
            y_test.flatten(), y_pred_test.flatten()
        )
        
        # Conformity
        scr, conf_details = self.calculate_conformity(self.predict)
        
        return {
            'n_neurons': n,
            'mse_train': mse_train, 'rmse_train': rmse_train, 'r2_train': r2_train,
            'mse_test': mse_test, 'rmse_test': rmse_test, 'r2_test': r2_test,
            'scr': scr, 'conformity_details': conf_details,
            'training_time': time.time() - iter_start
        }

    def get_params(self) -> Optional[Dict]:
        if self.param is None:
            return None
        return {k: v.copy() for k, v in self.param.items()}

    def set_params(self, param: Dict):
        self.param = {k: v.copy() for k, v in param.items()}
        self.n_neurons = param['W1'].shape[0]
```

### M4 (RIXMConstrainedNumpy) — adicionar em src/rixm_constrained.py após train()

```python
    # =========================================================================
    # VCD SUPPORT
    # =========================================================================

    def constructive_step(self, n: int, x_train: np.ndarray, y_train: np.ndarray,
                          x_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Execute ONE constructive step with constraints (no weight reuse).
        
        Args:
            n: Number of neurons
            x_train: (n_features, n_samples) - transposed
            y_train: (1, n_samples)
            x_test: (n_features, n_samples) - transposed
            y_test: (1, n_samples)
        """
        iter_start = time.time()
        
        # Always from scratch (RIXM has NO weight reuse)
        self.param = self._initialize_parameters(n)
        self.n_neurons = n
        
        # Train with reinitialization until 100% SCR
        # NOTE: M4's _train_single_model does NOT take previous_param
        success, n_iter, n_attempts = self._train_single_model(
            x_train, y_train, x_test, n
        )
        
        training_time = time.time() - iter_start
        
        if not success:
            scr_achieved = self._check_conformity_on_data(x_test)
            return {
                'n_neurons': n,
                'success': False,
                'scr': scr_achieved,
                'n_attempts': n_attempts,
                'n_iterations': n_iter,
                'training_time': training_time
            }
        
        # Training metrics
        y_pred_train = self.predict(x_train)
        mse_train, rmse_train, r2_train = calculate_all_metrics(
            y_train.flatten(), y_pred_train.flatten()
        )
        
        # Test metrics
        y_pred_test = self.predict(x_test)
        mse_test, rmse_test, r2_test = calculate_all_metrics(
            y_test.flatten(), y_pred_test.flatten()
        )
        
        # Conformity
        scr, conf_details = self.calculate_conformity(self.predict)
        
        return {
            'n_neurons': n,
            'success': True,
            'mse_train': mse_train, 'rmse_train': rmse_train, 'r2_train': r2_train,
            'mse_test': mse_test, 'rmse_test': rmse_test, 'r2_test': r2_test,
            'scr': scr, 'conformity_details': conf_details,
            'n_attempts': n_attempts, 'n_iterations': n_iter,
            'training_time': training_time
        }

    def get_params(self) -> Optional[Dict]:
        if self.param is None:
            return None
        return copy.deepcopy(self.param)

    def set_params(self, param: Dict):
        self.param = copy.deepcopy(param)
        self.n_neurons = param['W1'].shape[0]
```

### M5 (ELMNumpy) — adicionar em src/elm.py após train()

ATENÇÃO: ELM usa formato de dados diferente internamente.
constructive_step recebe dados transpostos (padrão da interface VCD),
mas _train_single_model espera (n_samples, n_features).

```python
    # =========================================================================
    # VCD SUPPORT
    # =========================================================================

    def constructive_step(self, n: int, x_train: np.ndarray, y_train: np.ndarray,
                          x_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Execute ONE constructive step for ELM (no weight reuse).
        
        IMPORTANT: This method receives data in TRANSPOSED format
        (n_features, n_samples) for interface consistency, but internally
        converts to ELM's native format (n_samples, n_features).
        
        Args:
            n: Number of neurons
            x_train: (n_features, n_samples) - transposed
            y_train: (1, n_samples)
            x_test: (n_features, n_samples) - transposed
            y_test: (1, n_samples)
        """
        iter_start = time.time()
        
        # Convert to ELM's native format
        X_train = x_train.T   # (n_samples, n_features)
        Y_train = y_train.flatten()  # (n_samples,)
        X_test = x_test.T
        Y_test = y_test.flatten()
        
        # Initialize from scratch
        self.param = self._initialize_parameters(n)
        self.n_neurons = n
        
        # Train via pseudo-inverse (single-pass)
        mse_train, rmse_train, r2_train = self._train_single_model(X_train, Y_train)
        
        # Test metrics
        H_test = self._hidden_nodes(X_test)
        y_pred_test = np.dot(H_test, self.param['output_weights'])
        mse_test, rmse_test, r2_test = calculate_all_metrics(Y_test, y_pred_test)
        
        # Conformity (uses predict() which handles transposed format)
        scr, conf_details = self.calculate_conformity(self.predict)
        
        return {
            'n_neurons': n,
            'mse_train': mse_train, 'rmse_train': rmse_train, 'r2_train': r2_train,
            'mse_test': mse_test, 'rmse_test': rmse_test, 'r2_test': r2_test,
            'scr': scr, 'conformity_details': conf_details,
            'training_time': time.time() - iter_start
        }

    def get_params(self) -> Optional[Dict]:
        if self.param is None:
            return None
        return {
            'input_weights': self.param['input_weights'].copy(),
            'biases': self.param['biases'].copy(),
            'output_weights': self.param['output_weights'].copy() 
                if self.param['output_weights'] is not None else None
        }

    def set_params(self, param: Dict):
        self.param = {
            'input_weights': param['input_weights'].copy(),
            'biases': param['biases'].copy(),
            'output_weights': param['output_weights'].copy()
                if param['output_weights'] is not None else None
        }
        self.n_neurons = param['input_weights'].shape[1]
```

### M6 (ELMConstrainedNumpy) — adicionar em src/elm_constrained.py após train()

ATENÇÃO: Similar ao M5 com conversão de formato + lógica de reinicialização.
`_train_single_model` e `_check_conformity_on_data` esperam (n_samples, n_features).

```python
    # =========================================================================
    # VCD SUPPORT
    # =========================================================================

    def constructive_step(self, n: int, x_train: np.ndarray, y_train: np.ndarray,
                          x_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Execute ONE constructive step for ELM Constrained (no weight reuse).
        
        IMPORTANT: Receives data in TRANSPOSED format (n_features, n_samples)
        for interface consistency, but converts internally.
        
        Args:
            n: Number of neurons
            x_train: (n_features, n_samples) - transposed
            y_train: (1, n_samples)
            x_test: (n_features, n_samples) - transposed
            y_test: (1, n_samples)
        """
        iter_start = time.time()
        
        # Convert to ELM's native format
        X_train = x_train.T   # (n_samples, n_features)
        Y_train = y_train.flatten()  # (n_samples,)
        X_test = x_test.T
        
        # Initialize from scratch
        self.param = self._initialize_parameters(n)
        self.n_neurons = n
        
        # Train with reinitialization until 100% SCR
        success, n_iter, n_attempts = self._train_single_model(
            X_train, Y_train, X_test, n
        )
        
        training_time = time.time() - iter_start
        
        if not success:
            scr_achieved = self._check_conformity_on_data(X_test)
            return {
                'n_neurons': n,
                'success': False,
                'scr': scr_achieved,
                'n_attempts': n_attempts,
                'n_iterations': n_iter,
                'training_time': training_time
            }
        
        # Training metrics
        H_train = self._hidden_nodes(X_train)
        y_pred_train = np.dot(H_train, self.param['output_weights'])
        mse_train, rmse_train, r2_train = calculate_all_metrics(Y_train, y_pred_train)
        
        # Test metrics
        Y_test_flat = y_test.flatten()
        H_test = self._hidden_nodes(X_test)
        y_pred_test = np.dot(H_test, self.param['output_weights'])
        mse_test, rmse_test, r2_test = calculate_all_metrics(Y_test_flat, y_pred_test)
        
        # Conformity
        scr, conf_details = self.calculate_conformity(self.predict)
        
        return {
            'n_neurons': n,
            'success': True,
            'mse_train': mse_train, 'rmse_train': rmse_train, 'r2_train': r2_train,
            'mse_test': mse_test, 'rmse_test': rmse_test, 'r2_test': r2_test,
            'scr': scr, 'conformity_details': conf_details,
            'n_attempts': n_attempts, 'n_iterations': n_iter,
            'training_time': training_time
        }

    def get_params(self) -> Optional[Dict]:
        if self.param is None:
            return None
        return {
            'input_weights': self.param['input_weights'].copy(),
            'biases': self.param['biases'].copy(),
            'output_weights': self.param['output_weights'].copy()
                if self.param['output_weights'] is not None else None
        }

    def set_params(self, param: Dict):
        self.param = {
            'input_weights': param['input_weights'].copy(),
            'biases': param['biases'].copy(),
            'output_weights': param['output_weights'].copy()
                if param['output_weights'] is not None else None
        }
        self.n_neurons = param['input_weights'].shape[1]
```

## Modificações em run_dynamic_cv.py

### 1. Atualizar create_model() para suportar métodos 3-6

```python
def create_model(method_id: int, **kwargs):
    """Create a model instance for the given method."""
    import io, contextlib
    
    try:
        from unconstrained import WILCARUnconstrainedNumpy
        from constrained import WILCARConstrainedNumpy
        from rixm import RIXMNumpy
        from rixm_constrained import RIXMConstrainedNumpy
        from elm import ELMNumpy
        from elm_constrained import ELMConstrainedNumpy
    except ImportError:
        from wilcar.unconstrained import WILCARUnconstrainedNumpy
        from wilcar.constrained import WILCARConstrainedNumpy
        from rixm import RIXMNumpy
        from rixm_constrained import RIXMConstrainedNumpy
        from elm import ELMNumpy
        from elm_constrained import ELMConstrainedNumpy
    
    classes = {
        1: WILCARUnconstrainedNumpy,
        2: WILCARConstrainedNumpy,
        3: RIXMNumpy,
        4: RIXMConstrainedNumpy,
        5: ELMNumpy,
        6: ELMConstrainedNumpy,
    }
    
    if method_id not in classes:
        raise ValueError(f"Method {method_id} not supported")
    
    with contextlib.redirect_stdout(io.StringIO()):
        model = classes[method_id](**kwargs)
    
    return model
```

### 2. Atualizar VCD_METHOD_CONFIGS para incluir M3-M6

```python
VCD_METHOD_CONFIGS = {
    'computer_hardware': {
        1: VCDMethodConfig(max_neurons=750, patience=150),
        2: VCDMethodConfig(max_neurons=200, patience=20),
        3: VCDMethodConfig(max_neurons=750, patience=150),
        4: VCDMethodConfig(max_neurons=200, patience=20),
        5: VCDMethodConfig(max_neurons=750, patience=150),
        6: VCDMethodConfig(max_neurons=150, patience=15),
    },
    'fish_toxicity': {
        1: VCDMethodConfig(max_neurons=750, patience=150),
        2: VCDMethodConfig(max_neurons=200, patience=20),
        3: VCDMethodConfig(max_neurons=750, patience=150),
        4: VCDMethodConfig(max_neurons=200, patience=20),
        5: VCDMethodConfig(max_neurons=750, patience=150),
        6: VCDMethodConfig(max_neurons=150, patience=15),
    },
    'aquatic_toxicity': {
        1: VCDMethodConfig(max_neurons=750, patience=150),
        2: VCDMethodConfig(max_neurons=200, patience=20),
        3: VCDMethodConfig(max_neurons=750, patience=150),
        4: VCDMethodConfig(max_neurons=200, patience=20),
        5: VCDMethodConfig(max_neurons=750, patience=150),
        6: VCDMethodConfig(max_neurons=150, patience=15),
    }
}
```

### 3. Atualizar ESTIMATED_TIME_PER_STEP

```python
ESTIMATED_TIME_PER_STEP = {
    'computer_hardware': {
        1: 0.5, 2: 3.0, 3: 0.5, 4: 3.0, 5: 0.01, 6: 0.5
    },
    'fish_toxicity': {
        1: 1.0, 2: 8.0, 3: 1.0, 4: 8.0, 5: 0.02, 6: 2.0
    },
    'aquatic_toxicity': {
        1: 0.8, 2: 6.0, 3: 0.8, 4: 6.0, 5: 0.01, 6: 1.5
    },
}
```

### 4. Remover a validação que restringe a métodos 1 e 2

No `main()`, remover/atualizar o bloco:
```python
# ANTES (remover):
for m in args.methods:
    if m not in [1, 2]:
        print(f"⚠️  Method {m} not yet supported for VCD...")

# DEPOIS:
for m in args.methods:
    if m not in [1, 2, 3, 4, 5, 6]:
        print(f"⚠️  Method {m} not valid. Available: 1-6")
```

E atualizar o default de methods no argparse:
```python
parser.add_argument('--methods', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                    help="Methods to run (1-6)")
```

### 5. Atualizar _vcd_fold_worker para M3-M6

O worker paralelo precisa de atenção especial:

- Para M3/M4: `set_params` funciona igual M1/M2 (mesma estrutura de params)
  mas como NÃO há weight reuse, o estado anterior é irrelevante — cada n
  reinicializa do zero no constructive_step. Ainda assim, set_params precisa
  funcionar porque o worker recria o modelo a cada chamada.
  
- Para M3/M4: Não há _vcd_previous_param para restaurar (ao contrário de M2)

- Para M5/M6: get_params/set_params lidam com estrutura diferente
  ('input_weights'/'biases'/'output_weights'). O worker não precisa saber
  a estrutura — ele só chama get_params/set_params genericamente.

- PONTO CRÍTICO: Para M3/M4/M5/M6, como NÃO há weight reuse, o fold_state
  retornado pelo worker é usado apenas para consistência da interface.
  O constructive_step sempre reinicializa. Isso significa que na prática,
  para esses métodos, poderíamos até ignorar fold_state e não chamar
  set_params. Mas manter a interface uniforme é mais limpo.

No worker, o bloco que restaura estado do M2:
```python
    # Restore state from previous step
    if fold_state is not None:
        model.set_params(fold_state)
        model.n_neurons = fold_state['W1'].shape[0]  # ← PROBLEMA para ELM!
```

Precisa ser atualizado para:
```python
    # Restore state from previous step
    if fold_state is not None:
        model.set_params(fold_state)
        # n_neurons is set inside set_params()
```

E verificar que set_params em cada classe já seta n_neurons corretamente
(nos patches acima, todos fazem isso).

Também, o bloco que restaura _vcd_previous_param para M2:
```python
    if method_id == 2 and fold_state is not None:
        model._vcd_previous_param = copy.deepcopy(fold_state)
```

Deve se manter APENAS para M2 (WILCAR Constrained). Não aplicar a M4 ou M6.

### 6. Atualizar METHOD_NAMES (se ainda não incluir todos)

Verificar que METHOD_NAMES tem todos os 6 métodos.

## Ordem de implementação

1. Adicionar constructive_step/get_params/set_params a cada arquivo:
   - src/rixm.py (M3)
   - src/rixm_constrained.py (M4)
   - src/elm.py (M5)
   - src/elm_constrained.py (M6)

2. Atualizar src/run_dynamic_cv.py:
   - create_model()
   - VCD_METHOD_CONFIGS
   - ESTIMATED_TIME_PER_STEP
   - argparse validation
   - _vcd_fold_worker (tratar fold_state genérico)

3. Testar cada método individualmente com computer_hardware:
   ```bash
   python src/run_dynamic_cv.py --methods 3 --dataset computer_hardware --base-dir . 2>&1 | head -30
   python src/run_dynamic_cv.py --methods 4 --dataset computer_hardware --base-dir . 2>&1 | head -30
   python src/run_dynamic_cv.py --methods 5 --dataset computer_hardware --base-dir . 2>&1 | head -30
   python src/run_dynamic_cv.py --methods 6 --dataset computer_hardware --base-dir . 2>&1 | head -30
   ```
   Para cada um, verificar que os primeiros ~10 neurônios rodam sem erro
   e que J_T está sendo calculado. Cancelar após confirmação (Ctrl+C).

4. Teste rápido integrado:
   ```bash
   python src/run_dynamic_cv.py --methods 5 --dataset computer_hardware --base-dir .
   ```
   M5 (ELM) é o mais rápido (~0.01s/step), então roda em segundos.

## Checklist de verificação

- [ ] M3 constructive_step: inicializa do zero, backprop, retorna métricas
- [ ] M4 constructive_step: inicializa do zero, SLSQP + reinit, retorna com 'success'
- [ ] M5 constructive_step: converte formato, pseudo-inverse, retorna métricas
- [ ] M6 constructive_step: converte formato, SLSQP + reinit, retorna com 'success'
- [ ] get_params/set_params: M3/M4 usam W1/b1/W2/b2, M5/M6 usam input_weights/biases/output_weights
- [ ] set_params seta n_neurons corretamente em todas as classes
- [ ] create_model() importa e instancia todos os 6 métodos
- [ ] _vcd_fold_worker funciona genericamente (não assume estrutura de params)
- [ ] _vcd_previous_param só é restaurado para M2 (não M4/M6)
- [ ] VCD_METHOD_CONFIGS tem configurações para M3-M6
- [ ] Argparse aceita métodos 1-6
