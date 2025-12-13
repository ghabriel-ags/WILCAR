"""
WILCAR Unconstrained (Método 1) - Sem restrições nos sinais dos ganhos
======================================================================
Universidade Federal da Bahia - Mestrado em Engenharia Industrial
Autor: Ghabriel Anton Gomes de Sá

Características:
- Pesos iniciais do 1º neurônio via linearização/Taylor
- Demais neurônios via Xavier initialization
- Treinamento via backpropagation
- Abordagem construtivista (adiciona neurônios incrementalmente)
"""

import numpy as np
import time
import copy
from datetime import datetime
from typing import Dict, Tuple, Optional

# Importações do projeto
import sys
from pathlib import Path

# Adicionar src ao path se necessário
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from base import BaseNeuralNetwork, TrainingConfig, TrainingResults, TORCH_AVAILABLE
from utils.metrics import calculate_all_metrics

# PyTorch (opcional)
if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim


# =============================================================================
# IMPLEMENTAÇÃO NUMPY (CPU)
# =============================================================================

class WILCARUnconstrainedNumpy(BaseNeuralNetwork):
    """
    WILCAR sem restrições - Implementação NumPy (CPU).
    """
    
    METHOD_NAME = "WILCAR_Unconstrained"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Obter pesos iniciais via linearização
        print("\n🔧 Calculando pesos iniciais via linearização...")
        self.initial_weights = self._initialize_first_hidden_neuron()
    
    def _initialize_first_hidden_neuron(self) -> np.ndarray:
        """
        Obtém pesos iniciais via linearização e análise de Taylor.
        """
        from scipy.optimize import fsolve
        import statsmodels.api as sm
        
        # Regressão linear múltipla
        inputs_with_const = sm.add_constant(self.train_inputs, prepend=True)
        ols_result = sm.OLS(self.train_targets, inputs_with_const).fit()
        
        # Coeficientes da regressão
        k = ols_result.params[1:]  # Coeficientes das variáveis
        b = ols_result.params[0]   # Intercepto
        
        # Ponto de equilíbrio (média das entradas)
        x_M = np.mean(self.train_inputs, axis=0)
        
        # Sistema de equações
        def equations(x):
            w, beta = x[:-1], x[-1]
            z = np.dot(w, x_M) + beta
            sig = self.sigmoid(z)
            sig_prime = sig * (1 - sig)
            
            # Equação do intercepto
            f = [sig - sig_prime * np.dot(w, x_M) - b]
            
            # Equações dos coeficientes
            for i in range(self.n_inputs):
                f.append(sig_prime * w[i] - k[i])
            
            return f
        
        # Resolver com múltiplos chutes iniciais
        best_result = None
        best_residual = float('inf')
        
        for trial in range(5):
            if trial == 0:
                x0 = np.append(k, b)
            else:
                x0 = np.random.randn(self.n_inputs + 1) * 0.5
            
            result, info, ier, _ = fsolve(equations, x0, full_output=True)
            residual = np.linalg.norm(info['fvec'])
            
            if ier == 1 and residual < best_residual:
                best_result = result
                best_residual = residual
        
        if best_residual > 1e-6:
            print(f"⚠️ Convergência pode não ser ótima (resíduo = {best_residual:.2e})")
        else:
            print(f"✅ Pesos iniciais obtidos (resíduo = {best_residual:.2e})")
        
        return best_result
    
    def _initialize_parameters(self, n_neurons: int) -> Dict[str, np.ndarray]:
        """Inicializa parâmetros da rede com n neurônios ocultos."""
        param = {
            'W1': np.random.randn(n_neurons, self.n_inputs) * np.sqrt(1. / self.n_inputs),
            'b1': np.zeros((n_neurons, 1)),
            'W2': np.random.randn(1, n_neurons) * np.sqrt(1. / n_neurons),
            'b2': np.zeros((1, 1))
        }
        
        # Aplicar pesos iniciais ao primeiro neurônio
        if self.initial_weights is not None:
            param['W1'][0, :] = self.initial_weights[:-1]
            param['b1'][0, 0] = self.initial_weights[-1]
        
        return param
    
    def _expand_network(self, n_neurons: int):
        """Expande a rede adicionando um neurônio oculto."""
        if n_neurons <= self.n_neurons:
            return
        
        # Adicionar neurônio à camada oculta
        new_w1 = np.random.randn(1, self.n_inputs) * np.sqrt(1. / self.n_inputs)
        self.param['W1'] = np.vstack([self.param['W1'], new_w1])
        self.param['b1'] = np.vstack([self.param['b1'], np.zeros((1, 1))])
        
        # Expandir conexões da camada de saída
        new_w2 = np.random.randn(1, 1) * np.sqrt(1. / n_neurons)
        self.param['W2'] = np.hstack([self.param['W2'], new_w2])
        
        self.n_neurons = n_neurons
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass."""
        Z1 = self.param['W1'].dot(x) + self.param['b1']
        A1 = self.sigmoid(Z1)
        Z2 = self.param['W2'].dot(A1) + self.param['b2']
        A2 = self.sigmoid(Z2)
        return A2, A1, Z1
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Faz predição para novos dados."""
        output, _, _ = self.forward(x)
        return output
    
    def _train_single_model(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """Treina modelo atual via backpropagation."""
        lr = self.config.learning_rate
        m = x.shape[1]
        
        best_loss = float('inf')
        no_improve = 0
        
        for epoch in range(self.config.iterations):
            # Forward
            A2, A1, Z1 = self.forward(x)
            
            # Loss
            current_loss = np.mean((A2 - y) ** 2)
            
            # Early stopping
            if current_loss < best_loss - self.config.tolerance:
                best_loss = current_loss
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= self.config.patience_early_stopping:
                break
            
            # Backward
            dZ2 = A2 - y
            dW2 = dZ2.dot(A1.T) / m
            db2 = np.sum(dZ2, axis=1, keepdims=True) / m
            
            dA1 = self.param['W2'].T.dot(dZ2)
            dZ1 = dA1 * A1 * (1 - A1)
            dW1 = dZ1.dot(x.T) / m
            db1 = np.sum(dZ1, axis=1, keepdims=True) / m
            
            # Update
            self.param['W1'] -= lr * dW1
            self.param['b1'] -= lr * db1
            self.param['W2'] -= lr * dW2
            self.param['b2'] -= lr * db2
            
            # Learning rate decay
            lr = max(lr * self.config.lr_decay, self.config.min_lr)
        
        # Métricas finais
        y_pred = self.predict(x)
        return calculate_all_metrics(y.flatten(), y_pred.flatten())
    
    def train(self) -> TrainingResults:
        """Executa treinamento construtivista completo."""
        results = TrainingResults()
        results.method_name = self.METHOD_NAME
        results.dataset_name = self.dataset_name
        results.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Preparar dados
        x_train = self.train_inputs.T
        y_train = self.train_targets.reshape(1, -1)
        x_test = self.test_inputs.T
        y_test = self.test_targets.reshape(1, -1)
        
        # Early stopping construtivista
        best_r2_so_far = -float('inf')
        no_improve_count = 0
        best_model_params = None
        
        start_time = time.time()
        
        print("\n" + "=" * 70)
        print(f"TREINAMENTO - {self.METHOD_NAME}")
        print("=" * 70)
        
        for n in range(1, self.config.max_neurons + 1):
            iter_start = time.time()
            
            # Inicializar ou expandir rede
            if n == 1:
                self.param = self._initialize_parameters(n)
                self.n_neurons = n
            else:
                self._expand_network(n)
            
            # Treinar
            mse_train, rmse_train, r2_train = self._train_single_model(x_train, y_train)
            
            # Métricas de teste
            y_pred_test = self.predict(x_test)
            mse_test, rmse_test, r2_test = calculate_all_metrics(y_test.flatten(), y_pred_test.flatten())
            
            # Conformidade
            tcs, conf_details = self.calculate_conformity(self.predict)
            
            # Armazenar
            results.mse_train.append(mse_train)
            results.rmse_train.append(rmse_train)
            results.r2_train.append(r2_train)
            results.mse_test.append(mse_test)
            results.rmse_test.append(rmse_test)
            results.r2_test.append(r2_test)
            results.conformity_rates.append(tcs)
            
            iter_time = time.time() - iter_start
            results.iteration_times.append(iter_time)
            
            # Atualizar melhor modelo
            if r2_test > results.best_test_r2:
                results.best_test_r2 = r2_test
                results.best_test_mse = mse_test
                results.best_test_rmse = rmse_test
                results.best_train_r2 = r2_train
                results.best_train_mse = mse_train
                results.best_train_rmse = rmse_train
                results.best_neurons = n
                results.best_conformity = tcs
                results.conformity_details = conf_details
                best_model_params = {k: v.copy() for k, v in self.param.items()}
            
            # Log
            if self.config.verbose >= 1:
                print(f"n={n:3d} | Train R²={r2_train:.4f} | Test R²={r2_test:.4f} | "
                      f"TCS={tcs:5.1%} | Time={iter_time:5.1f}s")
            
            # Early stopping construtivista
            if r2_test > best_r2_so_far:
                best_r2_so_far = r2_test
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= self.config.patience_constructive:
                print(f"\n⏹️ Early stopping: {self.config.patience_constructive} iterações sem melhora")
                break
        
        results.total_time = time.time() - start_time
        results.avg_time_per_model = np.mean(results.iteration_times)
        
        # Restaurar melhor modelo
        if best_model_params is not None:
            self.param = best_model_params
            self.n_neurons = results.best_neurons
        
        # Resumo
        print("\n" + "=" * 70)
        print(f"✅ Treinamento concluído em {results.total_time:.1f}s")
        print(f"🏆 Melhor: n={results.best_neurons} | R²={results.best_test_r2:.4f} | TCS={results.best_conformity:.1%}")
        print("=" * 70)
        
        return results


# =============================================================================
# IMPLEMENTAÇÃO PYTORCH (GPU)
# =============================================================================

if TORCH_AVAILABLE:
    
    class SFNNTorch(nn.Module):
        """Single Hidden Layer Feedforward Neural Network em PyTorch."""
        
        def __init__(self, n_inputs: int, n_hidden: int):
            super().__init__()
            self.hidden = nn.Linear(n_inputs, n_hidden)
            self.output = nn.Linear(n_hidden, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.sigmoid(self.hidden(x))
            x = self.sigmoid(self.output(x))
            return x
    
    
    class WILCARUnconstrainedTorch(BaseNeuralNetwork):
        """
        WILCAR sem restrições - Implementação PyTorch com suporte a GPU.
        """
        
        METHOD_NAME = "WILCAR_Unconstrained_GPU"
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Obter pesos iniciais via linearização (mesmo método)
            print("\n🔧 Calculando pesos iniciais via linearização...")
            self.initial_weights = self._initialize_first_hidden_neuron()
            
            self.model = None
            self.optimizer = None
        
        def _initialize_first_hidden_neuron(self) -> np.ndarray:
            """Mesmo método de linearização da versão NumPy."""
            from scipy.optimize import fsolve
            import statsmodels.api as sm
            
            inputs_with_const = sm.add_constant(self.train_inputs, prepend=True)
            ols_result = sm.OLS(self.train_targets, inputs_with_const).fit()
            
            k = ols_result.params[1:]
            b = ols_result.params[0]
            x_M = np.mean(self.train_inputs, axis=0)
            
            def equations(x):
                w, beta = x[:-1], x[-1]
                z = np.dot(w, x_M) + beta
                sig = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
                sig_prime = sig * (1 - sig)
                
                f = [sig - sig_prime * np.dot(w, x_M) - b]
                for i in range(self.n_inputs):
                    f.append(sig_prime * w[i] - k[i])
                return f
            
            best_result = None
            best_residual = float('inf')
            
            for trial in range(5):
                x0 = np.append(k, b) if trial == 0 else np.random.randn(self.n_inputs + 1) * 0.5
                result, info, ier, _ = fsolve(equations, x0, full_output=True)
                residual = np.linalg.norm(info['fvec'])
                
                if ier == 1 and residual < best_residual:
                    best_result = result
                    best_residual = residual
            
            if best_residual > 1e-6:
                print(f"⚠️ Convergência pode não ser ótima (resíduo = {best_residual:.2e})")
            else:
                print(f"✅ Pesos iniciais obtidos (resíduo = {best_residual:.2e})")
            
            return best_result
        
        def _create_model(self, n_neurons: int) -> nn.Module:
            """Cria modelo PyTorch."""
            model = SFNNTorch(self.n_inputs, n_neurons)
            
            # Aplicar pesos iniciais ao primeiro neurônio
            if self.initial_weights is not None:
                with torch.no_grad():
                    model.hidden.weight.data[0, :] = torch.tensor(
                        self.initial_weights[:-1], dtype=torch.float32)
                    model.hidden.bias.data[0] = torch.tensor(
                        self.initial_weights[-1], dtype=torch.float32)
            
            return model.to(self.device)
        
        def predict(self, x: np.ndarray) -> np.ndarray:
            """Predição (aceita numpy, retorna numpy)."""
            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(x.T, dtype=torch.float32).to(self.device)
                y_pred = self.model(x_tensor)
                return y_pred.cpu().numpy().T
        
        def train(self) -> TrainingResults:
            """Treinamento completo com PyTorch."""
            results = TrainingResults()
            results.method_name = self.METHOD_NAME
            results.dataset_name = self.dataset_name
            results.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Converter para tensores
            x_train = torch.tensor(self.train_inputs, dtype=torch.float32).to(self.device)
            y_train = torch.tensor(self.train_targets.reshape(-1, 1), dtype=torch.float32).to(self.device)
            x_test = torch.tensor(self.test_inputs, dtype=torch.float32).to(self.device)
            y_test_tensor = torch.tensor(self.test_targets.reshape(-1, 1), dtype=torch.float32).to(self.device)
            
            best_r2_so_far = -float('inf')
            no_improve_count = 0
            best_model_state = None
            
            start_time = time.time()
            
            device_str = "GPU" if self.device == 'cuda' else "CPU"
            print("\n" + "=" * 70)
            print(f"TREINAMENTO - {self.METHOD_NAME} ({device_str})")
            print("=" * 70)
            
            for n in range(1, self.config.max_neurons + 1):
                iter_start = time.time()
                
                # Criar novo modelo com n neurônios
                self.model = self._create_model(n)
                self.n_neurons = n
                
                # Otimizador
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
                scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.lr_decay)
                
                criterion = nn.MSELoss()
                
                # Treinar
                self.model.train()
                best_loss = float('inf')
                no_improve_epoch = 0
                
                for epoch in range(self.config.iterations):
                    self.optimizer.zero_grad()
                    y_pred = self.model(x_train)
                    loss = criterion(y_pred, y_train)
                    loss.backward()
                    self.optimizer.step()
                    scheduler.step()
                    
                    current_loss = loss.item()
                    if current_loss < best_loss - self.config.tolerance:
                        best_loss = current_loss
                        no_improve_epoch = 0
                    else:
                        no_improve_epoch += 1
                    
                    if no_improve_epoch >= self.config.patience_early_stopping:
                        break
                
                # Métricas
                self.model.eval()
                with torch.no_grad():
                    y_pred_train = self.model(x_train).cpu().numpy()
                    y_pred_test = self.model(x_test).cpu().numpy()
                
                mse_train, rmse_train, r2_train = calculate_all_metrics(
                    self.train_targets, y_pred_train.flatten())
                mse_test, rmse_test, r2_test = calculate_all_metrics(
                    self.test_targets, y_pred_test.flatten())
                
                tcs, conf_details = self.calculate_conformity(self.predict)
                
                # Armazenar
                results.mse_train.append(mse_train)
                results.rmse_train.append(rmse_train)
                results.r2_train.append(r2_train)
                results.mse_test.append(mse_test)
                results.rmse_test.append(rmse_test)
                results.r2_test.append(r2_test)
                results.conformity_rates.append(tcs)
                
                iter_time = time.time() - iter_start
                results.iteration_times.append(iter_time)
                
                # Melhor modelo
                if r2_test > results.best_test_r2:
                    results.best_test_r2 = r2_test
                    results.best_test_mse = mse_test
                    results.best_test_rmse = rmse_test
                    results.best_train_r2 = r2_train
                    results.best_train_mse = mse_train
                    results.best_train_rmse = rmse_train
                    results.best_neurons = n
                    results.best_conformity = tcs
                    results.conformity_details = conf_details
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                
                if self.config.verbose >= 1:
                    print(f"n={n:3d} | Train R²={r2_train:.4f} | Test R²={r2_test:.4f} | "
                          f"TCS={tcs:5.1%} | Time={iter_time:5.1f}s")
                
                # Early stopping construtivista
                if r2_test > best_r2_so_far:
                    best_r2_so_far = r2_test
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if no_improve_count >= self.config.patience_constructive:
                    print(f"\n⏹️ Early stopping: {self.config.patience_constructive} iterações sem melhora")
                    break
            
            results.total_time = time.time() - start_time
            results.avg_time_per_model = np.mean(results.iteration_times)
            
            # Restaurar melhor modelo
            if best_model_state is not None:
                self.model = self._create_model(results.best_neurons)
                self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            
            print("\n" + "=" * 70)
            print(f"✅ Treinamento concluído em {results.total_time:.1f}s")
            print(f"🏆 Melhor: n={results.best_neurons} | R²={results.best_test_r2:.4f} | TCS={results.best_conformity:.1%}")
            print("=" * 70)
            
            return results


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_wilcar_unconstrained(train_inputs, train_targets, test_inputs, test_targets,
                                expected_signals, config: TrainingConfig, 
                                dataset_name: str = "unknown"):
    """
    Factory que cria a melhor implementação disponível.
    
    Args:
        train_inputs, train_targets: Dados de treino
        test_inputs, test_targets: Dados de teste
        expected_signals: Sinais esperados dos ganhos
        config: TrainingConfig
        dataset_name: Nome do dataset
    
    Returns:
        Instância de WILCARUnconstrained (NumPy ou PyTorch)
    """
    if config.use_gpu and TORCH_AVAILABLE:
        import torch
        if torch.cuda.is_available():
            print("🚀 Usando implementação PyTorch com GPU")
            return WILCARUnconstrainedTorch(
                train_inputs, train_targets, test_inputs, test_targets,
                expected_signals, config, dataset_name)
        else:
            print("⚠️ GPU não disponível. Usando PyTorch com CPU.")
            return WILCARUnconstrainedTorch(
                train_inputs, train_targets, test_inputs, test_targets,
                expected_signals, config, dataset_name)
    else:
        print("💻 Usando implementação NumPy (CPU)")
        return WILCARUnconstrainedNumpy(
            train_inputs, train_targets, test_inputs, test_targets,
            expected_signals, config, dataset_name)


# Alias para compatibilidade
WILCARUnconstrained = WILCARUnconstrainedNumpy