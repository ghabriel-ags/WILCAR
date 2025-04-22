# Universidade Federal da Bahia
## Escola Politécnica
### Programa de Pós-Graduação em Engenharia Industrial
#### Mestrado em Engenharia Industrial

**Discente:** Ghabriel Anton Gomes de Sá  
**Orientadores:** Marcelo Embiruçu & Cristiano Fontes  

## Uma Abordagem Combinando uma Estimativa de Pesos Iniciais Baseada em Linearização e Algoritmo Construtivista
Este projeto implementa um método sistemático que visa a identificação de modelos de redes neurais de propagação direta com uma única camada oculta com múltiplas entradas e única saída em problemas de regressão (ou classificação binária) com possibilidade de restrições nos sinais dos ganhos.

### Método Proposto

#### Inicialização dos Pesos
O método para a obtenção dos pesos iniciais propõe que:
1. Realize-se a **linearização**, via série de Taylor, do modelo neural não-linear - ou seja, uma rede neural que se resume a apenas um neurônio (de saída) -, onde o ponto de equilíbrio para a linearização é a média dos valores das variáveis de entrada e saída do conjunto de dados que se pretende modelar.
2. Aplique-se a **Regressão Linear Múltipla**, via Método dos Mínimos Quadrados, ao conjunto de dados que se pretende modelar;
3. Compare-se matematicamente os resultados de (1) e (2) a fim de se obter uma estimativa para os pesos iniciais da rede neural inicial - uma rede com 1 neurônio oculto e 1 neurônio de saída -, nesse caso, os pesos que conectam os neurônios de entrada ao neurônio oculto único.

#### Expansão da Rede Neural

A rede neural se expande seguindo uma abordagem **construtivista**, adicionando neurônios ocultos progressivamente, até um dado limite, e utilizando os pesos obtidos em treinamnetos anteriores como ponto de partida para novos treinamentos. O método de Xavier é utilizado para estimar os pesos dos novos neurônios ocultos adicionados, enquanto os **biases** são inicializados como zero.

## Estrutura do Projeto

- **data/**: Dados brutos, processados e externos.
- **notebooks/**: Jupyter notebooks para exploração e experimentação.
- **src/**: Código fonte do projeto.
- **tests/**: Testes unitários e de integração.
- **env/**: Arquivos de configuração de ambientes (requirements.txt, environment.yml).

## Instalação

Instale as dependências utilizando pip:
```bash
pip install -r env/requirements.txt
```

Ou conda:
```bash
conda env create -f env/environment.yml
```