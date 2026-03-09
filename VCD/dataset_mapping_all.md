# Mapeamento Completo dos Datasets para VCD

## Contexto

A VCD está implementada para 3 datasets (computer_hardware, fish_toxicity, aquatic_toxicity).
Precisamos estender para todos os 11 problemas de regressão do Apêndice A da dissertação.

Nota: Energy Efficiency gera DOIS problemas (Heating Load e Cooling Load) com vetores de sinais idênticos.

## Datasets existentes no código (já configurados)

| # | Dataset Key | CSV Filename | n | p | Expected Signals |
|---|------------|-------------|---|---|-----------------|
| A.2 | computer_hardware | computer_hardware | 209 | 6 | [-1, +1, +1, +1, +1, +1] |
| A.7 | fish_toxicity | qsar_fish_toxicity | 908 | 6 | [-1, +1, -1, +1, +1, +1] |
| A.6 | aquatic_toxicity | qsar_aquatic_toxicity | 546 | 8 | [+1, -1, +1, +1, +1, -1, +1, +1] |

## Novos datasets a adicionar

| # | Dataset Key | CSV Filename | n | p | Expected Signals | Constraint Type |
|---|------------|-------------|---|---|-----------------|-----------------|
| A.1 | airfoil_self_noise | airfoil_self_noise | 1503 | 5 | [-1, 0, -1, +1, -1] | Partial (4/5) |
| A.3a | energy_heating | energy_efficiency_hl | 768 | 8 | [+1, -1, +1, -1, +1, 0, +1, 0] | Partial (6/8) |
| A.3b | energy_cooling | energy_efficiency_cl | 768 | 8 | [+1, -1, +1, -1, +1, 0, +1, 0] | Partial (6/8) |
| A.4 | lavender_friction | lavender_friction | 625 | 3 | [0, +1, +1] | Partial (2/3) |
| A.5 | optical_network | optical_interconnection | 630 | 3 | [-1, 0, +1] | Partial (2/3) |
| A.8 | real_estate | real_estate_valuation | 414 | 3 | [-1, -1, +1] | Full (3/3) |
| A.9 | synchronous_machine | synchronous_machine | 557 | 4 | [+1, -1, +1, +1] | Full (4/4) |
| A.10 | yacht_hydrodynamics | yacht_hydrodynamics | 308 | 6 | [0, 0, 0, 0, 0, +1] | Partial (1/6) |

## IMPORTANTE: Energy Efficiency (A.3)

Este dataset tem 8 inputs e 2 outputs (Heating Load e Cooling Load).
Os sinais esperados são IDÊNTICOS para ambos os targets.

Opções de implementação:
- **Opção A (recomendada)**: Dois CSVs separados onde a última coluna é o target:
  - `energy_efficiency_hl.csv` — 8 inputs + Heating Load
  - `energy_efficiency_cl.csv` — 8 inputs + Cooling Load
- **Opção B**: Um único CSV com 10 colunas, e o script seleciona qual output usar

Se os CSVs ainda não existem nesse formato, será necessário criá-los a partir do
dataset original (que tem 10 colunas: 8 inputs, HL, CL).

## Implementação no run_dynamic_cv.py

### 1. DATASET_FILES — mapeia dataset_key → csv_filename (sem .csv)

```python
DATASET_FILES = {
    # Existing
    'computer_hardware': 'computer_hardware',
    'fish_toxicity': 'qsar_fish_toxicity',
    'aquatic_toxicity': 'qsar_aquatic_toxicity',
    # New
    'airfoil_self_noise': 'airfoil_self_noise',
    'energy_heating': 'energy_efficiency_hl',
    'energy_cooling': 'energy_efficiency_cl',
    'lavender_friction': 'lavender_friction',
    'optical_network': 'optical_interconnection',
    'real_estate': 'real_estate_valuation',
    'synchronous_machine': 'synchronous_machine',
    'yacht_hydrodynamics': 'yacht_hydrodynamics',
}
```

### 2. EXPECTED_SIGNALS

```python
EXPECTED_SIGNALS = {
    # Existing
    'computer_hardware': [-1, 1, 1, 1, 1, 1],
    'fish_toxicity': [-1, 1, -1, 1, 1, 1],
    'aquatic_toxicity': [1, -1, 1, 1, 1, -1, 1, 1],
    # New
    'airfoil_self_noise': [-1, 0, -1, 1, -1],
    'energy_heating': [1, -1, 1, -1, 1, 0, 1, 0],
    'energy_cooling': [1, -1, 1, -1, 1, 0, 1, 0],
    'lavender_friction': [0, 1, 1],
    'optical_network': [-1, 0, 1],
    'real_estate': [-1, -1, 1],
    'synchronous_machine': [1, -1, 1, 1],
    'yacht_hydrodynamics': [0, 0, 0, 0, 0, 1],
}
```

### 3. VCD_METHOD_CONFIGS — configurações por dataset/método

Critérios para max_neurons e patience:
- Datasets pequenos (n < 400): max_neurons=300, patience=50-100
- Datasets médios (n 400-700): max_neurons=500, patience=100-150  
- Datasets grandes (n > 700): max_neurons=750, patience=150
- Constrained methods (2,4,6): sempre menores (max_neurons/3, patience/5 aprox.)
- ELM+R (M6): max_neurons=150, patience=15 (padrão existente)

```python
VCD_METHOD_CONFIGS = {
    # === EXISTING ===
    'computer_hardware': {  # n=209, p=6
        1: VCDMethodConfig(max_neurons=750, patience=150),
        2: VCDMethodConfig(max_neurons=200, patience=20),
        3: VCDMethodConfig(max_neurons=750, patience=150),
        4: VCDMethodConfig(max_neurons=200, patience=20),
        5: VCDMethodConfig(max_neurons=750, patience=150),
        6: VCDMethodConfig(max_neurons=150, patience=15),
    },
    'fish_toxicity': {  # n=908, p=6
        1: VCDMethodConfig(max_neurons=750, patience=150),
        2: VCDMethodConfig(max_neurons=200, patience=20),
        3: VCDMethodConfig(max_neurons=750, patience=150),
        4: VCDMethodConfig(max_neurons=200, patience=20),
        5: VCDMethodConfig(max_neurons=750, patience=150),
        6: VCDMethodConfig(max_neurons=150, patience=15),
    },
    'aquatic_toxicity': {  # n=546, p=8
        1: VCDMethodConfig(max_neurons=750, patience=150),
        2: VCDMethodConfig(max_neurons=200, patience=20),
        3: VCDMethodConfig(max_neurons=750, patience=150),
        4: VCDMethodConfig(max_neurons=200, patience=20),
        5: VCDMethodConfig(max_neurons=750, patience=150),
        6: VCDMethodConfig(max_neurons=150, patience=15),
    },
    # === NEW ===
    'airfoil_self_noise': {  # n=1503, p=5
        1: VCDMethodConfig(max_neurons=750, patience=150),
        2: VCDMethodConfig(max_neurons=200, patience=20),
        3: VCDMethodConfig(max_neurons=750, patience=150),
        4: VCDMethodConfig(max_neurons=200, patience=20),
        5: VCDMethodConfig(max_neurons=750, patience=150),
        6: VCDMethodConfig(max_neurons=150, patience=15),
    },
    'energy_heating': {  # n=768, p=8
        1: VCDMethodConfig(max_neurons=750, patience=150),
        2: VCDMethodConfig(max_neurons=200, patience=20),
        3: VCDMethodConfig(max_neurons=750, patience=150),
        4: VCDMethodConfig(max_neurons=200, patience=20),
        5: VCDMethodConfig(max_neurons=750, patience=150),
        6: VCDMethodConfig(max_neurons=150, patience=15),
    },
    'energy_cooling': {  # n=768, p=8
        1: VCDMethodConfig(max_neurons=750, patience=150),
        2: VCDMethodConfig(max_neurons=200, patience=20),
        3: VCDMethodConfig(max_neurons=750, patience=150),
        4: VCDMethodConfig(max_neurons=200, patience=20),
        5: VCDMethodConfig(max_neurons=750, patience=150),
        6: VCDMethodConfig(max_neurons=150, patience=15),
    },
    'lavender_friction': {  # n=625, p=3
        1: VCDMethodConfig(max_neurons=500, patience=100),
        2: VCDMethodConfig(max_neurons=200, patience=20),
        3: VCDMethodConfig(max_neurons=500, patience=100),
        4: VCDMethodConfig(max_neurons=200, patience=20),
        5: VCDMethodConfig(max_neurons=500, patience=150),
        6: VCDMethodConfig(max_neurons=150, patience=15),
    },
    'optical_network': {  # n=630, p=3
        1: VCDMethodConfig(max_neurons=500, patience=100),
        2: VCDMethodConfig(max_neurons=200, patience=20),
        3: VCDMethodConfig(max_neurons=500, patience=100),
        4: VCDMethodConfig(max_neurons=200, patience=20),
        5: VCDMethodConfig(max_neurons=500, patience=150),
        6: VCDMethodConfig(max_neurons=150, patience=15),
    },
    'real_estate': {  # n=414, p=3
        1: VCDMethodConfig(max_neurons=500, patience=100),
        2: VCDMethodConfig(max_neurons=200, patience=20),
        3: VCDMethodConfig(max_neurons=500, patience=100),
        4: VCDMethodConfig(max_neurons=200, patience=20),
        5: VCDMethodConfig(max_neurons=500, patience=150),
        6: VCDMethodConfig(max_neurons=150, patience=15),
    },
    'synchronous_machine': {  # n=557, p=4
        1: VCDMethodConfig(max_neurons=500, patience=100),
        2: VCDMethodConfig(max_neurons=200, patience=20),
        3: VCDMethodConfig(max_neurons=500, patience=100),
        4: VCDMethodConfig(max_neurons=200, patience=20),
        5: VCDMethodConfig(max_neurons=500, patience=150),
        6: VCDMethodConfig(max_neurons=150, patience=15),
    },
    'yacht_hydrodynamics': {  # n=308, p=6
        1: VCDMethodConfig(max_neurons=300, patience=100),
        2: VCDMethodConfig(max_neurons=200, patience=20),
        3: VCDMethodConfig(max_neurons=300, patience=100),
        4: VCDMethodConfig(max_neurons=200, patience=20),
        5: VCDMethodConfig(max_neurons=300, patience=150),
        6: VCDMethodConfig(max_neurons=150, patience=15),
    },
}
```

### 4. ESTIMATED_TIME_PER_STEP (seconds per constructive step per fold)

```python
ESTIMATED_TIME_PER_STEP = {
    # Existing
    'computer_hardware':    {1: 0.5, 2: 3.0, 3: 0.5, 4: 3.0, 5: 0.01, 6: 0.5},
    'fish_toxicity':        {1: 1.0, 2: 8.0, 3: 1.0, 4: 8.0, 5: 0.02, 6: 2.0},
    'aquatic_toxicity':     {1: 0.8, 2: 6.0, 3: 0.8, 4: 6.0, 5: 0.01, 6: 1.5},
    # New (estimates based on n_samples and n_inputs)
    'airfoil_self_noise':   {1: 1.5, 2: 10.0, 3: 1.5, 4: 10.0, 5: 0.03, 6: 3.0},
    'energy_heating':       {1: 1.2, 2: 8.0, 3: 1.2, 4: 8.0, 5: 0.02, 6: 2.5},
    'energy_cooling':       {1: 1.2, 2: 8.0, 3: 1.2, 4: 8.0, 5: 0.02, 6: 2.5},
    'lavender_friction':    {1: 0.4, 2: 2.0, 3: 0.4, 4: 2.0, 5: 0.01, 6: 0.3},
    'optical_network':      {1: 0.4, 2: 2.0, 3: 0.4, 4: 2.0, 5: 0.01, 6: 0.3},
    'real_estate':          {1: 0.3, 2: 1.5, 3: 0.3, 4: 1.5, 5: 0.01, 6: 0.3},
    'synchronous_machine':  {1: 0.4, 2: 2.5, 3: 0.4, 4: 2.5, 5: 0.01, 6: 0.4},
    'yacht_hydrodynamics':  {1: 0.3, 2: 2.0, 3: 0.3, 4: 2.0, 5: 0.01, 6: 0.3},
}
```

### 5. Update default datasets list and argparse choices

```python
ALL_DATASETS = [
    'computer_hardware',
    'fish_toxicity',
    'aquatic_toxicity',
    'airfoil_self_noise',
    'energy_heating',
    'energy_cooling',
    'lavender_friction',
    'optical_network',
    'real_estate',
    'synchronous_machine',
    'yacht_hydrodynamics',
]

# In VCDConfig:
datasets: List[str] = field(default_factory=lambda: ALL_DATASETS)

# In argparse:
parser.add_argument('--dataset', type=str, default=None,
                    choices=ALL_DATASETS,
                    help="Run specific dataset (default: all)")
```

### 6. Also update run_cross_validation.py with the same mappings

The static k-fold CV script should also be updated with the same
DATASET_FILES, EXPECTED_SIGNALS, and METHOD_CONFIGS dictionaries
so both scripts are consistent.

## Pre-requisites: CSV files in data/raw/

Before running, verify that all CSV files exist in data/raw/:
```bash
ls data/raw/*.csv
```

Expected files (confirm exact names):
- computer_hardware.csv
- qsar_fish_toxicity.csv
- qsar_aquatic_toxicity.csv
- airfoil_self_noise.csv
- energy_efficiency_hl.csv (or needs to be created from original)
- energy_efficiency_cl.csv (or needs to be created from original)
- lavender_friction.csv
- optical_interconnection.csv
- real_estate_valuation.csv
- synchronous_machine.csv
- yacht_hydrodynamics.csv

All CSVs should be semicolon-separated, no header, last column = target.

If energy_efficiency exists as a single file with 10 columns (8 inputs + HL + CL),
create the two separate files:
```python
import pandas as pd
data = pd.read_csv('data/raw/energy_efficiency.csv', header=None, sep=';')
# Assuming cols 0-7 are inputs, col 8 is HL, col 9 is CL
data.iloc[:, :9].to_csv('data/raw/energy_efficiency_hl.csv', header=False, index=False, sep=';')
data[list(range(8)) + [9]].to_csv('data/raw/energy_efficiency_cl.csv', header=False, index=False, sep=';')
```

## Summary of changes needed

1. **Verify CSV files exist** in data/raw/ with correct names and format
2. **Create energy_efficiency_hl.csv and energy_efficiency_cl.csv** if needed
3. **Update run_dynamic_cv.py**: DATASET_FILES, EXPECTED_SIGNALS, VCD_METHOD_CONFIGS, 
   ESTIMATED_TIME_PER_STEP, ALL_DATASETS, argparse choices
4. **Update run_cross_validation.py** with same mappings for consistency
5. **Create _config.json files** in data/raw/ for each new dataset (optional but recommended)
6. **Test one new dataset**: 
   `python src/run_dynamic_cv.py --methods 5 --dataset real_estate --base-dir .`
   (M5/ELM is fastest, real_estate is small)
