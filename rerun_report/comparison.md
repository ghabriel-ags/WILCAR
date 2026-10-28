# Re-run Heart Failure / Pima (out/2026) vs. março/2026

Código: branch `dissertation/rerun-hf-pima`, commit `bedbf220011c9eff62e7328d288a83f6bf752614`. Única mudança de entrada vs. março: os dois CSVs corrigidos (HF idade, 2 células; Pima DPF, 47 células).

Runs de março: cv_20260322_092703, cv_20260329_233245, cv_20260330_090253, cv_20260323_194416, cv_20260322_112130, cv_20260329_095115, cv_20260323_195252, vcd_20260328_200812, vcd_20260329_001412.  
Runs novas: vcd_20261026_113311, vcd_20261026_114430, cv_20261027_092850, cv_20261027_094521, cv_20261027_151025, cv_20261027_171128, cv_20261027_181331, vcd_20261027_004830_alg, vcd_20261027_004851_bcd, vcd_20261027_005421_pima, vcd_20261027_011812_bco, vcd_20261027_073947_reg, cv_20261027_083525_hf_m56, cv_20261027_083525_pima_m56, cv_20261027_083525_hf_m4, vcd_20261027_011812_hf, cv_20261027_092317_composed.

Threads do BLAS por run (política: K-fold com 20, DCV do WILCAR-C com 4 por worker):  
- `vcd_20261026_113311` (dcv_hf_m1): default (no thread variable set: 16 in the main process, 16 // 5 = 3 per DCV worker)  
- `vcd_20261026_114430` (dcv_pima_m1): default (no thread variable set: 16 in the main process, 16 // 5 = 3 per DCV worker)  
- `cv_20261027_092850` (cv_alg_m2_t20): 20 in the main process (with_blas_threads.py, threadpoolctl; March default)  
- `cv_20261027_094521` (cv_pima_m3_t20_r2): 20 in the main process (with_blas_threads.py, threadpoolctl; March default)  
- `cv_20261027_151025` (cv_hf_m12_t20_r2): 20 in the main process (with_blas_threads.py, threadpoolctl; March default)  
- `cv_20261027_171128` (cv_hf_m3_t20_r2): 20 in the main process (with_blas_threads.py, threadpoolctl; March default)  
- `cv_20261027_181331` (cv_pima_m12_t20_r2): 20 in the main process (with_blas_threads.py, threadpoolctl; March default)  
- `vcd_20261027_004830_alg` (dcvm2_alg_t4): OMP/OPENBLAS/MKL_NUM_THREADS=4 (4 per DCV worker, as in March; parent process also 4)  
- `vcd_20261027_004851_bcd` (dcvm2_bcd_t4): OMP/OPENBLAS/MKL_NUM_THREADS=4 (4 per DCV worker, as in March; parent process also 4)  
- `vcd_20261027_005421_pima` (dcvm2_pima_t4): OMP/OPENBLAS/MKL_NUM_THREADS=4 (4 per DCV worker, as in March; parent process also 4)  
- `vcd_20261027_011812_bco` (dcvm2_bco_t4): OMP/OPENBLAS/MKL_NUM_THREADS=4 (4 per DCV worker, as in March; parent process also 4)  
- `vcd_20261027_073947_reg` (dcvm2_reg_t4): OMP/OPENBLAS/MKL_NUM_THREADS=4 (4 per DCV worker, as in March; parent process also 4)  
- `cv_20261027_083525_hf_m56` (cv_hf_m56_t20): 20 in the main process (with_blas_threads.py, threadpoolctl; March default)  
- `cv_20261027_083525_pima_m56` (cv_pima_m56_t20): 20 in the main process (with_blas_threads.py, threadpoolctl; March default)  
- `cv_20261027_083525_hf_m4` (cv_hf_m4_t20): 20 in the main process (with_blas_threads.py, threadpoolctl; March default)  
- `vcd_20261027_011812_hf` (dcvm2_hf_t4): OMP/OPENBLAS/MKL_NUM_THREADS=4 (4 per DCV worker, as in March; parent process also 4)  
- `cv_20261027_092317_composed` (cv_pima_m4_t20_composed): 20 in the main process of every fold (with_blas_threads.py, threadpoolctl; March default): folds 1, 2, 3 in cv_pima_m4_t20, folds 4, 5 in their own processes; the composition trains nothing

## Checagens de sanidade

**Todas as checagens passaram.**

- ✅ Heart Failure M2 WILCAR-C (k-fold): SCR por fold = ['100%', '100%', '100%', '100%', '100%']
- ✅ Heart Failure M4 RIXM-C (k-fold): SCR por fold = ['100%', '100%', '100%', '100%', '100%']
- ✅ Heart Failure M6 ELM-C (k-fold): SCR por fold = ['100%', '100%', '100%', '100%', '100%']
- ✅ Heart Failure DCV M2 WILCAR-C: SCR por fold = ['100%', '100%', '100%', '100%', '100%'], final_SCR = 100%
- ✅ Heart Failure DCV M1 WILCAR: final_n = 1 ≤ n* = 3
- ✅ Heart Failure DCV M2 WILCAR-C: final_n = 1 ≤ n* = 1
- ✅ Pima M2 WILCAR-C (k-fold): SCR por fold = ['100%', '100%', '100%', '100%', '100%']
- ✅ Pima M4 RIXM-C (k-fold): SCR por fold = ['100%', '100%', '100%', '100%', '100%']
- ✅ Pima M6 ELM-C (k-fold): SCR por fold = ['100%', '100%', '100%', '100%', '100%']
- ✅ Pima DCV M2 WILCAR-C: SCR por fold = ['100%', '100%', '100%', '100%', '100%'], final_SCR = 100%
- ✅ Pima DCV M1 WILCAR: final_n = 3 ≤ n* = 4
- ✅ Pima DCV M2 WILCAR-C: final_n = 1 ≤ n* = 2
- ✅ Algerian M2 WILCAR-C (k-fold, re-run): 5 folds sem erro, SCR por fold = ['100%', '100%', '100%', '100%', '100%']
- ✅ nenhum NaN nas tabelas de resultado

## K-fold (5 folds): março → outubro

Δ = novo − antigo. Valores: média ± desvio sobre os 5 folds.

### Heart Failure

| Método | Acc mar | Acc out | Δ | MCC mar | MCC out | Δ | F1 mar | F1 out | Δ | n* mar | n* out | SCR % mar | SCR % out | t/fold mar (s) | t/fold out (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 WILCAR | 0.846 ± 0.048 | 0.839 ± 0.039 | -0.007 | 0.645 ± 0.111 | 0.638 ± 0.079 | -0.007 | 0.732 ± 0.097 | 0.725 ± 0.077 | -0.007 | 10.6 ± 11.0 | 1.6 ± 0.5 | 82.86 | 94.29 | 199 | 156 |
| M2 WILCAR-C | 0.823 ± 0.038 | 0.823 ± 0.022 | +0.000 | 0.598 ± 0.079 | 0.609 ± 0.067 | +0.010 | 0.692 ± 0.083 | 0.703 ± 0.073 | +0.011 | 2.0 ± 1.4 | 3.4 ± 4.3 | 100.00 | 100.00 | 1173 | 1286 |
| M3 RIXM | 0.849 ± 0.039 | 0.849 ± 0.031 | +0.000 | 0.651 ± 0.100 | 0.658 ± 0.091 | +0.008 | 0.726 ± 0.090 | 0.740 ± 0.082 | +0.013 | 145.0 ± 42.1 | 168.6 ± 69.1 | 91.43 | 91.43 | 658 | 737 |
| M4 RIXM-C ¹ | 0.803 ± 0.043 | 0.816 ± 0.024 | +0.013 | 0.530 ± 0.114 | 0.597 ± 0.049 | +0.068 | 0.646 ± 0.112 | 0.683 ± 0.061 | +0.037 | 1.8 ± 0.8 | 2.0 ± 1.2 | 100.00 | 100.00 | 204 | 529 |
| M5 ELM | 0.873 ± 0.038 | 0.859 ± 0.026 | -0.013 | 0.706 ± 0.105 | 0.678 ± 0.076 | -0.027 | 0.785 ± 0.083 | 0.764 ± 0.061 | -0.021 | 69.0 ± 37.1 | 49.8 ± 24.1 | 68.57 | 80.00 | 26 | 125 |
| M6 ELM-C | 0.843 ± 0.043 | 0.836 ± 0.022 | -0.007 | 0.634 ± 0.107 | 0.628 ± 0.067 | -0.006 | 0.727 ± 0.083 | 0.725 ± 0.060 | -0.002 | 25.8 ± 6.6 | 33.8 ± 13.6 | 100.00 | 100.00 | 77 | 361 |

- Ordenação por MCC (igual):
  - março:   ELM > RIXM > WILCAR > ELM-C > WILCAR-C > RIXM-C
  - outubro: ELM > RIXM > WILCAR > ELM-C > WILCAR-C > RIXM-C
- Ordenação por Accuracy (igual):
  - março:   ELM > RIXM > WILCAR > ELM-C > WILCAR-C > RIXM-C
  - outubro: ELM > RIXM > WILCAR > ELM-C > WILCAR-C > RIXM-C
- Ordenação por F1 (**mudou**):
  - março:   ELM > WILCAR > ELM-C > RIXM > WILCAR-C > RIXM-C
  - outubro: ELM > RIXM > ELM-C > WILCAR > WILCAR-C > RIXM-C

### Pima

| Método | Acc mar | Acc out | Δ | MCC mar | MCC out | Δ | F1 mar | F1 out | Δ | n* mar | n* out | SCR % mar | SCR % out | t/fold mar (s) | t/fold out (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 WILCAR | 0.786 ± 0.020 | 0.786 ± 0.024 | +0.000 | 0.518 ± 0.030 | 0.516 ± 0.032 | -0.002 | 0.672 ± 0.026 | 0.656 ± 0.025 | -0.015 | 47.6 ± 51.3 | 49.0 ± 47.4 | 96.67 | 93.33 | 801 | 646 |
| M2 WILCAR-C | 0.771 ± 0.030 | 0.776 ± 0.020 | +0.005 | 0.496 ± 0.036 | 0.512 ± 0.048 | +0.015 | 0.654 ± 0.048 | 0.666 ± 0.046 | +0.012 | 3.0 ± 2.0 | 5.8 ± 4.9 | 100.00 | 100.00 | 2456 | 1324 |
| M3 RIXM | 0.776 ± 0.026 | 0.784 ± 0.028 | +0.008 | 0.488 ± 0.059 | 0.508 ± 0.061 | +0.020 | 0.605 ± 0.070 | 0.625 ± 0.074 | +0.021 | 236.0 ± 150.3 | 311.2 ± 180.0 | 100.00 | 96.67 | 2544 | 3896 |
| M4 RIXM-C | 0.776 ± 0.010 | 0.781 ± 0.031 | +0.005 | 0.503 ± 0.026 | 0.515 ± 0.050 | +0.012 | 0.663 ± 0.038 | 0.672 ± 0.043 | +0.009 | 4.4 ± 4.0 | 7.0 ± 7.2 | 100.00 | 100.00 | 8181 | 15619 |
| M5 ELM | 0.786 ± 0.021 | 0.786 ± 0.013 | +0.000 | 0.510 ± 0.066 | 0.515 ± 0.047 | +0.006 | 0.648 ± 0.069 | 0.655 ± 0.052 | +0.007 | 68.4 ± 48.5 | 73.6 ± 55.7 | 66.67 | 83.33 | 36 | 266 |
| M6 ELM-C | 0.771 ± 0.020 | 0.772 ± 0.022 | +0.001 | 0.475 ± 0.055 | 0.477 ± 0.056 | +0.003 | 0.598 ± 0.086 | 0.605 ± 0.069 | +0.008 | 18.0 ± 17.7 | 22.6 ± 11.8 | 100.00 | 100.00 | 23 | 119 |

- Ordenação por MCC (igual):
  - março:   WILCAR > ELM > RIXM-C > WILCAR-C > RIXM > ELM-C
  - outubro: WILCAR > ELM > RIXM-C > WILCAR-C > RIXM > ELM-C
- Ordenação por Accuracy (**mudou**):
  - março:   ELM > WILCAR > RIXM-C > RIXM > WILCAR-C > ELM-C
  - outubro: WILCAR > ELM > RIXM > RIXM-C > WILCAR-C > ELM-C
- Ordenação por F1 (**mudou**):
  - março:   WILCAR > RIXM-C > WILCAR-C > ELM > RIXM > ELM-C
  - outubro: RIXM-C > WILCAR-C > WILCAR > ELM > RIXM > ELM-C

¹ Heart Failure M4: max_neurons=5, patience=5 (override, as in March)

## DCV (WILCAR / WILCAR-C, J_T = soma da BCE de teste)

| Dataset | Método | n* mar | n* out | Acc mar | Acc out | Δ | MCC mar | MCC out | Δ | F1 mar | F1 out | SCR % mar | SCR % out | final_n mar/out | tempo mar/out (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Heart Failure | M1 WILCAR | 2 | 3 | 0.809 ± 0.043 | 0.829 ± 0.020 | +0.020 | 0.563 | 0.624 | +0.060 | 0.666 | 0.715 | 94.29 | 80.00 | 2/1 | 3.4/10.2 |
| Heart Failure | M2 WILCAR-C | 1 | 1 | 0.799 ± 0.033 | 0.819 ± 0.024 | +0.020 | 0.555 | 0.596 | +0.041 | 0.677 | 0.708 | 100.00 | 100.00 | 1/1 | 242.4/534.7 |
| Pima | M1 WILCAR | 4 | 4 | 0.766 ± 0.025 | 0.768 ± 0.025 | +0.003 | 0.472 | 0.475 | +0.004 | 0.642 | 0.635 | 93.33 | 90.00 | 4/3 | 13.3/37.2 |
| Pima | M2 WILCAR-C | 3 | 2 | 0.756 ± 0.012 | 0.767 ± 0.022 | +0.010 | 0.449 | 0.476 | +0.026 | 0.620 | 0.644 | 100.00 | 100.00 | 3/1 | 213.4/23.7 |

- Heart Failure, DCV por MCC: março WILCAR > WILCAR-C; outubro WILCAR > WILCAR-C (igual)
- Pima, DCV por MCC: março WILCAR > WILCAR-C; outubro WILCAR-C > WILCAR (**mudou**)

## Correlação ponto-bisserial com o alvo (antes → depois da correção)

| Dataset | Variável | sinal esperado | r antes | r depois |
|---|---|---|---|---|
| Heart Failure | age | +1 | 0.032 | 0.254 |
| Pima | DiabetesPedigreeFunction | +1 | 0.112 | 0.174 |

As demais variáveis não mudam (tabela completa em `correlations.csv`).

- Heart Failure: n = 299, classe 0 = 203, classe 1 = 96
- Pima: n = 768, classe 0 = 500, classe 1 = 268

## Algerian — WILCAR-C (5 folds)

Re-run só do M2 com o código corrigido; o WILCAR (M1) não teve falha e não foi re-rodado.

| Versão | folds | Acc | MCC | F1 | n* | SCR % | t/fold (s) |
|---|---|---|---|---|---|---|---|
| março (4 folds: 2-5) | 4 | 0.949 ± 0.026 | 0.894 ± 0.057 | 0.956 ± 0.022 | 1.25 ± 0.50 | 100.00 | 53 |
| outubro (5 folds) | 5 | 0.947 ± 0.023 | 0.892 ± 0.049 | 0.954 ± 0.019 | 5.00 ± 8.40 | 100.00 | 10420 |

Por fold (março → outubro):

| Fold | n* mar | n* out | Acc mar | Acc out | MCC mar | MCC out | F1 mar | F1 out | SCR mar | SCR out | idêntico? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | erro | 20 | erro | 0.939 | erro | 0.883 | erro | 0.945 | erro | 1.00 | — |
| 2 | 1 | 1 | 0.918 | 0.918 | 0.833 | 0.833 | 0.929 | 0.929 | 1.00 | 1.00 | sim |
| 3 | 1 | 1 | 0.959 | 0.959 | 0.921 | 0.921 | 0.957 | 0.957 | 1.00 | 1.00 | sim |
| 4 | 1 | 1 | 0.979 | 0.979 | 0.958 | 0.958 | 0.982 | 0.982 | 1.00 | 1.00 | sim |
| 5 | 2 | 2 | 0.938 | 0.938 | 0.862 | 0.862 | 0.955 | 0.955 | 1.00 | 1.00 | sim |

Erro de março no fold 1: `cannot reshape array of size 2 into shape (1,13)`

### Causa do erro e correção

**Causa.** No WILCAR-C, quando nenhuma das até 100 reinicializações atinge 100% de SCR num tamanho `n`, o
laço construtivo de `train()` pula esse tamanho (`continue`) e mantém em `previous_param` o último modelo viável,
com `k < n` neurônios. No tamanho seguinte, `_expand_network(previous_param, n + 1)` acrescentava sempre **um**
neurônio, produzindo uma rede de `k + 1` neurônios, enquanto `_vector_to_params(vec, n + 1)` desempacotava o vetor
de parâmetros como se ele tivesse `n + 1`. No fold 1 do Algerian (p = 10), o tamanho 12 foi inviável depois de um
modelo viável de 11 neurônios. No tamanho 13, o vetor tinha 12·(10 + 2) + 1 = 145 elementos e, ao desempacotá-lo
com n = 13, sobraram só 2 elementos para `W2` (1 × 13): `cannot reshape array of size 2 into shape (1,13)`.
Esse caminho sempre levava a erro, porque qualquer tamanho pulado (exceto o último) seguido de outro passo deixa
o vetor curto. Nenhuma run que terminou passou por ele. O mesmo caminho existia no DCV
(`constructive_step`, que também pula tamanhos inviáveis), mas nunca foi atingido nas runs de março.

**Correção** (`src/wilcar/constrained.py`, `_expand_network`). A rede passa a ser completada até `n` neurônios
a partir do último modelo viável, com um neurônio novo por tamanho pulado. Isso preserva o reaproveitamento de
pesos e a semântica "pular o tamanho inviável e seguir". No caso regular (modelo anterior com `n − 1` neurônios),
o laço executa exatamente uma vez, com as mesmas chamadas ao gerador aleatório e na mesma ordem. O comportamento
é bit a bit idêntico ao de março em todos os casos que já funcionavam: teste
`tests/test_wilcar_constrained.py` e comparação por fold, código corrigido vs. código sem correção, nas mesmas
condições.
