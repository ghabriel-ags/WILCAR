# Re-rodadas de outubro/2026 (dissertação, Capítulo 6)

Branch `dissertation/rerun-hf-pima`, worktree `WILCAR-dissertacao`, a partir de `main` (`0e54bfe`, 21/03/2026).

| Commit | Conteúdo |
|---|---|
| `13de119` | alterações de março no runner e no relatório (4 arquivos da working copy usada nas runs de março) |
| `e63e54c` | CSVs corrigidos: Heart Failure (idade, 2 células) e Pima (DPF, 47 células) |
| `060805b` | README: vetores de sinal, invocação do DCV, n do Algerian |
| `42b31cb` | correção do WILCAR-C no K-fold (`_expand_network` após tamanhos inviáveis; erro do Algerian, fold 1) |
| `a30890a` | correção do reaproveitamento de pesos do WILCAR-C no DCV paralelo e regra de passos descartados |

Ambiente: micromamba `mestrado_ghabriel_pei` (o mesmo de março), Python 3.14.0, NumPy 2.3.5, SciPy 1.16.3,
scikit-learn 1.7.2, pandas 2.3.3, OpenBLAS 0.3.30 (pthreads), 16 CPUs lógicas (em março, 20; ver abaixo). Os
pacotes numéricos não mudaram desde a criação do ambiente (nov/2025). O torch, instalado em set/2026 para outro
projeto, não altera os resultados.

## Threads do BLAS

Resumo da política, para emular março:
- **K-fold:** 20 threads no processo principal, emuladas com `logs/rerun_2026-10/with_blas_threads.py`.
- **DCV do WILCAR-C:** 4 threads por worker (`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=4`).
- **DCV do WILCAR (M1):** default (3 por worker), idêntico a 4.

As runs de março não definiam variáveis de thread, então valiam os defaults. Na época, o WSL expunha as 20 CPUs
lógicas do host (sem `processors=` no `.wslconfig`): 20 threads no processo principal e 20 // 5 = 4 em cada worker
do DCV (o joblib/loky limita os workers a `cpu_count // n_jobs`). Hoje o `.wslconfig` fixa `processors=16` (era 8
em maio/2026), e os defaults dão 16 e 16 // 5 = 3. A variável `OPENBLAS_NUM_THREADS` fica limitada às CPUs
visíveis, mas `openblas_set_num_threads`, chamada pelo `threadpoolctl` no wrapper, não. Assim, 20 threads podem
ser emuladas sem mudar código.

### Evidências gerais

- **1 thread muda os resultados:** WILCAR-C, Algerian, fold 2: n\* 1 → 3; fold 4: n\* 1 → 2. Os folds 3 e 5 são
  idênticos a março com threads default, 2, 4 e 16.
- **2 threads não equivale ao default:** breast, WILCAR-C, fold 3 com 2 threads dá n\* = 12 (março: 4).
- **16 threads reproduzem março na maioria dos folds, mas não em todos:**
  - Idênticos com 16 threads: HF e Pima com os CSVs antigos, 9 de 9 folds (HF M1 f3, M2 f5, M3 f3, M5 f4, M6 f2;
    Pima M1 f4, M2 f1, M5 f1, M6 f1), e os 5 folds do breast WILCAR-C.
  - Diferentes: no computer_hardware (WILCAR-C, `cv_20260310_140954`), os folds 1, 2 e 5 diferem na 8ª–10ª casa
    com 16 threads e são idênticos a março com 20.
  - Com os CSVs corrigidos, o ELM-C do HF muda no fold 3 (tabela abaixo).
- **DCV:** o J_T(n) do WILCAR com n = 1..3 em modo paralelo é idêntico a março para Algerian, breast, BCD, HF e Pima
  (os dois últimos com os CSVs antigos). Na regressão, o DCV completo do WILCAR em `optical_network` e
  `computer_hardware` reproduz `vcd_20260310_140932` (histórico e resumo).
- **Determinismo:** com um número fixo de threads, repetições do mesmo fold, inclusive com a máquina carregada,
  dão resultados idênticos. Os tempos por fold dependem da carga e não são comparáveis com março.

Nota técnica: para comparar valores com março lendo os CSVs, use `pd.read_csv(..., float_precision="round_trip")`.
O parser padrão do pandas pode errar o último bit.

### K-fold: 20 threads

O K-fold foi rodado primeiro com 16 threads (o default). Uma verificação com 20 threads, mesmo comando e mesmo commit
(`42b31cb`), comparada bit a bit com essas runs, deu:

| Run (CSVs corrigidos) | 20 vs. 16 threads |
|---|---|
| Pima, ELM-C (M6), 5 folds | idêntico |
| Pima, ELM (M5), 5 folds | folds 1, 2 e 5 idênticos. Folds 3 e 4: R²/MSE diferem na 13ª–14ª casa (até 2,5 × 10⁻¹³; ex.: R² de teste −0,32543178352805757 vs. −0,32543178352781066). n\*, Acc, F1 e MCC idênticos |
| HF, ELM (M5), 5 folds | folds 2 e 4 idênticos. Folds 1, 3 e 5: R²/MSE diferem na 15ª–16ª casa (até 3,6 × 10⁻¹⁵). n\*, Acc, F1 e MCC idênticos |
| **HF, ELM-C (M6), 5 folds** | folds 1, 2, 4 e 5 idênticos. **Fold 3: n\* 45 → 49, F1 0,800 → 0,818, MCC 0,702 → 0,733** (Acc 0,867 igual). Média: MCC 0,622 → 0,628, F1 0,722 → 0,725 |
| HF, RIXM-C (M4), 5 folds | idêntico |

**Validação da emulação.**
- `OPENBLAS_NUM_THREADS=20` não basta: o OpenBLAS limita a variável às 16 CPUs visíveis, e o `threadpool_info()`
  do processo mostra `num_threads` = 16. Com o wrapper, mostra 20 (OpenBLAS 0.3.30, pthreads).
- O fold 3 do ELM-C no HF com o CSV antigo (o de março, md5 idêntico) deu idêntico a `cv_20260323_194416` nas
  três variantes: variável = 20 (efetivo 16), wrapper com 20 e default 16. Esse fold não distingue 16 de 20 com os
  dados antigos.
- O teste que distingue é o do computer_hardware (ver *Evidências gerais*): os folds 1, 2 e 5 diferem de março com
  16 threads e são idênticos com 20.

Como 16 vs. 20 threads muda resultados na precisão reportada, todo o K-fold de HF, Pima e Algerian vale na versão
com 20 threads:
- **M5/M6 de HF e Pima e o M4 do HF:** são as próprias runs de verificação. Rodaram em cópias `git archive` de
  `42b31cb` no scratchpad e foram promovidas para `results/cross_validation/`, com o rótulo no nome.
- **HF M1/M2 e M3, Pima M1/M2, M3 e M4 e Algerian M2:** foram lançados no worktree pelo `launch.sh` (jobs `*_t20`).
  Para reduzir a disputa de CPU entre threads do OpenBLAS, rodam ao mesmo tempo só o DCV do HF M2 (nice 10), o
  Algerian (nice 15) e o Pima M4 (nice 17). HF M1/M2 e M3, Pima M1/M2 e M3 foram encerrados após cerca de 17 min
  e seguem um por vez em `queue_t20.sh` (nice 19, rótulos `*_t20_r2`). A concorrência não altera resultados, só
  o tempo de parede.

As runs com 16 threads continuam em `results/` e ficam marcadas como substituídas (`superseded_by`) quando a de 20
termina sem erro. Até lá, o relatório usa a de 16, e a lista de pendências está no fim deste arquivo.

Runs `*_t20` do worktree já terminadas, comparadas bit a bit com as de 16 threads (`scripts/compare_kfold_runs.py`:
todas as colunas de `fold_results.csv`, menos o tempo de treino):

| Run | Término | 20 vs. 16 threads |
|---|---|---|
| Pima, RIXM (M3), `cv_20261027_094521` (`cv_pima_m3_t20_r2`) | 27/10 15h10, exit 0, 324,7 min | idêntico nos 5 folds |

### DCV: 4 threads por worker no WILCAR-C

A diferença de threads apareceu primeiro no BCD: o J_T(1) do WILCAR-C não batia com março. Em n = 1 não há
reaproveitamento de pesos, e o código desse passo é o mesmo de março. Passo n = 1 do DCV do WILCAR-C no BCD, por
threads por worker:

| threads por worker | 1 | 2 | 3 (default hoje) | 4 (março) | 16 |
|---|---|---|---|---|---|
| J_T(1) | 1,197967 | 1,248308 | 1,248308 | **1,1996397449136573** | 1,1996397449136573 |
| igual a março | não | não | não | **sim** | sim |

Os outros datasets dão o mesmo J_T(1) com 3 ou 4 threads, porque o BCD tem 30 atributos e os demais têm até 13.
Para n ≥ 2, cada DCV de outubro (3 threads) foi re-rodado com 4 threads e comparado bit a bit:

- **WILCAR (M1), HF e Pima:** idênticos com 4 threads por worker, e também com o pai em 20 threads e os workers em
  4 (condição de março), inclusive o retreino final. Ficam as runs com o default. As verificações estão em
  `logs/rerun_2026-10/verifications.tsv` e no manifest (`jobs[].verifications`).
- **WILCAR-C (M2):** com o reaproveitamento de pesos, diferenças de arredondamento se propagam pelo SLSQP.
  - Algerian, breast e Pima: diferenças na 8ª–9ª casa do J_T a partir de n = 3 (Pima: n = 7). n\* e métricas finais
    iguais.
  - HF: trajetória diverge a partir de n = 5. Com 4 threads, o passo n = 16 foi descartado (3,7 h) e o n = 17 foi
    aceito (4,5 h), encerrando pela paciência (17 passos contra 16). n\* = 1, J_T\*, métricas em n\* e retreino
    final são idênticos à versão com 3 threads.
  - Regressão: os 11 datasets diferem a partir de n = 4 a 11, na 8ª–10ª casa do J_T. n\* é igual e as métricas
    coincidem até a 6ª casa, exceto no aquatic_toxicity: n\* 40 → 33, R² 0,3859 → 0,3868, 48 passos (não chega
    mais a n = 50). synchronous_machine: n\* = 30, R² = 0,5073 ± 0,9485 nas duas versões, com o fold 0 preso.
  - BCD: n\* 4 → 1, MCC 0,936 → 0,868.

Os DCVs do WILCAR-C valem na versão com 4 threads. Essas runs saíram de cópias `git archive` do commit `a30890a`
no scratchpad da sessão, com o mesmo comando e `nice -n 10`, e foram copiadas para `results/dynamic_cv/` com o
rótulo do dataset no nome (ex.: `vcd_20261027_004851_bcd`).

**Processo pai.** O pai dessas runs também ficou com 4 threads (em março, 20). Ele faz o retreino final com
`--retrain`. Com o pai em 20 e os workers em 4, pelo wrapper, deram idêntico: o DCV do Algerian (WILCAR-C), os
DCVs do HF e do Pima (WILCAR) e o retreino do HF (WILCAR-C, n\* = 1). As verificações do pai para breast, Pima e
BCD (WILCAR-C) foram interrompidas para dar prioridade às rodadas oficiais.

A proveniência está em `logs/rerun_2026-10/promoted.tsv` e no manifest: `supersedes`/`superseded_by`, comando,
diretório de origem, commit e threads por job.

## Configuração (igual à de março)

- K-fold: 5 folds, seed 0, shuffle, defaults do código (`max_neurons`/paciência 750/150 para WILCAR e ELM, 500/150
  para RIXM, 50/15 para os métodos com restrição). Exceção: **HF RIXM-C** com `--max-neurons 5 --patience 5`, como
  em março (`cv_20260330_090253`).
- Pima RIXM e RIXM-C rodam em processos separados. Cada modelo é re-semeado com `seed + 100·fold + método`, então a
  separação não muda resultados, só o tempo de parede.
- DCV de classificação com `--retrain`. DCV de regressão sem `--retrain`. **BCD WILCAR-C** com `--max-neurons 5`,
  como em março (`vcd_20260328_200827`).
- Todos os DCVs de março rodaram em modo paralelo (5 workers). Ver a impressão digital acima.

## Correções de código e não regressão

- **K-fold, WILCAR-C (`42b31cb`):** ver `algerian_fix_note.md`. Testes em `tests/test_wilcar_constrained.py`. A/B
  corrigido vs. sem correção, nas mesmas condições, dá resultado bit a bit idêntico (computer_hardware folds 1–3,
  breast fold 3). Com o código corrigido, os 5 folds do breast são idênticos a março (`cv_20260321_015941`):
  folds 2 e 3 com threads default, folds 1, 4 e 5 com 2 threads. O fold 2 levou 5,8 h.
  computer_hardware (`cv_20260310_140954`): com 16 threads (o default de hoje), os folds 1, 2 e 5 diferem de março
  na 8ª–10ª casa, com n\* igual, e os folds 3 e 4 são idênticos. Com 20 threads, o default de março, emulado com
  `threadpoolctl.threadpool_limits(20)` porque a variável de ambiente fica limitada às 16 CPUs visíveis, os folds 1,
  2 e 5 também são idênticos. A diferença vem do número de threads, não do código.
- **DCV, WILCAR-C (`a30890a`):** ver `dcv_wilcarc_reuse.md`. Testes em `tests/test_dcv_wilcarc_reuse.py`.
- **Re-semeadura por passo no worker do DCV (não alterada):** o worker recria o modelo a cada n com semente
  `seed + 100·k`. Por isso, o neurônio novo do WILCAR (M1) recebe a mesma inicialização aleatória a cada n no mesmo
  fold. Mudar isso alteraria o M1 de março, então foi mantido e apenas registrado aqui.

## Pendências

Para commits posteriores (estado em 27/10, 15h10):

- **K-fold com 20 threads:** Algerian M2 (`cv_20261027_092850`) e Pima M4 (`cv_20261027_092317`) em execução. Na
  fila `queue_t20.sh`: HF M1/M2 (em execução desde 15h10), HF M3, Pima M1/M2; o Pima M3 já terminou. Até
  terminarem, o relatório usa as versões com 16 threads desses grupos.
- **K-fold Algerian WILCAR-C com 16 threads** (`cv_20261026_113259`): encerrado por decisão em 27/10 às 09h40,
  depois de 22 h no fold 1. Pela política, não entra nas tabelas.
