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

Resumo: **K-fold e DCV do WILCAR (M1) com o default** (nenhuma variável exportada, como em março). **DCV do
WILCAR-C (M2) com `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=4`**, que reproduz o número de threads
dos workers de março.

### K-fold: default

As runs de março não definiram `OMP_NUM_THREADS`, `MKL_NUM_THREADS` nem `OPENBLAS_NUM_THREADS`. Com 1 thread, o
OpenBLAS segue outro caminho de cálculo e os resultados mudam. Por isso o K-fold usa o default (16 threads hoje,
20 em março). Evidências:

- **WILCAR-C, Algerian, folds 3 e 5:** idênticos a março (bit a bit) com threads default, 2, 4 e 16. Com 1 thread,
  diferentes (fold 2: n\* 1 → 3; fold 4: n\* 1 → 2).
- **HF e Pima com os CSVs antigos (pré-correção):** 9 de 9 folds, cobrindo os métodos 1, 2, 3, 5 e 6, são idênticos a
  março com threads default (HF M1 f3, M2 f5, M3 f3, M5 f4, M6 f2; Pima M1 f4, M2 f1, M5 f1, M6 f1). O ambiente
  reproduz março exatamente, então as diferenças da re-rodada vêm só da correção dos dados.
- **2 threads não equivale ao default em geral:** breast, WILCAR-C, fold 3 com 2 threads dá n\* = 12 (março: 4).
  Com threads default, o fold é idêntico a março.
- **DCV:** o J_T(n) do WILCAR com n = 1..3 em modo paralelo é idêntico a março para Algerian, breast, BCD, HF e Pima
  (os dois últimos com os CSVs antigos). Na regressão, o DCV completo do WILCAR em `optical_network` e
  `computer_hardware` reproduz `vcd_20260310_140932` (histórico e resumo).

Com threads default, as execuções são determinísticas: repetições do mesmo fold, inclusive com a máquina carregada,
dão resultados idênticos. Os tempos por fold, porém, dependem da carga e não são comparáveis com março.

Nota técnica: para comparar valores com março lendo os CSVs, use `pd.read_csv(..., float_precision="round_trip")`.
O parser padrão do pandas pode errar o último bit.

### DCV: 4 threads por worker no WILCAR-C

Sem variáveis de thread, o joblib (loky) limita cada um dos 5 workers do DCV a `cpu_count // 5` threads. Em março, o
WSL expunha as 20 CPUs lógicas do host (sem `processors=` no `.wslconfig`), então os workers tinham 20 // 5 = 4
threads. Hoje o `.wslconfig` fixa `processors=16` (era 8 em maio/2026), então o default dá 16 // 5 = 3.

A diferença apareceu no BCD: o J_T(1) do WILCAR-C não batia com março. Em n = 1 não há reaproveitamento de pesos,
e o código desse passo é o mesmo de março. Passo n = 1 do DCV do WILCAR-C no BCD, por threads por worker:

| threads por worker | 1 | 2 | 3 (default hoje) | 4 (março) | 16 |
|---|---|---|---|---|---|
| J_T(1) | 1,197967 | 1,248308 | 1,248308 | **1,1996397449136573** | 1,1996397449136573 |
| igual a março | não | não | não | **sim** | sim |

Os outros datasets dão o mesmo J_T(1) com 3 ou 4 threads, porque o BCD tem 30 atributos e os demais têm até 13.
Para n ≥ 2, cada DCV de outubro (3 threads) foi re-rodado com 4 threads e comparado bit a bit:

- **WILCAR (M1), HF e Pima:** idênticos. Ficam as runs com o default.
- **WILCAR-C (M2):** com o reaproveitamento de pesos, diferenças de arredondamento se propagam pelo SLSQP.
  - Algerian, breast e Pima: diferenças na 8ª–9ª casa do J_T a partir de n = 3 (Pima: n = 7). n\* e métricas finais
    iguais.
  - HF: trajetória diverge a partir de n = 5, com n\* = 1 igual. A versão com 4 threads teve um passo descartado
    (n = 16, 3,7 h) e ainda roda, por isso **este commit usa a do HF com 3 threads** (`vcd_20261026_122110`). A
    de 4 threads entra num commit posterior.
  - Regressão: os 11 datasets diferem a partir de n = 4 a 11, na 8ª–10ª casa do J_T. n\* é igual e as métricas
    coincidem até a 6ª casa, exceto no aquatic_toxicity: n\* 40 → 33, R² 0,3859 → 0,3868, 48 passos (não chega
    mais a n = 50). synchronous_machine: n\* = 30, R² = 0,5073 ± 0,9485 nas duas versões, com o fold 0 preso.
  - BCD: n\* 4 → 1, MCC 0,936 → 0,868.

Para manter as condições de março, os DCVs do WILCAR-C passaram a valer na versão com 4 threads. Essas runs saíram
de cópias `git archive` do commit `a30890a` no scratchpad da sessão, com o mesmo comando e `nice -n 10`, e foram
copiadas para `results/dynamic_cv/` com o rótulo do dataset no nome (ex.: `vcd_20261027_004851_bcd`).

A proveniência está em `logs/rerun_2026-10/promoted.tsv` e no manifest: `supersedes`/`superseded_by`, comando,
diretório de origem e threads por job. As runs com o default continuam em `results/`, marcadas como substituídas.
O processo pai também fica com 4 threads (em março, 20), mas no BCD 4 e 16 threads dão o mesmo resultado.

### Pendente: 20 threads no K-fold e no processo pai do DCV

Em março, o default do OpenBLAS era 20 threads, nos processos do K-fold e no processo pai do DCV, que faz o retreino
final com `--retrain`. Hoje o K-fold usa 16 (o default), e o pai do DCV usa 4. Com 16 threads, os folds de HF,
Pima (métodos 1, 2, 3, 5 e 6, com os CSVs antigos) e breast verificados são idênticos a março. O computer_hardware
mostra, porém, que há casos sensíveis (ver abaixo), e o RIXM-C não foi verificado.

Próximo passo, sem mudar código: re-rodar o K-fold de HF e Pima e os DCVs de classificação com 20 threads no
processo principal, por um wrapper com `threadpoolctl`, e comparar bit a bit. Os que diferirem serão promovidos
num commit posterior, junto com o K-fold do Algerian e o DCV do HF com 4 threads.

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
