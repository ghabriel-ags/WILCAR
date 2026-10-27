Scripts de diagnóstico e verificação usados nesta re-rodada, copiados do scratchpad da sessão. Os caminhos absolutos
dentro deles apontam para esse scratchpad, que é temporário; ficam aqui como registro.

- `dcv_t4_queue.sh`: re-rodada dos DCVs com 4 threads por worker em cópias `git archive` de `a30890a` e comparação
  com as runs de 3 threads (`compare_dcv_runs.py`). Os DCVs de BCD e de regressão foram lançados à parte, com o
  mesmo comando registrado em `../promoted.tsv`.
- `bcd_n1_threads.sh`: passo n = 1 do DCV do BCD com 1, 2, 3, 4 e 16 threads e com `--no-parallel`.
- `run_one_fold.py` + `compare_kfold_folds.py`: um fold do K-fold (`BLAS_THREADS=20` emula o default de março;
  `UNFIXED=1` restaura o `_expand_network` anterior a `42b31cb`) e comparação exata com os `fold_results.csv` de março.
- `diag_sync.py` + `analyze_sync.py`: réplicas instrumentadas do DCV do WILCAR-C no synchronous_machine
  (`DIAG_THREADS=4`) e figuras/tabela por fold.
- `diag_kfold_vs_dcv.py`, `diag_n1_factorial.py`: fold 0 do synchronous_machine, K-fold vs. DCV, e teste fatorial em n = 1.
- `kfold_dcv_t20_verification_queue.sh` + `compare_kfold_runs.py`: verificação com 20 threads no processo
  principal (`../with_blas_threads.py`) do K-fold de HF e Pima e dos DCVs de classificação, comparada bit a bit com
  as runs oficiais. HF M5/M6, Pima M5/M6 e HF M4 terminaram e foram promovidos. Os demais foram encerrados e
  re-rodados oficialmente (`../queue_t20.sh`).
- `../validation_t20_hf_m6_f3.txt`: validação da emulação de 20 threads (`threadpool_info()` e resultado do fold 3
  do ELM-C no HF com o CSV antigo, nas variantes `OPENBLAS_NUM_THREADS=20`, wrapper e default).
