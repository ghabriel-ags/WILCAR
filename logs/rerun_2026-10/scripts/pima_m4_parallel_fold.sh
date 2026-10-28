#!/usr/bin/env bash
# Pima RIXM-C (M4), one fold in its own process: corrected CSV, code of the archived commit (COMMIT), 20 BLAS threads
# through with_blas_threads.py (as the main run cv_pima_m4_t20), default max_neurons/patience. Usage: run_fold.sh <fold>
cd "$(dirname "$0")"; PY=/home/ghabr/micromamba/envs/mestrado_ghabriel_pei/bin/python; f=$1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
cmd="nice -n 17 $PY logs/rerun_2026-10/with_blas_threads.py 20 run_fold_record.py pima__binary_ 4 $f fold$f.pkl"
echo "start=$(date --iso-8601=seconds) commit=$(cat COMMIT) cmd=$cmd" > fold$f.info
$cmd > fold$f.log 2>&1
echo "exit=$? end=$(date --iso-8601=seconds)" > fold$f.exit
