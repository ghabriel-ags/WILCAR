#!/usr/bin/env bash
R=$(dirname "$0"); PY=/home/ghabr/micromamba/envs/mestrado_ghabriel_pei/bin/python
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
CMD="src/run_dynamic_cv.py --task classification -y --dataset Diagnostic_Breast_Cancer__binary_ --methods 2 --max-neurons 1"
v() { local d=$1; shift; (cd $R/$d && env "$@" nice -n 10 $PY $CMD $EXTRA > ../$d.log 2>&1); }
v t2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 & v t4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 & wait
v t16 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 & EXTRA=--no-parallel v seq X=1 & wait
for d in def t1 t2 t4 t16 seq; do echo "$d: $(sed -n 2p $R/$d/results/dynamic_cv/*/Diagnostic_Breast_Cancer__binary_/method_2/vcd_history.csv | cut -d, -f1-2)"; done
echo "march: 1,1.1996397449136573"
