#!/usr/bin/env bash
# Diagnostic: re-run the finished October DCVs with 4 BLAS threads per worker (March: 20 CPUs // 5 workers = 4)
# in scratch copies of HEAD, then compare with the official October runs (3 threads per worker).
Q=$(dirname "$(readlink -f "$0")"); PY=/home/ghabr/micromamba/envs/mestrado_ghabriel_pei/bin/python
W=/home/ghabr/projects/WILCAR-dissertacao
run() {  # label official_dir args...
  local l=$1 off=$2; shift 2
  mkdir "$Q/$l" && git -C $W archive HEAD | tar -x -C "$Q/$l"
  (cd "$Q/$l" && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10 $PY src/run_dynamic_cv.py "$@" > "$Q/$l.log" 2>&1)
  echo "$l exit=$? $(date +%T)" >> "$Q/done.txt"
  $PY "$Q/compare.py" "$W/$off" "$Q/$l"/results/dynamic_cv/vcd_* >> "$Q/compare.txt" 2>&1
}
C="--task classification -y --dataset"
run alg results/dynamic_cv/vcd_20261026_120731 $C Algerian_forest_fires__binary_ --methods 2 --retrain &
run hf_m1 results/dynamic_cv/vcd_20261026_113311 $C heart_failure_clinical_records_dataset__binary_ --methods 1 --retrain &
wait
run pima_m1 results/dynamic_cv/vcd_20261026_114430 $C pima__binary_ --methods 1 --retrain &
run pima results/dynamic_cv/vcd_20261026_122207 $C pima__binary_ --methods 2 --retrain &
wait
run hf results/dynamic_cv/vcd_20261026_122110 $C heart_failure_clinical_records_dataset__binary_ --methods 2 --retrain &
run bco results/dynamic_cv/vcd_20261026_122030 $C breast__binary_ --methods 2 --retrain &
wait
run reg results/dynamic_cv/vcd_20261026_122026 --task regression --methods 2 -y
echo ALLDONE >> "$Q/done.txt"
