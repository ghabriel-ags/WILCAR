#!/usr/bin/env bash
# Verification with the March BLAS thread counts: 20 threads in the main process (WSL then exposed 20 CPUs) and,
# in the DCV, 4 per worker. Each job runs the official command in a git-archive copy of the commit of the official
# run, through logs/rerun_2026-10/with_blas_threads.py, and is compared bit for bit with the official run.
Q=$(dirname "$(readlink -f "$0")"); PY=/home/ghabr/micromamba/envs/mestrado_ghabriel_pei/bin/python
W=/home/ghabr/projects/WILCAR-dissertacao; WRAP=$W/logs/rerun_2026-10/with_blas_threads.py
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
job() {  # label commit official_dir dcv(0/1) script args...
  local l=$1 c=$2 off=$3 dcv=$4; shift 4
  mkdir "$Q/$l" && git -C $W archive $c | tar -x -C "$Q/$l"
  if [ $dcv = 1 ]; then
    (cd "$Q/$l" && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10 $PY $WRAP 20 "$@" > "$Q/$l.log" 2>&1)
  else
    (cd "$Q/$l" && nice -n 10 $PY $WRAP 20 "$@" > "$Q/$l.log" 2>&1)
  fi
  echo "$l exit=$? $(date --iso-8601=seconds)" >> "$Q/done.txt"
  local cmp=compare_kfold_runs.py; [ $dcv = 1 ] && cmp=../t4q/compare.py
  [ -n "$off" ] && $PY "$Q/$cmp" "$W/$off" "$Q/$l"/results/*/[cv]*_* >> "$Q/compare.txt" 2>&1
}
C="--task classification -y --dataset"; HF=heart_failure_clinical_records_dataset__binary_; R=results/cross_validation
job cv_pima_m4 42b31cb $R/cv_20261026_113244 0 run_cross_validation.py $C pima__binary_ --methods 4 &
job cv_pima_m3 42b31cb $R/cv_20261026_113247 0 run_cross_validation.py $C pima__binary_ --methods 3 &
job cv_pima_m12 42b31cb $R/cv_20261026_113250 0 run_cross_validation.py $C pima__binary_ --methods 1 2 &
job cv_hf_m12 42b31cb $R/cv_20261026_113253 0 run_cross_validation.py $C $HF --methods 1 2 &
job cv_hf_m3 42b31cb $R/cv_20261026_113255 0 run_cross_validation.py $C $HF --methods 3 &
job cv_hf_m4 42b31cb $R/cv_20261026_113302 0 run_cross_validation.py $C $HF --methods 4 --max-neurons 5 --patience 5 &
job cv_pima_m56 42b31cb $R/cv_20261026_113305 0 run_cross_validation.py $C pima__binary_ --methods 5 6 &
job cv_hf_m56 42b31cb $R/cv_20261026_113308 0 run_cross_validation.py $C $HF --methods 5 6 &
D=results/dynamic_cv; DCV=src/run_dynamic_cv.py
(
job dcv_hf_m2_retrain a30890a "" 1 $DCV $C $HF --methods 2 --retrain --max-neurons 1
job dcv_hf_m1 42b31cb $D/vcd_20261026_113311 1 $DCV $C $HF --methods 1 --retrain
job dcvm2_alg a30890a $D/vcd_20261027_004830_alg 1 $DCV $C Algerian_forest_fires__binary_ --methods 2 --retrain
job dcvm2_bco a30890a $D/vcd_20261027_011812_bco 1 $DCV $C breast__binary_ --methods 2 --retrain
job dcvm2_pima a30890a $D/vcd_20261027_005421_pima 1 $DCV $C pima__binary_ --methods 2 --retrain
job dcv_pima_m1 42b31cb $D/vcd_20261026_114430 1 $DCV $C pima__binary_ --methods 1 --retrain
job dcvm2_bcd a30890a $D/vcd_20261027_004851_bcd 1 $DCV $C Diagnostic_Breast_Cancer__binary_ --methods 2 --retrain --max-neurons 5
) &
wait
echo ALLDONE >> "$Q/done.txt"
