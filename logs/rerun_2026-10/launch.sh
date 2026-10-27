#!/usr/bin/env bash
# Re-runs for the dissertation (October 2026). Usage: logs/rerun_2026-10/launch.sh <label>...
#
# Part 1 (CSV fix, commit 42b31cb or later): Heart Failure and Pima K-fold (6 methods) and DCV of WILCAR (M1),
# Algerian K-fold of WILCAR-C (M2), whose fold 1 failed in March.
# Same configuration as March 2026: code defaults, except HF RIXM-C (M4), which ran in March with
# --max-neurons 5 --patience 5 (cv_20260330_090253) and therefore runs alone with those overrides.
# Pima M3 and M4 run as separate processes (each model reseeds from seed + 100*fold + method, so the split
# does not change results; it only shortens the wall-clock time).
#
# BLAS threads are left at their defaults on purpose: the March runs did not set them, and with
# OMP/OPENBLAS_NUM_THREADS=1 (or 2) OpenBLAS takes other code paths and the results change; with the
# defaults, folds re-run on the pre-fix CSVs are bit-for-bit identical to March.
#
# Each job writes <label>.exit with its exit code and end time; processes.tsv records the launches.
set -u
cd "$(dirname "$0")/../.."
L=logs/rerun_2026-10
PY=/home/ghabr/micromamba/envs/mestrado_ghabriel_pei/bin/python
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS

launch() {  # label log command...
  local label=$1 log=$2; shift 2
  if grep -q "^$label	" "$L/processes.tsv" 2>/dev/null; then echo "skip $label (already launched)"; return; fi
  nohup bash -c '"$@" > "$0" 2>&1; echo "exit=$? end=$(date --iso-8601=seconds)" > "'"$L/$label"'.exit"' \
    "$L/$log" "$@" > /dev/null 2>&1 &
  printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$!" "$(date --iso-8601=seconds)" "$L/$log" "$*" >> "$L/processes.tsv"
  echo "launched $label (pid $!)"
  sleep 3   # output folders are timestamped to the second
}

HF=heart_failure_clinical_records_dataset__binary_
PI=pima__binary_
AL=Algerian_forest_fires__binary_
BO=breast__binary_
BD=Diagnostic_Breast_Cancer__binary_
CV="$PY run_cross_validation.py --task classification -y"
DCV="$PY src/run_dynamic_cv.py --task classification -y"

for label in "$@"; do
  case $label in
    cv_pima_m3)  launch $label "cv_${PI}_m3.log"  $CV --dataset $PI --methods 3 ;;
    cv_pima_m4)  launch $label "cv_${PI}_m4.log"  $CV --dataset $PI --methods 4 ;;
    cv_pima_m12) launch $label "cv_${PI}_m12.log" $CV --dataset $PI --methods 1 2 ;;
    cv_pima_m56) launch $label "cv_${PI}_m56.log" $CV --dataset $PI --methods 5 6 ;;
    cv_hf_m12)   launch $label "cv_${HF}_m12.log" $CV --dataset $HF --methods 1 2 ;;
    cv_hf_m3)    launch $label "cv_${HF}_m3.log"  $CV --dataset $HF --methods 3 ;;
    cv_hf_m4)    launch $label "cv_${HF}_m4.log"  $CV --dataset $HF --methods 4 --max-neurons 5 --patience 5 ;;
    cv_hf_m56)   launch $label "cv_${HF}_m56.log" $CV --dataset $HF --methods 5 6 ;;
    cv_alg_m2)   launch $label "cv_${AL}_m2.log"  $CV --dataset $AL --methods 2 ;;
    dcv_hf_m1)   launch $label "dcv_${HF}_m1.log"   $DCV --dataset $HF --methods 1 --retrain ;;
    dcv_pima_m1) launch $label "dcv_${PI}_m1.log"   $DCV --dataset $PI --methods 1 --retrain ;;
    # Part 2: DCV of WILCAR-C (M2) with weight reuse restored in the parallel worker; same flags as March
    # (classification with --retrain; BCD ran in March with --max-neurons 5, vcd_20260328_200827;
    # the 11 regression datasets in one process without --retrain, as vcd_20260310_140932)
    dcvm2_alg)   launch $label "dcvm2_${AL}.log"    $DCV --dataset $AL --methods 2 --retrain ;;
    dcvm2_bco)   launch $label "dcvm2_${BO}.log"    $DCV --dataset $BO --methods 2 --retrain ;;
    dcvm2_bcd)   launch $label "dcvm2_${BD}.log"    $DCV --dataset $BD --methods 2 --retrain --max-neurons 5 ;;
    dcvm2_hf)    launch $label "dcvm2_${HF}.log"    $DCV --dataset $HF --methods 2 --retrain ;;
    dcvm2_pima)  launch $label "dcvm2_${PI}.log"    $DCV --dataset $PI --methods 2 --retrain ;;
    dcvm2_reg)   launch $label "dcvm2_regression.log" $PY src/run_dynamic_cv.py --task regression --methods 2 -y ;;
    *) echo "unknown job $label"; exit 1 ;;
  esac
done
