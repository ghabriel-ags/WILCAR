#!/usr/bin/env bash
# K-fold jobs with 20 BLAS threads that were stopped on 27/10 to reduce CPU contention, relaunched one at a time
# (labels <label>_r2; same commands). Priority: DCV HF M2 > Algerian K-fold > Pima M4 > this queue, so the queue
# runs under `nice -n 4` (launch.sh adds 15: nice 19). Concurrency changes only wall-clock times, not results.
# Usage: nohup setsid nice -n 4 logs/rerun_2026-10/queue_t20.sh > logs/rerun_2026-10/queue_t20.log 2>&1 &
cd "$(dirname "$0")/../.."
L=logs/rerun_2026-10
for label in cv_pima_m3_t20_r2 cv_hf_m12_t20_r2 cv_hf_m3_t20_r2 cv_pima_m12_t20_r2; do
  bash $L/launch.sh $label
  until [ -f $L/$label.exit ]; do sleep 60; done
  echo "$(date --iso-8601=seconds) $label $(cat $L/$label.exit)"
done
echo "$(date --iso-8601=seconds) queue done"
