# Revision for *Neural Networks* — nested protocol, analytic constraints, baselines

This package re-implements the six constructive methods for **binary classification** under the
protocol planned for the *Neural Networks* submission. The dissertation code in `src/` is untouched.

## What changed relative to `src/`

| Topic | `src/` (dissertation) | `revision/` |
| --- | --- | --- |
| Model selection | n\* = best **test** MCC; DCV selects on test folds | n\* on a validation split (or inner folds, DCV) **inside the outer training set**; outer test used once |
| Reinitialisation check (SCR = 100 %) | on the **test** set | on training anchors only |
| Scaling | MinMaxScaler fitted on the whole dataset | fitted on the outer training set |
| Gains in SLSQP | finite differences, numerical Jacobian | closed-form gains and closed-form gradient (`sfnn.gains_and_grad`) |
| Constraint | mean gain over the training data ≥ 0.05 | `--mode mean` (same), `multi` (gain ≥ 1e-3 at 10 k-means anchors), `global` (s_i·W2_j·W1_ji ≥ 0 for all j: **certified** monotonicity) |
| ELM-C output | BCE on a clipped linear output | sigmoid output + BCE (logistic ELM) |
| New neuron (WILCAR) | output weight ~ N(0, 1/n) | same by default; `--zero-out` starts it at 0 (n+1 network = n network) |
| Evaluation | accuracy, F1, MCC, SCR at the mean | + AUC, Brier, violation rate of the expected signs on test points and on 10 000 uniform points of the domain |
| Replicates | 5-fold, seed 0 | 5-fold × 5 repetitions (25 outer folds), Friedman / Holm / Wilcoxon |

Baselines (same outer folds, hyper-parameters on the same validation split): logistic regression free / with
sign-bounded coefficients (`LR`, `LR-C`), XGBoost and LightGBM free / with `monotone_constraints`, MLP (sklearn).

## Running

```bash
pip install -r requirements.txt xgboost lightgbm joblib
python -m pytest -q tests_revision                       # gradient checks + certified-monotonicity test

# smoke test (minutes)
python -m revision.run --datasets heart_failure --reps 1 --dcv --fast --tag smoke --jobs 4

# main campaign (5 datasets × 13 methods × 25 folds + DCV); resumable
python -m revision.run --all --reps 5 --jobs 18 --tag main
# constraint-mode ablations (constrained methods only)
python -m revision.run --methods WILCAR-C RIXM-C ELM-C --dcv --mode multi  --reps 5 --jobs 18 --tag multi
python -m revision.run --methods WILCAR-C RIXM-C ELM-C --dcv --mode global --reps 5 --jobs 18 --tag global
# weight-reuse / zero-init ablation
python -m revision.run --methods WILCAR WILCAR-C --dcv --zero-out --reps 5 --jobs 18 --tag zeroout

python -m revision.analyze --tag main                    # summary, ranks, Friedman/Holm, Wilcoxon, CD diagram
```

Every task appends one JSON line to `results/revision/results_<tag>.jsonl`; rerunning the same command skips
finished tasks, so an interrupted campaign can be resumed.

## Notes for the manuscript

* `mean` mode reproduces the dissertation constraint. It constrains the **average** gain, so point-wise
  violations remain possible; the smoke test on Heart Failure shows them (`viol_test` > 0). Report this honestly
  and present `global` (certified) and/or `multi` as the method that backs any "monotonicity" claim.
* `scr_legacy` is the dissertation SCR (sign of the average finite-difference gain on the test set, δ = 0.1),
  kept only for continuity with Chapters 5–6.
* In `global` mode ELM-C can only zero or sign-restrict output weights of fixed random units, so it degrades
  strongly; this is a structural limitation worth one sentence.
