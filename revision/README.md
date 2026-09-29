# Revision for *Neural Networks* — nested protocol, certified monotonicity, literature baselines

This package re-implements the six constructive methods for **binary classification** under the
protocol planned for the *Neural Networks* submission. The dissertation code in `src/` is untouched.

## What changed relative to `src/`

| Topic | `src/` (dissertation) | `revision/` |
| --- | --- | --- |
| Model selection | n\* = best **test** MCC; DCV selects on test folds | n\* on a validation split (or inner folds, DCV) **inside the outer training set**; outer test used once |
| Reinitialisation check (SCR = 100 %) | on the **test** set | on training anchors only |
| Scaling | MinMaxScaler fitted on the whole dataset | fitted on the outer training set |
| Gains in SLSQP | finite differences, numerical Jacobian | closed-form gains and closed-form gradient (`sfnn.gains_and_grad`) |
| Constraint | mean gain over the training data ≥ 0.05 | `--mode mean` (same; ≤ 1000 random anchors), `multi` (gain ≥ 1e-3 at 10 k-means anchors), `global` (s_i·W2_j·W1_ji ≥ 0: certified, restrictive), **`cegis`** (counterexample-guided anchors + branch-and-bound certificate: certified, flexible) |
| Certificate | — | every SLFN (constrained or not) is checked by a sound branch-and-bound on [0,1]^p (`certify.py`): `certified` / `counterexample` / `unknown` |
| Optimiser (unconstrained) | full-batch gradient descent, lr 0.1 | L-BFGS on the analytic gradient (same quasi-Newton family as SLSQP); `--optimizer gd` restores the dissertation |
| Regularisation | none (implicit, early stopping of GD) | Gaussian prior on W1, W2 (MAP): mean BCE + (λ₀/N)‖W‖², λ₀ = 0.3 (= 1e-3 at N = 300), in **both** constrained and unconstrained objectives (`--l2`) |
| Constructive budget | max 750 neurons, patience 150 | max 300, patience 30 (unconstrained); 50 / 15 (constrained); 3600 s per selection loop (`--time-budget`) |
| ELM | zero hidden biases, no output bias | random hidden biases U(-1,1) + output bias (standard ELM); `--elm-no-bias` restores the dissertation |
| ELM-C output | BCE on a clipped linear output | sigmoid output + BCE (logistic ELM); `global`/`cegis`: sign-aligned hidden weights, β ≥ 0 (monotone ELM) |
| New neuron (WILCAR) | output weight ~ N(0, 1/n) | same by default; `--zero-out` starts it at 0 (n+1 network = n network) |
| Evaluation | accuracy, F1, MCC, SCR at the mean | + AUC, Brier, violation rate on test points and on 10 000 uniform points, certificate status |
| Replicates | 5-fold, seed 0 | 5-fold × 5 repetitions (25 outer folds), Friedman / Holm / Wilcoxon |
| Datasets | 5 (Appendix B) | + COMPAS, Heart Disease, Loan Defaulter (monotone-NN benchmarks, Zenodo 7968969) |

Baselines (same outer folds, hyper-parameters on the same validation split):
* free / monotone classical: `LR`, `LR-C` (sign-bounded coefficients), `XGB(-C)`, `LGBM(-C)` (`monotone_constraints`), `MLP`;
* monotone neural networks (all certified by construction): `MINMAX` (Sill, 1997), `CMNN` (Runje & Shankaranarayana, ICML 2023,
  authors' `mononet`), `LMN` (Nolte et al., ICLR 2023, authors' `monotonicnetworks`).

## Changes of 2026-09-29 (affect every constrained SLFN: WILCAR-C, RIXM-C in all modes)

* **Initialisation of the constrained problems** (`Cfg.constrained_init="free"`): SLSQP starts from the unconstrained L-BFGS
  optimum (from the method's usual initialisation) instead of the raw initialisation. From a random, infeasible start SLSQP
  often stopped in poor local minima (e.g. anchor mode on a synthetic target: error 0.107 vs 0.040).
* **Feasibility tolerance** `feas_tol=1e-6` (was 1e-8, tighter than SLSQP's own tolerance): feasible solutions were declared
  infeasible and re-initialised, wasting restarts and breaking the cegis loop.
* **Certificate**: gradient enclosure + monotonicity test of interval global optimisation (Hansen & Walster) in the input-space
  branch and bound; cegis repair rounds 20 and a **sign-constrained fallback**, so every cegis network is certified
  (`cert_fallback` records when the fallback was used).
* **Synthetic ground-truth datasets** `syn_*` (`--syn`), metric `mae_true` against the known probability.

Results of constrained methods computed before these changes must be recomputed:
`python -m revision.purge --tag <tag> --methods WILCAR-C RIXM-C ELM-C`, then rerun the same command.

## The `cegis` mode (candidate main contribution)

For a sigmoid SLFN, `s_i ∂p/∂x_i = σ'(z₂) · h_i(x)` with `h_i(x) = Σ_j c_ij σ'(w_j·x + b_j)`, `c_ij = s_i W2_j W1_ji`.
1. SLSQP with point-wise gain constraints at the anchors (k-means centres, then counterexamples);
2. adversarial search: multi-start L-BFGS-B of `h_i` on [0,1]^p (analytic gradient); violators become anchors; repeat;
3. after refit, sound **branch-and-bound**: in x-space with natural-interval and mean-value bounds, or, when n < p, in
   pre-activation space where `h` is separable (exact box minimum) and an LP discards boxes outside the zonotope
   `{W1 x + b}`; counterexamples are fed back (repair) until certified.

Unlike `global`, output and hidden weights keep free signs (individual hidden units may be non-monotone as long as the
combination is), so the constraint costs little accuracy. Pilot (Heart Failure, fast config): WILCAR-C (DCV)
MCC 0.596 with 5/5 folds certified, vs CMNN 0.569, LMN 0.538, MINMAX 0.555, unconstrained WILCAR 0.58–0.59.

## Running

```bash
pip install -r requirements.txt && pip install --pre -r revision/requirements.txt
python -m revision.fetch_benchmarks                      # COMPAS / Heart Disease / Loan into data/benchmarks
python -m pytest -q tests_revision                       # 10 tests: gradients, certifier soundness, modes

# smoke test (minutes)
python -m revision.run --datasets heart_failure --reps 1 --dcv --fast --tag smoke --jobs 14

# campaign; resumable (every task appends one JSON line; rerunning the same command resumes)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m revision.run --all --reps 5 --jobs 14 --tag main          # all datasets, all methods, mean mode + DCV
python -m revision.run --all --methods WILCAR-C RIXM-C ELM-C --mode cegis  --reps 5 --jobs 14 --tag cegis
python -m revision.run --all --methods WILCAR-C RIXM-C ELM-C --mode global --reps 5 --jobs 14 --tag global
python -m revision.run --all --methods WILCAR-C RIXM-C ELM-C --mode multi  --reps 5 --jobs 14 --tag multi
python -m revision.run --methods WILCAR WILCAR-C --dcv --zero-out --reps 5 --jobs 14 --tag zeroout
python -m revision.run --methods WILCAR WILCAR-C RIXM RIXM-C --dcv --l2 0 --reps 5 --jobs 14 --tag l2zero
python -m revision.run --methods WILCAR RIXM --dcv --optimizer gd --l2 0 --reps 5 --jobs 14 --tag gd
# literature protocol: official train/test split of the benchmarks, 5 seeds (compare with published accuracy)
python -m revision.run --lit --reps 5 --dcv --mode cegis --jobs 14 --tag lit

python -m revision.analyze --tag main      # summary, certification, ranks, Friedman/Holm, Wilcoxon, CD diagram
```
See `CLAUDE.md` at the repository root for the experiment queue, decisions and open questions.

## Notes for the manuscript

* **Optimiser pilot (fast config, 4 datasets × 10 folds, WILCAR / RIXM):** mean MCC with GD 0.724 / 0.407, with L-BFGS
  0.703 / 0.698 (l2 = 0) and 0.729 / 0.731 (l2 = 1e-3); l2 = 1e-2 collapses to the majority class. Under GD, RIXM
  (random restart at every size) is badly under-trained, while WILCAR's weight reuse accumulates optimisation steps
  across sizes. **The dissertation's WILCAR > RIXM gap is therefore largely an optimiser effect.** With a proper
  optimiser, WILCAR's case must rest on (i) cost per size (warm start), (ii) robustness to a weak optimiser, (iii) the
  constrained/certified setting. The `gd` ablation documents this honestly. The penalty was fixed from this pilot
  (reps 0–1) and then expressed as a prior precision λ₀ = 0.3, i.e. λ₀/N on the mean loss (a fixed 1e-3 over-regularised
  COMPAS, N ≈ 4 000: accuracy 0.649 vs 0.663–0.668 with λ₀/N; small datasets unchanged, mean MCC 0.72 vs 0.73).
  Reported with the `l2zero` sensitivity run.
* `mean` mode constrains the **average** gain; point-wise violations remain possible and are detected by the certifier
  (`cert_status = counterexample`). Present `cegis` (or `global`) as the method behind any "monotone" claim.
* `scr_legacy` is the dissertation SCR (sign of the average finite-difference gain on the test set, δ = 0.1),
  kept only for continuity with Chapters 5–6.
* The certificate holds on [0,1]^p, the training box after min-max scaling; test points outside it are extrapolation.
* Loan Defaulter is stratified-subsampled to 20 000 rows for cross-validation (constructive SLSQP cost); with `--lit`
  the official test set is kept whole and the training set is subsampled to 20 000.

## Additions of 2026-09-29 (afternoon)
* `--starts k`: every constructive step of every SLFN (free or constrained) is repeated from k initialisations and the
  feasible result with the lowest training objective is kept. Used with k = 3 on the synthetic targets, where single
  starts often land in a poor basin (e.g. MAE 0.111 instead of 0.023 on "or") for free and constrained networks alike;
  real-data runs keep k = 1.
* `python -m revision.cert_bench`: ablation of the certificate (interval / + mean value / + monotonicity test /
  + pre-activation space) on 168 trained SLFNs; soundness cross-checked by dense sampling. Pilot: cegis-step networks
  decided in 75 / 86 / 89 / 96 % of the cases; 20 % of the mean-constrained networks have counterexamples.

