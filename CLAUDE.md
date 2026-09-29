# CLAUDE.md — WILCAR / Neural Networks revision (handoff for experiment orchestration)

Reply to the user in **Portuguese**. The user (Ghabriel) is a data scientist and MSc graduate (PEI/UFBA) with strong
Python; do not over-explain basics. Target journal: *Neural Networks* (Elsevier), subscription route (no APC).

## Goal
Turn the binary-classification chapter removed from the dissertation (constructive SLFNs WILCAR / RIXM / ELM with
optional monotonicity constraints and Dynamic Cross-Validation) into a *Neural Networks* paper. Revision plan (Claude Doc,
partly superseded by this file): https://claude.ai/code/artifact/f94f43e2-5731-41fc-b076-02c9ffcbc102

## Repository
* Branch `revision/neural-networks`. `src/` = dissertation code, **do not modify**. All new work is in `revision/`
  (read `revision/README.md` first: every protocol change vs. the dissertation is tabulated there) and `tests_revision/`.
* `revision/sfnn.py` analytic SLFN (loss, gains = input Jacobian, gradient of gains) · `methods.py` constructive methods,
  constraint modes `mean|multi|global|cegis` · `certify.py` adversarial search + sound branch-and-bound certificate ·
  `mono_baselines.py` MINMAX / CMNN / LMN · `protocol.py` nested CV, baselines, metrics · `data.py` datasets ·
  `run.py` resumable runner · `analyze.py` statistics · `fetch_benchmarks.py` Zenodo download.
* Raw results: `results/revision/results_<tag>.jsonl` (git-ignored). After each finished tag run
  `python -m revision.analyze --tag <tag>` and commit `results/revision/<tag>/*.csv|txt|png` with `git add -f`.

## Environment (user's machine)
* WSL2 Ubuntu, repo at `~/projects/thesis`, micromamba env `mestrado_ghabriel_pei` (Python 3.14), 16 logical CPUs.
* Always: `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`; `--jobs 14`; long runs inside `tmux`
  (session `wilcar`) with `2>&1 | tee <tag>.log`. Windows sleep must be off and a WSL window kept open.
* First-time setup: `pip install --pre -r revision/requirements.txt` (torch CPU, monotonicnetworks, mononet),
  `python -m revision.fetch_benchmarks`, `python -m pytest -q tests_revision` (expect 10 passed).
* Check `python -m revision.fetch_benchmarks` output: heart_disease positive rate and column names must match
  `BENCHMARKS` in `data.py` (loader asserts the monotone columns exist).

## Experiment queue (priority order). Everything is resumable: rerunning a command skips finished tasks and retries errors.
| # | Tag | Command (prefix `python -m revision.run`) | Purpose |
|---|---|---|---|
| 0 | smoke | `--datasets heart_failure --reps 1 --dcv --fast --tag smoke` | sanity, ~3 min |
| 1 | pilot | `--datasets compas loan heart_disease --reps 1 --folds 1 --dcv --fast --tag pilot` | cost on the new benchmarks; decide reps for compas/loan |
| 2 | main | `--datasets algerian breast_original breast_diagnostic heart_failure pima heart_disease --dcv --reps 5 --tag main` then `--datasets compas loan --dcv --reps 2 --tag main` (same tag, fewer reps if pilot is slow) | all 16 methods, mean mode |
| 3 | cegis | `--all --methods WILCAR-C RIXM-C ELM-C --mode cegis --reps 5 --tag cegis` (compas/loan: `--reps 2`) | **candidate main method** |
| 4 | global | same as 3 with `--mode global --tag global` | certified-by-sign ablation |
| 5 | lit | `--lit --reps 5 --dcv --mode cegis --tag lit` and `--lit --reps 5 --methods WILCAR RIXM MLP LR-C XGB-C LGBM-C MINMAX CMNN LMN --tag lit` | official splits, compare with published accuracy (COMPAS ≈ 69 %, Heart ≈ 89 %, Loan ≈ 65 %, CMNN/LMN papers) |
| 6 | multi | same as 3 with `--mode multi --tag multi` | ablation |
| 7 | l2zero | `--methods WILCAR WILCAR-C RIXM RIXM-C --dcv --l2 0 --reps 5 --tag l2zero` | sensitivity of the pilot-chosen l2 |
| 8 | gd | `--methods WILCAR RIXM --dcv --optimizer gd --l2 0 --reps 5 --tag gd` | dissertation optimiser (explains WILCAR > RIXM in the thesis) |
| 9 | zeroout | `--methods WILCAR WILCAR-C --dcv --zero-out --reps 5 --tag zeroout` | weight-reuse ablation |
Monitor: `wc -l results/revision/results_<tag>.jsonl`, `grep -c '"error"' ...`, `grep budget_hit` (selection loops stopped by
the 3600 s budget), `tail <tag>.log`. On errors read the `trace` field of the JSON line, fix, add a test, rerun.

## Key facts established so far (fast-config pilots, in `revision/README.md`)
* With L-BFGS + L2 prior (λ₀ = 0.3, penalty λ₀/N) the dissertation's WILCAR > RIXM advantage disappears (it was a GD effect) → WILCAR's argument must be
  cost/warm start, robustness, and the constrained/certified setting.
* `mean` mode leaves point-wise violations; `cegis` gives 100 % certified models at no MCC cost on Heart Failure / Pima /
  Breast Diagnostic pilots and beats CMNN / LMN / MINMAX on Heart Failure (0.596 vs 0.569 / 0.538 / 0.555).
* ELM(-C) is weak in every mode; keep it as a reference, not as a contender.
* COMPAS (1 fold, full config): SLFNs ≈ LR level (acc 0.663–0.668), CMNN 0.670, LMN 0.679, XGB 0.681; every task < 2 min.
  COMPAS is nearly linear; literature SOTA ≈ 0.69.

## Open decisions (ask the user)
1. Main method = `cegis` (expected) vs `global`; `mean` becomes "dissertation constraint" ablation.
2. Reps for compas/loan after the pilot (2 or 5); Loan is subsampled to 20 000 rows (declared in the paper).
3. Paper framing: "Constructive single-hidden-layer networks with certified partial monotonicity" (WILCAR + CEGIS + DCV).

## Analysis to produce when runs finish
* Table: MCC / AUC / Brier mean ± sd per dataset × method (main + cegis + global), certified %, viol_domain, n*, time.
* Friedman + Iman–Davenport + Nemenyi CD diagram over the 8 datasets; Holm vs best; Wilcoxon constrained vs free pairs.
* Cost: time_s and n* of WILCAR vs RIXM (warm start), selection curves (`curve_len`), DCV vs holdout.
* Literature table from `lit` next to published numbers of CMNN / LMN / Certified MNN / COMET.
