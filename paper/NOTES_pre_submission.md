# Pre-submission notes (2026-09-29) — to address when closing the next complete version

Decision (Ghabriel, 2026-09-29): wait for the experiment campaigns; address everything below when closing the next
"initial final version with nothing missing". Nothing here has been implemented yet.

## 1. Chances at Neural Networks (judgement, not statistics)

* Journal: ~1,200 papers/year, Q1 (41/210 in AI), no published acceptance rate; median 9 days to first decision (many
  desk rejections), ~78 days review, ~7 months submission→acceptance. In scope: NN published MonoKAN (certified monotone
  KAN, 2025); constructive networks have a history there. Suggested section: *Learning Systems*.
* Current estimate: ~30–40 % acceptance at NN on first submission; ~50 % with the fixes of §3.
* Scenarios after the cegis campaign:

| Scenario | Desk rejection | Accept after review | Overall |
|---|---|---|---|
| A: cegis ≈ free MCC, fallback < 10 %, ≥ CMNN/LMN, within ~1 pt of published SOTA | ~15 % | ~55–60 % | ~45–50 % |
| B: moderate MCC cost, or fallback 20–40 %, or clearly below SOTA | ~25 % | ~35–40 % | ~25–30 % |
| C: high cost or fallback dominant | ~35 % | < 20 % | < 15 % |

* Fallback journals: Neurocomputing → EAAI / Knowledge-Based Systems (Q1 very likely with current material).

## 2. Risks (most to least serious)

1. Perceived relevance: "one hidden layer in 2026"; SLSQP cubic cost, tens of units, Loan subsampled. Position the paper:
   small/medium tabular data, exact guarantee, cost, parity with CMNN/LMN.
2. Missing SOTA: **Sartor, Sinigaglia & Susto, ICML 2025** ("Advancing Constrained Monotonic Neural Networks: Achieving
   Universal Approximation Beyond Bounded Activations", arXiv 2505.02537): COMPAS 69.5 %, Heart Disease 94 %, Loan
   65.4 %. Must enter related work and the literature table.
3. Results pending: the story depends on cegis MCC vs free, fallback rate, distance to CMNN/LMN, and cost on BCD/Loan.
4. CEGIS novelty vs COMET (counterexamples as training data) and Certified MNN (MILP + penalty): state the differences
   (hard constraints, no penalty weight, exact pre-activation certificate, guaranteed output).
5. Theorem 1 may be called "elementary": stress the question it answers (sign-constrained one-layer nets are not
   universal; Mikulincer 2025 needs 4 layers for threshold nets; certified one-layer sigmoid nets with margin are).
6. Certificate rigour: "exact up to rounding" → check certified networks with rigorous interval arithmetic
   (e.g. mpmath.iv) or directed rounding.
7. Constructive angle may look incidental (WILCAR ≈ RIXM with L-BFGS): give numbers — anchors/rounds per size,
   certificate time vs n, amortised verification.
8. Description of the NCA constraint (see §4).

## 3. Actions, by return on effort

1. Cite Sartor et al. 2025 (related work + literature table).
2. New experiment: data scarcity / extrapolation — learning curves with 10–50 % of the training data on 2–3 real sets
   and the synthetic targets; show where the correct constraint helps (turns "guarantee at no cost" into "guarantee
   with a benefit").
3. Rigorous interval re-check of the certified networks (one sentence in the paper).
4. Numbers for the constructive argument (anchors, rounds, certificate time by size) — replaces the TODO in §7.
5. Positioning in intro and cover letter; Theorem 1 as answer to an open question.
6. Paragraph linking cegis to semi-infinite programming / exchange methods (Blankenship & Falk 1976; Hettich &
   Kortanek 1993) and to flexibility analysis in process design (Halemane & Grossmann 1983; Swaney & Grossmann 1985);
   bounding of sigmoid networks over boxes (Schweidtmann & Mitsos 2019). Verify bibliographic data/DOIs before adding.
7. One plain-language sentence at the start of the certificate section ("lower-bound the derivative on each box; if
   positive the box is proved; otherwise split"); optional figure of positive/negative σ' bumps and the worst case.
8. Caption of Fig. 1: say that on this target the average-gain constraint is inactive and coincides with panel (b).
9. Report MCC before/after repair (n* is selected with networks that did not yet go through repair); consider a small
   sensitivity to ε_p.

## 4. Open question: what exactly did the NCA paper constrain?

* Dissertation code (`src/*/constrained.py`): average over the training points of the finite-difference gain
  (δ = 0.1) with margin 0.05; the first WILCAR unit uses the gain at the mean input; SCR = sign of the average
  finite-difference gain on the test set.
* The published NCA text (Sá, Fontes & Embiruçu 2026, NCA 38:726, doi 10.1007/s00521-026-12462-9) was NOT checked
  (Springer page unreachable). If it defines the gain at the operating point (mean input) instead of the average,
  adjust: Introduction ("average finite-difference gain"), Section 4.3 (mean constraint = earlier work; SCR), Table 1
  row "mean (Sá et al., 2026)". The hierarchy argument (average/point ⊂ anchors ⊂ whole domain) is unchanged.
* Action: get the PDF from Ghabriel and check the equation of the constraint and the SCR definition.

## 5. Co-authors (reading guidance)

* Both advisors are chemical engineers (process control/identification of polymerisation reactors; Embiruçu: PhD
  COPPE/UFRJ, NLP/estimation/MPC; Fontes: PhD Unicamp, postdoc Waterloo in time-series pattern recognition, fuzzy
  clustering, NNs). They co-authored WILCAR (EAAI 2021, Evolutionary Intelligence) and NCA.
* Expected understanding: high for motivation, constructive nets, gain constraints, SLSQP, Figs 1–2; medium for the
  protocol/statistics, CEGIS (high with the flexibility-analysis analogy) and the branch-and-bound; low for the proof of
  Theorem 1 and the monotone-NN/verification literature. Nobody in the group will audit the proofs → ask an external
  reader (maths/CS) or an independent critical review of Appendix A and Section 4.6.
* Correct reading of Fig. 1 for them: (b) unconstrained (EI 2022) — and also what the NCA average-gain constraint
  gives on this target (inactive); (c) classical sign constraint on the weights (Archer & Wang; Daniels & Velikova),
  NOT the NCA method; (d) cegis.
* Honest claim: cegis is not more accurate than the NCA constraint in general; it adds a point-wise guarantee at about
  the same accuracy, and is more accurate than the classical method that gives the same guarantee. Possible exception:
  BCD, where the NCA margin ε_m = 0.05 on 27 inputs cost MCC.

## 6. Length (done in v6)

Journal layout (`paper/make_journal.sh`): 17 pages incl. references; submission PDF 35; Word 42; Supplementary 7.
NN has no page limit for full articles (abstract ≤ 250 words).
