# Prior knowledge on gain signs / monotonicity in data-driven process models (PSE, chemical engineering, industrial control)

Scope: how gain-sign / monotonicity / qualitative prior knowledge has been imposed on neural or grey-box models in
process engineering (identification, soft sensors, MPC models), and how "guarantees" were or were not obtained.
Purpose: position the WILCAR/CEGIS binary-classification paper (certified gain signs on the whole input domain) in a
literature that process-engineering co-authors and readers recognise, as a continuation of the authors' NCA 2026 paper
(gain signs as hard constraints at a reference operating point, via finite differences, on constructive SLFNs).

Note on sources: all DOIs below were checked against the publisher page, Semantic Scholar or OpenAlex metadata
unless marked "(verify)". The Springer page of the authors' own NCA paper (doi 10.1007/s00521-026-12462-9) could not
be fetched in this session (proxy rate limit), so its description is taken from the task briefing only.

## Key question 1 — Works that constrain steady-state gains or their signs in neural / black-box process models

### Takeaway
Gain-sign knowledge has been used in process identification since the early 1990s, but in three distinct "guarantee
regimes": (i) linear grey-box models where the sign of a stationary gain is a parameter inequality and therefore holds
globally (Tulleken 1993); (ii) nonlinear ANN models where gains are constrained by penalties at (training or synthetic)
points, with no domain-wide guarantee (Hartman 2000 / Pavilion, Thompson & Kramer 1994, Johansen 1996, recent soft
sensors such as Cheng et al. 2024); and (iii) architectures that bound gains everywhere by construction, which is the
industrial-MPC answer (Turner & Guiver's bounded derivative network, 2005; Siemens' monotonic MLPs, Lang 2005). The
"verify-and-refine until a semi-infinite constraint holds" approach — the closest analogue of a CEGIS loop — appears in
manufacturing regression (Kurnatowski et al. 2021) rather than in the chemical-process literature.

### Cited Findings

**Early black-box and hybrid lineage (why gains of ANN process models became an issue)**
- Bhat & McAvoy (1990), *Computers & Chemical Engineering* 14(4–5):573–583, doi 10.1016/0098-1354(90)87028-N, is the
  reference first paper on back-propagation networks for dynamic modelling and control of chemical processes (pH CSTR);
  it is the root of the black-box ANN lineage that later works tried to constrain —
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/009813549087028N);
  [IEEE Xplore copy](https://ieeexplore.ieee.org/document/4790399/).
- Psichogios & Ungar (1992), "A hybrid neural network-first principles approach to process modeling", *AIChE Journal*
  38(10):1499–1511, doi 10.1002/aic.690381003: serial hybrid in which the ANN estimates an unmeasured parameter (e.g.
  growth rate) inside the first-principles balances of a fed-batch bioreactor; the hybrid interpolates and extrapolates
  better and needs less data than a pure ANN. It constrains the model by *structure*, not by gain inequalities —
  [Wiley](https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/aic.690381003);
  [ADS record](https://ui.adsabs.harvard.edu/abs/1992AIChE..38.1499P/abstract).
- Thompson & Kramer (1994), "Modeling chemical processes using prior knowledge and neural networks", *AIChE Journal*
  40(8):1328–1340, doi 10.1002/aic.690400806 (≈600 citations). Reconstructed abstract: prior knowledge enters as simple
  first-principles equations; mass and component balances enforce *equality* constraints; "in addition, inequality
  constraints are imposed during parameter estimation"; the network controls extrapolation in regions lacking data;
  application to fed-batch penicillin fermentation; prior knowledge yields less data needed, more consistent predictions
  and more reliable extrapolation — [OpenAlex metadata](https://api.openalex.org/works/doi:10.1002/aic.690400806);
  [Wiley](https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/aic.690400806). The inequality constraints are enforced
  at the parameter-estimation stage (on data), i.e. no statement of validity over the whole input space.

**Gain-sign knowledge in grey-box / semi-parametric identification (control community)**
- Tulleken (1993), "Grey-box modelling and identification using physical knowledge and Bayesian techniques",
  *Automatica* 29(2):285–308, doi 10.1016/0005-1098(93)90124-C. Physical knowledge — process *stability* and the *sign
  of stationary gains* — is translated into linear inequality constraints on the parameters of a black-box (linear)
  model; a Bayesian treatment with uniform or piecewise-linear priors on the constrained parameter set gives MAP and
  posterior-mean estimators with explicit solutions in the Gaussian case; tested on a distillation process, giving
  "considerable variance reductions at the cost of a somewhat larger bias", motivated by adaptive control —
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/000510989390124C);
  [dblp](https://dblp.org/rec/journals/automatica/Tulleken93.html). Because the model is linear in the inputs, the
  gain-sign constraint on parameters is automatically a global guarantee — the guarantee is free, which is exactly what
  is lost when the model becomes a nonlinear ANN.
- Johansen (1996), "Identification of non-linear systems using empirical data and prior knowledge—an optimization
  approach", *Automatica* 32(3):337–356, doi 10.1016/0005-1098(95)00146-8. Semi-parametric identification where
  the least-squares criterion is augmented with penalty terms (smoothness) and "useful types of prior knowledge other
  than smoothness" are added "as a term in the criterion or as a constraint"; the paper acknowledges that "complicated
  constraints or penalty terms" make the optimal solution hard to derive and proposes a practical numerical procedure;
  framework relates to RBFs, splines and neural networks —
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0005109895001468). (From the abstract alone it is
  not confirmed that gain-sign constraints are among the examples; the full text should be checked before citing it for
  that specific point.)

**Gain constraints on ANN models for industrial MPC (Pavilion / Aspen lineage)**
- Hartman (2000), "Training Feedforward Neural Networks with Gain Constraints", *Neural Computation* 12(4):811–829,
  doi 10.1162/089976600300015600. Abstract: "Inaccurate input-output gains (partial derivatives of outputs with respect
  to inputs) are common in neural network models when input variables are correlated or when data are incomplete or
  inaccurate"; "inequality or equality-bound constraints on the gains of the learned mapping" are "implemented as penalty
  terms added to the objective function, and training is done using gradient descent", with "adaptive and robust
  procedures" when constraints conflict with data, and "extrapolation training, which can dramatically improve
  generalization"; the algorithm "has been advantageously applied to dozens of models currently in commercial use" —
  [MIT Press](https://direct.mit.edu/neco/article/12/4/811/6352/Training-Feedforward-Neural-Networks-with-Gain);
  [ML Anthology record](https://mlanthology.org/neco/2000/hartman2000neco-training/). No domain-wide guarantee is
  claimed: the penalty acts at the points where it is evaluated. The method is patented by Pavilion Technologies,
  US 7,058,617 B1, "Method and apparatus for training a system model with gain constraints" —
  [Google Patents](https://patents.google.com/patent/US7058617B1).
- Qin & Badgwell (2003), "A survey of industrial model predictive control technology", *Control Engineering Practice*
  11(7):733–764, doi 10.1016/S0967-0661(02)00186-7 (verify DOI; volume/pages confirmed). On Pavilion's Process Perfecter:
  the process is decomposed into a steady-state part obeying a nonlinear static (ANN) model and a deviation part with a
  linear dynamic model; the static gain used in control is a linear interpolation of initial and final steady-state gains
  and "bounds on K_s^i and K_s^f can be applied"; "bounds are enforced on the model gains in order to improve the quality
  of the neural network for control applications". On Aspen Target: "the model derivatives fall to zero as the network
  extrapolates beyond the range of its training data set", handled by an on-line model confidence index that gradually
  turns the ANN term off and falls back to the linear model —
  [PDF (CMU mirror)](https://cepac.cheme.cmu.edu/pasilectures/darciodolak/Review_article_2.pdf). This is the standard
  reference for *why* gain signs/bounds matter in industrial nonlinear MPC (gain inversion, zero-gain saturation).
- Turner & Guiver (2005), "Introducing the bounded derivative network—superceding the application of neural networks in
  control", *Journal of Process Control* 15(4):407–415, doi 10.1016/j.jprocont.2004.08.001. The BDN is "the analytical
  integral of a neural network": its gains are themselves a (bounded-activation) network, so minimum and maximum gains on
  each input/output pair — and the "gain trajectory" — can be specified and hold *by construction* everywhere, removing
  saturation ("zero gain" regions) and "arbitrary gain sign reversals" of ordinary ANNs; constrained optimisation with
  process-knowledge constraints is used in training; demonstrated on a commercial polypropylene process (melt-flow index
  from hydrogen and catalyst flows, R² = 0.93) for multivariable control —
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0959152404000915). Companion conference paper:
  Turner, Guiver & Lines (2003), "Introducing the state space bounded derivative network for commercial transition
  control", American Control Conference —
  [IEEE Xplore](https://ieeexplore.ieee.org/iel5/8775/27842/01242587.pdf) (DOI not retrieved; verify).
- Lang (2005), "Monotonic multi-layer perceptron networks as universal approximators", ICANN 2005, LNCS 3697 (Springer),
  doi 10.1007/11550907_6 (Siemens): monotonic MLPs via sign-constrained weights, with a universal-approximation argument —
  [Springer](https://link.springer.com/chapter/10.1007/11550907_6). Industrial follow-ups by the same group include
  Minin, Velikova, Lang & Daniels (ICANN 2008, doi 10.1007/978-3-540-87559-8_62, and the *Neural Networks* 2010 version
  already in refs.bib) and a "monotonic recurrent bounded derivative neural network" (Minin & Lang) —
  [Semantic Scholar](https://www.semanticscholar.org/paper/Monotonic-Recurrent-Bounded-Derivative-Neural-Minin-Lang/7177d6ee637bd025ed18f493fa5281c8d8f4f3b3).

**Recent process-engineering works (2021–2025)**
- Cheng, Yu, Wang, Jiang & Cao (2024), "Semi-supervised soft sensor method for fermentation processes based on physical
  monotonicity and variational autoencoders", *Engineering Applications of Artificial Intelligence* 137(A):109065,
  doi 10.1016/j.engappai.2024.109065. Monotonicity between process variables (temperature, pH, pressure, flow) and
  penicillin titre is "incorporated into the loss function of VAEs for regression"; a soft penalty, so monotonicity is
  encouraged, not guaranteed; validated on simulated and industrial penicillin data against five baselines —
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197624012235).
- Mousa, Negny, Ouaret, Di Pretoro & Montastruc (2025), "Incorporating physical constraints inside neural networks to
  improve their accuracy and physical reliability for chemical engineering unit operations modeling", *Computers &
  Chemical Engineering* 199:109156, doi 10.1016/j.compchemeng.2025.109156. Compares soft constraints (penalty residuals)
  with a hard approach that couples the ANN with data reconciliation to "exactly impose" conservation constraints; flash
  drum and distillation (toluene–biphenyl, NRTL) case studies; the constraints are balance equalities, not gain signs,
  and the abstract does not discuss validity over the input domain —
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0098135425001607).
- Kurnatowski, Schmid, Link, Zache, Morand, Kraft, Schmidt & Stoll (2021), "Compensating Data Shortages in
  Manufacturing with Monotonicity Knowledge", *Algorithms* 14(12):345, doi 10.3390/a14120345 (Fraunhofer ITWM/IWU/IWM).
  Monotonic regression is posed as a *semi-infinite* optimisation problem with constraints σ_j ∂_{x_j} ŷ_w(x) ≥ 0 for all
  x in the domain; an adaptive discretisation algorithm iteratively refines the finite constraint set "until monotonicity
  is satisfied"; compared with projection, rearrangement and tilting monotonisation and with constrained SVR / ANN
  training; the models are smooth ("no after-training monotonization step"); applications: laser glass bending and
  press hardening of sheet metal; lower RMSE than the alternatives on sparse data —
  [MDPI](https://www.mdpi.com/1999-4893/14/12/345); [arXiv/ar5iv](https://ar5iv.labs.arxiv.org/html/2010.15955).
  Related theory: Schmid, "Approximation, characterization, and continuity of multivariate monotonic regression
  functions", *Analysis and Applications*, doi 10.1142/S0219530521500299 —
  [World Scientific](https://www.worldscientific.com/doi/10.1142/S0219530521500299); and Link et al., "Capturing and
  incorporating expert knowledge into machine learning models for quality prediction in manufacturing", *Journal of
  Intelligent Manufacturing*, doi 10.1007/s10845-022-01975-4 —
  [Springer](https://link.springer.com/article/10.1007/s10845-022-01975-4).
- Sharma & Liu (2022), "A hybrid science-guided machine learning approach for modeling chemical processes: A review",
  *AIChE Journal* 68(5):e17609, doi 10.1002/aic.17609: organises hybrid modelling into ML-enhancing-science and
  science-guided ML, the latter split into "science-guided design, learning and refinement" —
  [Wiley](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.17609); [arXiv preprint](https://arxiv.org/abs/2112.01475).
  Whether it treats monotonicity/gain-sign constraints explicitly was not confirmed from the abstract.

### Inferences
- The chemical-process literature contains the *motivation* (gain inversion and zero-gain extrapolation of ANN models are
  unacceptable in MPC — Qin & Badgwell 2003; Turner & Guiver 2005) and two incomplete answers: penalty-at-points
  (Hartman 2000; Thompson & Kramer 1994; Cheng et al. 2024), which is the regime of the authors' NCA 2026 paper
  (finite-difference gain sign at a reference operating point), and architecture-by-construction (BDN, monotonic MLPs
  with sign-constrained weights), which guarantees the sign globally but restricts the hypothesis class.
- A third regime — train a free architecture, then *certify* the sign over the whole box and repair with counter-
  examples — has no direct precedent in Comput. Chem. Eng. / J. Process Control / AIChE J. as far as this search went;
  the nearest neighbours are Kurnatowski et al. 2021 (adaptive discretisation of a semi-infinite constraint, manufacturing)
  and the ML-side CEGIS works already in refs.bib (Sivaraman et al. 2020; Liu et al. 2020). This is a defensible novelty
  claim for a process-engineering audience, phrased as "closing the gap between Pavilion-style gain constraints and
  BDN-style global guarantees without changing the architecture".
- Tulleken 1993 is the right historical anchor for "sign of stationary gains as prior knowledge": in his linear setting
  the global guarantee is trivial, so the paper can frame the certified-sign problem as recovering, for nonlinear SLFNs,
  the property linear grey-box models had for free.

### Gaps
- No peer-reviewed PSE paper was found that *verifies* (formally, over a box) the gain signs of a trained ANN process
  model; absence of evidence after ~15 searches, not evidence of absence. A targeted search in J. Process Control /
  IECR for "monotonic" + "soft sensor" + "guarantee" could still be worth one more pass.
- Hartman 2000's full text (where exactly the penalty is evaluated — data points vs. sampled grid) was behind a 403; the
  abstract confirms penalties + gradient descent + "extrapolation training" but not the sampling scheme.
- Johansen 1996: not confirmed from the abstract that gain-sign/monotonicity is among the prior-knowledge examples.
- Lindskog & Ljung (semi-physical modelling), Joerding & Meador 1991 (*Neural Networks*, "Encoding prior information in
  feedforward networks"), Lampinen & Selonen 1997 and Hussain 1999 (review of ANN in process control) were not retrieved
  with verified DOIs in this session; cite only after checking.

## Key question 2 — Physics-guided / informed ML reviews: where monotonicity and shape constraints sit, and how guarantees are discussed

### Takeaway
The three canonical reviews classify monotonicity/shape knowledge as "algebraic (inequality) constraints" integrated
either into the hypothesis set (architecture), the learning algorithm (loss penalties) or checked on the final
hypothesis; none of them, at the abstract level retrieved here, offers or demands a formal domain-wide guarantee — the
guarantee question is left to the architecture/verification literature.

### Cited Findings
- von Rueden et al. (2023), "Informed Machine Learning – A Taxonomy and Survey of Integrating Prior Knowledge into
  Learning Systems", *IEEE Transactions on Knowledge and Data Engineering* 35(1):614–633 (online 2021),
  doi 10.1109/TKDE.2021.3079836. Three taxonomy axes: knowledge *source*, knowledge *representation* ("algebraic
  equations, logic rules, or simulation results", among others) and *integration* stage (training data, hypothesis set,
  learning algorithm, final hypothesis); informed ML is defined by the "additional integration of prior knowledge into
  the training process" for data-scarce settings; the abstract does not discuss formal guarantees or verification —
  [arXiv](https://arxiv.org/abs/1903.12394); [Lamarr Institute record](https://lamarr-institute.org/publication/informed-machine-learning-a-taxonomy-and-survey-of-integrating-prior-knowledge-into-learning-systems/).
  (Already in refs.bib as `vonRueden2023`.)
- Karniadakis, Kevrekidis, Lu, Perdikaris, Wang & Yang (2021), "Physics-informed machine learning", *Nature Reviews
  Physics* 3:422–440, doi 10.1038/s42254-021-00314-5 — [Nature](https://www.nature.com/articles/s42254-021-00314-5).
  Distinguishes observational, inductive (architecture-embedded) and learning (loss-embedded) biases; monotonicity /
  gain-sign priors are an inductive bias when built into the architecture and a learning bias when penalised (classification
  by the reviewer; the abstract itself does not list monotonicity).
- Willard, Jia, Xu, Steinbach & Kumar (2022/2023), "Integrating Scientific Knowledge with Machine Learning for
  Engineering and Environmental Systems", *ACM Computing Surveys* 55(4), article 66, 1–37, doi 10.1145/3514228 —
  [ACM DL](https://dl.acm.org/doi/10.1145/3514228); [arXiv](https://arxiv.org/abs/2003.04919). Surveys
  physics-guided loss functions, physics-guided initialisation, architecture design and hybrid models for engineering
  systems.
- Sharma & Liu (2022), *AIChE Journal* 68(5):e17609, doi 10.1002/aic.17609 (see Q1) is the chemical-engineering-specific
  review in the same family — [Wiley](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.17609).
- For the manufacturing side, Fraunhofer's work frames monotonicity as "expert knowledge" compensating data shortages and
  solves it as a semi-infinite programme (Kurnatowski et al. 2021, doi 10.3390/a14120345) —
  [MDPI](https://www.mdpi.com/1999-4893/14/12/345).

### Inferences
- In von Rueden's vocabulary the NCA 2026 paper integrates "algebraic inequality" knowledge into the *learning algorithm*
  (hard constraint at a reference point); the new paper additionally integrates it into the *final hypothesis* stage
  (certificate + repair). Using that taxonomy explicitly in the Introduction gives PSE readers a map they already know.
- None of the reviews treat "guaranteed on the whole domain" as a distinct category; the paper can note this as a gap in
  the review literature (soft vs. hard constraint is usually the only distinction drawn).

### Gaps
- The exact pages where von Rueden et al. and Willard et al. mention monotonicity were not retrieved (abstract-level
  access only); the statement that both list monotonicity among algebraic/shape constraints is from the reviewer's
  knowledge of the papers and should be checked against the PDFs before being cited as such.

## Key question 3 — Flexibility/feasibility analysis and deterministic global optimisation with embedded ANNs as conceptual analogues

### Takeaway
"Guarantee over a region by locating the worst point" is a 40-year-old PSE idea: Halemane & Grossmann's max–min–max
feasibility test and Swaney & Grossmann's flexibility index locate critical uncertain points to certify a design over a
box, and Schweidtmann & Mitsos (2019) give the modern tool — deterministic global optimisation of problems with embedded
ANNs via McCormick relaxations and branch-and-bound — that makes a sound certificate of an ANN property over a box
computable. Both map one-to-one onto the counter-example / branch-and-bound certificate of the new paper.

### Cited Findings
- Halemane & Grossmann (1983), "Optimal process design under uncertainty", *AIChE Journal* 29(3):425–433,
  doi 10.1002/aic.690290312 (≈450 citations) — [Semantic Scholar metadata](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1002/aic.690290312?fields=title,venue,year,authors,externalIds,citationCount).
  Formulates the feasibility of a design over a set of uncertain parameters as the max–min–max problem
  max_θ min_z max_j f_j(d,z,θ) ≤ 0 and shows that for convex problems the critical θ lie at vertices (vertex enumeration
  as a finite certificate); this is the PSE archetype of "worst-case point search certifies a region".
- Swaney & Grossmann (1985), "An index for operational flexibility in chemical process design. Part I: Formulation and
  theory", *AIChE Journal* 31(4):621–630, doi 10.1002/aic.690310412 —
  [Wiley](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.690310412); [scite record](https://scite.ai/reports/an-index-for-operational-flexibility-1JOnME).
  Defines the flexibility index as the largest scaled hyper-rectangle of uncertain parameters over which feasible
  operation is guaranteed, computed by searching for the critical (worst) parameter point.
- Later flexibility literature keeps the critical-point / active-set structure (e.g. Jiang et al. 2018, "New algorithm
  for the flexibility index problem of quadratic systems", *AIChE J.*, doi 10.1002/aic.16143 —
  [Wiley](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.16143); Zhao et al. 2018, "Analytical and triangular
  solutions to operational flexibility analysis using quantifier elimination", *AIChE J.*, doi 10.1002/aic.16207 —
  [Wiley](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.16207)); the quantifier-elimination paper is a direct
  bridge to "for all x in the box" statements.
- Schweidtmann & Mitsos (2019), "Deterministic Global Optimization with Artificial Neural Networks Embedded", *Journal of
  Optimization Theory and Applications* 180(3):925–948, doi 10.1007/s10957-018-1396-0. Reduced-space formulation,
  McCormick relaxations of the network propagated through the algorithm, tight convex/concave envelopes of tanh, solved
  by branch-and-bound in an in-house deterministic global solver (MAiNGO); guarantees global optimality over the whole
  input box; case studies: illustrative function, fermentation process, compressor plant, cumene process design;
  faster than BARON/SCIP — [Springer](https://link.springer.com/article/10.1007/s10957-018-1396-0);
  [RWTH record](https://www.avt.rwth-aachen.de/cms/avt/forschung/sonstiges/publikationen/~iavo/details/?file=745476&lidx=1).
  Follow-ups: Wilhelm et al., "Convex and concave envelopes of artificial neural network activation functions for
  deterministic global optimization", *J. Global Optimization*, doi 10.1007/s10898-022-01228-x —
  [Springer](https://link.springer.com/article/10.1007/s10898-022-01228-x); Schweidtmann et al., "Obey validity limits of
  data-driven models through topological data analysis and one-class classification", *Optimization and Engineering*,
  doi 10.1007/s11081-021-09608-0 — [Springer](https://link.springer.com/article/10.1007/s11081-021-09608-0).
- Kahrs & Marquardt (2007), "The validity domain of hybrid models and its application in process optimization",
  *Chemical Engineering and Processing: Process Intensification* 46(11):1054–1066, doi 10.1016/j.cep.2007.02.031
  (≈100 citations) — [OpenAlex metadata](https://api.openalex.org/works/doi:10.1016/j.cep.2007.02.031);
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0255270107001134). Addresses the complementary
  question — on which region of input space an ANN-based hybrid model may be trusted at all — and uses that region as a
  constraint in optimisation (abstract not retrievable; description from title and secondary citations).
- Qin & Badgwell (2003, see Q1) document the industrial practice of *not* certifying: Aspen Target's on-line model
  confidence index switches the ANN off outside the data range instead of guaranteeing its gains —
  [PDF](https://cepac.cheme.cmu.edu/pasilectures/darciodolak/Review_article_2.pdf).

### Inferences
- The paper's certificate (adversarial search for a sign violation + sound branch-and-bound over the box) can be
  presented to PSE readers as a *flexibility test for a model property*: the "uncertain parameters" are the inputs, the
  "constraint" is σ_j ∂f/∂x_j ≥ 0, the counter-example is the critical point, and the branch-and-bound bound plays the
  role of the convexity/vertex argument of Halemane & Grossmann. Schweidtmann & Mitsos supply the precedent that
  bounding an SLFN's (tanh/sigmoid) outputs and derivatives over a box with relaxations is standard PSE machinery.
- Kurnatowski et al. 2021 (adaptive discretisation of a semi-infinite monotonicity constraint) and the flexibility
  "active-set / critical-point" algorithms are the same iterative scheme as CEGIS: solve on a finite point set, find the
  worst violator, add it, repeat. Citing both makes the CEGIS loop look native to process engineering.

### Gaps
- The Halemane & Grossmann abstract was elided by the publisher in the metadata retrieved; the max–min–max description
  above is from the reviewer's knowledge of the paper and the Grossmann flexibility literature (standard, but re-check
  page numbers against the Wiley page before citing). Grossmann & Floudas (1987, *Comput. Chem. Eng.* 11(6):675–693,
  active-set strategy) was not retrieved in this session (verify DOI 10.1016/0098-1354(87)87011-4 before use).
- No PSE paper was found that applies deterministic global optimisation specifically to *verify monotonicity / gain signs*
  of a trained ANN (Schweidtmann & Mitsos optimise an objective with the ANN embedded; the verification use is an
  extension the new paper can claim).
