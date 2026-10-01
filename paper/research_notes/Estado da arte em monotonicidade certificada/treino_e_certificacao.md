# Estado da arte: treino guiado por contraexemplos com restrições de derivada e certificação de monotonicidade / sinal de derivada em redes treinadas

Notas de pesquisa (até 2026-10-01). Observações gerais sobre o método: cada entrada separa o que o método **garante** (sound / completo / global na caixa) do que apenas **promove**. Quando um DOI/ID não pôde ser confirmado numa fonte primária, isso está marcado como "[não verificado]". A página do MonoKAN (arXiv 2409.11078) não pôde ser lida (proxy devolveu 429); o que consta sobre ele vem de citações em outros artigos e do que já está em `paper/refs.bib`.

---

## Pergunta 1 — Quem certifica monotonicidade ou sinal de derivada de redes comuns, o que exatamente cada método garante e como escala?

### Takeaway
Há apenas quatro famílias que *certificam* (sound) monotonicidade de uma rede comum sobre um domínio inteiro: (i) MILP sobre a estrutura linear-por-partes (Liu et al. 2020 — exato por par de camadas, só ReLU, blocos de 2 camadas, ~4 s para 100 neurônios); (ii) envelopes/OMT por ponto (COMET, Sivaraman et al. 2020 — garantia só no predictor "envelope", não na rede treinada); (iii) cobertura Lipschitz–Voronoi da derivada (LipVor, Polo-Molina et al., TNNLS 2025 — sound e completo em princípio para ativações C², mas o número de iterações cresce exponencialmente com a dimensão; só demonstrado em 2D e 4D); (iv) propagação de limitantes lineares/intervalares do Jacobiano (RecurJac 2019, Shi et al. NeurIPS 2022 com BaB, Pasado 2023) — sound, polinomial, mas usada para monotonicidade apenas **local** (uma feature variando com as outras fixas) e, na prática, só com ReLU. Nenhuma delas combina (a) ativação sigmoide, (b) global na caixa, (c) B&B com teste de monotonicidade intervalar e forma de valor médio, (d) especialização a uma camada oculta.

### Cited Findings

**Liu, Han, Zhang, Liu — "Certified Monotonic Neural Networks", NeurIPS 2020 (arXiv:2011.10219)**
- Certificado = resolver `U_α := min_{x∈X, ℓ∈α} ∂_{x_ℓ} f(x)` como MILP sobre padrões de ativação ReLU; "if w^T_i x + bi ≥ 0, then zi must be one...if w^T_i x + bi ≤ 0, then zi must be zero" — a codificação é exata (sound e completa para o bloco certificado) — [arXiv 2011.10219](https://arxiv.org/pdf/2011.10219)
- Para redes profundas a verificação é por **pares de camadas** ("two-layer component f_{2k:2k−1}"), condição suficiente mas não necessária → procedimento global **não completo** para profundidade > 2 — [arXiv 2011.10219](https://arxiv.org/pdf/2011.10219)
- Treino: penalidade heurística `R(f) = E_{x∼Uni(X)} Σ_ℓ max(0, −∂_{x_ℓ} f(x))²` com 1 024 amostras uniformes por iteração; loop "treina → verifica → se falha, λ×10 e repete" (Algoritmo 1) — [arXiv 2011.10219](https://arxiv.org/pdf/2011.10219)
- Escala: "verification can be done in less than 4 seconds with 100 neurons in the first layer" (máquina de 48 núcleos/192 GB); "Even for a small network with 3 hidden layers and 20 neurons in each hidden layer, there is more than 800 binary variables. Current MILP solvers will fail to solve this problem in limited time (e.g. 1 hour)"; "the sampling could fail if the dimension of the input, d, is large (e.g. d = 276)" — [arXiv 2011.10219](https://arxiv.org/pdf/2011.10219)
- Datasets: COMPAS (6 172 × 13, 4 monótonas), Blog Feedback (54 270 × 276, 8), Loan Defaulter (488 909 × 28, 5), Chest X-Ray; larguras 40/100/200, profundidade 1 ou 3 — [arXiv 2011.10219](https://arxiv.org/pdf/2011.10219)
- Argumento para verificar derivadas e não pares: "verifying monotonicity casts a more significant challenge because it is a global property on the whole domain rather than a local neighborhood" — [arXiv 2011.10219](https://arxiv.org/pdf/2011.10219)

**Sivaraman, Farnadi, Millstein, Van den Broeck — "Counterexample-Guided Learning of Monotonic Neural Networks" (COMET), NeurIPS 2020 (arXiv:2006.08852)**
- Verificador: SMT em aritmética real linear com ReLU codificada como "y > 0 → z = y and y ≤ 0 → z = 0"; usa **OMT** (Optimization Modulo Theories) para o contraexemplo de violação máxima — [arXiv 2006.08852](https://arxiv.org/pdf/2006.08852)
- A consulta é **por ponto e por feature** (local): "the upper envelope search is over the interval [L, x^t[i]) and the lower envelope search is over the interval (x^t[i], U]" — [arXiv 2006.08852](https://arxiv.org/pdf/2006.08852)
- Garantia: só o predictor-envelope é monótono (Teorema 1: "For any function f and set of features S, the upper envelope f_S^u is monotonic in S"); o treino guiado **não** garante: "Although the algorithm improves quality, it does not guarantee monotonic predictions" — [arXiv 2006.08852](https://arxiv.org/pdf/2006.08852)
- Uso dos contraexemplos: dados adicionais de treino (rótulo = média, na regressão; mesmo rótulo do ponto-fonte, na classificação) — restrição **suave**, não dura — [arXiv 2006.08852](https://arxiv.org/pdf/2006.08852)
- Escala: redes de 3 camadas com 12–16 neurônios; Auto MPG, Boston Housing, Heart Disease, Adult; busca de pares arbitrários custou 48,29 min; "time taken grows with model size" — [arXiv 2006.08852](https://arxiv.org/pdf/2006.08852); código em [GitHub COMET](https://github.com/AishwaryaSivaraman/COMET)

**Polo-Molina, Alfaya, Portela — "A Mathematical Certification for Positivity Conditions in Neural Networks with Applications to Partial Monotonicity and Trustworthy AI" (LipVor), IEEE TNNLS 2025, DOI 10.1109/TNNLS.2025.3614431 (arXiv:2406.08525)**
- O que faz: certifica positividade de uma função (a derivada parcial de um MLP) num domínio compacto a partir de avaliações em pontos finitos, usando a constante de Lipschitz da derivada e um diagrama de Voronoi para provar cobertura: "the problem reduces to verifying if...the union of the respective balls...covers Ω" — [arXiv 2406.08525](https://arxiv.org/html/2406.08525); registo da publicação em [PubMed 41086072](https://pubmed.ncbi.nlm.nih.gov/41086072/) e [DOI](https://doi.org/10.1109/tnnls.2025.3614431)
- Sound (Teorema II.2, condição suficiente) e "completo" no sentido de terminar com certificado ou contraexemplo em N iterações (Teorema II.3) — [arXiv 2406.08525](https://arxiv.org/html/2406.08525)
- Ativações C² (sigmoide, tanh), MLPs de profundidade arbitrária; constante de Lipschitz da derivada parcial por limitante recursivo sobre normas das matrizes de pesos e da Hessiana da ativação (Teorema III.2): "this paper introduces, for the first time, a specific estimation of the Lipschitz constant of the partial derivative of a neural network" — [arXiv 2406.08525](https://arxiv.org/html/2406.08525)
- Escala: `N ≤ Vol(Ω)/Vol(B(ε/2L))` — exponencial na dimensão; experimentos só em 2D (581 iterações) e 4D (ESL, 488 instâncias, 423 iterações); sem tempos de parede — [arXiv 2406.08525](https://arxiv.org/html/2406.08525)
- Posicionamento: "the first study presenting an external certification algorithm to certify whether a trained unconstrained ANN...is partially monotonic without considering constrained architectures or piece-wise linear activation functions"; critica Liu 2020 por "only valid for a limited set of activation functions such as ReLU" e MILP "computationally expensive"; critica Sharma & Wehrheim por não garantir nem achar contraexemplos nem provar — [arXiv 2406.08525](https://arxiv.org/html/2406.08525)
- Código: https://github.com/alejandropolo/LipschitzNN — [arXiv 2406.08525](https://arxiv.org/html/2406.08525)

**Zhang, Zhang, Hsieh — "RecurJac", AAAI 2019 (arXiv:1810.11783)**
- Limitantes elemento-a-elemento (superior e inferior) do Jacobiano sobre uma bola em torno de um ponto; sound (Teorema 1); ativações gerais "(leaky-)ReLU, sigmoid family (including sigmoid, arctan, hyperbolic tangent, etc)"; custo polinomial, "at most H times" o de Fast-Lip — [arXiv 1810.11783](https://arxiv.org/html/1810.11783)
- Não discute monotonicidade explicitamente; a aplicação mais próxima é "we can guarantee that no stationary points exist inside a certain region" — [arXiv 1810.11783](https://arxiv.org/html/1810.11783)

**Shi, Wang, Zhang, Kolter, Hsieh — "Efficiently Computing Local Lipschitz Constants of Neural Networks via Bound Propagation", NeurIPS 2022 (arXiv:2210.07394)**
- Limitantes do Jacobiano de Clarke por propagação linear (CROWN) num grafo backward de ordem superior; RecurJac é "a special case of ours with weaker relaxations"; admite Branch-and-Bound; abstract anuncia "an application on provable monotonicity analysis" — [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0ff54b4ec4f70b3ae12c8621ca8a49f4-Abstract-Conference.html)
- A aplicação de monotonicidade é **local**: Adult, MLP de 4 camadas × 512, "For each continuous feature j, we set an input domain where only this particular feature can be varied between the minimum and maximum possible values in the dataset, while we keep other features fixed"; certifica 40 % (Age) / 58 % (Education) dos pontos vs 32 % / 55 % do RecurJac; sem comparação com MILP; relaxações só para ReLU nos experimentos — [arXiv 2210.07394](https://arxiv.org/pdf/2210.07394)

**Laurel, Qian, Misailovic et al. — "Synthesizing Precise Static Analyzers for Automatic Differentiation" (Pasado), OOPSLA 2023, DOI 10.1145/3622867**
- Interpretação abstrata (zonotopos/intervalos) de regras de AD; sound; exemplo de monotonicidade certificada "by certifying that the derivative is always non-zero" (modelo climático via ODE, não rede neural); para redes, só limitantes Lipschitz de uma CNN — [OOPSLA 2023](https://jsl1994.github.io/papers/OOPSLA2023Final.pdf); predecessor DeepJ POPL 2022 ([Laurel et al. 2022](https://misailo.cs.illinois.edu/papers/diff_oopsla22.pdf))

**Sharma & Wehrheim — "Higher Income, Larger Loan? Monotonicity Testing of Machine Learning Models", ISSTA 2020, DOI 10.1145/3395363.3397352 (arXiv:2002.12278)**
- Teste caixa-preta: aproxima o modelo por árvore de decisão, gera contraexemplos com Z3, refina; **nenhuma garantia** de monotonicidade — só deteta violações; 90 modelos, 10 datasets — [arXiv 2002.12278](https://arxiv.org/pdf/2002.12278); [ACM DL](https://dl.acm.org/doi/10.1145/3395363.3397352)

**Verificação com BaB para não linearidades gerais (contexto, não monotonicidade)**
- GenBaB (Shi, Jin, Kolter, Jana, Hsieh, Zhang, TACAS 2025, arXiv:2405.21063): BaB para sigmoide/tanh/sin/GeLU com "pre-optimize branching points" em espaço de pré-ativação; venceu VNN-COMP 2023/2024; não trata derivadas/Jacobianos — [arXiv 2405.21063](https://arxiv.org/abs/2405.21063)
- Tanaka & Yatabe, "Learn and Verify" (arXiv:2601.19818, 2026): aritmética intervalar com arredondamento dirigido (INTLAB) e subdivisão adaptativa para certificar desigualdades diferenciais de PINNs SIREN (4×30) — sound e global em [0,T]; sem retroalimentação de contraexemplos — [arXiv 2601.19818](https://arxiv.org/html/2601.19818v1)

**Monotonicidade por construção (referência para a frase "certified by sign")**
- CMNN (Runje & Shankaranarayana, ICML 2023) e Sartor et al. (ICML 2025) garantem monotonicidade por restrição de sinal dos pesos + ativações monótonas; Wiedemann, Jacquier & Gonon (arXiv:2606.13803, 2026) resumem: "architectural constraints remain the appropriate default where available and applicable" e reconhecem que métodos suaves têm "Off-grid feasibility is not formally certified" — [arXiv 2606.13803](https://arxiv.org/html/2606.13803)
- MonoKAN (Polo-Molina, Alfaya, Portela; arXiv:2409.11078): KAN com splines de Hermite cúbicas e restrições de positividade, monotonicidade certificada por construção — listado em [busca](https://arxiv.org/html/2409.11078v1) (página não lida: 429)

### Inferences
- O certificado do manuscrito (B&B intervalar com forma de valor médio + teste de monotonicidade de Hansen & Walster, global em [0,1]^p, para sigmoide) ocupa um nicho não coberto: Liu 2020 é ReLU/MILP; LipVor é sigmoide mas exponencial em p e sem B&B/contratação de caixas; Shi 2022/RecurJac são sound e polinomiais, mas usados para monotonicidade local (uma feature de cada vez) e demonstrados com ReLU; Pasado é interpretação abstrata genérica sem aplicação a redes monótonas.
- A afirmação de "primeiro certificador externo para ANN sem arquitetura restrita e sem ativação linear-por-partes" já foi feita por Polo-Molina et al. (TNNLS 2025) — o manuscrito deve citá-los e distinguir-se por: B&B + contratação (vs. cobertura por bolas), especialização a 1 camada oculta (limitante separável exato em pré-ativação + LP sobre o zonotopo), escalabilidade em p demonstrada nos 8 datasets, e integração no laço de treino.
- Shi et al. 2022 têm uma variante com BaB para Jacobiano que, em princípio, poderia certificar monotonicidade global de redes sigmoides; não há evidência publicada disso. Convém mencionar como "poderia estender-se" e não como "faz".

### Gaps
- Não consegui ler o MonoKAN (429) para confirmar o que a sua revisão de literatura diz sobre certificação de redes comuns.
- Não encontrei um artigo que aplique CROWN/∂-CROWN a monotonicidade global de redes sigmoides; ∂-CROWN (Eiras et al., "Efficient Error Certification for Physics-Informed Neural Networks", ICML 2024, arXiv:2305.10157 [não verificado nesta sessão]) limita derivadas de PINNs tanh e deveria ser citado como técnica de limitação de derivadas, mas não trata monotonicidade.
- Não encontrei tempos de parede do LipVor nem experimentos acima de 4D — o manuscrito pode afirmar que a comparação de escalabilidade com LipVor é inexistente na literatura.

---

## Pergunta 2 — Treino guiado por contraexemplos (CEGIS/CEGAR) com restrições duras para redes neurais: quem já fez e com que garantias?

### Takeaway
O padrão aprendiz–verificador existe desde 2016–2019 para funções de Lyapunov/barreira (Ravanbakhsh & Sankaranarayanan; Chang et al.; Abate/Peruffo/FOSSIL; Dai et al.) e para monotonicidade só na forma suave (COMET). O único precedente em que os contraexemplos entram como **restrições duras** no problema do aprendiz é Ravanbakhsh & Sankaranarayanan (candidato linear nos coeficientes → LP de viabilidade com as condições de Lyapunov impostas nos pontos-testemunha, com terminação em O(r²) via centro do elipsoide de volume máximo). Para monotonicidade, nenhum trabalho encontrado usa (a) restrições duras de sinal do gradiente em âncoras, (b) busca adversarial multi-start, (c) herança de âncoras entre tamanhos de rede construtiva, (d) verificador sound global como critério de paragem, (e) fallback por sinal que garante 100 % de modelos certificados.

### Cited Findings

**Ravanbakhsh & Sankaranarayanan — "Learning Control Lyapunov Functions from Counterexamples and Demonstrations", RSS 2017 (arXiv:1705.09619); versão estendida Autonomous Robots 43:275–307, 2019, DOI 10.1007/s10514-018-9791-9**
- Aprendiz = LP: o espaço de candidatos é o "set of all candidates c s.t. V_c satisfies the CLF condition (3) for every point in the finite set W_j" com restrições duras "V_c(x_i) > 0 ∧ ∇V_c · f(x_i, u_i) < 0" — [arXiv 1705.09619](https://arxiv.org/pdf/1705.09619)
- Terminação: escolhendo o centro do elipsoide de volume máximo, "Vol(C_{j+1}) ≤ (1 − 1/r) Vol(C_j)" ⇒ "the learning loop terminates in at most ... = O(r²) iterations" — [arXiv 1705.09619](https://arxiv.org/pdf/1705.09619)
- Verificador: dReal (δ-completo) ou relaxação SDP sound mas incompleta ("If the relaxed optimization problems ... yield a zero solution, then the given candidate V_i(x) is in fact a CLF", Lema 3.5) — [arXiv 1705.09619](https://arxiv.org/pdf/1705.09619); [Springer](https://link.springer.com/article/10.1007/s10514-018-9791-9)

**Chang, Roohi, Gao — "Neural Lyapunov Control", NeurIPS 2019**
- "alternates between optimizing the empirical loss (19) and searching for counterexamples to guide training"; "SMT-based learner-verifier architecture using the dReal SMT solver"; contraexemplos adicionados ao conjunto de treino (restrição suave via perda) — [Dawson, Gao, Fan, survey IEEE T-RO 2023, arXiv 2202.11762](https://arxiv.org/html/2202.11762v2)

**Abate, Ahmed, Giacobbe, Peruffo — "Formal Synthesis of Lyapunov Neural Networks", IEEE L-CSS 2021; Peruffo, Ahmed, Abate — barreiras neurais (TACAS 2021); FOSSIL (HSCC 2021) e Fossil 2.0 (Edwards, Peruffo, Abate, HSCC 2024)**
- Laço: "the learner is responsible for training the certificate neural network, while the verifier periodically checks the learned certificate. If the certificate is valid, the verifier stops the training early; otherwise, it provides counterexamples to enrich the training dataset"; "SMT-based methods have been demonstrated on networks of up to 30 neurons" — [survey arXiv 2202.11762](https://arxiv.org/html/2202.11762v2); [AAGP21 PDF](https://www.cs.ox.ac.uk/people/alessandro.abate/publications/AAGP21.pdf)
- Barreiras neurais contínuas com aprendizagem guiada por contraexemplos: [ACM TECS 2023, DOI 10.1145/3609125](https://dl.acm.org/doi/abs/10.1145/3609125)
- O FOSSIL Barr3 é usado como benchmark de certificado por Wiedemann et al. 2026, que citam "FOSSIL (Formal Certificate Synthesis) [Edwards et al. 2024]" e "Neural Lyapunov Functions [Dai et al. 2020, 2021]: Counter-example guided synthesis" — [arXiv 2606.13803](https://arxiv.org/html/2606.13803)

**Dai, Landry, Yang, Pavone, Tedrake — "Lyapunov-stable neural-network control", RSS 2021**
- Verificador MIP exato para ReLU: "the piecewise linear nature of (leaky) ReLU activation implies that the input and output of a (leaky) ReLU network satisfy mixed-integer linear constraints"; certifica quando os ótimos de (10a)/(10b) são 0 — [RSS 2021](https://www.roboticsproceedings.org/rss17/p063.pdf)
- Dois laços: Algoritmo 1 acumula contraexemplos em X₁, X₂ e minimiza perda-hinge (suave); Algoritmo 2 é bi-nível: "we first solve the inner maximization problem using MIP solvers, and then compute the gradient of the MIP optimal objective w.r.t the variables θ" — [RSS 2021](https://www.roboticsproceedings.org/rss17/p063.pdf)
- Escala: Lyapunov de (8,4,4) a (16,12,8) neurônios; o survey observa "the MILP-based method in [64] is demonstrated on networks containing only 16 neurons" — [arXiv 2202.11762](https://arxiv.org/html/2202.11762v2)

**COMET (Sivaraman et al. 2020)** — ver P1: contraexemplos de monotonicidade obtidos por OMT **são adicionados como dados rotulados** (restrição suave) e o laço "does not guarantee monotonic predictions" — [arXiv 2006.08852](https://arxiv.org/pdf/2006.08852)

**Penalidades de gradiente em pontos amostrados (o "adversarial training" de monotonicidade que existe é por amostragem, não por busca)**
- Gupta et al. 2019, "How to Incorporate Monotonicity in Deep Networks While Preserving Flexibility?" — perda ponto-a-ponto no gradiente — [arXiv 1909.10662](https://arxiv.org/abs/1909.10662)
- Monteiro et al., UAI 2022: penalidade `max(0, −∂h/∂x_i)²` em Ω_train, Ω_random ou Ω_mixup; "monotonicity is achieved in the regions where it is enforced"; "random draws from any distribution will likely lie in the boundaries of the space"; violação medida só empiricamente (taxa ρ̂); sem certificação — [PMLR v180](https://proceedings.mlr.press/v180/monteiro22a/monteiro22a.pdf)
- Liu et al. 2020: penalidade em 1 024 pontos uniformes + laço "aumenta λ até certificar" (ver P1) — a única combinação penalidade-amostrada + certificado sound encontrada — [arXiv 2011.10219](https://arxiv.org/pdf/2011.10219)

**Wiedemann, Jacquier, Gonon — "Neural Slack Variables for Shape Constraints", arXiv:2606.13803 (junho 2026)**
- Método suave (rede auxiliar de folga ≥ 0 casada ao perfil de restrição `C[f^θ](x)`), motivado pelo "constraint drifting": "For the penalty method, the violation term supplies no gradient wherever the network is already feasible. Subsequent training steps can then reintroduce violations" — [arXiv 2606.13803](https://arxiv.org/html/2606.13803)
- Sem certificação: "Off-grid feasibility is not formally certified"; lista métodos primal-dual (Ramirez et al. 2025), multiplicadores neurais (Narasimhan et al. 2020), HardNet (Min & Azizan 2025), PAC-constraints (Chamon & Ribeiro 2020) — [arXiv 2606.13803](https://arxiv.org/html/2606.13803)

**Aumento de dados guiado por contraexemplos (CEGIS para dados, não para restrições)**
- Dreossi et al., "Counterexample-Guided Data Augmentation", IJCAI 2018 [não verificado nesta sessão; citar com DOI 10.24963/ijcai.2018/286 após conferência] — contraexemplos de um falsificador entram como dados de treino, sem restrições duras.

### Inferences
- A diferença precisa a enunciar no manuscrito: COMET, Chang 2019, FOSSIL e Dai (Alg. 1) usam os contraexemplos como **dados/perda** (suave); Ravanbakhsh & Sankaranarayanan e Dai (Alg. 2) são os únicos com formulação **dura/bi-nível**, mas para certificados de Lyapunov com verificador SMT/SDP/MIP e candidatos lineares ou ReLU. O manuscrito transfere a versão dura (SLSQP com restrições de desigualdade nas âncoras) para o sinal de ganhos de uma SLFN sigmoide, com verificador intervalar sound — combinação não encontrada.
- A "herança de âncoras entre tamanhos" não tem precedente nos laços CEGIS acima (todos treinam uma arquitetura fixa); pode ser apresentada como contribuição ligada ao caráter construtivo (warm start de restrições, não só de pesos).
- A garantia "todo modelo devolvido é certificado" (via fallback de sinal) é mais forte do que a de COMET (só o envelope), Liu 2020 (pode não certificar e para), e dos laços Lyapunov (podem não convergir). Deve ser dita como "garantia de saída", distinguindo-a de garantia de que o laço CEGIS converge.

### Gaps
- Não encontrei nenhum artigo intitulado ou descrito como "adversarial training for monotonicity" com busca de pior caso por otimização (todos amostram). Se existir, está fora dos resultados obtidos.
- A terminação do laço CEGIS do manuscrito não tem análogo publicado para restrições não lineares nos parâmetros (SLSQP não é LP); o argumento de Ravanbakhsh & Sankaranarayanan só vale para candidatos lineares em c.

---

## Pergunta 3 — Programação semi-infinita / métodos de troca aplicados a regressão com restrições de forma e a redes; limitação de redes sigmoides sobre caixas em otimização global

### Takeaway
A leitura "troca/discretização de SIP" do laço CEGIS está bem ancorada: o grupo do Fraunhofer ITWM (Kurnatowski, Schmid, Link, Poursanidis) formula regressão com restrições de forma (monotonicidade, convexidade) como SIP e resolve por discretização adaptativa / pontos viáveis, mas sempre com modelos **lineares nos parâmetros** (polinómios), problema convexo e verificação do pior caso por amostragem (10 000 pontos). Para redes com ativações sigmoides, a limitação sound sobre caixas vem da otimização global determinística (Schweidtmann & Mitsos, JOTA 2019: relaxações de McCormick em espaço reduzido, ramificando apenas nas entradas da rede) — mas para a *saída* da rede, não para o sinal da derivada.

### Cited Findings

**Fundamentos SIP (já em `refs.bib`: Hettich & Kortanek 1993, Blankenship & Falk 1976)**
- Poursanidis, Link, Schmid, Teicher, "Incorporating Shape Knowledge into Regression Models" (arXiv:2409.17084) citam como base: Hettich & Kortanek (1993), Reemtsen (1998), Blankenship & Falk (1976, discretização), López (2007, discretização adaptativa), Tsoukalas & Mitsos (2011, pontos viáveis adaptativos), Djelassi et al. — [ar5iv 2409.17084](https://ar5iv.labs.arxiv.org/html/2409.17084)

**Kurnatowski, Schmid et al. — "A semi-infinite optimization approach to shape-constrained regression" (2021; ResearchGate 353038146) e Schmid — "Approximate solutions of convex semi-infinite optimization problems in finitely many iterations" (arXiv:2105.08417)**
- Algoritmo de discretização adaptativa para SIP convexos com "considerably smaller discretizations", terminação finita num ponto viável, precisão arbitrária, "Continuity is the only regularity assumption"; aplicação a "parametric and non-parametric regression problems under shape constraints"; não menciona redes neurais — [arXiv 2105.08417](https://arxiv.org/abs/2105.08417v1); [ResearchGate](https://www.researchgate.net/publication/353038146_A_semi-infinite_optimization_approach_to_shape-constrained_regression) (429 ao tentar ler o texto completo)
- SIASCOR (Poursanidis et al. 2024): modelos `ŷ_w(x) = wᵀφ(x)` polinomiais anisotrópicos; restrições "restrict affine-linear combinations of partial derivatives to satisfy g_i(w,x) ≤ 0"; "guarantees optimality up to arbitrary precision and strict fulfillment of the shape constraints"; o pior caso é verificado por amostragem ("we sampled 10000 random points in X and tested the shape compliance on 100 equidistant points along the relevant axis"); dimensões 3–5, 60–125 amostras — [ar5iv 2409.17084](https://ar5iv.labs.arxiv.org/html/2409.17084)
- Versão em capítulo Springer: [10.1007/978-3-031-83097-6_7](https://link.springer.com/chapter/10.1007/978-3-031-83097-6_7) (não lida)

**Schweidtmann & Mitsos — "Deterministic Global Optimization with Artificial Neural Networks Embedded", J. Optim. Theory Appl. 180(3):925–948, 2019, DOI 10.1007/s10957-018-1396-0**
- "relaxations of algorithms using McCormick relaxations in a reduced space" que exploram "the convex and concave envelopes of the nonlinear activation function" (tanh); ramifica apenas nas entradas da rede; solver MAiNGO; sem menção a monotonicidade ou derivadas — [Springer](https://link.springer.com/article/10.1007/s10957-018-1396-0); [RWTH AVT](https://www.avt.rwth-aachen.de/cms/avt/forschung/sonstiges/publikationen/~iavo/details/?file=745476&lidx=1)
- Revisão posterior "Global optimization with neural networks embedded: theory, applications, and future work" — [RWTH AVT](https://www.avt.rwth-aachen.de/cms/avt/forschung/sonstiges/publikationen/~iavo/details/?file=795238&lidx=1) (não lida)

**Interval global optimisation (contexto do certificado)**
- Hansen & Walster 2004 e Moore, Kearfott & Cloud 2009 já estão em `refs.bib`; o manuscrito usa extensão natural + forma de valor médio + teste de monotonicidade — ferramentas canónicas desses livros (não foi preciso nova fonte).
- Tanaka & Yatabe 2026 (P1) mostram que subdivisão adaptativa + aritmética intervalar continua a ser a via usada para certificar desigualdades diferenciais de redes suaves — [arXiv 2601.19818](https://arxiv.org/html/2601.19818v1)

### Inferences
- O manuscrito pode afirmar com segurança: (i) a leitura SIP está correta e tem precedente só com modelos lineares nos parâmetros e verificação por amostragem; (ii) o seu "subproblema de nível inferior" é resolvido de forma sound (B&B intervalar), ao contrário de SIASCOR/SIAMOR que amostram; (iii) é a primeira aplicação encontrada de SIP/troca a restrições de sinal de gradiente em SLFN sigmoides com certificado global.
- A ligação a Schweidtmann & Mitsos é útil para justificar "ramificar só no espaço de entrada" (reduced space) e os envelopes de σ'/σ''; o manuscrito vai além ao ramificar alternativamente no espaço de pré-ativação quando n < p.

### Gaps
- Não consegui confirmar autores/venue exatos do artigo "A semi-infinite optimization approach to shape-constrained regression" (ResearchGate 429); provável: Kurnatowski, Schmid et al., 2021 [não verificado].
- Não encontrei trabalho que use McCormick/relaxações convexas para limitar a **derivada** de uma rede tanh/sigmoide (só a saída).
- Não encontrei artigos de "learning shape-constrained neural networks via semi-infinite programming" com redes (todos os encontrados usam polinómios).

---

## Pergunta 4 — Especificidade de uma camada oculta: separabilidade em pré-ativação e B&B sobre zonotopos

### Takeaway
Não encontrei nenhum trabalho que explore, para certificar sinal de derivada, a separabilidade de `h_i(x)=Σ_j c_ij σ'(w_j·x+b_j)` em pré-ativação, nem B&B em espaço de pré-ativação com LP de pertença ao zonotopo `{Wx+b : x ∈ caixa}`. Os ingredientes existem separadamente: zonotopos como domínio abstrato para redes (AI2, DeepZ — incluindo sigmoide/tanh), ramificação em espaço de pré-ativação com pontos pré-otimizados (GenBaB), e certificação por par de camadas (Liu 2020 reduz a profundidade a blocos de 2 camadas, i.e., exatamente o caso de uma camada oculta, mas por MILP/ReLU).

### Cited Findings
- Liu et al. 2020 verificam redes profundas decompondo-as em blocos de duas camadas ("two-layer component f_{2k:2k−1}") — logo o caso SLFN é o átomo da certificação deles, resolvido por MILP com uma variável binária por neurônio — [arXiv 2011.10219](https://arxiv.org/pdf/2011.10219)
- GenBaB (TACAS 2025) ramifica em pontos pré-otimizados das não linearidades (pré-ativação) com "lookup table" — [arXiv 2405.21063](https://arxiv.org/abs/2405.21063)
- Pasado/DeepJ usam zonotopos como domínio abstrato para derivadas — [OOPSLA 2023](https://jsl1994.github.io/papers/OOPSLA2023Final.pdf)
- LipVor (TNNLS 2025) cobre o domínio de entrada por bolas; não usa a estrutura de pré-ativação — [arXiv 2406.08525](https://arxiv.org/html/2406.08525)

### Inferences
- A observação "a imagem afim de uma caixa é um zonotopo" é clássica (AI2 Gehr et al. 2018 / DeepZ Singh et al. 2018 — citações a confirmar); o passo novo do manuscrito é usar o limitante separável *exato* da soma de termos unidimensionais sobre uma caixa de pré-ativações e descartar caixas por LP de pertença ao zonotopo — isto é um B&B em dimensão n (< p) com relaxação sound. É defensável como contribuição técnica específica à SLFN; recomenda-se citar DeepZ/AI2 para o uso de zonotopos e GenBaB para ramificação em pré-ativação, e afirmar explicitamente que "não encontrámos uso prévio".

### Gaps
- Não verifiquei nesta sessão as referências AI2 (Gehr et al., IEEE S&P 2018) e DeepZ (Singh et al., NeurIPS 2018); conferir antes de citar.
- Não encontrei literatura sobre "optimização de função separável sobre zonotopo" aplicada a redes; se o manuscrito quiser um ancoradouro teórico, procurar em otimização global (funções separáveis sobre politopos) fora do escopo desta pesquisa.
