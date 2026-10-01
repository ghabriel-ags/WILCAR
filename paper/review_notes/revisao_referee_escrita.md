# Revisão simulada (referee + editor de texto) — `paper/main.tex` v7, 2026-10-01

Base: `main.tex` (19 p. preprint, 10 p. no layout da revista, ~2 950 palavras de corpo), `supplement.tex` (11 p.),
`tables/*.tex`, figuras em `paper/figures/`, `NOTES_pre_submission.md`, e os dois artigos anteriores do grupo (NCA 2026,
EI 2022) para estilo. Nenhum arquivo do manuscrito foi alterado.

---

## Parte 1 — Parecer de referee (Neural Networks)

### Resumo do manuscrito

Os autores tratam de classificação binária com sinais de ganho conhecidos (monotonicidade parcial) em redes de uma camada
oculta (SFNN) treinadas construtivamente (WILCAR, RIXM, ELM). Contribuições declaradas: (i) Teorema 1 — SFNNs sigmoides
monótonas na caixa, com margem de ganho positiva, são aproximadores universais de funções contínuas parcialmente
monótonas, enquanto a parametrização por sinais dos pesos (Archer & Wang; Daniels & Velikova) não é; (ii) treino
guiado por contraexemplos ("cegis"): restrições de sinal do ganho em âncoras que crescem com os pontos violados de uma
busca adversarial, mais reparo e *fallback* para restrição de sinal, de modo que toda rede devolvida é monótona;
(iii) certificado *branch-and-bound* sólido, especializado para uma camada oculta e exato no espaço de pré-ativação;
(iv) experimentos em 7 alvos sintéticos e 8 conjuntos reais, CV aninhada, 13 métodos na tabela principal (27 no
suplemento), e *splits* oficiais dos benchmarks COMPAS / Heart Disease / Loan.

### Pontos fortes

1. O problema é bem motivado e o texto é curto e direto para os padrões da revista (10 páginas no layout de duas colunas).
2. A hierarquia de restrições (média → âncora → sinal → cegis) é clara e a Tabela `modes` é um bom resumo "o que cada
   restrição garante".
3. O certificado é simples de entender (cota inferior em caixa; prova, contraexemplo ou divide), com prova de solidez e
   terminação, e a ablação (Tabela S cert_bench) é informativa (pré-ativação é o que decide os casos difíceis).
4. Protocolo honesto: CV aninhada, mesmos *splits* para todos, baselines re-treinados sob protocolo comum, números
   publicados mostrados lado a lado, estatística de Demšar, e os autores admitem que na prática o cegis não supera a
   restrição de sinal em dados reais.
5. Figura 1 é uma boa ilustração do argumento de expressividade.

### Preocupações maiores (ordem de gravidade)

**M1. Significância para o leitor de *Neural Networks*: o método proposto não ganha de ninguém onde importa.**
Na Tabela 2 o melhor rank é RIXM sem restrição (4.5); o WILCAR-cegis fica em 5.8, atrás de RIXM-cegis, mean, CMNN e
XGBoost; árvores vencem em 4 dos 8 conjuntos. O próprio texto diz que em dados reais `sign` e `cegis` dão a mesma acurácia
porque as redes selecionadas têm 1–5 unidades. Portanto o aparato cegis + certificado só mostra benefício nos alvos
sintéticos — e lá Min-Max, LMN e LightGBM-monótono têm erro menor (0.041–0.042 vs 0.053). O resultado "30 % menor
que sinal" do resumo é verdadeiro mas é o único recorte em que o método vence; o leitor descobre na Tabela 3 que três
baselines monótonos são melhores. O artigo precisa de um argumento de valor que não seja acurácia: custo/tempo, garantia
exata (vs. Lipschitz/arquitetura), *warm start*, ou um regime (dados escassos, extrapolação) em que a restrição correta
ajuda — o item "data scarcity" das notas é exatamente o experimento que falta.

**M2. Novidade frente a COMET (Sivaraman et al. 2020) e Certified MNN (Liu et al. 2020).**
COMET é literalmente "counterexample-guided monotonicity enforcement" (SMT → contraexemplos → re-treino) e Liu et al.
certificam por MILP com penalidade. O manuscrito os cita em uma linha na introdução e nunca diz em que o cegis difere
(restrição dura em vez de dado/penalidade; certificado exato de uma camada em vez de SMT/MILP; saída sempre monótona por
*fallback*; sem peso de penalidade). Um referee que conheça COMET classificará o método como variante. Também falta
posicionar frente a Polo-Molina et al. 2026 (cobertura Lipschitz) e à literatura de programação semi-infinita (o texto
cita Hettich 1993, bom, mas só uma frase).

**M3. Teorema 1 é elementar e a afirmação do resumo "We show that sign constraints ... cannot approximate" é um resultado
conhecido (Daniels & Velikova 2010; Gupta et al. 2016), aqui apenas ilustrado (Fig. 1, Fig. S1).** O resumo e a
introdução devem dizer "recall/illustrate", e o teorema deve ser vendido pelo que ele acrescenta: margem de ganho que torna
o certificado completo (Prop. term.) e o contraste com a necessidade de profundidade para redes monótonas de limiar
(Mikulincer & Reichman 2025 está no `.bib` e não é citado). Sem isso, um referee escreve "the theorem is a routine
application of Hornik (1991)".

**M4. Fairness dos baselines re-treinados.** Os CMNN/LMN "our runs" ficam 8–9 pontos abaixo dos publicados em Heart
Disease (80.7/83.6 vs 89/89.6) e 1.4–1.6 abaixo em COMPAS. As grades são pequenas (CMNN largura 8–32, 2 camadas; LMN
largura 16–32; Min-Max K∈{2,4,8}; XGB 2 profundidades × 2 n_trees), com Adam lr fixo e orçamento de 1 h. O texto diz
"fall short by similar margins" como se isso absolvesse a comparação; um referee lerá o contrário: *os baselines estão
subajustados, logo "match CMNN" não está demonstrado*. É preciso (a) alargar as grades dos três estruturais ao que os
artigos originais usam ou (b) justificar por igualdade de orçamento e mostrar sensibilidade. Heart Disease tem 61 casos de
teste (1 caso = 1.6 pontos) — dizer isso na tabela, não só no texto.

**M5. Estatística.** (a) Wilcoxon sobre 185 *outer folds* agrupados de 8 conjuntos: *folds* de CV repetida 5×5 não são
independentes (Nadeau & Bengio 2003) e agrupar conjuntos mistura escalas; p < 1e-4 para +0.017 MCC é inflado. Demšar
recomenda comparar por conjunto. (b) O diagrama CD (Fig. S2) tem CD = 6.45 com 13 métodos em 8 conjuntos: a barra une
todos os 13 métodos — o próprio diagrama diz "nenhuma diferença". O texto afirma, via Holm vs. melhor, que Min-Max e LMN
são piores. Os dois não são contraditórios (Holm é mais poderoso), mas o referee verá a figura e dirá que o artigo se
contradiz; a legenda deve dizer isso explicitamente ou a figura deve sair. (c) Os desvios-padrão na Tabela S main_mcc_full
são sobre *folds* de CV repetida (subestimam a variância).

**M6. Resultados escondidos no suplemento que contradizem a narrativa.** WILCAR-cegis com *hold-out* dá MCC 0.866 (sd 0.26)
em BCD (mean 0.896, anchor 0.865), i.e., alguns *folds* colapsaram (MCC ≈ 0), com tempos de 2 460–4 518 s e 22 de 740
laços de seleção interrompidos pelo orçamento de 1 h. A tabela principal mostra só a linha DCV (0.952). É preciso dizer o
que aconteceu em BCD (27 entradas restritas; SLSQP) e reportar o efeito do orçamento sobre os métodos afetados — senão
parece seleção de linha favorável.

**M7. Reparo e *fallback* não são avaliados.** n* é escolhido na DCV com redes que ainda não passaram pelo reparo; 2.7 %
(15/555) das redes vêm do *fallback* (classe diferente). Reportar MCC antes/depois do reparo, quantas âncoras/rodadas por
tamanho, e o MCC das redes de *fallback*. Também: durante a seleção (DCV) as redes intermediárias não são certificadas —
dizer isso.

**M8. Rigor do certificado.** "exact up to floating-point rounding" não é um certificado sólido no sentido da comunidade
de verificação; não há arredondamento dirigido nem aritmética intervalar rigorosa. Uma re-checagem com `mpmath.iv` (ou
margem explícita ε_cert > 0 na decisão) e uma frase bastam; sem isso, "sound" no título da seção será contestado.

**M9. Escala e relevância ("one hidden layer in 2026").** Dezenas de unidades, dezenas de milhares de linhas, Loan
subamostrado a 20 000 e SLSQP cúbico. O artigo deve assumir o nicho (tabular pequeno/médio, garantia exata) na
introdução e dizer os limites de tamanho com números (n, p, N) — hoje só aparece na última frase da discussão.

**M10. Restrição "mean" e o artigo anterior (NCA).** O NCA (Eq. 13–14 do artigo) impõe o ganho por diferenças finitas em
*um* ponto de referência (média das entradas de teste), ε = 0.05, δ = 0.1 — o manuscrito acerta ao chamá-lo de "anchor
com A = ponto de referência". Mas então a restrição "mean" (média do ganho sobre as entradas de treino, ε_m = 0.05) não é
atribuída a nada e só é justificada por Härdle & Stoker 1989. Diga de onde ela vem (dissertação) e por que está aqui
(ablação "restrição global mais fraca"). Note que ε_m = 0.05 vs ε_p = 1e-3 são margens incomparáveis; em BCD a restrição
mean custa 0.014 MCC — provavelmente a margem — e o texto diz "the mean constraint changes little".

### Preocupações menores

1. Tabela 1: taxa de positivos de LOAN = `[pending]`; `[TODO]` em CRediT, DOI de repositório, financiamento e declaração
   de IA; Tabela S signals com `[TODO]` na ordem dos atributos de LOAN. Nada disso pode ir à submissão.
2. Resumo: "35–41 % of the unconstrained networks are monotone" — a Tabela `modes` dá 35 % (real) e a Tabela 3 dá 35 %
   (sintético); de onde vem 41 %? E `numbers.tex` gera `MeanCertified = 78.0 %` / `MeanViolDomain = 2.55 %` enquanto a
   Tabela `modes` e o texto dizem 75 % / 3.0 % — populações diferentes (todas as redes mean vs WILCAR-DCV)? Alinhar ou
   explicar.
3. Introdução: "compare the approach with thirteen alternatives" — a Tabela 2 tem 13 métodos *incluindo* os dois
   propostos (11 alternativas); o suplemento tem 27.
4. "Rank" da Tabela 2 (sobre 13) e da Tabela S main_mcc_full (sobre 27) usam o mesmo nome; WILCAR-cegis-DCV é 5.8 numa e
   11.1 noutra.
5. ELM-C é definido na Tabela S settings mas não aparece em nenhuma tabela; DCV-1SE está no Algoritmo S2 e nunca é usado.
   Remover ambos ou reportar.
6. Tempo: o texto dá mediana (22 s), a Tabela `modes` mediana, a Tabela S size médias (até 4 518 s). Dizer que a média é
   dominada por BCD/LOAN já no texto da Tabela S size.
7. Heart Disease com sd 0.0 em cinco *seeds* para WILCAR/LR: explicar (métodos determinísticos dado o *split*).
8. "Certified MNN" (Liu 2020) e "Activation-switch MNN" (Sartor 2025) são nomes dos autores do manuscrito, não dos
   artigos; usar os nomes dos artigos ou dizer "(our label)".
9. Faltam na discussão os limites do Teorema 1 (f com valores em (0,1), caixa fechada; não diz nada sobre tamanho da rede).
10. Mikulincer & Reichman 2025 e Kitouni et al. 2023 estão no `.bib` e não são citados; LMN deveria citar Kitouni 2023
    (artigo original) além de Nolte 2023.
11. Keywords: "Formal verification" é forte para um B&B de uma camada; "Certified monotonicity" é mais honesto.
12. Prop. "Certified output" chama-se *certified* mas o caso de *fallback* é "monótono por construção", não certificado
    pelo B&B — o resumo diz "every network is certified". Escolher: "monotone" para a garantia, "certified" só para o B&B.

### Recomendação

**Major revision.** Se enviado como está, risco de *desk rejection* ≈ 25–30 %, pelos motivos abaixo; se passar ao
*review*, dois referees pedirão M1–M5. Com M1 (experimento que mostre benefício), M2 (parágrafo de posicionamento),
M4 (grades maiores) e M5 (estatística por conjunto, legenda do CD) o artigo tem chance razoável.

Razões mais prováveis de *desk rejection*:
1. Incrementalidade percebida: extensão do próprio artigo NCA 2026 (mesmos métodos, mesma restrição, agora em
   classificação) com um método que não melhora a acurácia dos baselines.
2. Escopo "redes de uma camada oculta com dezenas de unidades" lido como fora do interesse atual da revista.
3. Marcadores `[TODO]`/`[pending]` e ausência de repositório (se não forem resolvidos).
4. Resumo que promete ("We show ... cannot approximate") o que a literatura já tem.

---

## Parte 2 — Edição de texto (mais curto, mais claro)

### Diagnóstico geral

O texto já é enxuto; o problema não é volume, é (a) frases com três orações encadeadas, (b) o mesmo conteúdo em
resumo / bullets da introdução / discussão / conclusão, (c) três ou quatro nomes para a mesma coisa.

**Redundâncias a cortar**
- Conclusão, frases 1–3, repetem o resumo quase literalmente (universalidade; 30 %; "as accurate as ... more accurate
  than"). Deixar na conclusão só o que não está no resumo: limites (uma camada, dezenas de unidades) e próximo passo.
- Discussão, parágrafo 1 ("The constraints form a hierarchy ...") repete a Seção 2 ("Neither constrains the gain between
  the points. ... The sign constraint holds everywhere"). Reduzir a uma frase e manter só "Growth suits the scheme".
- Bullet 1 da introdução e Teorema 1 e primeira frase da conclusão dizem o mesmo três vezes.
- Seção 5.2, último parágrafo ("On real data the sign and cegis constraints give the same accuracy") repete o parágrafo
  anterior ("The sign and cegis constraints are monotone in every fold, at no cost in MCC"). Fundir.

**Terminologia inconsistente (escolher um e aplicar em texto, tabelas e figuras)**
| Conceito | Variantes encontradas | Sugestão |
|---|---|---|
| rede sem restrição | unconstrained (texto), none (tabelas), free (Fig. S2 CD, Tabela 3 "andfree") | prosa: *unconstrained*; coluna Constraint: *none*; corrigir "free" na Fig. S2 |
| a rede | SFNN, single-hidden-layer network, one-hidden-layer, network, model | *SFNN* sempre que for a rede de uma camada; *network* só como pronome |
| método proposto | certified WILCAR, WILCAR cegis DCV, cegis mode, counterexample-guided, "Certified SFNNs (this work)" | definir uma vez: "WILCAR under the cegis constraint (certified WILCAR)"; grupo da Tabela 2: "Counterexample-guided (this work)" — "Certified SFNNs" é enganoso porque as de sinal também são 100 % certificadas |
| restrição de sinal | sign, sign constraints, sign-constrained, non-negative weights, global (código) | *sign* |
| âncora | anchor, point constraint (`eq:point`), reference point, multi (código) | *anchor*; manter "reference point" só quando falar do NCA |
| garantia | certified / monotone / proved monotone / by construction | *monotone on B* para a propriedade; *certified* só para o resultado do B&B |
| erro sintético | MAE (figura), mean absolute error (tabela), "error" (texto) | MAE definido uma vez |

**Jargão a trocar**: "projected by SLSQP onto the anchor constraints" → "re-solved by SLSQP under the anchor constraints";
"searches the class of Theorem 1 directly" → "aims at the class of Theorem 1"; "gain-sign conformity" → "monotonicity";
"operating point" (só na introdução) → "reference point".

### Top 25 edições (original → substituição)

1. **Resumo.** `We train constructive single feedforward neural networks, grown one hidden unit at a time, to respect such gain signs on the whole input domain and to prove it.`
   → `We train constructive single feedforward neural networks (SFNN), grown one hidden unit at a time, to respect such gain signs on the whole input domain, and certify that they do.`

2. **Resumo.** `We show that sign constraints on the weights, the classical way to guarantee monotonicity, cannot approximate simple monotone targets with one hidden layer, whereas networks that are monotone on the domain are universal approximators.`
   → `Sign constraints on the weights, the classical guarantee, are not universal with one hidden layer; we prove that SFNNs monotone on the domain, with a gain margin, are.`

3. **Resumo.** `To find them, gain-sign constraints are imposed at anchor points that grow with the counterexamples of an adversarial search, and the final network is checked by a sound branch-and-bound certificate; if the check fails, a sign-constrained solution is returned, so every network is certified.`
   → `Training imposes the gain signs at anchor points, adds the counterexamples of an adversarial search to them, and ends with a sound branch-and-bound certificate. If the certificate fails, a sign-constrained network is returned, so every network is monotone.`

4. **Resumo.** `and are certified in every fold, whereas only 35--41\% of the unconstrained networks are monotone.`
   → `and are monotone in every fold, whereas only 35\% of the unconstrained networks are.` (ou citar a fonte do 41 %)

5. **Intro §1.** `Properly validated networks may still contradict such gain signs \citep{Sa2026}, and their predictions then cannot be defended to a clinician or a regulator.`
   → `Networks that pass validation may still contradict such gain signs \citep{Sa2026}; their predictions are then hard to defend to a clinician or a regulator.`

6. **Intro §1.** `they fix a specialised architecture in advance.`
   → `they fix the architecture in advance.`

7. **Intro §2.** `Single feedforward neural networks (SFNN), with one hidden layer, remain a natural choice for small and medium tabular data, and constructive algorithms define their number of hidden units by growing the hidden layer`
   → `Single feedforward neural networks (SFNN; one hidden layer) remain a natural choice for small and medium tabular data; constructive algorithms set their size by growing the hidden layer`

8. **Intro §2.** `Such constraints hold where they are imposed. The natural way to impose them everywhere, non-negative (sign-constrained) weights, is not universal for monotone functions with one hidden layer \citep{DanielsVelikova2010,Gupta2016}. This work closes the gap for binary classification:`
   → `Such constraints hold only where they are imposed, and the classical way to impose them everywhere, non-negative weights, is not universal with one hidden layer \citep{DanielsVelikova2010,Gupta2016}. This work closes the gap for binary classification:`

9. **Intro, bullet 1.** `single-hidden-layer sigmoid networks that are monotone on the input box, with a positive gain margin, are universal approximators of continuous partially monotone functions (\cref{thm:ua}); the obstruction lies in the sign parametrisation, not in the depth;`
   → `SFNNs that are monotone on the input box, with a positive gain margin, are universal approximators of continuous partially monotone functions (\cref{thm:ua}): the obstruction is the sign parametrisation, not the depth;`

10. **Intro, bullet 2.** `a counterexample-guided training imposes the gain signs at anchor points that grow with the violations found by an adversarial search, and a repair-and-fallback step makes every returned network certified (\cref{sec:cegis});`
    → `counterexample-guided training imposes the gain signs at anchor points, adds the violations found by an adversarial search to them, and a repair-and-fallback step makes every returned network monotone (\cref{sec:cegis});`

11. **Intro, bullet 4.** `synthetic targets with a known probability and eight real datasets, under nested cross-validation, compare the approach with thirteen alternatives, including Min-Max, constrained monotonic and Lipschitz monotonic networks.`
    → `seven synthetic targets with known probability and eight real datasets, under nested cross-validation, compare the approach with eleven alternatives, including Min-Max, constrained monotonic and Lipschitz monotonic networks.`

12. **§2, após Eq. 2–4.** `The anchor constraint fixes the gain sign at chosen points; with $\A$ the reference point it is the constraint of \citet{Sa2026}, and here $\A$ holds ten $k$-means centres. Neither constrains the gain between the points.`
    → `The anchor constraint fixes the gain sign at chosen points: at the single reference point of \citet{Sa2026}, or, here, at ten $k$-means centres of the training inputs. Neither constraint says anything between the points.`

13. **§2, Fig. 1.** `That costs expressiveness (\cref{fig:surfaces}). The target of \cref{fig:surfaces}, increasing in both inputs, has an L-shaped decision boundary.`
    → `That costs expressiveness. The target of \cref{fig:surfaces} increases in both inputs and has an L-shaped decision boundary.`

14. **§2, Fig. 1.** `Under \eqref{eq:sign} the logit is a sum of ridges increasing in both inputs, the boundary cuts the corner (panel c), and the error is three times that of the unconstrained network, which forms the corner but decreases in some input where the probability is flat and the data carry little information (panel b).`
    → `Under \eqref{eq:sign} the logit is a sum of ridges increasing in both inputs, so the boundary cuts the corner (panel c) and the error triples. The unconstrained network forms the corner (panel b) but decreases in some input where the probability is flat and the data are uninformative.`

15. **§3.1.** `\emph{WILCAR} \citep{Sa2022,Fontes2021} initialises the first unit by linearising the network around an equilibrium point (the mean input) and approximating the optimal linear regressor, and \emph{reuses weights}: the network with $n$ units starts from the one with $n-1$ units plus a new unit.`
    → `\emph{WILCAR} \citep{Sa2022,Fontes2021} initialises the first unit from the linearisation of the network at the mean input, matched to the optimal linear regressor, and \emph{reuses weights}: the network with $n$ units starts from the one with $n-1$ units plus a new unit.`

16. **§3.1.** `the cross-entropy with a Gaussian prior of precision $\lambda_0$ on the weights, by L-BFGS \citep{LiuNocedal1989}; the constrained versions start from that optimum and solve the constrained problem by SLSQP \citep{Kraft1994}, with exact gradients of the gains (Supplementary \cref{S-sup:gradients}).`
    → `the cross-entropy with an $L_2$ penalty (a Gaussian prior of precision $\lambda_0$), by L-BFGS \citep{LiuNocedal1989}. The constrained versions start from that optimum and use SLSQP \citep{Kraft1994} with exact gradients of the gains (Supplementary \cref{S-sup:gradients}).`

17. **§3.2.** `The \emph{cegis} mode (\cref{alg:cegis}), named after counterexample-guided inductive synthesis \citep{SolarLezama2006}, searches the class of \cref{thm:ua} directly. At size $n$ the network is projected by SLSQP onto the anchor constraints \eqref{eq:point}.`
    → `The \emph{cegis} mode (\cref{alg:cegis}), named after counterexample-guided inductive synthesis \citep{SolarLezama2006}, aims at the class of \cref{thm:ua}. At size $n$ the network is re-solved by SLSQP under the anchor constraints \eqref{eq:point}.`

18. **§3.2, acrescentar (posicionamento, M2) após "most violated points."**
    → `Unlike COMET \citep{Sivaraman2020}, which adds SMT counterexamples to the training data, and the MILP-certified networks of \citet{Liu2020}, which penalise violations, the counterexamples enter as hard constraints, there is no penalty weight, and the output is always monotone.`

19. **§3.3.** `Undecided cases (a minimum exactly at zero, or the box budget) go to the fallback. The certificate applies to any SFNN, so it also measures how often unconstrained and mean- or anchor-constrained networks are monotone.`
    → `Undecided cases (a minimum exactly at zero, or the box budget) go to the fallback. The certificate applies to any SFNN, so it also tells how often unconstrained, mean- and anchor-constrained networks happen to be monotone.`

20. **§3.4.** `The criterion needs validation curves that change smoothly with $n$, which weight reuse favours; DCV is therefore used with WILCAR, which is also run with a stratified 80/20 hold-out split and the validation MCC, the selection used by the other methods.`
    → `The criterion needs validation curves that change smoothly with $n$, which weight reuse favours, so DCV is used with WILCAR only. All other methods, and WILCAR as a control, select on a stratified 80/20 hold-out split by validation MCC.`

21. **§4, Synthetic.** `Since the question is what each class of networks can represent, every constructive step keeps the best of three initialisations on these targets.`
    → `Because the question here is what each class of networks can represent, each constructive step keeps the best of three initialisations.`

22. **§4, Metrics.** `Gain-sign conformity is measured on the whole domain: the share of (point, constrained input) pairs with a wrong-sign gain on $10^4$ uniform points of $\B$, and the outcome of the certificate.`
    → `Monotonicity is measured on the whole domain: the share of (point, constrained input) pairs with a wrong-sign gain over $10^4$ uniform points of $\B$ (viol.), and the outcome of the certificate (cert.).`

23. **§5.2.** `The sign and cegis constraints are monotone in every fold, at no cost in MCC. In cegis mode, 97.3\% of the networks were certified by the branch and bound and 2.7\% (15 of 555) were returned by the fallback.`
    → `Networks under the sign and cegis constraints are monotone in every fold, at no cost in MCC; in cegis mode 97.3\% were certified by the branch and bound and 2.7\% (15 of 555) came from the fallback.`
    (e fundir o parágrafo "On real data the sign and cegis constraints give the same accuracy..." com este.)

24. **§5.4 Discussion, parágrafo 1.** `The constraints form a hierarchy: the mean constraint fixes the sign of the global effect of each input, the anchor constraint and the reference-point constraint of our earlier work fix it at chosen points, and the sign and cegis constraints fix it everywhere. The weaker constraints often yield monotone networks, but only the certificate can tell which. Sign constraints give the guarantee through the parameters, at the price shown in \cref{fig:surfaces} and \cref{tab:synth}; the cegis mode gives it through verification and keeps the flexibility of \cref{thm:ua}. Growth suits the scheme:`
    → `The mean and anchor constraints often yield monotone networks, but only the certificate can tell which. Sign constraints give the guarantee through the parameters, at the price of \cref{fig:surfaces} and \cref{tab:synth}; cegis gives it through verification and keeps the flexibility of \cref{thm:ua}. Growth suits the scheme:`

25. **§6 Conclusions (inteira).** `Single-hidden-layer networks that are monotone on their domain are universal approximators of continuous partially monotone functions, while sign-constrained ones are not. Counterexample-guided constraints, a sound certificate and a sign-constrained fallback reach that class with constructive networks. Every network returned is certified; on synthetic targets the certified networks have 30\% lower error than sign-constrained ones, and on real data they are as accurate as unconstrained networks and CMNN, and more accurate than Min-Max and LMN. Extending the certificate to deeper constructive networks is a natural next step.`
    → `Constructive SFNNs can be made monotone on the whole input domain, with a certificate, at no cost in accuracy on real data and at a third less error than sign constraints on synthetic targets. The approach is limited to one hidden layer, tens of hidden units and tens of thousands of rows by the certificate and the quadratic programs; extending the certificate to deeper constructive networks is the natural next step.`

Outras edições rápidas: "Min-Max" vs "Min-Max Net" (Tabela lit) → "Min-Max"; "k-means centres" definir como "of the
training inputs" na primeira ocorrência; "the event of interest" (§2) pode sair; "all imposed as hard constraints of the
training problem" → "all as hard constraints"; "operating point" (intro) → "reference point".

---

## Parte 3 — Figuras, tabelas, algoritmo e legendas

**Figura 1 (`fig_surfaces`)**
- A legenda interna da figura tem "red arrow: unit not monotone on its own"; a legenda LaTeX não menciona setas
  vermelhas. Incluir: "red arrows: units that decrease in some input".
- Painel (a) intitulado "true P(Y=1|x)" na figura e "(a) Target" na legenda — alinhar.
- Em (e) as 8 setas se sobrepõem perto da origem (unidades quase nulas): o leitor não vê "8 unidades". Dizer na legenda
  "units of small norm overlap at the origin" ou usar escala log/normalizar.
- Faltou (item 8 das notas): dizer que nesse alvo a restrição *mean* é inativa e coincide com (b) — evita que o
  co-autor leia (c) como "o método do NCA". Mesmo cuidado para "(c) sign constraints" → "(c) sign constraints on the
  weights \eqref{eq:sign}".
- "MAE" nos títulos vs "mean absolute error" na legenda: definir MAE na legenda.

**Figura S1 (`fig_capacity`)**: ok; marcadores abertos/fechados pouco distinguíveis em "wave" (todos parecem cheios).
Legenda usa "unconstrained"; consistente com o texto. Checar se de fato os cinzentos de "wave" são certificados (a
legenda afirma "(non-monotone) unconstrained ... on and" e não diz nada sobre wave).

**Figura S2 (CD)**: (i) rótulos "RIXM free", "WILCAR free DCV" → "unconstrained"/"none"; (ii) com CD = 6.45 uma única
barra une os 13 métodos — a legenda deve dizer "the Nemenyi test separates no pair at this N; Holm's procedure against
the best (text) is more powerful and separates Min-Max and LMN", senão a figura contradiz o texto; (iii) considerar
remover a figura e manter só os ranks na Tabela 2.

**Tabela 1 (datasets)**: `[pending]` em LOAN; a nota de rodapé usa "constructive SLSQP fits" — jargão; "the cost of the
constrained fits". Falta dizer o tamanho do conjunto de teste oficial (61 em Heart Disease) que o texto usa.

**Tabela 2 (main_mcc)**
- Grupo "Certified SFNNs (this work)" é enganoso: as linhas "sign" também são 100 % monótonas. Renomear "Counterexample-
  guided (this work)" / "Other SFNNs" / "Monotone baselines".
- Coluna "Constraint" mistura modos (none/mean/sign/cegis), "monotone" (árvores) e "structural" (arquitetura); "sign"
  para LR significa coeficientes com sinal limitado — ok mas dizer na legenda "LR: coefficient signs".
- "Selection" = DCV/hold-out: para os baselines é seleção de hiper-parâmetros, não de unidades; legenda diz "Selection
  of the number of hidden units".
- "Rank" sobre 13 métodos aqui e sobre 27 na Tabela S main_mcc_full com o mesmo nome.
- Negrito em empates (BCD: LR 0.954 vs RIXM 0.954).

**Tabela `modes`**: boa. "cert. (%)" para `sign` é "by construction" (as notas dizem que `cert_status` desse modo estava
errado no arquivo) — dizer "100 (by construction)". Tempo mediano (22 s) aqui vs médias (até 4 518 s) na Tabela S size:
um leitor que compare as duas vai estranhar; acrescentar "median" no cabeçalho e uma frase no texto da Tabela S size.
Os números gerados em `numbers.tex` (MeanCertified 78.0 %, MeanViolDomain 2.55 %) divergem da tabela (75 %, 3.0 %).

**Tabela 3 (synthetic)**: "cert." aqui é "share of final models monotone ... certified or by construction" — mesmo nome,
definição ligeiramente diferente da Tabela `modes`. Em negrito, Min-Max/LGBM/LMN vencem a média — o texto do resumo
(30 % vs sinal) deveria admitir isso em uma oração.

**Tabela `literature`**: "Certified MNN" e "Activation-switch MNN" são rótulos dos autores; dizer "(label ours)" ou usar
os nomes dos artigos. sd 0.0 em Heart Disease para 4 linhas: explicar (métodos determinísticos; teste de 61 casos).
"LMN 65.44 (0.03)" com duas casas enquanto o resto tem uma.

**Algoritmo 1**: cobre só um passo construtivo; reparo e *fallback* (as partes que garantem a Prop. "Certified output")
estão apenas no texto. Acrescentar duas linhas (`after the final size: run certificate; repair ≤ R_rep; else fallback`)
ou renomear a legenda "Constructive step n (repair and fallback after the final refit: text)". Dizer que os modelos
intermediários da DCV não são certificados.

**Algoritmo S1 (B&B)**: linha 2 "pre-activation space when n<p and the input-space search exhausts N/4 boxes" é a única
menção à regra de troca de espaço — mover para o texto da seção S2. Linha "If c_j ≥ 0 for all j return certified"
corresponde à Prop. sign — citar.

**Algoritmo S2 (DCV)**: DCV-1SE não é usado em lugar nenhum — remover as linhas 9–10.

**Tabela S settings**: ELM-C descrito mas não reportado; "Restarts R=100" conflita com o uso de R para R_cex/R_rep —
renomear.

**Relação com a restrição do NCA (ponto de referência)**: o texto está correto (NCA = ganho por diferenças finitas em um
ponto de referência, a média das entradas; o manuscrito trata isso como "anchor com A = ponto de referência"). Mas em
três lugares a relação pode ser mal lida: (1) Tabela `modes` não tem linha "reference point (Sá et al. 2026)" — o leitor
pode achar que "mean" é o NCA; (2) a introdução diz "the constraint may be repeated at several points" — isso é uma
sugestão do NCA, não algo feito lá; dizer "could be repeated"; (3) na discussão "the anchor constraint and the
reference-point constraint of our earlier work" lista como duas coisas o que a Seção 2 definiu como uma.
