# Revisão matemática e de consistência artigo ↔ código (Neural Networks)

Arquivos revisados: `paper/main.tex`, `paper/supplement.tex`, `revision/{sfnn,methods,certify,protocol,mono_baselines,data,cert_bench}.py`,
`/tmp/claude-0/nca.txt` (artigo NCA, Seção 4.1). Data: 2026-10-01. Nenhum arquivo além deste foi alterado.

Veredicto geral: Lema 1, Proposição 1, Teorema 1, Proposição 2, Teorema 2 e as cotas (a)–(d) estão **corretos**; os
gradientes do Suplemento S4 batem com `sfnn.gains_and_grad`; as constantes de σ'' (máximo √3/18 em −t*, t* = ln(2+√3)) e o
invólucro `_d2_range` foram verificados numericamente (0 violações em 20 000 intervalos). Os problemas estão na
**Proposição 3** (prova com lacuna e hipótese que o código não cumpre), na redação de "exato", e em vários detalhes de
algoritmo cujo texto não corresponde ao código. Numerados por gravidade.

---

## 1. Proposição 3 (terminação): prova com lacuna e hipótese "breadth-first" que o código não satisfaz — GRAVE

**Local.** `main.tex` l. 277–279; `supplement.tex` l. 102–110; `certify.py` l. 129–131 (`_bb_one`) e l. 176–180 (`_bb_z`).

**(a) Lacuna na prova.** A prova afirma que "the splitting rule halves a coordinate of largest weighted width, so under
breadth-first processing the radius of every box at depth k tends to zero". Isso não decorre da regra: o peso
max{|G⁻_d|,|G⁺_d|} é o invólucro de ∂_d h na caixa e pode ficar arbitrariamente pequeno (por exemplo quando ∂_d h ≡ 0 ao
longo do segmento restante na coordenada d, o que ocorre com unidades duplicadas de sinais opostos ou em geral quando
σ'' se cancela), de modo que uma coordenada com Σ_j|c_j W_jd| > 0 pode nunca ser escolhida e o raio da caixa **não** tende a
zero. O argumento "at finite depth one of them lies inside Q*" também depende do raio → 0.

A proposição é verdadeira, mas por outro argumento, que recomendo substituir na prova. Sejam w_d = max{|G⁻_d|,|G⁺_d|} e
M = max_d r_d w_d na caixa. Como os invólucros intervalares são monótonos por inclusão, r_d w_d é não-crescente ao longo de
qualquer ramo. Ao longo de um ramo infinito, se M ≥ c > 0 sempre, cada escolha da coordenada d divide r_d por 2 e
w_d ≤ Γ_d := (√3/18)Σ_j|c_j W_jd|, logo após k_d escolhas r_d w_d ≤ 2^{−k_d}Γ_d < c: cada coordenada só pode ser escolhida um
número finito de vezes, contradição. Logo M → 0 e Σ_d r_d w_d ≤ pM → 0 ao longo de todo ramo. Então:
(i) se min_B h = δ > 0, o termo do valor médio h(m) − Σ_d r_d w_d ≥ δ − pM > 0 em profundidade finita em todo ramo; a árvore
é finitamente ramificada sem ramo infinito, logo finita (König) — **termina em qualquer ordem de processamento**;
(ii) se min_B h < 0, defina o ramo Q_0 = B, Q_{k+1} = metade de (a face de) Q_k que contém um minimizador x_k de h sobre Q_k;
como a face preserva o mínimo, min_{Q_k} h = min_B h < 0, e |h(m_k) − h(x_k)| ≤ Σ_d r_d w_d → 0, logo h(m_k) < 0 em
profundidade finita k*; com processamento breadth-first todas as caixas de profundidade ≤ k* são processadas em tempo
finito. (O teste de monotonicidade é compatível porque a face contém um minimizador da caixa.)

Texto sugerido (substituir o parágrafo da prova no suplemento):
> *Proof.* Write $w_d=\max\{|G^-_d|,|G^+_d|\}$ and $\Gamma_d=\frac{\sqrt3}{18}\sum_j|c_jW_{jd}|\ge w_d$. Interval
> enclosures are inclusion-monotone, so $r_dw_d$ is non-increasing along every branch of the search tree. Along an
> infinite branch, $M_k=\max_dr_dw_d\to0$: otherwise $M_k\ge c>0$ for all $k$, and since each choice of coordinate $d$
> halves $r_d$ while $w_d\le\Gamma_d$, every coordinate can be chosen only finitely often while $r_dw_d\ge c$, which
> contradicts infinitely many choices among $p$ coordinates. Hence $\sum_dr_dw_d\le pM_k\to0$. If $\min_\B h=\delta>0$, the
> mean-value term of \eqref{M-eq:lbx} is at least $\delta-pM_k>0$ at finite depth on every branch; the tree is finitely
> branching with no infinite branch, hence finite, and the search terminates in any processing order. If $\min_\B h<0$,
> let $Q_0=\B$ and let $Q_{k+1}$ be the half of (the face replacing) $Q_k$ that contains a minimiser $x_k$ of $h$ over
> $Q_k$; faces preserve the minimum, so $\min_{Q_k}h=\min_\B h<0$, and $|h(m_k)-h(x_k)|\le\sum_dr_dw_d\to0$, so the centre
> $m_k$ is a counterexample at some finite depth $k^\ast$. Breadth-first processing reaches depth $k^\ast$ after finitely
> many boxes. $\square$

**(b) O código não é breadth-first.** `_bb_one` retira as **últimas** `take` caixas da lista (`L[-take:]`) e acrescenta os
filhos ao fim (`np.vstack([L, l, l2])`): é uma busca em profundidade por lotes de 4096 (BFS só enquanto a fronteira cabe
num lote). No espaço de pré-ativação (`_bb_z`) a ordem é best-first por cota inferior (heap). Com min_B h < 0 e uma
tangência (h = 0) noutra região, a busca LIFO pode gastar o orçamento no ramo da tangência e devolver `unknown` (→ fallback
desnecessário) onde a BFS encontraria o contraexemplo. Correção mínima no código (uma linha): `l, u = L[:take].copy(),
U[:take].copy(); L, U = L[take:], U[take:]` (FIFO). Alternativa no texto: declarar a ordem usada e reformular a
Proposição 3 como "termina em qualquer ordem quando min_B h > 0; com min_B h < 0 a busca breadth-first (ou qualquer ordem
justa) encontra um contraexemplo".

**(c) Enunciado.** O Algoritmo S1 tem orçamento N, logo sempre termina; o conteúdo da proposição é **completude**.
Sugestão para `main.tex` l. 278: "If $\min_\B h_i\neq0$ and boxes are processed breadth-first, the input-space search
returns \textsc{certified} or \textsc{counterexample} after finitely many boxes (for a sufficient box budget)." E em
l. 186–187 e no fim da prova do Teorema 1 (supl. l. 78): "complete for such networks, given a sufficient budget".

## 2. "Exact up to floating-point rounding" e soundness em ponto flutuante — IMPORTANTE

**Local.** `main.tex` l. 385–386 ("the certificate is exact up to floating-point rounding"); l. 120 ("exact in
pre-activation space"); `certify.py` l. 108–114, 139, 184–186.

(a) "Exact" é a palavra errada: o certificado é **sound** (Teorema 2) e **completo** apenas sob as hipóteses da
Proposição 3 e do orçamento; casos "unknown" existem (o próprio texto os envia ao fallback). Substituir por: "the
certificate is sound up to floating-point rounding and decides every network whose gain margin $\min_\B h_i$ is non-zero,
given the box budget".

(b) As cotas são avaliadas em precisão dupla **sem arredondamento dirigido** (nenhum uso de `np.nextafter`/intervalos
com arredondamento para fora), e o descarte é `LB ≥ 0` estrito. "Up to rounding" é portanto honesto, mas convém
(i) dizê-lo explicitamente no suplemento (Seção S2) e (ii) opcionalmente tornar o certificado robusto a arredondamento
com uma folga: descartar só se `LB ≥ κ` com κ ≈ 1e-12·(Σ_j|c_j|) (em `_box_bounds`/`_bb_one` l. 139: `open_ = LB < kappa`;
em `_bb_z` l. 179 e 198 idem). Como os modelos cegis têm margem ε_p, a folga não custa nada.

(c) No espaço de pré-ativação, o descarte por "R não encontra Z" depende de `linprog(..., method="highs")` devolver
`status == 2`. A declaração de inviabilidade do HiGHS usa uma tolerância primal (≈1e-7); uma caixa que intersecta Z
numa fatia mais fina que isso pode ser descartada indevidamente. Soundness vale "até a tolerância do LP", não só "até o
arredondamento". Correção sound no código (`certify.py` l. 184): relaxar as desigualdades, `b_ub=np.concatenate([zu - b1,
b1 - zl]) + 1e-6` (alarga R; inviabilidade do LP relaxado implica inviabilidade do original). E acrescentar ao texto da
Seção S2: "boxes are discarded by the LP only when the relaxed system (tolerance $10^{-6}$) is infeasible".

(d) "Exact in pre-activation space" (l. 120, 265–266): o mínimo de φ sobre a caixa R é exato, mas R ⊇ R ∩ Z, logo a cota
usada é uma relaxação. Sugestão l. 120: "with exact box bounds in pre-activation space"; l. 266: "so that its minimum on a
box $R$ is computed exactly (a lower bound for the reachable part $R\cap Z$)".

## 3. Quando a busca vai ao espaço de pré-ativação — divergência texto principal ↔ código

**Local.** `main.tex` l. 264–265: "When $n<p$ the search runs in pre-activation space". Código (`certify.py` l. 212–214):
quando n < p a busca em espaço de entrada roda primeiro com orçamento N/4 (50 000 caixas) e só se devolver `unknown`
passa ao espaço de pré-ativação (3000 LPs). O Algoritmo S1 (comentário da l. 131 do suplemento) descreve corretamente.
Substituir em `main.tex`: "When $n<p$ and the input-space search has not decided within a quarter of the box budget, the
search continues in pre-activation space, where ...". Também na Discussão (l. 381–382, "the pre-activation search works in
dimension n") deixar claro que isso só vale quando n < p.

## 4. Busca adversária: número de pontos e quantos contraexemplos viram âncoras

**Local.** `main.tex` l. 221–223; `supplement.tex` l. 248; `certify.py` l. 65–85 (`adversarial`).

(a) O texto diz "every point with $h_i<0$ becomes an anchor"; o código devolve **um** ponto por entrada restrita por
rodada (o minimizador mais negativo entre todos os starts, `best`). O Algoritmo 1 (l. 242) já está correto ("$\hat x_i$").
Corrigir l. 223: "the most negative minimiser $\hat x_i$ of each $h_i$, when $h_i(\hat x_i)<0$, becomes an anchor and the
network is re-solved".

(b) Tabela S-settings: "adversarial search from 16 starting points". Código: 16 pontos uniformes **mais** os 8
(`starts // 2`) pontos de treino com menor $h_i$ → 24 starts. Corrigir: "16 uniform starting points plus the 8 training
points with the smallest $h_i$".

(c) Os starts são L-BFGS-B com `maxiter=200` (não documentado; opcional).

## 5. Repair e fallback: três diferenças em relação ao texto

**Local.** `main.tex` l. 226–228, 282; `methods.py` l. 360–386 (`certify_and_repair`), l. 388–407 (`_sign_fallback`).

(a) O repair só ocorre enquanto `status == "counterexample"`; um `unknown` (orçamento ou mínimo em zero) vai **direto** ao
fallback, sem reparos. O texto l. 282 ("Undecided cases ... go to the fallback") concorda, mas l. 227–228 ("If the
network is still not certified after $R_{\rm rep}$ repairs") sugere que sempre há $R_{\rm rep}$ tentativas. Sugestão:
"If the certificate returns a counterexample, it becomes an anchor and the network is re-solved (\emph{repair}), up to
$R_{\rm rep}$ times; if the certificate is still not \textsc{certified}, or returns \textsc{unknown}, the weights with
$c_{ij}<0$ are set to zero ...".

(b) Cada reparo re-executa também as rodadas adversárias (`_cegis_loop`, l. 378) antes de certificar de novo — mencionar
("followed by the counterexample rounds of Algorithm 1").

(c) Em `_solve(..., prev=None, warm=v)` (l. 376): se o warm start falha, os reinícios usam `_init_theta(n, p, None)`, ou
seja, uma reinicialização completa (para WILCAR, primeira unidade da linearização + aleatórias), perdendo a rede
construída. Decisão de projeto aceitável, mas o texto "warm-started" não a cobre; ou passar `prev` (a rede de tamanho
n−1 não está disponível nesse ponto) ou declarar.

(d) Proposição 2 está correta **para WILCAR-C/RIXM-C em modo cegis**: `_sign_fallback` zera no fim qualquer produto
residual negativo (l. 405–406), de modo que \eqref{eq:sign} vale exatamente e `certify` devolve `certified` de imediato.
Vale apenas para a rede final do `refit`; as redes do laço de seleção não são certificadas (coerente com o texto).

## 6. ELM / ELM-C: o que o texto não diz

**Local.** `main.tex` l. 208–209, 315–316; `supplement.tex` l. 251; `methods.py` l. 155–157, 254, 415–441, 448.

(a) Para ELM, `cegis` é False (l. 254): em modo cegis o ELM-C usa as restrições de sinal (`_global_constraints(elm=...)`),
isto é, **ELM-C "cegis" = ELM-C "sign"**. A frase "WILCAR, RIXM and ELM, unconstrained and with the four constraints" é
imprecisa para ELM (três restrições). Acrescentar: "(for ELM the cegis mode reduces to the sign constraint, since the
hidden layer is not trained)".

(b) Modelos de saída diferentes: ELM livre = saída **linear** por mínimos quadrados com previsão `clip(z, 0, 1)`
(l. 425, 448); ELM-C = saída **logística** com BCE por SLSQP a partir de `pinv(Hb) @ 4(y−0.5)` (l. 427–438). A comparação
ELM vs ELM-C confunde restrição com modelo de saída. Declarar na Tabela S-settings: "ELM: linear output by least squares,
probabilities clipped to [0,1]; ELM-C: logistic output (BCE) solved by SLSQP".

(c) Em modos mean/anchor o ELM-C restringe ganhos de uma rede ReLU (subgradiente `D`, `sfnn.py` l. 144) — restrição
não-suave no SLSQP; mencionar ou aceitar.

## 7. Modo `sign` (global): viabilidade só até 10⁻⁶ — Proposição 1 não vale exatamente para as redes `global`

**Local.** `methods.py` l. 340 (`viol <= feas_tol`, 1e-6), l. 221–250; `certify.py` l. 124.

No modo global, `_solve` aceita $c_{ij}\ge-10^{-6}$. A rede devolvida pode ter produtos ligeiramente negativos; então
`np.all(c >= 0)` falha e o branch-and-bound completo roda (podendo, em princípio, achar um contraexemplo de ordem 1e-9 ou
devolver `unknown`). O fallback já faz o zeramento exato (l. 405–406); fazer o mesmo no fim de `_solve` quando
`constraint_mode == "global"` (ou em `refit`) garante \eqref{eq:sign} exatamente e torna a afirmação "sign ... monotone in
every fold" verdadeira por construção. Alternativa no texto: "sign constraints are enforced to the SLSQP tolerance and
then verified by the certificate".

## 8. Descrição do trabalho anterior (NCA) e distinção do modo `mean` — CORRETA, com uma ressalva de redação

**Local.** `main.tex` l. 107–111, 160–162, 376–377; `nca.txt` l. 372–404 (Seção 4.1).

O NCA impõe $s_j G_j \ge \epsilon$ com $\epsilon=0.05$ e $G_j=(y(x_{ss}+\delta e_j)-y(x_{ss}))/\delta$, $\delta=0.1$, em
**um** ponto de referência $x_{ss}$ = média das entradas (de teste), e diz que a restrição "can be extended by enforcing
the sign constraint at multiple representative points". A descrição do artigo (l. 107–111) é fiel. O modo `mean` (média
dos ganhos sobre ≤1000 pontos de treino) é corretamente apresentado como distinto (média dos ganhos ≠ ganho na média;
l. 376–377 está certo). Ressalvas:

(a) l. 161–162 "with $\A$ the reference point it is the constraint of \citet{Sa2026}": lá o ganho é uma **diferença
finita progressiva de passo 0.1** (um secante de 10 % do domínio), não a derivada, e com ε = 0.05. Sugestão: "with $\A$ a
single reference point (and the derivative replaced by the forward difference of step 0.1 used there) it is the
constraint of \citet{Sa2026}".

(b) A inicialização da primeira unidade do WILCAR-C (`methods.py` l. 132–150, `start`) resolve a linearização sob a
restrição $s_i\,\sigma'(w^\top x_M+\beta)\,w_i\ge\varepsilon_m$ no ponto médio $x_M$ — exatamente uma restrição de ponto de
referência à la NCA, usada em **todos** os modos e com ε_m = 0.05. Não está documentado; acrescentar uma frase na Seção 3.1
ou na Tabela S-settings ("WILCAR-C: the linearised first unit is solved under the gain-sign constraint at the mean
input, margin $\varepsilon_m$").

## 9. Complexidade: O(np) por caixa, exponencial em min{n,p}

**Local.** `main.tex` l. 281–282.

(a) O(np) por caixa está correto no espaço de entrada (`_box_bounds`: produtos (B,n,p); o código chama três vezes por
lote, teste de monotonicidade ×2 + cota, ainda O(np)). No espaço de pré-ativação cada caixa custa **um LP** em p variáveis
com 2n restrições, não O(np). Sugestão: "Bounding a box costs $O(np)$ operations in input space and one linear program in
pre-activation space."

(b) "Exponential in min{n,p} in the worst case": o número de caixas para resolver o domínio a resolução ε cresce como
ε^{−p} no espaço de entrada e ε^{−n} no de pré-ativação; como a busca em pré-ativação só é usada quando n < p (e após N/4
caixas em x), a frase é aceitável se reescrita: "grows exponentially with the dimension of the space searched ($p$ in
input space, $n$ in pre-activation space) at a fixed resolution".

## 10. Teorema 1: prova correta; três retoques

**Local.** `main.tex` l. 180–187; `supplement.tex` l. 60–78.

(a) Ordem das escolhas: fixar primeiro δ (sup_B|F_δ−F| < ε), depois η < ε/(|I|+1) (digamos), e só então a rede de Hornik;
o texto escolhe "η and δ small enough" no fim, o que está certo mas vale explicitar que $z_o$ depende de η.
(b) Hornik (1991, Thm 3) exige ψ não-constante com derivadas limitadas até ordem m; a logística satisfaz (ok). A classe
de Hornik não inclui o viés de saída $c$; a densidade de um superconjunto segue trivialmente — uma meia-linha basta
("adding the output bias $c$ enlarges the class").
(c) A última frase ("Finite termination of Algorithm S1 follows from Prop. 3") não pertence à prova do Teorema 1 e
depende do orçamento; mover para um `Remark` após a Proposição 3: "By Theorem 1 the networks of that class have
$\min_\B h_i>0$, so the certificate decides them (Proposition 3)".
(d) Abstract l. 68–69 "We show that sign constraints ... cannot approximate simple monotone targets": a não-universalidade
é citada (Daniels & Velikova 2010) e ilustrada (Fig. 1), não demonstrada aqui. Sugestão: "We recall that ... and show on
simple targets that ...".

## 11. Definição 1, Lema 1, Proposição 1, Teorema 2 e cotas — verificados

* Def. 1 "iff" para $p_\theta$: correto ($C^1$ e $\B$ convexo; na fronteira $x_i=1$ usar o quociente à esquerda). Opcional:
  "(by integrating $g_i$ along the segment, and taking one-sided difference quotients)".
* Lema 1 e $c_{ij}$: batem com `sfnn.gains` e `certify.coefficients`.
* Prop. 1: correta ("on $\R^p$" é mais forte que a Def. 1, que é em $\B$ — fine).
* Cotas (a)–(d): corretas. $[\alpha_j,\beta_j]$ = `zc ∓ zr` (l. 105–106); mínimo de σ' nos extremos, máximo em
  `clip(0, zl, zu)` (l. 108; `np.clip(0.0, zl, zu)` com escalar e vetores funciona, verificado); invólucro de σ''
  (`_d2_range`, verificado numericamente); forma do valor médio e teste de monotonicidade (l. 114, 136–137) corretos —
  a atribuição sequencial `u = where(inc, l, u); l = where(dec, u, l)` está certa porque `inc` e `dec` são disjuntos.
* Teorema 2: a prova cobre face/divisão/descarte e o LP; contraexemplos são sempre pontos de $\B$ (`mid`, ou `clip(r.x)`).
  Falta apenas a ressalva do item 2(c).
* Gradientes S4: idênticos a `gains_and_grad` (termos $t_2$, $s_2$, $[i=l]$). "Checked against finite differences in the
  test suite": sim (`test_gain_param_grad`, `test_elm_gain_grad`).
* Tabela S-synthdef "wave": ganhos do logit em [1.5, 10.5] — correto ($6\pm4.5\cos$).

## 12. Protocolo / hiperparâmetros: correspondências e pequenos desvios

Conferidos e **corretos**: hold-out 80/20 estratificado com MCC de validação e patience (30/15) (`select_holdout`);
DCV com K = 4 (`k_in=4`), soma das BCE de validação, J = ∞ se algum fold inviável, patience, regra 1-SE com √K·sd (ddof=1);
refit no conjunto de treino externo (hold-out e DCV); L-BFGS(-B) 1000 it. → SLSQP 1000 it., ftol 1e-6, viabilidade 1e-6;
ε_m = 0.05 (≤1000 pontos), ε_p = 1e-3, λ₀ = 0.3 com penalidade λ₀/N em W e v (sem vieses, como na Eq. loss); 10 centros
k-means (`n_init=4`); R_cex = 10, R_rep = 20; orçamentos 2·10⁵ / 3000; 3600 s por laço; `--starts 3` só nos sintéticos
(seleção pelo objetivo regularizado); grades dos baselines e Adam 5e-3/256/patience 50/1000 épocas; 168 redes do cert_bench
(7×3×4×2).

Desvios menores:
* l. 208 "RIXM re-initialises all weights at every size \citep{Glorot2010}": a inicialização é $N(0,1/\text{fan-in})$
  (`W1 ~ N(0,1/p)`, `W2 ~ N(0,1/n)`), não Glorot. Citar LeCun et al. (1998) ou escrever "Gaussian weights with variance
  $1/\text{fan-in}$".
* l. 320 "from their authors' packages" vale para CMNN e LMN; Min-Max é implementação própria (`_minmax`). Escrever
  "CMNN and LMN from their authors' packages; Min-Max re-implemented".
* Tabela S-settings "R = 100 random restarts": `max_reinit=100` inclui o warm start (99 reinícios aleatórios).
* l. 324–325 "share of pairs with a wrong-sign gain on 10⁴ uniform points": medido por diferença finita central de passo
  10⁻³ com tolerância −10⁻⁹ (`monotonicity`), para todos os métodos; dizer "finite-difference gain".
* Tabela S-hyper LGBM: `min_child_samples=5`, `subsample_freq=1` não listados.
* Supl. S-signals LOAN: o vetor do código é `[-1, +1, -1, -1, +1]` (−,+,−,−,+), que não corresponde à ordem "três (−) depois
  dois (+)" da tabela — o TODO já existe; resolver antes da submissão.
* Prop. 2 só é invocada para a rede final; no laço de seleção as redes cegis podem ser inviáveis/não certificadas (ok).

## 13. Sugestões de texto consolidadas (main.tex)

* l. 120: "and with exact box bounds in pre-activation space".
* l. 223: "the most negative minimiser $\hat x_i$ of each $h_i$, when $h_i(\hat x_i)<0$, becomes an anchor".
* l. 226–228: ver item 5(a).
* l. 264–265: "When $n<p$ and the input-space search has not decided within a quarter of the box budget, the search
  continues in pre-activation space".
* l. 278: "... the input-space search returns \textsc{certified} or \textsc{counterexample} after finitely many boxes".
* l. 281: "Bounding a box costs $O(np)$ operations in input space and one linear program in pre-activation space. The
  number of boxes grows exponentially with the dimension of the space searched ($p$ or $n$) at a fixed resolution, but ...".
* l. 385–386: "The guarantee holds on the training box $\B$; the certificate is sound up to floating-point rounding and
  the feasibility tolerance of the linear programs, and it decides every network with a non-zero gain margin given the
  box budget."
