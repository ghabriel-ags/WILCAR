# Auditoria numérica do manuscrito (main.tex + tabelas + supplement.tex)

Data: 2026-10-01. Fonte: `results/revision/results_{main,cegis,global,multi,lit,syn,syn_cegis,syn_global}.jsonl`
e `cert_bench.jsonl`, recomputados com pandas/scipy de forma independente de `make_tables.py`/`analyze.py`.
Convenção: média sobre folds por dataset, depois média sobre datasets, salvo indicação. O `main.pdf` (23:46) é
posterior às tabelas (23:38) e aos resultados (23:19) — está sincronizado com as fontes.

Contagens de base (verificadas): main 3330 linhas, 18 métodos × 185 folds (7 datasets × 5 reps × 5 folds + Loan
2 reps × 5 folds = 185); cegis/global/multi 740 = 4 configurações × 185; lit 270 = 18 × 3 datasets × 5 seeds
(1 fold); syn 3150 = 18 × 7 alvos × 25 folds; syn_cegis/syn_global 525 = 3 × 175. Nenhuma linha com `error`.

## 1. Discrepâncias e pontos a corrigir

Nenhum número está errado em sentido estrito; os itens abaixo são imprecisões de reporte, denominadores
inconsistentes ou ambiguidades que um revisor pode apontar.

1. **Sec. 5.2, Wilcoxon WILCAR certificado vs. livre, "p = 0.07".** Recomputado: p = 0.070 com
   `zero_method="zsplit"` (o que `analyze.py` usa), mas p = 0.091 com o default do SciPy (`wilcox`) e p = 0.051
   com `pratt`. A diferença importa porque **103 das 185 diferenças de MCC são exatamente zero** (ALG 15, BCO 15,
   BCD 18, HF 16, PIM 13, HD 23, COM 3, LOAN 0 — folds de teste pequenos com predições idênticas). As outras três
   comparações são robustas ao método (CMNN 0.20/0.20/0.20; LMN 1.8e-6/2.0e-6/1.7e-6; Min-Max 3.1e-5/1.7e-5/3.1e-5).
   Correção sugerida: em "Metrics" (Sec. 4) acrescentar "…Wilcoxon signed-rank test over outer folds (zero
   differences split between the ranks, Pratt–zsplit)" ou, mais simples, "(p = 0.07, 103 ties)". O valor em si
   está correto sob a convenção usada.

2. **Sec. 5.2, "22 of 740 selection loops reached the one-hour budget".** Correto para a tag `cegis` inteira, mas
   os 740 incluem 185 corridas de ELM-C em modo cegis, que não aparecem em nenhuma tabela; a frase anterior usa o
   denominador 555 (WILCAR-C DCV + WILCAR-C hold-out + RIXM-C hold-out). Os 22 estouros são todos de WILCAR-C
   (BCD/DCV 14, LOAN/DCV 4, LOAN/hold-out 4); ELM-C e RIXM-C têm 0. Sugestão: "22 of the 555 selection loops"
   (ou "22 of the 185 loops of certified WILCAR with DCV reached the budget: 14 on BCD, 4 on Loan", já que
   a Tabela modes é só WILCAR/DCV — ver item 3).

3. **Sec. 5.2, "97.3 % … 2.7 % (15 of 555) were returned by the fallback"** vs. **Tabela modes (cert. = 100 para
   cegis)**. Ambos corretos, mas com populações diferentes: a tabela é WILCAR/DCV (185 folds, 3 fallbacks = 1.6 %);
   o texto agrega WILCAR-C DCV (3), WILCAR-C hold-out (7) e RIXM-C hold-out (5). Os 15 fallbacks: Loan 14 (RIXM-C
   HO 5, WILCAR-C DCV 3, WILCAR-C HO 6) e BCD/WILCAR-C HO 1. Sugestão: "Over the 555 cegis networks of the three
   configurations (WILCAR with DCV and hold-out, RIXM with hold-out), 97.3 % were certified by the branch and
   bound and 2.7 % (15, fourteen of them on Loan Defaulter) were returned by the fallback."

4. **Sec. 1, "compare the approach with thirteen alternatives".** A Tabela main_mcc tem 13 métodos no total
   (2 certificados + 11 alternativas); a tag `main` tem 13 métodos distintos que não são SFNN restritas
   (CMNN, ELM, LGBM, LGBM-C, LMN, LR, LR-C, MINMAX, MLP, RIXM, WILCAR, XGB, XGB-C), mas a tabela completa do
   suplemento tem 26 linhas. Sugestão: "with eleven alternatives (Table 2; twenty-six configurations in the
   Supplementary Material)" ou "with thirteen other methods" com remissão à tabela completa.

5. **Sec. 5.2, "AUC and Brier score lead to the same ranking".** Os grupos coincidem (Min-Max, LMN e MLP no fundo),
   mas o topo muda: por MCC o melhor rank é RIXM livre (4.5); por AUC é WILCAR mean/DCV (4.75), LightGBM-C (5.25),
   XGB-C (5.5); por Brier WILCAR mean/DCV (4.875), WILCAR cegis/DCV (5.0), CMNN (5.25). Sugestão: "lead to the
   same picture" / "the same groups".

6. **Tabela main_mcc, rank de "RIXM cegis hold-out" = 5.2.** Valor exato 5.25; o f-string do Python arredonda
   5.25 → 5.2 (half-to-even). Arredondamento convencional dá 5.3. Idem CMNN 5.625 → 5.6 (correto nas duas
   convenções). Opcional: usar `Decimal`/`round half up` em `make_tables.py` ou duas casas no rank.

7. **Tabela main_mcc, legenda: "$5\times5$-fold … $5\times2$ for LOAN".** Ordem inconsistente (reps×folds vs.
   folds×reps). Dados: 5 folds × 5 reps; Loan 5 folds × 2 reps. Sugestão: "$5\times5$-fold … $2\times5$-fold for
   LOAN" ou "(5-fold, five repetitions; two for LOAN)".

8. **Sec. 5.3, "our retrained CMNN and LMN fall short of them by similar margins".** Margens publicadas − nossas:
   WILCAR cegis 1.6 / 12.0 / 0.4 (COMPAS / HD / Loan); CMNN 1.4 / 8.3 / 0.3; LMN 1.6 / 6.0 / 0.7. Em Heart
   Disease (8 e 6 pontos vs. 12) "similar" é generoso. Sugestão: "fall short of their published figures by
   1.4–1.6 points on COMPAS, 6–8 points on Heart Disease and under one point on Loan Defaulter."
   Nota: "up to 1.6 points higher on COMPAS" usa WILCAR cegis (69.5 − 67.9); contra RIXM cegis seria 1.3.

9. **Tabela datasets: LOAN "Positive (%)" = `[pending]`** (o ficheiro `train_loan.csv` não está em
   `data/benchmarks` nesta máquina; `make_tables.py` cai no ramo `known`). Precisa do Zenodo para gerar; marcador
   vermelho ainda no PDF. Restam também 4 `[TODO]` (CRediT, DOI, funding, declaração de IA) e, no suplemento, o
   TODO da ordem das cinco features de Loan — `data.py` usa sinais `[-1, +1, -1, -1, +1]` para as cinco primeiras
   colunas, enquanto a Tabela S-signals descreve "credit score, length of employment, annual income (−);
   bankruptcies, DTI (+)" (padrão −,−,−,+,+). Conferir a ordem das colunas no release antes de submeter.

10. **`tables/numbers.tex`**: as macros `\CegisBB` (97.3 %), `\CegisFallback` (2.7 %), `\MeanCertified` (78.0 %),
    `\MeanViolDomain` (2.55 %), `\FreeViolDomain` (10.9 %) existem mas o texto codifica os números à mão.
    `\MeanCertified` = 78.0 % (WILCAR-C + RIXM-C, todas as seleções) difere dos 75 % da Tabela modes (WILCAR/DCV);
    se a macro for usada algum dia, essa diferença aparece. Sugestão: usar as macros no texto ou removê-las.

11. **Sec. 5.2, Holm vs. Nemenyi.** Holm (texto): Min-Max p_holm = 0.016, LMN 0.023, MLP 0.10, LR-C 0.56 — confere.
    A Figura S-cd (Nemenyi, CD = 6.45 para k = 13, N = 8) não separa nenhum par (amplitude dos ranks 4.5–10.75 =
    6.25 < 6.45). Não é erro, mas a legenda da figura poderia dizer que Holm vs. melhor (texto) é mais potente que
    Nemenyi, para o leitor não ver contradição.

## 2. Verificado como correto

Resumo/abstract
- "30 % lower error" (0.053 vs 0.076): cegis 0.05274, sign 0.07625 → redução 30.8 %; menor em todos os 7 alvos
  (55/27/27/13/26/47/24 %).
- "certified in all 175 runs": syn_cegis WILCAR-C/DCV 175/175 `certified`, 0 fallback (também 175/175 nas
  configurações hold-out).
- "35–41 % of the unconstrained networks are monotone": WILCAR/DCV 34.6 %, RIXM/hold-out 36.8 %, WILCAR/hold-out
  41.1 % (certificados pelo B&B).
- "match unconstrained networks and CMNN, outperform Min-Max and LMN": diferenças +0.002 (p 0.07), −0.004 (p 0.20),
  +0.019 (p 3e-5), +0.017 (p 2e-6).

Tabela main_mcc (13 linhas × 8 datasets): todas as 104 células conferem a 3 casas; negritos corretos (BCD: LR-C
0.95376 > RIXM 0.95374; BCO: RIXM cegis 0.93602 > WILCAR mean 0.93556); ranks 5.8/5.2(5.25)/6.4/5.5/6.8/4.5/9.5/
8.1/5.9/6.4/10.8/5.6/10.5 conferem. Tabela main_mcc_full: sd conferidos por amostragem (WILCAR cegis DCV).

Testes: Friedman χ² = 26.78 (p = 0.008); Iman–Davenport F = 2.708, p = 0.0039 → "p = 0.004" ✓; melhor método
RIXM livre (rank 4.5) ✓; Holm: Min-Max 0.0159 → "0.016" ✓, LMN 0.0227 → "0.023" ✓; MLP 0.10 e demais ≥ 0.56
→ "statistically tied" ✓. Wilcoxon (185 folds, zsplit): +0.00201 / −0.00448 / +0.01726 / +0.01861, p = 0.070 /
0.202 / 1.8e-6 / 3.1e-5 ✓ ("p < 10⁻⁴" ✓).

AUC (média dos datasets): WILCAR cegis 0.8770, WILCAR livre 0.8770, CMNN 0.8767 → "0.877" ✓; LMN 0.8727 ✓;
Min-Max 0.8691 ✓; XGB-C 0.8811, LGBM-C 0.8811 → "0.881" ✓. "Trees ahead on ALG and HF, networks on HD" ✓
(AUC ALG 0.998 vs 0.991; HF 0.906 vs 0.857; HD 0.914 vs 0.890).

Tabela modes (WILCAR/DCV, 185 folds): none 0.6310/11.04 %/34.6 %/18.2 s; mean 0.6301/2.99/74.6/87.7 s;
anchor 0.6337/3.05/63.8/16.4 s; sign 0.6321/0/100 (por construção)/29.7 s; cegis 0.6329/0/100/21.5 s — todas
as células conferem (viol. agregada sobre folds, como no script). Texto: "35 %, 11 %", "75 %", "64 %", "22 s vs
18 s" ✓; "mean is dominated by BCD and Loan" ✓ (medianas cegis BCD 4679 s, Loan 3784 s; média 836 s vs 60 s).

Fallback: 15 de 555 (2.7 %), 540 certificados pelo B&B (97.3 %) ✓. Budget: 22 de 740 ✓ (14 BCD, 8 Loan).
n*: mediana 5 (cegis DCV), 1 (cegis hold-out; também RIXM cegis) ✓; "BCD (27 constrained inputs)" ✓.

Tabela synthetic: 70 células e coluna "mean" conferem; cert. 100/100/38 (37.7)/35 (34.9)/100… ✓; negritos ✓
(mean: Min-Max 0.04147 < LightGBM 0.04148). Texto: livre 0.044 ✓; mean 0.046 ("changes little") ✓; "Min-Max, LMN
and trees 0.041–0.044" ✓ (0.0415, 0.0420, 0.0415, 0.0445); SFNNs perdem mais em "or" (0.078) e "and3" (0.080) ✓.

Tabela literature (bloco inferior, 5 seeds, 1 fold): todas as 27 células conferem (média e sd). Texto:
"67.9–68.2 vs 67.6–67.8" ✓, "80.3–82.0 vs 80.3–83.6" ✓, "65.0 vs 64.7–65.0" ✓, "within 0.5 on Loan" (0.44) ✓,
"up to 12 points on Heart Disease" (94 − 82.0) ✓, "official test set has 61 patients" ✓ (test_heart.csv: 61 linhas;
81.97 % = 50/61). Valores publicados do bloco superior coincidem com os que recordo de Runje 2023, Liu 2020,
Sivaraman 2020, Nolte 2023 e Sartor 2025 — não verificados nas fontes.

Tabela datasets: n/p/constrained/positive conferem para os 7 datasets carregáveis (ALG 243/10/9/56.4 … COM
6172/13/4/45.5); COMPAS 6172 = 4937 + 1235 ✓.

Tabela size (n* e tempo médios, 9 linhas × 8 datasets): todas conferem. Tabela cert_bench: 168 redes (56 por
tipo = 7 datasets × 4 tamanhos × 2 seeds), 12 linhas conferem; "decided 96 %" (96.4) e "median below one
millisecond" (0.05 ms) ✓; legenda "71 %, 20 %, 0 %" ✓.

Fig. 1 (fig_surfaces.pdf): MAE 0.021 (livre) / 0.067 (sign) / 0.026 (cegis) → "three times" (3.2×) ✓; "two of its
units decrease in one input each" ✓ (2 de 8). Fig. S-cd: 13 métodos da Tabela 2, CD = 6.45 ✓.

Protocolo: "5-fold repeated five times (twice for Loan)" ✓; "official splits with five seeds" ✓; "185 outer folds" ✓;
"25 outer folds" (sintéticos) ✓; alvos sintéticos da Tabela S-synthdef coincidem com `data.py`.

Referências cruzadas: todos os `\cref{S-…}` do main (alg:bb, alg:dcv, fig:capacity, sup:gradients, tab:certbench,
tab:main_auc, tab:main_brier, tab:settings, tab:hyper, tab:signals, tab:size, tab:synthdef, tab:main_mcc_full) e
todos os `M-…` do suplemento (tab:main_mcc, fig:surfaces, eq:lbx, eq:lbz, eq:loss, lem:gain, sec:dcv,
sec:certificate) existem; `main.log` sem referências indefinidas. Legendas das tabelas descrevem o conteúdo
(populações, métricas, negritos, ranks) corretamente, com a ressalva dos itens 3 e 7.
