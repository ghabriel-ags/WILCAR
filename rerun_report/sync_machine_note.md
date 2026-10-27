## synchronous_machine: um fold preso numa solução ruim de n = 1

**Versão final (4 threads por worker, `vcd_20261027_073947_reg`):** R² = 0,5073 ± 0,9485, n\* = 30, 45 passos,
nenhum descartado, não chega a n_max. **O fold 0 continua preso** (R² de teste −1,390, de treino −1,463); os outros
quatro folds ficam entre 0,974 e 0,985. A versão com 3 threads (`vcd_20261026_122026`, substituída) dá o mesmo
resultado até a 9ª casa decimal.

Em março, sem reaproveitamento, o DCV deu 0,987 ± 0,003 (n\* = 9); no K-fold do cap. 5, 0,990 ± 0,004. O
diagnóstico, sem nenhuma mudança de código, mostra que **não é erro da correção, e sim comportamento do método com
reaproveitamento**.

**Onde e como degenera.** O fold 0 degenera **já em n = 1**: R² de treino −1,463 e de teste −1,390 (MSE 0,148 e
0,152). A solução foi aceita na 1ª tentativa, porque atingiu SCR = 100%. O SLSQP reportou `success=True`, status 0,
"Optimization terminated successfully", com nit = 9 e fun = 0,148. Ou seja, convergiu formalmente para um ponto
estacionário ruim. Nos 44 passos seguintes o fold 0 parte desse estado, o SLSQP converge de novo com sucesso, e o
R² de treino fica entre −1,4631 e −1,4629.

**Não é erro da correção:**

- **Estados.** Instrumentado o caminho paralelo de produção (réplica bit a bit da run final), cada fold recebe
  sempre o próprio último estado aceito: 0 violações, e nenhum estado veio de outro fold.
- **Descarte.** Nenhum passo foi descartado, então a regra de descarte não foi exercida.
- **Partição e normalização.** São as de março: J_T(1) é idêntico a março (0,18095377423488446), e os dados
  normalizados do fold 0 são idênticos entre K-fold e DCV.
- **`--no-parallel`.** Com instâncias persistentes e uma sequência aleatória contínua, o fold 0 escapa em n = 2,
  mas o **fold 1** fica preso da mesma forma (R² de teste −1,89 em todos os passos). Resultado: n\* = 41,
  R² = 0,412 ± 1,151, e o laço chega a n_max = 50. O fenômeno aparece no fold que tiver o sorteio desfavorável.

**Por que março e o K-fold não mostram o problema.** Em março, sem reaproveitamento, cada n reinicializava a rede:
a solução ruim de n = 1 era um ponto isolado e o fold se recuperava em n = 2. O K-fold usa outra semente para o
fold 0 do M2 (`seed + 100·fold + método` = 2; no DCV, `seed + 100·fold` = 0). Partição, normalização e pesos da
linearização são os mesmos. Um teste fatorial em n = 1 (pesos da linearização das sementes 0 ou 2 × sequência
aleatória das sementes 0 ou 2) mostra que o desfecho depende só da sequência aleatória, isto é, do valor inicial de
W2 do único neurônio:

- semente 0 (DCV): W2 = −0,213, R² de treino = −1,463;
- semente 2 (K-fold): W2 = +0,088, R² de treino = 0,990 (nit = 82).

Com a semente 2, o fold 0 do K-fold fica em R² ≈ 0,990 de n = 1 a n = 22, com n\* = 7 e R² de teste = 0,98796,
como em março.

**Implicação.** O critério de aceitação (100% de SCR) não exige qualidade de ajuste. Com reaproveitamento de pesos,
uma solução ruim aceita em n = 1 é herdada por todos os passos seguintes daquele fold. Nada foi alterado: nem
critério, nem configuração, nem a re-semeadura.

Curvas por fold (R² e MSE de treino e de teste por n; março, outubro paralelo e outubro `--no-parallel`):
`sync_folds_r2.png`, `sync_folds_mse.png`. Valores em `sync_folds.csv`. As curvas vêm de execuções instrumentadas
com 4 threads do BLAS, como os workers de março. A de outubro (paralelo) reproduz bit a bit a run final, e a de março
reproduz bit a bit `vcd_20260310_140932` (todos os J_T(n), n\* = 9, 24 passos). Com 3 threads, a réplica de março
desviava até 6 × 10⁻⁷ em termos relativos para n ≥ 2. Isso mostra que a run de regressão de 10/03 também tinha
4 threads por worker. No painel de março, o fold 0 também é ruim em n = 1, mas se recupera em n = 2, porque a rede
é reinicializada.

![R² por fold](sync_folds_r2.png)

![MSE por fold (escala log)](sync_folds_mse.png)
