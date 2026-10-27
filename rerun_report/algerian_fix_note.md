### Causa do erro e correção

**Causa.** No WILCAR-C, quando nenhuma das até 100 reinicializações atinge 100% de SCR num tamanho `n`, o
laço construtivo de `train()` pula esse tamanho (`continue`) e mantém em `previous_param` o último modelo viável,
com `k < n` neurônios. No tamanho seguinte, `_expand_network(previous_param, n + 1)` acrescentava sempre **um**
neurônio, produzindo uma rede de `k + 1` neurônios, enquanto `_vector_to_params(vec, n + 1)` desempacotava o vetor
de parâmetros como se ele tivesse `n + 1`. No fold 1 do Algerian (p = 10), o tamanho 12 foi inviável depois de um
modelo viável de 11 neurônios. No tamanho 13, o vetor tinha 12·(10 + 2) + 1 = 145 elementos e, ao desempacotá-lo
com n = 13, sobraram só 2 elementos para `W2` (1 × 13): `cannot reshape array of size 2 into shape (1,13)`.
Esse caminho sempre levava a erro, porque qualquer tamanho pulado (exceto o último) seguido de outro passo deixa
o vetor curto. Nenhuma run que terminou passou por ele. O mesmo caminho existia no DCV
(`constructive_step`, que também pula tamanhos inviáveis), mas nunca foi atingido nas runs de março.

**Correção** (`src/wilcar/constrained.py`, `_expand_network`). A rede passa a ser completada até `n` neurônios
a partir do último modelo viável, com um neurônio novo por tamanho pulado. Isso preserva o reaproveitamento de
pesos e a semântica "pular o tamanho inviável e seguir". No caso regular (modelo anterior com `n − 1` neurônios),
o laço executa exatamente uma vez, com as mesmas chamadas ao gerador aleatório e na mesma ordem. O comportamento
é bit a bit idêntico ao de março em todos os casos que já funcionavam: teste
`tests/test_wilcar_constrained.py` e comparação por fold, código corrigido vs. código sem correção, nas mesmas
condições.
