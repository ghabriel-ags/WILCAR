"""n = 1 of fold 0 (synchronous_machine, WILCAR-C): linearization initial weights (from seed 0 or 2) x random stream
for the remaining draws (W2 etc.; seed 0 or 2). Read-only diagnosis."""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).parent)]
import diag_kfold_vs_dcv as d  # runs the comparison prints again (cheap), then reuse its helpers
import numpy as np
P = d.P
w = {s: d.model(s).initial_weights.copy() for s in (0, 2)}
P("initial weights seed 0:", np.round(w[0], 4).tolist())
P("initial weights seed 2:", np.round(w[2], 4).tolist())
for wseed in (0, 2):
    for rseed in (0, 2):
        m = d.model(rseed)              # model created with rseed: its RNG stream continues from that seed
        m.initial_weights = w[wseed].copy()
        ok, att, r2tr, r2te, last = d.n1(m)
        P(f"weights from seed {wseed}, stream seed {rseed}: R2_train={r2tr:.4f} R2_test={r2te:.4f} nit={last[4]} fun={last[5]:.6f}")
