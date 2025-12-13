"""
WILCAR Module - Métodos WILCAR (com e sem restrições)
=====================================================

Método 1: WILCAR Unconstrained (sem restrições nos sinais dos ganhos)
Método 2: WILCAR Constrained (com restrições nos sinais dos ganhos)
"""

from .unconstrained import (
    WILCARUnconstrained,
    WILCARUnconstrainedNumpy,
    create_wilcar_unconstrained
)

from .constrained import (
    WILCARConstrained,
    WILCARConstrainedNumpy,
    create_wilcar_constrained
)

__all__ = [
    # Método 1 - Sem restrições
    'WILCARUnconstrained',
    'WILCARUnconstrainedNumpy',
    'create_wilcar_unconstrained',
    
    # Método 2 - Com restrições
    'WILCARConstrained',
    'WILCARConstrainedNumpy',
    'create_wilcar_constrained'
]