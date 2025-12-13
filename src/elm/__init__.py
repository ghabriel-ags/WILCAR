"""
ELM Module - Extreme Learning Machine
======================================

Method 5: ELM (traditional - random fixed hidden weights, pseudo-inverse)
Method 6: ELM Constrained (with gain sign constraints via SLSQP)
"""

from .elm import (
    ELM,
    ELMNumpy,
    create_elm
)

from .elm_constrained import (
    ELMConstrained,
    ELMConstrainedNumpy,
    create_elm_constrained
)

__all__ = [
    # Method 5 - ELM
    'ELM',
    'ELMNumpy',
    'create_elm',
    
    # Method 6 - ELM Constrained
    'ELMConstrained',
    'ELMConstrainedNumpy',
    'create_elm_constrained'
]