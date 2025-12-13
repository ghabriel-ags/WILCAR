"""
RIXM Module - Random Initialization Xavier Method
==================================================

Method 3: RIXM (baseline - random initialization, no weight reuse)
Method 4: RIXM Constrained (baseline with gain sign constraints)
"""

from .rixm import (
    RIXM,
    RIXMNumpy,
    create_rixm
)

from .rixm_constrained import (
    RIXMConstrained,
    RIXMConstrainedNumpy,
    create_rixm_constrained
)

__all__ = [
    # Method 3 - RIXM (baseline)
    'RIXM',
    'RIXMNumpy',
    'create_rixm',
    
    # Method 4 - RIXM Constrained
    'RIXMConstrained',
    'RIXMConstrainedNumpy',
    'create_rixm_constrained'
]