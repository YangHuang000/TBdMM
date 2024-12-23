import pytest
import numpy as np
from ase import Atoms
from ase.build import fcc111
from TBdMM.calculations import TBdMM

def test_invalid_order():
    atoms = fcc111('Pt', size=(2, 2, 4), vacuum=10.0)
    mn, H = TBdMM(atoms, site_index=12, n=0)
    assert mn is None
    assert H is None

def test_valid_call():
    # Minimal example, just ensure no crash
    atoms = fcc111('Pt', size=(2, 2, 4), vacuum=10.0)
    mn, H = TBdMM(atoms, site_index=12, n=2)
    assert mn is not None
    assert H is not None
    assert H.shape[0] == 6*50, "Should have Natm*6 dimension"