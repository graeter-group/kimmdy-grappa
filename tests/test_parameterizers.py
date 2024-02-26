import pytest

# pytest.importorskip("kimmdy-grappa")
pytest.importorskip("grappa")

import os
import math
from pathlib import Path
from copy import deepcopy

from kimmdy.topology.topology import Topology
from kimmdy.topology.atomic import MultipleDihedrals
from kimmdy.parsing import read_top


from grappa_interface import GrappaInterface


def test_parameterize_topology(tmp_path):
    os.chdir(tmp_path.resolve())
    parameterizer = GrappaInterface()
    top = Topology(read_top(Path(__file__).parent / "Ala_out.top"), parameterizer)

    curr_top = deepcopy(top)
    curr_top.needs_parameterization = True
    curr_top.update_parameters()

    assert top != curr_top

    # if charge assignment gets into grappa
    # assert len(curr_top.atoms) == 21
    # charge_sum = 0
    # for atom in curr_top.atoms.values():
    #     charge_sum += int(atom.charge)
    # assert math.isclose(charge_sum,0,rel_tol=1e-5)

    assert len(curr_top.bonds) == 20
    for bond in curr_top.bonds.values():
        assert bond.c0 is not None
        assert bond.c1 is not None

    assert len(curr_top.angles) == 33
    for angle in curr_top.angles.values():
        assert angle.c0 is not None
        assert angle.c1 is not None

    assert len(curr_top.proper_dihedrals) == 34
    for multiple_dihedral in curr_top.proper_dihedrals.values():
        assert len(multiple_dihedral.dihedrals) in [3, 6]

    assert len(curr_top.improper_dihedrals) == 15
    for multiple_dihedral in curr_top.improper_dihedrals.values():
        assert isinstance(multiple_dihedral, MultipleDihedrals)
        assert len(multiple_dihedral.dihedrals) == 3
