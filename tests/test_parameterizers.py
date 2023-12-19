import pytest

pytest.importorskip("grappa_interface")
pytest.importorskip("grappa")

import os
from pathlib import Path
from copy import deepcopy
import numpy as np
import json

from grappa_interface import (
    convert_parameters,
    build_molecule,
    apply_parameters,
    GrappaInterface,
)

from grappa.grappa import Grappa
from grappa.data.Molecule import Molecule
from grappa.data.Parameters import Parameters
from grappa.utils.loading_utils import load_model

from kimmdy.topology.topology import Topology
from kimmdy.constants import AA3
from kimmdy.parsing import write_json, read_json, read_top, write_top


# generic functions
def array_eq(arr1, arr2):
    return (isinstance(arr1, np.ndarray) and
            isinstance(arr2, np.ndarray) and
            arr1.shape == arr2.shape and
            (arr1 == arr2).all())


## fixtures ##
@pytest.fixture
def grappa_input():
    return Molecule.from_json(Path(__file__).parent / "in_alanine.json")


@pytest.fixture
def grappa_output_raw():
    ref_dict = read_json(Path(__file__).parents[0] / "out_raw_alanine.json")
    for k,v in ref_dict.items():
        ref_dict[k] = np.asarray(v)
    output_raw= Parameters.from_dict(ref_dict)
    return output_raw

@pytest.fixture
def grappa_output_converted():
    ref_dict = read_json(Path(__file__).parents[0] / "out_converted_alanine.json")
    output_converted= Parameters.from_dict(ref_dict)
    return output_converted

## test scripts ##
def test_generate_input():
    top = Topology(read_top(Path(__file__).parent / "Ala_out.top"))
    mol = build_molecule(top)
    mol.to_json( Path(__file__).parents[0] / "tmp" / "in.json")

    assert len(mol.atoms) == 21
    assert len(mol.bonds) == 20
    assert len(mol.impropers) == 15
    feature_keys = list(mol.additional_features.keys())
    assert all(x in feature_keys for x in ['is_radical','ring_encoding'])
    assert mol.atomic_numbers[8] == 6
    assert mol.additional_features['is_radical'][8] == 1


def test_predict_parameters(grappa_input,grappa_output_raw):
    # load model, tag will be changed to be more permanent
    model_tag =  'https://github.com/LeifSeute/test_torchhub/releases/download/test_release_radicals/radical_model_12142023.pth'
    model = load_model(model_tag)

    # initialize class that handles ML part
    grappa = Grappa(model,device='cpu')
    parameters = grappa.predict(grappa_input)

    parameters_dict = parameters.to_dict()
    write_json(parameters_dict,  Path(__file__).parents[0] / "tmp" / "out_raw.json")
   
    # check for equality per attribute
    for k in grappa_output_raw.__annotations__.keys():
        assert array_eq(getattr(grappa_output_raw,k),getattr(parameters,k))
                                          
                            
def test_convert_parameters(grappa_output_raw, grappa_output_converted):
    parameters = convert_parameters(grappa_output_raw)

    parameters_dict = parameters.to_dict()
    write_json(parameters_dict, Path(__file__).parents[0] / "tmp" / "out_converted.json")

    assert parameters == grappa_output_converted    


def test_apply_parameters(grappa_output_converted):
    top = Topology(read_top(Path(__file__).parent / "Ala_out.top"))
    partial_charges = np.zeros_like(grappa_output_converted.atoms,dtype=float).tolist()
    apply_parameters(top, grappa_output_converted, partial_charges)

    write_top(top.to_dict(), Path(__file__).parents[0] / "tmp" / "out_parameterized.top")


def test_parameterize_topology(tmp_path):
    os.chdir(tmp_path.resolve())
    parameterizer = GrappaInterface()
    top = Topology(read_top(Path(__file__).parent / "Ala_out.top"), parameterizer)

    curr_top = deepcopy(top)
    curr_top.needs_parameterization = True
    curr_top.update_parameters()
    assert top != curr_top
