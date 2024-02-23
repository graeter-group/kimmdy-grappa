'''NOTE: MODEL_FROM_URL IS OUTDATED!'''

import pytest

# pytest.importorskip("kimmdy-grappa")
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
    load_model,
)

from grappa.grappa import Grappa
from grappa.data.Molecule import Molecule
from grappa.data.Parameters import Parameters

from kimmdy.topology.topology import Topology
from kimmdy.constants import AA3
from kimmdy.parsing import write_json, read_json, read_top, write_top


# generic functions
def array_eq(arr1, arr2):
    return (
        isinstance(arr1, np.ndarray)
        and isinstance(arr2, np.ndarray)
        and arr1.shape == arr2.shape
        and (arr1 == arr2).all()
    )


## fixtures ##
@pytest.fixture
def grappa_input():
    return Molecule.from_json(Path(__file__).parent / "in_alanine.json")


@pytest.fixture
def grappa_output_raw():
    ref_dict = read_json(Path(__file__).parents[0] / "out_raw_alanine.json")
    for k, v in ref_dict.items():
        ref_dict[k] = np.asarray(v)
    output_raw = Parameters.from_dict(ref_dict)
    return output_raw


@pytest.fixture
def grappa_output_converted():
    ref_dict = read_json(Path(__file__).parents[0] / "out_converted_alanine.json")
    output_converted = Parameters.from_dict(ref_dict)
    return output_converted


## test scripts ##
def test_generate_input():
    top = Topology(read_top(Path(__file__).parent / "Ala_out.top"))
    mol = build_molecule(top)
    out_path = Path(__file__).parents[0] / "tmp" / "in.json"
    out_path.parents[0].mkdir(parents=True, exist_ok=True)
    mol.to_json(out_path)

    assert len(mol.atoms) == 21
    assert len(mol.bonds) == 20
    assert len(mol.impropers) == 15
    feature_keys = list(mol.additional_features.keys())
    assert all(x in feature_keys for x in ["is_radical", "ring_encoding"])
    assert mol.atomic_numbers[8] == 6
    assert mol.additional_features["is_radical"][8] == 1


def test_predict_parameters(grappa_input, grappa_output_raw):
    model = load_model()

    # initialize class that handles ML part
    grappa = Grappa(model, device="cpu")
    parameters = grappa.predict(grappa_input)

    parameters_dict = parameters.to_dict()
    out_path = Path(__file__).parents[0] / "tmp" / "out_raw.json"
    out_path.parents[0].mkdir(parents=True, exist_ok=True)
    write_json(parameters_dict, out_path)

    # check for equality per attribute
    for k in grappa_output_raw.__annotations__.keys():
        assert array_eq(getattr(grappa_output_raw, k), getattr(parameters, k))


def test_convert_parameters(grappa_output_raw, grappa_output_converted):
    parameters = convert_parameters(grappa_output_raw)

    parameters_dict = parameters.to_dict()
    out_path = Path(__file__).parents[0] / "tmp" / "out_converted.json"
    out_path.parents[0].mkdir(parents=True, exist_ok=True)
    write_json(parameters_dict, out_path)

    assert parameters == grappa_output_converted


def test_apply_parameters(grappa_output_converted):
    top = Topology(read_top(Path(__file__).parent / "Ala_out.top"))
    apply_parameters(top, grappa_output_converted)

    out_path = Path(__file__).parents[0] / "tmp" / "out_parameterized.json"
    out_path.parents[0].mkdir(parents=True, exist_ok=True)
    write_top(top.to_dict(), out_path)


def test_parameterize_topology(tmp_path):
    os.chdir(tmp_path.resolve())
    parameterizer = GrappaInterface()
    top = Topology(read_top(Path(__file__).parent / "Ala_out.top"), parameterizer)

    curr_top = deepcopy(top)
    curr_top.needs_parameterization = True
    curr_top.update_parameters()
    assert top != curr_top
