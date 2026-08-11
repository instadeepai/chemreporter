# Copyright 2026 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import patch

import ase
import pytest
from ase.build import molecule

from chemreporter.source_database_tools.database_item import DatasetItem
from chemreporter.source_database_tools.exceptions import SourceDatabaseReaderUsageError
from chemreporter.source_database_tools.open_catalyst_20.oc20_source_fields import (
    get_oc20_source_additional_fields,
)
from chemreporter.source_database_tools.source_db_implementations import (
    correct_database_name,
)


def _make_dataset_item(atoms):
    return DatasetItem(
        database_name="oc20-s2ef",
        split_name="train",
        key="test_key",
        atoms=atoms,
        name_mapping={},
        additional_fields=[],
    )


@patch(
    "chemreporter.source_database_tools.open_catalyst_20.oc20_source_fields.get_catalyst_data"
)
def test_additional_fields_net_charge_and_spin_even_electrons(mock_get_catalyst_data):
    """An even total electron count yields a singlet with zero net charge."""
    mock_get_catalyst_data.return_value = {}
    atoms = molecule("H2O")  # 10 electrons total
    dataset_item = _make_dataset_item(atoms)

    result = get_oc20_source_additional_fields(dataset_item)

    assert result["net_charge"] == 0
    assert result["spin_multiplicity"] == 1


@patch(
    "chemreporter.source_database_tools.open_catalyst_20.oc20_source_fields.get_catalyst_data"
)
def test_additional_fields_spin_odd_electrons(mock_get_catalyst_data):
    """An odd total electron count yields a doublet."""
    mock_get_catalyst_data.return_value = {}
    atoms = ase.Atoms("H", positions=[[0.0, 0.0, 0.0]])  # 1 electron
    dataset_item = _make_dataset_item(atoms)

    result = get_oc20_source_additional_fields(dataset_item)

    assert result["net_charge"] == 0
    assert result["spin_multiplicity"] == 2


@patch(
    "chemreporter.source_database_tools.open_catalyst_20.oc20_source_fields.get_catalyst_data"
)
def test_additional_fields_composition(mock_get_catalyst_data):
    """The composition field reports the chemical formula of the atoms."""
    mock_get_catalyst_data.return_value = {}
    atoms = molecule("H2O")
    dataset_item = _make_dataset_item(atoms)

    result = get_oc20_source_additional_fields(dataset_item)

    assert result["composition"] == atoms.get_chemical_formula()


@patch(
    "chemreporter.source_database_tools.open_catalyst_20.oc20_source_fields.get_catalyst_data"
)
def test_additional_fields_merges_catalyst_data(mock_get_catalyst_data):
    """Fields from get_catalyst_data are merged into the result."""
    mock_get_catalyst_data.return_value = {"catalyst_bulk_id": 42}
    atoms = molecule("H2O")
    dataset_item = _make_dataset_item(atoms)

    result = get_oc20_source_additional_fields(dataset_item)

    mock_get_catalyst_data.assert_called_once_with(atoms)
    assert result["catalyst_bulk_id"] == 42


@patch(
    "chemreporter.source_database_tools.open_catalyst_20.oc20_source_fields.get_catalyst_data"
)
def test_additional_fields_catalyst_data_overrides_earlier_fields(
    mock_get_catalyst_data,
):
    """get_catalyst_data is merged last, so it can override earlier fields."""
    mock_get_catalyst_data.return_value = {"net_charge": 99, "spin_multiplicity": 99}
    atoms = molecule("H2O")
    dataset_item = _make_dataset_item(atoms)

    result = get_oc20_source_additional_fields(dataset_item)

    assert result["net_charge"] == 99
    assert result["spin_multiplicity"] == 99


def test_correct_database_name_extracts_subset_from_last_token():
    """The subset is taken from the final delimiter-separated token."""
    dataset_item = _make_dataset_item(molecule("H2O"))
    dataset_item.database_name = "oc20-s2ef"

    result = correct_database_name(dataset_item)

    assert result == {"database_name": "oc20", "subset": "s2ef"}


def test_correct_database_name_supports_multiple_delimiters():
    """Any non-alphanumeric run counts as a delimiter, including multiple."""
    dataset_item = _make_dataset_item(molecule("H2O"))
    dataset_item.database_name = "oc20_2020__is2res"

    result = correct_database_name(dataset_item)

    assert result == {"database_name": "oc20", "subset": "2020-is2res"}


def test_correct_database_name_raises_without_delimiter():
    """A name with no delimiter cannot yield a subset."""
    dataset_item = _make_dataset_item(molecule("H2O"))
    dataset_item.database_name = "oc20"

    with pytest.raises(SourceDatabaseReaderUsageError):
        correct_database_name(dataset_item)
