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
"""Tests for DatabaseProcessor and GraphBasedProcessingConfig."""

from unittest.mock import Mock, patch

import ase
import numpy as np
import polars as pl
import pytest

from chemreporter.config_schemas import GraphBasedProcessingConfig
from chemreporter.database_processors.converter import DatabaseProcessor
from chemreporter.query_database_tools.table_schemas import (
    AMINO_ACID_CODES,
    NUCLEOBASE_CODES,
)


def _amino_only_rdkit_stub() -> dict:
    d = {f"num_{c.lower()}": 0 for c in AMINO_ACID_CODES}
    d["is_protein"] = False
    for c in NUCLEOBASE_CODES:
        d[f"num_{c.lower()}"] = 0
    d["is_nucleobase"] = False
    return d


def _full_graph_rdkit_stub() -> dict:
    fp = {f"fingerprint_{i}": 0 for i in range(1024)}
    return {
        **_amino_only_rdkit_stub(),
        "logp": -0.5,
        "tpsa": 0.0,
        "smiles": "C",
        **fp,
    }


class TestGraphBasedProcessingConfig:
    """Test GraphBasedProcessingConfig class."""

    def test_default_initialization(self):
        """Test default config initialization."""
        config = GraphBasedProcessingConfig()

        assert config.enable is False
        assert config.nb_atoms_limit == 200
        assert config.subsets_skip_list == []

    def test_custom_initialization(self):
        """Test custom config initialization."""
        config = GraphBasedProcessingConfig(
            enable=True, nb_atoms_limit=150, subsets_skip_list=["subset1", "subset2"]
        )

        assert config.enable is True
        assert config.nb_atoms_limit == 150
        assert config.subsets_skip_list == ["subset1", "subset2"]

    def test_partial_custom_initialization(self):
        """Test partial custom config initialization."""
        config = GraphBasedProcessingConfig(enable=True)

        assert config.enable is True
        assert config.nb_atoms_limit == 200
        assert config.subsets_skip_list == []


class TestDatabaseProcessor:
    """Test DatabaseProcessor class."""

    @pytest.fixture
    def default_config(self):
        """Default configuration for testing."""
        return GraphBasedProcessingConfig()

    @pytest.fixture
    def enabled_config(self):
        """Enabled configuration for testing."""
        return GraphBasedProcessingConfig(
            enable=True, nb_atoms_limit=10, subsets_skip_list=[]
        )

    @pytest.fixture
    def mock_dataset_item(self):
        """Create a mock dataset item."""
        mock_item = Mock()
        mock_item.key = "test_key_1"

        # Create mock atoms
        mock_atoms = Mock(spec=ase.Atoms)
        mock_atoms.info = {
            "data_id": "test_set",
            "charge": 0,
            "num_atoms": 5,
            "spin": 0.0,
            "source": "test_source",
            "mulliken_charges": [0.1, -0.1, 0.2, -0.2, 0.0],
        }
        mock_atoms.arrays = {}

        mock_atoms.positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ])
        mock_atoms.get_positions.return_value = mock_atoms.positions
        mock_atoms.get_atomic_numbers.return_value = np.array([6, 1, 1, 1, 1])
        mock_atoms.get_chemical_symbols.return_value = ["C", "H", "H", "H", "H"]
        mock_atoms.get_center_of_mass.return_value = np.array([0.4, 0.4, 0.4])
        mock_atoms.get_masses.return_value = np.array([
            12.011,
            1.008,
            1.008,
            1.008,
            1.008,
        ])
        # Mock calculator results
        mock_atoms._calc = Mock()
        mock_atoms._calc.results = {
            "forces": np.array([
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [-0.2, -0.3, -0.4],
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1],
            ])
        }

        # Support subscripting for count_water_molecules (returns same mock)
        mock_atoms.__getitem__ = Mock(return_value=mock_atoms)
        mock_atoms.copy.return_value = mock_atoms
        # Set proper numpy arrays for cell/pbc
        mock_atoms.cell = np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
        mock_atoms.pbc = np.array([False, False, False])

        mock_item.atoms = mock_atoms
        mock_item.database_name = "test_db"
        mock_item.split_name = "test_split"
        mock_item.name_mapping = {}
        mock_item.additional_fields = []  # Empty list for additional property functions
        return mock_item

    def test_processor_initialization(self, default_config):
        """Test processor initialization."""
        processor = DatabaseProcessor(default_config)

        assert processor.graph_properties_config == default_config

    @patch("chemreporter.database_processors.converter.count_water_molecules")
    @patch("chemreporter.database_processors.converter.is_molecular_structure_valid")
    @patch("chemreporter.database_processors.converter.get_molecular_weight")
    @patch("chemreporter.database_processors.converter.get_unique_chemical_symbols")
    @patch("chemreporter.database_processors.converter.process_ase_info")
    @patch("chemreporter.database_processors.converter.get_net_forces_norm")
    @patch("chemreporter.database_processors.converter.get_max_forces_norm")
    @patch("chemreporter.database_processors.converter.calculate_dipole_moment")
    def test_run_without_rdkit(
        self,
        mock_dipole,
        mock_max_force,
        mock_sum_force,
        mock_extract,
        mock_unique_symbols,
        mock_molecular_weight,
        mock_structure_valid,
        mock_count_water,
        default_config,
        mock_dataset_item,
    ):
        """Test process method without RDKit properties."""
        # Setup mocks
        mock_extract.return_value = {
            "subset": "test_set",
            "composition": "CH4",
            "net_charge": 0,
            "num_atoms": 5,
            "spin_multiplicity": 0.0,
            "atomic_numbers": [6, 1, 1, 1, 1],
            "energy": -40.5,
        }
        mock_sum_force.return_value = 0.5
        mock_max_force.return_value = 1.0
        mock_dipole.return_value = 0.8
        mock_unique_symbols.return_value = "['C', 'H']"
        mock_molecular_weight.return_value = 16.04
        mock_structure_valid.return_value = True
        mock_count_water.return_value = 0

        processor = DatabaseProcessor(default_config)
        result = processor.process([mock_dataset_item])

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1
        mock_extract.assert_called_once()

    @patch("chemreporter.database_processors.converter.count_water_molecules")
    @patch("chemreporter.database_processors.converter.is_molecular_structure_valid")
    @patch("chemreporter.database_processors.converter.get_molecular_weight")
    @patch("chemreporter.database_processors.converter.get_unique_chemical_symbols")
    @patch("chemreporter.database_processors.converter.get_fpgen")
    @patch("chemreporter.database_processors.converter.process_ase_info")
    @patch("chemreporter.database_processors.converter.get_net_forces_norm")
    @patch("chemreporter.database_processors.converter.get_max_forces_norm")
    @patch("chemreporter.database_processors.converter.calculate_dipole_moment")
    @patch(
        "chemreporter.database_processors.converter.calculate_graph_derived_properties"
    )
    def test_run_with_rdkit_small_molecule(
        self,
        mock_calc_props,
        mock_dipole,
        mock_max_force,
        mock_sum_force,
        mock_extract,
        mock_fpgen,
        mock_unique_symbols,
        mock_molecular_weight,
        mock_structure_valid,
        mock_count_water,
        enabled_config,
        mock_dataset_item,
    ):
        """Test process method with RDKit properties for small molecule."""
        # Setup mocks
        mock_extract.return_value = {
            "subset": "test_set",
            "composition": "CH4",
            "net_charge": 0,
            "num_atoms": 5,
            "spin_multiplicity": 0.0,
            "atomic_numbers": [6, 1, 1, 1, 1],
            "energy": -40.5,
        }
        mock_sum_force.return_value = 0.5
        mock_max_force.return_value = 1.0
        mock_dipole.return_value = 0.8
        mock_molecular_weight.return_value = 16.04
        mock_calc_props.return_value = _full_graph_rdkit_stub()
        mock_unique_symbols.return_value = "['C', 'H']"
        mock_structure_valid.return_value = True
        mock_count_water.return_value = 0
        mock_fpgen.return_value = Mock()

        processor = DatabaseProcessor(enabled_config)
        # skip schema validation
        processor._validated_schema = True
        result = processor.process([mock_dataset_item])

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1
        # Should calculate RDKit properties for molecules under the limit
        mock_calc_props.assert_called_once()

    @patch("chemreporter.database_processors.converter.count_water_molecules")
    @patch("chemreporter.database_processors.converter.is_molecular_structure_valid")
    @patch("chemreporter.database_processors.converter.get_molecular_weight")
    @patch("chemreporter.database_processors.converter.get_unique_chemical_symbols")
    @patch("chemreporter.database_processors.converter.get_fpgen")
    @patch("chemreporter.database_processors.converter.process_ase_info")
    @patch("chemreporter.database_processors.converter.get_net_forces_norm")
    @patch("chemreporter.database_processors.converter.get_max_forces_norm")
    @patch("chemreporter.database_processors.converter.calculate_dipole_moment")
    @patch(
        "chemreporter.database_processors.converter.calculate_graph_derived_properties"
    )
    def test_run_with_rdkit_large_molecule(
        self,
        mock_calc_props,
        mock_dipole,
        mock_max_force,
        mock_sum_force,
        mock_extract,
        mock_fpgen,
        mock_unique_symbols,
        mock_molecular_weight,
        mock_structure_valid,
        mock_count_water,
        enabled_config,
        mock_dataset_item,
    ):
        """Test process method skips RDKit for large molecules."""
        # Setup mocks - molecule with 500 atoms (over limit)
        mock_structure_valid.return_value = True
        mock_count_water.return_value = 0
        mock_extract.return_value = {
            "subset": "test_set",
            "composition": "C500",
            "net_charge": 0,
            "num_atoms": 500,  # Over the limit
            "spin_multiplicity": 0.0,
            "atomic_numbers": [6] * 500,
            "energy": -2000.0,
        }
        mock_sum_force.return_value = 0.5
        mock_max_force.return_value = 1.0
        mock_dipole.return_value = 0.8
        mock_molecular_weight.return_value = 6000.0
        mock_unique_symbols.return_value = "['C']"
        mock_fpgen.return_value = Mock()
        mock_calc_props.return_value = _amino_only_rdkit_stub()

        processor = DatabaseProcessor(enabled_config)
        # skip schema validation
        processor._validated_schema = True
        result = processor.process([mock_dataset_item])

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1
        mock_calc_props.assert_called_once()
        assert mock_calc_props.call_args[0][2] is None

    @patch("chemreporter.database_processors.converter.count_water_molecules")
    @patch("chemreporter.database_processors.converter.is_molecular_structure_valid")
    @patch("chemreporter.database_processors.converter.get_molecular_weight")
    @patch("chemreporter.database_processors.converter.get_unique_chemical_symbols")
    @patch("chemreporter.database_processors.converter.get_fpgen")
    @patch("chemreporter.database_processors.converter.process_ase_info")
    @patch("chemreporter.database_processors.converter.get_net_forces_norm")
    @patch("chemreporter.database_processors.converter.get_max_forces_norm")
    @patch("chemreporter.database_processors.converter.calculate_dipole_moment")
    @patch(
        "chemreporter.database_processors.converter.calculate_graph_derived_properties"
    )
    def test_run_with_skip_list(
        self,
        mock_calc_props,
        mock_dipole,
        mock_max_force,
        mock_sum_force,
        mock_extract,
        mock_fpgen,
        mock_unique_symbols,
        mock_molecular_weight,
        mock_structure_valid,
        mock_count_water,
        mock_dataset_item,
    ):
        """Test process method skips subsets in skip list."""
        # Setup config with skip list
        config = GraphBasedProcessingConfig(
            enable=True, nb_atoms_limit=10, subsets_skip_list=["test_set"]
        )

        # Setup mocks
        mock_structure_valid.return_value = True
        mock_count_water.return_value = 0
        mock_extract.return_value = {
            "subset": "test_set",  # This is in skip list
            "composition": "CH4",
            "net_charge": 0,
            "num_atoms": 5,
            "spin_multiplicity": 0.0,
            "atomic_numbers": [6, 1, 1, 1, 1],
            "energy": -40.5,
        }
        mock_sum_force.return_value = 0.5
        mock_max_force.return_value = 1.0
        mock_dipole.return_value = 0.8
        mock_molecular_weight.return_value = 16.04
        mock_unique_symbols.return_value = "['C', 'H']"
        mock_fpgen.return_value = Mock()
        mock_calc_props.return_value = _amino_only_rdkit_stub()

        processor = DatabaseProcessor(config)
        result = processor.process([mock_dataset_item])

        assert isinstance(result, pl.DataFrame)
        mock_calc_props.assert_called_once()
        assert mock_calc_props.call_args[0][2] is None

    @patch("chemreporter.database_processors.converter.count_water_molecules")
    @patch("chemreporter.database_processors.converter.is_molecular_structure_valid")
    @patch("chemreporter.database_processors.converter.get_molecular_weight")
    @patch("chemreporter.database_processors.converter.get_unique_chemical_symbols")
    @patch("chemreporter.database_processors.converter.process_ase_info")
    @patch("chemreporter.database_processors.converter.get_net_forces_norm")
    @patch("chemreporter.database_processors.converter.get_max_forces_norm")
    @patch("chemreporter.database_processors.converter.calculate_dipole_moment")
    def test_run_multiple_items(
        self,
        mock_dipole,
        mock_max_force,
        mock_sum_force,
        mock_extract,
        mock_unique_symbols,
        mock_molecular_weight,
        mock_structure_valid,
        mock_count_water,
        default_config,
        mock_dataset_item,
    ):
        """Test process method with multiple dataset items."""
        # Setup mocks
        mock_structure_valid.return_value = True
        mock_count_water.return_value = 0
        mock_extract.side_effect = [
            {
                "subset": "test_set",
                "composition": "CH4",
                "net_charge": 0,
                "num_atoms": 5,
                "spin_multiplicity": 0.0,
                "atomic_numbers": [6, 1, 1, 1, 1],
                "energy": -40.5,
            },
            {
                "subset": "test_set_2",
                "composition": "OH2",
                "net_charge": 1,
                "num_atoms": 3,
                "spin_multiplicity": 0.5,
                "atomic_numbers": [8, 1, 1],
                "energy": -76.4,
            },
        ]
        mock_sum_force.return_value = 0.5
        mock_max_force.return_value = 1.0
        mock_dipole.return_value = 0.8
        mock_unique_symbols.return_value = "['C', 'H']"
        mock_molecular_weight.return_value = 18.0
        mock_structure_valid.return_value = True
        mock_count_water.return_value = 0
        # Create second mock item
        mock_item_2 = Mock()
        mock_item_2.key = "test_key_2"
        mock_item_2.atoms = mock_dataset_item.atoms
        mock_item_2.database_name = "test_db"
        mock_item_2.split_name = "test_split"
        mock_item_2.name_mapping = {}
        mock_item_2.additional_fields = []

        processor = DatabaseProcessor(default_config)
        result = processor.process([mock_dataset_item, mock_item_2])

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2

    @patch("chemreporter.database_processors.converter.count_water_molecules")
    @patch("chemreporter.database_processors.converter.is_molecular_structure_valid")
    @patch("chemreporter.database_processors.converter.get_unique_chemical_symbols")
    @patch("chemreporter.database_processors.converter.process_ase_info")
    @patch("chemreporter.database_processors.converter.get_net_forces_norm")
    @patch("chemreporter.database_processors.converter.get_max_forces_norm")
    @patch("chemreporter.database_processors.converter.calculate_dipole_moment")
    @patch("chemreporter.database_processors.converter.get_molecular_weight")
    def test_run_creates_dataframe_with_schema(
        self,
        mock_molecular_weight,
        mock_dipole,
        mock_max_force,
        mock_sum_force,
        mock_extract,
        mock_unique_symbols,
        mock_structure_valid,
        mock_count_water,
        default_config,
        mock_dataset_item,
    ):
        """Test that process creates DataFrame with proper schema."""
        # Setup mocks
        mock_extract.return_value = {
            "subset": "test_set",
            "composition": "CH4",
            "net_charge": 0,
            "num_atoms": 5,
            "spin_multiplicity": 0.0,
            "atomic_numbers": [6, 1, 1, 1, 1],
            "energy": -40.5,
        }
        mock_sum_force.return_value = 0.5
        mock_max_force.return_value = 1.0
        mock_dipole.return_value = 0.8
        mock_unique_symbols.return_value = "['C', 'H']"
        mock_molecular_weight.return_value = 16.0
        mock_structure_valid.return_value = True
        mock_count_water.return_value = 0

        processor = DatabaseProcessor(default_config)
        result = processor.process([mock_dataset_item])

        assert isinstance(result, pl.DataFrame)
        # Check that key columns exist
        assert "molecular_weight" in result.columns
        assert "entry_key" in result.columns
        assert result["molecular_weight"].null_count() == 0
        assert result["entry_key"].null_count() == 0

    def test_run_empty_list(
        self,
        default_config,
    ):
        """Test process method with empty list."""
        processor = DatabaseProcessor(default_config)
        result = processor.process([])

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    @patch("chemreporter.database_processors.converter.count_water_molecules")
    @patch("chemreporter.database_processors.converter.is_molecular_structure_valid")
    @patch("chemreporter.database_processors.converter.get_molecular_weight")
    @patch("chemreporter.database_processors.converter.get_unique_chemical_symbols")
    @patch("chemreporter.database_processors.converter.process_ase_info")
    @patch("chemreporter.database_processors.converter.get_net_forces_norm")
    @patch("chemreporter.database_processors.converter.get_max_forces_norm")
    @patch("chemreporter.database_processors.converter.calculate_dipole_moment")
    def test_process_clears_input_chunk_in_place(
        self,
        mock_dipole,
        mock_max_force,
        mock_sum_force,
        mock_extract,
        mock_unique_symbols,
        mock_molecular_weight,
        mock_structure_valid,
        mock_count_water,
        default_config,
        mock_dataset_item,
    ):
        """Regression: process empties data_chunk so atoms do not survive chunks.

        Guards the memory-leak fix that clears the input list in place; without
        it, ASE atoms stay referenced by the caller's loop variable across the
        next (expensive) chunk fetch.
        """
        mock_extract.return_value = {
            "subset": "test_set",
            "composition": "CH4",
            "net_charge": 0,
            "num_atoms": 5,
            "spin_multiplicity": 0.0,
            "atomic_numbers": [6, 1, 1, 1, 1],
            "energy": -40.5,
        }
        mock_sum_force.return_value = 0.5
        mock_max_force.return_value = 1.0
        mock_dipole.return_value = 0.8
        mock_unique_symbols.return_value = "['C', 'H']"
        mock_molecular_weight.return_value = 16.04
        mock_structure_valid.return_value = True
        mock_count_water.return_value = 0

        processor = DatabaseProcessor(default_config)
        data_chunk = [mock_dataset_item]
        processor.process(data_chunk)

        assert data_chunk == []
