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
"""Main converter class for Source Database to Query Database conversion."""

import gc
import logging
from typing import Any

import ase
import polars as pl

from chemreporter.analysis.ase_utils import (
    calculate_dipole_moment,
    get_max_forces_norm,
    get_molecular_weight,
    get_net_forces_norm,
    get_unique_chemical_symbols,
    process_ase_info,
)
from chemreporter.analysis.rdkit_utils import (
    calculate_graph_derived_properties,
    get_fpgen,
)
from chemreporter.analysis.structure_check import (
    count_water_molecules,
    is_molecular_structure_valid,
)
from chemreporter.config_schemas import (
    GraphBasedProcessingConfig,
    SourceDatabaseMetadata,
)
from chemreporter.query_database_tools.table_schemas import (
    FULL_SCHEMA,
    get_schema_dict,
)

logger = logging.getLogger("chemreporter")


class DatabaseProcessor:
    """Main converter class.

    internally uses ASE atoms objects
    """

    def __init__(
        self,
        graph_properties_config: GraphBasedProcessingConfig | None = None,
        database_info: SourceDatabaseMetadata | None = None,
    ):
        """Initialize converter with configuration.

        Args:
            graph_properties_config: Graph-based processing configuration object
            database_info: Optional source database metadata (basis_set, functional,
                correction_term); populates DFT metadata columns when provided
        """
        if graph_properties_config is None:
            graph_properties_config = GraphBasedProcessingConfig()
        self.graph_properties_config = graph_properties_config
        self.all_processed_keys: list[str] = []
        self._validated_schema = False  # Track if we've already validated
        self.database_info: dict[str, Any] = (
            database_info.model_dump() if database_info else {}
        )

    def process(self, data_chunk: list[ase.Atoms]) -> pl.DataFrame:
        """Convert ASE database file to configured storage backend.

        Clears ``data_chunk`` in place after the DataFrame is built so ASE
        atoms do not stay alive across chunks.

        Args:
            data_chunk: List of ASE atoms objects

        Returns:
            A Polars DataFrame with the properties of the ASE atoms object

        Raises:
            ValueError: If some columns are entirely null in the first chunk
        """
        datachunk_properties = []
        graph_enabled = self.graph_properties_config.enable
        if graph_enabled:
            fpgen = get_fpgen()
        else:
            fpgen = None
        for dataset_item in data_chunk:
            atoms = dataset_item.atoms
            atoms_properties: dict[str, Any] = {}
            atoms_properties.update(**self.database_info)
            atoms_properties.update({
                "entry_key": dataset_item.key,
                "database_name": dataset_item.database_name,
                "split_name": dataset_item.split_name,
            })

            atoms_properties.update(
                process_ase_info(atoms, fields_name_mapping=dataset_item.name_mapping)
            )

            for get_property_function in dataset_item.additional_fields:
                atoms_properties.update(**get_property_function(dataset_item))

            atoms_properties.update({
                "net_force_norm": get_net_forces_norm(atoms),
                "max_force_norm": get_max_forces_norm(atoms),
                "dipole_moment_magnitude": calculate_dipole_moment(atoms),
                "atomic_symbols": get_unique_chemical_symbols(atoms),
                "molecular_weight": get_molecular_weight(atoms),
                "is_molecular_structure_valid": is_molecular_structure_valid(atoms),
                "num_water_molecules": count_water_molecules(atoms),
            })

            if graph_enabled:
                is_candidate = self.is_graph_properties_candidate(atoms_properties)
                atoms_properties["graph_properties_candidate"] = is_candidate

                # Use fpgen only if it's a candidate (under atom limit and not skipped)
                # Otherwise pass None to skip fingerprints but still get other
                # RDKit-based properties.
                effective_fpgen = fpgen if is_candidate else None

                graph_properties = calculate_graph_derived_properties(
                    atoms, atoms_properties["net_charge"], effective_fpgen
                )
                # flag error if graph properties are not computed
                if len(graph_properties) == 0 or None in graph_properties.values():
                    atoms_properties["error_graph_properties"] = True
                else:
                    atoms_properties["error_graph_properties"] = False
                atoms_properties.update(**graph_properties)
            else:
                # When graph properties are disabled, mark all as non-candidates
                atoms_properties["graph_properties_candidate"] = False

            datachunk_properties.append(atoms_properties)

        query_db_chunk = pl.DataFrame(
            datachunk_properties, schema=get_schema_dict(FULL_SCHEMA)
        )

        # Drop per-chunk accumulators so they do not survive into the next chunk.
        del datachunk_properties
        data_chunk.clear()
        gc.collect()

        return query_db_chunk

    def is_graph_properties_candidate(self, properties: dict[str, Any]) -> bool:
        """Check if the atoms object is a candidate for graph properties computation.

        Args:
            properties: Properties of the ASE atoms object

        Returns:
            True if the atoms object is a candidate for graph
            properties computation, False otherwise
        """
        num_atoms = properties["num_atoms"]
        subset = properties["subset"]
        if self.graph_properties_config.enable is False:
            return False
        if num_atoms > self.graph_properties_config.nb_atoms_limit:
            return False
        if subset in self.graph_properties_config.subsets_skip_list:
            return False
        return True
