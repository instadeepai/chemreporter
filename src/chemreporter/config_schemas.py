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
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from chemreporter.query_database_tools.query_tools import resolve_allowlist_columns


class GraphBasedProcessingConfig(BaseModel):
    """Configuration class for graph-based processing.

    Attributes:
        enable: Whether to enable graph-based processing. Defaults to False.
        nb_atoms_limit: The maximum number of atoms to process. Defaults to 200.
        subsets_skip_list: A list of subsets to skip.
    """

    enable: bool = False
    nb_atoms_limit: int = Field(default=200)
    subsets_skip_list: list[str] = Field(default_factory=list)


class SourceDatabaseMetadata(BaseModel):
    """Per-source-dataset DFT metadata (basis set, functional, dispersion correction).

    Metadata is optional, but if provided, every entry processed from the source
    database is set to these values in the query database.

    Attributes:
        basis_set: The basis set used for the DFT calculations.
        functional: The functional used for the DFT calculations.
        correction_term: The correction term used for the DFT calculations.
    """

    basis_set: str = ""
    functional: str = ""
    correction_term: str = ""


class ProcessDBConfig(BaseModel):
    """Configuration for the process command.

    Attributes:
        source_database_path: The (local or remote) path to the source database.
        query_database_path: The (local or remote) path to the query database.
        database_name: The name of the database. (User provided, will be used for all
            lines in the database).
        split_name: The name of the split. (User provided, will be used for all lines
            in the database). Defaults to "other".
        database_format: The format of the database.
        processing_chunk_size: The size of the processing chunk. Defaults to 50000.
        source_database_metadata: The DFT metadata (basis set, functional, correction
            term) for the source database.
        graph_based_processing: The nested configuration for graph-based processing.
    """

    source_database_path: str
    query_database_path: str
    database_name: str = Field(min_length=1)
    split_name: str = Field(default="other", min_length=1)
    database_format: str
    processing_chunk_size: int = Field(default=50000, gt=0)
    source_database_metadata: SourceDatabaseMetadata = Field(
        default_factory=SourceDatabaseMetadata
    )
    graph_based_processing: GraphBasedProcessingConfig = Field(
        default_factory=GraphBasedProcessingConfig
    )


class RestrictToConfig(BaseModel):
    """Allowlist filter applied after the SQL query (semi-join).

    Attributes:
        columns: The columns to filter on.
        path_to_values: The path to a single ".npy" or ".npz" file with the values
            to filter on. A ".npy" file only supports a single column. A ".npz"
            file supports one or more columns, stored as one named 1D array per
            column (key = column name).

    Example:
        1. Single column, ".npy" file
        columns: ["smiles"]
        path_to_values: path_to_file.npy
        File content: 1D array of smiles values, e.g. ["C1CCCCC1", "C1CCCCCC1"]

        2. Multiple columns, ".npz" file
        columns: ["smiles", "charge"]
        path_to_values: path_to_file.npz
        File content: one named 1D array per column, e.g.
        smiles=["C1CCCCC1", "C1CCCCCC1"], charge=[0, 1]
    """

    columns: list[str]
    path_to_values: str = Field(min_length=1)

    @field_validator("columns", mode="before")
    @classmethod
    def _normalize_columns(cls, value: Union[str, List[str]]) -> list[str]:
        """Normalize allowlist column names to a non-empty list.

        Args:
            value: A single column name or a list of names.

        Returns:
            Normalized column names.
        """
        return resolve_allowlist_columns(value)


class QueryDBConfig(BaseModel):
    """Configuration for the query command.

    Attributes:
        query_database_path: The path to the query database.
        results_path: The path to the results directory.
        query: The query to execute.
        sampling: Sampling options applied to the query results. Recognized
            keys: "method" (str, built-in sampler "random" or "all", or a
            path to a plugin file defining "custom_sampling_function";
            defaults to "random"), "n_samples" (int, number of entries to
            sample; if omitted, all matching entries are returned), "seed"
            (int, optional seed for built-in random sampling), "kwargs"
            (dict, extra arguments forwarded to a custom sampler), and
            "required_columns" (str or list of str, columns required by the
            custom sampler function. If not provided, only entry_keys are passed).
            Example: {"n_samples": 100000, "method": "random"}.
        actions: Post-query actions to run. Each item is either an action name
            (str) to run with no extra arguments, or a single-key dict mapping
            the action name to its argument. Supported action names:
            "make_statistics" and "make_histograms" (argument: list of column
            names), and "extract_smiles" (no argument). Example:
            ["extract_smiles", {"make_statistics": ["logp", "num_atoms"]}].
        restrict_to: The restrict to configuration.
    """

    query_database_path: str
    results_path: str
    query: str = Field(min_length=1)
    sampling: Dict[str, Any] = Field(default_factory=dict)
    actions: List[Any] = Field(default_factory=list)
    restrict_to: Optional[RestrictToConfig] = None


class ExportHDF5Config(BaseModel):
    """Configuration for the export command.

    Attributes:
        query_database_path: The (local or remote) path to the query database.
        keys_path: The (local or remote) path to the numpy file containing the keys.
        output_path: The (local or remote) path to the output HDF5 file.
        num_workers: The number of workers to use during export. (set multiprocessing
            to True if over 1).
        extras_fields: The extra fields to include in the output HDF5 file.
        num_files_to_export: The number of files to export. (Setting this to 1 might
            lead to memory issues on large datasets). Defaults to 1.
    """

    query_database_path: str
    keys_path: str
    output_path: str
    num_workers: int = Field(default=1, ge=1)
    extras_fields: Optional[List[str]] = None
    num_files_to_export: int = Field(default=1, ge=1)
