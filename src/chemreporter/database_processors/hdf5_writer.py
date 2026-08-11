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
import logging
import os
from typing import Any, List

import ase
import h5py
import numpy as np
import polars as pl
from ase.stress import voigt_6_to_full_3x3_stress

from chemreporter.source_database_tools.database_reader import (
    SourceDatabaseReader,
    parse_key,
)

logger = logging.getLogger("chemreporter")


def _ase_atoms_to_hdf5(
    atoms: ase.Atoms,
    dataset_file: h5py.File,
    stem: str,
    fields_hdf5_export: dict[str, str],
    extras_fields: dict | None = None,
):
    """Write an ASE atoms object to a hdf5 file.

    Every field in the atoms object is written to the hdf5 file.

    Args:
        atoms: ASE atoms object.
        dataset_file: HDF5 file to write to.
        stem: Stem of the hdf5 file.
        fields_hdf5_export: Dictionary of fields to export to the hdf5 file.
        extras_fields: Optional dictionary of extra fields to write to hdf5.

    Raises:
        ValueError: If forces are not present in the atoms object.
    """
    # extract all necessary fields
    positions = atoms.positions
    elements = atoms.numbers
    energy = atoms._calc.results.get("energy", None)
    forces = atoms._calc.results.get("forces", None)
    pbc = atoms._calc.results.get("pbc", None)
    if pbc is None:
        pbc = np.asarray(atoms.pbc, dtype=np.float32)
    stress = atoms._calc.results.get("stress", None)

    if forces is None:
        raise ValueError("Forces are not present in the atoms object")

    current_data = dataset_file.create_group(stem)

    # Standard fields
    if energy is not None:
        current_data.attrs["energy"] = energy

    # Required datasets
    current_data.create_dataset("elements", data=elements)
    current_data.create_dataset("positions", data=positions, dtype=np.float64)
    current_data.create_dataset("forces", data=forces, dtype=np.float64)
    current_data.create_dataset("pbc", data=pbc, dtype=np.float32)
    if stress is not None:
        stress = np.asarray(stress, dtype=np.float64)
        if stress.shape == (6,):
            stress = voigt_6_to_full_3x3_stress(stress)
        current_data.create_dataset("stress", data=stress)

    # Dynamically export all atoms.info fields with original names
    for key in atoms.info.keys():
        if key not in fields_hdf5_export.keys():
            continue
        field_name = fields_hdf5_export[key]
        value = atoms.info[key]
        # Skip None values (they cause object dtype errors in HDF5)
        if value is None:
            continue
        if type(value) is np.ndarray:
            current_data.create_dataset(field_name, data=value, dtype=np.float64)
        else:
            current_data.attrs[field_name] = value

    if extras_fields is not None and len(extras_fields) > 0:
        current_data.create_group("extras")
        for field_name, value in extras_fields.items():
            if field_name == "entry_key":
                continue
            if value is None:
                continue
            if isinstance(value, (np.ndarray, list)):
                current_data["extras"].create_dataset(
                    field_name, data=np.array(value), dtype=np.float64
                )
            else:
                current_data["extras"].attrs[field_name] = value


def make_entry_key_lookup(
    extras_fields: pl.DataFrame | None,
) -> dict[str, dict[str, Any]]:
    """Convert a polars DataFrame to a dictionary for faster lookup.
    The dictionary is indexed by the entry_key and contains the values.

    Args:
        extras_fields: Polars DataFrame.

    Returns:
        Dictionary of extras fields.
    """
    if extras_fields is None or extras_fields.is_empty():
        return {}

    value_cols = [c for c in extras_fields.columns if c != "entry_key"]

    return dict(
        zip(
            extras_fields["entry_key"],
            extras_fields.select(value_cols).to_dicts(),
        )
    )


def write_hdf5(
    key_entries: list[str],
    source_db_readers: List[SourceDatabaseReader] | SourceDatabaseReader,
    hdf5_path: str,
    extras_fields: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Write a hdf5 file from a collection of key entries.

    Args:
        key_entries: List of key entries.
        source_db_readers: List of source database readers.
        hdf5_path: Path to the hdf5 file.
        extras_fields: Dictionary of extras fields. (entry_key -> {field_name -> value})
            Can be created using make_entry_key_lookup.

    Raises:
        RuntimeError: If the key entry is not found in any source database reader.
    """
    # Reproduce old behaviour :
    if isinstance(source_db_readers, SourceDatabaseReader):
        source_db_readers = [source_db_readers]

    with h5py.File(hdf5_path, "w", track_order=False, libver="latest") as f:
        # first identify all keys thst comes from the same files
        processed_keys = set()

        key_entries_set = set(key_entries)
        for key_entry in key_entries_set:
            if key_entry in processed_keys:
                continue
            # only added line compared to write_hdf5_from_indexes
            key_entry_parsed = parse_key(key_entry)

            for source_db_reader in source_db_readers:
                if (
                    source_db_reader.database_name == key_entry_parsed.database_name
                    and source_db_reader.split_name == key_entry_parsed.split_name
                ):
                    for system, system_key in source_db_reader.iter_atoms(
                        key_from_file=key_entry,
                        key_entries=key_entries_set,
                    ):
                        processed_keys.add(system_key)
                        _ase_atoms_to_hdf5(
                            atoms=system,
                            dataset_file=f,
                            stem=system_key,
                            fields_hdf5_export=source_db_reader.source_db_impl.fields_hdf5_export,
                            extras_fields=extras_fields.get(system_key)
                            if extras_fields
                            else None,
                        )
                    break  # Stop after first match
            else:
                raise RuntimeError(
                    f"Key {key_entry} not found in any source database reader\n"
                    f"key_entry: {key_entry_parsed.database_name} -"
                    f"{key_entry_parsed.split_name}\n"
                    f"source_db_readers: {source_db_readers}\n"
                )


def merge_hdf5_files(
    input_files: list[os.PathLike], output_file_path: os.PathLike
) -> None:
    """Merge list of HDF5 files into one and save to local file.

    Args:
        input_files: list of paths to HDF5 files to merge
        output_file_path: Path to the local file to save the merged HDF5 file
    """
    with h5py.File(output_file_path, "w", libver="latest", track_order=False) as out_h5:
        for input_file in input_files:
            with h5py.File(input_file, "r") as src_h5:
                for key in src_h5.keys():
                    src_h5.copy(key, out_h5)
