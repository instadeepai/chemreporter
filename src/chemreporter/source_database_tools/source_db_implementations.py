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
"""Reader for source database currently  FairChem AseDBDataset."""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, List

import ase
from ase.calculators.singlepoint import SinglePointCalculator
from fairchem.core.datasets import AseDBDataset

from chemreporter.source_database_tools.database_item import DatasetItem
from chemreporter.source_database_tools.exceptions import SourceDatabaseReaderUsageError
from chemreporter.source_database_tools.omol25.reactivity_source_field_parser import (
    parse_ani2x_reactivity_source,
    parse_metal_complexes_reactivity_source,
    parse_reactivity_reactivity_source,
    parse_rgd_reactivity_source,
    parse_trans1x_reactivity_source,
)
from chemreporter.source_database_tools.open_catalyst_20.oc20_source_fields import (
    get_oc20_source_additional_fields,
)
from chemreporter.source_database_tools.open_catalyst_20.supplementary_info_impl import (  # noqa: E501
    OC20_DATA_MAPPING_FILE,
    load_oc20_supplementary_info,
)

logger = logging.getLogger("chemreporter")

NEUTRAL_DATABASE_CHARGE = 0
CLOSED_SHELL_DATABASE_SPIN_MULTIPLICITY = 1

FIELDS = {
    "aselmdb_omol": {
        "subset": "data_id",
        "net_charge": "charge",
        "num_atoms": "num_atoms",
        "spin_multiplicity": "spin",
        "composition": "composition",
    },
    "aselmdb_omat": {
        "subset": "task_type",
        "composition": "composition_reduced",
    },
    "aselmdb_odac": {},
    "xyz": {
        "subset": "config_type",
        "net_charge": "total_charge",
        "num_atoms": "num_atoms",
        "spin_multiplicity": "spin",
    },
    "xyz_oc": {
        "num_atoms": "num_atoms",
        # no net charge and spin multiplicity present
    },
    "aselmdb_omc": {},
    "aselmdb_oc": {
        "num_atoms": "num_atoms",
    },
}


FIELDS_HDF5_EXPORT = {
    "aselmdb_omol": {
        "data_id": "subset",
        "charge": "charge",
        "spin": "spin_multiplicity",
        "num_atoms": "num_atoms",
        "mulliken_charges": "mulliken_charges",
        "lowdin_charges": "lowdin_charges",
    },
    "xyz": {
        "config_type": "subset",
        "total_charge": "charge",
        "spin": "spin_multiplicity",
        "num_atoms": "num_atoms",
    },
    "aselmdb_odac": {},
    "aselmdb_omat": {
        "task_type": "subset",
        "spin": "spin_multiplicity",
        "stress": "stress",
    },
    "xyz_oc": {
        "num_atoms": "num_atoms",
    },
    "aselmdb_omc": {
        "stress": "stress",
    },
    "aselmdb_oc": {
        "num_atoms": "num_atoms",
    },
}


def correct_database_name(dataset_item: DatasetItem) -> dict[str, Any]:
    """Map dataset name to database_name and subset from the last token.

    Returns:
        ``database_name`` and ``subset`` for downstream keys.

    Raises:
        SourceDatabaseReaderUsageError: If the name has no delimiter for subset.
    """
    database_name = dataset_item.database_name
    parts = re.split(r"[^a-zA-Z0-9]+", database_name)
    if len(parts) > 1:
        subset = "-".join(parts[1:])
        database_name = parts[0]
    else:
        raise SourceDatabaseReaderUsageError(
            "For open catalyst 20 database and omat, the database "
            "name should contain at least one non-alphanumeric "
            "character to deduce the subset name. "
            "e. g. oc20-s2ef"
            "e. g. omat24-rattled-1000 "
        )
    return {"database_name": database_name, "subset": subset}


class SourceDatabaseImplementation(ABC):
    """Abstract class for source database implementation."""

    def __init__(self, database_format: str):
        """Initialize the source database implementation.

        Args:
            database_format: The format of the database.
        """
        self.database_format = database_format
        self.fields_name_mapping = FIELDS[database_format]
        self.fields_hdf5_export = FIELDS_HDF5_EXPORT[database_format]
        self.get_additional_fields: list[Callable[[DatasetItem], dict[str, Any]]] = []

    @abstractmethod
    def read_file(self, file_path: Path) -> Any:
        """Read the file at the given path.

        Caveat : this object needs to have a __len__ method that
            returns the number of items in the object.

        Args:
            file_path: The path to the file.

        Returns:
            Any: Object that can be read by get_atoms_from_db_pointer
        """
        pass

    @abstractmethod
    def get_atoms_from_db_pointer(self, db_pointer: Any, index: int) -> ase.Atoms:
        """Get the atoms from the database pointer.

        Args:
            db_pointer: Object returned by read_file.
            index: The index of the atoms object.

        Returns:
            ase.Atoms: The atoms object.
        """
        pass

    def close(self, db_pointer: Any) -> None:
        """Close the database pointer and release any associated resources.

        Args:
            db_pointer: Object returned by read_file.
        """
        pass

    def read_supplementary_info(
        self,
        files_dir: Path,
        files_index: dict[int, Path],
        download_function: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        """Read supplementary information for the database.

        Default implementation does nothing. Subclasses can override this
        method to read supplementary information if needed.

        Note : file_index correspond to the database files index.
        Suplementary info files names will be deduced from the db filenames.

        Args:
            files_dir: Path to the directory containing the database files.
            files_index: Dictionary of the db files with the index of the db files.
            download_function: Optional function to download the files.
            **kwargs: Additional keyword arguments for subclass implementations.
        """
        pass


class AseDBDatasetImplementation(SourceDatabaseImplementation):
    """Implementation of the source database implementation for ase.db.dataset files."""

    def read_file(self, file_path: Path) -> AseDBDataset:
        """Read the file at the given path.

        Args:
            file_path: The path to the file.

        Returns:
            AseDBDataset: The current file as an AseDBDataset object.

        Raises:
            FileNotFoundError: If the source database path is not local
            and no download function is provided.
        """
        return AseDBDataset({"src": str(file_path)})

    def get_atoms_from_db_pointer(
        self, db_pointer: AseDBDataset, index: int
    ) -> ase.Atoms:
        """Get the atoms from the database pointer.

        Args:
            db_pointer: AseDBDataset object.
            index: The index of the atoms object.

        Returns:
            ase.Atoms: The atoms object.
        """
        atoms = db_pointer.get_atoms(index)

        # AseDBDataset may return atoms without an attached calculator depending
        # on backend/version. Rehydrate from persisted payload when available.
        calc = getattr(atoms, "calc", None) or getattr(atoms, "_calc", None)
        if calc is None:
            results = {}
            if "energy" in atoms.info and atoms.info["energy"] is not None:
                results["energy"] = atoms.info["energy"]
            if "forces" in atoms.arrays and atoms.arrays["forces"] is not None:
                results["forces"] = atoms.arrays["forces"]
            if results:
                atoms.calc = SinglePointCalculator(atoms, **results)  # pyright: ignore[reportUndefinedVariable]

        return atoms

    def close(self, db_pointer: AseDBDataset) -> None:
        """Close the AseDBDataset and its underlying LMDB environments.

        Args:
            db_pointer: The AseDBDataset object.
        """
        for db in getattr(db_pointer, "dbs", []):
            if hasattr(db, "env") and hasattr(db.env, "close"):
                db.env.close()
            if hasattr(db, "close"):
                db.close()
        del db_pointer


class AselmdbDatabaseImplementationOmol25(AseDBDatasetImplementation):
    """Implementation of the source database implementation for aselmdb files."""

    def __init__(self) -> None:
        """Initialize the aselmdb database implementation."""
        super().__init__("aselmdb_omol")
        self.get_additional_fields = [self.get_additional_ase_info_omol25]
        self.file_extension = "aselmdb"

    def get_additional_ase_info_omol25(
        self, dataset_item: DatasetItem
    ) -> dict[str, Any]:
        """Get the additional ASE info fields for OMOL25.

        Args:
            dataset_item: DatasetItem object.

        Returns:
            dict: The additional ASE info fields.
        """
        atoms = dataset_item.atoms
        source_info_string = atoms.info.get("source", None)
        reference_source_info_string = atoms.info.get("reference_source", None)
        subset_name = atoms.info.get("data_id", None)
        # reactivity -subsets
        match subset_name:
            case "rgd":
                return parse_rgd_reactivity_source(source_info_string)
            case "reactivity":
                return parse_reactivity_reactivity_source(source_info_string)
            case "ani1":
                return parse_ani2x_reactivity_source(source_info_string)
            case "trans1x":
                return parse_trans1x_reactivity_source(source_info_string)
            case "metal_complexes":
                return parse_metal_complexes_reactivity_source(
                    source_info_string, reference_source_info_string
                )
            case _:
                return {}


class AselmdbDatabaseImplementationOmc25(AseDBDatasetImplementation):
    """ASE LMDB reader for OMC25 (Open Molecular Crystals)."""

    def __init__(self) -> None:
        """Initialize the OMC25 aselmdb implementation."""
        SourceDatabaseImplementation.__init__(self, "aselmdb_omc")
        self.get_additional_fields = [self.omc25_derived_fields]

    def omc25_derived_fields(self, dataset_item: DatasetItem) -> dict[str, Any]:
        """Get additional ASE info fields for OMC25 aselmdb entries.

        Net charge and spin are inferred via OpenBabel from geometry.

        Args:
            dataset_item: DatasetItem object.

        Returns:
            dict: The additional ASE info fields.
        """
        atoms = dataset_item.atoms

        return {
            "composition": atoms.get_chemical_formula(),
            "num_atoms": len(atoms),
            "net_charge": NEUTRAL_DATABASE_CHARGE,
            "spin_multiplicity": CLOSED_SHELL_DATABASE_SPIN_MULTIPLICITY,
        }


class AselmdbDatabaseImplementationOmat(AseDBDatasetImplementation):
    """Implementation of the source database implementation for aselmdb files."""

    def __init__(self) -> None:
        """Initialize the aselmdb database implementation."""
        super().__init__("aselmdb_omat")
        self.get_additional_fields = [
            self.get_additional_ase_info_omat,
            correct_database_name,
        ]
        self.file_extension = "aselmdb"

    def get_additional_ase_info_omat(self, dataset_item: DatasetItem) -> dict[str, Any]:
        """Get the additional ASE info fields for OMAT.

        Args:
            dataset_item: DatasetItem object.

        Returns:
            dict: The additional ASE info fields.
        """
        out = {}
        atoms = dataset_item.atoms
        out["num_atoms"] = len(atoms)
        out["net_charge"] = NEUTRAL_DATABASE_CHARGE
        return out


class AselmdbDatabaseImplementationOc20(
    AseDBDatasetImplementation,
):
    """ASE LMDB (FairChem) for OC20; shares field logic with ``xyz_oc20``."""

    def __init__(self) -> None:
        """Initialize OC20 aselmdb implementation."""
        super().__init__("aselmdb_oc")
        self.get_additional_fields = [
            get_oc20_source_additional_fields,
            correct_database_name,
        ]
        self.file_extension = "aselmdb"
        self._current_oc20_shard_filename: str | None = None

    def read_file(self, file_path: Path) -> AseDBDataset:
        """Open the LMDB shard; OC20 sidecar data is tied to this file.

        If supplementary metadata exists for this shard, it is merged later in
        ``get_atoms_from_db_pointer``. Missing supplementary files are allowed.

        Returns:
            ``AseDBDataset`` for this shard.
        """
        fname = Path(file_path).name
        self._current_oc20_shard_filename = fname
        return super().read_file(file_path)

    def get_atoms_from_db_pointer(
        self, db_pointer: AseDBDataset, index: int
    ) -> ase.Atoms:
        """Return atoms for ``index`` with per-frame fields from the sidecar row."""
        atoms = super().get_atoms_from_db_pointer(db_pointer, index)

        if "num_atoms" not in atoms.info:
            atoms.info["num_atoms"] = len(atoms)
        atoms.info["net_charge"] = NEUTRAL_DATABASE_CHARGE
        atoms.info["spin_multiplicity"] = CLOSED_SHELL_DATABASE_SPIN_MULTIPLICITY
        return atoms


class AselmdbDatabaseImplementationOdac23(AseDBDatasetImplementation):
    """Implementation of the source database implementation for aselmdb files."""

    def __init__(self) -> None:
        """Initialize the aselmdb database implementation."""
        super().__init__("aselmdb_odac")
        self.get_additional_fields = [self.get_additional_ase_info_odac]
        self.file_extension = "aselmdb_odac"

    def get_additional_ase_info_odac(self, dataset_item: DatasetItem) -> dict[str, Any]:
        """Get the additional ASE info fields for ODAC.

        Args:
            dataset_item: DatasetItem object.

        Returns:
            dict: The additional ASE info fields.
        """
        out = {}
        atoms = dataset_item.atoms
        out["num_atoms"] = len(atoms)
        out["net_charge"] = NEUTRAL_DATABASE_CHARGE
        out["spin_multiplicity"] = CLOSED_SHELL_DATABASE_SPIN_MULTIPLICITY
        out["composition"] = dataset_item.atoms.get_chemical_formula()
        return out


class XyzDatabaseImplementation(SourceDatabaseImplementation):
    """Implementation of the source database implementation for xyz files."""

    def __init__(self, database_format: str = "xyz"):
        """Initialize the xyz database implementation."""
        super().__init__(database_format)

    def read_file(self, file_path: Path) -> List[ase.Atoms]:
        """Read the file at the given path.

        Args:
            file_path: The path to the file.
            supplementary_info_file: Optional path to the supplementary info file.

        Returns:
            List[ase.Atoms]: List of ASE Atoms objects.
        """
        atoms_list = ase.io.read(str(file_path), index=":", format="extxyz")

        for atoms in atoms_list:
            results = {}
            if "energy" in atoms.info:
                results["energy"] = atoms.info["energy"]
            if "forces" in atoms.arrays:
                results["forces"] = atoms.arrays["forces"]

            if results:
                atoms.calc = SinglePointCalculator(atoms, **results)  # pyright: ignore[reportUndefinedVariable]

        return atoms_list

    def get_atoms_from_db_pointer(
        self, db_pointer: List[ase.Atoms], index: int
    ) -> ase.Atoms:
        """Get the atoms from the database pointer.

        Args:
            db_pointer: List of ASE Atoms objects.
            index: The index of the atoms object.

        Returns:
            ase.Atoms: The atoms object.
        """
        return db_pointer[index]


class XyzDatabaseImplementationOc20(XyzDatabaseImplementation):
    """Implementation of the source database implementation for open catalyst database.

    Usefull link :
        https://fair-chem.github.io/oc20/#dataset-changelog
    """

    def __init__(self) -> None:
        """Initialize the xyz database implementation."""
        SourceDatabaseImplementation.__init__(self, "xyz_oc")

        self.get_additional_fields = [
            get_oc20_source_additional_fields,
        ]

        self.parent_directory = None
        self.supplementary_info_file_name = OC20_DATA_MAPPING_FILE

        self.supplementary_info: dict[str, Any] = {}

    def read_supplementary_info(
        self,
        files_dir: Path,
        files_index: dict[int, Path],
        download_function: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        """Load OC20 pickle + per-shard sidecars into ``self.supplementary_info``."""
        self.supplementary_info = load_oc20_supplementary_info(
            files_dir,
            files_index,
            download_function,
        )

    def read_file(self, file_path: Path) -> List[ase.Atoms]:
        """Read the file at the given path.

        Args:
            file_path: The path to the file..

        Returns:
            List[ase.Atoms]: List of ASE Atoms objects.

        Raises:
            ValueError: If the supplementary info is not found/read.
        """
        atoms_list = super().read_file(file_path)

        for i, atoms in enumerate(atoms_list):
            # Use file_path.name for supplementary info lookup
            file_name = Path(file_path).name

            if file_name in self.supplementary_info:
                info_dict = self.supplementary_info[file_name].get(i, {})
            else:
                raise ValueError(f" '{file_name}' not found in supplementary info.")

            atoms.info.update(info_dict)

            if "num_atoms" not in atoms.info:
                atoms.info["num_atoms"] = len(atoms)
            atoms.info["net_charge"] = NEUTRAL_DATABASE_CHARGE
            atoms.info["spin_multiplicity"] = CLOSED_SHELL_DATABASE_SPIN_MULTIPLICITY

        return atoms_list
