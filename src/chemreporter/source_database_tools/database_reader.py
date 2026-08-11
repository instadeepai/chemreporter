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

import gc
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generator, Literal

import ase
from pydantic import Field

from chemreporter.path_utils import is_local_path
from chemreporter.source_database_tools.database_item import DatasetItem
from chemreporter.source_database_tools.exceptions import SourceDatabaseReaderUsageError
from chemreporter.source_database_tools.source_db_implementations import (
    AselmdbDatabaseImplementationOc20,
    AselmdbDatabaseImplementationOdac23,
    AselmdbDatabaseImplementationOmat,
    AselmdbDatabaseImplementationOmc25,
    AselmdbDatabaseImplementationOmol25,
    SourceDatabaseImplementation,
    XyzDatabaseImplementation,
    XyzDatabaseImplementationOc20,
)

logger = logging.getLogger("chemreporter")


DatabaseFormat = Literal[
    "aselmdb_omol",
    "aselmdb_omat",
    "aselmdb_oc",
    "aselmdb_odac",
    "aselmdb_omc",
    "xyz",
    "xyz_oc",
]
DatabaseFileExtension = Literal["extxyz.xz", "extxyz", "aselmdb"]


@dataclass
class KeyEntry:
    """KeyEntry class.

    Args:
        database_name: Name of the database.
        split_name: Name of the split.
        source_file_name: Name of the source file.
        index: Index of the key entry.
        subset: Subset of the entry. (Optional)
    """

    database_name: str
    split_name: str
    source_file_name: str
    index: int

    def to_key(self) -> str:
        """Convert the key entry to a key string.

        Returns:
            str: The key string.
        """
        return (
            f"{self.database_name}_"
            f"{self.split_name}_"
            f"{self.source_file_name.replace('_', '$')}_"
            f"{self.index}"
        )


def parse_key(key_entry: str) -> KeyEntry:
    """Parse the key entry.

    Args:
        key_entry: Key entry to parse.

    Returns:
        KeyEntry: KeyEntry object.

    Raises:
        SourceDatabaseReaderUsageError: If the key entry is invalid.
        ValueError: If the key entry is invalid.
    """
    key_parts = key_entry.split("_")
    if len(key_parts) != 4:
        raise ValueError("key should be of length 4")

    database_name, split_name, source_file_name, index = key_parts
    return KeyEntry(
        database_name=database_name,
        split_name=split_name,
        source_file_name=source_file_name.replace("$", "_"),
        index=int(index),
    )


class SourceDatabaseReader:
    """Reader for source database.

    This class read the source database and return a chunk of atoms.
    If the source database is not local, the user must provide a download function
    to the AseDBReader class
    Through the class, "current" refers to the file being read in the original database.
    While "local" refers to the file being accessed. They might be the same or
    different.

    Args:
        db_path: Path to the source database
        chunk_size: Size of the chunk to read
        download_function: Function to download the database files
    """

    def __init__(
        self,
        database_name: str,
        split_name: str,
        db_path: os.PathLike,
        chunk_size: int = 1,
        download_function: Callable | None = None,
        database_format: DatabaseFormat = "aselmdb_omol",
    ):
        """Initialize the SourceDatabaseReader.

        Args:
            database_name: Name of the database (used in key generation).
            split_name: Name of the split (e.g., 'train', 'val', 'test').
            db_path: Path to the source database.
            chunk_size: Size of the chunk to read.
            download_function: Function to download the database files.
            database_format: Format of the database files.

        Raises:
            SourceDatabaseReaderUsageError: If the database format is not supported.
        """
        self._file_current_index = -1
        self.chunk_size = chunk_size
        self.download_function = download_function
        # files_path can be one file or a glob pattern
        # or the parent dir path of the files

        self.files_path = db_path

        self.database_name = database_name.replace("_", "-").lower()
        self.split_name = split_name.replace("_", "-").lower()

        self._current_index = 0
        self._files_index: dict[int, Path] = Field(default_factory=dict)
        self.database_format: DatabaseFormat = database_format
        self.file_extension: DatabaseFileExtension | DatabaseFormat = database_format

        # Declare the type as the base class
        self.source_db_impl: SourceDatabaseImplementation
        match database_format:
            case "aselmdb_omol":
                self.source_db_impl = AselmdbDatabaseImplementationOmol25()
                self.file_extension = "aselmdb"
            case "aselmdb_omat":
                self.source_db_impl = AselmdbDatabaseImplementationOmat()
                self.file_extension = "aselmdb"
            case "aselmdb_oc":
                self.source_db_impl = AselmdbDatabaseImplementationOc20()
                self.file_extension = "aselmdb"
            case "aselmdb_odac":
                self.source_db_impl = AselmdbDatabaseImplementationOdac23()
                self.file_extension = "aselmdb"
            case "xyz":
                self.source_db_impl = XyzDatabaseImplementation()
            case "aselmdb_omc":
                self.source_db_impl = AselmdbDatabaseImplementationOmc25()
                self.file_extension = "aselmdb"
            case "xyz_oc":
                self.source_db_impl = XyzDatabaseImplementationOc20()
                self.file_extension = "extxyz.xz"
            case _:
                raise SourceDatabaseReaderUsageError(
                    f"Unsupported database format: {database_format}"
                )

        self.num_files = 0  # initialized by _index_all_files()
        self.files_dir: Path = Path()  # initialized by _index_all_files()
        self._index_all_files()

        self.source_db_impl.read_supplementary_info(
            download_function=self.download_function,
            files_dir=self.files_dir,
            files_index=self._files_index,
        )

        self.temp_local_file: Path = Path()
        self.temp_supplementary_info_file: Path = Path()
        self.temp_dir: Path | None = None

    def _index_all_files(self) -> None:
        """Index all ASE database files.

        - update self.num_files
        - update self._files_index

        Raises:
            FileNotFoundError: If no database files found at self.files_path

        """
        if self.files_path.is_dir():  # type: ignore[attr-defined]
            self._files_index = dict(
                enumerate(self.files_path.glob(f"*.{self.file_extension}"))  # type: ignore[attr-defined]
            )  # type: ignore[attr-defined]
            self.files_dir = self.files_path  # type: ignore[assignment]
        else:
            # able to read glob pattern of single files
            glob_pattern = self.files_path.name  # type: ignore[attr-defined]
            path_glob = self.files_path.parent  # type: ignore[attr-defined]
            self.files_dir = path_glob
            self._files_index = dict(enumerate(path_glob.glob(glob_pattern)))
        self.num_files = len(self._files_index)
        logger.debug("Indexed %s  database files", self.num_files)
        if self.num_files == 0:
            raise FileNotFoundError(f"No database files found at {self.files_path}")

    def _local_path_name(self, index: int) -> Path:
        """Resolve a file index to a local path.

        Args:
            index: The index of the file in self._files_index.

        Returns:
            Path: The local path to the file.
        """
        # Path(CloudPath) implicitly downloads via os.fspath() - check first.
        file_path = self._files_index[index]

        if is_local_path(file_path):
            return Path(file_path)

        if self.temp_dir is not None:
            return self.temp_dir / file_path.name

        return Path(file_path)

    def read_file_iter(self) -> Any:
        """Read the current file and increment self._file_current_index.

        Returns:
            Any: The current file as an implementation-specific object.

        Raises:
            StopIteration: If the current file index is greater than the num of files.
            SourceDatabaseReaderUsageError: If the source database path is not local
                and no download function is provided.
        """
        if self._file_current_index >= self.num_files - 1:
            logger.debug("Done ! ALL FILES READ")
            return

        self._file_current_index += 1
        file_path = self._files_index[self._file_current_index]
        local_file_path = self._local_path_name(self._file_current_index)

        if not is_local_path(file_path):
            if self.download_function:
                self.download_function(file_path, local_file_path)
            else:
                raise SourceDatabaseReaderUsageError(
                    "If source database path is not local, "
                    "user MUST provide a download function "
                    "to the AseDBReader class"
                )

        return self.source_db_impl.read_file(local_file_path)

    def _resolve_target_keys(
        self,
        key_from_file: str,
        key_entries: set[str],
    ) -> tuple[Path, list[str], list[int]]:
        """Resolve the file and keys/indexes from ``key_entries`` sharing its file.

        Args:
            key_from_file: A key representing the file to fetch from.
            key_entries: Set of key entries to fetch.

        Returns:
            tuple[Path, list[str], list[int]]: The source file path, and the
            matching key entries and indexes found in ``key_entries``.

        Raises:
            SourceDatabaseReaderUsageError: If the database name or split name mismatch.
        """
        target_key_parsed = parse_key(key_from_file)

        if (
            target_key_parsed.database_name != self.database_name
            or target_key_parsed.split_name != self.split_name
        ):
            raise SourceDatabaseReaderUsageError("Database name or split name mismatch")

        file_path = (
            self.files_dir
            / f"{target_key_parsed.source_file_name}.{self.file_extension}"
        )
        extracted_key_entries = []
        extracted_indexes = []
        logger.debug("key_from_file: %s", key_from_file)
        logger.debug("Fetching all atoms from %s", file_path)
        for key_entry in key_entries:
            key_parsed = parse_key(key_entry)
            if (
                key_parsed.database_name == self.database_name
                and key_parsed.split_name == self.split_name
            ):
                if target_key_parsed.source_file_name == key_parsed.source_file_name:
                    extracted_key_entries.append(key_entry)
                    extracted_indexes.append(key_parsed.index)  # type: ignore

        return file_path, extracted_key_entries, extracted_indexes

    @contextmanager
    def _open_db_pointer(self, file_path: Path) -> Generator[Any, None, None]:
        """Open the db pointer for ``file_path``, downloading it first if needed.

        The file is downloaded into a temporary directory that is cleaned up
        once the caller is done with the db pointer.

        Args:
            file_path: The source file to open.

        Yields:
            Any: The implementation-specific db pointer for ``file_path``.
        """
        with tempfile.TemporaryDirectory() as _temp_dir_str:
            if self.download_function:
                temp_dir = Path(_temp_dir_str)
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_local_file = temp_dir / file_path.name
                self.download_function(file_path, temp_local_file)
                yield self.source_db_impl.read_file(temp_local_file)
            else:
                yield self.source_db_impl.read_file(file_path)

    def fetch_atoms(
        self,
        key_from_file: str,
        key_entries: set[str],
    ) -> tuple[list[ase.Atoms], list[str]]:
        """Fetch all atoms from the database for the given key entries.

        Args:
            key_from_file: A key representing the file to fetch from.
            key_entries: Set of key entries to fetch.

        Returns:
            tuple[list[ase.Atoms], list[str]]: The atoms objects and their keys.

        Raises:
            SourceDatabaseReaderUsageError: If the database name or split name mismatch.
        """
        file_path, extracted_key_entries, extracted_indexes = self._resolve_target_keys(
            key_from_file, key_entries
        )
        with self._open_db_pointer(file_path) as db_pointer:
            atoms_list = [
                self.source_db_impl.get_atoms_from_db_pointer(db_pointer, index)
                for index in extracted_indexes
            ]

        return atoms_list, extracted_key_entries

    def iter_atoms(
        self,
        key_from_file: str,
        key_entries: set[str],
    ) -> Generator[tuple[ase.Atoms, str], None, None]:
        """Iterate over all atoms from the database for the given key entries.

        Args:
            key_from_file: A key representing the file to fetch from.
            key_entries: Set of key entries to fetch.

        Yields:
            tuple[list[ase.Atoms], list[str]]: The atoms objects and their keys.

        Raises:
            SourceDatabaseReaderUsageError: If the database name or split name mismatch.
        """
        file_path, extracted_key_entries, extracted_indexes = self._resolve_target_keys(
            key_from_file, key_entries
        )
        with self._open_db_pointer(file_path) as db_pointer:
            for index, key in zip(extracted_indexes, extracted_key_entries):
                atoms = self.source_db_impl.get_atoms_from_db_pointer(db_pointer, index)
                yield atoms, key

    def fetch_atoms_from_key_index(self, key_entry: str) -> ase.Atoms:
        """Fetch a single atoms object from the database by key.

        Args:
            key_entry: The key entry to fetch.

        Returns:
            ase.Atoms: The atoms object.

        Raises:
            SourceDatabaseReaderUsageError: If the database name or split name mismatch.
        """
        key_parsed = parse_key(key_entry)
        if (
            key_parsed.database_name != self.database_name
            or key_parsed.split_name != self.split_name
        ):
            raise SourceDatabaseReaderUsageError("Database name or split name mismatch")
        file_path = (
            self.files_dir / f"{key_parsed.source_file_name}.{self.file_extension}"
        )
        db_pointer = self.source_db_impl.read_file(file_path)
        atoms = self.source_db_impl.get_atoms_from_db_pointer(
            db_pointer, key_parsed.index
        )
        self.source_db_impl.close(db_pointer)
        return atoms

    def file_chunk(self, chunk_size: int) -> Generator[list[DatasetItem], None, None]:
        """Generate a chunk of atoms from the database.

        Actions:
            - read the current file.
            - generate a chunk of atoms.

        Args:
            chunk_size: Size of the chunk to read.

        Yields:
            a generator of a list of DatasetItem objects.
        """
        is_database_local = is_local_path(self.files_path)

        for _ in range(self.num_files):
            db_pointer = self.read_file_iter()
            if db_pointer:
                for index in range(0, len(db_pointer), chunk_size):
                    end_index = min(index + chunk_size, len(db_pointer))

                    atoms_list = []
                    for i in range(index, end_index):
                        filename = self._files_index[
                            self._file_current_index
                        ].name.replace(f".{self.file_extension}", "")
                        key_entry = KeyEntry(
                            database_name=self.database_name,
                            split_name=self.split_name,
                            source_file_name=filename,
                            index=i,
                        ).to_key()
                        atoms = self.source_db_impl.get_atoms_from_db_pointer(
                            db_pointer, i
                        )
                        atoms_list.append(
                            DatasetItem(
                                database_name=self.database_name,
                                split_name=self.split_name,
                                key=key_entry,
                                atoms=atoms,
                                name_mapping=self.source_db_impl.fields_name_mapping,
                                additional_fields=self.source_db_impl.get_additional_fields,
                            )
                        )
                    yield atoms_list
                del db_pointer
                gc.collect()
                local_file_path = self._local_path_name(self._file_current_index)
                if not is_database_local and local_file_path.exists():
                    local_file_path.unlink()
            else:
                return  # StopIteration

    def __iter__(self) -> Generator[list[DatasetItem], None, None]:
        """Iterate through all atoms in the database, optionally in chunks.

        Yields:
            Generator[list[DatasetItem], None, None]: A generator of lists of
            DatasetItem objects.
        """
        # Loop through all files in the database
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a placeholder - actual file name will be used in read_file_iter
            self.temp_dir = Path(temp_dir)
            self.temp_supplementary_info_file = Path(temp_dir) / "supplementary_info"

            # Yield chunks of atoms
            for chunk in self.file_chunk(self.chunk_size):
                yield chunk

    def get_source_database_info(self) -> dict[str, Any]:
        """Get the source database configuration.

        Returns:
            dict[str, Any]: The source database configuration.
        """
        return {
            "database_name": self.database_name,
            "split_name": self.split_name,
            "file_extension": self.database_format,
            "files_path": str(self.files_path),
        }
