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

import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import polars as pl
from cloudpathlib import AnyPath, CloudPath

from chemreporter.source_database_tools.database_reader import is_local_path

ALLOWED_IO_OVERRIDES = {
    "init_cloud_client",
    "download_file",
    "upload_file",
    "upload_folder",
    "write_parquet",
    "read_parquet",
}


class ChemReporterIO:
    """Default I/O handler for CLI commands.

    Subclass and override individual methods, then pass the class or an
    instance to the CLI via ``CHEMREPORTER_IO`` in an ``--io-plugin`` file.
    """

    def __init__(self, functions_dict: dict[str, Callable] | None = None):
        """Initialize the handler, optionally replacing methods from a dict.

        Args:
            functions_dict: Mapping of I/O method names to callables that
                replace the default implementations on this instance.

        Raises:
            ValueError: If ``functions_dict`` contains an unknown method name.
        """
        for name, func in (functions_dict or {}).items():
            if name not in ALLOWED_IO_OVERRIDES:
                raise ValueError(f"Invalid I/O function name: {name}")
            setattr(self, name, func)

    def init_cloud_client(self) -> None:
        """Initialize cloud client if needed.

        With cloudpathlib, explicit initialization is usually not required
        if standard AWS environment variables (AWS_ACCESS_KEY_ID, etc.) are set.
        """
        pass

    def download_file(self, path: CloudPath, local_file_path: Path) -> None:
        """Download a file from a cloud path to a local path.

        Falls back to a local copy when the source path is already local,
        unless source and destination resolve to the same absolute path.

        Args:
            path: Source path (cloud or local).
            local_file_path: Destination local path.
        """
        if is_local_path(path):
            if str(path.absolute()) != str(local_file_path.absolute()):
                shutil.copy(path, local_file_path)
            return
        path.download_to(local_file_path)

    def upload_file(self, local_file_path: Path, dest_path: CloudPath) -> None:
        """Upload a local file to a destination path.

        Falls back to a local copy when the destination path is already local,
        unless source and destination resolve to the same absolute path.

        Args:
            local_file_path: Path to the local file.
            dest_path: Destination path (cloud or local).
        """
        if is_local_path(dest_path):
            if str(dest_path.absolute()) != str(local_file_path.absolute()):
                shutil.copy(local_file_path, dest_path)
            return
        dest_path.upload_from(local_file_path)

    def upload_folder(self, local_folder: Path, dest_path: CloudPath) -> None:
        """Upload a local folder to a destination path.

        Falls back to a local copy when the destination path is already local,
        unless source and destination resolve to the same absolute path.

        Args:
            local_folder: Path to the local folder.
            dest_path: Destination path (cloud or local).
        """
        if is_local_path(dest_path):
            if str(dest_path.absolute()) != str(local_folder.absolute()):
                shutil.copytree(local_folder, dest_path, dirs_exist_ok=True)
            return
        dest_path.upload_from(local_folder)

    def write_parquet(self, df: pl.DataFrame, output_path: CloudPath) -> None:
        """Write a Polars DataFrame to a parquet file.

        Writes directly to disk when the destination is local; uploads via a
        temporary file when it is a cloud path.

        Args:
            df: DataFrame to write.
            output_path: Destination path (cloud or local).
        """
        if is_local_path(output_path):
            df.write_parquet(output_path)
            return

        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            df.write_parquet(tmp.name)
            output_path.upload_from(tmp.name)

    def read_parquet(self, dir_path: CloudPath, schema: dict[str, Any] | None = None):
        """Read Parquet files from a directory (local or cloud) as a lazy scan.

        Extra columns present in the files but absent from the schema are
        silently ignored.

        Args:
            dir_path: Directory containing the parquet files.
            schema: Column schema passed to Polars when scanning. When None,
                the schema is inferred from the files.

        Returns:
            Polars LazyFrame scanning all parquet files in the directory.

        Raises:
            FileNotFoundError: If the directory does not exist or is not a
                directory.
            FileNotFoundError: If the directory contains no parquet files.
        """
        _dir_path = AnyPath(dir_path)
        if not _dir_path.exists() or not _dir_path.is_dir():
            raise FileNotFoundError(
                f"The path {dir_path} does not exist or is not a directory"
            )

        _db_files_list = [
            p.as_uri() if hasattr(p, "as_uri") else str(p)
            for p in _dir_path.glob("*.parquet")
        ]
        if len(_db_files_list) == 0:
            raise FileNotFoundError(f"No parquet files found in {str(_dir_path)}")

        return pl.scan_parquet(_db_files_list, schema=schema, extra_columns="ignore")
