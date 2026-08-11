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
"""Shared local/remote path resolution for CLI commands."""

from pathlib import Path

from cloudpathlib import AnyPath

from chemreporter.path_utils import is_local_path


def resolve_local_path(io_handler, path, temp_dir) -> Path:
    """Return path as a local Path, downloading it into temp_dir first if remote."""
    any_path = AnyPath(path)
    if is_local_path(any_path):
        return Path(str(any_path))

    local_path = Path(temp_dir) / any_path.name
    io_handler.download_file(any_path, local_path)
    return local_path


def check_output_not_exists(output_path, num_files_to_export: int) -> None:
    """Raise FileExistsError if the export output(s) already exist.

    Raises:
        FileExistsError: If the output (or an existing part) already exists.
    """
    if num_files_to_export == 1:
        if output_path.exists():
            raise FileExistsError(
                f"Output path {output_path} already exists, not overwriting"
            )
    else:
        existing_files = list(
            output_path.parent.glob(f"{output_path.stem}_part_*.hdf5")
        )
        if existing_files:
            raise FileExistsError(
                f"Output path {existing_files} already exists, not overwriting"
            )
