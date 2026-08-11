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
"""Helpers shared by the CLI commands and the database tools."""

from pathlib import Path

from cloudpathlib import CloudPath


def is_local_path(path: str | Path | CloudPath) -> bool:
    """Check if the path is a local path.

    A path is remote when it is a cloudpathlib path or carries a URI scheme, so
    every provider supported by cloudpathlib (S3, GCS, Azure Blob) is reported
    as remote, not only S3. Relative paths and plain strings are accepted.

    Args:
        path: Local or cloud path to check.

    Returns:
        bool: True if the path is a local path, False otherwise.
    """
    return not isinstance(path, CloudPath) and "://" not in str(path)
