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
"""Shared helper for archiving run configs as local YAML files."""

from pathlib import Path


def write_yaml(local_folder: Path, file_name: str, content: str) -> Path:
    """Write content (pre-rendered YAML text) to local_folder / file_name.

    Returns:
        The path of the written file.
    """
    path = local_folder / file_name
    path.write_text(content, encoding="utf-8")
    return path
