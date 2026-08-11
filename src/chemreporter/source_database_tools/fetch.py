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
from pathlib import Path
from typing import Callable, List

import yaml

from chemreporter.source_database_tools.database_reader import SourceDatabaseReader

logger = logging.getLogger("chemreporter")


CONFIG_FILE_PATTERN = "*.yaml"


def fetch_config_files(
    dir_path: os.PathLike,
) -> List[dict[str, str]]:
    """Read source database config files from a directory into dicts.

    Caveat: the source database path must use the same path type as the
    query database path, since ``SourceDatabaseReader`` can't take string
    paths.

    Args:
        dir_path: Path to the directory containing the source database
            config files.

    Returns:
        List of parsed config dicts, one per config file.

    Raises:
        ValueError: If `dir_path` is a string S3 path.
    """
    if isinstance(dir_path, str) and dir_path.startswith("s3://"):
        raise ValueError("string S3 paths are not supported")
    elif isinstance(dir_path, str):
        _dir_path = Path(dir_path)
    else:
        _dir_path = dir_path  # type: ignore[assignment]
    config_files = _dir_path.glob(CONFIG_FILE_PATTERN)
    config_list = []
    for config_file in config_files:
        with config_file.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            config_list.append(config)
    return config_list


def fetch_source_database_readers(
    dir_path: os.PathLike, download_function: Callable | None = None
) -> List[SourceDatabaseReader]:
    """Fetch source database readers from a directory.

    Args:
        dir_path: Path to the directory containing the source database config files.
        download_function: Function to download the database.

    Returns:
        List of source database reader objects.
    """
    source_database_readers_config_list = fetch_config_files(dir_path)
    source_database_readers: List[SourceDatabaseReader] = []

    for config in source_database_readers_config_list:
        source_database_readers.append(
            SourceDatabaseReader(
                database_name=config["database_name"],
                split_name=config["split_name"],
                db_path=type(dir_path)(config["files_path"]),  # type: ignore[call-arg]
                database_format=config["file_extension"],  # type: ignore[arg-type]
                download_function=download_function,
            )
        )
    logger.info("Loaded %s source database readers:\n", len(source_database_readers))
    for reader in source_database_readers:
        logger.info(" - %s - %s \n", reader.database_name, reader.split_name)
    return source_database_readers
