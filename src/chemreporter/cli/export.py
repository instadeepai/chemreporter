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
import tempfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import polars as pl
from cloudpathlib import AnyPath
from omegaconf import OmegaConf

from chemreporter.cli.helpers.export_batches import _subset_extras, build_export_batches
from chemreporter.cli.helpers.hdf5_export import (
    _initialize_hdf5_worker,
    _write_hdf5_multiprocess_worker,
    merge_hdf5_files,
)
from chemreporter.cli.helpers.path_utils import (
    check_output_not_exists,
    resolve_local_path,
)
from chemreporter.cli.io_utils import ChemReporterIO
from chemreporter.config_schemas import ExportHDF5Config
from chemreporter.database_processors.hdf5_writer import make_entry_key_lookup
from chemreporter.path_utils import is_local_path
from chemreporter.query_database_tools.query_database import (
    QueryDatabaseHandler,
)
from chemreporter.source_database_tools.database_reader import parse_key
from chemreporter.source_database_tools.fetch import fetch_source_database_readers

logger = logging.getLogger("chemreporter")


def group_keys_by_source_database_split_and_file(
    keys_list,
) -> dict[str, list[str]]:
    """Group keys by source database, split, and file.

    We could have split the keys randomly and  given each group the same number
    of keys, but more efficient to group them by source database, split, and file.
    source files are only downloaded once limiting the number of concurrent downloads.

    Args:
        keys_list: List of keys.

    Returns:
        Dictionary mapping (database_name, split_name, source_file_name)
        to list of keys.
    """
    grouped_keys = defaultdict(list)
    for key in keys_list:
        parsed = parse_key(key)
        group_id = (
            f"{parsed.database_name}_{parsed.split_name}_{parsed.source_file_name}"
        )
        grouped_keys[group_id].append(key)
    return dict(grouped_keys)


def extract_extras_fields(
    keys_list: list[str],
    query_database_path: str,
    extras_fields: list[str],
    read_func,
) -> dict[str, dict[str, Any]]:
    """Extract extras fields from the query database.

    Args:
        keys_list: List of keys to filter.
        query_database_path: Path to the query database.
        extras_fields: names of the extra fields to extract.
        read_func: Function to read the query database.

    Returns:
        Dictionary mapping entry_key to extra field values.
    """
    query_database_handler = QueryDatabaseHandler(
        AnyPath(query_database_path), read_func=read_func
    )

    pl_df = query_database_handler.get_dataframe(
        indices=keys_list, columns=extras_fields
    )

    if isinstance(pl_df, pl.LazyFrame):
        pl_df = pl_df.collect()

    return make_entry_key_lookup(pl_df)


def run_main(config, io_handler: ChemReporterIO):
    """Main function to run the export script.

    Args:
        config: OmegaConf config object.
        io_handler: I/O handler for cloud/local file operations.

    Raises:
        FileExistsError: If the output path already exists.
    """
    io_handler.init_cloud_client()

    # Unpack and validate config
    validated_config = ExportHDF5Config(**OmegaConf.to_container(config, resolve=True))

    output_path = AnyPath(validated_config.output_path)
    check_output_not_exists(output_path, validated_config.num_files_to_export)

    num_workers = validated_config.num_workers

    if num_workers > 1:
        is_multiprocess = True
    else:
        is_multiprocess = False
    output_path_is_remote = not is_local_path(validated_config.output_path)

    with tempfile.TemporaryDirectory() as temp_output_dir:
        if output_path_is_remote or is_multiprocess:
            local_output_folder = AnyPath(temp_output_dir)
        else:
            local_output_folder = AnyPath(validated_config.output_path).parent

        keys_path = resolve_local_path(
            io_handler, validated_config.keys_path, temp_output_dir
        )
        keys = np.load(keys_path, allow_pickle=True)

        keys_list = keys.ravel().tolist()
        del keys

        if validated_config.extras_fields:
            logger.info("Extracting extras fields: %s", validated_config.extras_fields)
            extras_fields = extract_extras_fields(
                keys_list,
                str(validated_config.query_database_path),
                validated_config.extras_fields,
                io_handler.read_parquet,
            )
        else:
            extras_fields = None

        if is_multiprocess or validated_config.num_files_to_export > 1:
            grouped_keys = group_keys_by_source_database_split_and_file(keys_list)

            logger.info(
                "Grouped %s keys into %s groups",
                len(keys_list),
                len(grouped_keys),
            )
            del keys_list
            worker_kwargs = [
                {
                    "keys_list": group_keys,
                    "output_file": str(local_output_folder / f"group_{group_id}.hdf5"),
                    "extras_fields": _subset_extras(extras_fields, group_keys),
                }
                for group_id, group_keys in grouped_keys.items()
            ]
        else:
            worker_kwargs = [
                {
                    "keys_list": keys_list,
                    "output_file": str(local_output_folder / output_path.name),
                    "extras_fields": extras_fields,
                }
            ]
        if len(worker_kwargs) == 0:
            logger.error("No keys to export")
            return

        source_database_readers = fetch_source_database_readers(
            AnyPath(str(validated_config.query_database_path)),
            download_function=io_handler.download_file,
        )

        export_batches = build_export_batches(
            worker_kwargs,
            num_files_to_export=validated_config.num_files_to_export,
            is_multiprocess=is_multiprocess,
            output_path=output_path,
            local_output_folder=local_output_folder,
            extras_fields=extras_fields,
        )

        for output_path_name, batch_worker_kwargs in export_batches:
            with Pool(
                processes=num_workers,
                initializer=_initialize_hdf5_worker,
                initargs=(source_database_readers,),
                maxtasksperchild=1,
            ) as pool:
                for kwargs in batch_worker_kwargs:
                    kwargs["extras_fields"] = cast(
                        Dict[str, Dict[str, Any]] | None,
                        kwargs.get("extras_fields", extras_fields),
                    )

                pool.map(_write_hdf5_multiprocess_worker, batch_worker_kwargs)

            if is_multiprocess:
                local_hdf5_files = [
                    AnyPath(kwargs["output_file"]) for kwargs in batch_worker_kwargs
                ]
                logger.info("Merging %s files", len(local_hdf5_files))
                merge_hdf5_files(
                    local_hdf5_files, local_output_folder / output_path_name
                )
                for file in local_hdf5_files:
                    Path(file).unlink()

            if output_path_is_remote:
                io_handler.upload_file(
                    local_output_folder / output_path_name,
                    output_path.parent / output_path_name,
                )
                Path(local_output_folder / output_path_name).unlink()
