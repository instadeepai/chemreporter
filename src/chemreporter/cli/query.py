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
import time
from pathlib import Path

import numpy as np
from cloudpathlib import AnyPath
from omegaconf import OmegaConf

from chemreporter.cli.helpers.actions import (
    normalize_actions,
    run_actions,
    warn_unsupported_actions,
)
from chemreporter.cli.helpers.path_utils import resolve_local_path
from chemreporter.cli.helpers.sampling_config import resolve_query_sampling
from chemreporter.cli.helpers.yaml_utils import write_yaml
from chemreporter.cli.io_utils import ChemReporterIO
from chemreporter.config_schemas import QueryDBConfig
from chemreporter.query_database_tools.actions_tools import make_time_stamp
from chemreporter.query_database_tools.query_database import QueryDatabaseHandler
from chemreporter.query_database_tools.query_tools import load_allowlist_frame

logger = logging.getLogger("chemreporter")


def save_filtered_keys(filtered_keys, output_path):
    """Save filtered keys to a numpy file."""
    logger.info("Saving %s filtered keys", len(filtered_keys))
    np.save(output_path, filtered_keys)


def run_main(config, io_handler: ChemReporterIO):
    """Main function to run the filter script.

    Args:
        config: OmegaConf config object.
        io_handler: I/O handler for cloud/local file operations.

    Raises:
        ValueError: If no query is provided.
    """
    io_handler.init_cloud_client()

    # 0. Unpack and validate config
    validated_config = QueryDBConfig(**OmegaConf.to_container(config, resolve=True))

    db_path = AnyPath(validated_config.query_database_path)
    results_path = AnyPath(validated_config.results_path)

    timestamp = make_time_stamp()

    if not results_path.is_dir():
        file_name = results_path.name
        results_path = results_path.parent
    else:
        file_name = f"{timestamp}-filtered_keys.npy"

    query = validated_config.query
    sampling = validated_config.sampling
    sampling_params = resolve_query_sampling(sampling)

    actions_dict = normalize_actions(validated_config.actions)
    logger.debug("actions_dict: %s", actions_dict)

    # 1. Init query database
    db_query = QueryDatabaseHandler(db_path, read_func=io_handler.read_parquet)

    # 2. Perform sql query # -> return index of the filtered database
    logger.info("Performing sql query: %s", query)
    # -> return keys of the filtered database str
    t1 = time.perf_counter()
    query_kwargs: dict = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        local_folder = Path(temp_dir)

        if validated_config.restrict_to is not None:
            local_values = resolve_local_path(
                io_handler, validated_config.restrict_to.path_to_values, local_folder
            )
            allowlist = load_allowlist_frame(
                local_values,
                validated_config.restrict_to.columns,
            )
            query_kwargs["restrict_to"] = {
                "columns": validated_config.restrict_to.columns,
                "values": allowlist,
            }

        keys = db_query.query_to_keys(query, **sampling_params, **query_kwargs)
        t2 = time.perf_counter()
        logger.debug("Time taken to query database: %s seconds", t2 - t1)

        if len(keys) == 0:
            raise ValueError("No keys found. Fix your query and try again.")
        # Extract entry_key column as numpy array (flattened)

        save_filtered_keys(keys, local_folder / file_name)

        # Save config to local folder for archival, named after the keys file
        # so the two are easy to match up.
        write_yaml(
            local_folder,
            Path(file_name).stem + ".yaml",
            OmegaConf.to_yaml(config, resolve=True),
        )

        warn_unsupported_actions(actions_dict)
        run_actions(actions_dict, db_query, keys, local_folder)
        io_handler.upload_folder(local_folder, results_path)
