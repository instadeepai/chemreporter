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

import gc
import tempfile
import time
from pathlib import Path

import yaml
from cloudpathlib import AnyPath
from omegaconf import OmegaConf

from chemreporter.cli.helpers.yaml_utils import write_yaml
from chemreporter.cli.io_utils import ChemReporterIO
from chemreporter.config_schemas import ProcessDBConfig
from chemreporter.database_processors.converter import DatabaseProcessor
from chemreporter.query_database_tools.query_database import QueryDatabaseHandler
from chemreporter.source_database_tools.database_reader import SourceDatabaseReader


def run_main(config, io_handler: ChemReporterIO):
    """Run the main function.

    Args:
        config: OmegaConf config object
        io_handler: I/O handler for cloud/local file operations

    Raises:
        ValueError: If the config is invalid
    """
    io_handler.init_cloud_client()

    timestamp = time.strftime("%Y%m%d_%H%M")
    # 0. unpack and validate config

    validated_config = ProcessDBConfig(**OmegaConf.to_container(config, resolve=True))

    query_db_path = AnyPath(validated_config.query_database_path)
    source_db_path = AnyPath(validated_config.source_database_path)
    database_name = validated_config.database_name
    split_name = validated_config.split_name
    chunk_size = validated_config.processing_chunk_size

    # 1. Initialize source database Reader -> pointer to the source database
    src_db_reader = SourceDatabaseReader(
        database_name=database_name,
        split_name=split_name,
        db_path=source_db_path,
        chunk_size=chunk_size,
        download_function=io_handler.download_file,
        database_format=validated_config.database_format,  # type: ignore[arg-type]
    )

    # 2. Initialize the writing class for query database
    write_func = QueryDatabaseHandler(
        db_path=query_db_path,
        save_func=io_handler.write_parquet,
    )

    # 3. Initialize the converter: convert the database to Polars DataFrames
    db_processor = DatabaseProcessor(
        graph_properties_config=validated_config.graph_based_processing,
        database_info=validated_config.source_database_metadata,
    )

    for data_chunk in src_db_reader:
        df = db_processor.process(data_chunk)
        write_func.store(df)
        del df
        gc.collect()

    with tempfile.TemporaryDirectory() as temp_dir:
        local_folder = Path(temp_dir)
        config_yaml_path = write_yaml(
            local_folder,
            f"{timestamp}-config_process_db.yaml",
            yaml.dump(src_db_reader.get_source_database_info()),
        )

        io_handler.upload_file(
            config_yaml_path,
            AnyPath(query_db_path) / config_yaml_path.name,
        )
