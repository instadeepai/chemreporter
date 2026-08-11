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

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from chemreporter.cli.process import run_main


@pytest.fixture
def valid_process_db_config():
    """Return a valid mock configuration for process_db."""
    return OmegaConf.create({
        "source_database_path": "s3://dummy/source",
        "database_name": "dummy_db",
        "split_name": "train",
        "database_format": "aselmdb",
        "query_database_path": "s3://dummy/query",
        "source_database_metadata": {
            "basis_set": "dummy_basis",
            "functional": "dummy_functional",
            "correction_term": "dummy_correction",
        },
        "processing_chunk_size": 100,
        "graph_based_processing": {"enable": False},
    })


@patch("chemreporter.cli.process.SourceDatabaseReader")
@patch("chemreporter.cli.process.QueryDatabaseHandler")
@patch("chemreporter.cli.process.DatabaseProcessor")
@patch("chemreporter.cli.process.ChemReporterIO")
def test_process_db_run_main(
    mock_io_handler,
    mock_db_processor_cls,
    mock_query_handler_cls,
    mock_src_reader_cls,
    valid_process_db_config,
):
    """Test that process_db.run_main executes the expected flow."""
    # Mock the reader to yield one chunk of data
    mock_reader_instance = MagicMock()
    mock_reader_instance.__iter__.return_value = [["data_chunk_1"]]
    mock_reader_instance.get_source_database_info.return_value = {"info": "dummy"}
    mock_src_reader_cls.return_value = mock_reader_instance

    # Mock the processor
    mock_processor_instance = MagicMock()
    mock_processor_instance.process.return_value = "processed_df"
    mock_db_processor_cls.return_value = mock_processor_instance

    # Mock the query handler
    mock_handler_instance = MagicMock()
    mock_query_handler_cls.return_value = mock_handler_instance

    # Run the main function
    run_main(valid_process_db_config, mock_io_handler)

    # Assertions
    mock_io_handler.init_cloud_client.assert_called_once()
    mock_src_reader_cls.assert_called_once()
    mock_query_handler_cls.assert_called_once()
    mock_db_processor_cls.assert_called_once()
    passed_metadata = mock_db_processor_cls.call_args.kwargs["database_info"]
    assert passed_metadata.model_dump() == {
        "basis_set": "dummy_basis",
        "functional": "dummy_functional",
        "correction_term": "dummy_correction",
    }

    # Verify the chunk was processed and stored
    mock_processor_instance.process.assert_called_once_with(["data_chunk_1"])
    mock_handler_instance.store.assert_called_once_with("processed_df")

    # Verify config was uploaded
    mock_io_handler.upload_file.assert_called_once()
