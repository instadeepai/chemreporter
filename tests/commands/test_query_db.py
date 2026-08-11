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

import numpy as np
import pytest
from omegaconf import OmegaConf

from chemreporter.cli.query import run_main, save_filtered_keys


@pytest.fixture
def valid_query_db_config():
    """Return a valid mock configuration for query_db."""
    return OmegaConf.create({
        "query_database_path": "s3://dummy/query",
        "results_path": "s3://dummy/results",
        "query": "SELECT * FROM db",
        "sampling": {"n_samples": 10},
        "actions": ["extract_smiles"],
    })


@patch("chemreporter.cli.query.ChemReporterIO")
@patch("chemreporter.cli.query.AnyPath")
@patch("chemreporter.cli.query.QueryDatabaseHandler")
@patch("chemreporter.cli.query.save_filtered_keys")
@patch("chemreporter.cli.helpers.actions.extract_smiles")
def test_query_db_run_main(
    mock_extract_smiles,
    mock_save_keys,
    mock_query_handler_cls,
    mock_anypath,
    mock_io_handler,
    valid_query_db_config,
):
    """Test that query_db.run_main executes the expected flow."""
    # Mock AnyPath to avoid real cloud calls
    mock_path_instance = MagicMock()
    mock_path_instance.is_dir.return_value = True
    mock_anypath.return_value = mock_path_instance
    # Mock the handler
    mock_handler_instance = MagicMock()
    mock_handler_instance.query_to_keys.return_value = ["key1", "key2"]
    mock_query_handler_cls.return_value = mock_handler_instance

    # Run the main function
    run_main(valid_query_db_config, mock_io_handler)

    # Assertions
    mock_io_handler.init_cloud_client.assert_called_once()
    mock_query_handler_cls.assert_called_once()

    # Verify query was executed
    mock_handler_instance.query_to_keys.assert_called_once_with(
        "SELECT * FROM db",
        n_samples=10,
        sampling_method="random",
        seed=None,
        sampling_required_columns=None,
        sampling_kwargs=None,
    )

    # Verify keys were saved locally
    mock_save_keys.assert_called_once()

    # Verify the requested action was executed
    mock_extract_smiles.assert_called_once()

    # Verify results were uploaded
    mock_io_handler.upload_folder.assert_called_once()


@patch("chemreporter.cli.query.AnyPath")
@patch("chemreporter.cli.query.ChemReporterIO")
@patch("chemreporter.cli.query.QueryDatabaseHandler")
def test_query_db_no_keys_found(
    mock_query_handler_cls,
    mock_io_handler,
    mock_anypath,
    valid_query_db_config,
):
    """Test that a ValueError is raised if no keys are found."""
    # Mock AnyPath to avoid real cloud calls
    mock_path_instance = MagicMock()
    mock_path_instance.is_dir.return_value = True
    mock_anypath.return_value = mock_path_instance
    mock_handler_instance = MagicMock()
    # Return empty list of keys
    mock_handler_instance.query_to_keys.return_value = []
    mock_query_handler_cls.return_value = mock_handler_instance

    with pytest.raises(ValueError, match="No keys found"):
        run_main(valid_query_db_config, mock_io_handler)


def test_save_filtered_keys(tmp_path):
    """save_filtered_keys writes the keys to a numpy file."""
    output_path = tmp_path / "keys.npy"

    save_filtered_keys(["key1", "key2"], output_path)

    assert list(np.load(output_path)) == ["key1", "key2"]


@patch("chemreporter.cli.query.ChemReporterIO")
@patch("chemreporter.cli.query.AnyPath")
@patch("chemreporter.cli.query.QueryDatabaseHandler")
@patch("chemreporter.cli.query.save_filtered_keys")
@patch("chemreporter.cli.helpers.actions.make_histograms")
@patch("chemreporter.cli.helpers.actions.make_statistics")
def test_query_db_run_main_with_actions(
    mock_make_statistics,
    mock_make_histograms,
    mock_save_keys,
    mock_query_handler_cls,
    mock_anypath,
    mock_io_handler,
):
    """Dict-style and unsupported actions are dispatched and warned about."""
    config = OmegaConf.create({
        "query_database_path": "s3://dummy/query",
        "results_path": "s3://dummy/results.npy",
        "query": "SELECT * FROM db",
        "sampling": {"n_samples": 10},
        "actions": [
            {"make_statistics": ["col1"]},
            {"make_histograms": ["col1"]},
            "unsupported_action",
        ],
    })

    # results_path resolves to a file, not a directory, for this test.
    mock_path_instance = MagicMock()
    mock_path_instance.is_dir.return_value = False
    mock_path_instance.name = "results.npy"
    mock_anypath.return_value = mock_path_instance

    mock_handler_instance = MagicMock()
    mock_handler_instance.query_to_keys.return_value = ["key1", "key2"]
    mock_query_handler_cls.return_value = mock_handler_instance

    run_main(config, mock_io_handler)

    mock_make_statistics.assert_called_once()
    mock_make_histograms.assert_called_once()
    mock_save_keys.assert_called_once()


@patch("chemreporter.cli.query.ChemReporterIO")
@patch("chemreporter.cli.query.AnyPath")
@patch("chemreporter.cli.query.QueryDatabaseHandler")
@patch("chemreporter.cli.query.save_filtered_keys")
@patch("chemreporter.cli.query.load_allowlist_frame")
def test_query_db_restrict_to_local_path(
    mock_load_allowlist_frame,
    mock_save_keys,
    mock_query_handler_cls,
    mock_anypath,
    mock_io_handler,
    valid_query_db_config,
):
    """A local restrict_to allowlist is loaded without a download."""
    config = OmegaConf.merge(
        valid_query_db_config,
        {"restrict_to": {"columns": ["smiles"], "path_to_values": "allowlist.npz"}},
    )

    mock_path_instance = MagicMock()
    mock_path_instance.is_dir.return_value = True
    mock_path_instance.name = "allowlist.npz"
    # "://" absent means path_utils.is_local_path treats the mock as local.
    mock_path_instance.__str__.return_value = "allowlist.npz"
    mock_anypath.return_value = mock_path_instance

    mock_handler_instance = MagicMock()
    mock_handler_instance.query_to_keys.return_value = ["key1"]
    mock_query_handler_cls.return_value = mock_handler_instance

    run_main(config, mock_io_handler)

    mock_io_handler.download_file.assert_not_called()
    mock_load_allowlist_frame.assert_called_once()
    _, kwargs = mock_handler_instance.query_to_keys.call_args
    assert kwargs["restrict_to"]["columns"] == ["smiles"]


@patch("chemreporter.cli.query.ChemReporterIO")
@patch("chemreporter.cli.query.AnyPath")
@patch("chemreporter.cli.query.QueryDatabaseHandler")
@patch("chemreporter.cli.query.save_filtered_keys")
@patch("chemreporter.cli.query.load_allowlist_frame")
def test_query_db_restrict_to_remote_path_is_downloaded(
    mock_load_allowlist_frame,
    mock_save_keys,
    mock_query_handler_cls,
    mock_anypath,
    mock_io_handler,
    valid_query_db_config,
):
    """A remote restrict_to allowlist is downloaded before being loaded."""
    config = OmegaConf.merge(
        valid_query_db_config,
        {
            "restrict_to": {
                "columns": ["smiles"],
                "path_to_values": "s3://dummy/allowlist.npz",
            }
        },
    )

    mock_path_instance = MagicMock()
    mock_path_instance.is_dir.return_value = True
    mock_path_instance.name = "allowlist.npz"

    mock_path_instance.__str__.return_value = "s3://dummy/allowlist.npz"
    mock_anypath.return_value = mock_path_instance

    mock_handler_instance = MagicMock()
    mock_handler_instance.query_to_keys.return_value = ["key1"]
    mock_query_handler_cls.return_value = mock_handler_instance

    run_main(config, mock_io_handler)

    mock_io_handler.download_file.assert_called_once()
    mock_load_allowlist_frame.assert_called_once()
