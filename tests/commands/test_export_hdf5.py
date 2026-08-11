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

import h5py
import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from chemreporter.cli import export
from chemreporter.cli.helpers import hdf5_export


@pytest.fixture
def valid_export_hdf5_config():
    """Return a valid mock configuration for export_hdf5."""
    return OmegaConf.create({
        "query_database_path": "s3://dummy/query",
        "keys_path": "local_keys.npy",
        "output_path": "s3://dummy/output.hdf5",
        "num_workers": 1,
        "extras_fields": None,
    })


@patch("chemreporter.cli.export.AnyPath")
@patch("chemreporter.cli.export.np.load")
@patch("chemreporter.cli.export.fetch_source_database_readers")
@patch("chemreporter.cli.export.Pool")
def test_export_hdf5_run_main_single_worker(
    mock_pool_cls,
    mock_fetch_readers,
    mock_np_load,
    mock_anypath,
    valid_export_hdf5_config,
):
    """Test that export_hdf5.run_main executes the expected flow for single worker."""
    mock_io_handler = MagicMock()

    # Mock AnyPath to simulate output path not existing
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = False
    mock_dest = MagicMock()
    mock_dest.exists.return_value = False
    mock_parent = MagicMock()
    mock_parent.__truediv__.return_value = mock_dest

    mock_path_instance.name = "output.hdf5"
    mock_path_instance.parent = mock_parent
    mock_path_instance.configure_mock(**{
        "__str__.return_value": "s3://dummy/output.hdf5"
    })
    mock_anypath.return_value = mock_path_instance

    # Mock numpy load
    mock_keys = MagicMock()
    mock_keys.ravel().tolist.return_value = ["key1", "key2"]
    mock_np_load.return_value = mock_keys

    # Mock fetch readers
    mock_fetch_readers.return_value = ["reader1"]

    # Mock Pool
    mock_pool_instance = MagicMock()
    mock_pool_instance.__enter__.return_value = mock_pool_instance
    mock_pool_cls.return_value = mock_pool_instance

    with (
        patch("chemreporter.cli.export.Path.unlink"),
        patch(
            "chemreporter.cli.helpers.hdf5_export.write_hdf5_worker"
        ) as mock_write_worker,
    ):
        export.run_main(valid_export_hdf5_config, mock_io_handler)

    mock_io_handler.init_cloud_client.assert_called_once()
    mock_np_load.assert_called_once()
    mock_fetch_readers.assert_called_once()
    mock_pool_cls.assert_called_once()
    mock_write_worker.assert_not_called()
    mock_io_handler.upload_file.assert_called_once()


@patch("chemreporter.cli.helpers.hdf5_export.write_hdf5_worker")
def test_multiprocess_worker_reuses_initialized_state(mock_write_worker):
    """Worker tasks reuse state initialized once for their process."""
    source_database_readers = [MagicMock()]
    extras_fields = {"key1": {"field": 1}}
    kwargs = {
        "keys_list": ["key1"],
        "output_file": "group.hdf5",
        "extras_fields": extras_fields,
    }

    with patch.object(hdf5_export._HDF5WorkerState, "source_database_readers", None):
        hdf5_export._initialize_hdf5_worker(
            source_database_readers,
        )
        hdf5_export._write_hdf5_multiprocess_worker(kwargs)

    mock_write_worker.assert_called_once_with(
        kwargs,
        source_database_readers=source_database_readers,
        extras_fields=extras_fields,
    )


def test_multiprocess_worker_raises_when_not_initialized():
    """RuntimeError is raised when the worker process was not initialized."""
    kwargs = {"keys_list": ["key1"], "output_file": "group.hdf5", "extras_fields": None}

    with patch.object(hdf5_export._HDF5WorkerState, "source_database_readers", None):
        with pytest.raises(RuntimeError, match="not initialized"):
            hdf5_export._write_hdf5_multiprocess_worker(kwargs)


@patch("chemreporter.cli.helpers.hdf5_export.write_hdf5_worker")
def test_multiprocess_worker_logs_and_reraises_on_failure(mock_write_worker):
    """Errors from write_hdf5_worker are logged and then re-raised."""
    mock_write_worker.side_effect = ValueError("boom")
    source_database_readers = [MagicMock()]
    kwargs = {"keys_list": ["key1"], "output_file": "group.hdf5", "extras_fields": None}

    with patch.object(
        hdf5_export._HDF5WorkerState, "source_database_readers", source_database_readers
    ):
        with pytest.raises(ValueError, match="boom"):
            hdf5_export._write_hdf5_multiprocess_worker(kwargs)


def test_group_keys_by_source_database_split_and_file():
    """Keys are grouped by (database_name, split_name, source_file_name)."""
    keys_list = [
        "omol25_train_shard1_0",
        "omol25_train_shard1_1",
        "omol25_train_shard2_0",
        "omol25_valid_shard1_0",
    ]

    grouped = export.group_keys_by_source_database_split_and_file(keys_list)

    assert set(grouped.keys()) == {
        "omol25_train_shard1",
        "omol25_train_shard2",
        "omol25_valid_shard1",
    }
    assert grouped["omol25_train_shard1"] == [
        "omol25_train_shard1_0",
        "omol25_train_shard1_1",
    ]
    assert grouped["omol25_train_shard2"] == ["omol25_train_shard2_0"]
    assert grouped["omol25_valid_shard1"] == ["omol25_valid_shard1_0"]


@patch("chemreporter.cli.export.QueryDatabaseHandler")
def test_extract_extras_fields_with_dataframe(mock_handler_cls):
    """extract_extras_fields builds an entry_key lookup from a plain DataFrame."""
    mock_handler = MagicMock()
    mock_handler.get_dataframe.return_value = pl.DataFrame({
        "entry_key": ["key1", "key2"],
        "field1": [1, 2],
    })
    mock_handler_cls.return_value = mock_handler

    result = export.extract_extras_fields(
        ["key1", "key2"], "s3://dummy/query", ["field1"], MagicMock()
    )

    assert result == {"key1": {"field1": 1}, "key2": {"field1": 2}}


@patch("chemreporter.cli.export.QueryDatabaseHandler")
def test_extract_extras_fields_with_lazyframe(mock_handler_cls):
    """extract_extras_fields collects a LazyFrame before building the lookup."""
    mock_handler = MagicMock()
    mock_handler.get_dataframe.return_value = pl.DataFrame({
        "entry_key": ["key1"],
        "field1": [42],
    }).lazy()
    mock_handler_cls.return_value = mock_handler

    result = export.extract_extras_fields(
        ["key1"], "s3://dummy/query", ["field1"], MagicMock()
    )

    assert result == {"key1": {"field1": 42}}


def test_merge_hdf5_files(tmp_path):
    """merge_hdf5_files combines top-level groups from each input file."""
    file1 = tmp_path / "part1.hdf5"
    file2 = tmp_path / "part2.hdf5"
    output = tmp_path / "merged.hdf5"

    with h5py.File(file1, "w") as f:
        grp = f.create_group("entry1")
        grp.create_dataset("positions", data=np.zeros((2, 3)))

    with h5py.File(file2, "w") as f:
        grp = f.create_group("entry2")
        grp.create_dataset("positions", data=np.ones((2, 3)))

    hdf5_export.merge_hdf5_files([file1, file2], output)

    with h5py.File(output, "r") as f:
        assert set(f.keys()) == {"entry1", "entry2"}
        np.testing.assert_array_equal(f["entry2"]["positions"][:], np.ones((2, 3)))


@patch("chemreporter.cli.helpers.hdf5_export.write_hdf5")
def test_write_hdf5_worker_calls_write_hdf5(mock_write_hdf5):
    """write_hdf5_worker delegates to write_hdf5 with the expected arguments."""
    kwargs = {"keys_list": ["key1", "key2"], "output_file": "out.hdf5"}
    source_database_readers = [MagicMock()]
    extras_fields = {"key1": {"field": 1}}

    hdf5_export.write_hdf5_worker(
        kwargs,
        source_database_readers=source_database_readers,
        extras_fields=extras_fields,
    )

    mock_write_hdf5.assert_called_once_with(
        key_entries=["key1", "key2"],
        source_db_readers=source_database_readers,
        hdf5_path="out.hdf5",
        extras_fields=extras_fields,
    )


@patch("chemreporter.cli.export.AnyPath")
def test_run_main_raises_when_single_output_exists(
    mock_anypath, valid_export_hdf5_config
):
    """FileExistsError is raised when the single output file already exists."""
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = True
    mock_anypath.return_value = mock_path_instance
    mock_io_handler = MagicMock()

    with pytest.raises(FileExistsError, match="already exists"):
        export.run_main(valid_export_hdf5_config, mock_io_handler)


@patch("chemreporter.cli.export.AnyPath")
def test_run_main_raises_when_export_parts_exist(mock_anypath):
    """FileExistsError is raised when existing export parts are found."""
    config = OmegaConf.create({
        "query_database_path": "s3://dummy/query",
        "keys_path": "local_keys.npy",
        "output_path": "s3://dummy/output.hdf5",
        "num_workers": 1,
        "extras_fields": None,
        "num_files_to_export": 2,
    })
    mock_path_instance = MagicMock()
    mock_parent = MagicMock()
    mock_parent.glob.return_value = ["existing_part.hdf5"]
    mock_path_instance.parent = mock_parent
    mock_path_instance.stem = "output"
    mock_anypath.return_value = mock_path_instance
    mock_io_handler = MagicMock()

    with pytest.raises(FileExistsError, match="already exists"):
        export.run_main(config, mock_io_handler)


@patch("chemreporter.cli.export.fetch_source_database_readers")
@patch("chemreporter.cli.export.Pool")
def test_run_main_local_output_and_keys_path(
    mock_pool_cls, mock_fetch_readers, tmp_path
):
    """Local (non-cloud) output_path and keys_path take the local-path branches."""
    keys_path = tmp_path / "keys.npy"
    np.save(keys_path, np.array(["k_a_b_0", "k_a_b_1"]))
    output_path = tmp_path / "output.hdf5"

    config = OmegaConf.create({
        "query_database_path": "s3://dummy/query",
        "keys_path": str(keys_path),
        "output_path": str(output_path),
        "num_workers": 1,
        "extras_fields": None,
    })

    mock_io_handler = MagicMock()
    mock_fetch_readers.return_value = ["reader1"]

    mock_pool_instance = MagicMock()
    mock_pool_instance.__enter__.return_value = mock_pool_instance
    mock_pool_cls.return_value = mock_pool_instance

    export.run_main(config, mock_io_handler)

    mock_io_handler.download_file.assert_not_called()
    mock_io_handler.upload_file.assert_not_called()


@patch("chemreporter.cli.export.merge_hdf5_files")
@patch("chemreporter.cli.export.AnyPath")
@patch("chemreporter.cli.export.np.load")
@patch("chemreporter.cli.export.fetch_source_database_readers")
@patch("chemreporter.cli.export.Pool")
def test_run_main_multiprocess_groups_and_merges(
    mock_pool_cls,
    mock_fetch_readers,
    mock_np_load,
    mock_anypath,
    mock_merge,
):
    """Multiprocess runs group keys by source file and merge the per-group files."""
    config = OmegaConf.create({
        "query_database_path": "s3://dummy/query",
        "keys_path": "local_keys.npy",
        "output_path": "s3://dummy/output.hdf5",
        "num_workers": 2,
        "extras_fields": None,
    })
    mock_io_handler = MagicMock()

    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = False
    mock_dest = MagicMock()
    mock_dest.exists.return_value = False
    mock_parent = MagicMock()
    mock_parent.__truediv__.return_value = mock_dest
    mock_path_instance.name = "output.hdf5"
    mock_path_instance.parent = mock_parent
    mock_path_instance.configure_mock(**{
        "__str__.return_value": "s3://dummy/output.hdf5"
    })
    mock_anypath.return_value = mock_path_instance

    mock_keys = MagicMock()
    mock_keys.ravel().tolist.return_value = [
        "omol25_train_shard1_0",
        "omol25_train_shard2_0",
    ]
    mock_np_load.return_value = mock_keys
    mock_fetch_readers.return_value = ["reader1"]

    mock_pool_instance = MagicMock()
    mock_pool_instance.__enter__.return_value = mock_pool_instance
    mock_pool_cls.return_value = mock_pool_instance

    with patch("chemreporter.cli.export.Path.unlink"):
        export.run_main(config, mock_io_handler)

    # Two distinct source files -> two groups -> one multiprocess batch of 2 tasks
    call_args = mock_pool_instance.map.call_args
    assert len(call_args[0][1]) == 2
    mock_merge.assert_called_once()
    mock_io_handler.upload_file.assert_called_once()


@patch("chemreporter.cli.export.AnyPath")
@patch("chemreporter.cli.export.np.load")
@patch("chemreporter.cli.export.fetch_source_database_readers")
def test_run_main_returns_early_when_no_keys_multiprocess(
    mock_fetch_readers, mock_np_load, mock_anypath
):
    """run_main returns early when grouping produces no work to do."""
    config = OmegaConf.create({
        "query_database_path": "s3://dummy/query",
        "keys_path": "local_keys.npy",
        "output_path": "s3://dummy/output.hdf5",
        "num_workers": 2,
        "extras_fields": None,
    })
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = False
    mock_anypath.return_value = mock_path_instance

    mock_keys = MagicMock()
    mock_keys.ravel().tolist.return_value = []
    mock_np_load.return_value = mock_keys
    mock_io_handler = MagicMock()

    export.run_main(config, mock_io_handler)

    mock_fetch_readers.assert_not_called()


@patch("chemreporter.cli.export.extract_extras_fields")
@patch("chemreporter.cli.export.AnyPath")
@patch("chemreporter.cli.export.np.load")
@patch("chemreporter.cli.export.fetch_source_database_readers")
@patch("chemreporter.cli.export.Pool")
def test_run_main_multi_file_extras_fields(
    mock_pool_cls,
    mock_fetch_readers,
    mock_np_load,
    mock_anypath,
    mock_extract_extras,
):
    """Multi-file, non-multiprocess exports merge extras fields per output part."""
    config = OmegaConf.create({
        "query_database_path": "s3://dummy/query",
        "keys_path": "local_keys.npy",
        "output_path": "s3://dummy/output.hdf5",
        "num_workers": 1,
        "extras_fields": ["field1"],
        "num_files_to_export": 2,
    })
    mock_io_handler = MagicMock()

    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = False
    mock_dest = MagicMock()
    mock_dest.exists.return_value = False
    mock_parent = MagicMock()
    mock_parent.glob.return_value = []
    mock_parent.__truediv__.return_value = mock_dest
    mock_path_instance.name = "output.hdf5"
    mock_path_instance.stem = "output"
    mock_path_instance.parent = mock_parent
    mock_path_instance.configure_mock(**{
        "__str__.return_value": "s3://dummy/output.hdf5"
    })
    mock_anypath.return_value = mock_path_instance

    mock_keys = MagicMock()
    mock_keys.ravel().tolist.return_value = [
        "omol25_train_shard1_0",
        "omol25_train_shard2_0",
    ]
    mock_np_load.return_value = mock_keys
    mock_fetch_readers.return_value = ["reader1"]
    mock_extract_extras.return_value = {
        "omol25_train_shard1_0": {"field1": 1},
        "omol25_train_shard2_0": {"field1": 2},
    }

    mock_pool_instance = MagicMock()
    mock_pool_instance.__enter__.return_value = mock_pool_instance
    mock_pool_cls.return_value = mock_pool_instance

    with patch("chemreporter.cli.export.Path.unlink"):
        export.run_main(config, mock_io_handler)

    mock_extract_extras.assert_called_once()
    assert mock_pool_instance.map.call_count == 2

    # Each part's extras must be scoped to its own keys, not the full lookup.
    first_batch = mock_pool_instance.map.call_args_list[0][0][1]
    second_batch = mock_pool_instance.map.call_args_list[1][0][1]
    assert first_batch[0]["extras_fields"] == {"omol25_train_shard1_0": {"field1": 1}}
    assert second_batch[0]["extras_fields"] == {"omol25_train_shard2_0": {"field1": 2}}


@patch("chemreporter.cli.export.fetch_source_database_readers")
@patch("chemreporter.cli.export.Pool")
def test_run_main_downloads_remote_keys_path(
    mock_pool_cls, mock_fetch_readers, tmp_path
):
    """A remote keys_path is downloaded to a local file before being loaded."""
    output_path = tmp_path / "output.hdf5"

    config = OmegaConf.create({
        "query_database_path": "s3://dummy/query",
        "keys_path": "s3://dummy/keys.npy",
        "output_path": str(output_path),
        "num_workers": 1,
        "extras_fields": None,
    })

    def _download(remote_path, local_path):
        np.save(local_path, np.array(["k_a_b_0", "k_a_b_1"]))

    mock_io_handler = MagicMock()
    mock_io_handler.download_file.side_effect = _download
    mock_fetch_readers.return_value = ["reader1"]

    mock_pool_instance = MagicMock()
    mock_pool_instance.__enter__.return_value = mock_pool_instance
    mock_pool_cls.return_value = mock_pool_instance

    export.run_main(config, mock_io_handler)

    mock_io_handler.download_file.assert_called_once()
    called_remote_path, called_local_path = mock_io_handler.download_file.call_args[0]
    assert str(called_remote_path) == "s3://dummy/keys.npy"
    assert called_local_path.name == "keys.npy"


@patch("chemreporter.cli.export.AnyPath")
@patch("chemreporter.cli.export.np.load")
@patch("chemreporter.cli.export.fetch_source_database_readers")
@patch("chemreporter.cli.export.Pool")
def test_run_main_multi_file_multiprocess_batches(
    mock_pool_cls, mock_fetch_readers, mock_np_load, mock_anypath
):
    """Multi-file export with multiprocess batches keys per output part."""
    config = OmegaConf.create({
        "query_database_path": "s3://dummy/query",
        "keys_path": "local_keys.npy",
        "output_path": "s3://dummy/output.hdf5",
        "num_workers": 2,
        "extras_fields": None,
        "num_files_to_export": 2,
    })
    mock_io_handler = MagicMock()

    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = False
    mock_dest = MagicMock()
    mock_dest.exists.return_value = False
    mock_parent = MagicMock()
    mock_parent.glob.return_value = []
    mock_parent.__truediv__.return_value = mock_dest
    mock_path_instance.name = "output.hdf5"
    mock_path_instance.stem = "output"
    mock_path_instance.parent = mock_parent
    mock_path_instance.configure_mock(**{
        "__str__.return_value": "s3://dummy/output.hdf5"
    })
    mock_anypath.return_value = mock_path_instance

    mock_keys = MagicMock()
    mock_keys.ravel().tolist.return_value = [
        "omol25_train_shard1_0",
        "omol25_train_shard2_0",
    ]
    mock_np_load.return_value = mock_keys
    mock_fetch_readers.return_value = ["reader1"]

    mock_pool_instance = MagicMock()
    mock_pool_instance.__enter__.return_value = mock_pool_instance
    mock_pool_cls.return_value = mock_pool_instance

    with (
        patch("chemreporter.cli.export.Path.unlink"),
        patch("chemreporter.cli.export.merge_hdf5_files") as mock_merge,
    ):
        export.run_main(config, mock_io_handler)

    assert mock_pool_instance.map.call_count == 2
    assert mock_merge.call_count == 2


@patch("chemreporter.cli.export.AnyPath")
@patch("chemreporter.cli.export.np.load")
@patch("chemreporter.cli.export.fetch_source_database_readers")
@patch("chemreporter.cli.export.Pool")
def test_run_main_skips_existing_export_part(
    mock_pool_cls, mock_fetch_readers, mock_np_load, mock_anypath
):
    """Existing output parts are skipped rather than regenerated."""
    config = OmegaConf.create({
        "query_database_path": "s3://dummy/query",
        "keys_path": "local_keys.npy",
        "output_path": "s3://dummy/output.hdf5",
        "num_workers": 1,
        "extras_fields": None,
        "num_files_to_export": 2,
    })
    mock_io_handler = MagicMock()

    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = False
    mock_dest = MagicMock()
    mock_dest.exists.return_value = True
    mock_parent = MagicMock()
    mock_parent.glob.return_value = []
    mock_parent.__truediv__.return_value = mock_dest
    mock_path_instance.name = "output.hdf5"
    mock_path_instance.stem = "output"
    mock_path_instance.parent = mock_parent
    mock_path_instance.configure_mock(**{
        "__str__.return_value": "s3://dummy/output.hdf5"
    })
    mock_anypath.return_value = mock_path_instance

    mock_keys = MagicMock()
    mock_keys.ravel().tolist.return_value = [
        "omol25_train_shard1_0",
        "omol25_train_shard2_0",
    ]
    mock_np_load.return_value = mock_keys
    mock_fetch_readers.return_value = ["reader1"]

    export.run_main(config, mock_io_handler)

    mock_pool_cls.assert_not_called()
