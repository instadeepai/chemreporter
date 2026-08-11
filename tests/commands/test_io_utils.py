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

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from cloudpathlib import CloudPath

from chemreporter.cli.io_utils import ChemReporterIO


def test_init_rejects_unknown_override():
    """Constructing with an unknown override name raises ValueError."""
    with pytest.raises(ValueError, match="Invalid I/O function name"):
        ChemReporterIO(functions_dict={"not_a_real_method": lambda: None})


def test_init_applies_known_override():
    """Constructing with a known override name replaces the bound method."""

    def my_download(path, local_file_path):
        return "custom"

    io_handler = ChemReporterIO(functions_dict={"download_file": my_download})

    assert io_handler.download_file is my_download


def test_init_cloud_client_is_a_noop():
    """init_cloud_client is a no-op and never raises."""
    assert ChemReporterIO().init_cloud_client() is None


def test_download_file_delegates_to_path():
    """download_file calls download_to on the given cloud path."""
    mock_path = MagicMock()

    with patch("chemreporter.cli.io_utils.is_local_path", return_value=False):
        ChemReporterIO().download_file(mock_path, Path("/tmp/foo"))

    mock_path.download_to.assert_called_once_with(Path("/tmp/foo"))


def test_upload_file_delegates_to_dest_path():
    """upload_file calls upload_from on the destination path if CloudPath."""
    mock_dest_path = MagicMock(spec=CloudPath)
    local_file_path = Path("/tmp/foo")

    with patch("chemreporter.cli.io_utils.is_local_path", return_value=False):
        ChemReporterIO().upload_file(local_file_path, mock_dest_path)

    mock_dest_path.upload_from.assert_called_once_with(local_file_path)


def test_upload_folder_delegates_to_dest_path():
    """upload_folder calls upload_from on the destination path if CloudPath."""
    mock_dest_path = MagicMock(spec=CloudPath)
    local_folder = Path("/tmp/some_folder")

    with patch("chemreporter.cli.io_utils.is_local_path", return_value=False):
        ChemReporterIO().upload_folder(local_folder, mock_dest_path)

    mock_dest_path.upload_from.assert_called_once_with(local_folder)


def test_download_file_skips_copy_for_same_local_path(tmp_path):
    """download_file does not call shutil.copy when source and dest are the same."""
    local_file = tmp_path / "data.bin"
    local_file.write_bytes(b"x")

    with (
        patch("chemreporter.cli.io_utils.is_local_path", return_value=True),
        patch("chemreporter.cli.io_utils.shutil.copy") as mock_copy,
    ):
        ChemReporterIO().download_file(local_file, local_file)

    mock_copy.assert_not_called()


def test_download_file_copies_when_local_paths_differ(tmp_path):
    """download_file calls shutil.copy when local source and dest paths differ."""
    src = tmp_path / "src.bin"
    dst = tmp_path / "dst.bin"
    src.write_bytes(b"x")

    with (
        patch("chemreporter.cli.io_utils.is_local_path", return_value=True),
        patch("chemreporter.cli.io_utils.shutil.copy") as mock_copy,
    ):
        ChemReporterIO().download_file(src, dst)

    mock_copy.assert_called_once_with(src, dst)


def test_upload_file_skips_copy_for_same_local_path(tmp_path):
    """upload_file does not call shutil.copy when source and dest are the same path."""
    local_file = tmp_path / "data.bin"
    local_file.write_bytes(b"x")

    with (
        patch("chemreporter.cli.io_utils.is_local_path", return_value=True),
        patch("chemreporter.cli.io_utils.shutil.copy") as mock_copy,
    ):
        ChemReporterIO().upload_file(local_file, local_file)

    mock_copy.assert_not_called()


def test_upload_file_copies_when_local_paths_differ(tmp_path):
    """upload_file calls shutil.copy when source and dest are different local paths."""
    src = tmp_path / "src.bin"
    dst = tmp_path / "dst.bin"
    src.write_bytes(b"x")

    with (
        patch("chemreporter.cli.io_utils.is_local_path", return_value=True),
        patch("chemreporter.cli.io_utils.shutil.copy") as mock_copy,
    ):
        ChemReporterIO().upload_file(src, dst)

    mock_copy.assert_called_once_with(src, dst)


def test_upload_folder_skips_copy_for_same_local_path(tmp_path):
    """upload_folder does not call shutil.copy when source and dest are the same."""
    with (
        patch("chemreporter.cli.io_utils.is_local_path", return_value=True),
        patch("chemreporter.cli.io_utils.shutil.copy") as mock_copy,
    ):
        ChemReporterIO().upload_folder(tmp_path, tmp_path)

    mock_copy.assert_not_called()


def test_upload_folder_copies_when_local_paths_differ(tmp_path):
    """upload_folder copies a real directory tree when local paths differ."""
    src = tmp_path / "src_folder"
    src.mkdir()
    (src / "part0.parquet").write_bytes(b"x")
    dst = tmp_path / "dst_folder"

    with patch("chemreporter.cli.io_utils.is_local_path", return_value=True):
        ChemReporterIO().upload_folder(src, dst)

    assert (dst / "part0.parquet").read_bytes() == b"x"


def test_upload_folder_copies_into_existing_destination(tmp_path):
    """upload_folder copies into a destination directory that already exists."""
    src = tmp_path / "src_folder"
    src.mkdir()
    (src / "part0.parquet").write_bytes(b"x")
    dst = tmp_path / "dst_folder"
    dst.mkdir()

    with patch("chemreporter.cli.io_utils.is_local_path", return_value=True):
        ChemReporterIO().upload_folder(src, dst)

    assert (dst / "part0.parquet").read_bytes() == b"x"


def test_write_parquet_uploads_temp_file_with_dataframe_contents():
    """write_parquet writes to a temp file then uploads it if CloudPath."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    mock_output_path = MagicMock(spec=CloudPath)
    captured = {}

    def _capture(path):
        captured["df"] = pl.read_parquet(path)

    mock_output_path.upload_from.side_effect = _capture

    with patch("chemreporter.cli.io_utils.is_local_path", return_value=False):
        ChemReporterIO().write_parquet(df, mock_output_path)

    mock_output_path.upload_from.assert_called_once()
    assert captured["df"].equals(df)


def test_write_parquet_writes_directly_to_local_path(tmp_path):
    """write_parquet writes straight to disk when the destination is local."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    output_path = tmp_path / "out.parquet"

    ChemReporterIO().write_parquet(df, output_path)

    assert pl.read_parquet(output_path).equals(df)


def test_read_parquet_returns_lazyframe_of_all_files(tmp_path):
    """read_parquet scans every parquet file in the directory."""
    pl.DataFrame({"a": [1, 2]}).write_parquet(tmp_path / "part0.parquet")
    pl.DataFrame({"a": [3, 4]}).write_parquet(tmp_path / "part1.parquet")

    result = ChemReporterIO().read_parquet(tmp_path)

    assert isinstance(result, pl.LazyFrame)
    collected = result.collect().sort("a")
    assert collected["a"].to_list() == [1, 2, 3, 4]


def test_read_parquet_with_schema(tmp_path):
    """read_parquet accepts an explicit schema for scanning."""
    pl.DataFrame({"a": [1, 2]}).write_parquet(tmp_path / "part0.parquet")

    result = ChemReporterIO().read_parquet(tmp_path, schema={"a": pl.Int64})

    assert result.collect()["a"].to_list() == [1, 2]


def test_read_parquet_raises_when_directory_missing(tmp_path):
    """read_parquet raises FileNotFoundError when the directory does not exist."""
    missing_dir = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        ChemReporterIO().read_parquet(missing_dir)


def test_read_parquet_raises_when_no_parquet_files(tmp_path):
    """read_parquet raises FileNotFoundError when the directory has no parquet files."""
    with pytest.raises(FileNotFoundError, match="No parquet files found"):
        ChemReporterIO().read_parquet(tmp_path)
