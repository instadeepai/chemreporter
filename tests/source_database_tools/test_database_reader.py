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
"""Tests for SourceDatabaseReader class."""

import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import ase
import numpy as np
import pytest
from ase.build import molecule
from ase.db import connect
from cloudpathlib import AnyPath, AzureBlobPath, GSPath, S3Path

from chemreporter.cli.io_utils import ChemReporterIO
from chemreporter.source_database_tools.database_item import DatasetItem
from chemreporter.source_database_tools.database_reader import (
    SourceDatabaseReader,
    is_local_path,
    parse_key,
)
from chemreporter.source_database_tools.exceptions import SourceDatabaseReaderUsageError
from chemreporter.source_database_tools.source_db_implementations import (
    AselmdbDatabaseImplementationOc20,
    AselmdbDatabaseImplementationOdac23,
    AselmdbDatabaseImplementationOmat,
    AselmdbDatabaseImplementationOmc25,
)

# Path to the small test database fixture (created by create_test_db.py)
# This db has 10 entries
TEST_DB_FIXTURE_FILE = Path(__file__).parent / "data" / "small-test.aselmdb"
TEST_DB_FIXTURE_DIR = Path(__file__).parent / "data"
TEST_DB_FIXTURE_GLOB = Path(__file__).parent / "data" / "*.aselmdb"
XYZ_TEST_DB_FIXTURE_FILE = Path(__file__).parent / "data" / "xyz-test.xyz"


@pytest.fixture
def small_test_db_multiple(tmp_path):
    """Create a small test database with multiple files."""
    db_dir = tmp_path / "test_db_multi"
    db_dir.mkdir()

    with connect(str(TEST_DB_FIXTURE_FILE)) as source_db:
        assert len(source_db) == 10  # this is the source database
        # Create first database file with entries 0-1
        db_file1 = db_dir / "test1.aselmdb"
        with connect(str(db_file1)) as target_db:
            for i in range(2):
                atoms = source_db.get_atoms(i + 1)
                target_db.write(atoms)

        # Create second database file with entries 2-3
        db_file2 = db_dir / "test2.aselmdb"
        with connect(str(db_file2)) as target_db:
            for i in range(2, 4):
                atoms = source_db.get_atoms(i + 1)
                target_db.write(atoms)

        # Create third database file with entries 4-10
        db_file3 = db_dir / "test3.aselmdb"
        with connect(str(db_file3)) as target_db:
            for i in range(4, 10):
                atoms = source_db.get_atoms(i + 1)
                target_db.write(atoms)

    return Path(db_dir)


@pytest.fixture
def empty_db_dir(tmp_path):
    """Create an empty directory for testing."""
    db_dir = tmp_path / "empty_db"
    db_dir.mkdir()
    return db_dir


@pytest.mark.parametrize(
    "path",
    [
        Path("/tmp/local/db.aselmdb"),
        Path("relative/local/db.aselmdb"),
        AnyPath("/tmp/local/db.aselmdb"),
    ],
    ids=["posix-absolute", "posix-relative", "anypath-local"],
)
def test_is_local_path_true_for_local_paths(path):
    """Local pathlib / AnyPath values are treated as local."""
    assert is_local_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        S3Path("s3://bucket/prefix/db.aselmdb"),
        S3Path("s3://bucket/"),
        GSPath("gs://bucket/prefix/db.aselmdb"),
        GSPath("gs://bucket/"),
        AzureBlobPath("az://account/container/prefix/db.aselmdb"),
        AzureBlobPath("az://account/container/"),
        AnyPath("s3://bucket/prefix/db.aselmdb"),
        AnyPath("gs://bucket/prefix/db.aselmdb"),
        AnyPath("az://account/container/prefix/db.aselmdb"),
    ],
    ids=[
        "s3-file",
        "s3-prefix",
        "gs-file",
        "gs-prefix",
        "azure-file",
        "azure-prefix",
        "anypath-s3",
        "anypath-gs",
        "anypath-azure",
    ],
)
def test_is_local_path_false_for_cloud_paths(path):
    """Cloudpathlib S3 / GCS / Azure paths are treated as non-local."""
    assert is_local_path(path) is False


def test_local_path_name_does_not_touch_cloud_path(monkeypatch, small_test_db_multiple):
    """Regression: must not implicitly download via os.fspath() on CloudPath."""
    reader = SourceDatabaseReader(
        database_name="small-test",
        split_name="train",
        db_path=small_test_db_multiple,
        chunk_size=2,
    )
    reader.temp_dir = Path("/some/tmp/dir")
    reader._files_index[0] = S3Path("s3://bucket/prefix/data0000.aselmdb")

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("must not touch cloud storage")

    monkeypatch.setattr(S3Path, "__fspath__", _fail_if_called)

    local_path = reader._local_path_name(0)

    assert local_path == reader.temp_dir / "data0000.aselmdb"


def test_local_path_name_falls_back_to_wrapped_path_when_temp_dir_unset(
    monkeypatch, small_test_db_multiple
):
    """When temp_dir is still None, the cloud path itself gets wrapped in Path()."""
    reader = SourceDatabaseReader(
        database_name="small-test",
        split_name="train",
        db_path=small_test_db_multiple,
        chunk_size=2,
    )
    assert reader.temp_dir is None

    cloud_path = S3Path("s3://bucket/prefix/data0000.aselmdb")
    monkeypatch.setattr(S3Path, "__fspath__", lambda self: str(self._no_prefix))
    reader._files_index[0] = cloud_path

    local_path = reader._local_path_name(0)

    assert local_path == Path(cloud_path)


def test_parse_key_invalid_length_raises_value_error():
    """parse_key rejects keys that don't split into exactly 4 parts."""
    with pytest.raises(ValueError, match="length 4"):
        parse_key("not_four_parts")


class TestDatasetItem:
    """Test DatasetItem dataclass."""

    def test_dataset_item_creation(self):
        """Test creating a DatasetItem."""
        atoms = molecule("H2O")
        item = DatasetItem(
            database_name="dbname",
            split_name="train",
            name_mapping={"subset": "data_id", "net_charge": "charge"},
            key="test_key",
            atoms=atoms,
            additional_fields=[],
        )

        assert item.database_name == "dbname"
        assert item.split_name == "train"
        assert item.key == "test_key"
        assert isinstance(item.atoms, ase.Atoms)
        assert len(item.atoms) == 3  # H2O has 3 atoms


class TestSourceDatabaseReaderInitialization:
    """Test SourceDatabaseReader initialization."""

    def test_init_with_directory(self):
        """Test initialization with a directory path."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=TEST_DB_FIXTURE_DIR,
            chunk_size=2,
        )

        assert reader.chunk_size == 2
        assert reader.num_files == 1
        assert reader.files_dir == TEST_DB_FIXTURE_DIR
        assert reader._file_current_index == -1

    def test_init_with_multiple_files(self, small_test_db_multiple):
        """Test initialization with directory containing multiple files."""
        directory = Path(small_test_db_multiple)
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=directory,
            chunk_size=1,
        )
        assert reader.num_files == 3
        assert reader.files_dir == directory

    def test_init_with_glob_pattern(self):
        """Test initialization with glob pattern."""
        glob_path = TEST_DB_FIXTURE_GLOB
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=glob_path,
            chunk_size=2,
        )
        assert reader.chunk_size == 2
        assert reader.files_dir == TEST_DB_FIXTURE_DIR
        assert reader.num_files == 1

    def test_init_with_specific_file(self):
        """Test initialization with specific file."""
        db_file = TEST_DB_FIXTURE_FILE
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=db_file,
            chunk_size=2,
        )
        assert reader.files_dir == TEST_DB_FIXTURE_DIR
        assert reader.num_files == 1

    def test_init_empty_directory_raises_error(self, empty_db_dir):
        """Test initialization with empty directory raises error."""
        with pytest.raises(FileNotFoundError, match="No database files found"):
            SourceDatabaseReader(
                database_name="small-test",
                split_name="train",
                db_path=empty_db_dir,
                chunk_size=2,
            )

    def test_init_with_download_function(
        self,
    ):
        """Test initialization with download function."""
        download_fn = Mock()
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=TEST_DB_FIXTURE_FILE,
            chunk_size=2,
            download_function=download_fn,
        )
        assert reader.download_function == download_fn

    @pytest.mark.parametrize(
        "database_format, expected_impl_cls",
        [
            ("aselmdb_omat", AselmdbDatabaseImplementationOmat),
            ("aselmdb_oc", AselmdbDatabaseImplementationOc20),
            ("aselmdb_odac", AselmdbDatabaseImplementationOdac23),
            ("aselmdb_omc", AselmdbDatabaseImplementationOmc25),
        ],
    )
    def test_init_selects_expected_implementation_per_format(
        self, small_test_db_multiple, database_format, expected_impl_cls
    ):
        """Each aselmdb database_format wires up its matching implementation class."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=2,
            database_format=database_format,
        )
        assert isinstance(reader.source_db_impl, expected_impl_cls)
        assert reader.file_extension == "aselmdb"

    def test_init_unsupported_format_raises_usage_error(self, small_test_db_multiple):
        """An unsupported database_format hits the match statement's default case."""
        with pytest.raises(SourceDatabaseReaderUsageError, match="Unsupported"):
            SourceDatabaseReader(
                database_name="small-test",
                split_name="train",
                db_path=small_test_db_multiple,
                chunk_size=2,
                database_format="bogus_format",
            )


class TestSourceDatabaseReaderIndexing:
    """Test file indexing functionality."""

    def test_index_all_files_directory(self, small_test_db_multiple):
        """Test indexing all files in a directory."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=2,
            database_format="aselmdb_omol",
        )

        assert len(reader._files_index) == 3
        assert all(isinstance(p, Path) for p in reader._files_index.values())
        assert all(p.suffix == ".aselmdb" for p in reader._files_index.values())

    def test_index_all_files_glob(self, small_test_db_multiple):
        """Test indexing with glob pattern."""
        glob_path = small_test_db_multiple / "*.aselmdb"
        reader = SourceDatabaseReader(
            database_name="small",
            split_name="train",
            db_path=glob_path,
            chunk_size=2,
            database_format="aselmdb_omol",
        )
        assert reader.num_files == 3


class TestSourceDatabaseReaderFileReading:
    """Test file reading functionality."""

    def test_read_file_iter_single_file(self):
        """Test reading a single file with read_file_iter."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=TEST_DB_FIXTURE_FILE,
            chunk_size=2,
        )

        dataset = reader.read_file_iter()
        assert dataset is not None
        assert len(dataset) == 10  # 10 molecules in the test database
        assert reader._file_current_index == 0

    def test_read_file_iter_multiple_files(self, small_test_db_multiple):
        """Test reading multiple files sequentially with read_file_iter."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=2,
        )

        # Read first file
        dataset1 = reader.read_file_iter()
        assert reader._file_current_index == 0

        # Read second file
        dataset2 = reader.read_file_iter()
        assert reader._file_current_index == 1

        # Read third file
        dataset3 = reader.read_file_iter()
        assert reader._file_current_index == 2

        # files are not read in order because of glob pattern
        assert len(dataset1) + len(dataset2) + len(dataset3) == 10
        assert sorted([len(dataset1), len(dataset2), len(dataset3)]) == [2, 2, 6]

    def test_read_file_direct(self):
        """Test reading a file directly."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=TEST_DB_FIXTURE_FILE,
            chunk_size=2,
        )
        db_file = TEST_DB_FIXTURE_FILE
        dataset = reader.source_db_impl.read_file(db_file)

        assert dataset is not None
        assert len(dataset) == 10

    def test_read_file_iter_returns_none_once_exhausted(self):
        """After all files are read, read_file_iter returns None instead of raising."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=TEST_DB_FIXTURE_FILE,
            chunk_size=2,
        )
        assert reader.read_file_iter() is not None
        assert reader.read_file_iter() is None

    def test_read_file_iter_raises_without_download_function_for_remote_path(
        self, monkeypatch, small_test_db_multiple
    ):
        """A remote file with no download_function raises a usage error."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=2,
        )
        monkeypatch.setattr(S3Path, "__fspath__", lambda self: str(self._no_prefix))
        reader._files_index[0] = S3Path("s3://bucket/prefix/data0000.aselmdb")

        with pytest.raises(SourceDatabaseReaderUsageError, match="download function"):
            reader.read_file_iter()

    def test_read_file_iter_does_not_call_download_function_for_local_path(
        self, small_test_db_multiple
    ):
        """download_function is never called when the source database is local."""
        download_fn = Mock()
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=2,
            download_function=download_fn,
        )

        reader.read_file_iter()

        download_fn.assert_not_called()

    def test_read_file_iter_with_chemreporter_io_does_not_copy_local_db(
        self, small_test_db_multiple
    ):
        """ChemReporterIO.download_file never copies the database for a local path.

        Mirrors the wiring in process.py where io_handler.download_file is passed
        as the download_function. When the source database already lives on disk,
        shutil.copy must not be called at any point during iteration.
        """
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=2,
            download_function=ChemReporterIO().download_file,
        )

        with patch("chemreporter.cli.io_utils.shutil.copy") as mock_copy:
            reader.read_file_iter()

        mock_copy.assert_not_called()

    def test_read_file_iter_downloads_remote_file_then_reads_it(
        self, tmp_path, small_test_db_multiple
    ):
        """A remote file with a download_function is downloaded, then read."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=2,
        )
        real_file = reader._files_index[0]
        reader.temp_dir = tmp_path
        reader._files_index[0] = S3Path("s3://bucket/prefix/remote.aselmdb")
        download_fn = Mock(
            side_effect=lambda remote, local: shutil.copy(str(real_file), str(local))
        )
        reader.download_function = download_fn

        dataset = reader.read_file_iter()

        download_fn.assert_called_once()
        assert dataset is not None
        assert len(dataset) > 0


class TestSourceDatabaseReaderChunking:
    """Test chunking functionality."""

    def test_file_chunk_single_chunk(self, small_test_db_multiple):
        """Test reading a file in a single chunk."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=10,
        )

        chunks = list(reader.file_chunk(chunk_size=10))

        assert len(chunks) == 3  # 3 files in total
        assert len(chunks[0]) in [2, 6]  # All molecules in one chunk

    def test_file_chunk_multiple_chunks(self, small_test_db_multiple):
        """Test reading a file in multiple chunks."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=1,
        )

        chunks = list(reader.file_chunk(chunk_size=1))

        assert len(chunks) == 10  # 10 molecules, 1 per chunk
        assert all(len(chunk) == 1 for chunk in chunks)

    def test_file_chunk_multiple_chunks_xyz(self):
        """Test reading a file in multiple chunks."""
        reader = SourceDatabaseReader(
            db_path=XYZ_TEST_DB_FIXTURE_FILE,
            chunk_size=2,
            database_format="xyz",
            database_name="small",
            split_name="train",
        )

        chunks = list(reader.file_chunk(chunk_size=2))

        assert len(chunks) == 13  # 13 molecules in total
        assert all(len(chunk) == 2 for chunk in chunks)

    def test_file_chunk_partial_last_chunk(self, small_test_db_multiple):
        """Test reading a file where last chunk is partial."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=2,
        )

        chunks = list(reader.file_chunk(chunk_size=3))
        # one file has 6 molecules, the 2 others have 2 molecules
        # makes 4 files in total
        assert len(chunks) == 4
        # all molecules are in the chunks : 10 molecules in total
        assert np.sum([len(chunk) for chunk in chunks]) == 10
        # chunks are not in order because of glob pattern
        assert sorted([len(chunk) for chunk in chunks]) == [2, 2, 3, 3]

    def test_file_chunk_creates_dataset_items(self, small_test_db_multiple):
        """Test that file_chunk creates DatasetItem objects."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=2,
        )

        chunks = list(reader.file_chunk(chunk_size=2))

        # Check first chunk
        assert all(isinstance(item, DatasetItem) for item in chunks[0])
        assert all(hasattr(item, "key") for item in chunks[0])
        assert all(hasattr(item, "atoms") for item in chunks[0])

    def test_file_chunk_key_format(self, small_test_db_multiple):
        """Test that file_chunk creates correct key format."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=1,
        )
        chunk = next(reader.file_chunk(chunk_size=5))
        file_name = reader._files_index[reader._file_current_index].stem
        keys = [f"small-test_train_{file_name}_{i}" for i in range(5)]
        assert len(chunk) in [2, 5]
        assert sorted([chunk[i].key for i in range(len(chunk))]) == sorted(
            keys[: len(chunk)]
        )
        chunk = next(reader.file_chunk(chunk_size=1))
        file_name = reader._files_index[reader._file_current_index].stem
        keys = [f"small-test_train_{file_name}_{i}" for i in range(1)]

        assert len(chunk) == 1
        assert [chunk[i].key for i in range(len(chunk))] == keys

        chunk = next(reader.file_chunk(chunk_size=2))
        file_name = reader._files_index[reader._file_current_index].stem
        keys = [f"small-test_train_{file_name}_{i}" for i in range(2)]
        assert len(chunk) == 2
        assert sorted([chunk[i].key for i in range(2)]) == sorted(keys)

    def test_file_chunk_creates_dataset_items_xyz(self):
        """Test that file_chunk creates DatasetItem objects."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=XYZ_TEST_DB_FIXTURE_FILE,
            chunk_size=2,
            database_format="xyz",
        )

        chunk = next(reader.file_chunk(chunk_size=5))
        file_name = reader._files_index[reader._file_current_index].stem
        keys = [f"small-test_train_{file_name}_{i}" for i in range(5)]

        assert len(chunk) == 5
        assert sorted([chunk[i].key for i in range(len(chunk))]) == sorted(keys)

        # Verify all items are DatasetItem objects with correct attributes
        assert all(isinstance(item, DatasetItem) for item in chunk)
        assert all(hasattr(item, "key") for item in chunk)
        assert all(hasattr(item, "atoms") for item in chunk)
        assert all(isinstance(item.atoms, ase.Atoms) for item in chunk)

    def test_file_chunk_does_not_unlink_local_file(self, small_test_db_multiple):
        """Local files survive file_chunk even when download_function is set."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=10,
            download_function=Mock(),
        )
        files_before = list(reader._files_index.values())
        assert all(f.exists() for f in files_before)

        list(reader.file_chunk(chunk_size=10))

        assert all(f.exists() for f in files_before)

    @patch("chemreporter.source_database_tools.database_reader.is_local_path")
    def test_file_chunk_unlinks_downloaded_file(
        self, mock_is_local_path, small_test_db_multiple
    ):
        """Files ARE cleaned up after each chunk when the path is remote."""
        mock_is_local_path.return_value = False
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=10,
            download_function=Mock(),
        )
        files_before = list(reader._files_index.values())
        assert all(f.exists() for f in files_before)

        list(reader.file_chunk(chunk_size=10))

        assert all(not f.exists() for f in files_before)


class TestSourceDatabaseReaderIteration:
    """Test iteration functionality."""

    def test_iter_creates_valid_dataset_items_aselmdb(self):
        """Test that iteration creates valid DatasetItem objects."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=TEST_DB_FIXTURE_FILE,
            chunk_size=2,
            database_format="aselmdb_omol",
        )

        for chunk in reader:
            for item in chunk:
                assert isinstance(item, DatasetItem)
                assert isinstance(item.key, str)
                assert isinstance(item.atoms, ase.Atoms)
                assert len(item.atoms) > 0

    def test_iter_creates_valid_dataset_items_xyz(self):
        """Test that iteration creates valid DatasetItem objects."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=XYZ_TEST_DB_FIXTURE_FILE,
            chunk_size=2,
            database_format="xyz",
        )

        for chunk in reader:
            for item in chunk:
                assert isinstance(item, DatasetItem)
                assert isinstance(item.key, str)
                assert isinstance(item.atoms, ase.Atoms)
                assert len(item.atoms) > 0


class TestSourceDatabaseReaderFetchAtoms:
    """Test fetch_atoms_from_key_index functionality."""

    def test_fetch_atoms_different_indices(self, small_test_db_multiple):
        """Test fetching different atoms by index."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=2,
            database_format="aselmdb_omol",
        )

        index_0 = "small-test_train_test1_0"
        index_1 = "small-test_train_test1_1"
        atoms0, keys0 = reader.fetch_atoms(index_0, key_entries=[index_0])
        atoms1, keys1 = reader.fetch_atoms(index_1, key_entries=[index_1])

        assert isinstance(atoms0, list)
        assert isinstance(atoms1, list)
        assert len(atoms0) == 1
        assert len(atoms1) == 1
        assert isinstance(atoms0[0], ase.Atoms)
        assert isinstance(atoms1[0], ase.Atoms)

    def test_fetch_atoms_different_indices_xyz(self):
        """Test fetching different atoms by index."""
        reader = SourceDatabaseReader(
            database_name="xyz-test",
            db_path=XYZ_TEST_DB_FIXTURE_FILE,
            chunk_size=2,
            split_name="train",
            database_format="xyz",
        )
        index_0 = "xyz-test_train_xyz-test_0"
        index_1 = "xyz-test_train_xyz-test_1"
        atoms0, key_entries0 = reader.fetch_atoms(index_0, key_entries=[index_0])
        atoms1, key_entries1 = reader.fetch_atoms(index_1, key_entries=[index_1])

        assert isinstance(atoms0, list)
        assert isinstance(atoms1, list)
        assert len(atoms0) == len(key_entries0)
        assert len(atoms1) == len(key_entries1)

    def test_fetch_atoms_raises_on_database_or_split_mismatch(
        self, small_test_db_multiple
    ):
        """A key from a different database/split raises a usage error."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=2,
            database_format="aselmdb_omol",
        )
        mismatched_key = "other-db_train_test1_0"

        with pytest.raises(SourceDatabaseReaderUsageError, match="mismatch"):
            reader.fetch_atoms(mismatched_key, key_entries=[mismatched_key])

    def test_fetch_atoms_uses_download_function_when_set(self, small_test_db_multiple):
        """When a download_function is provided, fetch_atoms routes through it."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=2,
            database_format="aselmdb_omol",
        )
        download_fn = Mock(side_effect=shutil.copy)
        reader.download_function = download_fn

        index_0 = "small-test_train_test1_0"
        atoms0, keys0 = reader.fetch_atoms(index_0, key_entries=[index_0])

        download_fn.assert_called_once()
        assert len(atoms0) == 1
        assert isinstance(atoms0[0], ase.Atoms)


class TestSourceDatabaseReaderIterAtoms:
    """Test iter_atoms functionality."""

    def test_iter_atoms_matches_fetch_atoms(self, small_test_db_multiple):
        """iter_atoms yields the same (atoms, key) pairs fetch_atoms returns."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=2,
            database_format="aselmdb_omol",
        )
        index_0 = "small-test_train_test1_0"

        results = list(reader.iter_atoms(index_0, key_entries={index_0}))

        assert len(results) == 1
        atoms, key = results[0]
        assert isinstance(atoms, ase.Atoms)
        assert key == index_0

    def test_iter_atoms_raises_on_database_or_split_mismatch(
        self, small_test_db_multiple
    ):
        """A key from a different database/split raises a usage error."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=2,
            database_format="aselmdb_omol",
        )
        mismatched_key = "other-db_train_test1_0"

        with pytest.raises(SourceDatabaseReaderUsageError, match="mismatch"):
            list(reader.iter_atoms(mismatched_key, key_entries={mismatched_key}))

    def test_iter_atoms_uses_download_function_when_set(self, small_test_db_multiple):
        """When a download_function is provided, iter_atoms routes through it."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=2,
            database_format="aselmdb_omol",
        )
        download_fn = Mock(side_effect=shutil.copy)
        reader.download_function = download_fn

        index_0 = "small-test_train_test1_0"
        results = list(reader.iter_atoms(index_0, key_entries={index_0}))

        download_fn.assert_called_once()
        assert len(results) == 1
        assert isinstance(results[0][0], ase.Atoms)


class TestSourceDatabaseReaderFetchAtomsFromKeyIndex:
    """Test fetch_atoms_from_key_index functionality."""

    def test_fetch_atoms_from_key_index_returns_single_atoms(
        self, small_test_db_multiple
    ):
        """A valid key returns a single ase.Atoms object."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=2,
            database_format="aselmdb_omol",
        )

        atoms = reader.fetch_atoms_from_key_index("small-test_train_test1_0")

        assert isinstance(atoms, ase.Atoms)
        assert len(atoms) > 0

    def test_fetch_atoms_from_key_index_raises_on_mismatch(
        self, small_test_db_multiple
    ):
        """A key from a different database/split raises a usage error."""
        reader = SourceDatabaseReader(
            database_name="small-test",
            split_name="train",
            db_path=small_test_db_multiple,
            chunk_size=2,
            database_format="aselmdb_omol",
        )

        with pytest.raises(SourceDatabaseReaderUsageError, match="mismatch"):
            reader.fetch_atoms_from_key_index("other-db_train_test1_0")


def test_get_source_database_info_returns_expected_keys(small_test_db_multiple):
    """get_source_database_info reports the reader's configuration."""
    reader = SourceDatabaseReader(
        database_name="small-test",
        split_name="train",
        db_path=small_test_db_multiple,
        chunk_size=2,
        database_format="aselmdb_omol",
    )

    info = reader.get_source_database_info()

    assert info == {
        "database_name": "small-test",
        "split_name": "train",
        "file_extension": "aselmdb_omol",
        "files_path": str(small_test_db_multiple),
    }
