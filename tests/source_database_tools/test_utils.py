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
"""Tests for source_database_tools utils module."""

from pathlib import Path

import pytest

from chemreporter.source_database_tools.database_reader import SourceDatabaseReader
from chemreporter.source_database_tools.utils import (
    retrieve_source_database_data_from_keys,
)

# Path to the small test database fixture
TEST_DB_FIXTURE_FILE = Path(__file__).parent / "data" / "small-test.aselmdb"


@pytest.fixture
def source_db_reader() -> SourceDatabaseReader:
    """Create a SourceDatabaseReader for the small-test database."""
    return SourceDatabaseReader(
        database_name="small",
        split_name="train",
        db_path=TEST_DB_FIXTURE_FILE,
        chunk_size=10,
        database_format="aselmdb_omol",
    )


class TestRetrieveSourceDatabaseDataFromKeys:
    """Tests for retrieve_source_database_data_from_keys function."""

    def test_retrieve_single_key(self, source_db_reader: SourceDatabaseReader):
        """Test retrieving data for a single key."""
        key = "small_train_small-test_0"
        result = retrieve_source_database_data_from_keys(
            source_db_readers=[source_db_reader],
            key_entries=[key],
            data_field="charge",
        )

        assert isinstance(result, dict)
        assert key in result
        assert len(result) == 1

    def test_retrieve_multiple_keys(self, source_db_reader: SourceDatabaseReader):
        """Test retrieving data for multiple keys."""
        keys = [
            "small_train_small-test_0",
            "small_train_small-test_1",
            "small_train_small-test_2",
        ]
        result = retrieve_source_database_data_from_keys(
            source_db_readers=[source_db_reader],
            key_entries=keys,
            data_field="charge",
        )

        assert isinstance(result, dict)
        assert len(result) == 3
        for key in keys:
            assert key in result

    def test_retrieve_different_data_fields(
        self, source_db_reader: SourceDatabaseReader
    ):
        """Test retrieving different data fields."""
        key = "small_train_small-test_0"

        # Test with different fields that exist in the database
        for field in ["charge", "spin", "num_atoms"]:
            result = retrieve_source_database_data_from_keys(
                source_db_readers=[source_db_reader],
                key_entries=[key],
                data_field=field,
            )
            assert key in result

    def test_key_not_found_raises_runtime_error(
        self, source_db_reader: SourceDatabaseReader
    ):
        """Test that RuntimeError is raised when key is not found."""
        # Key with wrong database name
        invalid_key = "wrong-db_train_small-test_0"

        with pytest.raises(
            RuntimeError, match="not found in any source database reader"
        ):
            retrieve_source_database_data_from_keys(
                source_db_readers=[source_db_reader],
                key_entries=[invalid_key],
                data_field="charge",
            )

    def test_key_with_wrong_split_raises_runtime_error(
        self, source_db_reader: SourceDatabaseReader
    ):
        """Test that RuntimeError is raised when split name doesn't match."""
        # Key with wrong split name
        invalid_key = "small_wrong-split_small-test_0"

        with pytest.raises(
            RuntimeError, match="not found in any source database reader"
        ):
            retrieve_source_database_data_from_keys(
                source_db_readers=[source_db_reader],
                key_entries=[invalid_key],
                data_field="charge",
            )

    def test_empty_key_list_returns_empty_dict(
        self, source_db_reader: SourceDatabaseReader
    ):
        """Test that empty key list returns empty dict."""
        result = retrieve_source_database_data_from_keys(
            source_db_readers=[source_db_reader],
            key_entries=[],
            data_field="charge",
        )

        assert result == {}
