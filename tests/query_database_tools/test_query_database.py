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
"""Tests for QueryDatabaseHandler class."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from chemreporter.query_database_tools.query_database import (
    QueryDatabaseHandler,
    QueryDatabaseInputError,
    default_lazy_read_function,
    default_save_function,
    return_db_paths,
    simple_read_function,
)
from chemreporter.query_database_tools.query_tools import load_allowlist_frame
from chemreporter.query_database_tools.table_schemas import FULL_SCHEMA, get_schema_dict

TEST_DB_PATH = Path(__file__).parent / "data"


@pytest.fixture
def sample_structures_data():
    """Create sample structures data for testing."""
    return {
        "entry_key": ["mol_1", "mol_2", "mol_3", "mol_4", "mol_5"],
        "entry_index": [1, 2, 3, 4, 5],
        "subset": ["spice", "tr1x", "anix", "spice", "biomolecules"],
        "net_charge": [0, 1, -1, 0, 0],
        "spin": [0.0, 0.5, 0.0, 0.0, 0.5],
        "num_atoms": [10, 15, 8, 20, 12],
        "atomic_numbers": [
            [6, 1, 1, 1, 1, 8, 1, 1, 1, 1],
            [7, 1, 1, 1, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [8, 1, 1, 6, 6, 6, 6, 1],
            [6] * 20,
            [6, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1],
        ],
        "dipole_moment_magnitude": [1.2, 2.3, 0.5, 1.8, 0.9],
        "net_force_norm": [0.0001, 0.0005, 0.002, 0.0003, 0.0008],
        "max_force_norm": [0.5, 1.2, 2.0, 0.8, 1.5],
        "element_symbol": [
            "CHO",
            "NHO",
            "CHO",
            "CH",
            "CHO",
        ],
        "graph_properties_candidate": [True, True, False, True, True],
        "error_graph_properties": [False, False, True, False, False],
        "molecular_weight": [44.0, 60.0, 32.0, 240.0, 58.0],
        "logp": [0.5, -0.3, -1.2, 3.5, 0.8],
        "tpsa": [20.0, 35.0, 15.0, 0.0, 25.0],
        "smiles": ["CCO", "NCCO", "C=O", "C" * 20, "CCCO"],
    }


@pytest.fixture
def sample_structures_df(sample_structures_data):
    """Create a sample structures DataFrame."""
    # Create basic columns without fingerprints
    df = pl.DataFrame(sample_structures_data)

    for i in range(1024):
        df = df.with_columns(pl.lit(0, dtype=pl.Int64).alias(f"fingerprint_{i}"))

    return df


@pytest.fixture
def test_db_dir(tmp_path, sample_structures_df):
    """Create a test database directory with parquet files."""
    db_dir = tmp_path / "test_query_db"
    db_dir.mkdir()

    # Split data into two files
    df1 = sample_structures_df.slice(0, 3)
    df2 = sample_structures_df.slice(3, 2)

    # Save as parquet files
    df1.write_parquet(db_dir / "query_database_000001.parquet")
    df2.write_parquet(db_dir / "query_database_000002.parquet")

    return db_dir


@pytest.fixture
def empty_db_dir(tmp_path):
    """Create an empty database directory."""
    db_dir = tmp_path / "empty_query_db"
    db_dir.mkdir()
    return db_dir


class TestDefaultFunctions:
    """Test default helper functions."""

    def test_default_save_function(self, tmp_path, sample_structures_df):
        """Test default save function."""
        save_path = tmp_path / "test_save.parquet"
        default_save_function(sample_structures_df, str(save_path))

        assert save_path.exists()

        # Read back and verify
        df_read = pl.read_parquet(save_path)
        assert len(df_read) == len(sample_structures_df)

    def test_default_lazy_read_function(self, test_db_dir):
        """Test default lazy read function."""
        lazy_df = default_lazy_read_function(test_db_dir)

        assert isinstance(lazy_df, pl.LazyFrame)

        # Collect and verify
        df = lazy_df.collect()
        assert len(df) == 5  # Total of 5 rows across 2 files

    def test_default_lazy_read_function_with_schema(self, test_db_dir):
        """Test default lazy read function with schema."""
        schema = get_schema_dict(FULL_SCHEMA)
        lazy_df = default_lazy_read_function(test_db_dir, schema=schema)

        assert isinstance(lazy_df, pl.LazyFrame)


class TestQueryDatabaseHandlerInitialization:
    """Test QueryDatabaseHandler initialization."""

    def test_init_basic(self, test_db_dir):
        """Test basic initialization."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        assert handler.db_path == test_db_dir
        assert handler.table_name == "query_database"

    def test_init_with_custom_functions(self, test_db_dir):
        """Test initialization with custom save/read functions."""

        def custom_save(df, path):
            pass

        def custom_read(path):
            return pl.LazyFrame()

        handler = QueryDatabaseHandler(
            db_path=test_db_dir, save_func=custom_save, read_func=custom_read
        )

        assert handler.save_function == custom_save
        assert handler.read_function == custom_read


class TestQueryDatabaseHandlerStore:
    """Test storing data."""

    def test_store_dataframe(self, empty_db_dir, sample_structures_df):
        """Test storing a DataFrame."""
        handler = QueryDatabaseHandler(db_path=empty_db_dir)

        handler.store(sample_structures_df)

        # Check that file was created
        files = list(empty_db_dir.glob("*.parquet"))
        assert len(files) == 1

        # Verify content
        df_read = pl.read_parquet(files[0])
        assert len(df_read) == len(sample_structures_df)

    def test_store_multiple_dataframes(self, empty_db_dir, sample_structures_df):
        """Test storing multiple DataFrames."""
        handler = QueryDatabaseHandler(db_path=empty_db_dir)

        handler.store(sample_structures_df.slice(0, 2))
        handler.store(sample_structures_df.slice(2, 3))

        # Check that two files were created
        files = list(empty_db_dir.glob("*.parquet"))
        assert len(files) == 2

    def test_store_empty_dataframe(self, empty_db_dir):
        """Test storing an empty DataFrame."""
        handler = QueryDatabaseHandler(db_path=empty_db_dir)

        empty_df = pl.DataFrame()
        handler.store(empty_df)

        # No file should be created
        files = list(empty_db_dir.glob("*.parquet"))
        assert len(files) == 0


class TestQueryDatabaseHandlerQueries:
    """Test SQL query functionality."""

    def test_query_to_keys_where(self, test_db_dir):
        """Test querying with WHERE clause."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        query = " num_atoms > 10"
        result = handler.query_to_keys(query)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

        # Verify the results
        df = handler.get_dataframe(indices=result, columns=["num_atoms"]).collect()
        assert all(df["num_atoms"] > 10)

    def test_query_to_keys_complex(self, test_db_dir):
        """Test applying complex WHERE conditions."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        query = "net_force_norm < 0.001 AND max_force_norm < 1.0"
        result = handler.query_to_keys(query)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_query_to_keys_subset_filter(self, test_db_dir):
        """Test filtering by subset."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        query = "subset = 'spice'"
        result = handler.query_to_keys(query)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2  # 2 spice samples

    def test_query_to_keys_empty_query(self, test_db_dir):
        """Test empty query raises error."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        with pytest.raises(QueryDatabaseInputError):
            handler.query_to_keys("")

    def test_query_with_custom_sampling_method(self, test_db_dir):
        """Custom sampling method callable returns the requested number of keys."""

        def take_first_n(frame, n_samples, **kwargs):
            collected = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
            return collected["entry_key"].to_list()[:n_samples]

        handler = QueryDatabaseHandler(db_path=test_db_dir)

        result = handler.query_to_keys(
            "num_atoms > 5",
            n_samples=3,
            sampling_method=take_first_n,
            sampling_required_columns=["fingerprint_bits"],
        )

        assert len(result) == 3
        assert len(set(result.tolist())) == 3

    def test_query_to_keys_restrict_to_npz_single(self, test_db_dir, tmp_path):
        """Semi-join on a single-column .npz allowlist file."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)
        npz_path = tmp_path / "train.npz"
        np.savez(npz_path, smiles=np.array(["CCO", "C=O"]))
        allowlist = load_allowlist_frame(npz_path, "smiles")
        result = handler.query_to_keys(
            "num_atoms > 0",
            restrict_to={"columns": ["smiles"], "values": allowlist},
        )
        assert set(result.tolist()) == {"mol_1", "mol_3"}

    def test_query_to_keys_restrict_to_npz(self, test_db_dir, tmp_path):
        """Semi-join on a multi-column .npz allowlist file."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)
        npz_path = tmp_path / "train.npz"
        np.savez(
            npz_path,
            smiles=np.array(["CCO", "NCCO"]),
            net_charge=np.array([0, 1]),
        )
        allowlist = load_allowlist_frame(npz_path, ["smiles", "net_charge"])
        result = handler.query_to_keys(
            "num_atoms > 0",
            restrict_to={
                "columns": ["smiles", "net_charge"],
                "values": allowlist,
            },
        )
        assert set(result.tolist()) == {"mol_1", "mol_2"}

    def test_query_to_keys_restrict_to_dataframe_left(self, test_db_dir, tmp_path):
        """Semi-join works when the reader returns a DataFrame, not LazyFrame."""

        def read_as_dataframe(db_path, schema=None):
            return default_lazy_read_function(db_path, schema=schema).collect()

        handler = QueryDatabaseHandler(db_path=test_db_dir, read_func=read_as_dataframe)
        npz_path = tmp_path / "train.npz"
        np.savez(npz_path, smiles=np.array(["CCO", "C=O"]))
        allowlist = load_allowlist_frame(npz_path, "smiles")
        # Intentionally mismatched integer width on a multi-column path
        allowlist_multi = pl.DataFrame({
            "smiles": ["CCO"],
            "net_charge": pl.Series([0], dtype=pl.Int32),
        })
        result = handler.query_to_keys(
            "num_atoms > 0",
            restrict_to={
                "columns": ["smiles", "net_charge"],
                "values": allowlist_multi,
            },
        )
        assert set(result.tolist()) == {"mol_1"}
        result_single = handler.query_to_keys(
            "num_atoms > 0",
            restrict_to={"columns": ["smiles"], "values": allowlist},
        )
        assert set(result_single.tolist()) == {"mol_1", "mol_3"}

    def test_query_to_keys_restrict_to_rejects_bad_values(self, test_db_dir):
        """Reject malformed restrict_to payloads."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)
        with pytest.raises(ValueError, match="columns"):
            handler.query_to_keys(
                "num_atoms > 0",
                restrict_to={"values": pl.DataFrame({"smiles": ["CCO"]})},
            )
        with pytest.raises(ValueError, match="DataFrame"):
            handler.query_to_keys(
                "num_atoms > 0",
                restrict_to={"columns": ["smiles"], "values": ["CCO"]},
            )


class TestQueryDatabaseHandlerGetDataframe:
    """Test getting dataframe with specific columns."""

    def test_get_dataframe_with_columns(self, test_db_dir):
        """Test getting specific columns for given indices."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        # Get keys first
        query = "num_atoms > 10"
        keys = handler.query_to_keys(query)

        # Get specific columns for those keys
        df = handler.get_dataframe(indices=keys, columns=["num_atoms"])

        assert isinstance(df, pl.LazyFrame)
        df = df.collect()
        assert "entry_key" in df.columns  # Always included
        assert "num_atoms" in df.columns
        assert len(df) == len(keys)

    def test_get_dataframe_multiple_columns(self, test_db_dir):
        """Test getting multiple columns."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        query = "num_atoms > 10"
        keys = handler.query_to_keys(query)

        # Get multiple columns
        df = handler.get_dataframe(
            indices=keys, columns=["num_atoms", "molecular_weight", "net_force_norm"]
        ).collect()

        assert isinstance(df, pl.DataFrame)
        assert "entry_key" in df.columns
        assert "num_atoms" in df.columns
        assert "molecular_weight" in df.columns
        assert "net_force_norm" in df.columns


class TestQueryDatabaseHandlerFilters:
    """Test filter functionality."""

    def test_numeric_range_filter(self, test_db_dir):
        """Test filtering by numeric range."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        query = "num_atoms >= 10 AND num_atoms <= 15"
        keys = handler.query_to_keys(query)

        df = handler.get_dataframe(indices=keys, columns=["num_atoms"]).collect()
        assert all((df["num_atoms"] >= 10) & (df["num_atoms"] <= 15))

    def test_string_filter(self, test_db_dir):
        """Test filtering by string column."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        query = "subset = 'anix'"
        keys = handler.query_to_keys(query)

        df = handler.get_dataframe(indices=keys, columns=["subset"]).collect()
        assert len(df) == 1
        assert df["subset"][0] == "anix"

    def test_multiple_conditions(self, test_db_dir):
        """Test multiple filter conditions."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        query = "subset = 'spice' AND net_force_norm < 0.001"
        keys = handler.query_to_keys(query)

        df = handler.get_dataframe(
            indices=keys, columns=["subset", "net_force_norm"]
        ).collect()
        assert all(df["subset"] == "spice")
        assert all(df["net_force_norm"] < 0.001)

    def test_atomic_symbols_filter(self, test_db_dir):
        """Test multiple filter conditions."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        query = "element_symbol = 'CHO'"
        keys = handler.query_to_keys(query)

        df = handler.get_dataframe(indices=keys, columns=["element_symbol"]).collect()
        assert all(df["element_symbol"] == "CHO")

    def test_atomic_symbols_logical_filter(self, test_db_dir):
        """Test atomic symbols logical filter."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        query = "element_symbol ~ '^(C|H|O|Na|Cl)+$'"
        keys = handler.query_to_keys(query)

        df = handler.get_dataframe(indices=keys, columns=["element_symbol"]).collect()
        assert all((df["element_symbol"] == "CHO") | (df["element_symbol"] == "CH"))


class TestQueryDatabaseHandlerEdgeCases:
    """Test edge cases and error handling."""

    def test_query_empty_result(self, test_db_dir):
        """Test query that returns no results."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        query = "num_atoms > 1000"
        keys = handler.query_to_keys(query)

        assert isinstance(keys, np.ndarray)
        assert len(keys) == 0

    def test_query_with_missing_column(self, test_db_dir):
        """Test query with column that doesn't exist."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        # This should raise an error when trying to execute
        query = "nonexistent_column > 10"

        # Should raise during query execution
        with pytest.raises((Exception, pl.exceptions.ColumnNotFoundError)):
            handler.query_to_keys(query)

    def test_read_empty_directory(self, empty_db_dir):
        """Test reading from empty directory raises FileNotFoundError."""
        handler = QueryDatabaseHandler(db_path=empty_db_dir)

        with pytest.raises(FileNotFoundError, match="No db files found"):
            handler.query_to_keys("num_atoms > 0")


class TestQueryDatabaseHandlerSchema:
    """Test schema-related functionality."""

    def test_read_with_schema_subset(self, test_db_dir):
        """Test that subset column is properly read."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        lazy_df = handler.read_function(handler.db_path)
        df = lazy_df.collect()

        assert "subset" in df.columns
        assert df["subset"].dtype == pl.String

    def test_schema_columns_present(self, test_db_dir):
        """Test that expected schema columns are present."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        lazy_df = handler.read_function(handler.db_path)
        df = lazy_df.collect()

        expected_columns = [
            "entry_key",
            "subset",
            "net_charge",
            "num_atoms",
            "net_force_norm",
            "max_force_norm",
        ]

        for col in expected_columns:
            assert col in df.columns

    def test_schema_numeric_types(self, test_db_dir):
        """Test that numeric columns have correct types."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        lazy_df = handler.read_function(handler.db_path)
        df = lazy_df.collect()

        # Check some numeric columns
        assert df["num_atoms"].dtype in [pl.Int64, pl.Int32]
        assert df["net_force_norm"].dtype in [pl.Float64, pl.Float32]
        assert df["max_force_norm"].dtype in [pl.Float64, pl.Float32]


class TestRealParquetDatabase:
    """Integration tests using real parquet database."""

    def test_read_real_database(self):
        """Test reading from real parquet database."""
        handler = QueryDatabaseHandler(db_path=TEST_DB_PATH)

        # Try to read the database
        lazy_df = handler.read_function(handler.db_path)
        assert isinstance(lazy_df, pl.LazyFrame)

        # Collect and check basic properties
        df = lazy_df.collect()
        assert len(df) > 0
        assert "entry_key" in df.columns

    def test_query_real_database_num_atoms(self):
        """Test querying real database by num_atoms."""
        handler = QueryDatabaseHandler(db_path=TEST_DB_PATH)

        # Query for molecules with more than 10 atoms
        query = "num_atoms > 10"
        result = handler.query_to_keys(query)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_query_real_database_subset(self):
        """Test querying real database by subset."""
        handler = QueryDatabaseHandler(db_path=TEST_DB_PATH)

        # First, check what subsets are available
        lazy_df = handler.read_function(handler.db_path)
        df = lazy_df.select("subset").collect()
        unique_subsets = df["subset"].unique().to_list()

        first_subset = unique_subsets[0]
        query = f"subset = '{first_subset}'"
        result = handler.query_to_keys(query)

        assert len(result) > 0

    def test_get_dataframe_real_database(self):
        """Test getting specific columns from real database."""
        handler = QueryDatabaseHandler(db_path=TEST_DB_PATH)

        # Get all entry keys (filter out None values)
        lazy_df = handler.read_function(handler.db_path)
        all_keys_series = lazy_df.select("entry_key").collect()["entry_key"]
        all_keys = [k for k in all_keys_series.to_list() if k is not None]

        if len(all_keys) < 10:
            pytest.skip("Not enough valid entry keys in database")

        # Get first 10 entries with specific columns
        sample_keys = all_keys[:10]
        columns = ["num_atoms", "molecular_weight"]

        df = handler.get_dataframe(indices=sample_keys, columns=columns)

        assert isinstance(df, pl.LazyFrame)
        df = df.collect()
        assert len(df) == len(sample_keys)
        assert "entry_key" in df.columns
        assert "num_atoms" in df.columns
        assert "molecular_weight" in df.columns

    def test_query_real_database_complex(self):
        """Test complex query on real database."""
        handler = QueryDatabaseHandler(db_path=TEST_DB_PATH)

        # Complex query with multiple conditions
        query = "num_atoms > 5 AND num_atoms < 50"
        result = handler.query_to_keys(query)

        assert len(result) > 0

        # Verify the results match the query (result is already a numpy array)
        df = handler.get_dataframe(indices=result, columns=["num_atoms"]).collect()
        assert all((df["num_atoms"] > 5) & (df["num_atoms"] < 50))

    def test_query_real_database_with_sampling(self):
        """Test querying real database with random sampling."""
        handler = QueryDatabaseHandler(db_path=TEST_DB_PATH)

        # Query with sampling
        query = "num_atoms > 10"
        n_samples = 50

        result = handler.query_to_keys(
            query, n_samples=n_samples, sampling_method="random"
        )

        # Should return at most n_samples entries
        assert len(result) <= n_samples
        assert len(result) > 0


class TestReturnDbPaths:
    """Test return_db_paths function edge cases."""

    def test_return_db_paths_with_file(self, tmp_path, sample_structures_df):
        """Test return_db_paths with a file path."""
        file_path = tmp_path / "test.parquet"
        sample_structures_df.write_parquet(file_path)

        result = return_db_paths(file_path)
        assert result is not None
        assert len(result) == 1
        assert str(file_path) in result

    def test_return_db_paths_invalid_path(self, tmp_path):
        """Test return_db_paths with invalid path."""
        invalid_path = tmp_path / "nonexistent"
        result = return_db_paths(invalid_path)
        assert result is None


class TestSimpleReadFunction:
    """Test simple_read_function."""

    def test_simple_read_function_without_schema(self, test_db_dir):
        """Test simple_read_function without schema."""
        df = simple_read_function(test_db_dir)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 5

    def test_simple_read_function_no_files(self, tmp_path):
        """Test simple_read_function with no parquet files."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No db files found"):
            simple_read_function(empty_dir)


class TestStoreWithDuplicateKeys:
    """Test store method with duplicate keys."""

    def test_store_duplicate_keys_raises_error(self, test_db_dir, sample_structures_df):
        """Test that storing duplicate keys raises QueryDatabaseInputError.

        Note: test_db_dir already contains mol_1 through mol_5,
        so attempting to store sample_structures_df will raise an error.
        """
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        # Try to store keys that already exist in test_db_dir (should raise error)
        with pytest.raises(QueryDatabaseInputError):
            handler.store(sample_structures_df, check_keys=True)

    def test_store_duplicate_keys_no_check(self, empty_db_dir, sample_structures_df):
        """Test storing duplicate keys with check_keys=False."""
        handler = QueryDatabaseHandler(db_path=empty_db_dir)

        # Store first time
        handler.store(sample_structures_df)

        # Store again with check_keys=False (should succeed)
        handler.store(sample_structures_df, check_keys=False)

        files = list(empty_db_dir.glob("*.parquet"))
        assert len(files) == 2


class TestGetDataframeEdgeCases:
    """Test get_dataframe edge cases."""

    def test_get_dataframe_with_2d_indices(self, test_db_dir):
        """Test get_dataframe with 2D array indices."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        # Create 2D indices (as might come from some operations)
        keys_2d = [["mol_1"], ["mol_2"], ["mol_3"]]

        df = handler.get_dataframe(indices=keys_2d, columns=["num_atoms"])

        assert isinstance(df, pl.LazyFrame)
        assert len(df.collect()) == 3

    def test_get_dataframe_with_numpy_array(self, test_db_dir):
        """Test get_dataframe with numpy array indices."""
        handler = QueryDatabaseHandler(db_path=test_db_dir)

        # Create numpy array indices
        keys_np = np.array(["mol_1", "mol_2"])

        df = handler.get_dataframe(indices=keys_np, columns=["num_atoms"])

        assert isinstance(df, pl.LazyFrame)
        assert len(df.collect()) == 2


class TestExistingKeysProperty:
    """Test _existing_keys cached property."""

    def test_existing_keys_empty_database(self, empty_db_dir):
        """Test _existing_keys with empty database."""
        handler = QueryDatabaseHandler(db_path=empty_db_dir)

        keys = handler._existing_keys

        assert isinstance(keys, set)
        assert len(keys) == 0
