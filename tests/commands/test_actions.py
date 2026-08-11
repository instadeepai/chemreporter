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

from chemreporter.cli.helpers.actions import run_actions


@patch("chemreporter.cli.helpers.actions.make_statistics")
def test_run_actions_skips_make_statistics_when_bare(mock_make_statistics):
    """A bare 'make_statistics' action (no columns) is not run."""
    run_actions({"make_statistics": None}, MagicMock(), ["key1"], "out")

    mock_make_statistics.assert_not_called()


@patch("chemreporter.cli.helpers.actions.make_histograms")
def test_run_actions_skips_make_histograms_when_bare(mock_make_histograms):
    """A bare 'make_histograms' action (no columns) is not run."""
    run_actions({"make_histograms": None}, MagicMock(), ["key1"], "out")

    mock_make_histograms.assert_not_called()


@patch("chemreporter.cli.helpers.actions.make_statistics")
def test_run_actions_runs_make_statistics_with_columns(mock_make_statistics):
    """make_statistics runs with the requested column_names when given."""
    db_query = MagicMock()
    run_actions({"make_statistics": ["col1"]}, db_query, ["key1"], "out")

    mock_make_statistics.assert_called_once_with(
        db_query, keys=["key1"], output_path="out", column_names=["col1"]
    )
