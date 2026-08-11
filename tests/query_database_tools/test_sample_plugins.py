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
"""Tests for sampling plugin loading from a Python file path."""

from pathlib import Path

import pytest

from chemreporter.cli.helpers.sampling_config import resolve_query_sampling
from chemreporter.query_database_tools.sample_plugins import load_sampler_from_path

REPO_ROOT = Path(__file__).resolve().parents[2]
GREEDY_SAMPLER_PATH = REPO_ROOT / "sampling_examples/greedy_sampling.py"


def test_load_sampler_loads_custom_sampling_function():
    """The plugin file's custom_sampling_function callable is loaded."""
    sampler = load_sampler_from_path(GREEDY_SAMPLER_PATH)
    assert sampler.__name__ == "run_greedy_sampler"


def test_load_sampler_missing_file():
    """Missing plugin path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_sampler_from_path("/tmp/does-not-exist-sampler.py")


def test_load_sampler_missing_entrypoint(tmp_path):
    """Plugin file without a custom_sampling_function raises ValueError."""
    plugin = tmp_path / "empty_sampler.py"
    plugin.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="custom_sampling_function"):
        load_sampler_from_path(plugin)


def test_resolve_query_sampling_from_path():
    """A non-builtin method is loaded as a custom sampler plugin."""
    params = resolve_query_sampling({
        "n_samples": None,
        "method": str(GREEDY_SAMPLER_PATH),
        "required_columns": "fingerprint_bits",
        "kwargs": {"seed": 42, "stop_when_exhausted": True},
    })
    assert params["n_samples"] is None
    assert params["sampling_method"].__name__ == "run_greedy_sampler"
    assert params["sampling_required_columns"] == ["fingerprint_bits"]
    assert params["sampling_kwargs"]["stop_when_exhausted"] is True


def test_resolve_query_sampling_missing_plugin_path():
    """A non-builtin method that isn't a valid path raises."""
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_query_sampling({"method": "not/a/real/path.py"})
