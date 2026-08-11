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

import sys
from unittest.mock import MagicMock

import pytest

from chemreporter.cli import io_utils
from chemreporter.main import (
    EXTERNAL_IO_VARIABLE_NAME,
    generate_config_template,
    load_io_plugin,
    main,
)


def test_load_io_plugin_function_dict(tmp_path):
    """Test CHEMREPORTER_IO dict of function overrides."""
    plugin_file = tmp_path / "dict_plugin.py"
    plugin_file.write_text(f"""
def my_download(path, local_path):
    return "downloaded_by_plugin"

{EXTERNAL_IO_VARIABLE_NAME} = {{
    "download_file": my_download
}}
""")

    handler = load_io_plugin(str(plugin_file))
    assert handler.download_file("dummy_path", "dummy_local") == "downloaded_by_plugin"


def test_load_io_plugin_implicit_functions(tmp_path):
    """Test implicit matching on top-level function names."""
    plugin_file = tmp_path / "implicit_plugin.py"
    plugin_file.write_text("""
def download_file(path, local_path):
    return "downloaded_implicitly"
""")

    handler = load_io_plugin(str(plugin_file))
    assert handler.download_file("dummy", "dummy") == "downloaded_implicitly"


def test_load_io_plugin_file_not_found():
    """Test that a non-existent plugin file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="I/O plugin file not found"):
        load_io_plugin("non_existent_plugin.py")


def test_load_io_plugin_missing_plugin_definition(tmp_path):
    """Test that a plugin with no handler or overrides raises ValueError."""
    plugin_file = tmp_path / "missing_plugin.py"
    plugin_file.write_text("def unrelated(): pass")

    with pytest.raises(ValueError, match="Plugin file must define CHEMREPORTER_IO"):
        load_io_plugin(str(plugin_file))


def test_load_io_plugin_invalid_handler_type(tmp_path):
    """Test that an invalid CHEMREPORTER_IO export raises ValueError."""
    plugin_file = tmp_path / "invalid_handler.py"
    plugin_file.write_text(f"""
{EXTERNAL_IO_VARIABLE_NAME} = ["download_file"]
""")

    with pytest.raises(ValueError, match="must be a dict of function overrides"):
        load_io_plugin(str(plugin_file))


def test_load_io_plugin_ignores_unknown_overrides(tmp_path, caplog):
    """Test that unknown override keys are ignored and log a warning."""
    plugin_file = tmp_path / "unknown_override.py"
    plugin_file.write_text(f"""
def unknown_func():
    pass

{EXTERNAL_IO_VARIABLE_NAME} = {{
    "unknown_func": unknown_func
}}
""")

    load_io_plugin(str(plugin_file))
    assert "Ignoring unknown I/O override: unknown_func" in caplog.text


def test_generate_config_template_valid(capsys):
    """Test that generating a config template prints the expected YAML."""
    generate_config_template("process")
    captured = capsys.readouterr()

    assert "# Template configuration for process" in captured.out
    assert "source_database_path: '### REQUIRED ###'" in captured.out
    assert "database_name: '### REQUIRED ###'" in captured.out
    assert "split_name: other" in captured.out
    assert "processing_chunk_size: 50000" in captured.out
    assert "  enable: false" in captured.out
    assert "  nb_atoms_limit: 200" in captured.out
    assert "  subsets_skip_list: []" in captured.out


def test_generate_config_template_invalid():
    """Test that generating a template for an unknown command raises ValueError."""
    with pytest.raises(ValueError, match="Unknown command: invalid-cmd"):
        generate_config_template("invalid-cmd")


def test_main_dispatch_generate_config(monkeypatch, capsys):
    """Test that main() correctly dispatches the generate-config command and exits."""
    monkeypatch.setattr(sys, "argv", ["chemreporter", "generate-config", "export"])

    with pytest.raises(SystemExit) as e:
        main()

    assert e.value.code == 0
    captured = capsys.readouterr()
    assert "# Template configuration for export" in captured.out


def test_main_dispatch_command(monkeypatch, tmp_path):
    """Test that main() passes config and io_handler to process.run_main."""
    config_path = tmp_path / "dummy.yaml"
    config_path.write_text("source_database_path: /tmp/db\n")
    monkeypatch.setattr(
        sys, "argv", ["chemreporter", "process", "-c", str(config_path)]
    )

    mock_load = MagicMock(return_value="mocked_config_object")
    monkeypatch.setattr("chemreporter.main.OmegaConf.load", mock_load)

    mock_run_main = MagicMock()
    monkeypatch.setattr("chemreporter.main.process.run_main", mock_run_main)

    main()

    mock_load.assert_called_once_with(str(config_path))
    config_arg, io_handler_arg = mock_run_main.call_args[0]
    assert config_arg == "mocked_config_object"
    assert isinstance(io_handler_arg, io_utils.ChemReporterIO)


def test_main_errors_when_config_file_missing(monkeypatch, tmp_path, capsys):
    """Test that main() exits with a CLI error when the config does not exist."""
    missing_config = tmp_path / "does_not_exist.yaml"
    monkeypatch.setattr(
        sys, "argv", ["chemreporter", "process", "-c", str(missing_config)]
    )

    with pytest.raises(SystemExit) as e:
        main()

    assert e.value.code == 2
    captured = capsys.readouterr()
    assert f"Configuration file not found: {missing_config}" in captured.err
