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

import argparse
import logging
import runpy
import sys
from pathlib import Path
from typing import Any, Callable, get_args, get_origin

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from chemreporter.cli import export, io_utils, process, query
from chemreporter.config_schemas import ExportHDF5Config, ProcessDBConfig, QueryDBConfig

logger = logging.getLogger("chemreporter")

EXTERNAL_IO_VARIABLE_NAME = "CHEMREPORTER_IO"
_REQUIRED_TEMPLATE_MARKER = "### REQUIRED ###"


def load_io_plugin(plugin_path: str) -> io_utils.ChemReporterIO:
    """Load a custom I/O handler from a Python plugin file.

    Plugins may export either:
    - ``CHEMREPORTER_IO``: a dict mapping I/O method names to callables
    - top-level functions named after I/O methods (implicit fallback)

    Args:
        plugin_path: Path to a Python plugin file.

    Returns:
        A ``ChemReporterIO`` instance with the plugin overrides applied.

    Raises:
        FileNotFoundError: If the plugin file does not exist.
        ValueError: If the plugin does not define a valid I/O configuration.
    """
    path = Path(plugin_path)
    if not path.exists():
        raise FileNotFoundError(f"I/O plugin file not found: {plugin_path}")

    globals_dict = runpy.run_path(str(path))

    if EXTERNAL_IO_VARIABLE_NAME in globals_dict:
        external_functions = globals_dict[EXTERNAL_IO_VARIABLE_NAME]
        if not isinstance(external_functions, dict):
            raise ValueError(
                f"{EXTERNAL_IO_VARIABLE_NAME} must be a dict of function overrides; "
                f"got {type(external_functions)!r}."
            )
        functions_dict: dict[str, Callable] = {}
        for name, func in external_functions.items():
            if name not in io_utils.ALLOWED_IO_OVERRIDES:
                logger.warning("Ignoring unknown I/O override: %s", name)
                continue
            if not callable(func):
                continue
            functions_dict[name] = func

    else:
        functions_dict = {}
        for name in io_utils.ALLOWED_IO_OVERRIDES:
            if name in globals_dict and callable(globals_dict[name]):
                functions_dict[name] = globals_dict[name]

    if not functions_dict:
        if EXTERNAL_IO_VARIABLE_NAME not in globals_dict:
            raise ValueError(
                f"Plugin file must define {EXTERNAL_IO_VARIABLE_NAME} "
                f"or at least one top-level I/O function."
            )

    _print_io_functions(functions_dict)
    return io_utils.ChemReporterIO(functions_dict)


def _print_io_functions(functions_dict: dict[str, Callable]):
    logger.info("Loaded %s I/O functions ", list(functions_dict.keys()))
    not_overridden_functions = set(io_utils.ALLOWED_IO_OVERRIDES) - set(
        functions_dict.keys()
    )
    if not_overridden_functions:
        logger.warning(
            "keeping default I/O functions for %s. Not overridden functions: %s",
            not_overridden_functions,
            not_overridden_functions,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chemreporter",
        description="chemreporter command line interface",
    )

    parser.add_argument(
        "--io-plugin",
        type=str,
        help=(
            "Path to a Python file exporting CHEMREPORTER_IO (a dict mapping "
            "I/O method names to functions), or top-level functions named "
            "after the I/O methods to override."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command", help="available commands", required=True
    )

    parser_process = subparsers.add_parser(
        "process", help="Process raw database into chemreporter format"
    )
    parser_process.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the YAML configuration file for process",
    )

    parser_query = subparsers.add_parser(
        "query", help="Query the chemreporter database"
    )
    parser_query.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the YAML configuration file for query",
    )

    parser_export = subparsers.add_parser(
        "export", help="Export processed database to HDF5"
    )
    parser_export.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the YAML configuration file for export",
    )

    parser_generate = subparsers.add_parser(
        "generate-config", help="Generate a template YAML configuration file"
    )
    parser_generate.add_argument(
        "command_name",
        choices=["process", "query", "export"],
        help="The command to generate a template for",
    )

    return parser


def _unwrap_optional_annotation(annotation: Any) -> Any:
    """Return the non-None type from an optional annotation.

    Args:
        annotation: A field type annotation, optionally wrapped in ``Optional``.

    Returns:
        The inner annotation when wrapped in ``Optional``; otherwise ``annotation``.
    """
    origin = get_origin(annotation)
    if origin is None:
        return annotation

    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1:
        return args[0]
    return annotation


def _is_model_class(annotation: Any) -> bool:
    """Return whether an annotation refers to a Pydantic model class.

    Args:
        annotation: A field type annotation.

    Returns:
        ``True`` when ``annotation`` is a ``BaseModel`` subclass.
    """
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _template_value_for_field(field_info: FieldInfo) -> Any:
    """Build one template value from a Pydantic field definition.

    Nested ``BaseModel`` fields are expanded recursively. Required scalar fields
    use a placeholder marker; optional fields use Pydantic defaults when set.

    Args:
        field_info: Metadata for one model field.

    Returns:
        A placeholder string, default value, or nested template dictionary.
    """
    annotation = _unwrap_optional_annotation(field_info.annotation)
    if _is_model_class(annotation):
        return _template_for_model(annotation)

    if field_info.is_required():
        return _REQUIRED_TEMPLATE_MARKER

    if field_info.default is not PydanticUndefined:
        return field_info.default

    if field_info.default_factory is not None:
        return field_info.default_factory()

    return ""


def _template_for_model(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Build a nested template dictionary from a Pydantic model class.

    Args:
        model_cls: Config schema model to walk.

    Returns:
        Mapping of field names to placeholders or default values.
    """
    return {
        field_name: _template_value_for_field(field_info)
        for field_name, field_info in model_cls.model_fields.items()
    }


def generate_config_template(command_name: str):
    """Generate and print a YAML template from the Pydantic schema.

    Args:
        command_name: The name of the command to generate a template for.

    Raises:
        ValueError: If the command name is unknown.
    """
    if command_name == "process":
        schema_cls = ProcessDBConfig
    elif command_name == "query":
        schema_cls = QueryDBConfig
    elif command_name == "export":
        schema_cls = ExportHDF5Config
    else:
        raise ValueError(f"Unknown command: {command_name}")

    template_dict = _template_for_model(schema_cls)

    print(f"# Template configuration for {command_name}\n")
    print(yaml.dump(template_dict, sort_keys=False, default_flow_style=False))


def main():
    """Parse command line arguments and dispatch to the selected command."""
    parser = _parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        force=True,
    )

    if args.command == "generate-config":
        generate_config_template(args.command_name)
        sys.exit(0)

    if args.io_plugin:
        io_handler = load_io_plugin(args.io_plugin)
    else:
        io_handler = io_utils.ChemReporterIO()

    if not hasattr(args, "config") or not args.config:
        parser.error(
            f"the following arguments are required: -c/--config "
            f"for command {args.command}"
        )
    if not Path(args.config).exists():
        parser.error(f"Configuration file not found: {args.config}")
    config = OmegaConf.load(args.config)

    if args.command == "process":
        process.run_main(config, io_handler)
    elif args.command == "query":
        query.run_main(config, io_handler)
    elif args.command == "export":
        export.run_main(config, io_handler)


if __name__ == "__main__":
    main()
