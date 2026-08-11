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
"""Post-query action normalization and dispatch for ``cli.query``."""

import logging

from chemreporter.query_database_tools.actions_tools import (
    extract_smiles,
    make_histograms,
    make_statistics,
)

logger = logging.getLogger("chemreporter")

LIST_OF_SUPPORTED_ACTIONS = [
    "make_statistics",
    "make_histograms",
    "extract_smiles",
]


def normalize_actions(actions) -> dict[str, list[str] | str | None]:
    """Normalize actions (bare names or {name: args} dicts) into one dict.

    Returns:
        Dict mapping action name to its arguments (None if bare).
    """
    actions_dict: dict[str, list[str] | str | None] = {}
    for action in actions:
        if isinstance(action, str):
            actions_dict[action] = None
        else:
            actions_dict.update(action)
    return actions_dict


def warn_unsupported_actions(actions_dict: dict[str, list[str] | str | None]) -> None:
    """Log a warning for every action outside LIST_OF_SUPPORTED_ACTIONS."""
    for action_name in actions_dict:
        if action_name not in LIST_OF_SUPPORTED_ACTIONS:
            logger.warning("Action %s not supported, will be ignored", action_name)
            logger.warning(
                "Supported actions are: " + ", ".join(LIST_OF_SUPPORTED_ACTIONS)
            )


def run_actions(
    actions_dict: dict[str, list[str] | str | None], db_query, keys, output_path
) -> None:
    """Run each requested, supported action against the filtered keys."""
    if "make_statistics" in actions_dict:
        column_names = actions_dict["make_statistics"]
        if column_names is not None:
            logger.info("Running action: make_statistics")
            make_statistics(
                db_query,
                keys=keys,
                output_path=output_path,
                column_names=column_names,
            )

    if "make_histograms" in actions_dict:
        columns = actions_dict["make_histograms"]
        if columns is not None:
            make_histograms(
                db_query,
                entry_keys=keys,
                output_path=output_path,
                column_names=columns,
            )

    if "extract_smiles" in actions_dict:
        extract_smiles(db_query, keys=keys, output_path=output_path)
