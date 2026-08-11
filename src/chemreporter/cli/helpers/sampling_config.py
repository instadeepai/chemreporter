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
from chemreporter.query_database_tools.sample_plugins import load_sampler_from_path

FINGERPRINT_COLUMN_PRESET = "fingerprint_bits"
BUILTIN_SAMPLING_METHODS = frozenset({"random", "all"})


def _resolve_sampling_columns(sampling: dict) -> list[str] | None:
    """Resolve fingerprint column names for custom sampling methods.

    Args:
        sampling: Hydra ``sampling`` config block.

    Returns:
        Column names to pass to ``query_to_keys``, or ``None``.
    """
    columns = sampling.get("required_columns")
    if columns is None:
        return None
    if isinstance(columns, str):
        if columns == FINGERPRINT_COLUMN_PRESET:
            return [FINGERPRINT_COLUMN_PRESET]
        return [columns]
    return list(columns)


def resolve_query_sampling(sampling: dict) -> dict:
    """Map ``sampling`` config to ``query_to_keys`` sampling arguments.

    The config's ``method`` key is either a built-in name (``random`` or
    ``all``) or a path to a Python plugin file defining a
    ``custom_sampling_function``. Extra sampler arguments come from the
    config's ``kwargs`` key. The returned dict uses ``query_to_keys``'s
    internal ``sampling_method*`` argument names, with ``sampling_method``
    holding either the built-in name or the loaded custom sampler callable.

    Args:
        sampling: Hydra ``sampling`` config block.

    Returns:
        Keyword arguments for ``QueryDatabaseHandler.query_to_keys``.

    Raises:
        FileNotFoundError: If ``method`` is a path that doesn't exist.
        ValueError: If the plugin file doesn't define ``custom_sampling_function``.
    """
    n_samples = sampling.get("n_samples", None)
    sampling_method = sampling.get("method", "random")
    seed = sampling.get("seed", None)

    if sampling_method not in BUILTIN_SAMPLING_METHODS:
        sampler = load_sampler_from_path(sampling_method)
        kwargs = sampling.get("kwargs", {})
        return {
            "n_samples": n_samples,
            "seed": seed,
            "sampling_method": sampler,
            "sampling_required_columns": _resolve_sampling_columns(sampling),
            "sampling_kwargs": dict(kwargs or {}),
        }

    return {
        "n_samples": n_samples,
        "sampling_method": sampling_method,
        "seed": seed,
        "sampling_required_columns": None,
        "sampling_kwargs": None,
    }
