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
"""Load user-provided sampling callables from a Python file."""

from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path

CUSTOM_SAMPLER_ENTRYPOINT = "custom_sampling_function"


def load_sampler_from_path(plugin_path: str | Path) -> Callable[..., list[str]]:
    """Load the ``custom_sampling_function`` callable from a Python plugin file.

    Args:
        plugin_path: Path to a Python file defining a ``custom_sampling_function``.

    Returns:
        Sampling callable ``(frame, n_samples, **kwargs) -> list[str]``.

    Raises:
        FileNotFoundError: If ``plugin_path`` does not exist.
        ValueError: If ``custom_sampling_function`` is missing or not callable.
    """
    path = Path(plugin_path)
    if not path.exists():
        raise FileNotFoundError(f"Sampling plugin file not found: {plugin_path}")

    globals_dict = runpy.run_path(str(path))
    if CUSTOM_SAMPLER_ENTRYPOINT not in globals_dict:
        raise ValueError(
            f"Sampling plugin {plugin_path} must define a "
            f"{CUSTOM_SAMPLER_ENTRYPOINT!r} function."
        )
    candidate = globals_dict[CUSTOM_SAMPLER_ENTRYPOINT]
    if not callable(candidate):
        raise ValueError(
            f"{CUSTOM_SAMPLER_ENTRYPOINT!r} in {plugin_path} is not callable."
        )
    return candidate
