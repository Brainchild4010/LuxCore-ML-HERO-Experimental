# SPDX-FileCopyrightText: 2025 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""Preset command."""

import re
from enum import (
    Enum,
)

from .utils import run_cmake


# Preset stuff
class PresetType(Enum):
    """CMake preset types."""

    CONFIG = "configure"
    BUILD = "build"
    TEST = "test"
    PACKAGE = "package"


PRESET_PATTERN = re.compile(r"\s*\"([a-zA-Z\-]+)\".*")


def get_presets(
    preset_type: PresetType,
):
    """CMake get presets for a given type."""
    preset_type = str(preset_type.value)
    cmd = [f"--list-presets={preset_type}"]
    res = run_cmake(
        cmd,
        capture_output=True,
        text=True,
    )
    presets = [
        preset[1]
        for line in res.stdout.splitlines()
        if (preset := PRESET_PATTERN.fullmatch(line)) is not None
    ]
    return presets


def get_all_presets():
    """CMake get all presets."""
    presets = {
        preset_type.value: get_presets(preset_type)
        for preset_type in PresetType.__members__.values()
    }
    return presets


def list_presets(
    _,
):
    """List all presets."""
    print(get_all_presets())
