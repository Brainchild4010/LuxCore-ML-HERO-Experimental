# SPDX-FileCopyrightText: 2025 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""Build and install commands."""

from .constants import PARAMS
from .utils import run_cmake, fail
from .presets import get_presets, PresetType


def build(
    args,
):
    """CMake build."""
    preset = _get_preset_from_build_type(PARAMS.BUILD_TYPE)
    cmd = [
        "--build",
        f"--preset {preset}",
        f"--target {args.target}",
    ]
    run_cmake(cmd)


def install(
    args,
):
    """CMake install."""
    cmd = [
        "--install",
        str(PARAMS.BUILD_DIR),
        f"--prefix {PARAMS.INSTALL_DIR}",
        f"--config {PARAMS.BUILD_TYPE}",
        f"--component {args.target}",
    ]
    run_cmake(cmd)


def build_and_install(
    args,
):
    """CMake build and install."""
    build(args)
    install(args)


_PRESETS = {
    "Release": "conan-release",
    "Debug": "conan-debug",
    "RelWithDebInfo": "conan-relwithdebinfo",
    "MinSizeRel": "conan-minsizerel",
}


def _get_preset_from_build_type(
    build_type,
):
    """Get conan preset from build type."""
    try:
        preset = _PRESETS[build_type]
    except KeyError:
        fail(
            "Unknown build type '%s'. Valid values (case sensitive) are: '%s'",
            build_type,
            _PRESETS.keys(),
        )
    if preset not in (presets := get_presets(PresetType.BUILD)):
        fail("Preset '%s' missing. Available presets: '%s'", preset, presets)
    return preset
