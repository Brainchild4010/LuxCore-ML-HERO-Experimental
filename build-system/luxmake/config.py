# SPDX-FileCopyrightText: 2025 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""Config command.

This command has also been extended to export compile commands for CMake
(`compile_commands.json` file), mainly for syntaxic checkers.
"""

from .constants import PARAMS
from .utils import run_cmake, fail, logger


def config(
    _,  # args (unused)
):
    """CMake config."""
    # Check whether presets exist
    presets = PARAMS.BINARY_DIR / "build" / "generators" / "CMakePresets.json"
    if not presets.exists():
        fail(
            "Cannot find presets file ('%s'). "
            "Have you run 'make deps' beforehand?",
            str(presets.absolute()),
        )

    # Prepare and run command
    cmd = [
        "--preset conan-default",
        f"-DCMAKE_INSTALL_PREFIX={str(PARAMS.INSTALL_DIR)}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"-S {str(PARAMS.SOURCE_DIR)}",
    ]
    run_cmake(cmd)

    # Info
    compile_commands_file = (
        PARAMS.BINARY_DIR / "build" / "compile_commands.json"
    )
    logger.info(
        "Compile commands file generated at: '%s'", compile_commands_file
    )
