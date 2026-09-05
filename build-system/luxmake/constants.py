# SPDX-FileCopyrightText: 2025 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""Constants for luxmake."""

import os
import pathlib


class Parameters:
    """Singleton class for global parameters handling."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    _internals = {"DEFAULT_BUILD_TYPE": "Release"}

    # Rest of the class is setter/getter, with transparent intention,
    # so we will deactivate pylint
    # pylint:disable=invalid-name
    # pylint:disable=missing-function-docstring

    @property
    def DEFAULT_BUILD_TYPE(self):
        return self._internals["DEFAULT_BUILD_TYPE"]

    @DEFAULT_BUILD_TYPE.setter
    def DEFAULT_BUILD_TYPE(self, build_type):
        self._internals["DEFAULT_BUILD_TYPE"] = build_type

    # Environment variables that control the build
    @property
    def SOURCE_DIR(self):
        return pathlib.Path(
            os.getenv(
                "LUX_SOURCE_DIR",
                os.getcwd(),
            )
        )

    @property
    def BINARY_DIR(self):
        return pathlib.Path(
            os.getenv(
                "LUX_BINARY_DIR",
                "out",
            )
        )

    @property
    def WHEEL_HOOK(self):
        return os.getenv(  # Hook to execute after wheel-test build
            "LUX_WHEEL_HOOK",
            "",
        )

    @property
    def BUILD_TYPE(self):
        return os.getenv("LUX_BUILD_TYPE", self.DEFAULT_BUILD_TYPE)

    @property
    def BUILD_DIR(self):
        return self.BINARY_DIR / "build"

    @property
    def INSTALL_DIR(self):
        return self.BINARY_DIR / "install" / self.BUILD_TYPE

    @property
    def WHEELHOUSE_DIR(self):
        return self.INSTALL_DIR / "wheel"  # Where the wheel will be created


PARAMS = Parameters()
