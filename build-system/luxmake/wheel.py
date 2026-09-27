# SPDX-FileCopyrightText: 2025 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0

"""Make wheel command."""

import sys
import json
import subprocess
import tempfile
import shutil
import platform
import os
import shlex
import itertools
import re
import sysconfig
import logging
import runpy
from datetime import datetime, timezone
from pathlib import Path

from .constants import PARAMS
from .utils import logger, pack, fail, Colors, get_dep_version, run_module
from .build import build_and_install
from .config import config
from .windows import win_recompose

# ---------------------------------------------------------------------------
# ML LuxCore HERO build metadata
#
# BUILD / PHASE / FEATURE and compile switches are read automatically from
# src/slg/engines/bidircpu/bidircputhread.cpp.
# ---------------------------------------------------------------------------

ML_HERO_COMPILE_FLAGS = (
    "ML_HERO_GLASS_PER_LANE_WEIGHT",
    "ML_HERO_QUARTER_CYCLING",
)

# Following snippets are intended to wheel reconstruction
_WHEEL_SNIPPET = """\
Wheel-Version: 1.0
Generator: fake 0.0.0
Root-Is-Purelib: false
Tag: {}
"""

_METADATA_SNIPPET = """\
Metadata-Version: 2.2
Name: pyluxcore
Version: {}
Summary: LuxCore Python bindings
Keywords: raytracing,ray tracing,rendering,pbr,physical based rendering,path tracing
Author: LuxCoreRender
Requires-Python: >=3.10
Requires-Dist: {}; sys_platform != "darwin"
"""

_ENTRYPOINTS_SNIPPET = """\
[console_scripts]
pyluxcoretest = pyluxcoretest:main
pyluxcore-console = pyluxcoretools.console.cmd:main
pyluxcore-maketx = pyluxcoretools.maketx.cmd:main
pyluxcore-merge = pyluxcoretools.merge.cmd:main
pyluxcore-netmenu = pyluxcoretools.netmenu.cmd:main
pyluxcore-netconsole = pyluxcoretools.netconsole.cmd:main
pyluxcore-netnode = pyluxcoretools.netnode.cmd:main

[gui_scripts]
pyluxcore-netconsole-ui = pyluxcoretools.netconsole.ui:main
pyluxcore-netnode-ui = pyluxcoretools.netnode.ui:main
"""


def _compute_platform_tag():
    """Compute tag.

    This tag may not be totally correct. Do not use in production.
    https://packaging.python.org/en/latest/specifications/platform-compatibility-tags
    """
    system, machine = platform.system(), platform.machine()
    if system == "Linux":
        return "linux_x86_64"
    if system == "Windows":
        if machine.lower() == "arm64":
            return "win_arm64"
        return "win_amd64"
    if system == "Darwin" and machine == "x86_64":
        return "macosx_13_0"
    if system == "Darwin" and machine == "arm64":
        return "macosx_14_2"

    return fail("Unknown platform/system: '%s' / '%s'", platform, machine)


def _get_lib_paths():
    """Get library paths for dependencies."""
    base = PARAMS.BINARY_DIR / "dependencies" / "full_deploy" / "host"
    paths_bin = (str(p.absolute()) for p in base.rglob("**/bin"))
    paths_lib = (str(p.absolute()) for p in base.rglob("**/lib"))
    paths = itertools.chain(paths_bin, paths_lib)
    result = [["-l", Path(p)] for p in paths]
    result = list(itertools.chain.from_iterable(result))
    return result


def _check_repairwheel():
    output = run_module("repairwheel", ["-V"])
    logger.info("repairwheel version: %s", output)
    version = output.split(".")
    if version < ["0", "7", "0"]:
        fail("repairwheel >= 0.7.0 is required")



def _read_ml_hero_source_metadata():
    """Read BUILD / PHASE / FEATURE from bidircputhread.cpp."""
    source_file = (
        PARAMS.SOURCE_DIR
        / "src"
        / "slg"
        / "engines"
        / "bidircpu"
        / "bidircputhread.cpp"
    )

    result = {
        "build": "UNKNOWN",
        "phase": "UNKNOWN",
        "feature": "UNKNOWN",
    }

    if not source_file.is_file():
        logger.warning("ML HERO metadata: source file not found: %s", source_file)
        return result

    try:
        source = source_file.read_text(encoding="utf-8", errors="ignore")
    except OSError as err:
        logger.warning("ML HERO metadata: could not read %s: %s", source_file, err)
        return result

    define_map = {
        "build": "ML_HERO_BUILD",
        "phase": "ML_HERO_TEST_PHASE",
        "feature": "ML_HERO_FEATURE",
    }

    for key, define_name in define_map.items():
        match = re.search(
            rf'^\s*#\s*define\s+{re.escape(define_name)}\s+"([^"]*)"',
            source,
            flags=re.MULTILINE,
        )
        if match:
            result[key] = match.group(1)

    return result


def _read_ml_hero_compile_flags():
    """Read selected ML HERO #define values from the current source tree."""
    values = {name: None for name in ML_HERO_COMPILE_FLAGS}

    # The current HERO A/B switches live in bidircputhread.cpp.
    candidates = (
        PARAMS.SOURCE_DIR / "src" / "slg" / "engines" / "bidircpu" / "bidircputhread.cpp",
    )

    for source_file in candidates:
        if not source_file.is_file():
            continue

        try:
            source = source_file.read_text(encoding="utf-8", errors="ignore")
        except OSError as err:
            logger.warning("ML HERO metadata: could not read %s: %s", source_file, err)
            continue

        for name in values:
            match = re.search(
                rf"^\s*#\s*define\s+{re.escape(name)}\s+([^\s/]+)",
                source,
                flags=re.MULTILINE,
            )
            if not match:
                continue

            raw_value = match.group(1).strip()
            try:
                values[name] = int(raw_value, 0)
            except ValueError:
                values[name] = raw_value

    return values


def _write_ml_hero_build_info(wheeltree, version, python_tag, platform_tag):
    """Write ml_hero_build.json into the pyluxcore package in the wheel."""
    package_dir = wheeltree / "pyluxcore"
    package_dir.mkdir(parents=True, exist_ok=True)

    source_metadata = _read_ml_hero_source_metadata()

    build_info = {
        "schema": 1,
        "project": "ML LuxCore HERO",
        "build": source_metadata["build"],
        "phase": source_metadata["phase"],
        "feature": source_metadata["feature"],
        "luxcore_version": version,
        "python_tag": python_tag,
        "platform_tag": platform_tag,
        "build_type": PARAMS.DEFAULT_BUILD_TYPE,
        "build_date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "compile_flags": _read_ml_hero_compile_flags(),
    }

    output_file = package_dir / "ml_hero_build.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(build_info, f, indent=2, sort_keys=True)
        f.write("\n")

    logger.info(
        "ML HERO metadata: %s / %s -> %s",
        source_metadata["build"],
        source_metadata["phase"],
        output_file,
    )
    logger.info(
        "ML HERO compile flags: %s",
        build_info["compile_flags"],
    )


def make_wheel(args):
    """Build a wheel."""
    _check_repairwheel()

    # ML HERO test wheels are built as Release.
    PARAMS.DEFAULT_BUILD_TYPE = "Release"

    args.target = "pyluxcore"
    config(args)
    build_and_install(args)

    disclaimer = (
        f"{Colors.WARNING2}"
        "This command builds a TEST wheel, "
        "not fully compliant to standard "
        "and only intended for test. "
        f"{Colors.WARNING3}"
        "DO NOT USE IN PRODUCTION."
        f"{Colors.ENDC}"
    )
    logger.warning(disclaimer)

    build_settings_file = Path("build-system", "build-settings.json")
    with open(build_settings_file, encoding="utf-8") as in_file:
        default_version = json.load(in_file)["DefaultVersion"]
    version = ".".join(default_version[i] for i in ("major", "minor", "patch"))

    vinfo = sys.version_info
    python_tag = f"cp{vinfo.major}{vinfo.minor}"
    abi_tag = python_tag
    platform_tag = _compute_platform_tag()
    tag = f"{python_tag}-{abi_tag}-{platform_tag}"

    logger.info("Making wheel for version '%s' and tag '%s'", version, tag)
    with (
        tempfile.TemporaryDirectory() as wheeltree,
        tempfile.TemporaryDirectory() as raw_wheel,
    ):
        wheeltree = Path(wheeltree)
        raw_wheel_dir = Path(raw_wheel)

        # Check Python version in extension. Ignore unrelated files and only
        # inspect the actual pyluxcore .pyd.
        extension_path = PARAMS.INSTALL_DIR / "pyluxcore"
        extensions = [
            f.name
            for f in extension_path.iterdir()
            if f.is_file()
            and f.name.startswith("pyluxcore")
            and f.suffix.lower() == ".pyd"
        ]

        if not extensions:
            raise RuntimeError(f"No pyluxcore extension in {extension_path}")

        extension = extensions[0]

        ext_version = None
        match = re.search(r"\.[^.]*?(\d+)", extension)
        if match:
            ext_version = match.group(1)
        else:
            logger.warning(
                f"{Colors.WARNING2}"
                "Could not get version tag from extension name "
                f"('{extension}'). "
                "Compatibility between extension tag and wheel tag could not "
                "be verified."
                f"{Colors.ENDC}"
            )

        soabi = sysconfig.get_config_var("SOABI")
        abi_match = re.search(r"(\d+)", soabi or "")
        abi_version = abi_match.group(1) if abi_match else None

        if (
            ext_version is not None
            and abi_version is not None
            and ext_version != abi_version
        ):
            raise RuntimeError(
                "Cannot build wheel: "
                f"Extension Python version ({ext_version}) is different "
                f"from Wheel Python version ({abi_version})."
            )

        dist_info = wheeltree / f"pyluxcore-{version}.dist-info"
        dist_info.mkdir(exist_ok=True)

        with open(dist_info / "WHEEL", "w", encoding="utf-8") as f:
            f.write(_WHEEL_SNIPPET.format(tag))

        with open(dist_info / "METADATA", "w", encoding="utf-8") as f:
            nvrtc_version = get_dep_version("nvrtc")
            logger.info("NVRTC version: %s", nvrtc_version)
            major = int(nvrtc_version.split(".")[0])
            requirement = (
                f"nvidia-cuda-nvrtc-cu{major}=={nvrtc_version}"
                if major <= 12
                else f"nvidia-cuda-nvrtc=={nvrtc_version}"
            )
            f.write(_METADATA_SNIPPET.format(version, requirement))

        with open(dist_info / "entry_points.txt", "w", encoding="utf-8") as f:
            f.write(_ENTRYPOINTS_SNIPPET)

        shutil.copytree(
            PARAMS.SOURCE_DIR / "python" / "pyluxcore",
            wheeltree / "pyluxcore",
            dirs_exist_ok=True,
        )
        shutil.copytree(
            PARAMS.SOURCE_DIR / "python" / "pyluxcoretest",
            wheeltree / "pyluxcoretest",
            dirs_exist_ok=True,
        )
        shutil.copytree(
            PARAMS.SOURCE_DIR / "python" / "pyluxcoretools",
            wheeltree / "pyluxcoretools",
            dirs_exist_ok=True,
        )
        shutil.copytree(
            PARAMS.INSTALL_DIR / "pyluxcore",
            wheeltree / "pyluxcore",
            dirs_exist_ok=True,
        )
        shutil.copytree(
            PARAMS.INSTALL_DIR / "pyluxcore.libs",
            wheeltree / "pyluxcore.libs",
            dirs_exist_ok=True,
        )

        # ML HERO: generate build provenance after all pyluxcore files have
        # been copied, so the JSON is guaranteed to be part of the wheel.
        _write_ml_hero_build_info(
            wheeltree,
            version,
            python_tag,
            platform_tag,
        )

        logger.info("Packing wheel")
        pack(wheeltree, raw_wheel)
        wheelname = f"pyluxcore-{version}-{tag}.whl"

        wheel_lib_dir = PARAMS.INSTALL_DIR / "lib"
        logger.info("Repairing wheel")
        input_path = raw_wheel_dir / wheelname
        logging.basicConfig(level=logging.DEBUG)
        repair_args = [
            "-l",
            wheel_lib_dir,
            *_get_lib_paths(),
            "-o",
            PARAMS.WHEELHOUSE_DIR,
            input_path,
        ]
        run_module("repairwheel", repair_args)

        if platform.system() == "Windows":
            args.wheel = PARAMS.WHEELHOUSE_DIR / wheelname
            win_recompose(args)

        if PARAMS.WHEEL_HOOK:
            logger.info("Executing hook: %s", PARAMS.WHEEL_HOOK)
            try:
                result = subprocess.check_output(
                    shlex.split(PARAMS.WHEEL_HOOK), text=True
                )
            except subprocess.CalledProcessError as err:
                fail(err)
            logger.info(result)
