###############################################################################
# Copyright 1998-2025 by authors (see AUTHORS.txt)
#
#   This file is part of LuxCoreRender.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import sys
import argparse
import logging

import pyluxcore
from .._utils import loghandler

logger = logging.getLogger(loghandler.loggerName + ".luxcoremaketx")


def LuxCoreMakeTx(argv):
    parser = argparse.ArgumentParser(
        prog="pyluxcore-maketx",
        description="PyLuxCoreMakeTx",
    )
    parser.add_argument(
        "srcImageFileName", help="source image file name to convert"
    )
    parser.add_argument(
        "dstImageFileName", help="destination image TX file name"
    )

    # Parse command line arguments
    args = parser.parse_args(argv)

    if not args.srcImageFileName:
        raise TypeError("Source image file name must be specified")
    if not args.dstImageFileName:
        raise TypeError("Destination image file name must be specified")

    logger.info(
        "Converting [%s] to [%s]",
        args.srcImageFileName,
        args.dstImageFileName,
    )
    pyluxcore.MakeTx(args.srcImageFileName, args.dstImageFileName)

    logger.info("Done.")


def main():
    argv = sys.argv
    try:
        pyluxcore.Init(loghandler.LuxCoreLogHandler)
        logger.info("LuxCore %s", pyluxcore.Version())

        LuxCoreMakeTx(argv[1:])
    finally:
        pyluxcore.SetLogHandler(None)
