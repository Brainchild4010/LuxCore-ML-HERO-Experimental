################################################################################
# Copyright 1998-2020 by authors (see AUTHORS.txt)
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
################################################################################

message(STATUS "Checking compiler version...")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14.0)
        message(FATAL_ERROR "GCC >= 14.0 is required, found ${CMAKE_CXX_COMPILER_VERSION}")
    endif()

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 20.0)
        message(FATAL_ERROR "Clang/LLVM >= 20.0 is required, found ${CMAKE_CXX_COMPILER_VERSION}")
    endif()

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # MSVC_VERSION is in a weird format: e.g. 1929 -> Visual Studio 2019 version 16.9
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.40)
        message(FATAL_ERROR "MSVC >= 19.40 (at least Visual Studio 2022 version 17.10) is required, found ${CMAKE_CXX_COMPILER_VERSION}")
    endif()

else()
    message(WARNING "Unknown compiler '${CMAKE_CXX_COMPILER_ID}', skipping version check.")
endif()
