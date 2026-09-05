/***************************************************************************
 * Copyright 1998-2025 by authors (see AUTHORS.txt)                        *
 *                                                                         *
 *   This file is part of LuxCoreRender.                                   *
 *                                                                         *
 * Licensed under the Apache License, Version 2.0 (the "License");         *
 * you may not use this file except in compliance with the License.        *
 * You may obtain a copy of the License at                                 *
 *                                                                         *
 *     http://www.apache.org/licenses/LICENSE-2.0                          *
 *                                                                         *
 * Unless required by applicable law or agreed to in writing, software     *
 * distributed under the License is distributed on an "AS IS" BASIS,       *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.*
 * See the License for the specific language governing permissions and     *
 * limitations under the License.                                          *
 ***************************************************************************/

// This file is intended to gather all usings (pointers, refs etc.) for
// luxrays classes

#pragma once

#include <memory>
#include <thread>
#include <optional>
#include "luxrays/utils/observer_ptr.h"

// Below are the reference/pointer macro types to use
// Please note that ownership is restricted to unique and shared, and
// explicitely exclude raw pointers
#define DECLARE_SUBTYPES(T) \
class T; \
using T##Ref = T&;                                 /* Reference */\
using T##ConstRef = const T&;                      /* Const reference */\
using T##UPtr = std::unique_ptr<T>;				   /* Unique pointer (OWNING) */\
using T##ConstUPtr = std::unique_ptr<const T>;     /* Const unique point (OWNING) */\
using T##RPtr = const std::unique_ptr<T> &;		   /* Reference to unique */\
using T##ConstRPtr = const std::unique_ptr<const T> &; /* Reference to const unique */\
using T##SPtr = std::shared_ptr<T>;                /* Shared pointer (OWNING) */\
using T##ConstSPtr = std::shared_ptr<const T>;     /* Shared const pointer (OWNING) */\
using T##Ptr = luxrays::observer_ptr<T>;		   /* Raw pointer */\
using T##ConstPtr = luxrays::observer_ptr<const T>;   /* Raw const pointer */\
using T##Ptr = luxrays::observer_ptr<T> ;          /* Raw pointer */\
using T##ConstPtr = luxrays::observer_ptr<const T>;/* Raw const pointer */\




namespace luxrays {

DECLARE_SUBTYPES(Accelerator);
DECLARE_SUBTYPES(BBox);
DECLARE_SUBTYPES(Blob);
DECLARE_SUBTYPES(BSphere);
DECLARE_SUBTYPES(Context);
DECLARE_SUBTYPES(DataSet);
DECLARE_SUBTYPES(Device);
DECLARE_SUBTYPES(DeviceDescription);
DECLARE_SUBTYPES(HardwareDevice);
DECLARE_SUBTYPES(IntersectionDevice)
DECLARE_SUBTYPES(Mesh);
DECLARE_SUBTYPES(ExtMesh);
DECLARE_SUBTYPES(ExtTriangleMesh);
DECLARE_SUBTYPES(ExtInstanceTriangleMesh);
DECLARE_SUBTYPES(ExtMotionTriangleMesh);
DECLARE_SUBTYPES(Matrix4x4);
DECLARE_SUBTYPES(NamedObject);
DECLARE_SUBTYPES(Normal);
DECLARE_SUBTYPES(Point);
DECLARE_SUBTYPES(Ray);
DECLARE_SUBTYPES(RayHit);
DECLARE_SUBTYPES(RGBColor);
DECLARE_SUBTYPES(Triangle);
DECLARE_SUBTYPES(TriangleMesh);
DECLARE_SUBTYPES(UV);
DECLARE_SUBTYPES(Vector);
DECLARE_SUBTYPES(Property);
DECLARE_SUBTYPES(Properties);
DECLARE_SUBTYPES(RandomGenerator);

using JThread = std::jthread;
using JThreadUPtr = std::unique_ptr<std::jthread>;


}

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
