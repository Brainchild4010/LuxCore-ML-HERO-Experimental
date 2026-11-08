/***************************************************************************
 * Copyright 1998-2026 by authors (see AUTHORS.txt)                        *
 *                                                                         *
 *   This file is part of LuxCoreRender.                                   *
 *                                                                         *
 * Licensed under the Apache License, Version 2.0 (the "License");         *
 * you may not use this file except in compliance with the License.        *
 * You may obtain a copy of the License at                                 *
 *                                                                         *
 * http://www.apache.org/licenses/LICENSE-2.0                              *
 *                                                                         *
 * Unless required by applicable law or agreed to in writing, software     *
 * distributed under the License is distributed on an "AS IS" BASIS,       *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.*
 ***************************************************************************/

#ifndef _SLG_ZSOBOL_SEQUENCE_H
#define _SLG_ZSOBOL_SEQUENCE_H

#include <cstdint>

#include "slg/slg.h"
#include "slg/samplers/sampler.h"

namespace slg {

// 52 columns match the extended index width used by the reference Sobol
// matrices while retaining a 32-bit output value.
#define ZSOBOL_BITS 52

class ZSobolSequence {
public:
	ZSobolSequence();
	~ZSobolSequence();

	void RequestSamples(const u_int size);

	// scrambleSeed is deliberately explicit. The later sampler layer owns the
	// policy that hashes renderer seed and logical dimension together.
	float GetSample(const std::uint64_t sampleIndex, const u_int dimension,
			const u_int scrambleSeed) const;
	u_int GetRawSample(const std::uint64_t sampleIndex, const u_int dimension) const;

	u_int GetRequestedDimensions() const { return requestedDimensions; }

private:
	u_int *directions;
	u_int requestedDimensions;
};

}

#endif // _SLG_ZSOBOL_SEQUENCE_H

