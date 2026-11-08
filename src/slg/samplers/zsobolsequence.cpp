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

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "slg/samplers/sobolsequence.h"
#include "slg/samplers/zsobolindex.h"
#include "slg/samplers/zsobolsequence.h"

using namespace slg;

ZSobolSequence::ZSobolSequence() : directions(nullptr), requestedDimensions(0u) {
}

ZSobolSequence::~ZSobolSequence() {
	delete[] directions;
}

void ZSobolSequence::RequestSamples(const u_int size) {
	if (size == 0u || size > SOBOL_MAX_DIMENSIONS)
		throw std::runtime_error("ZSOBOL requires 1.." +
			std::to_string(SOBOL_MAX_DIMENSIONS) + " sample dimensions; requested " +
			std::to_string(size) + ". Reduce path depths.");

	std::unique_ptr<u_int[]> next(new u_int[static_cast<std::size_t>(size) * ZSOBOL_BITS]);
	SobolSequence::GenerateDirectionVectors(next.get(), size, ZSOBOL_BITS);
	delete[] directions;
	directions = next.release();
	requestedDimensions = size;
}

u_int ZSobolSequence::GetRawSample(const std::uint64_t sampleIndex,
		const u_int dimension) const {
	if (!directions || dimension >= requestedDimensions)
		throw std::out_of_range("ZSOBOL sample dimension was not requested");
	if (sampleIndex >= (std::uint64_t(1) << ZSOBOL_BITS))
		throw std::out_of_range("ZSOBOL sample index exceeds 52 columns");

	const std::size_t offset = static_cast<std::size_t>(dimension) * ZSOBOL_BITS;
	u_int result = 0u;
	std::uint64_t bits = sampleIndex;

	for (u_int column = 0u; bits; bits >>= 1u, ++column) {
		if (bits & 1u)
			result ^= directions[offset + column];
	}

	return result;
}

float ZSobolSequence::GetSample(const std::uint64_t sampleIndex,
		const u_int dimension, const u_int scrambleSeed) const {
	const u_int raw = GetRawSample(sampleIndex, dimension);
	const u_int scrambled = ZSobolIndex::FastOwenScramble(raw, scrambleSeed);
	const double normalized = static_cast<double>(scrambled) * (1.0 / 4294967296.0);

	// Conversion of the largest uint32_t to float can round to exactly 1.
	return std::min(static_cast<float>(normalized), std::nextafter(1.f, 0.f));
}

