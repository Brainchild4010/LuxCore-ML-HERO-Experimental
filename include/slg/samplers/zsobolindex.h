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

#ifndef _SLG_ZSOBOL_INDEX_H
#define _SLG_ZSOBOL_INDEX_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace slg {

//------------------------------------------------------------------------------
// ZSobolIndex
//
// Pure integer reference implementation of the hierarchical base-4 index
// permutation used by the PBRT-v4 ZSobol sampler. It deliberately contains no
// renderer, RNG or Sobol-direction-vector state so it can be tested in
// isolation before it is connected to a LuxCore sampler.
//------------------------------------------------------------------------------

class ZSobolIndex {
public:
	ZSobolIndex(const std::uint32_t samplesPerPixel,
			const std::uint32_t filmWidth, const std::uint32_t filmHeight) {
		if (!samplesPerPixel || (samplesPerPixel & (samplesPerPixel - 1u)))
			throw std::invalid_argument("ZSobol samples per pixel must be a power of two");
		if (!filmWidth || !filmHeight)
			throw std::invalid_argument("ZSobol film resolution must be non-zero");

		log2SamplesPerPixel = Log2Pow2(samplesPerPixel);
		configuredSamplesPerPixel = samplesPerPixel;
		configuredFilmWidth = filmWidth;
		configuredFilmHeight = filmHeight;
		const std::uint32_t resolution = RoundUpPow2(std::max(filmWidth, filmHeight));
		const std::uint32_t log4SamplesPerPixel = (log2SamplesPerPixel + 1u) / 2u;
		nBase4Digits = Log2Pow2(resolution) + log4SamplesPerPixel;

		// The largest shift used by GetIndex() must fit a uint64_t. A 64-bit
		// index is intentional: LuxCore's existing Sobol pass is only 32 bit.
		indexBits = 2u * nBase4Digits - (log2SamplesPerPixel & 1u);
		if (indexBits > 64u)
			throw std::invalid_argument("ZSobol extended Morton index exceeds 64 bits");
	}

	std::uint64_t GetMortonIndex(const std::uint32_t pixelX,
			const std::uint32_t pixelY, const std::uint32_t sampleInPixel) const {
		if (pixelX >= configuredFilmWidth || pixelY >= configuredFilmHeight)
			throw std::out_of_range("ZSobol pixel lies outside the configured film");
		if (sampleInPixel >= configuredSamplesPerPixel)
			throw std::out_of_range("ZSobol sample index exceeds configured samples per pixel");

		return (EncodeMorton2(pixelX, pixelY) << log2SamplesPerPixel) | sampleInPixel;
	}

	std::uint64_t GetIndex(const std::uint32_t pixelX, const std::uint32_t pixelY,
			const std::uint32_t sampleInPixel, const std::uint32_t dimension) const {
		return PermuteIndex(GetMortonIndex(pixelX, pixelY, sampleInPixel), dimension);
	}

	std::uint64_t PermuteIndex(const std::uint64_t mortonIndex,
			const std::uint32_t dimension) const {
		static const std::uint8_t permutations[24][4] = {
			{0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1},
			{0, 3, 2, 1}, {0, 3, 1, 2}, {1, 0, 2, 3}, {1, 0, 3, 2},
			{1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 2, 0}, {1, 3, 0, 2},
			{2, 1, 0, 3}, {2, 1, 3, 0}, {2, 0, 1, 3}, {2, 0, 3, 1},
			{2, 3, 0, 1}, {2, 3, 1, 0}, {3, 1, 2, 0}, {3, 1, 0, 2},
			{3, 2, 1, 0}, {3, 2, 0, 1}, {3, 0, 2, 1}, {3, 0, 1, 2}
		};

		std::uint64_t sampleIndex = 0;
		const bool oddLog2Samples = (log2SamplesPerPixel & 1u) != 0u;
		const int lastDigit = oddLog2Samples ? 1 : 0;

		for (int i = static_cast<int>(nBase4Digits) - 1; i >= lastDigit; --i) {
			const int digitShift = 2 * i - (oddLog2Samples ? 1 : 0);
			std::uint32_t digit = static_cast<std::uint32_t>((mortonIndex >> digitShift) & 3u);
			const std::uint64_t higherDigits = mortonIndex >> (digitShift + 2);
			const std::uint32_t permutation = static_cast<std::uint32_t>(
				(MixBits(higherDigits ^ (std::uint64_t(0x55555555u) * dimension)) >> 24) % 24u);
			digit = permutations[permutation][digit];
			sampleIndex |= std::uint64_t(digit) << digitShift;
		}

		if (oddLog2Samples) {
			const std::uint32_t digit = static_cast<std::uint32_t>(mortonIndex & 1u);
			sampleIndex |= digit ^ static_cast<std::uint32_t>(
				MixBits((mortonIndex >> 1) ^ (std::uint64_t(0x55555555u) * dimension)) & 1u);
		}

		return sampleIndex;
	}

	std::uint32_t GetLog2SamplesPerPixel() const { return log2SamplesPerPixel; }
	std::uint32_t GetBase4DigitCount() const { return nBase4Digits; }
	std::uint32_t GetIndexBitCount() const { return indexBits; }

	static std::uint64_t MixBits(std::uint64_t value) {
		value ^= value >> 30;
		value *= UINT64_C(0xbf58476d1ce4e5b9);
		value ^= value >> 27;
		value *= UINT64_C(0x94d049bb133111eb);
		value ^= value >> 31;
		return value;
	}

	static std::uint32_t FastOwenScramble(std::uint32_t value, const std::uint32_t seed) {
		value = ReverseBits32(value);
		value ^= value * 0x3d20adeau;
		value += seed;
		value *= (seed >> 16) | 1u;
		value ^= value * 0x05526c56u;
		value ^= value * 0x53a22864u;
		return ReverseBits32(value);
	}

private:
	static std::uint32_t Log2Pow2(std::uint32_t value) {
		std::uint32_t result = 0;
		while (value > 1u) {
			value >>= 1;
			++result;
		}
		return result;
	}

	static std::uint32_t RoundUpPow2(std::uint32_t value) {
		if (value > (std::uint32_t(1) << 31))
			throw std::invalid_argument("ZSobol film extent is too large");
		--value;
		value |= value >> 1;
		value |= value >> 2;
		value |= value >> 4;
		value |= value >> 8;
		value |= value >> 16;
		return value + 1u;
	}

	static std::uint64_t Part1By1(std::uint32_t value) {
		std::uint64_t result = value;
		result = (result | (result << 16)) & UINT64_C(0x0000ffff0000ffff);
		result = (result | (result << 8)) & UINT64_C(0x00ff00ff00ff00ff);
		result = (result | (result << 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
		result = (result | (result << 2)) & UINT64_C(0x3333333333333333);
		result = (result | (result << 1)) & UINT64_C(0x5555555555555555);
		return result;
	}

	static std::uint64_t EncodeMorton2(const std::uint32_t x, const std::uint32_t y) {
		return Part1By1(x) | (Part1By1(y) << 1);
	}

	static std::uint32_t ReverseBits32(std::uint32_t value) {
		value = ((value & 0x55555555u) << 1) | ((value >> 1) & 0x55555555u);
		value = ((value & 0x33333333u) << 2) | ((value >> 2) & 0x33333333u);
		value = ((value & 0x0f0f0f0fu) << 4) | ((value >> 4) & 0x0f0f0f0fu);
		value = (value << 24) | ((value & 0xff00u) << 8) |
			((value >> 8) & 0xff00u) | (value >> 24);
		return value;
	}

	std::uint32_t log2SamplesPerPixel;
	std::uint32_t nBase4Digits;
	std::uint32_t indexBits;
	std::uint32_t configuredSamplesPerPixel;
	std::uint32_t configuredFilmWidth;
	std::uint32_t configuredFilmHeight;
};

}

#endif // _SLG_ZSOBOL_INDEX_H

