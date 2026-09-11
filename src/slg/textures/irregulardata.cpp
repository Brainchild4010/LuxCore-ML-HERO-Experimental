/***************************************************************************
 * Copyright 1998-2020 by authors (see AUTHORS.txt)                        *
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

#include <algorithm>

#include "luxrays/core/color/spds/irregular.h"

#include "slg/textures/irregulardata.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Irregular data texture
//------------------------------------------------------------------------------

IrregularDataTexture::IrregularDataTexture(const u_int n,
		const float *wl, const float *dt,
		const float res, bool em) :
	waveLengths(n), data(n), resolution(res), emission(em)
{
	copy(wl, wl + n, waveLengths.begin());
	copy(dt, dt + n, data.begin());

	IrregularSPD spd(&waveLengths[0], &data[0], n, resolution);

	if (emission) {
		ColorSystem colorSpace;
		rgb = colorSpace.ToRGBConstrained(spd.ToXYZ()).Clamp(0.f);
	} else {
		ColorSystem colorSpace(.63f, .34f, .31f, .595f, .155f, .07f,
			1.f / 3.f, 1.f / 3.f, 1.f);
		rgb = colorSpace.ToRGBConstrained(spd.ToNormalizedXYZ()).Clamp(0.f);
	}
}

float IrregularDataTexture::GetSpectralValue(const float waveLength) const {
	if (waveLengths.empty() || data.empty())
		return 0.f;

	if (waveLength <= waveLengths.front())
		return data.front();
	if (waveLength >= waveLengths.back())
		return data.back();

	const auto upperIt = std::lower_bound(waveLengths.begin(), waveLengths.end(), waveLength);
	const size_t upperIndex = static_cast<size_t>(upperIt - waveLengths.begin());

	if (*upperIt == waveLength)
		return data[upperIndex];

	const size_t lowerIndex = upperIndex - 1;
	const float wl0 = waveLengths[lowerIndex];
	const float wl1 = waveLengths[upperIndex];
	const float v0 = data[lowerIndex];
	const float v1 = data[upperIndex];

	const float delta = wl1 - wl0;
	if (delta <= 0.f)
		return v0;

	const float t = (waveLength - wl0) / delta;
	return v0 + (v1 - v0) * t;
}

PropertiesUPtr IrregularDataTexture::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.textures." + name + ".type")("irregulardata"));
	props->Set(Property("scene.textures." + name + ".wavelengths")(waveLengths));
	props->Set(Property("scene.textures." + name + ".data")(data));
	props->Set(Property("scene.textures." + name + ".resolution")(resolution));
	props->Set(Property("scene.textures." + name + ".emission")(emission));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
