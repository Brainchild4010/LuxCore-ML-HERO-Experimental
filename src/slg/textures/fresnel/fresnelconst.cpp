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

#include "slg/textures/fresnel/fresnelconst.h"

using namespace std;
using namespace luxrays;
using namespace slg;

// ML-HERO helpers are currently implemented as global functions in glass.cpp.
// Keep these declarations global too (do not put them in namespace slg).
bool GetMLHeroEnabled();
float GetMLCurrentWaveLength();
void MarkMLDispersionUsed();

//------------------------------------------------------------------------------
// Fresnel const texture
//------------------------------------------------------------------------------

float FresnelConstTexture::GetSpectralValue(const vector<float> &values, const float waveLength) const {
	if (waveLengths.empty() || values.empty() || (waveLengths.size() != values.size()))
		return 0.f;

	if (waveLength <= waveLengths.front())
		return values.front();
	if (waveLength >= waveLengths.back())
		return values.back();

	const auto upper = lower_bound(waveLengths.begin(), waveLengths.end(), waveLength);
	const size_t i1 = upper - waveLengths.begin();
	const size_t i0 = i1 - 1;

	const float wl0 = waveLengths[i0];
	const float wl1 = waveLengths[i1];
	const float t = (waveLength - wl0) / (wl1 - wl0);

	return Lerp(t, values[i0], values[i1]);
}

float FresnelConstTexture::GetFloatValue(const HitPoint &hitPoint) const {
	return 0.f;
}

Spectrum FresnelConstTexture::GetSpectrumValue(const HitPoint &hitPoint) const {
	return GeneralEvaluate(n, k, .5f);
}

float FresnelConstTexture::Y() const {
	return 0.f;
}

float FresnelConstTexture::Filter() const {
	return 0.f;
}

bool FresnelConstTexture::HasSpectralData() const {
	return !waveLengths.empty() &&
			(waveLengths.size() == nSpectral.size()) &&
			(waveLengths.size() == kSpectral.size());
}

bool FresnelConstTexture::GetNKAtWaveLength(const HitPoint &hitPoint, const float waveLength,
		Spectrum *eta, Spectrum *kk) const {
	if (!HasSpectralData() || (waveLength < 380.f) || (waveLength > 780.f))
		return false;

	const float nValue = Max(.001f, GetSpectralValue(nSpectral, waveLength));
	const float kValue = Max(.001f, GetSpectralValue(kSpectral, waveLength));

	if (eta)
		*eta = Spectrum(nValue);
	if (kk)
		*kk = Spectrum(kValue);

	return true;
}

Spectrum FresnelConstTexture::Evaluate(const HitPoint &hitPoint, const float cosi) const {
	// CPU ML-HERO path: use explicit spectral n/k for the active HERO lane.
	if (::GetMLHeroEnabled()) {
		Spectrum eta, kk;
		if (GetNKAtWaveLength(hitPoint, ::GetMLCurrentWaveLength(), &eta, &kk)) {
			::MarkMLDispersionUsed();
			return GeneralEvaluate(eta, kk, cosi);
		}
	}

	return GeneralEvaluate(n, k, cosi);
}

PropertiesUPtr FresnelConstTexture::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.textures." + name + ".type")("fresnelconst"));
	props->Set(Property("scene.textures." + name + ".n")(n));
	props->Set(Property("scene.textures." + name + ".k")(k));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
