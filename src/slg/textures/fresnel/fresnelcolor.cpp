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

#include <cmath>

#include "slg/textures/fresnel/fresnelcolor.h"
#include "luxrays/core/color/spds/rgbrefl.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Fresnel color texture
//------------------------------------------------------------------------------

float FresnelColorTexture::GetFloatValue(const HitPoint &hitPoint) const {
	return GetKr().GetFloatValue(hitPoint);
}

Spectrum FresnelColorTexture::GetSpectrumValue(const HitPoint &hitPoint) const {
	return GetKr().GetSpectrumValue(hitPoint);
}

float FresnelColorTexture::Y() const {
	return GetKr().Y();
}

float FresnelColorTexture::Filter() const {
	return GetKr().Filter();
}

Spectrum FresnelColorTexture::Evaluate(const HitPoint &hitPoint, const float cosi) const {
	const Spectrum c = GetKr().GetSpectrumValue(hitPoint);

	const Spectrum n = ApproxN(c);
	const Spectrum k = ApproxK(c);

	return GeneralEvaluate(n, k, cosi);
}

// ML HERO phase 15nu: FresnelColor stores normal-incidence reflectance as an
// RGB texture, not measured n/k curves.  Reconstruct that authored Kr at the
// requested wavelength and then apply the exact same ApproxN/ApproxK model as
// the native RGB Evaluate() above.  Keeping this separate from GetNKAtWaveLength
// is intentional: these n/k values are reconstructed, not measured metadata.
bool FresnelColorTexture::EvaluateAtWaveLength(const HitPoint &hitPoint,
		const float cosi, const float waveLength, Spectrum *fresnel) const {
	if (!fresnel || !std::isfinite(waveLength) ||
			(waveLength < 380.f) || (waveLength > 780.f))
		return false;

	const Spectrum kr = GetKr().GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	if (kr.IsNaN() || kr.IsInf())
		return false;

	const RGBReflSPD krSPD(RGBColor(kr.c[0], kr.c[1], kr.c[2]));
	// LuxCore's stock Smits table ends at 720 nm and treats the exact upper
	// boundary as outside in some builds.  Hold its last interior value through
	// HERO's 720..780 nm extension, matching the established HERO RGB path.
	const float krAtWaveLength = Clamp(
		krSPD.Sample(Clamp(waveLength, 380.f, 719.999f)), 0.f, .999f);
	const Spectrum eta(ApproxN(krAtWaveLength));
	const Spectrum k(ApproxK(krAtWaveLength));

	*fresnel = GeneralEvaluate(eta, k, cosi).Clamp(0.f, 1.f);
	return !fresnel->IsNaN() && !fresnel->IsInf();
}

PropertiesUPtr FresnelColorTexture::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.textures." + name + ".type")("fresnelcolor"));
	props->Set(Property("scene.textures." + name + ".kr")(GetKr().GetSDLValue()));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
