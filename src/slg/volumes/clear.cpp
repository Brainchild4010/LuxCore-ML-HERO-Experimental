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

#include <cstddef>

#include "slg/volumes/clear.h"
#include "slg/bsdf/bsdf.h"
#include "slg/textures/irregulardata.h"

// ML HERO state is currently implemented as global thread-local helpers in
// glass.cpp. Keep these declarations global to match the existing definitions.
bool GetMLHeroEnabled();
float GetMLCurrentWaveLength();
void MarkMLDispersionUsed();

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// ClearVolume
//------------------------------------------------------------------------------

ClearVolume::ClearVolume(
	TextureConstRef iorTex,
	TextureConstPtr emiTex,
	TextureConstRef a
) : Volume(iorTex, emiTex), sigmaA(a) {}

Spectrum ClearVolume::SigmaA(const HitPoint &hitPoint) const {
	// ML HERO: evaluate the original IrregularData spectrum at the current
	// per-thread hero wavelength. Standard LuxCore RGB behavior stays intact
	// for all other textures and when HERO is disabled.
	if (::GetMLHeroEnabled()) {
		const float waveLength = ::GetMLCurrentWaveLength();
		if ((waveLength >= 380.f) && (waveLength <= 780.f)) {
			const IrregularDataTexture *irregular =
				dynamic_cast<const IrregularDataTexture *>(&GetSigmaA());
			if (irregular) {
				const float value = Max(0.f, irregular->GetSpectralValue(waveLength));
				::MarkMLDispersionUsed();
				return Spectrum(value);
			}
		}
	}

	return GetSigmaA().GetSpectrumValue(hitPoint).Clamp();
}

Spectrum ClearVolume::SigmaS(const HitPoint &hitPoint) const {
	return Spectrum();
}

float ClearVolume::Scatter(const Ray &ray, const float u,
		const bool scatteredStart, Spectrum *connectionThroughput,
		Spectrum *connectionEmission) const {
	// Point where to evaluate the volume
	HitPoint hitPoint;
	hitPoint.Init();
	hitPoint.fixedDir = ray.d;
	hitPoint.p = ray.o;
	hitPoint.geometryN = hitPoint.interpolatedN = hitPoint.shadeN = Normal(-ray.d);
	hitPoint.passThroughEvent = u;
	
	const float distance = ray.maxt - ray.mint;
	Spectrum transmittance(1.f);

	const Spectrum sigma = SigmaT(hitPoint);
	if (!sigma.Black()) {
		const Spectrum tau = (distance * sigma).Clamp();
		transmittance = Exp(-tau);
	}

	// Apply volume transmittance
	*connectionThroughput *= transmittance;

	// Apply volume emission
	if (volumeEmissionTex)
		*connectionEmission += *connectionThroughput * distance * volumeEmissionTex->GetSpectrumValue(hitPoint).Clamp();
	
	return -1.f;
}

Spectrum ClearVolume::Albedo(const HitPoint &hitPoint) const {
	throw runtime_error("Internal error: called ClearVolume::Albedo()");
}

Spectrum ClearVolume::Evaluate(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
		float *directPdfW, float *reversePdfW) const {
	throw runtime_error("Internal error: called ClearVolume::Evaluate()");
}

Spectrum ClearVolume::Sample(const HitPoint &hitPoint,
		const Vector &localFixedDir, Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event) const {
	throw runtime_error("Internal error: called ClearVolume::Sample()");
}

void ClearVolume::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	throw runtime_error("Internal error: called ClearVolume::Pdf()");
}

void ClearVolume::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Volume::AddReferencedTextures(referencedTexs);

	GetSigmaA().AddReferencedTextures(referencedTexs);
}

void ClearVolume::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Volume::UpdateTextureReferences(oldTex, newTex);

	updtex(sigmaA, oldTex, newTex);
}

PropertiesUPtr ClearVolume::ToProperties() const {
	PropertiesUPtr props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.volumes." + name + ".type")("clear"));
	props->Set(Property("scene.volumes." + name + ".absorption")(GetSigmaA().GetSDLValue()));
	props->Set(Volume::ToProperties());

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
