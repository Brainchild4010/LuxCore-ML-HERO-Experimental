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

#include "slg/materials/velvet.h"
#include "slg/engines/mlhero_velvet_audit15lq.h"
#include "slg/engines/mlhero_velvet_spectral15nn.h"
#include "slg/engines/mlhero_glossy2_spectral15nj.h"

#include <limits>

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Velvet material
//------------------------------------------------------------------------------

VelvetMaterial::VelvetMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump,
		TextureConstPtr kd, TextureConstPtr p1, TextureConstPtr p2, TextureConstPtr p3,
		TextureConstPtr thickness) :
			Material(frontTransp, backTransp, emitted, bump), Kd(kd),
			P1(p1), P2(p2), P3(p3), Thickness(thickness) {
	glossiness = 1.f;
}

Spectrum VelvetMaterial::Albedo(const HitPoint &hitPoint) const {
	return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
}

Spectrum VelvetMaterial::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {


	if (directPdfW)
		*directPdfW = fabsf((hitPoint.fromLight ? localEyeDir.z : localLightDir.z) * INV_PI);

	if (reversePdfW)
		*reversePdfW = fabsf((hitPoint.fromLight ? localLightDir.z : localEyeDir.z) * INV_PI);

	*event = DIFFUSE | REFLECT;
	
	const float A1 = P1->GetFloatValue(hitPoint);
	const float A2 = P2->GetFloatValue(hitPoint);
	const float A3 = P3->GetFloatValue(hitPoint);
	const float delta = Thickness->GetFloatValue(hitPoint);
	
	const float cosv = -Dot(localLightDir, localEyeDir);

	// Compute phase function

	// Clamp the physical BRDF (page 7), not the direction-dependent
	// cosine-weighted transport value.
	const float p = mlhero15nn::PhaseFactor(A1, A2, A3, delta, cosv,
		fabsf(localLightDir.z), fabsf(localEyeDir.z));

	return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * p;
}

Spectrum VelvetMaterial::Sample(const HitPoint &hitPoint,
	const Vector &localFixedDir, Vector *localSampledDir,
	const float u0, const float u1, const float passThroughEvent,
	float *pdfW, BSDFEvent *event) const {
	if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	*localSampledDir = Sgn(localFixedDir.z) * CosineSampleHemisphere(u0, u1, pdfW);
	if (fabsf(CosTheta(*localSampledDir)) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	*event = DIFFUSE | REFLECT;
	
	float A1 = P1->GetFloatValue(hitPoint);
	float A2 = P2->GetFloatValue(hitPoint);
	float A3 = P3->GetFloatValue(hitPoint);
	float delta = Thickness->GetFloatValue(hitPoint);
	slg::mlhero15lq::Attempt audit15lq(hitPoint.fromLight, GetID(), A1, A2, A3,
		delta, u0, u1, passThroughEvent, slg::mlhero15kk::V(localFixedDir));
	
	const float cosv = -Dot(localFixedDir, *localSampledDir);

	// Compute phase function

	const float absLightCos = hitPoint.fromLight ? fabsf(localFixedDir.z) :
		fabsf(localSampledDir->z);
	const float absEyeCos = hitPoint.fromLight ? fabsf(localSampledDir->z) :
		fabsf(localFixedDir.z);

	// Use the same reciprocal BRDF clamp as Evaluate().
	const float p = mlhero15nn::PhaseFactor(A1, A2, A3, delta, cosv,
		absLightCos, absEyeCos);

	if (audit15lq.Active())
		audit15lq.Accept(slg::mlhero15kk::V(*localSampledDir), *pdfW, p);

	return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * (p / *pdfW);
}

bool VelvetMaterial::PrepareMLHeroSpectralEndpoint15nn(
		const HitPoint &hitPoint, const Vector &localLightDir,
		const Vector &localEyeDir,
		MLHeroVelvetSpectralEndpoint15nn *prepared) const {
	if (!prepared)
		return false;

	prepared->event = DIFFUSE | REFLECT;
	prepared->kd = Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float a1 = P1->GetFloatValue(hitPoint);
	const float a2 = P2->GetFloatValue(hitPoint);
	const float a3 = P3->GetFloatValue(hitPoint);
	const float thickness = Thickness->GetFloatValue(hitPoint);
	const float cosv = -Dot(localLightDir, localEyeDir);
	prepared->phaseFactor = mlhero15nn::PhaseFactor(a1, a2, a3,
		thickness, cosv, fabsf(localLightDir.z), fabsf(localEyeDir.z));

	return prepared->kd.IsValid() && std::isfinite(a1) &&
		std::isfinite(a2) && std::isfinite(a3) &&
		std::isfinite(thickness) && std::isfinite(cosv) &&
		std::isfinite(prepared->phaseFactor) &&
		prepared->phaseFactor >= 0.f;
}

bool VelvetMaterial::EvaluateMLHeroSpectralEndpoint15nn(
		const MLHeroVelvetSpectralEndpoint15nn &prepared,
		const float waveLength, const bool basisCompensation,
		float *value) {
	if (!value || !std::isfinite(waveLength) ||
			!std::isfinite(prepared.phaseFactor) ||
			prepared.phaseFactor < 0.f || !prepared.kd.IsValid())
		return false;

	const float kd = mlhero15nj::Reflectance(prepared.kd, waveLength,
		basisCompensation);
	const double result = double(kd) * double(prepared.phaseFactor);
	if (!std::isfinite(result) || result < 0.0 ||
			result > double(std::numeric_limits<float>::max()))
		return false;
	*value = float(result);
	return true;
}

void VelvetMaterial::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	if (directPdfW)
		*directPdfW = fabsf((hitPoint.fromLight ? localEyeDir.z : localLightDir.z) * INV_PI);

	if (reversePdfW)
		*reversePdfW = fabsf((hitPoint.fromLight ? localLightDir.z : localEyeDir.z) * INV_PI);
}

void VelvetMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	Kd->AddReferencedTextures(referencedTexs);
	P1->AddReferencedTextures(referencedTexs);
	P2->AddReferencedTextures(referencedTexs);
	P3->AddReferencedTextures(referencedTexs);
	Thickness->AddReferencedTextures(referencedTexs);
}

void VelvetMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	if (Kd == &oldTex)
		Kd = &newTex;
	if (P1 == &oldTex)
		P1 = &newTex;
	if (P2 == &oldTex)
		P2 = &newTex;
	if (P3 == &oldTex)
		P3 = &newTex;
	if (Thickness == &oldTex)
		Thickness = &newTex;
}

PropertiesUPtr VelvetMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const std::string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("velvet"));
	props->Set(Property("scene.materials." + name + ".kd")(Kd->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".p1")(P1->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".p2")(P2->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".p3")(P3->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".thickness")(Thickness->GetSDLValue()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
