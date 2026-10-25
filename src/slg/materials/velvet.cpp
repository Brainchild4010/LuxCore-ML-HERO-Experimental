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

using namespace std;
using namespace luxrays;
using namespace slg;

namespace {

// Material::Evaluate() returns f(light, eye) * |cos(light)|.  Clamp the
// physical BRDF before applying that transport cosine so the result remains
// reciprocal when light and eye directions are exchanged.
static inline float VelvetCosWeightedValue(const float phaseThickness,
		const float absLightCos, const float absEyeCos) {
	const float cosineProduct = absLightCos * absEyeCos;
	if (!(cosineProduct > 0.f))
		return 0.f;

	float brdf = phaseThickness / cosineProduct;
	if (brdf > 1.f)
		brdf = 1.f;
	else if (brdf < 0.f)
		brdf = 0.f;
	return brdf * absLightCos;
}

}

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

	const float B = 3.0f * cosv;

	const float polynomial = 1.0f + A1 * cosv + A2 * 0.5f * (B * cosv - 1.0f) +
		A3 * 0.5f * (5.0f * cosv * cosv * cosv - B);
	const float phaseThickness = polynomial * delta / (4.0f * M_PI);

	// Clamp the physical BRDF (page 7), not the direction-dependent
	// cosine-weighted transport value.
	const float p = VelvetCosWeightedValue(phaseThickness,
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

	const float B = 3.0f * cosv;

	const float polynomial = 1.0f + A1 * cosv + A2 * 0.5f * (B * cosv - 1.0f) +
		A3 * 0.5f * (5.0f * cosv * cosv * cosv - B);
	const float phaseThickness = polynomial * delta / (4.0f * M_PI);
	const float absLightCos = hitPoint.fromLight ? fabsf(localFixedDir.z) :
		fabsf(localSampledDir->z);
	const float absEyeCos = hitPoint.fromLight ? fabsf(localSampledDir->z) :
		fabsf(localFixedDir.z);

	// Use the same reciprocal BRDF clamp as Evaluate().
	const float p = VelvetCosWeightedValue(phaseThickness, absLightCos, absEyeCos);

	if (audit15lq.Active())
		audit15lq.Accept(slg::mlhero15kk::V(*localSampledDir), *pdfW, p);

	return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * (p / *pdfW);
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
