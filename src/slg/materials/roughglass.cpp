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

#include "slg/textures/fresnel/fresneltexture.h"
#include "slg/materials/roughglass.h"
#include "slg/materials/thinfilmcoating.h"
#include "slg/usings.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// ML HERO dispersion state shared with GlassMaterial (CPU only, Test A)
//------------------------------------------------------------------------------

extern thread_local float mlDispersionWaveLength;
extern thread_local bool mlDispersionUsed;
extern thread_local bool mlHeroEnabled;

static float RoughGlassMLGetWaveLength() {
	if ((mlDispersionWaveLength >= 380.f) && (mlDispersionWaveLength <= 780.f))
		return mlDispersionWaveLength;

	// Fallback only for safety. CPU Path/Bidir Goldstand normally initializes
	// the wavelength before tracing the sample.
	return 550.f;
}

static float RoughGlassMLWaveLength2IOR(const float waveLength,
		const float IOR, const float B) {
	// Same d-line corrected Cauchy convention used by the current HERO Glass.
	const float A = IOR - B / Sqr(587.56f / 1000.f);
	return A + B / Sqr(waveLength / 1000.f);
}

static float RoughGlassMLGetInteriorIOR(const HitPoint &hitPoint,
		TextureConstPtr interiorIor, TextureConstPtr cauchyB) {
	const float nt = ExtractInteriorIors(hitPoint, interiorIor);
	const float b = cauchyB ? cauchyB->GetFloatValue(hitPoint) : 0.f;

	if (mlHeroEnabled && (b > 0.f))
		return RoughGlassMLWaveLength2IOR(RoughGlassMLGetWaveLength(), nt, b);

	// LuxCore Standard RoughGlass has no spectral Cauchy dispersion.
	return nt;
}


//------------------------------------------------------------------------------
// ML Fine RoughGlass - CPU Test B
//
// For isotropic RoughGlass, LuxCore's Schlick NDF is mathematically the same
// isotropic GGX/Trowbridge-Reitz NDF when:
//
//     alpha^2 = roughness = u * v
//
// The stock sampler draws half-vectors from the full NDF. At grazing angles
// this can generate many hidden/back-facing microfacets and therefore many
// rejected/black samples. Test B samples the *visible* GGX normal distribution
// (VNDF) for the isotropic case, while keeping the existing RoughGlass BRDF,
// HERO IOR, Fresnel, D and G terms unchanged.
//
// Anisotropic RoughGlass intentionally falls back to the original LuxCore
// Schlick sampler in this first test.
//------------------------------------------------------------------------------

static bool RoughGlassMLUseVNDF(const float anisotropy) {
	return fabsf(anisotropy) < 1e-6f;
}

static float RoughGlassMLGGXG1(const float alpha, const Vector &v) {
	const float cosTheta = fabsf(v.z);
	if (cosTheta <= 0.f)
		return 0.f;

	const float sinTheta2 = Max(0.f, 1.f - cosTheta * cosTheta);
	if (sinTheta2 <= 0.f)
		return 1.f;

	const float tanTheta2 = sinTheta2 / (cosTheta * cosTheta);
	const float root = sqrtf(1.f + alpha * alpha * tanTheta2);
	return 2.f / (1.f + root);
}

static float RoughGlassMLVisibleNormalPdf(const float roughness,
		const Vector &fixedDir, const Vector &wh) {
	const float cosTheta = fabsf(fixedDir.z);
	if (cosTheta <= DEFAULT_COS_EPSILON_STATIC)
		return 0.f;

	const float alpha = sqrtf(Max(roughness, 1e-18f));
	const float D = SchlickDistribution_D(roughness, wh, 0.f);
	const float G1 = RoughGlassMLGGXG1(alpha, fixedDir);
	return D * G1 * fabsf(Dot(fixedDir, wh)) / cosTheta;
}

static void RoughGlassMLSampleVisibleNormal(const float roughness,
		const Vector &fixedDir, const float u0, const float u1,
		Vector *wh, float *d, float *pdf) {
	const float alpha = sqrtf(Max(roughness, 1e-18f));

	// Work in the upper hemisphere. RoughGlass later keeps the half-vector
	// convention wh.z >= 0 for both entering and exiting paths.
	Vector V = (fixedDir.z >= 0.f) ? fixedDir : -fixedDir;

	// Stretch the view direction.
	Vector Vh = Normalize(Vector(alpha * V.x, alpha * V.y, V.z));

	// Build an orthonormal basis around Vh.
	const float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	const Vector T1 = (lensq > 0.f) ?
			Vector(-Vh.y, Vh.x, 0.f) / sqrtf(lensq) :
			Vector(1.f, 0.f, 0.f);
	const Vector T2 = Cross(Vh, T1);

	// Sample a disk and warp it to the visible hemisphere.
	const float r = sqrtf(Clamp(u0, 0.f, 1.f));
	const float phi = 2.f * M_PI * u1;
	float t1 = r * cosf(phi);
	float t2 = r * sinf(phi);
	const float s = .5f * (1.f + Vh.z);
	t2 = (1.f - s) * sqrtf(Max(0.f, 1.f - t1 * t1)) + s * t2;

	const float z = sqrtf(Max(0.f, 1.f - t1 * t1 - t2 * t2));
	const Vector Nh = t1 * T1 + t2 * T2 + z * Vh;

	// Unstretch.
	*wh = Normalize(Vector(alpha * Nh.x, alpha * Nh.y, Max(0.f, Nh.z)));
	if (wh->z < 0.f)
		*wh = -*wh;

	*d = SchlickDistribution_D(roughness, *wh, 0.f);
	*pdf = RoughGlassMLVisibleNormalPdf(roughness, fixedDir, *wh);
}


//------------------------------------------------------------------------------
// ML Fine RoughGlass - CPU Test C: smooth Near-Delta transition
//
// The user-facing roughness is represented by sqrt(u * v) for the transition.
// Isotropic U=V therefore maps directly to the value shown in Blender.
//
// <= 0.0005 : 100% exact Glass/HERO (delta)
// >= 0.0050 : 100% RoughGlass/VNDF
// between    : smooth stochastic mixture
//
// This is a true BSDF mixture: the continuous RoughGlass lobe is weighted by
// roughWeight in Evaluate/Pdf, while Sample chooses either the delta or glossy
// lobe with the same probabilities. This avoids a hard visual switch.
//------------------------------------------------------------------------------

static float RoughGlassMLNearDeltaWeight(const float u, const float v) {
	const float uiRoughness = sqrtf(Max(0.f, u * v));
	const float deltaStart = 0.0005f;
	const float deltaEnd = 0.005f;

	if (uiRoughness <= deltaStart)
		return 1.f;
	if (uiRoughness >= deltaEnd)
		return 0.f;

	const float x = (deltaEnd - uiRoughness) / (deltaEnd - deltaStart);
	// Smoothstep for a soft transition.
	return x * x * (3.f - 2.f * x);
}

static Spectrum RoughGlassMLSampleDelta(const HitPoint &hitPoint,
		const Vector &localFixedDir, Vector *localSampledDir,
		const float passThroughEvent,
		const Spectrum &kr, const Spectrum &kt,
		const float nc, const float nt,
		const float localFilmThickness, const float localFilmIor,
		float *pdfW, BSDFEvent *event) {
	// Exact HERO Glass transmission candidate.
	Spectrum trans;
	Vector transLocalSampledDir;
	if (!kt.Black()) {
		const float ntc = nt / nc;
		const float cosTheta = CosTheta(localFixedDir);
		const bool entering = (cosTheta > 0.f);
		const float eta = entering ? (nc / nt) : ntc;
		const float eta2 = eta * eta;
		const float sini2 = SinTheta2(localFixedDir);
		const float sint2 = eta2 * sini2;

		if (sint2 < 1.f) {
			const float cost = sqrtf(Max(0.f, 1.f - sint2)) *
					(entering ? -1.f : 1.f);
			transLocalSampledDir = Vector(-eta * localFixedDir.x,
					-eta * localFixedDir.y, cost);

			float ce;
			if (!hitPoint.fromLight)
				ce = (1.f - FresnelTexture::CauchyEvaluate(ntc, cost)) * eta2;
			else {
				const float absCosSampledDir =
						fabsf(CosTheta(transLocalSampledDir));
				ce = (1.f - FresnelTexture::CauchyEvaluate(ntc, cosTheta)) *
						fabsf(CosTheta(localFixedDir) / absCosSampledDir);
			}

			trans = kt * ce;
		}
	}

	// Exact HERO Glass reflection candidate.
	Spectrum refl;
	Vector reflLocalSampledDir;
	if (!kr.Black()) {
		const float cosTheta = CosTheta(localFixedDir);
		reflLocalSampledDir = Vector(-localFixedDir.x, -localFixedDir.y,
				localFixedDir.z);

		const float ntc = nt / nc;
		refl = kr * FresnelTexture::CauchyEvaluate(ntc, cosTheta);

		if (localFilmThickness > 0.f)
			refl *= CalcFilmColor(localFixedDir, localFilmThickness,
					localFilmIor);
	}

	float threshold;
	if (!refl.Black()) {
		if (!trans.Black()) {
			const float reflFilter = refl.Filter();
			const float transFilter = trans.Filter();
			threshold = transFilter / (reflFilter + transFilter);
			threshold = Clamp(threshold, .25f, .75f);
		} else
			threshold = 0.f;
	} else {
		if (!trans.Black())
			threshold = 1.f;
		else
			return Spectrum();
	}

	Spectrum result;
	if (passThroughEvent < threshold) {
		*localSampledDir = transLocalSampledDir;
		*event = SPECULAR | TRANSMIT;
		*pdfW = threshold;
		result = trans;
	} else {
		*localSampledDir = reflLocalSampledDir;
		*event = SPECULAR | REFLECT;
		*pdfW = 1.f - threshold;
		result = refl;
	}

	return result / *pdfW;
}

//------------------------------------------------------------------------------
// RoughGlass material
//
// LuxRender RoughGlass material porting.
//------------------------------------------------------------------------------

RoughGlassMaterial::RoughGlassMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump,
		TextureConstPtr refl, TextureConstPtr trans,
		TextureConstPtr exteriorIorFact, TextureConstPtr interiorIorFact,
		TextureConstPtr B,
		const bool enableFineRoughGlass,
		TextureConstPtr u, TextureConstPtr v,
		TextureConstPtr filmThickness, TextureConstPtr filmIor) :
			Material(frontTransp, backTransp, emitted, bump), Kr(refl), Kt(trans),
			exteriorIor(exteriorIorFact), interiorIor(interiorIorFact), cauchyB(B),
			fineRoughGlass(enableFineRoughGlass),
			nu(u), nv(v), filmThickness(filmThickness), filmIor(filmIor) {
	glossiness = ComputeGlossiness(nu, nv);
}

Spectrum RoughGlassMaterial::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);

	const bool isKtBlack = kt.Black();
	const bool isKrBlack = kr.Black();
	if (isKtBlack && isKrBlack)
		return Spectrum();

	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float nt = RoughGlassMLGetInteriorIOR(hitPoint, interiorIor, cauchyB);
	const float ntc = nt / nc;

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;
	const float deltaWeight = fineRoughGlass ? RoughGlassMLNearDeltaWeight(u, v) : 0.f;
	const float roughWeight = 1.f - deltaWeight;

	// The exact delta lobe has no finite solid-angle Evaluate() contribution.
	if (roughWeight <= 0.f)
		return Spectrum();

	const float threshold = isKrBlack ? 1.f : (isKtBlack ? 0.f : .5f);
	if (localLightDir.z * localEyeDir.z < 0.f) {
		// Transmit

		const bool entering = (CosTheta(localLightDir) > 0.f);
		const float eta = entering ? (nc / nt) : ntc;

		Vector wh = eta * localLightDir + localEyeDir;
		if (wh.z < 0.f)
			wh = -wh;

		const float lengthSquared = wh.LengthSquared();
		if (!(lengthSquared > 0.f))
			return Spectrum();
		wh /= sqrtf(lengthSquared);
		const float cosThetaI = fabsf(CosTheta(localEyeDir));
		const float cosThetaIH = AbsDot(localEyeDir, wh);
		const float cosThetaOH = Dot(localLightDir, wh);

		const float D = SchlickDistribution_D(roughness, wh, anisotropy);
		const float G = SchlickDistribution_G(roughness, localLightDir, localEyeDir);
		const float directSpecPdf = (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
				RoughGlassMLVisibleNormalPdf(roughness,
						hitPoint.fromLight ? localLightDir : localEyeDir, wh) :
				SchlickDistribution_Pdf(roughness, wh, anisotropy);
		const float reverseSpecPdf = (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
				RoughGlassMLVisibleNormalPdf(roughness,
						hitPoint.fromLight ? localEyeDir : localLightDir, wh) :
				SchlickDistribution_Pdf(roughness, wh, anisotropy);
		const float F = FresnelTexture::CauchyEvaluate(ntc, cosThetaOH);

		if (directPdfW)
			*directPdfW = roughWeight * threshold * directSpecPdf * (hitPoint.fromLight ? fabsf(cosThetaIH) : (fabsf(cosThetaOH) * eta * eta)) / lengthSquared;

		if (reversePdfW)
			*reversePdfW = roughWeight * threshold * reverseSpecPdf * (hitPoint.fromLight ? (fabsf(cosThetaOH) * eta * eta) : fabsf(cosThetaIH)) / lengthSquared;

		const Spectrum result = roughWeight * (fabsf(cosThetaOH) * cosThetaIH * D *
			G / (cosThetaI * lengthSquared)) *
			kt * (1.f - F);

		*event = GLOSSY | TRANSMIT;

		return result;
	} else {
		// Reflect
		const float cosThetaO = fabsf(CosTheta(localLightDir));
		const float cosThetaI = fabsf(CosTheta(localEyeDir));
		if (cosThetaO == 0.f || cosThetaI == 0.f)
			return Spectrum();
		Vector wh = localLightDir + localEyeDir;
		if (wh == Vector(0.f))
			return Spectrum();
		wh = Normalize(wh);
		if (wh.z < 0.f)
			wh = -wh;

		float cosThetaH = Dot(localEyeDir, wh);
		const float D = SchlickDistribution_D(roughness, wh, anisotropy);
		const float G = SchlickDistribution_G(roughness, localLightDir, localEyeDir);
		const float directSpecPdf = (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
				RoughGlassMLVisibleNormalPdf(roughness,
						hitPoint.fromLight ? localLightDir : localEyeDir, wh) :
				SchlickDistribution_Pdf(roughness, wh, anisotropy);
		const float reverseSpecPdf = (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
				RoughGlassMLVisibleNormalPdf(roughness,
						hitPoint.fromLight ? localEyeDir : localLightDir, wh) :
				SchlickDistribution_Pdf(roughness, wh, anisotropy);
		const float F = FresnelTexture::CauchyEvaluate(ntc, cosThetaH);

		if (directPdfW)
			*directPdfW = roughWeight * (1.f - threshold) * directSpecPdf / (4.f * AbsDot(localLightDir, wh));

		if (reversePdfW)
			*reversePdfW = roughWeight * (1.f - threshold) * reverseSpecPdf / (4.f * AbsDot(localLightDir, wh));

		const Spectrum result = roughWeight * (D * G / (4.f * cosThetaI)) * kr * F;

		*event = GLOSSY | REFLECT;
		
		const float localFilmThickness = filmThickness ? filmThickness->GetFloatValue(hitPoint) : 0.f;
		if (localFilmThickness > 0.f) {
			const float localFilmIor = filmIor ? filmIor->GetFloatValue(hitPoint) : 1.f;
			return result * CalcFilmColor(localEyeDir, localFilmThickness, localFilmIor);
		}
		return result;
	}
}

Spectrum RoughGlassMaterial::Sample(const HitPoint &hitPoint,
		const Vector &localFixedDir, Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event) const {
	if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);

	const bool isKtBlack = kt.Black();
	const bool isKrBlack = kr.Black();
	if (isKtBlack && isKrBlack)
		return Spectrum();

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;
	const float deltaWeight = fineRoughGlass ? RoughGlassMLNearDeltaWeight(u, v) : 0.f;
	const float roughWeight = 1.f - deltaWeight;

	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float nt = RoughGlassMLGetInteriorIOR(hitPoint, interiorIor, cauchyB);
	const float ntc = nt / nc;

	// Smooth stochastic near-delta mixture. A delta sample uses exact Glass/HERO;
	// otherwise remap u0 to the RoughGlass/VNDF interval.
	if ((deltaWeight >= 1.f) || (u0 < deltaWeight)) {
		const float localFilmThickness =
				filmThickness ? filmThickness->GetFloatValue(hitPoint) : 0.f;
		const float localFilmIor =
				(localFilmThickness > 0.f && filmIor) ?
				filmIor->GetFloatValue(hitPoint) : 1.f;

		Spectrum deltaResult = RoughGlassMLSampleDelta(hitPoint,
				localFixedDir, localSampledDir, passThroughEvent,
				kr, kt, nc, nt, localFilmThickness, localFilmIor,
				pdfW, event);

		if (!deltaResult.Black()) {
			*pdfW *= Max(deltaWeight, 1e-9f);

			const float cauchyBValue =
					cauchyB ? cauchyB->GetFloatValue(hitPoint) : 0.f;
			if (mlHeroEnabled && (cauchyBValue > 0.f))
				mlDispersionUsed = true;
		}

		return deltaResult;
	}

	const float roughU0 = (roughWeight > 0.f) ?
			Clamp((u0 - deltaWeight) / roughWeight, 0.f, 1.f) : 0.f;

	Vector wh;
	float d, specPdf;
	if (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy))
		RoughGlassMLSampleVisibleNormal(roughness, localFixedDir,
				roughU0, u1, &wh, &d, &specPdf);
	else
		SchlickDistribution_SampleH(roughness, anisotropy,
				roughU0, u1, &wh, &d, &specPdf);
	if (wh.z < 0.f)
		wh = -wh;
	const float cosThetaOH = Dot(localFixedDir, wh);

	const float coso = fabsf(localFixedDir.z);

	// Decide to transmit or reflect
	float threshold;
	if (!isKrBlack) {
		if (!isKtBlack)
			threshold = .5f;
		else
			threshold = 0.f;
	} else {
		if (!isKtBlack)
			threshold = 1.f;
		else
			return Spectrum();
	}

	Spectrum result;
	if (passThroughEvent < threshold) {
		// Transmit

		const bool entering = (CosTheta(localFixedDir) > 0.f);
		const float eta = entering ? (nc / nt) : ntc;
		const float eta2 = eta * eta;
		const float sinThetaIH2 = eta2 * Max(0.f, 1.f - cosThetaOH * cosThetaOH);
		if (sinThetaIH2 >= 1.f)
			return Spectrum();
		float cosThetaIH = sqrtf(1.f - sinThetaIH2);
		if (entering)
			cosThetaIH = -cosThetaIH;
		const float length = eta * cosThetaOH + cosThetaIH;
		*localSampledDir = length * wh - eta * localFixedDir;

		const float lengthSquared = length * length;
		*pdfW = specPdf * fabsf(cosThetaIH) / lengthSquared;
		if (*pdfW <= 0.f)
			return Spectrum();

		const float cosi = fabsf(localSampledDir->z);

		const float G = SchlickDistribution_G(roughness, localFixedDir, *localSampledDir);
		float factor = (d / specPdf) * G * fabsf(cosThetaOH) / threshold;

		if (!hitPoint.fromLight) {
			const float F = FresnelTexture::CauchyEvaluate(ntc, cosThetaIH);
			result = (factor / coso) * kt * (1.f - F);
		} else {
			const float F = FresnelTexture::CauchyEvaluate(ntc, cosThetaOH);
			result = (factor / cosi) * kt * (1.f - F);
		}

		*pdfW *= threshold;
		*event = GLOSSY | TRANSMIT;
	} else {
		// Reflect
		*pdfW = specPdf / (4.f * fabsf(cosThetaOH));
		if (*pdfW <= 0.f)
			return Spectrum();

		*localSampledDir = 2.f * cosThetaOH * wh - localFixedDir;

		const float cosi = fabsf(localSampledDir->z);
		if ((cosi < DEFAULT_COS_EPSILON_STATIC) || (localFixedDir.z * localSampledDir->z < 0.f))
			return Spectrum();

		const float G = SchlickDistribution_G(roughness, localFixedDir, *localSampledDir);
		float factor = (d / specPdf) * G * fabsf(cosThetaOH) / (1.f - threshold);

		const float F = FresnelTexture::CauchyEvaluate(ntc, cosThetaOH);
		factor /= (!hitPoint.fromLight) ? coso : cosi;
		result = factor * F * kr;
		
		const float localFilmThickness = filmThickness ? filmThickness->GetFloatValue(hitPoint) : 0.f;
		if (localFilmThickness > 0.f) {
			const float localFilmIor = filmIor ? filmIor->GetFloatValue(hitPoint) : 1.f;
			result *= CalcFilmColor(localFixedDir, localFilmThickness, localFilmIor);
		}

		*pdfW *= (1.f - threshold);
		*event = GLOSSY | REFLECT;
	}

	// Account for the probability of selecting the continuous RoughGlass lobe.
	*pdfW *= Max(roughWeight, 1e-9f);

	// ML HERO: mark this path as dispersive only after RoughGlass has
	// successfully sampled an actual event. The existing Path/Bidir HERO
	// weighting then applies the wavelength RGB estimator exactly once.
	const float cauchyBValue = cauchyB ? cauchyB->GetFloatValue(hitPoint) : 0.f;
	if (mlHeroEnabled && (cauchyBValue > 0.f))
		mlDispersionUsed = true;

	return result;
}

void RoughGlassMaterial::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	if (directPdfW)
		*directPdfW = 0.f;
	if (reversePdfW)
		*reversePdfW = 0.f;

	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);

	const bool isKtBlack = kt.Black();
	const bool isKrBlack = kr.Black();
	if (isKtBlack && isKrBlack)
		return;

	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float nt = RoughGlassMLGetInteriorIOR(hitPoint, interiorIor, cauchyB);
	const float ntc = nt / nc;

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;
	const float deltaWeight = fineRoughGlass ? RoughGlassMLNearDeltaWeight(u, v) : 0.f;
	const float roughWeight = 1.f - deltaWeight;

	// The delta lobe has zero finite solid-angle Pdf().
	if (roughWeight <= 0.f)
		return;

	const float threshold = isKrBlack ? 1.f : (isKtBlack ? 0.f : .5f);
	if (localLightDir.z * localEyeDir.z < 0.f) {
		// Transmit

		const bool entering = (CosTheta(localLightDir) > 0.f);
		const float eta = entering ? (nc / nt) : ntc;

		Vector wh = eta * localLightDir + localEyeDir;
		if (wh.z < 0.f)
			wh = -wh;

		const float lengthSquared = wh.LengthSquared();
		if (!(lengthSquared > 0.f))
			return;

		wh /= sqrtf(lengthSquared);
		const float cosThetaIH = AbsDot(localEyeDir, wh);
		const float cosThetaOH = AbsDot(localLightDir, wh);

		const float directSpecPdf = (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
				RoughGlassMLVisibleNormalPdf(roughness,
						hitPoint.fromLight ? localLightDir : localEyeDir, wh) :
				SchlickDistribution_Pdf(roughness, wh, anisotropy);
		const float reverseSpecPdf = (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
				RoughGlassMLVisibleNormalPdf(roughness,
						hitPoint.fromLight ? localEyeDir : localLightDir, wh) :
				SchlickDistribution_Pdf(roughness, wh, anisotropy);

		if (directPdfW)
			*directPdfW = roughWeight * threshold * directSpecPdf *
					(hitPoint.fromLight ? cosThetaIH : (cosThetaOH * eta * eta)) /
					lengthSquared;

		if (reversePdfW)
			*reversePdfW = roughWeight * threshold * reverseSpecPdf *
					(hitPoint.fromLight ? (cosThetaOH * eta * eta) : cosThetaIH) /
					lengthSquared;
	} else {
		// Reflect
		const float cosThetaO = fabsf(CosTheta(localLightDir));
		const float cosThetaI = fabsf(CosTheta(localEyeDir));
		if (cosThetaO == 0.f || cosThetaI == 0.f)
			return;

		Vector wh = localLightDir + localEyeDir;
		if (wh == Vector(0.f))
			return;
		wh = Normalize(wh);
		if (wh.z < 0.f)
			wh = -wh;

		const float directSpecPdf = (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
				RoughGlassMLVisibleNormalPdf(roughness,
						hitPoint.fromLight ? localLightDir : localEyeDir, wh) :
				SchlickDistribution_Pdf(roughness, wh, anisotropy);
		const float reverseSpecPdf = (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
				RoughGlassMLVisibleNormalPdf(roughness,
						hitPoint.fromLight ? localEyeDir : localLightDir, wh) :
				SchlickDistribution_Pdf(roughness, wh, anisotropy);

		if (directPdfW)
			*directPdfW = roughWeight * (1.f - threshold) * directSpecPdf / (4.f * AbsDot(localLightDir, wh));

		if (reversePdfW)
			*reversePdfW = roughWeight * (1.f - threshold) * reverseSpecPdf / (4.f * AbsDot(localLightDir, wh));
	}
}

void RoughGlassMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	Kr->AddReferencedTextures(referencedTexs);
	Kt->AddReferencedTextures(referencedTexs);
	if (exteriorIor)
		exteriorIor->AddReferencedTextures(referencedTexs);
	if (interiorIor)
		interiorIor->AddReferencedTextures(referencedTexs);
	if (cauchyB)
		cauchyB->AddReferencedTextures(referencedTexs);
	nu->AddReferencedTextures(referencedTexs);
	nv->AddReferencedTextures(referencedTexs);
	if (filmThickness)
		filmThickness->AddReferencedTextures(referencedTexs);
	if (filmIor)
		filmIor->AddReferencedTextures(referencedTexs);
}

void RoughGlassMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	bool updateGlossiness = false;
	if (Kr == &oldTex)
		Kr = &newTex;
	if (Kt == &oldTex)
		Kt = &newTex;
	if (exteriorIor == &oldTex)
		exteriorIor = &newTex;
	if (interiorIor == &oldTex)
		interiorIor = &newTex;
	if (cauchyB == &oldTex)
		cauchyB = &newTex;
	if (nu == &oldTex) {
		nu = &newTex;
		updateGlossiness = true;
	}
	if (nv == &oldTex) {
		nv = &newTex;
		updateGlossiness = true;
	}
	if (filmThickness == &oldTex)
		filmThickness = &newTex;
	if (filmIor == &oldTex)
		filmIor = &newTex;

	if (updateGlossiness)
		glossiness = ComputeGlossiness(nu, nv);
}

PropertiesUPtr RoughGlassMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("roughglass"));
	props->Set(Property("scene.materials." + name + ".kr")(Kr->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".kt")(Kt->GetSDLValue()));
	if (exteriorIor)
		props->Set(Property("scene.materials." + name + ".exteriorior")(exteriorIor->GetSDLValue()));
	if (interiorIor)
		props->Set(Property("scene.materials." + name + ".interiorior")(interiorIor->GetSDLValue()));
	if (cauchyB)
		props->Set(Property("scene.materials." + name + ".cauchyb")(cauchyB->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".fineroughglass.enable")(fineRoughGlass));
	props->Set(Property("scene.materials." + name + ".uroughness")(nu->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".vroughness")(nv->GetSDLValue()));
	if (filmThickness)
		props->Set(Property("scene.materials." + name + ".filmthickness")(filmThickness->GetSDLValue()));
	if (filmIor)
		props->Set(Property("scene.materials." + name + ".filmior")(filmIor->GetSDLValue()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
