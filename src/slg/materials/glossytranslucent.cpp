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
#include "slg/materials/glossytranslucent.h"
#include "slg/engines/mlhero_glossytranslucent_pdf_fix15lk.h"
#include "slg/engines/mlhero_glossytranslucent_material_audit15ll.h"

using namespace std;
using namespace luxrays;
using namespace slg;

bool GetMLHeroEnabled();

static slg::mlhero15ll::RGB MLHeroGlossyTranslucentRGB15ll(
		const Spectrum &value) {
	slg::mlhero15ll::RGB result;
	result.r = value.c[0];
	result.g = value.c[1];
	result.b = value.c[2];
	return result;
}

static void MLHeroGlossyTranslucentEvaluation15ll(const HitPoint &hitPoint,
		const unsigned materialID, const bool reflection, const Vector &localLightDir,
		const Vector &localEyeDir, const float sideTest,
		const float roughness, const float anisotropy, const float materialIndex,
		const float materialDepth, const bool mbounce, const Vector &rawHalf,
		const Vector &half, const Spectrum &ks, const Spectrum &alpha,
		const Spectrum &base, const Spectrum &absorption,
		const Spectrum &fresnel, const Spectrum &coating,
		const Spectrum &result, const float *directPdfW,
		const float *reversePdfW) {
	if (!slg::mlhero15ll::Enabled())
		return;

	const bool isNonfinite = result.IsNaN() || result.IsInf();
	const float rawHalfLength = rawHalf.Length();
	const float cosI = fabsf(localLightDir.z);
	const float cosO = fabsf(localEyeDir.z);
	slg::mlhero15ll::Record record;
	record.side = hitPoint.fromLight ? 1u : 0u;
	record.material = materialID;
	record.mbounce = mbounce ? 1u : 0u;
	record.light = slg::mlhero15ll::V(localLightDir);
	record.eye = slg::mlhero15ll::V(localEyeDir);
	record.fixed = slg::mlhero15ll::V(hitPoint.fromLight ? localLightDir : localEyeDir);
	record.sampled = slg::mlhero15ll::V(hitPoint.fromLight ? localEyeDir : localLightDir);
	record.rawHalf = slg::mlhero15ll::V(rawHalf);
	record.half = slg::mlhero15ll::V(half);
	record.sideTest = sideTest;
	record.rawHalfLength = rawHalfLength;
	record.cosI = cosI;
	record.cosO = cosO;
	record.roughness = roughness;
	record.anisotropy = anisotropy;
	record.index = materialIndex;
	record.depth = materialDepth;
	record.fresnelCos = AbsDot(hitPoint.fromLight ? localEyeDir : localLightDir, half);
	record.directPdf = directPdfW ? *directPdfW : NAN;
	record.reversePdf = reversePdfW ? *reversePdfW : NAN;
	record.ks = MLHeroGlossyTranslucentRGB15ll(ks);
	record.alpha = MLHeroGlossyTranslucentRGB15ll(alpha);
	record.base = MLHeroGlossyTranslucentRGB15ll(base);
	record.absorption = MLHeroGlossyTranslucentRGB15ll(absorption);
	record.fresnel = MLHeroGlossyTranslucentRGB15ll(fresnel);
	record.coating = MLHeroGlossyTranslucentRGB15ll(coating);
	record.result = MLHeroGlossyTranslucentRGB15ll(result);
	if (isNonfinite && reflection && std::isfinite(half.x) && std::isfinite(half.y) &&
			std::isfinite(half.z)) {
		record.d = SchlickDistribution_D(roughness, half, anisotropy);
		record.g = SchlickDistribution_G(roughness, localLightDir, localEyeDir);
		record.cosFixedHalf = Dot(hitPoint.fromLight ? localLightDir : localEyeDir,
			half);
	}

	slg::mlhero15ll::ObserveEvaluation(reflection, isNonfinite,
		rawHalfLength < DEFAULT_EPSILON_STATIC,
		(cosI < DEFAULT_COS_EPSILON_STATIC) ||
			(cosO < DEFAULT_COS_EPSILON_STATIC), record);
}

// Phase 15lk: report the density of the actual Schlick inverse-CDF sampler.
// The double unit-direction pair carries forward the sharp-lobe precision
// hardening validated for Glossy2 and GlossyCoating.
static float MLHeroGlossyTranslucentCoatingPdf15lk(const float roughness,
		const float anisotropy, const Vector &localFixedDir,
		const Vector &localSampledDir) {
	const float nativePdf = SchlickBSDF_CoatingPdf(roughness, anisotropy,
			localFixedDir, localSampledDir);
	if (!GetMLHeroEnabled() || !(localFixedDir.z * localSampledDir.z > 0.f))
		return nativePdf;

	const slg::mlhero15kk::V fixed = slg::mlhero15kk::Unit(
			slg::mlhero15kk::V(localFixedDir));
	const slg::mlhero15kk::V sampled = slg::mlhero15kk::Unit(
			slg::mlhero15kk::V(localSampledDir));
	const slg::mlhero15kk::V wh = slg::mlhero15kk::Unit(fixed + sampled);
	const double jacobianDenominator = 4. * std::fabs(
			slg::mlhero15kk::Dot(fixed, wh));
	if (!(jacobianDenominator > 0.))
		return 0.f;

	const double halfPdf = slg::mlhero15lb::HalfPdf(roughness, anisotropy,
			wh.x, wh.y, wh.z);
	return float(halfPdf / jacobianDenominator);
}

// Phase 15lm: use one precise Schlick inverse map for both faces. This removes
// the float/native epsilon split observed on the front face and retains the
// 15lk back-face orientation repair. The same u0/u1 values are used.
static Spectrum MLHeroGlossyTranslucentCoatingSample15lm(
		const bool fromLight, const Spectrum &ks, const float roughness,
		const float anisotropy, const bool mbounce, const Vector &localFixedDir,
		Vector *localSampledDir, const float u0, const float u1,
		float *coatingPdf) {
	const slg::mlhero15kk::V fixed = slg::mlhero15kk::Unit(
			slg::mlhero15kk::V(localFixedDir));
	slg::mlhero15kk::V wh = slg::mlhero15ky::SampleHalf(
			roughness, anisotropy, u0, u1);
	if (fixed.z < 0.)
		wh = wh * -1.;
	const double cosWH = slg::mlhero15kk::Dot(fixed, wh);
	const slg::mlhero15kk::V sampled = slg::mlhero15kk::Unit(
			wh * (2. * cosWH) - fixed);
	*localSampledDir = Vector(float(sampled.x), float(sampled.y), float(sampled.z));

	if (!(cosWH > 0.) ||
			(fabsf(localSampledDir->z) < DEFAULT_COS_EPSILON_STATIC) ||
			!(localFixedDir.z * localSampledDir->z > 0.f))
		return Spectrum();

	*coatingPdf = MLHeroGlossyTranslucentCoatingPdf15lk(roughness,
			anisotropy, localFixedDir, *localSampledDir);
	if (!(*coatingPdf > 0.f))
		return Spectrum();

	return SchlickBSDF_CoatingF(fromLight, ks, roughness, anisotropy,
			mbounce, localFixedDir, *localSampledDir);
}

//------------------------------------------------------------------------------
// Glossy Translucent material
//
// LuxRender GlossyTranslucent material porting.
//------------------------------------------------------------------------------

GlossyTranslucentMaterial::GlossyTranslucentMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump,
		TextureConstPtr kd, TextureConstPtr kt, TextureConstPtr ks, TextureConstPtr ks2,
		TextureConstPtr u, TextureConstPtr u2, TextureConstPtr v, TextureConstPtr v2,
		TextureConstPtr ka, TextureConstPtr ka2, TextureConstPtr d, TextureConstPtr d2,
		TextureConstPtr i, TextureConstPtr i2, const bool mbounce, const bool mbounce2) :
			Material(frontTransp, backTransp, emitted, bump), Kd(kd), Kt(kt), Ks(ks), Ks_bf(ks2), nu(u), nu_bf(u2),
			nv(v), nv_bf(v2), Ka(ka), Ka_bf(ka2), depth(d), depth_bf(d2), index(i),
			index_bf(i2), multibounce(mbounce), multibounce_bf(mbounce2) {
	glossiness = Min(ComputeGlossiness(nu, nv), ComputeGlossiness(nu_bf, nv_bf));
}

Spectrum GlossyTranslucentMaterial::Albedo(const HitPoint &hitPoint) const {
	return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
}

Spectrum GlossyTranslucentMaterial::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
	const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;
	const float cosi = fabsf(localLightDir.z);
	const float coso = fabsf(localEyeDir.z);

	const Frame frame(hitPoint.GetFrame());
	const float sideTest = Dot(frame.ToWorld(localFixedDir), hitPoint.geometryN) * Dot(frame.ToWorld(localSampledDir), hitPoint.geometryN);
	// Phase 15lm: Sample() already excludes tangent directions. Apply the same
	// support rule to Evaluate() before Schlick or absorption can form 0 / 0.
	if ((cosi < DEFAULT_COS_EPSILON_STATIC) ||
			(coso < DEFAULT_COS_EPSILON_STATIC)) {
		if (sideTest > DEFAULT_COS_EPSILON_STATIC)
			*event = GLOSSY | REFLECT;
		else if (sideTest < -DEFAULT_COS_EPSILON_STATIC)
			*event = DIFFUSE | TRANSMIT;
		else
			*event = NONE;
		if (directPdfW)
			*directPdfW = 0.f;
		if (reversePdfW)
			*reversePdfW = 0.f;
		return Spectrum();
	}

	if (sideTest < -DEFAULT_COS_EPSILON_STATIC) {
		// Transmission
		*event = DIFFUSE | TRANSMIT;

		if (directPdfW)
			*directPdfW = fabsf(localSampledDir.z) * (INV_PI * .5f);
		if (reversePdfW)
			*reversePdfW = fabsf(localFixedDir.z) * (INV_PI * .5f);

		Vector H(localLightDir.x + localEyeDir.x, localLightDir.y + localEyeDir.y,
			localLightDir.z - localEyeDir.z);
		const float hLength = H.Length();
		Spectrum S1, S2;
		if (hLength < DEFAULT_EPSILON_STATIC) {
			// Match the established GlossyCoating degenerate-transmission policy:
			// an undefined half vector contributes no coating Fresnel attenuation.
			S1 = Spectrum(0.f);
			S2 = Spectrum(0.f);
		} else {
			H /= hLength;
			const float fresnelCos = AbsDot(localLightDir, H);
			Spectrum ks = Ks->GetSpectrumValue(hitPoint);
			float i = index->GetFloatValue(hitPoint);
			if (i > 0.f) {
				const float ti = (i - 1.f) / (i + 1.f);
				ks *= ti * ti;
			}
			ks = ks.Clamp(0.f, 1.f);
			S1 = FresnelTexture::SchlickEvaluate(ks, fresnelCos);

			ks = Ks_bf->GetSpectrumValue(hitPoint);
			i = index_bf->GetFloatValue(hitPoint);
			if (i > 0.f) {
				const float ti = (i - 1.f) / (i + 1.f);
				ks *= ti * ti;
			}
			ks = ks.Clamp(0.f, 1.f);
			S2 = FresnelTexture::SchlickEvaluate(ks, fresnelCos);
		}
		Spectrum S(Sqrt((Spectrum(1.f) - S1) * (Spectrum(1.f) - S2)));
		Spectrum transmissionAbsorption;
		if (localLightDir.z > 0.f) {
			transmissionAbsorption = Exp(Ka->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * -(depth->GetFloatValue(hitPoint) / cosi) +
				Ka_bf->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * -(depth_bf->GetFloatValue(hitPoint) / coso));
		} else {
			transmissionAbsorption = Exp(Ka->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * -(depth->GetFloatValue(hitPoint) / coso) +
				Ka_bf->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * -(depth_bf->GetFloatValue(hitPoint) / cosi));
		}
		S *= transmissionAbsorption;

		const Spectrum transmissionKt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
		const Spectrum transmissionOneMinusKd = Spectrum(1.f) -
			Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
		const Spectrum result = (INV_PI * cosi) * S * transmissionKt *
			transmissionOneMinusKd;
		const Spectrum transmissionBase = transmissionKt * transmissionOneMinusKd;
		MLHeroGlossyTranslucentEvaluation15ll(hitPoint, GetID(), false,
			localLightDir, localEyeDir, sideTest, 0.f, 0.f, 0.f, 0.f, false,
			Vector(localLightDir.x + localEyeDir.x,
				localLightDir.y + localEyeDir.y,
				localLightDir.z - localEyeDir.z), H, S1, S2,
			transmissionBase, transmissionAbsorption, S, Spectrum(), result,
			directPdfW, reversePdfW);
		return result;
	} else if (sideTest > DEFAULT_COS_EPSILON_STATIC) {
		// Reflection
		*event = GLOSSY | REFLECT;

		const Spectrum baseF = Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * INV_PI * cosi;
		Spectrum ks, alpha;
		float i, u, v, d;
		bool mbounce;
		if (localEyeDir.z >= 0.f) {
			ks = Ks->GetSpectrumValue(hitPoint);
			i = index->GetFloatValue(hitPoint);
			u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
			v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
			alpha = Ka->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
			d = depth->GetFloatValue(hitPoint);
			mbounce = multibounce;
		} else {
			ks = Ks_bf->GetSpectrumValue(hitPoint);
			i = index_bf->GetFloatValue(hitPoint);
			u = Clamp(nu_bf->GetFloatValue(hitPoint), 1e-9f, 1.f);
			v = Clamp(nv_bf->GetFloatValue(hitPoint), 1e-9f, 1.f);
			alpha = Ka_bf->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
			d = depth_bf->GetFloatValue(hitPoint);
			mbounce = multibounce_bf;
		}

		if (i > 0.f) {
			const float ti = (i - 1.f) / (i + 1.f);
			ks *= ti * ti;
		}
		ks = ks.Clamp(0.f, 1.f);

		const float u2 = u * u;
		const float v2 = v * v;
		const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
		const float roughness = u * v;

		if (directPdfW) {
			const float wCoating = SchlickBSDF_CoatingWeight(ks, localFixedDir);
			const float wBase = 1.f - wCoating;

			*directPdfW = .5f * (wBase * fabsf(localSampledDir.z * INV_PI) +
				wCoating * MLHeroGlossyTranslucentCoatingPdf15lk(roughness, anisotropy, localFixedDir, localSampledDir));
		}

		if (reversePdfW) {
			const float wCoatingR = SchlickBSDF_CoatingWeight(ks, localSampledDir);
			const float wBaseR = 1.f - wCoatingR;

			*reversePdfW = .5f * (wBaseR * fabsf(localFixedDir.z * INV_PI) +
				wCoatingR * MLHeroGlossyTranslucentCoatingPdf15lk(roughness, anisotropy, localSampledDir, localFixedDir));
		}

		// Absorption
		const Spectrum absorption = CoatingAbsorption(cosi, coso, alpha, d);

		// Coating fresnel factor
		const Vector rawH(localFixedDir + localSampledDir);
		const Vector H(Normalize(rawH));
		const Spectrum S = FresnelTexture::SchlickEvaluate(ks, AbsDot(localSampledDir, H));

		const Spectrum coatingF = SchlickBSDF_CoatingF(hitPoint.fromLight, ks, roughness, anisotropy, mbounce,
			localFixedDir, localSampledDir);

		// Blend in base layer Schlick style
		// assumes coating bxdf takes fresnel factor S into account

		const Spectrum result = coatingF + absorption * (Spectrum(1.f) - S) * baseF;
		MLHeroGlossyTranslucentEvaluation15ll(hitPoint, GetID(), true,
			localLightDir, localEyeDir, sideTest, roughness, anisotropy, i, d,
			mbounce, rawH, H, ks, alpha, baseF, absorption, S, coatingF,
			result, directPdfW, reversePdfW);
		return result;
	} else
		return Spectrum();
}

Spectrum GlossyTranslucentMaterial::Sample(const HitPoint &hitPoint,
	const Vector &localFixedDir, Vector *localSampledDir,
	const float u0, const float u1, const float passThroughEvent,
	float *pdfW, BSDFEvent *event) const {
	if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	if (passThroughEvent < .5f) {
		// Reflection
		Spectrum ks, alpha;
		float i, u, v, d;
		bool mbounce;
		if (localFixedDir.z >= 0.f) {
			ks = Ks->GetSpectrumValue(hitPoint);
			i = index->GetFloatValue(hitPoint);
			u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
			v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
			alpha = Ka->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
			d = depth->GetFloatValue(hitPoint);
			mbounce = multibounce;
		} else {
			ks = Ks_bf->GetSpectrumValue(hitPoint);
			i = index_bf->GetFloatValue(hitPoint);
			u = Clamp(nu_bf->GetFloatValue(hitPoint), 1e-9f, 1.f);
			v = Clamp(nv_bf->GetFloatValue(hitPoint), 1e-9f, 1.f);
			alpha = Ka_bf->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
			d = depth_bf->GetFloatValue(hitPoint);
			mbounce = multibounce_bf;
		}
		if (i > 0.f) {
			const float ti = (i - 1.f) / (i + 1.f);
			ks *= ti * ti;
		}
		ks = ks.Clamp(0.f, 1.f);

		const float u2 = u * u;
		const float v2 = v * v;
		const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
		const float roughness = u * v;

		const float wCoating = SchlickBSDF_CoatingWeight(ks, localFixedDir);
		const float wBase = 1.f - wCoating;
		const float branchU15lk = 2.f * passThroughEvent;
		const bool sampledBase15lk = branchU15lk < wBase;
		slg::mlhero15lk::Attempt audit15lk(hitPoint.fromLight, GetID(), sampledBase15lk,
			roughness, anisotropy, u0, u1, branchU15lk, wBase, ks.Filter(),
			slg::mlhero15kk::V(localFixedDir));

		float basePdf = 0.f, coatingPdf = 0.f;
		Spectrum baseF, coatingF;

		if (2.f * passThroughEvent < wBase) {
			// Sample base BSDF (Matte BSDF)
			*localSampledDir = Sgn(localFixedDir.z) * CosineSampleHemisphere(u0, u1, &basePdf);

			const float absCosSampledDir = fabsf(localSampledDir->z);
			if (absCosSampledDir < DEFAULT_COS_EPSILON_STATIC) {
				audit15lk.Reject(slg::mlhero15lk::BASE_REJECT);
				return Spectrum();
			}

			baseF = Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * INV_PI * fabsf(hitPoint.fromLight ? localFixedDir.z : absCosSampledDir);

			// Evaluate coating BSDF (Schlick BSDF)
			coatingF = SchlickBSDF_CoatingF(hitPoint.fromLight, ks, roughness, anisotropy, mbounce,
				localFixedDir, *localSampledDir);
			coatingPdf = MLHeroGlossyTranslucentCoatingPdf15lk(roughness, anisotropy,
				localFixedDir, *localSampledDir);
		} else {
			// Sample coating BSDF (Schlick BSDF)
			if (GetMLHeroEnabled()) {
				coatingF = MLHeroGlossyTranslucentCoatingSample15lm(
					hitPoint.fromLight, ks, roughness, anisotropy, mbounce,
					localFixedDir, localSampledDir, u0, u1, &coatingPdf);
			} else {
				float nativeCoatingPdf15lm = 0.f;
				coatingF = SchlickBSDF_CoatingSampleF(hitPoint.fromLight, ks,
					roughness, anisotropy, mbounce, localFixedDir,
					localSampledDir, u0, u1, &nativeCoatingPdf15lm);
				if (!coatingF.Black()) {
					// HERO-off remains exactly on the native sampler and density.
					coatingF *= nativeCoatingPdf15lm;
					coatingPdf = MLHeroGlossyTranslucentCoatingPdf15lk(
						roughness, anisotropy, localFixedDir, *localSampledDir);
				}
			}
			if (coatingF.Black()) {
				audit15lk.Reject(slg::mlhero15lk::COATING_REJECT);
				return Spectrum();
			}

			const float absCosSampledDir = fabsf(localSampledDir->z);
			if (absCosSampledDir < DEFAULT_COS_EPSILON_STATIC) {
				audit15lk.Reject(slg::mlhero15lk::COATING_REJECT);
				return Spectrum();
			}

			// Evaluate base BSDF (Matte BSDF)
			basePdf = absCosSampledDir * INV_PI;
			baseF = Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * INV_PI * fabsf(hitPoint.fromLight ? localFixedDir.z : absCosSampledDir);
		}
		*event = GLOSSY | REFLECT;

		const Frame frame(hitPoint.GetFrame());
		const float sideTest = Dot(frame.ToWorld(localFixedDir), hitPoint.geometryN) * Dot(frame.ToWorld(*localSampledDir), hitPoint.geometryN);
		if (!(sideTest > DEFAULT_COS_EPSILON_STATIC)) {
			audit15lk.Reject(slg::mlhero15lk::GEOMETRY_REJECT);
			return Spectrum();
		}

		*pdfW = .5f * (coatingPdf * wCoating + basePdf * wBase);
		if (audit15lk.Active()) {
			const float wCoatingReverse15lk = SchlickBSDF_CoatingWeight(ks, *localSampledDir);
			audit15lk.Accept(slg::mlhero15kk::V(*localSampledDir), *pdfW, coatingPdf,
				1.f - wCoatingReverse15lk);
		}

		// Absorption
		const float cosi = fabsf(localSampledDir->z);
		const float coso = fabsf(localFixedDir.z);
		const Spectrum absorption = CoatingAbsorption(cosi, coso, alpha, d);

		// Coating fresnel factor
		const Vector H(Normalize(localFixedDir + *localSampledDir));
		const Spectrum S = FresnelTexture::SchlickEvaluate(ks, AbsDot(*localSampledDir, H));

		// Blend in base layer Schlick style
		// coatingF already takes fresnel factor S into account

		return (coatingF + absorption * (Spectrum(1.f) - S) * baseF) / *pdfW;
	} else {
		// Transmission
		*localSampledDir = -Sgn(localFixedDir.z) * CosineSampleHemisphere(u0, u1, pdfW);

		const Frame frame(hitPoint.dpdu, hitPoint.dpdv, hitPoint.shadeN);
		const float sideTest = Dot(frame.ToWorld(localFixedDir), hitPoint.geometryN) * Dot(frame.ToWorld(*localSampledDir), hitPoint.geometryN);
		if (!(sideTest < -DEFAULT_COS_EPSILON_STATIC))
			return Spectrum();

		*pdfW *= .5f;

		const float absCosSampledDir = fabsf(localSampledDir->z);
		if (absCosSampledDir < DEFAULT_COS_EPSILON_STATIC)
			return Spectrum();

		// pdfW and the pdf computed inside Evaluate() are exactly the same so
		// it doesn't matter if it is overwritten
		return Evaluate(hitPoint,
				hitPoint.fromLight ? localFixedDir : *localSampledDir,
				hitPoint.fromLight ? *localSampledDir : localFixedDir,
				event, pdfW, NULL) / *pdfW;
	}
}

void GlossyTranslucentMaterial::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
	const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;
	if ((fabsf(localLightDir.z) < DEFAULT_COS_EPSILON_STATIC) ||
			(fabsf(localEyeDir.z) < DEFAULT_COS_EPSILON_STATIC)) {
		if (directPdfW)
			*directPdfW = 0.f;
		if (reversePdfW)
			*reversePdfW = 0.f;
		return;
	}

	if (localFixedDir.z * localSampledDir.z < 0.f) {
		// Transmission
		if (directPdfW)
			*directPdfW = fabsf(localSampledDir.z) * (INV_PI * .5f);
		if (reversePdfW)
			*reversePdfW = fabsf(localFixedDir.z) * (INV_PI * .5f);
	} else if (localFixedDir.z * localSampledDir.z > 0.f) {
		// Reflection
		Spectrum ks;
		float i, u, v;
		if (localEyeDir.z >= 0.f) {
			ks = Ks->GetSpectrumValue(hitPoint);
			i = index->GetFloatValue(hitPoint);
			u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
			v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
		} else {
			ks = Ks_bf->GetSpectrumValue(hitPoint);
			i = index_bf->GetFloatValue(hitPoint);
			u = Clamp(nu_bf->GetFloatValue(hitPoint), 1e-9f, 1.f);
			v = Clamp(nv_bf->GetFloatValue(hitPoint), 1e-9f, 1.f);
		}

		if (i > 0.f) {
			const float ti = (i - 1.f) / (i + 1.f);
			ks *= ti * ti;
		}
		ks = ks.Clamp(0.f, 1.f);

		const float u2 = u * u;
		const float v2 = v * v;
		const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
		const float roughness = u * v;

		if (directPdfW) {
			const float wCoating = SchlickBSDF_CoatingWeight(ks, localFixedDir);
			const float wBase = 1.f - wCoating;

			*directPdfW = .5f * (wBase * fabsf(localSampledDir.z * INV_PI) +
				wCoating * MLHeroGlossyTranslucentCoatingPdf15lk(roughness, anisotropy, localFixedDir, localSampledDir));
		}

		if (reversePdfW) {
			const float wCoatingR = SchlickBSDF_CoatingWeight(ks, localSampledDir);
			const float wBaseR = 1.f - wCoatingR;

			*reversePdfW = .5f * (wBaseR * fabsf(localFixedDir.z * INV_PI) +
				wCoatingR * MLHeroGlossyTranslucentCoatingPdf15lk(roughness, anisotropy, localSampledDir, localFixedDir));
		}
	} else {
		if (directPdfW)
			*directPdfW = 0.f;
		if (reversePdfW)
			*reversePdfW = 0.f;
	}
}

void GlossyTranslucentMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	Kd->AddReferencedTextures(referencedTexs);
	Kt->AddReferencedTextures(referencedTexs);
	Ks->AddReferencedTextures(referencedTexs);
	Ks_bf->AddReferencedTextures(referencedTexs);
	nu->AddReferencedTextures(referencedTexs);
	nu_bf->AddReferencedTextures(referencedTexs);
	nv->AddReferencedTextures(referencedTexs);
	nv_bf->AddReferencedTextures(referencedTexs);
	Ka->AddReferencedTextures(referencedTexs);
	Ka_bf->AddReferencedTextures(referencedTexs);
	depth->AddReferencedTextures(referencedTexs);
	depth_bf->AddReferencedTextures(referencedTexs);
	index->AddReferencedTextures(referencedTexs);
	index_bf->AddReferencedTextures(referencedTexs);
}

void GlossyTranslucentMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	bool updateGlossiness = false;
	if (Kd == &oldTex)
		Kd = &newTex;
	if (Kt == &oldTex)
		Kt = &newTex;
	if (Ks == &oldTex)
		Ks = &newTex;
	if (Ks_bf == &oldTex)
		Ks_bf = &newTex;
	if (nu == &oldTex) {
		nu = &newTex;
		updateGlossiness = true;
	}
	if (nu_bf == &oldTex) {
		nu_bf = &newTex;
		updateGlossiness = true;
	}
	if (nv == &oldTex) {
		nv = &newTex;
		updateGlossiness = true;
	}
	if (nv_bf == &oldTex) {
		nv_bf = &newTex;
		updateGlossiness = true;
	}
	if (Ka == &oldTex)
		Ka = &newTex;
	if (Ka_bf == &oldTex)
		Ka_bf = &newTex;
	if (depth == &oldTex)
		depth = &newTex;
	if (depth_bf == &oldTex)
		depth_bf = &newTex;
	if (index == &oldTex)
		index = &newTex;
	if (index_bf == &oldTex)
		index_bf = &newTex;

	if (updateGlossiness)
		glossiness = Min(ComputeGlossiness(nu, nv), ComputeGlossiness(nu_bf, nv_bf));
}

PropertiesUPtr GlossyTranslucentMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("glossytranslucent"));
	props->Set(Property("scene.materials." + name + ".kd")(Kd->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".kt")(Kt->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ks")(Ks->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ks_bf")(Ks_bf->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".uroughness")(nu->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".uroughness_bf")(nu_bf->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".vroughness")(nv->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".vroughness_bf")(nv_bf->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ka")(Ka->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ka_bf")(Ka_bf->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".d")(depth->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".d_bf")(depth_bf->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".index")(index->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".index_bf")(index_bf->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".multibounce")(multibounce));
	props->Set(Property("scene.materials." + name + ".multibounce_bf")(multibounce_bf));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
