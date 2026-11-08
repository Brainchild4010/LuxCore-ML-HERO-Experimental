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
#include "slg/materials/glossy2.h"
#include "slg/engines/mlhero_glossy2_pdf_precision15lg.h"
#include "slg/engines/mlhero_glossy2_spectral15nj.h"

using namespace std;
using namespace luxrays;
using namespace slg;

bool GetMLHeroEnabled();

// Phase 15lg: use one double-precision direction pair for half-vector density
// and reflection Jacobian.  This removes rare sharp-lobe float reconstruction
// outliers while preserving the 15lf density correction and HERO-off behavior.
static float MLHeroGlossy2CoatingPdf15lg(const float roughness,
		const float anisotropy, const Vector &localFixedDir,
		const Vector &localSampledDir) {
	if (!GetMLHeroEnabled())
		return SchlickBSDF_CoatingPdf(roughness, anisotropy,
				localFixedDir, localSampledDir);

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

//------------------------------------------------------------------------------
// Glossy2 material
//
// LuxRender Glossy2 material porting.
//------------------------------------------------------------------------------

Glossy2Material::Glossy2Material(TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump,
		TextureConstPtr kd, TextureConstPtr ks, TextureConstPtr u, TextureConstPtr v,
		TextureConstPtr ka, TextureConstPtr d, TextureConstPtr i, const bool mbounce, const bool doublesided) :
			Material(frontTransp, backTransp, emitted, bump), Kd(kd), Ks(ks), nu(u), nv(v),
			Ka(ka), depth(d), index(i), multibounce(mbounce), doublesided (doublesided) {
	glossiness = ComputeGlossiness(nu, nv);
}

Spectrum Glossy2Material::Albedo(const HitPoint &hitPoint) const {
	return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
}

Spectrum Glossy2Material::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
	const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;

	const Spectrum baseF = Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * INV_PI * fabsf(localLightDir.z);

	if ((!doublesided) && (localEyeDir.z <= 0.f)) {
		// Back face: no coating

		if (directPdfW)
			*directPdfW = fabsf (localSampledDir.z * INV_PI);

		if (reversePdfW)
			*reversePdfW = fabsf (localFixedDir.z * INV_PI);

		*event = DIFFUSE | REFLECT;
		return baseF;
	}

	// Front face: coating+base
	*event = GLOSSY | REFLECT;

	Spectrum ks = Ks->GetSpectrumValue(hitPoint);
	const float i = index->GetFloatValue(hitPoint);
	if (i > 0.f) {
		const float ti = (i - 1.f) / (i + 1.f);
		ks *= ti * ti;
	}
	ks = ks.Clamp(0.f, 1.f);

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;

	if (directPdfW) {
		if ((!doublesided) && (localFixedDir.z < 0.f)) {
			// Backface
			*directPdfW = fabsf (localSampledDir.z * INV_PI);
		}
		else {
			const float wCoating = SchlickBSDF_CoatingWeight (ks, localFixedDir);
			const float wBase = 1.f - wCoating;

			*directPdfW = wBase * fabsf (localSampledDir.z * INV_PI) +
				wCoating * MLHeroGlossy2CoatingPdf15lg(roughness, anisotropy, localFixedDir, localSampledDir);

		}
	}

	if (reversePdfW) {
		if ((!doublesided) && (localSampledDir.z < 0.f)) {
			// Backface
			*reversePdfW = fabsf (localFixedDir.z * INV_PI);
		}
		else {
			const float wCoatingR = SchlickBSDF_CoatingWeight (ks, localSampledDir);
			const float wBaseR = 1.f - wCoatingR;

			*reversePdfW = wBaseR * fabsf (localFixedDir.z * INV_PI) +
				wCoatingR * MLHeroGlossy2CoatingPdf15lg(roughness, anisotropy, localSampledDir, localFixedDir);
		}
	}
	if ((!doublesided) && (localFixedDir.z < 0.f)) {
		// Backface, no coating
		return baseF;
	}


	// Absorption
	const float cosi = fabsf(localSampledDir.z);
	const float coso = fabsf(localFixedDir.z);
	const Spectrum alpha = Ka->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float d = depth->GetFloatValue(hitPoint);
	const Spectrum absorption = CoatingAbsorption(cosi, coso, alpha, d);

	// Coating fresnel factor
	const Vector H(Normalize(localFixedDir + localSampledDir));
	const Spectrum S = FresnelTexture::SchlickEvaluate(ks, AbsDot(localSampledDir, H));

	const Spectrum coatingF = SchlickBSDF_CoatingF(hitPoint.fromLight, ks, roughness, anisotropy, multibounce, localFixedDir, localSampledDir);

	// Blend in base layer Schlick style
	// assumes coating bxdf takes fresnel factor S into account

	return coatingF + absorption * (Spectrum(1.f) - S) * baseF;
}

Spectrum Glossy2Material::Sample(const HitPoint &hitPoint,
	const Vector &localFixedDir, Vector *localSampledDir,
	const float u0, const float u1, const float passThroughEvent,
	float *pdfW, BSDFEvent *event) const {
	if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	if ((!doublesided) && (localFixedDir.z <= 0.f)) {
		// Back face
		*localSampledDir = -CosineSampleHemisphere (u0, u1, pdfW);

		const float absCosSampledDir = fabsf (localSampledDir->z);
		if (absCosSampledDir < DEFAULT_COS_EPSILON_STATIC)
			return Spectrum ();
		*event = DIFFUSE | REFLECT;
		if (hitPoint.fromLight)
			return Kd->GetSpectrumValue (hitPoint) * fabsf (localFixedDir.z / absCosSampledDir);
		else
			return Kd->GetSpectrumValue (hitPoint);
	}

	Spectrum ks = Ks->GetSpectrumValue(hitPoint);
	const float i = index->GetFloatValue(hitPoint);
	if (i > 0.f) {
		const float ti = (i - 1.f) / (i + 1.f);
		ks *= ti * ti;
	}
	ks = ks.Clamp(0.f, 1.f);

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;

	// Coating is used only on the front face
	const float wCoating = SchlickBSDF_CoatingWeight(ks, localFixedDir);
	const float wBase = 1.f - wCoating;
	const bool sampledBase15lg = passThroughEvent < wBase;
	slg::mlhero15lg::Attempt audit15lg(hitPoint.fromLight, GetID(), sampledBase15lg,
		doublesided, roughness, anisotropy, u0, u1, passThroughEvent, wBase,
		ks.c[0], ks.c[1], ks.c[2],
		slg::mlhero15kk::V(localFixedDir));

	float basePdf, coatingPdf;
	Spectrum baseF, coatingF;

	if (passThroughEvent < wBase) {
		// Sample base BSDF (Matte BSDF)
		*localSampledDir = Sgn(localFixedDir.z) * CosineSampleHemisphere(u0, u1, &basePdf);

		const float absCosSampledDir = fabsf(localSampledDir->z);
		if (absCosSampledDir < DEFAULT_COS_EPSILON_STATIC) {
			audit15lg.Reject(slg::mlhero15lg::BASE_REJECT);
			return Spectrum();
		}

		baseF = Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * INV_PI * fabsf(hitPoint.fromLight ? localFixedDir.z : absCosSampledDir);

		// Evaluate coating BSDF (Schlick BSDF)
		coatingF = SchlickBSDF_CoatingF(hitPoint.fromLight, ks, roughness, anisotropy, multibounce, localFixedDir, *localSampledDir);
		coatingPdf = MLHeroGlossy2CoatingPdf15lg(roughness, anisotropy, localFixedDir, *localSampledDir);
	} else {
		// Sample coating BSDF (Schlick BSDF)
		float nativeCoatingPdf15lg;
		coatingF = SchlickBSDF_CoatingSampleF(hitPoint.fromLight, ks, roughness, anisotropy, multibounce,
				localFixedDir, localSampledDir, u0, u1, &nativeCoatingPdf15lg);
		if (coatingF.Black()) {
			audit15lg.Reject(slg::mlhero15lg::COATING_REJECT);
			return Spectrum();
		}

		const float absCosSampledDir = fabsf(localSampledDir->z);
		if (absCosSampledDir < DEFAULT_COS_EPSILON_STATIC) {
			audit15lg.Reject(slg::mlhero15lg::COATING_REJECT);
			return Spectrum();
		}

		// Reconstruct the unchanged coating BSDF with the native density first;
		// only the estimator probability is replaced by the generated density.
		coatingF *= nativeCoatingPdf15lg;
		coatingPdf = GetMLHeroEnabled() ?
			MLHeroGlossy2CoatingPdf15lg(roughness, anisotropy,
					localFixedDir, *localSampledDir) : nativeCoatingPdf15lg;

		// Evaluate base BSDF (Matte BSDF)
		basePdf = absCosSampledDir * INV_PI;
		baseF = Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * INV_PI * fabsf(hitPoint.fromLight ? localFixedDir.z : absCosSampledDir);
	}

	*event = GLOSSY | REFLECT;

	*pdfW = coatingPdf * wCoating + basePdf * wBase;
	if (audit15lg.Active()) {
		const float wCoatingReverse15lg = SchlickBSDF_CoatingWeight(ks, *localSampledDir);
		audit15lg.Accept(slg::mlhero15kk::V(*localSampledDir), *pdfW, coatingPdf,
			1.f - wCoatingReverse15lg);
	}

	// Absorption
	const float cosi = fabsf(localSampledDir->z);
	const float coso = fabsf(localFixedDir.z);
	const Spectrum alpha = Ka->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float d = depth->GetFloatValue(hitPoint);
	const Spectrum absorption = CoatingAbsorption(cosi, coso, alpha, d);

	// Coating fresnel factor
	const Vector H(Normalize(localFixedDir + *localSampledDir));
	const Spectrum S = FresnelTexture::SchlickEvaluate(ks, AbsDot(*localSampledDir, H));

	// Blend in base layer Schlick style
	// coatingF already takes fresnel factor S into account

	return (coatingF + absorption * (Spectrum(1.f) - S) * baseF) / *pdfW;
}


bool Glossy2Material::EvaluateMLHeroDebugSampleComponents(const HitPoint &hitPoint,
		const Vector &localFixedDir, const Vector &localSampledDir,
		const float pdfW,
		Spectrum *kdValue, Spectrum *ksValue,
		Spectrum *schlickS, Spectrum *absorptionValue,
		Spectrum *baseContribution, Spectrum *coatingContribution,
		Spectrum *combinedResult) const {
	if (!kdValue || !ksValue || !schlickS || !absorptionValue ||
			!baseContribution || !coatingContribution || !combinedResult ||
			(pdfW <= 0.f))
		return false;

	// Phase 15ac is aimed at the front-face Glossy2 path.
	if ((!doublesided) && (localFixedDir.z <= 0.f))
		return false;

	const Spectrum kd = Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);

	Spectrum ks = Ks->GetSpectrumValue(hitPoint);
	const float i = index->GetFloatValue(hitPoint);
	if (i > 0.f) {
		const float ti = (i - 1.f) / (i + 1.f);
		ks *= ti * ti;
	}
	ks = ks.Clamp(0.f, 1.f);

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) :
		u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;

	const float absCosSampledDir = fabsf(localSampledDir.z);
	const Spectrum baseF = kd * INV_PI *
		fabsf(hitPoint.fromLight ? localFixedDir.z : absCosSampledDir);

	const float cosi = fabsf(localSampledDir.z);
	const float coso = fabsf(localFixedDir.z);
	const Spectrum alpha = Ka->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float d = depth->GetFloatValue(hitPoint);
	const Spectrum absorption = CoatingAbsorption(cosi, coso, alpha, d);

	const Vector H(Normalize(localFixedDir + localSampledDir));
	const Spectrum S = FresnelTexture::SchlickEvaluate(
		ks, AbsDot(localSampledDir, H));

	const Spectrum coatingF = SchlickBSDF_CoatingF(
		hitPoint.fromLight, ks, roughness, anisotropy, multibounce,
		localFixedDir, localSampledDir);

	const Spectrum coating = coatingF / pdfW;
	const Spectrum base = absorption * (Spectrum(1.f) - S) * baseF / pdfW;

	*kdValue = kd;
	*ksValue = ks;
	*schlickS = S;
	*absorptionValue = absorption;
	*baseContribution = base;
	*coatingContribution = coating;
	*combinedResult = coating + base;

	return true;
}

bool Glossy2Material::EvaluateMLHeroEvaluateComponents15nd(
		const HitPoint &hitPoint, const Vector &localLightDir,
		const Vector &localEyeDir, BSDFEvent *event, Spectrum *kdValue,
		Spectrum *baseContribution, Spectrum *coatingContribution,
		Spectrum *combinedResult) const {
	if (!event || !kdValue || !baseContribution || !coatingContribution ||
			!combinedResult)
		return false;

	const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
	const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;
	const Spectrum kd = Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum baseF = kd * INV_PI * fabsf(localLightDir.z);

	if ((!doublesided) && (localEyeDir.z <= 0.f)) {
		*event = DIFFUSE | REFLECT;
		*kdValue = kd;
		*baseContribution = baseF;
		*coatingContribution = Spectrum(0.f);
		*combinedResult = baseF;
		return true;
	}

	*event = GLOSSY | REFLECT;
	if ((!doublesided) && (localFixedDir.z < 0.f)) {
		*kdValue = kd;
		*baseContribution = baseF;
		*coatingContribution = Spectrum(0.f);
		*combinedResult = baseF;
		return true;
	}

	Spectrum ks = Ks->GetSpectrumValue(hitPoint);
	const float i = index->GetFloatValue(hitPoint);
	if (i > 0.f) {
		const float ti = (i - 1.f) / (i + 1.f);
		ks *= ti * ti;
	}
	ks = ks.Clamp(0.f, 1.f);

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) :
		u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;

	const float cosi = fabsf(localSampledDir.z);
	const float coso = fabsf(localFixedDir.z);
	const Spectrum alpha = Ka->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float d = depth->GetFloatValue(hitPoint);
	const Spectrum absorption = CoatingAbsorption(cosi, coso, alpha, d);
	const Vector H(Normalize(localFixedDir + localSampledDir));
	const Spectrum S = FresnelTexture::SchlickEvaluate(
		ks, AbsDot(localSampledDir, H));
	const Spectrum coatingF = SchlickBSDF_CoatingF(hitPoint.fromLight,
		ks, roughness, anisotropy, multibounce, localFixedDir,
		localSampledDir);
	const Spectrum base = absorption * (Spectrum(1.f) - S) * baseF;

	*kdValue = kd;
	*baseContribution = base;
	*coatingContribution = coatingF;
	*combinedResult = coatingF + base;
	return true;
}

bool Glossy2Material::EvaluateMLHeroAtWaveLength15nj(
		const HitPoint &hitPoint, const Vector &localLightDir,
		const Vector &localEyeDir, const float waveLength,
		const bool basisCompensation, BSDFEvent *event, float *value) const {
	if (!event || !value)
		return false;
	MLHeroGlossy2SpectralEndpoint15nk prepared;
	if (!PrepareMLHeroSpectralEndpoint15nk(hitPoint, localLightDir,
			localEyeDir, &prepared) ||
			!EvaluateMLHeroSpectralEndpoint15nk(prepared, waveLength,
				basisCompensation, value))
		return false;
	*event = prepared.event;
	return true;
}

bool Glossy2Material::PrepareMLHeroSpectralEndpoint15nk(
		const HitPoint &hitPoint, const Vector &localLightDir,
		const Vector &localEyeDir,
		MLHeroGlossy2SpectralEndpoint15nk *prepared) const {
	if (!prepared)
		return false;

	const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
	const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;
	prepared->kd = Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	prepared->baseGeometry = INV_PI * fabsf(localLightDir.z);
	prepared->fromLight = hitPoint.fromLight;
	prepared->multibounce = multibounce;
	prepared->localFixedDir = localFixedDir;
	prepared->localSampledDir = localSampledDir;
	prepared->wrapperFactor = 1.f;

	if ((!doublesided) && (localEyeDir.z <= 0.f)) {
		prepared->event = DIFFUSE | REFLECT;
		prepared->baseOnly = true;
		return true;
	}

	prepared->event = GLOSSY | REFLECT;
	if ((!doublesided) && (localFixedDir.z < 0.f)) {
		prepared->baseOnly = true;
		return true;
	}

	prepared->ks = Ks->GetSpectrumValue(hitPoint);
	const float i = index->GetFloatValue(hitPoint);
	if (i > 0.f) {
		const float ti = (i - 1.f) / (i + 1.f);
		prepared->ks *= ti * ti;
	}
	prepared->ks = prepared->ks.Clamp(0.f, 1.f);
	prepared->ka = Ka->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	prepared->anisotropy = (u2 < v2) ? (1.f - u2 / v2) :
		u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	prepared->roughness = u * v;

	prepared->cosi = fabsf(localSampledDir.z);
	prepared->coso = fabsf(localFixedDir.z);
	prepared->depth = depth->GetFloatValue(hitPoint);
	prepared->baseOnly = false;
	return prepared->kd.IsValid() && prepared->ks.IsValid() &&
		prepared->ka.IsValid() && std::isfinite(prepared->baseGeometry) &&
		std::isfinite(prepared->roughness) &&
		std::isfinite(prepared->anisotropy) &&
		std::isfinite(prepared->depth) && prepared->cosi > 0.f &&
		prepared->coso > 0.f;
}

bool Glossy2Material::EvaluateMLHeroSpectralEndpoint15nk(
		const MLHeroGlossy2SpectralEndpoint15nk &prepared,
		const float waveLength, const bool basisCompensation,
		float *value) {
	if (!value || !std::isfinite(waveLength) ||
			!std::isfinite(prepared.wrapperFactor) ||
			prepared.wrapperFactor < 0.f)
		return false;
	const float kd = mlhero15nj::Reflectance(prepared.kd,
		waveLength, basisCompensation);
	const float baseF = kd * prepared.baseGeometry;
	if (prepared.baseOnly) {
		*value = prepared.wrapperFactor * baseF;
		return std::isfinite(*value) && *value >= 0.f;
	}

	const float ks = mlhero15nj::Reflectance(prepared.ks,
		waveLength, basisCompensation);
	const float alpha = mlhero15nj::Reflectance(prepared.ka,
		waveLength, basisCompensation);
	const Spectrum absorption = CoatingAbsorption(prepared.cosi,
		prepared.coso, Spectrum(alpha), prepared.depth);
	const Vector hInput = prepared.localFixedDir + prepared.localSampledDir;
	if (!(hInput.LengthSquared() > 0.f))
		return false;
	const Vector h = Normalize(hInput);
	if (!std::isfinite(h.x) || !std::isfinite(h.y) || !std::isfinite(h.z))
		return false;
	const Spectrum schlick = FresnelTexture::SchlickEvaluate(
		Spectrum(ks), AbsDot(prepared.localSampledDir, h));
	const Spectrum coating = SchlickBSDF_CoatingF(prepared.fromLight,
		Spectrum(ks), prepared.roughness, prepared.anisotropy,
		prepared.multibounce, prepared.localFixedDir,
		prepared.localSampledDir);
	const Spectrum spectralValue = prepared.wrapperFactor *
		(coating + absorption * (Spectrum(1.f) - schlick) * baseF);
	if (spectralValue.IsNaN() || spectralValue.IsInf() ||
		!std::isfinite(spectralValue.c[0]) || spectralValue.c[0] < 0.f)
		return false;
	*value = spectralValue.c[0];
	return true;
}

bool Glossy2Material::EvaluateMLHeroNativeComponentsPrepared15nk(
		const MLHeroGlossy2SpectralEndpoint15nk &prepared,
		Spectrum *baseContribution, Spectrum *coatingContribution,
		Spectrum *combinedResult) {
	if (!baseContribution || !coatingContribution || !combinedResult ||
			!std::isfinite(prepared.wrapperFactor) ||
			prepared.wrapperFactor < 0.f)
		return false;
	const Spectrum baseF = prepared.kd * prepared.baseGeometry;
	if (prepared.baseOnly) {
		*baseContribution = baseF * prepared.wrapperFactor;
		*coatingContribution = Spectrum(0.f);
		*combinedResult = *baseContribution;
		return combinedResult->IsValid();
	}

	const Vector hInput = prepared.localFixedDir + prepared.localSampledDir;
	if (!(hInput.LengthSquared() > 0.f))
		return false;
	const Vector h = Normalize(hInput);
	if (!std::isfinite(h.x) || !std::isfinite(h.y) || !std::isfinite(h.z))
		return false;
	const Spectrum absorption = CoatingAbsorption(prepared.cosi,
		prepared.coso, prepared.ka, prepared.depth);
	const Spectrum schlick = FresnelTexture::SchlickEvaluate(prepared.ks,
		AbsDot(prepared.localSampledDir, h));
	const Spectrum coating = SchlickBSDF_CoatingF(prepared.fromLight,
		prepared.ks, prepared.roughness, prepared.anisotropy,
		prepared.multibounce, prepared.localFixedDir,
		prepared.localSampledDir);
	*baseContribution = prepared.wrapperFactor * absorption *
		(Spectrum(1.f) - schlick) * baseF;
	*coatingContribution = prepared.wrapperFactor * coating;
	*combinedResult = *baseContribution + *coatingContribution;
	return baseContribution->IsValid() && coatingContribution->IsValid() &&
		combinedResult->IsValid();
}

void Glossy2Material::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
	const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;

	Spectrum ks = Ks->GetSpectrumValue(hitPoint);
	const float i = index->GetFloatValue(hitPoint);
	if (i > 0.f) {
		const float ti = (i - 1.f) / (i + 1.f);
		ks *= ti * ti;
	}
	ks = ks.Clamp(0.f, 1.f);

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;

	if (directPdfW) {
		if ((!doublesided) && (localFixedDir.z < 0.f)) {
			// Backface
			*directPdfW = fabsf (localSampledDir.z * INV_PI);
		}
		else {
			const float wCoating = SchlickBSDF_CoatingWeight (ks, localFixedDir);
			const float wBase = 1.f - wCoating;

			*directPdfW = wBase * fabsf (localSampledDir.z * INV_PI) +
				wCoating * MLHeroGlossy2CoatingPdf15lg(roughness, anisotropy, localFixedDir, localSampledDir);
		}
	}

	if (reversePdfW) {
		if ((!doublesided) && (localSampledDir.z < 0.f)) {
			// Backface

			*reversePdfW = fabsf (localFixedDir.z * INV_PI);
		}
		else {
			const float wCoatingR = SchlickBSDF_CoatingWeight (ks, localSampledDir);
			const float wBaseR = 1.f - wCoatingR;

			*reversePdfW = wBaseR * fabsf (localFixedDir.z * INV_PI) +
				wCoatingR * MLHeroGlossy2CoatingPdf15lg(roughness, anisotropy, localSampledDir, localFixedDir);
		}
	}
}

void Glossy2Material::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	Kd->AddReferencedTextures(referencedTexs);
	Ks->AddReferencedTextures(referencedTexs);
	nu->AddReferencedTextures(referencedTexs);
	nv->AddReferencedTextures(referencedTexs);
	Ka->AddReferencedTextures(referencedTexs);
	depth->AddReferencedTextures(referencedTexs);
	index->AddReferencedTextures(referencedTexs);
}

void Glossy2Material::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	bool updateGlossiness = false;
	if (Kd == &oldTex)
		Kd = &newTex;
	if (Ks == &oldTex)
		Ks = &newTex;
	if (nu == &oldTex) {
		nu = &newTex;
		updateGlossiness = true;
	}
	if (nv == &oldTex) {
		nv = &newTex;
		updateGlossiness = true;
	}
	if (Ka == &oldTex)
		Ka = &newTex;
	if (depth == &oldTex)
		depth = &newTex;
	if (index == &oldTex)
		index = &newTex;

	if (updateGlossiness)
		glossiness = ComputeGlossiness(nu, nv);
}

PropertiesUPtr Glossy2Material::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("glossy2"));
	props->Set(Property("scene.materials." + name + ".kd")(Kd->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ks")(Ks->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".uroughness")(nu->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".vroughness")(nv->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ka")(Ka->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".d")(depth->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".index")(index->GetSDLValue()));
	props->Set(Property ("scene.materials." + name + ".multibounce")(multibounce));
	props->Set(Property ("scene.materials." + name + ".doublesided")(doublesided));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
