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
#include "slg/materials/carpaint.h"
#include "slg/engines/mlhero_carpaint_pdf_audit15lo.h"

using namespace std;
using namespace luxrays;
using namespace slg;

bool GetMLHeroEnabled();

// Phase 15lp: SchlickDistribution_SampleH() draws half vectors from the
// inverse-CDF density established in 15lb, not from Schlick D(h).  CarPaint
// is an equal-weight mixture, so every active glossy layer must contribute
// that generated density to the directional proposal.  Reconstruct the half
// vector from the emitted direction pair in double precision so Sample(),
// Evaluate() and Pdf() agree even for the extremely sharp third paint layer.
static float MLHeroCarPaintGlossyPdf15lp(const bool l1, const float rough1,
		const bool l2, const float rough2, const bool l3, const float rough3,
		const Vector &localFixedDir, const Vector &localSampledDir) {
	const slg::mlhero15kk::V fixed = slg::mlhero15kk::Unit(
			slg::mlhero15kk::V(localFixedDir));
	const slg::mlhero15kk::V sampled = slg::mlhero15kk::Unit(
			slg::mlhero15kk::V(localSampledDir));
	const slg::mlhero15kk::V wh = slg::mlhero15kk::Unit(fixed + sampled);
	const double denominator = 4. * std::fabs(slg::mlhero15kk::Dot(fixed, wh));
	if (!(denominator > 0.) || wh.Norm2() == 0.)
		return 0.f;

	double halfPdf = 0.;
	if (l1)
		halfPdf += slg::mlhero15lb::HalfPdf(rough1, 0., wh.x, wh.y, wh.z);
	if (l2)
		halfPdf += slg::mlhero15lb::HalfPdf(rough2, 0., wh.x, wh.y, wh.z);
	if (l3)
		halfPdf += slg::mlhero15lb::HalfPdf(rough3, 0., wh.x, wh.y, wh.z);
	return float(halfPdf / denominator);
}

static float MLHeroCarPaintSelectedHalfPdf15lp(const int sampled,
		const float rough1, const float rough2, const float rough3,
		const Vector &localFixedDir, const Vector &localSampledDir) {
	if (sampled < 1 || sampled > 3)
		return 0.f;
	const slg::mlhero15kk::V fixed = slg::mlhero15kk::Unit(
			slg::mlhero15kk::V(localFixedDir));
	const slg::mlhero15kk::V sampledDir = slg::mlhero15kk::Unit(
			slg::mlhero15kk::V(localSampledDir));
	const slg::mlhero15kk::V wh = slg::mlhero15kk::Unit(fixed + sampledDir);
	const float roughness = sampled == 1 ? rough1 : (sampled == 2 ? rough2 : rough3);
	return float(slg::mlhero15lb::HalfPdf(roughness, 0., wh.x, wh.y, wh.z));
}

//------------------------------------------------------------------------------
// CarPaint material
//
// LuxRender carpaint material porting.
//------------------------------------------------------------------------------

CarPaintMaterial::CarPaintMaterial(
	TextureConstPtr frontTransp, TextureConstPtr backTransp,
	TextureConstPtr emitted, TextureConstPtr bump,
	TextureConstPtr kd, TextureConstPtr ks1, TextureConstPtr ks2, TextureConstPtr ks3, TextureConstPtr m1, TextureConstPtr m2, TextureConstPtr m3,
	TextureConstPtr r1, TextureConstPtr r2, TextureConstPtr r3, TextureConstPtr ka, TextureConstPtr d) :
	Material(frontTransp, backTransp, emitted, bump), Kd(kd), Ks1(ks1), Ks2(ks2), Ks3(ks3), M1(m1), M2(m2), M3(m3),
	R1(r1), R2(r2), R3(r3),	Ka(ka), depth(d) {
	ComputeGlossiness(M1, M2, M3);
}

Spectrum CarPaintMaterial::Albedo(const HitPoint &hitPoint) const {
	return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
}

Spectrum CarPaintMaterial::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	Vector H = Normalize(localLightDir + localEyeDir);
	if (H == Vector(0.f))
	{
		if (directPdfW)
			*directPdfW = 0.f;
		if (reversePdfW)
			*reversePdfW = 0.f;
		return Spectrum();
	}
	if (H.z < 0.f)
		H = -H;

	float pdf = 0.f;
	int n = 1; // already counts the diffuse layer
	bool l1 = false, l2 = false, l3 = false;
	float rough1 = 0.f, rough2 = 0.f, rough3 = 0.f;

	// Absorption
	const float cosi = fabsf(localLightDir.z);
	const float coso = fabsf(localEyeDir.z);
	const Spectrum alpha = Ka->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float d = depth->GetFloatValue(hitPoint);
	const Spectrum absorption = CoatingAbsorption(cosi, coso, alpha, d);

	// Diffuse layer
	Spectrum result = absorption * Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * INV_PI * fabsf(localLightDir.z);

	// 1st glossy layer
	const Spectrum ks1 = Ks1->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float m1 = M1->GetFloatValue(hitPoint);
	if (ks1.Filter() > 0.f && m1 > 0.f)
	{
		l1 = true;
		rough1 = m1 * m1;
		const float r1 = R1->GetFloatValue(hitPoint);
		result += (SchlickDistribution_D(rough1, H, 0.f) * SchlickDistribution_G(rough1, localLightDir, localEyeDir) / (4.f * coso)) *
				(ks1 * FresnelTexture::SchlickEvaluate(r1, Dot(localEyeDir, H)));
		pdf += SchlickDistribution_Pdf(rough1, H, 0.f);
		++n;
	}
	const Spectrum ks2 = Ks2->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float m2 = M2->GetFloatValue(hitPoint);
	if (ks2.Filter() > 0.f && m2 > 0.f)
	{
		l2 = true;
		rough2 = m2 * m2;
		const float r2 = R2->GetFloatValue(hitPoint);
		result += (SchlickDistribution_D(rough2, H, 0.f) * SchlickDistribution_G(rough2, localLightDir, localEyeDir) / (4.f * coso)) *
				(ks2 * FresnelTexture::SchlickEvaluate(r2, Dot(localEyeDir, H)));
		pdf += SchlickDistribution_Pdf(rough2, H, 0.f);
		++n;
	}
	const Spectrum ks3 = Ks3->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float m3 = M3->GetFloatValue(hitPoint);
	if (ks3.Filter() > 0.f && m3 > 0.f)
	{
		l3 = true;
		rough3 = m3 * m3;
		const float r3 = R3->GetFloatValue(hitPoint);
		result += (SchlickDistribution_D(rough3, H, 0.f) * SchlickDistribution_G(rough3, localLightDir, localEyeDir) / (4.f * coso)) *
				(ks3 * FresnelTexture::SchlickEvaluate(r3, Dot(localEyeDir, H)));
		pdf += SchlickDistribution_Pdf(rough3, H, 0.f);
		++n;
	}

	// Front face: coating+base
	*event = GLOSSY | REFLECT;

	// Finish pdf computation. HERO uses the density of the actual inverse-CDF
	// sampler and the emitted direction pair; HERO-off keeps LuxCore's native
	// CarPaint behavior unchanged.
	const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
	const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;
	if (GetMLHeroEnabled()) {
		if (directPdfW)
			*directPdfW = (MLHeroCarPaintGlossyPdf15lp(l1, rough1, l2, rough2,
				l3, rough3, localFixedDir, localSampledDir) +
				fabsf(localSampledDir.z) * INV_PI) / n;
		if (reversePdfW)
			*reversePdfW = (MLHeroCarPaintGlossyPdf15lp(l1, rough1, l2, rough2,
				l3, rough3, localSampledDir, localFixedDir) +
				fabsf(localFixedDir.z) * INV_PI) / n;
	} else {
		pdf /= 4.f * AbsDot(localLightDir, H);
		if (directPdfW)
			*directPdfW = (pdf + fabsf(localSampledDir.z) * INV_PI) / n;
		if (reversePdfW)
			*reversePdfW = (pdf + fabsf(localFixedDir.z) * INV_PI) / n;
	}

	return result;
}

Spectrum CarPaintMaterial::Sample(const HitPoint &hitPoint,
	const Vector &localFixedDir, Vector *localSampledDir,
	const float u0, const float u1, const float passThroughEvent,
	float *pdfW, BSDFEvent *event) const {
	if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	// Test presence of components
	int n = 1; // already count the diffuse layer
	int sampled = 0; // sampled layer
	Spectrum result(0.f);
	float pdf = 0.f;
	bool l1 = false, l2 = false, l3 = false;
	// 1st glossy layer
	const Spectrum ks1 = Ks1->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float m1 = M1->GetFloatValue(hitPoint);
	if (ks1.Filter() > 0.f && m1 > 0.f)
	{
		l1 = true;
		++n;
	}
	// 2nd glossy layer
	const Spectrum ks2 = Ks2->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float m2 = M2->GetFloatValue(hitPoint);
	if (ks2.Filter() > 0.f && m2 > 0.f)
	{
		l2 = true;
		++n;
	}
	// 3rd glossy layer
	const Spectrum ks3 = Ks3->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float m3 = M3->GetFloatValue(hitPoint);
	if (ks3.Filter() > 0.f && m3 > 0.f) {
		l3 = true;
		++n;
	}
	const float rough1 = l1 ? m1 * m1 : 0.f;
	const float rough2 = l2 ? m2 * m2 : 0.f;
	const float rough3 = l3 ? m3 * m3 : 0.f;

	// Phase 15lp: select a compact active-component slot in HERO.  The native
	// conditional chain assigns missing or later-layer intervals incorrectly
	// (for example, with all layers active layer 3 consumes layer 2's slot).
	// HERO-off deliberately retains that legacy selection for compatibility.
	int selectedComponent15lp = -1;
	if (GetMLHeroEnabled()) {
		int slot = int(passThroughEvent * n);
		if (slot < 0)
			slot = 0;
		else if (slot >= n)
			slot = n - 1;
		if (slot == 0)
			selectedComponent15lp = 0;
		else {
			if (l1 && --slot == 0)
				selectedComponent15lp = 1;
			else if (l2 && --slot == 0)
				selectedComponent15lp = 2;
			else if (l3 && --slot == 0)
				selectedComponent15lp = 3;
		}
	} else if (passThroughEvent < 1.f / n)
		selectedComponent15lp = 0;
	else if (passThroughEvent < 2.f / n && l1)
		selectedComponent15lp = 1;
	else if ((passThroughEvent < 2.f / n ||
			(!l1 && passThroughEvent < 3.f / n)) && l2)
		selectedComponent15lp = 2;
	else if (l3)
		selectedComponent15lp = 3;

	// ML HERO phase 15lo: bounded, diagnostic-only CarPaint proposal audit.
	// Phase 15lp feeds it the corrected HERO branch so the existing audit now
	// acts as the post-fix verifier for every active-layer combination.
	const u_int activeMask15lo = (l1 ? 1u : 0u) | (l2 ? 2u : 0u) | (l3 ? 4u : 0u);
	const u_int selectedBranch15lo = selectedComponent15lp >= 0 ?
		u_int(selectedComponent15lp) : slg::mlhero15lo::BASE;
	slg::mlhero15lo::Attempt audit15lo(hitPoint.fromLight, GetID(), selectedBranch15lo,
		activeMask15lo, n, rough1, rough2, rough3, u0, u1, passThroughEvent,
		slg::mlhero15kk::V(localFixedDir));
	double sampledHalfClaim15lo = 0.;

	Vector wh;
	float cosWH;
	if (selectedComponent15lp == 0) {
		// Sample diffuse layer
		*localSampledDir = Sgn(localFixedDir.z) * CosineSampleHemisphere(u0, u1, &pdf);
		if (fabsf(CosTheta(*localSampledDir)) < DEFAULT_COS_EPSILON_STATIC) {
			audit15lo.Reject(slg::mlhero15lo::DIRECTION_REJECT);
			return Spectrum();
		}

		// Absorption
		const float cosi = fabsf(localFixedDir.z);
		const float coso = fabsf(localSampledDir->z);
		const Spectrum alpha = Ka->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
		const float d = depth->GetFloatValue(hitPoint);
		const Spectrum absorption = CoatingAbsorption(cosi, coso, alpha, d);

		// Evaluate base BSDF
		const float diffuseValue = fabsf((hitPoint.fromLight ?
			localFixedDir.z : localSampledDir->z)) * INV_PI;
		result = absorption * Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) *
			(GetMLHeroEnabled() ? diffuseValue : pdf);

		wh = Normalize(*localSampledDir + localFixedDir);
		if (wh.z < 0.f)
			wh = -wh;
		cosWH = AbsDot(localFixedDir, wh);
	} else if (selectedComponent15lp == 1) {
		// Sample 1st glossy layer
		sampled = 1;
		float d;
		SchlickDistribution_SampleH(rough1, 0.f, u0, u1, &wh, &d, &pdf);
		sampledHalfClaim15lo = pdf;
		cosWH = Dot(localFixedDir, wh);
		*localSampledDir = 2.f * cosWH * wh - localFixedDir;
		cosWH = fabsf(cosWH);

		if ((localSampledDir->z < DEFAULT_COS_EPSILON_STATIC) ||
			(localFixedDir.z * localSampledDir->z < 0.f)) {
			audit15lo.Reject(slg::mlhero15lo::DIRECTION_REJECT);
			return Spectrum();
		}

		pdf /= 4.f * cosWH;
		if (pdf <= 0.f) {
			audit15lo.Reject(slg::mlhero15lo::PDF_REJECT);
			return Spectrum();
		}

		result = ks1 * FresnelTexture::SchlickEvaluate(R1->GetFloatValue(hitPoint), cosWH);

		const float G = SchlickDistribution_G(rough1, localFixedDir, *localSampledDir);
		if (!hitPoint.fromLight)
			//CoatingF(sw, *wi, wo, f_);
			result *= d * G / (4.f * fabsf(localFixedDir.z));
		else
			//CoatingF(sw, wo, *wi, f_);
			result *= d * G / (4.f * fabsf(localSampledDir->z));
	} else if (selectedComponent15lp == 2) {
		// Sample 2nd glossy layer
		sampled = 2;
		float d;
		SchlickDistribution_SampleH(rough2, 0.f, u0, u1, &wh, &d, &pdf);
		sampledHalfClaim15lo = pdf;
		cosWH = Dot(localFixedDir, wh);
		*localSampledDir = 2.f * cosWH * wh - localFixedDir;
		cosWH = fabsf(cosWH);

		if ((localSampledDir->z < DEFAULT_COS_EPSILON_STATIC) ||
			(localFixedDir.z * localSampledDir->z < 0.f)) {
			audit15lo.Reject(slg::mlhero15lo::DIRECTION_REJECT);
			return Spectrum();
		}

		pdf /= 4.f * cosWH;
		if (pdf <= 0.f) {
			audit15lo.Reject(slg::mlhero15lo::PDF_REJECT);
			return Spectrum();
		}

		result = ks2 * FresnelTexture::SchlickEvaluate(R2->GetFloatValue(hitPoint), cosWH);

		const float G = SchlickDistribution_G(rough2, localFixedDir, *localSampledDir);
		if (!hitPoint.fromLight)
			//CoatingF(sw, *wi, wo, f_);
			result *= d * G / (4.f * fabsf(localFixedDir.z));
		else
			//CoatingF(sw, wo, *wi, f_);
			result *= d * G / (4.f * fabsf(localSampledDir->z));
	} else if (selectedComponent15lp == 3) {
		// Sample 3rd glossy layer
		sampled = 3;
		float d;
		SchlickDistribution_SampleH(rough3, 0.f, u0, u1, &wh, &d, &pdf);
		sampledHalfClaim15lo = pdf;
		cosWH = Dot(localFixedDir, wh);
		*localSampledDir = 2.f * cosWH * wh - localFixedDir;
		cosWH = fabsf(cosWH);

		if ((localSampledDir->z < DEFAULT_COS_EPSILON_STATIC) ||
			(localFixedDir.z * localSampledDir->z < 0.f)) {
			audit15lo.Reject(slg::mlhero15lo::DIRECTION_REJECT);
			return Spectrum();
		}

		pdf /= 4.f * cosWH;
		if (pdf <= 0.f) {
			audit15lo.Reject(slg::mlhero15lo::PDF_REJECT);
			return Spectrum();
		}

		result = ks3 * FresnelTexture::SchlickEvaluate(R3->GetFloatValue(hitPoint), cosWH);

		const float G = SchlickDistribution_G(rough3, localFixedDir, *localSampledDir);
		if (!hitPoint.fromLight)
			//CoatingF(sw, *wi, wo, f_);
			result *= d * G / (4.f * fabsf(localFixedDir.z));
		else
			//CoatingF(sw, wo, *wi, f_);
			result *= d * G / (4.f * fabsf(localSampledDir->z));
	} else {
		// Sampling issue
		audit15lo.Reject(slg::mlhero15lo::DIRECTION_REJECT);
		return Spectrum();
	}
	*event = GLOSSY | REFLECT;
	// Add other components
	// Diffuse
	if (sampled != 0) {
		// Absorption
		const float cosi = fabsf(localFixedDir.z);
		const float coso = fabsf(localSampledDir->z);
		const Spectrum alpha = Ka->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
		const float d = depth->GetFloatValue(hitPoint);
		const Spectrum absorption = CoatingAbsorption(cosi, coso, alpha, d);

		// This is the BSDF's light-direction cosine.  In native CarPaint the
		// same value is also added to the proposal PDF, which is wrong on a
		// light path; phase 15lp replaces the complete HERO PDF below.
		const float pdf0 = fabsf((hitPoint.fromLight ? localFixedDir.z : localSampledDir->z) * INV_PI);
		pdf += pdf0;
		result += absorption * Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * pdf0;
	}
	// 1st glossy
	if (l1 && sampled != 1) {
		const float d1 = SchlickDistribution_D(rough1, wh, 0.f);
		const float pdf1 = SchlickDistribution_Pdf(rough1, wh, 0.f) / (4.f * cosWH);
		if (pdf1 > 0.f) {
			result += ks1 * (d1 *
				SchlickDistribution_G(rough1, localFixedDir, *localSampledDir) /
				(4.f * (hitPoint.fromLight ? fabsf(localSampledDir->z) : fabsf(localFixedDir.z)))) *
				FresnelTexture::SchlickEvaluate(R1->GetFloatValue(hitPoint), cosWH);
			pdf += pdf1;
		}
	}
	// 2nd glossy
	if (l2 && sampled != 2) {
		const float d2 = SchlickDistribution_D(rough2, wh, 0.f);
		const float pdf2 = SchlickDistribution_Pdf(rough2, wh, 0.f) / (4.f * cosWH);
		if (pdf2 > 0.f) {
			result += ks2 * (d2 *
				SchlickDistribution_G(rough2, localFixedDir, *localSampledDir) /
				(4.f * (hitPoint.fromLight ? fabsf(localSampledDir->z) : fabsf(localFixedDir.z)))) *
				FresnelTexture::SchlickEvaluate(R2->GetFloatValue(hitPoint), cosWH);
			pdf += pdf2;
		}
	}
	// 3rd glossy
	if (l3 && sampled != 3) {
		const float d3 = SchlickDistribution_D(rough3, wh, 0.f);
		const float pdf3 = SchlickDistribution_Pdf(rough3, wh, 0.f) / (4.f * cosWH);
		if (pdf3 > 0.f) {
			result += ks3 * (d3 *
				SchlickDistribution_G(rough3, localFixedDir, *localSampledDir) /
				(4.f * (hitPoint.fromLight ? fabsf(localSampledDir->z) : fabsf(localFixedDir.z)))) *
				FresnelTexture::SchlickEvaluate(R3->GetFloatValue(hitPoint), cosWH);
			pdf += pdf3;
		}
	}
	// Phase 15lp HERO correction.  Evaluate the physical BSDF at the emitted
	// float direction pair (rather than at the pre-rounding sampled half
	// vector), keep its light-direction cosine separate from the proposal's
	// sampled-direction cosine, and use the true inverse-CDF mixture density.
	if (GetMLHeroEnabled()) {
		const Vector &localLightDir = hitPoint.fromLight ? localFixedDir : *localSampledDir;
		const Vector &localEyeDir = hitPoint.fromLight ? *localSampledDir : localFixedDir;
		Vector evalH = Normalize(localLightDir + localEyeDir);
		if (evalH.z < 0.f)
			evalH = -evalH;
		const float cosEye = fabsf(localEyeDir.z);
		const float cosLight = fabsf(localLightDir.z);
		const Spectrum alpha = Ka->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
		const float coatingDepth = depth->GetFloatValue(hitPoint);
		const Spectrum absorption = CoatingAbsorption(cosLight, cosEye, alpha, coatingDepth);
		result = absorption * Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) *
			INV_PI * cosLight;
		if (l1)
			result += (SchlickDistribution_D(rough1, evalH, 0.f) *
				SchlickDistribution_G(rough1, localLightDir, localEyeDir) / (4.f * cosEye)) *
				(ks1 * FresnelTexture::SchlickEvaluate(R1->GetFloatValue(hitPoint),
					Dot(localEyeDir, evalH)));
		if (l2)
			result += (SchlickDistribution_D(rough2, evalH, 0.f) *
				SchlickDistribution_G(rough2, localLightDir, localEyeDir) / (4.f * cosEye)) *
				(ks2 * FresnelTexture::SchlickEvaluate(R2->GetFloatValue(hitPoint),
					Dot(localEyeDir, evalH)));
		if (l3)
			result += (SchlickDistribution_D(rough3, evalH, 0.f) *
				SchlickDistribution_G(rough3, localLightDir, localEyeDir) / (4.f * cosEye)) *
				(ks3 * FresnelTexture::SchlickEvaluate(R3->GetFloatValue(hitPoint),
					Dot(localEyeDir, evalH)));

		pdf = MLHeroCarPaintGlossyPdf15lp(l1, rough1, l2, rough2, l3, rough3,
			localFixedDir, *localSampledDir) + fabsf(localSampledDir->z) * INV_PI;
		sampledHalfClaim15lo = MLHeroCarPaintSelectedHalfPdf15lp(sampled,
			rough1, rough2, rough3, localFixedDir, *localSampledDir);
	}

	// Adjust pdf and result
	*pdfW = pdf / n;
	if (audit15lo.Active()) {
		const double claimed1_15lo = l1 ? (GetMLHeroEnabled() ?
			MLHeroCarPaintSelectedHalfPdf15lp(1, rough1, rough2, rough3,
				localFixedDir, *localSampledDir) : SchlickDistribution_Pdf(rough1, wh, 0.f)) : 0.;
		const double claimed2_15lo = l2 ? (GetMLHeroEnabled() ?
			MLHeroCarPaintSelectedHalfPdf15lp(2, rough1, rough2, rough3,
				localFixedDir, *localSampledDir) : SchlickDistribution_Pdf(rough2, wh, 0.f)) : 0.;
		const double claimed3_15lo = l3 ? (GetMLHeroEnabled() ?
			MLHeroCarPaintSelectedHalfPdf15lp(3, rough1, rough2, rough3,
				localFixedDir, *localSampledDir) : SchlickDistribution_Pdf(rough3, wh, 0.f)) : 0.;
		audit15lo.Accept(slg::mlhero15kk::V(*localSampledDir), *pdfW,
			sampledHalfClaim15lo, claimed1_15lo, claimed2_15lo, claimed3_15lo);
	}
	return result / *pdfW;
}

void CarPaintMaterial::Pdf(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir,
	float *directPdfW, float *reversePdfW) const {
	Vector H = Normalize(localLightDir + localEyeDir);
	if (H == Vector(0.f))
	{
		if (directPdfW)
			*directPdfW = 0.f;
		if (reversePdfW)
			*reversePdfW = 0.f;
		return;
	}
	if (H.z < 0.f)
		H = -H;

	float pdf = 0.f;
	int n = 1; // already counts the diffuse layer
	bool l1 = false, l2 = false, l3 = false;
	float rough1 = 0.f, rough2 = 0.f, rough3 = 0.f;

	// First specular lobe
	const Spectrum ks1 = Ks1->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float m1 = M1->GetFloatValue(hitPoint);
	if (ks1.Filter() > 0.f && m1 > 0.f)
	{
		l1 = true;
		rough1 = m1 * m1;
		pdf += SchlickDistribution_Pdf(rough1, H, 0.f);
		++n;
	}

	// Second specular lobe
	const Spectrum ks2 = Ks2->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float m2 = M2->GetFloatValue(hitPoint);
	if (ks2.Filter() > 0.f && m2 > 0.f)
	{
		l2 = true;
		rough2 = m2 * m2;
		pdf += SchlickDistribution_Pdf(rough2, H, 0.f);
		++n;
	}

	// Third specular lobe
	const Spectrum ks3 = Ks3->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float m3 = M3->GetFloatValue(hitPoint);
	if (ks3.Filter() > 0.f && m3 > 0.f)
	{
		l3 = true;
		rough3 = m3 * m3;
		pdf += SchlickDistribution_Pdf(rough3, H, 0.f);
		++n;
	}

	// Finish pdf computation
	const Vector &localFixedDir = hitPoint.fromLight ? localLightDir : localEyeDir;
	const Vector &localSampledDir = hitPoint.fromLight ? localEyeDir : localLightDir;
	if (GetMLHeroEnabled()) {
		if (directPdfW)
			*directPdfW = (MLHeroCarPaintGlossyPdf15lp(l1, rough1, l2, rough2,
				l3, rough3, localFixedDir, localSampledDir) +
				fabsf(localSampledDir.z) * INV_PI) / n;
		if (reversePdfW)
			*reversePdfW = (MLHeroCarPaintGlossyPdf15lp(l1, rough1, l2, rough2,
				l3, rough3, localSampledDir, localFixedDir) +
				fabsf(localFixedDir.z) * INV_PI) / n;
	} else {
		pdf /= 4.f * AbsDot(localLightDir, H);
		if (directPdfW)
			*directPdfW = (pdf + fabsf(localSampledDir.z) * INV_PI) / n;
		if (reversePdfW)
			*reversePdfW = (pdf + fabsf(localFixedDir.z) * INV_PI) / n;
	}
}

void CarPaintMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	Kd->AddReferencedTextures(referencedTexs);
	Ks1->AddReferencedTextures(referencedTexs);
	Ks2->AddReferencedTextures(referencedTexs);
	Ks3->AddReferencedTextures(referencedTexs);
	M1->AddReferencedTextures(referencedTexs);
	M2->AddReferencedTextures(referencedTexs);
	M3->AddReferencedTextures(referencedTexs);
	R1->AddReferencedTextures(referencedTexs);
	R2->AddReferencedTextures(referencedTexs);
	R3->AddReferencedTextures(referencedTexs);
	Ka->AddReferencedTextures(referencedTexs);
	depth->AddReferencedTextures(referencedTexs);
}

void CarPaintMaterial::UpdateTextureReferences(
	TextureConstRef oldTex,
	TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	bool updateGlossiness = false;
	if (Kd == &oldTex)
		Kd = &newTex;
	if (Ks1 == &oldTex)
		Ks1 = &newTex;
	if (Ks2 == &oldTex)
		Ks2 = &newTex;
	if (Ks3 == &oldTex)
		Ks3 = &newTex;
	if (M1 == &oldTex) {
		M1 = &newTex;
		updateGlossiness = true;
	}
	if (M2 == &oldTex) {
		M2 = &newTex;
		updateGlossiness = true;
	}
	if (M3 == &oldTex) {
		M3 = &newTex;
		updateGlossiness = true;
	}
	if (R1 == &oldTex)
		R1 = &newTex;
	if (R2 == &oldTex)
		R2 = &newTex;
	if (R3 == &oldTex)
		R3 = &newTex;
	if (Ka == &oldTex)
		Ka = &newTex;
	if (depth == &oldTex)
		depth = &newTex;
	
	if (updateGlossiness)
		ComputeGlossiness(M1, M2, M3);
}

PropertiesUPtr CarPaintMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("carpaint"));
	props->Set(Property("scene.materials." + name + ".kd")(Kd->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ks1")(Ks1->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ks2")(Ks2->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ks3")(Ks3->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".m1")(M1->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".m2")(M2->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".m3")(M3->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".r1")(R1->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".r2")(R2->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".r3")(R3->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".ka")(Ka->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".d")(depth->GetSDLValue()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}

const struct CarPaintMaterial::CarPaintData CarPaintMaterial::data[8] = {
  {"ford f8",
   {0.0012f, 0.0015f, 0.0018f},
   {0.0049f, 0.0076f, 0.0120f},
   {0.0100f, 0.0130f, 0.0180f},
   {0.0070f, 0.0065f, 0.0077f},
    0.1500f, 0.0870f, 0.9000f,
    0.3200f, 0.1100f, 0.0130f},
  {"polaris silber",
   {0.0550f, 0.0630f, 0.0710f},
   {0.0650f, 0.0820f, 0.0880f},
   {0.1100f, 0.1100f, 0.1300f},
   {0.0080f, 0.0130f, 0.0150f},
    1.0000f, 0.9200f, 0.9000f,
    0.3800f, 0.1700f, 0.0130f},
  {"opel titan",
   {0.0110f, 0.0130f, 0.0150f},
   {0.0570f, 0.0660f, 0.0780f},
   {0.1100f, 0.1200f, 0.1300f},
   {0.0095f, 0.0140f, 0.0160f},
    0.8500f, 0.8600f, 0.9000f,
    0.3800f, 0.1700f, 0.0140f},
  {"bmw339",
   {0.0120f, 0.0150f, 0.0160f},
   {0.0620f, 0.0760f, 0.0800f},
   {0.1100f, 0.1200f, 0.1200f},
   {0.0083f, 0.0150f, 0.0160f},
    0.9200f, 0.8700f, 0.9000f,
    0.3900f, 0.1700f, 0.0130f},
  {"2k acrylack",
   {0.4200f, 0.3200f, 0.1000f},
   {0.0000f, 0.0000f, 0.0000f},
   {0.0280f, 0.0260f, 0.0060f},
   {0.0170f, 0.0075f, 0.0041f},
    1.0000f, 0.9000f, 0.1700f,
    0.8800f, 0.8000f, 0.0150f},
  {"white",
   {0.6100f, 0.6300f, 0.5500f},
   {2.6e-06f, 0.00031f, 3.1e-08f},
   {0.0130f, 0.0110f, 0.0083f},
   {0.0490f, 0.0420f, 0.0370f},
    0.0490f, 0.4500f, 0.1700f,
    1.0000f, 0.1500f, 0.0150f},
  {"blue",
   {0.0079f, 0.0230f, 0.1000f},
   {0.0011f, 0.0015f, 0.0019f},
   {0.0250f, 0.0300f, 0.0430f},
   {0.0590f, 0.0740f, 0.0820f},
    1.0000f, 0.0940f, 0.1700f,
    0.1500f, 0.0430f, 0.0200f},
  {"blue matte",
   {0.0099f, 0.0360f, 0.1200f},
   {0.0032f, 0.0045f, 0.0059f},
   {0.1800f, 0.2300f, 0.2800f},
   {0.0400f, 0.0490f, 0.0510f},
    1.0000f, 0.0460f, 0.1700f,
    0.1600f, 0.0750f, 0.0340f}
};
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
