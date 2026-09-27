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
#include "slg/materials/glass.h"
#include "slg/materials/thinfilmcoating.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// Glass material
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// ML dispersion experiment: keep one wavelength for the complete BIDIR sample
//------------------------------------------------------------------------------

thread_local float mlDispersionWaveLength = -1.f;
thread_local bool mlDispersionUsed = false;
thread_local float mlDispersionCurrentCauchyB = 0.f;
thread_local GlassDispersionModel mlDispersionCurrentModel = GLASS_DISPERSION_CAUCHY;
thread_local GlassSellmeierPreset mlDispersionCurrentSellmeierPreset = GLASS_SELLMEIER_N_BK7;
thread_local float mlDispersionSampleWeight = 1.f;
thread_local bool mlHeroEnabled = true;

// ML HERO multi-wavelength packet infrastructure. Lane 0 is always the
// existing primary HERO wavelength, so wavelengthcount=1 keeps the old
// renderer behaviour bit-for-bit at this stage. Companion lanes are prepared
// here but are not evaluated by the BSDF/path throughput yet.
static const u_int ML_HERO_MAX_WAVELENGTHS = 8u;
thread_local u_int mlHeroWavelengthCount = 1u;
thread_local u_int mlHeroActiveLane = 0u;
thread_local float mlHeroWaveLengths[ML_HERO_MAX_WAVELENGTHS] = { -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f };
thread_local float mlHeroSampleWeights[ML_HERO_MAX_WAVELENGTHS] = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };

void SetMLHeroEnabled(const bool enabled) {
	mlHeroEnabled = enabled;
	mlDispersionWaveLength = -1.f;
	mlDispersionUsed = false;
	mlDispersionCurrentCauchyB = 0.f;
	mlDispersionCurrentModel = GLASS_DISPERSION_CAUCHY;
	mlDispersionCurrentSellmeierPreset = GLASS_SELLMEIER_N_BK7;
	mlDispersionSampleWeight = 1.f;
	mlHeroWavelengthCount = 1u;
	mlHeroActiveLane = 0u;
	for (u_int i = 0; i < ML_HERO_MAX_WAVELENGTHS; ++i) {
		mlHeroWaveLengths[i] = -1.f;
		mlHeroSampleWeights[i] = 1.f;
	}
}

bool GetMLHeroEnabled() {
	return mlHeroEnabled;
}

void SetMLHeroWavelengthCount(const u_int count) {
	mlHeroWavelengthCount = Clamp(count, 1u, ML_HERO_MAX_WAVELENGTHS);
	mlHeroActiveLane = 0u;
}

u_int GetMLHeroWavelengthCount() {
	return mlHeroWavelengthCount;
}

void SetMLHeroWaveLengthAt(const u_int lane, const float waveLength, const float sampleWeight) {
	if (lane >= ML_HERO_MAX_WAVELENGTHS)
		return;

	mlHeroWaveLengths[lane] = waveLength;
	mlHeroSampleWeights[lane] = sampleWeight;
}

float GetMLHeroWaveLengthAt(const u_int lane) {
	return (lane < mlHeroWavelengthCount) ? mlHeroWaveLengths[lane] : -1.f;
}

float GetMLHeroSampleWeightAt(const u_int lane) {
	return (lane < mlHeroWavelengthCount) ? mlHeroSampleWeights[lane] : 1.f;
}

void SetMLHeroActiveLane(const u_int lane) {
	if (lane >= mlHeroWavelengthCount)
		return;

	mlHeroActiveLane = lane;
	mlDispersionWaveLength = mlHeroWaveLengths[lane];
	mlDispersionSampleWeight = mlHeroSampleWeights[lane];
	mlDispersionUsed = false;
}

u_int GetMLHeroActiveLane() {
	return mlHeroActiveLane;
}

void SetMLDispersionWaveLength(const float waveLength) {
	mlDispersionWaveLength = waveLength;
	mlDispersionSampleWeight = 1.f;
	mlHeroActiveLane = 0u;
	mlHeroWaveLengths[0] = waveLength;
	mlHeroSampleWeights[0] = 1.f;
	// Start a new hero-wavelength sample.
	mlDispersionUsed = false;
}

void SetMLDispersionWaveLength(const float waveLength, const float sampleWeight) {
	mlDispersionWaveLength = waveLength;
	mlDispersionSampleWeight = sampleWeight;
	mlHeroActiveLane = 0u;
	mlHeroWaveLengths[0] = waveLength;
	mlHeroSampleWeights[0] = sampleWeight;
	// Start a new hero-wavelength sample.
	mlDispersionUsed = false;
}

float GetMLCurrentWaveLength() {
	return mlDispersionWaveLength;
}

void MarkMLDispersionUsed() {
	mlDispersionUsed = true;
}

static float GetMLDispersionWaveLength(const float fallbackU) {
	if ((mlDispersionWaveLength >= 380.f) && (mlDispersionWaveLength <= 780.f))
		return mlDispersionWaveLength;

	return Lerp(fallbackU, 380.f, 780.f);
}

GlassMaterial::GlassMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump,
		TextureConstPtr refl, TextureConstPtr trans,
		TextureConstPtr exteriorIorFact, TextureConstPtr interiorIorFact,
		TextureConstPtr B, const GlassDispersionModel dispModel,
		const GlassSellmeierPreset smPreset,
		TextureConstPtr filmThickness, TextureConstPtr filmIor) :
			Material(frontTransp, backTransp, emitted, bump),
			Kr(refl), Kt(trans), exteriorIor(exteriorIorFact), interiorIor(interiorIorFact),
			cauchyB(B), dispersionModel(dispModel), sellmeierPreset(smPreset),
			filmThickness(filmThickness), filmIor(filmIor) {
}

Spectrum GlassMaterial::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	return Spectrum();
}

static Spectrum WaveLength2RGB(const float waveLength) {
	float r, g, b;
	if ((waveLength >= 380.f) && (waveLength < 440.f)) {
		r = -(waveLength - 440.f) / (440.f - 380.f);
		g = 0.f;
		b = 1.f;
	} else if ((waveLength >= 440.f) && (waveLength < 490.f)) {
		r = 0.f;
		g = (waveLength - 440.f) / (490.f - 440.f);
		b = 1.f;
	} else if ((waveLength >= 490.f) && (waveLength < 510.f)) {
		r = 0.f;
		g = 1.f;
		b = -(waveLength - 510.f) / (510.f - 490.f);
	} else if ((waveLength >= 510.f) && (waveLength < 580.f)) {
		r = (waveLength - 510.f) / (580.f - 510.f);
		g = 1.f;
		b = 0.f;
	} else if ((waveLength >= 580.f) && (waveLength < 645.f)) {
		r = 1.f;
		g = -(waveLength - 645.f) / (645.f - 580.f);
		b = 0.f;
	} else if ((waveLength >= 645.f) && (waveLength < 780.f)) {
		r = 1.f;
		g = 0.f;
		b = 0.f;
	} else
		return Spectrum();

	// Original LuxCore wavelength-to-RGB mapping, including its edge falloff.
	// Keeping the hero-wavelength path logic unchanged makes this an isolated
	// A/B test of the color estimator only.
	float factor;
	if ((waveLength >= 380.f) && (waveLength < 420.f))
		factor = .3f + .7f * (waveLength - 380.f) / (420.f - 380.f);
	else if ((waveLength >= 420.f) && (waveLength < 700.f))
		factor = 1.f;
	else
		factor = .3f + .7f * (780.f - waveLength) / (780.f - 700.f);

	const Spectrum result = Spectrum(r, g, b) * factor;

	// Original LuxCore normalization: a uniform average over 380..779 nm is white.
	const Spectrum normFactor(1.f / .5652729f, 1.f / .36875f, 1.f / .265375f);
	return result * normFactor;
}

Spectrum GetMLDispersionSampleColor() {
	if (!mlDispersionUsed || (mlDispersionWaveLength < 380.f) || (mlDispersionWaveLength > 780.f))
		return Spectrum(1.f);

	return WaveLength2RGB(mlDispersionWaveLength) * mlDispersionSampleWeight;
}

// ML-HERO multi-wave helper: return the wavelength-to-RGB estimator weight for
// one explicitly selected packet lane. This does not change the active HERO
// lane and is therefore safe to use for diagnostic radiance reconstruction.
Spectrum GetMLHeroSampleColorAt(const u_int lane) {
	if (lane >= mlHeroWavelengthCount)
		return Spectrum();

	const float waveLength = mlHeroWaveLengths[lane];
	if ((waveLength < 380.f) || (waveLength > 780.f))
		return Spectrum();

	return WaveLength2RGB(waveLength) * mlHeroSampleWeights[lane];
}

bool GetMLDispersionUsed() {
	return mlDispersionUsed;
}

static float WaveLength2IORStandard(const float waveLength, const float IOR, const float B) {
	// Original LuxCore convention: user IOR is Cauchy-A.
	return IOR + B / Sqr(waveLength / 1000.f);
}

static float WaveLength2IOR(const float waveLength, const float IOR, const float B) {
	// Cauchy's equation for relationship between the refractive index and wavelength
	// note: Cauchy's lambda is expressed in micrometers while waveLength is in nanometers

	// This is the formula suggested by Neo here, with a changed naming convention from B->A and C-> B:
	// https://github.com/LuxCoreRender/BlendLuxCore/commit/d3fed046ab62e18226e410b42a16ca1bccefb530#commitcomment-26617643
	
	// Compute Cauchy-A assuming the user input IOR at 587.56 nm 
	// (Fraunhofer d-line, Helium, used in one definition of the Abbe number)
	//const float A = IOR - B / Sqr(587.56f / 1000.f);

	// Use the user input IOR directly as Cauchy-A. Equivalent to the B used by old LuxRender.
	// Compute Cauchy-A assuming the user input IOR at 587.56 nm
	// (Fraunhofer d-line)
	const float A = IOR - B / Sqr(587.56f / 1000.f);

	// Old behaviour:
	// const float A = IOR;

	// Cauchy's equation
	const float cauchyEq = A + B / Sqr(waveLength / 1000.f);

	return cauchyEq;
}

//------------------------------------------------------------------------------
// ML Sellmeier runtime model/presets
//------------------------------------------------------------------------------

struct MLSellmeierCoefficients {
	float B1, B2, B3;
	float C1, C2, C3;
};

static MLSellmeierCoefficients GetMLSellmeierCoefficients(const GlassSellmeierPreset preset) {
	switch (preset) {
		case GLASS_SELLMEIER_FUSED_SILICA:
			return {0.6961663f, 0.4079426f, 0.8974794f,
				0.00467914826f, 0.0135120631f, 97.9340025f};
		case GLASS_SELLMEIER_SF10:
			return {1.62153902f, 0.256287842f, 1.64447552f,
				0.0122241457f, 0.0595736775f, 147.468793f};
		case GLASS_SELLMEIER_SF11:
			return {1.73759695f, 0.313747346f, 1.89878101f,
				0.013188707f, 0.0623068142f, 155.23629f};
		case GLASS_SELLMEIER_N_BK7:
		default:
			return {1.03961212f, 0.231792344f, 1.01046945f,
				0.00600069867f, 0.0200179144f, 103.560653f};
	}
}

static float WaveLength2IORSellmeier(const float waveLength,
		const GlassSellmeierPreset preset) {
	const MLSellmeierCoefficients c = GetMLSellmeierCoefficients(preset);
	const float lambda = waveLength / 1000.f;
	const float lambda2 = lambda * lambda;

	const float n2 = 1.f +
			c.B1 * lambda2 / (lambda2 - c.C1) +
			c.B2 * lambda2 / (lambda2 - c.C2) +
			c.B3 * lambda2 / (lambda2 - c.C3);

	return sqrtf(Max(1.f, n2));
}

static float MLWaveLength2IOR(const float waveLength, const float IOR, const float cauchyB) {
	if (mlDispersionCurrentModel == GLASS_DISPERSION_SELLMEIER)
		return WaveLength2IORSellmeier(waveLength, mlDispersionCurrentSellmeierPreset);

	return WaveLength2IOR(waveLength, IOR, cauchyB);
}

Spectrum GlassMaterial::EvalSpecularReflection(const HitPoint &hitPoint,
		const Vector &localFixedDir, const Spectrum &kr,
		const float nc, const float nt,
		Vector *localSampledDir, 
		const float localFilmThickness, const float localFilmIor) {
	if (kr.Black())
		return Spectrum();

	const float cosTheta = CosTheta(localFixedDir);
	*localSampledDir = Vector(-localFixedDir.x, -localFixedDir.y, localFixedDir.z);

	// ML Fresnel test: use the same hero-wavelength IOR for reflection
	// that transmission already uses. This keeps Fresnel reflection and
	// refraction spectrally consistent for dispersive glass.
	float lnt = nt;
	if (mlHeroEnabled && ((mlDispersionCurrentModel == GLASS_DISPERSION_SELLMEIER) ||
			(mlDispersionCurrentCauchyB > 0.f))) {
		const float waveLength = GetMLDispersionWaveLength(0.5f);
		lnt = MLWaveLength2IOR(waveLength, nt, mlDispersionCurrentCauchyB);
		mlDispersionUsed = true;
	}

	const float ntc = lnt / nc;
	const Spectrum result = kr * FresnelTexture::CauchyEvaluate(ntc, cosTheta);

	if (localFilmThickness > 0.f) {
		const Spectrum filmColor = CalcFilmColor(localFixedDir, localFilmThickness, localFilmIor);
		return result * filmColor;
	}
	return result;
}

Spectrum GlassMaterial::EvalSpecularTransmission(const HitPoint &hitPoint,
		const Vector &localFixedDir, const float u0,
		const Spectrum &kt, const float nc, const float nt, const float cauchyB,
		Vector *localSampledDir) {
	if (kt.Black())
		return Spectrum();

	// Compute transmitted ray direction
	Spectrum lkt;
	float lnt;
	if ((mlDispersionCurrentModel == GLASS_DISPERSION_SELLMEIER) || (cauchyB > 0.f)) {
		if (mlHeroEnabled) {
			// ML HERO: one fixed path wavelength with the selected IOR model.
			const float waveLength = GetMLDispersionWaveLength(u0);
			lnt = MLWaveLength2IOR(waveLength, nt, cauchyB);
			lkt = kt;
			mlDispersionUsed = true;
		} else if (mlDispersionCurrentModel == GLASS_DISPERSION_SELLMEIER) {
			// Standard per-bounce wavelength, but with Sellmeier IOR.
			const float waveLength = Lerp(u0, 380.f, 780.f);
			lnt = WaveLength2IORSellmeier(waveLength, mlDispersionCurrentSellmeierPreset);
			lkt = kt * WaveLength2RGB(waveLength);
		} else {
			// LuxCore Standard: original per-bounce Cauchy wavelength and RGB weighting.
			const float waveLength = Lerp(u0, 380.f, 780.f);
			lnt = WaveLength2IORStandard(waveLength, nt, cauchyB);
			lkt = kt * WaveLength2RGB(waveLength);
		}
	} else {
		lnt = nt;
		lkt = kt;
	}

	const float ntc = lnt / nc;
	const float cosTheta = CosTheta(localFixedDir);
	const bool entering = (cosTheta > 0.f);
	const float eta = entering ? (nc / lnt) : ntc;
	const float eta2 = eta * eta;
	const float sini2 = SinTheta2(localFixedDir);
	const float sint2 = eta2 * sini2;

	// Handle total internal reflection for transmission
	if (sint2 >= 1.f)
		return Spectrum();

	const float cost = sqrtf(Max(0.f, 1.f - sint2)) * (entering ? -1.f : 1.f);
	*localSampledDir = Vector(-eta * localFixedDir.x, -eta * localFixedDir.y, cost);

	float ce;
	if (!hitPoint.fromLight)
		ce = (1.f - FresnelTexture::CauchyEvaluate(ntc, cost)) * eta2;
	else {
		const float absCosSampledDir = fabsf(CosTheta(*localSampledDir));
		ce = (1.f - FresnelTexture::CauchyEvaluate(ntc, cosTheta)) * fabsf(CosTheta(localFixedDir) / absCosSampledDir);
	}

	return lkt * ce;
}


bool GlassMaterial::EvaluateMLHeroDebugAtWaveLength(const HitPoint &hitPoint,
		const Vector &localFixedDir, const Vector &actualLocalSampledDir,
		const BSDFEvent sampledEvent, const float samplePdfW,
		const float waveLength,
		float *exteriorIOR, float *interiorIORBase, float *interiorIORLambda,
		float *fresnelR, float *etaOut, float *eta2Out, float *transportFactor,
		float *directionDelta, Spectrum *sampleMultiplier,
		float *cosFixedOut, float *sinI2Out, float *sinT2Out, bool *tirOut,
		float *laneEventPdfWOut) const {
	if (!exteriorIOR || !interiorIORBase || !interiorIORLambda || !fresnelR ||
			!etaOut || !eta2Out || !transportFactor || !directionDelta ||
			!sampleMultiplier || (samplePdfW <= 0.f))
		return false;

	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float ntBase = ExtractInteriorIors(hitPoint, interiorIor);
	const float cauchyBValue = cauchyB ? cauchyB->GetFloatValue(hitPoint) : 0.f;

	float ntLambda = ntBase;
	if (dispersionModel == GLASS_DISPERSION_SELLMEIER)
		ntLambda = WaveLength2IORSellmeier(waveLength, sellmeierPreset);
	else if (cauchyBValue > 0.f)
		ntLambda = WaveLength2IOR(waveLength, ntBase, cauchyBValue);

	const float ntc = ntLambda / nc;
	const float cosFixed = CosTheta(localFixedDir);
	const bool entering = (cosFixed > 0.f);
	const float eta = entering ? (nc / ntLambda) : ntc;
	const float eta2 = eta * eta;
	const float sinI2 = SinTheta2(localFixedDir);
	const float sinT2 = eta2 * sinI2;
	const bool tir = (sinT2 >= 1.f);

	*exteriorIOR = nc;
	*interiorIORBase = ntBase;
	*interiorIORLambda = ntLambda;
	*etaOut = eta;
	*eta2Out = eta2;
	*directionDelta = 0.f;
	if (cosFixedOut) *cosFixedOut = cosFixed;
	if (sinI2Out) *sinI2Out = sinI2;
	if (sinT2Out) *sinT2Out = sinT2;
	if (tirOut) *tirOut = tir;

	// ML HERO phase 15ba diagnostic:
	// Reconstruct the wavelength-specific probability with which this same
	// REFLECT/TRANSMIT event would have been sampled at this lane. The active
	// renderer still uses samplePdfW from the HERO wavelength; this value is
	// diagnostic only and is used to expose spectral PDF mismatch.
	Spectrum laneRefl = kr * FresnelTexture::CauchyEvaluate(ntc, cosFixed);
	const float localFilmThickness15ba = filmThickness ? filmThickness->GetFloatValue(hitPoint) : 0.f;
	const float localFilmIor15ba = (localFilmThickness15ba > 0.f && filmIor) ?
			filmIor->GetFloatValue(hitPoint) : 1.f;
	if (localFilmThickness15ba > 0.f)
		laneRefl *= CalcFilmColor(localFixedDir, localFilmThickness15ba, localFilmIor15ba);

	Spectrum laneTrans(0.f);
	if (!tir) {
		const float cost15ba = sqrtf(Max(0.f, 1.f - sinT2)) * (entering ? -1.f : 1.f);
		float ce15ba;
		if (!hitPoint.fromLight) {
			const float f15ba = FresnelTexture::CauchyEvaluate(ntc, cost15ba);
			ce15ba = (1.f - f15ba) * eta2;
		} else {
			const float f15ba = FresnelTexture::CauchyEvaluate(ntc, cosFixed);
			const float absCosExpected15ba = fabsf(cost15ba);
			ce15ba = (absCosExpected15ba > 1e-20f) ?
					((1.f - f15ba) * fabsf(cosFixed / absCosExpected15ba)) : 0.f;
		}
		laneTrans = kt * ce15ba;
	}

	float threshold15ba;
	if (!laneRefl.Black()) {
		if (!laneTrans.Black()) {
			const float reflFilter15ba = laneRefl.Filter();
			const float transFilter15ba = laneTrans.Filter();
			threshold15ba = transFilter15ba / (reflFilter15ba + transFilter15ba);
			threshold15ba = Clamp(threshold15ba, .25f, .75f);
		} else
			threshold15ba = 0.f;
	} else {
		threshold15ba = !laneTrans.Black() ? 1.f : 0.f;
	}

	if (laneEventPdfWOut) {
		if (sampledEvent & TRANSMIT)
			*laneEventPdfWOut = threshold15ba;
		else if (sampledEvent & REFLECT)
			*laneEventPdfWOut = 1.f - threshold15ba;
		else
			*laneEventPdfWOut = 0.f;
	}

	if (sampledEvent & REFLECT) {
		const float f = FresnelTexture::CauchyEvaluate(ntc, cosFixed);
		*fresnelR = f;
		*transportFactor = f;
		Spectrum result = kr * f;

		const float localFilmThickness = filmThickness ? filmThickness->GetFloatValue(hitPoint) : 0.f;
		const float localFilmIor = (localFilmThickness > 0.f && filmIor) ? filmIor->GetFloatValue(hitPoint) : 1.f;
		if (localFilmThickness > 0.f)
			result *= CalcFilmColor(localFixedDir, localFilmThickness, localFilmIor);

		const Vector expectedDir(-localFixedDir.x, -localFixedDir.y, localFixedDir.z);
		*directionDelta = (expectedDir - actualLocalSampledDir).Length();
		*sampleMultiplier = result / samplePdfW;
		return true;
	}

	if (sampledEvent & TRANSMIT) {
		// Keep the existing 15ay behavior unchanged: a lane that would be TIR
		// cannot evaluate the shared transmitted direction and falls back to
		// the shared HERO multiplier in the caller. 15ba still reports tir/sinT2.
		if (tir)
			return false;

		const float cost = sqrtf(Max(0.f, 1.f - sinT2)) * (entering ? -1.f : 1.f);
		const Vector expectedDir(-eta * localFixedDir.x, -eta * localFixedDir.y, cost);
		*directionDelta = (expectedDir - actualLocalSampledDir).Length();

		float f;
		float ce;
		if (!hitPoint.fromLight) {
			f = FresnelTexture::CauchyEvaluate(ntc, cost);
			ce = (1.f - f) * eta2;
		} else {
			f = FresnelTexture::CauchyEvaluate(ntc, cosFixed);
			const float absCosExpected = fabsf(CosTheta(expectedDir));
			ce = (1.f - f) * fabsf(cosFixed / absCosExpected);
		}

		*fresnelR = f;
		*transportFactor = ce;
		*sampleMultiplier = kt * ce / samplePdfW;
		return true;
	}

	return false;
}

Spectrum GlassMaterial::Sample(const HitPoint &hitPoint,
		const Vector &localFixedDir, Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event) const {
	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);

	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float nt = ExtractInteriorIors(hitPoint, interiorIor);

	const float cauchyBValue = cauchyB ? cauchyB->GetFloatValue(hitPoint) : 0.f;

	// Make the current material dispersion strength available to the
	// reflection Fresnel evaluation without changing the public material API.
	mlDispersionCurrentCauchyB = cauchyBValue;
	mlDispersionCurrentModel = dispersionModel;
	mlDispersionCurrentSellmeierPreset = sellmeierPreset;

	Vector transLocalSampledDir; 
	const Spectrum trans = EvalSpecularTransmission(hitPoint, localFixedDir, u0,
			kt, nc, nt, cauchyBValue, &transLocalSampledDir);
	
	const float localFilmThickness = filmThickness ? filmThickness->GetFloatValue(hitPoint) : 0.f;
	const float localFilmIor = (localFilmThickness > 0.f && filmIor) ? filmIor->GetFloatValue(hitPoint) : 1.f;
	Vector reflLocalSampledDir;
	const Spectrum refl = EvalSpecularReflection(hitPoint, localFixedDir,
			kr, nc, nt, &reflLocalSampledDir, localFilmThickness, localFilmIor);

	// Decide to transmit or reflect
	float threshold;
	if (!refl.Black()) {
		if (!trans.Black()) {
			// Importance sampling
			const float reflFilter = refl.Filter();
			const float transFilter = trans.Filter();
			threshold = transFilter / (reflFilter + transFilter);

			// A place an upper and lower limit to not under sample
			// reflection or transmission
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
		// Transmit

		*localSampledDir = transLocalSampledDir;

		*event = SPECULAR | TRANSMIT;
		*pdfW = threshold;
	
		result = trans;
	} else {
		// Reflect

		*localSampledDir = reflLocalSampledDir;

		*event = SPECULAR | REFLECT;
		*pdfW = 1.f - threshold;
		
		result = refl;
	}
	
	return result / *pdfW;
}

void GlassMaterial::Pdf(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir,
	float *directPdfW, float *reversePdfW) const {
	if (directPdfW)
		*directPdfW = 0.f;
	if (reversePdfW)
		*reversePdfW = 0.f;
}

void GlassMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	Kr->AddReferencedTextures(referencedTexs);
	Kt->AddReferencedTextures(referencedTexs);
	if (exteriorIor)
		exteriorIor->AddReferencedTextures(referencedTexs);
	if (interiorIor)
		interiorIor->AddReferencedTextures(referencedTexs);
	if (filmThickness)
		filmThickness->AddReferencedTextures(referencedTexs);
	if (filmIor)
		filmIor->AddReferencedTextures(referencedTexs);
}

void GlassMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	if (Kr == &oldTex)
		Kr = &newTex;
	if (Kt == &oldTex)
		Kt = &newTex;
	if (exteriorIor == &oldTex)
		exteriorIor = &newTex;
	if (interiorIor == &oldTex)
		interiorIor = &newTex;
	if (filmThickness == &oldTex)
		filmThickness = &newTex;
	if (filmIor == &oldTex)
		filmIor = &newTex;
}

PropertiesUPtr GlassMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("glass"));
	props->Set(Property("scene.materials." + name + ".kr")(Kr->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".kt")(Kt->GetSDLValue()));
	if (exteriorIor)
		props->Set(Property("scene.materials." + name + ".exteriorior")(exteriorIor->GetSDLValue()));
	if (interiorIor)
		props->Set(Property("scene.materials." + name + ".interiorior")(interiorIor->GetSDLValue()));
	if (cauchyB)
		props->Set(Property("scene.materials." + name + ".cauchyb")(cauchyB->GetSDLValue()));

	props->Set(Property("scene.materials." + name + ".dispersionmodel")(
			dispersionModel == GLASS_DISPERSION_SELLMEIER ? "sellmeier" : "cauchy"));
	if (dispersionModel == GLASS_DISPERSION_SELLMEIER) {
		string presetName = "n_bk7";
		switch (sellmeierPreset) {
			case GLASS_SELLMEIER_FUSED_SILICA: presetName = "fused_silica"; break;
			case GLASS_SELLMEIER_SF10: presetName = "sf10"; break;
			case GLASS_SELLMEIER_SF11: presetName = "sf11"; break;
			case GLASS_SELLMEIER_N_BK7:
			default: presetName = "n_bk7"; break;
		}
		props->Set(Property("scene.materials." + name + ".sellmeierpreset")(presetName));
	}
	if (filmThickness)
		props->Set(Property("scene.materials." + name + ".filmthickness")(filmThickness->GetSDLValue()));
	if (filmIor)
		props->Set(Property("scene.materials." + name + ".filmior")(filmIor->GetSDLValue()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
