#include "slg/engines/mlhero_volume_direction15kp.h"
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
// 15ep: legacy spectral-color bookkeeping is broader than packet fallback.
// Explicit Metal2 lane helpers support a packet; Glass/RoughGlass and the
// existing generic single-wavelength users still require the legacy fallback.
thread_local bool mlHeroPacketFallbackUsed = false;
thread_local float mlDispersionCurrentCauchyB = 0.f;
thread_local GlassDispersionModel mlDispersionCurrentModel = GLASS_DISPERSION_CAUCHY;
thread_local GlassSellmeierPreset mlDispersionCurrentSellmeierPreset = GLASS_SELLMEIER_N_BK7;
thread_local float mlDispersionSampleWeight = 1.f;
thread_local bool mlHeroEnabled = true;
thread_local bool mlHeroRoughGlassFresnelImportance = false;

// 15ii / Diagnose 50: exact internal Sample -> EvalSpecularTransmission TIR audit.
// Stored per thread so a BIDIR caller can read the exact native sample result.
// No sampling, PDF, direction, weight or renderer decision is modified.
#include <atomic>
struct MLHeroGlassTIRState15ii {
    unsigned int reason; // 0=not-tested, 1=TIR, 2=transmission-computed
    unsigned int selected; // 0=black, 1=reflect, 2=transmit
    float wavelength, nc, ntBase, ntLambda, cosFixed, eta, sinT2, threshold;
    unsigned int fromLight;
};
static thread_local MLHeroGlassTIRState15ii mlHeroGlassTIR15ii = {};
static std::atomic<unsigned long long> mlHeroGlassTIRTotal15ii(0);
static std::atomic<unsigned long long> mlHeroGlassTIRRejected15ii(0);
static std::atomic<unsigned long long> mlHeroGlassTIRBlack15ii(0);
// Exposed only to BIDIR diagnostics; called after the real GLASS BSDF sample.
bool GetMLHeroGlassTIRSample15ii(unsigned int *reason, unsigned int *selected,
        float *wavelength, float *nc, float *ntBase, float *ntLambda,
        float *cosFixed, float *eta, float *sinT2, float *threshold,
        unsigned int *fromLight, unsigned long long *total,
        unsigned long long *tirCount, unsigned long long *blackCount) {
    if (!reason || !selected || !wavelength || !nc || !ntBase || !ntLambda ||
            !cosFixed || !eta || !sinT2 || !threshold || !fromLight ||
            !total || !tirCount || !blackCount) return false;
    const MLHeroGlassTIRState15ii &s = mlHeroGlassTIR15ii;
    *reason=s.reason; *selected=s.selected; *wavelength=s.wavelength;
    *nc=s.nc; *ntBase=s.ntBase; *ntLambda=s.ntLambda;
    *cosFixed=s.cosFixed; *eta=s.eta; *sinT2=s.sinT2;
    *threshold=s.threshold; *fromLight=s.fromLight;
    *total=mlHeroGlassTIRTotal15ii.load(std::memory_order_relaxed);
    *tirCount=mlHeroGlassTIRRejected15ii.load(std::memory_order_relaxed);
    *blackCount=mlHeroGlassTIRBlack15ii.load(std::memory_order_relaxed);
    return true;
}

// 15iw: counterfactual TIR for packet wavelengths at the REAL parent hit.
// No child BSDF Sample, no ray construction and no renderer changes.
struct MLHeroLaneTIR15iw {
    unsigned valid, count, parentTIR, differing, invalid;
    float wavelength[8], nInterior[8], sinT2[8];
    unsigned tir[8], laneValid[8];
};
static thread_local MLHeroLaneTIR15iw mlHeroLaneTIR15iw = {};
bool GetMLHeroLaneTIR15iw(unsigned *valid, unsigned *count, unsigned *parentTIR,
        unsigned *differing, unsigned *invalid, float *wavelength,
        float *nInterior, float *sinT2, unsigned *tir, unsigned *laneValid) {
    if (!valid || !count || !parentTIR || !differing || !invalid ||
            !wavelength || !nInterior || !sinT2 || !tir || !laneValid) return false;
    const MLHeroLaneTIR15iw &a=mlHeroLaneTIR15iw;
    *valid=a.valid; *count=a.count; *parentTIR=a.parentTIR;
    *differing=a.differing; *invalid=a.invalid;
    for (unsigned i=0;i<8;++i) {
        wavelength[i]=a.wavelength[i]; nInterior[i]=a.nInterior[i];
        sinT2[i]=a.sinT2[i]; tir[i]=a.tir[i]; laneValid[i]=a.laneValid[i];
    }
    return true;
}

// 15iz: actual parent Kr/Kt textures, counterfactual per-lane Fresnel and
// native-style discrete event MASS. No additional texture reads, sampling or rays.
// This uses RGB Spectrum.Filter(), not independently measured spectral Kr(lambda).
struct MLHeroLaneMass15iz {
    unsigned valid, count, invalid;
    float krFilter, ktFilter;
    float reflectFilter[8], transmitFilter[8], transmitMass[8], reflectMass[8];
    unsigned laneValid[8], reflectSupport[8], transmitSupport[8], bothBlack[8];
};
static thread_local MLHeroLaneMass15iz mlHeroLaneMass15iz = {};
bool GetMLHeroLaneMass15iz(unsigned *valid,unsigned *count,unsigned *invalid,
        float *krFilter,float *ktFilter,float *reflectFilter,float *transmitFilter,
        float *transmitMass,float *reflectMass,unsigned *laneValid,
        unsigned *reflectSupport,unsigned *transmitSupport,unsigned *bothBlack) {
    if (!valid||!count||!invalid||!krFilter||!ktFilter||!reflectFilter||
            !transmitFilter||!transmitMass||!reflectMass||!laneValid||
            !reflectSupport||!transmitSupport||!bothBlack) return false;
    const MLHeroLaneMass15iz &a=mlHeroLaneMass15iz;
    *valid=a.valid;*count=a.count;*invalid=a.invalid;
    *krFilter=a.krFilter;*ktFilter=a.ktFilter;
    for (unsigned i=0;i<8;++i) {
        reflectFilter[i]=a.reflectFilter[i];transmitFilter[i]=a.transmitFilter[i];
        transmitMass[i]=a.transmitMass[i];reflectMass[i]=a.reflectMass[i];
        laneValid[i]=a.laneValid[i];reflectSupport[i]=a.reflectSupport[i];
        transmitSupport[i]=a.transmitSupport[i];bothBlack[i]=a.bothBlack[i];
    }
    return true;
}

// 15jb: genuine native counterfactual GlassMaterial::Sample calls. These
// calls are bounded, isolated and MUST NOT be used for production transport.
struct MLHeroNativeChild15jb {
    unsigned valid, count, fail;
    unsigned sampled[8], event[8], expectedEvent[8], match[8];
    float pdf[8], expectedPdf[8], directionNorm[8], directionResidual[8], snellResidual[8];
    unsigned directionPass[8], tirBlocked[8];
};
static thread_local MLHeroNativeChild15jb mlHeroNativeChild15jb = {};
static thread_local bool mlHeroChildReentry15jb = false;
static std::atomic<unsigned> mlHeroChildBudget15jb[2] = {}; // retained, no quota ownership
// 15je: bounded gate set by actual BIDIR parent call, never by arbitrary Sample.
static thread_local bool mlHeroNativeChildArmed15je = false;
void MLHeroSetNativeGlassChildArmed15je(bool enabled) {
    mlHeroNativeChildArmed15je = enabled;
}
// 15jd: reset native diagnostic Child Sample budget at the start of each render.
// Call only while no render workers are executing GlassMaterial::Sample.
void MLHeroResetNativeGlassChildBudget15jd() {
    // 15kd: reset native TIR statistics as well as the retained child budget.
    // The BIDIR engine calls this before starting ANY of its workers.
    mlHeroGlassTIRTotal15ii.store(0ull, std::memory_order_relaxed);
    mlHeroGlassTIRRejected15ii.store(0ull, std::memory_order_relaxed);
    mlHeroGlassTIRBlack15ii.store(0ull, std::memory_order_relaxed);
    mlHeroChildBudget15jb[0].store(0u, std::memory_order_relaxed);
    mlHeroChildBudget15jb[1].store(0u, std::memory_order_relaxed);
}

bool GetMLHeroNativeChild15jb(unsigned *valid,unsigned *count,unsigned *fail,
        unsigned *sampled,unsigned *event,unsigned *expectedEvent,unsigned *match,
        float *pdf,float *expectedPdf,float *directionNorm,
        float *directionResidual,float *snellResidual,unsigned *directionPass,unsigned *tirBlocked) {
    if (!valid||!count||!fail||!sampled||!event||!expectedEvent||!match||
            !pdf||!expectedPdf||!directionNorm||!directionResidual||!snellResidual||
            !directionPass||!tirBlocked) return false;
    const MLHeroNativeChild15jb &a=mlHeroNativeChild15jb;
    *valid=a.valid;*count=a.count;*fail=a.fail;
    for (unsigned i=0;i<8;++i) {
        sampled[i]=a.sampled[i];event[i]=a.event[i];
        expectedEvent[i]=a.expectedEvent[i];match[i]=a.match[i];
        pdf[i]=a.pdf[i];expectedPdf[i]=a.expectedPdf[i];
        directionNorm[i]=a.directionNorm[i];
        directionResidual[i]=a.directionResidual[i];snellResidual[i]=a.snellResidual[i];
        directionPass[i]=a.directionPass[i];tirBlocked[i]=a.tirBlocked[i];
    }
    return true;
}

// 15is: Native decision invariant snapshot. No synthetic Sample() call.
// Each validity flag describes only the actual parent surface hit.
struct MLHeroNativeSpecial15is {
    unsigned valid, tir, reflectionPresent, transmissionPresent, bothBlack;
    unsigned supportConsistent, thresholdConsistent, selectedConsistent, pdfExpected;
    float expectedThreshold, expectedSelectedPdf, actualSelectedPdf;
};
static thread_local MLHeroNativeSpecial15is mlHeroNativeSpecial15is = {};
bool GetMLHeroNativeSpecial15is(unsigned *valid, unsigned *tir,
        unsigned *reflectionPresent, unsigned *transmissionPresent, unsigned *bothBlack,
        unsigned *supportConsistent, unsigned *thresholdConsistent,
        unsigned *selectedConsistent, unsigned *pdfExpected,
        float *expectedThreshold, float *expectedSelectedPdf, float *actualSelectedPdf) {
    if (!valid || !tir || !reflectionPresent || !transmissionPresent || !bothBlack ||
            !supportConsistent || !thresholdConsistent || !selectedConsistent || !pdfExpected ||
            !expectedThreshold || !expectedSelectedPdf || !actualSelectedPdf) return false;
    const MLHeroNativeSpecial15is &s=mlHeroNativeSpecial15is;
    *valid=s.valid; *tir=s.tir;
    *reflectionPresent=s.reflectionPresent; *transmissionPresent=s.transmissionPresent;
    *bothBlack=s.bothBlack; *supportConsistent=s.supportConsistent;
    *thresholdConsistent=s.thresholdConsistent; *selectedConsistent=s.selectedConsistent;
    *pdfExpected=s.pdfExpected; *expectedThreshold=s.expectedThreshold;
    *expectedSelectedPdf=s.expectedSelectedPdf; *actualSelectedPdf=s.actualSelectedPdf;
    return true;
}

// 15ir: read-only observation of actual native GlassMaterial::Sample weights.
struct MLHeroNativeLobe15ir {
    unsigned valid, reflectionPresent, transmissionPresent, black;
    float reflectionFilter, transmissionFilter, threshold, passThrough;
};
static thread_local MLHeroNativeLobe15ir mlHeroNativeLobe15ir = {};
bool GetMLHeroNativeLobe15ir(unsigned *valid, unsigned *reflectionPresent,
        unsigned *transmissionPresent, unsigned *black, float *reflectionFilter,
        float *transmissionFilter, float *threshold, float *passThrough) {
    if (!valid || !reflectionPresent || !transmissionPresent || !black ||
            !reflectionFilter || !transmissionFilter || !threshold || !passThrough) return false;
    const MLHeroNativeLobe15ir &v = mlHeroNativeLobe15ir;
    *valid=v.valid; *reflectionPresent=v.reflectionPresent;
    *transmissionPresent=v.transmissionPresent; *black=v.black;
    *reflectionFilter=v.reflectionFilter; *transmissionFilter=v.transmissionFilter;
    *threshold=v.threshold; *passThrough=v.passThrough;
    return true;
}

// 15kd: per-worker observations are separate from the shared counters.
// This touches diagnostics only, never wavelength or transport state.
void MLHeroResetGlassDiagnosticThreadState15kd() {
    mlHeroGlassTIR15ii = {};
    mlHeroLaneTIR15iw = {};
    mlHeroLaneMass15iz = {};
    mlHeroNativeChild15jb = {};
    mlHeroNativeSpecial15is = {};
    mlHeroNativeLobe15ir = {};
    mlHeroChildReentry15jb = false;
    mlHeroNativeChildArmed15je = false;
}

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
	mlHeroPacketFallbackUsed = false;
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
	mlHeroPacketFallbackUsed = false;
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
	mlHeroPacketFallbackUsed = false;
}

void SetMLDispersionWaveLength(const float waveLength, const float sampleWeight) {
	mlDispersionWaveLength = waveLength;
	mlDispersionSampleWeight = sampleWeight;
	mlHeroActiveLane = 0u;
	mlHeroWaveLengths[0] = waveLength;
	mlHeroSampleWeights[0] = sampleWeight;
	// Start a new hero-wavelength sample.
	mlDispersionUsed = false;
	mlHeroPacketFallbackUsed = false;
}

float GetMLCurrentWaveLength() {
	return mlDispersionWaveLength;
}

void MarkMLDispersionUsed() {
	mlDispersionUsed = true;
	mlHeroPacketFallbackUsed = true;
}

// Preserve HERO-1.0 coloring without retiring a supported Metal2 packet.
void MarkMLHeroSpectralMetalUsed() {
	mlDispersionUsed = true;
}

bool GetMLHeroPacketFallbackUsed() {
	return mlHeroPacketFallbackUsed;
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
		mlHeroPacketFallbackUsed = true;
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
			mlHeroPacketFallbackUsed = true;
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

	// 15ii: record exact wavelength-specific native refraction decision.
	mlHeroGlassTIR15ii.reason = (sint2 >= 1.f) ? 1u : 2u;
	mlHeroGlassTIR15ii.wavelength = (mlHeroEnabled && ((mlDispersionCurrentModel == GLASS_DISPERSION_SELLMEIER) || (cauchyB > 0.f))) ? GetMLDispersionWaveLength(u0) :
		((mlDispersionCurrentModel == GLASS_DISPERSION_SELLMEIER) || (cauchyB > 0.f)) ? Lerp(u0, 380.f, 780.f) : -1.f;
	mlHeroGlassTIR15ii.nc = nc;
	mlHeroGlassTIR15ii.ntBase = nt;
	mlHeroGlassTIR15ii.ntLambda = lnt;
	mlHeroGlassTIR15ii.cosFixed = cosTheta;
	mlHeroGlassTIR15ii.eta = eta;
	mlHeroGlassTIR15ii.sinT2 = sint2;
	mlHeroGlassTIR15ii.fromLight = hitPoint.fromLight ? 1u : 0u;
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


bool GlassMaterial::HasMLHeroDispersion(const HitPoint &hitPoint) const {
    return dispersionModel == GLASS_DISPERSION_SELLMEIER ||
        (cauchyB && cauchyB->GetFloatValue(hitPoint) > 0.f);
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
	// 15ii reset ensures no previous surface data can leak into this event.
    if (!mlHeroChildReentry15jb)
        mlHeroNativeChild15jb=MLHeroNativeChild15jb();
	mlHeroGlassTIR15ii = MLHeroGlassTIRState15ii();
	mlHeroNativeLobe15ir = MLHeroNativeLobe15ir();
	mlHeroNativeSpecial15is = MLHeroNativeSpecial15is();
	mlHeroLaneTIR15iw = MLHeroLaneTIR15iw();
	mlHeroLaneMass15iz = MLHeroLaneMass15iz();
    if (!mlHeroChildReentry15jb)
        mlHeroGlassTIRTotal15ii.fetch_add(1ull, std::memory_order_relaxed);
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
	// 15iw uses the same native material IOR inputs and local direction.
    // Packet lanes are counterfactual: at most eight read-only Snell decisions.
    MLHeroLaneTIR15iw &audit15iw=mlHeroLaneTIR15iw;
    audit15iw.count=Min(mlHeroWavelengthCount,8u);
    audit15iw.parentTIR=(mlHeroGlassTIR15ii.reason==1u) ? 1u : 0u;
    const double cos15iw=(double)CosTheta(localFixedDir);
    const double sinI2_15iw=Max(0.,1.-cos15iw*cos15iw);
    const bool entering15iw=cos15iw>0.;
    const bool dispersion15iw=(dispersionModel==GLASS_DISPERSION_SELLMEIER || cauchyBValue>0.f);
    for (unsigned i15iw=0;i15iw<audit15iw.count;++i15iw) {
        const float wavelength15iw=mlHeroWaveLengths[i15iw];
        audit15iw.wavelength[i15iw]=wavelength15iw;
        const bool validWl15iw=std::isfinite((double)wavelength15iw) &&
            wavelength15iw>=380.f && wavelength15iw<=780.f;
        const float n15iw=(!dispersion15iw || !validWl15iw) ? nt :
            (dispersionModel==GLASS_DISPERSION_SELLMEIER ?
                WaveLength2IORSellmeier(wavelength15iw,sellmeierPreset) :
                WaveLength2IOR(wavelength15iw,nt,cauchyBValue));
        audit15iw.nInterior[i15iw]=n15iw;
        const bool valid15iw=validWl15iw &&
            std::isfinite((double)n15iw) && n15iw>0.f &&
            std::isfinite((double)nc) && nc>0.f;
        audit15iw.laneValid[i15iw]=valid15iw ? 1u : 0u;
        if (!valid15iw) {++audit15iw.invalid;continue;}
        const double eta15iw=entering15iw ? (double)nc/n15iw : (double)n15iw/nc;
        const double sinT2_15iw=eta15iw*eta15iw*sinI2_15iw;
        audit15iw.sinT2[i15iw]=(float)sinT2_15iw;
        audit15iw.tir[i15iw]=(sinT2_15iw>=1.) ? 1u : 0u;
        if (audit15iw.tir[i15iw]!=audit15iw.parentTIR) ++audit15iw.differing;
    }
    audit15iw.valid=(audit15iw.count>0u && audit15iw.invalid==0u) ? 1u : 0u;
    // 15iz: recompute the existing native material sampling policy with
    // cached, REAL Kr/Kt texture evaluations and each lane's material IOR.
    // The native parent event remains untouched; these are diagnostic masses.
    MLHeroLaneMass15iz &mass15iz=mlHeroLaneMass15iz;
    mass15iz.count=audit15iw.count;
    mass15iz.krFilter=kr.Filter();mass15iz.ktFilter=kt.Filter();
    const float filmThickness15iz=filmThickness ? filmThickness->GetFloatValue(hitPoint) : 0.f;
    const float filmIor15iz=(filmThickness15iz>0.f && filmIor) ?
        filmIor->GetFloatValue(hitPoint) : 1.f;
    for (unsigned i15iz=0;i15iz<mass15iz.count;++i15iz) {
        if (!audit15iw.laneValid[i15iz]) { ++mass15iz.invalid; continue; }
        const float ni15iz=audit15iw.nInterior[i15iz];
        const float ntc15iz=ni15iz/nc;
        const float cos15iz=CosTheta(localFixedDir);
        const float fresnel15iz=FresnelTexture::CauchyEvaluate(ntc15iz,cos15iz);
        Spectrum refl15iz=kr*fresnel15iz;
        if (filmThickness15iz>0.f)
            refl15iz*=CalcFilmColor(localFixedDir,filmThickness15iz,filmIor15iz);
        Spectrum trans15iz(0.f);
        if (!audit15iw.tir[i15iz]) {
            const float cost15iz=sqrtf(Max(0.f,1.f-audit15iw.sinT2[i15iz]));
            const float eta15iz=entering15iw ? nc/ni15iz : ni15iz/nc;
            float ce15iz=0.f;
            if (!hitPoint.fromLight)
                ce15iz=(1.f-FresnelTexture::CauchyEvaluate(ntc15iz,
                    cost15iz*(entering15iw?-1.f:1.f)))*eta15iz*eta15iz;
            else if (cost15iz>1.e-20f)
                ce15iz=(1.f-fresnel15iz)*fabsf(cos15iz/cost15iz);
            trans15iz=kt*ce15iz;
        }
        const bool r15iz=!refl15iz.Black(),t15iz=!trans15iz.Black();
        mass15iz.reflectSupport[i15iz]=r15iz?1u:0u;
        mass15iz.transmitSupport[i15iz]=t15iz?1u:0u;
        mass15iz.bothBlack[i15iz]=(!r15iz&&!t15iz)?1u:0u;
        mass15iz.reflectFilter[i15iz]=r15iz?refl15iz.Filter():0.f;
        mass15iz.transmitFilter[i15iz]=t15iz?trans15iz.Filter():0.f;
        float pTx15iz=0.f;
        if (!r15iz) pTx15iz=t15iz?1.f:0.f;
        else if (t15iz) pTx15iz=Clamp(mass15iz.transmitFilter[i15iz]/
                (mass15iz.reflectFilter[i15iz]+mass15iz.transmitFilter[i15iz]),.25f,.75f);
        mass15iz.transmitMass[i15iz]=pTx15iz;
        mass15iz.reflectMass[i15iz]=(!r15iz&&!t15iz)?0.f:1.f-pTx15iz;
        const bool validMass15iz=std::isfinite((double)fresnel15iz) &&
            std::isfinite((double)mass15iz.reflectFilter[i15iz]) &&
            std::isfinite((double)mass15iz.transmitFilter[i15iz]) &&
            std::isfinite((double)pTx15iz);
        mass15iz.laneValid[i15iz]=validMass15iz?1u:0u;
        if (!validMass15iz) ++mass15iz.invalid;
    }
    mass15iz.valid=(mass15iz.count>0u && mass15iz.invalid==0u)?1u:0u;
    if (!mlHeroChildReentry15jb && mlHeroGlassTIR15ii.reason == 1u)
        mlHeroGlassTIRRejected15ii.fetch_add(1ull, std::memory_order_relaxed);
	
	const float localFilmThickness = filmThickness ? filmThickness->GetFloatValue(hitPoint) : 0.f;
	const float localFilmIor = (localFilmThickness > 0.f && filmIor) ? filmIor->GetFloatValue(hitPoint) : 1.f;
	Vector reflLocalSampledDir;
	const Spectrum refl = EvalSpecularReflection(hitPoint, localFixedDir,
			kr, nc, nt, &reflLocalSampledDir, localFilmThickness, localFilmIor);

	// 15ir snapshot: actual computed BSDF lobe filters; no extra evaluations.
	mlHeroNativeLobe15ir.valid = 1u;
	mlHeroNativeLobe15ir.reflectionPresent = refl.Black() ? 0u : 1u;
	mlHeroNativeLobe15ir.transmissionPresent = trans.Black() ? 0u : 1u;
	mlHeroNativeLobe15ir.black = (refl.Black() && trans.Black()) ? 1u : 0u;
	mlHeroNativeLobe15ir.reflectionFilter = refl.Black() ? 0.f : refl.Filter();
	mlHeroNativeLobe15ir.transmissionFilter = trans.Black() ? 0.f : trans.Filter();
	mlHeroNativeLobe15ir.passThrough = passThroughEvent;
	// 15is: independently derive the discrete event mass from the lobe support.
	// This observes the REAL native lobe evaluations (no second BSDF evaluation).
	MLHeroNativeSpecial15is &special15is=mlHeroNativeSpecial15is;
	special15is.valid=1u;
	special15is.tir=(mlHeroGlassTIR15ii.reason == 1u) ? 1u : 0u;
	special15is.reflectionPresent=refl.Black() ? 0u : 1u;
	special15is.transmissionPresent=trans.Black() ? 0u : 1u;
	special15is.bothBlack=(!special15is.reflectionPresent && !special15is.transmissionPresent) ? 1u : 0u;
	special15is.supportConsistent=(!special15is.tir || !special15is.transmissionPresent) ? 1u : 0u;
	special15is.expectedThreshold= special15is.bothBlack ? -1.f :
		(!special15is.reflectionPresent ? 1.f : (!special15is.transmissionPresent ? 0.f :
			Clamp(trans.Filter() / (refl.Filter() + trans.Filter()), .25f, .75f)));
	special15is.thresholdConsistent=special15is.bothBlack ? 1u : 0u;
	special15is.selectedConsistent=special15is.bothBlack ? 1u : 0u;
	special15is.pdfExpected=0u;
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
		else {
			mlHeroGlassTIR15ii.selected = 0u;
            if (!mlHeroChildReentry15jb)
                mlHeroGlassTIRBlack15ii.fetch_add(1ull, std::memory_order_relaxed);
			special15is.thresholdConsistent=1u;
			special15is.selectedConsistent=1u;
			return Spectrum();
		}
	}

	mlHeroGlassTIR15ii.threshold = threshold;
	mlHeroNativeLobe15ir.threshold = threshold;
	special15is.thresholdConsistent=(fabsf(special15is.expectedThreshold-threshold) <= 2.e-6f) ? 1u : 0u;
	special15is.selectedConsistent=((passThroughEvent < special15is.expectedThreshold) == (passThroughEvent < threshold)) ? 1u : 0u;
	special15is.expectedSelectedPdf=(passThroughEvent < special15is.expectedThreshold) ? special15is.expectedThreshold : (1.f-special15is.expectedThreshold);

	Spectrum result;
	if (passThroughEvent < threshold) {
		mlHeroGlassTIR15ii.selected = 2u;
		// Transmit

		*localSampledDir = transLocalSampledDir;

		*event = SPECULAR | TRANSMIT;
		*pdfW = threshold;
	
		result = trans;
	} else {
		// Reflect
		mlHeroGlassTIR15ii.selected = 1u;

		*localSampledDir = reflLocalSampledDir;

		*event = SPECULAR | REFLECT;
		*pdfW = 1.f - threshold;
		
		result = refl;
	}
	
	special15is.actualSelectedPdf=*pdfW;
	special15is.pdfExpected=(fabsf(special15is.expectedSelectedPdf-*pdfW) <= 2.e-6f) ? 1u : 0u;
    // 15jb: parent BSDF already finished. Replay native Child Sample with the
    // SAME valid real HitPoint, local direction and independent TLS wavelength.
    // Explicit save/restore of every known Sample-owned TLS state is essential.
    if (!mlHeroChildReentry15jb && mlHeroNativeChildArmed15je &&
            mlHeroEnabled && mlHeroWavelengthCount>1u) {
        const MLHeroGlassTIRState15ii savedTIR=mlHeroGlassTIR15ii;
        const MLHeroNativeLobe15ir savedLobe=mlHeroNativeLobe15ir;
        const MLHeroNativeSpecial15is savedSpecial=mlHeroNativeSpecial15is;
        const MLHeroLaneTIR15iw savedLanes=mlHeroLaneTIR15iw;
        const MLHeroLaneMass15iz savedMass=mlHeroLaneMass15iz;
        const float savedWavelength=mlDispersionWaveLength;
        const float savedSampleWeight=mlDispersionSampleWeight;
        const bool savedUsed=mlDispersionUsed;
        const bool savedFallback=mlHeroPacketFallbackUsed;
        const float savedB=mlDispersionCurrentCauchyB;
        const GlassDispersionModel savedModel=mlDispersionCurrentModel;
        const GlassSellmeierPreset savedPreset=mlDispersionCurrentSellmeierPreset;
        const u_int savedActiveLane=mlHeroActiveLane;
        MLHeroNativeChild15jb audit={};
        audit.count=Min(savedMass.count,8u);
        mlHeroChildReentry15jb=true;
        for (unsigned i=0;i<audit.count;++i) {
            if (!savedMass.laneValid[i] || !savedLanes.laneValid[i]) continue;
            mlDispersionWaveLength=savedLanes.wavelength[i];
            mlHeroActiveLane=i;
            mlDispersionSampleWeight=mlHeroSampleWeights[i];
            mlDispersionUsed=false;
            mlHeroPacketFallbackUsed=false;
            Vector childDir;
            float childPdf=0.f;
            BSDFEvent childEvent=(BSDFEvent)0;
            const Spectrum childThroughput=this->Sample(hitPoint,localFixedDir,
                &childDir,u0,u1,passThroughEvent,&childPdf,&childEvent);
            const unsigned eventCode=childThroughput.Black()?0u:
                ((childEvent & TRANSMIT)?2u:((childEvent & REFLECT)?1u:0u));
            const unsigned expected=savedMass.bothBlack[i]?0u:
                (passThroughEvent<savedMass.transmitMass[i]?2u:1u);
            const float expectedPdf=expected==0u?0.f:
                (expected==2u?savedMass.transmitMass[i]:savedMass.reflectMass[i]);
            const float norm=eventCode?childDir.Length():0.f;
            // 15jc: independent double-precision local Snell/reflection equation.
            // No helper direction or sampled direction is used to construct the reference.
            const double ci15jc=(double)localFixedDir.z;
            const double ni15jc=(double)savedLanes.nInterior[i];
            const double ext15jc=(double)savedTIR.nc;
            const bool enter15jc=ci15jc>0.;
            const double eta15jc=enter15jc?ext15jc/ni15jc:ni15jc/ext15jc;
            const double sinI215jc=Max(0.,1.-ci15jc*ci15jc);
            const double sinT215jc=eta15jc*eta15jc*sinI215jc;
            const bool tir15jc=sinT215jc>=1.;
            Vector predicted15jc;
            if(eventCode==1u)
                predicted15jc=Vector(-localFixedDir.x,-localFixedDir.y,localFixedDir.z);
            else if(eventCode==2u && !tir15jc)
                predicted15jc=Vector((float)(-eta15jc*localFixedDir.x),
                    (float)(-eta15jc*localFixedDir.y),
                    (float)(sqrt(Max(0.,1.-sinT215jc))*(enter15jc?-1.:1.)));
            const double residual15jc=eventCode?
                sqrt((double)(childDir.x-predicted15jc.x)*(childDir.x-predicted15jc.x)+
                     (double)(childDir.y-predicted15jc.y)*(childDir.y-predicted15jc.y)+
                     (double)(childDir.z-predicted15jc.z)*(childDir.z-predicted15jc.z)) : 0.;
            const double snell15jc=(eventCode==2u && !tir15jc)?
                fabs((double)(childDir.x*childDir.x+childDir.y*childDir.y)-sinT215jc):0.;
            const bool directionOk15jc=eventCode==0u ||
                ((!tir15jc || eventCode!=2u) && std::isfinite(residual15jc) &&
                 residual15jc<2.e-4 && std::isfinite(snell15jc) && snell15jc<2.e-4);
            audit.directionResidual[i]=(float)residual15jc;
            audit.snellResidual[i]=(float)snell15jc;
            audit.directionPass[i]=directionOk15jc?1u:0u;
            audit.tirBlocked[i]=(tir15jc && eventCode!=2u)?1u:0u;
            const bool ok=eventCode==expected &&
                fabsf(childPdf-expectedPdf)<2.e-5f &&
                (eventCode==0u || (std::isfinite((double)norm) && fabsf(norm-1.f)<1.e-3f)) &&
                !(savedLanes.tir[i] && eventCode==2u) && directionOk15jc;
            audit.sampled[i]=1u;audit.event[i]=eventCode;
            audit.expectedEvent[i]=expected;audit.match[i]=ok?1u:0u;
            audit.pdf[i]=childPdf;audit.expectedPdf[i]=expectedPdf;
            audit.directionNorm[i]=norm;
            if (!ok) ++audit.fail;
        }
        mlHeroChildReentry15jb=false;
        mlHeroGlassTIR15ii=savedTIR;
        mlHeroNativeLobe15ir=savedLobe;
        mlHeroNativeSpecial15is=savedSpecial;
        mlHeroLaneTIR15iw=savedLanes;
        mlHeroLaneMass15iz=savedMass;
        mlDispersionWaveLength=savedWavelength;
        mlDispersionSampleWeight=savedSampleWeight;
        mlDispersionUsed=savedUsed;
        mlHeroPacketFallbackUsed=savedFallback;
        mlDispersionCurrentCauchyB=savedB;
        mlDispersionCurrentModel=savedModel;
        mlDispersionCurrentSellmeierPreset=savedPreset;
        mlHeroActiveLane=savedActiveLane;
        audit.valid=1u;
        mlHeroNativeChild15jb=audit;
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

// 15kt BEGIN: diagnostic evaluation only; the native Sample/Pdf/Evaluate stay unchanged.
slg::mlhero15kt::Evidence GlassMaterial::EvaluateMLHeroReverseMass15kt(
        const HitPoint &forwardHit,const HitPoint &reverseHit,
        const Vector &fixed,const Vector &reverseFixed,
        const BSDFEvent event,const float nativeForwardMass) const {
    using namespace slg::mlhero15kt;
    Evidence a;a.status=INVALID_INPUT;
    a.reverseFromLight=reverseHit.fromLight;a.forwardIntoObject=forwardHit.intoObject;
    a.reverseIntoObject=reverseHit.intoObject;a.fixedCosine=fixed.z;a.reverseCosine=reverseFixed.z;
    const bool transmit=(event&TRANSMIT)!=0,reflect=(event&REFLECT)!=0;
    if(!mlHeroEnabled||!(event&SPECULAR)||transmit==reflect||
            !std::isfinite(nativeForwardMass)||nativeForwardMass<=0.f)return a;
    for(unsigned c=0;c<3;++c)if(!std::isfinite(fixed[c])||!std::isfinite(reverseFixed[c]))return a;
    if(!(fixed.LengthSquared()>0.f)||!(reverseFixed.LengthSquared()>0.f))return a;
    // The existing native helpers write only these TLS values. Restore even on early return.
    struct Restore {
        MLHeroGlassTIRState15ii tir=mlHeroGlassTIR15ii;
        bool used=mlDispersionUsed,fallback=mlHeroPacketFallbackUsed;
        float b=mlDispersionCurrentCauchyB;
        GlassDispersionModel model=mlDispersionCurrentModel;
        GlassSellmeierPreset preset=mlDispersionCurrentSellmeierPreset;
        ~Restore(){mlHeroGlassTIR15ii=tir;mlDispersionUsed=used;mlHeroPacketFallbackUsed=fallback;
            mlDispersionCurrentCauchyB=b;mlDispersionCurrentModel=model;mlDispersionCurrentSellmeierPreset=preset;}
    } restore;
    a.wavelength=mlDispersionWaveLength;
    struct Lobes {float threshold=0.f,mass=0.f,nc=0.f,nt=0.f;bool tir=false;Vector direction;};
    auto evaluate=[&](const HitPoint &hp,const Vector &local,Lobes &out)->bool {
        const Spectrum kr=Kr->GetSpectrumValue(hp).Clamp(0.f,1.f);
        const Spectrum kt=Kt->GetSpectrumValue(hp).Clamp(0.f,1.f);
        const float nc=ExtractExteriorIors(hp,exteriorIor),nt=ExtractInteriorIors(hp,interiorIor);
        const float b=cauchyB?cauchyB->GetFloatValue(hp):0.f;
        if(!std::isfinite(nc)||!std::isfinite(nt)||nc<=0.f||nt<=0.f||!std::isfinite(b))return false;
        if((dispersionModel==GLASS_DISPERSION_SELLMEIER||b>0.f)&&
                !(mlDispersionWaveLength>=380.f&&mlDispersionWaveLength<=780.f)){
            a.status=INVALID_WAVELENGTH;return false;
        }
        mlDispersionCurrentCauchyB=b;mlDispersionCurrentModel=dispersionModel;
        mlDispersionCurrentSellmeierPreset=sellmeierPreset;
        Vector transmitted,reflected;
        // No sampler call: 0.5 is irrelevant with a valid fixed HERO wavelength.
        const Spectrum t=EvalSpecularTransmission(hp,local,.5f,kt,nc,nt,b,&transmitted);
        out.tir=!kt.Black()&&mlHeroGlassTIR15ii.reason==1u;
        const float thickness=filmThickness?filmThickness->GetFloatValue(hp):0.f;
        const float film=(thickness>0.f&&filmIor)?filmIor->GetFloatValue(hp):1.f;
        const Spectrum r=EvalSpecularReflection(hp,local,kr,nc,nt,&reflected,thickness,film);
        for(unsigned c=0;c<3;++c)if(!std::isfinite(r.c[c])||!std::isfinite(t.c[c])||r.c[c]<0.f||t.c[c]<0.f)return false;
        if(!r.Black()&&!t.Black()&&(!std::isfinite(r.Filter())||!std::isfinite(t.Filter())||
                !std::isfinite(r.Filter()+t.Filter())||!(r.Filter()+t.Filter()>0.f)))return false;
        if(!r.Black())out.threshold=!t.Black()?Clamp(t.Filter()/(r.Filter()+t.Filter()),.25f,.75f):0.f;
        else out.threshold=!t.Black()?1.f:0.f;
        out.mass=(r.Black()&&t.Black())?0.f:(transmit?out.threshold:1.f-out.threshold);
        if(transmit&&!t.Black())out.direction=transmitted;
        else if(reflect&&!r.Black())out.direction=reflected;
        out.nc=nc;out.nt=nt;
        return std::isfinite(out.threshold)&&std::isfinite(out.mass)&&out.mass>=0.f&&out.mass<=1.f;
    };
    a.status=INVALID_LOBES;Lobes forward,reverse;
    if(!evaluate(forwardHit,fixed,forward)||!evaluate(reverseHit,reverseFixed,reverse))return a;
    a.status=EVALUATED;a.forwardMass=forward.mass;a.reverseMass=reverse.mass;
    a.forwardThreshold=forward.threshold;a.reverseThreshold=reverse.threshold;
    a.forwardMatches=CloseMass(forward.mass,nativeForwardMass);a.reverseTir=reverse.tir;
    a.nc=reverse.nc;a.nt=reverse.nt;
    if(reverse.mass>0.f){
        const Vector predicted=slg::mlhero15kp::Normalize(reverse.direction);
        const double x=double(predicted.x)-fixed.x,y=double(predicted.y)-fixed.y,z=double(predicted.z)-fixed.z;
        a.directionResidual=std::sqrt(x*x+y*y+z*z);
        a.supported=std::isfinite(a.directionResidual)&&a.directionResidual<=1.e-4;
    }
    return a;
}
// 15kt END
