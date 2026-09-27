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

// NOTE: this is code is heavily based on Tomas Davidovic's SmallVCM
// (http://www.davidovic.cz and http://www.smallvcm.com)

#include <thread>
#include <cassert>
#include <cstdio>
#include <atomic>
#include <cmath>
#include <mutex>
#include <boost/format.hpp>

#include "luxrays/utils/thread.h"
#include "luxrays/core/color/spds/rgbillum.h"
#include "luxrays/core/color/spds/rgbrefl.h"
#include "luxrays/core/color/spds/data/rgbD65_MLHERO_380_780.h"
#include "luxrays/core/color/spds/data/xyzbasis.h"

#include "slg/engines/bidircpu/bidircpu.h"
#include "slg/cameras/camera.h"
#include "slg/lights/light.h"

// ML dispersion experiment: implemented in glass.cpp
void SetMLHeroEnabled(const bool enabled);
void SetMLDispersionWaveLength(const float waveLength);
void SetMLDispersionWaveLength(const float waveLength, const float sampleWeight);
void SetMLHeroWavelengthCount(const u_int count);
void SetMLHeroWaveLengthAt(const u_int lane, const float waveLength, const float sampleWeight);
luxrays::Spectrum GetMLDispersionSampleColor();

using namespace std;
using namespace luxrays;
using namespace slg;
using namespace std::literals::chrono_literals;

// ML HERO phase 15cq: legacy diagnostics are no longer compile-disabled.
// Runtime diagnostics controls decide what is active; this marker stays only for old log text compatibility.
#define ML_HERO_FAST_BUILD 0

// ML HERO phase 15t:
// Serialize all writes to ML_HERO_debug.txt so complete diagnostic blocks from
// different render threads can no longer interleave.
static std::recursive_mutex mlHeroDebugLogMutex;

// ML HERO phase 15cq: runtime diagnostic tiers. These are mirrored from the render-engine
// properties at thread start so old helper functions can remain available without carrying
// a render-engine pointer. Master OFF is the fast normal-render path.
static std::atomic<bool> mlHeroDiagnostics15cq(false);
static std::atomic<bool> mlHeroCurrentDiagnostics15cq(true);
static std::atomic<bool> mlHeroLegacyDiagnostics15cq(false);
static std::atomic<bool> mlHeroHeavyDiagnostics15cq(false);

static inline bool MLHeroDiagnosticsEnabled15cq() {
    return mlHeroDiagnostics15cq.load(std::memory_order_relaxed);
}
static inline bool MLHeroCurrentDiagnosticsEnabled15cq() {
    return MLHeroDiagnosticsEnabled15cq() && mlHeroCurrentDiagnostics15cq.load(std::memory_order_relaxed);
}
static inline bool MLHeroLegacyDiagnosticsEnabled15cq() {
    return MLHeroDiagnosticsEnabled15cq() && mlHeroLegacyDiagnostics15cq.load(std::memory_order_relaxed);
}
static inline bool MLHeroCurrentHeavyDiagnosticsEnabled15cq() {
    return MLHeroCurrentDiagnosticsEnabled15cq() && mlHeroHeavyDiagnostics15cq.load(std::memory_order_relaxed);
}
static inline bool MLHeroLegacyHeavyDiagnosticsEnabled15cq() {
    return MLHeroLegacyDiagnosticsEnabled15cq() && mlHeroHeavyDiagnostics15cq.load(std::memory_order_relaxed);
}

// ML HERO phase 15bd: support-aware Glass estimator diagnostics extending 15bc.
// Diagnostics only; these counters never modify path throughput or sampling.
struct MLHeroGlassOutlier15ba {
	bool valid = false;
	float correction = 1.f;
	float absDelta = 0.f;
	float lambda = 0.f;
	float heroLambda = 0.f;
	u_int depth = 0u;
	u_int caseIndex = 0u;
	float heroEventPdfW = 0.f;
	float laneEventPdfW = 0.f;
	float pdfRatioHeroOverLane = 0.f;
	float pdfAdjustedCorrection = 0.f;
	float spectralMixPdfW = 0.f;
	float spectralMixAdjustedCorrection = 0.f;
	bool supportMatch = true;
	float supportAwareCorrection = 0.f;
	float nc = 0.f;
	float ntBase = 0.f;
	float ntLambda = 0.f;
	float eta = 0.f;
	float eta2 = 0.f;
	float cosFixed = 0.f;
	float sinI2 = 0.f;
	float sinT2 = 0.f;
	bool tir = false;
	float heroIOR = 0.f;
	float baseSinT2 = 0.f;
	float heroSinT2 = 0.f;
	bool baseTir = false;
	bool heroTir = false;
	float heroDirDelta = 0.f;
	float baseDirDelta = 0.f;
	float laneDirAngle = 0.f;
	float heroDirAngle = 0.f;
	float baseDirAngle = 0.f;
	float fresnelR = 0.f;
	float transport = 0.f;
	float dirDelta = 0.f;
	float sharedMultiplier = 0.f;
	float laneMultiplier = 0.f;
};

struct MLHeroGlassCorrectionStats15az {
	unsigned long long glassBounces = 0ull;
	unsigned long long evaluatedLanes = 0ull;
	unsigned long long laneOver5Pct = 0ull;
	unsigned long long laneOver10Pct = 0ull;
	unsigned long long laneOver25Pct = 0ull;
	unsigned long long bounceOver5Pct = 0ull;
	unsigned long long bounceOver10Pct = 0ull;
	unsigned long long bounceOver25Pct = 0ull;
	unsigned long long caseBounces[4] = {0ull, 0ull, 0ull, 0ull};
	unsigned long long tirLanes = 0ull;
	unsigned long long nearTir95Lanes = 0ull;
	unsigned long long nearTir99Lanes = 0ull;
	unsigned long long pdfMismatch20PctLanes = 0ull;
	unsigned long long baseHeroTirMismatchLanes = 0ull;
	unsigned long long heroLaneTirMismatchLanes = 0ull;
	unsigned long long baseLaneTirMismatchLanes = 0ull;
	unsigned long long allThreeTirAgreeLanes = 0ull;
	unsigned long long nearHero99TransmitLanes = 0ull;
	unsigned long long rawOver25NearHero99 = 0ull;
	unsigned long long lanePdfAdjustedOver25NearHero99 = 0ull;
	unsigned long long spectralMixAdjustedOver25NearHero99 = 0ull;
	float maxRawCorrectionNearHero99 = 1.f;
	float maxLanePdfAdjustedCorrectionNearHero99 = 1.f;
	float maxSpectralMixAdjustedCorrectionNearHero99 = 1.f;
	unsigned long long eyeTransmitNearHero99Lanes = 0ull;
	unsigned long long eyeTransmitNearHero99SupportMatch = 0ull;
	unsigned long long eyeTransmitNearHero99SupportMismatch = 0ull;
	unsigned long long eyeTransmitNearHero99LanePdfOver25Supported = 0ull;
	unsigned long long eyeTransmitSupportMismatchTotal = 0ull;
	unsigned long long lightTransmitSupportMismatchTotal = 0ull;
	float maxEyeTransmitNearHero99LanePdfSupported = 1.f;
	unsigned long long heroOnlyTerminations = 0ull;
	unsigned long long heroOnlyEyeTerminations = 0ull;
	unsigned long long heroOnlyLightTerminations = 0ull;
	unsigned long long heroOnlyRatioOver105 = 0ull;
	unsigned long long heroOnlyRatioOver125 = 0ull;
	unsigned long long heroOnlyRatioOver150 = 0ull;
	unsigned long long heroOnlyRatioUnder095 = 0ull;
	double heroOnlyBrightnessRatioSum = 0.0;
	double heroOnlyIdealScaleSum = 0.0;
	double heroOnlyPacketAverageSum = 0.0;
	double heroOnlyHeroLaneSum = 0.0;
	float heroOnlyBrightnessRatioMin = 1e30f;
	float heroOnlyBrightnessRatioMax = 0.f;
	float heroOnlyIdealScaleMin = 1e30f;
	float heroOnlyIdealScaleMax = 0.f;
	float minCorrection = 1.f;
	float maxCorrection = 1.f;
	float maxAbsDelta = 0.f;
	float extremeCorrection = 1.f;
	float extremeLambda = 0.f;
	u_int extremeDepth = 0u;
	u_int extremeCase = 0u;
	MLHeroGlassOutlier15ba topOutliers[16];
};

static MLHeroGlassCorrectionStats15az mlHeroGlassCorrectionStats15az;
static std::atomic<u_int> mlHero15azFinishedThreads(0u);
static std::atomic<unsigned long long> mlHero15bhPdfTerminations(0ull);
static std::atomic<unsigned long long> mlHero15bhPdfEyeTerminations(0ull);
static std::atomic<unsigned long long> mlHero15bhPdfLightTerminations(0ull);
static std::atomic<unsigned long long> mlHero15bhVisibleConnections(0ull);
static std::atomic<unsigned long long> mlHero15bhVisibleEyeTerminated(0ull);
static std::atomic<unsigned long long> mlHero15bhVisibleLightTerminated(0ull);
static std::atomic<unsigned long long> mlHero15bhVisibleBothTerminated(0ull);
static std::atomic<unsigned long long> mlHero15bhVisibleNeitherTerminated(0ull);

// ML HERO phase 15bi: measure the *actual* ConnectVertices contribution by
// termination state. Rendering is unchanged. For the both-terminated class we
// also accumulate a diagnostic candidate equal to current / packetCount, which
// corresponds to removing one of the two packet-to-HERO normalization factors.
// Categories: 0=neither, 1=eye-only, 2=light-only, 3=both.
static std::atomic<unsigned long long> mlHero15biConnectContributionCount[4];
static std::atomic<double> mlHero15biConnectR[4];
static std::atomic<double> mlHero15biConnectG[4];
static std::atomic<double> mlHero15biConnectB[4];
static std::atomic<double> mlHero15biConnectY[4];
static std::atomic<double> mlHero15biConnectAbsY[4];

static inline void MLHeroRecordConnectContribution15bi(const Spectrum &c,
		const bool eyeTerminated, const bool lightTerminated) {
    if (!MLHeroLegacyHeavyDiagnosticsEnabled15cq()) return;
	u_int category = 0u;
	if (eyeTerminated && lightTerminated)
		category = 3u;
	else if (eyeTerminated)
		category = 1u;
	else if (lightTerminated)
		category = 2u;

	const double r = c.c[0];
	const double g = c.c[1];
	const double b = c.c[2];
	const double y = .2126 * r + .7152 * g + .0722 * b;
	mlHero15biConnectContributionCount[category].fetch_add(1ull, std::memory_order_relaxed);
	mlHero15biConnectR[category].fetch_add(r, std::memory_order_relaxed);
	mlHero15biConnectG[category].fetch_add(g, std::memory_order_relaxed);
	mlHero15biConnectB[category].fetch_add(b, std::memory_order_relaxed);
	mlHero15biConnectY[category].fetch_add(y, std::memory_order_relaxed);
	mlHero15biConnectAbsY[category].fetch_add(fabs(y), std::memory_order_relaxed);
}


// ML HERO phase 15bj: classify the actual contribution of every major BIDIR
// estimator by HERO termination state and by reconstruction path. This remains
// diagnostic-only; no contribution is modified.
//
// Estimators: 0=ConnectVertices, 1=DirectLightSampling,
//             2=ConnectToEye, 3=DirectHitLight.
// Categories: 0=neither, 1=eye-only, 2=light-only, 3=both.
static std::atomic<unsigned long long> mlHero15bjEstimatorCount[4][4];
static std::atomic<unsigned long long> mlHero15bjEstimatorPacketDivCount[4][4];
static std::atomic<unsigned long long> mlHero15bjEstimatorClassicCount[4][4];
static std::atomic<double> mlHero15bjEstimatorR[4][4];
static std::atomic<double> mlHero15bjEstimatorG[4][4];
static std::atomic<double> mlHero15bjEstimatorB[4][4];
static std::atomic<double> mlHero15bjEstimatorY[4][4];
static std::atomic<double> mlHero15bjEstimatorAbsY[4][4];

// ML HERO phase 15bk: measure whether single-HERO CIE->RGB contributions
// acquire an upward bias if negative RGB channels are later clipped.
// Diagnostic only: the renderer still receives the original contribution.
//
// Only finite contributions are accumulated here so old deep-path INF/NaN
// events cannot poison the full-render summary.
static std::atomic<unsigned long long> mlHero15bkFiniteCount[4][4];
static std::atomic<unsigned long long> mlHero15bkNonFiniteCount[4][4];
static std::atomic<unsigned long long> mlHero15bkAnyNegativeCount[4][4];
static std::atomic<unsigned long long> mlHero15bkNegativeChannelCount[4][4];
static std::atomic<double> mlHero15bkRawY[4][4];
static std::atomic<double> mlHero15bkRawAbsY[4][4];
static std::atomic<double> mlHero15bkPositiveClampedY[4][4];
static std::atomic<double> mlHero15bkPositiveClampedAbsY[4][4];
static std::atomic<double> mlHero15bkClampDeltaY[4][4];

// ML HERO phase 15bn: compare a naive event-PDF-only spectral MIS weight with
// the support-aware weight required by a perfect dispersive (Dirac) interface.
// Diagnostic only: no throughput, PDF or MIS value used by the renderer is changed.
static std::atomic<unsigned long long> mlHero15bnTerminationCount(0ull);
static std::atomic<unsigned long long> mlHero15bnEventWeightBelow099(0ull);
static std::atomic<unsigned long long> mlHero15bnEventWeightBelow075(0ull);
static std::atomic<unsigned long long> mlHero15bnEventWeightBelow050(0ull);
static std::atomic<unsigned long long> mlHero15bnEventWeightBelow025(0ull);
static std::atomic<unsigned long long> mlHero15bnSupportWeightNearOne(0ull);
static std::atomic<unsigned long long> mlHero15bnSupportCompetitorCount(0ull);
static std::atomic<double> mlHero15bnEventWeightSum(0.0);
static std::atomic<double> mlHero15bnSupportWeightSum(0.0);
static std::atomic<double> mlHero15bnEventDenomSum(0.0);
static std::atomic<double> mlHero15bnSupportDenomSum(0.0);
static std::atomic<unsigned long long> mlHero15bnDetailedLogCount(0ull);

// ML HERO phase 15bo: diagnostic upper-bound test for LuxCore BIDIR MIS after
// a dispersive HERO-only termination.  The existing LuxCore MIS quantities are
// still computed exactly as before.  We only compare them against a deliberately
// support-pruned candidate that removes the alternate strategy on the side of a
// subpath that already crossed a perfect dispersive Dirac event.
//
// This is NOT yet a production estimator.  It answers a narrower question:
// can stale alternate-technique competition after HERO termination explain the
// observed over-brightness?  Since pruning competitors can only increase the MIS
// weight, a candidate/current ratio >= 1 is expected if that hypothesis applies.
static std::atomic<unsigned long long> mlHero15boMisCount[4][4];
static std::atomic<unsigned long long> mlHero15boRatioOver101[4][4];
static std::atomic<unsigned long long> mlHero15boRatioOver105[4][4];
static std::atomic<unsigned long long> mlHero15boRatioOver125[4][4];
static std::atomic<unsigned long long> mlHero15boRatioOver150[4][4];
static std::atomic<double> mlHero15boCurrentMisSum[4][4];
static std::atomic<double> mlHero15boCandidateMisSum[4][4];
static std::atomic<double> mlHero15boRatioSum[4][4];
static std::atomic<double> mlHero15boRemovedWeightSum[4][4];

static inline void MLHeroRecordMisPrune15bo(const u_int estimator,
        const bool eyeTerminated, const bool lightTerminated,
        const float currentMis, const float candidateMis,
        const float removedWeight) {
    if (!MLHeroLegacyHeavyDiagnosticsEnabled15cq()) return;
    if ((estimator >= 4u) || !std::isfinite(currentMis) || !std::isfinite(candidateMis) ||
            (currentMis <= 0.f) || (candidateMis <= 0.f))
        return;
    u_int category = 0u;
    if (eyeTerminated && lightTerminated)
        category = 3u;
    else if (eyeTerminated)
        category = 1u;
    else if (lightTerminated)
        category = 2u;
    const double ratio = (double)candidateMis / (double)currentMis;
    mlHero15boMisCount[estimator][category].fetch_add(1ull, std::memory_order_relaxed);
    mlHero15boCurrentMisSum[estimator][category].fetch_add(currentMis, std::memory_order_relaxed);
    mlHero15boCandidateMisSum[estimator][category].fetch_add(candidateMis, std::memory_order_relaxed);
    mlHero15boRatioSum[estimator][category].fetch_add(ratio, std::memory_order_relaxed);
    mlHero15boRemovedWeightSum[estimator][category].fetch_add(removedWeight, std::memory_order_relaxed);
    if (ratio > 1.01) mlHero15boRatioOver101[estimator][category].fetch_add(1ull, std::memory_order_relaxed);
    if (ratio > 1.05) mlHero15boRatioOver105[estimator][category].fetch_add(1ull, std::memory_order_relaxed);
    if (ratio > 1.25) mlHero15boRatioOver125[estimator][category].fetch_add(1ull, std::memory_order_relaxed);
    if (ratio > 1.50) mlHero15boRatioOver150[estimator][category].fetch_add(1ull, std::memory_order_relaxed);
}


// ML HERO phase 15bp: correlate the HERO-only packet-selection ratio captured at
// the dispersive termination with the *actual visible contribution energy* later
// produced by each BIDIR estimator.  Render behavior is unchanged.
static std::atomic<unsigned long long> mlHero15bpCount[4][4];
static std::atomic<double> mlHero15bpRatioSum[4][4];
static std::atomic<double> mlHero15bpYSum[4][4];
static std::atomic<double> mlHero15bpAbsYSum[4][4];
static std::atomic<double> mlHero15bpPositiveYSum[4][4];
static std::atomic<double> mlHero15bpRatioTimesY[4][4];
static std::atomic<double> mlHero15bpRatioTimesAbsY[4][4];
static std::atomic<double> mlHero15bpRatioTimesPositiveY[4][4];
static std::atomic<double> mlHero15bpAbsYOver105[4][4];
static std::atomic<double> mlHero15bpAbsYOver125[4][4];
static std::atomic<double> mlHero15bpAbsYOver150[4][4];
static std::atomic<double> mlHero15bpAbsYUnder095[4][4];

// ML HERO phase 15bs: localize the visible contribution-weighted HERO-selection
// bias by surviving HERO wavelength, subpath side and current contribution depth.
// Diagnostic only: the renderer estimator is untouched.
static const u_int ML_HERO_15BS_WAVELENGTH_BINS = 7u;
static const u_int ML_HERO_15BS_DEPTH_BINS = 6u; // depth 1,2,3,4,5,6+
static std::atomic<unsigned long long> mlHero15bsCount[4][2][ML_HERO_15BS_WAVELENGTH_BINS][ML_HERO_15BS_DEPTH_BINS];
static std::atomic<double> mlHero15bsRatioSum[4][2][ML_HERO_15BS_WAVELENGTH_BINS][ML_HERO_15BS_DEPTH_BINS];
static std::atomic<double> mlHero15bsYSum[4][2][ML_HERO_15BS_WAVELENGTH_BINS][ML_HERO_15BS_DEPTH_BINS];
static std::atomic<double> mlHero15bsAbsYSum[4][2][ML_HERO_15BS_WAVELENGTH_BINS][ML_HERO_15BS_DEPTH_BINS];
static std::atomic<double> mlHero15bsRatioTimesAbsY[4][2][ML_HERO_15BS_WAVELENGTH_BINS][ML_HERO_15BS_DEPTH_BINS];
static std::atomic<double> mlHero15bsRatioTimesY[4][2][ML_HERO_15BS_WAVELENGTH_BINS][ML_HERO_15BS_DEPTH_BINS];
static std::atomic<unsigned long long> mlHero15bsRatioOver105[4][2][ML_HERO_15BS_WAVELENGTH_BINS][ML_HERO_15BS_DEPTH_BINS];
static std::atomic<unsigned long long> mlHero15bsRatioUnder095[4][2][ML_HERO_15BS_WAVELENGTH_BINS][ML_HERO_15BS_DEPTH_BINS];

// ML HERO phase 15bt: compare the fixed xN single-HERO reconstruction scale
// with a counterfactual packet-average reference, using the *later visible*
// contribution as the energy weight.  The reference is diagnostic only and
// assumes linear throughput scaling after the first HERO-only termination.
static std::atomic<unsigned long long> mlHero15btTerminationCount[2][ML_HERO_15BS_WAVELENGTH_BINS];
static std::atomic<double> mlHero15btTerminationRatioSum[2][ML_HERO_15BS_WAVELENGTH_BINS];
static std::atomic<double> mlHero15btHeroScalarSum[2][ML_HERO_15BS_WAVELENGTH_BINS];
static std::atomic<double> mlHero15btPacketAverageSum[2][ML_HERO_15BS_WAVELENGTH_BINS];
static std::atomic<double> mlHero15btLaneSelectionPdfSum[2][ML_HERO_15BS_WAVELENGTH_BINS];
static std::atomic<double> mlHero15btWavelengthPdfSum[2][ML_HERO_15BS_WAVELENGTH_BINS];
static std::atomic<double> mlHero15btCurrentScaleSum[2][ML_HERO_15BS_WAVELENGTH_BINS];
static std::atomic<double> mlHero15btLocalIdealScaleSum[2][ML_HERO_15BS_WAVELENGTH_BINS];

static std::atomic<unsigned long long> mlHero15btVisibleCount[4][2][ML_HERO_15BS_WAVELENGTH_BINS];
static std::atomic<double> mlHero15btActualAbsY[4][2][ML_HERO_15BS_WAVELENGTH_BINS];
static std::atomic<double> mlHero15btReferenceAbsY[4][2][ML_HERO_15BS_WAVELENGTH_BINS];
static std::atomic<double> mlHero15btRequiredScaleTimesAbsY[4][2][ML_HERO_15BS_WAVELENGTH_BINS];
static std::atomic<unsigned long long> mlHero15btInvalidVisibleCount(0ull);
static std::atomic<unsigned long long> mlHero15btDetailCount(0ull);

// ML HERO phase 15bu: propagate an explicit packet-reference throughput beside
// the surviving HERO throughput. This avoids reconstructing the reference later
// by dividing the visible contribution by heroOverPacket.
static std::atomic<unsigned long long> mlHero15buVisibleCount[4][2][ML_HERO_15BS_WAVELENGTH_BINS];
static std::atomic<double> mlHero15buActualAbsY[4][2][ML_HERO_15BS_WAVELENGTH_BINS];
static std::atomic<double> mlHero15buReferenceAbsY[4][2][ML_HERO_15BS_WAVELENGTH_BINS];
static std::atomic<double> mlHero15buRequiredScaleTimesAbsY[4][2][ML_HERO_15BS_WAVELENGTH_BINS];
static std::atomic<unsigned long long> mlHero15buRejectedCount(0ull);
static std::atomic<double> mlHero15buRejectedAbsY(0.0);

static inline int MLHeroWavelengthBin15bs(const float lambda) {
    if (!std::isfinite(lambda) || (lambda < 380.f) || (lambda > 780.f))
        return -1;
    if (lambda < 450.f) return 0;
    if (lambda < 500.f) return 1;
    if (lambda < 550.f) return 2;
    if (lambda < 600.f) return 3;
    if (lambda < 650.f) return 4;
    if (lambda < 700.f) return 5;
    return 6;
}

static inline u_int MLHeroDepthBin15bs(const u_int depth) {
    if (depth <= 1u) return 0u;
    if (depth == 2u) return 1u;
    if (depth == 3u) return 2u;
    if (depth == 4u) return 3u;
    if (depth == 5u) return 4u;
    return 5u;
}

static inline void MLHeroRecordWavelengthBias15bs(const u_int estimator,
        const u_int side, const PathVertexVM &v, const Spectrum &c) {
    if (!MLHeroLegacyHeavyDiagnosticsEnabled15cq()) return;
    if ((estimator >= 4u) || (side >= 2u) || !v.mlHeroSecondaryWavelengthsTerminated ||
            (v.mlHeroLaneCount == 0u))
        return;
    const double ratio = (double)v.mlHeroTerminationBrightnessRatio;
    const float lambda = v.mlHeroLaneWaveLength[0];
    const int wb = MLHeroWavelengthBin15bs(lambda);
    if ((wb < 0) || !std::isfinite(ratio) || (ratio <= 0.0))
        return;
    const double r = c.c[0], g = c.c[1], b = c.c[2];
    const double y = .2126 * r + .7152 * g + .0722 * b;
    if (!std::isfinite(y))
        return;
    const double absY = fabs(y);

    // Phase 15bu paired-reference diagnostic. The packet-average reference is
    // propagated explicitly in PathVertexVM. At a visible contribution we only
    // need the directly carried actual/reference throughput ratio; no division
    // by the stored termination brightness ratio is used.
    const double scale15bu = (double)v.mlHeroTerminationCurrentScale;
    const double actualTP15bu = (double)v.mlHeroLaneThroughput[0].c[0];
    const double referenceTP15bu = v.mlHero15buReferenceThroughput;
    if (std::isfinite(scale15bu) && (scale15bu > 0.0) &&
            std::isfinite(actualTP15bu) && std::isfinite(referenceTP15bu) &&
            (fabs(actualTP15bu) > 1e-30) && (fabs(referenceTP15bu) > 1e-30)) {
        const double actualOverReference15bu = actualTP15bu / referenceTP15bu;
        const double referenceAbsY15bu = absY / fabs(actualOverReference15bu);
        const double requiredScale15bu = scale15bu / fabs(actualOverReference15bu);
        // Keep the aggregate robust against vanishing-HERO firefly cases. Those
        // are counted separately instead of dominating the mean by 1e20..1e30.
        if (std::isfinite(referenceAbsY15bu) && std::isfinite(requiredScale15bu) &&
                (fabs(actualOverReference15bu) >= (1.0 / 16.0)) &&
                (fabs(actualOverReference15bu) <= 16.0)) {
            mlHero15buVisibleCount[estimator][side][wb].fetch_add(1ull, std::memory_order_relaxed);
            mlHero15buActualAbsY[estimator][side][wb].fetch_add(absY, std::memory_order_relaxed);
            mlHero15buReferenceAbsY[estimator][side][wb].fetch_add(referenceAbsY15bu, std::memory_order_relaxed);
            mlHero15buRequiredScaleTimesAbsY[estimator][side][wb].fetch_add(requiredScale15bu * absY, std::memory_order_relaxed);
        } else {
            mlHero15buRejectedCount.fetch_add(1ull, std::memory_order_relaxed);
            mlHero15buRejectedAbsY.fetch_add(absY, std::memory_order_relaxed);
        }
    } else {
        mlHero15buRejectedCount.fetch_add(1ull, std::memory_order_relaxed);
        mlHero15buRejectedAbsY.fetch_add(absY, std::memory_order_relaxed);
    }

    const u_int db = MLHeroDepthBin15bs(v.depth);
    mlHero15bsCount[estimator][side][wb][db].fetch_add(1ull, std::memory_order_relaxed);
    mlHero15bsRatioSum[estimator][side][wb][db].fetch_add(ratio, std::memory_order_relaxed);
    mlHero15bsYSum[estimator][side][wb][db].fetch_add(y, std::memory_order_relaxed);
    mlHero15bsAbsYSum[estimator][side][wb][db].fetch_add(absY, std::memory_order_relaxed);
    mlHero15bsRatioTimesAbsY[estimator][side][wb][db].fetch_add(ratio * absY, std::memory_order_relaxed);
    mlHero15bsRatioTimesY[estimator][side][wb][db].fetch_add(ratio * y, std::memory_order_relaxed);
    if (ratio > 1.05) mlHero15bsRatioOver105[estimator][side][wb][db].fetch_add(1ull, std::memory_order_relaxed);
    if (ratio < .95) mlHero15bsRatioUnder095[estimator][side][wb][db].fetch_add(1ull, std::memory_order_relaxed);
}

// ML HERO phase 15cl: lightweight shader/RGBReflSPD usage map.
// This deliberately keeps the FAST BUILD active. Only a few relaxed atomic
// counters are touched per qualifying bounce; no per-event file I/O is done.
static const u_int MLHERO_MATERIAL_TYPE_COUNT_15CL = 22u;
static std::atomic<unsigned long long> mlHero15clReflectCandidate[MLHERO_MATERIAL_TYPE_COUNT_15CL];
static std::atomic<unsigned long long> mlHero15clGenericRGBReflSPD[MLHERO_MATERIAL_TYPE_COUNT_15CL];
static std::atomic<unsigned long long> mlHero15clMatteCompensated[MLHERO_MATERIAL_TYPE_COUNT_15CL];
static std::atomic<unsigned long long> mlHero15clGlossy2Split[MLHERO_MATERIAL_TYPE_COUNT_15CL];
// ML HERO phase 15co: count actual generic compensation use over the full render
// and collect a small deterministic A/B sample set per material. The expensive
// 1 nm RGB->SPD->CIE/RGB roundtrip is intentionally capped so FAST BUILD stays fast.
static const u_int MLHERO_15CO_SAMPLE_BUDGET = 64u;
static std::atomic<unsigned long long> mlHero15cnGenericCompensated[MLHERO_MATERIAL_TYPE_COUNT_15CL];
static std::atomic<unsigned int> mlHero15coSampleAttempts[MLHERO_MATERIAL_TYPE_COUNT_15CL];
static std::atomic<unsigned long long> mlHero15coSampleCount[MLHERO_MATERIAL_TYPE_COUNT_15CL];
static std::atomic<double> mlHero15coErrBeforeSum[MLHERO_MATERIAL_TYPE_COUNT_15CL];
static std::atomic<double> mlHero15coErrAfterSum[MLHERO_MATERIAL_TYPE_COUNT_15CL];
static std::atomic<double> mlHero15coErrBeforeMax[MLHERO_MATERIAL_TYPE_COUNT_15CL];
static std::atomic<double> mlHero15coErrAfterMax[MLHERO_MATERIAL_TYPE_COUNT_15CL];

// ML HERO phase 15cq: bounded Glossy2 Kd/full-BSDF validation stats.
static constexpr u_int MLHERO_15CQ_GLOSSY2_SAMPLE_BUDGET = 64u;
static std::atomic<unsigned long long> mlHero15cqGlossy2Events(0ull);
static std::atomic<u_int> mlHero15cqGlossy2SampleAttempts(0u);
static std::atomic<unsigned long long> mlHero15cqGlossy2SampleCount(0ull);
static std::atomic<double> mlHero15cqGlossy2KdErrBeforeSum(0.0);
static std::atomic<double> mlHero15cqGlossy2KdErrAfterSum(0.0);
static std::atomic<double> mlHero15cqGlossy2KdErrBeforeMax(0.0);
static std::atomic<double> mlHero15cqGlossy2KdErrAfterMax(0.0);
static std::atomic<double> mlHero15cqGlossy2FullErrBeforeSum(0.0);
static std::atomic<double> mlHero15cqGlossy2FullErrAfterSum(0.0);
static std::atomic<double> mlHero15cqGlossy2FullErrBeforeMax(0.0);
static std::atomic<double> mlHero15cqGlossy2FullErrAfterMax(0.0);

static inline void MLHeroAtomicMax15co(std::atomic<double> &dst, const double value) {
    double current = dst.load(std::memory_order_relaxed);
    while ((value > current) && !dst.compare_exchange_weak(current, value,
            std::memory_order_relaxed, std::memory_order_relaxed)) { }
}

static inline const char *MLHeroMaterialTypeName15cl(const u_int t) {
    switch (t) {
        case MATTE: return "MATTE";
        case MIRROR: return "MIRROR";
        case GLASS: return "GLASS";
        case ARCHGLASS: return "ARCHGLASS";
        case MIX: return "MIX";
        case NULLMAT: return "NULLMAT";
        case MATTETRANSLUCENT: return "MATTETRANSLUCENT";
        case GLOSSY2: return "GLOSSY2";
        case METAL2: return "METAL2";
        case ROUGHGLASS: return "ROUGHGLASS";
        case VELVET: return "VELVET";
        case CLOTH: return "CLOTH";
        case CARPAINT: return "CARPAINT";
        case ROUGHMATTE: return "ROUGHMATTE";
        case ROUGHMATTETRANSLUCENT: return "ROUGHMATTETRANSLUCENT";
        case GLOSSYTRANSLUCENT: return "GLOSSYTRANSLUCENT";
        case GLOSSYCOATING: return "GLOSSYCOATING";
        case DISNEY: return "DISNEY";
        case TWOSIDED: return "TWOSIDED";
        case HOMOGENEOUS_VOL: return "HOMOGENEOUS_VOL";
        case CLEAR_VOL: return "CLEAR_VOL";
        case HETEROGENEOUS_VOL: return "HETEROGENEOUS_VOL";
        default: return "UNKNOWN";
    }
}

// ML HERO central diagnostics configuration. Change the phase in ONE place;
// the log filename and banner follow automatically.
#define ML_HERO_BUILD "HERO_84"
#define ML_HERO_TEST_PHASE "15cr"
#define ML_HERO_FEATURE "Diagnostics UI/log cleanup; reflectance fixes unchanged; no renderer-physics changes"
#define ML_HERO_LOG_FILE "C:\\LuxCoreCustom\\ML_HERO_debug_" ML_HERO_TEST_PHASE ".txt"

static inline FILE *MLHeroOpenCurrentLog15cq(const char *mode) {
    return MLHeroCurrentDiagnosticsEnabled15cq() ? fopen(ML_HERO_LOG_FILE, mode) : nullptr;
}
static inline FILE *MLHeroOpenLegacyLog15cq(const char *mode) {
    return MLHeroLegacyDiagnosticsEnabled15cq() ? fopen(ML_HERO_LOG_FILE, mode) : nullptr;
}

// ML HERO phase 15ce: full-render Matte packet convergence diagnostic.
// Diagnostic only. Matte events are grouped from Kd itself so the test does
// not depend on object/material names. The packet estimator is compared with
// the classic RGB BSDF sample over the whole render.
enum MLHeroMatteClass15ce {
    MLHERO_MATTE_NEUTRAL_15CE = 0,
    MLHERO_MATTE_RED_15CE = 1,
    MLHERO_MATTE_GREEN_15CE = 2,
    MLHERO_MATTE_BLUE_15CE = 3,
    MLHERO_MATTE_OTHER_15CE = 4,
    MLHERO_MATTE_CLASS_COUNT_15CE = 5
};
static std::atomic<unsigned long long> mlHero15ceCount[MLHERO_MATTE_CLASS_COUNT_15CE];
static std::atomic<double> mlHero15ceClassicR[MLHERO_MATTE_CLASS_COUNT_15CE];
static std::atomic<double> mlHero15ceClassicG[MLHERO_MATTE_CLASS_COUNT_15CE];
static std::atomic<double> mlHero15ceClassicB[MLHERO_MATTE_CLASS_COUNT_15CE];
static std::atomic<double> mlHero15cePacketR[MLHERO_MATTE_CLASS_COUNT_15CE];
static std::atomic<double> mlHero15cePacketG[MLHERO_MATTE_CLASS_COUNT_15CE];
static std::atomic<double> mlHero15cePacketB[MLHERO_MATTE_CLASS_COUNT_15CE];
static std::atomic<double> mlHero15ceAbsErrR[MLHERO_MATTE_CLASS_COUNT_15CE];
static std::atomic<double> mlHero15ceAbsErrG[MLHERO_MATTE_CLASS_COUNT_15CE];
static std::atomic<double> mlHero15ceAbsErrB[MLHERO_MATTE_CLASS_COUNT_15CE];
static std::atomic<double> mlHero15ceSqErrR[MLHERO_MATTE_CLASS_COUNT_15CE];
static std::atomic<double> mlHero15ceSqErrG[MLHERO_MATTE_CLASS_COUNT_15CE];
static std::atomic<double> mlHero15ceSqErrB[MLHERO_MATTE_CLASS_COUNT_15CE];

// ML HERO phase 15cf: accumulate the actual Matte Kd used by each 15ce class.
// At render end the mean Kd is reconstructed through RGBReflSPD and integrated
// deterministically over 380..780 nm at 1 nm resolution (trapezoidal endpoints).
// This isolates RGB->SPD->CIE/RGB conversion from Monte Carlo packet sampling.
static std::atomic<double> mlHero15cfKdR[MLHERO_MATTE_CLASS_COUNT_15CE];
static std::atomic<double> mlHero15cfKdG[MLHERO_MATTE_CLASS_COUNT_15CE];
static std::atomic<double> mlHero15cfKdB[MLHERO_MATTE_CLASS_COUNT_15CE];

static inline u_int MLHeroClassifyMatte15ce(const Spectrum &kd) {
    const float minC = Min(kd.c[0], Min(kd.c[1], kd.c[2]));
    const float maxC = Max(kd.c[0], Max(kd.c[1], kd.c[2]));
    if ((maxC - minC) <= .02f)
        return MLHERO_MATTE_NEUTRAL_15CE;
    if ((kd.c[0] > kd.c[1] * 1.25f) && (kd.c[0] > kd.c[2] * 1.25f))
        return MLHERO_MATTE_RED_15CE;
    if ((kd.c[1] > kd.c[0] * 1.25f) && (kd.c[1] > kd.c[2] * 1.25f))
        return MLHERO_MATTE_GREEN_15CE;
    if ((kd.c[2] > kd.c[0] * 1.25f) && (kd.c[2] > kd.c[1] * 1.25f))
        return MLHERO_MATTE_BLUE_15CE;
    return MLHERO_MATTE_OTHER_15CE;
}

static inline void MLHeroRecordMatteConvergence15ce(const Spectrum &kd,
        const Spectrum &classicValue, const Spectrum &packetValue) {
    if (!MLHeroLegacyHeavyDiagnosticsEnabled15cq()) return;
    const u_int c = MLHeroClassifyMatte15ce(kd);
    const double er = (double)packetValue.c[0] - (double)classicValue.c[0];
    const double eg = (double)packetValue.c[1] - (double)classicValue.c[1];
    const double eb = (double)packetValue.c[2] - (double)classicValue.c[2];
    mlHero15ceCount[c].fetch_add(1ull, std::memory_order_relaxed);
    mlHero15cfKdR[c].fetch_add(kd.c[0], std::memory_order_relaxed);
    mlHero15cfKdG[c].fetch_add(kd.c[1], std::memory_order_relaxed);
    mlHero15cfKdB[c].fetch_add(kd.c[2], std::memory_order_relaxed);
    mlHero15ceClassicR[c].fetch_add(classicValue.c[0], std::memory_order_relaxed);
    mlHero15ceClassicG[c].fetch_add(classicValue.c[1], std::memory_order_relaxed);
    mlHero15ceClassicB[c].fetch_add(classicValue.c[2], std::memory_order_relaxed);
    mlHero15cePacketR[c].fetch_add(packetValue.c[0], std::memory_order_relaxed);
    mlHero15cePacketG[c].fetch_add(packetValue.c[1], std::memory_order_relaxed);
    mlHero15cePacketB[c].fetch_add(packetValue.c[2], std::memory_order_relaxed);
    mlHero15ceAbsErrR[c].fetch_add(fabs(er), std::memory_order_relaxed);
    mlHero15ceAbsErrG[c].fetch_add(fabs(eg), std::memory_order_relaxed);
    mlHero15ceAbsErrB[c].fetch_add(fabs(eb), std::memory_order_relaxed);
    mlHero15ceSqErrR[c].fetch_add(er * er, std::memory_order_relaxed);
    mlHero15ceSqErrG[c].fetch_add(eg * eg, std::memory_order_relaxed);
    mlHero15ceSqErrB[c].fetch_add(eb * eb, std::memory_order_relaxed);
}

// ML HERO phase 15by: focused diagnostic for ConnectVertices where an eye and a
// light subpath can independently terminate to HERO-only after perfect
// dispersion.  Rendering is NOT modified.  We record the actual contribution,
// a simple both-terminated /N counterfactual, and the termination scale state
// carried by both vertices.  The /N result is a diagnostic candidate only; it
// is not assumed to be the correct estimator until the measured data supports
// that conclusion.
static std::atomic<unsigned long long> mlHero15byCount[4];
static std::atomic<unsigned long long> mlHero15byPacketDividedCount[4];
static std::atomic<double> mlHero15byY[4];
static std::atomic<double> mlHero15byAbsY[4];
static std::atomic<double> mlHero15byCandidateY[4];
static std::atomic<double> mlHero15byCandidateAbsY[4];
static std::atomic<double> mlHero15byEyeScaleSum[4];
static std::atomic<double> mlHero15byLightScaleSum[4];
static std::atomic<double> mlHero15byNaiveNetScaleSum[4];
static std::atomic<unsigned long long> mlHero15byDetailTicket(0ull);

static inline void MLHeroRecordDualTermination15by(const Spectrum &c,
        const PathVertexVM &eyeVertex, const PathVertexVM &lightVertex,
        const u_int laneCount, const bool packetDivided) {
    if (!MLHeroLegacyHeavyDiagnosticsEnabled15cq()) return;
    const bool eyeTerminated = eyeVertex.mlHeroSecondaryWavelengthsTerminated;
    const bool lightTerminated = lightVertex.mlHeroSecondaryWavelengthsTerminated;

    u_int category = 0u;
    if (eyeTerminated && lightTerminated)
        category = 3u;
    else if (eyeTerminated)
        category = 1u;
    else if (lightTerminated)
        category = 2u;

    const double r = c.c[0];
    const double g = c.c[1];
    const double b = c.c[2];
    const double y = .2126 * r + .7152 * g + .0722 * b;
    const double absY = fabs(y);
    const double n = (laneCount > 0u) ? (double)laneCount : 1.0;
    const double candidateScale = (category == 3u) ? (1.0 / n) : 1.0;
    const double eyeScale = eyeTerminated ? (double)eyeVertex.mlHeroTerminationCurrentScale : 1.0;
    const double lightScale = lightTerminated ? (double)lightVertex.mlHeroTerminationCurrentScale : 1.0;
    const double reconstructionScale = packetDivided ? (1.0 / n) : 1.0;
    const double naiveNetScale = eyeScale * lightScale * reconstructionScale;

    mlHero15byCount[category].fetch_add(1ull, std::memory_order_relaxed);
    if (packetDivided)
        mlHero15byPacketDividedCount[category].fetch_add(1ull, std::memory_order_relaxed);
    mlHero15byY[category].fetch_add(y, std::memory_order_relaxed);
    mlHero15byAbsY[category].fetch_add(absY, std::memory_order_relaxed);
    mlHero15byCandidateY[category].fetch_add(y * candidateScale, std::memory_order_relaxed);
    mlHero15byCandidateAbsY[category].fetch_add(absY * candidateScale, std::memory_order_relaxed);
    mlHero15byEyeScaleSum[category].fetch_add(eyeScale, std::memory_order_relaxed);
    mlHero15byLightScaleSum[category].fetch_add(lightScale, std::memory_order_relaxed);
    mlHero15byNaiveNetScaleSum[category].fetch_add(naiveNetScale, std::memory_order_relaxed);

    if (category == 3u) {
        const unsigned long long ticket = mlHero15byDetailTicket.fetch_add(1ull, std::memory_order_relaxed);
        if (ticket < 32ull) {
            std::lock_guard<std::recursive_mutex> lock(mlHeroDebugLogMutex);
            FILE *f = MLHeroOpenLegacyLog15cq("a");
            if (f) {
                fprintf(f,
                    "phase15by DUAL_DETAIL index=%llu eyeDepth=%u lightDepth=%u lanes=%u packetDivided=%u "
                    "eyeScale=%.9g lightScale=%.9g reconstructionScale=%.9g naiveNetScale=%.9g "
                    "actualRGB=(%.9g,%.9g,%.9g) actualY=%.12g candidateDivN_Y=%.12g\n",
                    ticket, eyeVertex.depth, lightVertex.depth, laneCount,
                    packetDivided ? 1u : 0u,
                    eyeScale, lightScale, reconstructionScale, naiveNetScale,
                    r, g, b, y, y * candidateScale);
                fclose(f);
            }
        }
    }
}

// ML HERO phase 15bz: runtime A/B correction for ConnectVertices. OFF preserves
// HERO_65. ON divides only doubly HERO-only contributions by the packet size.
static std::atomic<unsigned long long> mlHero15bzAppliedCount(0ull);
static std::atomic<double> mlHero15bzBaselineY(0.0);
static std::atomic<double> mlHero15bzCompensatedY(0.0);
static std::atomic<unsigned long long> mlHero15bzDetailTicket(0ull);

static inline bool MLHeroApplyDualTerminationCompensation15bz(Spectrum &c,
        const PathVertexVM &eyeVertex, const PathVertexVM &lightVertex,
        const u_int laneCount) {
    if (!eyeVertex.mlHeroSecondaryWavelengthsTerminated ||
            !lightVertex.mlHeroSecondaryWavelengthsTerminated || (laneCount == 0u))
        return false;

    const double yBefore = .2126 * c.c[0] + .7152 * c.c[1] + .0722 * c.c[2];
    c /= (float)laneCount;
    const double yAfter = .2126 * c.c[0] + .7152 * c.c[1] + .0722 * c.c[2];
    mlHero15bzAppliedCount.fetch_add(1ull, std::memory_order_relaxed);
    mlHero15bzBaselineY.fetch_add(yBefore, std::memory_order_relaxed);
    mlHero15bzCompensatedY.fetch_add(yAfter, std::memory_order_relaxed);

    const unsigned long long ticket = mlHero15bzDetailTicket.fetch_add(1ull, std::memory_order_relaxed);
    if (ticket < 16ull) {
        std::lock_guard<std::recursive_mutex> lock(mlHeroDebugLogMutex);
        FILE *f = MLHeroOpenLegacyLog15cq("a");
        if (f) {
            fprintf(f, "phase15bz APPLY index=%llu eyeDepth=%u lightDepth=%u lanes=%u baselineY=%.12g compensatedY=%.12g ratio=%.9g\n",
                ticket, eyeVertex.depth, lightVertex.depth, laneCount, yBefore, yAfter,
                (fabs(yBefore) > 1e-30) ? (yAfter / yBefore) : 0.0);
            fclose(f);
        }
    }
    return true;
}


// ML HERO phase 15bq: verify that the termination brightness-ratio state is
// actually stored, survives later bounces/copies, and reaches the visible
// BIDIR estimators.  This is diagnostic-only and deliberately logs a small
// number of events immediately instead of relying only on the end-of-render
// summary barrier.
static std::atomic<unsigned long long> mlHero15bqStoreCount(0ull);
static std::atomic<unsigned long long> mlHero15bqPropagateCount(0ull);
static std::atomic<unsigned long long> mlHero15bqReadCount[4];
static std::atomic<unsigned long long> mlHero15bqInvalidRatioCount(0ull);
static std::atomic<unsigned long long> mlHero15bqSnapshotTicket(0ull);

static inline bool MLHero15bqShouldDetail(const unsigned long long index) {
    return (index < 16ull) || ((index != 0ull) && ((index & (index - 1ull)) == 0ull) && (index <= 1048576ull));
}

static inline void MLHeroLogState15bq(const char *stage, const unsigned long long index,
        const u_int estimator, const PathVertexVM &v) {
    if (!MLHeroLegacyHeavyDiagnosticsEnabled15cq()) return;
    if (!MLHero15bqShouldDetail(index))
        return;
    std::lock_guard<std::recursive_mutex> lock15bq(mlHeroDebugLogMutex);
    FILE *f15bq = MLHeroOpenLegacyLog15cq("a");
    if (!f15bq)
        return;
    const char *estimatorName15bq[4] = { "ConnectVertices", "DirectLightSampling", "ConnectToEye", "DirectHitLight" };
    fprintf(f15bq,
        "ML HERO phase 15bq STATE stage=%s index=%llu estimator=%s depth=%u fromLight=%u terminated=%u active=%u laneCount=%u ratio=%.9g heroTP=%.9g\n",
        stage, index, (estimator < 4u) ? estimatorName15bq[estimator] : "n/a",
        v.depth, v.bsdf.hitPoint.fromLight ? 1u : 0u,
        v.mlHeroSecondaryWavelengthsTerminated ? 1u : 0u,
        v.mlHeroActiveWavelengthCount, v.mlHeroLaneCount,
        v.mlHeroTerminationBrightnessRatio,
        (v.mlHeroLaneCount > 0u) ? v.mlHeroLaneThroughput[0].c[0] : 0.f);
    fclose(f15bq);
}

static inline void MLHeroWriteSnapshot15bq(const unsigned long long ticket) {
    if (!MLHeroLegacyHeavyDiagnosticsEnabled15cq()) return;
    if (!MLHero15bqShouldDetail(ticket))
        return;
    std::lock_guard<std::recursive_mutex> lock15bq(mlHeroDebugLogMutex);
    FILE *f15bq = MLHeroOpenLegacyLog15cq("a");
    if (!f15bq)
        return;
    fprintf(f15bq,
        "phase15bq snapshot ticket=%llu stores=%llu propagates=%llu readsCV=%llu readsDLS=%llu readsCTE=%llu readsDHL=%llu invalidRatio=%llu\n",
        ticket,
        mlHero15bqStoreCount.load(std::memory_order_relaxed),
        mlHero15bqPropagateCount.load(std::memory_order_relaxed),
        mlHero15bqReadCount[0].load(std::memory_order_relaxed),
        mlHero15bqReadCount[1].load(std::memory_order_relaxed),
        mlHero15bqReadCount[2].load(std::memory_order_relaxed),
        mlHero15bqReadCount[3].load(std::memory_order_relaxed),
        mlHero15bqInvalidRatioCount.load(std::memory_order_relaxed));
    fclose(f15bq);
}


// ML HERO phase 15br: isolate the effect of the 15ay per-lane Glass weights
// from the later HERO-only xN normalization.  The current render remains
// unchanged.  At each first Mode-1 dispersive termination we compare the
// actually weighted packet/HERO lane against a counterfactual packet in which
// the exact same pre-Glass lane throughputs received LuxCore's shared Glass
// scalar instead.  This directly tests whether the per-lane weighting has
// already shifted energy before the secondary wavelengths are terminated.
static std::atomic<unsigned long long> mlHero15brTerminationCount(0ull);
static std::atomic<unsigned long long> mlHero15brEyeCount(0ull);
static std::atomic<unsigned long long> mlHero15brLightCount(0ull);
static std::atomic<unsigned long long> mlHero15brInvalidCount(0ull);
static std::atomic<unsigned long long> mlHero15brDetailCount(0ull);
static std::atomic<double> mlHero15brHeroRatioSum(0.0);
static std::atomic<double> mlHero15brPacketRatioSum(0.0);
static std::atomic<double> mlHero15brActualHeroSum(0.0);
static std::atomic<double> mlHero15brSharedHeroSum(0.0);
static std::atomic<double> mlHero15brActualPacketAvgSum(0.0);
static std::atomic<double> mlHero15brSharedPacketAvgSum(0.0);
static std::atomic<unsigned long long> mlHero15brHeroOver101(0ull);
static std::atomic<unsigned long long> mlHero15brHeroOver105(0ull);
static std::atomic<unsigned long long> mlHero15brHeroOver125(0ull);
static std::atomic<unsigned long long> mlHero15brHeroOver150(0ull);
static std::atomic<unsigned long long> mlHero15brHeroUnder099(0ull);
static std::atomic<unsigned long long> mlHero15brPacketOver101(0ull);
static std::atomic<unsigned long long> mlHero15brPacketOver105(0ull);
static std::atomic<unsigned long long> mlHero15brPacketOver125(0ull);
static std::atomic<unsigned long long> mlHero15brPacketUnder099(0ull);

static inline void MLHeroRecordContributionWeightedBias15bp(
        const u_int estimator, const Spectrum &c,
        const bool eyeTerminated, const bool lightTerminated,
        const float eyeRatio, const float lightRatio) {
    if (!MLHeroLegacyHeavyDiagnosticsEnabled15cq()) return;
    if ((estimator >= 4u) || (!eyeTerminated && !lightTerminated))
        return;

    u_int category = 0u;
    if (eyeTerminated && lightTerminated)
        category = 3u;
    else if (eyeTerminated)
        category = 1u;
    else
        category = 2u;

    // If both independently terminated subpaths meet, both stochastic packet
    // selections are present in the visible contribution, so use their product
    // as the diagnostic effective ratio.  Single-sided categories use that side.
    const double ratio = eyeTerminated && lightTerminated ?
        (double)eyeRatio * (double)lightRatio :
        (eyeTerminated ? (double)eyeRatio : (double)lightRatio);
    const double r = c.c[0];
    const double g = c.c[1];
    const double b = c.c[2];
    const double y = .2126 * r + .7152 * g + .0722 * b;
    if (!std::isfinite(ratio) || !std::isfinite(y) || (ratio < 0.0))
        return;

    const double absY = fabs(y);
    const double positiveY = (y > 0.0) ? y : 0.0;
    mlHero15bpCount[estimator][category].fetch_add(1ull, std::memory_order_relaxed);
    mlHero15bpRatioSum[estimator][category].fetch_add(ratio, std::memory_order_relaxed);
    mlHero15bpYSum[estimator][category].fetch_add(y, std::memory_order_relaxed);
    mlHero15bpAbsYSum[estimator][category].fetch_add(absY, std::memory_order_relaxed);
    mlHero15bpPositiveYSum[estimator][category].fetch_add(positiveY, std::memory_order_relaxed);
    mlHero15bpRatioTimesY[estimator][category].fetch_add(ratio * y, std::memory_order_relaxed);
    mlHero15bpRatioTimesAbsY[estimator][category].fetch_add(ratio * absY, std::memory_order_relaxed);
    mlHero15bpRatioTimesPositiveY[estimator][category].fetch_add(ratio * positiveY, std::memory_order_relaxed);
    if (ratio > 1.05) mlHero15bpAbsYOver105[estimator][category].fetch_add(absY, std::memory_order_relaxed);
    if (ratio > 1.25) mlHero15bpAbsYOver125[estimator][category].fetch_add(absY, std::memory_order_relaxed);
    if (ratio > 1.50) mlHero15bpAbsYOver150[estimator][category].fetch_add(absY, std::memory_order_relaxed);
    if (ratio < 0.95) mlHero15bpAbsYUnder095[estimator][category].fetch_add(absY, std::memory_order_relaxed);
}

static inline u_int MLHeroTerminationCategory15bj(
		const bool eyeTerminated, const bool lightTerminated) {
	if (eyeTerminated && lightTerminated)
		return 3u;
	else if (eyeTerminated)
		return 1u;
	else if (lightTerminated)
		return 2u;
	return 0u;
}

static inline void MLHeroRecordEstimatorContribution15bj(
		const u_int estimator, const Spectrum &c,
		const bool eyeTerminated, const bool lightTerminated,
		const bool packetDivided, const bool classicFallback) {
    if (!MLHeroLegacyHeavyDiagnosticsEnabled15cq()) return;
	if (estimator >= 4u)
		return;
	const u_int category = MLHeroTerminationCategory15bj(
		eyeTerminated, lightTerminated);
	const double r = c.c[0];
	const double g = c.c[1];
	const double b = c.c[2];
	const double y = .2126 * r + .7152 * g + .0722 * b;

	mlHero15bjEstimatorCount[estimator][category].fetch_add(
		1ull, std::memory_order_relaxed);
	if (packetDivided)
		mlHero15bjEstimatorPacketDivCount[estimator][category].fetch_add(
			1ull, std::memory_order_relaxed);
	if (classicFallback)
		mlHero15bjEstimatorClassicCount[estimator][category].fetch_add(
			1ull, std::memory_order_relaxed);

	mlHero15bjEstimatorR[estimator][category].fetch_add(r, std::memory_order_relaxed);
	mlHero15bjEstimatorG[estimator][category].fetch_add(g, std::memory_order_relaxed);
	mlHero15bjEstimatorB[estimator][category].fetch_add(b, std::memory_order_relaxed);
	mlHero15bjEstimatorY[estimator][category].fetch_add(y, std::memory_order_relaxed);
	mlHero15bjEstimatorAbsY[estimator][category].fetch_add(fabs(y), std::memory_order_relaxed);

	// Phase 15bk: compare the raw RGB contribution with a hypothetical
	// positive-only RGB clamp. This does NOT alter the contribution.
	if (std::isfinite(r) && std::isfinite(g) && std::isfinite(b) && std::isfinite(y)) {
		mlHero15bkFiniteCount[estimator][category].fetch_add(
			1ull, std::memory_order_relaxed);

		const bool negR = r < 0.0;
		const bool negG = g < 0.0;
		const bool negB = b < 0.0;
		const unsigned int negativeChannels =
			(negR ? 1u : 0u) + (negG ? 1u : 0u) + (negB ? 1u : 0u);
		if (negativeChannels > 0u)
			mlHero15bkAnyNegativeCount[estimator][category].fetch_add(
				1ull, std::memory_order_relaxed);
		if (negativeChannels > 0u)
			mlHero15bkNegativeChannelCount[estimator][category].fetch_add(
				(unsigned long long)negativeChannels, std::memory_order_relaxed);

		const double cr = (r > 0.0) ? r : 0.0;
		const double cg = (g > 0.0) ? g : 0.0;
		const double cb = (b > 0.0) ? b : 0.0;
		const double clampedY = .2126 * cr + .7152 * cg + .0722 * cb;

		mlHero15bkRawY[estimator][category].fetch_add(y, std::memory_order_relaxed);
		mlHero15bkRawAbsY[estimator][category].fetch_add(fabs(y), std::memory_order_relaxed);
		mlHero15bkPositiveClampedY[estimator][category].fetch_add(
			clampedY, std::memory_order_relaxed);
		mlHero15bkPositiveClampedAbsY[estimator][category].fetch_add(
			fabs(clampedY), std::memory_order_relaxed);
		mlHero15bkClampDeltaY[estimator][category].fetch_add(
			clampedY - y, std::memory_order_relaxed);
	} else {
		mlHero15bkNonFiniteCount[estimator][category].fetch_add(
			1ull, std::memory_order_relaxed);
	}
}

static inline u_int MLHeroGlassCaseIndex15az(const bool fromLight, const BSDFEvent event) {
	if (event & TRANSMIT)
		return fromLight ? 1u : 0u;
	return fromLight ? 3u : 2u;
}

static inline const char *MLHeroGlassCaseName15az(const u_int index) {
	static const char *names[4] = {
		"EYE_TRANSMIT", "LIGHT_TRANSMIT", "EYE_REFLECT", "LIGHT_REFLECT"
	};
	return names[(index < 4u) ? index : 0u];
}

static inline void MLHeroInsertGlassOutlier15ba(
		MLHeroGlassCorrectionStats15az *stats,
		const MLHeroGlassOutlier15ba &candidate) {
	if (!candidate.valid)
		return;

	for (u_int pos = 0u; pos < 16u; ++pos) {
		if (!stats->topOutliers[pos].valid ||
				(candidate.absDelta > stats->topOutliers[pos].absDelta)) {
			for (u_int j = 15u; j > pos; --j)
				stats->topOutliers[j] = stats->topOutliers[j - 1u];
			stats->topOutliers[pos] = candidate;
			return;
		}
	}
}

// ML HERO phase 15bx:
// Mode-1 HERO-only dispersive termination is independent from the runtime
// Glass per-lane weighting switch.  The switch now controls only which Glass
// throughput multiplier is applied before termination; it no longer disables
// the actual HERO-only wavelength termination.
// Quarter-phase cycling remains an independent runtime property.

// ML HERO phase 15be:
// Render-mode switch for dispersive specular Glass.
// 0 = current per-lane Glass weighting (phase 15ay behavior)
// 1 = HERO-only after a dispersive SPECULAR TRANSMIT bounce
//     (secondary wavelength lanes are terminated; lane 0 is scaled by the
//      packet size so the existing 1/N packet reconstruction remains normalized)
// 2 = experimental lane-event-PDF correction approximation
//     (same shared HERO geometry; NOT a full path-space spectral MIS solution)

u_int GetMLHeroWavelengthCount();
float GetMLHeroWaveLengthAt(const u_int lane);
float GetMLHeroSampleWeightAt(const u_int lane);
luxrays::Spectrum GetMLHeroSampleColorAt(const u_int lane);

// ML HERO phase 15g: sample the experimental 380..780 nm illuminant basis.
// This reproduces RGBIllumSPD's Smits mixing logic, but evaluates our
// separate ML-HERO table. It is diagnostic only in phase 15g.
static float MLHeroRGBIllumSample380_780(const RGBColor &s, const float lambda) {
	if ((lambda < mlhero_illumrgb2spect_start) || (lambda > mlhero_illumrgb2spect_end))
		return 0.f;

	const float delta = (mlhero_illumrgb2spect_end - mlhero_illumrgb2spect_start) /
			(mlhero_illumrgb2spect_bins - 1);
	const float pos = (lambda - mlhero_illumrgb2spect_start) / delta;
	u_int i0 = Min((u_int)Floor2UInt(pos), mlhero_illumrgb2spect_bins - 1);
	u_int i1 = Min(i0 + 1, mlhero_illumrgb2spect_bins - 1);
	const float t = Clamp(pos - i0, 0.f, 1.f);

	auto evalBin = [&s](const u_int i) {
		float v = 0.f;
		const float r = s.c[0];
		const float g = s.c[1];
		const float b = s.c[2];

		auto add = [&v, i](const float w, const float *basis) {
			v += w * basis[i];
		};

		if ((r <= g) && (r <= b)) {
			add(r, mlhero_illumrgb2spect_white);
			if (g <= b) {
				add(g - r, mlhero_illumrgb2spect_cyan);
				add(b - g, mlhero_illumrgb2spect_blue);
			} else {
				add(b - r, mlhero_illumrgb2spect_cyan);
				add(g - b, mlhero_illumrgb2spect_green);
			}
		} else if ((g <= r) && (g <= b)) {
			add(g, mlhero_illumrgb2spect_white);
			if (r <= b) {
				add(r - g, mlhero_illumrgb2spect_magenta);
				add(b - r, mlhero_illumrgb2spect_blue);
			} else {
				add(b - g, mlhero_illumrgb2spect_magenta);
				add(r - b, mlhero_illumrgb2spect_red);
			}
		} else {
			add(b, mlhero_illumrgb2spect_white);
			if (r <= g) {
				add(r - b, mlhero_illumrgb2spect_yellow);
				add(g - r, mlhero_illumrgb2spect_green);
			} else {
				add(g - b, mlhero_illumrgb2spect_yellow);
				add(r - g, mlhero_illumrgb2spect_red);
			}
		}

		v *= mlhero_illumrgb2spect_scale;
		return Max(0.f, v);
	};

	const float v0 = evalBin(i0);
	const float v1 = evalBin(i1);
	return Lerp(t, v0, v1);
}

static void MLHeroInitLaneThroughput(PathVertexVM *vertex, const Spectrum &initial) {
	vertex->mlHeroLaneCount = Clamp(GetMLHeroWavelengthCount(), 1u, 8u);
	for (u_int lane = 0; lane < vertex->mlHeroLaneCount; ++lane) {
		vertex->mlHeroLaneThroughput[lane] = initial;
		vertex->mlHeroLaneWaveLength[lane] = GetMLHeroWaveLengthAt(lane);
		vertex->mlHeroLaneSampleWeight[lane] = GetMLHeroSampleWeightAt(lane);
		const float sampleWeight15bh = vertex->mlHeroLaneSampleWeight[lane];
		vertex->mlHeroLaneWaveLengthPdf[lane] = (sampleWeight15bh > 1e-20f) ?
			(1.f / (400.f * sampleWeight15bh)) : 0.f;
		vertex->mlHeroLaneGlassEventPdfProduct[lane] = 1.0;
		vertex->mlHeroLaneDiracSupportPdfProduct[lane] = 1.0;
	}
	for (u_int lane = vertex->mlHeroLaneCount; lane < 8u; ++lane) {
		vertex->mlHeroLaneWaveLengthPdf[lane] = 0.f;
		vertex->mlHeroLaneGlassEventPdfProduct[lane] = 0.0;
		vertex->mlHeroLaneDiracSupportPdfProduct[lane] = 0.0;
	}
	vertex->mlHeroSecondaryWavelengthsTerminated = false;
	vertex->mlHeroActiveWavelengthCount = vertex->mlHeroLaneCount;
	vertex->mlHeroTerminationBrightnessRatio = 1.f;
	vertex->mlHeroTerminationHeroLambda = 0.f;
	vertex->mlHeroTerminationPacketAverage = 0.f;
	vertex->mlHeroTerminationHeroScalar = 0.f;
	vertex->mlHeroTerminationLaneSelectionPdf = 0.f;
	vertex->mlHeroTerminationWavelengthPdf = 0.f;
	vertex->mlHeroTerminationCurrentScale = 0.f;
	vertex->mlHero15buReferenceThroughput = 0.0;
	vertex->mlHeroDebugTracked = false;
	vertex->mlHeroDebugLoggedBounces = 0u;
}


// ML HERO phase 15i:
// Initialize LIGHT-path lanes from the emitted RGB using the experimental
// 380..780 nm illuminant SPD. The classic RGB throughput remains untouched.
// Each lane stores a scalar spectral throughput in the existing Spectrum
// container as (s,s,s), so later diagnostics can detect when an RGB factor
// accidentally re-introduces color into a wavelength lane.
static void MLHeroInitSpectralLightLaneThroughput(PathVertexVM *vertex,
		const Spectrum &rawEmit, const float lightEmitPdfW) {
	vertex->mlHeroLaneCount = Clamp(GetMLHeroWavelengthCount(), 1u, 8u);

	for (u_int lane = 0; lane < vertex->mlHeroLaneCount; ++lane) {
		const float lambda = GetMLHeroWaveLengthAt(lane);
		const float sampleWeight = GetMLHeroSampleWeightAt(lane);
		const float rawSPD = MLHeroRGBIllumSample380_780(rawEmit, lambda);
		const float scalar = (lightEmitPdfW > 0.f) ? (rawSPD / lightEmitPdfW) : 0.f;
		vertex->mlHeroLaneThroughput[lane] = Spectrum(scalar);
		vertex->mlHeroLaneWaveLength[lane] = lambda;
		vertex->mlHeroLaneSampleWeight[lane] = sampleWeight;
		vertex->mlHeroLaneWaveLengthPdf[lane] = (sampleWeight > 1e-20f) ?
			(1.f / (400.f * sampleWeight)) : 0.f;
		vertex->mlHeroLaneGlassEventPdfProduct[lane] = 1.0;
		vertex->mlHeroLaneDiracSupportPdfProduct[lane] = 1.0;
	}
	for (u_int lane = vertex->mlHeroLaneCount; lane < 8u; ++lane) {
		vertex->mlHeroLaneWaveLengthPdf[lane] = 0.f;
		vertex->mlHeroLaneGlassEventPdfProduct[lane] = 0.0;
		vertex->mlHeroLaneDiracSupportPdfProduct[lane] = 0.0;
	}
	vertex->mlHeroSecondaryWavelengthsTerminated = false;
	vertex->mlHeroActiveWavelengthCount = vertex->mlHeroLaneCount;
	vertex->mlHeroTerminationBrightnessRatio = 1.f;
	vertex->mlHeroTerminationHeroLambda = 0.f;
	vertex->mlHeroTerminationPacketAverage = 0.f;
	vertex->mlHeroTerminationHeroScalar = 0.f;
	vertex->mlHeroTerminationLaneSelectionPdf = 0.f;
	vertex->mlHeroTerminationWavelengthPdf = 0.f;
	vertex->mlHeroTerminationCurrentScale = 0.f;
	vertex->mlHero15buReferenceThroughput = 0.0;

	vertex->mlHeroDebugTracked = false;
	vertex->mlHeroDebugLoggedBounces = 0u;
}


static void MLHeroMultiplyLaneThroughput(PathVertexVM *vertex, const Spectrum &factor) {
	for (u_int lane = 0; lane < vertex->mlHeroLaneCount; ++lane)
		vertex->mlHeroLaneThroughput[lane] *= factor;
}

static void MLHeroDivideLaneThroughput(PathVertexVM *vertex, const float factor) {
	for (u_int lane = 0; lane < vertex->mlHeroLaneCount; ++lane)
		vertex->mlHeroLaneThroughput[lane] /= factor;
}

// ML HERO phase 15w:
// LuxCore's stock RGBReflSPD table ends at 720 nm. SPD::Sample() returns 0
// outside that range, which would incorrectly kill HERO lanes above 720 nm.
// Keep the original 380..720 nm reconstruction untouched and extend the
// far-red tail to 780 nm by holding the 720 nm endpoint value.
static inline float MLHeroRGBReflSample380_780(const RGBReflSPD &spd,
		const float waveLength) {
	// RGBReflSPD::Sample() treats the upper table edge as out-of-range, so
	// sampling exactly 720 nm can return zero.  Use a point just inside the
	// stock table and hold that endpoint through the HERO 720..780 nm tail.
	return spd.Sample(Clamp(waveLength, 380.f, 719.999f));
}

// ML HERO phase 15cc:
// Controlled Matte experiment.  Reflectance is physically non-negative, while
// the stock RGB->Smits reconstruction can contain small negative lobes for
// saturated RGB colours.  Clamp only MATTE here; all other generic reflective
// materials keep their previous 15v behaviour.
static inline float MLHeroMatteReflSampleNonNegative380_780(
		const RGBReflSPD &spd, const float waveLength) {
	return Max(0.f, MLHeroRGBReflSample380_780(spd, waveLength));
}

// ML HERO phase 15cj: experimental Matte basis compensation.
// HERO_75 measured the effective clamped RGB->spectral->RGB basis and its
// inverse. For this A/B test, apply that inverse to the Matte RGB multiplier
// and rebuild the wavelength-domain reflectance as a linear combination of
// the three unit-primary RGBReflSPD spectra. The final reflectance is clamped
// non-negative, matching the existing Matte rule.
static inline float MLHeroMatteBasisCompensatedSample15cj(
		const Spectrum &rgb, const float waveLength) {
	// Columns of the inverse CLAMP_basis measured in phase15ci.
	const float coeffR = 0.81789457798f * rgb.c[0] +
		0.006252348423f * rgb.c[1] + 0.007401828188f * rgb.c[2];
	const float coeffG = 0.02175390534f * rgb.c[0] +
		1.00719189644f * rgb.c[1] + 0.01943249628f * rgb.c[2];
	const float coeffB = -0.000957758341f * rgb.c[0] -
		0.02197206020f * rgb.c[1] + 1.10662686825f * rgb.c[2];

	static const RGBReflSPD redSPD15cj(RGBColor(1.f, 0.f, 0.f));
	static const RGBReflSPD greenSPD15cj(RGBColor(0.f, 1.f, 0.f));
	static const RGBReflSPD blueSPD15cj(RGBColor(0.f, 0.f, 1.f));

	const float raw =
		coeffR * MLHeroRGBReflSample380_780(redSPD15cj, waveLength) +
		coeffG * MLHeroRGBReflSample380_780(greenSPD15cj, waveLength) +
		coeffB * MLHeroRGBReflSample380_780(blueSPD15cj, waveLength);
	return Max(0.f, raw);
}

// ML HERO phase 15cn: use the same measured CLAMP-basis inverse for the
// generic HERO reflectance fallback. This is deliberately HERO-local: stock
// RGBReflSPD and all non-HERO LuxCore rendering remain unchanged. Matte keeps
// its independent 15cj switch so the two experiments can be compared cleanly.
static inline float MLHeroGenericBasisCompensatedSample15cn(
		const Spectrum &rgb, const float waveLength) {
	return MLHeroMatteBasisCompensatedSample15cj(rgb, waveLength);
}

// ML HERO phase 15aj:
// A material reflectance cannot physically be negative. The stock Smits
// RGBReflSPD basis can produce small negative lobes for saturated RGB colors.
// For the controlled Glossy2 Kd path only, test a non-negative reconstruction.
// Keep the generic whole-BSDF fallback unchanged so this experiment stays
// isolated to actual material reflectance.
static inline float MLHeroGlossy2ReflSampleNonNegative380_780(
		const RGBReflSPD &spd, const float waveLength) {
	return Max(0.f, spd.Sample(Clamp(waveLength, 380.f, 720.f)));
}

// ML HERO phase 15cp: Glossy2 Kd-only basis compensation.  This intentionally
// reuses the measured 15ci CLAMP-basis inverse, but applies it only to the
// diffuse/base Kd reflectance component of the dedicated Glossy2 split.
// Coating, Schlick/Fresnel support and residual transport stay unchanged.
static inline float MLHeroGlossy2BasisCompensatedSample15cp(
		const Spectrum &kd, const float waveLength) {
	return MLHeroMatteBasisCompensatedSample15cj(kd, waveLength);
}

static bool MLHeroIsScalarSpectrum(const Spectrum &s, const float eps = 1e-6f) {
	const float m = Max(1.f, Max(fabsf(s.c[0]), Max(fabsf(s.c[1]), fabsf(s.c[2]))));
	return (fabsf(s.c[0] - s.c[1]) <= eps * m) && (fabsf(s.c[0] - s.c[2]) <= eps * m);
}

// ML HERO phase 15al:
// Glossy2 support terms such as Ks and SchlickS can be intended achromatic
// but differ by a few ULPs after color-management/material math. For the
// controlled Glossy2 decomposition only, use a genuinely relative scalar test.
// This is intentionally NOT used by the generic HERO scalar classification.
static bool MLHeroIsNearScalarGlossy2Support(const Spectrum &s,
		const float relEps = 1e-4f) {
	const float maxAbs = Max(fabsf(s.c[0]), Max(fabsf(s.c[1]), fabsf(s.c[2])));
	const float scale = Max(1e-8f, maxAbs);
	const float spread = Max(fabsf(s.c[0] - s.c[1]),
		Max(fabsf(s.c[0] - s.c[2]), fabsf(s.c[1] - s.c[2])));
	return spread <= relEps * scale;
}

static bool MLHeroAllLanesScalar(const PathVertexVM &vertex) {
	for (u_int lane = 0; lane < vertex.mlHeroLaneCount; ++lane) {
		if (!MLHeroIsScalarSpectrum(vertex.mlHeroLaneThroughput[lane]))
			return false;
	}
	return true;
}

// ML HERO phase 15j fix2:
// Low-overhead multi-bounce scalar-purity trace.
// Log only the FIRST occurrence of each (stage, depth) pair globally.
// This avoids per-path file I/O, which previously throttled rendering badly.
static void MLHeroLogLightScalarTrace(const PathVertexVM &vertex, const char *stage) {
    if (!MLHeroLegacyHeavyDiagnosticsEnabled15cq()) return;
	if (GetMLHeroWavelengthCount() <= 1u)
		return;
	if ((vertex.depth < 1u) || (vertex.depth > 12u))
		return;

	// 3 stages x 13 depth slots. Bit set = already logged.
	// stage ids: after-connection=0, before-bounce=1, after-bounce=2
	u_int stageId = 3u;
	if (!strcmp(stage, "after-connection"))
		stageId = 0u;
	else if (!strcmp(stage, "before-bounce"))
		stageId = 1u;
	else if (!strcmp(stage, "after-bounce"))
		stageId = 2u;
	if (stageId > 2u)
		return;

	const u_int bitIndex = stageId * 13u + vertex.depth;
	const uint64_t bit = 1ull << bitIndex;

	static std::atomic<uint64_t> loggedMask(0ull);
	uint64_t oldMask = loggedMask.load(std::memory_order_relaxed);
	while (true) {
		if (oldMask & bit)
			return;
		if (loggedMask.compare_exchange_weak(oldMask, oldMask | bit,
				std::memory_order_relaxed, std::memory_order_relaxed))
			break;
	}

	const bool allScalar = MLHeroAllLanesScalar(vertex);

	std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock0(mlHeroDebugLogMutex);
	FILE *f = MLHeroOpenLegacyLog15cq("a");
	if (!f)
		return;

	fprintf(f, "ML HERO phase 15j LIGHT TRACE: stage=%s depth=%u count=%u allScalar=%u\n",
		stage, vertex.depth, vertex.mlHeroLaneCount, allScalar ? 1u : 0u);

	for (u_int lane = 0; lane < vertex.mlHeroLaneCount; ++lane) {
		const Spectrum &s = vertex.mlHeroLaneThroughput[lane];
		fprintf(f,
			"phase15j lane %u: lambda=%.9g scalar=%u TP=(%.9g, %.9g, %.9g)\n",
			lane, vertex.mlHeroLaneWaveLength[lane],
			MLHeroIsScalarSpectrum(s) ? 1u : 0u,
			s.c[0], s.c[1], s.c[2]);
	}

	if (!allScalar) {
		static std::atomic<bool> lostLogged(false);
		bool expected = false;
		if (lostLogged.compare_exchange_strong(expected, true))
			fprintf(f, "ML HERO phase 15j FIRST SCALAR PURITY LOSS: stage=%s depth=%u\n",
				stage, vertex.depth);
	}

	fclose(f);
}

static void MLHeroLogLanePurityBlock(const char *label, const PathVertexVM &vertex) {
    if (!MLHeroLegacyHeavyDiagnosticsEnabled15cq()) return;
	std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock1(mlHeroDebugLogMutex);
	FILE *f = MLHeroOpenLegacyLog15cq("a");
	if (!f)
		return;
	fprintf(f, "ML HERO phase 15d %s: depth=%u count=%u allScalar=%u\n", label, vertex.depth, vertex.mlHeroLaneCount, MLHeroAllLanesScalar(vertex) ? 1u : 0u);
	for (u_int lane = 0; lane < vertex.mlHeroLaneCount; ++lane) {
		const Spectrum &s = vertex.mlHeroLaneThroughput[lane];
		fprintf(f, "phase15d lane %u: lambda=%.9g nm scalar=%u throughput=(%.9g, %.9g, %.9g)\n",
			lane, vertex.mlHeroLaneWaveLength[lane], MLHeroIsScalarSpectrum(s) ? 1u : 0u, s.c[0], s.c[1], s.c[2]);
	}
	fclose(f);
}

// Phase 8 diagnostic estimator. Reconstruct one RGB contribution from the
// per-wavelength lane throughputs without touching the classic BIDIR result.
// Each lane is converted through the same wavelength-to-RGB estimator used by
// the existing single-HERO path, then the packet is averaged.
static Spectrum MLHeroEstimateLaneRadiance(const PathVertexVM &vertex) {
	if (vertex.mlHeroLaneCount <= 1u)
		return vertex.throughput;

	Spectrum result;
	for (u_int lane = 0; lane < vertex.mlHeroLaneCount; ++lane)
		result += vertex.mlHeroLaneThroughput[lane] * GetMLHeroSampleColorAt(lane);

	return result / (float)vertex.mlHeroLaneCount;
}


// ML HERO phase 15h:
// Convert one wavelength sample to the RGB contribution implied by LuxCore's
// own CIE XYZ matching functions and DefaultColorSystem.
//
// For uniform sampling over 380..780 nm, the Monte Carlo estimator is:
//
//   RGB ~= (1/N) * sum_i [ SPD(lambda_i) *
//          RGB(683 * 400 * CIE_XYZ(lambda_i)) * sampleWeight_i ]
//
// This mirrors SPD::ToXYZ() (1 nm sum * 683) but evaluates it with HERO lanes.
static Spectrum MLHeroCIEEstimatorColor(const float lambda, const float sampleWeight) {
	if ((lambda < 380.f) || (lambda > 780.f))
		return Spectrum();

	const float ciePos = lambda - (float)CIEstart;
	const u_int i0 = Min((u_int)Floor2UInt(ciePos), nCIE - 1);
	const u_int i1 = Min(i0 + 1u, nCIE - 1);
	const float t = Clamp(ciePos - i0, 0.f, 1.f);

	const float x = Lerp(t, CIE_X[i0], CIE_X[i1]);
	const float y = Lerp(t, CIE_Y[i0], CIE_Y[i1]);
	const float z = Lerp(t, CIE_Z[i0], CIE_Z[i1]);

	// 400 nm is the width of the HERO interval [380, 780].
	const XYZColor xyz = XYZColor(x, y, z) * (683.f * 400.f);
	const RGBColor rgb = ColorSystem::DefaultColorSystem.ToRGB(xyz);

	return Spectrum(rgb.c[0], rgb.c[1], rgb.c[2]) * sampleWeight;
}

// ML HERO phase 15an:
// A reflectance SPD is dimensionless. LuxCore's SPD::ToNormalizedXYZ() does
// NOT apply the radiometric 683 lm/W factor used by SPD::ToXYZ(); instead it
// divides the CIE integral by the integral of CIE_Y. For a HERO Monte Carlo
// estimate over [380,780], the matching per-lane factor is therefore
// 400 / sum(CIE_Y), with the lane importance-sampling correction applied
// separately. Diagnostic only for now.
static Spectrum MLHeroCIENormalizedReflectanceEstimatorColor(
		const float lambda, const float sampleWeight) {
	if ((lambda < 380.f) || (lambda > 780.f))
		return Spectrum();

	const float ciePos = lambda - (float)CIEstart;
	const u_int i0 = Min((u_int)Floor2UInt(ciePos), nCIE - 1);
	const u_int i1 = Min(i0 + 1u, nCIE - 1);
	const float t = Clamp(ciePos - i0, 0.f, 1.f);

	const float x = Lerp(t, CIE_X[i0], CIE_X[i1]);
	const float y = Lerp(t, CIE_Y[i0], CIE_Y[i1]);
	const float z = Lerp(t, CIE_Z[i0], CIE_Z[i1]);

	float cieYIntegral = 0.f;
	for (u_int i = 0; i < nCIE; ++i)
		cieYIntegral += CIE_Y[i];

	const float normalization = 400.f / Max(1e-20f, cieYIntegral);
	const XYZColor xyz = XYZColor(x, y, z) * normalization;
	const RGBColor rgb = ColorSystem::DefaultColorSystem.ToRGB(xyz);

	return Spectrum(rgb.c[0], rgb.c[1], rgb.c[2]) * sampleWeight;
}

// Deterministic 1 nm A/B reconstruction used only once per encountered material
// type. It is diagnostic-only and therefore cheap enough for FAST BUILD.
static Spectrum MLHeroIntegrateGenericCompensated15cn(const Spectrum &rgb) {
	Spectrum sum;
	for (u_int i = 0u; i <= 400u; ++i) {
		const float lambda = 380.f + (float)i;
		const float trapWeight = ((i == 0u) || (i == 400u)) ? .5f : 1.f;
		const float refl = MLHeroGenericBasisCompensatedSample15cn(rgb, lambda);
		const Spectrum cie = MLHeroCIENormalizedReflectanceEstimatorColor(lambda, 1.f);
		sum += cie * (refl * (trapWeight / 400.f));
	}
	return sum;
}

// ML HERO phase 15cf:
// Deterministic reference integration for Matte RGB reflectance reconstruction.
// The per-wavelength normalized CIE helper is the same conversion used by the
// packet diagnostic, but here it is integrated at 1 nm with trapezoidal endpoint
// weights. Dividing by 400 converts the helper's uniform-MC 400 nm factor back
// to the corresponding 1 nm integral contribution.
static Spectrum MLHeroIntegrateMatteReflectance15cf(const RGBReflSPD &spd,
		const bool nonNegative) {
	Spectrum sum15cf;
	for (u_int i15cf = 0u; i15cf <= 400u; ++i15cf) {
		const float lambda15cf = 380.f + (float)i15cf;
		const float trapWeight15cf = ((i15cf == 0u) || (i15cf == 400u)) ? .5f : 1.f;
		const float refl15cf = nonNegative ?
			MLHeroMatteReflSampleNonNegative380_780(spd, lambda15cf) :
			MLHeroRGBReflSample380_780(spd, lambda15cf);
		const Spectrum cie15cf = MLHeroCIENormalizedReflectanceEstimatorColor(lambda15cf, 1.f);
		sum15cf += cie15cf * (refl15cf * (trapWeight15cf / 400.f));
	}
	return sum15cf;
}

// ML HERO phase 15cg:
// Decompose the same deterministic Matte reconstruction one stage earlier,
// before DefaultColorSystem::ToRGB(). This lets us distinguish an RGBReflSPD
// basis/normalization mismatch from the XYZ->RGB matrix itself.
static XYZColor MLHeroIntegrateMatteReflectanceXYZ15cg(const RGBReflSPD &spd,
		const bool nonNegative) {
	float cieYIntegral15cg = 0.f;
	for (u_int i15cg = 0u; i15cg < nCIE; ++i15cg)
		cieYIntegral15cg += CIE_Y[i15cg];
	const float invY15cg = 1.f / Max(1e-20f, cieYIntegral15cg);

	float x15cg = 0.f, y15cg = 0.f, z15cg = 0.f;
	for (u_int i15cg = 0u; i15cg <= 400u; ++i15cg) {
		const float lambda15cg = 380.f + (float)i15cg;
		const float trapWeight15cg = ((i15cg == 0u) || (i15cg == 400u)) ? .5f : 1.f;
		const float refl15cg = nonNegative ?
			MLHeroMatteReflSampleNonNegative380_780(spd, lambda15cg) :
			MLHeroRGBReflSample380_780(spd, lambda15cg);

		const float ciePos15cg = lambda15cg - (float)CIEstart;
		const u_int i015cg = Min((u_int)Floor2UInt(ciePos15cg), nCIE - 1);
		const u_int i115cg = Min(i015cg + 1u, nCIE - 1);
		const float t15cg = Clamp(ciePos15cg - i015cg, 0.f, 1.f);
		x15cg += refl15cg * Lerp(t15cg, CIE_X[i015cg], CIE_X[i115cg]) * trapWeight15cg;
		y15cg += refl15cg * Lerp(t15cg, CIE_Y[i015cg], CIE_Y[i115cg]) * trapWeight15cg;
		z15cg += refl15cg * Lerp(t15cg, CIE_Z[i015cg], CIE_Z[i115cg]) * trapWeight15cg;
	}
	return XYZColor(x15cg * invY15cg, y15cg * invY15cg, z15cg * invY15cg);
}

static Spectrum MLHeroXYZToRGB15cg(const XYZColor &xyz15cg) {
	const RGBColor rgb15cg = ColorSystem::DefaultColorSystem.ToRGB(xyz15cg);
	return Spectrum(rgb15cg.c[0], rgb15cg.c[1], rgb15cg.c[2]);
}

// ML HERO phase 15ch:
// Diagnostic-only white-point remap. Recover the XYZ white expected by the
// active DefaultColorSystem by inverting its XYZ->RGB matrix for RGB=(1,1,1),
// then compare a simple diagonal XYZ white remap against the current path.
static XYZColor MLHeroDefaultRGBWhiteXYZ15ch() {
	const Spectrum cx = MLHeroXYZToRGB15cg(XYZColor(1.f, 0.f, 0.f));
	const Spectrum cy = MLHeroXYZToRGB15cg(XYZColor(0.f, 1.f, 0.f));
	const Spectrum cz = MLHeroXYZToRGB15cg(XYZColor(0.f, 0.f, 1.f));
	const double a00=cx.c[0], a01=cy.c[0], a02=cz.c[0];
	const double a10=cx.c[1], a11=cy.c[1], a12=cz.c[1];
	const double a20=cx.c[2], a21=cy.c[2], a22=cz.c[2];
	const double det = a00*(a11*a22-a12*a21)-a01*(a10*a22-a12*a20)+a02*(a10*a21-a11*a20);
	if (fabs(det) < 1e-20)
		return XYZColor(1.f, 1.f, 1.f);
	const double invDet=1.0/det;
	const double x=((a11*a22-a12*a21) + (a02*a21-a01*a22) + (a01*a12-a02*a11))*invDet;
	const double y=((a12*a20-a10*a22) + (a00*a22-a02*a20) + (a02*a10-a00*a12))*invDet;
	const double z=((a10*a21-a11*a20) + (a01*a20-a00*a21) + (a00*a11-a01*a10))*invDet;
	return XYZColor((float)x,(float)y,(float)z);
}

static XYZColor MLHeroDiagonalWhiteRemap15ch(const XYZColor &xyz,
		const XYZColor &sourceWhite, const XYZColor &targetWhite) {
	return XYZColor(
		xyz.c[0] * targetWhite.c[0] / Max(1e-20f, sourceWhite.c[0]),
		xyz.c[1] * targetWhite.c[1] / Max(1e-20f, sourceWhite.c[1]),
		xyz.c[2] * targetWhite.c[2] / Max(1e-20f, sourceWhite.c[2]));
}

// ML HERO phase 15ci:
// Treat the continuous RGBReflSPD -> normalized CIE XYZ -> DefaultColorSystem RGB
// reconstruction of unit R/G/B as an effective 3x3 basis matrix. Diagnostic only.
// Inverting that matrix lets us test whether one global linear RGB correction can
// recover arbitrary RGB inputs, or whether RGBReflSPD's piecewise basis introduces
// residual nonlinear/color-order dependence.
static Spectrum MLHeroApplyRGBMatrix15ci(const Spectrum &c0, const Spectrum &c1,
		const Spectrum &c2, const Spectrum &v) {
	return c0 * v.c[0] + c1 * v.c[1] + c2 * v.c[2];
}

static bool MLHeroInvertRGBMatrix15ci(const Spectrum &c0, const Spectrum &c1,
		const Spectrum &c2, Spectrum *ic0, Spectrum *ic1, Spectrum *ic2, double *detOut) {
	const double a00=c0.c[0], a01=c1.c[0], a02=c2.c[0];
	const double a10=c0.c[1], a11=c1.c[1], a12=c2.c[1];
	const double a20=c0.c[2], a21=c1.c[2], a22=c2.c[2];
	const double det=a00*(a11*a22-a12*a21)-a01*(a10*a22-a12*a20)+a02*(a10*a21-a11*a20);
	if (detOut) *detOut=det;
	if (fabs(det)<1e-20) {
		*ic0=Spectrum(); *ic1=Spectrum(); *ic2=Spectrum();
		return false;
	}
	const double d=1.0/det;
	// Inverse columns, so MLHeroApplyRGBMatrix15ci() can use them directly.
	*ic0=Spectrum((float)((a11*a22-a12*a21)*d),
		(float)((a12*a20-a10*a22)*d),
		(float)((a10*a21-a11*a20)*d));
	*ic1=Spectrum((float)((a02*a21-a01*a22)*d),
		(float)((a00*a22-a02*a20)*d),
		(float)((a01*a20-a00*a21)*d));
	*ic2=Spectrum((float)((a01*a12-a02*a11)*d),
		(float)((a02*a10-a00*a12)*d),
		(float)((a00*a11-a01*a10)*d));
	return true;
}

static Spectrum MLHeroContinuousMatteRGB15ci(const Spectrum &rgb, const bool nonNegative) {
	const RGBReflSPD spd15ci(RGBColor(rgb.c[0], rgb.c[1], rgb.c[2]));
	return MLHeroXYZToRGB15cg(MLHeroIntegrateMatteReflectanceXYZ15cg(spd15ci, nonNegative));
}

//------------------------------------------------------------------------------
// ML HERO Sampling 2.0 / 3.0 for CPU BIDIR.
// Same 16-bin distributions and PDF correction as CPU Path O1/O2.
//------------------------------------------------------------------------------
static float MLBidirHeroSampling2Sample(const float u, float *sampleWeight) {
    static const float binProb[16] = {
        0.0126331286f, 0.0133533552f, 0.0166194938f, 0.0275044480f,
        0.0568548852f, 0.1352293344f, 0.1936849374f, 0.1927480102f,
        0.1493873570f, 0.0880972731f, 0.0418778055f, 0.0203700156f,
        0.0139428652f, 0.0126807265f, 0.0125154610f, 0.0125009033f
    };

    const float uu = Clamp(u, 0.f, 0.99999994f);
    float cdf0 = 0.f;
    for (u_int i = 0; i < 16u; ++i) {
        const float cdf1 = cdf0 + binProb[i];
        if ((uu < cdf1) || (i == 15u)) {
            const float localU = Clamp((uu - cdf0) / binProb[i], 0.f, 0.99999994f);
            *sampleWeight = 1.f / (16.f * binProb[i]);
            return 380.f + (i + localU) * 25.f;
        }
        cdf0 = cdf1;
    }
    *sampleWeight = 1.f;
    return 780.f;
}

static float MLBidirHeroSampling3Sample(const float u, float *sampleWeight) {
    static const float binProb[16] = {
        0.05676103f, 0.08028577f, 0.08038660f, 0.10020402f,
        0.10460621f, 0.06266096f, 0.07141445f, 0.08194949f,
        0.07852106f, 0.06112917f, 0.04512863f, 0.04199809f,
        0.04186904f, 0.03748120f, 0.03102849f, 0.02457578f
    };

    const float uu = Clamp(u, 0.f, 0.99999994f);
    float cdf0 = 0.f;
    for (u_int i = 0; i < 16u; ++i) {
        const float cdf1 = cdf0 + binProb[i];
        if ((uu < cdf1) || (i == 15u)) {
            const float localU = Clamp((uu - cdf0) / binProb[i], 0.f, 0.99999994f);
            *sampleWeight = 1.f / (16.f * binProb[i]);
            return 380.f + (i + localU) * 25.f;
        }
        cdf0 = cdf1;
    }
    *sampleWeight = 1.f;
    return 780.f;
}

//------------------------------------------------------------------------------
// BiDirCPU RenderThread
//------------------------------------------------------------------------------

const Film::FilmChannels BiDirCPURenderThread::eyeSampleResultsChannels({
	Film::RADIANCE_PER_PIXEL_NORMALIZED, Film::ALPHA, Film::DEPTH,
	Film::POSITION, Film::GEOMETRY_NORMAL, Film::SHADING_NORMAL, Film::MATERIAL_ID,
	Film::UV, Film::OBJECT_ID, Film::SAMPLECOUNT, Film::CONVERGENCE,
	Film::MATERIAL_ID_COLOR, Film::ALBEDO, Film::AVG_SHADING_NORMAL, Film::NOISE
});

const Film::FilmChannels BiDirCPURenderThread::lightSampleResultsChannels({
	Film::RADIANCE_PER_SCREEN_NORMALIZED
}); 

BiDirCPURenderThread::BiDirCPURenderThread(BiDirCPURenderEngine *engine,
		const u_int index, IntersectionDevice *device) :
		CPUNoTileRenderThread(engine, index, device) {
}

void BiDirCPURenderThread::AOVWarmUp(
	std::stop_token stop_token,
	const luxrays::RandomGeneratorUPtr & rndGen
) {
	if (threadIndex == 0)
		SLG_LOG("[BiDirCPURenderThread::" << threadIndex << "] AOV warmup started");

	const double start = WallClockTime();
	double lastProgressPrint = start;

	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;
	auto& scene = engine->renderConfig.GetScene();
	auto& camera = scene.GetCamera();

	SobolSampler sampler(rndGen, engine->GetFilm(), engine->GetSampleSplatter(), true, 0.f, 0.f,
		16, 16, 1, 1,
		engine->aovWarmupSamplerSharedData
	);

	// Request the samples
	const u_int sampleBootSize = 5;
	const u_int sampleStepSize = 3;
	const u_int sampleSize = 
		sampleBootSize + // To generate eye ray
		engine->maxEyePathDepth * sampleStepSize; // For each path vertex
	sampler.RequestSamples(ONLY_AOV_SAMPLE, sampleSize);

	// Initialize SampleResult 
	vector<SampleResult> sampleResults(1);
	SampleResult &sampleResult = sampleResults[0];
	const Film::FilmChannels sampleResultsChannels({
		Film::ALBEDO, Film::AVG_SHADING_NORMAL
	});

	sampleResult.Init(&sampleResultsChannels, engine->GetFilm().GetRadianceGroupCount());

	// Initialize the max. path depth
	PathDepthInfo maxPathDepthInfo;
	maxPathDepthInfo.depth = engine->maxEyePathDepth;
	maxPathDepthInfo.diffuseDepth = engine->maxEyePathDepth;
	maxPathDepthInfo.glossyDepth = engine->maxEyePathDepth;
	maxPathDepthInfo.specularDepth = engine->maxEyePathDepth;

	while (sampler.GetPassCount() < engine->aovWarmupSPP) {
		if (stop_token.stop_requested())
			return;

		sampleResult.filmX = sampler.GetSample(0);
		sampleResult.filmY = sampler.GetSample(1);

		const float timeSample = sampler.GetSample(4);
		const float time = scene.GetCamera().GenerateRayTime(timeSample);

		Ray eyeRay;
		PathVolumeInfo volInfo;
		camera.GenerateRay(time,
				sampleResult.filmX, sampleResult.filmY, &eyeRay,
				&volInfo, sampler.GetSample(2), sampler.GetSample(3));

		sampleResult.albedo = Spectrum(); // Just in case albedoToDo is never true
		sampleResult.shadingNormal = Normal();

		Spectrum pathThroughput(1.f);
		u_int depth = 0;
		BSDF bsdf;
		while (depth < engine->maxEyePathDepth) {
			sampleResult.firstPathVertex = (depth == 0);

			const u_int sampleOffset = sampleBootSize + depth * sampleStepSize;

			// NOTE: I account for volume emission only with path tracing (i.e. here and
			// not in any other place)
			RayHit eyeRayHit;
			Spectrum connectionThroughput;
			const bool hit = scene.Intersect(device,
					EYE_RAY | (sampleResult.firstPathVertex ? CAMERA_RAY : INDIRECT_RAY),
					&volInfo, sampler.GetSample(sampleOffset),
					&eyeRay, &eyeRayHit, &bsdf,
					&connectionThroughput, &pathThroughput, &sampleResult);
			pathThroughput *= connectionThroughput;

			if (!hit) {
				// Nothing was hit
				break;
			}

			//------------------------------------------------------------------
			// Something was hit
			//------------------------------------------------------------------

			if (bsdf.IsAlbedoEndPoint(engine->albedoSpecularSetting,
					engine->albedoSpecularGlossinessThreshold)) {
				sampleResult.albedo = pathThroughput * bsdf.Albedo();
				sampleResult.shadingNormal = bsdf.hitPoint.shadeN;
				break;
			}

			// Check if I reached the max. depth
			if (depth == engine->maxEyePathDepth)
				break;

			//------------------------------------------------------------------
			// Build the next vertex path ray
			//------------------------------------------------------------------

			Vector sampledDir;
			float cosSampledDir, lastPdfW;
			BSDFEvent lastBSDFEvent;
			const Spectrum bsdfSample = bsdf.Sample(&sampledDir,
						sampler.GetSample(sampleOffset + 1),
						sampler.GetSample(sampleOffset + 2),
						&lastPdfW, &cosSampledDir, &lastBSDFEvent);

			assert (!bsdfSample.IsNaN() && !bsdfSample.IsInf());
			if (bsdfSample.Black())
				break;
			assert (!isnan(lastPdfW) && !isinf(lastPdfW));

			pathThroughput *= bsdfSample;
			assert (!pathThroughput.IsNaN() && !pathThroughput.IsInf());

			// Update volume information
			volInfo.Update(lastBSDFEvent, bsdf);

			eyeRay.Update(bsdf.GetRayOrigin(sampledDir), sampledDir);
		}

		sampler.NextSample(sampleResults);

		if (threadIndex == 0) {
			const double end = WallClockTime();
			const double delta = end - lastProgressPrint;

			if (delta > 2.0) {
				const u_int *filmSubRegion = engine->GetFilm().GetSubRegion();
				const u_int subRegionWidth = filmSubRegion[1] - filmSubRegion[0] + 1;
				const u_int subRegionHeight = filmSubRegion[3] - filmSubRegion[2] + 1;

				const double samplesSec = sampler.GetPassCount() * (double)(subRegionWidth * subRegionHeight) / (1000000.0 * (end -start));

				SLG_LOG("[BiDirCPURenderThread::" << threadIndex << "] AOV warmup progress: " <<
						sampler.GetPassCount() << "/" << engine->aovWarmupSPP << " pass"
						" (" << boost::str(boost::format("%3.2fM") % samplesSec) << " samples/sec)");
				
				lastProgressPrint = WallClockTime();
			}
		}
#ifdef WIN32
		// Work around Windows bad scheduling
        std::this_thread::yield();
#endif
	}
	
	if (threadIndex == 0) {
		const double end = WallClockTime();
		
		const u_int *filmSubRegion = engine->GetFilm().GetSubRegion();
		const u_int subRegionWidth = filmSubRegion[1] - filmSubRegion[0] + 1;
		const u_int subRegionHeight = filmSubRegion[3] - filmSubRegion[2] + 1;
		
		const double samplesSec = engine->aovWarmupSPP * (double)(subRegionWidth * subRegionHeight) / (1000000.0 * (end -start));
		
		SLG_LOG("[BiDirCPURenderThread::" << threadIndex << "] AOV warmup done: " <<
				boost::str(boost::format("%3.2fM") % samplesSec) << " samples/sec");
	}
}

SampleResult &BiDirCPURenderThread::AddResult(vector<SampleResult> &sampleResults, const bool fromLight) const {
	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;

	const u_int size = sampleResults.size();
	sampleResults.resize(size + 1);

	SampleResult &sampleResult = sampleResults[size];

	sampleResult.Init(
			fromLight ? &lightSampleResultsChannels : &eyeSampleResultsChannels,
			engine->GetFilm().GetRadianceGroupCount());

	return sampleResult;
}

void BiDirCPURenderThread::ConnectVertices(const float time,
		const PathVertexVM &eyeVertex, const PathVertexVM &lightVertex,
		SampleResult &eyeSampleResult, const float u0) const {
	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;
	auto& scene = engine->renderConfig.GetScene();

	Vector p2pDir(lightVertex.bsdf.hitPoint.p - eyeVertex.bsdf.hitPoint.p);
	const float p2pDistance2 = p2pDir.LengthSquared();
	const float p2pDistance = sqrtf(p2pDistance2);
	p2pDir /= p2pDistance;

	// Check eye vertex BSDF
	float eyeBsdfPdfW, eyeBsdfRevPdfW;
	BSDFEvent eyeEvent;
	const Spectrum eyeBsdfEval = eyeVertex.bsdf.Evaluate(p2pDir, &eyeEvent, &eyeBsdfPdfW, &eyeBsdfRevPdfW);

	if (!eyeBsdfEval.Black()) {
		// Check light vertex BSDF
		float lightBsdfPdfW, lightBsdfRevPdfW;
		BSDFEvent lightEvent;
		const Spectrum lightBsdfEval = lightVertex.bsdf.Evaluate(-p2pDir, &lightEvent, &lightBsdfPdfW, &lightBsdfRevPdfW);

		if (!lightBsdfEval.Black()) {
			// Check if the 2 surfaces can see each other
			const float cosThetaAtCamera = Dot(eyeVertex.bsdf.hitPoint.shadeN, p2pDir);
			const float cosThetaAtLight = Dot(lightVertex.bsdf.hitPoint.shadeN, -p2pDir);
			// Was:
			//  const float geometryTerm = cosThetaAtCamera * cosThetaAtLight / eyeDistance2;
			//
			// but now BSDF::Evaluate() follows LuxRender habit to return the
			// result multiplied by cosThetaAtLight
			const float geometryTerm = 1.f / p2pDistance2;

			// Trace ray between the two vertices
			const Point shadowRayOrig = eyeVertex.bsdf.GetRayOrigin(p2pDir);
			const Vector shadowRayOrigP2P(lightVertex.bsdf.hitPoint.p - shadowRayOrig);
			const float shadowRayDistanceSquared = shadowRayOrigP2P.LengthSquared();
			const float shadowRayDistance = sqrtf(shadowRayDistanceSquared);
			const Vector shadowRayDir = shadowRayOrigP2P / shadowRayDistance;

			Ray p2pRay(shadowRayOrig, shadowRayDir,
					0.f,
					shadowRayDistance,
					time);
			p2pRay.UpdateMinMaxWithEpsilon();

			RayHit p2pRayHit;
			BSDF bsdfConn;
			Spectrum connectionThroughput;
			PathVolumeInfo volInfo = eyeVertex.volInfo; // I need to use a copy here
			// For the connection event, we need to evaluate the volume based
			// on whether the shadow ray is going into the object or not
			bool connectionIntoObject = (Dot(lightVertex.bsdf.hitPoint.geometryN, -shadowRayDir) < 0.f);
			volInfo.SetCurrentVolume(connectionIntoObject ?
				lightVertex.bsdf.hitPoint.interiorVolume :
				lightVertex.bsdf.hitPoint.exteriorVolume
			);
			if (!scene.Intersect(device, LIGHT_RAY | INDIRECT_RAY | SHADOW_RAY, &volInfo, u0, &p2pRay, &p2pRayHit, &bsdfConn,
					&connectionThroughput)) {
				// Nothing was hit, the light path vertex is visible

				const bool eyeTerm15bi = eyeVertex.mlHeroSecondaryWavelengthsTerminated;
				const bool lightTerm15bi = lightVertex.mlHeroSecondaryWavelengthsTerminated;

				if (MLHeroLegacyHeavyDiagnosticsEnabled15cq() && (engine->mlHeroGlassMode == 1)) {
					mlHero15bhVisibleConnections.fetch_add(1ull, std::memory_order_relaxed);
					if (eyeTerm15bi) mlHero15bhVisibleEyeTerminated.fetch_add(1ull, std::memory_order_relaxed);
					if (lightTerm15bi) mlHero15bhVisibleLightTerminated.fetch_add(1ull, std::memory_order_relaxed);
					if (eyeTerm15bi && lightTerm15bi)
						mlHero15bhVisibleBothTerminated.fetch_add(1ull, std::memory_order_relaxed);
					else if (!eyeTerm15bi && !lightTerm15bi)
						mlHero15bhVisibleNeitherTerminated.fetch_add(1ull, std::memory_order_relaxed);
				}

				if (eyeVertex.depth >= engine->rrDepth) {
					// Russian Roulette
					const float prob = RenderEngine::RussianRouletteProb(eyeBsdfEval, engine->rrImportanceCap);
					eyeBsdfPdfW *= prob;
					eyeBsdfRevPdfW *= prob;
				}

				if (lightVertex.depth >= engine->rrDepth) {
					// Russian Roulette
					const float prob = RenderEngine::RussianRouletteProb(lightBsdfEval, engine->rrImportanceCap);
					lightBsdfPdfW *= prob;
					lightBsdfRevPdfW *= prob;
				}

				// Convert pdfs to area pdfs
				const float eyeBsdfPdfA = PdfWtoA(eyeBsdfPdfW, p2pDistance, cosThetaAtLight);
				const float lightBsdfPdfA = PdfWtoA(lightBsdfPdfW,  p2pDistance, cosThetaAtCamera);

				// MIS weights
				const float lightWeight = MIS(eyeBsdfPdfA) *
					(misVmWeightFactor + lightVertex.dVCM + lightVertex.dVC * MIS(lightBsdfRevPdfW));
				const float eyeWeight = MIS(lightBsdfPdfA) *
					(misVmWeightFactor + eyeVertex.dVCM + eyeVertex.dVC * MIS(eyeBsdfRevPdfW));

				const float misWeight = 1.f / (lightWeight + 1.f + eyeWeight);

				// ML HERO phase 15bo: if a stored eye/light subpath already crossed
				// a perfect dispersive HERO-only event, test an upper-bound MIS
				// candidate where alternate strategies on that same side are removed.
				// Render behavior is unchanged.
				if (MLHeroLegacyHeavyDiagnosticsEnabled15cq() && (engine->mlHeroGlassMode == 1)) {
					const float prunedLightWeight15bo = lightTerm15bi ? 0.f : lightWeight;
					const float prunedEyeWeight15bo = eyeTerm15bi ? 0.f : eyeWeight;
					const float candidateMis15bo = 1.f / (prunedLightWeight15bo + 1.f + prunedEyeWeight15bo);
					const float removed15bo = (lightTerm15bi ? lightWeight : 0.f) + (eyeTerm15bi ? eyeWeight : 0.f);
					MLHeroRecordMisPrune15bo(0u, eyeTerm15bi, lightTerm15bi, misWeight, candidateMis15bo, removed15bo);
				}

				// ML HERO phase 15m:
				// Controlled ConnectVertices experiment with TWO CIE candidates:
				//
				// A) CONTROL-CIE:
				//    uses scalar eye/light lane throughputs + the CLASSIC endpoint BSDF
				//    scalar for every wavelength lane.
				//
				// B) METAL2-CIE:
				//    identical to A, except Metal2 endpoints use explicit BSDF(lambda).
				//
				// This cleanly separates general spectral/CIE normalization from the
				// wavelength-dependent Metal2 endpoint response.
				if ((eyeVertex.mlHeroLaneCount > 1u) && (lightVertex.mlHeroLaneCount > 1u)) {
					const u_int laneCount = Min(eyeVertex.mlHeroLaneCount, lightVertex.mlHeroLaneCount);

					const bool eyeTPScalar = MLHeroAllLanesScalar(eyeVertex);
					const bool lightTPScalar = MLHeroAllLanesScalar(lightVertex);
					const bool connectionScalar = MLHeroIsScalarSpectrum(connectionThroughput);
					const bool classicEyeBsdfScalar = MLHeroIsScalarSpectrum(eyeBsdfEval);
					const bool classicLightBsdfScalar = MLHeroIsScalarSpectrum(lightBsdfEval);

					bool packetMatch = (eyeVertex.mlHeroLaneCount == lightVertex.mlHeroLaneCount);
					for (u_int lane = 0; packetMatch && (lane < laneCount); ++lane) {
						const float dLambda = fabsf(eyeVertex.mlHeroLaneWaveLength[lane] -
							lightVertex.mlHeroLaneWaveLength[lane]);
						const float dWeight = fabsf(eyeVertex.mlHeroLaneSampleWeight[lane] -
							lightVertex.mlHeroLaneSampleWeight[lane]);
						packetMatch = (dLambda < 1e-4f) && (dWeight < 1e-6f);
					}

					bool metalEndpointsUsable = true;
					bool anyEyeExplicitMetal2 = false;
					bool anyLightExplicitMetal2 = false;

					float metalEyeBsdfScalar[8] = { 0.f };
					float metalLightBsdfScalar[8] = { 0.f };

					for (u_int lane = 0; lane < laneCount; ++lane) {
						const float waveLength = eyeVertex.mlHeroLaneWaveLength[lane];

						Spectrum eyeLaneBsdf = eyeBsdfEval;
						Spectrum explicitEyeBsdf;
						if (eyeVertex.bsdf.EvaluateMLHeroMetal2AtWaveLength(
								p2pDir, waveLength, &explicitEyeBsdf)) {
							eyeLaneBsdf = explicitEyeBsdf;
							anyEyeExplicitMetal2 = true;
						}

						Spectrum lightLaneBsdf = lightBsdfEval;
						Spectrum explicitLightBsdf;
						if (lightVertex.bsdf.EvaluateMLHeroMetal2AtWaveLength(
								-p2pDir, waveLength, &explicitLightBsdf)) {
							lightLaneBsdf = explicitLightBsdf;
							anyLightExplicitMetal2 = true;
						}

						if (!MLHeroIsScalarSpectrum(eyeLaneBsdf) ||
								!MLHeroIsScalarSpectrum(lightLaneBsdf)) {
							metalEndpointsUsable = false;
							break;
						}

						metalEyeBsdfScalar[lane] = eyeLaneBsdf.c[0];
						metalLightBsdfScalar[lane] = lightLaneBsdf.c[0];
					}

					const bool commonGate =
						packetMatch &&
						eyeTPScalar &&
						lightTPScalar &&
						connectionScalar &&
						classicEyeBsdfScalar &&
						classicLightBsdfScalar;

					if (commonGate) {
						const float connectionScalarValue = connectionThroughput.c[0];
						const float classicEyeBsdfValue = eyeBsdfEval.c[0];
						const float classicLightBsdfValue = lightBsdfEval.c[0];

						Spectrum controlCIE;
						Spectrum metal2CIE;

						float controlPathScalar[8] = { 0.f };
						float metalPathScalar[8] = { 0.f };

						for (u_int lane = 0; lane < laneCount; ++lane) {
							const float waveLength = eyeVertex.mlHeroLaneWaveLength[lane];
							const float sampleWeight = eyeVertex.mlHeroLaneSampleWeight[lane];

							const float eyeScalar = eyeVertex.mlHeroLaneThroughput[lane].c[0];
							const float lightScalar = lightVertex.mlHeroLaneThroughput[lane].c[0];
							const Spectrum cieColor =
								MLHeroCIEEstimatorColor(waveLength, sampleWeight);

							controlPathScalar[lane] =
								eyeScalar * classicEyeBsdfValue *
								lightScalar * classicLightBsdfValue *
								connectionScalarValue *
								(misWeight * geometryTerm);

							controlCIE += cieColor * controlPathScalar[lane];

							if (metalEndpointsUsable) {
								metalPathScalar[lane] =
									eyeScalar * metalEyeBsdfScalar[lane] *
									lightScalar * metalLightBsdfScalar[lane] *
									connectionScalarValue *
									(misWeight * geometryTerm);

								metal2CIE += cieColor * metalPathScalar[lane];
							}
						}

						controlCIE /= (float)laneCount;
						if (metalEndpointsUsable)
							metal2CIE /= (float)laneCount;

						const Spectrum classicRadiance =
							(misWeight * geometryTerm) *
							eyeVertex.throughput * eyeBsdfEval *
							connectionThroughput *
							lightBsdfEval * lightVertex.throughput;

						// Prefer a sample that actually exercises Metal2 on at least one endpoint.
						if (anyEyeExplicitMetal2 || anyLightExplicitMetal2) {
							static std::atomic<bool> mlHeroPhase15mLogged(false);
							bool expected = false;
							if (mlHeroPhase15mLogged.compare_exchange_strong(expected, true)) {
								std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock2(mlHeroDebugLogMutex);
								FILE *f = MLHeroOpenLegacyLog15cq("a");
								if (f) {
									fprintf(f,
										"ML HERO phase 15m CONNECTVERTICES CONTROL TEST: "
										"eyeDepth=%u lightDepth=%u count=%u eyeMetal2=%u lightMetal2=%u\n",
										eyeVertex.depth, lightVertex.depth, laneCount,
										anyEyeExplicitMetal2 ? 1u : 0u,
										anyLightExplicitMetal2 ? 1u : 0u);

									fprintf(f,
										"phase15m checks: packetMatch=%u eyeTP=%u lightTP=%u connection=%u "
										"classicEyeBSDF=%u classicLightBSDF=%u metalEndpoints=%u "
										"misWeight=%.9g geometryTerm=%.9g connectionScalar=%.9g\n",
										packetMatch ? 1u : 0u,
										eyeTPScalar ? 1u : 0u,
										lightTPScalar ? 1u : 0u,
										connectionScalar ? 1u : 0u,
										classicEyeBsdfScalar ? 1u : 0u,
										classicLightBsdfScalar ? 1u : 0u,
										metalEndpointsUsable ? 1u : 0u,
										misWeight, geometryTerm, connectionScalarValue);

									fprintf(f,
										"phase15m classic endpoint BSDF scalars: eye=%.9g light=%.9g\n",
										classicEyeBsdfValue, classicLightBsdfValue);

									fprintf(f, "phase15m classic radiance=(%.9g, %.9g, %.9g)\n",
										classicRadiance.c[0], classicRadiance.c[1], classicRadiance.c[2]);
									fprintf(f, "phase15m CONTROL CIE candidate=(%.9g, %.9g, %.9g)\n",
										controlCIE.c[0], controlCIE.c[1], controlCIE.c[2]);

									if (metalEndpointsUsable) {
										fprintf(f, "phase15m METAL2 CIE candidate=(%.9g, %.9g, %.9g)\n",
											metal2CIE.c[0], metal2CIE.c[1], metal2CIE.c[2]);
									}

									for (u_int lane = 0; lane < laneCount; ++lane) {
										const float lambda = eyeVertex.mlHeroLaneWaveLength[lane];
										const float weight = eyeVertex.mlHeroLaneSampleWeight[lane];
										const Spectrum cieColor =
											MLHeroCIEEstimatorColor(lambda, weight);

										fprintf(f,
											"phase15m lane %u: lambda=%.9g weight=%.9g "
											"eyeTP=%.9g lightTP=%.9g "
											"controlEyeBSDF=%.9g controlLightBSDF=%.9g "
											"metalEyeBSDF=%.9g metalLightBSDF=%.9g "
											"controlScalar=%.9g metalScalar=%.9g\n",
											lane, lambda, weight,
											eyeVertex.mlHeroLaneThroughput[lane].c[0],
											lightVertex.mlHeroLaneThroughput[lane].c[0],
											classicEyeBsdfValue, classicLightBsdfValue,
											metalEyeBsdfScalar[lane], metalLightBsdfScalar[lane],
											controlPathScalar[lane], metalPathScalar[lane]);

										fprintf(f,
											"  cieColor=(%.9g, %.9g, %.9g)\n",
											cieColor.c[0], cieColor.c[1], cieColor.c[2]);
									}

									fclose(f);
								}
							}
						}
					}
				}

				// ML HERO phase 15n: validate the explicit Metal2(lambda) factorization
				// against the measured ExtremeGold/Sopra n/k input.
				if ((eyeVertex.mlHeroLaneCount > 1u) && (lightVertex.mlHeroLaneCount > 1u)) {
					static std::atomic<bool> mlHeroPhase15nLogged(false);

					const bool eyeIsMetal2 = (eyeVertex.bsdf.GetMaterialType() == METAL2);
					const bool lightIsMetal2 = (lightVertex.bsdf.GetMaterialType() == METAL2);

					if (eyeIsMetal2 || lightIsMetal2) {
						bool expected = false;
						if (mlHeroPhase15nLogged.compare_exchange_strong(expected, true)) {
							std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock3(mlHeroDebugLogMutex);
							FILE *f = MLHeroOpenLegacyLog15cq("a");
							if (f) {
								const u_int laneCount = Min(eyeVertex.mlHeroLaneCount,
										lightVertex.mlHeroLaneCount);

								fprintf(f,
									"ML HERO phase 15n METAL2 FACTOR VALIDATION: "
									"eyeDepth=%u lightDepth=%u count=%u eyeMetal2=%u lightMetal2=%u\n",
									eyeVertex.depth, lightVertex.depth, laneCount,
									eyeIsMetal2 ? 1u : 0u, lightIsMetal2 ? 1u : 0u);

								fprintf(f,
									"phase15n classic endpoint BSDF: eye=(%.9g, %.9g, %.9g) "
									"light=(%.9g, %.9g, %.9g)\n",
									eyeBsdfEval.c[0], eyeBsdfEval.c[1], eyeBsdfEval.c[2],
									lightBsdfEval.c[0], lightBsdfEval.c[1], lightBsdfEval.c[2]);

								for (u_int lane = 0; lane < laneCount; ++lane) {
									const float lambda = eyeVertex.mlHeroLaneWaveLength[lane];

									Spectrum eta, kk, fresnel, result;
									float microfacet = 0.f;
									float wrapper = 0.f;
									float cosWH = 0.f;
									bool ok = false;
									const char *endpoint = "NONE";

									if (eyeIsMetal2) {
										ok = eyeVertex.bsdf.EvaluateMLHeroMetal2DebugAtWaveLength(
											p2pDir, lambda,
											&eta, &kk, &fresnel,
											&microfacet, &wrapper, &cosWH, &result);
										endpoint = "EYE";
									} else if (lightIsMetal2) {
										ok = lightVertex.bsdf.EvaluateMLHeroMetal2DebugAtWaveLength(
											-p2pDir, lambda,
											&eta, &kk, &fresnel,
											&microfacet, &wrapper, &cosWH, &result);
										endpoint = "LIGHT";
									}

									if (ok) {
										const Spectrum reconstructed =
											fresnel * (microfacet * wrapper);

										fprintf(f,
											"phase15n lane %u endpoint=%s lambda=%.9g "
											"n=(%.9g, %.9g, %.9g) k=(%.9g, %.9g, %.9g) "
											"cosWH=%.9g\n",
											lane, endpoint, lambda,
											eta.c[0], eta.c[1], eta.c[2],
											kk.c[0], kk.c[1], kk.c[2],
											cosWH);

										fprintf(f,
											"  Fresnel=(%.9g, %.9g, %.9g) "
											"microfacet=%.9g wrapper=%.9g "
											"BSDF=(%.9g, %.9g, %.9g) "
											"reconstructed=(%.9g, %.9g, %.9g)\n",
											fresnel.c[0], fresnel.c[1], fresnel.c[2],
											microfacet, wrapper,
											result.c[0], result.c[1], result.c[2],
											reconstructed.c[0],
											reconstructed.c[1],
											reconstructed.c[2]);
									} else {
										fprintf(f,
											"phase15n lane %u endpoint=%s lambda=%.9g DEBUG_EVAL_FAILED\n",
											lane, endpoint, lambda);
									}
								}

								fclose(f);
							}
						}
					}
				}

				// ML HERO phase 15o:
				// Activate the spectral CIE ConnectVertices contribution ONLY when the
				// packet is clean and every factor can be evaluated as a scalar lane.
				// Otherwise fall back to the original classic RGB contribution.
				bool mlHero15oUsedSpectral = false;

				if ((eyeVertex.mlHeroLaneCount > 1u) && (lightVertex.mlHeroLaneCount > 1u)) {
					const u_int laneCount = Min(eyeVertex.mlHeroLaneCount, lightVertex.mlHeroLaneCount);

					const bool eyeTPScalar = MLHeroAllLanesScalar(eyeVertex);
					const bool lightTPScalar = MLHeroAllLanesScalar(lightVertex);
					const bool connectionScalar = MLHeroIsScalarSpectrum(connectionThroughput);

					bool packetMatch = (eyeVertex.mlHeroLaneCount == lightVertex.mlHeroLaneCount);
					for (u_int lane = 0; packetMatch && (lane < laneCount); ++lane) {
						const float dLambda = fabsf(eyeVertex.mlHeroLaneWaveLength[lane] -
							lightVertex.mlHeroLaneWaveLength[lane]);
						const float dWeight = fabsf(eyeVertex.mlHeroLaneSampleWeight[lane] -
							lightVertex.mlHeroLaneSampleWeight[lane]);
						packetMatch = (dLambda < 1e-4f) && (dWeight < 1e-6f);
					}

					bool endpointsUsable = true;
					bool anyExplicitMetal2 = false;
					float eyeBsdfScalar[8] = { 0.f };
					float lightBsdfScalar[8] = { 0.f };

					for (u_int lane = 0; lane < laneCount; ++lane) {
						const float waveLength = eyeVertex.mlHeroLaneWaveLength[lane];

						Spectrum eyeLaneBsdf = eyeBsdfEval;
						Spectrum explicitEyeBsdf;
						if (eyeVertex.bsdf.EvaluateMLHeroMetal2AtWaveLength(
								p2pDir, waveLength, &explicitEyeBsdf)) {
							eyeLaneBsdf = explicitEyeBsdf;
							anyExplicitMetal2 = true;
						}

						Spectrum lightLaneBsdf = lightBsdfEval;
						Spectrum explicitLightBsdf;
						if (lightVertex.bsdf.EvaluateMLHeroMetal2AtWaveLength(
								-p2pDir, waveLength, &explicitLightBsdf)) {
							lightLaneBsdf = explicitLightBsdf;
							anyExplicitMetal2 = true;
						}

						if (!MLHeroIsScalarSpectrum(eyeLaneBsdf) ||
								!MLHeroIsScalarSpectrum(lightLaneBsdf)) {
							endpointsUsable = false;
							break;
						}

						eyeBsdfScalar[lane] = eyeLaneBsdf.c[0];
						lightBsdfScalar[lane] = lightLaneBsdf.c[0];
					}

					if (packetMatch && eyeTPScalar && lightTPScalar &&
							connectionScalar && endpointsUsable && anyExplicitMetal2) {
						const float connectionScalarValue = connectionThroughput.c[0];

						Spectrum ciePacket;
						for (u_int lane = 0; lane < laneCount; ++lane) {
							const float waveLength = eyeVertex.mlHeroLaneWaveLength[lane];
							const float sampleWeight = eyeVertex.mlHeroLaneSampleWeight[lane];

							const float eyeScalar = eyeVertex.mlHeroLaneThroughput[lane].c[0];
							const float lightScalar = lightVertex.mlHeroLaneThroughput[lane].c[0];

							const float lanePathScalar =
								eyeScalar * eyeBsdfScalar[lane] *
								lightScalar * lightBsdfScalar[lane] *
								connectionScalarValue *
								(misWeight * geometryTerm);

							ciePacket += MLHeroCIEEstimatorColor(waveLength, sampleWeight) *
								lanePathScalar;
						}

						ciePacket /= (float)laneCount;
						if (engine->mlHeroGlassMode == 1) {
							MLHeroRecordConnectContribution15bi(ciePacket, eyeTerm15bi, lightTerm15bi);
							MLHeroRecordDualTermination15by(ciePacket, eyeVertex, lightVertex, laneCount, true);
							if (engine->mlHeroDualTerminationCompensation)
								MLHeroApplyDualTerminationCompensation15bz(ciePacket, eyeVertex, lightVertex, laneCount);
						}
							MLHeroRecordEstimatorContribution15bj(0u, ciePacket, eyeTerm15bi, lightTerm15bi, true, false);
							if (MLHeroLegacyHeavyDiagnosticsEnabled15cq()) { const unsigned long long read15bq = mlHero15bqReadCount[0].fetch_add(1ull, std::memory_order_relaxed); MLHeroLogState15bq("READ_EYE", read15bq, 0u, eyeVertex); MLHeroLogState15bq("READ_LIGHT", read15bq, 0u, lightVertex); const unsigned long long snap15bq = mlHero15bqSnapshotTicket.fetch_add(1ull, std::memory_order_relaxed); MLHeroWriteSnapshot15bq(snap15bq); }
							MLHeroRecordContributionWeightedBias15bp(0u, ciePacket, eyeTerm15bi, lightTerm15bi,
								eyeVertex.mlHeroTerminationBrightnessRatio, lightVertex.mlHeroTerminationBrightnessRatio);
							MLHeroRecordWavelengthBias15bs(0u, 0u, eyeVertex, ciePacket);
							MLHeroRecordWavelengthBias15bs(0u, 1u, lightVertex, ciePacket);
						eyeSampleResult.radiance[lightVertex.lightID] += ciePacket;
						mlHero15oUsedSpectral = true;

						static std::atomic<bool> mlHeroPhase15pLogged(false);
						bool expected = false;
						if (mlHeroPhase15pLogged.compare_exchange_strong(expected, true)) {
							std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock4(mlHeroDebugLogMutex);
							FILE *f = MLHeroOpenLegacyLog15cq("a");
							if (f) {
								const Spectrum classicRadiance =
									(misWeight * geometryTerm) *
									eyeVertex.throughput * eyeBsdfEval *
									connectionThroughput *
									lightBsdfEval * lightVertex.throughput;

								fprintf(f,
									"ML HERO phase 15p SAME-HIT VISIBLE CONNECTVERTICES BREAKDOWN: "
									"eyeDepth=%u lightDepth=%u count=%u\n",
									eyeVertex.depth, lightVertex.depth, laneCount);

								fprintf(f,
									"phase15p gate: packetMatch=%u eyeTP=%u lightTP=%u connection=%u "
									"endpoints=%u explicitMetal2=%u\n",
									packetMatch ? 1u : 0u,
									eyeTPScalar ? 1u : 0u,
									lightTPScalar ? 1u : 0u,
									connectionScalar ? 1u : 0u,
									endpointsUsable ? 1u : 0u,
									anyExplicitMetal2 ? 1u : 0u);

								fprintf(f,
									"phase15p classic fallback reference=(%.9g, %.9g, %.9g)\n",
									classicRadiance.c[0], classicRadiance.c[1], classicRadiance.c[2]);

								Spectrum sumContrib;
								for (u_int lane = 0; lane < laneCount; ++lane) {
									const float waveLength = eyeVertex.mlHeroLaneWaveLength[lane];
									const float sampleWeight = eyeVertex.mlHeroLaneSampleWeight[lane];
									const float eyeScalar = eyeVertex.mlHeroLaneThroughput[lane].c[0];
									const float lightScalar = lightVertex.mlHeroLaneThroughput[lane].c[0];

									const float lanePathScalar =
										eyeScalar * eyeBsdfScalar[lane] *
										lightScalar * lightBsdfScalar[lane] *
										connectionScalarValue *
										(misWeight * geometryTerm);

									const Spectrum cieColor =
										MLHeroCIEEstimatorColor(waveLength, sampleWeight);
									const Spectrum rawContribution = cieColor * lanePathScalar;
									const Spectrum avgContribution = rawContribution / (float)laneCount;

									sumContrib += avgContribution;

									fprintf(f,
										"phase15p lane %u: lambda=%.9g weight=%.9g "
										"eyeTP=%.9g eyeBSDF=%.9g lightTP=%.9g lightBSDF=%.9g "
										"connection=%.9g mis=%.9g geom=%.9g laneScalar=%.9g\n",
										lane, waveLength, sampleWeight,
										eyeScalar, eyeBsdfScalar[lane],
										lightScalar, lightBsdfScalar[lane],
										connectionScalarValue, misWeight, geometryTerm,
										lanePathScalar);

									fprintf(f,
										"  cieColor=(%.9g, %.9g, %.9g) "
										"rawContribution=(%.9g, %.9g, %.9g) "
										"avgContribution=(%.9g, %.9g, %.9g)\n",
										cieColor.c[0], cieColor.c[1], cieColor.c[2],
										rawContribution.c[0], rawContribution.c[1], rawContribution.c[2],
										avgContribution.c[0], avgContribution.c[1], avgContribution.c[2]);
								}

								fprintf(f,
									"phase15p summed lane contributions=(%.9g, %.9g, %.9g)\n",
									sumContrib.c[0], sumContrib.c[1], sumContrib.c[2]);

								fprintf(f,
									"phase15p visible CIE packet=(%.9g, %.9g, %.9g)\n",
									ciePacket.c[0], ciePacket.c[1], ciePacket.c[2]);

								fprintf(f,
									"phase15p closure error=(%.9g, %.9g, %.9g)\n",
									sumContrib.c[0] - ciePacket.c[0],
									sumContrib.c[1] - ciePacket.c[1],
									sumContrib.c[2] - ciePacket.c[2]);

								fclose(f);
							}
						}
					}
				}

				if (!mlHero15oUsedSpectral) {
					Spectrum classicConnectContribution15bi = (misWeight * geometryTerm) *
						eyeVertex.throughput * eyeBsdfEval *
						connectionThroughput * lightBsdfEval * lightVertex.throughput;
					if (engine->mlHeroGlassMode == 1) {
						const u_int dualLaneCount15bz = Min(eyeVertex.mlHeroLaneCount, lightVertex.mlHeroLaneCount);
						MLHeroRecordConnectContribution15bi(classicConnectContribution15bi, eyeTerm15bi, lightTerm15bi);
						MLHeroRecordDualTermination15by(classicConnectContribution15bi, eyeVertex, lightVertex, dualLaneCount15bz, false);
						if (engine->mlHeroDualTerminationCompensation)
							MLHeroApplyDualTerminationCompensation15bz(classicConnectContribution15bi, eyeVertex, lightVertex, dualLaneCount15bz);
					}
						MLHeroRecordEstimatorContribution15bj(0u, classicConnectContribution15bi, eyeTerm15bi, lightTerm15bi, false, true);
						if (MLHeroLegacyHeavyDiagnosticsEnabled15cq()) { const unsigned long long read15bq = mlHero15bqReadCount[0].fetch_add(1ull, std::memory_order_relaxed); MLHeroLogState15bq("READ_EYE", read15bq, 0u, eyeVertex); MLHeroLogState15bq("READ_LIGHT", read15bq, 0u, lightVertex); const unsigned long long snap15bq = mlHero15bqSnapshotTicket.fetch_add(1ull, std::memory_order_relaxed); MLHeroWriteSnapshot15bq(snap15bq); }
						MLHeroRecordContributionWeightedBias15bp(0u, classicConnectContribution15bi, eyeTerm15bi, lightTerm15bi,
							eyeVertex.mlHeroTerminationBrightnessRatio, lightVertex.mlHeroTerminationBrightnessRatio);
						MLHeroRecordWavelengthBias15bs(0u, 0u, eyeVertex, classicConnectContribution15bi);
						MLHeroRecordWavelengthBias15bs(0u, 1u, lightVertex, classicConnectContribution15bi);
					eyeSampleResult.radiance[lightVertex.lightID] += classicConnectContribution15bi;
				}
			}
		}
	}
}

void BiDirCPURenderThread::ConnectToEye(const float time,
		const PathVertexVM &lightVertex, const float u0,
		const Point &lensPoint, vector<SampleResult> &sampleResults) const {
	// I don't connect camera invisible objects with the eye
	if (lightVertex.bsdf.IsCameraInvisible())
		return;

	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;
	auto& scene = engine->renderConfig.GetScene();

	// Test if the point-camera connection is valid
	float filmX, filmY;
	bool sampleSuccess;
	Ray eyeRay;
	Vector eyeDir;
	float eyeDistance = 0;
    if (scene.GetCamera().GetType() == Camera::ORTHOGRAPHIC){
		// Orthographic camera need to be handled separately,
		// lensPoint can not be pre-calculated in this case
		Point p = lightVertex.bsdf.hitPoint.p;
		eyeDir = scene.GetCamera().GetDir();
		// calculate distance from vertex to camera plane
		const float D = -eyeDir.x*lensPoint.x - eyeDir.y*lensPoint.y - eyeDir.z*lensPoint.z;
		eyeDistance = eyeDir.x*p.x + eyeDir.y*p.y + eyeDir.z*p.z + D;
		eyeDistance = fabsf(eyeDistance);
		eyeRay = Ray(lightVertex.bsdf.hitPoint.p, eyeDir,
			0.f,
			eyeDistance,
			time);
		// Do not clamp the ray here because of the check inside ProjectToImage
		sampleSuccess = scene.GetCamera().ProjectToImage(&eyeRay, &filmX, &filmY);
	} else {
		eyeDir = Vector(lightVertex.bsdf.hitPoint.p - lensPoint);
		eyeDistance = eyeDir.Length();
		eyeDir /= eyeDistance;
		eyeRay = Ray(lensPoint, eyeDir,
			0.f,
			eyeDistance,
			time);
		// Do not clamp the ray here because of the check inside GetSamplePosition
		sampleSuccess = scene.GetCamera().GetSamplePosition(&eyeRay, &filmX, &filmY);
	}

	if (!sampleSuccess)
		return;

	// Test if the bsdf evaluates to black
	float bsdfPdfW, bsdfRevPdfW;
	BSDFEvent event;
	const Spectrum bsdfEval = lightVertex.bsdf.Evaluate(-eyeDir, &event, &bsdfPdfW, &bsdfRevPdfW);

	if (bsdfEval.Black())
		return;

	// Trace a shadow ray
	scene.GetCamera().ClampRay(&eyeRay); // Clamp the ray here (see comment above)
	// I have to flip the direction of the traced ray because
	// the information inside PathVolumeInfo are about the path from
	// the light toward the camera (i.e. ray.o would be in the wrong
	// place).
	Ray traceRay(lightVertex.bsdf.GetRayOrigin(-eyeRay.d), -eyeRay.d,
			eyeDistance - eyeRay.maxt,
			eyeDistance - eyeRay.mint,
			time);
	traceRay.UpdateMinMaxWithEpsilon();

	RayHit traceRayHit;
	BSDF bsdfConn;
	Spectrum connectionThroughput;
	PathVolumeInfo volInfo = lightVertex.volInfo; // I need to use a copy here
	const bool shadowIntersection = scene.Intersect(device, LIGHT_RAY | CAMERA_RAY, &volInfo, u0, &traceRay, &traceRayHit, &bsdfConn,
			&connectionThroughput);

	if (shadowIntersection)
		return;

	if (lightVertex.depth >= engine->rrDepth) {
		// Russian Roulette
		const float prob = RenderEngine::RussianRouletteProb(bsdfEval, engine->rrImportanceCap);
		bsdfRevPdfW *= prob;
	}

	const float cosToCamera = Dot(lightVertex.bsdf.hitPoint.shadeN, -eyeDir);
	float cameraPdfW, fluxToRadianceFactor;
	scene.GetCamera().GetPDF(eyeRay, eyeDistance, filmX, filmY, &cameraPdfW, &fluxToRadianceFactor);
	const float cameraPdfA = PdfWtoA(cameraPdfW, eyeDistance, cosToCamera);
	// Was:
	//  const float fluxToRadianceFactor = cameraPdfA;
	//
	// but now BSDF::Evaluate() follows LuxRender habit to return the
	// result multiplied by cosThetaToLight
	//
	// However this is not true for volumes (see bug
	// report http://forums.luxcorerender.org/viewtopic.php?f=4&t=1146&start=10#p13491)
	fluxToRadianceFactor *= lightVertex.bsdf.IsVolume() ? fabsf(cosToCamera) : 1.f;

	const float weightLight = MIS(cameraPdfA) *
		(misVmWeightFactor + lightVertex.dVCM + lightVertex.dVC * MIS(bsdfRevPdfW));
	const float misWeight = 1.f / (weightLight + 1.f);

	// ML HERO phase 15bo: connection-to-eye alternate strategies live on the
	// stored light subpath.  If that subpath is HERO-terminated, compare with
	// the support-pruned upper bound (weightLight removed).
	if (engine->mlHeroGlassMode == 1) {
		const bool lightTerm15bo = lightVertex.mlHeroSecondaryWavelengthsTerminated;
		const float candidateMis15bo = lightTerm15bo ? 1.f : misWeight;
		MLHeroRecordMisPrune15bo(2u, false, lightTerm15bo, misWeight, candidateMis15bo, lightTerm15bo ? weightLight : 0.f);
	}

	// ML HERO phase 15s:
	// ConnectToEye is another visible contribution path in BIDIR.  The light
	// vertex already owns scalar spectral lane throughput, so keep the packet
	// spectral through the endpoint and reconstruct RGB only at the end.
	Spectrum radiance;
	bool mlHero15sUsedSpectral = false;

	if ((lightVertex.mlHeroLaneCount > 1u) &&
			MLHeroAllLanesScalar(lightVertex) &&
			MLHeroIsScalarSpectrum(connectionThroughput)) {
		const u_int laneCount = lightVertex.mlHeroLaneCount;
		const float connectionScalar = connectionThroughput.c[0];

		bool endpointUsable = true;
		bool explicitMetal2 = false;
		float laneBsdfScalar[8] = { 0.f };

		for (u_int lane = 0; lane < laneCount; ++lane) {
			const float waveLength = lightVertex.mlHeroLaneWaveLength[lane];

			Spectrum laneBsdf = bsdfEval;
			Spectrum explicitBsdf;
			if (lightVertex.bsdf.EvaluateMLHeroMetal2AtWaveLength(
					-eyeDir, waveLength, &explicitBsdf)) {
				laneBsdf = explicitBsdf;
				explicitMetal2 = true;
			}

			if (!MLHeroIsScalarSpectrum(laneBsdf)) {
				endpointUsable = false;
				break;
			}

			laneBsdfScalar[lane] = laneBsdf.c[0];
		}

		if (endpointUsable) {
			Spectrum ciePacket;

			for (u_int lane = 0; lane < laneCount; ++lane) {
				const float waveLength = lightVertex.mlHeroLaneWaveLength[lane];
				const float sampleWeight = lightVertex.mlHeroLaneSampleWeight[lane];
				const float lightScalar =
					lightVertex.mlHeroLaneThroughput[lane].c[0];

				const float laneScalar =
					lightScalar *
					laneBsdfScalar[lane] *
					connectionScalar *
					(misWeight * fluxToRadianceFactor);

				ciePacket += MLHeroCIEEstimatorColor(
					waveLength, sampleWeight) * laneScalar;
			}

			ciePacket /= (float)laneCount;
			radiance = ciePacket;
			mlHero15sUsedSpectral = true;

			static std::atomic<bool> mlHeroPhase15sLogged(false);
			bool expected = false;
			if (mlHeroPhase15sLogged.compare_exchange_strong(expected, true)) {
				std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock5(mlHeroDebugLogMutex);
				FILE *f = MLHeroOpenLegacyLog15cq("a");
				if (f) {
					const Spectrum classicRadiance =
						(misWeight * fluxToRadianceFactor) *
						connectionThroughput *
						lightVertex.throughput * bsdfEval;

					fprintf(f,
						"ML HERO phase 15s CONNECTTOEYE SPECTRAL ACTIVE: "
						"lightDepth=%u count=%u explicitMetal2=%u\n",
						lightVertex.depth, laneCount,
						explicitMetal2 ? 1u : 0u);

					fprintf(f,
						"phase15s gate: lightTP=1 connection=1 endpoint=1 "
						"mis=%.9g fluxFactor=%.9g\n",
						misWeight, fluxToRadianceFactor);

					fprintf(f,
						"phase15s classic radiance=(%.9g, %.9g, %.9g)\n",
						classicRadiance.c[0],
						classicRadiance.c[1],
						classicRadiance.c[2]);

					Spectrum checkSum;
					for (u_int lane = 0; lane < laneCount; ++lane) {
						const float waveLength =
							lightVertex.mlHeroLaneWaveLength[lane];
						const float sampleWeight =
							lightVertex.mlHeroLaneSampleWeight[lane];
						const float lightScalar =
							lightVertex.mlHeroLaneThroughput[lane].c[0];

						const float laneScalar =
							lightScalar *
							laneBsdfScalar[lane] *
							connectionScalar *
							(misWeight * fluxToRadianceFactor);

						const Spectrum avgContribution =
							MLHeroCIEEstimatorColor(
								waveLength, sampleWeight) *
							(laneScalar / (float)laneCount);
						checkSum += avgContribution;

						fprintf(f,
							"phase15s lane %u: lambda=%.9g "
							"lightTP=%.9g bsdf=%.9g connection=%.9g "
							"laneScalar=%.9g avgRGB=(%.9g, %.9g, %.9g)\n",
							lane, waveLength,
							lightScalar,
							laneBsdfScalar[lane],
							connectionScalar,
							laneScalar,
							avgContribution.c[0],
							avgContribution.c[1],
							avgContribution.c[2]);
					}

					fprintf(f,
						"phase15s summed lane contributions=(%.9g, %.9g, %.9g)\n",
						checkSum.c[0], checkSum.c[1], checkSum.c[2]);

					fprintf(f,
						"phase15s visible CIE radiance=(%.9g, %.9g, %.9g)\n",
						radiance.c[0], radiance.c[1], radiance.c[2]);

					fprintf(f,
						"phase15s closure error=(%.9g, %.9g, %.9g)\n",
						checkSum.c[0] - radiance.c[0],
						checkSum.c[1] - radiance.c[1],
						checkSum.c[2] - radiance.c[2]);

					fclose(f);
				}
			}
		}
	}

	if (!mlHero15sUsedSpectral) {
		radiance = (misWeight * fluxToRadianceFactor) *
			connectionThroughput * lightVertex.throughput * bsdfEval;
	}

	MLHeroRecordEstimatorContribution15bj(
		2u, radiance, false,
		lightVertex.mlHeroSecondaryWavelengthsTerminated,
		mlHero15sUsedSpectral, !mlHero15sUsedSpectral);
	if (MLHeroLegacyHeavyDiagnosticsEnabled15cq()) { const unsigned long long read15bq = mlHero15bqReadCount[2].fetch_add(1ull, std::memory_order_relaxed); MLHeroLogState15bq("READ_LIGHT", read15bq, 2u, lightVertex); const unsigned long long snap15bq = mlHero15bqSnapshotTicket.fetch_add(1ull, std::memory_order_relaxed); MLHeroWriteSnapshot15bq(snap15bq); }
	MLHeroRecordContributionWeightedBias15bp(2u, radiance, false,
		lightVertex.mlHeroSecondaryWavelengthsTerminated, 1.f,
		lightVertex.mlHeroTerminationBrightnessRatio);
	MLHeroRecordWavelengthBias15bs(2u, 1u, lightVertex, radiance);

	SampleResult &sampleResult = AddResult(sampleResults, true);
	sampleResult.filmX = filmX;
	sampleResult.filmY = filmY;

	// Add radiance from the light source
	sampleResult.radiance[lightVertex.lightID] = radiance;
}

void BiDirCPURenderThread::DirectLightSampling(const float time,
		const float u0, const float u1, const float u2,
		const float u3, const float u4,
		const PathVertexVM &eyeVertex,
		SampleResult &eyeSampleResult) const {
	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;
	auto& scene = engine->renderConfig.GetScene();

	if (!eyeVertex.bsdf.IsDelta()) {
		// Pick a light source to sample
		const Normal landingNormal = eyeVertex.bsdf.hitPoint.intoObject ? eyeVertex.bsdf.hitPoint.geometryN : -eyeVertex.bsdf.hitPoint.geometryN;
		float lightPickPdf;
		auto light = scene.GetLightSources().GetEmitLightStrategy().SampleLights(
			scene, u0,
			eyeVertex.bsdf.hitPoint.p,
			landingNormal,
			eyeVertex.bsdf.IsVolume(),
			&lightPickPdf
		);

		if (light) {
			Ray shadowRay;
			float directPdfW, emissionPdfW, cosThetaAtLight;
			const Spectrum lightRadiance = light->Illuminate(scene, eyeVertex.bsdf,
					time, u1, u2, u3, shadowRay, directPdfW, &emissionPdfW,
					&cosThetaAtLight);

			if (!lightRadiance.Black()) {
				BSDFEvent event;
				float bsdfPdfW, bsdfRevPdfW;
				const Spectrum bsdfEval = eyeVertex.bsdf.Evaluate(shadowRay.d, &event, &bsdfPdfW, &bsdfRevPdfW);

				if (!bsdfEval.Black()) {
					RayHit shadowRayHit;
					BSDF shadowBsdf;
					Spectrum connectionThroughput;
					PathVolumeInfo volInfo = eyeVertex.volInfo; // I need to use a copy here
					// Check if the light source is visible
					const bool occluded = scene.Intersect(device, EYE_RAY | SHADOW_RAY, &volInfo, u4,
							&shadowRay, &shadowRayHit, &shadowBsdf, &connectionThroughput);

					if (!occluded) {
						// I'm ignoring volume emission because it is not sampled in
						// direct light step.

						// If the light source is not intersectable, it can not be
						// sampled with BSDF
						bsdfPdfW *= (light->IsEnvironmental() || light->IsIntersectable()) ? 1.f : 0.f;

						// The +1 is there to account the current path vertex used for DL
						if (eyeVertex.depth + 1 >= engine->rrDepth) {
							// Russian Roulette
							const float prob = RenderEngine::RussianRouletteProb(bsdfEval, engine->rrImportanceCap);
							bsdfPdfW *= prob;
							bsdfRevPdfW *= prob;
						}

						const float cosThetaToLight = AbsDot(shadowRay.d, eyeVertex.bsdf.hitPoint.shadeN);
						const float directLightSamplingPdfW = directPdfW * lightPickPdf;

						// emissionPdfA / directPdfA = emissionPdfW / directPdfW
						const float weightLight = MIS(bsdfPdfW / directLightSamplingPdfW);
						const float weightCamera = MIS(emissionPdfW * cosThetaToLight / (directPdfW * cosThetaAtLight)) *
								(misVmWeightFactor + eyeVertex.dVCM + eyeVertex.dVC * MIS(bsdfRevPdfW));
						// Disable MIS if we have gone trough a shadow transparent object
						const float misWeight = shadowBsdf.hitPoint.throughShadowTransparency ?
							1.f : (1.f / (weightLight + 1.f + weightCamera));

						// ML HERO phase 15bo: for a HERO-terminated eye subpath, the
						// camera-side history term is the part that can cross the prior
						// dispersive Dirac event.  Keep the local BSDF-vs-light competitor.
						if ((engine->mlHeroGlassMode == 1) && !shadowBsdf.hitPoint.throughShadowTransparency) {
							const bool eyeTerm15bo = eyeVertex.mlHeroSecondaryWavelengthsTerminated;
							const float candidateMis15bo = 1.f / (weightLight + 1.f + (eyeTerm15bo ? 0.f : weightCamera));
							MLHeroRecordMisPrune15bo(1u, eyeTerm15bo, false, misWeight, candidateMis15bo, eyeTerm15bo ? weightCamera : 0.f);
						}

						// Was:
						//  const float factor = cosThetaToLight / directLightSamplingPdfW;
						//
						// but now BSDF::Evaluate() follows LuxRender habit to return the
						// result multiplied by cosThetaToLight
						const float factor = 1.f / directLightSamplingPdfW;

						// ML HERO phase 15q:
						// DirectLightSampling now follows the same spectral architecture
						// validated by ConnectVertices:
						//
						//   scalar eye lane TP
						// * scalar BSDF(lambda)
						// * scalar shadow/connection throughput
						// * RGB lightRadiance -> illuminant SPD(lambda)
						// * classic scalar MIS/PDF factor
						// -> CIE reconstruction at the end.
						//
						// Unsupported/non-scalar cases fall back to original classic RGB.
						Spectrum mlHeroDirectLightRadiance;
						bool mlHero15qUsedSpectral = false;

						if (eyeVertex.mlHeroLaneCount > 1u) {
							const u_int laneCount = eyeVertex.mlHeroLaneCount;
							const bool eyeTPScalar = MLHeroAllLanesScalar(eyeVertex);
							const bool connectionScalar = MLHeroIsScalarSpectrum(connectionThroughput);

							bool endpointUsable = true;
							bool anyExplicitMetal2 = false;
							float laneBsdfScalar[8] = { 0.f };

							for (u_int lane = 0; lane < laneCount; ++lane) {
								const float waveLength = eyeVertex.mlHeroLaneWaveLength[lane];

								Spectrum laneBsdfEval = bsdfEval;
								Spectrum explicitMetal2;
								if (eyeVertex.bsdf.EvaluateMLHeroMetal2AtWaveLength(
										shadowRay.d, waveLength, &explicitMetal2)) {
									laneBsdfEval = explicitMetal2;
									anyExplicitMetal2 = true;
								}

								if (!MLHeroIsScalarSpectrum(laneBsdfEval)) {
									endpointUsable = false;
									break;
								}

								laneBsdfScalar[lane] = laneBsdfEval.c[0];
							}

							if (eyeTPScalar && connectionScalar && endpointUsable) {
								const float connectionScalarValue = connectionThroughput.c[0];
								const RGBColor lightRGB(lightRadiance.c[0],
									lightRadiance.c[1], lightRadiance.c[2]);

								Spectrum ciePacket;

								for (u_int lane = 0; lane < laneCount; ++lane) {
									const float waveLength = eyeVertex.mlHeroLaneWaveLength[lane];
									const float sampleWeight = eyeVertex.mlHeroLaneSampleWeight[lane];

									const float eyeScalar =
										eyeVertex.mlHeroLaneThroughput[lane].c[0];
									const float lightSPD =
										MLHeroRGBIllumSample380_780(lightRGB, waveLength);

									const float laneScalar =
										eyeScalar *
										laneBsdfScalar[lane] *
										connectionScalarValue *
										lightSPD *
										(misWeight * factor);

									ciePacket += MLHeroCIEEstimatorColor(
										waveLength, sampleWeight) * laneScalar;
								}

								ciePacket /= (float)laneCount;
								mlHeroDirectLightRadiance = ciePacket;
								mlHero15qUsedSpectral = true;

								static std::atomic<bool> mlHeroPhase15qLogged(false);
								bool expected = false;
								if (mlHeroPhase15qLogged.compare_exchange_strong(
										expected, true)) {
									std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock6(mlHeroDebugLogMutex);
									FILE *f = MLHeroOpenLegacyLog15cq("a");
									if (f) {
										const Spectrum classicRadiance =
											(misWeight * factor) *
											eyeVertex.throughput *
											connectionThroughput *
											lightRadiance * bsdfEval;

										fprintf(f,
											"ML HERO phase 15q DIRECTLIGHTSAMPLING SPECTRAL ACTIVE: "
											"depth=%u count=%u explicitMetal2=%u\n",
											eyeVertex.depth, laneCount,
											anyExplicitMetal2 ? 1u : 0u);

										fprintf(f,
											"phase15q gate: eyeTP=%u connection=%u endpoint=%u\n",
											eyeTPScalar ? 1u : 0u,
											connectionScalar ? 1u : 0u,
											endpointUsable ? 1u : 0u);

										fprintf(f,
											"phase15q classic radiance=(%.9g, %.9g, %.9g)\n",
											classicRadiance.c[0],
											classicRadiance.c[1],
											classicRadiance.c[2]);

										Spectrum checkSum;
										for (u_int lane = 0; lane < laneCount; ++lane) {
											const float waveLength =
												eyeVertex.mlHeroLaneWaveLength[lane];
											const float sampleWeight =
												eyeVertex.mlHeroLaneSampleWeight[lane];
											const float eyeScalar =
												eyeVertex.mlHeroLaneThroughput[lane].c[0];
											const float lightSPD =
												MLHeroRGBIllumSample380_780(
													lightRGB, waveLength);
											const float laneScalar =
												eyeScalar *
												laneBsdfScalar[lane] *
												connectionScalarValue *
												lightSPD *
												(misWeight * factor);

											const Spectrum avgContribution =
												MLHeroCIEEstimatorColor(
													waveLength, sampleWeight) *
												(laneScalar / (float)laneCount);
											checkSum += avgContribution;

											fprintf(f,
												"phase15q lane %u: lambda=%.9g "
												"eyeTP=%.9g bsdf=%.9g lightSPD=%.9g "
												"connection=%.9g mis=%.9g factor=%.9g "
												"laneScalar=%.9g avgRGB=(%.9g, %.9g, %.9g)\n",
												lane, waveLength,
												eyeScalar,
												laneBsdfScalar[lane],
												lightSPD,
												connectionScalarValue,
												misWeight, factor,
												laneScalar,
												avgContribution.c[0],
												avgContribution.c[1],
												avgContribution.c[2]);
										}

										fprintf(f,
											"phase15q summed lane contributions=(%.9g, %.9g, %.9g)\n",
											checkSum.c[0], checkSum.c[1], checkSum.c[2]);

										fprintf(f,
											"phase15q visible CIE radiance=(%.9g, %.9g, %.9g)\n",
											mlHeroDirectLightRadiance.c[0],
											mlHeroDirectLightRadiance.c[1],
											mlHeroDirectLightRadiance.c[2]);

										fprintf(f,
											"phase15q closure error=(%.9g, %.9g, %.9g)\n",
											checkSum.c[0] - mlHeroDirectLightRadiance.c[0],
											checkSum.c[1] - mlHeroDirectLightRadiance.c[1],
											checkSum.c[2] - mlHeroDirectLightRadiance.c[2]);

										fclose(f);
									}
								}
							}
						}

						if (!mlHero15qUsedSpectral) {
							mlHeroDirectLightRadiance = (misWeight * factor) *
								eyeVertex.throughput *
								connectionThroughput *
								lightRadiance * bsdfEval;
						}

						MLHeroRecordEstimatorContribution15bj(
							1u, mlHeroDirectLightRadiance,
							eyeVertex.mlHeroSecondaryWavelengthsTerminated, false,
							mlHero15qUsedSpectral, !mlHero15qUsedSpectral);
						if (MLHeroLegacyHeavyDiagnosticsEnabled15cq()) { const unsigned long long read15bq = mlHero15bqReadCount[1].fetch_add(1ull, std::memory_order_relaxed); MLHeroLogState15bq("READ_EYE", read15bq, 1u, eyeVertex); const unsigned long long snap15bq = mlHero15bqSnapshotTicket.fetch_add(1ull, std::memory_order_relaxed); MLHeroWriteSnapshot15bq(snap15bq); }
						MLHeroRecordContributionWeightedBias15bp(1u, mlHeroDirectLightRadiance,
							eyeVertex.mlHeroSecondaryWavelengthsTerminated, false,
							eyeVertex.mlHeroTerminationBrightnessRatio, 1.f);
						MLHeroRecordWavelengthBias15bs(1u, 0u, eyeVertex, mlHeroDirectLightRadiance);

						eyeSampleResult.radiance[light->GetID()] += mlHeroDirectLightRadiance;

						assert (eyeSampleResult.IsValid());
					}
				}
			}
		}
	}
}

void BiDirCPURenderThread::DirectHitLight(
	LightSourceConstRef light,
	const Spectrum &lightRadiance,
	const float directPdfA,
	const float emissionPdfW,
	const PathVertexVM &eyeVertex,
	Spectrum *radiance
) const {
	if (lightRadiance.Black())
		return;

	if (eyeVertex.depth == 1) {
		// ML HERO phase 15r:
		// Direct camera hit on a finite/environment light. For multi-wave mode,
		// convert the RGB light radiance to an illuminant SPD at each stored lane
		// and reconstruct RGB through CIE only at the end.
		if ((eyeVertex.mlHeroLaneCount > 1u) && MLHeroAllLanesScalar(eyeVertex)) {
			const u_int laneCount = eyeVertex.mlHeroLaneCount;
			const RGBColor lightRGB(lightRadiance.c[0],
				lightRadiance.c[1], lightRadiance.c[2]);

			Spectrum ciePacket;
			for (u_int lane = 0; lane < laneCount; ++lane) {
				const float waveLength = eyeVertex.mlHeroLaneWaveLength[lane];
				const float sampleWeight = eyeVertex.mlHeroLaneSampleWeight[lane];
				const float eyeScalar = eyeVertex.mlHeroLaneThroughput[lane].c[0];
				const float lightSPD =
					MLHeroRGBIllumSample380_780(lightRGB, waveLength);

				ciePacket += MLHeroCIEEstimatorColor(
					waveLength, sampleWeight) * (eyeScalar * lightSPD);
			}

			ciePacket /= (float)laneCount;
			MLHeroRecordEstimatorContribution15bj(
				3u, ciePacket,
				eyeVertex.mlHeroSecondaryWavelengthsTerminated, false,
				true, false);
			if (MLHeroLegacyHeavyDiagnosticsEnabled15cq()) { const unsigned long long read15bq = mlHero15bqReadCount[3].fetch_add(1ull, std::memory_order_relaxed); MLHeroLogState15bq("READ_EYE", read15bq, 3u, eyeVertex); const unsigned long long snap15bq = mlHero15bqSnapshotTicket.fetch_add(1ull, std::memory_order_relaxed); MLHeroWriteSnapshot15bq(snap15bq); }
			MLHeroRecordContributionWeightedBias15bp(3u, ciePacket,
				eyeVertex.mlHeroSecondaryWavelengthsTerminated, false,
				eyeVertex.mlHeroTerminationBrightnessRatio, 1.f);
			MLHeroRecordWavelengthBias15bs(3u, 0u, eyeVertex, ciePacket);
			*radiance += ciePacket;

			static std::atomic<bool> mlHeroPhase15rDepth1Logged(false);
			bool expected = false;
			if (mlHeroPhase15rDepth1Logged.compare_exchange_strong(expected, true)) {
				std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock7(mlHeroDebugLogMutex);
				FILE *f = MLHeroOpenLegacyLog15cq("a");
				if (f) {
					const Spectrum classicRadiance =
						eyeVertex.throughput * lightRadiance;

					fprintf(f,
						"ML HERO phase 15r DIRECTHITLIGHT SPECTRAL ACTIVE: "
						"depth=1 count=%u mis=1\n", laneCount);

					fprintf(f,
						"phase15r classic radiance=(%.9g, %.9g, %.9g)\n",
						classicRadiance.c[0],
						classicRadiance.c[1],
						classicRadiance.c[2]);

					Spectrum checkSum;
					for (u_int lane = 0; lane < laneCount; ++lane) {
						const float waveLength =
							eyeVertex.mlHeroLaneWaveLength[lane];
						const float sampleWeight =
							eyeVertex.mlHeroLaneSampleWeight[lane];
						const float eyeScalar =
							eyeVertex.mlHeroLaneThroughput[lane].c[0];
						const float lightSPD =
							MLHeroRGBIllumSample380_780(
								lightRGB, waveLength);

						const Spectrum avgContribution =
							MLHeroCIEEstimatorColor(
								waveLength, sampleWeight) *
							((eyeScalar * lightSPD) / (float)laneCount);
						checkSum += avgContribution;

						fprintf(f,
							"phase15r lane %u: lambda=%.9g "
							"eyeTP=%.9g lightSPD=%.9g "
							"avgRGB=(%.9g, %.9g, %.9g)\n",
							lane, waveLength,
							eyeScalar, lightSPD,
							avgContribution.c[0],
							avgContribution.c[1],
							avgContribution.c[2]);
					}

					fprintf(f,
						"phase15r summed lane contributions=(%.9g, %.9g, %.9g)\n",
						checkSum.c[0], checkSum.c[1], checkSum.c[2]);

					fprintf(f,
						"phase15r visible CIE radiance=(%.9g, %.9g, %.9g)\n",
						ciePacket.c[0], ciePacket.c[1], ciePacket.c[2]);

					fprintf(f,
						"phase15r closure error=(%.9g, %.9g, %.9g)\n",
						checkSum.c[0] - ciePacket.c[0],
						checkSum.c[1] - ciePacket.c[1],
						checkSum.c[2] - ciePacket.c[2]);

					fclose(f);
				}
			}
		} else {
			const Spectrum mlHero15bjContribution =
				eyeVertex.throughput * lightRadiance;
			MLHeroRecordEstimatorContribution15bj(
				3u, mlHero15bjContribution,
				eyeVertex.mlHeroSecondaryWavelengthsTerminated, false,
				false, true);
			if (MLHeroLegacyHeavyDiagnosticsEnabled15cq()) { const unsigned long long read15bq = mlHero15bqReadCount[3].fetch_add(1ull, std::memory_order_relaxed); MLHeroLogState15bq("READ_EYE", read15bq, 3u, eyeVertex); const unsigned long long snap15bq = mlHero15bqSnapshotTicket.fetch_add(1ull, std::memory_order_relaxed); MLHeroWriteSnapshot15bq(snap15bq); }
			MLHeroRecordContributionWeightedBias15bp(3u, mlHero15bjContribution,
				eyeVertex.mlHeroSecondaryWavelengthsTerminated, false,
				eyeVertex.mlHeroTerminationBrightnessRatio, 1.f);
			MLHeroRecordWavelengthBias15bs(3u, 0u, eyeVertex, mlHero15bjContribution);
			*radiance += mlHero15bjContribution;
		}

		return;
	}

	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;
	auto& scene = engine->renderConfig.GetScene();

	const float lightPickPdf = scene.GetLightSources().GetEmitLightStrategy().SampleLightPdf(
		light,
		eyeVertex.bsdf.hitPoint.p,
		eyeVertex.bsdf.hitPoint.geometryN,
		eyeVertex.bsdf.IsVolume()
	);

	// MIS weight
	const float weightCamera = MIS(directPdfA * lightPickPdf) * eyeVertex.dVCM +
		MIS(emissionPdfW * lightPickPdf) * eyeVertex.dVC;
	const float misWeight = 1.f / (weightCamera + 1.f);

	// ML HERO phase 15bo: a direct-hit path whose eye subpath already crossed
	// a perfect dispersive HERO-only event is compared against the upper bound
	// where the camera-side alternate-technique history is removed.
	if (engine->mlHeroGlassMode == 1) {
		const bool eyeTerm15bo = eyeVertex.mlHeroSecondaryWavelengthsTerminated;
		const float candidateMis15bo = eyeTerm15bo ? 1.f : misWeight;
		MLHeroRecordMisPrune15bo(3u, eyeTerm15bo, false, misWeight, candidateMis15bo, eyeTerm15bo ? weightCamera : 0.f);
	}

	// ML HERO phase 15r:
	// Replace the old Phase 9 RGB packet estimate with the same scalar-SPD +
	// CIE architecture used by DirectLightSampling and ConnectVertices.
	if ((eyeVertex.mlHeroLaneCount > 1u) && MLHeroAllLanesScalar(eyeVertex)) {
		const u_int laneCount = eyeVertex.mlHeroLaneCount;
		const RGBColor lightRGB(lightRadiance.c[0],
			lightRadiance.c[1], lightRadiance.c[2]);

		Spectrum ciePacket;
		for (u_int lane = 0; lane < laneCount; ++lane) {
			const float waveLength = eyeVertex.mlHeroLaneWaveLength[lane];
			const float sampleWeight = eyeVertex.mlHeroLaneSampleWeight[lane];
			const float eyeScalar = eyeVertex.mlHeroLaneThroughput[lane].c[0];
			const float lightSPD =
				MLHeroRGBIllumSample380_780(lightRGB, waveLength);

			ciePacket += MLHeroCIEEstimatorColor(
				waveLength, sampleWeight) *
				(eyeScalar * lightSPD * misWeight);
		}

		ciePacket /= (float)laneCount;
		MLHeroRecordEstimatorContribution15bj(
			3u, ciePacket,
			eyeVertex.mlHeroSecondaryWavelengthsTerminated, false,
			true, false);
		if (MLHeroLegacyHeavyDiagnosticsEnabled15cq()) { const unsigned long long read15bq = mlHero15bqReadCount[3].fetch_add(1ull, std::memory_order_relaxed); MLHeroLogState15bq("READ_EYE", read15bq, 3u, eyeVertex); const unsigned long long snap15bq = mlHero15bqSnapshotTicket.fetch_add(1ull, std::memory_order_relaxed); MLHeroWriteSnapshot15bq(snap15bq); }
		MLHeroRecordContributionWeightedBias15bp(3u, ciePacket,
			eyeVertex.mlHeroSecondaryWavelengthsTerminated, false,
			eyeVertex.mlHeroTerminationBrightnessRatio, 1.f);
		MLHeroRecordWavelengthBias15bs(3u, 0u, eyeVertex, ciePacket);
		*radiance += ciePacket;

		static std::atomic<bool> mlHeroPhase15rLogged(false);
		bool expected = false;
		if (mlHeroPhase15rLogged.compare_exchange_strong(expected, true)) {
			std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock8(mlHeroDebugLogMutex);
			FILE *f = MLHeroOpenLegacyLog15cq("a");
			if (f) {
				const Spectrum classicRadiance =
					misWeight * eyeVertex.throughput * lightRadiance;

				fprintf(f,
					"ML HERO phase 15r DIRECTHITLIGHT SPECTRAL ACTIVE: "
					"depth=%u count=%u mis=%.9g\n",
					eyeVertex.depth, laneCount, misWeight);

				fprintf(f,
					"phase15r classic radiance=(%.9g, %.9g, %.9g)\n",
					classicRadiance.c[0],
					classicRadiance.c[1],
					classicRadiance.c[2]);

				Spectrum checkSum;
				for (u_int lane = 0; lane < laneCount; ++lane) {
					const float waveLength =
						eyeVertex.mlHeroLaneWaveLength[lane];
					const float sampleWeight =
						eyeVertex.mlHeroLaneSampleWeight[lane];
					const float eyeScalar =
						eyeVertex.mlHeroLaneThroughput[lane].c[0];
					const float lightSPD =
						MLHeroRGBIllumSample380_780(
							lightRGB, waveLength);
					const float laneScalar =
						eyeScalar * lightSPD * misWeight;

					const Spectrum avgContribution =
						MLHeroCIEEstimatorColor(
							waveLength, sampleWeight) *
						(laneScalar / (float)laneCount);
					checkSum += avgContribution;

					fprintf(f,
						"phase15r lane %u: lambda=%.9g "
						"eyeTP=%.9g lightSPD=%.9g mis=%.9g "
						"laneScalar=%.9g avgRGB=(%.9g, %.9g, %.9g)\n",
						lane, waveLength,
						eyeScalar, lightSPD, misWeight,
						laneScalar,
						avgContribution.c[0],
						avgContribution.c[1],
						avgContribution.c[2]);
				}

				fprintf(f,
					"phase15r summed lane contributions=(%.9g, %.9g, %.9g)\n",
					checkSum.c[0], checkSum.c[1], checkSum.c[2]);

				fprintf(f,
					"phase15r visible CIE radiance=(%.9g, %.9g, %.9g)\n",
					ciePacket.c[0], ciePacket.c[1], ciePacket.c[2]);

				fprintf(f,
					"phase15r closure error=(%.9g, %.9g, %.9g)\n",
					checkSum.c[0] - ciePacket.c[0],
					checkSum.c[1] - ciePacket.c[1],
					checkSum.c[2] - ciePacket.c[2]);

				fclose(f);
			}
		}
	} else {
		const Spectrum mlHero15bjContribution =
			misWeight * eyeVertex.throughput * lightRadiance;
		MLHeroRecordEstimatorContribution15bj(
			3u, mlHero15bjContribution,
			eyeVertex.mlHeroSecondaryWavelengthsTerminated, false,
			false, true);
		if (MLHeroLegacyHeavyDiagnosticsEnabled15cq()) { const unsigned long long read15bq = mlHero15bqReadCount[3].fetch_add(1ull, std::memory_order_relaxed); MLHeroLogState15bq("READ_EYE", read15bq, 3u, eyeVertex); const unsigned long long snap15bq = mlHero15bqSnapshotTicket.fetch_add(1ull, std::memory_order_relaxed); MLHeroWriteSnapshot15bq(snap15bq); }
		MLHeroRecordContributionWeightedBias15bp(3u, mlHero15bjContribution,
			eyeVertex.mlHeroSecondaryWavelengthsTerminated, false,
			eyeVertex.mlHeroTerminationBrightnessRatio, 1.f);
		MLHeroRecordWavelengthBias15bs(3u, 0u, eyeVertex, mlHero15bjContribution);
		*radiance += mlHero15bjContribution;
	}

	assert (radiance->IsValid());
}

void BiDirCPURenderThread::DirectHitLight(
	const bool finiteLightSource,
	const PathVertexVM &eyeVertex,
	SampleResult &eyeSampleResult
) const {
	float directPdfA, emissionPdfW;
	if (finiteLightSource) {
		const Spectrum lightRadiance = eyeVertex.bsdf.GetEmittedRadiance(
			&directPdfA, &emissionPdfW
		);

		DirectHitLight(
			*eyeVertex.bsdf.GetLightSource(),
			lightRadiance,
			directPdfA,
			emissionPdfW,
			eyeVertex, &eyeSampleResult.radiance[eyeVertex.bsdf.GetLightID()]
		);
	} else {
		BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;
		auto& scene = engine->renderConfig.GetScene();

		for(EnvLightSource& el: scene.GetLightSources().GetEnvLightSources()) {
			const Spectrum lightRadiance = el.GetRadiance(scene,
					(eyeVertex.depth == 1) ? nullptr : &eyeVertex.bsdf,
					eyeVertex.bsdf.hitPoint.fixedDir, &directPdfA, &emissionPdfW);

			DirectHitLight(el, lightRadiance, directPdfA, emissionPdfW,
					eyeVertex, &eyeSampleResult.radiance[el.GetID()]);
		}
	}
}

bool BiDirCPURenderThread::TraceLightPath(const float time,
		const SamplerUPtr& sampler, CameraConstRef camera,
		vector<PathVertexVM> &lightPathVertices,
		vector<SampleResult> &sampleResults) const {
	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;
	auto& scene = engine->renderConfig.GetScene();

	// Select one light source
	// BiDir can use only a single strategy, emit in this case
	float lightPickPdf;
	auto light = scene.GetLightSources().GetEmitLightStrategy().
			SampleLights(scene, sampler->GetSample(2), &lightPickPdf);
	if (!light)
		return false;

	// Initialize the light path
	PathVertexVM lightVertex;
	lightVertex.lightID = light->GetID();
	
	float lightEmitPdfW, lightDirectPdfW, cosThetaAtLight;
	Ray lightRay;
	lightVertex.throughput = light->Emit(scene,
			time, sampler->GetSample(5), sampler->GetSample(6),
			sampler->GetSample(7), sampler->GetSample(8), sampler->GetSample(9),
			lightRay, lightEmitPdfW,
			&lightDirectPdfW, &cosThetaAtLight);
	if (!lightVertex.throughput.Black()) {
		lightVertex.volInfo.AddVolume(light->volume);

		// ML HERO phase 15e: capture the light source contribution before and after
		// BIDIR's PDF normalization. Log environmental and finite/intersectable
		// sources separately so an HDRI and an area light can both be identified.
		const Spectrum mlHeroRawLightEmit = lightVertex.throughput;
		const float mlHeroRawEmitPdfW = lightEmitPdfW;
		const float mlHeroRawDirectPdfW = lightDirectPdfW;

		lightEmitPdfW *= lightPickPdf;
		lightDirectPdfW *= lightPickPdf;

		lightVertex.throughput /= lightEmitPdfW;
		lightVertex.depth = 0u;

		// ML HERO phase 15i ACTIVE (diagnostic light-lane architecture only):
		// keep the classic RGB throughput unchanged, but initialize the parallel
		// HERO light lanes as true scalar SPD samples.
		if (GetMLHeroWavelengthCount() > 1u)
			MLHeroInitSpectralLightLaneThroughput(&lightVertex, mlHeroRawLightEmit, lightEmitPdfW);
		else
			MLHeroInitLaneThroughput(&lightVertex, lightVertex.throughput);
		if (GetMLHeroWavelengthCount() > 1u) {
			static std::atomic<bool> loggedEnvLightInit(false);
			static std::atomic<bool> loggedFiniteLightInit(false);
			std::atomic<bool> &logFlag = light->IsEnvironmental() ? loggedEnvLightInit : loggedFiniteLightInit;
			bool expected = false;
			if (logFlag.compare_exchange_strong(expected, true)) {
				std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock9(mlHeroDebugLogMutex);
				FILE *f = MLHeroOpenLegacyLog15cq("a");
				if (f) {
					fprintf(f, "ML HERO phase 15e LIGHT_SOURCE_INIT: id=%u environmental=%u intersectable=%u\n",
						light->GetID(), light->IsEnvironmental() ? 1u : 0u, light->IsIntersectable() ? 1u : 0u);
					fprintf(f, "phase15e raw Emit=(%.9g, %.9g, %.9g) scalar=%u\n",
						mlHeroRawLightEmit.c[0], mlHeroRawLightEmit.c[1], mlHeroRawLightEmit.c[2], MLHeroIsScalarSpectrum(mlHeroRawLightEmit) ? 1u : 0u);
					fprintf(f, "phase15e pdfs: lightPickPdf=%.9g rawEmitPdfW=%.9g rawDirectPdfW=%.9g combinedEmitPdfW=%.9g combinedDirectPdfW=%.9g cosThetaAtLight=%.9g\n",
						lightPickPdf, mlHeroRawEmitPdfW, mlHeroRawDirectPdfW, lightEmitPdfW, lightDirectPdfW, cosThetaAtLight);
					fprintf(f, "phase15e normalized light throughput=(%.9g, %.9g, %.9g) scalar=%u\n",
						lightVertex.throughput.c[0], lightVertex.throughput.c[1], lightVertex.throughput.c[2], MLHeroIsScalarSpectrum(lightVertex.throughput) ? 1u : 0u);
					fprintf(f, "phase15e emitted ray dir=(%.9g, %.9g, %.9g)\n", lightRay.d.x, lightRay.d.y, lightRay.d.z);

					// ML HERO phase 15i:
					// The parallel LIGHT lanes are now true scalar SPD samples.
					// Reconstruct them with the CIE estimator to verify that their
					// packet energy is consistent with the classic RGB throughput.
					Spectrum phase15iCIERecon;
					fprintf(f, "ML HERO phase 15i SCALAR LIGHT LANE INIT ACTIVE:\n");
					for (u_int lane = 0; lane < lightVertex.mlHeroLaneCount; ++lane) {
						const float lambda = lightVertex.mlHeroLaneWaveLength[lane];
						const float weight = lightVertex.mlHeroLaneSampleWeight[lane];
						const Spectrum &laneTP = lightVertex.mlHeroLaneThroughput[lane];
						const float scalar = laneTP.c[0];
						const Spectrum cieColor = MLHeroCIEEstimatorColor(lambda, weight);
						phase15iCIERecon += cieColor * scalar;

						fprintf(f,
							"phase15i lane %u: lambda=%.9g weight=%.9g scalarTP=%.9g "
							"isScalar=%u cieColor=(%.9g,%.9g,%.9g)\n",
							lane, lambda, weight, scalar,
							MLHeroIsScalarSpectrum(laneTP) ? 1u : 0u,
							cieColor.c[0], cieColor.c[1], cieColor.c[2]);
					}
					phase15iCIERecon /= (float)lightVertex.mlHeroLaneCount;

					const Spectrum phase15iErr = phase15iCIERecon - lightVertex.throughput;
					fprintf(f, "phase15i classic normalized throughput=(%.9g, %.9g, %.9g)\n",
						lightVertex.throughput.c[0], lightVertex.throughput.c[1], lightVertex.throughput.c[2]);
					fprintf(f, "phase15i scalar-lane CIE recon=(%.9g, %.9g, %.9g) error=(%.9g, %.9g, %.9g)\n",
						phase15iCIERecon.c[0], phase15iCIERecon.c[1], phase15iCIERecon.c[2],
						phase15iErr.c[0], phase15iErr.c[1], phase15iErr.c[2]);
					fclose(f);
				}
				MLHeroLogLanePurityBlock(light->IsEnvironmental() ? "LIGHT_INIT_ENV" : "LIGHT_INIT_FINITE", lightVertex);
			}
		}
		assert (!lightVertex.throughput.IsNaN() && !lightVertex.throughput.IsInf());

		// I don't store the light vertex 0 because direct lighting will take
		// care of these kind of paths
		lightVertex.dVCM = MIS(lightDirectPdfW / lightEmitPdfW);
		// If the light source is not intersectable, it can not be
		// sampled with BSDF
		if (light->IsEnvironmental() || light->IsIntersectable()) {
			const float usedCosLight = light->IsEnvironmental() ? 1.f : cosThetaAtLight;
			lightVertex.dVC = MIS(usedCosLight / lightEmitPdfW);
		} else
			lightVertex.dVC = 0.f;
		lightVertex.dVM = lightVertex.dVC * misVcWeightFactor;

		lightVertex.depth = 1;
		while (lightVertex.depth <= engine->maxLightPathDepth) {
			const u_int sampleOffset = sampleBootSize + (lightVertex.depth - 1) * sampleLightStepSize;

			RayHit nextEventRayHit;
			Spectrum connectionThroughput;
			const bool hit = scene.Intersect(device, LIGHT_RAY | INDIRECT_RAY,
					&lightVertex.volInfo, sampler->GetSample(sampleOffset),
					&lightRay, &nextEventRayHit, &lightVertex.bsdf,
					&connectionThroughput);

			if (hit) {
				// Something was hit
				
				// Check if it is something with a not black shadow transparency
				// and stop if it has. Direct light sampling will take care of
				// this kind of paths.
				if (!lightVertex.bsdf.GetPassThroughShadowTransparency().Black() & !lightVertex.bsdf.GetPassThroughShadowTransparencyOverride())
					break;

				// Update the new light vertex
				lightVertex.throughput *= connectionThroughput;
				MLHeroMultiplyLaneThroughput(&lightVertex, connectionThroughput);

				// ML HERO phase 15j: trace scalar purity after the connection factor.
				MLHeroLogLightScalarTrace(lightVertex, "after-connection");
		
				// Infinite lights use MIS based on solid angle instead of area
				if((lightVertex.depth > 1) || !light->IsEnvironmental())
					lightVertex.dVCM *= MIS(nextEventRayHit.t * nextEventRayHit.t);
				const float factor = 1.f / MIS(AbsDot(lightVertex.bsdf.hitPoint.shadeN, lightRay.d));
				lightVertex.dVCM *= factor;
				lightVertex.dVC *= factor;
				lightVertex.dVM *= factor;

				// Store the vertex only if it isn't specular
				if (!lightVertex.bsdf.IsDelta()) {
					lightPathVertices.push_back(lightVertex);

					//----------------------------------------------------------
					// Try to connect the light path vertex with the eye
					//----------------------------------------------------------

					// Sample a point on the camera lens
					Point lensPoint;
					camera.SampleLens(time, sampler->GetSample(3), sampler->GetSample(4), &lensPoint);

					ConnectToEye(time, lightVertex, sampler->GetSample(sampleOffset + 1),
							lensPoint, sampleResults);
				}

				if (lightVertex.depth >= engine->maxLightPathDepth)
					break;

				//--------------------------------------------------------------
				// Build the next vertex path ray
				//--------------------------------------------------------------

				MLHeroLogLightScalarTrace(lightVertex, "before-bounce");

				if (!Bounce(time, sampler, sampleOffset + 2, &lightVertex, &lightRay))
					break;

				// ML HERO phase 15j: trace scalar purity over multiple BSDF bounces.
				MLHeroLogLightScalarTrace(lightVertex, "after-bounce");
			} else {
				// Ray lost in space...
				break;
			}
		}
	}
	
	return true;
}

bool BiDirCPURenderThread::Bounce(const float time, const SamplerUPtr& sampler,
		const u_int sampleOffset, PathVertexVM *pathVertex, Ray *nextEventRay) const {
	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;

	// phase15bu: capture the surviving HERO throughput at the start of this bounce.
	// The reference throughput is propagated with the exact same scalar transport
	// multiplier after the bounce, avoiding any reconstruction via hero/packet ratio.
	const bool mlHero15buTerminatedAtBounceStart = pathVertex->mlHeroSecondaryWavelengthsTerminated;
	const double mlHero15buActualBeforeBounce =
		(mlHero15buTerminatedAtBounceStart && (pathVertex->mlHeroLaneCount > 0u)) ?
		(double)pathVertex->mlHeroLaneThroughput[0].c[0] : 0.0;

	bool mlHeroBeforeScalar[8] = { true, true, true, true, true, true, true, true };
	Spectrum mlHeroBeforeTP[8];
	if (GetMLHeroWavelengthCount() > 1u) {
		for (u_int lane = 0; lane < pathVertex->mlHeroLaneCount; ++lane) {
			mlHeroBeforeTP[lane] = pathVertex->mlHeroLaneThroughput[lane];
			mlHeroBeforeScalar[lane] = MLHeroIsScalarSpectrum(mlHeroBeforeTP[lane]);
		}
	}

	Vector sampledDir;
	BSDFEvent &event = pathVertex->bsdfEvent;
	float bsdfPdfW, cosSampledDir;
	const Spectrum bsdfSample = pathVertex->bsdf.Sample(&sampledDir,
			sampler->GetSample(sampleOffset),
			sampler->GetSample(sampleOffset + 1),
			&bsdfPdfW, &cosSampledDir, &event);
	if (bsdfSample.Black())
		return false;

	float bsdfRevPdfW;
	if ((GetMLHeroWavelengthCount() > 1u) && (event & SPECULAR)) {
		static std::atomic<bool> mlHeroPhase15xFirstSpecularLogged(false);
		bool expected = false;
		if (mlHeroPhase15xFirstSpecularLogged.compare_exchange_strong(expected, true)) {
			std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock15xSpec(mlHeroDebugLogMutex);
			FILE *f = MLHeroOpenLegacyLog15cq("a");
			if (f) {
				const float bsdfMin = Min(bsdfSample.c[0], Min(bsdfSample.c[1], bsdfSample.c[2]));
				const float bsdfMax = Max(bsdfSample.c[0], Max(bsdfSample.c[1], bsdfSample.c[2]));
				fprintf(f,
					"ML HERO phase 15x FIRST SPECULAR BOUNCE: depth=%u event=%u "
					"REFLECT=%u TRANSMIT=%u DIFFUSE=%u GLOSSY=%u SPECULAR=%u "
					"bsdfPdfW=%.9g cosSampledDir=%.9g bsdfSpread=%.9g "
					"classicBSDF=(%.9g, %.9g, %.9g)\n",
					pathVertex->depth, (u_int)event,
					(event & REFLECT) ? 1u : 0u,
					(event & TRANSMIT) ? 1u : 0u,
					(event & DIFFUSE) ? 1u : 0u,
					(event & GLOSSY) ? 1u : 0u,
					(event & SPECULAR) ? 1u : 0u,
					bsdfPdfW, cosSampledDir, bsdfMax - bsdfMin,
					bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2]);
				fclose(f);
			}
		}
	}


	// ML HERO phase 15ay: A/B validation for spectral Glass weighting. Keep the
	// actual HERO direction, event and renderer multiplier untouched, but compare
	// the current shared HERO multiplier with an ideal per-lane spectral multiplier
	// (including BSDF::Sample() adjoint correction on light sub-paths). Diagnostic
	// only: no throughput, event probability or sampled direction is changed here.
	if ((GetMLHeroWavelengthCount() > 1u) &&
			(pathVertex->bsdf.GetMaterialType() == GLASS) && (event & SPECULAR)) {
		const bool fromLight15ay = pathVertex->bsdf.hitPoint.fromLight;
		const bool reflect15ay = (event & REFLECT) != 0;
		const bool transmit15ay = (event & TRANSMIT) != 0;

		static std::atomic<bool> mlHero15ayEyeTransmitLogged(false);
		static std::atomic<bool> mlHero15ayLightTransmitLogged(false);
		static std::atomic<bool> mlHero15ayEyeReflectLogged(false);
		static std::atomic<bool> mlHero15ayLightReflectLogged(false);
		std::atomic<bool> *logGate15ay = nullptr;
		const char *case15ay = "UNKNOWN";
		if (transmit15ay && !fromLight15ay) {
			logGate15ay = &mlHero15ayEyeTransmitLogged;
			case15ay = "EYE_TRANSMIT";
		} else if (transmit15ay && fromLight15ay) {
			logGate15ay = &mlHero15ayLightTransmitLogged;
			case15ay = "LIGHT_TRANSMIT";
		} else if (reflect15ay && !fromLight15ay) {
			logGate15ay = &mlHero15ayEyeReflectLogged;
			case15ay = "EYE_REFLECT";
		} else if (reflect15ay && fromLight15ay) {
			logGate15ay = &mlHero15ayLightReflectLogged;
			case15ay = "LIGHT_REFLECT";
		}

		bool doLog15ay = false;
		if (logGate15ay) {
			bool expected15ay = false;
			doLog15ay = logGate15ay->compare_exchange_strong(expected15ay, true);
		}
		if (doLog15ay) {
			std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock15ay(mlHeroDebugLogMutex);
			FILE *f15ay = MLHeroOpenLegacyLog15cq("a");
			if (f15ay) {
				const float actualScalar15ay = bsdfSample.Filter();
				const float absDotFixedNG15ay = AbsDot(pathVertex->bsdf.hitPoint.fixedDir,
					pathVertex->bsdf.hitPoint.geometryN);
				const float absDotSampledNG15ay = AbsDot(sampledDir,
					pathVertex->bsdf.hitPoint.geometryN);
				const float adjointCorrection15ay = fromLight15ay ?
					((absDotFixedNG15ay > 1e-20f) ?
						(absDotSampledNG15ay / absDotFixedNG15ay) : 0.f) : 1.f;
				const char *transportMode15ay = fromLight15ay ? "IMPORTANCE" : "RADIANCE";

				fprintf(f15ay,
					"ML HERO phase 15ay GLASS WRAPPED ACTUAL VS IDEAL: case=%s transportMode=%s depth=%u name=%s event=%u REFLECT=%u TRANSMIT=%u fromLight=%u lanes=%u pdfW=%.9g cosSampledDir=%.9g absDotFixedNG=%.9g absDotSampledNG=%.9g adjointCorrection=%.9g actualMultiplier=(%.9g, %.9g, %.9g) actualScalar=%.9g\n",
					case15ay, transportMode15ay, pathVertex->depth,
					pathVertex->bsdf.GetMaterialName().c_str(), (u_int)event,
					reflect15ay ? 1u : 0u, transmit15ay ? 1u : 0u, fromLight15ay ? 1u : 0u,
					pathVertex->mlHeroLaneCount, bsdfPdfW, cosSampledDir,
					absDotFixedNG15ay, absDotSampledNG15ay, adjointCorrection15ay,
					bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2], actualScalar15ay);

				float maxAbsWrappedRelError15ay = 0.f;
				float maxAbsMaterialRelError15ay = 0.f;
				float maxDirDelta15ay = 0.f;
				float minCandidateCorrection15ay = 1e30f;
				float maxCandidateCorrection15ay = 0.f;
				Spectrum currentSharedPacket15ay;
				Spectrum perLaneCandidatePacket15ay;
				for (u_int lane = 0; lane < pathVertex->mlHeroLaneCount; ++lane) {
					float nc15ay = 0.f, ntBase15ay = 0.f, ntLambda15ay = 0.f;
					float fresnel15ay = 0.f, eta15ay = 0.f, eta2_15ay = 0.f;
					float transport15ay = 0.f, dirDelta15ay = 0.f;
					Spectrum materialIdealMultiplier15ay;
					const float lambda15ay = pathVertex->mlHeroLaneWaveLength[lane];
					const bool ok15ay = pathVertex->bsdf.EvaluateMLHeroGlassDebugAtWaveLength(
						sampledDir, event, bsdfPdfW, lambda15ay,
						&nc15ay, &ntBase15ay, &ntLambda15ay,
						&fresnel15ay, &eta15ay, &eta2_15ay, &transport15ay,
						&dirDelta15ay, &materialIdealMultiplier15ay);

					const float materialIdealScalar15ay = materialIdealMultiplier15ay.Filter();
					const Spectrum wrappedIdealMultiplier15ay =
						materialIdealMultiplier15ay * adjointCorrection15ay;
					const float wrappedIdealScalar15ay = wrappedIdealMultiplier15ay.Filter();

					const float actualOverMaterialIdeal15ay =
						(ok15ay && fabsf(materialIdealScalar15ay) > 1e-20f) ?
						(actualScalar15ay / materialIdealScalar15ay) : 0.f;
					const float materialRelError15ay =
						(ok15ay && fabsf(materialIdealScalar15ay) > 1e-20f) ?
						((actualScalar15ay - materialIdealScalar15ay) / materialIdealScalar15ay) : 0.f;
					const float actualOverWrappedIdeal15ay =
						(ok15ay && fabsf(wrappedIdealScalar15ay) > 1e-20f) ?
						(actualScalar15ay / wrappedIdealScalar15ay) : 0.f;
					const float wrappedRelError15ay =
						(ok15ay && fabsf(wrappedIdealScalar15ay) > 1e-20f) ?
						((actualScalar15ay - wrappedIdealScalar15ay) / wrappedIdealScalar15ay) : 0.f;
					const float candidateCorrection15ay =
						(ok15ay && fabsf(actualScalar15ay) > 1e-20f) ?
						(wrappedIdealScalar15ay / actualScalar15ay) : 1.f;

					if (ok15ay) {
						minCandidateCorrection15ay = Min(minCandidateCorrection15ay, candidateCorrection15ay);
						maxCandidateCorrection15ay = Max(maxCandidateCorrection15ay, candidateCorrection15ay);
						const Spectrum cieColor15ay = MLHeroCIEEstimatorColor(lambda15ay, 1.f);
						currentSharedPacket15ay += mlHeroBeforeTP[lane] * actualScalar15ay * cieColor15ay;
						perLaneCandidatePacket15ay += mlHeroBeforeTP[lane] * wrappedIdealScalar15ay * cieColor15ay;
					}

					maxAbsMaterialRelError15ay = Max(maxAbsMaterialRelError15ay, fabsf(materialRelError15ay));
					maxAbsWrappedRelError15ay = Max(maxAbsWrappedRelError15ay, fabsf(wrappedRelError15ay));
					maxDirDelta15ay = Max(maxDirDelta15ay, dirDelta15ay);

					fprintf(f15ay,
						"phase15ay lane %u: lambda=%.9g ok=%u nc=%.9g ntBase=%.9g ntLambda=%.9g fresnelR=%.9g eta=%.9g eta2=%.9g materialTransport=%.9g dirDelta=%.9g materialIdealScalar=%.9g adjointCorrection=%.9g wrappedIdealScalar=%.9g actualScalar=%.9g actualOverMaterialIdeal=%.9g materialRelError=%.9g actualOverWrappedIdeal=%.9g wrappedRelError=%.9g candidateCorrection=%.9g wrappedIdealMultiplier=(%.9g, %.9g, %.9g)\n",
						lane, lambda15ay, ok15ay ? 1u : 0u,
						nc15ay, ntBase15ay, ntLambda15ay, fresnel15ay,
						eta15ay, eta2_15ay, transport15ay, dirDelta15ay,
						materialIdealScalar15ay, adjointCorrection15ay, wrappedIdealScalar15ay,
						actualScalar15ay, actualOverMaterialIdeal15ay, materialRelError15ay,
						actualOverWrappedIdeal15ay, wrappedRelError15ay, candidateCorrection15ay,
						wrappedIdealMultiplier15ay.c[0], wrappedIdealMultiplier15ay.c[1],
						wrappedIdealMultiplier15ay.c[2]);
				}
				if (pathVertex->mlHeroLaneCount > 0u) {
					currentSharedPacket15ay /= (float)pathVertex->mlHeroLaneCount;
					perLaneCandidatePacket15ay /= (float)pathVertex->mlHeroLaneCount;
				}
				const Spectrum packetDelta15ay = perLaneCandidatePacket15ay - currentSharedPacket15ay;
				const float currentPacketFilter15ay = currentSharedPacket15ay.Filter();
				const float candidatePacketFilter15ay = perLaneCandidatePacket15ay.Filter();
				const float packetFilterRelDelta15ay = (fabsf(currentPacketFilter15ay) > 1e-20f) ?
					((candidatePacketFilter15ay - currentPacketFilter15ay) / currentPacketFilter15ay) : 0.f;

				fprintf(f15ay,
					"phase15ay candidatePacket: currentShared=(%.9g, %.9g, %.9g) perLaneIdeal=(%.9g, %.9g, %.9g) delta=(%.9g, %.9g, %.9g) currentFilter=%.9g candidateFilter=%.9g filterRelDelta=%.9g correctionRange=[%.9g, %.9g]\n",
					currentSharedPacket15ay.c[0], currentSharedPacket15ay.c[1], currentSharedPacket15ay.c[2],
					perLaneCandidatePacket15ay.c[0], perLaneCandidatePacket15ay.c[1], perLaneCandidatePacket15ay.c[2],
					packetDelta15ay.c[0], packetDelta15ay.c[1], packetDelta15ay.c[2],
					currentPacketFilter15ay, candidatePacketFilter15ay, packetFilterRelDelta15ay,
					minCandidateCorrection15ay, maxCandidateCorrection15ay);
				fprintf(f15ay,
					"phase15ay summary: case=%s transportMode=%s maxAbsMaterialRelError=%.9g maxAbsWrappedRelError=%.9g maxDirDelta=%.9g adjointCorrection=%.9g note=A_B_compare_shared_HERO_vs_per_lane_spectral_weight_same_HERO_geometry\n",
					case15ay, transportMode15ay, maxAbsMaterialRelError15ay,
					maxAbsWrappedRelError15ay, maxDirDelta15ay, adjointCorrection15ay);
				fclose(f15ay);
			}
		}
	}

	if (event & SPECULAR)
		bsdfRevPdfW = bsdfPdfW;
	else
		pathVertex->bsdf.Pdf(sampledDir, NULL, &bsdfRevPdfW);

	if (pathVertex->depth >= engine->rrDepth) {
		// Russian Roulette
		const float prob = RenderEngine::RussianRouletteProb(bsdfSample, engine->rrImportanceCap);
		if (prob < sampler->GetSample(sampleOffset + 2))
			return false;

		pathVertex->throughput /= prob;
		MLHeroDivideLaneThroughput(pathVertex, prob);
	}

	// Was:
	//  pathVertex->throughput *= bsdfSample * (cosSampledDir / bsdfPdfW);
	//
	// but now BSDF::Sample() follows LuxRender habit to return the
	// result multiplied by cosSampledDir / bsdfPdfW
	// ML-HERO phase 6: propagate a parallel diagnostic throughput for every
	// wavelength lane. For measured Metal2, evaluate the same sampled bounce
	// explicitly at each lane wavelength. All other materials currently share
	// the classic BSDF multiplier. These lanes still do NOT contribute to film.
	bool usedExplicitMetal2 = false;
	bool usedGenericReflectanceSPD = false;
	const bool mlHeroNearScalarCanonicalized =
		(GetMLHeroWavelengthCount() > 1u) &&
		MLHeroIsScalarSpectrum(bsdfSample) &&
		((bsdfSample.c[0] != bsdfSample.c[1]) ||
		 (bsdfSample.c[1] != bsdfSample.c[2]));
	if (GetMLHeroWavelengthCount() > 1u) {
		// ML HERO phase 15z:
		// Bounce evaluation must use the wavelength packet owned by this
		// PathVertexVM. The thread-local/global packet may already have moved on
		// to another sampled path by the time this vertex is evaluated.
		static std::atomic<bool> mlHeroPhase15zOwnershipLogged(false);
		bool expected15z = false;
		if (mlHeroPhase15zOwnershipLogged.compare_exchange_strong(expected15z, true)) {
			std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock15z(mlHeroDebugLogMutex);
			FILE *f = MLHeroOpenLegacyLog15cq("a");
			if (f) {
				fprintf(f, "ML HERO phase 15z VERTEX-OWNED BOUNCE WAVELENGTHS ACTIVE: depth=%u count=%u\n",
					pathVertex->depth, pathVertex->mlHeroLaneCount);
				for (u_int lane = 0; lane < pathVertex->mlHeroLaneCount; ++lane) {
					fprintf(f, "phase15z lane %u: vertexLambda=%.9g globalLambda=%.9g delta=%.9g\n",
						lane,
						pathVertex->mlHeroLaneWaveLength[lane],
						GetMLHeroWaveLengthAt(lane),
						pathVertex->mlHeroLaneWaveLength[lane] - GetMLHeroWaveLengthAt(lane));
				}
				fclose(f);
			}
		}

		// ML HERO phase 15v:
		// For ordinary colored reflective bounces, do not copy the RGB BSDF
		// multiplier into every wavelength lane. Convert the sampled RGB
		// reflectance multiplier to a Smits reflectance SPD and sample it at
		// each lane wavelength. Keep specular events out of this first step so
		// Glass/Fresnel remains on its dedicated spectral path.
		const bool genericReflectanceCandidate =
			(event & REFLECT) && !(event & SPECULAR) &&
			!MLHeroIsScalarSpectrum(bsdfSample);

		// ML HERO phase 15ad: the 15ac one-shot was consumed by an achromatic
		// Glossy2 material ("back"). Evaluate each Glossy2 hit first and only
		// latch the one-shot when Kd is genuinely chromatic. This should catch
		// the fully green test sphere without changing render math.
		if (pathVertex->bsdf.GetMaterialType() == GLOSSY2) {
			Spectrum kd15ad, ks15ad, s15ad, absorption15ad;
			Spectrum base15ad, coating15ad, combined15ad;
			float wrapper15ad = 1.f;

			const bool ok15ad =
				pathVertex->bsdf.EvaluateMLHeroGlossy2DebugSampleComponents(
					sampledDir, bsdfPdfW,
					&kd15ad, &ks15ad, &s15ad, &absorption15ad,
					&base15ad, &coating15ad, &wrapper15ad, &combined15ad);

			const float kdMin15ad = Min(kd15ad.c[0], Min(kd15ad.c[1], kd15ad.c[2]));
			const float kdMax15ad = Max(kd15ad.c[0], Max(kd15ad.c[1], kd15ad.c[2]));
			const bool chromaticKd15ad = ok15ad && ((kdMax15ad - kdMin15ad) > 1e-4f);

			if (chromaticKd15ad) {
				static std::atomic<bool> mlHeroPhase15adLogged(false);
				bool expected15ad = false;
				if (mlHeroPhase15adLogged.compare_exchange_strong(expected15ad, true)) {
					const Spectrum closure15ad = combined15ad - bsdfSample;
					const float closureMax15ad = Max(fabsf(closure15ad.c[0]),
						Max(fabsf(closure15ad.c[1]), fabsf(closure15ad.c[2])));
					const float classicMax15ad = Max(fabsf(bsdfSample.c[0]),
						Max(fabsf(bsdfSample.c[1]), fabsf(bsdfSample.c[2])));
					const float relClosure15ad = (classicMax15ad > 1e-12f) ?
						(closureMax15ad / classicMax15ad) : 0.f;

					std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock15ad(
						mlHeroDebugLogMutex);
					FILE *f15ad = MLHeroOpenLegacyLog15cq("a");
					if (f15ad) {
						fprintf(f15ad,
							"ML HERO phase 15ad CHROMATIC GLOSSY2 COMPONENT SPLIT: "
							"depth=%u event=%u name=%s pdfW=%.9g cosSampledDir=%.9g wrapper=%.9g\n",
							pathVertex->depth, (u_int)event,
							pathVertex->bsdf.GetMaterialName().c_str(),
							bsdfPdfW, cosSampledDir, wrapper15ad);
						fprintf(f15ad,
							"phase15ad kd=(%.9g, %.9g, %.9g) ks=(%.9g, %.9g, %.9g)\n",
							kd15ad.c[0], kd15ad.c[1], kd15ad.c[2],
							ks15ad.c[0], ks15ad.c[1], ks15ad.c[2]);
						fprintf(f15ad,
							"phase15ad SchlickS=(%.9g, %.9g, %.9g) "
							"absorption=(%.9g, %.9g, %.9g)\n",
							s15ad.c[0], s15ad.c[1], s15ad.c[2],
							absorption15ad.c[0], absorption15ad.c[1], absorption15ad.c[2]);
						fprintf(f15ad,
							"phase15ad base=(%.9g, %.9g, %.9g) "
							"coating=(%.9g, %.9g, %.9g)\n",
							base15ad.c[0], base15ad.c[1], base15ad.c[2],
							coating15ad.c[0], coating15ad.c[1], coating15ad.c[2]);
						fprintf(f15ad,
							"phase15ad combined=(%.9g, %.9g, %.9g) "
							"classicBSDF=(%.9g, %.9g, %.9g) "
							"closure=(%.9g, %.9g, %.9g) relativeClosure=%.9g\n",
							combined15ad.c[0], combined15ad.c[1], combined15ad.c[2],
							bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2],
							closure15ad.c[0], closure15ad.c[1], closure15ad.c[2],
							relClosure15ad);
						fclose(f15ad);
					}
				}
			}
		}

		// ML HERO phase 15aa: diagnostic only.
		// Compare the sampled RGB BSDF multiplier with the material's Albedo().
		// BSDF::Sample() already contains cos/pdf transport factors, so feeding
		// bsdfSample directly to RGBReflSPD can overdrive the reconstruction.
		// If bsdfSample/albedo is approximately the same scalar in all channels,
		// we have a clean way to separate spectral color from achromatic transport.
		if (genericReflectanceCandidate) {
			static std::atomic<bool> mlHeroPhase15abLogged(false);
			bool expected15ab = false;
			if (mlHeroPhase15abLogged.compare_exchange_strong(expected15ab, true)) {
				const Spectrum albedo15ab = pathVertex->bsdf.Albedo();
				const float denom15ab =
					albedo15ab.c[0] * albedo15ab.c[0] +
					albedo15ab.c[1] * albedo15ab.c[1] +
					albedo15ab.c[2] * albedo15ab.c[2];
				const float factor15ab = (denom15ab > 1e-12f) ?
					((bsdfSample.c[0] * albedo15ab.c[0] +
					  bsdfSample.c[1] * albedo15ab.c[1] +
					  bsdfSample.c[2] * albedo15ab.c[2]) / denom15ab) : 0.f;
				const Spectrum reconstructed15ab = albedo15ab * factor15ab;
				const Spectrum residual15ab = bsdfSample - reconstructed15ab;
				const float residualMax15ab = Max(fabsf(residual15ab.c[0]),
					Max(fabsf(residual15ab.c[1]), fabsf(residual15ab.c[2])));
				const float bsdfMax15ab = Max(fabsf(bsdfSample.c[0]),
					Max(fabsf(bsdfSample.c[1]), fabsf(bsdfSample.c[2])));
				const float relResidual15ab = (bsdfMax15ab > 1e-12f) ?
					(residualMax15ab / bsdfMax15ab) : 0.f;

				std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock15ab(mlHeroDebugLogMutex);
				FILE *f15ab = MLHeroOpenLegacyLog15cq("a");
				if (f15ab) {
					fprintf(f15ab,
						"ML HERO phase 15ab REFLECTANCE DECOMPOSITION: depth=%u event=%u name=%s\n",
						pathVertex->depth, (u_int)event,
						pathVertex->bsdf.GetMaterialName().c_str());
					fprintf(f15ab,
						"phase15ab bsdf=(%.9g, %.9g, %.9g) albedo=(%.9g, %.9g, %.9g) factor=%.9g\n",
						bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2],
						albedo15ab.c[0], albedo15ab.c[1], albedo15ab.c[2], factor15ab);
					fprintf(f15ab,
						"phase15ab reconstructed=(%.9g, %.9g, %.9g) residual=(%.9g, %.9g, %.9g) "
						"maxAbsResidual=%.9g relativeResidual=%.9g\n",
						reconstructed15ab.c[0], reconstructed15ab.c[1], reconstructed15ab.c[2],
						residual15ab.c[0], residual15ab.c[1], residual15ab.c[2],
						residualMax15ab, relResidual15ab);
					fclose(f15ab);
				}
			}

			static std::atomic<bool> mlHeroPhase15aaLogged(false);
			bool expected15aa = false;
			if (mlHeroPhase15aaLogged.compare_exchange_strong(expected15aa, true)) {
				const Spectrum albedo = pathVertex->bsdf.Albedo();
				const float ratioR = (fabsf(albedo.c[0]) > 1e-9f) ? (bsdfSample.c[0] / albedo.c[0]) : 0.f;
				const float ratioG = (fabsf(albedo.c[1]) > 1e-9f) ? (bsdfSample.c[1] / albedo.c[1]) : 0.f;
				const float ratioB = (fabsf(albedo.c[2]) > 1e-9f) ? (bsdfSample.c[2] / albedo.c[2]) : 0.f;
				const float ratioMin = Min(ratioR, Min(ratioG, ratioB));
				const float ratioMax = Max(ratioR, Max(ratioG, ratioB));

				std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock15aa(mlHeroDebugLogMutex);
				FILE *f = MLHeroOpenLegacyLog15cq("a");
				if (f) {
					fprintf(f,
						"ML HERO phase 15aa GENERIC REFLECTANCE FACTOR DIAGNOSTIC: "
						"depth=%u event=%u materialID=%u materialType=%u glossiness=%.9g name=%s\n",
						pathVertex->depth, (u_int)event,
						pathVertex->bsdf.GetMaterialID(),
						(u_int)pathVertex->bsdf.GetMaterialType(),
						pathVertex->bsdf.GetGlossiness(),
						pathVertex->bsdf.GetMaterialName().c_str());
					fprintf(f,
						"phase15aa flags: REFLECT=%u TRANSMIT=%u DIFFUSE=%u GLOSSY=%u SPECULAR=%u\n",
						(event & REFLECT) ? 1u : 0u,
						(event & TRANSMIT) ? 1u : 0u,
						(event & DIFFUSE) ? 1u : 0u,
						(event & GLOSSY) ? 1u : 0u,
						(event & SPECULAR) ? 1u : 0u);
					fprintf(f,
						"phase15aa classicBSDF=(%.9g, %.9g, %.9g) albedo=(%.9g, %.9g, %.9g)\n",
						bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2],
						albedo.c[0], albedo.c[1], albedo.c[2]);
					fprintf(f,
						"phase15aa bsdf_over_albedo=(%.9g, %.9g, %.9g) ratioSpread=%.9g "
						"bsdfPdfW=%.9g cosSampledDir=%.9g\n",
						ratioR, ratioG, ratioB, ratioMax - ratioMin,
						bsdfPdfW, cosSampledDir);
					fclose(f);
				}
			}
		}

		// ML HERO phase 15ce: Matte spectral Kd convergence diagnostic.
		// Diagnostic only. Every Matte event is reconstructed from the same
		// corrected/non-negative reflectance helper used by the render path and
		// accumulated for a full-render mean comparison against classic RGB.
		if (MLHeroLegacyHeavyDiagnosticsEnabled15cq() && (pathVertex->bsdf.GetMaterialType() == MATTE) && (event & DIFFUSE) &&
				(GetMLHeroWavelengthCount() > 0u) && (pathVertex->mlHeroLaneCount > 0u)) {
			const Spectrum kd15ce = pathVertex->bsdf.Albedo().Clamp(0.f, 1.f);
			const float kdDenom15ce = kd15ce.c[0] * kd15ce.c[0] +
				kd15ce.c[1] * kd15ce.c[1] + kd15ce.c[2] * kd15ce.c[2];
			float scalar15ce = 0.f;
			Spectrum reconClassic15ce;
			Spectrum residual15ce;
			if (kdDenom15ce > 1e-12f) {
				scalar15ce = (bsdfSample.c[0] * kd15ce.c[0] +
					bsdfSample.c[1] * kd15ce.c[1] + bsdfSample.c[2] * kd15ce.c[2]) /
					kdDenom15ce;
				reconClassic15ce = kd15ce * scalar15ce;
				residual15ce = bsdfSample - reconClassic15ce;
			}

			const RGBColor kdRGB15ce(kd15ce.c[0], kd15ce.c[1], kd15ce.c[2]);
			RGBReflSPD kdSPD15ce(kdRGB15ce);
			Spectrum packetRaw15ce;
			Spectrum packetClamped15ce;
			float minSPD15ce = 1e30f;
			float maxSPD15ce = -1e30f;
			for (u_int lane15ce = 0; lane15ce < pathVertex->mlHeroLaneCount; ++lane15ce) {
				const float lambda15ce = pathVertex->mlHeroLaneWaveLength[lane15ce];
				const float sampleWeight15ce = pathVertex->mlHeroLaneSampleWeight[lane15ce];
				const float raw15ce = MLHeroRGBReflSample380_780(kdSPD15ce, lambda15ce);
				const float clamped15ce = engine->mlHeroMatteBasisCompensation ?
					MLHeroMatteBasisCompensatedSample15cj(kd15ce, lambda15ce) :
					MLHeroMatteReflSampleNonNegative380_780(kdSPD15ce, lambda15ce);
				minSPD15ce = Min(minSPD15ce, raw15ce);
				maxSPD15ce = Max(maxSPD15ce, raw15ce);
				const Spectrum cie15ce = MLHeroCIENormalizedReflectanceEstimatorColor(
					lambda15ce, sampleWeight15ce);
				packetRaw15ce += cie15ce * (raw15ce * scalar15ce);
				packetClamped15ce += cie15ce * (clamped15ce * scalar15ce);
			}
			packetRaw15ce /= (float)pathVertex->mlHeroLaneCount;
			packetClamped15ce /= (float)pathVertex->mlHeroLaneCount;
			MLHeroRecordMatteConvergence15ce(kd15ce, bsdfSample, packetClamped15ce);

			// Keep a small number of detailed packets for spot checking while the
			// full-render counters above continue to collect every Matte event.
			static std::atomic<u_int> mlHeroPhase15ceLogged(0u);
			const u_int ticket15ce = mlHeroPhase15ceLogged.fetch_add(1u);
			if (ticket15ce < 24u) {
				const Spectrum errRaw15ce = packetRaw15ce - bsdfSample;
				const Spectrum errClamped15ce = packetClamped15ce - bsdfSample;
				const u_int class15ce = MLHeroClassifyMatte15ce(kd15ce);
				static const char *classNames15ce[MLHERO_MATTE_CLASS_COUNT_15CE] = {
					"NEUTRAL", "RED", "GREEN", "BLUE", "OTHER"
				};
				std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock15ce(mlHeroDebugLogMutex);
				FILE *f15ce = MLHeroOpenLegacyLog15cq("a");
				if (f15ce) {
					fprintf(f15ce,
						"ML HERO phase 15ce MATTE SPECTRAL KD DIAGNOSTIC: ticket=%u depth=%u "
						"fromLight=%u lanes=%u class=%s name=%s\n",
						ticket15ce, pathVertex->depth, pathVertex->bsdf.hitPoint.fromLight ? 1u : 0u,
						pathVertex->mlHeroLaneCount, classNames15ce[class15ce],
						pathVertex->bsdf.GetMaterialName().c_str());
					fprintf(f15ce,
						"phase15ce samplingHelper=MLHeroRGBReflSample380_780 matteNonNegative=1 matteBasisCompensation=%u farRedEndpoint=719.999 fullRenderAggregation=1\n",
						engine->mlHeroMatteBasisCompensation ? 1u : 0u);
					fprintf(f15ce,
						"phase15ce Kd=(%.9g, %.9g, %.9g) classicBSDF=(%.9g, %.9g, %.9g) "
						"packetRaw=(%.9g, %.9g, %.9g) packetApplied=(%.9g, %.9g, %.9g) "
						"rawErr=(%.9g, %.9g, %.9g) appliedErr=(%.9g, %.9g, %.9g) "
						"scalar=%.9g spdRange=[%.9g, %.9g] pdfW=%.9g cos=%.9g\n",
						kd15ce.c[0], kd15ce.c[1], kd15ce.c[2],
						bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2],
						packetRaw15ce.c[0], packetRaw15ce.c[1], packetRaw15ce.c[2],
						packetClamped15ce.c[0], packetClamped15ce.c[1], packetClamped15ce.c[2],
						errRaw15ce.c[0], errRaw15ce.c[1], errRaw15ce.c[2],
						errClamped15ce.c[0], errClamped15ce.c[1], errClamped15ce.c[2],
						scalar15ce, minSPD15ce, maxSPD15ce, bsdfPdfW, cosSampledDir);
					fclose(f15ce);
				}
			}
		}

		// ML HERO phase 15ae:
		// Controlled activation for Glossy2 when the layered decomposition is
		// simple enough to separate safely:
		//   - Kd may be chromatic and is reconstructed as a reflectance SPD
		//   - Ks / Schlick Fresnel / absorption / coating must be achromatic
		//   - the small reconstruction residual must also be achromatic
		//
		// This avoids spectralizing the final mixed RGB BSDF multiplier.
		bool glossy2SpectralActive15ae = false;
		float glossy2BaseFactor15ae = 0.f;
		float glossy2CoatingScalar15ae = 0.f;
		float glossy2ResidualScalar15ae = 0.f;
		Spectrum glossy2Kd15ae;
		std::unique_ptr<RGBReflSPD> glossy2KdSPD15ae;

		// Phase 15af diagnostic state for the first chromatic Glossy2 sample.
		bool glossy2Seen15af = false;
		bool glossy2Ok15af = false;
		bool glossy2ScalarSupport15af = false;
		bool glossy2BaseFit15af = false;
		float glossy2BaseRelError15af = 0.f;
		Spectrum glossy2Ks15af, glossy2S15af, glossy2Abs15af;
		Spectrum glossy2Base15af, glossy2Coating15af, glossy2Residual15af;

		if ((pathVertex->bsdf.GetMaterialType() == GLOSSY2) &&
				(event & REFLECT) && !(event & SPECULAR)) {
			Spectrum kd15ae, ks15ae, s15ae, absorption15ae;
			Spectrum base15ae, coating15ae, combined15ae;
			float wrapper15ae = 1.f;

			const bool ok15ae =
				pathVertex->bsdf.EvaluateMLHeroGlossy2DebugSampleComponents(
					sampledDir, bsdfPdfW,
					&kd15ae, &ks15ae, &s15ae, &absorption15ae,
					&base15ae, &coating15ae, &wrapper15ae, &combined15ae);

			if (ok15ae) {
				const Spectrum residual15ae = bsdfSample - combined15ae;

				const float kdMin15af = Min(kd15ae.c[0], Min(kd15ae.c[1], kd15ae.c[2]));
				const float kdMax15af = Max(kd15ae.c[0], Max(kd15ae.c[1], kd15ae.c[2]));
				const bool chromaticKd15af = ((kdMax15af - kdMin15af) > 1e-4f);
				if (chromaticKd15af) {
					glossy2Seen15af = true;
					glossy2Ok15af = true;
					glossy2Kd15ae = kd15ae;
					glossy2Ks15af = ks15ae;
					glossy2S15af = s15ae;
					glossy2Abs15af = absorption15ae;
					glossy2Base15af = base15ae;
					glossy2Coating15af = coating15ae;
					glossy2Residual15af = residual15ae;
				}

				const float kdDenom15ae =
					kd15ae.c[0] * kd15ae.c[0] +
					kd15ae.c[1] * kd15ae.c[1] +
					kd15ae.c[2] * kd15ae.c[2];

				const float classicMax15ai = Max(1e-9f,
					Max(fabsf(bsdfSample.c[0]),
						Max(fabsf(bsdfSample.c[1]), fabsf(bsdfSample.c[2]))));

				const float residualMean15ai =
					(residual15ae.c[0] + residual15ae.c[1] + residual15ae.c[2]) / 3.f;
				const Spectrum residualChroma15ai =
					residual15ae - Spectrum(residualMean15ai);
				const float residualChromaMax15ai = Max(fabsf(residualChroma15ai.c[0]),
					Max(fabsf(residualChroma15ai.c[1]), fabsf(residualChroma15ai.c[2])));
				const float residualChromaRel15ai =
					residualChromaMax15ai / classicMax15ai;

				const bool residualScalar15ai = MLHeroIsScalarSpectrum(residual15ae);
				const bool residualChromaTiny15ai = (residualChromaRel15ai <= 1e-4f);

				const bool scalarSupport15ae =
					MLHeroIsNearScalarGlossy2Support(ks15ae) &&
					MLHeroIsNearScalarGlossy2Support(s15ae) &&
					MLHeroIsNearScalarGlossy2Support(absorption15ae) &&
					MLHeroIsNearScalarGlossy2Support(coating15ae) &&
					(residualScalar15ai || residualChromaTiny15ai);

				if (glossy2Seen15af)
					glossy2ScalarSupport15af = scalarSupport15ae;

				if (glossy2Seen15af) {
					static std::atomic<bool> mlHeroPhase15alSupportLogged(false);
					bool expected15al = false;
					if (mlHeroPhase15alSupportLogged.compare_exchange_strong(
							expected15al, true)) {
						const float ksMax15al = Max(fabsf(ks15ae.c[0]),
							Max(fabsf(ks15ae.c[1]), fabsf(ks15ae.c[2])));
						const float ksSpread15al = Max(fabsf(ks15ae.c[0] - ks15ae.c[1]),
							Max(fabsf(ks15ae.c[0] - ks15ae.c[2]),
								fabsf(ks15ae.c[1] - ks15ae.c[2])));
						const float coatMax15al = Max(fabsf(coating15ae.c[0]),
							Max(fabsf(coating15ae.c[1]), fabsf(coating15ae.c[2])));
						const float coatSpread15al = Max(
							fabsf(coating15ae.c[0] - coating15ae.c[1]),
							Max(fabsf(coating15ae.c[0] - coating15ae.c[2]),
								fabsf(coating15ae.c[1] - coating15ae.c[2])));

						std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock15al(
							mlHeroDebugLogMutex);
						FILE *f15al = MLHeroOpenLegacyLog15cq("a");
						if (f15al) {
							fprintf(f15al,
								"ML HERO phase 15al GLOSSY2 NEAR-SCALAR SUPPORT: "
								"depth=%u event=%u name=%s activeSupport=%u "
								"ksRelSpread=%.9g coatingRelSpread=%.9g\n",
								pathVertex->depth, (u_int)event,
								pathVertex->bsdf.GetMaterialName().c_str(),
								scalarSupport15ae ? 1u : 0u,
								ksSpread15al / Max(1e-8f, ksMax15al),
								coatSpread15al / Max(1e-8f, coatMax15al));
							fclose(f15al);
						}
					}
				}

				if (scalarSupport15ae && (kdDenom15ae > 1e-12f)) {
					glossy2BaseFactor15ae =
						(base15ae.c[0] * kd15ae.c[0] +
						 base15ae.c[1] * kd15ae.c[1] +
						 base15ae.c[2] * kd15ae.c[2]) / kdDenom15ae;

					const Spectrum baseRecon15ae = kd15ae * glossy2BaseFactor15ae;
					const Spectrum baseError15ae = base15ae - baseRecon15ae;

					const float baseErrMax15ae = Max(fabsf(baseError15ae.c[0]),
						Max(fabsf(baseError15ae.c[1]), fabsf(baseError15ae.c[2])));
					const float baseMax15ae = Max(1e-9f,
						Max(fabsf(base15ae.c[0]),
							Max(fabsf(base15ae.c[1]), fabsf(base15ae.c[2]))));
					const float baseRelError15ae = baseErrMax15ae / baseMax15ae;

					if (glossy2Seen15af) {
						glossy2BaseRelError15af = baseRelError15ae;
						glossy2BaseFit15af = (baseRelError15ae <= 1e-4f);
					}

					// Require the base contribution itself to be Kd times one
					// scalar transport factor before activating.
					if ((baseErrMax15ae / baseMax15ae) <= 1e-4f) {
						glossy2Kd15ae = kd15ae;
						glossy2CoatingScalar15ae =
							(coating15ae.c[0] + coating15ae.c[1] + coating15ae.c[2]) / 3.f;
						// Preserve the scalar mean of the residual in all accepted cases.
						// Only the tiny chromatic deviation is discarded.
						glossy2ResidualScalar15ae = residualMean15ai;

						const RGBColor kdRGB15ae(
							kd15ae.c[0], kd15ae.c[1], kd15ae.c[2]);
						glossy2KdSPD15ae.reset(new RGBReflSPD(kdRGB15ae));
						glossy2SpectralActive15ae = true;
					}
				}
			}

			// ML HERO phase 15ah: Glossy2 back-face path.
			// Glossy2 intentionally skips its coating on the back side and returns
			// a diffuse Kd-only sample (event REFLECT|DIFFUSE). The front-side
			// decomposition helper returns false there, so handle this simple case
			// directly as Kd * scalar transport.
			if (!ok15ae && (event & DIFFUSE)) {
				const Spectrum kdBack15ah = pathVertex->bsdf.Albedo();
				const float kdDenomBack15ah =
					kdBack15ah.c[0] * kdBack15ah.c[0] +
					kdBack15ah.c[1] * kdBack15ah.c[1] +
					kdBack15ah.c[2] * kdBack15ah.c[2];

				if (kdDenomBack15ah > 1e-12f) {
					const float factorBack15ah =
						(bsdfSample.c[0] * kdBack15ah.c[0] +
						 bsdfSample.c[1] * kdBack15ah.c[1] +
						 bsdfSample.c[2] * kdBack15ah.c[2]) / kdDenomBack15ah;
					const Spectrum reconstructedBack15ah = kdBack15ah * factorBack15ah;
					const Spectrum errorBack15ah = bsdfSample - reconstructedBack15ah;
					const float errorMaxBack15ah = Max(fabsf(errorBack15ah.c[0]),
						Max(fabsf(errorBack15ah.c[1]), fabsf(errorBack15ah.c[2])));
					const float bsdfMaxBack15ah = Max(1e-9f,
						Max(fabsf(bsdfSample.c[0]),
							Max(fabsf(bsdfSample.c[1]), fabsf(bsdfSample.c[2]))));

					if ((errorMaxBack15ah / bsdfMaxBack15ah) <= 1e-5f) {
						glossy2Kd15ae = kdBack15ah;
						glossy2BaseFactor15ae = factorBack15ah;
						glossy2CoatingScalar15ae = 0.f;
						glossy2ResidualScalar15ae = 0.f;

						const RGBColor kdRGBBack15ah(
							kdBack15ah.c[0], kdBack15ah.c[1], kdBack15ah.c[2]);
						glossy2KdSPD15ae.reset(new RGBReflSPD(kdRGBBack15ah));
						glossy2SpectralActive15ae = true;
					}
				}
			}
		}

		// Phase 15af: log the first chromatic Glossy2 sample and tell us
		// exactly why 15ae activated or rejected it.
		if (glossy2Seen15af) {
			static std::atomic<bool> mlHeroPhase15afLogged(false);
			bool expected15af = false;
			if (mlHeroPhase15afLogged.compare_exchange_strong(expected15af, true)) {
				std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock15af(
					mlHeroDebugLogMutex);
				FILE *f15af = MLHeroOpenLegacyLog15cq("a");
				if (f15af) {
					fprintf(f15af,
						"ML HERO phase 15af CHROMATIC GLOSSY2 ACTIVATION CHECK: "
						"depth=%u event=%u name=%s active=%u ok=%u "
						"scalarSupport=%u baseFit=%u baseRelError=%.9g\n",
						pathVertex->depth, (u_int)event,
						pathVertex->bsdf.GetMaterialName().c_str(),
						glossy2SpectralActive15ae ? 1u : 0u,
						glossy2Ok15af ? 1u : 0u,
						glossy2ScalarSupport15af ? 1u : 0u,
						glossy2BaseFit15af ? 1u : 0u,
						glossy2BaseRelError15af);
					fprintf(f15af,
						"phase15af kd=(%.9g, %.9g, %.9g) "
						"ks=(%.9g, %.9g, %.9g)\n",
						glossy2Kd15ae.c[0], glossy2Kd15ae.c[1], glossy2Kd15ae.c[2],
						glossy2Ks15af.c[0], glossy2Ks15af.c[1], glossy2Ks15af.c[2]);
					fprintf(f15af,
						"phase15af SchlickS=(%.9g, %.9g, %.9g) "
						"absorption=(%.9g, %.9g, %.9g)\n",
						glossy2S15af.c[0], glossy2S15af.c[1], glossy2S15af.c[2],
						glossy2Abs15af.c[0], glossy2Abs15af.c[1], glossy2Abs15af.c[2]);
					fprintf(f15af,
						"phase15af base=(%.9g, %.9g, %.9g) "
						"coating=(%.9g, %.9g, %.9g) "
						"residual=(%.9g, %.9g, %.9g)\n",
						glossy2Base15af.c[0], glossy2Base15af.c[1], glossy2Base15af.c[2],
						glossy2Coating15af.c[0], glossy2Coating15af.c[1], glossy2Coating15af.c[2],
						glossy2Residual15af.c[0], glossy2Residual15af.c[1], glossy2Residual15af.c[2]);
					fclose(f15af);
				}
			}
		}

		const u_int materialType15cl = (u_int)pathVertex->bsdf.GetMaterialType();
		if (MLHeroCurrentDiagnosticsEnabled15cq() && genericReflectanceCandidate && (materialType15cl < MLHERO_MATERIAL_TYPE_COUNT_15CL))
			mlHero15clReflectCandidate[materialType15cl].fetch_add(1ull, std::memory_order_relaxed);
		if (MLHeroCurrentDiagnosticsEnabled15cq() && glossy2SpectralActive15ae && (materialType15cl < MLHERO_MATERIAL_TYPE_COUNT_15CL))
			mlHero15clGlossy2Split[materialType15cl].fetch_add(1ull, std::memory_order_relaxed);

		std::unique_ptr<RGBReflSPD> genericReflectanceSPD;

		// Phase 15ag: catch the first Glossy2 sample that STILL falls back to
		// the old whole-BSDF RGBReflSPD path. 15af proved that the green sphere
		// can activate the new split on at least one bounce, so this tells us
		// why a later bounce may still miss it.
		if (genericReflectanceCandidate && !glossy2SpectralActive15ae &&
				(pathVertex->bsdf.GetMaterialType() == GLOSSY2)) {
			static std::atomic<bool> mlHeroPhase15agLogged(false);
			bool expected15ag = false;
			if (mlHeroPhase15agLogged.compare_exchange_strong(expected15ag, true)) {
				Spectrum kd15ag, ks15ag, s15ag, absorption15ag;
				Spectrum base15ag, coating15ag, combined15ag;
				float wrapper15ag = 1.f;

				const bool ok15ag =
					pathVertex->bsdf.EvaluateMLHeroGlossy2DebugSampleComponents(
						sampledDir, bsdfPdfW,
						&kd15ag, &ks15ag, &s15ag, &absorption15ag,
						&base15ag, &coating15ag, &wrapper15ag, &combined15ag);

				const Spectrum residual15ag = bsdfSample - combined15ag;

				const bool ksScalar15ag = ok15ag && MLHeroIsScalarSpectrum(ks15ag);
				const bool sScalar15ag = ok15ag && MLHeroIsScalarSpectrum(s15ag);
				const bool absScalar15ag = ok15ag && MLHeroIsScalarSpectrum(absorption15ag);
				const bool coatingScalar15ag = ok15ag && MLHeroIsScalarSpectrum(coating15ag);
				const bool residualScalar15ag = ok15ag && MLHeroIsScalarSpectrum(residual15ag);

				float baseRelError15ag = -1.f;
				bool baseFit15ag = false;
				if (ok15ag) {
					const float kdDenom15ag =
						kd15ag.c[0] * kd15ag.c[0] +
						kd15ag.c[1] * kd15ag.c[1] +
						kd15ag.c[2] * kd15ag.c[2];

					if (kdDenom15ag > 1e-12f) {
						const float baseFactor15ag =
							(base15ag.c[0] * kd15ag.c[0] +
							 base15ag.c[1] * kd15ag.c[1] +
							 base15ag.c[2] * kd15ag.c[2]) / kdDenom15ag;
						const Spectrum baseRecon15ag = kd15ag * baseFactor15ag;
						const Spectrum baseError15ag = base15ag - baseRecon15ag;

						const float baseErrMax15ag = Max(fabsf(baseError15ag.c[0]),
							Max(fabsf(baseError15ag.c[1]), fabsf(baseError15ag.c[2])));
						const float baseMax15ag = Max(1e-9f,
							Max(fabsf(base15ag.c[0]),
								Max(fabsf(base15ag.c[1]), fabsf(base15ag.c[2]))));

						baseRelError15ag = baseErrMax15ag / baseMax15ag;
						baseFit15ag = (baseRelError15ag <= 1e-4f);
					}
				}

				std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock15ag(
					mlHeroDebugLogMutex);
				FILE *f15ag = MLHeroOpenLegacyLog15cq("a");
				if (f15ag) {
					fprintf(f15ag,
						"ML HERO phase 15ag GLOSSY2 GENERIC FALLBACK: "
						"depth=%u event=%u name=%s ok=%u "
						"ksScalar=%u sScalar=%u absScalar=%u "
						"coatingScalar=%u residualScalar=%u "
						"baseFit=%u baseRelError=%.9g\n",
						pathVertex->depth, (u_int)event,
						pathVertex->bsdf.GetMaterialName().c_str(),
						ok15ag ? 1u : 0u,
						ksScalar15ag ? 1u : 0u,
						sScalar15ag ? 1u : 0u,
						absScalar15ag ? 1u : 0u,
						coatingScalar15ag ? 1u : 0u,
						residualScalar15ag ? 1u : 0u,
						baseFit15ag ? 1u : 0u,
						baseRelError15ag);

					fprintf(f15ag,
						"phase15ag classicBSDF=(%.9g, %.9g, %.9g) "
						"kd=(%.9g, %.9g, %.9g) ks=(%.9g, %.9g, %.9g)\n",
						bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2],
						kd15ag.c[0], kd15ag.c[1], kd15ag.c[2],
						ks15ag.c[0], ks15ag.c[1], ks15ag.c[2]);

					fprintf(f15ag,
						"phase15ag SchlickS=(%.9g, %.9g, %.9g) "
						"absorption=(%.9g, %.9g, %.9g)\n",
						s15ag.c[0], s15ag.c[1], s15ag.c[2],
						absorption15ag.c[0], absorption15ag.c[1], absorption15ag.c[2]);

					fprintf(f15ag,
						"phase15ag base=(%.9g, %.9g, %.9g) "
						"coating=(%.9g, %.9g, %.9g) "
						"residual=(%.9g, %.9g, %.9g)\n",
						base15ag.c[0], base15ag.c[1], base15ag.c[2],
						coating15ag.c[0], coating15ag.c[1], coating15ag.c[2],
						residual15ag.c[0], residual15ag.c[1], residual15ag.c[2]);

					fclose(f15ag);
				}
			}
		}

		if (genericReflectanceCandidate && !glossy2SpectralActive15ae) {
			const RGBColor bsdfRGB(bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2]);
			genericReflectanceSPD.reset(new RGBReflSPD(bsdfRGB));
			if (MLHeroCurrentDiagnosticsEnabled15cq() && (materialType15cl < MLHERO_MATERIAL_TYPE_COUNT_15CL)) {
				mlHero15clGenericRGBReflSPD[materialType15cl].fetch_add(1ull, std::memory_order_relaxed);

				// ML HERO phase 15co: deterministic bounded A/B sampling per generic material.
				// MATTE remains excluded because its independent 15cj experiment owns it.
				if (MLHeroCurrentHeavyDiagnosticsEnabled15cq() && (materialType15cl != MATTE)) {
					const u_int sampleIndex15co = mlHero15coSampleAttempts[materialType15cl].fetch_add(1u, std::memory_order_relaxed);
					if (sampleIndex15co < MLHERO_15CO_SAMPLE_BUDGET) {
						const Spectrum baseline15co = MLHeroIntegrateMatteReflectance15cf(*genericReflectanceSPD, false);
						const Spectrum candidate15co = MLHeroIntegrateGenericCompensated15cn(bsdfSample);
						const Spectrum errBase15co = baseline15co - bsdfSample;
						const Spectrum errCand15co = candidate15co - bsdfSample;
						const double maxBase15co = Max(fabsf(errBase15co.c[0]), Max(fabsf(errBase15co.c[1]), fabsf(errBase15co.c[2])));
						const double maxCand15co = Max(fabsf(errCand15co.c[0]), Max(fabsf(errCand15co.c[1]), fabsf(errCand15co.c[2])));
						mlHero15coSampleCount[materialType15cl].fetch_add(1ull, std::memory_order_relaxed);
						mlHero15coErrBeforeSum[materialType15cl].fetch_add(maxBase15co, std::memory_order_relaxed);
						mlHero15coErrAfterSum[materialType15cl].fetch_add(maxCand15co, std::memory_order_relaxed);
						MLHeroAtomicMax15co(mlHero15coErrBeforeMax[materialType15cl], maxBase15co);
						MLHeroAtomicMax15co(mlHero15coErrAfterMax[materialType15cl], maxCand15co);

						// Keep a single human-readable representative line per material.
						if (sampleIndex15co == 0u) {
							std::lock_guard<std::recursive_mutex> lock15co(mlHeroDebugLogMutex);
							FILE *f15co = MLHeroOpenCurrentLog15cq("a");
							if (f15co) {
								fprintf(f15co, "ML HERO phase 15co GENERIC REFLECTANCE BASIS A/B: material=%s id=%u switch=%u sourceRGB=(%.9g, %.9g, %.9g) baselineRecon=(%.9g, %.9g, %.9g) candidateRecon=(%.9g, %.9g, %.9g) maxErrBefore=%.9g maxErrAfter=%.9g\n",
									MLHeroMaterialTypeName15cl(materialType15cl), materialType15cl, engine->mlHeroGenericReflectanceCompensation ? 1u : 0u,
									bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2],
									baseline15co.c[0], baseline15co.c[1], baseline15co.c[2],
									candidate15co.c[0], candidate15co.c[1], candidate15co.c[2], maxBase15co, maxCand15co);
								fclose(f15co);
							}
						}
					}
				}
			}
		}

		// ML HERO phase 15ay: optionally replace only the Glass packet weights.
		// The classic path throughput, sampled event, PDF and direction stay exactly
		// as LuxCore sampled them. Only the parallel HERO lane throughput receives
		// the wavelength-specific Glass multiplier.
		const bool glassSpecularCandidate15bx =
			(pathVertex->bsdf.GetMaterialType() == GLASS) && (event & SPECULAR);
		const bool glassPerLaneWeightActive15ay =
			engine->mlHeroGlassPerLaneWeight && glassSpecularCandidate15bx;
		// ML HERO phase 15ca: Mode 2 lane-PDF correction is an independent mode,
		// not a side effect of the per-lane Glass-weight switch.  When PLW is OFF,
		// Mode 2 still evaluates the wavelength-specific event PDF but starts from
		// the shared HERO Glass multiplier.
		const bool mode2LanePdfActive15ca =
			(engine->mlHeroGlassMode == 2) && glassSpecularCandidate15bx;
		const bool glassLaneEvalActive15ca =
			glassPerLaneWeightActive15ay || mode2LanePdfActive15ca;
		// HERO_64 / 15bx: Mode 1 must still detect a dispersive Glass transmit
		// when per-lane weighting is disabled.  This detection is deliberately
		// separate from the throughput-weighting switch.
		const bool mode1NeedsIndependentDispersionDetection15bx =
			(engine->mlHeroGlassMode == 1) && glassSpecularCandidate15bx &&
			(event & TRANSMIT) && !glassPerLaneWeightActive15ay;
		float glassAdjointCorrection15ay = 1.f;
		if (glassPerLaneWeightActive15ay && pathVertex->bsdf.hitPoint.fromLight) {
			const float absDotFixedNG15ay = AbsDot(pathVertex->bsdf.hitPoint.fixedDir,
				pathVertex->bsdf.hitPoint.geometryN);
			const float absDotSampledNG15ay = AbsDot(sampledDir,
				pathVertex->bsdf.hitPoint.geometryN);
			glassAdjointCorrection15ay = (absDotFixedNG15ay > 1e-20f) ?
				(absDotSampledNG15ay / absDotFixedNG15ay) : 0.f;
		}

		// ML HERO phase 15az: collect full-render correction statistics while
		// leaving the 15ay weighting behavior unchanged.
		float minCorrection15az = 1e30f;
		float maxCorrection15az = -1e30f;
		float maxAbsDelta15az = 0.f;
		float extremeCorrection15az = 1.f;
		float extremeLambda15az = 0.f;
		u_int evaluatedLanes15az = 0u;
		u_int lanesOver5Pct15az = 0u;
		u_int lanesOver10Pct15az = 0u;
		u_int lanesOver25Pct15az = 0u;
		u_int tirLanes15ba = 0u;
		u_int nearTir95Lanes15ba = 0u;
		u_int nearTir99Lanes15ba = 0u;
		u_int pdfMismatch20PctLanes15ba = 0u;
		u_int baseHeroTirMismatchLanes15bb = 0u;
		u_int heroLaneTirMismatchLanes15bb = 0u;
		u_int baseLaneTirMismatchLanes15bb = 0u;
		u_int allThreeTirAgreeLanes15bb = 0u;
		u_int nearHero99TransmitLanes15bc = 0u;
		u_int rawOver25NearHero99_15bc = 0u;
		u_int lanePdfAdjustedOver25NearHero99_15bc = 0u;
		u_int spectralMixAdjustedOver25NearHero99_15bc = 0u;
		float maxRawCorrectionNearHero99_15bc = 1.f;
		float maxLanePdfAdjustedCorrectionNearHero99_15bc = 1.f;
		float maxSpectralMixAdjustedCorrectionNearHero99_15bc = 1.f;
		u_int eyeTransmitNearHero99Lanes15bd = 0u;
		u_int eyeTransmitNearHero99SupportMatch15bd = 0u;
		u_int eyeTransmitNearHero99SupportMismatch15bd = 0u;
		u_int eyeTransmitNearHero99LanePdfOver25Supported15bd = 0u;
		u_int eyeTransmitSupportMismatchTotal15bd = 0u;
		u_int lightTransmitSupportMismatchTotal15bd = 0u;
		float maxEyeTransmitNearHero99LanePdfSupported15bd = 1.f;
		MLHeroGlassOutlier15ba localOutlier15ba;
		const float heroLambda15ba = (pathVertex->mlHeroLaneCount > 0u) ?
			pathVertex->mlHeroLaneWaveLength[0] : 0.f;

		float heroNc15bb = 0.f, heroNtBase15bb = 0.f, heroNtLambda15bb = 0.f;
		float heroFresnel15bb = 0.f, heroEta15bb = 0.f, heroEta2_15bb = 0.f;
		float heroTransport15bb = 0.f, heroDirDelta15bb = 0.f;
		float heroCosFixed15bb = 0.f, heroSinI2_15bb = 0.f, heroSinT2_15bb = 0.f;
		float heroEventPdfW15bb = 0.f;
		bool heroTir15bb = false;
		Spectrum heroMultiplier15bb;
		bool heroGlassEvalOk15bx = false;
		if ((glassLaneEvalActive15ca || mode1NeedsIndependentDispersionDetection15bx) &&
				(heroLambda15ba > 0.f))
			heroGlassEvalOk15bx = pathVertex->bsdf.EvaluateMLHeroGlassDebugAtWaveLength(
				sampledDir, event, bsdfPdfW, heroLambda15ba,
				&heroNc15bb, &heroNtBase15bb, &heroNtLambda15bb,
				&heroFresnel15bb, &heroEta15bb, &heroEta2_15bb, &heroTransport15bb,
				&heroDirDelta15bb, &heroMultiplier15bb,
				&heroCosFixed15bb, &heroSinI2_15bb, &heroSinT2_15bb, &heroTir15bb,
				&heroEventPdfW15bb);

		const Vector localFixed15bb = pathVertex->bsdf.GetFrame().ToLocal(pathVertex->bsdf.hitPoint.fixedDir);
		const Vector localSampled15bb = pathVertex->bsdf.GetFrame().ToLocal(sampledDir);

		// ML HERO phase 15bc: counterfactual equal-weight spectral mixture density.
		// This is diagnostic only. The renderer still samples the event with the HERO
		// wavelength PDF. A future unbiased implementation would have to sample from
		// the same mixture (or use the corresponding MIS construction).
		float spectralMixEventPdfW15bc = 0.f;
		if (glassPerLaneWeightActive15ay && (pathVertex->mlHeroLaneCount > 0u)) {
			for (u_int mixLane15bc = 0u; mixLane15bc < pathVertex->mlHeroLaneCount; ++mixLane15bc) {
				float mixNc15bc = 0.f, mixNtBase15bc = 0.f, mixNtLambda15bc = 0.f;
				float mixFresnel15bc = 0.f, mixEta15bc = 0.f, mixEta2_15bc = 0.f;
				float mixTransport15bc = 0.f, mixDirDelta15bc = 0.f;
				float mixCosFixed15bc = 0.f, mixSinI2_15bc = 0.f, mixSinT2_15bc = 0.f;
				float mixEventPdf15bc = 0.f;
				bool mixTir15bc = false;
				Spectrum mixMultiplier15bc;
				pathVertex->bsdf.EvaluateMLHeroGlassDebugAtWaveLength(
					sampledDir, event, bsdfPdfW, pathVertex->mlHeroLaneWaveLength[mixLane15bc],
					&mixNc15bc, &mixNtBase15bc, &mixNtLambda15bc,
					&mixFresnel15bc, &mixEta15bc, &mixEta2_15bc, &mixTransport15bc,
					&mixDirDelta15bc, &mixMultiplier15bc,
					&mixCosFixed15bc, &mixSinI2_15bc, &mixSinT2_15bc, &mixTir15bc,
					&mixEventPdf15bc);
				spectralMixEventPdfW15bc += mixEventPdf15bc;
			}
			spectralMixEventPdfW15bc /= (float)pathVertex->mlHeroLaneCount;
		}

		bool dispersiveSpecularTransmit15be = false;
		// HERO_64 / 15bx: when per-lane Glass weighting is OFF, do a lightweight
		// wavelength scan only to decide whether Mode 1 must terminate the packet.
		// The scan does not alter throughput, PDF, sampled direction or event.
		if (mode1NeedsIndependentDispersionDetection15bx && heroGlassEvalOk15bx &&
				(event & TRANSMIT) && (pathVertex->mlHeroLaneCount > 1u)) {
			for (u_int lane15bx = 1u; lane15bx < pathVertex->mlHeroLaneCount; ++lane15bx) {
				float nc15bx = 0.f, ntBase15bx = 0.f, ntLambda15bx = 0.f;
				float fresnel15bx = 0.f, eta15bx = 0.f, eta2_15bx = 0.f;
				float transport15bx = 0.f, dirDelta15bx = 0.f;
				float cosFixed15bx = 0.f, sinI2_15bx = 0.f, sinT2_15bx = 0.f;
				float laneEventPdfW15bx = 0.f;
				bool tir15bx = false;
				Spectrum multiplier15bx;
				const bool ok15bx = pathVertex->bsdf.EvaluateMLHeroGlassDebugAtWaveLength(
					sampledDir, event, bsdfPdfW, pathVertex->mlHeroLaneWaveLength[lane15bx],
					&nc15bx, &ntBase15bx, &ntLambda15bx,
					&fresnel15bx, &eta15bx, &eta2_15bx, &transport15bx,
					&dirDelta15bx, &multiplier15bx,
					&cosFixed15bx, &sinI2_15bx, &sinT2_15bx, &tir15bx,
					&laneEventPdfW15bx);
				if (ok15bx && (fabsf(ntLambda15bx - heroNtLambda15bb) > 1e-5f)) {
					dispersiveSpecularTransmit15be = true;
					break;
				}
			}
		}

		// Phase 15br keeps the pre-Glass lane throughput and the exact multiplier
		// applied to each lane.  This is diagnostic-only and lets us reconstruct a
		// shared-weight counterfactual at the same bounce without a second render.
		float lanePreGlassScalar15br[8] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
		float laneAppliedGlassMultiplier15br[8] = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
		const float sharedGlassMultiplier15br =
			(bsdfSample.c[0] + bsdfSample.c[1] + bsdfSample.c[2]) / 3.f;
		for (u_int lane = 0; lane < pathVertex->mlHeroLaneCount; ++lane) {
			lanePreGlassScalar15br[lane] = pathVertex->mlHeroLaneThroughput[lane].c[0];
			Spectrum laneSample;
			if (glassLaneEvalActive15ca) {
				float nc15ay = 0.f, ntBase15ay = 0.f, ntLambda15ay = 0.f;
				float fresnel15ay = 0.f, eta15ay = 0.f, eta2_15ay = 0.f;
				float transport15ay = 0.f, dirDelta15ay = 0.f;
				float cosFixed15ba = 0.f, sinI2_15ba = 0.f, sinT2_15ba = 0.f;
				float laneEventPdfW15ba = 0.f;
				bool tir15ba = false;
				Spectrum materialIdealMultiplier15ay;
				const float laneLambda15ba = pathVertex->mlHeroLaneWaveLength[lane];
				const bool ok15ay = pathVertex->bsdf.EvaluateMLHeroGlassDebugAtWaveLength(
					sampledDir, event, bsdfPdfW, laneLambda15ba,
					&nc15ay, &ntBase15ay, &ntLambda15ay,
					&fresnel15ay, &eta15ay, &eta2_15ay, &transport15ay,
					&dirDelta15ay, &materialIdealMultiplier15ay,
					&cosFixed15ba, &sinI2_15ba, &sinT2_15ba, &tir15ba,
					&laneEventPdfW15ba);

				if (ok15ay && (event & TRANSMIT) &&
						(fabsf(ntLambda15ay - heroNtLambda15bb) > 1e-5f))
					dispersiveSpecularTransmit15be = true;

				// Phase 15ba diagnostics are evaluated even if the lane itself
				// would be TIR and therefore falls back to shared-HERO weighting.
				if (tir15ba) ++tirLanes15ba;
				if ((sinT2_15ba >= .95f) && (sinT2_15ba < 1.f)) ++nearTir95Lanes15ba;
				if ((sinT2_15ba >= .99f) && (sinT2_15ba < 1.f)) ++nearTir99Lanes15ba;
				if ((bsdfPdfW > 1e-20f) && (laneEventPdfW15ba > 1e-20f) &&
						(fabsf(laneEventPdfW15ba / bsdfPdfW - 1.f) > .20f))
					++pdfMismatch20PctLanes15ba;

				// 15bb: compare path support at base, HERO and lane IOR.
				const bool entering15bb = (cosFixed15ba > 0.f);
				const float baseEta15bb = (nc15ay > 1e-20f && ntBase15ay > 1e-20f) ?
					(entering15bb ? (nc15ay / ntBase15ay) : (ntBase15ay / nc15ay)) : 0.f;
				const float baseSinT2_15bb = baseEta15bb * baseEta15bb * sinI2_15ba;
				const bool baseTir15bb = (baseSinT2_15bb >= 1.f);

				if (baseTir15bb != heroTir15bb) ++baseHeroTirMismatchLanes15bb;
				if (heroTir15bb != tir15ba) ++heroLaneTirMismatchLanes15bb;
				if (baseTir15bb != tir15ba) ++baseLaneTirMismatchLanes15bb;
				if ((baseTir15bb == heroTir15bb) && (heroTir15bb == tir15ba))
					++allThreeTirAgreeLanes15bb;

				float baseDirDelta15bb = 0.f, baseDirAngle15bb = 0.f;
				float heroDirAngle15bb = 0.f, laneDirAngle15bb = 0.f;
				if (event & TRANSMIT) {
					if (!baseTir15bb) {
						const float baseCost15bb = sqrtf(Max(0.f, 1.f - baseSinT2_15bb)) * (entering15bb ? -1.f : 1.f);
						const Vector baseExpected15bb(-baseEta15bb * localFixed15bb.x, -baseEta15bb * localFixed15bb.y, baseCost15bb);
						baseDirDelta15bb = (baseExpected15bb - localSampled15bb).Length();
						baseDirAngle15bb = acosf(Clamp(Dot(baseExpected15bb, localSampled15bb), -1.f, 1.f));
					}
					if (!heroTir15bb) {
						const float heroCost15bb = sqrtf(Max(0.f, 1.f - heroSinT2_15bb)) * (heroCosFixed15bb > 0.f ? -1.f : 1.f);
						const Vector heroExpected15bb(-heroEta15bb * localFixed15bb.x, -heroEta15bb * localFixed15bb.y, heroCost15bb);
						heroDirAngle15bb = acosf(Clamp(Dot(heroExpected15bb, localSampled15bb), -1.f, 1.f));
					}
					if (!tir15ba) {
						const float laneCost15bb = sqrtf(Max(0.f, 1.f - sinT2_15ba)) * (cosFixed15ba > 0.f ? -1.f : 1.f);
						const Vector laneExpected15bb(-eta15ay * localFixed15bb.x, -eta15ay * localFixed15bb.y, laneCost15bb);
						laneDirAngle15bb = acosf(Clamp(Dot(laneExpected15bb, localSampled15bb), -1.f, 1.f));
					}
				}

				// Phase 15bn: accumulate the wavelength-dependent Glass event probability
				// along this path.  For a perfect specular transmission, also track whether
				// this exact sampled HERO direction has Dirac support at each lane.  This is
				// the distinction made by the spectral MIS derivation: event probability
				// alone is not the probability of generating the same perfect-specular path.
				if (ok15ay) {
					const double laneEventP15bn = (laneEventPdfW15ba > 0.f) ? (double)laneEventPdfW15ba : 0.0;
					pathVertex->mlHeroLaneGlassEventPdfProduct[lane] *= laneEventP15bn;
					bool diracSupport15bn = (laneEventP15bn > 0.0);
					if (event & TRANSMIT)
						diracSupport15bn = diracSupport15bn && !tir15ba && (dirDelta15ay <= 1e-5f);
					if (diracSupport15bn)
						pathVertex->mlHeroLaneDiracSupportPdfProduct[lane] *= laneEventP15bn;
					else
						pathVertex->mlHeroLaneDiracSupportPdfProduct[lane] = 0.0;
				}

				if (ok15ay) {
					const Spectrum wrappedIdealMultiplier15ay =
						materialIdealMultiplier15ay * glassAdjointCorrection15ay;
					const float laneGlassMultiplier15ay =
						(wrappedIdealMultiplier15ay.c[0] + wrappedIdealMultiplier15ay.c[1] + wrappedIdealMultiplier15ay.c[2]) / 3.f;

					const float sharedGlassMultiplier15az =
						(bsdfSample.c[0] + bsdfSample.c[1] + bsdfSample.c[2]) / 3.f;
					if (fabsf(sharedGlassMultiplier15az) > 1e-20f) {
						const float correction15az = laneGlassMultiplier15ay / sharedGlassMultiplier15az;
						const float lanePdfAdjustedCorrection15bc =
							(laneEventPdfW15ba > 1e-20f) ?
							(correction15az * bsdfPdfW / laneEventPdfW15ba) : 0.f;
						const float spectralMixAdjustedCorrection15bc =
							(spectralMixEventPdfW15bc > 1e-20f) ?
							(correction15az * bsdfPdfW / spectralMixEventPdfW15bc) : 0.f;

						// ML HERO phase 15bd: a transmission lane only shares the HERO event support
						// when both HERO and lane agree about TIR/non-TIR.  We do not apply any
						// correction here; this merely classifies lanes before evaluating the
						// already promising lane-PDF candidate.  A support mismatch would require
						// a different sampling/MIS construction, not a throughput clamp.
						const bool supportMatch15bd = !(event & TRANSMIT) ||
							(heroTir15bb == tir15ba);
						const float supportAwareCorrection15bd = supportMatch15bd ?
							lanePdfAdjustedCorrection15bc : 0.f;
						const bool eyeTransmit15bd = (event & TRANSMIT) &&
							!pathVertex->bsdf.hitPoint.fromLight;
						if (eyeTransmit15bd && !supportMatch15bd)
							++eyeTransmitSupportMismatchTotal15bd;
						else if ((event & TRANSMIT) && pathVertex->bsdf.hitPoint.fromLight && !supportMatch15bd)
							++lightTransmitSupportMismatchTotal15bd;

						if (eyeTransmit15bd && !heroTir15bb &&
							(heroSinT2_15bb >= .99f) && (heroSinT2_15bb < 1.f)) {
							++eyeTransmitNearHero99Lanes15bd;
							if (supportMatch15bd) {
								++eyeTransmitNearHero99SupportMatch15bd;
								if (fabsf(lanePdfAdjustedCorrection15bc - 1.f) > .25f)
									++eyeTransmitNearHero99LanePdfOver25Supported15bd;
								maxEyeTransmitNearHero99LanePdfSupported15bd = Max(
									maxEyeTransmitNearHero99LanePdfSupported15bd, lanePdfAdjustedCorrection15bc);
							} else {
								++eyeTransmitNearHero99SupportMismatch15bd;
							}
						}
						const float absDelta15az = fabsf(correction15az - 1.f);

						if ((event & TRANSMIT) && !heroTir15bb &&
								(heroSinT2_15bb >= .99f) && (heroSinT2_15bb < 1.f)) {
							++nearHero99TransmitLanes15bc;
							if (fabsf(correction15az - 1.f) > .25f) ++rawOver25NearHero99_15bc;
							if (fabsf(lanePdfAdjustedCorrection15bc - 1.f) > .25f) ++lanePdfAdjustedOver25NearHero99_15bc;
							if (fabsf(spectralMixAdjustedCorrection15bc - 1.f) > .25f) ++spectralMixAdjustedOver25NearHero99_15bc;
							maxRawCorrectionNearHero99_15bc = Max(maxRawCorrectionNearHero99_15bc, correction15az);
							maxLanePdfAdjustedCorrectionNearHero99_15bc = Max(maxLanePdfAdjustedCorrectionNearHero99_15bc, lanePdfAdjustedCorrection15bc);
							maxSpectralMixAdjustedCorrectionNearHero99_15bc = Max(maxSpectralMixAdjustedCorrectionNearHero99_15bc, spectralMixAdjustedCorrection15bc);
						}
						minCorrection15az = Min(minCorrection15az, correction15az);
						maxCorrection15az = Max(maxCorrection15az, correction15az);
						++evaluatedLanes15az;
						if (absDelta15az > .05f) ++lanesOver5Pct15az;
						if (absDelta15az > .10f) ++lanesOver10Pct15az;
						if (absDelta15az > .25f) ++lanesOver25Pct15az;
						if (absDelta15az > maxAbsDelta15az) {
							maxAbsDelta15az = absDelta15az;
							extremeCorrection15az = correction15az;
							extremeLambda15az = laneLambda15ba;

							localOutlier15ba.valid = true;
							localOutlier15ba.correction = correction15az;
							localOutlier15ba.absDelta = absDelta15az;
							localOutlier15ba.lambda = laneLambda15ba;
							localOutlier15ba.heroLambda = heroLambda15ba;
							localOutlier15ba.depth = pathVertex->depth;
							localOutlier15ba.caseIndex = MLHeroGlassCaseIndex15az(
								pathVertex->bsdf.hitPoint.fromLight, event);
							localOutlier15ba.heroEventPdfW = bsdfPdfW;
							localOutlier15ba.laneEventPdfW = laneEventPdfW15ba;
							localOutlier15ba.pdfRatioHeroOverLane =
								(laneEventPdfW15ba > 1e-20f) ?
								(bsdfPdfW / laneEventPdfW15ba) : 0.f;
							localOutlier15ba.pdfAdjustedCorrection = lanePdfAdjustedCorrection15bc;
							localOutlier15ba.spectralMixPdfW = spectralMixEventPdfW15bc;
							localOutlier15ba.spectralMixAdjustedCorrection = spectralMixAdjustedCorrection15bc;
							localOutlier15ba.supportMatch = supportMatch15bd;
							localOutlier15ba.supportAwareCorrection = supportAwareCorrection15bd;
							localOutlier15ba.nc = nc15ay;
							localOutlier15ba.ntBase = ntBase15ay;
							localOutlier15ba.ntLambda = ntLambda15ay;
							localOutlier15ba.eta = eta15ay;
							localOutlier15ba.eta2 = eta2_15ay;
							localOutlier15ba.cosFixed = cosFixed15ba;
							localOutlier15ba.sinI2 = sinI2_15ba;
							localOutlier15ba.sinT2 = sinT2_15ba;
							localOutlier15ba.tir = tir15ba;
							localOutlier15ba.heroIOR = heroNtLambda15bb;
							localOutlier15ba.baseSinT2 = baseSinT2_15bb;
							localOutlier15ba.heroSinT2 = heroSinT2_15bb;
							localOutlier15ba.baseTir = baseTir15bb;
							localOutlier15ba.heroTir = heroTir15bb;
							localOutlier15ba.heroDirDelta = heroDirDelta15bb;
							localOutlier15ba.baseDirDelta = baseDirDelta15bb;
							localOutlier15ba.laneDirAngle = laneDirAngle15bb;
							localOutlier15ba.heroDirAngle = heroDirAngle15bb;
							localOutlier15ba.baseDirAngle = baseDirAngle15bb;
							localOutlier15ba.fresnelR = fresnel15ay;
							localOutlier15ba.transport = transport15ay;
							localOutlier15ba.dirDelta = dirDelta15ay;
							localOutlier15ba.sharedMultiplier = sharedGlassMultiplier15az;
							localOutlier15ba.laneMultiplier = laneGlassMultiplier15ay;
						}
					}

					// 15ca: PLW controls only the base Glass multiplier. Mode 2 controls
					// the lane-event-PDF correction independently.
					float appliedGlassMultiplier15be = engine->mlHeroGlassPerLaneWeight ?
						laneGlassMultiplier15ay : sharedGlassMultiplier15br;
					if (engine->mlHeroGlassMode == 2) {
						// Experimental approximation: correct only the discrete Glass event
						// probability for this wavelength while retaining the shared HERO ray.
						// This is intentionally not presented as full spectral path-space MIS.
						const bool sameEventSupport15be = !(event & TRANSMIT) ||
							(heroTir15bb == tir15ba);
						if (sameEventSupport15be && (laneEventPdfW15ba > 1e-20f))
							appliedGlassMultiplier15be *= bsdfPdfW / laneEventPdfW15ba;
						else {
							const float sharedGlassMultiplier15be =
								(bsdfSample.c[0] + bsdfSample.c[1] + bsdfSample.c[2]) / 3.f;
							appliedGlassMultiplier15be = sharedGlassMultiplier15be;
						}
					}

					laneAppliedGlassMultiplier15br[lane] = appliedGlassMultiplier15be;
					pathVertex->mlHeroLaneThroughput[lane] *= Spectrum(appliedGlassMultiplier15be);
				} else {
					// Safety fallback: preserve the previous shared-HERO behavior.
					const float scalarBsdf15ay =
						(bsdfSample.c[0] + bsdfSample.c[1] + bsdfSample.c[2]) / 3.f;
					laneAppliedGlassMultiplier15br[lane] = scalarBsdf15ay;
					pathVertex->mlHeroLaneThroughput[lane] *= Spectrum(scalarBsdf15ay);
				}
			} else if (pathVertex->bsdf.EvaluateMLHeroMetal2SampleAtWaveLength(sampledDir,
					pathVertex->mlHeroLaneWaveLength[lane], &laneSample)) {
				pathVertex->mlHeroLaneThroughput[lane] *= laneSample;
				usedExplicitMetal2 = true;
			} else if (glossy2SpectralActive15ae && glossy2KdSPD15ae) {
				const float lambda15ae = pathVertex->mlHeroLaneWaveLength[lane];
				const float kdSpectral15ae = engine->mlHeroGlossy2BasisCompensation ?
					MLHeroGlossy2BasisCompensatedSample15cp(glossy2Kd15ae, lambda15ae) :
					MLHeroGlossy2ReflSampleNonNegative380_780(*glossy2KdSPD15ae, lambda15ae);
				const float laneGlossy215ae =
					glossy2CoatingScalar15ae +
					glossy2ResidualScalar15ae +
					glossy2BaseFactor15ae * kdSpectral15ae;

				pathVertex->mlHeroLaneThroughput[lane] *= Spectrum(laneGlossy215ae);
			} else if (genericReflectanceSPD) {
				const float lambda15cc = pathVertex->mlHeroLaneWaveLength[lane];
				const float laneReflectanceRaw15cc =
					MLHeroRGBReflSample380_780(*genericReflectanceSPD, lambda15cc);
				const bool matte15cc = (pathVertex->bsdf.GetMaterialType() == MATTE);
				const float laneReflectance15cc = matte15cc ?
					(engine->mlHeroMatteBasisCompensation ?
						MLHeroMatteBasisCompensatedSample15cj(bsdfSample, lambda15cc) :
						MLHeroMatteReflSampleNonNegative380_780(*genericReflectanceSPD, lambda15cc)) :
					(engine->mlHeroGenericReflectanceCompensation ?
						MLHeroGenericBasisCompensatedSample15cn(bsdfSample, lambda15cc) :
						laneReflectanceRaw15cc);
				pathVertex->mlHeroLaneThroughput[lane] *= Spectrum(laneReflectance15cc);
				usedGenericReflectanceSPD = true;
			} else {
				// ML HERO phase 15y:
				// Some nominally achromatic BSDFs differ by ~1e-6 between RGB
				// channels due to floating-point math. Multiplying that tiny RGB
				// noise into a lane can later push the lane outside the scalar
				// tolerance. If the BSDF is already classified as scalar, collapse
				// it to one exact scalar value before multiplying the HERO lane.
				if (MLHeroIsScalarSpectrum(bsdfSample)) {
					const float scalarBsdf =
						(bsdfSample.c[0] + bsdfSample.c[1] + bsdfSample.c[2]) / 3.f;
					pathVertex->mlHeroLaneThroughput[lane] *= Spectrum(scalarBsdf);
				} else {
					pathVertex->mlHeroLaneThroughput[lane] *= bsdfSample;
				}
			}
		}

		if (MLHeroCurrentDiagnosticsEnabled15cq() && usedGenericReflectanceSPD && engine->mlHeroMatteBasisCompensation &&
				(pathVertex->bsdf.GetMaterialType() == MATTE) &&
				(materialType15cl < MLHERO_MATERIAL_TYPE_COUNT_15CL))
			mlHero15clMatteCompensated[materialType15cl].fetch_add(1ull, std::memory_order_relaxed);

		if (MLHeroCurrentDiagnosticsEnabled15cq() && usedGenericReflectanceSPD && engine->mlHeroGenericReflectanceCompensation &&
				(pathVertex->bsdf.GetMaterialType() != MATTE) &&
				(materialType15cl < MLHERO_MATERIAL_TYPE_COUNT_15CL))
			mlHero15cnGenericCompensated[materialType15cl].fetch_add(1ull, std::memory_order_relaxed);

		// ML HERO phase 15be mode 1: after a truly dispersive, perfectly-specular
		// transmission bounce, keep only the HERO wavelength (lane 0).  Since the
		// rest of this experimental renderer reconstructs packet radiance as 1/N,
		// scale the surviving lane by N before zeroing the secondary lanes.
		if ((engine->mlHeroGlassMode == 1) &&
				(event & TRANSMIT) && dispersiveSpecularTransmit15be &&
				(pathVertex->mlHeroLaneCount > 1u)) {
			const u_int packetCount15be = pathVertex->mlHeroLaneCount;
			bool hadLiveSecondary15be = false;
			for (u_int lane15be = 1u; lane15be < packetCount15be; ++lane15be) {
				if (!pathVertex->mlHeroLaneThroughput[lane15be].Black()) {
					hadLiveSecondary15be = true;
					break;
				}
			}

			float packetScalarSum15bf = 0.f;
			for (u_int lane15bf = 0u; lane15bf < packetCount15be; ++lane15bf)
				packetScalarSum15bf += pathVertex->mlHeroLaneThroughput[lane15bf].c[0];
			const float heroScalar15bf = pathVertex->mlHeroLaneThroughput[0].c[0];
			const float packetAverage15bf =
				(packetCount15be > 0u) ? (packetScalarSum15bf / (float)packetCount15be) : 0.f;

			// Phase 15br counterfactual: the exact same pre-Glass lane packet, but
			// every lane receives LuxCore's shared Glass scalar.  The xN step below
			// cancels with the existing 1/N reconstruction, so comparing the HERO
			// scalar here also compares the final reconstructed Mode-1 energy.
			float sharedPacketScalarSum15br = 0.f;
			for (u_int lane15br = 0u; lane15br < packetCount15be; ++lane15br)
				sharedPacketScalarSum15br += lanePreGlassScalar15br[lane15br] * sharedGlassMultiplier15br;
			const float sharedPacketAverage15br = (packetCount15be > 0u) ?
				(sharedPacketScalarSum15br / (float)packetCount15be) : 0.f;
			const float sharedHeroScalar15br = lanePreGlassScalar15br[0] * sharedGlassMultiplier15br;
			const float heroPerLaneOverShared15br = (fabsf(sharedHeroScalar15br) > 1e-30f) ?
				(heroScalar15bf / sharedHeroScalar15br) : 0.f;
			const float packetPerLaneOverShared15br = (fabsf(sharedPacketAverage15br) > 1e-30f) ?
				(packetAverage15bf / sharedPacketAverage15br) : 0.f;

			const float heroOnlyBrightnessRatio15bf =
				(packetAverage15bf > 1e-20f) ? (heroScalar15bf / packetAverage15bf) : 0.f;
			const float idealPacketScale15bf =
				(heroScalar15bf > 1e-20f) ? (packetScalarSum15bf / heroScalar15bf) : 0.f;

			// Keep these diagnostics in the outer scope because the first-activation
			// log below is emitted after the termination block.
			float heroPdfBefore15bh = pathVertex->mlHeroLaneWaveLengthPdf[0];
			float heroPdfAfter15bh = heroPdfBefore15bh;

			// Phase 15bn: whole-path spectral technique weight, restricted to the
			// wavelength-dependent Glass probabilities currently available in LuxCore.
			// Two versions are reported: event-only (known to be insufficient for a
			// perfect Dirac interface) and exact-direction-support-aware.
			double eventDenom15bn = 0.0, supportDenom15bn = 0.0;
			u_int supportCompetitors15bn = 0u;
			for (u_int lane15bn = 0u; lane15bn < packetCount15be; ++lane15bn) {
				eventDenom15bn += pathVertex->mlHeroLaneGlassEventPdfProduct[lane15bn];
				supportDenom15bn += pathVertex->mlHeroLaneDiracSupportPdfProduct[lane15bn];
				if ((lane15bn > 0u) && (pathVertex->mlHeroLaneDiracSupportPdfProduct[lane15bn] > 0.0))
					++supportCompetitors15bn;
			}
			const double heroEventProduct15bn = pathVertex->mlHeroLaneGlassEventPdfProduct[0];
			const double heroSupportProduct15bn = pathVertex->mlHeroLaneDiracSupportPdfProduct[0];
			const double eventWeight15bn = (eventDenom15bn > 0.0) ? heroEventProduct15bn / eventDenom15bn : 0.0;
			const double supportWeight15bn = (supportDenom15bn > 0.0) ? heroSupportProduct15bn / supportDenom15bn : 0.0;

			if (hadLiveSecondary15be) {
				// Phase 15br: accumulate the actual per-lane-vs-shared Glass energy shift
				// at the exact first HERO-only termination.  No value is applied to the render.
				const bool valid15br = std::isfinite(heroPerLaneOverShared15br) &&
					std::isfinite(packetPerLaneOverShared15br) &&
					std::isfinite(heroScalar15bf) && std::isfinite(sharedHeroScalar15br) &&
					std::isfinite(packetAverage15bf) && std::isfinite(sharedPacketAverage15br);
				if (valid15br) {
					mlHero15brTerminationCount.fetch_add(1ull, std::memory_order_relaxed);
					if (pathVertex->bsdf.hitPoint.fromLight) mlHero15brLightCount.fetch_add(1ull, std::memory_order_relaxed);
					else mlHero15brEyeCount.fetch_add(1ull, std::memory_order_relaxed);
					mlHero15brHeroRatioSum.fetch_add((double)heroPerLaneOverShared15br, std::memory_order_relaxed);
					mlHero15brPacketRatioSum.fetch_add((double)packetPerLaneOverShared15br, std::memory_order_relaxed);
					mlHero15brActualHeroSum.fetch_add((double)heroScalar15bf, std::memory_order_relaxed);
					mlHero15brSharedHeroSum.fetch_add((double)sharedHeroScalar15br, std::memory_order_relaxed);
					mlHero15brActualPacketAvgSum.fetch_add((double)packetAverage15bf, std::memory_order_relaxed);
					mlHero15brSharedPacketAvgSum.fetch_add((double)sharedPacketAverage15br, std::memory_order_relaxed);
					if (heroPerLaneOverShared15br > 1.01f) mlHero15brHeroOver101.fetch_add(1ull, std::memory_order_relaxed);
					if (heroPerLaneOverShared15br > 1.05f) mlHero15brHeroOver105.fetch_add(1ull, std::memory_order_relaxed);
					if (heroPerLaneOverShared15br > 1.25f) mlHero15brHeroOver125.fetch_add(1ull, std::memory_order_relaxed);
					if (heroPerLaneOverShared15br > 1.50f) mlHero15brHeroOver150.fetch_add(1ull, std::memory_order_relaxed);
					if (heroPerLaneOverShared15br < .99f) mlHero15brHeroUnder099.fetch_add(1ull, std::memory_order_relaxed);
					if (packetPerLaneOverShared15br > 1.01f) mlHero15brPacketOver101.fetch_add(1ull, std::memory_order_relaxed);
					if (packetPerLaneOverShared15br > 1.05f) mlHero15brPacketOver105.fetch_add(1ull, std::memory_order_relaxed);
					if (packetPerLaneOverShared15br > 1.25f) mlHero15brPacketOver125.fetch_add(1ull, std::memory_order_relaxed);
					if (packetPerLaneOverShared15br < .99f) mlHero15brPacketUnder099.fetch_add(1ull, std::memory_order_relaxed);

					const unsigned long long detail15br = mlHero15brDetailCount.fetch_add(1ull, std::memory_order_relaxed);
					if (detail15br < 16ull) {
						std::lock_guard<std::recursive_mutex> lock15br(mlHeroDebugLogMutex);
						FILE *f15br = MLHeroOpenLegacyLog15cq("a");
						if (f15br) {
							fprintf(f15br, "ML HERO phase 15br GLASS_WEIGHT_COMPARE: sample=%llu depth=%u fromLight=%u heroLambda=%.9g sharedMultiplier=%.9g heroAppliedMultiplier=%.9g heroPre=%.9g actualHero=%.9g sharedHero=%.9g heroPerLaneOverShared=%.9g actualPacketAvg=%.9g sharedPacketAvg=%.9g packetPerLaneOverShared=%.9g currentXnScale=%u\n",
								detail15br, pathVertex->depth, pathVertex->bsdf.hitPoint.fromLight ? 1u : 0u,
								pathVertex->mlHeroLaneWaveLength[0], sharedGlassMultiplier15br,
								laneAppliedGlassMultiplier15br[0], lanePreGlassScalar15br[0],
								heroScalar15bf, sharedHeroScalar15br, heroPerLaneOverShared15br,
								packetAverage15bf, sharedPacketAverage15br, packetPerLaneOverShared15br, packetCount15be);
							fclose(f15br);
						}
					}
				} else {
					mlHero15brInvalidCount.fetch_add(1ull, std::memory_order_relaxed);
				}

				mlHero15bnTerminationCount.fetch_add(1ull, std::memory_order_relaxed);
				mlHero15bnEventWeightSum.fetch_add(eventWeight15bn, std::memory_order_relaxed);
				mlHero15bnSupportWeightSum.fetch_add(supportWeight15bn, std::memory_order_relaxed);
				mlHero15bnEventDenomSum.fetch_add(eventDenom15bn, std::memory_order_relaxed);
				mlHero15bnSupportDenomSum.fetch_add(supportDenom15bn, std::memory_order_relaxed);
				if (eventWeight15bn < .99) mlHero15bnEventWeightBelow099.fetch_add(1ull, std::memory_order_relaxed);
				if (eventWeight15bn < .75) mlHero15bnEventWeightBelow075.fetch_add(1ull, std::memory_order_relaxed);
				if (eventWeight15bn < .50) mlHero15bnEventWeightBelow050.fetch_add(1ull, std::memory_order_relaxed);
				if (eventWeight15bn < .25) mlHero15bnEventWeightBelow025.fetch_add(1ull, std::memory_order_relaxed);
				if (fabs(supportWeight15bn - 1.0) <= 1e-6) mlHero15bnSupportWeightNearOne.fetch_add(1ull, std::memory_order_relaxed);
				if (supportCompetitors15bn > 0u) mlHero15bnSupportCompetitorCount.fetch_add(1ull, std::memory_order_relaxed);

				const unsigned long long detailIndex15bn = mlHero15bnDetailedLogCount.fetch_add(1ull, std::memory_order_relaxed);
				if (detailIndex15bn < 8ull) {
					std::lock_guard<std::recursive_mutex> lock15bn(mlHeroDebugLogMutex);
					FILE *f15bn = MLHeroOpenLegacyLog15cq("a");
					if (f15bn) {
						fprintf(f15bn, "ML HERO phase 15bn PATH_SPECTRAL_MIS: sample=%llu depth=%u fromLight=%u heroLambda=%.9g eventWeight=%.12g supportWeight=%.12g eventDenom=%.12g supportDenom=%.12g supportCompetitors=%u currentXnScale=%u eventCandidateRelativeScale=%.12g diracCandidateRelativeScale=%.12g\n",
							detailIndex15bn, pathVertex->depth, pathVertex->bsdf.hitPoint.fromLight ? 1u : 0u,
							pathVertex->mlHeroLaneWaveLength[0], eventWeight15bn, supportWeight15bn, eventDenom15bn, supportDenom15bn,
							supportCompetitors15bn, packetCount15be, eventWeight15bn, supportWeight15bn);
						for (u_int lane15bn = 0u; lane15bn < packetCount15be; ++lane15bn)
							fprintf(f15bn, "phase15bn lane=%u lambda=%.9g eventPdfProduct=%.12g diracSupportPdfProduct=%.12g liveTP=%u\n",
								lane15bn, pathVertex->mlHeroLaneWaveLength[lane15bn],
								pathVertex->mlHeroLaneGlassEventPdfProduct[lane15bn], pathVertex->mlHeroLaneDiracSupportPdfProduct[lane15bn],
								pathVertex->mlHeroLaneThroughput[lane15bn].Black() ? 0u : 1u);
						fclose(f15bn);
					}
				}
			}

			// Apply the packet-to-single-wavelength normalization only once.  On later
			// dispersive Glass bounces the secondary lanes are already zero.
			if (hadLiveSecondary15be) {
				heroPdfAfter15bh = (packetCount15be > 0u) ?
					(heroPdfBefore15bh / (float)packetCount15be) : heroPdfBefore15bh;
				pathVertex->mlHeroLaneWaveLengthPdf[0] = heroPdfAfter15bh;
				for (u_int lane15bh = 1u; lane15bh < packetCount15be; ++lane15bh)
					pathVertex->mlHeroLaneWaveLengthPdf[lane15bh] = 0.f;
				pathVertex->mlHeroSecondaryWavelengthsTerminated = true;
				pathVertex->mlHeroActiveWavelengthCount = 1u;
				pathVertex->mlHeroTerminationBrightnessRatio = heroOnlyBrightnessRatio15bf;
				pathVertex->mlHeroTerminationHeroLambda = pathVertex->mlHeroLaneWaveLength[0];
				pathVertex->mlHeroTerminationPacketAverage = packetAverage15bf;
				pathVertex->mlHeroTerminationHeroScalar = heroScalar15bf;
				pathVertex->mlHeroTerminationLaneSelectionPdf = (packetCount15be > 0u) ? (1.f / (float)packetCount15be) : 0.f;
				pathVertex->mlHeroTerminationWavelengthPdf = heroPdfBefore15bh;
				pathVertex->mlHeroTerminationCurrentScale = (float)packetCount15be;
				pathVertex->mlHero15buReferenceThroughput = (double)packetAverage15bf * (double)packetCount15be;

				// Phase 15bt: first-termination metadata, binned before any later visible
				// contribution can duplicate this vertex.
				{
					const u_int side15bt = pathVertex->bsdf.hitPoint.fromLight ? 1u : 0u;
					const int wb15bt = MLHeroWavelengthBin15bs(pathVertex->mlHeroTerminationHeroLambda);
					if ((wb15bt >= 0) && std::isfinite(heroOnlyBrightnessRatio15bf) && (heroOnlyBrightnessRatio15bf > 1e-30f)) {
						mlHero15btTerminationCount[side15bt][wb15bt].fetch_add(1ull, std::memory_order_relaxed);
						mlHero15btTerminationRatioSum[side15bt][wb15bt].fetch_add((double)heroOnlyBrightnessRatio15bf, std::memory_order_relaxed);
						mlHero15btHeroScalarSum[side15bt][wb15bt].fetch_add((double)heroScalar15bf, std::memory_order_relaxed);
						mlHero15btPacketAverageSum[side15bt][wb15bt].fetch_add((double)packetAverage15bf, std::memory_order_relaxed);
						mlHero15btLaneSelectionPdfSum[side15bt][wb15bt].fetch_add((double)pathVertex->mlHeroTerminationLaneSelectionPdf, std::memory_order_relaxed);
						mlHero15btWavelengthPdfSum[side15bt][wb15bt].fetch_add((double)pathVertex->mlHeroTerminationWavelengthPdf, std::memory_order_relaxed);
						mlHero15btCurrentScaleSum[side15bt][wb15bt].fetch_add((double)pathVertex->mlHeroTerminationCurrentScale, std::memory_order_relaxed);
						mlHero15btLocalIdealScaleSum[side15bt][wb15bt].fetch_add((double)pathVertex->mlHeroTerminationCurrentScale / (double)heroOnlyBrightnessRatio15bf, std::memory_order_relaxed);

						const unsigned long long detail15bt = mlHero15btDetailCount.fetch_add(1ull, std::memory_order_relaxed);
						if (detail15bt < 16ull) {
							std::lock_guard<std::recursive_mutex> lock15bt(mlHeroDebugLogMutex);
							FILE *f15bt = MLHeroOpenLegacyLog15cq("a");
							if (f15bt) {
								fprintf(f15bt, "ML HERO phase 15bt TERMINATION: sample=%llu depth=%u side=%s lambda=%.9g laneSelectionPdf=%.12g wavelengthPdf=%.12g packetAverage=%.12g heroScalar=%.12g heroOverPacket=%.12g currentScale=%.12g localIdealScale=%.12g\n",
									detail15bt, pathVertex->depth, side15bt ? "light" : "eye",
									pathVertex->mlHeroTerminationHeroLambda, pathVertex->mlHeroTerminationLaneSelectionPdf,
									pathVertex->mlHeroTerminationWavelengthPdf, packetAverage15bf, heroScalar15bf,
									heroOnlyBrightnessRatio15bf, pathVertex->mlHeroTerminationCurrentScale,
									pathVertex->mlHeroTerminationCurrentScale / heroOnlyBrightnessRatio15bf);
								fclose(f15bt);
							}
						}
					}
				}
                {
                    const unsigned long long storeIndex15bq = mlHero15bqStoreCount.fetch_add(1ull, std::memory_order_relaxed);
                    if (!std::isfinite(pathVertex->mlHeroTerminationBrightnessRatio) ||
                            (pathVertex->mlHeroTerminationBrightnessRatio < 0.f))
                        mlHero15bqInvalidRatioCount.fetch_add(1ull, std::memory_order_relaxed);
                    MLHeroLogState15bq("STORE", storeIndex15bq, 4u, *pathVertex);
                }
				mlHero15bhPdfTerminations.fetch_add(1ull, std::memory_order_relaxed);
				if (pathVertex->bsdf.hitPoint.fromLight)
					mlHero15bhPdfLightTerminations.fetch_add(1ull, std::memory_order_relaxed);
				else
					mlHero15bhPdfEyeTerminations.fetch_add(1ull, std::memory_order_relaxed);
				{
					std::lock_guard<std::recursive_mutex> mlHero15bfStatsLock(mlHeroDebugLogMutex);
					MLHeroGlassCorrectionStats15az &stats15bf = mlHeroGlassCorrectionStats15az;
					++stats15bf.heroOnlyTerminations;
					if (pathVertex->bsdf.hitPoint.fromLight)
						++stats15bf.heroOnlyLightTerminations;
					else
						++stats15bf.heroOnlyEyeTerminations;
					stats15bf.heroOnlyBrightnessRatioSum += heroOnlyBrightnessRatio15bf;
					stats15bf.heroOnlyIdealScaleSum += idealPacketScale15bf;
					stats15bf.heroOnlyPacketAverageSum += packetAverage15bf;
					stats15bf.heroOnlyHeroLaneSum += heroScalar15bf;
					if (heroOnlyBrightnessRatio15bf < stats15bf.heroOnlyBrightnessRatioMin)
						stats15bf.heroOnlyBrightnessRatioMin = heroOnlyBrightnessRatio15bf;
					if (heroOnlyBrightnessRatio15bf > stats15bf.heroOnlyBrightnessRatioMax)
						stats15bf.heroOnlyBrightnessRatioMax = heroOnlyBrightnessRatio15bf;
					if (idealPacketScale15bf < stats15bf.heroOnlyIdealScaleMin)
						stats15bf.heroOnlyIdealScaleMin = idealPacketScale15bf;
					if (idealPacketScale15bf > stats15bf.heroOnlyIdealScaleMax)
						stats15bf.heroOnlyIdealScaleMax = idealPacketScale15bf;
					if (heroOnlyBrightnessRatio15bf > 1.05f) ++stats15bf.heroOnlyRatioOver105;
					if (heroOnlyBrightnessRatio15bf > 1.25f) ++stats15bf.heroOnlyRatioOver125;
					if (heroOnlyBrightnessRatio15bf > 1.50f) ++stats15bf.heroOnlyRatioOver150;
					if (heroOnlyBrightnessRatio15bf < 0.95f) ++stats15bf.heroOnlyRatioUnder095;
				}
				pathVertex->mlHeroLaneThroughput[0] *= (float)packetCount15be;
			}
			for (u_int lane15be = 1u; lane15be < packetCount15be; ++lane15be)
				pathVertex->mlHeroLaneThroughput[lane15be] = Spectrum();

			static std::atomic<bool> mlHero15beFirstTerminationLogged(false);
			bool expected15be = false;
			if (hadLiveSecondary15be && mlHero15beFirstTerminationLogged.compare_exchange_strong(expected15be, true)) {
				std::lock_guard<std::recursive_mutex> mlHero15beLogLock(mlHeroDebugLogMutex);
				FILE *f15be = MLHeroOpenLegacyLog15cq("a");
				if (f15be) {
					fprintf(f15be,
						"ML HERO phase 15bh HERO_ONLY_ACTIVE: depth=%u fromLight=%u "
						"event=%u heroLambda=%.9g packetCount=%u packetScalarSum=%.9g heroScalar=%.9g packetAverage=%.9g brightnessRatio=%.9g idealScale=%.9g "
						"heroPdfBefore=%.9g heroPdfAfterTerminate=%.9g pdfEstimatorScale=%.9g currentThroughputScale=%u\n",
						pathVertex->depth, pathVertex->bsdf.hitPoint.fromLight ? 1u : 0u,
						(u_int)event, pathVertex->mlHeroLaneWaveLength[0], packetCount15be,
						packetScalarSum15bf, heroScalar15bf, packetAverage15bf,
						heroOnlyBrightnessRatio15bf, idealPacketScale15bf,
						heroPdfBefore15bh, heroPdfAfter15bh,
						(heroPdfAfter15bh > 1e-30f) ? (heroPdfBefore15bh / heroPdfAfter15bh) : 0.f,
						packetCount15be);
					fclose(f15be);
				}
			}
		}

		if (MLHeroLegacyDiagnosticsEnabled15cq() && glassPerLaneWeightActive15ay && (evaluatedLanes15az > 0u)) {
			const u_int caseIndex15az = MLHeroGlassCaseIndex15az(
				pathVertex->bsdf.hitPoint.fromLight, event);
			std::lock_guard<std::recursive_mutex> mlHeroStatsLock15az(mlHeroDebugLogMutex);
			MLHeroGlassCorrectionStats15az &stats15az = mlHeroGlassCorrectionStats15az;
			++stats15az.glassBounces;
			stats15az.evaluatedLanes += evaluatedLanes15az;
			stats15az.laneOver5Pct += lanesOver5Pct15az;
			stats15az.laneOver10Pct += lanesOver10Pct15az;
			stats15az.laneOver25Pct += lanesOver25Pct15az;
			stats15az.tirLanes += tirLanes15ba;
			stats15az.nearTir95Lanes += nearTir95Lanes15ba;
			stats15az.nearTir99Lanes += nearTir99Lanes15ba;
			stats15az.pdfMismatch20PctLanes += pdfMismatch20PctLanes15ba;
			stats15az.baseHeroTirMismatchLanes += baseHeroTirMismatchLanes15bb;
			stats15az.heroLaneTirMismatchLanes += heroLaneTirMismatchLanes15bb;
			stats15az.baseLaneTirMismatchLanes += baseLaneTirMismatchLanes15bb;
			stats15az.allThreeTirAgreeLanes += allThreeTirAgreeLanes15bb;
			stats15az.nearHero99TransmitLanes += nearHero99TransmitLanes15bc;
			stats15az.rawOver25NearHero99 += rawOver25NearHero99_15bc;
			stats15az.lanePdfAdjustedOver25NearHero99 += lanePdfAdjustedOver25NearHero99_15bc;
			stats15az.spectralMixAdjustedOver25NearHero99 += spectralMixAdjustedOver25NearHero99_15bc;
			stats15az.maxRawCorrectionNearHero99 = Max(stats15az.maxRawCorrectionNearHero99, maxRawCorrectionNearHero99_15bc);
			stats15az.maxLanePdfAdjustedCorrectionNearHero99 = Max(stats15az.maxLanePdfAdjustedCorrectionNearHero99, maxLanePdfAdjustedCorrectionNearHero99_15bc);
			stats15az.maxSpectralMixAdjustedCorrectionNearHero99 = Max(stats15az.maxSpectralMixAdjustedCorrectionNearHero99, maxSpectralMixAdjustedCorrectionNearHero99_15bc);
			stats15az.eyeTransmitNearHero99Lanes += eyeTransmitNearHero99Lanes15bd;
			stats15az.eyeTransmitNearHero99SupportMatch += eyeTransmitNearHero99SupportMatch15bd;
			stats15az.eyeTransmitNearHero99SupportMismatch += eyeTransmitNearHero99SupportMismatch15bd;
			stats15az.eyeTransmitNearHero99LanePdfOver25Supported += eyeTransmitNearHero99LanePdfOver25Supported15bd;
			stats15az.eyeTransmitSupportMismatchTotal += eyeTransmitSupportMismatchTotal15bd;
			stats15az.lightTransmitSupportMismatchTotal += lightTransmitSupportMismatchTotal15bd;
			stats15az.maxEyeTransmitNearHero99LanePdfSupported = Max(
				stats15az.maxEyeTransmitNearHero99LanePdfSupported, maxEyeTransmitNearHero99LanePdfSupported15bd);
			if (maxAbsDelta15az > .05f) ++stats15az.bounceOver5Pct;
			if (maxAbsDelta15az > .10f) ++stats15az.bounceOver10Pct;
			if (maxAbsDelta15az > .25f) ++stats15az.bounceOver25Pct;
			++stats15az.caseBounces[caseIndex15az];
			MLHeroInsertGlassOutlier15ba(&stats15az, localOutlier15ba);
			stats15az.minCorrection = Min(stats15az.minCorrection, minCorrection15az);
			stats15az.maxCorrection = Max(stats15az.maxCorrection, maxCorrection15az);
			if (maxAbsDelta15az > stats15az.maxAbsDelta) {
				stats15az.maxAbsDelta = maxAbsDelta15az;
				stats15az.extremeCorrection = extremeCorrection15az;
				stats15az.extremeLambda = extremeLambda15az;
				stats15az.extremeDepth = pathVertex->depth;
				stats15az.extremeCase = caseIndex15az;
			}
		}

		if (MLHeroLegacyHeavyDiagnosticsEnabled15cq() && glossy2SpectralActive15ae && glossy2KdSPD15ae &&
				(pathVertex->bsdf.GetMaterialType() == GLOSSY2)) {
			static std::atomic<bool> mlHeroPhase15aiChromaticLogged(false);

			const float kdMin15ai = Min(glossy2Kd15ae.c[0],
				Min(glossy2Kd15ae.c[1], glossy2Kd15ae.c[2]));
			const float kdMax15ai = Max(glossy2Kd15ae.c[0],
				Max(glossy2Kd15ae.c[1], glossy2Kd15ae.c[2]));
			const bool chromaticKd15ai = ((kdMax15ai - kdMin15ai) > 1e-4f);

			if (chromaticKd15ai) {
				bool expected15ai = false;
				if (mlHeroPhase15aiChromaticLogged.compare_exchange_strong(
						expected15ai, true)) {
					std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock15ai(
						mlHeroDebugLogMutex);
					FILE *f15ai = MLHeroOpenLegacyLog15cq("a");
					if (f15ai) {
						fprintf(f15ai,
							"ML HERO phase 15al CHROMATIC GLOSSY2 RAW VS CLAMPED: "
							"depth=%u event=%u name=%s Kd=(%.9g, %.9g, %.9g) "
							"baseFactor=%.9g coatingScalar=%.9g residualScalar=%.9g\n",
							pathVertex->depth, (u_int)event,
							pathVertex->bsdf.GetMaterialName().c_str(),
							glossy2Kd15ae.c[0], glossy2Kd15ae.c[1], glossy2Kd15ae.c[2],
							glossy2BaseFactor15ae,
							glossy2CoatingScalar15ae,
							glossy2ResidualScalar15ae);

						float mostNegative15ak = 0.f;
						float removedNegativeSum15ak = 0.f;

						// ML HERO phase 15am:
						// Quantify what the non-negative Kd clamp changes after the same
						// CIE packet reconstruction used by the active HERO transport.
						// Diagnostic only: no rendering state is changed here.
						Spectrum rawCIE15am;
						Spectrum clampedCIE15am;
						Spectrum rawNormalizedCIE15an;
						Spectrum clampedNormalizedCIE15an;
						// ML HERO phase 15ao: also reconstruct Kd itself, so we can
						// separate RGB->SPD basis error from packet Monte Carlo error.
						Spectrum packetKdRaw15ao;
						Spectrum packetKdClamped15ao;
						float cieYIntegral15an = 0.f;
						for (u_int cieIndex15an = 0; cieIndex15an < nCIE; ++cieIndex15an)
							cieYIntegral15an += CIE_Y[cieIndex15an];
						const float radiometricScale15an = 683.f * 400.f;
						const float reflectanceScale15an =
							400.f / Max(1e-20f, cieYIntegral15an);

						for (u_int lane15ak = 0; lane15ak < pathVertex->mlHeroLaneCount; ++lane15ak) {
							const float lambda15ak =
								pathVertex->mlHeroLaneWaveLength[lane15ak];
							const float sampleWeight15am =
								pathVertex->mlHeroLaneSampleWeight[lane15ak];
							const float kdRaw15ak =
								MLHeroRGBReflSample380_780(*glossy2KdSPD15ae, lambda15ak);
							const float kdClamped15ak =
								MLHeroGlossy2ReflSampleNonNegative380_780(
									*glossy2KdSPD15ae, lambda15ak);
							const float laneRaw15ak =
								glossy2CoatingScalar15ae +
								glossy2ResidualScalar15ae +
								glossy2BaseFactor15ae * kdRaw15ak;
							const float laneClamped15ak =
								glossy2CoatingScalar15ae +
								glossy2ResidualScalar15ae +
								glossy2BaseFactor15ae * kdClamped15ak;

							if (kdRaw15ak < mostNegative15ak)
								mostNegative15ak = kdRaw15ak;
							if (kdRaw15ak < 0.f)
								removedNegativeSum15ak += -kdRaw15ak;

							const Spectrum cieColor15am =
								MLHeroCIEEstimatorColor(lambda15ak, sampleWeight15am);
							rawCIE15am += cieColor15am * laneRaw15ak;
							clampedCIE15am += cieColor15am * laneClamped15ak;
							const Spectrum normalizedReflectanceColor15an =
								MLHeroCIENormalizedReflectanceEstimatorColor(
									lambda15ak, sampleWeight15am);
							rawNormalizedCIE15an +=
								normalizedReflectanceColor15an * laneRaw15ak;
							clampedNormalizedCIE15an +=
								normalizedReflectanceColor15an * laneClamped15ak;
							packetKdRaw15ao +=
								normalizedReflectanceColor15an * kdRaw15ak;
							packetKdClamped15ao +=
								normalizedReflectanceColor15an * kdClamped15ak;

							fprintf(f15ai,
								"phase15al lane %u: lambda=%.9g kdRaw=%.9g "
								"kdClamped=%.9g laneRaw=%.9g laneClamped=%.9g\n",
								lane15ak, lambda15ak, kdRaw15ak, kdClamped15ak,
								laneRaw15ak, laneClamped15ak);
						}
						fprintf(f15ai,
							"phase15al summary: mostNegativeKd=%.9g "
							"removedNegativeSum=%.9g\n",
							mostNegative15ak, removedNegativeSum15ak);

						if (pathVertex->mlHeroLaneCount > 0u) {
							rawCIE15am /= (float)pathVertex->mlHeroLaneCount;
							clampedCIE15am /= (float)pathVertex->mlHeroLaneCount;
							rawNormalizedCIE15an /= (float)pathVertex->mlHeroLaneCount;
							clampedNormalizedCIE15an /= (float)pathVertex->mlHeroLaneCount;
							packetKdRaw15ao /= (float)pathVertex->mlHeroLaneCount;
							packetKdClamped15ao /= (float)pathVertex->mlHeroLaneCount;

							const Spectrum clampDelta15am = clampedCIE15am - rawCIE15am;
							const Spectrum rawError15am = rawCIE15am - bsdfSample;
							const Spectrum clampedError15am = clampedCIE15am - bsdfSample;

							const float maxClampDelta15am = Max(fabsf(clampDelta15am.c[0]),
								Max(fabsf(clampDelta15am.c[1]), fabsf(clampDelta15am.c[2])));
							const float maxRawError15am = Max(fabsf(rawError15am.c[0]),
								Max(fabsf(rawError15am.c[1]), fabsf(rawError15am.c[2])));
							const float maxClampedError15am = Max(fabsf(clampedError15am.c[0]),
								Max(fabsf(clampedError15am.c[1]), fabsf(clampedError15am.c[2])));

							// Linear-RGB luminance proxy. This is reported as a diagnostic
							// brightness delta, not used in transport.
							const float rawY15am =
								0.2126f * rawCIE15am.c[0] +
								0.7152f * rawCIE15am.c[1] +
								0.0722f * rawCIE15am.c[2];
							const float clampedY15am =
								0.2126f * clampedCIE15am.c[0] +
								0.7152f * clampedCIE15am.c[1] +
								0.0722f * clampedCIE15am.c[2];
							const float relativeYDelta15am =
								(clampedY15am - rawY15am) / Max(1e-9f, fabsf(rawY15am));

							fprintf(f15ai,
								"ML HERO phase 15am GLOSSY2 CLAMP ENERGY CHECK: "
								"depth=%u event=%u name=%s count=%u\n",
								pathVertex->depth, (u_int)event,
								pathVertex->bsdf.GetMaterialName().c_str(),
								pathVertex->mlHeroLaneCount);
							fprintf(f15ai,
								"phase15am classic RGB=(%.9g, %.9g, %.9g)\n",
								bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2]);
							fprintf(f15ai,
								"phase15am raw CIE=(%.9g, %.9g, %.9g)\n",
								rawCIE15am.c[0], rawCIE15am.c[1], rawCIE15am.c[2]);
							fprintf(f15ai,
								"phase15am clamped CIE=(%.9g, %.9g, %.9g)\n",
								clampedCIE15am.c[0], clampedCIE15am.c[1], clampedCIE15am.c[2]);
							fprintf(f15ai,
								"phase15am clamp delta=(%.9g, %.9g, %.9g) "
								"maxAbs=%.9g relativeLuminanceDelta=%.9g\n",
								clampDelta15am.c[0], clampDelta15am.c[1], clampDelta15am.c[2],
								maxClampDelta15am, relativeYDelta15am);
							fprintf(f15ai,
								"phase15am raw error vs classic=(%.9g, %.9g, %.9g) maxAbs=%.9g\n",
								rawError15am.c[0], rawError15am.c[1], rawError15am.c[2],
								maxRawError15am);
							fprintf(f15ai,
								"phase15am clamped error vs classic=(%.9g, %.9g, %.9g) maxAbs=%.9g\n",
								clampedError15am.c[0], clampedError15am.c[1], clampedError15am.c[2],
								maxClampedError15am);

							const Spectrum rawNormalizedError15an =
								rawNormalizedCIE15an - bsdfSample;
							const Spectrum clampedNormalizedError15an =
								clampedNormalizedCIE15an - bsdfSample;
							const float maxRawNormalizedError15an =
								Max(fabsf(rawNormalizedError15an.c[0]),
									Max(fabsf(rawNormalizedError15an.c[1]),
										fabsf(rawNormalizedError15an.c[2])));
							const float maxClampedNormalizedError15an =
								Max(fabsf(clampedNormalizedError15an.c[0]),
									Max(fabsf(clampedNormalizedError15an.c[1]),
										fabsf(clampedNormalizedError15an.c[2])));

							fprintf(f15ai,
								"ML HERO phase 15an GLOSSY2 CIE NORMALIZATION CHECK: "
								"depth=%u event=%u name=%s count=%u\n",
								pathVertex->depth, (u_int)event,
								pathVertex->bsdf.GetMaterialName().c_str(),
								pathVertex->mlHeroLaneCount);
							fprintf(f15ai,
								"phase15an scales: cieYIntegral=%.9g radiometric683x400=%.9g "
								"normalizedReflectance400overY=%.9g ratio=%.9g\n",
								cieYIntegral15an, radiometricScale15an, reflectanceScale15an,
								radiometricScale15an / Max(1e-20f, reflectanceScale15an));
							fprintf(f15ai,
								"phase15an classic RGB=(%.9g, %.9g, %.9g)\n",
								bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2]);
							fprintf(f15ai,
								"phase15an radiometric raw=(%.9g, %.9g, %.9g) "
								"clamped=(%.9g, %.9g, %.9g)\n",
								rawCIE15am.c[0], rawCIE15am.c[1], rawCIE15am.c[2],
								clampedCIE15am.c[0], clampedCIE15am.c[1], clampedCIE15am.c[2]);
							fprintf(f15ai,
								"phase15an normalized reflectance raw=(%.9g, %.9g, %.9g) "
								"clamped=(%.9g, %.9g, %.9g)\n",
								rawNormalizedCIE15an.c[0], rawNormalizedCIE15an.c[1],
								rawNormalizedCIE15an.c[2],
								clampedNormalizedCIE15an.c[0], clampedNormalizedCIE15an.c[1],
								clampedNormalizedCIE15an.c[2]);
							fprintf(f15ai,
								"phase15an normalized raw error=(%.9g, %.9g, %.9g) maxAbs=%.9g\n",
								rawNormalizedError15an.c[0], rawNormalizedError15an.c[1],
								rawNormalizedError15an.c[2], maxRawNormalizedError15an);
							fprintf(f15ai,
								"phase15an normalized clamped error=(%.9g, %.9g, %.9g) maxAbs=%.9g\n",
								clampedNormalizedError15an.c[0], clampedNormalizedError15an.c[1],
								clampedNormalizedError15an.c[2], maxClampedNormalizedError15an);

							// ML HERO phase 15ao:
							// Dense 1 nm (trapezoidal) roundtrip of the exact same reconstructed
							// reflectance. Comparing this against the 8-lane packet separates
							// basis/roundtrip bias from finite-packet Monte Carlo error.
							Spectrum denseKdRaw15ao;
							Spectrum denseKdClamped15ao;
							Spectrum denseFullRaw15ao;
							Spectrum denseFullClamped15ao;
							for (u_int nm15ao = 380u; nm15ao <= 780u; ++nm15ao) {
								const float lambda15ao = (float)nm15ao;
								const float edgeWeight15ao =
									((nm15ao == 380u) || (nm15ao == 780u)) ? .5f : 1.f;
								// The HERO normalized estimator contains the 400 nm interval
								// width. Divide by 400 to obtain a 1 nm quadrature weight.
								const Spectrum oneNmColor15ao =
									MLHeroCIENormalizedReflectanceEstimatorColor(lambda15ao, 1.f) / 400.f;
								const float kdRaw15ao =
									MLHeroRGBReflSample380_780(*glossy2KdSPD15ae, lambda15ao);
								const float kdClamped15ao =
									MLHeroGlossy2ReflSampleNonNegative380_780(
										*glossy2KdSPD15ae, lambda15ao);
								const float fullRaw15ao = glossy2CoatingScalar15ae +
									glossy2ResidualScalar15ae + glossy2BaseFactor15ae * kdRaw15ao;
								const float fullClamped15ao = glossy2CoatingScalar15ae +
									glossy2ResidualScalar15ae + glossy2BaseFactor15ae * kdClamped15ao;

								denseKdRaw15ao += oneNmColor15ao * (edgeWeight15ao * kdRaw15ao);
								denseKdClamped15ao += oneNmColor15ao * (edgeWeight15ao * kdClamped15ao);
								denseFullRaw15ao += oneNmColor15ao * (edgeWeight15ao * fullRaw15ao);
								denseFullClamped15ao += oneNmColor15ao * (edgeWeight15ao * fullClamped15ao);
							}

							const Spectrum packetKdRawError15ao = packetKdRaw15ao - glossy2Kd15ae;
							const Spectrum denseKdRawError15ao = denseKdRaw15ao - glossy2Kd15ae;
							const Spectrum packetVsDenseKd15ao = packetKdRaw15ao - denseKdRaw15ao;
							const Spectrum packetVsDenseFull15ao = rawNormalizedCIE15an - denseFullRaw15ao;
							const Spectrum denseFullError15ao = denseFullRaw15ao - bsdfSample;

							fprintf(f15ai,
								"ML HERO phase 15ao GLOSSY2 DENSE VS HERO ROUNDTRIP: "
								"depth=%u event=%u name=%s packetCount=%u\n",
								pathVertex->depth, (u_int)event,
								pathVertex->bsdf.GetMaterialName().c_str(),
								pathVertex->mlHeroLaneCount);
							fprintf(f15ai,
								"phase15ao target Kd=(%.9g, %.9g, %.9g)\n",
								glossy2Kd15ae.c[0], glossy2Kd15ae.c[1], glossy2Kd15ae.c[2]);
							fprintf(f15ai,
								"phase15ao packet Kd raw=(%.9g, %.9g, %.9g) clamped=(%.9g, %.9g, %.9g)\n",
								packetKdRaw15ao.c[0], packetKdRaw15ao.c[1], packetKdRaw15ao.c[2],
								packetKdClamped15ao.c[0], packetKdClamped15ao.c[1], packetKdClamped15ao.c[2]);
							fprintf(f15ai,
								"phase15ao dense1nm Kd raw=(%.9g, %.9g, %.9g) clamped=(%.9g, %.9g, %.9g)\n",
								denseKdRaw15ao.c[0], denseKdRaw15ao.c[1], denseKdRaw15ao.c[2],
								denseKdClamped15ao.c[0], denseKdClamped15ao.c[1], denseKdClamped15ao.c[2]);
							fprintf(f15ai,
								"phase15ao packet Kd error=(%.9g, %.9g, %.9g) "
								"dense Kd error=(%.9g, %.9g, %.9g)\n",
								packetKdRawError15ao.c[0], packetKdRawError15ao.c[1], packetKdRawError15ao.c[2],
								denseKdRawError15ao.c[0], denseKdRawError15ao.c[1], denseKdRawError15ao.c[2]);
							fprintf(f15ai,
								"phase15ao packet-minus-dense Kd=(%.9g, %.9g, %.9g)\n",
								packetVsDenseKd15ao.c[0], packetVsDenseKd15ao.c[1], packetVsDenseKd15ao.c[2]);
							fprintf(f15ai,
								"phase15ao classic BSDF=(%.9g, %.9g, %.9g)\n",
								bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2]);
							fprintf(f15ai,
								"phase15ao packet full raw=(%.9g, %.9g, %.9g) dense1nm full raw=(%.9g, %.9g, %.9g)\n",
								rawNormalizedCIE15an.c[0], rawNormalizedCIE15an.c[1], rawNormalizedCIE15an.c[2],
								denseFullRaw15ao.c[0], denseFullRaw15ao.c[1], denseFullRaw15ao.c[2]);
							fprintf(f15ai,
								"phase15ao dense full error=(%.9g, %.9g, %.9g) "
								"packet-minus-dense full=(%.9g, %.9g, %.9g)\n",
								denseFullError15ao.c[0], denseFullError15ao.c[1], denseFullError15ao.c[2],
								packetVsDenseFull15ao.c[0], packetVsDenseFull15ao.c[1], packetVsDenseFull15ao.c[2]);

							// ML HERO phase 15ap:
							// Scan all unique packet phases for Sampling 1.0 (Linear).
							// With N lanes, shifting the base offset by 1/N only permutes
							// the same wavelength set, so the unique phase interval is
							// [0, 1/N). 64 midpoint phases over that interval provide
							// a deterministic estimate of packet bias vs. packet-phase noise.
							const u_int phaseCount15ap = 64u;
							const u_int laneCount15ap = pathVertex->mlHeroLaneCount;
							Spectrum meanKd15ap;
							Spectrum meanFull15ap;
							Spectrum sumSqKd15ap;
							Spectrum sumSqFull15ap;
							Spectrum minKd15ap;
							Spectrum maxKd15ap;
							Spectrum minFull15ap;
							Spectrum maxFull15ap;

							for (u_int phase15ap = 0; phase15ap < phaseCount15ap; ++phase15ap) {
								const float baseU15ap =
									((float)phase15ap + .5f) /
									((float)phaseCount15ap * (float)laneCount15ap);
								Spectrum packetKd15ap;
								Spectrum packetFull15ap;

								for (u_int lane15ap = 0; lane15ap < laneCount15ap; ++lane15ap) {
									const float u15ap = baseU15ap +
										(float)lane15ap / (float)laneCount15ap;
									const float lambda15ap = Lerp(u15ap, 380.f, 780.f);
									const Spectrum reflColor15ap =
										MLHeroCIENormalizedReflectanceEstimatorColor(lambda15ap, 1.f);
									const float kdRaw15ap =
										MLHeroRGBReflSample380_780(*glossy2KdSPD15ae, lambda15ap);
									const float fullRaw15ap = glossy2CoatingScalar15ae +
										glossy2ResidualScalar15ae + glossy2BaseFactor15ae * kdRaw15ap;

									packetKd15ap += reflColor15ap * kdRaw15ap;
									packetFull15ap += reflColor15ap * fullRaw15ap;
								}

								packetKd15ap /= (float)laneCount15ap;
								packetFull15ap /= (float)laneCount15ap;

								if (phase15ap == 0u) {
									minKd15ap = maxKd15ap = packetKd15ap;
									minFull15ap = maxFull15ap = packetFull15ap;
								} else {
									for (u_int c15ap = 0; c15ap < 3u; ++c15ap) {
										minKd15ap.c[c15ap] = Min(minKd15ap.c[c15ap], packetKd15ap.c[c15ap]);
										maxKd15ap.c[c15ap] = Max(maxKd15ap.c[c15ap], packetKd15ap.c[c15ap]);
										minFull15ap.c[c15ap] = Min(minFull15ap.c[c15ap], packetFull15ap.c[c15ap]);
										maxFull15ap.c[c15ap] = Max(maxFull15ap.c[c15ap], packetFull15ap.c[c15ap]);
									}
								}

								meanKd15ap += packetKd15ap;
								meanFull15ap += packetFull15ap;
								const Spectrum errKd15ap = packetKd15ap - denseKdRaw15ao;
								const Spectrum errFull15ap = packetFull15ap - denseFullRaw15ao;
								for (u_int c15ap = 0; c15ap < 3u; ++c15ap) {
									sumSqKd15ap.c[c15ap] += errKd15ap.c[c15ap] * errKd15ap.c[c15ap];
									sumSqFull15ap.c[c15ap] += errFull15ap.c[c15ap] * errFull15ap.c[c15ap];
								}
							}

							meanKd15ap /= (float)phaseCount15ap;
							meanFull15ap /= (float)phaseCount15ap;
							Spectrum rmsKd15ap;
							Spectrum rmsFull15ap;
							for (u_int c15ap = 0; c15ap < 3u; ++c15ap) {
								rmsKd15ap.c[c15ap] = sqrtf(sumSqKd15ap.c[c15ap] / (float)phaseCount15ap);
								rmsFull15ap.c[c15ap] = sqrtf(sumSqFull15ap.c[c15ap] / (float)phaseCount15ap);
							}

							const Spectrum meanMinusDenseKd15ap = meanKd15ap - denseKdRaw15ao;
							const Spectrum meanMinusTargetKd15ap = meanKd15ap - glossy2Kd15ae;
							const Spectrum meanMinusDenseFull15ap = meanFull15ap - denseFullRaw15ao;
							const Spectrum meanMinusClassicFull15ap = meanFull15ap - bsdfSample;

							fprintf(f15ai,
								"ML HERO phase 15ap GLOSSY2 PACKET OFFSET BIAS CHECK: "
								"uniqueOffsets=%u lanes=%u phaseRange=[0, %.9g)\n",
								phaseCount15ap, laneCount15ap, 1.f / (float)laneCount15ap);
							fprintf(f15ai,
								"phase15ap Kd mean=(%.9g, %.9g, %.9g) dense=(%.9g, %.9g, %.9g) target=(%.9g, %.9g, %.9g)\n",
								meanKd15ap.c[0], meanKd15ap.c[1], meanKd15ap.c[2],
								denseKdRaw15ao.c[0], denseKdRaw15ao.c[1], denseKdRaw15ao.c[2],
								glossy2Kd15ae.c[0], glossy2Kd15ae.c[1], glossy2Kd15ae.c[2]);
							fprintf(f15ai,
								"phase15ap Kd mean-minus-dense=(%.9g, %.9g, %.9g) mean-minus-target=(%.9g, %.9g, %.9g)\n",
								meanMinusDenseKd15ap.c[0], meanMinusDenseKd15ap.c[1], meanMinusDenseKd15ap.c[2],
								meanMinusTargetKd15ap.c[0], meanMinusTargetKd15ap.c[1], meanMinusTargetKd15ap.c[2]);
							fprintf(f15ai,
								"phase15ap Kd RMS-vs-dense=(%.9g, %.9g, %.9g) min=(%.9g, %.9g, %.9g) max=(%.9g, %.9g, %.9g)\n",
								rmsKd15ap.c[0], rmsKd15ap.c[1], rmsKd15ap.c[2],
								minKd15ap.c[0], minKd15ap.c[1], minKd15ap.c[2],
								maxKd15ap.c[0], maxKd15ap.c[1], maxKd15ap.c[2]);
							fprintf(f15ai,
								"phase15ap full mean=(%.9g, %.9g, %.9g) dense=(%.9g, %.9g, %.9g) classic=(%.9g, %.9g, %.9g)\n",
								meanFull15ap.c[0], meanFull15ap.c[1], meanFull15ap.c[2],
								denseFullRaw15ao.c[0], denseFullRaw15ao.c[1], denseFullRaw15ao.c[2],
								bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2]);
							fprintf(f15ai,
								"phase15ap full mean-minus-dense=(%.9g, %.9g, %.9g) mean-minus-classic=(%.9g, %.9g, %.9g) RMS-vs-dense=(%.9g, %.9g, %.9g)\n",
								meanMinusDenseFull15ap.c[0], meanMinusDenseFull15ap.c[1], meanMinusDenseFull15ap.c[2],
								meanMinusClassicFull15ap.c[0], meanMinusClassicFull15ap.c[1], meanMinusClassicFull15ap.c[2],
								rmsFull15ap.c[0], rmsFull15ap.c[1], rmsFull15ap.c[2]);
							fprintf(f15ai,
								"phase15ap full min=(%.9g, %.9g, %.9g) max=(%.9g, %.9g, %.9g)\n",
								minFull15ap.c[0], minFull15ap.c[1], minFull15ap.c[2],
								maxFull15ap.c[0], maxFull15ap.c[1], maxFull15ap.c[2]);

							// ML HERO phase 15aq:
							// Test variance-reduction candidates without changing the render path.
							// Candidate A averages two packet phases separated by half of the
							// unique phase interval (antithetic half-shift). Candidate B averages
							// four quarter-stratified phases. In a later implementation these can
							// be interleaved across consecutive samples/passes so the lane count
							// remains 8 per sample.
							const u_int phaseCount15aq = 64u;
							const u_int laneCount15aq = pathVertex->mlHeroLaneCount;
							const float phaseRange15aq = 1.f / (float)laneCount15aq;
							Spectrum sumSqPairKd15aq, sumSqPairFull15aq;
							Spectrum sumSqQuadKd15aq, sumSqQuadFull15aq;
							Spectrum maxAbsPairKd15aq, maxAbsPairFull15aq;
							Spectrum maxAbsQuadKd15aq, maxAbsQuadFull15aq;

							for (u_int phase15aq = 0; phase15aq < phaseCount15aq; ++phase15aq) {
								const float baseU15aq =
									((float)phase15aq + .5f) /
									((float)phaseCount15aq * (float)laneCount15aq);
								Spectrum packetKd15aq[4];
								Spectrum packetFull15aq[4];

								for (u_int stratum15aq = 0; stratum15aq < 4u; ++stratum15aq) {
									float phaseU15aq = baseU15aq +
										(float)stratum15aq * phaseRange15aq * .25f;
									while (phaseU15aq >= phaseRange15aq)
										phaseU15aq -= phaseRange15aq;

									for (u_int lane15aq = 0; lane15aq < laneCount15aq; ++lane15aq) {
										const float u15aq = phaseU15aq +
											(float)lane15aq / (float)laneCount15aq;
										const float lambda15aq = Lerp(u15aq, 380.f, 780.f);
										const Spectrum reflColor15aq =
											MLHeroCIENormalizedReflectanceEstimatorColor(lambda15aq, 1.f);
										const float kdRaw15aq =
											MLHeroRGBReflSample380_780(*glossy2KdSPD15ae, lambda15aq);
										const float fullRaw15aq = glossy2CoatingScalar15ae +
											glossy2ResidualScalar15ae + glossy2BaseFactor15ae * kdRaw15aq;

										packetKd15aq[stratum15aq] += reflColor15aq * kdRaw15aq;
										packetFull15aq[stratum15aq] += reflColor15aq * fullRaw15aq;
									}

									packetKd15aq[stratum15aq] /= (float)laneCount15aq;
									packetFull15aq[stratum15aq] /= (float)laneCount15aq;
								}

								const Spectrum pairKd15aq =
									(packetKd15aq[0] + packetKd15aq[2]) * .5f;
								const Spectrum pairFull15aq =
									(packetFull15aq[0] + packetFull15aq[2]) * .5f;
								const Spectrum quadKd15aq =
									(packetKd15aq[0] + packetKd15aq[1] +
									 packetKd15aq[2] + packetKd15aq[3]) * .25f;
								const Spectrum quadFull15aq =
									(packetFull15aq[0] + packetFull15aq[1] +
									 packetFull15aq[2] + packetFull15aq[3]) * .25f;

								const Spectrum errPairKd15aq = pairKd15aq - denseKdRaw15ao;
								const Spectrum errPairFull15aq = pairFull15aq - denseFullRaw15ao;
								const Spectrum errQuadKd15aq = quadKd15aq - denseKdRaw15ao;
								const Spectrum errQuadFull15aq = quadFull15aq - denseFullRaw15ao;

								for (u_int c15aq = 0; c15aq < 3u; ++c15aq) {
									sumSqPairKd15aq.c[c15aq] += errPairKd15aq.c[c15aq] * errPairKd15aq.c[c15aq];
									sumSqPairFull15aq.c[c15aq] += errPairFull15aq.c[c15aq] * errPairFull15aq.c[c15aq];
									sumSqQuadKd15aq.c[c15aq] += errQuadKd15aq.c[c15aq] * errQuadKd15aq.c[c15aq];
									sumSqQuadFull15aq.c[c15aq] += errQuadFull15aq.c[c15aq] * errQuadFull15aq.c[c15aq];
									maxAbsPairKd15aq.c[c15aq] = Max(maxAbsPairKd15aq.c[c15aq], fabsf(errPairKd15aq.c[c15aq]));
									maxAbsPairFull15aq.c[c15aq] = Max(maxAbsPairFull15aq.c[c15aq], fabsf(errPairFull15aq.c[c15aq]));
									maxAbsQuadKd15aq.c[c15aq] = Max(maxAbsQuadKd15aq.c[c15aq], fabsf(errQuadKd15aq.c[c15aq]));
									maxAbsQuadFull15aq.c[c15aq] = Max(maxAbsQuadFull15aq.c[c15aq], fabsf(errQuadFull15aq.c[c15aq]));
								}
							}

							Spectrum rmsPairKd15aq, rmsPairFull15aq, rmsQuadKd15aq, rmsQuadFull15aq;
							Spectrum reductionPairKd15aq, reductionPairFull15aq;
							Spectrum reductionQuadKd15aq, reductionQuadFull15aq;
							for (u_int c15aq = 0; c15aq < 3u; ++c15aq) {
								rmsPairKd15aq.c[c15aq] = sqrtf(sumSqPairKd15aq.c[c15aq] / (float)phaseCount15aq);
								rmsPairFull15aq.c[c15aq] = sqrtf(sumSqPairFull15aq.c[c15aq] / (float)phaseCount15aq);
								rmsQuadKd15aq.c[c15aq] = sqrtf(sumSqQuadKd15aq.c[c15aq] / (float)phaseCount15aq);
								rmsQuadFull15aq.c[c15aq] = sqrtf(sumSqQuadFull15aq.c[c15aq] / (float)phaseCount15aq);
								reductionPairKd15aq.c[c15aq] = rmsKd15ap.c[c15aq] / Max(1e-20f, rmsPairKd15aq.c[c15aq]);
								reductionPairFull15aq.c[c15aq] = rmsFull15ap.c[c15aq] / Max(1e-20f, rmsPairFull15aq.c[c15aq]);
								reductionQuadKd15aq.c[c15aq] = rmsKd15ap.c[c15aq] / Max(1e-20f, rmsQuadKd15aq.c[c15aq]);
								reductionQuadFull15aq.c[c15aq] = rmsFull15ap.c[c15aq] / Max(1e-20f, rmsQuadFull15aq.c[c15aq]);
							}

							fprintf(f15ai,
								"ML HERO phase 15aq GLOSSY2 ANTITHETIC PHASE VARIANCE CHECK: "
								"offsets=%u lanes=%u phaseRange=[0, %.9g)\n",
								phaseCount15aq, laneCount15aq, phaseRange15aq);
							fprintf(f15ai,
								"phase15aq single Kd RMS=(%.9g, %.9g, %.9g) pair-half RMS=(%.9g, %.9g, %.9g) reduction=(%.9g, %.9g, %.9g)\n",
								rmsKd15ap.c[0], rmsKd15ap.c[1], rmsKd15ap.c[2],
								rmsPairKd15aq.c[0], rmsPairKd15aq.c[1], rmsPairKd15aq.c[2],
								reductionPairKd15aq.c[0], reductionPairKd15aq.c[1], reductionPairKd15aq.c[2]);
							fprintf(f15ai,
								"phase15aq quad Kd RMS=(%.9g, %.9g, %.9g) reduction=(%.9g, %.9g, %.9g) pairMaxAbs=(%.9g, %.9g, %.9g) quadMaxAbs=(%.9g, %.9g, %.9g)\n",
								rmsQuadKd15aq.c[0], rmsQuadKd15aq.c[1], rmsQuadKd15aq.c[2],
								reductionQuadKd15aq.c[0], reductionQuadKd15aq.c[1], reductionQuadKd15aq.c[2],
								maxAbsPairKd15aq.c[0], maxAbsPairKd15aq.c[1], maxAbsPairKd15aq.c[2],
								maxAbsQuadKd15aq.c[0], maxAbsQuadKd15aq.c[1], maxAbsQuadKd15aq.c[2]);
							fprintf(f15ai,
								"phase15aq single full RMS=(%.9g, %.9g, %.9g) pair-half RMS=(%.9g, %.9g, %.9g) reduction=(%.9g, %.9g, %.9g)\n",
								rmsFull15ap.c[0], rmsFull15ap.c[1], rmsFull15ap.c[2],
								rmsPairFull15aq.c[0], rmsPairFull15aq.c[1], rmsPairFull15aq.c[2],
								reductionPairFull15aq.c[0], reductionPairFull15aq.c[1], reductionPairFull15aq.c[2]);
							fprintf(f15ai,
								"phase15aq quad full RMS=(%.9g, %.9g, %.9g) reduction=(%.9g, %.9g, %.9g) pairMaxAbs=(%.9g, %.9g, %.9g) quadMaxAbs=(%.9g, %.9g, %.9g)\n",
								rmsQuadFull15aq.c[0], rmsQuadFull15aq.c[1], rmsQuadFull15aq.c[2],
								reductionQuadFull15aq.c[0], reductionQuadFull15aq.c[1], reductionQuadFull15aq.c[2],
								maxAbsPairFull15aq.c[0], maxAbsPairFull15aq.c[1], maxAbsPairFull15aq.c[2],
								maxAbsQuadFull15aq.c[0], maxAbsQuadFull15aq.c[1], maxAbsQuadFull15aq.c[2]);

							// ML HERO phase 15ar:
							// Simulate practical quarter-phase cycling across consecutive samples.
							// Both strategies use exactly 8 wavelengths per sample. The plain baseline
							// advances packet phase with a deterministic low-discrepancy rotation;
							// the cycling strategy uses the same group base, then visits the four
							// quarter phases before choosing a new group base. This tests correlation
							// control only; it does not alter the renderer path yet.
							const u_int trialCount15ar = 64u;
							const u_int sampleCounts15ar[3] = { 4u, 8u, 16u };
							const float goldenStep15ar = 0.6180339887498948482f;
							Spectrum sumSqPlainKd15ar[3], sumSqCycleKd15ar[3];
							Spectrum sumSqPlainFull15ar[3], sumSqCycleFull15ar[3];
							Spectrum maxAbsPlainKd15ar[3], maxAbsCycleKd15ar[3];
							Spectrum maxAbsPlainFull15ar[3], maxAbsCycleFull15ar[3];

							for (u_int trial15ar = 0; trial15ar < trialCount15ar; ++trial15ar) {
								const float trialBase15ar = ((float)trial15ar + .5f) /
									((float)trialCount15ar * (float)laneCount15aq);

								for (u_int nIndex15ar = 0; nIndex15ar < 3u; ++nIndex15ar) {
									const u_int sampleCount15ar = sampleCounts15ar[nIndex15ar];
									Spectrum plainKd15ar, plainFull15ar;
									Spectrum cycleKd15ar, cycleFull15ar;

									for (u_int sample15ar = 0; sample15ar < sampleCount15ar; ++sample15ar) {
										float plainPhase15ar = trialBase15ar +
											phaseRange15aq * fmodf((float)sample15ar * goldenStep15ar, 1.f);
										while (plainPhase15ar >= phaseRange15aq)
											plainPhase15ar -= phaseRange15aq;

										const u_int group15ar = sample15ar / 4u;
										const u_int quarter15ar = sample15ar & 3u;
										float cycleGroupBase15ar = trialBase15ar +
											phaseRange15aq * fmodf((float)group15ar * goldenStep15ar, 1.f);
										while (cycleGroupBase15ar >= phaseRange15aq)
											cycleGroupBase15ar -= phaseRange15aq;
										float cyclePhase15ar = cycleGroupBase15ar +
											(float)quarter15ar * phaseRange15aq * .25f;
										while (cyclePhase15ar >= phaseRange15aq)
											cyclePhase15ar -= phaseRange15aq;

										Spectrum packetPlainKd15ar, packetPlainFull15ar;
										Spectrum packetCycleKd15ar, packetCycleFull15ar;

										for (u_int lane15ar = 0; lane15ar < laneCount15aq; ++lane15ar) {
											const float laneU15ar = (float)lane15ar / (float)laneCount15aq;
											const float lambdaPlain15ar = Lerp(plainPhase15ar + laneU15ar, 380.f, 780.f);
											const float lambdaCycle15ar = Lerp(cyclePhase15ar + laneU15ar, 380.f, 780.f);

											const float kdPlain15ar = MLHeroRGBReflSample380_780(*glossy2KdSPD15ae, lambdaPlain15ar);
											const float kdCycle15ar = MLHeroRGBReflSample380_780(*glossy2KdSPD15ae, lambdaCycle15ar);
											const float fullPlain15ar = glossy2CoatingScalar15ae + glossy2ResidualScalar15ae +
												glossy2BaseFactor15ae * kdPlain15ar;
											const float fullCycle15ar = glossy2CoatingScalar15ae + glossy2ResidualScalar15ae +
												glossy2BaseFactor15ae * kdCycle15ar;

											const Spectrum reflPlain15ar = MLHeroCIENormalizedReflectanceEstimatorColor(lambdaPlain15ar, 1.f);
											const Spectrum reflCycle15ar = MLHeroCIENormalizedReflectanceEstimatorColor(lambdaCycle15ar, 1.f);
											packetPlainKd15ar += reflPlain15ar * kdPlain15ar;
											packetPlainFull15ar += reflPlain15ar * fullPlain15ar;
											packetCycleKd15ar += reflCycle15ar * kdCycle15ar;
											packetCycleFull15ar += reflCycle15ar * fullCycle15ar;
										}

										packetPlainKd15ar /= (float)laneCount15aq;
										packetPlainFull15ar /= (float)laneCount15aq;
										packetCycleKd15ar /= (float)laneCount15aq;
										packetCycleFull15ar /= (float)laneCount15aq;

										plainKd15ar += packetPlainKd15ar;
										plainFull15ar += packetPlainFull15ar;
										cycleKd15ar += packetCycleKd15ar;
										cycleFull15ar += packetCycleFull15ar;
									}

									plainKd15ar /= (float)sampleCount15ar;
									plainFull15ar /= (float)sampleCount15ar;
									cycleKd15ar /= (float)sampleCount15ar;
									cycleFull15ar /= (float)sampleCount15ar;

									const Spectrum errPlainKd15ar = plainKd15ar - denseKdRaw15ao;
									const Spectrum errCycleKd15ar = cycleKd15ar - denseKdRaw15ao;
									const Spectrum errPlainFull15ar = plainFull15ar - denseFullRaw15ao;
									const Spectrum errCycleFull15ar = cycleFull15ar - denseFullRaw15ao;

									for (u_int c15ar = 0; c15ar < 3u; ++c15ar) {
										sumSqPlainKd15ar[nIndex15ar].c[c15ar] += errPlainKd15ar.c[c15ar] * errPlainKd15ar.c[c15ar];
										sumSqCycleKd15ar[nIndex15ar].c[c15ar] += errCycleKd15ar.c[c15ar] * errCycleKd15ar.c[c15ar];
										sumSqPlainFull15ar[nIndex15ar].c[c15ar] += errPlainFull15ar.c[c15ar] * errPlainFull15ar.c[c15ar];
										sumSqCycleFull15ar[nIndex15ar].c[c15ar] += errCycleFull15ar.c[c15ar] * errCycleFull15ar.c[c15ar];
										maxAbsPlainKd15ar[nIndex15ar].c[c15ar] = Max(maxAbsPlainKd15ar[nIndex15ar].c[c15ar], fabsf(errPlainKd15ar.c[c15ar]));
										maxAbsCycleKd15ar[nIndex15ar].c[c15ar] = Max(maxAbsCycleKd15ar[nIndex15ar].c[c15ar], fabsf(errCycleKd15ar.c[c15ar]));
										maxAbsPlainFull15ar[nIndex15ar].c[c15ar] = Max(maxAbsPlainFull15ar[nIndex15ar].c[c15ar], fabsf(errPlainFull15ar.c[c15ar]));
										maxAbsCycleFull15ar[nIndex15ar].c[c15ar] = Max(maxAbsCycleFull15ar[nIndex15ar].c[c15ar], fabsf(errCycleFull15ar.c[c15ar]));
									}
								}
							}

							fprintf(f15ai,
								"ML HERO phase 15ar GLOSSY2 QUARTER-PHASE CYCLING SIMULATION: trials=%u lanes=%u samples={4,8,16}\n",
								trialCount15ar, laneCount15aq);
							for (u_int nIndex15ar = 0; nIndex15ar < 3u; ++nIndex15ar) {
								Spectrum rmsPlainKd15ar, rmsCycleKd15ar, rmsPlainFull15ar, rmsCycleFull15ar;
								Spectrum reductionKd15ar, reductionFull15ar;
								for (u_int c15ar = 0; c15ar < 3u; ++c15ar) {
									rmsPlainKd15ar.c[c15ar] = sqrtf(sumSqPlainKd15ar[nIndex15ar].c[c15ar] / (float)trialCount15ar);
									rmsCycleKd15ar.c[c15ar] = sqrtf(sumSqCycleKd15ar[nIndex15ar].c[c15ar] / (float)trialCount15ar);
									rmsPlainFull15ar.c[c15ar] = sqrtf(sumSqPlainFull15ar[nIndex15ar].c[c15ar] / (float)trialCount15ar);
									rmsCycleFull15ar.c[c15ar] = sqrtf(sumSqCycleFull15ar[nIndex15ar].c[c15ar] / (float)trialCount15ar);
									reductionKd15ar.c[c15ar] = rmsPlainKd15ar.c[c15ar] / Max(1e-20f, rmsCycleKd15ar.c[c15ar]);
									reductionFull15ar.c[c15ar] = rmsPlainFull15ar.c[c15ar] / Max(1e-20f, rmsCycleFull15ar.c[c15ar]);
								}

								fprintf(f15ai,
									"phase15ar N=%u Kd plainRMS=(%.9g, %.9g, %.9g) cycleRMS=(%.9g, %.9g, %.9g) reduction=(%.9g, %.9g, %.9g) plainMax=(%.9g, %.9g, %.9g) cycleMax=(%.9g, %.9g, %.9g)\n",
									sampleCounts15ar[nIndex15ar],
									rmsPlainKd15ar.c[0], rmsPlainKd15ar.c[1], rmsPlainKd15ar.c[2],
									rmsCycleKd15ar.c[0], rmsCycleKd15ar.c[1], rmsCycleKd15ar.c[2],
									reductionKd15ar.c[0], reductionKd15ar.c[1], reductionKd15ar.c[2],
									maxAbsPlainKd15ar[nIndex15ar].c[0], maxAbsPlainKd15ar[nIndex15ar].c[1], maxAbsPlainKd15ar[nIndex15ar].c[2],
									maxAbsCycleKd15ar[nIndex15ar].c[0], maxAbsCycleKd15ar[nIndex15ar].c[1], maxAbsCycleKd15ar[nIndex15ar].c[2]);
								fprintf(f15ai,
									"phase15ar N=%u full plainRMS=(%.9g, %.9g, %.9g) cycleRMS=(%.9g, %.9g, %.9g) reduction=(%.9g, %.9g, %.9g) plainMax=(%.9g, %.9g, %.9g) cycleMax=(%.9g, %.9g, %.9g)\n",
									sampleCounts15ar[nIndex15ar],
									rmsPlainFull15ar.c[0], rmsPlainFull15ar.c[1], rmsPlainFull15ar.c[2],
									rmsCycleFull15ar.c[0], rmsCycleFull15ar.c[1], rmsCycleFull15ar.c[2],
									reductionFull15ar.c[0], reductionFull15ar.c[1], reductionFull15ar.c[2],
									maxAbsPlainFull15ar[nIndex15ar].c[0], maxAbsPlainFull15ar[nIndex15ar].c[1], maxAbsPlainFull15ar[nIndex15ar].c[2],
									maxAbsCycleFull15ar[nIndex15ar].c[0], maxAbsCycleFull15ar[nIndex15ar].c[1], maxAbsCycleFull15ar[nIndex15ar].c[2]);
							}
						}
						fclose(f15ai);
					}
				}
			}
		}

		if (MLHeroCurrentDiagnosticsEnabled15cq() && glossy2SpectralActive15ae && glossy2KdSPD15ae) {
			mlHero15cqGlossy2Events.fetch_add(1ull, std::memory_order_relaxed);
			if (MLHeroCurrentHeavyDiagnosticsEnabled15cq()) {
				const u_int sampleIndex15cq = mlHero15cqGlossy2SampleAttempts.fetch_add(1u, std::memory_order_relaxed);
				if (sampleIndex15cq < MLHERO_15CQ_GLOSSY2_SAMPLE_BUDGET) {
					const Spectrum kdBaseline15cq = MLHeroIntegrateMatteReflectance15cf(*glossy2KdSPD15ae, false);
					const Spectrum kdCandidate15cq = MLHeroIntegrateGenericCompensated15cn(glossy2Kd15ae);
					const Spectrum kdErrBefore15cq = kdBaseline15cq - glossy2Kd15ae;
					const Spectrum kdErrAfter15cq = kdCandidate15cq - glossy2Kd15ae;
					const double kdBefore15cq = Max(fabsf(kdErrBefore15cq.c[0]), Max(fabsf(kdErrBefore15cq.c[1]), fabsf(kdErrBefore15cq.c[2])));
					const double kdAfter15cq = Max(fabsf(kdErrAfter15cq.c[0]), Max(fabsf(kdErrAfter15cq.c[1]), fabsf(kdErrAfter15cq.c[2])));
					const Spectrum scalarSupport15cq(glossy2CoatingScalar15ae + glossy2ResidualScalar15ae);
					const Spectrum fullBaseline15cq = scalarSupport15cq + kdBaseline15cq * glossy2BaseFactor15ae;
					const Spectrum fullCandidate15cq = scalarSupport15cq + kdCandidate15cq * glossy2BaseFactor15ae;
					const Spectrum fullErrBefore15cq = fullBaseline15cq - bsdfSample;
					const Spectrum fullErrAfter15cq = fullCandidate15cq - bsdfSample;
					const double fullBefore15cq = Max(fabsf(fullErrBefore15cq.c[0]), Max(fabsf(fullErrBefore15cq.c[1]), fabsf(fullErrBefore15cq.c[2])));
					const double fullAfter15cq = Max(fabsf(fullErrAfter15cq.c[0]), Max(fabsf(fullErrAfter15cq.c[1]), fabsf(fullErrAfter15cq.c[2])));
					mlHero15cqGlossy2SampleCount.fetch_add(1ull, std::memory_order_relaxed);
					mlHero15cqGlossy2KdErrBeforeSum.fetch_add(kdBefore15cq, std::memory_order_relaxed);
					mlHero15cqGlossy2KdErrAfterSum.fetch_add(kdAfter15cq, std::memory_order_relaxed);
					MLHeroAtomicMax15co(mlHero15cqGlossy2KdErrBeforeMax, kdBefore15cq);
					MLHeroAtomicMax15co(mlHero15cqGlossy2KdErrAfterMax, kdAfter15cq);
					mlHero15cqGlossy2FullErrBeforeSum.fetch_add(fullBefore15cq, std::memory_order_relaxed);
					mlHero15cqGlossy2FullErrAfterSum.fetch_add(fullAfter15cq, std::memory_order_relaxed);
					MLHeroAtomicMax15co(mlHero15cqGlossy2FullErrBeforeMax, fullBefore15cq);
					MLHeroAtomicMax15co(mlHero15cqGlossy2FullErrAfterMax, fullAfter15cq);
				}
			}
		}

		if (MLHeroCurrentHeavyDiagnosticsEnabled15cq() && glossy2SpectralActive15ae && glossy2KdSPD15ae) {
			// ML HERO phase 15cp/15cq: deterministic 1 nm A/B for the first active Glossy2 hit.
			// Baseline uses the current non-negative Kd SPD; candidate uses the same
			// 15ci basis correction now available to the render path.  Full-BSDF values
			// re-add the untouched coating and residual scalar terms.
			static std::atomic<bool> mlHeroPhase15cpLogged(false);
			bool expected15cp = false;
			if (mlHeroPhase15cpLogged.compare_exchange_strong(expected15cp, true)) {
				Spectrum kdBaseline15cp, kdCandidate15cp;
				for (u_int i15cp = 0u; i15cp <= 400u; ++i15cp) {
					const float lambda15cp = 380.f + (float)i15cp;
					const float trap15cp = ((i15cp == 0u) || (i15cp == 400u)) ? .5f : 1.f;
					const Spectrum cie15cp = MLHeroCIENormalizedReflectanceEstimatorColor(lambda15cp, 1.f);
					const float baseKd15cp = MLHeroGlossy2ReflSampleNonNegative380_780(*glossy2KdSPD15ae, lambda15cp);
					const float candKd15cp = MLHeroGlossy2BasisCompensatedSample15cp(glossy2Kd15ae, lambda15cp);
					kdBaseline15cp += cie15cp * (baseKd15cp * (trap15cp / 400.f));
					kdCandidate15cp += cie15cp * (candKd15cp * (trap15cp / 400.f));
				}
				const Spectrum kdErrBefore15cp = kdBaseline15cp - glossy2Kd15ae;
				const Spectrum kdErrAfter15cp = kdCandidate15cp - glossy2Kd15ae;
				const float kdMaxBefore15cp = Max(fabsf(kdErrBefore15cp.c[0]), Max(fabsf(kdErrBefore15cp.c[1]), fabsf(kdErrBefore15cp.c[2])));
				const float kdMaxAfter15cp = Max(fabsf(kdErrAfter15cp.c[0]), Max(fabsf(kdErrAfter15cp.c[1]), fabsf(kdErrAfter15cp.c[2])));
				const Spectrum scalarSupport15cp(glossy2CoatingScalar15ae + glossy2ResidualScalar15ae);
				const Spectrum fullBaseline15cp = scalarSupport15cp + kdBaseline15cp * glossy2BaseFactor15ae;
				const Spectrum fullCandidate15cp = scalarSupport15cp + kdCandidate15cp * glossy2BaseFactor15ae;
				const Spectrum fullErrBefore15cp = fullBaseline15cp - bsdfSample;
				const Spectrum fullErrAfter15cp = fullCandidate15cp - bsdfSample;
				const float fullMaxBefore15cp = Max(fabsf(fullErrBefore15cp.c[0]), Max(fabsf(fullErrBefore15cp.c[1]), fabsf(fullErrBefore15cp.c[2])));
				const float fullMaxAfter15cp = Max(fabsf(fullErrAfter15cp.c[0]), Max(fabsf(fullErrAfter15cp.c[1]), fabsf(fullErrAfter15cp.c[2])));

				std::lock_guard<std::recursive_mutex> lock15cp(mlHeroDebugLogMutex);
				FILE *f15cp = MLHeroOpenCurrentLog15cq("a");
				if (f15cp) {
					fprintf(f15cp, "ML HERO phase 15cq GLOSSY2 KD BASIS A/B: switch=%u depth=%u name=%s\n",
						engine->mlHeroGlossy2BasisCompensation ? 1u : 0u, pathVertex->depth, pathVertex->bsdf.GetMaterialName().c_str());
					fprintf(f15cp, "phase15cr Kd source=(%.9g, %.9g, %.9g) baselineRecon=(%.9g, %.9g, %.9g) candidateRecon=(%.9g, %.9g, %.9g) maxErrBefore=%.9g maxErrAfter=%.9g\n",
						glossy2Kd15ae.c[0], glossy2Kd15ae.c[1], glossy2Kd15ae.c[2],
						kdBaseline15cp.c[0], kdBaseline15cp.c[1], kdBaseline15cp.c[2],
						kdCandidate15cp.c[0], kdCandidate15cp.c[1], kdCandidate15cp.c[2], kdMaxBefore15cp, kdMaxAfter15cp);
					fprintf(f15cp, "phase15cq full classic=(%.9g, %.9g, %.9g) baseline=(%.9g, %.9g, %.9g) candidate=(%.9g, %.9g, %.9g) maxErrBefore=%.9g maxErrAfter=%.9g baseFactor=%.9g coatingScalar=%.9g residualScalar=%.9g\n",
						bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2],
						fullBaseline15cp.c[0], fullBaseline15cp.c[1], fullBaseline15cp.c[2],
						fullCandidate15cp.c[0], fullCandidate15cp.c[1], fullCandidate15cp.c[2],
						fullMaxBefore15cp, fullMaxAfter15cp, glossy2BaseFactor15ae, glossy2CoatingScalar15ae, glossy2ResidualScalar15ae);
					fclose(f15cp);
				}
			}

			static std::atomic<bool> mlHeroPhase15ahLogged(false);
			bool expected15ah = false;
			if (mlHeroPhase15ahLogged.compare_exchange_strong(expected15ah, true)) {
				std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock15ah(
					mlHeroDebugLogMutex);
				FILE *f15ah = MLHeroOpenLegacyLog15cq("a");
				if (f15ah) {
					fprintf(f15ah,
						"ML HERO phase 15al GLOSSY2 SPLIT ACTIVE: depth=%u event=%u "
						"name=%s Kd=(%.9g, %.9g, %.9g) baseFactor=%.9g "
						"coatingScalar=%.9g residualScalar=%.9g\n",
						pathVertex->depth, (u_int)event,
						pathVertex->bsdf.GetMaterialName().c_str(),
						glossy2Kd15ae.c[0], glossy2Kd15ae.c[1], glossy2Kd15ae.c[2],
						glossy2BaseFactor15ae,
						glossy2CoatingScalar15ae,
						glossy2ResidualScalar15ae);
					fclose(f15ah);
				}
			}

			static std::atomic<bool> mlHeroPhase15aeLogged(false);
			bool expected15ae = false;
			if (mlHeroPhase15aeLogged.compare_exchange_strong(expected15ae, true)) {
				std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock15ae(
					mlHeroDebugLogMutex);
				FILE *f15ae = MLHeroOpenLegacyLog15cq("a");
				if (f15ae) {
					fprintf(f15ae,
						"ML HERO phase 15ae GLOSSY2 SPECTRAL KD ACTIVE: "
						"depth=%u event=%u name=%s baseFactor=%.9g "
						"coatingScalar=%.9g residualScalar=%.9g\n",
						pathVertex->depth, (u_int)event,
						pathVertex->bsdf.GetMaterialName().c_str(),
						glossy2BaseFactor15ae,
						glossy2CoatingScalar15ae,
						glossy2ResidualScalar15ae);
					fprintf(f15ae,
						"phase15ae kd=(%.9g, %.9g, %.9g) "
						"classicBSDF=(%.9g, %.9g, %.9g)\n",
						glossy2Kd15ae.c[0], glossy2Kd15ae.c[1], glossy2Kd15ae.c[2],
						bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2]);

					for (u_int lane15ae = 0; lane15ae < pathVertex->mlHeroLaneCount; ++lane15ae) {
						const float lambda15ae =
							pathVertex->mlHeroLaneWaveLength[lane15ae];
						const float kdRaw15aj =
							MLHeroRGBReflSample380_780(*glossy2KdSPD15ae, lambda15ae);
						const float kdBaseline15cp =
							MLHeroGlossy2ReflSampleNonNegative380_780(*glossy2KdSPD15ae, lambda15ae);
						const float kdApplied15cp = engine->mlHeroGlossy2BasisCompensation ?
							MLHeroGlossy2BasisCompensatedSample15cp(glossy2Kd15ae, lambda15ae) :
							kdBaseline15cp;
						const float laneGlossy215ae =
							glossy2CoatingScalar15ae +
							glossy2ResidualScalar15ae +
							glossy2BaseFactor15ae * kdApplied15cp;
						fprintf(f15ae,
							"phase15cq lane %u: lambda=%.9g kdRaw=%.9g kdBaseline=%.9g kdApplied=%.9g "
							"switch=%u laneBSDF=%.9g\n",
							lane15ae, lambda15ae, kdRaw15aj, kdBaseline15cp, kdApplied15cp,
							engine->mlHeroGlossy2BasisCompensation ? 1u : 0u, laneGlossy215ae);
					}
					fclose(f15ae);
				}
			}
		}

		if (mlHeroNearScalarCanonicalized) {
			static std::atomic<bool> mlHeroPhase15yLogged(false);
			bool expected = false;
			if (mlHeroPhase15yLogged.compare_exchange_strong(expected, true)) {
				std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock15y(mlHeroDebugLogMutex);
				FILE *f = MLHeroOpenLegacyLog15cq("a");
				if (f) {
					const float scalarBsdf =
						(bsdfSample.c[0] + bsdfSample.c[1] + bsdfSample.c[2]) / 3.f;
					const float bsdfMin = Min(bsdfSample.c[0], Min(bsdfSample.c[1], bsdfSample.c[2]));
					const float bsdfMax = Max(bsdfSample.c[0], Max(bsdfSample.c[1], bsdfSample.c[2]));
					fprintf(f,
						"ML HERO phase 15y NEAR-SCALAR CANONICALIZED: depth=%u event=%u "
						"spread=%.9g scalar=%.9g classicBSDF=(%.9g, %.9g, %.9g)\n",
						pathVertex->depth, (u_int)event,
						bsdfMax - bsdfMin, scalarBsdf,
						bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2]);
					fclose(f);
				}
			}
		}

		if (usedGenericReflectanceSPD) {
			static std::atomic<bool> mlHeroPhase15vLogged(false);
			bool expected = false;
			if (mlHeroPhase15vLogged.compare_exchange_strong(expected, true)) {
				std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock15v(mlHeroDebugLogMutex);
				FILE *f = MLHeroOpenLegacyLog15cq("a");
				if (f) {
					fprintf(f,
						"ML HERO phase 15v GENERIC REFLECTANCE SPD ACTIVE: "
						"depth=%u event=%u classicBSDF=(%.9g, %.9g, %.9g)\n",
						pathVertex->depth, (u_int)event,
						bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2]);

					const bool matte15cc = (pathVertex->bsdf.GetMaterialType() == MATTE);
					fprintf(f, "phase15cj matteNonNegative=%u matteBasisCompensation=%u farRedEndpoint=719.999\n",
						matte15cc ? 1u : 0u, engine->mlHeroMatteBasisCompensation ? 1u : 0u);
					for (u_int lane = 0; lane < pathVertex->mlHeroLaneCount; ++lane) {
						const float lambda = pathVertex->mlHeroLaneWaveLength[lane];
						const float laneReflectanceRaw15cc = MLHeroRGBReflSample380_780(*genericReflectanceSPD, lambda);
						const float laneReflectanceApplied15cc = matte15cc ?
							(engine->mlHeroMatteBasisCompensation ?
								MLHeroMatteBasisCompensatedSample15cj(bsdfSample, lambda) :
								MLHeroMatteReflSampleNonNegative380_780(*genericReflectanceSPD, lambda)) :
							laneReflectanceRaw15cc;
						const Spectrum &laneTP = pathVertex->mlHeroLaneThroughput[lane];
						fprintf(f,
							"phase15v lane %u: lambda=%.9g reflRaw=%.9g reflApplied=%.9g "
							"laneTP=(%.9g, %.9g, %.9g) scalar=%u\n",
							lane, lambda, laneReflectanceRaw15cc, laneReflectanceApplied15cc,
							laneTP.c[0], laneTP.c[1], laneTP.c[2],
							MLHeroIsScalarSpectrum(laneTP) ? 1u : 0u);
					}
					fclose(f);
				}
			}
		}
	} else
		MLHeroMultiplyLaneThroughput(pathVertex, bsdfSample);

	pathVertex->throughput *= bsdfSample;
	if (GetMLHeroWavelengthCount() > 1u) {
		bool introducedDivergence = false;
		bool alreadyDivergent = false;
		for (u_int lane = 0; lane < pathVertex->mlHeroLaneCount; ++lane) {
			const bool afterScalar = MLHeroIsScalarSpectrum(pathVertex->mlHeroLaneThroughput[lane]);
			introducedDivergence |= mlHeroBeforeScalar[lane] && !afterScalar;
			alreadyDivergent |= !mlHeroBeforeScalar[lane];
		}
		static std::atomic<bool> loggedIntroduced(false);
		static std::atomic<bool> loggedAlready(false);
		bool doLog = false;
		const char *reason = nullptr;
		if (introducedDivergence) {
			bool expected = false;
			if (loggedIntroduced.compare_exchange_strong(expected, true)) { doLog = true; reason = "FIRST_SCALAR_TO_RGB_DIVERGENCE"; }
		} else if (alreadyDivergent) {
			bool expected = false;
			if (loggedAlready.compare_exchange_strong(expected, true)) { doLog = true; reason = "ENTERED_BOUNCE_ALREADY_RGB_DIVERGENT"; }
		}
		if (doLog) {
			std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock10(mlHeroDebugLogMutex);
			FILE *f = MLHeroOpenLegacyLog15cq("a");
			if (f) {
				const float bsdfMin = Min(bsdfSample.c[0], Min(bsdfSample.c[1], bsdfSample.c[2]));
				const float bsdfMax = Max(bsdfSample.c[0], Max(bsdfSample.c[1], bsdfSample.c[2]));
				const float bsdfSpread = bsdfMax - bsdfMin;
				fprintf(f,
					"ML HERO phase 15x %s: depth=%u event=%u "
					"REFLECT=%u TRANSMIT=%u DIFFUSE=%u GLOSSY=%u SPECULAR=%u "
					"explicitMetal2=%u genericReflSPD=%u "
					"bsdfPdfW=%.9g cosSampledDir=%.9g bsdfSpread=%.9g "
					"classicBSDF=(%.9g, %.9g, %.9g)\n",
					reason, pathVertex->depth, (u_int)event,
					(event & REFLECT) ? 1u : 0u,
					(event & TRANSMIT) ? 1u : 0u,
					(event & DIFFUSE) ? 1u : 0u,
					(event & GLOSSY) ? 1u : 0u,
					(event & SPECULAR) ? 1u : 0u,
					usedExplicitMetal2 ? 1u : 0u,
					usedGenericReflectanceSPD ? 1u : 0u,
					bsdfPdfW, cosSampledDir, bsdfSpread,
					bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2]);
				for (u_int lane = 0; lane < pathVertex->mlHeroLaneCount; ++lane) {
					const Spectrum &after = pathVertex->mlHeroLaneThroughput[lane];
					fprintf(f, "phase15d bounce lane %u: lambda=%.9g nm beforeScalar=%u before=(%.9g, %.9g, %.9g) afterScalar=%u after=(%.9g, %.9g, %.9g)\n", lane, pathVertex->mlHeroLaneWaveLength[lane], mlHeroBeforeScalar[lane] ? 1u : 0u, mlHeroBeforeTP[lane].c[0], mlHeroBeforeTP[lane].c[1], mlHeroBeforeTP[lane].c[2], MLHeroIsScalarSpectrum(after) ? 1u : 0u, after.c[0], after.c[1], after.c[2]);
				}
				fclose(f);
			}
		}
	}
	assert (!pathVertex->throughput.IsNaN() && !pathVertex->throughput.IsInf());

	// New MIS weights
	if (event & SPECULAR) {
		pathVertex->dVCM = 0.f;
		// Was:
		//  const float factor = MIS(cosSampledDir / bsdfPdfW) * MIS(bsdfRevPdfW);
		//
		// but bsdfPdfW = bsdfRevPdfW for specular material.
		assert (bsdfPdfW == bsdfRevPdfW);
		const float factor = MIS(cosSampledDir);
		pathVertex->dVC *= factor;
		pathVertex->dVM *= factor;
	} else {
		pathVertex->dVC = MIS(cosSampledDir / bsdfPdfW) * (pathVertex->dVC *
				MIS(bsdfRevPdfW) + pathVertex->dVCM + misVmWeightFactor);
		pathVertex->dVM = MIS(cosSampledDir / bsdfPdfW) * (pathVertex->dVM *
				MIS(bsdfRevPdfW) + pathVertex->dVCM * misVcWeightFactor + 1.f);
		pathVertex->dVCM = MIS(1.f / bsdfPdfW);
	}

	// Update volume information
	pathVertex->volInfo.Update(event, pathVertex->bsdf);

	*nextEventRay = Ray(pathVertex->bsdf.GetRayOrigin(sampledDir), sampledDir);
	nextEventRay->UpdateMinMaxWithEpsilon();
	nextEventRay->time = time;

    if (mlHero15buTerminatedAtBounceStart && pathVertex->mlHeroSecondaryWavelengthsTerminated &&
            (pathVertex->mlHeroLaneCount > 0u) && std::isfinite(mlHero15buActualBeforeBounce) &&
            (fabs(mlHero15buActualBeforeBounce) > 1e-30)) {
        const double actualAfter15bu = (double)pathVertex->mlHeroLaneThroughput[0].c[0];
        const double transportMultiplier15bu = actualAfter15bu / mlHero15buActualBeforeBounce;
        if (std::isfinite(transportMultiplier15bu))
            pathVertex->mlHero15buReferenceThroughput *= transportMultiplier15bu;
    }

    if (pathVertex->mlHeroSecondaryWavelengthsTerminated) {
        const unsigned long long propagateIndex15bq = mlHero15bqPropagateCount.fetch_add(1ull, std::memory_order_relaxed);
        if (!std::isfinite(pathVertex->mlHeroTerminationBrightnessRatio) ||
                (pathVertex->mlHeroTerminationBrightnessRatio < 0.f))
            mlHero15bqInvalidRatioCount.fetch_add(1ull, std::memory_order_relaxed);
        MLHeroLogState15bq("PROPAGATE", propagateIndex15bq, 4u, *pathVertex);
    }

    ++pathVertex->depth;

    return true;
}

void BiDirCPURenderThread::RenderFunc(std::stop_token stop_token) {
#ifndef NDEBUG
	SLG_LOG("[BiDirCPURenderThread::" << threadIndex << "] Rendering thread started");
#endif

	//--------------------------------------------------------------------------
	// Initialization
	//--------------------------------------------------------------------------

	// This is really used only by Windows for 64+ threads support
	SetThreadGroupAffinity(threadIndex);

	BiDirCPURenderEngine *engine = (BiDirCPURenderEngine *)renderEngine;
	mlHeroDiagnostics15cq.store(engine->mlHeroDiagnostics, std::memory_order_relaxed);
	mlHeroCurrentDiagnostics15cq.store(engine->mlHeroCurrentDiagnostics, std::memory_order_relaxed);
	mlHeroLegacyDiagnostics15cq.store(engine->mlHeroLegacyDiagnostics, std::memory_order_relaxed);
	mlHeroHeavyDiagnostics15cq.store(engine->mlHeroHeavyDiagnostics, std::memory_order_relaxed);
	
	if (threadIndex == 0) {
		{
			std::lock_guard<std::recursive_mutex> mlHeroStatsResetLock15az(mlHeroDebugLogMutex);
			mlHeroGlassCorrectionStats15az = MLHeroGlassCorrectionStats15az();
			mlHero15azFinishedThreads.store(0u);
			mlHero15bhPdfTerminations.store(0ull);
			mlHero15bhPdfEyeTerminations.store(0ull);
			mlHero15bhPdfLightTerminations.store(0ull);
			mlHero15bhVisibleConnections.store(0ull);
			mlHero15bhVisibleEyeTerminated.store(0ull);
			mlHero15bhVisibleLightTerminated.store(0ull);
			mlHero15bhVisibleBothTerminated.store(0ull);
			mlHero15bhVisibleNeitherTerminated.store(0ull);
			for (u_int c15ce = 0u; c15ce < MLHERO_MATTE_CLASS_COUNT_15CE; ++c15ce) {
				mlHero15ceCount[c15ce].store(0ull, std::memory_order_relaxed);
				mlHero15ceClassicR[c15ce].store(0.0, std::memory_order_relaxed);
				mlHero15ceClassicG[c15ce].store(0.0, std::memory_order_relaxed);
				mlHero15ceClassicB[c15ce].store(0.0, std::memory_order_relaxed);
				mlHero15cePacketR[c15ce].store(0.0, std::memory_order_relaxed);
				mlHero15cePacketG[c15ce].store(0.0, std::memory_order_relaxed);
				mlHero15cePacketB[c15ce].store(0.0, std::memory_order_relaxed);
				mlHero15ceAbsErrR[c15ce].store(0.0, std::memory_order_relaxed);
				mlHero15ceAbsErrG[c15ce].store(0.0, std::memory_order_relaxed);
				mlHero15ceAbsErrB[c15ce].store(0.0, std::memory_order_relaxed);
				mlHero15ceSqErrR[c15ce].store(0.0, std::memory_order_relaxed);
				mlHero15ceSqErrG[c15ce].store(0.0, std::memory_order_relaxed);
				mlHero15ceSqErrB[c15ce].store(0.0, std::memory_order_relaxed);
				mlHero15cfKdR[c15ce].store(0.0, std::memory_order_relaxed);
				mlHero15cfKdG[c15ce].store(0.0, std::memory_order_relaxed);
				mlHero15cfKdB[c15ce].store(0.0, std::memory_order_relaxed);
			}
			for (u_int t15cl = 0u; t15cl < MLHERO_MATERIAL_TYPE_COUNT_15CL; ++t15cl) {
				mlHero15clReflectCandidate[t15cl].store(0ull, std::memory_order_relaxed);
				mlHero15clGenericRGBReflSPD[t15cl].store(0ull, std::memory_order_relaxed);
				mlHero15clMatteCompensated[t15cl].store(0ull, std::memory_order_relaxed);
				mlHero15clGlossy2Split[t15cl].store(0ull, std::memory_order_relaxed);
				mlHero15cnGenericCompensated[t15cl].store(0ull, std::memory_order_relaxed);
				mlHero15coSampleAttempts[t15cl].store(0u, std::memory_order_relaxed);
				mlHero15coSampleCount[t15cl].store(0ull, std::memory_order_relaxed);
				mlHero15coErrBeforeSum[t15cl].store(0.0, std::memory_order_relaxed);
				mlHero15coErrAfterSum[t15cl].store(0.0, std::memory_order_relaxed);
				mlHero15coErrBeforeMax[t15cl].store(0.0, std::memory_order_relaxed);
				mlHero15coErrAfterMax[t15cl].store(0.0, std::memory_order_relaxed);
			}
			mlHero15cqGlossy2Events.store(0ull, std::memory_order_relaxed);
			mlHero15cqGlossy2SampleAttempts.store(0u, std::memory_order_relaxed);
			mlHero15cqGlossy2SampleCount.store(0ull, std::memory_order_relaxed);
			mlHero15cqGlossy2KdErrBeforeSum.store(0.0, std::memory_order_relaxed);
			mlHero15cqGlossy2KdErrAfterSum.store(0.0, std::memory_order_relaxed);
			mlHero15cqGlossy2KdErrBeforeMax.store(0.0, std::memory_order_relaxed);
			mlHero15cqGlossy2KdErrAfterMax.store(0.0, std::memory_order_relaxed);
			mlHero15cqGlossy2FullErrBeforeSum.store(0.0, std::memory_order_relaxed);
			mlHero15cqGlossy2FullErrAfterSum.store(0.0, std::memory_order_relaxed);
			mlHero15cqGlossy2FullErrBeforeMax.store(0.0, std::memory_order_relaxed);
			mlHero15cqGlossy2FullErrAfterMax.store(0.0, std::memory_order_relaxed);
		}

		// ML-HERO phase 11: always start a fresh diagnostic log with an
		// unambiguous build/test header. This also makes it immediately obvious
		// whether Blender loaded the newly built pyluxcore wheel.
		std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock11(mlHeroDebugLogMutex);
		FILE *f = engine->mlHeroDiagnostics ? fopen(ML_HERO_LOG_FILE, "w") : nullptr;
		if (f) {
			fprintf(f, "ML HERO v2.11.2 | Test phase: %s | Feature: %s\n", ML_HERO_TEST_PHASE, ML_HERO_FEATURE);
			fprintf(f, "ML HERO phase 15bl CONFIG: path.mlhero.glassmode=%d (0=current per-lane, 1=HERO-only after dispersive transmit, 2=experimental lane-PDF)\n", engine->mlHeroGlassMode);
			fprintf(f, "ML HERO phase 15bz CONFIG: path.mlhero.glassperlaneweight=%u\n", engine->mlHeroGlassPerLaneWeight ? 1u : 0u);
			fprintf(f, "ML HERO phase 15bz CONFIG: path.mlhero.quartercycling=%u\n", engine->mlHeroQuarterCycling ? 1u : 0u);
			fprintf(f, "ML HERO phase 15bz CONFIG: path.mlhero.dualterminationcompensation=%u\n", engine->mlHeroDualTerminationCompensation ? 1u : 0u);
			fprintf(f, "ML HERO phase 15cj CONFIG: path.mlhero.mattebasiscompensation=%u\n", engine->mlHeroMatteBasisCompensation ? 1u : 0u);
			fprintf(f, "ML HERO phase 15co CONFIG: path.mlhero.genericreflectancecompensation=%u\n", engine->mlHeroGenericReflectanceCompensation ? 1u : 0u);
			fprintf(f, "ML HERO phase 15cr CONFIG: path.mlhero.glossy2basiscompensation=%u\n", engine->mlHeroGlossy2BasisCompensation ? 1u : 0u);
			fprintf(f, "ML HERO phase 15cr DIAGNOSTICS: master=%u current=%u legacy=%u heavy=%u\n",
				engine->mlHeroDiagnostics ? 1u : 0u, engine->mlHeroCurrentDiagnostics ? 1u : 0u,
				engine->mlHeroLegacyDiagnostics ? 1u : 0u, engine->mlHeroHeavyDiagnostics ? 1u : 0u);
			fprintf(f, "ML HERO phase 15bz CONFIG: mode1TerminationIndependentOfPerLaneWeight=1\n");
			fprintf(f, "ML HERO phase 15bz CONFIG: compensationRule=bothTerminated_ConnectVertices_divideByWavelengthCount\n");
			fprintf(f, "ML HERO phase 15ca CONFIG: mode2LanePdfIndependentOfPerLaneWeight=1\n");
			fprintf(f, "Build date/time: %s %s\n", __DATE__, __TIME__);
			fprintf(f, "Wavelength count: %u\n", (u_int)engine->mlHeroWavelengthCount);
			fprintf(f, "ML HERO phase 15cr CONFIG: diagnostics_runtime_gated=1 masterOff_fast_path=1 rendererPhysicsUnchanged=1\n");
			fprintf(f, "------------------------------------------------------------\n");
			fclose(f);
		}
	}
	
	// (engine->seedBase + 1) seed is used for sharedRndGen

	auto rndGen = std::make_unique<RandomGenerator>(engine->seedBase + 1 + threadIndex);
	auto& scene = engine->renderConfig.GetScene();
	auto& camera = scene.GetCamera();
	PhotonGICache *photonGICache = engine->photonGICache;

	// Albedo and Normal AOV warm up
	if (engine->aovWarmupSPP > 0)
		AOVWarmUp(stop_token, rndGen);

	// Setup the sampler
	auto sampler = engine->renderConfig.AllocSampler(
		rndGen,
		engine->GetFilm(),
		engine->GetSampleSplatter(),
		engine->samplerSharedData,
		Properties()
	);
	const u_int sampleSize =
		sampleBootSize + // To generate the initial light vertex and trace eye ray
		engine->maxLightPathDepth * sampleLightStepSize + // For each light vertex
		engine->maxEyePathDepth * sampleEyeStepSize + 1; // Dedicated HERO wavelength sample dimension
		sampler->SetThreadIndex(threadIndex);
	sampler->RequestSamples(PIXEL_NORMALIZED_AND_SCREEN_NORMALIZED, sampleSize);

	VarianceClamping varianceClamping(engine->sqrtVarianceClampMaxValue);

	// Disable vertex merging
	misVmWeightFactor = 0.f;
	misVcWeightFactor = 0.f;

	vector<SampleResult> sampleResults;
	vector<PathVertexVM> lightPathVertices;

	// ML HERO phase 15as: state for real quarter-phase cycling.
	// Only the packet phase inside one 1/N interval is held/cycled across
	// four consecutive samples. The current Sobol stratum is preserved on
	// every sample so lane 0 remains distributed across the full spectrum.
	float mlHeroQuarterCycleGroupPhase15as = 0.f;
	bool mlHeroQuarterCycleGroupValid15as = false;

	for(u_int steps = 0; !stop_token.stop_requested(); ++steps) {
		// Check if we are in pause mode
		if (engine->pauseMode) {
			// Check every 100ms if I have to continue the rendering
			while (!stop_token.stop_requested() && engine->pauseMode)
				std::this_thread::sleep_for(100ms);

			if (stop_token.stop_requested())
				break;
		}

		sampleResults.clear();
		lightPathVertices.clear();

		const float timeSample = sampler->GetSample(12);
		const float time = scene.GetCamera().GenerateRayTime(timeSample);

		SetMLHeroEnabled(engine->mlHeroEnabled);
		// ML HERO phase 15u:
		// When HERO is disabled, force the shared packet state back to one lane.
		// All multi-wave contribution branches are gated by laneCount > 1, so
		// this restores the original/classic BIDIR path instead of accidentally
		// leaving an 8-lane packet active from the UI setting.
		SetMLHeroWavelengthCount(engine->mlHeroEnabled ? engine->mlHeroWavelengthCount : 1u);

		// ML HERO: lane 0 remains the original HERO wavelength used by the
		// complete light + eye path pair. Additional lanes are deterministic
		// stratified companions derived from the same sampler dimension.
		// They are only prepared here; path throughput still evaluates lane 0
		// until the next multi-spectral packet patch.
		if (!engine->mlHeroEnabled && (threadIndex == 0) && (steps == 0)) {
			std::lock_guard<std::recursive_mutex> mlHeroDebugLogLockHeroOff(mlHeroDebugLogMutex);
			FILE *f = MLHeroOpenLegacyLog15cq("a");
			if (f) {
				fprintf(f,
					"ML HERO phase 15u HERO OFF CLASSIC ACTIVE: enabled=0 packetCount=%u\n",
					GetMLHeroWavelengthCount());
				fclose(f);
			}
		}

		if (engine->mlHeroEnabled) {
			const u_int mlHeroSampleIndex = sampleSize - 1;
			const float mlHeroSobolU0 = sampler->GetSample(mlHeroSampleIndex);
			float mlHeroU0 = mlHeroSobolU0;

			// ML HERO phase 15cm: enable the validated quarter-phase cycle in the
			// real renderer for Sampling 1.0 (Linear), independent of packet lane count.
			//
			// A packet with N uniformly spaced lanes is invariant to shifts of 1/N
			// apart from lane permutation. Therefore split Sobol U0 into:
			//   stratum = floor(U0*N)/N       (which lane is lane 0)
			//   phase   = U0 mod (1/N)        (packet position)
			// We keep the CURRENT Sobol stratum on every sample, but for each group
			// of four samples visit phase offsets 0, 1/4, 2/4, 3/4 of the unique
			// packet interval. This preserves lane-0 full-spectrum coverage while
			// applying the variance reduction measured in phases 15aq/15ar.
			const bool mlHeroQuarterCycling15as =
				engine->mlHeroQuarterCycling &&
				(engine->mlHeroSamplingMode == 1) &&
				(engine->mlHeroWavelengthCount >= 2);
			u_int mlHeroQuarter15as = 0u;
			float mlHeroSobolStratum15as = 0.f;
			float mlHeroPacketPhase15as = 0.f;

			if (mlHeroQuarterCycling15as) {
				const float laneCount15as = (float)engine->mlHeroWavelengthCount;
				const float phaseRange15as = 1.f / laneCount15as;
				const u_int stratumIndex15as = Min((u_int)Floor2Int(mlHeroSobolU0 * laneCount15as),
					(u_int)engine->mlHeroWavelengthCount - 1u);
				mlHeroSobolStratum15as = (float)stratumIndex15as * phaseRange15as;
				const float sobolLocalPhase15as = mlHeroSobolU0 - mlHeroSobolStratum15as;
				mlHeroQuarter15as = steps & 3u;

				if (!mlHeroQuarterCycleGroupValid15as || (mlHeroQuarter15as == 0u)) {
					mlHeroQuarterCycleGroupPhase15as = sobolLocalPhase15as;
					mlHeroQuarterCycleGroupValid15as = true;
				}

				mlHeroPacketPhase15as = mlHeroQuarterCycleGroupPhase15as +
					(float)mlHeroQuarter15as * phaseRange15as * .25f;
				while (mlHeroPacketPhase15as >= phaseRange15as)
					mlHeroPacketPhase15as -= phaseRange15as;

				mlHeroU0 = mlHeroSobolStratum15as + mlHeroPacketPhase15as;
				mlHeroU0 = Clamp(mlHeroU0, 0.f, 0.99999994f);
			} else {
				mlHeroQuarterCycleGroupValid15as = false;
			}

			for (u_int lane = 0; lane < (u_int)engine->mlHeroWavelengthCount; ++lane) {
				float mlU = mlHeroU0 + (float)lane / (float)engine->mlHeroWavelengthCount;
				mlU -= Floor2Int(mlU);
				mlU = Clamp(mlU, 0.f, 0.99999994f);

				float mlSampleWeight = 1.f;
				float mlWaveLength;
				if (engine->mlHeroSamplingMode == 3)
					mlWaveLength = MLBidirHeroSampling3Sample(mlU, &mlSampleWeight);
				else if (engine->mlHeroSamplingMode == 2)
					mlWaveLength = MLBidirHeroSampling2Sample(mlU, &mlSampleWeight);
				else
					mlWaveLength = Lerp(mlU, 380.f, 780.f);

				if (lane == 0)
					SetMLDispersionWaveLength(mlWaveLength, mlSampleWeight);
				else
					SetMLHeroWaveLengthAt(lane, mlWaveLength, mlSampleWeight);
			}
			// ML HERO phase 15at: log the first 16 real packet starts on thread 0.
			// This is intentionally more than one 4-sample group so we can verify
			// 0,1,2,3 repetition, group-base refresh, and the actual cycled U0.
			if ((threadIndex == 0) && (steps < 16u)) {
				std::lock_guard<std::recursive_mutex> mlHeroDebugLogLock12(mlHeroDebugLogMutex);
				FILE *f = MLHeroOpenLegacyLog15cq("a");
				if (f) {
					const float laneCount15at = (float)engine->mlHeroWavelengthCount;
					const float phaseRange15at = (laneCount15at > 0.f) ? 1.f / laneCount15at : 0.f;
					const float sobolLocalPhase15at = mlHeroSobolU0 - mlHeroSobolStratum15as;
					fprintf(f, "ML HERO phase 15at REAL QUARTER CYCLING SEQ: step=%u active=%u runtimeSwitch=%u mode=%u lanes=%u quarter=%u expectedQuarter=%u sobolU0=%.9g stratum=%.9g sobolLocal=%.9g groupBase=%.9g quarterOffset=%.9g packetPhase=%.9g cycledU0=%.9g lane0=%.9g\n",
						steps, mlHeroQuarterCycling15as ? 1u : 0u, engine->mlHeroQuarterCycling ? 1u : 0u,
						engine->mlHeroSamplingMode, (u_int)engine->mlHeroWavelengthCount,
						mlHeroQuarter15as, steps & 3u, mlHeroSobolU0, mlHeroSobolStratum15as,
						sobolLocalPhase15at, mlHeroQuarterCycleGroupPhase15as,
						(float)mlHeroQuarter15as * phaseRange15at * .25f, mlHeroPacketPhase15as,
						mlHeroU0, GetMLHeroWaveLengthAt(0));
					if (steps == 0u) {
						fprintf(f, "ML HERO packet count: %u\n", GetMLHeroWavelengthCount());
						for (u_int lane = 0; lane < GetMLHeroWavelengthCount(); ++lane) {
							fprintf(f, "lane %u: lambda=%.9g nm, weight=%.9g\n",
								lane, GetMLHeroWaveLengthAt(lane), GetMLHeroSampleWeightAt(lane));
						}
					}
					fclose(f);
				}
			}
		}
	
	
		/*
		// Sample a point on the camera lens
		Point lensPoint;
		if (!camera.SampleLens(time, sampler->GetSample(3), sampler->GetSample(4),
				&lensPoint)) {
			assert (SampleResult::IsAllValid(sampleResults));

			sampler->NextSample(sampleResults);
			continue;
		}
		*/

		//----------------------------------------------------------------------
		// Trace light path
		//----------------------------------------------------------------------

		const bool validLightPath = TraceLightPath(time, sampler, camera, lightPathVertices, sampleResults);
		assert (SampleResult::IsAllValid(sampleResults));

		if (validLightPath) {
			//------------------------------------------------------------------
			// Trace eye path
			//------------------------------------------------------------------

			PathVertexVM eyeVertex;
			SampleResult &eyeSampleResult = AddResult(sampleResults, false);

			eyeSampleResult.filmX = sampler->GetSample(0);
			eyeSampleResult.filmY = sampler->GetSample(1);
			Ray eyeRay;
			camera.GenerateRay(time,
					eyeSampleResult.filmX, eyeSampleResult.filmY, &eyeRay,
					&eyeVertex.volInfo, sampler->GetSample(10), sampler->GetSample(11));

			// Required by MIS weights update
			eyeVertex.bsdf.hitPoint.fixedDir = -eyeRay.d;
			eyeVertex.throughput = Spectrum(1.f);
			eyeVertex.depth = 1;
			MLHeroInitLaneThroughput(&eyeVertex, eyeVertex.throughput);
			if (GetMLHeroWavelengthCount() > 1u) {
				static std::atomic<bool> loggedEyeInit(false);
				bool expected = false;
				if (loggedEyeInit.compare_exchange_strong(expected, true))
					MLHeroLogLanePurityBlock("EYE_INIT", eyeVertex);
			}
			float cameraPdfW;
			scene.GetCamera().GetPDF(eyeRay, 0.f, eyeSampleResult.filmX, eyeSampleResult.filmY, &cameraPdfW, nullptr);
			eyeVertex.dVCM = MIS(1.f / cameraPdfW);
			eyeVertex.dVC = 0.f;
			eyeVertex.dVM = 0.f;

			bool albedoToDo = true;
			eyeSampleResult.albedo = Spectrum(); // Just in case albedoToDo is never true
			eyeSampleResult.shadingNormal = Normal();
			bool photonGICausticCacheUsed = false;
			bool isTransmittedEyePath = true;
			while (eyeVertex.depth <= engine->maxEyePathDepth) {
				eyeSampleResult.firstPathVertex = (eyeVertex.depth == 1);
				eyeSampleResult.lastPathVertex = (eyeVertex.depth == engine->maxEyePathDepth);

				const u_int sampleOffset = sampleBootSize + engine->maxLightPathDepth * sampleLightStepSize +
					(eyeVertex.depth - 1) * sampleEyeStepSize;

				// NOTE: I account for volume emission only with path tracing (i.e. here and
				// not in any other place)
				RayHit eyeRayHit;
				Spectrum connectionThroughput;
				const bool hit = scene.Intersect(device,
						EYE_RAY | (eyeSampleResult.firstPathVertex ? CAMERA_RAY : INDIRECT_RAY),
						&eyeVertex.volInfo, sampler->GetSample(sampleOffset),
						&eyeRay, &eyeRayHit, &eyeVertex.bsdf,
						&connectionThroughput, &eyeVertex.throughput, &eyeSampleResult);

				if (!hit) {
					// Nothing was hit, look for infinitelight

					// This is a trick, you can not have a BSDF of something that has
					// not been hit. DirectHitInfiniteLight must be aware of this.
					eyeVertex.bsdf.hitPoint.fixedDir = -eyeRay.d;
					eyeVertex.throughput *= connectionThroughput;
					MLHeroMultiplyLaneThroughput(&eyeVertex, connectionThroughput);

					DirectHitLight(false, eyeVertex, eyeSampleResult);

					if (eyeSampleResult.firstPathVertex) {
						eyeSampleResult.alpha = 0.f;
						eyeSampleResult.depth = numeric_limits<float>::infinity();
						eyeSampleResult.position = Point(
								numeric_limits<float>::infinity(),
								numeric_limits<float>::infinity(),
								numeric_limits<float>::infinity());
						eyeSampleResult.geometryNormal = Normal();
						eyeSampleResult.shadingNormal = Normal();
						eyeSampleResult.materialID = 0;
						eyeSampleResult.objectID = 0;
						eyeSampleResult.uv = UV(numeric_limits<float>::infinity(),
								numeric_limits<float>::infinity());
					} else if (isTransmittedEyePath) {
						// I set to 0.0 also the alpha all purely transmitted paths hitting nothing
						eyeSampleResult.alpha = 0.f;
					}
					break;
				}
				eyeVertex.throughput *= connectionThroughput;
				MLHeroMultiplyLaneThroughput(&eyeVertex, connectionThroughput);

				// Something was hit

				if (albedoToDo && eyeVertex.bsdf.IsAlbedoEndPoint(engine->albedoSpecularSetting,
						engine->albedoSpecularGlossinessThreshold)) {
					eyeSampleResult.albedo = eyeVertex.throughput * eyeVertex.bsdf.Albedo();
					eyeSampleResult.shadingNormal = eyeVertex.bsdf.hitPoint.shadeN;
					albedoToDo = false;
				}

				float t_MIS;
				if (eyeSampleResult.firstPathVertex) {
					eyeSampleResult.alpha = 1.f;
					eyeSampleResult.depth = eyeRayHit.t;
					eyeSampleResult.position = eyeVertex.bsdf.hitPoint.p;
					eyeSampleResult.geometryNormal = eyeVertex.bsdf.hitPoint.geometryN;
					eyeSampleResult.materialID = eyeVertex.bsdf.GetMaterialID();
					eyeSampleResult.objectID = eyeVertex.bsdf.GetObjectID();
					eyeSampleResult.uv = eyeVertex.bsdf.hitPoint.GetUV(0);
					// for the camera ray, we need to add the clipping distance
					// because eyeRayHit.t is measured from clipping start.
					// Otherwise, the brightness near the front clipping plane may be distorted.
					t_MIS = eyeRayHit.t + camera.clipHither;
				} 
				else{
					t_MIS = eyeRayHit.t;
				}

				// Update MIS constants
				const float factor = 1.f / MIS(AbsDot(eyeVertex.bsdf.hitPoint.shadeN, eyeVertex.bsdf.hitPoint.fixedDir));
				eyeVertex.dVCM *= MIS(t_MIS * t_MIS) * factor;
				eyeVertex.dVC *= factor;
				eyeVertex.dVM *= factor;

				// Check if it is a light source
				if (eyeVertex.bsdf.IsLightSource() &&
					// Avoid to render caustic path if PhotonGI caustic cache
					// has been used (for SDS paths)
					!photonGICausticCacheUsed){
					DirectHitLight(true, eyeVertex, eyeSampleResult);
				}

				// Note: pass-through check is done inside Scene::Intersect()
		
				//--------------------------------------------------------------
				// Check if I can use the photon cache
				//--------------------------------------------------------------

				if (photonGICache) {
					const bool isPhotonGIEnabled = photonGICache->IsPhotonGIEnabled(eyeVertex.bsdf);

					// Check if the cache is enabled for this material
					if (isPhotonGIEnabled) {
						// TODO: add support for AOVs (possible ?)

						if (photonGICache->IsCausticEnabled() && (eyeVertex.depth > 1)) {
							const SpectrumGroup causticRadiance = photonGICache->ConnectWithCausticPaths(eyeVertex.bsdf);

							if (!causticRadiance.Black())
								eyeSampleResult.radiance.AddWeighted(eyeVertex.throughput, causticRadiance);
							
							photonGICausticCacheUsed = true;
						}
					}
				}

				//--------------------------------------------------------------
				// Direct light sampling
				//--------------------------------------------------------------

				DirectLightSampling(time,
						sampler->GetSample(sampleOffset + 1),
						sampler->GetSample(sampleOffset + 2),
						sampler->GetSample(sampleOffset + 3),
						sampler->GetSample(sampleOffset + 4),
						sampler->GetSample(sampleOffset + 5),
						eyeVertex, eyeSampleResult);

				assert (eyeSampleResult.IsValid());

				//--------------------------------------------------------------
				// Connect vertex path ray with all light path vertices
				//--------------------------------------------------------------

				if (!eyeVertex.bsdf.IsDelta()) {
					for (vector<PathVertexVM>::const_iterator lightPathVertex = lightPathVertices.begin();
							lightPathVertex < lightPathVertices.end(); ++lightPathVertex)
						ConnectVertices(time, eyeVertex, *lightPathVertex, eyeSampleResult,
								sampler->GetSample(sampleOffset + 6));
					
					assert (eyeSampleResult.IsValid());
				}

				//--------------------------------------------------------------
				// Build the next vertex path ray
				//--------------------------------------------------------------

				if (!Bounce(time, sampler, sampleOffset + 7, &eyeVertex, &eyeRay))
					break;
				
				isTransmittedEyePath = isTransmittedEyePath && (eyeVertex.bsdfEvent & TRANSMIT);

#ifdef WIN32
				// Work around Windows bad scheduling
                std::this_thread::yield();
#endif
			}
		}
		
		// ML HERO phase 15bv:
		// The legacy single-HERO renderer colors the completed BIDIR sample here.
		// Multi-wavelength BIDIR contributions are already reconstructed to RGB in
		// ConnectVertices/DirectLightSampling/ConnectToEye/DirectHitLight via
		// MLHeroCIEEstimatorColor(). Applying the legacy HERO color again would tint
		// the whole sample whenever any dispersive Glass path set mlDispersionUsed.
		// Keep the old behavior only for the single-wavelength path.
		if (engine->mlHeroEnabled && (engine->mlHeroWavelengthCount == 1)) {
			const Spectrum mlDispersionColor = GetMLDispersionSampleColor();
			for (u_int i = 0; i < sampleResults.size(); ++i) {
				for (u_int j = 0; j < engine->GetFilm().GetRadianceGroupCount(); ++j)
					sampleResults[i].radiance[j] = sampleResults[i].radiance[j] * mlDispersionColor;
			}
		}

		assert (SampleResult::IsAllValid(sampleResults));

		// Variance clamping
		if (varianceClamping.hasClamping()) {
			for(u_int i = 0; i < sampleResults.size(); ++i)
				varianceClamping.Clamp(engine->GetFilm(), sampleResults[i]);

			assert (SampleResult::IsAllValid(sampleResults));
		}

		sampler->NextSample(sampleResults);

		// Check halt conditions
		if (engine->GetFilm().GetConvergence() == 1.f)
			break;

		if (photonGICache) {
			const u_int spp = engine->GetFilm().GetTotalEyeSampleCount() / engine->GetFilm().GetPixelCount();
			photonGICache->Update(threadIndex, spp);
                }
	} // ~for

	const u_int finishedThreads15az = mlHero15azFinishedThreads.fetch_add(1u) + 1u;
	const u_int expectedThreads15az = (u_int)engine->renderThreads.size();

	// ML HERO phase 15cq: compact current-diagnostics summary.
	if ((finishedThreads15az == expectedThreads15az) && MLHeroCurrentDiagnosticsEnabled15cq()) {
		std::lock_guard<std::recursive_mutex> mlHeroCurrentSummaryLock15cq(mlHeroDebugLogMutex);
		FILE *f15cq = MLHeroOpenCurrentLog15cq("a");
		if (f15cq) {
			fprintf(f15cq, "------------------------------------------------------------\n");
			fprintf(f15cq, "ML HERO phase 15cr CURRENT DIAGNOSTICS SUMMARY heavy=%u\n", mlHeroHeavyDiagnostics15cq.load(std::memory_order_relaxed) ? 1u : 0u);
			fprintf(f15cq, "phase15cr columns: materialType candidate genericRGBReflSPD matteCompensated glossy2Split genericCompensated\n");
			for (u_int t15cq = 0u; t15cq < MLHERO_MATERIAL_TYPE_COUNT_15CL; ++t15cq) {
				const unsigned long long c = mlHero15clReflectCandidate[t15cq].load(std::memory_order_relaxed);
				const unsigned long long g = mlHero15clGenericRGBReflSPD[t15cq].load(std::memory_order_relaxed);
				const unsigned long long m = mlHero15clMatteCompensated[t15cq].load(std::memory_order_relaxed);
				const unsigned long long s2 = mlHero15clGlossy2Split[t15cq].load(std::memory_order_relaxed);
				const unsigned long long gc = mlHero15cnGenericCompensated[t15cq].load(std::memory_order_relaxed);
				if (c || g || m || s2 || gc)
					fprintf(f15cq, "phase15cr material=%s id=%u candidate=%llu genericRGBReflSPD=%llu matteCompensated=%llu glossy2Split=%llu genericCompensated=%llu\n", MLHeroMaterialTypeName15cl(t15cq), t15cq, c, g, m, s2, gc);
			}
			fprintf(f15cq, "phase15cr glossy2FullRenderCompensated=%llu\n", mlHero15cqGlossy2Events.load(std::memory_order_relaxed));
			if (MLHeroCurrentHeavyDiagnosticsEnabled15cq()) {
				const unsigned long long gn15cq = mlHero15cqGlossy2SampleCount.load(std::memory_order_relaxed);
				if (gn15cq) {
					const double gd15cq = (double)gn15cq;
					const double kdBefore = mlHero15cqGlossy2KdErrBeforeSum.load(std::memory_order_relaxed) / gd15cq;
					const double kdAfter = mlHero15cqGlossy2KdErrAfterSum.load(std::memory_order_relaxed) / gd15cq;
					const double fullBefore = mlHero15cqGlossy2FullErrBeforeSum.load(std::memory_order_relaxed) / gd15cq;
					const double fullAfter = mlHero15cqGlossy2FullErrAfterSum.load(std::memory_order_relaxed) / gd15cq;
					fprintf(f15cq, "ML HERO phase 15cr GLOSSY2 BASIS FULL-RENDER SUMMARY samples=%llu sampleBudget=%u\n", gn15cq, MLHERO_15CQ_GLOSSY2_SAMPLE_BUDGET);
					fprintf(f15cq, "phase15cr Kd meanErrBefore=%.9g meanErrAfter=%.9g meanReduction=%.6f%% worstBefore=%.9g worstAfter=%.9g\n", kdBefore, kdAfter, kdBefore > 0.0 ? (1.0-kdAfter/kdBefore)*100.0 : 0.0, mlHero15cqGlossy2KdErrBeforeMax.load(std::memory_order_relaxed), mlHero15cqGlossy2KdErrAfterMax.load(std::memory_order_relaxed));
					fprintf(f15cq, "phase15cr FullBSDF meanErrBefore=%.9g meanErrAfter=%.9g meanReduction=%.6f%% worstBefore=%.9g worstAfter=%.9g\n", fullBefore, fullAfter, fullBefore > 0.0 ? (1.0-fullAfter/fullBefore)*100.0 : 0.0, mlHero15cqGlossy2FullErrBeforeMax.load(std::memory_order_relaxed), mlHero15cqGlossy2FullErrAfterMax.load(std::memory_order_relaxed));
				}
				fprintf(f15cq, "ML HERO phase 15co GENERIC REFLECTANCE A/B MATERIAL SUMMARY sampleBudgetPerMaterial=%u\n", MLHERO_15CO_SAMPLE_BUDGET);
				for (u_int t15co = 0u; t15co < MLHERO_MATERIAL_TYPE_COUNT_15CL; ++t15co) {
					const unsigned long long n = mlHero15coSampleCount[t15co].load(std::memory_order_relaxed);
					if (!n) continue;
					const double before = mlHero15coErrBeforeSum[t15co].load(std::memory_order_relaxed) / (double)n;
					const double after = mlHero15coErrAfterSum[t15co].load(std::memory_order_relaxed) / (double)n;
					const double reduction = before > 0.0 ? (1.0 - after / before) * 100.0 : 0.0;
					fprintf(f15cq, "phase15co material=%s id=%u samples=%llu meanMaxErrBefore=%.9g meanMaxErrAfter=%.9g meanErrorReduction=%.6f%% worstErrBefore=%.9g worstErrAfter=%.9g\n", MLHeroMaterialTypeName15cl(t15co), t15co, n, before, after, reduction, mlHero15coErrBeforeMax[t15co].load(std::memory_order_relaxed), mlHero15coErrAfterMax[t15co].load(std::memory_order_relaxed));
				}
			} else {
				fprintf(f15cq, "phase15cr heavyDiagnostics=0 expensive_dense_and_multi_trial_diagnostics_skipped\n");
			}
			fprintf(f15cq, "phase15cr note=reflectance_fixes_unchanged; Glossy2_coating_Fresnel_residual_unchanged; diagnostics_cleanup_only\n");
			fprintf(f15cq, "ML HERO phase 15cr DIAGNOSTIC STATUS: mode=current master=1 current=1 legacy=%u heavy=%u physicsChanges=0\n", mlHeroLegacyDiagnostics15cq.load(std::memory_order_relaxed) ? 1u : 0u, mlHeroHeavyDiagnostics15cq.load(std::memory_order_relaxed) ? 1u : 0u);
			fclose(f15cq);
		}
	}

	if ((finishedThreads15az == expectedThreads15az) && MLHeroLegacyDiagnosticsEnabled15cq()) {
		std::lock_guard<std::recursive_mutex> mlHeroSummaryLock15az(mlHeroDebugLogMutex);
		FILE *f15az = MLHeroOpenLegacyLog15cq("a");
		if (f15az) {
			const MLHeroGlassCorrectionStats15az &st15az = mlHeroGlassCorrectionStats15az;
			const double laneDen15az = (st15az.evaluatedLanes > 0ull) ? (double)st15az.evaluatedLanes : 1.0;
			const double bounceDen15az = (st15az.glassBounces > 0ull) ? (double)st15az.glassBounces : 1.0;
			fprintf(f15az, "------------------------------------------------------------\n");
			fprintf(f15az, "ML HERO phase 15ce MATTE FULL-RENDER PACKET CONVERGENCE SUMMARY\n");
			{
				static const char *classNames15ce[MLHERO_MATTE_CLASS_COUNT_15CE] = {
					"NEUTRAL", "RED", "GREEN", "BLUE", "OTHER"
				};
				for (u_int c15ce = 0u; c15ce < MLHERO_MATTE_CLASS_COUNT_15CE; ++c15ce) {
					const unsigned long long n15ce = mlHero15ceCount[c15ce].load(std::memory_order_relaxed);
					if (!n15ce)
						continue;
					const double dn15ce = (double)n15ce;
					const double cr15ce = mlHero15ceClassicR[c15ce].load(std::memory_order_relaxed);
					const double cg15ce = mlHero15ceClassicG[c15ce].load(std::memory_order_relaxed);
					const double cb15ce = mlHero15ceClassicB[c15ce].load(std::memory_order_relaxed);
					const double pr15ce = mlHero15cePacketR[c15ce].load(std::memory_order_relaxed);
					const double pg15ce = mlHero15cePacketG[c15ce].load(std::memory_order_relaxed);
					const double pb15ce = mlHero15cePacketB[c15ce].load(std::memory_order_relaxed);
					const double classicY15ce = .2126 * cr15ce + .7152 * cg15ce + .0722 * cb15ce;
					const double packetY15ce = .2126 * pr15ce + .7152 * pg15ce + .0722 * pb15ce;
					fprintf(f15az,
						"phase15ce class=%s count=%llu classicMean=(%.12g,%.12g,%.12g) packetMean=(%.12g,%.12g,%.12g) meanErr=(%.12g,%.12g,%.12g) Yratio=%.12g\n",
						classNames15ce[c15ce], n15ce,
						cr15ce / dn15ce, cg15ce / dn15ce, cb15ce / dn15ce,
						pr15ce / dn15ce, pg15ce / dn15ce, pb15ce / dn15ce,
						(pr15ce - cr15ce) / dn15ce, (pg15ce - cg15ce) / dn15ce, (pb15ce - cb15ce) / dn15ce,
						(fabs(classicY15ce) > 1e-30) ? packetY15ce / classicY15ce : 0.0);
					fprintf(f15az,
						"phase15ce class=%s MAE=(%.12g,%.12g,%.12g) RMSE=(%.12g,%.12g,%.12g) aggregateRGBRatio=(%.12g,%.12g,%.12g)\n",
						classNames15ce[c15ce],
						mlHero15ceAbsErrR[c15ce].load(std::memory_order_relaxed) / dn15ce,
						mlHero15ceAbsErrG[c15ce].load(std::memory_order_relaxed) / dn15ce,
						mlHero15ceAbsErrB[c15ce].load(std::memory_order_relaxed) / dn15ce,
						sqrt(mlHero15ceSqErrR[c15ce].load(std::memory_order_relaxed) / dn15ce),
						sqrt(mlHero15ceSqErrG[c15ce].load(std::memory_order_relaxed) / dn15ce),
						sqrt(mlHero15ceSqErrB[c15ce].load(std::memory_order_relaxed) / dn15ce),
						(fabs(cr15ce) > 1e-30) ? pr15ce / cr15ce : 0.0,
						(fabs(cg15ce) > 1e-30) ? pg15ce / cg15ce : 0.0,
						(fabs(cb15ce) > 1e-30) ? pb15ce / cb15ce : 0.0);
				}
				fprintf(f15az, "phase15ce note=phase15cj_runtime_AB_may_change_Matte_render_when_enabled; classification_from_Kd neutral_if_range<=0.02 otherwise dominant_channel_if_1.25x; packetApplied_matches_active_Matte_basis_compensation_switch; meanErr_should_approach_zero_if_active_estimator_matches_classic_RGB\n");
			}

			// ML HERO phase 15cf: deterministic continuous-spectrum reference.
			// This is intentionally independent of the sampled HERO packet phases.
			fprintf(f15az, "ML HERO phase 15cf MATTE CONTINUOUS 1NM RECONSTRUCTION SUMMARY\n");
			{
				static const char *classNames15cf[MLHERO_MATTE_CLASS_COUNT_15CE] = {
					"NEUTRAL", "RED", "GREEN", "BLUE", "OTHER"
				};
				for (u_int c15cf = 0u; c15cf < MLHERO_MATTE_CLASS_COUNT_15CE; ++c15cf) {
					const unsigned long long n15cf = mlHero15ceCount[c15cf].load(std::memory_order_relaxed);
					if (!n15cf)
						continue;
					const double dn15cf = (double)n15cf;
					const Spectrum meanKd15cf(
						(float)(mlHero15cfKdR[c15cf].load(std::memory_order_relaxed) / dn15cf),
						(float)(mlHero15cfKdG[c15cf].load(std::memory_order_relaxed) / dn15cf),
						(float)(mlHero15cfKdB[c15cf].load(std::memory_order_relaxed) / dn15cf));
					const RGBColor meanKdRGB15cf(meanKd15cf.c[0], meanKd15cf.c[1], meanKd15cf.c[2]);
					const RGBReflSPD meanSPD15cf(meanKdRGB15cf);
					const Spectrum rawIntegral15cf = MLHeroIntegrateMatteReflectance15cf(meanSPD15cf, false);
					const Spectrum clampedIntegral15cf = MLHeroIntegrateMatteReflectance15cf(meanSPD15cf, true);
					const double inputY15cf = .2126 * meanKd15cf.c[0] + .7152 * meanKd15cf.c[1] + .0722 * meanKd15cf.c[2];
					const double rawY15cf = .2126 * rawIntegral15cf.c[0] + .7152 * rawIntegral15cf.c[1] + .0722 * rawIntegral15cf.c[2];
					const double clampY15cf = .2126 * clampedIntegral15cf.c[0] + .7152 * clampedIntegral15cf.c[1] + .0722 * clampedIntegral15cf.c[2];
					fprintf(f15az,
						"phase15cf class=%s count=%llu meanKd=(%.12g,%.12g,%.12g) raw1nm=(%.12g,%.12g,%.12g) clamped1nm=(%.12g,%.12g,%.12g)\n",
						classNames15cf[c15cf], n15cf,
						meanKd15cf.c[0], meanKd15cf.c[1], meanKd15cf.c[2],
						rawIntegral15cf.c[0], rawIntegral15cf.c[1], rawIntegral15cf.c[2],
						clampedIntegral15cf.c[0], clampedIntegral15cf.c[1], clampedIntegral15cf.c[2]);
					fprintf(f15az,
						"phase15cf class=%s rawRGBRatio=(%.12g,%.12g,%.12g) clampedRGBRatio=(%.12g,%.12g,%.12g) rawYratio=%.12g clampedYratio=%.12g\n",
						classNames15cf[c15cf],
						(fabs(meanKd15cf.c[0]) > 1e-20f) ? rawIntegral15cf.c[0] / meanKd15cf.c[0] : 0.f,
						(fabs(meanKd15cf.c[1]) > 1e-20f) ? rawIntegral15cf.c[1] / meanKd15cf.c[1] : 0.f,
						(fabs(meanKd15cf.c[2]) > 1e-20f) ? rawIntegral15cf.c[2] / meanKd15cf.c[2] : 0.f,
						(fabs(meanKd15cf.c[0]) > 1e-20f) ? clampedIntegral15cf.c[0] / meanKd15cf.c[0] : 0.f,
						(fabs(meanKd15cf.c[1]) > 1e-20f) ? clampedIntegral15cf.c[1] / meanKd15cf.c[1] : 0.f,
						(fabs(meanKd15cf.c[2]) > 1e-20f) ? clampedIntegral15cf.c[2] / meanKd15cf.c[2] : 0.f,
						(fabs(inputY15cf) > 1e-30) ? rawY15cf / inputY15cf : 0.0,
						(fabs(inputY15cf) > 1e-30) ? clampY15cf / inputY15cf : 0.0);
				}
				fprintf(f15az, "phase15cf note=diagnostic_only_render_unchanged; deterministic_1nm_trapezoidal_integral_380_780; raw_vs_Matte_nonnegative_clamp; same_RGBReflSPD_and_normalized_CIE_to_RGB_path_as_packet_diagnostic\n");
			}

			// ML HERO phase 15cg: XYZ / white-reference / matrix decomposition.
			fprintf(f15az, "ML HERO phase 15cg MATTE XYZ / WHITE / MATRIX DECOMPOSITION SUMMARY\n");
			{
				static const char *classNames15cg[MLHERO_MATTE_CLASS_COUNT_15CE] = {
					"NEUTRAL", "RED", "GREEN", "BLUE", "OTHER"
				};

				const Spectrum matrixX15cg = MLHeroXYZToRGB15cg(XYZColor(1.f, 0.f, 0.f));
				const Spectrum matrixY15cg = MLHeroXYZToRGB15cg(XYZColor(0.f, 1.f, 0.f));
				const Spectrum matrixZ15cg = MLHeroXYZToRGB15cg(XYZColor(0.f, 0.f, 1.f));
				fprintf(f15az,
					"phase15cg DefaultColorSystem_XYZ_to_RGB columns: X=(%.12g,%.12g,%.12g) Y=(%.12g,%.12g,%.12g) Z=(%.12g,%.12g,%.12g)\n",
					matrixX15cg.c[0], matrixX15cg.c[1], matrixX15cg.c[2],
					matrixY15cg.c[0], matrixY15cg.c[1], matrixY15cg.c[2],
					matrixZ15cg.c[0], matrixZ15cg.c[1], matrixZ15cg.c[2]);

				const RGBReflSPD whiteSPD15cg(RGBColor(1.f, 1.f, 1.f));
				const XYZColor whiteXYZRaw15cg = MLHeroIntegrateMatteReflectanceXYZ15cg(whiteSPD15cg, false);
				const XYZColor whiteXYZClamp15cg = MLHeroIntegrateMatteReflectanceXYZ15cg(whiteSPD15cg, true);
				const Spectrum whiteRGBRaw15cg = MLHeroXYZToRGB15cg(whiteXYZRaw15cg);
				const Spectrum whiteRGBClamp15cg = MLHeroXYZToRGB15cg(whiteXYZClamp15cg);
				fprintf(f15az,
					"phase15cg UNIT_WHITE rawXYZ=(%.12g,%.12g,%.12g) rawRGB=(%.12g,%.12g,%.12g) clampedXYZ=(%.12g,%.12g,%.12g) clampedRGB=(%.12g,%.12g,%.12g)\n",
					whiteXYZRaw15cg.c[0], whiteXYZRaw15cg.c[1], whiteXYZRaw15cg.c[2],
					whiteRGBRaw15cg.c[0], whiteRGBRaw15cg.c[1], whiteRGBRaw15cg.c[2],
					whiteXYZClamp15cg.c[0], whiteXYZClamp15cg.c[1], whiteXYZClamp15cg.c[2],
					whiteRGBClamp15cg.c[0], whiteRGBClamp15cg.c[1], whiteRGBClamp15cg.c[2]);

				for (u_int c15cg = 0u; c15cg < MLHERO_MATTE_CLASS_COUNT_15CE; ++c15cg) {
					const unsigned long long n15cg = mlHero15ceCount[c15cg].load(std::memory_order_relaxed);
					if (!n15cg)
						continue;
					const double dn15cg = (double)n15cg;
					const Spectrum meanKd15cg(
						(float)(mlHero15cfKdR[c15cg].load(std::memory_order_relaxed) / dn15cg),
						(float)(mlHero15cfKdG[c15cg].load(std::memory_order_relaxed) / dn15cg),
						(float)(mlHero15cfKdB[c15cg].load(std::memory_order_relaxed) / dn15cg));
					const RGBReflSPD spd15cg(RGBColor(meanKd15cg.c[0], meanKd15cg.c[1], meanKd15cg.c[2]));
					const XYZColor rawXYZ15cg = MLHeroIntegrateMatteReflectanceXYZ15cg(spd15cg, false);
					const XYZColor clampXYZ15cg = MLHeroIntegrateMatteReflectanceXYZ15cg(spd15cg, true);
					const Spectrum rawRGB15cg = MLHeroXYZToRGB15cg(rawXYZ15cg);
					const Spectrum clampRGB15cg = MLHeroXYZToRGB15cg(clampXYZ15cg);
					const double rawSumXYZ15cg = rawXYZ15cg.c[0] + rawXYZ15cg.c[1] + rawXYZ15cg.c[2];
					const double clampSumXYZ15cg = clampXYZ15cg.c[0] + clampXYZ15cg.c[1] + clampXYZ15cg.c[2];
					fprintf(f15az,
						"phase15cg class=%s meanKd=(%.12g,%.12g,%.12g) rawXYZ=(%.12g,%.12g,%.12g) raw_xy=(%.12g,%.12g) rawRGB=(%.12g,%.12g,%.12g)\n",
						classNames15cg[c15cg], meanKd15cg.c[0], meanKd15cg.c[1], meanKd15cg.c[2],
						rawXYZ15cg.c[0], rawXYZ15cg.c[1], rawXYZ15cg.c[2],
						(fabs(rawSumXYZ15cg) > 1e-30) ? rawXYZ15cg.c[0] / rawSumXYZ15cg : 0.0,
						(fabs(rawSumXYZ15cg) > 1e-30) ? rawXYZ15cg.c[1] / rawSumXYZ15cg : 0.0,
						rawRGB15cg.c[0], rawRGB15cg.c[1], rawRGB15cg.c[2]);
					fprintf(f15az,
						"phase15cg class=%s clampXYZ=(%.12g,%.12g,%.12g) clamp_xy=(%.12g,%.12g) clampRGB=(%.12g,%.12g,%.12g) deltaXYZ_clamp_minus_raw=(%.12g,%.12g,%.12g)\n",
						classNames15cg[c15cg],
						clampXYZ15cg.c[0], clampXYZ15cg.c[1], clampXYZ15cg.c[2],
						(fabs(clampSumXYZ15cg) > 1e-30) ? clampXYZ15cg.c[0] / clampSumXYZ15cg : 0.0,
						(fabs(clampSumXYZ15cg) > 1e-30) ? clampXYZ15cg.c[1] / clampSumXYZ15cg : 0.0,
						clampRGB15cg.c[0], clampRGB15cg.c[1], clampRGB15cg.c[2],
						clampXYZ15cg.c[0] - rawXYZ15cg.c[0],
						clampXYZ15cg.c[1] - rawXYZ15cg.c[1],
						clampXYZ15cg.c[2] - rawXYZ15cg.c[2]);
				}
				fprintf(f15az, "phase15cg note=diagnostic_only_render_unchanged; XYZ_logged_before_DefaultColorSystem_ToRGB; UNIT_WHITE_exposes_reflectance_whitepoint_response; matrix_columns_are_ToRGB_of_X_Y_Z_unit_axes; compare_raw_vs_clamped_to_isolate_nonnegative_clamp_effect\n");
			}

			// ML HERO phase 15ch: white-point remap counterfactual, diagnostic only.
			fprintf(f15az, "ML HERO phase 15ch MATTE WHITE-POINT REMAP DIAGNOSTIC SUMMARY\n");
			{
				static const char *classNames15ch[MLHERO_MATTE_CLASS_COUNT_15CE] = {
					"NEUTRAL", "RED", "GREEN", "BLUE", "OTHER"
				};
				const RGBReflSPD whiteSPD15ch(RGBColor(1.f, 1.f, 1.f));
				const XYZColor sourceWhite15ch = MLHeroIntegrateMatteReflectanceXYZ15cg(whiteSPD15ch, true);
				const XYZColor targetWhite15ch = MLHeroDefaultRGBWhiteXYZ15ch();
				const Spectrum sourceWhiteRGB15ch = MLHeroXYZToRGB15cg(sourceWhite15ch);
				const Spectrum targetWhiteRGB15ch = MLHeroXYZToRGB15cg(targetWhite15ch);
				const XYZColor remappedWhite15ch = MLHeroDiagonalWhiteRemap15ch(sourceWhite15ch, sourceWhite15ch, targetWhite15ch);
				const Spectrum remappedWhiteRGB15ch = MLHeroXYZToRGB15cg(remappedWhite15ch);
				fprintf(f15az, "phase15ch sourceWhiteXYZ=(%.12g,%.12g,%.12g) sourceWhiteRGB=(%.12g,%.12g,%.12g) targetWhiteXYZ=(%.12g,%.12g,%.12g) targetWhiteRGB=(%.12g,%.12g,%.12g) remappedWhiteRGB=(%.12g,%.12g,%.12g)\n",
					sourceWhite15ch.c[0],sourceWhite15ch.c[1],sourceWhite15ch.c[2], sourceWhiteRGB15ch.c[0],sourceWhiteRGB15ch.c[1],sourceWhiteRGB15ch.c[2],
					targetWhite15ch.c[0],targetWhite15ch.c[1],targetWhite15ch.c[2], targetWhiteRGB15ch.c[0],targetWhiteRGB15ch.c[1],targetWhiteRGB15ch.c[2],
					remappedWhiteRGB15ch.c[0],remappedWhiteRGB15ch.c[1],remappedWhiteRGB15ch.c[2]);

				for (u_int c15ch=0u;c15ch<MLHERO_MATTE_CLASS_COUNT_15CE;++c15ch) {
					const unsigned long long n15ch=mlHero15ceCount[c15ch].load(std::memory_order_relaxed);
					if (!n15ch) continue;
					const double dn15ch=(double)n15ch;
					const Spectrum meanKd15ch((float)(mlHero15cfKdR[c15ch].load(std::memory_order_relaxed)/dn15ch),
						(float)(mlHero15cfKdG[c15ch].load(std::memory_order_relaxed)/dn15ch),
						(float)(mlHero15cfKdB[c15ch].load(std::memory_order_relaxed)/dn15ch));
					const RGBReflSPD spd15ch(RGBColor(meanKd15ch.c[0],meanKd15ch.c[1],meanKd15ch.c[2]));
					const XYZColor currentXYZ15ch=MLHeroIntegrateMatteReflectanceXYZ15cg(spd15ch,true);
					const Spectrum currentRGB15ch=MLHeroXYZToRGB15cg(currentXYZ15ch);
					const XYZColor adaptedXYZ15ch=MLHeroDiagonalWhiteRemap15ch(currentXYZ15ch,sourceWhite15ch,targetWhite15ch);
					const Spectrum adaptedRGB15ch=MLHeroXYZToRGB15cg(adaptedXYZ15ch);
					const double inputY15ch=.2126*meanKd15ch.c[0]+.7152*meanKd15ch.c[1]+.0722*meanKd15ch.c[2];
					const double currentY15ch=.2126*currentRGB15ch.c[0]+.7152*currentRGB15ch.c[1]+.0722*currentRGB15ch.c[2];
					const double adaptedY15ch=.2126*adaptedRGB15ch.c[0]+.7152*adaptedRGB15ch.c[1]+.0722*adaptedRGB15ch.c[2];
					fprintf(f15az, "phase15ch class=%s meanKd=(%.12g,%.12g,%.12g) currentRGB=(%.12g,%.12g,%.12g) adaptedXYZ=(%.12g,%.12g,%.12g) adaptedRGB=(%.12g,%.12g,%.12g) currentYratio=%.12g adaptedYratio=%.12g\n",
						classNames15ch[c15ch],meanKd15ch.c[0],meanKd15ch.c[1],meanKd15ch.c[2], currentRGB15ch.c[0],currentRGB15ch.c[1],currentRGB15ch.c[2],
						adaptedXYZ15ch.c[0],adaptedXYZ15ch.c[1],adaptedXYZ15ch.c[2], adaptedRGB15ch.c[0],adaptedRGB15ch.c[1],adaptedRGB15ch.c[2],
						(fabs(inputY15ch)>1e-30)?currentY15ch/inputY15ch:0.0, (fabs(inputY15ch)>1e-30)?adaptedY15ch/inputY15ch:0.0);
				}
				fprintf(f15az, "phase15ch note=diagnostic_only_render_unchanged; targetWhiteXYZ_is_inverse_DefaultColorSystem_matrix_times_RGB_111; adaptedXYZ_uses_simple_diagonal_white_remap_source_equal_energy_to_target; this_is_a_counterfactual_not_yet_an_applied_renderer_change\n");
			}

			// ML HERO phase 15ci: effective spectral RGB basis matrix + inverse test.
			fprintf(f15az, "ML HERO phase 15ci MATTE SPECTRAL RGB BASIS-MATRIX DIAGNOSTIC SUMMARY\n");
			{
				const Spectrum unitR15ci(1.f,0.f,0.f), unitG15ci(0.f,1.f,0.f), unitB15ci(0.f,0.f,1.f);
				const Spectrum rawR15ci=MLHeroContinuousMatteRGB15ci(unitR15ci,false);
				const Spectrum rawG15ci=MLHeroContinuousMatteRGB15ci(unitG15ci,false);
				const Spectrum rawB15ci=MLHeroContinuousMatteRGB15ci(unitB15ci,false);
				const Spectrum clampR15ci=MLHeroContinuousMatteRGB15ci(unitR15ci,true);
				const Spectrum clampG15ci=MLHeroContinuousMatteRGB15ci(unitG15ci,true);
				const Spectrum clampB15ci=MLHeroContinuousMatteRGB15ci(unitB15ci,true);
				Spectrum invRawR15ci,invRawG15ci,invRawB15ci,invClampR15ci,invClampG15ci,invClampB15ci;
				double detRaw15ci=0.0,detClamp15ci=0.0;
				const bool rawInvertible15ci=MLHeroInvertRGBMatrix15ci(rawR15ci,rawG15ci,rawB15ci,&invRawR15ci,&invRawG15ci,&invRawB15ci,&detRaw15ci);
				const bool clampInvertible15ci=MLHeroInvertRGBMatrix15ci(clampR15ci,clampG15ci,clampB15ci,&invClampR15ci,&invClampG15ci,&invClampB15ci,&detClamp15ci);
				fprintf(f15az,"phase15ci RAW_basis columns R=(%.12g,%.12g,%.12g) G=(%.12g,%.12g,%.12g) B=(%.12g,%.12g,%.12g) det=%.12g invertible=%u\n",
					rawR15ci.c[0],rawR15ci.c[1],rawR15ci.c[2],rawG15ci.c[0],rawG15ci.c[1],rawG15ci.c[2],rawB15ci.c[0],rawB15ci.c[1],rawB15ci.c[2],detRaw15ci,rawInvertible15ci?1u:0u);
				fprintf(f15az,"phase15ci CLAMP_basis columns R=(%.12g,%.12g,%.12g) G=(%.12g,%.12g,%.12g) B=(%.12g,%.12g,%.12g) det=%.12g invertible=%u\n",
					clampR15ci.c[0],clampR15ci.c[1],clampR15ci.c[2],clampG15ci.c[0],clampG15ci.c[1],clampG15ci.c[2],clampB15ci.c[0],clampB15ci.c[1],clampB15ci.c[2],detClamp15ci,clampInvertible15ci?1u:0u);
				if (rawInvertible15ci) fprintf(f15az,"phase15ci RAW_inverse columns R=(%.12g,%.12g,%.12g) G=(%.12g,%.12g,%.12g) B=(%.12g,%.12g,%.12g)\n",invRawR15ci.c[0],invRawR15ci.c[1],invRawR15ci.c[2],invRawG15ci.c[0],invRawG15ci.c[1],invRawG15ci.c[2],invRawB15ci.c[0],invRawB15ci.c[1],invRawB15ci.c[2]);
				if (clampInvertible15ci) fprintf(f15az,"phase15ci CLAMP_inverse columns R=(%.12g,%.12g,%.12g) G=(%.12g,%.12g,%.12g) B=(%.12g,%.12g,%.12g)\n",invClampR15ci.c[0],invClampR15ci.c[1],invClampR15ci.c[2],invClampG15ci.c[0],invClampG15ci.c[1],invClampG15ci.c[2],invClampB15ci.c[0],invClampB15ci.c[1],invClampB15ci.c[2]);

				struct MLHero15ciProbe { const char *name; Spectrum rgb; };
				const MLHero15ciProbe probes15ci[] = {
					{"WHITE",Spectrum(1.f,1.f,1.f)}, {"RED",Spectrum(1.f,0.f,0.f)}, {"GREEN",Spectrum(0.f,1.f,0.f)}, {"BLUE",Spectrum(0.f,0.f,1.f)},
					{"YELLOW",Spectrum(1.f,1.f,0.f)}, {"CYAN",Spectrum(0.f,1.f,1.f)}, {"MAGENTA",Spectrum(1.f,0.f,1.f)},
					{"WARM",Spectrum(.8f,.35f,.1f)}, {"COOL",Spectrum(.1f,.35f,.8f)}, {"MID",Spectrum(.35f,.55f,.2f)}
				};
				for (u_int p15ci=0u;p15ci<sizeof(probes15ci)/sizeof(probes15ci[0]);++p15ci) {
					const Spectrum in15ci=probes15ci[p15ci].rgb;
					const Spectrum actualRaw15ci=MLHeroContinuousMatteRGB15ci(in15ci,false);
					const Spectrum actualClamp15ci=MLHeroContinuousMatteRGB15ci(in15ci,true);
					const Spectrum predictedRaw15ci=MLHeroApplyRGBMatrix15ci(rawR15ci,rawG15ci,rawB15ci,in15ci);
					const Spectrum predictedClamp15ci=MLHeroApplyRGBMatrix15ci(clampR15ci,clampG15ci,clampB15ci,in15ci);
					const Spectrum correctedRaw15ci=rawInvertible15ci?MLHeroApplyRGBMatrix15ci(invRawR15ci,invRawG15ci,invRawB15ci,actualRaw15ci):Spectrum();
					const Spectrum correctedClamp15ci=clampInvertible15ci?MLHeroApplyRGBMatrix15ci(invClampR15ci,invClampG15ci,invClampB15ci,actualClamp15ci):Spectrum();
					fprintf(f15az,"phase15ci probe=%s input=(%.12g,%.12g,%.12g) rawActual=(%.12g,%.12g,%.12g) rawLinearPred=(%.12g,%.12g,%.12g) rawPredResidual=(%.12g,%.12g,%.12g) rawInverseRecovered=(%.12g,%.12g,%.12g)\n",
						probes15ci[p15ci].name,in15ci.c[0],in15ci.c[1],in15ci.c[2],actualRaw15ci.c[0],actualRaw15ci.c[1],actualRaw15ci.c[2],predictedRaw15ci.c[0],predictedRaw15ci.c[1],predictedRaw15ci.c[2],actualRaw15ci.c[0]-predictedRaw15ci.c[0],actualRaw15ci.c[1]-predictedRaw15ci.c[1],actualRaw15ci.c[2]-predictedRaw15ci.c[2],correctedRaw15ci.c[0],correctedRaw15ci.c[1],correctedRaw15ci.c[2]);
					fprintf(f15az,"phase15ci probe=%s clampActual=(%.12g,%.12g,%.12g) clampLinearPred=(%.12g,%.12g,%.12g) clampPredResidual=(%.12g,%.12g,%.12g) clampInverseRecovered=(%.12g,%.12g,%.12g)\n",
						probes15ci[p15ci].name,actualClamp15ci.c[0],actualClamp15ci.c[1],actualClamp15ci.c[2],predictedClamp15ci.c[0],predictedClamp15ci.c[1],predictedClamp15ci.c[2],actualClamp15ci.c[0]-predictedClamp15ci.c[0],actualClamp15ci.c[1]-predictedClamp15ci.c[1],actualClamp15ci.c[2]-predictedClamp15ci.c[2],correctedClamp15ci.c[0],correctedClamp15ci.c[1],correctedClamp15ci.c[2]);
				}
				fprintf(f15az,"phase15ci note=diagnostic_only_render_unchanged; unit_primary_reconstructions_form_effective_inputRGB_to_reconstructedRGB_matrix; inverseRecovered_tests_single_global_3x3_correction; mixed_probe_linearPred_residual_exposes_piecewise_RGBReflSPD_or_nonnegative_clamp_nonlinearity\n");
				fprintf(f15az,"ML HERO phase 15cj MATTE BASIS COMPENSATION RUNTIME A/B: enabled=%u basis=phase15ci_CLAMP_inverse implementation=linear_unit_primary_SPD_combination_then_nonnegative_clamp note=render_changes_only_Matte_HERO_lane_reflectance_when_enabled\n", engine->mlHeroMatteBasisCompensation ? 1u : 0u);
			}
			fprintf(f15az, "ML HERO phase 15cl RGBReflSPD SHADER-USAGE SUMMARY\n");
			fprintf(f15az, "phase15cl columns: materialType candidateNonSpecularChromatic actualGenericRGBReflSPD matteCompensated glossy2Split generic15coCompensated\n");
			unsigned long long totalCandidate15cl = 0ull, totalGeneric15cl = 0ull, totalMatteComp15cl = 0ull, totalGlossySplit15cl = 0ull;
			for (u_int t15cl = 0u; t15cl < MLHERO_MATERIAL_TYPE_COUNT_15CL; ++t15cl) {
				const unsigned long long c15cl = mlHero15clReflectCandidate[t15cl].load(std::memory_order_relaxed);
				const unsigned long long g15cl = mlHero15clGenericRGBReflSPD[t15cl].load(std::memory_order_relaxed);
				const unsigned long long m15cl = mlHero15clMatteCompensated[t15cl].load(std::memory_order_relaxed);
				const unsigned long long s15cl = mlHero15clGlossy2Split[t15cl].load(std::memory_order_relaxed);
				const unsigned long long c15cn = mlHero15cnGenericCompensated[t15cl].load(std::memory_order_relaxed);
				if (!(c15cl || g15cl || m15cl || s15cl))
					continue;
				fprintf(f15az, "phase15cl material=%s id=%u candidate=%llu genericRGBReflSPD=%llu matteCompensated=%llu glossy2Split=%llu generic15coCompensated=%llu\n",
					MLHeroMaterialTypeName15cl(t15cl), t15cl, c15cl, g15cl, m15cl, s15cl, c15cn);
				totalCandidate15cl += c15cl; totalGeneric15cl += g15cl; totalMatteComp15cl += m15cl; totalGlossySplit15cl += s15cl;
			}
			fprintf(f15az, "phase15cl totals candidate=%llu genericRGBReflSPD=%llu matteCompensated=%llu glossy2Split=%llu\n",
				totalCandidate15cl, totalGeneric15cl, totalMatteComp15cl, totalGlossySplit15cl);
			fprintf(f15az, "phase15cl interpretation: genericRGBReflSPD>0 means this material actually used the same generic RGB-to-reflectance-SPD path under BIDIR; matteCompensated is HERO_76 correction use; glossy2Split is the dedicated Glossy2 Kd/coating split and therefore did not use whole-BSDF generic fallback for those events. generic15coCompensated counts the runtime correction when enabled\n");

			fprintf(f15az, "ML HERO phase 15co GENERIC REFLECTANCE A/B MATERIAL SUMMARY sampleBudgetPerMaterial=%u\n", MLHERO_15CO_SAMPLE_BUDGET);
			for (u_int t15co = 0u; t15co < MLHERO_MATERIAL_TYPE_COUNT_15CL; ++t15co) {
				const unsigned long long n15co = mlHero15coSampleCount[t15co].load(std::memory_order_relaxed);
				if (!n15co) continue;
				const double denom15co = (double)n15co;
				const double meanBefore15co = mlHero15coErrBeforeSum[t15co].load(std::memory_order_relaxed) / denom15co;
				const double meanAfter15co = mlHero15coErrAfterSum[t15co].load(std::memory_order_relaxed) / denom15co;
				const double maxBefore15co = mlHero15coErrBeforeMax[t15co].load(std::memory_order_relaxed);
				const double maxAfter15co = mlHero15coErrAfterMax[t15co].load(std::memory_order_relaxed);
				const double meanReduction15co = (meanBefore15co > 0.0) ? (1.0 - meanAfter15co / meanBefore15co) * 100.0 : 0.0;
				fprintf(f15az, "phase15co material=%s id=%u samples=%llu fullRenderCompensated=%llu meanMaxErrBefore=%.9g meanMaxErrAfter=%.9g meanErrorReduction=%.6f%% worstErrBefore=%.9g worstErrAfter=%.9g\n",
					MLHeroMaterialTypeName15cl(t15co), t15co, n15co,
					mlHero15cnGenericCompensated[t15co].load(std::memory_order_relaxed),
					meanBefore15co, meanAfter15co, meanReduction15co, maxBefore15co, maxAfter15co);
			}
			fprintf(f15az, "phase15co note=fullRenderCompensated_is_exact_event_count; A/B error statistics use the first deterministic bounded sample set per material to preserve Fast-build performance\n");

			if (!MLHeroLegacyHeavyDiagnosticsEnabled15cq()) {
				fprintf(f15az, "ML HERO phase 15az FULL RENDER GLASS CORRECTION SUMMARY: legacy heavy diagnostic disabled (enable Heavy Diagnostics)\n");
			} else {
				fprintf(f15az, "ML HERO phase 15az FULL RENDER GLASS CORRECTION SUMMARY\n");
			fprintf(f15az, "phase15az totals: glassBounces=%llu evaluatedLanes=%llu\n",
				st15az.glassBounces, st15az.evaluatedLanes);
			fprintf(f15az,
				"phase15az laneThresholds: absCorrectionDelta>5%%=%llu (%.6f%%) >10%%=%llu (%.6f%%) >25%%=%llu (%.6f%%)\n",
				st15az.laneOver5Pct, 100.0 * (double)st15az.laneOver5Pct / laneDen15az,
				st15az.laneOver10Pct, 100.0 * (double)st15az.laneOver10Pct / laneDen15az,
				st15az.laneOver25Pct, 100.0 * (double)st15az.laneOver25Pct / laneDen15az);
			fprintf(f15az,
				"phase15az bounceThresholds: anyLane>5%%=%llu (%.6f%%) anyLane>10%%=%llu (%.6f%%) anyLane>25%%=%llu (%.6f%%)\n",
				st15az.bounceOver5Pct, 100.0 * (double)st15az.bounceOver5Pct / bounceDen15az,
				st15az.bounceOver10Pct, 100.0 * (double)st15az.bounceOver10Pct / bounceDen15az,
				st15az.bounceOver25Pct, 100.0 * (double)st15az.bounceOver25Pct / bounceDen15az);
			fprintf(f15az, "phase15az correctionRange: min=%.9g max=%.9g maxAbsDeltaFrom1=%.9g\n",
				st15az.minCorrection, st15az.maxCorrection, st15az.maxAbsDelta);
			fprintf(f15az, "phase15az strongest: correction=%.9g absDelta=%.9g lambda=%.9g depth=%u case=%s\n",
				st15az.extremeCorrection, st15az.maxAbsDelta, st15az.extremeLambda,
				st15az.extremeDepth, MLHeroGlassCaseName15az(st15az.extremeCase));
			fprintf(f15az, "phase15az cases: EYE_TRANSMIT=%llu LIGHT_TRANSMIT=%llu EYE_REFLECT=%llu LIGHT_REFLECT=%llu\n",
				st15az.caseBounces[0], st15az.caseBounces[1], st15az.caseBounces[2], st15az.caseBounces[3]);
			fprintf(f15az,
				"phase15ba tirAndPdf: tirLanes=%llu nearTIR_sinT2>=0.95=%llu nearTIR_sinT2>=0.99=%llu laneEventPdfMismatch>20%%=%llu\n",
				st15az.tirLanes, st15az.nearTir95Lanes, st15az.nearTir99Lanes,
				st15az.pdfMismatch20PctLanes);
			fprintf(f15az,
				"phase15bb pathSupport: baseHeroTirMismatch=%llu heroLaneTirMismatch=%llu baseLaneTirMismatch=%llu allThreeTirAgree=%llu\n",
				st15az.baseHeroTirMismatchLanes, st15az.heroLaneTirMismatchLanes,
				st15az.baseLaneTirMismatchLanes, st15az.allThreeTirAgreeLanes);
			const double nearHero99Den15bc = (st15az.nearHero99TransmitLanes > 0ull) ? (double)st15az.nearHero99TransmitLanes : 1.0;
			fprintf(f15az,
				"phase15bc nearHero99: lanes=%llu rawAbsDelta>25%%=%llu (%.6f%%) lanePdfAdjusted>25%%=%llu (%.6f%%) spectralMixAdjusted>25%%=%llu (%.6f%%)\n",
				st15az.nearHero99TransmitLanes,
				st15az.rawOver25NearHero99, 100.0 * (double)st15az.rawOver25NearHero99 / nearHero99Den15bc,
				st15az.lanePdfAdjustedOver25NearHero99, 100.0 * (double)st15az.lanePdfAdjustedOver25NearHero99 / nearHero99Den15bc,
				st15az.spectralMixAdjustedOver25NearHero99, 100.0 * (double)st15az.spectralMixAdjustedOver25NearHero99 / nearHero99Den15bc);
			fprintf(f15az,
				"phase15bc nearHero99Max: raw=%.9g lanePdfAdjusted=%.9g spectralMixAdjusted=%.9g\n",
				st15az.maxRawCorrectionNearHero99, st15az.maxLanePdfAdjustedCorrectionNearHero99,
				st15az.maxSpectralMixAdjustedCorrectionNearHero99);
			const double eyeNearHero99Den15bd = (st15az.eyeTransmitNearHero99Lanes > 0ull) ?
				(double)st15az.eyeTransmitNearHero99Lanes : 1.0;
			const double eyeNearHero99SupportedDen15bd = (st15az.eyeTransmitNearHero99SupportMatch > 0ull) ?
				(double)st15az.eyeTransmitNearHero99SupportMatch : 1.0;
			fprintf(f15az,
				"phase15bd EYE_TRANSMIT nearHero99Support: lanes=%llu supportMatch=%llu (%.6f%%) supportMismatch=%llu (%.6f%%) "
				"supportedLanePdfAdjusted>25%%=%llu (%.6f%%) maxSupportedLanePdfAdjusted=%.9g\n",
				st15az.eyeTransmitNearHero99Lanes,
				st15az.eyeTransmitNearHero99SupportMatch, 100.0 * (double)st15az.eyeTransmitNearHero99SupportMatch / eyeNearHero99Den15bd,
				st15az.eyeTransmitNearHero99SupportMismatch, 100.0 * (double)st15az.eyeTransmitNearHero99SupportMismatch / eyeNearHero99Den15bd,
				st15az.eyeTransmitNearHero99LanePdfOver25Supported,
				100.0 * (double)st15az.eyeTransmitNearHero99LanePdfOver25Supported / eyeNearHero99SupportedDen15bd,
				st15az.maxEyeTransmitNearHero99LanePdfSupported);
			fprintf(f15az,
				"phase15bd transmitSupportMismatchTotals: EYE_TRANSMIT=%llu LIGHT_TRANSMIT=%llu\n",
				st15az.eyeTransmitSupportMismatchTotal, st15az.lightTransmitSupportMismatchTotal);
			}
			const double heroOnlyDen15bf = (st15az.heroOnlyTerminations > 0ull) ? (double)st15az.heroOnlyTerminations : 1.0;
			if (st15az.heroOnlyTerminations > 0ull) {
				fprintf(f15az,
					"phase15bf heroOnlyEnergy: terminations=%llu eye=%llu light=%llu meanBrightnessRatio=%.9g minBrightnessRatio=%.9g maxBrightnessRatio=%.9g "
					"meanIdealScale=%.9g minIdealScale=%.9g maxIdealScale=%.9g meanPacketAverage=%.9g meanHeroScalar=%.9g\n",
					st15az.heroOnlyTerminations, st15az.heroOnlyEyeTerminations, st15az.heroOnlyLightTerminations,
					st15az.heroOnlyBrightnessRatioSum / heroOnlyDen15bf,
					st15az.heroOnlyBrightnessRatioMin, st15az.heroOnlyBrightnessRatioMax,
					st15az.heroOnlyIdealScaleSum / heroOnlyDen15bf,
					st15az.heroOnlyIdealScaleMin, st15az.heroOnlyIdealScaleMax,
					st15az.heroOnlyPacketAverageSum / heroOnlyDen15bf,
					st15az.heroOnlyHeroLaneSum / heroOnlyDen15bf);
				fprintf(f15az,
					"phase15bf heroOnlyThresholds: brightnessRatio>1.05=%llu (%.6f%%) >1.25=%llu (%.6f%%) >1.50=%llu (%.6f%%) <0.95=%llu (%.6f%%) currentPacketScale=%u\n",
					st15az.heroOnlyRatioOver105, 100.0 * (double)st15az.heroOnlyRatioOver105 / heroOnlyDen15bf,
					st15az.heroOnlyRatioOver125, 100.0 * (double)st15az.heroOnlyRatioOver125 / heroOnlyDen15bf,
					st15az.heroOnlyRatioOver150, 100.0 * (double)st15az.heroOnlyRatioOver150 / heroOnlyDen15bf,
					st15az.heroOnlyRatioUnder095, 100.0 * (double)st15az.heroOnlyRatioUnder095 / heroOnlyDen15bf,
					engine->mlHeroWavelengthCount);
			}
			const unsigned long long pdfTerms15bh = mlHero15bhPdfTerminations.load(std::memory_order_relaxed);
			const unsigned long long vis15bh = mlHero15bhVisibleConnections.load(std::memory_order_relaxed);
			fprintf(f15az,
				"phase15bh wavelengthPdf: terminations=%llu eye=%llu light=%llu packetCount=%u baseUniformPdf=%.9g terminatedUniformPdf=%.9g pdfEstimatorScale=%.9g note=diagnostic_metadata_only_current_xN_throughput_still_active\n",
				pdfTerms15bh,
				mlHero15bhPdfEyeTerminations.load(std::memory_order_relaxed),
				mlHero15bhPdfLightTerminations.load(std::memory_order_relaxed),
				engine->mlHeroWavelengthCount, 1.f / 400.f,
				(engine->mlHeroWavelengthCount > 0) ? (1.f / (400.f * (float)engine->mlHeroWavelengthCount)) : 0.f,
				(float)engine->mlHeroWavelengthCount);
			fprintf(f15az,
				"phase15bh bidirTerminationState: visibleConnections=%llu eyeTerminated=%llu (%.6f%%) lightTerminated=%llu (%.6f%%) bothTerminated=%llu (%.6f%%) neitherTerminated=%llu (%.6f%%)\n",
				vis15bh,
				mlHero15bhVisibleEyeTerminated.load(std::memory_order_relaxed),
				(vis15bh > 0ull) ? 100.0 * (double)mlHero15bhVisibleEyeTerminated.load(std::memory_order_relaxed) / (double)vis15bh : 0.0,
				mlHero15bhVisibleLightTerminated.load(std::memory_order_relaxed),
				(vis15bh > 0ull) ? 100.0 * (double)mlHero15bhVisibleLightTerminated.load(std::memory_order_relaxed) / (double)vis15bh : 0.0,
				mlHero15bhVisibleBothTerminated.load(std::memory_order_relaxed),
				(vis15bh > 0ull) ? 100.0 * (double)mlHero15bhVisibleBothTerminated.load(std::memory_order_relaxed) / (double)vis15bh : 0.0,
				mlHero15bhVisibleNeitherTerminated.load(std::memory_order_relaxed),
				(vis15bh > 0ull) ? 100.0 * (double)mlHero15bhVisibleNeitherTerminated.load(std::memory_order_relaxed) / (double)vis15bh : 0.0);
			// Phase 15bi: actual ConnectVertices energy split by termination state.
			const char *catName15bi[4] = {"neither", "eyeOnly", "lightOnly", "both"};
			double totalR15bi = 0.0, totalG15bi = 0.0, totalB15bi = 0.0;
			double totalY15bi = 0.0, totalAbsY15bi = 0.0;
			for (u_int c15bi = 0u; c15bi < 4u; ++c15bi) {
				totalR15bi += mlHero15biConnectR[c15bi].load(std::memory_order_relaxed);
				totalG15bi += mlHero15biConnectG[c15bi].load(std::memory_order_relaxed);
				totalB15bi += mlHero15biConnectB[c15bi].load(std::memory_order_relaxed);
				totalY15bi += mlHero15biConnectY[c15bi].load(std::memory_order_relaxed);
				totalAbsY15bi += mlHero15biConnectAbsY[c15bi].load(std::memory_order_relaxed);
			}
			fprintf(f15az,
				"phase15bi connectContributionTotals: R=%.12g G=%.12g B=%.12g Y=%.12g absY=%.12g note=actual_ConnectVertices_contributions_only\n",
				totalR15bi, totalG15bi, totalB15bi, totalY15bi, totalAbsY15bi);
			for (u_int c15bi = 0u; c15bi < 4u; ++c15bi) {
				const double r15bi = mlHero15biConnectR[c15bi].load(std::memory_order_relaxed);
				const double g15bi = mlHero15biConnectG[c15bi].load(std::memory_order_relaxed);
				const double b15bi = mlHero15biConnectB[c15bi].load(std::memory_order_relaxed);
				const double y15bi = mlHero15biConnectY[c15bi].load(std::memory_order_relaxed);
				const double absY15bi = mlHero15biConnectAbsY[c15bi].load(std::memory_order_relaxed);
				fprintf(f15az,
					"phase15bi category=%s contributions=%llu R=%.12g G=%.12g B=%.12g Y=%.12g absY=%.12g absYShare=%.6f%%\n",
					catName15bi[c15bi],
					mlHero15biConnectContributionCount[c15bi].load(std::memory_order_relaxed),
					r15bi, g15bi, b15bi, y15bi, absY15bi,
					(totalAbsY15bi > 0.0) ? 100.0 * absY15bi / totalAbsY15bi : 0.0);
			}
			const double bothR15bi = mlHero15biConnectR[3].load(std::memory_order_relaxed);
			const double bothG15bi = mlHero15biConnectG[3].load(std::memory_order_relaxed);
			const double bothB15bi = mlHero15biConnectB[3].load(std::memory_order_relaxed);
			const double bothY15bi = mlHero15biConnectY[3].load(std::memory_order_relaxed);
			const double bothAbsY15bi = mlHero15biConnectAbsY[3].load(std::memory_order_relaxed);
			const double packetScale15bi = (engine->mlHeroWavelengthCount > 0u) ? (double)engine->mlHeroWavelengthCount : 1.0;
			const double candidateR15bi = totalR15bi - bothR15bi + bothR15bi / packetScale15bi;
			const double candidateG15bi = totalG15bi - bothG15bi + bothG15bi / packetScale15bi;
			const double candidateB15bi = totalB15bi - bothB15bi + bothB15bi / packetScale15bi;
			const double candidateY15bi = totalY15bi - bothY15bi + bothY15bi / packetScale15bi;
			const double candidateAbsY15bi = totalAbsY15bi - bothAbsY15bi + bothAbsY15bi / packetScale15bi;
			fprintf(f15az,
				"phase15bi bothTerminationCandidate: packetScale=%.9g currentBothY=%.12g candidateBothY=%.12g currentTotalY=%.12g candidateTotalY=%.12g totalYRatio=%.9g currentTotalAbsY=%.12g candidateTotalAbsY=%.12g totalAbsYRatio=%.9g candidateRGB=(%.12g,%.12g,%.12g) note=diagnostic_only_both_contribution_divided_by_packetCount\n",
				packetScale15bi, bothY15bi, bothY15bi / packetScale15bi,
				totalY15bi, candidateY15bi, (fabs(totalY15bi) > 1e-30) ? candidateY15bi / totalY15bi : 0.0,
				totalAbsY15bi, candidateAbsY15bi, (totalAbsY15bi > 1e-30) ? candidateAbsY15bi / totalAbsY15bi : 0.0,
				candidateR15bi, candidateG15bi, candidateB15bi);


            // Phase 15by: focused dual HERO-only ConnectVertices diagnostic.
            const char *catName15by[4] = {"neither", "eyeOnly", "lightOnly", "both"};
            double totalAbsY15by = 0.0;
            double totalCandidateAbsY15by = 0.0;
            for (u_int c15by = 0u; c15by < 4u; ++c15by) {
                totalAbsY15by += mlHero15byAbsY[c15by].load(std::memory_order_relaxed);
                totalCandidateAbsY15by += mlHero15byCandidateAbsY[c15by].load(std::memory_order_relaxed);
            }
            fprintf(f15az,
                "phase15by DUAL_SUMMARY packetCount=%u totalAbsY=%.12g candidateTotalAbsY=%.12g candidateOverActual=%.9g note=diagnostic_only_render_unchanged\n",
                engine->mlHeroWavelengthCount, totalAbsY15by, totalCandidateAbsY15by,
                (totalAbsY15by > 1e-30) ? totalCandidateAbsY15by / totalAbsY15by : 0.0);
            for (u_int c15by = 0u; c15by < 4u; ++c15by) {
                const unsigned long long count15by = mlHero15byCount[c15by].load(std::memory_order_relaxed);
                const double den15by = (count15by > 0ull) ? (double)count15by : 1.0;
                const double absY15by = mlHero15byAbsY[c15by].load(std::memory_order_relaxed);
                const double candAbsY15by = mlHero15byCandidateAbsY[c15by].load(std::memory_order_relaxed);
                fprintf(f15az,
                    "phase15by category=%s count=%llu packetDivided=%llu Y=%.12g absY=%.12g absYShare=%.6f%% "
                    "candidateY=%.12g candidateAbsY=%.12g candidateOverActualAbsY=%.9g "
                    "meanEyeScale=%.9g meanLightScale=%.9g meanNaiveNetScale=%.9g\n",
                    catName15by[c15by], count15by,
                    mlHero15byPacketDividedCount[c15by].load(std::memory_order_relaxed),
                    mlHero15byY[c15by].load(std::memory_order_relaxed), absY15by,
                    (totalAbsY15by > 1e-30) ? 100.0 * absY15by / totalAbsY15by : 0.0,
                    mlHero15byCandidateY[c15by].load(std::memory_order_relaxed), candAbsY15by,
                    (absY15by > 1e-30) ? candAbsY15by / absY15by : 0.0,
                    mlHero15byEyeScaleSum[c15by].load(std::memory_order_relaxed) / den15by,
                    mlHero15byLightScaleSum[c15by].load(std::memory_order_relaxed) / den15by,
                    mlHero15byNaiveNetScaleSum[c15by].load(std::memory_order_relaxed) / den15by);
            }
            fprintf(f15az,
                "phase15by interpretation: category=both is the target; candidateDivN is a counterfactual only, not an applied correction; meanNaiveNetScale is eyeTerminationScale*lightTerminationScale*packetReconstructionScale and is descriptive, not proof of estimator bias\n");
            const unsigned long long applied15bz = mlHero15bzAppliedCount.load(std::memory_order_relaxed);
            const double baselineY15bz = mlHero15bzBaselineY.load(std::memory_order_relaxed);
            const double compensatedY15bz = mlHero15bzCompensatedY.load(std::memory_order_relaxed);
            fprintf(f15az,
                "phase15bz RUNTIME_SUMMARY enabled=%u applied=%llu baselineY=%.12g compensatedY=%.12g ratio=%.9g note=actual_render_compensation_when_enabled\n",
                engine->mlHeroDualTerminationCompensation ? 1u : 0u, applied15bz, baselineY15bz, compensatedY15bz,
                (fabs(baselineY15bz) > 1e-30) ? (compensatedY15bz / baselineY15bz) : 0.0);

			// Phase 15bj: compare the four major visible BIDIR estimators and make
			// the packet reconstruction state explicit.
			const char *estimatorName15bj[4] = {
				"ConnectVertices", "DirectLightSampling",
				"ConnectToEye", "DirectHitLight"
			};
			const char *categoryName15bj[4] = {
				"neither", "eyeOnly", "lightOnly", "both"
			};
			double allEstimatorAbsY15bj = 0.0;
			for (u_int e15bj = 0u; e15bj < 4u; ++e15bj)
				for (u_int c15bj = 0u; c15bj < 4u; ++c15bj)
					allEstimatorAbsY15bj += mlHero15bjEstimatorAbsY[e15bj][c15bj].load(std::memory_order_relaxed);

			fprintf(f15az,
				"phase15bj estimatorTotals: totalAbsY=%.12g note=actual_visible_contributions_all_four_major_BIDIR_estimators\n",
				allEstimatorAbsY15bj);

			for (u_int e15bj = 0u; e15bj < 4u; ++e15bj) {
				double estimatorY15bj = 0.0;
				double estimatorAbsY15bj = 0.0;
				unsigned long long estimatorCount15bj = 0ull;
				unsigned long long estimatorPacketDiv15bj = 0ull;
				unsigned long long estimatorClassic15bj = 0ull;

				for (u_int c15bj = 0u; c15bj < 4u; ++c15bj) {
					estimatorY15bj += mlHero15bjEstimatorY[e15bj][c15bj].load(std::memory_order_relaxed);
					estimatorAbsY15bj += mlHero15bjEstimatorAbsY[e15bj][c15bj].load(std::memory_order_relaxed);
					estimatorCount15bj += mlHero15bjEstimatorCount[e15bj][c15bj].load(std::memory_order_relaxed);
					estimatorPacketDiv15bj += mlHero15bjEstimatorPacketDivCount[e15bj][c15bj].load(std::memory_order_relaxed);
					estimatorClassic15bj += mlHero15bjEstimatorClassicCount[e15bj][c15bj].load(std::memory_order_relaxed);
				}

				fprintf(f15az,
					"phase15bj estimator=%s contributions=%llu packetDivided=%llu classicFallback=%llu Y=%.12g absY=%.12g absYShare=%.6f%%\n",
					estimatorName15bj[e15bj], estimatorCount15bj,
					estimatorPacketDiv15bj, estimatorClassic15bj,
					estimatorY15bj, estimatorAbsY15bj,
					(allEstimatorAbsY15bj > 0.0) ? 100.0 * estimatorAbsY15bj / allEstimatorAbsY15bj : 0.0);

				for (u_int c15bj = 0u; c15bj < 4u; ++c15bj) {
					const unsigned long long count15bj =
						mlHero15bjEstimatorCount[e15bj][c15bj].load(std::memory_order_relaxed);
					if (count15bj == 0ull)
						continue;

					const double r15bj = mlHero15bjEstimatorR[e15bj][c15bj].load(std::memory_order_relaxed);
					const double g15bj = mlHero15bjEstimatorG[e15bj][c15bj].load(std::memory_order_relaxed);
					const double b15bj = mlHero15bjEstimatorB[e15bj][c15bj].load(std::memory_order_relaxed);
					const double y15bj = mlHero15bjEstimatorY[e15bj][c15bj].load(std::memory_order_relaxed);
					const double absY15bj = mlHero15bjEstimatorAbsY[e15bj][c15bj].load(std::memory_order_relaxed);
					const unsigned long long packetDiv15bj =
						mlHero15bjEstimatorPacketDivCount[e15bj][c15bj].load(std::memory_order_relaxed);
					const unsigned long long classic15bj =
						mlHero15bjEstimatorClassicCount[e15bj][c15bj].load(std::memory_order_relaxed);

					fprintf(f15az,
						"phase15bj estimator=%s category=%s contributions=%llu packetDivided=%llu classicFallback=%llu R=%.12g G=%.12g B=%.12g Y=%.12g absY=%.12g estimatorAbsYShare=%.6f%% globalAbsYShare=%.6f%%\n",
						estimatorName15bj[e15bj], categoryName15bj[c15bj],
						count15bj, packetDiv15bj, classic15bj,
						r15bj, g15bj, b15bj, y15bj, absY15bj,
						(estimatorAbsY15bj > 0.0) ? 100.0 * absY15bj / estimatorAbsY15bj : 0.0,
						(allEstimatorAbsY15bj > 0.0) ? 100.0 * absY15bj / allEstimatorAbsY15bj : 0.0);
				}
			}


			// Phase 15bk: test the single-HERO RGB sign/clamp-bias hypothesis.
			// Report finite data only; INF/NaN events are counted separately.
			double rawTerminatedY15bk = 0.0;
			double clampedTerminatedY15bk = 0.0;
			double rawTerminatedAbsY15bk = 0.0;
			double clampedTerminatedAbsY15bk = 0.0;
			unsigned long long finiteTerminated15bk = 0ull;
			unsigned long long nonFiniteTerminated15bk = 0ull;
			unsigned long long anyNegativeTerminated15bk = 0ull;
			unsigned long long negativeChannelsTerminated15bk = 0ull;

			fprintf(f15az,
				"ML HERO phase 15bk HERO-ONLY RGB SIGN/CLAMP DIAGNOSTIC\n");

			for (u_int e15bk = 0u; e15bk < 4u; ++e15bk) {
				for (u_int c15bk = 0u; c15bk < 4u; ++c15bk) {
					const unsigned long long finite15bk =
						mlHero15bkFiniteCount[e15bk][c15bk].load(std::memory_order_relaxed);
					const unsigned long long nonFinite15bk =
						mlHero15bkNonFiniteCount[e15bk][c15bk].load(std::memory_order_relaxed);
					if ((finite15bk == 0ull) && (nonFinite15bk == 0ull))
						continue;

					const unsigned long long anyNegative15bk =
						mlHero15bkAnyNegativeCount[e15bk][c15bk].load(std::memory_order_relaxed);
					const unsigned long long negativeChannels15bk =
						mlHero15bkNegativeChannelCount[e15bk][c15bk].load(std::memory_order_relaxed);
					const double rawY15bk =
						mlHero15bkRawY[e15bk][c15bk].load(std::memory_order_relaxed);
					const double rawAbsY15bk =
						mlHero15bkRawAbsY[e15bk][c15bk].load(std::memory_order_relaxed);
					const double clampedY15bk =
						mlHero15bkPositiveClampedY[e15bk][c15bk].load(std::memory_order_relaxed);
					const double clampedAbsY15bk =
						mlHero15bkPositiveClampedAbsY[e15bk][c15bk].load(std::memory_order_relaxed);
					const double deltaY15bk =
						mlHero15bkClampDeltaY[e15bk][c15bk].load(std::memory_order_relaxed);

					fprintf(f15az,
						"phase15bk estimator=%s category=%s finite=%llu nonFinite=%llu anyNegative=%llu (%.6f%%) negativeChannels=%llu rawY=%.12g positiveClampedY=%.12g deltaY=%.12g clampedOverRawY=%.9g rawAbsY=%.12g positiveClampedAbsY=%.12g\n",
						estimatorName15bj[e15bk], categoryName15bj[c15bk],
						finite15bk, nonFinite15bk, anyNegative15bk,
						(finite15bk > 0ull) ? 100.0 * (double)anyNegative15bk / (double)finite15bk : 0.0,
						negativeChannels15bk, rawY15bk, clampedY15bk, deltaY15bk,
						(fabs(rawY15bk) > 1e-30) ? clampedY15bk / rawY15bk : 0.0,
						rawAbsY15bk, clampedAbsY15bk);

					// Categories 1..3 contain at least one terminated sub-path.
					if (c15bk != 0u) {
						finiteTerminated15bk += finite15bk;
						nonFiniteTerminated15bk += nonFinite15bk;
						anyNegativeTerminated15bk += anyNegative15bk;
						negativeChannelsTerminated15bk += negativeChannels15bk;
						rawTerminatedY15bk += rawY15bk;
						clampedTerminatedY15bk += clampedY15bk;
						rawTerminatedAbsY15bk += rawAbsY15bk;
						clampedTerminatedAbsY15bk += clampedAbsY15bk;
					}
				}
			}

			fprintf(f15az,
				"phase15bk terminatedTotals: finite=%llu nonFinite=%llu anyNegative=%llu (%.6f%%) negativeChannels=%llu rawY=%.12g positiveClampedY=%.12g deltaY=%.12g clampedOverRawY=%.9g rawAbsY=%.12g positiveClampedAbsY=%.12g note=hypothetical_positive_RGB_clamp_only_render_unchanged\n",
				finiteTerminated15bk, nonFiniteTerminated15bk,
				anyNegativeTerminated15bk,
				(finiteTerminated15bk > 0ull) ? 100.0 * (double)anyNegativeTerminated15bk / (double)finiteTerminated15bk : 0.0,
				negativeChannelsTerminated15bk,
				rawTerminatedY15bk, clampedTerminatedY15bk,
				clampedTerminatedY15bk - rawTerminatedY15bk,
				(fabs(rawTerminatedY15bk) > 1e-30) ? clampedTerminatedY15bk / rawTerminatedY15bk : 0.0,
				rawTerminatedAbsY15bk, clampedTerminatedAbsY15bk);

			{
				const unsigned long long n15bn = mlHero15bnTerminationCount.load(std::memory_order_relaxed);
				const double eventSum15bn = mlHero15bnEventWeightSum.load(std::memory_order_relaxed);
				const double supportSum15bn = mlHero15bnSupportWeightSum.load(std::memory_order_relaxed);
				fprintf(f15az, "ML HERO phase 15bn SPECTRAL PATH-PDF / DIRAC-SUPPORT MIS DIAGNOSTIC\n");
				fprintf(f15az, "phase15bn totals: terminations=%llu meanEventOnlyWeight=%.12g meanDiracSupportWeight=%.12g meanEventDenom=%.12g meanSupportDenom=%.12g supportWeightNearOne=%llu (%.6f%%) supportCompetitors=%llu (%.6f%%)\n",
					n15bn, n15bn ? eventSum15bn / (double)n15bn : 0.0, n15bn ? supportSum15bn / (double)n15bn : 0.0,
					n15bn ? mlHero15bnEventDenomSum.load(std::memory_order_relaxed) / (double)n15bn : 0.0,
					n15bn ? mlHero15bnSupportDenomSum.load(std::memory_order_relaxed) / (double)n15bn : 0.0,
					mlHero15bnSupportWeightNearOne.load(std::memory_order_relaxed),
					n15bn ? 100.0 * (double)mlHero15bnSupportWeightNearOne.load(std::memory_order_relaxed) / (double)n15bn : 0.0,
					mlHero15bnSupportCompetitorCount.load(std::memory_order_relaxed),
					n15bn ? 100.0 * (double)mlHero15bnSupportCompetitorCount.load(std::memory_order_relaxed) / (double)n15bn : 0.0);
				fprintf(f15az, "phase15bn eventOnlyThresholds: <0.99=%llu <0.75=%llu <0.50=%llu <0.25=%llu note=event_only_is_not_a_valid_perfect_Dirac_path_probability; support-aware weight is the relevant candidate\n",
					mlHero15bnEventWeightBelow099.load(std::memory_order_relaxed), mlHero15bnEventWeightBelow075.load(std::memory_order_relaxed),
					mlHero15bnEventWeightBelow050.load(std::memory_order_relaxed), mlHero15bnEventWeightBelow025.load(std::memory_order_relaxed));
			}

			fprintf(f15az, "ML HERO phase 15bo BIDIR MIS TERMINATION-SUPPORT PRUNING DIAGNOSTIC\n");
			const char *estimatorNames15bo[4] = { "ConnectVertices", "DirectLightSampling", "ConnectToEye", "DirectHitLight" };
			const char *categoryNames15bo[4] = { "neither", "eyeOnly", "lightOnly", "both" };
			for (u_int e15bo = 0u; e15bo < 4u; ++e15bo) {
				for (u_int c15bo = 0u; c15bo < 4u; ++c15bo) {
					const unsigned long long n15bo = mlHero15boMisCount[e15bo][c15bo].load(std::memory_order_relaxed);
					if (!n15bo)
						continue;
					const double currentSum15bo = mlHero15boCurrentMisSum[e15bo][c15bo].load(std::memory_order_relaxed);
					const double candidateSum15bo = mlHero15boCandidateMisSum[e15bo][c15bo].load(std::memory_order_relaxed);
					const double ratioSum15bo = mlHero15boRatioSum[e15bo][c15bo].load(std::memory_order_relaxed);
					fprintf(f15az, "phase15bo estimator=%s category=%s count=%llu meanCurrentMis=%.12g meanPrunedMis=%.12g meanPrunedOverCurrent=%.12g meanRemovedWeight=%.12g ratio>1.01=%llu ratio>1.05=%llu ratio>1.25=%llu ratio>1.50=%llu\n",
						estimatorNames15bo[e15bo], categoryNames15bo[c15bo], n15bo,
						currentSum15bo / (double)n15bo, candidateSum15bo / (double)n15bo, ratioSum15bo / (double)n15bo,
						mlHero15boRemovedWeightSum[e15bo][c15bo].load(std::memory_order_relaxed) / (double)n15bo,
						mlHero15boRatioOver101[e15bo][c15bo].load(std::memory_order_relaxed),
						mlHero15boRatioOver105[e15bo][c15bo].load(std::memory_order_relaxed),
						mlHero15boRatioOver125[e15bo][c15bo].load(std::memory_order_relaxed),
						mlHero15boRatioOver150[e15bo][c15bo].load(std::memory_order_relaxed));
				}
			}
			fprintf(f15az, "phase15bo note=diagnostic_upper_bound_only_render_unchanged; pruning removes alternate-strategy terms on HERO-terminated subpath sides; candidate>=current means this mechanism would brighten rather than explain existing over-brightness\n");

            fprintf(f15az, "ML HERO phase 15bq STATE PROPAGATION SUMMARY\n");
            fprintf(f15az, "phase15bq totals: stores=%llu propagates=%llu readsCV=%llu readsDLS=%llu readsCTE=%llu readsDHL=%llu invalidRatio=%llu\n",
                mlHero15bqStoreCount.load(std::memory_order_relaxed),
                mlHero15bqPropagateCount.load(std::memory_order_relaxed),
                mlHero15bqReadCount[0].load(std::memory_order_relaxed),
                mlHero15bqReadCount[1].load(std::memory_order_relaxed),
                mlHero15bqReadCount[2].load(std::memory_order_relaxed),
                mlHero15bqReadCount[3].load(std::memory_order_relaxed),
                mlHero15bqInvalidRatioCount.load(std::memory_order_relaxed));
			fprintf(f15az, "ML HERO phase 15bp VISIBLE-CONTRIBUTION-WEIGHTED HERO TERMINATION BIAS\n");
			const char *estimatorNames15bp[4] = { "ConnectVertices", "DirectLightSampling", "ConnectToEye", "DirectHitLight" };
			const char *categoryNames15bp[4] = { "neither", "eyeOnly", "lightOnly", "both" };
			for (u_int e15bp = 0u; e15bp < 4u; ++e15bp) {
				for (u_int c15bp = 1u; c15bp < 4u; ++c15bp) {
					const unsigned long long n15bp = mlHero15bpCount[e15bp][c15bp].load(std::memory_order_relaxed);
					if (!n15bp)
						continue;
					const double ratioSum15bp = mlHero15bpRatioSum[e15bp][c15bp].load(std::memory_order_relaxed);
					const double ySum15bp = mlHero15bpYSum[e15bp][c15bp].load(std::memory_order_relaxed);
					const double absYSum15bp = mlHero15bpAbsYSum[e15bp][c15bp].load(std::memory_order_relaxed);
					const double positiveYSum15bp = mlHero15bpPositiveYSum[e15bp][c15bp].load(std::memory_order_relaxed);
					const double weightedY15bp = mlHero15bpRatioTimesY[e15bp][c15bp].load(std::memory_order_relaxed);
					const double weightedAbsY15bp = mlHero15bpRatioTimesAbsY[e15bp][c15bp].load(std::memory_order_relaxed);
					const double weightedPositiveY15bp = mlHero15bpRatioTimesPositiveY[e15bp][c15bp].load(std::memory_order_relaxed);
					fprintf(f15az, "phase15bp estimator=%s category=%s count=%llu meanStoredRatio=%.12g signedYWeightedRatio=%.12g absYWeightedRatio=%.12g positiveYWeightedRatio=%.12g Y=%.12g absY=%.12g absYShareRatio>1.05=%.6f%% >1.25=%.6f%% >1.50=%.6f%% ratio<0.95=%.6f%%\n",
						estimatorNames15bp[e15bp], categoryNames15bp[c15bp], n15bp,
						ratioSum15bp / (double)n15bp,
						(fabs(ySum15bp) > 1e-30) ? weightedY15bp / ySum15bp : 0.0,
						(absYSum15bp > 1e-30) ? weightedAbsY15bp / absYSum15bp : 0.0,
						(positiveYSum15bp > 1e-30) ? weightedPositiveY15bp / positiveYSum15bp : 0.0,
						ySum15bp, absYSum15bp,
						(absYSum15bp > 1e-30) ? 100.0 * mlHero15bpAbsYOver105[e15bp][c15bp].load(std::memory_order_relaxed) / absYSum15bp : 0.0,
						(absYSum15bp > 1e-30) ? 100.0 * mlHero15bpAbsYOver125[e15bp][c15bp].load(std::memory_order_relaxed) / absYSum15bp : 0.0,
						(absYSum15bp > 1e-30) ? 100.0 * mlHero15bpAbsYOver150[e15bp][c15bp].load(std::memory_order_relaxed) / absYSum15bp : 0.0,
						(absYSum15bp > 1e-30) ? 100.0 * mlHero15bpAbsYUnder095[e15bp][c15bp].load(std::memory_order_relaxed) / absYSum15bp : 0.0);
				}
			}
			fprintf(f15az, "phase15bp note=diagnostic_only_render_unchanged; ratio_is_survivingHeroScalar_over_packetAverage_captured_at_first_HERO_only_termination; both_category_uses_eyeRatio_times_lightRatio; weighted_means_test_whether_high-energy_visible_contributions_correlate_with_bright_HERO_selections\n");

            fprintf(f15az, "ML HERO phase 15bs VISIBLE HERO BIAS BY WAVELENGTH / SIDE / DEPTH\n");
            {
                const char *estimatorNames15bs[4] = { "ConnectVertices", "DirectLightSampling", "ConnectToEye", "DirectHitLight" };
                const char *sideNames15bs[2] = { "eye", "light" };
                const char *waveNames15bs[ML_HERO_15BS_WAVELENGTH_BINS] = { "380-450", "450-500", "500-550", "550-600", "600-650", "650-700", "700-780" };
                const char *depthNames15bs[ML_HERO_15BS_DEPTH_BINS] = { "1", "2", "3", "4", "5", "6+" };
                double globalAbsY15bs[ML_HERO_15BS_WAVELENGTH_BINS] = { 0.0 };
                double globalRatioAbsY15bs[ML_HERO_15BS_WAVELENGTH_BINS] = { 0.0 };
                unsigned long long globalCount15bs[ML_HERO_15BS_WAVELENGTH_BINS] = { 0ull };
                double sideAbsY15bs[2][ML_HERO_15BS_WAVELENGTH_BINS] = { {0.0} };
                double sideRatioAbsY15bs[2][ML_HERO_15BS_WAVELENGTH_BINS] = { {0.0} };
                unsigned long long sideCount15bs[2][ML_HERO_15BS_WAVELENGTH_BINS] = { {0ull} };

                for (u_int e15bs = 0u; e15bs < 4u; ++e15bs) {
                    for (u_int s15bs = 0u; s15bs < 2u; ++s15bs) {
                        for (u_int w15bs = 0u; w15bs < ML_HERO_15BS_WAVELENGTH_BINS; ++w15bs) {
                            unsigned long long count15bs = 0ull;
                            double ratioSum15bs = 0.0, ySum15bs = 0.0, absYSum15bs = 0.0;
                            double weightedY15bs = 0.0, weightedAbsY15bs = 0.0;
                            unsigned long long over10515bs = 0ull, under09515bs = 0ull;
                            for (u_int d15bs = 0u; d15bs < ML_HERO_15BS_DEPTH_BINS; ++d15bs) {
                                const unsigned long long n15bs = mlHero15bsCount[e15bs][s15bs][w15bs][d15bs].load(std::memory_order_relaxed);
                                const double a15bs = mlHero15bsAbsYSum[e15bs][s15bs][w15bs][d15bs].load(std::memory_order_relaxed);
                                if (n15bs && (a15bs > 0.0)) {
                                    fprintf(f15az, "phase15bs depth estimator=%s side=%s lambda=%s depth=%s count=%llu meanRatio=%.12g absYWeightedRatio=%.12g signedYWeightedRatio=%.12g absY=%.12g ratioCount>1.05=%llu ratioCount<0.95=%llu\n",
                                        estimatorNames15bs[e15bs], sideNames15bs[s15bs], waveNames15bs[w15bs], depthNames15bs[d15bs], n15bs,
                                        mlHero15bsRatioSum[e15bs][s15bs][w15bs][d15bs].load(std::memory_order_relaxed) / (double)n15bs,
                                        mlHero15bsRatioTimesAbsY[e15bs][s15bs][w15bs][d15bs].load(std::memory_order_relaxed) / a15bs,
                                        (fabs(mlHero15bsYSum[e15bs][s15bs][w15bs][d15bs].load(std::memory_order_relaxed)) > 1e-30) ?
                                            mlHero15bsRatioTimesY[e15bs][s15bs][w15bs][d15bs].load(std::memory_order_relaxed) /
                                            mlHero15bsYSum[e15bs][s15bs][w15bs][d15bs].load(std::memory_order_relaxed) : 0.0,
                                        a15bs,
                                        mlHero15bsRatioOver105[e15bs][s15bs][w15bs][d15bs].load(std::memory_order_relaxed),
                                        mlHero15bsRatioUnder095[e15bs][s15bs][w15bs][d15bs].load(std::memory_order_relaxed));
                                }
                                count15bs += n15bs;
                                ratioSum15bs += mlHero15bsRatioSum[e15bs][s15bs][w15bs][d15bs].load(std::memory_order_relaxed);
                                ySum15bs += mlHero15bsYSum[e15bs][s15bs][w15bs][d15bs].load(std::memory_order_relaxed);
                                absYSum15bs += a15bs;
                                weightedY15bs += mlHero15bsRatioTimesY[e15bs][s15bs][w15bs][d15bs].load(std::memory_order_relaxed);
                                weightedAbsY15bs += mlHero15bsRatioTimesAbsY[e15bs][s15bs][w15bs][d15bs].load(std::memory_order_relaxed);
                                over10515bs += mlHero15bsRatioOver105[e15bs][s15bs][w15bs][d15bs].load(std::memory_order_relaxed);
                                under09515bs += mlHero15bsRatioUnder095[e15bs][s15bs][w15bs][d15bs].load(std::memory_order_relaxed);
                            }
                            if (count15bs && (absYSum15bs > 0.0)) {
                                fprintf(f15az, "phase15bs wavelength estimator=%s side=%s lambda=%s count=%llu meanRatio=%.12g absYWeightedRatio=%.12g signedYWeightedRatio=%.12g absY=%.12g ratioCount>1.05=%llu ratioCount<0.95=%llu\n",
                                    estimatorNames15bs[e15bs], sideNames15bs[s15bs], waveNames15bs[w15bs], count15bs,
                                    ratioSum15bs / (double)count15bs, weightedAbsY15bs / absYSum15bs,
                                    (fabs(ySum15bs) > 1e-30) ? weightedY15bs / ySum15bs : 0.0,
                                    absYSum15bs, over10515bs, under09515bs);
                            }
                            globalCount15bs[w15bs] += count15bs;
                            globalAbsY15bs[w15bs] += absYSum15bs;
                            globalRatioAbsY15bs[w15bs] += weightedAbsY15bs;
                            sideCount15bs[s15bs][w15bs] += count15bs;
                            sideAbsY15bs[s15bs][w15bs] += absYSum15bs;
                            sideRatioAbsY15bs[s15bs][w15bs] += weightedAbsY15bs;
                        }
                    }
                }
                for (u_int w15bs = 0u; w15bs < ML_HERO_15BS_WAVELENGTH_BINS; ++w15bs) {
                    if (globalCount15bs[w15bs] && (globalAbsY15bs[w15bs] > 0.0))
                        fprintf(f15az, "phase15bs GLOBAL lambda=%s count=%llu absYWeightedRatio=%.12g absY=%.12g\n",
                            waveNames15bs[w15bs], globalCount15bs[w15bs], globalRatioAbsY15bs[w15bs] / globalAbsY15bs[w15bs], globalAbsY15bs[w15bs]);
                    for (u_int s15bs = 0u; s15bs < 2u; ++s15bs)
                        if (sideCount15bs[s15bs][w15bs] && (sideAbsY15bs[s15bs][w15bs] > 0.0))
                            fprintf(f15az, "phase15bs SIDE side=%s lambda=%s count=%llu absYWeightedRatio=%.12g absY=%.12g\n",
                                sideNames15bs[s15bs], waveNames15bs[w15bs], sideCount15bs[s15bs][w15bs],
                                sideRatioAbsY15bs[s15bs][w15bs] / sideAbsY15bs[s15bs][w15bs], sideAbsY15bs[s15bs][w15bs]);
                }
                fprintf(f15az, "phase15bs note=diagnostic_only_render_unchanged; each terminated eye/light side is recorded independently using its surviving HERO wavelength; both-sided ConnectVertices contributes once to each side so wavelength correlation is not hidden by multiplying two ratios; depth is current visible-contribution vertex depth, not necessarily first-termination depth\n");
            }

            fprintf(f15az, "ML HERO phase 15bu DIRECT PACKET-REFERENCE THROUGHPUT PROPAGATION DIAGNOSTIC\n");
            {
                const char *est15bu[4] = { "ConnectVertices", "DirectLightSampling", "ConnectToEye", "DirectHitLight" };
                const char *side15bu[2] = { "eye", "light" };
                const char *wave15bu[ML_HERO_15BS_WAVELENGTH_BINS] = { "380-450", "450-500", "500-550", "550-600", "600-650", "650-700", "700-780" };
                double allA15bu = 0.0, allR15bu = 0.0, allS15bu = 0.0;
                unsigned long long allN15bu = 0ull;
                for (u_int w = 0; w < ML_HERO_15BS_WAVELENGTH_BINS; ++w) {
                    double aW=0.0, rW=0.0, sW=0.0; unsigned long long nW=0ull;
                    for (u_int e=0; e<4; ++e) for (u_int sd=0; sd<2; ++sd) {
                        const auto n = mlHero15buVisibleCount[e][sd][w].load(std::memory_order_relaxed);
                        if (!n) continue;
                        const double a = mlHero15buActualAbsY[e][sd][w].load(std::memory_order_relaxed);
                        const double r = mlHero15buReferenceAbsY[e][sd][w].load(std::memory_order_relaxed);
                        const double sw = mlHero15buRequiredScaleTimesAbsY[e][sd][w].load(std::memory_order_relaxed);
                        fprintf(f15az, "phase15bu visible estimator=%s side=%s lambda=%s count=%llu actualAbsY=%.12g directReferenceAbsY=%.12g actualOverReference=%.12g absYWeightedRequiredScale=%.12g\n",
                            est15bu[e], side15bu[sd], wave15bu[w], n, a, r, (r>1e-30)?a/r:0.0, (a>1e-30)?sw/a:0.0);
                        nW+=n; aW+=a; rW+=r; sW+=sw;
                    }
                    if (nW) fprintf(f15az, "phase15bu GLOBAL lambda=%s count=%llu actualOverReference=%.12g absYWeightedRequiredScale=%.12g actualAbsY=%.12g directReferenceAbsY=%.12g\n",
                        wave15bu[w], nW, (rW>1e-30)?aW/rW:0.0, (aW>1e-30)?sW/aW:0.0, aW, rW);
                    allN15bu+=nW; allA15bu+=aW; allR15bu+=rW; allS15bu+=sW;
                }
                fprintf(f15az, "phase15bu ALL count=%llu actualOverReference=%.12g absYWeightedRequiredScale=%.12g actualAbsY=%.12g directReferenceAbsY=%.12g rejectedCount=%llu rejectedAbsY=%.12g currentScale=8\n",
                    allN15bu, (allR15bu>1e-30)?allA15bu/allR15bu:0.0, (allA15bu>1e-30)?allS15bu/allA15bu:0.0, allA15bu, allR15bu,
                    mlHero15buRejectedCount.load(std::memory_order_relaxed), mlHero15buRejectedAbsY.load(std::memory_order_relaxed));
                fprintf(f15az, "phase15bu note=diagnostic_only_render_unchanged; reference throughput is initialized directly as packetAverage*N at first HERO-only termination and propagated by the same subsequent HERO transport multipliers; aggregate rejects only extreme actual/reference ratios outside [1/16,16] and reports their count/absY separately\n");
            }

            fprintf(f15az, "ML HERO phase 15bt HERO SELECTION-PDF / EMPIRICAL RECONSTRUCTION-SCALE DIAGNOSTIC\n");
            {
                const char *estimatorNames15bt[4] = { "ConnectVertices", "DirectLightSampling", "ConnectToEye", "DirectHitLight" };
                const char *sideNames15bt[2] = { "eye", "light" };
                const char *waveNames15bt[ML_HERO_15BS_WAVELENGTH_BINS] = { "380-450", "450-500", "500-550", "550-600", "600-650", "650-700", "700-780" };
                double globalActual15bt[ML_HERO_15BS_WAVELENGTH_BINS] = { 0.0 };
                double globalReference15bt[ML_HERO_15BS_WAVELENGTH_BINS] = { 0.0 };
                double globalRequiredScaleTimesAbsY15bt[ML_HERO_15BS_WAVELENGTH_BINS] = { 0.0 };
                unsigned long long globalVisibleCount15bt[ML_HERO_15BS_WAVELENGTH_BINS] = { 0ull };
                double allActual15bt = 0.0, allReference15bt = 0.0, allRequiredScaleTimesAbsY15bt = 0.0;
                unsigned long long allVisibleCount15bt = 0ull;

                for (u_int s15bt = 0u; s15bt < 2u; ++s15bt) {
                    for (u_int w15bt = 0u; w15bt < ML_HERO_15BS_WAVELENGTH_BINS; ++w15bt) {
                        const unsigned long long n15bt = mlHero15btTerminationCount[s15bt][w15bt].load(std::memory_order_relaxed);
                        if (!n15bt) continue;
                        fprintf(f15az, "phase15bt termination side=%s lambda=%s count=%llu meanHeroSelectionPdf=%.12g meanWavelengthPdf=%.12g meanPacketAverage=%.12g meanHeroScalar=%.12g meanHeroOverPacket=%.12g meanCurrentScale=%.12g meanLocalIdealScale=%.12g\n",
                            sideNames15bt[s15bt], waveNames15bt[w15bt], n15bt,
                            mlHero15btLaneSelectionPdfSum[s15bt][w15bt].load(std::memory_order_relaxed) / (double)n15bt,
                            mlHero15btWavelengthPdfSum[s15bt][w15bt].load(std::memory_order_relaxed) / (double)n15bt,
                            mlHero15btPacketAverageSum[s15bt][w15bt].load(std::memory_order_relaxed) / (double)n15bt,
                            mlHero15btHeroScalarSum[s15bt][w15bt].load(std::memory_order_relaxed) / (double)n15bt,
                            mlHero15btTerminationRatioSum[s15bt][w15bt].load(std::memory_order_relaxed) / (double)n15bt,
                            mlHero15btCurrentScaleSum[s15bt][w15bt].load(std::memory_order_relaxed) / (double)n15bt,
                            mlHero15btLocalIdealScaleSum[s15bt][w15bt].load(std::memory_order_relaxed) / (double)n15bt);
                    }
                }

                for (u_int e15bt = 0u; e15bt < 4u; ++e15bt) {
                    for (u_int s15bt = 0u; s15bt < 2u; ++s15bt) {
                        for (u_int w15bt = 0u; w15bt < ML_HERO_15BS_WAVELENGTH_BINS; ++w15bt) {
                            const unsigned long long n15bt = mlHero15btVisibleCount[e15bt][s15bt][w15bt].load(std::memory_order_relaxed);
                            if (!n15bt) continue;
                            const double actual15bt = mlHero15btActualAbsY[e15bt][s15bt][w15bt].load(std::memory_order_relaxed);
                            const double reference15bt = mlHero15btReferenceAbsY[e15bt][s15bt][w15bt].load(std::memory_order_relaxed);
                            const double requiredScaleWeighted15bt = mlHero15btRequiredScaleTimesAbsY[e15bt][s15bt][w15bt].load(std::memory_order_relaxed);
                            fprintf(f15az, "phase15bt visible estimator=%s side=%s lambda=%s count=%llu actualAbsY=%.12g packetReferenceAbsY=%.12g actualOverReference=%.12g absYWeightedRequiredScale=%.12g\n",
                                estimatorNames15bt[e15bt], sideNames15bt[s15bt], waveNames15bt[w15bt], n15bt,
                                actual15bt, reference15bt,
                                (reference15bt > 1e-30) ? actual15bt / reference15bt : 0.0,
                                (actual15bt > 1e-30) ? requiredScaleWeighted15bt / actual15bt : 0.0);
                            globalVisibleCount15bt[w15bt] += n15bt;
                            globalActual15bt[w15bt] += actual15bt;
                            globalReference15bt[w15bt] += reference15bt;
                            globalRequiredScaleTimesAbsY15bt[w15bt] += requiredScaleWeighted15bt;
                            allVisibleCount15bt += n15bt;
                            allActual15bt += actual15bt;
                            allReference15bt += reference15bt;
                            allRequiredScaleTimesAbsY15bt += requiredScaleWeighted15bt;
                        }
                    }
                }

                for (u_int w15bt = 0u; w15bt < ML_HERO_15BS_WAVELENGTH_BINS; ++w15bt) {
                    if (!globalVisibleCount15bt[w15bt]) continue;
                    fprintf(f15az, "phase15bt DIAGNOSTIC_AGGREGATE lambda=%s count=%llu actualOverReference=%.12g absYWeightedRequiredScale=%.12g actualAbsY=%.12g packetReferenceAbsY=%.12g\n",
                        waveNames15bt[w15bt], globalVisibleCount15bt[w15bt],
                        (globalReference15bt[w15bt] > 1e-30) ? globalActual15bt[w15bt] / globalReference15bt[w15bt] : 0.0,
                        (globalActual15bt[w15bt] > 1e-30) ? globalRequiredScaleTimesAbsY15bt[w15bt] / globalActual15bt[w15bt] : 0.0,
                        globalActual15bt[w15bt], globalReference15bt[w15bt]);
                }
                fprintf(f15az, "phase15bt DIAGNOSTIC_ALL count=%llu actualOverReference=%.12g absYWeightedRequiredScale=%.12g actualAbsY=%.12g packetReferenceAbsY=%.12g invalidVisible=%llu\n",
                    allVisibleCount15bt,
                    (allReference15bt > 1e-30) ? allActual15bt / allReference15bt : 0.0,
                    (allActual15bt > 1e-30) ? allRequiredScaleTimesAbsY15bt / allActual15bt : 0.0,
                    allActual15bt, allReference15bt, mlHero15btInvalidVisibleCount.load(std::memory_order_relaxed));
                fprintf(f15az, "phase15bt note=diagnostic_only_render_unchanged; heroSelectionPdf_is_discrete_1_over_packetCount; wavelengthPdf_is_pretermination_continuous_pdf; packetReferenceAbsY_is_actualAbsY_divided_by_survivingHeroScalar_over_packetAverage; requiredScale_assumes_linear_posttermination_throughput; DIAGNOSTIC_AGGREGATE_and_ALL_double-count_both-sided_ConnectVertices_and_are_not_a_renderer-wide_correction_factor\n");
            }

            fprintf(f15az, "ML HERO phase 15br GLASS PER-LANE VS SHARED-WEIGHT HERO-ONLY DIAGNOSTIC\n");
            {
                const unsigned long long n15br = mlHero15brTerminationCount.load(std::memory_order_relaxed);
                const double heroRatioSum15br = mlHero15brHeroRatioSum.load(std::memory_order_relaxed);
                const double packetRatioSum15br = mlHero15brPacketRatioSum.load(std::memory_order_relaxed);
                const double actualHeroSum15br = mlHero15brActualHeroSum.load(std::memory_order_relaxed);
                const double sharedHeroSum15br = mlHero15brSharedHeroSum.load(std::memory_order_relaxed);
                const double actualPacketSum15br = mlHero15brActualPacketAvgSum.load(std::memory_order_relaxed);
                const double sharedPacketSum15br = mlHero15brSharedPacketAvgSum.load(std::memory_order_relaxed);
                fprintf(f15az, "phase15br totals: terminations=%llu eye=%llu light=%llu invalid=%llu meanHeroPerLaneOverShared=%.12g aggregateHeroEnergyRatio=%.12g meanPacketPerLaneOverShared=%.12g aggregatePacketEnergyRatio=%.12g\n",
                    n15br,
                    mlHero15brEyeCount.load(std::memory_order_relaxed),
                    mlHero15brLightCount.load(std::memory_order_relaxed),
                    mlHero15brInvalidCount.load(std::memory_order_relaxed),
                    n15br ? heroRatioSum15br / (double)n15br : 0.0,
                    (fabs(sharedHeroSum15br) > 1e-30) ? actualHeroSum15br / sharedHeroSum15br : 0.0,
                    n15br ? packetRatioSum15br / (double)n15br : 0.0,
                    (fabs(sharedPacketSum15br) > 1e-30) ? actualPacketSum15br / sharedPacketSum15br : 0.0);
                fprintf(f15az, "phase15br heroThresholds: >1.01=%llu >1.05=%llu >1.25=%llu >1.50=%llu <0.99=%llu; packetThresholds: >1.01=%llu >1.05=%llu >1.25=%llu <0.99=%llu\n",
                    mlHero15brHeroOver101.load(std::memory_order_relaxed),
                    mlHero15brHeroOver105.load(std::memory_order_relaxed),
                    mlHero15brHeroOver125.load(std::memory_order_relaxed),
                    mlHero15brHeroOver150.load(std::memory_order_relaxed),
                    mlHero15brHeroUnder099.load(std::memory_order_relaxed),
                    mlHero15brPacketOver101.load(std::memory_order_relaxed),
                    mlHero15brPacketOver105.load(std::memory_order_relaxed),
                    mlHero15brPacketOver125.load(std::memory_order_relaxed),
                    mlHero15brPacketUnder099.load(std::memory_order_relaxed));
                fprintf(f15az, "phase15br note=diagnostic_only; shared_counterfactual_uses_same_preGlass_lane_throughputs_and_same_sampled_HERO_event_geometry; hero_ratio_directly_compares_reconstructed_Mode1_energy_before_common_xN_divN; HERO_64_runtime_glassperlaneweight_controls_weighting_only_and_no_longer_disables_Mode1_termination\n");
            }

			fprintf(f15az, "ML HERO phase 15bd TOP GLASS OUTLIERS (descending abs correction delta)\n");
			for (u_int i15ba = 0u; i15ba < 16u; ++i15ba) {
				const MLHeroGlassOutlier15ba &o15ba = st15az.topOutliers[i15ba];
				if (!o15ba.valid)
					break;
				fprintf(f15az,
					"phase15bd outlier[%u]: correction=%.9g absDelta=%.9g case=%s depth=%u "
					"heroLambda=%.9g laneLambda=%.9g heroPdf=%.9g lanePdf=%.9g "
					"heroOverLanePdf=%.9g pdfAdjustedCorrection=%.9g spectralMixPdf=%.9g spectralMixAdjustedCorrection=%.9g supportMatch=%u supportAwareCorrection=%.9g "
					"nc=%.9g baseIOR=%.9g heroIOR=%.9g laneIOR=%.9g eta=%.9g eta2=%.9g "
					"cosFixed=%.9g sinI2=%.9g baseSinT2=%.9g heroSinT2=%.9g laneSinT2=%.9g "
					"baseTIR=%u heroTIR=%u laneTIR=%u "
					"baseDirDelta=%.9g heroDirDelta=%.9g laneDirDelta=%.9g "
					"baseDirAngleRad=%.9g heroDirAngleRad=%.9g laneDirAngleRad=%.9g "
					"fresnelR=%.9g transport=%.9g sharedMultiplier=%.9g laneMultiplier=%.9g\n",
					i15ba, o15ba.correction, o15ba.absDelta,
					MLHeroGlassCaseName15az(o15ba.caseIndex), o15ba.depth,
					o15ba.heroLambda, o15ba.lambda,
					o15ba.heroEventPdfW, o15ba.laneEventPdfW,
					o15ba.pdfRatioHeroOverLane, o15ba.pdfAdjustedCorrection,
					o15ba.spectralMixPdfW, o15ba.spectralMixAdjustedCorrection,
					o15ba.supportMatch ? 1u : 0u, o15ba.supportAwareCorrection,
					o15ba.nc, o15ba.ntBase, o15ba.heroIOR, o15ba.ntLambda, o15ba.eta, o15ba.eta2,
					o15ba.cosFixed, o15ba.sinI2, o15ba.baseSinT2, o15ba.heroSinT2, o15ba.sinT2,
					o15ba.baseTir ? 1u : 0u, o15ba.heroTir ? 1u : 0u, o15ba.tir ? 1u : 0u,
					o15ba.baseDirDelta, o15ba.heroDirDelta, o15ba.dirDelta,
					o15ba.baseDirAngle, o15ba.heroDirAngle, o15ba.laneDirAngle,
					o15ba.fresnelR, o15ba.transport, o15ba.sharedMultiplier, o15ba.laneMultiplier);
			}
			fprintf(f15az, "phase15bp note=diagnostic_only_render_behavior_unchanged_vs_15bo; visible_contribution_weighted_HERO_bias_diagnostic_enabled; supportAwareCorrection is only reported for same-support lanes; support mismatches are classified separately and never clamped or applied; heroOnlyEnergy only reports packet-average vs surviving-hero brightness ratios\n");
			fprintf(f15az, "------------------------------------------------------------\n");
			fclose(f15az);
		}
	}

	threadDone = true;

	// This is done to stop threads pending on barrier wait
	// inside engine->photonGICache->Update(). This can happen when an
	// halt condition is satisfied.
	if (photonGICache)
		photonGICache->FinishUpdate(threadIndex);

#ifndef NDEBUG
	SLG_LOG("[BiDirCPURenderThread::" << threadIndex << "] Rendering thread halted");
#endif
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
