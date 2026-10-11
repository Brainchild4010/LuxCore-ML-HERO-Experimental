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


#include <atomic>
#include <cmath>
#include "slg/engines/mlhero_unit_reflectance15fx.h"
#include "slg/engines/mlhero_path_rr15fz.h"
#include "mlherolog.h"
#include <cstdio>
#include <mutex>
#include <memory>
// 15fh BEGIN diagnostic-only
#include "slg/engines/mlhero_diagnostics15fh.h"
#include "slg/lights/trianglelight.h"
// 15fh END


#include "luxrays/utils/properties.h"
#include "luxrays/core/color/spds/rgbrefl.h"
#include "luxrays/core/color/spds/rgbillum.h"
#include "luxrays/core/color/spds/data/xyzbasis.h"
#include "luxrays/core/color/spds/data/rgbD65_MLHERO_380_780.h"
#include "slg/lights/light.h"
#include "slg/usings.h"
#include "slg/engines/pathtracer.h"
#include "slg/engines/caches/photongi/photongicache.h"
#include "slg/samplers/metropolis.h"
#include "slg/utils/varianceclamping.h"
#include "slg/cameras/camera.h"

// ML HERO dispersion helpers implemented in glass.cpp.
// They are currently defined in the global namespace (same as the BIDIR patch).
void SetMLHeroEnabled(const bool enabled);
extern thread_local bool mlHeroRoughGlassFresnelImportance;
void SetMLHeroWavelengthCount(const u_int count);
void SetMLHeroWaveLengthAt(const u_int lane, const float waveLength, const float sampleWeight);
float GetMLHeroWaveLengthAt(const u_int lane);
float GetMLHeroSampleWeightAt(const u_int lane);
void SetMLHeroActiveLane(const u_int lane);
void SetMLDispersionWaveLength(const float waveLength);
void SetMLDispersionWaveLength(const float waveLength, const float sampleWeight);
luxrays::Spectrum GetMLDispersionSampleColor();
luxrays::Spectrum GetMLHeroSampleColorAt(const u_int lane);
bool GetMLDispersionUsed();
bool GetMLHeroPacketFallbackUsed();

using namespace std;
using namespace luxrays;
using namespace slg;

// 15fl: restrict paired endpoint reconstruction to ordinary flat area emitters.
static unsigned MLHeroTerminalLightGuard15fl(LightSourceConstRef light,const float time) {
    if(light.GetType()!=slg::TYPE_TRIANGLE)return 3u;
    const auto&tl=static_cast<const TriangleLight&>(light);
    if(tl.lightMaterial->GetEmissionFunc() || tl.lightMaterial->GetBumpTexture() ||
            tl.lightMaterial->GetEmittedTheta()!=90.f || light.GetAvgPassThroughTransparency()!=1.f)
        return 3u;
    auto&mesh=tl.sceneObject->GetExtMesh();
    if(mesh.HasNormals()) {
        Transform transform;mesh.GetLocal2World(time,transform);
        const Normal gn=mesh.GetGeometryNormal(transform,tl.triangleIndex);
        for(unsigned i=0;i<3;++i)
            if(Dot(gn,mesh.GetShadeNormal(transform,tl.triangleIndex,i))<1.f-1e-6f)return 6u;
    }
    return 0u;
}


//------------------------------------------------------------------------------
// ML HERO 15ev - CPU PATH parallel packet-throughput diagnostics
//------------------------------------------------------------------------------

#define ML_HERO_BUILD "HERO_170"
#define ML_HERO_TEST_PHASE "15gb"
#define ML_HERO_FEATURE "Central CPU diagnostic log filenames; renderer physics unchanged from 15ga"
#define ML_HERO_PATH_LOG_FILE slg::mlhero15gb::FilePath()

static std::mutex mlHeroPathDebugMutex15ev;
static std::atomic<unsigned long long> mlHeroPathEyeSamples15ev(0ull);
static std::atomic<unsigned long long> mlHeroPathPacketSamples15ev(0ull);
static std::atomic<unsigned long long> mlHeroPathQuarterSamples15ev(0ull);
static std::atomic<unsigned long long> mlHeroPathLogDetailBudget15ev(16ull);
static std::atomic<unsigned long long> mlHeroPathCompanionBounceCount15ev(0ull);
static std::atomic<unsigned long long> mlHeroPathScalarBounceCount15ev(0ull);
static std::atomic<unsigned long long> mlHeroPathSpectralReflectBounceCount15ev(0ull);
static std::atomic<unsigned long long> mlHeroPathUnsupportedBounceCount15ev(0ull);
static std::atomic<unsigned long long> mlHeroPathConnectionFallbackCount15ev(0ull);
static std::atomic<unsigned long long> mlHeroPathBounceLogBudget15ev(48ull);
static std::atomic<unsigned long long> mlHeroPathEndpointBudget15ev(96ull);
static std::atomic<unsigned long long> mlHeroPathEndpointDLS15ev(0ull);
static std::atomic<unsigned long long> mlHeroPathEndpointFinite15ev(0ull);
static std::atomic<unsigned long long> mlHeroPathEndpointInfinite15ev(0ull);
static std::atomic<unsigned long long> mlHeroPathEndpointUnsupported15ev(0ull);
static std::atomic<unsigned long long> mlHeroPathMaterialAuditBudget15ev(160ull);
static std::atomic<unsigned long long> mlHeroPathGlossy2AuditBudget15ev(96ull);
static std::atomic<unsigned long long> mlHeroPathGlossy2TransportBudget15ev(96ull);
static std::atomic<unsigned long long> mlHeroPathMetal2BounceBudget15ev(64ull);
static std::atomic<unsigned long long> mlHeroPathMetal2DLSBudget15ev(64ull);
static std::atomic<unsigned long long> mlHeroPathGlassReflectBudget15ev(32ull);
static std::atomic<unsigned long long> mlHeroPathGlassTransmitBudget15ev(32ull);
static std::atomic<unsigned long long> mlHeroPathGlassOrdinal15ev(0ull);

static std::atomic<unsigned long long> mlHeroRoughBounceBudget15ev(64ull);
static std::atomic<unsigned long long> mlHeroRoughDLSBudget15ev(64ull);

// 15fg BEGIN
static thread_local bool mlHeroPathComp15fg = true;
static thread_local bool mlHeroPathDiag15fg = false;
static thread_local bool mlHeroPathNativeSample15fg = false;
static std::atomic<unsigned long long> mlHero15fgPrepared[8], mlHero15fgRejected[8],
    mlHero15fgApplied[8], mlHero15fgDetails[8];
static std::atomic<unsigned long long> mlHero15fgEyeResults(0ull), mlHero15fgEyeNonfinite(0ull),
    mlHero15fgAppliedNonfinite(0ull), mlHero15fgRgbLanes(0ull);
static void MLHeroPathResetTranslucent15fg() {
    for (u_int i = 0u; i < 8u; ++i) {
        mlHero15fgPrepared[i].store(0ull); mlHero15fgRejected[i].store(0ull);
        mlHero15fgApplied[i].store(0ull); mlHero15fgDetails[i].store(0ull);
    }
    mlHero15fgEyeResults.store(0ull); mlHero15fgEyeNonfinite.store(0ull);
    mlHero15fgAppliedNonfinite.store(0ull); mlHero15fgRgbLanes.store(0ull);
}
// 15fg END

static void MLHeroPathResetContinuationDiagnostics15ev();
static void MLHeroPathLogIllumBasis15ev(FILE *f);

static void MLHeroPathResetDiagnostics15ev(const bool enabled, const int samplingMode,
        const int wavelengthCount, const bool quarterCycling, const bool compensation15fg,
        const bool diagnostics15fg, const bool lightTracing15fg) {
    mlHeroRoughBounceBudget15ev.store(64ull, std::memory_order_relaxed);
    mlHeroRoughDLSBudget15ev.store(64ull, std::memory_order_relaxed);
    mlHeroPathEyeSamples15ev.store(0ull, std::memory_order_relaxed);
    mlHeroPathPacketSamples15ev.store(0ull, std::memory_order_relaxed);
    mlHeroPathQuarterSamples15ev.store(0ull, std::memory_order_relaxed);
    mlHeroPathLogDetailBudget15ev.store(16ull, std::memory_order_relaxed);
    mlHeroPathCompanionBounceCount15ev.store(0ull, std::memory_order_relaxed);
    mlHeroPathScalarBounceCount15ev.store(0ull, std::memory_order_relaxed);
    mlHeroPathSpectralReflectBounceCount15ev.store(0ull, std::memory_order_relaxed);
    mlHeroPathUnsupportedBounceCount15ev.store(0ull, std::memory_order_relaxed);
    mlHeroPathConnectionFallbackCount15ev.store(0ull, std::memory_order_relaxed);
    mlHeroPathBounceLogBudget15ev.store(48ull, std::memory_order_relaxed);
    mlHeroPathEndpointBudget15ev.store(96ull, std::memory_order_relaxed);
    mlHeroPathEndpointDLS15ev.store(0ull, std::memory_order_relaxed);
    mlHeroPathEndpointFinite15ev.store(0ull, std::memory_order_relaxed);
    mlHeroPathEndpointInfinite15ev.store(0ull, std::memory_order_relaxed);
    mlHeroPathEndpointUnsupported15ev.store(0ull, std::memory_order_relaxed);
    mlHeroPathMaterialAuditBudget15ev.store(160ull, std::memory_order_relaxed);
    mlHeroPathGlossy2AuditBudget15ev.store(96ull, std::memory_order_relaxed);
    mlHeroPathGlossy2TransportBudget15ev.store(96ull, std::memory_order_relaxed);
    mlHeroPathMetal2BounceBudget15ev.store(64ull, std::memory_order_relaxed);
    mlHeroPathMetal2DLSBudget15ev.store(64ull, std::memory_order_relaxed);
    mlHeroPathGlassReflectBudget15ev.store(32ull, std::memory_order_relaxed);
    mlHeroPathGlassTransmitBudget15ev.store(32ull, std::memory_order_relaxed);
    mlHeroPathGlassOrdinal15ev.store(0ull, std::memory_order_relaxed);
    MLHeroPathResetContinuationDiagnostics15ev();
// 15fg BEGIN
    MLHeroPathResetTranslucent15fg();
// 15fg END


    std::lock_guard<std::mutex> lock(mlHeroPathDebugMutex15ev);
    if (!slg::mlhero15gb::Begin("PATH", enabled, ML_HERO_BUILD, ML_HERO_TEST_PHASE)) return;
    FILE *f = fopen(ML_HERO_PATH_LOG_FILE, "w");
    if (!f)
        return;

    fprintf(f, "ML HERO %s | Test phase: %s | Feature: %s\n",
            ML_HERO_BUILD, ML_HERO_TEST_PHASE, ML_HERO_FEATURE);
    fprintf(f, "ML HERO phase 15ev CONFIG: engine=PATHCPU heroEnabled=%d samplingMode=%d wavelengthCount=%d quarterCycling=%d\n",
            enabled ? 1 : 0, samplingMode, wavelengthCount, quarterCycling ? 1 : 0);
    fprintf(f, "ML HERO phase 15ev BASELINE: HERO_1_0_lane0_render=1 packetStart=1 companionTransport=PARALLEL_DIAGNOSTIC endpointSpectralAudit=1 endpointCIEReconstruction=ACTIVE legacyWhiteNormalization=DIAGNOSTIC matteSpectralTransport=ACTIVE glossy2SpectralTransport=ACTIVE metal2SpectralTransport=ACTIVE packetReconstruction=VALIDATED_PATHS dispersionPacket=HERO_LANE0_FALLBACK spectralContinuation=ACTIVE collapse=TX_ONLY reconstruction=CIE_ACTIVE_COUNT\n");
    fprintf(f, "ML HERO phase 15ev NOTE: Scalar Matte BSDF values stay wavelength-independent; chromatic Matte and supported Glossy2 Kd use CLAMP-basis compensation; Glossy2 coating/support stay scalar; unsupported layered cases fall back to HERO-1.0 lane0; CIE radiometric endpoint reconstruction remains active\n");
    fprintf(f, "ML HERO phase 15ev AUDIT: Glossy2 decomposition labels COATED/DIFFUSE_FALLBACK; independent GLOSSY2_TRANSPORT budget logs packet validity, actual per-lane helper results, applied factors including RR, and throughput before/after; GLOSSY2=7 METAL2=8; independent METAL2_BOUNCE/DLS budgets=64 each; full lane validation before update; GLASS diagnostic only: 32 REFLECT + 32 TRANSMIT samples with per-lane physics and actual transport status\n");
    fprintf(f, "ML HERO phase 15ev FIX: packetFallbackFlag=SEPARATE_FROM_LEGACY_SPECTRAL_FLAG Metal2=KEEP_SUPPORTED_PACKET RoughGlass=SPECTRAL_CONTINUATION_TX_COLLAPSE GenericSingleWavelength=KEEP_EXISTING_FALLBACK lightTable=MLHERO_380_780_MATCH_BIDIR\n");
    fprintf(f, "ML HERO phase 15ev LIGHT_TABLE: active=MLHERO_380_780 startNm=380 endNm=780 bins=41 mixing=MATCH_BIDIR lambdaClamp720=REMOVED legacyRGBIllumSPD=DIAGNOSTIC_ONLY reflectiveBasis=UNCHANGED\n");
    if (enabled)
        fprintf(f, "ML HERO phase 15ev ROUGHGLASS_CONFIG: spectralContinuation=ACTIVE start=BEFORE_DLS color=MATCH_BIDIR_15DF dispersionCollapse=TRANSMIT_ONLY nonDispersive=PACKET samplePdfOwner=HERO_SHARED rr=ONCE reconstruction=ACTIVE_COUNT noExtraN=1 diagnosticsBudgets=64_BOUNCE_64_DLS\n");
        fprintf(f, "ML HERO phase 15ev GLASS_CONFIG: start=BEFORE_DLS includesNonDispersive=1 color=MATCH_BIDIR_SPECTRAL sharedPdf=ACTUAL_HERO_SAMPLE rr=ONCE collapse=DISPERSIVE_TX_ONLY activeAverage=1/activeCount extraN=0 nativePacketAudit=LEGACY_PARALLEL_TO_ACTIVE_CONTINUATION\n");
// 15fg BEGIN
    fprintf(f, "phase15fg TRANSLUCENT_CONFIG heroEnabled=%u configuredWavelengths=%d compensation=%u diagnostics=%u basis=MATCH_BIDIR15ff_HERO_WHITE stages=packetBounce,packetDLS,continuationBounce,continuationDLS R_lambda_T_lambda_times_1minusR_lambda=1 rr=ONCE helper=BSDF15fe explicitWrapper=1 unitClamp=0_1 grayBypass=EXACT proposalPDF_MIS_UNCHANGED=1 noRGBFit=1 singleConfiguredLane=LEGACY lightTracingEnabled=%u lightTracingTransport=UNCHANGED_LEGACY BIDIR=UNCHANGED15ff GPU_KERNELS=UNCHANGED\n", enabled ? 1u : 0u, wavelengthCount, compensation15fg ? 1u : 0u, diagnostics15fg ? 1u : 0u, lightTracing15fg ? 1u : 0u);
// 15fg END
    MLHeroPathLogIllumBasis15ev(f);
    fprintf(f, "------------------------------------------------------------\n");
    fclose(f);
}

static void MLHeroPathLogPacket15ev(const unsigned long long sampleOrdinal,
        const int samplingMode, const int wavelengthCount, const bool quarterActive,
        const unsigned int quarter, const float spectralU, const float packetU0) {
    unsigned long long budget = mlHeroPathLogDetailBudget15ev.load(std::memory_order_relaxed);
    while (budget > 0ull && !mlHeroPathLogDetailBudget15ev.compare_exchange_weak(
            budget, budget - 1ull, std::memory_order_relaxed)) { }
    if (budget == 0ull)
        return;

    std::lock_guard<std::mutex> lock(mlHeroPathDebugMutex15ev);
    FILE *f = fopen(ML_HERO_PATH_LOG_FILE, "a");
    if (!f)
        return;

    fprintf(f, "phase15ev packet sample=%llu samplingMode=%d laneCount=%d quarterActive=%d quarter=%u spectralU=%.9g packetU0=%.9g heroLane=0",
            sampleOrdinal, samplingMode, wavelengthCount, quarterActive ? 1 : 0, quarter, spectralU, packetU0);
    for (int lane = 0; lane < wavelengthCount; ++lane)
        fprintf(f, " lane%dLambda=%.9g lane%dWeight=%.9g", lane, GetMLHeroWaveLengthAt((u_int)lane), lane, GetMLHeroSampleWeightAt((u_int)lane));
    fprintf(f, "\n");
    fclose(f);
}

static inline bool MLHeroPathIsScalarSpectrum15ev(const Spectrum &v, const float eps = 1e-6f) {
    const float m = Max(1.f, Max(fabsf(v.c[0]), Max(fabsf(v.c[1]), fabsf(v.c[2]))));
    return (fabsf(v.c[0] - v.c[1]) <= eps * m) && (fabsf(v.c[0] - v.c[2]) <= eps * m);
}

static inline float MLHeroPathRGBReflSample380_780_15ev(const RGBReflSPD &spd, const float lambda) {
    return spd.Sample(Clamp(lambda, 380.f, 719.999f));
}

// 15fg BEGIN
static inline float MLHeroPathTranslucentBasis15fg(const Spectrum &rgb,
        const float lambda) {
    const float r = 0.92828033230552f * rgb.c[0] +
        0.057298551042327f * rgb.c[1] + 0.015214109441961f * rgb.c[2];
    const float g = 0.022022679206621f * rgb.c[0] +
        0.96221704129375f * rgb.c[1] + 0.018368484576659f * rgb.c[2];
    const float b = -0.0043974443528483f * rgb.c[0] -
        0.020504057940599f * rgb.c[1] + 1.0229715165802f * rgb.c[2];
    static const RGBReflSPD red(RGBColor(1.f, 0.f, 0.f));
    static const RGBReflSPD green(RGBColor(0.f, 1.f, 0.f));
    static const RGBReflSPD blue(RGBColor(0.f, 0.f, 1.f));
    return Max(0.f, r * MLHeroPathRGBReflSample380_780_15ev(red, lambda) +
        g * MLHeroPathRGBReflSample380_780_15ev(green, lambda) +
        b * MLHeroPathRGBReflSample380_780_15ev(blue, lambda));
}

static bool MLHeroPathStrictNeutral15fg(const Spectrum &v) {
    return std::isfinite(v.c[0]) && v.c[0] >= 0.f &&
        v.c[0] == v.c[1] && v.c[1] == v.c[2];
}
static float MLHeroPathTranslucentUnit15fg(const Spectrum &rgb,
        const RGBReflSPD *spd, float lambda, bool compensation) {
    if (!spd) return Clamp(rgb.c[0], 0.f, 1.f);
    return Clamp(compensation ? MLHeroPathTranslucentBasis15fg(rgb, lambda) :
        MLHeroPathRGBReflSample380_780_15ev(*spd, lambda), 0.f, 1.f);
}
// All lanes are prepared and validated before any caller's array is changed.
// Native Sample/Evaluate own geometry, event support and proposal PDFs.
static bool MLHeroPathTranslucentLanes15fg(const BSDF &bsdf, const Vector &dir,
        BSDFEvent event, bool sampled, u_int count, float rrProb,
        bool compensation, u_int stage, float factors[8]) {
    if (stage >= 4u || !factors || bsdf.GetMaterialType() != MATTETRANSLUCENT) return false;
    const u_int index = stage * 2u + ((event & TRANSMIT) ? 1u : 0u);
    Spectrum kr, kt; float wrapper = 0.f;
    bool ok = count > 0u && count <= 8u && std::isfinite(rrProb) && rrProb > 0.f &&
        bsdf.GetMLHeroMatteTranslucentFactors15fe(dir, event, sampled, &kr, &kt, &wrapper);
    if (ok) {
        for (u_int j = 0u; j < 3u; ++j)
            ok = ok && std::isfinite(kr.c[j]) && kr.c[j] >= 0.f &&
                std::isfinite(kt.c[j]) && kt.c[j] >= 0.f;
        ok = ok && std::isfinite(wrapper) && wrapper >= 0.f;
        const Spectrum nativeTransmission = kt * (Spectrum(1.f) - kr);
        const bool fullyReflecting = MLHeroPathStrictNeutral15fg(kr) && kr.c[0] == 1.f;
        if (nativeTransmission.Black() && !kt.Black() && !fullyReflecting) ok = false;
    }
    std::unique_ptr<RGBReflSPD> rSPD, tSPD;
    if (ok) {
        if (!MLHeroPathStrictNeutral15fg(kr)) rSPD.reset(new RGBReflSPD(RGBColor(kr.c[0],kr.c[1],kr.c[2])));
        if ((event & TRANSMIT) && !MLHeroPathStrictNeutral15fg(kt)) tSPD.reset(new RGBReflSPD(RGBColor(kt.c[0],kt.c[1],kt.c[2])));
    }
    float pending[8] = {};
    for (u_int lane = 0u; ok && lane < count; ++lane) {
        const float lambda = GetMLHeroWaveLengthAt(lane);
        ok = std::isfinite(lambda) && lambda >= 380.f && lambda <= 780.f;
        if (!ok) break;
        const float r = MLHeroPathTranslucentUnit15fg(kr, rSPD.get(), lambda, compensation);
        const float t = (event & TRANSMIT) ? MLHeroPathTranslucentUnit15fg(kt, tSPD.get(), lambda, compensation) : 0.f;
        pending[lane] = wrapper * ((event & TRANSMIT) ? t * (1.f-r) : r) / rrProb;
        ok = std::isfinite(pending[lane]) && pending[lane] >= 0.f;
    }
    if (ok) for (u_int lane = 0u; lane < count; ++lane) factors[lane] = pending[lane];
    if (mlHeroPathDiag15fg) {
        mlHero15fgPrepared[index].fetch_add(1ull, std::memory_order_relaxed);
        if (!ok) mlHero15fgRejected[index].fetch_add(1ull, std::memory_order_relaxed);
        if (mlHero15fgDetails[index].fetch_add(1ull, std::memory_order_relaxed) < 12ull) {
            std::lock_guard<std::mutex> lock(mlHeroPathDebugMutex15ev);
            FILE *f=fopen(ML_HERO_PATH_LOG_FILE,"a");
            if (f) {
                fprintf(f,"phase15fg TRANSLUCENT_FACTORS stage=%u transmit=%u materialID=%u sampled=%u count=%u compensation=%u helperOK=%u wrapper=%.9g rrProb=%.9g Kr=(%.9g,%.9g,%.9g) Kt=(%.9g,%.9g,%.9g) actualApplied=SUMMARY_only",stage,(event&TRANSMIT)?1u:0u,bsdf.GetMaterialID(),sampled?1u:0u,count,compensation?1u:0u,ok?1u:0u,wrapper,rrProb,kr.c[0],kr.c[1],kr.c[2],kt.c[0],kt.c[1],kt.c[2]);
                for(u_int lane=0u;ok&&lane<count;++lane) fprintf(f," lane%uLambda=%.9g lane%uFactor=%.9g",lane,GetMLHeroWaveLengthAt(lane),lane,pending[lane]);
                fprintf(f,"\n"); fclose(f);
            }
        }
    }
    return ok;
}
static void MLHeroPathTranslucentApplied15fg(u_int stage, BSDFEvent event,
        const Spectrum *values, u_int count, bool laneThroughput) {
    if (!mlHeroPathDiag15fg || stage>=4u) return;
    mlHero15fgApplied[stage*2u+((event&TRANSMIT)?1u:0u)].fetch_add(1ull,std::memory_order_relaxed);
    for(u_int lane=0u;values&&lane<count&&lane<8u;++lane) {
        if(values[lane].IsNaN() || values[lane].IsInf()) mlHero15fgAppliedNonfinite.fetch_add(1ull,std::memory_order_relaxed);
        if(laneThroughput && !MLHeroPathStrictNeutral15fg(values[lane])) mlHero15fgRgbLanes.fetch_add(1ull,std::memory_order_relaxed);
    }
}
// 15fg END

// Same measured CLAMP-basis inverse used by the validated BIDIR HERO path.
// Kept local to PATH so HERO OFF and LuxCore's stock RGBReflSPD remain untouched.
static inline float MLHeroPathBasisCompensatedReflectance15ev(const Spectrum &rgb, const float lambda) {
    const float coeffR = 0.81789457798f * rgb.c[0] +
        0.006252348423f * rgb.c[1] + 0.007401828188f * rgb.c[2];
    const float coeffG = 0.02175390534f * rgb.c[0] +
        1.00719189644f * rgb.c[1] + 0.01943249628f * rgb.c[2];
    const float coeffB = -0.000957758341f * rgb.c[0] -
        0.02197206020f * rgb.c[1] + 1.10662686825f * rgb.c[2];

    static const RGBReflSPD redSPD(RGBColor(1.f, 0.f, 0.f));
    static const RGBReflSPD greenSPD(RGBColor(0.f, 1.f, 0.f));
    static const RGBReflSPD blueSPD(RGBColor(0.f, 0.f, 1.f));

    const float raw =
        coeffR * MLHeroPathRGBReflSample380_780_15ev(redSPD, lambda) +
        coeffG * MLHeroPathRGBReflSample380_780_15ev(greenSPD, lambda) +
        coeffB * MLHeroPathRGBReflSample380_780_15ev(blueSPD, lambda);
    return Max(0.f, raw);
}


// 15fx: this wrapper is only for material Kd, never a sampled BSDF multiplier.
static float MLHeroPathKdReflectance15fx(const Spectrum &kdInput,const float lambda) {
    const Spectrum kd=kdInput.Clamp(0.f,1.f);
    if (kd.c[0]==kd.c[1] && kd.c[1]==kd.c[2]) return kd.c[0];
    return MLHeroUnitReflectance15fx(MLHeroPathBasisCompensatedReflectance15ev(kd,lambda));
}

// 15ev: same scalar bypass and compensated chromatic basis as BIDIR 15df.
static float MLHeroPathRoughColor15ev(const Spectrum &value, const float wavelength) {
    if (MLHeroPathIsScalarSpectrum15ev(value))
        return (value.c[0] + value.c[1] + value.c[2]) / 3.f;
    return MLHeroPathBasisCompensatedReflectance15ev(value, wavelength);
}


// 15ev: split a Matte BSDF value into achromatic transport * spectral Kd.
// This mirrors the validated BIDIR Matte semantics without touching stock LuxCore.
static inline bool MLHeroPathMatteLaneMultiplier15ev(const Spectrum &kdInput,
        const Spectrum &bsdfValue, const float lambda, float *laneMultiplier,
        float *fitError = nullptr) {
    // 15ev: match BIDIR's scalar-BSDF bypass. A neutral Matte multiplier
    // is wavelength-independent, including direct-light evaluation and RR.
    // Chromatic Matte continues through the existing compensated Kd fit.
    if (MLHeroPathIsScalarSpectrum15ev(bsdfValue)) {
        *laneMultiplier = (bsdfValue.c[0] + bsdfValue.c[1] + bsdfValue.c[2]) / 3.f;
        if (fitError)
            *fitError = 0.f;
        return true;
    }

    const Spectrum kd = kdInput.Clamp(0.f, 1.f);
    const float denom = kd.c[0] * kd.c[0] + kd.c[1] * kd.c[1] + kd.c[2] * kd.c[2];
    if (denom <= 1e-12f)
        return false;

    const float scalar = (bsdfValue.c[0] * kd.c[0] +
            bsdfValue.c[1] * kd.c[1] + bsdfValue.c[2] * kd.c[2]) / denom;
    const Spectrum recon = kd * scalar;
    const Spectrum residual = bsdfValue - recon;
    const float maxResidual = Max(fabsf(residual.c[0]), Max(fabsf(residual.c[1]), fabsf(residual.c[2])));
    const float maxValue = Max(1e-9f, Max(fabsf(bsdfValue.c[0]), Max(fabsf(bsdfValue.c[1]), fabsf(bsdfValue.c[2]))));
    const float relError = maxResidual / maxValue;
    if (fitError)
        *fitError = relError;

    // Only accept the decomposition when the RGB BSDF really is Kd times one scalar.
    if (relError > 1e-4f)
        return false;

    *laneMultiplier = scalar * MLHeroPathKdReflectance15fx(kd, lambda);
    return true;
}

// 15ev: Glossy2 support terms should remain achromatic while only the diffuse/base Kd
// is spectralized. This mirrors the already validated Bidir decomposition, but keeps
// CPU Path conservative: if any support term or reconstruction residual is chromatic,
// the packet is rejected and the classic lane0 path remains authoritative.
static inline bool MLHeroPathNearScalarGlossy2Support15ev(const Spectrum &v,
        const float relEps = 1e-4f) {
    const float maxAbs = Max(fabsf(v.c[0]), Max(fabsf(v.c[1]), fabsf(v.c[2])));
    const float scale = Max(1e-8f, maxAbs);
    const float spread = Max(fabsf(v.c[0] - v.c[1]),
        Max(fabsf(v.c[0] - v.c[2]), fabsf(v.c[1] - v.c[2])));
    return spread <= relEps * scale;
}

static inline bool MLHeroPathGlossy2LaneMultiplier15ev(const BSDF &bsdf,
        const Vector &sampledDir, const float pdfW, const BSDFEvent event,
        const Spectrum &classicPhysicalValue, const float lambda,
        float *laneMultiplier, float *baseFitError = nullptr) {
    if (!laneMultiplier || (bsdf.GetMaterialType() != GLOSSY2) ||
            !(event & REFLECT) || (event & SPECULAR))
        return false;

    Spectrum kd, ks, schlickS, absorption;
    Spectrum base, coating, combined;
    float wrapper = 1.f;
    const bool ok = bsdf.EvaluateMLHeroGlossy2DebugSampleComponents(
        sampledDir, pdfW, &kd, &ks, &schlickS, &absorption,
        &base, &coating, &wrapper, &combined);

    if (ok) {
        const Spectrum residual = classicPhysicalValue - combined;
        const float classicMax = Max(1e-9f, Max(fabsf(classicPhysicalValue.c[0]),
            Max(fabsf(classicPhysicalValue.c[1]), fabsf(classicPhysicalValue.c[2]))));
        const float residualMean = (residual.c[0] + residual.c[1] + residual.c[2]) / 3.f;
        const Spectrum residualChroma = residual - Spectrum(residualMean);
        const float residualChromaMax = Max(fabsf(residualChroma.c[0]),
            Max(fabsf(residualChroma.c[1]), fabsf(residualChroma.c[2])));
        const bool residualOK = MLHeroPathNearScalarGlossy2Support15ev(residual) ||
            ((residualChromaMax / classicMax) <= 1e-4f);

        const bool scalarSupport =
            MLHeroPathNearScalarGlossy2Support15ev(ks) &&
            MLHeroPathNearScalarGlossy2Support15ev(schlickS) &&
            MLHeroPathNearScalarGlossy2Support15ev(absorption) &&
            MLHeroPathNearScalarGlossy2Support15ev(coating) && residualOK;
        if (!scalarSupport)
            return false;

        kd = kd.Clamp(0.f, 1.f);
        const float kdDenom = kd.c[0] * kd.c[0] + kd.c[1] * kd.c[1] + kd.c[2] * kd.c[2];
        if (kdDenom <= 1e-12f)
            return false;

        const float baseFactor = (base.c[0] * kd.c[0] + base.c[1] * kd.c[1] +
            base.c[2] * kd.c[2]) / kdDenom;
        const Spectrum baseRecon = kd * baseFactor;
        const Spectrum baseError = base - baseRecon;
        const float baseErrMax = Max(fabsf(baseError.c[0]),
            Max(fabsf(baseError.c[1]), fabsf(baseError.c[2])));
        const float baseMax = Max(1e-9f, Max(fabsf(base.c[0]),
            Max(fabsf(base.c[1]), fabsf(base.c[2]))));
        const float relError = baseErrMax / baseMax;
        if (baseFitError)
            *baseFitError = relError;
        if (relError > 1e-4f)
            return false;

        const float coatingScalar = (coating.c[0] + coating.c[1] + coating.c[2]) / 3.f;
        const float kdSpectral = MLHeroPathKdReflectance15fx(kd, lambda);
        *laneMultiplier = coatingScalar + residualMean + baseFactor * kdSpectral;
        return std::isfinite(*laneMultiplier) && (*laneMultiplier >= 0.f);
    }

    // Glossy2 back face: coating is skipped and the material reduces to Kd * scalar.
    if (event & DIFFUSE) {
        const Spectrum kdBack = bsdf.Albedo().Clamp(0.f, 1.f);
        const float kdDenom = kdBack.c[0] * kdBack.c[0] + kdBack.c[1] * kdBack.c[1] +
            kdBack.c[2] * kdBack.c[2];
        if (kdDenom <= 1e-12f)
            return false;
        const float factor = (classicPhysicalValue.c[0] * kdBack.c[0] +
            classicPhysicalValue.c[1] * kdBack.c[1] + classicPhysicalValue.c[2] * kdBack.c[2]) / kdDenom;
        const Spectrum recon = kdBack * factor;
        const Spectrum err = classicPhysicalValue - recon;
        const float errMax = Max(fabsf(err.c[0]), Max(fabsf(err.c[1]), fabsf(err.c[2])));
        const float valueMax = Max(1e-9f, Max(fabsf(classicPhysicalValue.c[0]),
            Max(fabsf(classicPhysicalValue.c[1]), fabsf(classicPhysicalValue.c[2]))));
        const float relError = errMax / valueMax;
        if (baseFitError)
            *baseFitError = relError;
        if (relError > 1e-5f)
            return false;
        *laneMultiplier = factor * MLHeroPathKdReflectance15fx(kdBack, lambda);
        return std::isfinite(*laneMultiplier) && (*laneMultiplier >= 0.f);
    }

    return false;
}

static void MLHeroPathLogGlossy2Decomposition15ev(const u_int depth, const BSDF &bsdf,
        const Vector &sampledDir, const float pdfW, const BSDFEvent event,
        const Spectrum &classicPhysicalValue, const float rrProb) {
    if ((bsdf.GetMaterialType() != GLOSSY2) || !(event & REFLECT) || (event & SPECULAR))
        return;

    unsigned long long budget = mlHeroPathGlossy2AuditBudget15ev.load(std::memory_order_relaxed);
    while (budget > 0ull && !mlHeroPathGlossy2AuditBudget15ev.compare_exchange_weak(
            budget, budget - 1ull, std::memory_order_relaxed)) { }
    if (budget == 0ull)
        return;

    Spectrum kd, ks, schlickS, absorption, base, coating, combined;
    float wrapper = 1.f;
    const bool componentsOK = bsdf.EvaluateMLHeroGlossy2DebugSampleComponents(
        sampledDir, pdfW, &kd, &ks, &schlickS, &absorption,
        &base, &coating, &wrapper, &combined);

    const Spectrum residual = componentsOK ? (classicPhysicalValue - combined) : Spectrum();
    const float residualMean = (residual.c[0] + residual.c[1] + residual.c[2]) / 3.f;
    const Spectrum residualChroma = residual - Spectrum(residualMean);
    const float classicMax = Max(1e-9f, Max(fabsf(classicPhysicalValue.c[0]),
        Max(fabsf(classicPhysicalValue.c[1]), fabsf(classicPhysicalValue.c[2]))));
    const float residualChromaMax = Max(fabsf(residualChroma.c[0]),
        Max(fabsf(residualChroma.c[1]), fabsf(residualChroma.c[2])));

    const bool ksScalar = componentsOK && MLHeroPathNearScalarGlossy2Support15ev(ks);
    const bool schlickScalar = componentsOK && MLHeroPathNearScalarGlossy2Support15ev(schlickS);
    const bool absorptionScalar = componentsOK && MLHeroPathNearScalarGlossy2Support15ev(absorption);
    const bool coatingScalar = componentsOK && MLHeroPathNearScalarGlossy2Support15ev(coating);
    const bool residualScalar = componentsOK && (MLHeroPathNearScalarGlossy2Support15ev(residual) ||
        ((residualChromaMax / classicMax) <= 1e-4f));

    Spectrum kdClamped = kd.Clamp(0.f, 1.f);
    const float kdDenom = kdClamped.c[0] * kdClamped.c[0] + kdClamped.c[1] * kdClamped.c[1] + kdClamped.c[2] * kdClamped.c[2];
    float baseFactor = 0.f;
    float baseFitError = -1.f;
    if (componentsOK && kdDenom > 1e-12f) {
        baseFactor = (base.c[0] * kdClamped.c[0] + base.c[1] * kdClamped.c[1] + base.c[2] * kdClamped.c[2]) / kdDenom;
        const Spectrum baseRecon = kdClamped * baseFactor;
        const Spectrum baseError = base - baseRecon;
        const float baseErrMax = Max(fabsf(baseError.c[0]), Max(fabsf(baseError.c[1]), fabsf(baseError.c[2])));
        const float baseMax = Max(1e-9f, Max(fabsf(base.c[0]), Max(fabsf(base.c[1]), fabsf(base.c[2]))));
        baseFitError = baseErrMax / baseMax;
    }

    float lane0 = 0.f;
    float helperFitError = 0.f;
    const bool accepted = MLHeroPathGlossy2LaneMultiplier15ev(bsdf, sampledDir, pdfW, event,
        classicPhysicalValue, GetMLHeroWaveLengthAt(0), &lane0, &helperFitError);

    std::lock_guard<std::mutex> lock(mlHeroPathDebugMutex15ev);
    FILE *f = fopen(ML_HERO_PATH_LOG_FILE, "a");
    if (!f)
        return;
    fprintf(f,
        "phase15ev GLOSSY2_DECOMP depth=%u event=%u rrProb=%.9g componentsOK=%d result=%s branch=%s "
        "kd=(%.9g,%.9g,%.9g) ks=(%.9g,%.9g,%.9g) schlickS=(%.9g,%.9g,%.9g) absorption=(%.9g,%.9g,%.9g) "
        "base=(%.9g,%.9g,%.9g) coating=(%.9g,%.9g,%.9g) combined=(%.9g,%.9g,%.9g) classicPhysical=(%.9g,%.9g,%.9g) "
        "residual=(%.9g,%.9g,%.9g) wrapper=%.9g checks[ks=%d schlick=%d absorption=%d coating=%d residual=%d] "
        "baseFactor=%.9g baseFitError=%.9g helperFitError=%.9g lane0Lambda=%.9g lane0Multiplier=%.9g\n",
        depth, (u_int)event, rrProb, componentsOK ? 1 : 0, accepted ? "ACCEPT" : "REJECT",
        componentsOK ? "COATED" : ((event & DIFFUSE) ? "DIFFUSE_FALLBACK" : "NO_COMPONENTS"),
        kd.c[0], kd.c[1], kd.c[2], ks.c[0], ks.c[1], ks.c[2], schlickS.c[0], schlickS.c[1], schlickS.c[2],
        absorption.c[0], absorption.c[1], absorption.c[2], base.c[0], base.c[1], base.c[2], coating.c[0], coating.c[1], coating.c[2],
        combined.c[0], combined.c[1], combined.c[2], classicPhysicalValue.c[0], classicPhysicalValue.c[1], classicPhysicalValue.c[2],
        residual.c[0], residual.c[1], residual.c[2], wrapper,
        ksScalar ? 1 : 0, schlickScalar ? 1 : 0, absorptionScalar ? 1 : 0, coatingScalar ? 1 : 0, residualScalar ? 1 : 0,
        baseFactor, baseFitError, helperFitError, GetMLHeroWaveLengthAt(0), lane0);
    fclose(f);
}

// Independent budget: floor bounces cannot hide the Glossy2 update.
// Factors and throughputs are captured at the actual transport call site.
static void MLHeroPathLogGlossy2Transport15ev(const u_int depth,
        const BSDFEvent event, const bool validBefore, const bool validAfter,
        const char *mode, const float rrProb, const u_int laneCount,
        const bool helperOK[8], const float appliedFactor[8],
        const Spectrum before[8], const Spectrum after[8]) {
    unsigned long long budget = mlHeroPathGlossy2TransportBudget15ev.load(std::memory_order_relaxed);
    while (budget > 0ull && !mlHeroPathGlossy2TransportBudget15ev.compare_exchange_weak(
            budget, budget - 1ull, std::memory_order_relaxed)) { }
    if (budget == 0ull)
        return;
    std::lock_guard<std::mutex> lock(mlHeroPathDebugMutex15ev);
    FILE *f = fopen(ML_HERO_PATH_LOG_FILE, "a");
    if (!f)
        return;
    fprintf(f, "phase15ev GLOSSY2_TRANSPORT depth=%u event=%u packetBefore=%d packetAfter=%d mode=%s factorsApplied=%d rrProb=%.9g laneCount=%u",
        depth, (u_int)event, validBefore ? 1 : 0, validAfter ? 1 : 0,
        mode, (validBefore && validAfter) ? 1 : 0, rrProb, laneCount);
    for (u_int lane = 0; lane < laneCount; ++lane)
        fprintf(f, " lane%uLambda=%.9g lane%uHelperOK=%d lane%uAppliedFactor=%.9g lane%uTPBefore=%.9g lane%uTPAfter=%.9g",
            lane, GetMLHeroWaveLengthAt(lane), lane, helperOK[lane] ? 1 : 0,
            lane, appliedFactor[lane], lane, before[lane].c[0], lane, after[lane].c[0]);
    fprintf(f, "\n");
    fclose(f);
}

// Shared Metal2 BSDF helpers return a monochromatic Spectrum. Reject the
// whole packet before any throughput update if a lane is invalid.
static bool MLHeroPathMetal2Lanes15ev(const BSDF &bsdf, const Vector &dir,
        const bool sampled, const u_int laneCount, const float rrProb,
        bool laneOK[8], float laneValue[8]) {
    bool allOK = laneCount > 0u && laneCount <= 8u &&
        std::isfinite(rrProb) && rrProb > 0.f;
    for (u_int lane = 0; lane < laneCount && lane < 8u; ++lane) {
        Spectrum value;
        const bool ok = sampled ?
            bsdf.EvaluateMLHeroMetal2SampleAtWaveLength(dir, GetMLHeroWaveLengthAt(lane), &value) :
            bsdf.EvaluateMLHeroMetal2AtWaveLength(dir, GetMLHeroWaveLengthAt(lane), &value);
        laneOK[lane] = ok && !value.IsNaN() && !value.IsInf() && !value.IsNeg() &&
            MLHeroPathIsScalarSpectrum15ev(value);
        if (laneOK[lane]) {
            laneValue[lane] = (value.c[0] + value.c[1] + value.c[2]) / 3.f;
            // Sample helper is pre-RR; survival compensation belongs here once.
            if (sampled && rrProb > 0.f)
                laneValue[lane] /= rrProb;
            laneOK[lane] = std::isfinite(laneValue[lane]) && laneValue[lane] >= 0.f;
        }
        allOK = allOK && laneOK[lane];
    }
    return allOK;
}

static void MLHeroPathLogMetal2Audit15ev(const char *kind, const BSDF &bsdf,
        const bool sampled, const bool packetBefore, const bool packetAfter,
        const bool helpersCalled, const bool applied, const char *mode,
        const float rrProb, const u_int laneCount, const bool laneOK[8],
        const float laneValue[8], const Spectrum before[8], const Spectrum after[8]) {
    std::atomic<unsigned long long> &counter = sampled ?
        mlHeroPathMetal2BounceBudget15ev : mlHeroPathMetal2DLSBudget15ev;
    unsigned long long budget = counter.load(std::memory_order_relaxed);
    while (budget > 0ull && !counter.compare_exchange_weak(
            budget, budget - 1ull, std::memory_order_relaxed)) { }
    if (budget == 0ull)
        return;
    std::lock_guard<std::mutex> lock(mlHeroPathDebugMutex15ev);
    FILE *f = fopen(ML_HERO_PATH_LOG_FILE, "a");
    if (!f)
        return;
    fprintf(f, "phase15ev METAL2_%s materialID=%u materialType=%u packetBefore=%d packetAfter=%d helpersCalled=%d applied=%d mode=%s rrProb=%.9g laneCount=%u legacySpectralUsed=%d packetFallbackUsed=%d",
        kind, bsdf.GetMaterialID(), (u_int)bsdf.GetMaterialType(),
        packetBefore ? 1 : 0, packetAfter ? 1 : 0, helpersCalled ? 1 : 0,
        applied ? 1 : 0, mode, rrProb, laneCount,
        GetMLDispersionUsed() ? 1 : 0, GetMLHeroPacketFallbackUsed() ? 1 : 0);
    for (u_int lane = 0; lane < laneCount; ++lane) {
        fprintf(f, " lane%uLambda=%.9g lane%uHelperOK=%d lane%uValue=%.9g",
            lane, GetMLHeroWaveLengthAt(lane), lane, laneOK[lane] ? 1 : 0, lane, laneValue[lane]);
        if (sampled)
            fprintf(f, " lane%uTPBefore=%.9g lane%uTPAfter=%.9g",
                lane, before[lane].c[0], lane, after[lane].c[0]);
    }
    fprintf(f, "\n");
    fclose(f);
}

// GLASS diagnostics only: the material's debug helper never replaces the
// sampled transport. Separate reflection/transmission budgets avoid starvation.
static unsigned long long MLHeroPathLogGlassSample15ev(const u_int depth,
        const BSDF &bsdf, const Vector &dir, const BSDFEvent event,
        const float pdfW, const Spectrum &classicSample, const bool packetValid,
        const bool snapshotTaken, const u_int laneCount) {
    if (bsdf.GetMaterialType() != GLASS || !(event & SPECULAR))
        return 0ull;
    std::atomic<unsigned long long> &counter = (event & TRANSMIT) ?
        mlHeroPathGlassTransmitBudget15ev : mlHeroPathGlassReflectBudget15ev;
    unsigned long long budget = counter.load(std::memory_order_relaxed);
    while (budget > 0ull && !counter.compare_exchange_weak(
            budget, budget - 1ull, std::memory_order_relaxed)) { }
    if (budget == 0ull)
        return 0ull;
    const unsigned long long id = mlHeroPathGlassOrdinal15ev.fetch_add(1ull,
        std::memory_order_relaxed) + 1ull;
    std::lock_guard<std::mutex> lock(mlHeroPathDebugMutex15ev);
    FILE *f = fopen(ML_HERO_PATH_LOG_FILE, "a");
    if (!f)
        return 0ull;
    fprintf(f, "phase15ev GLASS_SAMPLE id=%llu depth=%u materialID=%u event=%u branch=%s samplePdfW=%.9g packetBefore=%d snapshotBefore=%d dispersionUsed=%d packetFallbackUsed=%d laneCount=%u classicPreRR=(%.9g,%.9g,%.9g) scalar=%d\n",
        id, depth, bsdf.GetMaterialID(), (u_int)event,
        (event & TRANSMIT) ? "TRANSMIT" : "REFLECT", pdfW,
        packetValid ? 1 : 0, snapshotTaken ? 1 : 0, GetMLDispersionUsed() ? 1 : 0,
        GetMLHeroPacketFallbackUsed() ? 1 : 0, laneCount, classicSample.c[0], classicSample.c[1], classicSample.c[2],
        MLHeroPathIsScalarSpectrum15ev(classicSample) ? 1 : 0);
    for (u_int lane = 0; lane < laneCount; ++lane) {
        float exteriorIOR = 0.f, interiorBase = 0.f, interiorLambda = 0.f;
        float fresnel = 0.f, eta = 0.f, eta2 = 0.f, transport = 0.f;
        float delta = 0.f, cosFixed = 0.f, sinI2 = 0.f, sinT2 = 0.f, eventPdf = 0.f;
        bool tir = false;
        Spectrum candidate;
        const bool ok = bsdf.EvaluateMLHeroGlassDebugAtWaveLength(dir, event, pdfW,
            GetMLHeroWaveLengthAt(lane), &exteriorIOR, &interiorBase, &interiorLambda,
            &fresnel, &eta, &eta2, &transport, &delta, &candidate,
            &cosFixed, &sinI2, &sinT2, &tir, &eventPdf);
        fprintf(f, "phase15ev GLASS_LANE id=%llu lane=%u lambda=%.9g helperOK=%d exteriorIOR=%.9g interiorBase=%.9g interiorLambda=%.9g fresnelR=%.9g eta=%.9g eta2=%.9g transport=%.9g directionDelta=%.9g cosFixed=%.9g sinI2=%.9g sinT2=%.9g tir=%d laneEventPdfW=%.9g candidatePreRR=(%.9g,%.9g,%.9g) candidateApplied=0\n",
            id, lane, GetMLHeroWaveLengthAt(lane), ok ? 1 : 0,
            exteriorIOR, interiorBase, interiorLambda, fresnel, eta, eta2,
            transport, delta, cosFixed, sinI2, sinT2, tir ? 1 : 0, eventPdf,
            candidate.c[0], candidate.c[1], candidate.c[2]);
    }
    fclose(f);
    return id;
}

static void MLHeroPathLogGlassTransport15ev(const unsigned long long id,
        const char *mode, const bool packetValid, const bool snapshotTaken,
        const float rrProb, const u_int laneCount, const Spectrum before[8],
        const Spectrum after[8]) {
    if (id == 0ull)
        return;
    std::lock_guard<std::mutex> lock(mlHeroPathDebugMutex15ev);
    FILE *f = fopen(ML_HERO_PATH_LOG_FILE, "a");
    if (!f)
        return;
    fprintf(f, "phase15ev GLASS_TRANSPORT id=%llu mode=%s packetAfter=%d snapshotTaken=%d dispersionUsed=%d packetFallbackUsed=%d rrProb=%.9g laneCount=%u",
        id, mode, packetValid ? 1 : 0, snapshotTaken ? 1 : 0,
        GetMLDispersionUsed() ? 1 : 0, GetMLHeroPacketFallbackUsed() ? 1 : 0, rrProb, laneCount);
    for (u_int lane = 0; lane < laneCount; ++lane)
        fprintf(f, " lane%uTPBefore=%.9g lane%uTPAfter=%.9g",
            lane, before[lane].c[0], lane, after[lane].c[0]);
    fprintf(f, "\n");
    fclose(f);
}

static void MLHeroPathLogMaterialAudit15ev(const u_int depth, const BSDF &bsdf,
        const BSDFEvent sampledEvent, const Spectrum &bsdfSample, const float pdfW,
        const float cosSampledDir) {
    // 15ev is intentionally biased toward non-Matte materials so a small log budget
    // cannot be consumed by the large Matte parts of the Lu Yu scene.
    if (bsdf.GetMaterialType() == MATTE)
        return;

    unsigned long long budget = mlHeroPathMaterialAuditBudget15ev.load(std::memory_order_relaxed);
    while (budget > 0ull && !mlHeroPathMaterialAuditBudget15ev.compare_exchange_weak(
            budget, budget - 1ull, std::memory_order_relaxed)) { }
    if (budget == 0ull)
        return;

    const Spectrum albedo = bsdf.Albedo();
    const BSDFEvent eventTypes = bsdf.GetEventTypes();

    std::lock_guard<std::mutex> lock(mlHeroPathDebugMutex15ev);
    FILE *f = fopen(ML_HERO_PATH_LOG_FILE, "a");
    if (!f)
        return;

    fprintf(f,
        "phase15ev MATERIAL_AUDIT depth=%u materialID=%u materialType=%u name=\"%s\" "
        "sampledEvent=%u eventTypes=%u flags[R=%u T=%u D=%u G=%u S=%u] glossiness=%.9g "
        "albedo=(%.9g,%.9g,%.9g) bsdfSample=(%.9g,%.9g,%.9g) pdfW=%.9g cos=%.9g\n",
        depth, bsdf.GetMaterialID(), (u_int)bsdf.GetMaterialType(), bsdf.GetMaterialName().c_str(),
        (u_int)sampledEvent, (u_int)eventTypes,
        (sampledEvent & REFLECT) ? 1u : 0u,
        (sampledEvent & TRANSMIT) ? 1u : 0u,
        (sampledEvent & DIFFUSE) ? 1u : 0u,
        (sampledEvent & GLOSSY) ? 1u : 0u,
        (sampledEvent & SPECULAR) ? 1u : 0u,
        bsdf.GetGlossiness(),
        albedo.c[0], albedo.c[1], albedo.c[2],
        bsdfSample.c[0], bsdfSample.c[1], bsdfSample.c[2], pdfW, cosSampledDir);
    fclose(f);
}

static void MLHeroPathLogBounce15ev(const u_int depth, const BSDFEvent event,
        const MaterialType materialType, const char *mode, const bool packetValid,
        const u_int laneCount, const Spectrum laneTp[8]) {
    unsigned long long budget = mlHeroPathBounceLogBudget15ev.load(std::memory_order_relaxed);
    while (budget > 0ull && !mlHeroPathBounceLogBudget15ev.compare_exchange_weak(
            budget, budget - 1ull, std::memory_order_relaxed)) { }
    if (budget == 0ull)
        return;

    std::lock_guard<std::mutex> lock(mlHeroPathDebugMutex15ev);
    FILE *f = fopen(ML_HERO_PATH_LOG_FILE, "a");
    if (!f)
        return;

    fprintf(f, "phase15ev bounce depth=%u event=%u materialType=%u mode=%s packetValid=%d laneCount=%u",
            depth, (u_int)event, (u_int)materialType, mode, packetValid ? 1 : 0, laneCount);
    for (u_int lane = 0; lane < laneCount; ++lane)
        fprintf(f, " lane%uTP=%.9g lane%uLambda=%.9g", lane, laneTp[lane].c[0], lane, GetMLHeroWaveLengthAt(lane));
    fprintf(f, "\n");
    fclose(f);
}


//------------------------------------------------------------------------------
// ML HERO 15ev - CIE radiometric spectral endpoint reconstruction
//------------------------------------------------------------------------------

// Exact Smits mixing/interpolation used by BIDIR for the existing ML-HERO
// 380..780 nm illuminant table. Do not extend the stock 380..720 nm SPD by
// holding its last value: all active light endpoints sample this basis.
static float MLHeroPathIllumBasisSample15ev(const RGBColor &s, const float lambda) {
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

static inline float MLHeroPathRGBIllumSample15ev(const Spectrum &rgb, const float lambda) {
    const RGBColor nonNegative(Max(0.f, rgb.c[0]), Max(0.f, rgb.c[1]), Max(0.f, rgb.c[2]));
    return MLHeroPathIllumBasisSample15ev(nonNegative, lambda);
}

// Header-only audit: no RNG, no film writes, no extra per-path file access.
// The legacy table is evaluated here solely for the A/B diagnostic.
static void MLHeroPathLogIllumBasis15ev(FILE *f) {
    const char *names[] = {"WHITE", "RED", "GREEN", "BLUE", "YELLOW", "CYAN", "MAGENTA"};
    const Spectrum probes[] = {Spectrum(1.f, 1.f, 1.f), Spectrum(1.f, 0.f, 0.f),
        Spectrum(0.f, 1.f, 0.f), Spectrum(0.f, 0.f, 1.f), Spectrum(1.f, 1.f, 0.f),
        Spectrum(0.f, 1.f, 1.f), Spectrum(1.f, 0.f, 1.f)};
    const float wavelengths[] = {380.f, 550.f, 700.f, 720.f, 730.f, 750.f, 780.f};
    for (u_int color = 0u; color < 7u; ++color) {
        const RGBColor rgb(probes[color].c[0], probes[color].c[1], probes[color].c[2]);
        RGBIllumSPD legacy(rgb);
        for (u_int sample = 0u; sample < 7u; ++sample) {
            const float lambda = wavelengths[sample];
            const float active = MLHeroPathRGBIllumSample15ev(probes[color], lambda);
            const float legacyClamped = Max(0.f, legacy.Sample(Clamp(lambda, 380.f, 719.999f)));
            fprintf(f, "phase15ev ILLUM_BASIS_PROBE rgb=%s lambda=%.9g activeSPD=%.9g legacyClampedSPD=%.9g deltaSPD=%.9g table=MLHERO_380_780 appliedToFilm=0\n",
                names[color], lambda, active, legacyClamped, active - legacyClamped);
        }
    }
}


// Matching LuxCore SPD::ToXYZ semantics for a uniform HERO Monte Carlo estimate
// over [380,780] nm: (1/N) sum SPD(lambda) * RGB(683 * 400 * CIE_XYZ(lambda)).
static Spectrum MLHeroPathCIEEstimatorColor15ev(const float lambda, const float sampleWeight) {
    if ((lambda < 380.f) || (lambda > 780.f))
        return Spectrum();

    const float ciePos = lambda - (float)CIEstart;
    const u_int i0 = Min((u_int)Floor2UInt(ciePos), nCIE - 1);
    const u_int i1 = Min(i0 + 1u, nCIE - 1);
    const float t = Clamp(ciePos - i0, 0.f, 1.f);

    const float x = Lerp(t, CIE_X[i0], CIE_X[i1]);
    const float y = Lerp(t, CIE_Y[i0], CIE_Y[i1]);
    const float z = Lerp(t, CIE_Z[i0], CIE_Z[i1]);

    const XYZColor xyz = XYZColor(x, y, z) * (683.f * 400.f);
    const RGBColor rgb = ColorSystem::DefaultColorSystem.ToRGB(xyz);
    return Spectrum(rgb.c[0], rgb.c[1], rgb.c[2]) * sampleWeight;
}

static Spectrum MLHeroPathReconstructEndpoint15ev(const u_int laneCount,
        const float laneScalar[8], const bool supported, Spectrum *rawOut = nullptr,
        Spectrum *whiteOut = nullptr, Spectrum *legacyNormalizedOut = nullptr) {
    Spectrum rawPacketRGB;
    Spectrum whiteResponse;
    Spectrum legacyNormalizedPacketRGB;
    Spectrum ciePacketRGB;

    if (supported && laneCount > 0u) {
        const float invLaneCount = 1.f / (float)laneCount;
        const Spectrum whiteRGB(1.f);
        for (u_int lane = 0; lane < laneCount; ++lane) {
            const float lambda = GetMLHeroWaveLengthAt(lane);

            // 15eb/15ed reconstruction retained only as an A/B diagnostic.
            const Spectrum sampleColor = GetMLHeroSampleColorAt(lane);
            rawPacketRGB += sampleColor * (laneScalar[lane] * invLaneCount);
            const float whiteLane = MLHeroPathRGBIllumSample15ev(whiteRGB, lambda);
            whiteResponse += sampleColor * (whiteLane * invLaneCount);

            // 15ev active film reconstruction: radiometric CIE estimator.
            const Spectrum cieColor = MLHeroPathCIEEstimatorColor15ev(
                lambda, GetMLHeroSampleWeightAt(lane));
            ciePacketRGB += cieColor * (laneScalar[lane] * invLaneCount);
        }

        const float eps = 1e-20f;
        legacyNormalizedPacketRGB.c[0] = (fabsf(whiteResponse.c[0]) > eps) ? rawPacketRGB.c[0] / whiteResponse.c[0] : 0.f;
        legacyNormalizedPacketRGB.c[1] = (fabsf(whiteResponse.c[1]) > eps) ? rawPacketRGB.c[1] / whiteResponse.c[1] : 0.f;
        legacyNormalizedPacketRGB.c[2] = (fabsf(whiteResponse.c[2]) > eps) ? rawPacketRGB.c[2] / whiteResponse.c[2] : 0.f;
    }

    if (rawOut)
        *rawOut = rawPacketRGB;
    if (whiteOut)
        *whiteOut = whiteResponse;
    if (legacyNormalizedOut)
        *legacyNormalizedOut = legacyNormalizedPacketRGB;
    return ciePacketRGB;
}

// Spectral continuation is thread-local and reset per eye path. Logging budgets
// only select log rows. All supported paths use this transport, independently
// of logging. Actual ray directions, PDFs and RR decisions stay with Path.
struct MLHeroPathContinuation15ev {
    bool attempted = false, active = false, valid = false, collapsed = false;
    unsigned long long id = 0ull;
    u_int count = 0u;
    Spectrum tp[8];
    std::vector<Spectrum> cieRadiance;
};
static thread_local MLHeroPathContinuation15ev mlHeroContinuation15ev;
static std::atomic<unsigned long long> mlHeroGlassDetailBudget15ev(64ull);
static std::atomic<unsigned long long> mlHeroContinuationPathBudget15ev(64ull);
static std::atomic<unsigned long long> mlHeroContinuationOrdinal15ev(0ull);
static std::atomic<unsigned long long> mlHeroContinuationRowBudget15ev(512ull);

static void MLHeroPathResetContinuationDiagnostics15ev() {
    mlHeroGlassDetailBudget15ev.store(64ull, std::memory_order_relaxed);
    mlHeroContinuationPathBudget15ev.store(64ull, std::memory_order_relaxed);
    mlHeroContinuationOrdinal15ev.store(0ull, std::memory_order_relaxed);
    mlHeroContinuationRowBudget15ev.store(512ull, std::memory_order_relaxed);
}

static bool MLHeroPathContinuationClaim15ev(std::atomic<unsigned long long> &counter) {
    unsigned long long n = counter.load(std::memory_order_relaxed);
    while (n > 0ull && !counter.compare_exchange_weak(n, n - 1ull, std::memory_order_relaxed)) { }
    return n > 0ull;
}

static void MLHeroPathContinuationLog15ev(const char *kind, const u_int depth,
        const char *mode, const BSDFEvent event, const float rrProb,
        const float directionDelta, const Spectrum &classic,
        const Spectrum &candidate, const float *factor = nullptr,
        const u_int materialType = 0u, const bool applied = false) {
    const MLHeroPathContinuation15ev &s = mlHeroContinuation15ev;
    if (!s.active || s.id == 0ull || !MLHeroPathContinuationClaim15ev(mlHeroContinuationRowBudget15ev)) return;
    std::lock_guard<std::mutex> lock(mlHeroPathDebugMutex15ev);
    FILE *f = fopen(ML_HERO_PATH_LOG_FILE, "a");
    if (!f) return;
    fprintf(f, "phase15ev HERO_CONTINUATION kind=%s path=%llu depth=%u mode=%s event=%u materialType=%u valid=%d collapsed=%d activeCount=%u rrProb=%.9g maxDirectionDelta=%.9g estimatorScale=1 packetAverage=1/activeCount candidateApplied=%d classicFinalRGB=(%.9g,%.9g,%.9g) candidateCIE=(%.9g,%.9g,%.9g)",
        kind, s.id, depth, mode, (u_int)event, materialType, s.valid ? 1 : 0, s.collapsed ? 1 : 0,
        s.count, rrProb, directionDelta, applied ? 1 : 0, classic.c[0], classic.c[1], classic.c[2],
        candidate.c[0], candidate.c[1], candidate.c[2]);
    for (u_int lane = 0; lane < s.count; ++lane)
        fprintf(f, " lane%uLambda=%.9g lane%uTP=%.9g lane%uFactor=%.9g lane%uSampleWeight=%.9g",
            lane, GetMLHeroWaveLengthAt(lane), lane, s.tp[lane].c[0], lane,
            factor ? factor[lane] : 0.f, lane, GetMLHeroSampleWeightAt(lane));
    fprintf(f, "\n"); fclose(f);
}

static void MLHeroPathContinuationStart15ev(const BSDF &bsdf, const BSDFEvent event,
        const Spectrum &sample, const bool packetValid, const u_int count,
        const Spectrum tp[8], const u_int depth) {
    MLHeroPathContinuation15ev &s = mlHeroContinuation15ev;
    const bool rough = bsdf.GetMaterialType() == ROUGHGLASS;
    const bool glass = bsdf.GetMaterialType() == GLASS;
    if (s.attempted || (!rough && !glass) || sample.Black()) return;
    s.attempted = true;
    s.active = true; s.valid = packetValid && count > 1u && count <= 8u;
    if (MLHeroPathContinuationClaim15ev(mlHeroContinuationPathBudget15ev))
        s.id = mlHeroContinuationOrdinal15ev.fetch_add(1ull, std::memory_order_relaxed) + 1ull;
    s.count = count <= 8u ? count : 0u;
    for (u_int lane = 0; lane < s.count; ++lane) s.tp[lane] = tp[lane];
    MLHeroPathContinuationLog15ev("START", depth,
        s.valid ? (rough ? "COPY_PRE_ROUGHGLASS_PACKET" : "COPY_PRE_GLASS_PACKET") : "PRE_GLASS_PACKET_UNSUPPORTED",
        event, 1.f, 0.f, Spectrum(), Spectrum());
}

// Same supported material factors as the existing Path packet. All factors are
// calculated before any spectral throughput update. No scalar mean fallback
// is allowed for rejected chromatic/layered cases.
static bool MLHeroPathContinuationFactors15ev(const BSDF &bsdf, const Vector &dir,
        const BSDFEvent event, const float pdfW, const Spectrum &value,
        const bool sampled, const float rrProb, float factors[8]) {
    const u_int count = mlHeroContinuation15ev.count;
    if (count == 0u || count > 8u || !std::isfinite(rrProb) || rrProb <= 0.f) return false;
// 15fg BEGIN
    if (bsdf.GetMaterialType() == MATTETRANSLUCENT && (!sampled || mlHeroPathNativeSample15fg))
        return MLHeroPathTranslucentLanes15fg(bsdf, dir, event, sampled, count,
            rrProb, mlHeroPathComp15fg, sampled ? 2u : 3u, factors);
// 15fg END
    if (bsdf.GetMaterialType() == METAL2) {
        bool ok[8] = {};
        return MLHeroPathMetal2Lanes15ev(bsdf, dir, sampled, count, rrProb, ok, factors);
    }
    if (bsdf.GetMaterialType() == ROUGHGLASS) {
        if (sampled && (!std::isfinite(pdfW) || pdfW <= 0.f)) return false;
        for (u_int lane = 0; lane < count; ++lane) {
            Spectrum laneValue;
            if (!bsdf.EvaluateMLHeroRoughGlassAtWaveLength(dir, event,
                GetMLHeroWaveLengthAt(lane), &laneValue)) return false;
            factors[lane] = MLHeroPathRoughColor15ev(laneValue, GetMLHeroWaveLengthAt(lane)) /
                ((sampled ? pdfW : 1.f) * rrProb);
            if (!std::isfinite(factors[lane]) || factors[lane] < 0.f) return false;
        }
        return true;
    }
    const bool matte = bsdf.GetMaterialType() == MATTE && (event & DIFFUSE) &&
        (event & REFLECT) && !(event & SPECULAR);
    const bool glossy = bsdf.GetMaterialType() == GLOSSY2 && (event & REFLECT) && !(event & SPECULAR);
    for (u_int lane = 0; lane < count; ++lane) {
        bool ok;
        if (matte) ok = MLHeroPathMatteLaneMultiplier15ev(bsdf.Albedo().Clamp(0.f, 1.f),
            value, GetMLHeroWaveLengthAt(lane), &factors[lane]);
        else if (glossy) ok = MLHeroPathGlossy2LaneMultiplier15ev(bsdf, dir, pdfW,
            event, value, GetMLHeroWaveLengthAt(lane), &factors[lane]);
        else {
            ok = MLHeroPathIsScalarSpectrum15ev(value);
            factors[lane] = (value.c[0] + value.c[1] + value.c[2]) / 3.f;
        }
        factors[lane] /= rrProb;
        if (!ok || !std::isfinite(factors[lane]) || factors[lane] < 0.f) return false;
    }
    return true;
}

static void MLHeroPathContinuationConnection15ev(const Spectrum &connection, const u_int depth) {
    MLHeroPathContinuation15ev &s = mlHeroContinuation15ev;
    if (!s.active || !s.valid) return;
    if (!MLHeroPathIsScalarSpectrum15ev(connection)) {
        s.valid = false;
        MLHeroPathContinuationLog15ev("CONNECTION", depth, "CHROMATIC_CONNECTION_REJECT",
            (BSDFEvent)0, 1.f, 0.f, Spectrum(), Spectrum());
        return;
    }
    const float factor = (connection.c[0] + connection.c[1] + connection.c[2]) / 3.f;
    for (u_int lane = 0; lane < s.count; ++lane) s.tp[lane] *= Spectrum(factor);
}

static void MLHeroPathLogRough15ev(const BSDF &bsdf, const char *kind,
        const BSDFEvent event, const float rrProb, const float factors[8], const bool ok) {
    if (bsdf.GetMaterialType() != ROUGHGLASS) return;
    auto &budget = (kind[0] == 'B') ? mlHeroRoughBounceBudget15ev : mlHeroRoughDLSBudget15ev;
    unsigned long long remaining = budget.load(std::memory_order_relaxed);
    while (remaining && !budget.compare_exchange_weak(remaining, remaining - 1ull,
        std::memory_order_relaxed)) {}
    if (!remaining) return;
    FILE *f = fopen(ML_HERO_PATH_LOG_FILE, "a");
    if (!f) return;
    const auto &state = mlHeroContinuation15ev;
    fprintf(f, "phase15ev ROUGHGLASS_%s materialID=%u name=\"%s\" event=%u dispersion=%d valid=%d collapsed=%d activeCount=%u rrProb=%.9g helperOK=%d sampledPdfOwner=HERO_SHARED activeAverage=1/activeCount",
        kind, bsdf.GetMaterialID(), bsdf.GetMaterialName().c_str(), (u_int)event,
        bsdf.HasMLHeroRoughGlassDispersion() ? 1 : 0, state.valid ? 1 : 0,
        state.collapsed ? 1 : 0, state.count, rrProb, ok ? 1 : 0);
    for (u_int lane = 0; lane < state.count; ++lane)
        fprintf(f, " lane%uLambda=%.9g lane%uFactor=%.9g", lane,
            GetMLHeroWaveLengthAt(lane), lane, factors[lane]);
    fprintf(f, "\n"); fclose(f);
}

static void MLHeroPathContinuationBounce15ev(const BSDF &bsdf, const Vector &dir,
        const BSDFEvent event, const float pdfW, const Spectrum &preRR,
        const float rrProb, const u_int depth) {
    MLHeroPathContinuation15ev &s = mlHeroContinuation15ev;
    if (!s.active || !s.valid) return;
    float factors[8] = {};
    float maxDelta = 0.f;
    bool ok = true;
    const char *mode = "MATERIAL_SPECTRAL";
    if (bsdf.GetMaterialType() == ROUGHGLASS) {
        const bool dispersiveTX = (event & TRANSMIT) && bsdf.HasMLHeroRoughGlassDispersion();
        if (dispersiveTX && !s.collapsed) {
            for (u_int lane = 1; lane < s.count; ++lane) s.tp[lane] = Spectrum();
            s.count = 1u; s.collapsed = true;
        }
        if (event & SPECULAR) {
            // The finite-solid-angle helper cannot represent the exact delta lobe.
            // Its native sampled multiplier is safe for a single HERO or a
            // non-dispersive packet; keep the legacy path for dispersive reflection.
            ok = (s.collapsed || !bsdf.HasMLHeroRoughGlassDispersion()) &&
                std::isfinite(rrProb) && rrProb > 0.f;
            if (ok) for (u_int lane = 0; lane < s.count; ++lane) {
                factors[lane] = MLHeroPathRoughColor15ev(preRR, GetMLHeroWaveLengthAt(lane)) / rrProb;
                ok = ok && std::isfinite(factors[lane]) && factors[lane] >= 0.f;
            }
        } else
            ok = MLHeroPathContinuationFactors15ev(bsdf, dir, event, pdfW, preRR, true, rrProb, factors);
        mode = (event & TRANSMIT) ? (s.collapsed ? "ROUGHGLASS_TX_HERO_ONLY" : "ROUGHGLASS_TX_PACKET") :
            (s.collapsed ? "ROUGHGLASS_REFLECT_HERO_ONLY" : "ROUGHGLASS_REFLECT_PACKET");
    } else if (bsdf.GetMaterialType() == GLASS && (event & SPECULAR)) {
        const bool dispersion = bsdf.HasMLHeroGlassDispersion();
        const bool firstCollapse = (event & TRANSMIT) && dispersion && !s.collapsed;
        if (firstCollapse) {
            for (u_int lane = 1; lane < s.count; ++lane) s.tp[lane] = Spectrum();
            s.count = 1u; s.collapsed = true;
        }
        bool allHelpersOK = true;
        for (u_int lane = 0; lane < s.count; ++lane) {
            Spectrum multiplier;
            float delta = 0.f;
            const bool helperOK = bsdf.EvaluateMLHeroGlassSampleAtWaveLength(dir, event, pdfW,
                GetMLHeroWaveLengthAt(lane), &multiplier, &delta);
            allHelpersOK = allHelpersOK && helperOK;
            maxDelta = Max(maxDelta, delta);
            // Native Sample already contains the selected HERO PDF. Only use
            // it at a rejected numerical boundary, keeping wavelength color.
            factors[lane] = MLHeroPathRoughColor15ev(helperOK ? multiplier : preRR,
                GetMLHeroWaveLengthAt(lane)) / rrProb;
            ok = ok && std::isfinite(rrProb) && rrProb > 0.f &&
                std::isfinite(factors[lane]) && factors[lane] >= 0.f;
        }
        mode = (event & REFLECT) ? (s.collapsed ? "GLASS_REFLECT_HERO_ONLY" : "GLASS_REFLECT_PACKET") :
            (s.collapsed ? "GLASS_TX_HERO_ONLY" : "GLASS_TX_PACKET");
        if (MLHeroPathContinuationClaim15ev(mlHeroGlassDetailBudget15ev)) {
            std::lock_guard<std::mutex> lock(mlHeroPathDebugMutex15ev);
            FILE *f = fopen(ML_HERO_PATH_LOG_FILE, "a");
            if (f) {
                fprintf(f, "phase15ev GLASS_CONTINUATION materialID=%u name=\"%s\" event=%u dispersion=%u helperOK=%u nativeSampleFallback=%u firstCollapse=%u collapsed=%u activeCount=%u rrProb=%.9g heroDirectionDelta=%.9g heroFactor=%.9g colorTransport=SPECTRAL activeAverage=1/activeCount extraN=0\n",
                    bsdf.GetMaterialID(), bsdf.GetMaterialName().c_str(), (u_int)event,
                    dispersion ? 1u : 0u, allHelpersOK ? 1u : 0u, allHelpersOK ? 0u : 1u,
                    firstCollapse ? 1u : 0u, s.collapsed ? 1u : 0u, s.count, rrProb, maxDelta, factors[0]);
                fclose(f);
            }
        }
    } else {
        ok = MLHeroPathContinuationFactors15ev(bsdf, dir, event, pdfW, preRR, true, rrProb, factors);
        if (bsdf.GetMaterialType() == METAL2) mode = "METAL2_SPECTRAL";
        else if (bsdf.GetMaterialType() == MATTE) mode = "MATTE_SPECTRAL";
        else if (bsdf.GetMaterialType() == GLOSSY2) mode = "GLOSSY2_SPECTRAL";
// 15fg BEGIN
        else if (bsdf.GetMaterialType() == MATTETRANSLUCENT && mlHeroPathNativeSample15fg)
            mode = "MATTETRANSLUCENT_NATIVE_SPECTRAL";
// 15fg END
        else mode = "SCALAR_SHARED";
    }
    if (ok)
        for (u_int lane = 0; lane < s.count; ++lane) s.tp[lane] *= Spectrum(factors[lane]);
    else { s.valid = false; mode = "CONTINUATION_HELPER_REJECT"; }
// 15fg BEGIN
    if (ok && bsdf.GetMaterialType() == MATTETRANSLUCENT && mlHeroPathNativeSample15fg)
        MLHeroPathTranslucentApplied15fg(2u,event,s.tp,s.count,true);
// 15fg END
    MLHeroPathLogRough15ev(bsdf, "BOUNCE", event, rrProb, factors, ok);
    MLHeroPathContinuationLog15ev("BOUNCE", depth, mode, event, rrProb, maxDelta,
        Spectrum(), Spectrum(), factors, (u_int)bsdf.GetMaterialType(), ok);
}

static bool MLHeroPathContinuationEndpoint15ev(const char *kind, const u_int depth,
        Spectrum *applied, const Spectrum &classic, const Spectrum &light, const float scale,
        const Spectrum &connection, const BSDF *bsdf = nullptr,
        const Vector *dir = nullptr, const BSDFEvent event = (BSDFEvent)0,
        const float pdfW = 1.f, const Spectrum &bsdfValue = Spectrum(1.f)) {
    MLHeroPathContinuation15ev &s = mlHeroContinuation15ev;
    if (!s.active || !s.valid || !applied) return false;
    float factors[8] = {1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
    const bool supported = MLHeroPathIsScalarSpectrum15ev(connection) &&
        (!bsdf || (dir && MLHeroPathContinuationFactors15ev(*bsdf, *dir, event, pdfW,
            bsdfValue, false, 1.f, factors)));
    float lanes[8] = {};
    if (supported) {
        const float conn = (connection.c[0] + connection.c[1] + connection.c[2]) / 3.f;
        for (u_int lane = 0; lane < s.count; ++lane)
            lanes[lane] = s.tp[lane].c[0] * factors[lane] * scale * conn *
                MLHeroPathRGBIllumSample15ev(light, GetMLHeroWaveLengthAt(lane));
    }
    if (bsdf) MLHeroPathLogRough15ev(*bsdf, "DLS", event, 1.f, factors, supported);
    const Spectrum candidate = MLHeroPathReconstructEndpoint15ev(s.count, lanes, supported);
    // Log the legacy counterfactual alongside the CIE value submitted to film.
    MLHeroPathContinuationLog15ev(kind, depth, supported ? "CIE_ACTIVE" : "ENDPOINT_REJECT",
        event, 1.f, 0.f, classic * GetMLDispersionSampleColor(), candidate, factors,
        bsdf ? (u_int)bsdf->GetMaterialType() : 0u, supported);
    if (supported) *applied = candidate;
    return supported;
}

// Track only CIE contributions actually submitted to this radiance group.
// They stay outside the legacy RGB color operation even after a later fallback.
static void MLHeroPathContinuationRecord15ev(const u_int lightID, const Spectrum &value) {
    std::vector<Spectrum> &cie = mlHeroContinuation15ev.cieRadiance;
    if (cie.size() <= lightID) cie.resize(lightID + 1u);
    cie[lightID] += value;
}

static Spectrum MLHeroPathContinuationSplit15ev(const Spectrum &before,
        const Spectrum &total, const Spectrum &cie, const Spectrum &legacyColor) {
    return before + (total - before - cie) * legacyColor + cie;
}

static bool MLHeroPathTakeEndpointBudget15ev() {
    unsigned long long budget = mlHeroPathEndpointBudget15ev.load(std::memory_order_relaxed);
    while (budget > 0ull && !mlHeroPathEndpointBudget15ev.compare_exchange_weak(
            budget, budget - 1ull, std::memory_order_relaxed)) { }
    return budget > 0ull;
}

static void MLHeroPathLogEndpoint15ev(const char *kind, const Spectrum &classicContribution,
        const u_int laneCount, const float laneScalar[8], const bool supported) {
    if (!MLHeroPathTakeEndpointBudget15ev())
        return;

    Spectrum rawPacketRGB;
    Spectrum whiteResponse;
    Spectrum legacyNormalizedPacketRGB;
    const Spectrum ciePacketRGB = MLHeroPathReconstructEndpoint15ev(
        laneCount, laneScalar, supported, &rawPacketRGB, &whiteResponse, &legacyNormalizedPacketRGB);

    Spectrum ratio;
    const float ratioEps = 1e-12f;
    for (u_int c = 0; c < 3u; ++c)
        ratio.c[c] = (fabsf(classicContribution.c[c]) > ratioEps) ?
            ciePacketRGB.c[c] / classicContribution.c[c] : 0.f;

    std::lock_guard<std::mutex> lock(mlHeroPathDebugMutex15ev);
    FILE *f = fopen(ML_HERO_PATH_LOG_FILE, "a");
    if (!f)
        return;

    fprintf(f, "phase15ev endpoint kind=%s supported=%d laneCount=%u classicRGB=(%.9g,%.9g,%.9g) rawPacketRGB=(%.9g,%.9g,%.9g) whiteResponse=(%.9g,%.9g,%.9g) legacyNormalizedRGB=(%.9g,%.9g,%.9g) ciePacketRGB=(%.9g,%.9g,%.9g) ratio=(%.9g,%.9g,%.9g)",
        kind, supported ? 1 : 0, laneCount,
        classicContribution.c[0], classicContribution.c[1], classicContribution.c[2],
        rawPacketRGB.c[0], rawPacketRGB.c[1], rawPacketRGB.c[2],
        whiteResponse.c[0], whiteResponse.c[1], whiteResponse.c[2],
        legacyNormalizedPacketRGB.c[0], legacyNormalizedPacketRGB.c[1], legacyNormalizedPacketRGB.c[2],
        ciePacketRGB.c[0], ciePacketRGB.c[1], ciePacketRGB.c[2],
        ratio.c[0], ratio.c[1], ratio.c[2]);
    if (supported) {
        for (u_int lane = 0; lane < laneCount; ++lane)
            fprintf(f, " lane%u=%.9g@%.9gnm", lane, laneScalar[lane], GetMLHeroWaveLengthAt(lane));
    }
    fprintf(f, "\n");
    fclose(f);
}

//------------------------------------------------------------------------------
// ML HERO Sampling 2.0 - O1 CPU PATH diagnostic
// 16 wavelength bins; 80% CIE-Y-like photopic importance + 20% uniform floor.
// sampleWeight = p_uniform / p_sampling2, so the existing uniform-normalized
// WaveLength2RGB estimator remains energy-correct.
//------------------------------------------------------------------------------
static float MLHeroSampling2Sample(const float u, float *sampleWeight) {
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

//------------------------------------------------------------------------------
// ML HERO Sampling 3.0 - O2 CPU PATH diagnostic
//
// "RGB sensor weighted" here means importance sampled against LuxCore's own
// wavelength-to-RGB output response, not a physical camera sensor model.
// We integrate the current WaveLength2RGB response into 16 x 25 nm bins and
// use an 80% sensor-importance / 20% uniform mixture.
//
// sampleWeight = p_uniform / p_sampling3, preserving the same uniform-spectrum
// estimator used by Sampling 1.0 and keeping the estimator unbiased.
//------------------------------------------------------------------------------
static float MLHeroSampling3Sample(const float u, float *sampleWeight) {
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
// PathTracer
//------------------------------------------------------------------------------

PathTracerThreadState::PathTracerThreadState(IntersectionDevice *dev,
		const SamplerUPtr& eSampler,
		const SamplerUPtr& lSampler,
		SceneConstRef scn, FilmRef flm,
		const VarianceClamping *varClamping,
		const bool useFilmSplat) : device(dev),
		eyeSampler(eSampler), lightSampler(lSampler), scene(scn), film(flm),
		varianceClamping(varClamping) {
	// Initialize Eye SampleResults
	eyeSampleResults.resize(1);
	PathTracer::InitEyeSampleResults(film, eyeSampleResults, useFilmSplat);

	eyeSampleCount = 0.0;
	// Using 1.0 instead of 0.0 to avoid a division by zero
	lightSampleCount = 1.0;
}

PathTracerThreadState::~PathTracerThreadState() {
}

//------------------------------------------------------------------------------
// PathTracer
//------------------------------------------------------------------------------

const Film::FilmChannels PathTracer::eyeSampleResultsChannels({
	Film::RADIANCE_PER_PIXEL_NORMALIZED, Film::ALPHA, Film::DEPTH,
	Film::POSITION, Film::GEOMETRY_NORMAL, Film::SHADING_NORMAL, Film::MATERIAL_ID,
	Film::DIRECT_DIFFUSE, Film::DIRECT_DIFFUSE_REFLECT, Film::DIRECT_DIFFUSE_TRANSMIT,
	Film::DIRECT_GLOSSY, Film::DIRECT_GLOSSY_REFLECT, Film::DIRECT_GLOSSY_TRANSMIT,
	Film::EMISSION,
	Film::INDIRECT_DIFFUSE, Film::INDIRECT_DIFFUSE_REFLECT, Film::INDIRECT_DIFFUSE_TRANSMIT,
	Film::INDIRECT_GLOSSY, Film::INDIRECT_GLOSSY_REFLECT, Film::INDIRECT_GLOSSY_TRANSMIT,
	Film::INDIRECT_SPECULAR, Film::INDIRECT_SPECULAR_REFLECT, Film::INDIRECT_SPECULAR_TRANSMIT,
	Film::DIRECT_SHADOW_MASK, Film::INDIRECT_SHADOW_MASK, Film::UV, Film::RAYCOUNT,
	Film::IRRADIANCE, Film::OBJECT_ID, Film::SAMPLECOUNT, Film::CONVERGENCE,
	Film::MATERIAL_ID_COLOR, Film::ALBEDO, Film::AVG_SHADING_NORMAL, Film::NOISE
});

const Film::FilmChannels PathTracer::lightSampleResultsChannels({
	Film::RADIANCE_PER_SCREEN_NORMALIZED
}); 

PathTracer::PathTracer() : pixelFilterDistribution(nullptr),
		photonGICache(nullptr), mlHeroEnabled(false), mlHeroRoughGlassFresnelImportanceConfig(false), mlHeroSamplingMode(1), mlHeroWavelengthCount(1), mlHeroQuarterCycling(true), mlHeroGenericReflectanceCompensation(true), mlHeroDiagnostics15fg(false), mlHeroRRGlassEtaExperiment(false), mlHeroRRMaterialExperiment(false), mlHeroRRCPUPath15fz(false) {
}

PathTracer::~PathTracer() {
	delete pixelFilterDistribution;
}

// 15fg BEGIN
// 15fh BEGIN diagnostic-only
void PathTracer::ResetMLHeroEstimator15fh() const {
    MLHeroReset15fh(mlHeroEnabled && mlHeroWavelengthCount>1 && mlHeroDiagnostics15fg);
    mlHeroPathRRStats15fz=MLHeroPathRRStats15fz();
}
// 15fh END
void PathTracer::LogMLHeroMatteTranslucent15fg(const u_int threadIndex) const {
    if(!mlHeroDiagnostics15fg || !mlHeroEnabled) return;
    std::lock_guard<std::mutex> lock(mlHeroPathDebugMutex15ev);
    FILE*f=fopen(ML_HERO_PATH_LOG_FILE,"a"); if(!f)return;
    for(u_int i=0u;i<8u;++i)
        fprintf(f,"phase15fg TRANSLUCENT_SUMMARY thread=%u stage=%u transmit=%u preparations=%llu helperRejected=%llu applied=%llu globalCumulativeSnapshot=1 doNotSumSnapshots=1 stageNames=packetBounce,packetDLS,continuationBounce,continuationDLS\n",threadIndex,i/2u,i%2u,mlHero15fgPrepared[i].load(),mlHero15fgRejected[i].load(),mlHero15fgApplied[i].load());
    fprintf(f,"phase15fg TRANSPORT_AUDIT thread=%u eyeResults=%llu eyeNonfinite=%llu appliedNonfinite=%llu nonNeutralOrInvalidAppliedLanes=%llu globalCumulativeSnapshot=1 doNotSumSnapshots=1 onlyNewMaterialStages=1\n",threadIndex,mlHero15fgEyeResults.load(),mlHero15fgEyeNonfinite.load(),mlHero15fgAppliedNonfinite.load(),mlHero15fgRgbLanes.load());
    // 15fh BEGIN diagnostic-only
fprintf(f,"phase15fi DEPTH_CONFIG engine=PATHCPU thread=%u total=%u diffuse=%u glossy=%u specular=%u rrStart=%u rrLimit=%.9g rrCounter=diffusePlusGlossy rrAfterSampledBounce=1 rrSkipSpecularBaseline=1 glassOverrideSee15fzConfig=1 lastVertexTest=nextPossibleEvent lastVertexDLS=firstVertexOnly rendererPhysicsChanged=1\n",threadIndex,maxPathDepth.depth,maxPathDepth.diffuseDepth,maxPathDepth.glossyDepth,maxPathDepth.specularDepth,rrDepth,rrImportanceCap);
fprintf(f,"phase15fm PATH_MIS_CONFIG engine=PATHCPU thread=%u rendererPhysicsChanged=1 nativeBSDFEvalDlsPDF=RAW_DIRECTIONAL rrCompensation=THROUGHPUT_ONLY hybridEnabled=%u hybridAndBakeBaselineRetained=1 independentOfHeroAndDiagnostics=1 unchangedTransparencyFactor=1 BIDIR_UNCHANGED=1\n",threadIndex,hybridBackForwardEnable?1u:0u);
MLHeroWrite15fh(f,"PATHCPU",threadIndex);
    MLHeroPathRRWrite15fz(f,threadIndex,mlHeroRRCPUPath15fz,mlHeroRRGlassEtaExperiment,mlHeroRRMaterialExperiment);
// 15fh END
    fclose(f);
}
// 15fg END

void PathTracer::InitPixelFilterDistribution(const FilterUPtr& pixelFilter) {
	// Compile sample distribution
	delete pixelFilterDistribution;
	pixelFilterDistribution = new FilterDistribution(pixelFilter, 64);
}

void  PathTracer::DeletePixelFilterDistribution() {
	delete pixelFilterDistribution;
	pixelFilterDistribution = NULL;
}

void PathTracer::InitEyeSampleResults(FilmConstRef film, vector<SampleResult> &sampleResults,
		const bool useFilmSplat) {
	SampleResult &sampleResult = sampleResults[0];

	sampleResult.Init(&eyeSampleResultsChannels, film.GetRadianceGroupCount());
	sampleResult.useFilmSplat = useFilmSplat;
}

void PathTracer::ResetEyeSampleResults(vector<SampleResult> &sampleResults) {
	SampleResult &sampleResult = sampleResults[0];

	// Set to 0.0 all result colors
	sampleResult.emission = Spectrum();
	for (u_int i = 0; i < sampleResult.radiance.Size(); ++i)
		sampleResult.radiance[i] = Spectrum();
	sampleResult.directDiffuseReflect = Spectrum();
	sampleResult.directDiffuseTransmit = Spectrum();
	sampleResult.directGlossyReflect = Spectrum();
	sampleResult.directGlossyTransmit = Spectrum();
	sampleResult.indirectDiffuseReflect = Spectrum();
	sampleResult.indirectDiffuseTransmit = Spectrum();
	sampleResult.indirectGlossyReflect = Spectrum();
	sampleResult.indirectGlossyTransmit = Spectrum();
	sampleResult.indirectSpecularReflect = Spectrum();
	sampleResult.indirectSpecularTransmit = Spectrum();
	sampleResult.directShadowMask = 1.f;
	sampleResult.indirectShadowMask = 1.f;
	sampleResult.irradiance = Spectrum();
	sampleResult.albedo = Spectrum();
	sampleResult.isHoldout = false;

	sampleResult.rayCount = 0.f;
}

//------------------------------------------------------------------------------
// RenderEyeSample methods
//------------------------------------------------------------------------------

PathTracer::DirectLightResult PathTracer::DirectLightSampling(
		luxrays::IntersectionDevice *device, SceneConstRef scene,
		const float time,
		const float u0, const float u1, const float u2,
		const float u3, const float u4,
		const EyePathInfo &pathInfo, const Spectrum &pathThroughput,
		const BSDF &bsdf, SampleResult *sampleResult,
		const bool useBSDFEVal, const Spectrum *mlHeroLaneThroughput,
		const u_int mlHeroLaneCount, const bool mlHeroPacketValid) const {
	if (!bsdf.IsDelta()) {
		// Select the light strategy to use
		auto& lightStrategy =
			bsdf.IsShadowCatcherOnlyInfiniteLights() ?
			scene.GetLightSources().GetInfiniteLightStrategy() :
			scene.GetLightSources().GetIlluminateLightStrategy();

		// Pick a light source to sample
		const Normal landingNormal = bsdf.hitPoint.intoObject ? bsdf.hitPoint.shadeN : -bsdf.hitPoint.shadeN;
		float lightPickPdf;
		auto light = lightStrategy.SampleLights(
			scene,
			u0,
			bsdf.hitPoint.p,
			landingNormal,
			bsdf.IsVolume(),
			&lightPickPdf
		);

		if (light) {
			Ray shadowRay;
			float directPdfW;
			Spectrum lightRadiance = light->Illuminate(
				scene, bsdf, time, u1, u2, u3, shadowRay, directPdfW
			);
			verify (!lightRadiance.IsNaN() && !lightRadiance.IsInf());

			if (!lightRadiance.Black()) {
				verify (!isnan(directPdfW) && !isinf(directPdfW));

				BSDFEvent event;
				float bsdfPdfW;
				Spectrum bsdfEval;
				
				if (useBSDFEVal)
					bsdfEval = bsdf.Evaluate(shadowRay.d, &event, &bsdfPdfW);
				else {
					// This is used by BAKECPU and must be aligned with its BSDF sampling
					bsdfEval = Spectrum(Dot(shadowRay.d, bsdf.hitPoint.shadeN) * INV_PI);
					bsdfPdfW = INV_TWOPI;
					event = DIFFUSE | REFLECT;
				}
				verify (!bsdfEval.IsNaN() && !bsdfEval.IsInf());

				if (!bsdfEval.Black() &&
						(!hybridBackForwardEnable ||
						!pathInfo.IsCausticPath(event, bsdf.GetGlossiness(), hybridBackForwardGlossinessThreshold))) {
					verify (!isnan(bsdfPdfW) && !isinf(bsdfPdfW));
					
					// Create a new PathDepthInfo for the path to the light source
					PathDepthInfo directLightDepthInfo = pathInfo.depth;
					directLightDepthInfo.IncDepths(event);
					
					RayHit shadowRayHit;
					BSDF shadowBsdf;
					Spectrum connectionThroughput;
					// Create a new PathVolumeInfo for the path to the light source
					PathVolumeInfo volInfo = pathInfo.volume;
					// Check if the light source is visible
					if (!scene.Intersect(device, EYE_RAY | SHADOW_RAY, &volInfo, u4, &shadowRay,
							&shadowRayHit, &shadowBsdf, &connectionThroughput, nullptr,
							nullptr, true)) {
						// Add the light contribution only if it is not a shadow catcher
						// (because, if the light is visible, the material will be
						// transparent in the case of a shadow catcher).

						if (!bsdf.IsShadowCatcher()) {
							// I'm ignoring volume emission because it is not sampled in
							// direct light step.
							const float directLightSamplingPdfW = directPdfW * lightPickPdf;
							const float factor = 1.f / directLightSamplingPdfW;


                            // 15fm: native CPU PATH uses raw directional PDFs in both
                            // NEE and DirectHit. RR is compensated in throughput.
                            // BAKE proposals and hybrid light tracing keep their baseline.
                            // 15fm BEGIN live PDF policy
                            const bool rawPathMIS15fm = useBSDFEVal && !hybridBackForwardEnable;
    if (mlHeroState15fh.enabled) MLHeroRRAudit15fk(7,directLightDepthInfo.GetRRDepth(),bsdf.GetMaterialType(),event,bsdfEval,bsdfPdfW,0.f,rrImportanceCap,RenderEngine::RussianRouletteProb(bsdfEval,rrImportanceCap),(!rawPathMIS15fm && directLightDepthInfo.GetRRDepth() >= rrDepth),(!rawPathMIS15fm && directLightDepthInfo.GetRRDepth() >= rrDepth),false);
                            const float rawForwardPdf15fl=bsdfPdfW;
							if (!rawPathMIS15fm && directLightDepthInfo.GetRRDepth() >= rrDepth) {
								// Russian Roulette
								bsdfPdfW *= RenderEngine::RussianRouletteProb(bsdfEval, rrImportanceCap);
							}

                            // 15fm END live PDF policy
							// Account for material transparency
							bsdfPdfW *= light->GetAvgPassThroughTransparency();

							// MIS between direct light sampling and BSDF sampling
							//
							// Note: I have to avoid MIS on the last path vertex
							const bool misEnabled = !sampleResult->lastPathVertex &&
								(light->IsEnvironmental() || light->IsIntersectable()) &&
								CheckDirectHitVisibilityFlags(*light, directLightDepthInfo, event) &&
								!shadowBsdf.hitPoint.throughShadowTransparency;

							const float weight = misEnabled ? PowerHeuristic(directLightSamplingPdfW, bsdfPdfW) : 1.f;
                            if(mlHeroState15fh.enabled) {
                                unsigned skip15fl=MLHeroTerminalLightGuard15fl(*light,time);
                                if(bsdf.GetMaterialType()!=MATTE && bsdf.GetMaterialType()!=MATTETRANSLUCENT)skip15fl=3u;
                                if(!misEnabled || !useBSDFEVal || bsdf.IsShadowCatcher())skip15fl=1u;
                                else if(lightStrategy.GetType()==TYPE_DLS_CACHE)skip15fl=4u;
                                else if(bsdf.IsVolume())skip15fl=5u;
                                MLHeroTerminalMIS15fl(0u,directLightDepthInfo.GetRRDepth(),bsdf.GetMaterialType(),
                                    event,light->GetID(),skip15fl,weight,rawForwardPdf15fl,bsdfPdfW,
                                    directPdfW,0.,0.,1.,0.,0.,0.,lightPickPdf);
                            }
							const Spectrum incomingRadiance = bsdfEval * (weight * factor) * connectionThroughput * lightRadiance;
                            const Spectrum classicContribution15ev = pathThroughput * incomingRadiance;
                            Spectrum continuationContribution15ev;
                            const bool useContinuation15ev = MLHeroPathContinuationEndpoint15ev("DLS", pathInfo.depth.depth,
                                &continuationContribution15ev, classicContribution15ev, lightRadiance, weight * factor,
                                connectionThroughput, &bsdf, &shadowRay.d, event, bsdfPdfW, bsdfEval);
                            bool usePacketContribution15ev = false;
                            Spectrum packetContribution15ev;
                            float laneScalar15ev[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                            bool supported15ev = false;

                            if (mlHeroEnabled && mlHeroLaneThroughput && mlHeroPacketValid && (mlHeroLaneCount > 1u)) {
                                const bool connScalar15ev = MLHeroPathIsScalarSpectrum15ev(connectionThroughput);
                                const bool bsdfScalar15ev = MLHeroPathIsScalarSpectrum15ev(bsdfEval);
                                const bool matte15ev = (bsdf.GetMaterialType() == MATTE) &&
                                    (event & DIFFUSE) && (event & REFLECT) && !(event & SPECULAR);
                                const bool glossy215ev = (bsdf.GetMaterialType() == GLOSSY2) &&
                                    (event & REFLECT) && !(event & SPECULAR);
                                const Spectrum matteKd15ev = matte15ev ? bsdf.Albedo().Clamp(0.f, 1.f) : Spectrum();
                                float matteFitError15ev = 0.f;
                                float matteProbe15ev = 0.f;
                                const bool matteFit15ev = matte15ev && MLHeroPathMatteLaneMultiplier15ev(
                                    matteKd15ev, bsdfEval, GetMLHeroWaveLengthAt(0), &matteProbe15ev, &matteFitError15ev);
                                float glossy2FitError15ev = 0.f;
                                float glossy2Probe15ev = 0.f;
                                const bool glossy2Fit15ev = glossy215ev && MLHeroPathGlossy2LaneMultiplier15ev(
                                    bsdf, shadowRay.d, bsdfPdfW, event, bsdfEval, GetMLHeroWaveLengthAt(0),
                                    &glossy2Probe15ev, &glossy2FitError15ev);
                                const bool metal215ev = bsdf.GetMaterialType() == METAL2;
                                bool metal2LaneOK15ev[8] = {};
                                float metal2LaneValue15ev[8] = {};
                                const bool metal2Fit15ev = metal215ev && MLHeroPathMetal2Lanes15ev(
                                    bsdf, shadowRay.d, false, mlHeroLaneCount, 1.f,
                                    metal2LaneOK15ev, metal2LaneValue15ev);
// 15fg BEGIN
                                const bool translucent15fg = useBSDFEVal && bsdf.GetMaterialType() == MATTETRANSLUCENT;
                                float translucentFactors15fg[8] = {};
                                const bool translucentFit15fg = translucent15fg && MLHeroPathTranslucentLanes15fg(
                                    bsdf,shadowRay.d,event,false,mlHeroLaneCount,1.f,
                                    mlHeroGenericReflectanceCompensation,1u,translucentFactors15fg);
// 15fg END
                                supported15ev = connScalar15ev && (translucent15fg ? translucentFit15fg :
                                    (metal215ev ? metal2Fit15ev : (bsdfScalar15ev || matteFit15ev || glossy2Fit15ev)));

                                if (supported15ev) {
                                    const float conn15ev = (connectionThroughput.c[0] + connectionThroughput.c[1] + connectionThroughput.c[2]) / 3.f;
                                    const float bsdfScalarValue15ev = (bsdfEval.c[0] + bsdfEval.c[1] + bsdfEval.c[2]) / 3.f;
                                    for (u_int lane15ev = 0; lane15ev < mlHeroLaneCount; ++lane15ev) {
                                        const float lambda15ev = GetMLHeroWaveLengthAt(lane15ev);
                                        float bsdfLane15ev = bsdfScalarValue15ev;
// 15fg BEGIN
                                        if (translucentFit15fg)
                                            bsdfLane15ev = translucentFactors15fg[lane15ev];
                                        else
// 15fg END
                                        if (metal2Fit15ev)
                                            bsdfLane15ev = metal2LaneValue15ev[lane15ev];
                                        else if (matteFit15ev)
                                            MLHeroPathMatteLaneMultiplier15ev(matteKd15ev, bsdfEval, lambda15ev, &bsdfLane15ev);
                                        else if (glossy2Fit15ev)
                                            MLHeroPathGlossy2LaneMultiplier15ev(bsdf, shadowRay.d, bsdfPdfW, event,
                                                bsdfEval, lambda15ev, &bsdfLane15ev);
                                        const float lightLane15ev = MLHeroPathRGBIllumSample15ev(lightRadiance, lambda15ev);
                                        laneScalar15ev[lane15ev] = mlHeroLaneThroughput[lane15ev].c[0] *
                                            bsdfLane15ev * (weight * factor) * conn15ev * lightLane15ev;
                                    }
                                    packetContribution15ev = MLHeroPathReconstructEndpoint15ev(
                                        mlHeroLaneCount, laneScalar15ev, true);
                                    usePacketContribution15ev = true;
                                    mlHeroPathEndpointDLS15ev.fetch_add(1ull, std::memory_order_relaxed);
                                } else
                                    mlHeroPathEndpointUnsupported15ev.fetch_add(1ull, std::memory_order_relaxed);

                                if (metal215ev)
                                    MLHeroPathLogMetal2Audit15ev("DLS", bsdf, false,
                                        mlHeroPacketValid, mlHeroPacketValid, true, supported15ev,
                                        supported15ev ? "METAL2_SPECTRAL_DLS" : "METAL2_DLS_FALLBACK",
                                        1.f, mlHeroLaneCount, metal2LaneOK15ev, metal2LaneValue15ev,
                                        nullptr, nullptr);
                                MLHeroPathLogEndpoint15ev("DLS", classicContribution15ev,
                                    mlHeroLaneCount, laneScalar15ev, supported15ev);
                            }

                            if (useContinuation15ev) {
                                sampleResult->AddDirectLight(light->GetID(), event, Spectrum(1.f), continuationContribution15ev, 1.f);
                                MLHeroPathContinuationRecord15ev(light->GetID(), continuationContribution15ev);
// 15fg BEGIN
                                if (bsdf.GetMaterialType()==MATTETRANSLUCENT)
                                    MLHeroPathTranslucentApplied15fg(3u,event,&continuationContribution15ev,1u,false);
// 15fg END

                            } else if (usePacketContribution15ev) {
                                sampleResult->AddDirectLight(light->GetID(), event, Spectrum(1.f), packetContribution15ev, 1.f);
// 15fg BEGIN
                                if (useBSDFEVal && bsdf.GetMaterialType()==MATTETRANSLUCENT)
                                    MLHeroPathTranslucentApplied15fg(1u,event,&packetContribution15ev,1u,false);
// 15fg END
                            }
                            else
                                sampleResult->AddDirectLight(light->GetID(), event, pathThroughput, incomingRadiance, 1.f);

// 15fh BEGIN diagnostic-only
                            MLHeroRecord15fh(pathInfo.depth.depth==0?0u:1u,
                                useContinuation15ev?continuationContribution15ev:
                                    (usePacketContribution15ev?packetContribution15ev:classicContribution15ev),
                                useContinuation15ev||usePacketContribution15ev,pathInfo.depth.depth+1u);
// 15fh END
							// The first path vertex is not handled by AddDirectLight(). This is valid
							// for irradiance AOV only if it is not a SPECULAR material.
							//
							// Note: irradiance samples the light sources only here (i.e. no
							// direct hit, no MIS, it would be useless)
							//
							// Note: RR is ignored here because it can not happen on first path vertex
							if ((sampleResult->firstPathVertex) && !(bsdf.GetEventTypes() & SPECULAR))
								sampleResult->irradiance =
										(INV_PI * fabsf(Dot(bsdf.hitPoint.shadeN, shadowRay.d)) *
										factor) * connectionThroughput * lightRadiance;
						}

						return ILLUMINATED;
					} else
						return SHADOWED; 
				}
			}
		}
	}

	return NOT_VISIBLE;
}

bool PathTracer::CheckDirectHitVisibilityFlags(LightSourceConstRef lightSource, const PathDepthInfo &depthInfo,
		const BSDFEvent lastBSDFEvent) const {
	if (depthInfo.depth == 0)
		return true;

	if ((lastBSDFEvent & DIFFUSE) && lightSource.IsVisibleIndirectDiffuse())
		return true;
	if ((lastBSDFEvent & GLOSSY) && lightSource.IsVisibleIndirectGlossy())
		return true;
	if ((lastBSDFEvent & SPECULAR) && lightSource.IsVisibleIndirectSpecular())
		return true;

	return false;
}

void PathTracer::DirectHitFiniteLight(SceneConstRef scene,
		const EyePathInfo &pathInfo,
		const Spectrum &pathThroughput, const Ray &ray,
		const float distance, const BSDF &bsdf,
		SampleResult *sampleResult, const Spectrum *mlHeroLaneThroughput,
		const u_int mlHeroLaneCount, const bool mlHeroPacketValid) const {

	auto lightSource = bsdf.GetLightSource();

	// Check if the light source is visible according the settings
	if (!CheckDirectHitVisibilityFlags(*lightSource, pathInfo.depth, pathInfo.lastBSDFEvent) ||
			// If the material is shadow transparent, Direct Light sampling
			// will take care of transporting all emitted light
			bsdf.hitPoint.throughShadowTransparency)
		return;

	float directPdfA;
	const Spectrum emittedRadiance = bsdf.GetEmittedRadiance(&directPdfA);

	if (!emittedRadiance.Black()) {
		float weight;
		if (!(pathInfo.lastBSDFEvent & SPECULAR)) {
			auto& lightStrategy = scene.GetLightSources().GetIlluminateLightStrategy();
			const float lightPickProb = lightStrategy.SampleLightPdf(
				*lightSource,
				ray.o, pathInfo.lastShadeN, pathInfo.lastFromVolume);

			// This is a specific check to avoid fireflies with DLSC
			if ((lightPickProb == 0.f) && lightSource->IsDirectLightSamplingEnabled() &&
					(lightStrategy.GetType() == TYPE_DLS_CACHE))
				return;

			const float directPdfW = PdfAtoW(directPdfA, distance,
					AbsDot(bsdf.hitPoint.fixedDir, bsdf.hitPoint.shadeN));

			// MIS between BSDF sampling and direct light sampling
			weight = PowerHeuristic(pathInfo.lastBSDFPdfW * lightSource->GetAvgPassThroughTransparency(), directPdfW * lightPickProb);
		} else
			weight = 1.f;

        const Spectrum classicContribution15ev = pathThroughput * (weight * emittedRadiance);
        Spectrum continuationContribution15ev;
        const bool useContinuation15ev = MLHeroPathContinuationEndpoint15ev("FINITE", pathInfo.depth.depth,
            &continuationContribution15ev, classicContribution15ev, emittedRadiance, weight, Spectrum(1.f));
        bool usePacketContribution15ev = false;
        Spectrum packetContribution15ev;
        if (mlHeroEnabled && mlHeroLaneThroughput && mlHeroPacketValid && (mlHeroLaneCount > 1u)) {
            float laneScalar15ev[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (u_int lane15ev = 0; lane15ev < mlHeroLaneCount; ++lane15ev) {
                const float lambda15ev = GetMLHeroWaveLengthAt(lane15ev);
                laneScalar15ev[lane15ev] = mlHeroLaneThroughput[lane15ev].c[0] * weight *
                    MLHeroPathRGBIllumSample15ev(emittedRadiance, lambda15ev);
            }
            packetContribution15ev = MLHeroPathReconstructEndpoint15ev(
                mlHeroLaneCount, laneScalar15ev, true);
            usePacketContribution15ev = true;
            mlHeroPathEndpointFinite15ev.fetch_add(1ull, std::memory_order_relaxed);
            MLHeroPathLogEndpoint15ev("FINITE", classicContribution15ev,
                mlHeroLaneCount, laneScalar15ev, true);
        }

        if (useContinuation15ev) {
            sampleResult->AddEmission(bsdf.GetLightID(), Spectrum(1.f), continuationContribution15ev);
            MLHeroPathContinuationRecord15ev(bsdf.GetLightID(), continuationContribution15ev);
        } else if (usePacketContribution15ev)
            sampleResult->AddEmission(bsdf.GetLightID(), Spectrum(1.f), packetContribution15ev);
        else
            sampleResult->AddEmission(bsdf.GetLightID(), pathThroughput, weight * emittedRadiance);
// 15fh BEGIN diagnostic-only
        MLHeroRecord15fh(MLHeroHitBucket15fh(pathInfo.depth.depth),
            useContinuation15ev?continuationContribution15ev:
                (usePacketContribution15ev?packetContribution15ev:classicContribution15ev),
            useContinuation15ev||usePacketContribution15ev,pathInfo.depth.depth);
// 15fh END

	}
}

void PathTracer::DirectHitInfiniteLight(SceneConstRef scene,
		const EyePathInfo &pathInfo, const Spectrum &pathThroughput,
		const Ray &ray, const BSDF *bsdf, SampleResult *sampleResult,
		const Spectrum *mlHeroLaneThroughput, const u_int mlHeroLaneCount,
		const bool mlHeroPacketValid) const {
	// If the material is shadow transparent, Direct Light sampling
	// will take care of transporting all emitted light
	if (bsdf && bsdf->hitPoint.throughShadowTransparency)
		return;

	for(EnvLightSource& envLight: scene.GetLightSources().GetEnvLightSources()) {
		// Check if the light source is visible according the settings
		if (!CheckDirectHitVisibilityFlags(envLight, pathInfo.depth, pathInfo.lastBSDFEvent))
			continue;

		float directPdfW;
		const Spectrum envRadiance = envLight.GetRadiance(scene, bsdf, -ray.d, &directPdfW);
		if (!envRadiance.Black()) {
			float weight;
			if (!(pathInfo.lastBSDFEvent & SPECULAR)) {
				const float lightPickProb = scene.GetLightSources().GetIlluminateLightStrategy().
						SampleLightPdf(envLight, ray.o, pathInfo.lastShadeN, pathInfo.lastFromVolume);

				// MIS between BSDF sampling and direct light sampling
				weight = PowerHeuristic(pathInfo.lastBSDFPdfW, directPdfW * lightPickProb);
			} else
				weight = 1.f;

            const Spectrum classicContribution15ev = pathThroughput * (weight * envRadiance);
            Spectrum continuationContribution15ev;
            const bool useContinuation15ev = MLHeroPathContinuationEndpoint15ev("INFINITE", pathInfo.depth.depth,
                &continuationContribution15ev, classicContribution15ev, envRadiance, weight, Spectrum(1.f));
            bool usePacketContribution15ev = false;
            Spectrum packetContribution15ev;
            if (mlHeroEnabled && mlHeroLaneThroughput && mlHeroPacketValid && (mlHeroLaneCount > 1u)) {
                float laneScalar15ev[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                for (u_int lane15ev = 0; lane15ev < mlHeroLaneCount; ++lane15ev) {
                    const float lambda15ev = GetMLHeroWaveLengthAt(lane15ev);
                    laneScalar15ev[lane15ev] = mlHeroLaneThroughput[lane15ev].c[0] * weight *
                        MLHeroPathRGBIllumSample15ev(envRadiance, lambda15ev);
                }
                packetContribution15ev = MLHeroPathReconstructEndpoint15ev(
                    mlHeroLaneCount, laneScalar15ev, true);
                usePacketContribution15ev = true;
                mlHeroPathEndpointInfinite15ev.fetch_add(1ull, std::memory_order_relaxed);
                MLHeroPathLogEndpoint15ev("INFINITE", classicContribution15ev,
                    mlHeroLaneCount, laneScalar15ev, true);
            }

            if (useContinuation15ev) {
                sampleResult->AddEmission(envLight.GetID(), Spectrum(1.f), continuationContribution15ev);
                MLHeroPathContinuationRecord15ev(envLight.GetID(), continuationContribution15ev);
            } else if (usePacketContribution15ev)
                sampleResult->AddEmission(envLight.GetID(), Spectrum(1.f), packetContribution15ev);
            else
                sampleResult->AddEmission(envLight.GetID(), pathThroughput, weight * envRadiance);
// 15fh BEGIN diagnostic-only
        MLHeroRecord15fh(MLHeroHitBucket15fh(pathInfo.depth.depth),
            useContinuation15ev?continuationContribution15ev:
                (usePacketContribution15ev?packetContribution15ev:classicContribution15ev),
            useContinuation15ev||usePacketContribution15ev,pathInfo.depth.depth);
// 15fh END

		}
	}	
}

void PathTracer::GenerateEyeRay(CameraConstRef camera, FilmConstRef film, Ray &eyeRay,
		PathVolumeInfo &volInfo, Sampler& sampler, SampleResult &sampleResult) const {
	const float filmX = sampler.GetSample(0);
	const float filmY = sampler.GetSample(1);

	// Use fast pixel filtering, like the one used in TILEPATH.

	const u_int *subRegion = film.GetSubRegion();
	sampleResult.pixelX = Min(Floor2UInt(filmX), subRegion[1]);
	sampleResult.pixelY = Min(Floor2UInt(filmY), subRegion[3]);
	assert (sampleResult.pixelX >= subRegion[0]);
	assert (sampleResult.pixelX <= subRegion[1]);
	assert (sampleResult.pixelY >= subRegion[2]);
	assert (sampleResult.pixelY <= subRegion[3]);

	const float uSubPixelX = filmX - sampleResult.pixelX;
	const float uSubPixelY = filmY - sampleResult.pixelY;

	// Sample according the pixel filter distribution
	float distX, distY;
	pixelFilterDistribution->SampleContinuous(uSubPixelX, uSubPixelY, &distX, &distY);

	sampleResult.filmX = sampleResult.pixelX + .5f + distX;
	sampleResult.filmY = sampleResult.pixelY + .5f + distY;

	const float timeSample = sampler.GetSample(4);
	const float time = camera.GenerateRayTime(timeSample);

	camera.GenerateRay(time, sampleResult.filmX, sampleResult.filmY, &eyeRay, &volInfo,
		sampler.GetSample(2), sampler.GetSample(3));
}

//------------------------------------------------------------------------------
// RenderEyePath methods
//------------------------------------------------------------------------------

void PathTracer::RenderEyePath(IntersectionDevice *device,
		SceneConstRef scene, Sampler& sampler, EyePathInfo &pathInfo,
		Ray &eyeRay,  const luxrays::Spectrum &eyeTroughput,
		vector<SampleResult> &sampleResults) const {
	mlHeroContinuation15ev = MLHeroPathContinuation15ev();
    const bool spectralRR15fz=mlHeroRRCPUPath15fz && !hybridBackForwardEnable &&
        mlHeroEnabled && mlHeroWavelengthCount>1 &&
        (mlHeroRRGlassEtaExperiment || mlHeroRRMaterialExperiment);
    bool etaValid15fz=true;
    double eta15fz[8]={1.,1.,1.,1.,1.,1.,1.,1.};
// 15fg BEGIN
    mlHeroPathComp15fg=mlHeroGenericReflectanceCompensation;
    mlHeroPathDiag15fg=mlHeroDiagnostics15fg;
    mlHeroPathNativeSample15fg=false;
// 15fg END

	// To keep track of the number of rays traced
	const double deviceRayCount = device->GetTotalRaysCount();

	// This is used by light strategy
	pathInfo.lastShadeN = Normal(eyeRay.d);

	SampleResult &sampleResult = sampleResults[0];
	bool photonGIShowIndirectPathMixUsed = false;
	bool photonGICausticCacheUsed = false;
	bool photonGICacheEnabledOnLastHit = false;
	bool albedoToDo = true;
	sampleResult.albedo = Spectrum(); // Just in case albedoToDo is never true
	sampleResult.shadingNormal = Normal();
	Spectrum pathThroughput(eyeTroughput);

	// ML HERO 15ev: parallel packet throughput. Supported DLS/FINITE/INFINITE
	// endpoints now reconstruct normalized packet RGB into sampleResult. Unsupported
	// material/event cases preserve the proven HERO-1.0 lane-0 fallback.
	Spectrum mlHeroLaneThroughput15ev[8];
	const u_int mlHeroLaneCount15ev = mlHeroEnabled ? Clamp((u_int)mlHeroWavelengthCount, 1u, 8u) : 1u;
	bool mlHeroPacketTransportValid15ev = mlHeroEnabled && (mlHeroLaneCount15ev > 1u);
	for (u_int lane15ev = 0; lane15ev < 8u; ++lane15ev)
		mlHeroLaneThroughput15ev[lane15ev] = eyeTroughput;

	// ML HERO split-radiance test:
	// Radiance accumulated before the first dispersive GLASS event stays
	// neutral. Only radiance added after that event is HERO-colored.
	bool mlDispersionSnapshotTaken = false;
	std::vector<Spectrum> mlRadianceBeforeDispersion;

	BSDF bsdf;
	for (;;) {
		sampleResult.firstPathVertex = (pathInfo.depth.depth == 0);
		const u_int sampleOffset = eyeSampleBootSize + pathInfo.depth.depth * eyeSampleStepSize;

		RayHit eyeRayHit;
		Spectrum connectionThroughput;
		const float passThrough = sampler.GetSample(sampleOffset);
		const bool hit = scene.Intersect(device,
				EYE_RAY | (sampleResult.firstPathVertex ? CAMERA_RAY : INDIRECT_RAY),
				&pathInfo.volume, passThrough,
				&eyeRay, &eyeRayHit, &bsdf, &connectionThroughput,
				&pathThroughput, &sampleResult);
		pathThroughput *= connectionThroughput;
		MLHeroPathContinuationConnection15ev(connectionThroughput, pathInfo.depth.depth);
		if (mlHeroPacketTransportValid15ev) {
			if (MLHeroPathIsScalarSpectrum15ev(connectionThroughput)) {
				const float c15ev = (connectionThroughput.c[0] + connectionThroughput.c[1] + connectionThroughput.c[2]) / 3.f;
				for (u_int lane15ev = 0; lane15ev < mlHeroLaneCount15ev; ++lane15ev)
					mlHeroLaneThroughput15ev[lane15ev] *= Spectrum(c15ev);
			} else {
				mlHeroPacketTransportValid15ev = false;
				mlHeroPathConnectionFallbackCount15ev.fetch_add(1ull, std::memory_order_relaxed);
			}
		}
		// Note: pass-through check is done inside Scene::Intersect()

		const bool checkDirectLightHit =
				// Avoid to render caustic path if hybridBackForwardEnable
				(!hybridBackForwardEnable || !pathInfo.IsCausticPath()) &&
				// Avoid to render caustic path if PhotonGI caustic cache is enabled
				(!photonGICache ||
					photonGICache->IsDirectLightHitVisible(pathInfo, photonGICausticCacheUsed));

		if (!hit) {
			// Nothing was hit, look for env. lights
			if ((!(forceBlackBackground && pathInfo.isPassThroughPath) || !pathInfo.isPassThroughPath) &&
					checkDirectLightHit) {
				DirectHitInfiniteLight(scene, pathInfo, pathThroughput,
						eyeRay, sampleResult.firstPathVertex ? nullptr : &bsdf,
						&sampleResult, mlHeroLaneThroughput15ev, mlHeroLaneCount15ev, mlHeroPacketTransportValid15ev);
			}

			if (sampleResult.firstPathVertex) {
				sampleResult.alpha = 0.f;
				sampleResult.depth = numeric_limits<float>::infinity();
				sampleResult.position = Point(
						numeric_limits<float>::infinity(),
						numeric_limits<float>::infinity(),
						numeric_limits<float>::infinity());
				sampleResult.geometryNormal = Normal();
				sampleResult.shadingNormal = Normal();
				sampleResult.materialID = 0;
				sampleResult.objectID = 0;
				sampleResult.uv = UV(numeric_limits<float>::infinity(),
						numeric_limits<float>::infinity());
			} else if (!sampleResult.isHoldout && pathInfo.isTransmittedPath) {
				// I set to 0.0 also the alpha all purely transmitted paths hitting nothing
				sampleResult.alpha = 0.f;
			}
			break;
		}

		// Something was hit

		if (albedoToDo && bsdf.IsAlbedoEndPoint(albedoSpecularSetting, albedoSpecularGlossinessThreshold)) {
			sampleResult.albedo = pathThroughput * bsdf.Albedo();
			sampleResult.shadingNormal = bsdf.hitPoint.shadeN;
			albedoToDo = false;
		}

		if (sampleResult.firstPathVertex) {
			// The alpha value can be changed if the material is a shadow catcher (see below)
			sampleResult.alpha = bsdf.IsHoldout() ? 0.f : 1.f;
			sampleResult.depth = eyeRayHit.t;
			sampleResult.position = bsdf.hitPoint.p;
			sampleResult.geometryNormal = bsdf.hitPoint.geometryN;
			sampleResult.materialID = bsdf.GetMaterialID();
			sampleResult.objectID = bsdf.GetObjectID();
			sampleResult.uv = bsdf.hitPoint.GetUV(0);
			sampleResult.isHoldout = bsdf.IsHoldout();
		}
		sampleResult.lastPathVertex = pathInfo.depth.IsLastPathVertex(maxPathDepth, bsdf.GetEventTypes());

		//----------------------------------------------------------------------
		// Check if it is a baked material
		//----------------------------------------------------------------------

		if (bsdf.HasBakeMap(COMBINED)) {
			sampleResult.radiance[0] += pathThroughput * bsdf.GetBakeMapValue();
			break;
		} else if (bsdf.HasBakeMap(LIGHTMAP)) {
			sampleResult.radiance[0] += pathThroughput * bsdf.Albedo() * bsdf.GetBakeMapValue();
			break;
		}

		//----------------------------------------------------------------------
		// Check if it is a light source and I have to add light emission
		//----------------------------------------------------------------------

		if (bsdf.IsLightSource() && checkDirectLightHit) {
			DirectHitFiniteLight(scene, pathInfo, pathThroughput,
					eyeRay, eyeRayHit.t, bsdf, &sampleResult, mlHeroLaneThroughput15ev, mlHeroLaneCount15ev, mlHeroPacketTransportValid15ev);
		}

		//----------------------------------------------------------------------
		// Check if I can use the photon cache
		//----------------------------------------------------------------------

		if (photonGICache) {
			const bool isPhotonGIEnabled = photonGICache->IsPhotonGIEnabled(bsdf);

			// Check if one of the debug modes is enabled
			if (photonGICache->GetDebugType() == PhotonGIDebugType::PGIC_DEBUG_SHOWINDIRECT) {
				if (isPhotonGIEnabled) {
					const SpectrumGroup *group = photonGICache->GetIndirectRadiance(bsdf);
					if (group)
						sampleResult.radiance += *group;
				}
				break;
			} else if (photonGICache->GetDebugType() == PhotonGIDebugType::PGIC_DEBUG_SHOWCAUSTIC) {
				if (isPhotonGIEnabled)
					sampleResult.radiance += photonGICache->ConnectWithCausticPaths(bsdf);
				break;
			} else if (photonGICache->GetDebugType() == PhotonGIDebugType::PGIC_DEBUG_SHOWINDIRECTPATHMIX) {
				// Check if the cache is enabled for this material
				if (isPhotonGIEnabled) {
					if (photonGICacheEnabledOnLastHit &&
							(eyeRayHit.t > photonGICache->GetIndirectUsageThreshold(pathInfo.lastBSDFEvent,
								pathInfo.lastGlossiness,
								passThrough))) {
						sampleResult.radiance[0] = Spectrum(0.f, 0.f, 1.f);
						photonGIShowIndirectPathMixUsed = true;
						break;
					}

					photonGICacheEnabledOnLastHit = true;
				} else
					photonGICacheEnabledOnLastHit = false;
			} else {
				// Check if the cache is enabled for this material
				if (isPhotonGIEnabled) {
					// TODO: add support for AOVs (possible ?)

					if (photonGICache->IsCausticEnabled() && (!hybridBackForwardEnable || pathInfo.depth.depth != 0)) {
						const SpectrumGroup causticRadiance = photonGICache->ConnectWithCausticPaths(bsdf);

						if (!causticRadiance.Black()) {
							sampleResult.radiance.AddWeighted(pathThroughput, causticRadiance);
							photonGICausticCacheUsed = true;
						}
					}

					if (photonGICache->IsIndirectEnabled() && photonGICacheEnabledOnLastHit &&
							(eyeRayHit.t > photonGICache->GetIndirectUsageThreshold(pathInfo.lastBSDFEvent,
								pathInfo.lastGlossiness,
								// I hope to not introduce strange sample correlations
								// by using passThrough here
								passThrough))) {
						const SpectrumGroup *group = photonGICache->GetIndirectRadiance(bsdf);
						if (group)
							sampleResult.radiance.AddWeighted(pathThroughput, *group);
						// I can terminate the path, all done
						break;
					}

					photonGICacheEnabledOnLastHit = true;
				} else
					photonGICacheEnabledOnLastHit = false;
			}
		}

		//------------------------------------------------------------------
		// Direct light sampling
		//------------------------------------------------------------------

		// I avoid to do DL on the last vertex otherwise it introduces a lot of
		// noise because I can not use MIS.
		// I handle as a special case when the path vertex is both the first
		// and the last: I do direct light sampling without MIS.
		if (sampleResult.lastPathVertex && !sampleResult.firstPathVertex)
			break;

        if (mlHeroEnabled && (bsdf.GetMaterialType() == ROUGHGLASS || bsdf.GetMaterialType() == GLASS))
            MLHeroPathContinuationStart15ev(bsdf, GLOSSY, Spectrum(1.f),
                mlHeroPacketTransportValid15ev, mlHeroLaneCount15ev,
                mlHeroLaneThroughput15ev, pathInfo.depth.depth);
		const DirectLightResult directLightResult = DirectLightSampling(
				device, scene,
				eyeRay.time,
				sampler.GetSample(sampleOffset + 1),
				sampler.GetSample(sampleOffset + 2),
				sampler.GetSample(sampleOffset + 3),
				sampler.GetSample(sampleOffset + 4),
				sampler.GetSample(sampleOffset + 5),
				pathInfo, 
				pathThroughput, bsdf, &sampleResult, true,
				mlHeroLaneThroughput15ev, mlHeroLaneCount15ev, mlHeroPacketTransportValid15ev);

		if (sampleResult.lastPathVertex)
			break;

		//------------------------------------------------------------------
		// Build the next vertex path ray
		//------------------------------------------------------------------

		Vector sampledDir;
		float cosSampledDir;
		Spectrum bsdfSample;
		float bsdfPdfW;
// 15fg BEGIN
    mlHeroPathNativeSample15fg=false;
// 15fg END
		BSDFEvent bsdfEvent;
		if (bsdf.IsShadowCatcher() && (directLightResult != SHADOWED)) {
			bsdfSample = bsdf.ShadowCatcherSample(&sampledDir, &bsdfPdfW, &cosSampledDir, &bsdfEvent);

			if (sampleResult.firstPathVertex) {
				// In this case I have also to set the value of the alpha channel to 0.0
				sampleResult.alpha = 0.f;
			}
		} else {
			const Spectrum &shadowTransparency = bsdf.GetPassThroughShadowTransparency();
			if (!sampleResult.firstPathVertex && !shadowTransparency.Black() && !pathInfo.IsSpecularPath()) {
				sampledDir = -bsdf.hitPoint.fixedDir;
				bsdfSample = shadowTransparency;
				bsdfPdfW = pathInfo.lastBSDFPdfW;
				cosSampledDir = -1.f;
				bsdfEvent = pathInfo.lastBSDFEvent;
			} else {
				bsdfSample = bsdf.Sample(&sampledDir,
						sampler.GetSample(sampleOffset + 6),
						sampler.GetSample(sampleOffset + 7),
						&bsdfPdfW, &cosSampledDir, &bsdfEvent);
				pathInfo.isPassThroughPath = false;
// 15fg BEGIN
                mlHeroPathNativeSample15fg=true;
// 15fg END

			}
		}

		verify (!bsdfSample.IsNaN() && !bsdfSample.IsInf() && !bsdfSample.IsNeg());
		if (mlHeroEnabled)
			MLHeroPathContinuationStart15ev(bsdf, bsdfEvent, bsdfSample,
				mlHeroPacketTransportValid15ev, mlHeroLaneCount15ev,
				mlHeroLaneThroughput15ev, pathInfo.depth.depth + 1u);
		const unsigned long long glassAuditId15ev = mlHeroEnabled ?
			MLHeroPathLogGlassSample15ev(pathInfo.depth.depth + 1u, bsdf, sampledDir,
				bsdfEvent, bsdfPdfW, bsdfSample, mlHeroPacketTransportValid15ev,
				mlDispersionSnapshotTaken, mlHeroLaneCount15ev) : 0ull;
		Spectrum glassTpBefore15ev[8];
		if (glassAuditId15ev != 0ull)
			for (u_int lane = 0; lane < mlHeroLaneCount15ev; ++lane)
				glassTpBefore15ev[lane] = mlHeroLaneThroughput15ev[lane];
		if (bsdfSample.Black()) {
			MLHeroPathLogGlassTransport15ev(glassAuditId15ev, "BLACK_SAMPLE",
				mlHeroPacketTransportValid15ev, mlDispersionSnapshotTaken, 1.f,
				mlHeroLaneCount15ev, glassTpBefore15ev, mlHeroLaneThroughput15ev);
			break;
		}

		if (mlHeroEnabled)
			MLHeroPathLogMaterialAudit15ev(pathInfo.depth.depth + 1u, bsdf, bsdfEvent,
				bsdfSample, bsdfPdfW, cosSampledDir);

		// Snapshot before transport that requires the single-wavelength fallback.
		// 15ev: spectral Metal2 also sets the legacy color flag, but its
		// supported packet must stay active. Generic markers (e.g. spectral
		// volume absorption), single-lane rendering and already invalid packets
		// retain the existing legacy coloring behavior.
		if (mlHeroEnabled && !mlDispersionSnapshotTaken &&
				(GetMLHeroPacketFallbackUsed() ||
				 (!mlHeroPacketTransportValid15ev && GetMLDispersionUsed()))) {
			mlRadianceBeforeDispersion.resize(sampleResult.radiance.Size());
			for (u_int i = 0; i < sampleResult.radiance.Size(); ++i)
				mlRadianceBeforeDispersion[i] = sampleResult.radiance[i];

			mlDispersionSnapshotTaken = true;
		}

		// Dispersive Glass samples the actual ray with the HERO wavelength.
		// Companion wavelengths cannot share its transmission direction; even
		// reflection has wavelength-dependent Fresnel/branch probabilities.
		// Retire the packet for the rest of this path, including endpoints.
		// Keep the existing lane-0 RGB throughput and split-radiance estimator:
		// GetMLDispersionSampleColor already includes the HERO sampling weight.
		// There is no packet 1/N average here, so no extra N multiplier is needed.
		if (mlHeroEnabled && mlDispersionSnapshotTaken)
			mlHeroPacketTransportValid15ev = false;

		if (sampleResult.firstPathVertex)
			sampleResult.firstPathVertexEvent = bsdfEvent;

		pathInfo.AddVertex(bsdf, bsdfEvent, bsdfPdfW, hybridBackForwardGlossinessThreshold);

        // 15fz: retain camera eta history independently of diagnostic logging.
        const bool idealGlass15fz=bsdf.GetMaterialType()==GLASS && (bsdfEvent & SPECULAR);
        if(spectralRR15fz && (bsdfEvent & TRANSMIT)) {
            if(!idealGlass15fz || !mlHeroPathNativeSample15fg || !etaValid15fz)
                etaValid15fz=false;
            else {
                const u_int etaCount15fz=(mlHeroContinuation15ev.active && mlHeroContinuation15ev.collapsed) ||
                    bsdf.HasMLHeroGlassDispersion() ? 1u : mlHeroLaneCount15ev;
                double nextEta15fz[8];
                for(u_int i=0;i<8u;++i)nextEta15fz[i]=eta15fz[i];
                for(u_int i=0;i<etaCount15fz;++i) {
                    float nc=0,nt=0,ntLambda=0,fresnel=0,eta=0,eta2=0,transport=0,delta=0;
                    Spectrum debug;
                    if(!bsdf.EvaluateMLHeroGlassDebugAtWaveLength(sampledDir,bsdfEvent,bsdfPdfW,
                        GetMLHeroWaveLengthAt(i),&nc,&nt,&ntLambda,&fresnel,&eta,&eta2,&transport,&delta,&debug) ||
                        !MLHeroPathEtaAdvance15fz(nextEta15fz[i],eta2,&nextEta15fz[i]))etaValid15fz=false;
                }
                if(etaValid15fz)for(u_int i=0;i<8u;++i)eta15fz[i]=nextEta15fz[i];
            }
        }
        const bool materialScope15fz=bsdf.GetMaterialType()==MATTE ||
            bsdf.GetMaterialType()==METAL2 || bsdf.GetMaterialType()==GLOSSY2;
        const bool nativeRR15fz=pathInfo.UseRR(rrDepth);
        const bool glassRR15fz=spectralRR15fz && mlHeroRRGlassEtaExperiment &&
            idealGlass15fz && pathInfo.depth.depth>=rrDepth;
        const bool deferRR15fz=spectralRR15fz && mlHeroPathNativeSample15fg &&
            (glassRR15fz || (nativeRR15fz && mlHeroRRMaterialExperiment && materialScope15fz));

		// Russian Roulette

    if (mlHeroState15fh.enabled) MLHeroRRAudit15fk(6,pathInfo.depth.GetRRDepth(),bsdf.GetMaterialType(),bsdfEvent,bsdfSample,bsdfPdfW,0.f,rrImportanceCap,RenderEngine::RussianRouletteProb(bsdfSample,rrImportanceCap),pathInfo.UseRR(rrDepth),pathInfo.UseRR(rrDepth),true);
		float rrProb = 1.f;
		if (nativeRR15fz && !deferRR15fz) {
			 rrProb = RenderEngine::RussianRouletteProb(bsdfSample, rrImportanceCap);
			if (rrProb < sampler.GetSample(sampleOffset + 8)) {
				MLHeroPathLogGlassTransport15ev(glassAuditId15ev, "RR_TERMINATED",
					mlHeroPacketTransportValid15ev, mlDispersionSnapshotTaken, rrProb,
					mlHeroLaneCount15ev, glassTpBefore15ev, mlHeroLaneThroughput15ev);
				break;
			}

			// Increase path contribution
			bsdfSample /= rrProb;
		}

		if (mlHeroEnabled && (bsdf.GetMaterialType() == GLOSSY2))
			MLHeroPathLogGlossy2Decomposition15ev(pathInfo.depth.depth, bsdf, sampledDir, bsdfPdfW,
				bsdfEvent, bsdfSample * rrProb, rrProb);

		MLHeroPathContinuationBounce15ev(bsdf, sampledDir, bsdfEvent, bsdfPdfW,
			bsdfSample * rrProb, rrProb, pathInfo.depth.depth);

		const char *glassMode15ev = mlDispersionSnapshotTaken ?
			"DISPERSION_HERO_LANE0" : "PACKET_ALREADY_INVALID";
		const bool glossy2Audit15ev = mlHeroEnabled && (bsdf.GetMaterialType() == GLOSSY2);
		const bool glossy2ValidBefore15ev = mlHeroPacketTransportValid15ev;
		Spectrum glossy2TpBefore15ev[8];
		bool glossy2HelperOK15ev[8] = {};
		float glossy2AppliedFactor15ev[8] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
		const char *glossy2Mode15ev = "PACKET_ALREADY_INVALID";
		if (glossy2Audit15ev)
			for (u_int lane = 0; lane < mlHeroLaneCount15ev; ++lane)
				glossy2TpBefore15ev[lane] = mlHeroLaneThroughput15ev[lane];

		const bool metal2Audit15ev = mlHeroEnabled && (bsdf.GetMaterialType() == METAL2);
		const bool metal2ValidBefore15ev = mlHeroPacketTransportValid15ev;
		Spectrum metal2TpBefore15ev[8];
		bool metal2LaneOK15ev[8] = {};
		float metal2LaneValue15ev[8] = {};
		const char *metal2Mode15ev = "PACKET_ALREADY_INVALID";
		if (metal2Audit15ev)
			for (u_int lane = 0; lane < mlHeroLaneCount15ev; ++lane)
				metal2TpBefore15ev[lane] = mlHeroLaneThroughput15ev[lane];

		if (mlHeroPacketTransportValid15ev) {
			const bool scalar15ev = MLHeroPathIsScalarSpectrum15ev(bsdfSample);
			const bool matte15ev = (bsdf.GetMaterialType() == MATTE) &&
				(bsdfEvent & DIFFUSE) && (bsdfEvent & REFLECT) && !(bsdfEvent & SPECULAR);
			const bool glossy215ev = (bsdf.GetMaterialType() == GLOSSY2) &&
				(bsdfEvent & REFLECT) && !(bsdfEvent & SPECULAR);
			const char *mode15ev = "UNSUPPORTED";

// 15fg BEGIN
            const bool translucent15fg=bsdf.GetMaterialType()==MATTETRANSLUCENT && mlHeroPathNativeSample15fg;
            if(translucent15fg) {
                float factors15fg[8]={};
                const bool ok15fg=MLHeroPathTranslucentLanes15fg(bsdf,sampledDir,bsdfEvent,true,
                    mlHeroLaneCount15ev,rrProb,mlHeroGenericReflectanceCompensation,0u,factors15fg);
                if(ok15fg) {
                    for(u_int lane=0u;lane<mlHeroLaneCount15ev;++lane)
                        mlHeroLaneThroughput15ev[lane]*=Spectrum(factors15fg[lane]);
                    mode15ev="MATTETRANSLUCENT_NATIVE_SPECTRAL";
                    MLHeroPathTranslucentApplied15fg(0u,bsdfEvent,mlHeroLaneThroughput15ev,mlHeroLaneCount15ev,true);
                } else {
                    mlHeroPacketTransportValid15ev=false;
                    mlHeroPathUnsupportedBounceCount15ev.fetch_add(1ull,std::memory_order_relaxed);
                    mode15ev="MATTETRANSLUCENT_HELPER_REJECT";
                }
            } else
// 15fg END
			if (matte15ev) {
				const Spectrum kd15ev = bsdf.Albedo().Clamp(0.f, 1.f);
				float probe15ev = 0.f;
				float fitError15ev = 0.f;
				const bool fit15ev = MLHeroPathMatteLaneMultiplier15ev(kd15ev, bsdfSample,
					GetMLHeroWaveLengthAt(0), &probe15ev, &fitError15ev);
				if (fit15ev) {
					for (u_int lane15ev = 0; lane15ev < mlHeroLaneCount15ev; ++lane15ev) {
						float laneMultiplier15ev = 0.f;
						MLHeroPathMatteLaneMultiplier15ev(kd15ev, bsdfSample,
							GetMLHeroWaveLengthAt(lane15ev), &laneMultiplier15ev);
						mlHeroLaneThroughput15ev[lane15ev] *= Spectrum(laneMultiplier15ev);
					}
					if (scalar15ev) {
						mlHeroPathScalarBounceCount15ev.fetch_add(1ull, std::memory_order_relaxed);
						mode15ev = "MATTE_SCALAR";
					} else {
						mlHeroPathSpectralReflectBounceCount15ev.fetch_add(1ull, std::memory_order_relaxed);
						mode15ev = "MATTE_KD_SPECTRAL";
					}
				} else {
					mlHeroPacketTransportValid15ev = false;
					mlHeroPathUnsupportedBounceCount15ev.fetch_add(1ull, std::memory_order_relaxed);
					mode15ev = "MATTE_FIT_FAIL";
				}
			} else if (glossy215ev) {
				// Path CPU applies Russian-Roulette survival by dividing bsdfSample.
				// The Glossy2 component helper reconstructs the physical pre-RR BSDF,
				// so undo RR for the decomposition and reapply it to every lane once.
				const Spectrum glossy2Physical15ev = bsdfSample * rrProb;
				float probe15ev = 0.f;
				float fitError15ev = 0.f;
				const bool fit15ev = MLHeroPathGlossy2LaneMultiplier15ev(bsdf, sampledDir, bsdfPdfW,
					bsdfEvent, glossy2Physical15ev, GetMLHeroWaveLengthAt(0), &probe15ev, &fitError15ev);
				if (fit15ev) {
					const float invRR15ev = 1.f / rrProb;
					for (u_int lane15ev = 0; lane15ev < mlHeroLaneCount15ev; ++lane15ev) {
						float laneMultiplier15ev = 0.f;
						glossy2HelperOK15ev[lane15ev] = MLHeroPathGlossy2LaneMultiplier15ev(bsdf, sampledDir, bsdfPdfW, bsdfEvent,
							glossy2Physical15ev, GetMLHeroWaveLengthAt(lane15ev), &laneMultiplier15ev);
						glossy2AppliedFactor15ev[lane15ev] = laneMultiplier15ev * invRR15ev;
						mlHeroLaneThroughput15ev[lane15ev] *= Spectrum(glossy2AppliedFactor15ev[lane15ev]);
					}
					mlHeroPathSpectralReflectBounceCount15ev.fetch_add(1ull, std::memory_order_relaxed);
					mode15ev = "GLOSSY2_KD_SPECTRAL";
				} else {
					mlHeroPacketTransportValid15ev = false;
					mlHeroPathUnsupportedBounceCount15ev.fetch_add(1ull, std::memory_order_relaxed);
					mode15ev = "GLOSSY2_SPLIT_REJECT";
				}
			} else if (metal2Audit15ev) {
				const bool metal2Fit15ev = MLHeroPathMetal2Lanes15ev(bsdf, sampledDir, true,
					mlHeroLaneCount15ev, rrProb, metal2LaneOK15ev, metal2LaneValue15ev);
				if (metal2Fit15ev) {
					for (u_int lane = 0; lane < mlHeroLaneCount15ev; ++lane)
						mlHeroLaneThroughput15ev[lane] *= Spectrum(metal2LaneValue15ev[lane]);
					mlHeroPathSpectralReflectBounceCount15ev.fetch_add(1ull, std::memory_order_relaxed);
					mode15ev = "METAL2_SPECTRAL";
				} else {
					mlHeroPacketTransportValid15ev = false;
					mlHeroPathUnsupportedBounceCount15ev.fetch_add(1ull, std::memory_order_relaxed);
					mode15ev = "METAL2_HELPER_REJECT";
				}
			} else if (scalar15ev) {
				const float m15ev = (bsdfSample.c[0] + bsdfSample.c[1] + bsdfSample.c[2]) / 3.f;
				if (glossy2Audit15ev)
					for (u_int lane = 0; lane < mlHeroLaneCount15ev; ++lane)
						glossy2AppliedFactor15ev[lane] = m15ev;
				for (u_int lane15ev = 0; lane15ev < mlHeroLaneCount15ev; ++lane15ev)
					mlHeroLaneThroughput15ev[lane15ev] *= Spectrum(m15ev);
				mlHeroPathScalarBounceCount15ev.fetch_add(1ull, std::memory_order_relaxed);
				mode15ev = "SCALAR";
			} else {
				mlHeroPacketTransportValid15ev = false;
				mlHeroPathUnsupportedBounceCount15ev.fetch_add(1ull, std::memory_order_relaxed);
			}

			glassMode15ev = mode15ev;
			glossy2Mode15ev = mode15ev;
			metal2Mode15ev = mode15ev;
			mlHeroPathCompanionBounceCount15ev.fetch_add(1ull, std::memory_order_relaxed);
			MLHeroPathLogBounce15ev(pathInfo.depth.depth, bsdfEvent, bsdf.GetMaterialType(),
				mode15ev, mlHeroPacketTransportValid15ev, mlHeroLaneCount15ev, mlHeroLaneThroughput15ev);
		}

        // 15fz: the existing transport above ran with rrProb=1 on deferred bounces.
        // Select from ACTUAL post-material spectral lanes, then divide each live
        // representation and the classic multiplier exactly once. PATH collapse
        // uses count=1 without BIDIR's xN, so there is no /configuredN here.
        if(deferRR15fz) {
            const bool useContinuation15fz=mlHeroContinuation15ev.active;
            const bool valid15fz=etaValid15fz && (useContinuation15fz ?
                mlHeroContinuation15ev.valid : mlHeroPacketTransportValid15ev);
            const Spectrum *post15fz=useContinuation15fz ?
                mlHeroContinuation15ev.tp : mlHeroLaneThroughput15ev;
            const u_int count15fz=useContinuation15fz ?
                mlHeroContinuation15ev.count : mlHeroLaneCount15ev;
            double beta15fz=0.;
            bool candidateValid15fz=valid15fz && count15fz>0u && count15fz<=8u;
            for(u_int i=0;candidateValid15fz && i<count15fz;++i) {
                const Spectrum &v=post15fz[i];
                double b=0.;
                candidateValid15fz=MLHeroPathRRScalar15fz(v.c[0],v.c[1],v.c[2],eta15fz[i],&b);
                beta15fz=std::max(beta15fz,b);
            }
            const float baseline15fz=nativeRR15fz ?
                RenderEngine::RussianRouletteProb(bsdfSample,rrImportanceCap) : 1.f;
            bool selected15fz=false;
            rrProb=MLHeroPathRRSelect15fz(baseline15fz,candidateValid15fz,beta15fz,
                rrImportanceCap,&selected15fz);
            const float random15fz=sampler.GetSample(sampleOffset+8);
            const bool killed15fz=rrProb<=0.f || rrProb<random15fz;
            if(mlHeroDiagnostics15fg) {
                const unsigned kind15fz=glassRR15fz?0u:(bsdf.GetMaterialType()==MATTE?1u:
                    (bsdf.GetMaterialType()==METAL2?2u:3u));
                MLHeroPathRRRecord15fz(kind15fz,selected15fz,baseline15fz,rrProb,random15fz,killed15fz);
                if(mlHeroPathRRStats15fz.details++<24u) {
                    std::lock_guard<std::mutex> lock(mlHeroPathDebugMutex15ev);
                    FILE *f=fopen(ML_HERO_PATH_LOG_FILE,"a");
                    if(f) {
                        fprintf(f,"phase15fz PATH_RR_DECISION material=%u depth=%u nativeRR=%u glassRR=%u selected=%u validEta=%u continuation=%u activeLanes=%u beta=%.17g baselineQ=%.9g liveQ=%.9g random=%.9g killed=%u currentRRExcluded=1 compensationOnce=1\n",
                            (u_int)bsdf.GetMaterialType(),pathInfo.depth.depth,nativeRR15fz?1u:0u,
                            glassRR15fz?1u:0u,selected15fz?1u:0u,etaValid15fz?1u:0u,
                            useContinuation15fz?1u:0u,count15fz,beta15fz,baseline15fz,rrProb,random15fz,killed15fz?1u:0u);
                        fclose(f);
                    }
                }
            }
            if(killed15fz)break;
            bsdfSample/=rrProb;
            if(mlHeroPacketTransportValid15ev)
                for(u_int i=0;i<mlHeroLaneCount15ev;++i)mlHeroLaneThroughput15ev[i]/=rrProb;
            if(mlHeroContinuation15ev.active && mlHeroContinuation15ev.valid)
                for(u_int i=0;i<mlHeroContinuation15ev.count;++i)mlHeroContinuation15ev.tp[i]/=rrProb;
            for(u_int i=0;i<mlHeroLaneCount15ev;++i) {
                glossy2AppliedFactor15ev[i]/=rrProb;
                metal2LaneValue15ev[i]/=rrProb;
            }
        }

		if (metal2Audit15ev)
			MLHeroPathLogMetal2Audit15ev("BOUNCE", bsdf, true,
				metal2ValidBefore15ev, mlHeroPacketTransportValid15ev,
				metal2ValidBefore15ev, metal2ValidBefore15ev && mlHeroPacketTransportValid15ev,
				metal2Mode15ev, rrProb, mlHeroLaneCount15ev, metal2LaneOK15ev,
				metal2LaneValue15ev, metal2TpBefore15ev, mlHeroLaneThroughput15ev);

		if (glossy2Audit15ev)
			MLHeroPathLogGlossy2Transport15ev(pathInfo.depth.depth, bsdfEvent,
				glossy2ValidBefore15ev, mlHeroPacketTransportValid15ev, glossy2Mode15ev,
				rrProb, mlHeroLaneCount15ev, glossy2HelperOK15ev,
				glossy2AppliedFactor15ev, glossy2TpBefore15ev, mlHeroLaneThroughput15ev);

		MLHeroPathLogGlassTransport15ev(glassAuditId15ev, glassMode15ev,
			mlHeroPacketTransportValid15ev, mlDispersionSnapshotTaken, rrProb,
			mlHeroLaneCount15ev, glassTpBefore15ev, mlHeroLaneThroughput15ev);

		pathThroughput *= bsdfSample;
		verify (!pathThroughput.IsNaN() && !pathThroughput.IsInf());

		// This is valid for irradiance AOV only if it is not a SPECULAR material and
		// first path vertex. Set or update sampleResult.irradiancePathThroughput
		if (sampleResult.firstPathVertex) {
			if (!(bsdf.GetEventTypes() & SPECULAR))
				sampleResult.irradiancePathThroughput = INV_PI * AbsDot(bsdf.hitPoint.shadeN, sampledDir) / rrProb;
			else
				sampleResult.irradiancePathThroughput = Spectrum();
		} else
			sampleResult.irradiancePathThroughput *= bsdfSample;

		eyeRay.Update(bsdf.GetRayOrigin(sampledDir), sampledDir);
	}

	// ML HERO split-radiance (packet reconstruction is disabled after dispersion):
	// final = before + legacy_after * HERO_RGB + CIE_after
	// Applied CIE endpoints are kept out of the legacy HERO color operation.
	if (mlHeroEnabled && mlDispersionSnapshotTaken) {
		const Spectrum mlDispersionColor = GetMLDispersionSampleColor();

		for (u_int i = 0; i < sampleResult.radiance.Size(); ++i) {
			const Spectrum cie15ev = (i < mlHeroContinuation15ev.cieRadiance.size()) ?
                mlHeroContinuation15ev.cieRadiance[i] : Spectrum();
            sampleResult.radiance[i] = MLHeroPathContinuationSplit15ev(
                mlRadianceBeforeDispersion[i], sampleResult.radiance[i], cie15ev, mlDispersionColor);
		}
	}

	MLHeroPathContinuationLog15ev("END", pathInfo.depth.depth, "EYE_PATH_COMPLETE",
        (BSDFEvent)0, 1.f, 0.f, Spectrum(), Spectrum());
	 mlHeroContinuation15ev.active = false;
// 15fg BEGIN
    if(mlHeroDiagnostics15fg && mlHeroEnabled) {
        mlHero15fgEyeResults.fetch_add(1ull,std::memory_order_relaxed);
        bool finite15fg=true;
        for(u_int group=0u;group<sampleResult.radiance.Size();++group)
            finite15fg=finite15fg && !sampleResult.radiance[group].IsNaN() && !sampleResult.radiance[group].IsInf();
        if(!finite15fg) mlHero15fgEyeNonfinite.fetch_add(1ull,std::memory_order_relaxed);
    }
// 15fg END


	sampleResult.rayCount += (float)(device->GetTotalRaysCount() - deviceRayCount);

	if (sampleResult.isHoldout) {
		sampleResult.radiance.Clear();
		sampleResult.albedo = Spectrum();
	}

	if (photonGICache && (photonGICache->GetDebugType() == PhotonGIDebugType::PGIC_DEBUG_SHOWINDIRECTPATHMIX) &&
			!photonGIShowIndirectPathMixUsed)
		sampleResult.radiance[0] = Spectrum(1.f, 0.f, 0.f);
}

//------------------------------------------------------------------------------
// RenderEyeSample
//------------------------------------------------------------------------------

void PathTracer::RenderEyeSample(
	IntersectionDevice *device,
	SceneConstRef scene, FilmConstRef film,
	Sampler& sampler,
	vector<SampleResult> &sampleResults
) const {
	ResetEyeSampleResults(sampleResults);
	SetMLHeroEnabled(mlHeroEnabled);
	mlHeroRoughGlassFresnelImportance = mlHeroRoughGlassFresnelImportanceConfig;
	const unsigned long long mlHeroPathSampleOrdinal15ev =
		mlHeroPathEyeSamples15ev.fetch_add(1ull, std::memory_order_relaxed) + 1ull;

	EyePathInfo pathInfo;
	Ray eyeRay;
	GenerateEyeRay(scene.GetCamera(), film, eyeRay, pathInfo.volume, sampler, sampleResults[0]);

	// ML HERO phase 15ev - CPU PATH parallel packet-throughput stage.
	//
	// Supported scalar and reflective bounces propagate a wavelength packet.
	// Valid direct-light/emission endpoints use CIE reconstruction; unsupported
	// transport retains the legacy fallback. Metal2 does not itself retire the
	// packet merely because its lane-0 Fresnel sets legacy spectral bookkeeping.
	SetMLHeroWavelengthCount(mlHeroEnabled ? (u_int)mlHeroWavelengthCount : 1u);

	if (mlHeroEnabled) {
		const u_int mlStrataCount = 16u;
		const float mlFilmSampleX = sampler.GetSample(0);
		const float mlFilmSampleY = sampler.GetSample(1);

		const u_int mlPixelX = Floor2UInt(mlFilmSampleX);
		const u_int mlPixelY = Floor2UInt(mlFilmSampleY);
		const float mlSubPixelX = mlFilmSampleX - mlPixelX;
		const float mlSubPixelY = mlFilmSampleY - mlPixelY;

		const u_int mlBaseStratum = Min(Floor2UInt(mlSubPixelX * mlStrataCount), mlStrataCount - 1u);
		const u_int mlPixelHash = (mlPixelX * 73856093u) ^ (mlPixelY * 19349663u);
		const u_int mlStratum = (mlBaseStratum + (mlPixelHash % mlStrataCount)) % mlStrataCount;
		const float mlSpectralU = (mlStratum + Clamp(mlSubPixelY, 0.f, 0.99999994f)) / mlStrataCount;

		// Quarter Phase Cycling, kept independent of packet lane count exactly as
		// in BIDIR. A thread-local four-sample group is sufficient here because
		// each PATH render thread owns its sampler sequence.
		static thread_local u_int mlQuarterStep15ev = 0u;
		static thread_local float mlQuarterGroupPhase15ev = 0.f;
		static thread_local bool mlQuarterGroupValid15ev = false;

		float mlPacketU0 = mlSpectralU;
		const bool mlQuarterActive15ev = mlHeroQuarterCycling &&
				(mlHeroSamplingMode == 1) && (mlHeroWavelengthCount >= 2);
		const u_int mlQuarter15ev = mlQuarterStep15ev & 3u;

		if (mlQuarterActive15ev) {
			const float laneCount = (float)mlHeroWavelengthCount;
			const float phaseRange = 1.f / laneCount;
			const u_int stratumIndex = Min((u_int)Floor2Int(mlSpectralU * laneCount), (u_int)mlHeroWavelengthCount - 1u);
			const float stratum = (float)stratumIndex * phaseRange;
			const float localPhase = mlSpectralU - stratum;

			if (!mlQuarterGroupValid15ev || (mlQuarter15ev == 0u)) {
				mlQuarterGroupPhase15ev = localPhase;
				mlQuarterGroupValid15ev = true;
			}

			float packetPhase = mlQuarterGroupPhase15ev + (float)mlQuarter15ev * phaseRange * .25f;
			while (packetPhase >= phaseRange)
				packetPhase -= phaseRange;
			mlPacketU0 = Clamp(stratum + packetPhase, 0.f, 0.99999994f);
		} else
			mlQuarterGroupValid15ev = false;

		for (u_int lane = 0; lane < (u_int)mlHeroWavelengthCount; ++lane) {
			float mlU = mlPacketU0 + (float)lane / (float)mlHeroWavelengthCount;
			mlU -= Floor2Int(mlU);
			mlU = Clamp(mlU, 0.f, 0.99999994f);

			float mlSampleWeight = 1.f;
			float mlWaveLength;
			if (mlHeroSamplingMode == 3)
				mlWaveLength = MLHeroSampling3Sample(mlU, &mlSampleWeight);
			else if (mlHeroSamplingMode == 2)
				mlWaveLength = MLHeroSampling2Sample(mlU, &mlSampleWeight);
			else
				mlWaveLength = Lerp(mlU, 380.f, 780.f);

			if (lane == 0u)
				SetMLDispersionWaveLength(mlWaveLength, mlSampleWeight);
			else
				SetMLHeroWaveLengthAt(lane, mlWaveLength, mlSampleWeight);
		}
		SetMLHeroActiveLane(0u);
		mlHeroPathPacketSamples15ev.fetch_add(1ull, std::memory_order_relaxed);
		if (mlQuarterActive15ev)
			mlHeroPathQuarterSamples15ev.fetch_add(1ull, std::memory_order_relaxed);
		MLHeroPathLogPacket15ev(mlHeroPathSampleOrdinal15ev, mlHeroSamplingMode,
			mlHeroWavelengthCount, mlQuarterActive15ev, mlQuarter15ev, mlSpectralU, mlPacketU0);
		++mlQuarterStep15ev;
	}

// 15fh BEGIN diagnostic-only
    MLHeroCycle15fh();
    MLHeroBegin15fh(sampleResults[0].filmX,sampleResults[0].filmY,film.GetWidth(),film.GetHeight());
// 15fh END
	RenderEyePath(device, scene, sampler, pathInfo, eyeRay, Spectrum(1.f), sampleResults);
// 15fh BEGIN diagnostic-only
    MLHeroFinish15fh(sampleResults[0]);
// 15fh END


}

//------------------------------------------------------------------------------
// RenderLightSample methods
//------------------------------------------------------------------------------

SampleResult &PathTracer::AddLightSampleResult(vector<SampleResult> &sampleResults,
		FilmConstRef film) {
	const u_int size = sampleResults.size();
	sampleResults.resize(size + 1);

	SampleResult &sampleResult = sampleResults[size];
	sampleResult.Init(&lightSampleResultsChannels, film.GetRadianceGroupCount());

	return sampleResult;
}

void PathTracer::ConnectToEye(IntersectionDevice *device,
		SceneConstRef scene,
		FilmConstRef film, const float time,
		const float u0, const float u1, const float u2,
		const LightSource &light, const BSDF &bsdf, 
		const Spectrum &flux, const LightPathInfo &pathInfo,
		vector<SampleResult> &sampleResults) const {
	// I don't connect camera invisible objects with the eye
	if (bsdf.IsCameraInvisible() || bsdf.IsDelta())
		return;

	// Test if the point-camera connection is valid
	float filmX, filmY;
	bool sampleSuccess;
	Ray eyeRay;
	Vector eyeDir;
	float eyeDistance = 0;
	Point lensPoint = pathInfo.lensPoint;
    if (scene.GetCamera().GetType() == Camera::ORTHOGRAPHIC){
		// Orthographic camera needs to be handled separately,
		// lensPoint can not be used as a target here
		Point p = bsdf.hitPoint.p;
		eyeDir = scene.GetCamera().GetDir();
		// calculate distance from vertex to camera plane
		const float D = -eyeDir.x*lensPoint.x - eyeDir.y*lensPoint.y - eyeDir.z*lensPoint.z;
		eyeDistance = eyeDir.x*p.x + eyeDir.y*p.y + eyeDir.z*p.z + D;
		eyeDistance = fabsf(eyeDistance);
		eyeRay = Ray(bsdf.hitPoint.p, eyeDir,
			0.f,
			eyeDistance,
			time);
		// Do not clamp the ray here because of the check inside ProjectToImage
		sampleSuccess = scene.GetCamera().ProjectToImage(&eyeRay, &filmX, &filmY);
	} else {
		eyeDir = Vector(bsdf.hitPoint.p - lensPoint);
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
	BSDFEvent event;
	const Spectrum bsdfEval = bsdf.Evaluate(-eyeDir, &event);

	if (bsdfEval.Black())
		return;

	// Trace a shadow ray
	scene.GetCamera().ClampRay(&eyeRay); // Clamp the ray here (see comment above)
	// I have to flip the direction of the traced ray because
	// the information inside PathVolumeInfo are about the path from
	// the light toward the camera (i.e. ray.o would be in the wrong
	// place).
	Ray traceRay(bsdf.GetRayOrigin(-eyeRay.d), -eyeRay.d,
			eyeDistance - eyeRay.maxt,
			eyeDistance - eyeRay.mint,
			time);
	traceRay.UpdateMinMaxWithEpsilon();

	RayHit traceRayHit;
	BSDF bsdfConn;
	Spectrum connectionThroughput;
	PathVolumeInfo volInfo = pathInfo.volume; // I need to use a copy here
	const bool shadowIntersection = scene.Intersect(device, LIGHT_RAY | CAMERA_RAY, &volInfo, u0, &traceRay, &traceRayHit, &bsdfConn,
			&connectionThroughput);

	if (shadowIntersection)
		return;

	float fluxToRadianceFactor;
	scene.GetCamera().GetPDF(eyeRay, eyeDistance, filmX, filmY, nullptr, &fluxToRadianceFactor);

	SampleResult &sampleResult = AddLightSampleResult(sampleResults, film);
	sampleResult.filmX = filmX;
	sampleResult.filmY = filmY;

	sampleResult.pixelX = Floor2UInt(filmX);
	sampleResult.pixelY = Floor2UInt(filmY);

#if !defined(NDEBUG)
	const u_int *subRegion = film.GetSubRegion();
#endif
	assert (sampleResult.pixelX >= subRegion[0]);
	assert (sampleResult.pixelX <= subRegion[1]);
	assert (sampleResult.pixelY >= subRegion[2]);
	assert (sampleResult.pixelY <= subRegion[3]);

	sampleResult.isCaustic = pathInfo.IsCausticPath(event, bsdf.GetGlossiness(), hybridBackForwardGlossinessThreshold);

	// Add radiance from the light source
	sampleResult.radiance[light.GetID()] = connectionThroughput * flux * fluxToRadianceFactor * bsdfEval;
}

//------------------------------------------------------------------------------
// RenderLightSample
//------------------------------------------------------------------------------

void PathTracer::RenderLightSample(IntersectionDevice *device,
		SceneConstRef scene, FilmConstRef film,
		Sampler& sampler, vector<SampleResult> &sampleResults,
		const ConnectToEyeCallBackType &ConnectToEyeCallBack) const {
	sampleResults.clear();
	SetMLHeroEnabled(mlHeroEnabled);
	mlHeroRoughGlassFresnelImportance = mlHeroRoughGlassFresnelImportanceConfig;

	// ML CPU PATH HERO for the optional CPU light-tracing path. Keep this
	// state independent from the eye path and reuse the first light BSDF-X
	// sample as wavelength source (same no-extra-dimension strategy).
	if (mlHeroEnabled) {
		const float mlLightU = sampler.GetSample(lightSampleBootSize + 4);
		if (mlHeroSamplingMode == 3) {
			float mlSampleWeight;
			const float mlWaveLength = MLHeroSampling3Sample(mlLightU, &mlSampleWeight);
			SetMLDispersionWaveLength(mlWaveLength, mlSampleWeight);
		} else if (mlHeroSamplingMode == 2) {
			float mlSampleWeight;
			const float mlWaveLength = MLHeroSampling2Sample(mlLightU, &mlSampleWeight);
			SetMLDispersionWaveLength(mlWaveLength, mlSampleWeight);
		} else
			SetMLDispersionWaveLength(Lerp(mlLightU, 380.f, 780.f));
	}

	Spectrum lightPathFlux;

	const float timeSample = sampler.GetSample(8);
	const float time = scene.GetCamera().GenerateRayTime(timeSample);

	// Select one light source
	float lightPickPdf;
	auto light = scene.GetLightSources().GetEmitLightStrategy().
			SampleLights(scene, sampler.GetSample(0), &lightPickPdf);

	if (light) {
		// Initialize the light path
		Ray nextEventRay;
		float lightEmitPdfW;
		lightPathFlux = light->Emit(scene,
				time, sampler.GetSample(1), sampler.GetSample(2),
				sampler.GetSample(3), sampler.GetSample(4), sampler.GetSample(5),
				nextEventRay, lightEmitPdfW);

		if (lightPathFlux.Black())
			return;

		lightPathFlux /= lightEmitPdfW * lightPickPdf;
		assert (!lightPathFlux.IsNaN() && !lightPathFlux.IsInf());

		LightPathInfo pathInfo;

		/*
		// Sample a point on the camera lens
		if (!scene.GetCamera().SampleLens(time, sampler.GetSample(6), sampler.GetSample(7),
				&pathInfo.lensPoint))
			return;
		*/

		//----------------------------------------------------------------------
		// Trace the light path
		//----------------------------------------------------------------------

		while (pathInfo.depth.depth < maxPathDepth.depth) {
			const u_int sampleOffset = lightSampleBootSize +  pathInfo.depth.depth * lightSampleStepSize;

			RayHit nextEventRayHit;
			BSDF bsdf;
			Spectrum connectionThroughput;
			const bool hit = scene.Intersect(device, LIGHT_RAY | INDIRECT_RAY, &pathInfo.volume, sampler.GetSample(sampleOffset),
					&nextEventRay, &nextEventRayHit, &bsdf,
					&connectionThroughput);
			if (!hit) {
				// Ray lost in space...
				break;
			}

			// Check if it is something with a not black shadow transparency
			// and stop if it has. Direct light sampling will take care of
			// this kind of paths.
			if (!bsdf.GetPassThroughShadowTransparency().Black() & !bsdf.GetPassThroughShadowTransparencyOverride())
				break;

			// Something was hit

			lightPathFlux *= connectionThroughput;

			//--------------------------------------------------------------
			// Try to connect the light path vertex with the eye
			//--------------------------------------------------------------

			scene.GetCamera().SampleLens(time, sampler.GetSample(6), sampler.GetSample(7),
				&pathInfo.lensPoint);

			if (ConnectToEyeCallBack){
				ConnectToEyeCallBack(pathInfo, bsdf, light->GetID(), lightPathFlux, sampleResults);
			} else {
				ConnectToEye(device, scene, film,
						nextEventRay.time,
						sampler.GetSample(sampleOffset + 1),
						sampler.GetSample(sampleOffset + 2),
						sampler.GetSample(sampleOffset + 3),
						*light, bsdf, lightPathFlux, pathInfo, sampleResults);
			}

			if (pathInfo.depth.depth == maxPathDepth.depth - 1)
				break;

			//--------------------------------------------------------------
			// Build the next vertex path ray
			//--------------------------------------------------------------

			float bsdfPdf = 0.f;
			Vector sampledDir;
			BSDFEvent bsdfEvent;
			float cosSampleDir;
			Spectrum bsdfSample = bsdf.Sample(&sampledDir,
						sampler.GetSample(sampleOffset + 4),
						sampler.GetSample(sampleOffset + 5),
					&bsdfPdf, &cosSampleDir, &bsdfEvent);
			if (bsdfSample.Black())
				break;


			pathInfo.AddVertex(bsdf, bsdfEvent, hybridBackForwardGlossinessThreshold);

			// If it isn't anymore a (nearly) specular path, I can stop
			if (hybridBackForwardEnable && !pathInfo.IsSpecularPath() &&
					// This condition is added to "stabilize" Metropolis sampler
					// used in light tracing part of hybrid rendering. In this case
					// I render also some not-caustic sample to make an "easy"
					// computation of the avg. image luminance.
					(pathInfo.depth.diffuseDepth + pathInfo.depth.glossyDepth > 1))
				break;

			// Russian Roulette
			if (pathInfo.UseRR(rrDepth)) {
				// Russian Roulette
				const float rrProb = RenderEngine::RussianRouletteProb(bsdfSample, rrImportanceCap);
				if (rrProb < sampler.GetSample(sampleOffset + 6))
					break;

				// Increase path contribution
				bsdfSample /= rrProb;
			}

			lightPathFlux *= bsdfSample;
			verify (!lightPathFlux.IsNaN() && !lightPathFlux.IsInf());

			nextEventRay.Update(bsdf.GetRayOrigin(sampledDir), sampledDir);
		}
	}

	// ML HERO harmonized baseline: apply wavelength-to-RGB exactly once
	// to all SampleResults generated by this single light path.
	if (mlHeroEnabled) {
		const Spectrum mlDispersionColor = GetMLDispersionSampleColor();
		for (u_int i = 0; i < sampleResults.size(); ++i) {
			SampleResult &mlLightSampleResult = sampleResults[i];
			for (u_int j = 0; j < mlLightSampleResult.radiance.Size(); ++j)
				mlLightSampleResult.radiance[j] *= mlDispersionColor;
		}
	}
}

//------------------------------------------------------------------------------
// RenderSample
//------------------------------------------------------------------------------

bool PathTracer::HasToRenderEyeSample(PathTracerThreadState &state) const {
	// Check if I have to trace an eye or light path
	if (hybridBackForwardEnable) {
		const double ratio = state.eyeSampleCount / state.lightSampleCount;
		if ((hybridBackForwardPartition == 1.f) ||
				(ratio < hybridBackForwardPartition)) {
			// Trace an eye path
			state.eyeSampleCount += 1.0;
			return true;
		} else {
			// Trace a light path
			state.lightSampleCount += 1.0;
			return false;
		}
	} else {
		state.eyeSampleCount += 1.0;
		return true;
	}
}

void PathTracer::ApplyVarianceClamp(const PathTracerThreadState &state,
		vector<SampleResult> &sampleResults) const {
	// Variance clamping
	if (state.varianceClamping->hasClamping()) {
		for(u_int i = 0; i < sampleResults.size(); ++i) {
			SampleResult &sampleResult = sampleResults[i];

			// I clamp only eye paths samples (variance clamping would cut
			// SDS path values due to high scale of PSR samples)
			if (sampleResult.HasChannel(Film::RADIANCE_PER_PIXEL_NORMALIZED))
				state.varianceClamping->Clamp(state.GetFilm(), sampleResult);
		}
	}
}

void PathTracer::RenderSample(PathTracerThreadState &state) const {

	auto Render = [&]() {

		// Check if I have to trace an eye or light path
		if (HasToRenderEyeSample(state)) {
			// Trace an eye path
			auto& sampler = state.GetEyeSampler();
			auto& sampleResults = state.GetEyeSampleResults();
			RenderEyeSample(
				state.device,
				state.scene,
				state.GetFilm(),
				sampler,
				sampleResults
			);
			return std::make_tuple(std::ref(sampler), std::ref(sampleResults));
		} else {
			// Otherwise trace a light path
			auto& sampler = state.GetLightSampler();
			auto& sampleResults = state.GetLightSampleResults();
			RenderLightSample(
				state.device,
				state.scene,
				state.GetFilm(),
				sampler,
				sampleResults
			);
			return std::make_tuple(std::ref(sampler), std::ref(sampleResults));
		}
	};  // lambda

	// Render sample
	auto [sampler, sampleResults] = Render();
	assert(&sampleResults == &state.GetEyeSampleResults() || &sampleResults == &state.GetLightSampleResults());

	// Apply variance clamping
	ApplyVarianceClamp(state, sampleResults);

	sampler.NextSample(sampleResults);
}

//------------------------------------------------------------------------------
// ParseOptions
//------------------------------------------------------------------------------

void PathTracer::ParseOptions(
	luxrays::PropertiesConstRef cfg,
	const luxrays::Properties &defaultProps
) {
	// Path depth settings
	maxPathDepth.depth = Max(0, cfg.Get(defaultProps.Get("path.pathdepth.total")).Get<int>());
	maxPathDepth.diffuseDepth = Max(0, cfg.Get(defaultProps.Get("path.pathdepth.diffuse")).Get<int>());
	maxPathDepth.glossyDepth = Max(0, cfg.Get(defaultProps.Get("path.pathdepth.glossy")).Get<int>());
	maxPathDepth.specularDepth = Max(0, cfg.Get(defaultProps.Get("path.pathdepth.specular")).Get<int>());

	// For compatibility with the past
	if (cfg.IsDefined("path.maxdepth") &&
			!cfg.IsDefined("path.pathdepth.total") &&
			!cfg.IsDefined("path.pathdepth.diffuse") &&
			!cfg.IsDefined("path.pathdepth.glossy") &&
			!cfg.IsDefined("path.pathdepth.specular")) {
		const u_int maxDepth = Max(0, cfg.Get("path.maxdepth").Get<int>());
		maxPathDepth.depth = maxDepth;
		maxPathDepth.diffuseDepth = maxDepth;
		maxPathDepth.glossyDepth = maxDepth;
		maxPathDepth.specularDepth = maxDepth;
	}

	// ML HERO global spectral mode
	mlHeroRoughGlassFresnelImportanceConfig = cfg.Get(defaultProps.Get("path.mlhero.roughglass.fresnelimportance")).Get<bool>();
	mlHeroEnabled = cfg.Get(defaultProps.Get("path.mlhero.enable")).Get<bool>();
	mlHeroSamplingMode = Clamp(cfg.Get(defaultProps.Get("path.mlhero.samplingmode")).Get<int>(), 1, 3);
	mlHeroWavelengthCount = Clamp(cfg.Get(defaultProps.Get("path.mlhero.wavelengthcount")).Get<int>(), 1, 8);
	mlHeroQuarterCycling = cfg.Get(defaultProps.Get("path.mlhero.quartercycling")).Get<bool>();
    mlHeroRRGlassEtaExperiment=cfg.Get(defaultProps.Get("path.mlhero.rr.glassetaexperiment")).Get<bool>();
    mlHeroRRMaterialExperiment=cfg.Get(defaultProps.Get("path.mlhero.rr.materialexperiment")).Get<bool>();
    mlHeroRRCPUPath15fz=cfg.Get(Property("renderengine.type")("PATHCPU")).Get<string>()=="PATHCPU";


// 15fg BEGIN
    mlHeroGenericReflectanceCompensation=cfg.Get(defaultProps.Get("path.mlhero.genericreflectancecompensation")).Get<bool>();
    mlHeroDiagnostics15fg=cfg.Get(defaultProps.Get("path.mlhero.diagnostics")).Get<bool>() &&
        cfg.Get(defaultProps.Get("path.mlhero.currentdiagnostics")).Get<bool>();
// 15fg END
	MLHeroPathResetDiagnostics15ev(mlHeroEnabled, mlHeroSamplingMode, mlHeroWavelengthCount, mlHeroQuarterCycling,
        mlHeroGenericReflectanceCompensation,mlHeroDiagnostics15fg,
        cfg.Get(defaultProps.Get("path.hybridbackforward.enable")).Get<bool>());

	// Russian Roulette settings
	rrDepth = (u_int)Max(1, cfg.Get(defaultProps.Get("path.russianroulette.depth")).Get<int>());
	rrImportanceCap = Clamp(cfg.Get(defaultProps.Get("path.russianroulette.cap")).Get<double>(), 0.0, 1.0);

	// Clamping settings
	// clamping.radiance.maxvalue is the old radiance clamping, now converted in variance clamping
	sqrtVarianceClampMaxValue = cfg.Get(Property("path.clamping.radiance.maxvalue")(0.0)).Get<double>();
	if (cfg.IsDefined("path.clamping.variance.maxvalue"))
		sqrtVarianceClampMaxValue = cfg.Get(defaultProps.Get("path.clamping.variance.maxvalue")).Get<double>();
	sqrtVarianceClampMaxValue = Max(0.f, sqrtVarianceClampMaxValue);

	forceBlackBackground = cfg.Get(defaultProps.Get("path.forceblackbackground.enable")).Get<bool>();
	
	hybridBackForwardEnable = cfg.Get(defaultProps.Get("path.hybridbackforward.enable")).Get<bool>();
	// hybridBackForwardGlossinessThreshold is used by LIGHTCPU when PSR is enabled
	// so I have always to set the value
	hybridBackForwardGlossinessThreshold = .05f;
	if (hybridBackForwardEnable) {
		hybridBackForwardPartition = Clamp(cfg.Get(defaultProps.Get("path.hybridbackforward.partition")).Get<double>(), 0.0, 1.0);
		hybridBackForwardGlossinessThreshold = Clamp(cfg.Get(defaultProps.Get("path.hybridbackforward.glossinessthreshold")).Get<double>(), 0.0, 1.0);
	}

	// Albedo AOV settings
	albedoSpecularSetting = String2AlbedoSpecularSetting(cfg.Get(defaultProps.Get("path.albedospecular.type")).Get<string>());
	albedoSpecularGlossinessThreshold = Max(cfg.Get(defaultProps.Get("path.albedospecular.glossinessthreshold")).Get<double>(), 0.0);

	// Update eye sample size
	eyeSampleBootSize = 5;
	eyeSampleStepSize = 9;
	eyeSampleSize = 
		eyeSampleBootSize + // To generate eye ray
		(maxPathDepth.depth + 1) * eyeSampleStepSize; // For each path vertex
	
	// Update light sample size
	lightSampleBootSize = 9;
	lightSampleStepSize = 7;
	lightSampleSize = 
		lightSampleBootSize + // To generate eye ray
		maxPathDepth.depth * lightSampleStepSize; // For each path vertex
}

//------------------------------------------------------------------------------
// Static methods used by RenderEngineRegistry
//------------------------------------------------------------------------------

PropertiesUPtr PathTracer::ToProperties(const Properties &cfg) {
	auto props_ptr = std::make_unique<Properties>();
	auto& props = *props_ptr;

	if (cfg.IsDefined("path.maxdepth") &&
			!cfg.IsDefined("path.pathdepth.total") &&
			!cfg.IsDefined("path.pathdepth.diffuse") &&
			!cfg.IsDefined("path.pathdepth.glossy") &&
			!cfg.IsDefined("path.pathdepth.specular")) {
		const u_int maxDepth = Max(0, cfg.Get("path.maxdepth").Get<int>());
		props << 
				Property("path.pathdepth.total")(maxDepth) <<
				Property("path.pathdepth.diffuse")(maxDepth) <<
				Property("path.pathdepth.glossy")(maxDepth) <<
				Property("path.pathdepth.specular")(maxDepth);
	} else {
		props <<
				cfg.Get(GetDefaultProps()->Get("path.pathdepth.total")) <<
				cfg.Get(GetDefaultProps()->Get("path.pathdepth.diffuse")) <<
				cfg.Get(GetDefaultProps()->Get("path.pathdepth.glossy")) <<
				cfg.Get(GetDefaultProps()->Get("path.pathdepth.specular"));
	}

	props <<
			cfg.Get(GetDefaultProps()->Get("path.mlhero.enable")) <<
			cfg.Get(GetDefaultProps()->Get("path.mlhero.roughglass.fresnelimportance")) <<
			cfg.Get(GetDefaultProps()->Get("path.mlhero.samplingmode")) <<
			cfg.Get(GetDefaultProps()->Get("path.mlhero.wavelengthcount")) <<
			cfg.Get(GetDefaultProps()->Get("path.mlhero.quartercycling")) <<
            cfg.Get(GetDefaultProps()->Get("path.mlhero.rr.glassetaexperiment")) <<
            cfg.Get(GetDefaultProps()->Get("path.mlhero.rr.materialexperiment")) <<
// 15fg BEGIN
            cfg.Get(GetDefaultProps()->Get("path.mlhero.genericreflectancecompensation")) <<
            cfg.Get(GetDefaultProps()->Get("path.mlhero.diagnostics")) <<
            cfg.Get(GetDefaultProps()->Get("path.mlhero.currentdiagnostics")) <<
// 15fg END

			cfg.Get(GetDefaultProps()->Get("path.hybridbackforward.enable")) <<
			cfg.Get(GetDefaultProps()->Get("path.hybridbackforward.partition")) <<
			cfg.Get(GetDefaultProps()->Get("path.hybridbackforward.glossinessthreshold")) <<
			cfg.Get(GetDefaultProps()->Get("path.russianroulette.depth")) <<
			cfg.Get(GetDefaultProps()->Get("path.russianroulette.cap")) <<
			cfg.Get(GetDefaultProps()->Get("path.clamping.variance.maxvalue")) <<
			cfg.Get(GetDefaultProps()->Get("path.forceblackbackground.enable")) <<
			cfg.Get(GetDefaultProps()->Get("path.albedospecular.type")) <<
			cfg.Get(GetDefaultProps()->Get("path.albedospecular.glossinessthreshold")) <<
			*Sampler::ToProperties(cfg);

	return props_ptr;
}

PropertiesUPtr PathTracer::GetDefaultProps() {
	auto props = std::make_unique<Properties>();
	*props <<
			Property("path.mlhero.enable")(false) <<
		Property("path.mlhero.roughglass.fresnelimportance")(false) <<
			Property("path.mlhero.samplingmode")(1) <<
			Property("path.mlhero.wavelengthcount")(1) <<
			Property("path.mlhero.quartercycling")(true) <<
            Property("path.mlhero.rr.glassetaexperiment")(false) <<
            Property("path.mlhero.rr.materialexperiment")(false) <<
// 15fg BEGIN
            Property("path.mlhero.genericreflectancecompensation")(true) <<
            Property("path.mlhero.diagnostics")(false) <<
            Property("path.mlhero.currentdiagnostics")(true) <<
// 15fg END

			Property("path.hybridbackforward.enable")(false) <<
			Property("path.hybridbackforward.partition")(0.8) <<
			Property("path.hybridbackforward.glossinessthreshold")(.05f) <<
			Property("path.pathdepth.total")(6) <<
			Property("path.pathdepth.diffuse")(4) <<
			Property("path.pathdepth.glossy")(4) <<
			Property("path.pathdepth.specular")(6) <<
			Property("path.russianroulette.depth")(3) <<
			Property("path.russianroulette.cap")(.5f) <<
			Property("path.clamping.variance.maxvalue")(0.f) <<
			Property("path.forceblackbackground.enable")(false) <<
			Property("path.albedospecular.type")("REFLECT_TRANSMIT") <<
			Property("path.albedospecular.glossinessthreshold")(.05f);

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
