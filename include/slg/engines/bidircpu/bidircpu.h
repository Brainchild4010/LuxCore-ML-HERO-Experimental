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

#ifndef _SLG_BIDIRCPU_H
#define	_SLG_BIDIRCPU_H

#include "luxrays/utils/thread.h"
#include "slg/slg.h"
#include "slg/engines/cpurenderengine.h"
#include "slg/engines/caches/photongi/photongicache.h"
#include "slg/samplers/sampler.h"
#include "slg/film/film.h"
#include "slg/film/filmsamplesplatter.h"
#include "slg/film/sampleresult.h"
#include "slg/bsdf/bsdf.h"
#include "slg/volumes/volume.h"

namespace slg {

//------------------------------------------------------------------------------
// Bidirectional path tracing CPU render engine
//------------------------------------------------------------------------------

typedef struct {
	BSDF bsdf;
	BSDFEvent bsdfEvent;
	luxrays::Spectrum throughput;

	// ML-HERO phase 6: diagnostic per-wavelength path throughput. These lanes
	// are propagated alongside the classic RGB throughput but are not yet used
	// by radiance accumulation or MIS.
	u_int mlHeroLaneCount;
	luxrays::Spectrum mlHeroLaneThroughput[8];

	// ML HERO phase 15l: wavelength packet metadata belongs to the vertex.
	// This makes a stored eye/light vertex self-contained and prevents later
	// ConnectVertices evaluation from depending on mutable thread-local state.
	float mlHeroLaneWaveLength[8];
	float mlHeroLaneSampleWeight[8];

	// ML HERO phase 15bh: explicit wavelength-PDF state for future
	// TerminateSecondary-style dispersive paths. Diagnostic-only in 15bh.
	float mlHeroLaneWaveLengthPdf[8];
	bool mlHeroSecondaryWavelengthsTerminated;
	u_int mlHeroActiveWavelengthCount;

	// ML HERO phase 15bp: surviving-HERO scalar / packet-average ratio captured
	// at the first HERO-only dispersive termination. Diagnostic only.
	float mlHeroTerminationBrightnessRatio;

	// ML HERO phase 15bt: first-termination sampling/reconstruction metadata.
	// Diagnostic only; these values are never consumed by the renderer estimator.
	float mlHeroTerminationHeroLambda;
	float mlHeroTerminationPacketAverage;
	float mlHeroTerminationHeroScalar;
	float mlHeroTerminationLaneSelectionPdf;
	float mlHeroTerminationWavelengthPdf;
	float mlHeroTerminationCurrentScale;
	double mlHero15buReferenceThroughput;

	// ML HERO phase 15bn: diagnostic-only spectral path-probability state.
	// event product ignores perfect-specular directional support; support product
	// zeros a lane when the sampled HERO direction is impossible at that wavelength.
	double mlHeroLaneGlassEventPdfProduct[8];
	double mlHeroLaneDiracSupportPdfProduct[8];

	// ML-HERO phase 7 diagnostic: mark one path so the same lane state can
	// be observed over several consecutive BIDIR bounces.
	bool mlHeroDebugTracked;
	u_int mlHeroDebugLoggedBounces;

	u_int lightID, depth;

	// Check Iliyan Georgiev's latest technical report for the details of how
	// MIS weight computation works (http://www.iliyan.com/publications/ImplementingVCM)
	float dVCM; // MIS quantity used for vertex connection and merging
	float dVC;  // MIS quantity used for vertex connection
	float dVM;  // MIS quantity used for vertex merging

	// Volume rendering information
	PathVolumeInfo volInfo;
} PathVertexVM;

class BiDirCPURenderEngine;

class BiDirCPURenderThread : public CPUNoTileRenderThread {
public:
	BiDirCPURenderThread(BiDirCPURenderEngine *engine, const u_int index,
			luxrays::IntersectionDevice *device);

	friend class BiDirCPURenderEngine;

protected:
	// Used to offset Sampler data
	static const u_int sampleBootSize = 13;
	static const u_int sampleBootSizeVM = 12; // I'm using the same time for all rays in a single pass (so I need one less random variable)
	static const u_int sampleLightStepSize = 5;
	static const u_int sampleEyeStepSize = 10;

	static float MIS(const float a) {
		//return a; // Balance heuristic
		return a * a; // Power heuristic
	}

	virtual luxrays::JThreadUPtr AllocRenderThread() {
		auto t = std::make_unique<luxrays::JThread>(
			std::bind_front(&BiDirCPURenderThread::RenderFunc, this)
		);
		luxrays::SetThreadName(t, "LxBiDirCPU");
		return std::move(t);
	}

	void AOVWarmUp(std::stop_token stop_token, const luxrays::RandomGeneratorUPtr & rndGen);

	SampleResult &AddResult(std::vector<SampleResult> &sampleResults, const bool fromLight) const;
	void RenderFunc(std::stop_token stop_token);

	void DirectLightSampling(const float time,
		const float u0, const float u1, const float u2,
		const float u3, const float u4,
		const PathVertexVM &eyeVertex, SampleResult &eyeSampleResult) const;
	void DirectHitLight(const bool finiteLightSource, const PathVertexVM &eyeVertex,
		SampleResult &eyeSampleResult) const;
	void DirectHitLight(LightSourceConstRef light, const luxrays::Spectrum &lightRadiance,
		const float directPdfA, const float emissionPdfW,
		const PathVertexVM &eyeVertex, luxrays::Spectrum *radiance) const;

	void ConnectVertices(const float time,
		const PathVertexVM &eyeVertex, const PathVertexVM &BiDirVertex,
		SampleResult &eyeSampleResult, const float u0) const;
	void ConnectToEye(const float time,
		const PathVertexVM &BiDirVertex, const float u0,
		const luxrays::Point &lensPoint, std::vector<SampleResult> &sampleResults) const;

	bool TraceLightPath(const float time,
		const SamplerUPtr& sampler,
		CameraConstRef camera,
		std::vector<PathVertexVM> &lightPathVertices,
		std::vector<SampleResult> &sampleResults) const;
	bool Bounce(const float time, const SamplerUPtr& sampler, const u_int sampleOffset,
		PathVertexVM *pathVertex, luxrays::Ray *nextEventRay) const;

	float misVmWeightFactor; // Weight of vertex merging (used in VC)
    float misVcWeightFactor; // Weight of vertex connection (used in VM)
	float vmNormalization; // 1 / (Pi * radius^2 * light_path_count)

	static const Film::FilmChannels eyeSampleResultsChannels;
	static const Film::FilmChannels lightSampleResultsChannels;
};

class SobolSamplerSharedData;

class BiDirCPURenderEngine : public CPUNoTileRenderEngine {
public:
	BiDirCPURenderEngine(RenderConfigRef cfg);
	virtual ~BiDirCPURenderEngine();

	virtual RenderEngineType GetType() const { return GetObjectType(); }
	virtual std::string GetTag() const { return GetObjectTag(); }

	virtual RenderStateSPtr GetRenderState();

	//--------------------------------------------------------------------------
	// Static methods used by RenderEngineRegistry
	//--------------------------------------------------------------------------

	static RenderEngineType GetObjectType() { return BIDIRCPU; }
	static std::string GetObjectTag() { return "BIDIRCPU"; }
	static luxrays::PropertiesUPtr ToProperties(const luxrays::Properties &cfg);
	static RenderEngine *FromProperties(RenderConfigRef rcfg);

	// Signed because of the delta parameter
	u_int maxEyePathDepth, maxLightPathDepth;

	// Used for vertex merging, it enables VM if it is > 0
	u_int lightPathsCount;
	float baseRadius; // VM (i.e. SPPM) start radius parameter
	float radiusAlpha; // VM (i.e. SPPM) alpha parameter

	u_int rrDepth;
	float rrImportanceCap;

	// ML HERO global spectral mode
	bool mlHeroEnabled;
	
	
	// ML HERO wavelength sampling: 1=Linear, 2=CIE Weighted, 3=Sensor Weighted
	int mlHeroSamplingMode;
	// Number of HERO wavelengths evaluated per path (reserved for multi-wavelength mode)
	int mlHeroWavelengthCount;
	// Dispersive Glass transport: 0=current HERO, 1=HERO-only, 2=experimental lane-PDF
	int mlHeroGlassMode;
	// Runtime replacement for the former ML_HERO_GLASS_PER_LANE_WEIGHT compile switch
	bool mlHeroGlassPerLaneWeight;
	// Runtime replacement for the former ML_HERO_QUARTER_CYCLING compile switch
	bool mlHeroQuarterCycling;
	// Experimental BIDIR A/B compensation for doubly HERO-only ConnectVertices.
	bool mlHeroDualTerminationCompensation;
	// Experimental Matte spectral basis compensation A/B switch (HERO_76 / phase15cj).
	bool mlHeroMatteBasisCompensation;
	// Generic non-specular chromatic HERO reflectance basis compensation (HERO_81 / phase15co).
	bool mlHeroGenericReflectanceCompensation;
	// Glossy2 Kd-only spectral basis compensation, consolidated in HERO_83 / phase15cq.
	bool mlHeroGlossy2BasisCompensation;
	// ML HERO diagnostics runtime controls. Master OFF removes diagnostic logging/counters from normal renders.
	bool mlHeroDiagnostics;
	bool mlHeroCurrentDiagnostics;
	bool mlHeroLegacyDiagnostics;
	bool mlHeroHeavyDiagnostics;
// Clamping settings
	float sqrtVarianceClampMaxValue;

	// Albedo AOV settings
	AlbedoSpecularSetting albedoSpecularSetting;
	float albedoSpecularGlossinessThreshold;

	bool forceBlackBackground;

	friend class BiDirCPURenderThread;

protected:
	static luxrays::PropertiesUPtr GetDefaultProps();

	virtual void InitFilm();
	virtual void StartLockLess();
	virtual void StopLockLess();

	PhotonGICache *photonGICache;

	u_int aovWarmupSPP;
	// We'll use a shared_ptr for shared data since, by design, the
	// ownership is shared among several objects...
	SobolSamplerSharedDataSPtr aovWarmupSamplerSharedData;

private:
	CPURenderThreadUPtr NewRenderThread(const u_int index, luxrays::IntersectionDevice *device) {
		return std::make_unique<BiDirCPURenderThread>(this, index, device);
	}
};

}

#endif	/* _SLG_BIDIRCPU_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
