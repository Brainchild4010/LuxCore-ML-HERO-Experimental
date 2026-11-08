/***************************************************************************
 * Copyright 1998-2026 by authors (see AUTHORS.txt)                        *
 *                                                                         *
 *   This file is part of LuxCoreRender.                                   *
 *                                                                         *
 * Licensed under the Apache License, Version 2.0 (the "License");         *
 ***************************************************************************/

#ifndef _SLG_ZSOBOL_SAMPLER_H
#define _SLG_ZSOBOL_SAMPLER_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <tuple>

#include "luxrays/core/randomgen.h"
#include "luxrays/usings.h"
#include "luxrays/utils/atomic.h"

#include "slg/film/film.h"
#include "slg/samplers/sampler.h"
#include "slg/samplers/zsobolindex.h"
#include "slg/samplers/zsobolsequence.h"

namespace slg {

class ZSobolSamplerSharedData : public SamplerSharedData {
public:
	ZSobolSamplerSharedData(const luxrays::RandomGeneratorUPtr &rndGen, FilmPtr engineFilm);
	ZSobolSamplerSharedData(const u_int seed, FilmPtr engineFilm);
	virtual ~ZSobolSamplerSharedData();

	virtual void Reset();
	std::tuple<u_int, u_int> GetNewBucket(const u_int bucketCount);
	u_int GetNewPixelPass(const u_int pixelIndex = 0u);
	u_int GetPassCount(const u_int bucketCount) const;
	void MergeAdaptiveStats(const std::uint64_t candidates,
		const std::uint64_t acceptedByThreshold,
		const std::uint64_t skipped,
		const std::uint64_t forcedAccepts,
		const std::uint64_t invalidNoise);

	static std::unique_ptr<SamplerSharedData> FromProperties(
		const luxrays::Properties &cfg,
		const luxrays::RandomGeneratorUPtr &rndGen,
		FilmPtr film);

	FilmRef GetEngineFilm() { return *engineFilm; }
	bool HasEngineFilm() const { return bool(engineFilm); }
	u_int GetSeedBase() const { return *seedBase; }

private:
	std::shared_ptr<u_int> seedBase;
	std::shared_ptr<u_int> bucketIndex;
	FilmPtr engineFilm;
	std::vector<u_int> passPerPixel;
	std::atomic<std::uint64_t> adaptiveCandidates{0u};
	std::atomic<std::uint64_t> adaptiveAcceptedByThreshold{0u};
	std::atomic<std::uint64_t> adaptiveSkipped{0u};
	std::atomic<std::uint64_t> adaptiveForcedAccepts{0u};
	std::atomic<std::uint64_t> adaptiveInvalidNoise{0u};
};

class ZSobolSampler : public Sampler {
public:
	ZSobolSampler(const luxrays::RandomGeneratorUPtr &rnd, FilmPtr film,
		const FilmSampleSplatterUPtr &filmSplatter, const bool imageSamplesEnable,
		const float adaptiveStrength, const float adaptiveUserImportanceWeight,
		const u_int bucketSize, const u_int tileSize, const u_int superSampling,
		const u_int overlapping, const u_int samplesPerPixel,
		SamplerSharedDataSPtr samplerSharedData);
	virtual ~ZSobolSampler();

	virtual SamplerType GetType() const { return GetObjectType(); }
	virtual std::string GetTag() const { return GetObjectTag(); }
	virtual void RequestSamples(const SampleType sampleType, const u_int size);
	virtual float GetSample(const u_int index);
	virtual void NextSample(const std::vector<SampleResult> &sampleResults);
	virtual luxrays::PropertiesUPtr ToProperties() const;

	u_int GetPassCount() const;

	static SamplerType GetObjectType() { return ZSOBOL; }
	static std::string GetObjectTag() { return "ZSOBOL"; }
	static luxrays::PropertiesUPtr ToProperties(const luxrays::Properties &cfg);
	static SamplerUPtr FromProperties(const luxrays::Properties &cfg,
		const luxrays::RandomGeneratorUPtr &rndGen, FilmPtr film,
		const FilmSampleSplatterUPtr &filmSplatter,
		SamplerSharedDataSPtr sharedData);
	static slg::ocl::Sampler *FromPropertiesOCL(const luxrays::Properties &cfg);
	static void AddRequiredChannels(Film::FilmChannels &channels,
		const luxrays::Properties &cfg);

private:
	void InitNewSample();
	float GetZSobolSample(const u_int dimension) const;
	u_int GetScrambleSeed(const u_int dimension) const;

	static luxrays::PropertiesUPtr GetDefaultProps();
	static u_int ResolveSamplesPerPixel(const luxrays::Properties &cfg);

	std::shared_ptr<ZSobolSamplerSharedData> sharedData;
	ZSobolSequence zsobolSequence;
	std::unique_ptr<ZSobolIndex> zsobolIndex;
	float adaptiveStrength, adaptiveUserImportanceWeight;
	u_int bucketSize, tileSize, superSampling, overlapping;
	u_int samplesPerPixel;

	std::shared_ptr<u_int> bucketIndex;
	u_int pixelOffset, passOffset, pass;
	u_int pixelX, pixelY;
	float sample0, sample1;
	std::uint64_t adaptiveCandidates = 0u;
	std::uint64_t adaptiveAcceptedByThreshold = 0u;
	std::uint64_t adaptiveSkipped = 0u;
	std::uint64_t adaptiveForcedAccepts = 0u;
	std::uint64_t adaptiveInvalidNoise = 0u;
};

}

#endif // _SLG_ZSOBOL_SAMPLER_H

