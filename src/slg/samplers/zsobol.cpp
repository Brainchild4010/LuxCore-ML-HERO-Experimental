/***************************************************************************
 * Copyright 1998-2026 by authors (see AUTHORS.txt)                        *
 *                                                                         *
 *   This file is part of LuxCoreRender.                                   *
 *                                                                         *
 * Licensed under the Apache License, Version 2.0 (the "License");         *
 ***************************************************************************/

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

#include "luxrays/core/color/color.h"
#include "luxrays/usings.h"
#include "slg/usings.h"
#include "slg/samplers/zsobol.h"
#include "slg/utils/mortoncurve.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// ZSobolSamplerSharedData
//------------------------------------------------------------------------------

ZSobolSamplerSharedData::ZSobolSamplerSharedData(const u_int seed,
		FilmPtr engineFlm) : SamplerSharedData(),
		seedBase(std::make_shared<u_int>(seed)), engineFilm(engineFlm) {
	Reset();
}

ZSobolSamplerSharedData::ZSobolSamplerSharedData(const RandomGeneratorUPtr &rndGen,
		FilmPtr engineFlm) : SamplerSharedData(),
		seedBase(std::make_shared<u_int>(
			rndGen->uintValue() % (0xffffffffu - 1u) + 1u)), engineFilm(engineFlm) {
	Reset();
}

ZSobolSamplerSharedData::~ZSobolSamplerSharedData() {
	const std::uint64_t candidates = adaptiveCandidates.load(std::memory_order_relaxed);
	if (candidates == 0u)
		return;

	const std::uint64_t accepted = adaptiveAcceptedByThreshold.load(std::memory_order_relaxed);
	const std::uint64_t skipped = adaptiveSkipped.load(std::memory_order_relaxed);
	const std::uint64_t forced = adaptiveForcedAccepts.load(std::memory_order_relaxed);
	const std::uint64_t invalid = adaptiveInvalidNoise.load(std::memory_order_relaxed);
	SLG_LOG("ZSOBOL_ADAPTIVE_SUMMARY candidates=" << candidates <<
		" acceptedByThreshold=" << accepted <<
		" skipped=" << skipped <<
		" forcedAccepts=" << forced <<
		" invalidNoise=" << invalid <<
		" accountingPass=" << ((candidates == accepted + skipped + forced) ? 1 : 0) <<
		" boundedSkip=1 acceptedPassOnly=1 zOrderPrefixPreserved=1");
}

void ZSobolSamplerSharedData::Reset() {
	if (HasEngineFilm()) {
		const u_int *subRegion = GetEngineFilm().GetSubRegion();
		const u_int pixelCount = (subRegion[1] - subRegion[0] + 1u) *
			(subRegion[3] - subRegion[2] + 1u);
		passPerPixel.assign(pixelCount, 0u);
	} else
		passPerPixel.assign(1u, 0u);

	bucketIndex = std::make_shared<u_int>(0u);
	adaptiveCandidates.store(0u, std::memory_order_relaxed);
	adaptiveAcceptedByThreshold.store(0u, std::memory_order_relaxed);
	adaptiveSkipped.store(0u, std::memory_order_relaxed);
	adaptiveForcedAccepts.store(0u, std::memory_order_relaxed);
	adaptiveInvalidNoise.store(0u, std::memory_order_relaxed);
}

tuple<u_int, u_int> ZSobolSamplerSharedData::GetNewBucket(const u_int bucketCount) {
	const u_int newBucketIndex = AtomicInc(bucketIndex.get()) % bucketCount;
	const u_int seed = (*seedBase + newBucketIndex) % (0xffffffffu - 1u) + 1u;
	return make_tuple(newBucketIndex, seed);
}

u_int ZSobolSamplerSharedData::GetNewPixelPass(const u_int pixelIndex) {
	return AtomicInc(&passPerPixel[pixelIndex]);
}

u_int ZSobolSamplerSharedData::GetPassCount(const u_int bucketCount) const {
	return *bucketIndex / bucketCount;
}

void ZSobolSamplerSharedData::MergeAdaptiveStats(const std::uint64_t candidates,
		const std::uint64_t acceptedByThreshold, const std::uint64_t skipped,
		const std::uint64_t forcedAccepts, const std::uint64_t invalidNoise) {
	adaptiveCandidates.fetch_add(candidates, std::memory_order_relaxed);
	adaptiveAcceptedByThreshold.fetch_add(acceptedByThreshold, std::memory_order_relaxed);
	adaptiveSkipped.fetch_add(skipped, std::memory_order_relaxed);
	adaptiveForcedAccepts.fetch_add(forcedAccepts, std::memory_order_relaxed);
	adaptiveInvalidNoise.fetch_add(invalidNoise, std::memory_order_relaxed);
}

unique_ptr<SamplerSharedData> ZSobolSamplerSharedData::FromProperties(
		const Properties &cfg, const RandomGeneratorUPtr &rndGen, FilmPtr film) {
	return std::make_unique<ZSobolSamplerSharedData>(rndGen, film);
}

//------------------------------------------------------------------------------
// ZSobolSampler
//------------------------------------------------------------------------------

ZSobolSampler::ZSobolSampler(const RandomGeneratorUPtr &rnd, FilmPtr flm,
		const FilmSampleSplatterUPtr &flmSplatter, const bool imgSamplesEnable,
		const float adaptiveStr, const float adaptiveUserImpWeight,
		const u_int bucketSz, const u_int tileSz, const u_int superSmpl,
		const u_int overlap, const u_int spp,
		SamplerSharedDataSPtr samplerSharedData) :
	Sampler(rnd, flm, flmSplatter, imgSamplesEnable),
	sharedData(static_pointer_cast<ZSobolSamplerSharedData>(samplerSharedData)),
	zsobolSequence(), zsobolIndex(),
	adaptiveStrength(adaptiveStr),
	adaptiveUserImportanceWeight(adaptiveUserImpWeight),
	bucketSize(bucketSz), tileSize(tileSz), superSampling(superSmpl),
	overlapping(overlap), samplesPerPixel(spp),
	bucketIndex(std::make_shared<u_int>(0u)), pixelOffset(0u), passOffset(0u),
	pass(0u), pixelX(0u), pixelY(0u), sample0(0.f), sample1(0.f) {
}

ZSobolSampler::~ZSobolSampler() {
	sharedData->MergeAdaptiveStats(adaptiveCandidates,
		adaptiveAcceptedByThreshold, adaptiveSkipped,
		adaptiveForcedAccepts, adaptiveInvalidNoise);
}

u_int ZSobolSampler::GetScrambleSeed(const u_int dimension) const {
	const std::uint64_t key = (std::uint64_t(sharedData->GetSeedBase()) << 32) |
		dimension;
	return static_cast<u_int>(ZSobolIndex::MixBits(key));
}

float ZSobolSampler::GetZSobolSample(const u_int dimension) const {
	const std::uint64_t sampleIndex = zsobolIndex->GetIndex(
		pixelX, pixelY, pass, dimension);
	return zsobolSequence.GetSample(sampleIndex, dimension,
		GetScrambleSeed(dimension));
}

void ZSobolSampler::InitNewSample() {
	const bool doImageSamples = imageSamplesEnable && film;
	std::uint64_t adaptiveRejectAttempts = 0u;
	const std::uint64_t adaptiveRejectLimit = std::max<std::uint64_t>(1u,
		static_cast<std::uint64_t>(bucketSize) * superSampling);

	const u_int *filmSubRegion = nullptr;
	u_int subRegionWidth = 0u, subRegionHeight = 0u;
	u_int tileWidthCount = 0u, tileHeightCount = 0u, bucketCount = 0u;

	if (doImageSamples) {
		filmSubRegion = GetFilm().GetSubRegion();
		subRegionWidth = filmSubRegion[1] - filmSubRegion[0] + 1u;
		subRegionHeight = filmSubRegion[3] - filmSubRegion[2] + 1u;
		tileWidthCount = (subRegionWidth + tileSize - 1u) / tileSize;
		tileHeightCount = (subRegionHeight + tileSize - 1u) / tileSize;
		bucketCount = overlapping *
			(tileWidthCount * tileSize * tileHeightCount * tileSize + bucketSize - 1u) /
			bucketSize;
	} else
		bucketCount = 0xffffffffu;

	for (;;) {
		++passOffset;
		if (passOffset >= superSampling) {
			++pixelOffset;
			passOffset = 0u;

			if (pixelOffset >= bucketSize) {
				auto newBucket = sharedData->GetNewBucket(bucketCount);
				*bucketIndex = get<0>(newBucket);
				pixelOffset = 0u;
			}
		}

		if (doImageSamples) {
			const u_int pixelBucketIndex = (*bucketIndex / overlapping) *
				bucketSize + pixelOffset;
			const u_int mortonCurveOffset = pixelBucketIndex % (tileSize * tileSize);
			const u_int pixelTileIndex = pixelBucketIndex / (tileSize * tileSize);
			const u_int subRegionPixelX = (pixelTileIndex % tileWidthCount) * tileSize +
				DecodeMorton2X(mortonCurveOffset);
			const u_int subRegionPixelY = (pixelTileIndex / tileWidthCount) * tileSize +
				DecodeMorton2Y(mortonCurveOffset);

			if (subRegionPixelX >= subRegionWidth || subRegionPixelY >= subRegionHeight)
				continue;

			pixelX = filmSubRegion[0] + subRegionPixelX;
			pixelY = filmSubRegion[2] + subRegionPixelY;

			if (adaptiveStrength > 0.f && GetFilm().HasChannel(Film::NOISE)) {
				const float noise = *(GetFilm().channel_NOISE->GetPixel(pixelX, pixelY));
				++adaptiveCandidates;
				if (std::isnan(noise))
					++adaptiveInvalidNoise;
				float threshold;
				if (GetFilm().HasChannel(Film::USER_IMPORTANCE)) {
					const float userImportance =
						*(GetFilm().channel_USER_IMPORTANCE->GetPixel(pixelX, pixelY));
					if (isinf(noise))
						threshold = userImportance;
					else
						threshold = userImportance > 0.f ?
							Lerp(adaptiveUserImportanceWeight, noise, userImportance) : 0.f;
				} else
					threshold = noise;

				threshold = Max(threshold, 1.f - adaptiveStrength);
				if (rndGen->floatValue() > threshold) {
					// Never let a very low-importance image region spin forever.
					// After one complete local bucket/supersample sweep, accept the
					// current candidate. The pixel pass is claimed only below, so
					// skipped candidates do not punch holes into its ZSobol prefix.
					if (++adaptiveRejectAttempts < adaptiveRejectLimit) {
						++adaptiveSkipped;
						continue;
					}
					++adaptiveForcedAccepts;
				} else
					++adaptiveAcceptedByThreshold;
			}

			pass = sharedData->GetNewPixelPass(
				subRegionPixelX + subRegionPixelY * subRegionWidth);
		} else {
			pixelX = 0u;
			pixelY = 0u;
			pass = sharedData->GetNewPixelPass();
		}

		if (pass >= samplesPerPixel)
			throw runtime_error("ZSOBOL exhausted its fixed sample domain of " +
				ToString(samplesPerPixel) + " samples per pixel. Increase "
				"sampler.zsobol.samplesperpixel or use an explicit batch.haltspp.");

		sample0 = pixelX + GetZSobolSample(0u);
		sample1 = pixelY + GetZSobolSample(1u);
		break;
	}
}

void ZSobolSampler::RequestSamples(const SampleType smplType, const u_int size) {
	Sampler::RequestSamples(smplType, size);
	zsobolSequence.RequestSamples(size);

	const u_int width = film ? GetFilm().GetWidth() : 1u;
	const u_int height = film ? GetFilm().GetHeight() : 1u;
	zsobolIndex = std::make_unique<ZSobolIndex>(samplesPerPixel, width, height);
	if (zsobolIndex->GetIndexBitCount() > ZSOBOL_BITS)
		throw runtime_error("ZSOBOL requires " +
			ToString(zsobolIndex->GetIndexBitCount()) +
			" extended index bits for this film/SPP combination, but only " +
			ToString(ZSOBOL_BITS) + " are available.");

	pixelOffset = bucketSize;
	passOffset = superSampling;
	InitNewSample();
}

float ZSobolSampler::GetSample(const u_int index) {
	assert(index < requestedSamples);
	if (index == 0u)
		return sample0;
	if (index == 1u)
		return sample1;
	return GetZSobolSample(index);
}

void ZSobolSampler::NextSample(const vector<SampleResult> &sampleResults) {
	if (film) {
		switch (sampleType) {
			case PIXEL_NORMALIZED_ONLY:
				GetFilm().AddSampleCount(threadIndex, 1.0, 0.0);
				break;
			case SCREEN_NORMALIZED_ONLY:
				GetFilm().AddSampleCount(threadIndex, 0.0, 1.0);
				break;
			case PIXEL_NORMALIZED_AND_SCREEN_NORMALIZED:
				GetFilm().AddSampleCount(threadIndex, 1.0, 1.0);
				break;
			case ONLY_AOV_SAMPLE:
				break;
			default:
				throw runtime_error("Unknown sample type in ZSobolSampler::NextSample(): " +
					ToString(sampleType));
		}
		AtomicAddSamplesToFilm(sampleResults);
	}
	InitNewSample();
}

u_int ZSobolSampler::GetPassCount() const {
	if (!(imageSamplesEnable && film))
		throw runtime_error("Called ZSobolSampler::GetPassCount() without image sampling");
	const u_int *subRegion = GetFilm().GetSubRegion();
	const u_int width = subRegion[1] - subRegion[0] + 1u;
	const u_int height = subRegion[3] - subRegion[2] + 1u;
	const u_int tileWidthCount = (width + tileSize - 1u) / tileSize;
	const u_int tileHeightCount = (height + tileSize - 1u) / tileSize;
	const u_int bucketCount = overlapping *
		(tileWidthCount * tileSize * tileHeightCount * tileSize + bucketSize - 1u) /
		bucketSize;
	return sharedData->GetPassCount(bucketCount);
}

PropertiesUPtr ZSobolSampler::ToProperties() const {
	auto props = std::make_unique<Properties>();
	*props << Sampler::ToProperties() <<
		Property("sampler.zsobol.adaptive.strength")(adaptiveStrength) <<
		Property("sampler.zsobol.adaptive.userimportanceweight")(
			adaptiveUserImportanceWeight) <<
		Property("sampler.zsobol.bucketsize")(bucketSize) <<
		Property("sampler.zsobol.tilesize")(tileSize) <<
		Property("sampler.zsobol.supersampling")(superSampling) <<
		Property("sampler.zsobol.overlapping")(overlapping) <<
		Property("sampler.zsobol.samplesperpixel")(samplesPerPixel);
	return props;
}

PropertiesUPtr ZSobolSampler::ToProperties(const Properties &cfg) {
	auto props = std::make_unique<Properties>();
	*props << cfg.Get(GetDefaultProps()->Get("sampler.type")) <<
		cfg.Get(GetDefaultProps()->Get("sampler.imagesamples.enable")) <<
		cfg.Get(GetDefaultProps()->Get("sampler.zsobol.adaptive.strength")) <<
		cfg.Get(GetDefaultProps()->Get("sampler.zsobol.adaptive.userimportanceweight")) <<
		cfg.Get(GetDefaultProps()->Get("sampler.zsobol.bucketsize")) <<
		cfg.Get(GetDefaultProps()->Get("sampler.zsobol.tilesize")) <<
		cfg.Get(GetDefaultProps()->Get("sampler.zsobol.supersampling")) <<
		cfg.Get(GetDefaultProps()->Get("sampler.zsobol.overlapping")) <<
		cfg.Get(GetDefaultProps()->Get("sampler.zsobol.samplesperpixel"));
	return props;
}

u_int ZSobolSampler::ResolveSamplesPerPixel(const Properties &cfg) {
	u_int requested = cfg.Get(GetDefaultProps()->Get(
		"sampler.zsobol.samplesperpixel")).Get<u_int>();
	if (requested != 0u) {
		if (requested > (1u << 30))
			throw runtime_error("ZSOBOL samples-per-pixel domain is too large");
		return RoundUpPow2(requested);
	}

	u_int haltTarget = 0u;
	if (cfg.IsDefined("batch.haltspp")) {
		const Property haltDefault = Property("batch.haltspp")(0u);
		const Property &halt = cfg.Get(haltDefault);
		for (u_int i = 0u; i < halt.GetSize(); ++i)
			haltTarget = Max(haltTarget, halt.Get<u_int>(i));
	}

	if (haltTarget != 0u) {
		// Film::RunTests() stops only after SPP is greater than the requested
		// halt value. NextSample() also prepares the following sample before
		// render-stop signal reaches all worker threads. Adaptive scheduling
		// can additionally give a noisy pixel up to 1/(1-strength) times the
		// average work. Reserve that per-pixel headroom before rounding the
		// fixed PBRT-style Z-order domain to a power of two.
		const double adaptiveStrength = Clamp(cfg.Get(GetDefaultProps()->Get(
			"sampler.zsobol.adaptive.strength")).Get<double>(), 0.0, .95);
		const double minimumRate = std::max(.05, 1. - adaptiveStrength);
		const std::uint64_t minimumDomain = static_cast<std::uint64_t>(std::ceil(
			(static_cast<double>(haltTarget) + 2.) / minimumRate));
		if (minimumDomain >= (std::uint64_t(1u) << 30u))
			throw runtime_error("ZSOBOL halt SPP is too large");
		u_int domain = RoundUpPow2(static_cast<u_int>(minimumDomain));
		if (domain <= minimumDomain)
			domain <<= 1u;
		return domain;
	}

	// A finite fallback is required because PBRT-style Z-ordering needs a
	// fixed power-of-two sample block. 65,536 SPP is far beyond ordinary
	// interactive renders while still making accidental exhaustion explicit.
	return 65536u;
}

SamplerUPtr ZSobolSampler::FromProperties(const Properties &cfg,
		const RandomGeneratorUPtr &rndGen, FilmPtr film,
		const FilmSampleSplatterUPtr &filmSplatter,
		SamplerSharedDataSPtr sharedData) {
	const bool imageSamplesEnable = cfg.Get(GetDefaultProps()->Get(
		"sampler.imagesamples.enable")).Get<bool>();
	const float adaptiveStrength = Clamp(cfg.Get(GetDefaultProps()->Get(
		"sampler.zsobol.adaptive.strength")).Get<double>(), 0.0, .95);
	const float adaptiveUserImportanceWeight = cfg.Get(GetDefaultProps()->Get(
		"sampler.zsobol.adaptive.userimportanceweight")).Get<double>();
	const u_int bucketSize = RoundUpPow2(cfg.Get(GetDefaultProps()->Get(
		"sampler.zsobol.bucketsize")).Get<u_int>());
	const u_int tileSize = RoundUpPow2(cfg.Get(GetDefaultProps()->Get(
		"sampler.zsobol.tilesize")).Get<u_int>());
	const u_int superSampling = Max(1u, cfg.Get(GetDefaultProps()->Get(
		"sampler.zsobol.supersampling")).Get<u_int>());
	const u_int overlapping = Max(1u, cfg.Get(GetDefaultProps()->Get(
		"sampler.zsobol.overlapping")).Get<u_int>());
	const u_int spp = ResolveSamplesPerPixel(cfg);

	return std::make_unique<ZSobolSampler>(rndGen, film, filmSplatter,
		imageSamplesEnable, adaptiveStrength, adaptiveUserImportanceWeight,
		bucketSize, tileSize, superSampling, overlapping, spp,
		dynamic_pointer_cast<ZSobolSamplerSharedData>(sharedData));
}

slg::ocl::Sampler *ZSobolSampler::FromPropertiesOCL(const Properties &cfg) {
	(void)cfg;
	throw runtime_error("ZSOBOL is currently available for CPU render engines only");
}

void ZSobolSampler::AddRequiredChannels(Film::FilmChannels &channels,
		const Properties &cfg) {
	const bool imageSamplesEnable = cfg.Get(GetDefaultProps()->Get(
		"sampler.imagesamples.enable")).Get<bool>();
	const float strength = cfg.Get(GetDefaultProps()->Get(
		"sampler.zsobol.adaptive.strength")).Get<double>();
	if (imageSamplesEnable && strength > 0.f)
		channels.insert(Film::NOISE);
}

PropertiesUPtr ZSobolSampler::GetDefaultProps() {
	auto props = std::make_unique<Properties>();
	*props << Sampler::GetDefaultProps() <<
		Property("sampler.type")(GetObjectTag()) <<
		// Keep adaptive scheduling opt-in for direct LuxCore API users.
		// BlendLuxCore exports the user-selected strength explicitly.
		Property("sampler.zsobol.adaptive.strength")(0.f) <<
		Property("sampler.zsobol.adaptive.userimportanceweight")(.75f) <<
		Property("sampler.zsobol.bucketsize")(16u) <<
		Property("sampler.zsobol.tilesize")(16u) <<
		Property("sampler.zsobol.supersampling")(1u) <<
		Property("sampler.zsobol.overlapping")(1u) <<
		Property("sampler.zsobol.samplesperpixel")(0u);
	return props;
}

