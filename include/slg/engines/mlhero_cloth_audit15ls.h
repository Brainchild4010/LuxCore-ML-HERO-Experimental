#ifndef SLG_MLHERO_CLOTH_AUDIT15LS_H
#define SLG_MLHERO_CLOTH_AUDIT15LS_H

#include <atomic>
#include <cmath>
#include <cstdio>

#include "slg/engines/mlhero_rough_optics15kk.h"

namespace slg { namespace mlhero15ls {

using V = mlhero15kk::V;
constexpr unsigned Budget = 256u;
constexpr unsigned PresetCount = 6u;
constexpr double Tolerance = 2e-4;

inline double Relative(const double a, const double b) {
	return std::fabs(a - b) /
		std::fmax(1e-300, std::fmax(std::fabs(a), std::fabs(b)));
}

inline const char *PresetName(const unsigned preset) {
	static const char *names[PresetCount] = {
		"DENIM", "SILK_SHANTUNG", "SILK_CHARMEUSE",
		"COTTON_TWILL", "WOOL_GABARDINE", "POLYESTER"
	};
	return preset < PresetCount ? names[preset] : "INVALID";
}

struct Record {
	unsigned side = 0u, material = 0u, preset = 0u, yarnType = 0u;
	unsigned long long detail = 0ull;
	double u0 = 0., u1 = 0., branchU = 0.;
	double hitU = 0., hitV = 0., yarnU = 0., yarnV = 0., umax = 0.;
	V fixed, sampled;
	double reportedPdf = 0., cosinePdf = 0., reverseCosinePdf = 0.;
	double lightCos = 0., eyeCos = 0.;
	double nativeScale = 0., evaluateOrderScale = 0.;
	double specularReciprocityError = 0.;
	double nativeMaterialFactor = 1., expectedMaterialFactor = 1.;
	double materialFactorError = 0.;
};

struct LastSample {
	bool valid = false;
	unsigned side = 0u, preset = 0u;
};

using Reporter = void (*)(const Record &);
inline thread_local Reporter reporter = nullptr;
inline thread_local unsigned used[2]{};
inline thread_local LastSample lastSample;

inline std::atomic<unsigned long long> attempts[2]{}, accepted[2]{},
	invalidReference[2]{}, invalidPreset[2]{}, cosinePdfMismatch[2]{},
	materialFactorMismatch[2]{}, specularReciprocityMismatch[2]{},
	nonfiniteScale[2]{}, negativeScale[2]{}, details[2]{}, writeFailures[2]{};
inline std::atomic<unsigned long long> presetSamples[2][PresetCount]{},
	presetMaterialFactorMismatch[2][PresetCount]{},
	presetSpecularReciprocityMismatch[2][PresetCount]{},
	presetNonfiniteScale[2][PresetCount]{}, presetNegativeScale[2][PresetCount]{};

inline std::atomic<unsigned long long> queryCount[2]{}, queryEvalBlack[2]{},
	queryEventMismatch[2]{}, querySampleEvalPdfMismatch[2]{},
	querySamplePdfMismatch[2]{}, queryEvalPdfMismatch[2]{},
	queryReversePdfMismatch[2]{}, queryWeightMismatch[2]{},
	presetQueryCount[2][PresetCount]{}, presetQueryWeightMismatch[2][PresetCount]{};
inline std::atomic<double> sampleEvalPdfSum[2]{}, sampleEvalPdfMax[2]{},
	sampleQueryPdfSum[2]{}, sampleQueryPdfMax[2]{}, evalQueryPdfSum[2]{},
	evalQueryPdfMax[2]{}, reverseEvalQueryPdfSum[2]{}, reverseEvalQueryPdfMax[2]{},
	sampleWeightSum[2]{}, sampleWeightMax[2]{}, specularReciprocitySum[2]{},
	specularReciprocityMax[2]{}, materialFactorSum[2]{}, materialFactorMax[2]{};

inline void AtomicMax(std::atomic<double> &dst, const double value) {
	double current = dst.load(std::memory_order_relaxed);
	while (value > current && !dst.compare_exchange_weak(current, value,
			std::memory_order_relaxed, std::memory_order_relaxed)) { }
}

inline void Reset() {
	for (unsigned s = 0u; s < 2u; ++s) {
		attempts[s] = accepted[s] = invalidReference[s] = invalidPreset[s] = 0ull;
		cosinePdfMismatch[s] = materialFactorMismatch[s] = 0ull;
		specularReciprocityMismatch[s] = nonfiniteScale[s] = negativeScale[s] = 0ull;
		details[s] = writeFailures[s] = 0ull;
		queryCount[s] = queryEvalBlack[s] = queryEventMismatch[s] = 0ull;
		querySampleEvalPdfMismatch[s] = querySamplePdfMismatch[s] = 0ull;
		queryEvalPdfMismatch[s] = queryReversePdfMismatch[s] = queryWeightMismatch[s] = 0ull;
		sampleEvalPdfSum[s] = sampleEvalPdfMax[s] = 0.;
		sampleQueryPdfSum[s] = sampleQueryPdfMax[s] = 0.;
		evalQueryPdfSum[s] = evalQueryPdfMax[s] = 0.;
		reverseEvalQueryPdfSum[s] = reverseEvalQueryPdfMax[s] = 0.;
		sampleWeightSum[s] = sampleWeightMax[s] = 0.;
		specularReciprocitySum[s] = specularReciprocityMax[s] = 0.;
		materialFactorSum[s] = materialFactorMax[s] = 0.;
		for (unsigned p = 0u; p < PresetCount; ++p) {
			presetSamples[s][p] = presetMaterialFactorMismatch[s][p] = 0ull;
			presetSpecularReciprocityMismatch[s][p] = presetNonfiniteScale[s][p] = 0ull;
			presetNegativeScale[s][p] = presetQueryCount[s][p] = 0ull;
			presetQueryWeightMismatch[s][p] = 0ull;
		}
	}
}

struct WorkerScope {
	WorkerScope(const bool enabled, Reporter callback) {
		used[0] = used[1] = 0u;
		lastSample = LastSample();
		reporter = enabled ? callback : nullptr;
	}
	~WorkerScope() { reporter = nullptr; lastSample = LastSample(); }
	WorkerScope(const WorkerScope &) = delete;
	WorkerScope &operator=(const WorkerScope &) = delete;
};

inline void Begin(const bool enabled) { (void)enabled; lastSample = LastSample(); }
inline bool Active(const bool fromLight) {
	const unsigned side = fromLight ? 1u : 0u;
	return reporter && used[side] < Budget;
}
inline bool Consume(LastSample *sample) {
	if (!lastSample.valid) return false;
	*sample = lastSample;
	lastSample = LastSample();
	return true;
}

inline void Capture(const bool fromLight, const unsigned material, const unsigned preset,
		const unsigned yarnType, const double u0, const double u1, const double branchU,
		const double hitU, const double hitV, const double yarnU, const double yarnV,
		const double umax, const V fixed, const V sampled, const double reportedPdf,
		const double nativeScale, const double evaluateOrderScale,
		const double nativeMaterialFactor) {
	const unsigned side = fromLight ? 1u : 0u;
	if (!reporter || used[side] >= Budget) return;
	++used[side];
	attempts[side].fetch_add(1ull, std::memory_order_relaxed);
	accepted[side].fetch_add(1ull, std::memory_order_relaxed);

	Record r;
	r.side = side; r.material = material; r.preset = preset; r.yarnType = yarnType;
	r.u0 = u0; r.u1 = u1; r.branchU = branchU;
	r.hitU = hitU; r.hitV = hitV; r.yarnU = yarnU; r.yarnV = yarnV; r.umax = umax;
	r.fixed = mlhero15kk::Unit(fixed); r.sampled = mlhero15kk::Unit(sampled);
	r.reportedPdf = reportedPdf;
	r.cosinePdf = std::fabs(r.sampled.z) * 0.31830988618379067154;
	r.reverseCosinePdf = std::fabs(r.fixed.z) * 0.31830988618379067154;
	const V light = side ? r.fixed : r.sampled;
	const V eye = side ? r.sampled : r.fixed;
	r.lightCos = std::fabs(light.z); r.eyeCos = std::fabs(eye.z);
	r.nativeScale = nativeScale; r.evaluateOrderScale = evaluateOrderScale;
	r.nativeMaterialFactor = nativeMaterialFactor;
	r.specularReciprocityError = Relative(nativeScale, evaluateOrderScale);
	r.expectedMaterialFactor = side && r.eyeCos > 0. ? r.lightCos / r.eyeCos : 1.;
	r.materialFactorError = Relative(r.nativeMaterialFactor, r.expectedMaterialFactor);

	lastSample.valid = true; lastSample.side = side; lastSample.preset = preset;
	const bool validPreset = preset < PresetCount;
	if (validPreset)
		presetSamples[side][preset].fetch_add(1ull, std::memory_order_relaxed);
	else
		invalidPreset[side].fetch_add(1ull, std::memory_order_relaxed);
	const bool finiteScale = std::isfinite(nativeScale) && std::isfinite(evaluateOrderScale);
	const bool valid = validPreset && reportedPdf > 0. && r.cosinePdf > 0. &&
		std::isfinite(reportedPdf) && std::isfinite(r.materialFactorError) && finiteScale;
	if (!valid)
		invalidReference[side].fetch_add(1ull, std::memory_order_relaxed);
	if (Relative(reportedPdf, r.cosinePdf) > Tolerance)
		cosinePdfMismatch[side].fetch_add(1ull, std::memory_order_relaxed);
	if (r.materialFactorError > Tolerance) {
		materialFactorMismatch[side].fetch_add(1ull, std::memory_order_relaxed);
		if (validPreset) presetMaterialFactorMismatch[side][preset].fetch_add(1ull, std::memory_order_relaxed);
	}
	if (finiteScale && r.specularReciprocityError > Tolerance) {
		specularReciprocityMismatch[side].fetch_add(1ull, std::memory_order_relaxed);
		if (validPreset) presetSpecularReciprocityMismatch[side][preset].fetch_add(1ull, std::memory_order_relaxed);
	}
	if (!finiteScale) {
		nonfiniteScale[side].fetch_add(1ull, std::memory_order_relaxed);
		if (validPreset) presetNonfiniteScale[side][preset].fetch_add(1ull, std::memory_order_relaxed);
	}
	if ((std::isfinite(nativeScale) && nativeScale < 0.) ||
			(std::isfinite(evaluateOrderScale) && evaluateOrderScale < 0.)) {
		negativeScale[side].fetch_add(1ull, std::memory_order_relaxed);
		if (validPreset) presetNegativeScale[side][preset].fetch_add(1ull, std::memory_order_relaxed);
	}
	if (finiteScale) {
		specularReciprocitySum[side].fetch_add(r.specularReciprocityError, std::memory_order_relaxed);
		AtomicMax(specularReciprocityMax[side], r.specularReciprocityError);
	}
	materialFactorSum[side].fetch_add(r.materialFactorError, std::memory_order_relaxed);
	AtomicMax(materialFactorMax[side], r.materialFactorError);

	r.detail = details[side].fetch_add(1ull, std::memory_order_relaxed);
	if (r.detail < 24ull) reporter(r);
}

inline void RecordQuery(const LastSample &last, const bool evalBlack, const bool eventMismatch,
		const double samplePdf, const double evalPdf, const double queryPdf,
		const double evalReversePdf, const double queryReversePdf,
		const double sampleWeightError) {
	if (!last.valid) return;
	const unsigned side = last.side;
	const bool validPreset = last.preset < PresetCount;
	queryCount[side].fetch_add(1ull, std::memory_order_relaxed);
	if (validPreset) presetQueryCount[side][last.preset].fetch_add(1ull, std::memory_order_relaxed);
	if (evalBlack) queryEvalBlack[side].fetch_add(1ull, std::memory_order_relaxed);
	if (eventMismatch) queryEventMismatch[side].fetch_add(1ull, std::memory_order_relaxed);
	const double se = Relative(samplePdf, evalPdf), sq = Relative(samplePdf, queryPdf),
		eq = Relative(evalPdf, queryPdf), rq = Relative(evalReversePdf, queryReversePdf);
	if (se > Tolerance) querySampleEvalPdfMismatch[side].fetch_add(1ull, std::memory_order_relaxed);
	if (sq > Tolerance) querySamplePdfMismatch[side].fetch_add(1ull, std::memory_order_relaxed);
	if (eq > Tolerance) queryEvalPdfMismatch[side].fetch_add(1ull, std::memory_order_relaxed);
	if (rq > Tolerance) queryReversePdfMismatch[side].fetch_add(1ull, std::memory_order_relaxed);
	if (sampleWeightError > Tolerance) {
		queryWeightMismatch[side].fetch_add(1ull, std::memory_order_relaxed);
		if (validPreset) presetQueryWeightMismatch[side][last.preset].fetch_add(1ull, std::memory_order_relaxed);
	}
	sampleEvalPdfSum[side].fetch_add(se, std::memory_order_relaxed); AtomicMax(sampleEvalPdfMax[side], se);
	sampleQueryPdfSum[side].fetch_add(sq, std::memory_order_relaxed); AtomicMax(sampleQueryPdfMax[side], sq);
	evalQueryPdfSum[side].fetch_add(eq, std::memory_order_relaxed); AtomicMax(evalQueryPdfMax[side], eq);
	reverseEvalQueryPdfSum[side].fetch_add(rq, std::memory_order_relaxed); AtomicMax(reverseEvalQueryPdfMax[side], rq);
	sampleWeightSum[side].fetch_add(sampleWeightError, std::memory_order_relaxed);
	AtomicMax(sampleWeightMax[side], sampleWeightError);
}

inline void WriteRecord(FILE *f, const Record &r, const unsigned thread, const unsigned sample) {
	fprintf(f, "phase15ls CLOTH_SAMPLE side=%s detail=%llu thread=%u sample=%u materialID=%u preset=%s presetID=%u yarn=%s u=(%.17g,%.17g,%.17g) hitUV=(%.17g,%.17g) yarnUV=(%.17g,%.17g) umax=%.17g fixed=(%.17g,%.17g,%.17g) sampled=(%.17g,%.17g,%.17g) reportedPdf=%.17g cosinePdf=%.17g reverseCosinePdf=%.17g lightCos=%.17g eyeCos=%.17g nativeScale=%.17g evaluateOrderScale=%.17g specularReciprocityRelErr=%.17g nativeMaterialFactor=%.17g expectedMaterialFactor=%.17g materialFactorRelErr=%.17g diagnosticOnly=1 physicsChanged=0\n",
		r.side ? "LIGHT" : "EYE", r.detail, thread, sample, r.material,
		PresetName(r.preset), r.preset, r.yarnType ? "WEFT" : "WARP",
		r.u0, r.u1, r.branchU, r.hitU, r.hitV, r.yarnU, r.yarnV, r.umax,
		r.fixed.x, r.fixed.y, r.fixed.z, r.sampled.x, r.sampled.y, r.sampled.z,
		r.reportedPdf, r.cosinePdf, r.reverseCosinePdf, r.lightCos, r.eyeCos,
		r.nativeScale, r.evaluateOrderScale, r.specularReciprocityError,
		r.nativeMaterialFactor, r.expectedMaterialFactor, r.materialFactorError);
}

inline void WriteSummary(FILE *f) {
	for (unsigned side = 0u; side < 2u; ++side) {
		const unsigned long long a = accepted[side].load(std::memory_order_relaxed);
		const unsigned long long q = queryCount[side].load(std::memory_order_relaxed);
		const double ad = a ? double(a) : 1., qd = q ? double(q) : 1.;
		fprintf(f, "phase15ls CLOTH_SUMMARY side=%s attempts=%llu accepted=%llu invalidReference=%llu invalidPreset=%llu cosinePdfMismatches=%llu localMaterialFactorMismatches=%llu meanLocalMaterialFactorRelErr=%.9g worstLocalMaterialFactorRelErr=%.9g specularReciprocityMismatches=%llu meanSpecularReciprocityRelErr=%.9g worstSpecularReciprocityRelErr=%.9g nonfiniteScale=%llu negativeScale=%llu queryCount=%llu evalBlack=%llu eventMismatches=%llu sampleVsEvaluatePdfMismatches=%llu sampleVsPdfQueryMismatches=%llu evaluateVsPdfQueryMismatches=%llu reverseEvaluateVsPdfQueryMismatches=%llu sampleWeightMismatches=%llu meanSampleVsEvaluatePdfRelErr=%.9g worstSampleVsEvaluatePdfRelErr=%.9g meanSampleVsPdfQueryRelErr=%.9g worstSampleVsPdfQueryRelErr=%.9g meanEvaluateVsPdfQueryRelErr=%.9g worstEvaluateVsPdfQueryRelErr=%.9g meanReverseEvaluateVsPdfQueryRelErr=%.9g worstReverseEvaluateVsPdfQueryRelErr=%.9g meanSampleVsEvaluateOverPdfRelErr=%.9g worstSampleVsEvaluateOverPdfRelErr=%.9g detailsSelected=%llu writeFailures=%llu toleranceRelative=0.0002 extraSamplerDraws=0 rendererPhysicsChanged=LIGHT_SUBPATH_ESTIMATOR_WEIGHT\n",
			side ? "LIGHT" : "EYE", attempts[side].load(), a,
			invalidReference[side].load(), invalidPreset[side].load(), cosinePdfMismatch[side].load(),
			materialFactorMismatch[side].load(), materialFactorSum[side].load() / ad,
			materialFactorMax[side].load(), specularReciprocityMismatch[side].load(),
			specularReciprocitySum[side].load() / ad, specularReciprocityMax[side].load(),
			nonfiniteScale[side].load(), negativeScale[side].load(), q,
			queryEvalBlack[side].load(), queryEventMismatch[side].load(),
			querySampleEvalPdfMismatch[side].load(), querySamplePdfMismatch[side].load(),
			queryEvalPdfMismatch[side].load(), queryReversePdfMismatch[side].load(),
			queryWeightMismatch[side].load(), sampleEvalPdfSum[side].load() / qd,
			sampleEvalPdfMax[side].load(), sampleQueryPdfSum[side].load() / qd,
			sampleQueryPdfMax[side].load(), evalQueryPdfSum[side].load() / qd,
			evalQueryPdfMax[side].load(), reverseEvalQueryPdfSum[side].load() / qd,
			reverseEvalQueryPdfMax[side].load(), sampleWeightSum[side].load() / qd,
			sampleWeightMax[side].load(), details[side].load() < 24ull ? details[side].load() : 24ull,
			writeFailures[side].load());
		for (unsigned p = 0u; p < PresetCount; ++p)
			fprintf(f, "phase15ls CLOTH_PRESET_SUMMARY side=%s preset=%s presetID=%u samples=%llu localMaterialFactorMismatches=%llu specularReciprocityMismatches=%llu nonfiniteScale=%llu negativeScale=%llu queryCount=%llu sampleWeightMismatches=%llu diagnosticOnly=1 physicsChanged=0\n",
				side ? "LIGHT" : "EYE", PresetName(p), p, presetSamples[side][p].load(),
				presetMaterialFactorMismatch[side][p].load(),
				presetSpecularReciprocityMismatch[side][p].load(),
				presetNonfiniteScale[side][p].load(), presetNegativeScale[side][p].load(),
				presetQueryCount[side][p].load(), presetQueryWeightMismatch[side][p].load());
	}
}

} }

#endif
