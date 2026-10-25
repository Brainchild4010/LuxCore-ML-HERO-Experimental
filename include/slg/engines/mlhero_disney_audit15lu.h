#ifndef SLG_MLHERO_DISNEY_AUDIT15LU_H
#define SLG_MLHERO_DISNEY_AUDIT15LU_H

#include <atomic>
#include <cmath>
#include <cstdio>

#include "slg/engines/mlhero_rough_optics15kk.h"

namespace slg { namespace mlhero15lu {

using V = mlhero15kk::V;
constexpr unsigned Budget = 256u;
constexpr unsigned BranchCount = 3u;
constexpr double Pi = 3.14159265358979323846;
constexpr double Tolerance = 2e-4;

enum Branch : unsigned { METALLIC = 0u, CLEARCOAT = 1u, DIFFUSE = 2u };

inline const char *BranchName(const unsigned branch) {
	static const char *names[BranchCount] = {"METALLIC", "CLEARCOAT", "DIFFUSE"};
	return branch < BranchCount ? names[branch] : "INVALID";
}

inline double Relative(const double a, const double b) {
	return std::fabs(a - b) /
		std::fmax(1e-300, std::fmax(std::fabs(a), std::fabs(b)));
}

inline double Clamp01(const double value) {
	return std::fmax(0., std::fmin(1., value));
}

inline double GTR1(const double nDotH, const double a) {
	if (a >= 1.) return 1. / Pi;
	const double a2 = a * a;
	const double t = 1. + (a2 - 1.) * nDotH * nDotH;
	return (a2 - 1.) / (Pi * std::log(a2) * t);
}

struct PdfReference {
	double total = 0., diffuse = 0., metallic = 0., clearcoat = 0.;
	double ratioDiffuse = 0., ratioMetallic = 0., ratioClearcoat = 0.;
};

inline PdfReference ReferencePdf(const bool fromLight, const double roughnessIn,
		const double metallicIn, const double clearcoatIn, const double clearcoatGlossIn,
		const double anisotropicIn, const V lightIn, const V eyeIn) {
	PdfReference r;
	// Replay the native float direction density exactly.  A double half-vector
	// reconstruction is a different density for very sharp lobes and belongs
	// in a separate precision audit, not in this property/weight diagnosis.
	const float lx = static_cast<float>(lightIn.x), ly = static_cast<float>(lightIn.y),
		lz = static_cast<float>(lightIn.z);
	const float ex = static_cast<float>(eyeIn.x), ey = static_cast<float>(eyeIn.y),
		ez = static_cast<float>(eyeIn.z);
	if (!(lz * ez > 0.f)) return r;
	const float fx = fromLight ? lx : ex, fy = fromLight ? ly : ey,
		fz = fromLight ? lz : ez;
	const float sampledZ = fromLight ? ez : lz;
	float hx = lx + ex, hy = ly + ey, hz = lz + ez;
	const float hLength = std::sqrt(hx * hx + hy * hy + hz * hz);
	if (!(hLength > 0.f)) return r;
	hx /= hLength; hy /= hLength; hz /= hLength;
	const float fixedDotH = fx * hx + fy * hy + fz * hz;
	if (!(fixedDotH > 0.f)) return r;

	const float roughness = static_cast<float>(Clamp01(roughnessIn));
	const float metallic = static_cast<float>(Clamp01(metallicIn));
	const float clearcoat = static_cast<float>(Clamp01(clearcoatIn));
	const float clearcoatGloss = static_cast<float>(Clamp01(clearcoatGlossIn));
	const float anisotropic = static_cast<float>(Clamp01(anisotropicIn));
	const float norm = 1.f / (2.f - metallic + clearcoat);
	r.ratioMetallic = norm;
	r.ratioDiffuse = (1.f - metallic) * norm;
	r.ratioClearcoat = clearcoat * norm;

	r.diffuse = std::fabs(sampledZ) * static_cast<float>(1. / Pi);
	const float aspect = std::sqrt(1.f - .9f * anisotropic);
	const float roughness2 = roughness * roughness;
	const float ax = std::fmax(.001f, roughness2 / aspect);
	const float ay = std::fmax(.001f, roughness2 * aspect);
	const float nDotH = std::fabs(hz);
	const float denom = hx * hx / (ax * ax) + hy * hy / (ay * ay) + nDotH * nDotH;
	if (denom > 0.f) {
		const float distribution = nDotH /
			(Pi * ax * ay * denom * denom);
		r.metallic = distribution / (4.f * fixedDotH);
	}
	const float a = .1f * (1.f - clearcoatGloss) + .001f * clearcoatGloss;
	const float a2 = a * a;
	const float t = 1.f + (a2 - 1.f) * nDotH * nDotH;
	const float dr = (a2 - 1.f) /
		(Pi * std::log(a2) * t);
	r.clearcoat = dr * nDotH / (4.f * fixedDotH);
	const float total = static_cast<float>(r.ratioDiffuse) * static_cast<float>(r.diffuse) +
		static_cast<float>(r.ratioMetallic) * static_cast<float>(r.metallic) +
		static_cast<float>(r.ratioClearcoat) * static_cast<float>(r.clearcoat);
	r.total = total;
	return r;
}

struct Record {
	unsigned side = 0u, material = 0u, branch = 0u;
	unsigned long long detail = 0ull;
	double u0 = 0., u1 = 0., branchU = 0.;
	double roughness = 0., metallic = 0., specularTint = 0.;
	double clearcoat = 0., clearcoatGloss = 0., anisotropic = 0.;
	V fixed, sampled;
	double lightCos = 0., eyeCos = 0.;
	double ratioDiffuse = 0., ratioMetallic = 0., ratioClearcoat = 0.;
	double reportedPdf = 0., correctPdf = 0., legacyTintPdf = 0.;
	double reportedCorrectError = 0., reportedLegacyError = 0.;
};

struct LastSample {
	bool valid = false;
	unsigned side = 0u, branch = 0u;
	double correctPdf = 0., legacyTintPdf = 0., lightCos = 0.;
};

using Reporter = void (*)(const Record &);
inline thread_local Reporter reporter = nullptr;
inline thread_local unsigned used[2]{};
inline thread_local LastSample lastSample;

inline std::atomic<unsigned long long> attempts[2]{}, accepted[2]{},
	invalidReference[2]{}, invalidBranch[2]{}, nonfinite[2]{},
	reportedCorrectMismatch[2]{}, reportedLegacyMismatch[2]{},
	details[2]{}, writeFailures[2]{};
inline std::atomic<unsigned long long> branchSamples[2][BranchCount]{},
	branchWeightMismatch[2][BranchCount]{};
inline std::atomic<unsigned long long> queryCount[2]{}, evalBlack[2]{},
	eventMismatch[2]{}, sampleEvalPdfMismatch[2]{}, sampleQueryPdfMismatch[2]{},
	evalCorrectPdfMismatch[2]{}, queryCorrectPdfMismatch[2]{},
	queryLegacyPdfMismatch[2]{}, sampleWeightMismatch[2]{},
	doubleCosinePredictionMismatch[2]{};
inline std::atomic<double> reportedCorrectSum[2]{}, reportedCorrectMax[2]{},
	reportedLegacySum[2]{}, reportedLegacyMax[2]{}, weightSum[2]{}, weightMax[2]{},
	doubleCosineSum[2]{}, doubleCosineMax[2]{}, queryCorrectSum[2]{},
	queryCorrectMax[2]{}, queryLegacySum[2]{}, queryLegacyMax[2]{};

inline void AtomicMax(std::atomic<double> &dst, const double value) {
	double current = dst.load(std::memory_order_relaxed);
	while (value > current && !dst.compare_exchange_weak(current, value,
			std::memory_order_relaxed, std::memory_order_relaxed)) { }
}

inline void Reset() {
	for (unsigned s = 0u; s < 2u; ++s) {
		attempts[s] = accepted[s] = invalidReference[s] = invalidBranch[s] = 0ull;
		nonfinite[s] = reportedCorrectMismatch[s] = reportedLegacyMismatch[s] = 0ull;
		details[s] = writeFailures[s] = queryCount[s] = evalBlack[s] = 0ull;
		eventMismatch[s] = sampleEvalPdfMismatch[s] = sampleQueryPdfMismatch[s] = 0ull;
		evalCorrectPdfMismatch[s] = queryCorrectPdfMismatch[s] = 0ull;
		queryLegacyPdfMismatch[s] = sampleWeightMismatch[s] = 0ull;
		doubleCosinePredictionMismatch[s] = 0ull;
		reportedCorrectSum[s] = reportedCorrectMax[s] = 0.;
		reportedLegacySum[s] = reportedLegacyMax[s] = 0.;
		weightSum[s] = weightMax[s] = doubleCosineSum[s] = doubleCosineMax[s] = 0.;
		queryCorrectSum[s] = queryCorrectMax[s] = queryLegacySum[s] = queryLegacyMax[s] = 0.;
		for (unsigned b = 0u; b < BranchCount; ++b)
			branchSamples[s][b] = branchWeightMismatch[s][b] = 0ull;
	}
}

struct WorkerScope {
	WorkerScope(const bool enabled, Reporter callback) {
		used[0] = used[1] = 0u; lastSample = LastSample();
		reporter = enabled ? callback : nullptr;
	}
	~WorkerScope() { reporter = nullptr; lastSample = LastSample(); }
	WorkerScope(const WorkerScope &) = delete;
	WorkerScope &operator=(const WorkerScope &) = delete;
};

inline void Begin(const bool) { lastSample = LastSample(); }
inline bool Active(const bool fromLight) {
	return reporter && used[fromLight ? 1u : 0u] < Budget;
}
inline bool Consume(LastSample *sample) {
	if (!lastSample.valid) return false;
	*sample = lastSample; lastSample = LastSample(); return true;
}

inline void Capture(const bool fromLight, const unsigned material, const unsigned branch,
		const double u0, const double u1, const double branchU, const double roughness,
		const double metallic, const double specularTint, const double clearcoat,
		const double clearcoatGloss, const double anisotropic, const V fixedIn,
		const V sampledIn, const double reportedPdf, const double nativeLegacyTintPdf) {
	const unsigned side = fromLight ? 1u : 0u;
	if (!reporter || used[side] >= Budget) return;
	++used[side]; attempts[side].fetch_add(1ull, std::memory_order_relaxed);

	Record r; r.side = side; r.material = material; r.branch = branch;
	r.u0 = u0; r.u1 = u1; r.branchU = branchU; r.roughness = roughness;
	r.metallic = metallic; r.specularTint = specularTint; r.clearcoat = clearcoat;
	r.clearcoatGloss = clearcoatGloss; r.anisotropic = anisotropic;
	r.fixed = mlhero15kk::Unit(fixedIn); r.sampled = mlhero15kk::Unit(sampledIn);
	const V light = side ? fixedIn : sampledIn;
	const V eye = side ? sampledIn : fixedIn;
	r.lightCos = std::fabs(light.z); r.eyeCos = std::fabs(eye.z);
	const PdfReference correct = ReferencePdf(fromLight, roughness, metallic, clearcoat,
		clearcoatGloss, anisotropic, light, eye);
	r.ratioDiffuse = correct.ratioDiffuse; r.ratioMetallic = correct.ratioMetallic;
	r.ratioClearcoat = correct.ratioClearcoat; r.reportedPdf = reportedPdf;
	r.correctPdf = correct.total; r.legacyTintPdf = nativeLegacyTintPdf;
	r.reportedCorrectError = Relative(reportedPdf, r.correctPdf);
	r.reportedLegacyError = Relative(reportedPdf, r.legacyTintPdf);

	lastSample.valid = true; lastSample.side = side; lastSample.branch = branch;
	lastSample.correctPdf = reportedPdf; lastSample.legacyTintPdf = r.legacyTintPdf;
	lastSample.lightCos = r.lightCos;
	accepted[side].fetch_add(1ull, std::memory_order_relaxed);
	if (branch < BranchCount) branchSamples[side][branch].fetch_add(1ull, std::memory_order_relaxed);
	else invalidBranch[side].fetch_add(1ull, std::memory_order_relaxed);
	const bool finite = std::isfinite(reportedPdf) && std::isfinite(r.correctPdf) &&
		std::isfinite(r.legacyTintPdf) && reportedPdf > 0. && r.correctPdf > 0.;
	if (!finite) { invalidReference[side].fetch_add(1ull, std::memory_order_relaxed);
		nonfinite[side].fetch_add(1ull, std::memory_order_relaxed); }
	if (r.reportedCorrectError > Tolerance)
		reportedCorrectMismatch[side].fetch_add(1ull, std::memory_order_relaxed);
	if (r.reportedLegacyError > Tolerance)
		reportedLegacyMismatch[side].fetch_add(1ull, std::memory_order_relaxed);
	reportedCorrectSum[side].fetch_add(r.reportedCorrectError, std::memory_order_relaxed);
	AtomicMax(reportedCorrectMax[side], r.reportedCorrectError);
	reportedLegacySum[side].fetch_add(r.reportedLegacyError, std::memory_order_relaxed);
	AtomicMax(reportedLegacyMax[side], r.reportedLegacyError);
	r.detail = details[side].fetch_add(1ull, std::memory_order_relaxed);
	if (r.detail < 24ull) reporter(r);
}

inline void RecordQuery(const LastSample &last, const bool isEvalBlack,
		const bool isEventMismatch, const double samplePdf, const double evaluatePdf,
		const double queryPdf, const double weightError, const double doubleCosineError) {
	if (!last.valid) return;
	const unsigned s = last.side;
	queryCount[s].fetch_add(1ull, std::memory_order_relaxed);
	if (isEvalBlack) evalBlack[s].fetch_add(1ull, std::memory_order_relaxed);
	if (isEventMismatch) eventMismatch[s].fetch_add(1ull, std::memory_order_relaxed);
	if (Relative(samplePdf, evaluatePdf) > Tolerance) sampleEvalPdfMismatch[s].fetch_add(1ull, std::memory_order_relaxed);
	if (Relative(samplePdf, queryPdf) > Tolerance) sampleQueryPdfMismatch[s].fetch_add(1ull, std::memory_order_relaxed);
	if (Relative(evaluatePdf, last.correctPdf) > Tolerance) evalCorrectPdfMismatch[s].fetch_add(1ull, std::memory_order_relaxed);
	const double queryCorrectError = Relative(queryPdf, last.correctPdf);
	const double queryLegacyError = Relative(queryPdf, last.legacyTintPdf);
	if (queryCorrectError > Tolerance) queryCorrectPdfMismatch[s].fetch_add(1ull, std::memory_order_relaxed);
	if (queryLegacyError > Tolerance) queryLegacyPdfMismatch[s].fetch_add(1ull, std::memory_order_relaxed);
	if (weightError > Tolerance) { sampleWeightMismatch[s].fetch_add(1ull, std::memory_order_relaxed);
		if (last.branch < BranchCount) branchWeightMismatch[s][last.branch].fetch_add(1ull, std::memory_order_relaxed); }
	if (doubleCosineError > Tolerance) doubleCosinePredictionMismatch[s].fetch_add(1ull, std::memory_order_relaxed);
	weightSum[s].fetch_add(weightError, std::memory_order_relaxed); AtomicMax(weightMax[s], weightError);
	doubleCosineSum[s].fetch_add(doubleCosineError, std::memory_order_relaxed); AtomicMax(doubleCosineMax[s], doubleCosineError);
	queryCorrectSum[s].fetch_add(queryCorrectError, std::memory_order_relaxed); AtomicMax(queryCorrectMax[s], queryCorrectError);
	queryLegacySum[s].fetch_add(queryLegacyError, std::memory_order_relaxed); AtomicMax(queryLegacyMax[s], queryLegacyError);
}

inline void WriteRecord(FILE *f, const Record &r, const unsigned thread, const unsigned sample) {
	fprintf(f, "phase15lv DISNEY_SAMPLE side=%s detail=%llu thread=%u sample=%u materialID=%u branch=%s branchID=%u u=(%.17g,%.17g,%.17g) roughness=%.17g metallic=%.17g specularTint=%.17g clearcoat=%.17g clearcoatGloss=%.17g anisotropic=%.17g fixed=(%.17g,%.17g,%.17g) sampled=(%.17g,%.17g,%.17g) lightCos=%.17g eyeCos=%.17g ratios=(%.17g,%.17g,%.17g) reportedPdf=%.17g correctPdf=%.17g legacySpecularTintAsClearcoatPdf=%.17g reportedCorrectRelErr=%.17g reportedLegacyRelErr=%.17g diagnosticOnly=1 rendererPhysicsChanged=DISNEY_TRANSPORT_PDF_AND_TINT\n",
		r.side ? "LIGHT" : "EYE", r.detail, thread, sample, r.material,
		BranchName(r.branch), r.branch, r.u0, r.u1, r.branchU, r.roughness,
		r.metallic, r.specularTint, r.clearcoat, r.clearcoatGloss, r.anisotropic,
		r.fixed.x, r.fixed.y, r.fixed.z, r.sampled.x, r.sampled.y, r.sampled.z,
		r.lightCos, r.eyeCos, r.ratioMetallic, r.ratioClearcoat, r.ratioDiffuse,
		r.reportedPdf, r.correctPdf, r.legacyTintPdf,
		r.reportedCorrectError, r.reportedLegacyError);
}

inline void WriteSummary(FILE *f) {
	for (unsigned s = 0u; s < 2u; ++s) {
		const unsigned long long a = accepted[s].load(), q = queryCount[s].load();
		const double ad = a ? double(a) : 1., qd = q ? double(q) : 1.;
		fprintf(f, "phase15lv DISNEY_SUMMARY side=%s attempts=%llu accepted=%llu invalidReference=%llu invalidBranch=%llu nonfinite=%llu reportedVsCorrectPdfMismatches=%llu meanReportedCorrectPdfRelErr=%.9g worstReportedCorrectPdfRelErr=%.9g reportedVsLegacyTintPdfMismatches=%llu meanReportedLegacyPdfRelErr=%.9g worstReportedLegacyPdfRelErr=%.9g queryCount=%llu evalBlack=%llu eventMismatches=%llu sampleVsEvaluatePdfMismatches=%llu sampleVsPdfQueryMismatches=%llu evaluateVsCorrectPdfMismatches=%llu queryVsCorrectPdfMismatches=%llu meanQueryCorrectPdfRelErr=%.9g worstQueryCorrectPdfRelErr=%.9g queryVsLegacyTintPdfMismatches=%llu meanQueryLegacyPdfRelErr=%.9g worstQueryLegacyPdfRelErr=%.9g sampleWeightMismatches=%llu meanSampleWeightRelErr=%.9g worstSampleWeightRelErr=%.9g oldDoubleCosineReplayMismatches=%llu meanOldDoubleCosineReplayRelErr=%.9g worstOldDoubleCosineReplayRelErr=%.9g detailsSelected=%llu writeFailures=%llu toleranceRelative=0.0002 extraSamplerDraws=0 acceptedSamplePdfQueries=2 extraLegacyPdfEvaluations=1_PER_SELECTED rendererPhysicsChanged=DISNEY_TRANSPORT_PDF_AND_TINT\n",
			s ? "LIGHT" : "EYE", attempts[s].load(), a, invalidReference[s].load(),
			invalidBranch[s].load(), nonfinite[s].load(), reportedCorrectMismatch[s].load(),
			reportedCorrectSum[s].load() / ad, reportedCorrectMax[s].load(),
			reportedLegacyMismatch[s].load(), reportedLegacySum[s].load() / ad,
			reportedLegacyMax[s].load(), q, evalBlack[s].load(), eventMismatch[s].load(),
			sampleEvalPdfMismatch[s].load(), sampleQueryPdfMismatch[s].load(),
			evalCorrectPdfMismatch[s].load(), queryCorrectPdfMismatch[s].load(),
			queryCorrectSum[s].load() / qd, queryCorrectMax[s].load(),
			queryLegacyPdfMismatch[s].load(), queryLegacySum[s].load() / qd,
			queryLegacyMax[s].load(), sampleWeightMismatch[s].load(), weightSum[s].load() / qd,
			weightMax[s].load(), doubleCosinePredictionMismatch[s].load(),
			doubleCosineSum[s].load() / qd, doubleCosineMax[s].load(),
			details[s].load() < 24ull ? details[s].load() : 24ull, writeFailures[s].load());
		for (unsigned b = 0u; b < BranchCount; ++b)
			fprintf(f, "phase15lv DISNEY_BRANCH_SUMMARY side=%s branch=%s branchID=%u samples=%llu sampleWeightMismatches=%llu diagnosticOnly=1 rendererPhysicsChanged=DISNEY_TRANSPORT_PDF_AND_TINT\n",
				s ? "LIGHT" : "EYE", BranchName(b), b, branchSamples[s][b].load(),
				branchWeightMismatch[s][b].load());
	}
}

} }

#endif
