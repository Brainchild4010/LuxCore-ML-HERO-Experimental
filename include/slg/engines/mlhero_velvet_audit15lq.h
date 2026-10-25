#ifndef SLG_MLHERO_VELVET_AUDIT15LQ_H
#define SLG_MLHERO_VELVET_AUDIT15LQ_H

#include <atomic>
#include <cmath>
#include <cstdio>

#include "slg/engines/mlhero_rough_optics15kk.h"

namespace slg { namespace mlhero15lq {

using V = mlhero15kk::V;
constexpr unsigned Budget = 256u;
constexpr unsigned StatusCount = 3u;
enum Status : unsigned { DRAW_ONLY = 0u, DIRECTION_REJECT = 1u, ACCEPT = 2u };

inline double Relative(const double a, const double b) {
	return std::fabs(a - b) /
		std::fmax(1e-300, std::fmax(std::fabs(a), std::fabs(b)));
}
inline double Clamp01(const double v) { return std::fmax(0., std::fmin(1., v)); }

struct Record {
	unsigned side = 0u, material = 0u, status = DRAW_ONLY;
	unsigned long long detail = 0ull;
	double u0 = 0., u1 = 0., branchU = 0.;
	double a1 = 0., a2 = 0., a3 = 0., thickness = 0.;
	V fixed, sampled;
	double reportedPdf = 0., cosinePdf = 0., reverseCosinePdf = 0.;
	double cosv = 0., polynomial = 0., rawPhaseThickness = 0.;
	double rawBrdf = 0.;
	double claimedP = 0., referenceP = 0., swappedP = 0.;
	double reciprocityError = 0.;
};

struct LastSample {
	bool valid = false;
	unsigned side = 0u, status = DRAW_ONLY;
};

using Reporter = void (*)(const Record &);
inline thread_local Reporter reporter = nullptr;
inline thread_local unsigned used[2]{};
inline thread_local LastSample lastSample;

inline std::atomic<unsigned long long> attempts[2]{}, status[2][StatusCount]{},
	invalidReference[2]{}, cosinePdfMismatch[2]{}, phaseValueMismatch[2]{},
	upperClamp[2]{}, lowerClamp[2]{}, swappedUpperClamp[2]{}, swappedLowerClamp[2]{},
	reciprocityMismatch[2]{}, details[2]{}, writeFailures[2]{};
inline std::atomic<unsigned long long> queryCount[2]{}, queryEvalBlack[2]{},
	queryEventMismatch[2]{};
inline std::atomic<double> sampleEvalPdfSum[2]{}, sampleEvalPdfMax[2]{},
	sampleQueryPdfSum[2]{}, sampleQueryPdfMax[2]{}, evalQueryPdfSum[2]{},
	evalQueryPdfMax[2]{}, reverseEvalQueryPdfSum[2]{}, reverseEvalQueryPdfMax[2]{},
	sampleWeightSum[2]{}, sampleWeightMax[2]{}, reciprocitySum[2]{}, reciprocityMax[2]{};

inline void AtomicMax(std::atomic<double> &dst, const double value) {
	double current = dst.load(std::memory_order_relaxed);
	while (value > current && !dst.compare_exchange_weak(current, value,
			std::memory_order_relaxed, std::memory_order_relaxed)) { }
}

inline const char *StatusName(const unsigned value) {
	static const char *names[StatusCount] = {"DRAW_ONLY", "DIRECTION_REJECT", "ACCEPT"};
	return value < StatusCount ? names[value] : "UNKNOWN";
}

inline void Reset() {
	for (unsigned s = 0u; s < 2u; ++s) {
		attempts[s] = invalidReference[s] = cosinePdfMismatch[s] = phaseValueMismatch[s] = 0ull;
		upperClamp[s] = lowerClamp[s] = swappedUpperClamp[s] = swappedLowerClamp[s] = 0ull;
		reciprocityMismatch[s] = details[s] = writeFailures[s] = 0ull;
		queryCount[s] = queryEvalBlack[s] = queryEventMismatch[s] = 0ull;
		sampleEvalPdfSum[s] = sampleEvalPdfMax[s] = 0.;
		sampleQueryPdfSum[s] = sampleQueryPdfMax[s] = 0.;
		evalQueryPdfSum[s] = evalQueryPdfMax[s] = 0.;
		reverseEvalQueryPdfSum[s] = reverseEvalQueryPdfMax[s] = 0.;
		sampleWeightSum[s] = sampleWeightMax[s] = 0.;
		reciprocitySum[s] = reciprocityMax[s] = 0.;
		for (unsigned k = 0u; k < StatusCount; ++k) status[s][k] = 0ull;
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

inline void Begin(const bool enabled) { (void)enabled; lastSample = LastSample(); }
inline bool Consume(LastSample *sample) {
	if (!lastSample.valid) return false;
	*sample = lastSample; lastSample = LastSample(); return true;
}

struct Attempt {
	bool active = false;
	Record r;
	Attempt(const bool fromLight, const unsigned material, const double a1,
			const double a2, const double a3, const double thickness,
			const double u0, const double u1, const double branchU, const V fixed) {
		const unsigned s = fromLight ? 1u : 0u;
		if (!reporter || used[s] >= Budget) return;
		++used[s]; active = true; r.side = s; r.material = material;
		r.a1 = a1; r.a2 = a2; r.a3 = a3; r.thickness = thickness;
		r.u0 = u0; r.u1 = u1; r.branchU = branchU; r.fixed = mlhero15kk::Unit(fixed);
		attempts[s].fetch_add(1ull, std::memory_order_relaxed);
	}
	bool Active() const { return active; }
	void Reject(const Status reason) { if (active) r.status = reason; }
	void Accept(const V sampled, const double reportedPdf, const double claimedP) {
		if (!active) return;
		r.status = ACCEPT; r.sampled = mlhero15kk::Unit(sampled);
		r.reportedPdf = reportedPdf;
		const double invPi = 0.31830988618379067154;
		r.cosinePdf = std::fabs(r.sampled.z) * invPi;
		r.reverseCosinePdf = std::fabs(r.fixed.z) * invPi;
		const V light = r.side ? r.fixed : r.sampled;
		const V eye = r.side ? r.sampled : r.fixed;
		r.cosv = -mlhero15kk::Dot(light, eye);
		const double c2 = r.cosv * r.cosv;
		r.polynomial = 1. + r.a1 * r.cosv + r.a2 * .5 * (3. * c2 - 1.) +
			r.a3 * .5 * (5. * c2 * r.cosv - 3. * r.cosv);
		r.rawPhaseThickness = r.polynomial * r.thickness / (4. * 3.14159265358979323846);
		const double eyeCos = std::fabs(eye.z), lightCos = std::fabs(light.z);
		r.claimedP = claimedP;
		if (eyeCos > 0. && lightCos > 0.) {
			r.rawBrdf = r.rawPhaseThickness / (eyeCos * lightCos);
			const double brdf = Clamp01(r.rawBrdf);
			r.referenceP = brdf * lightCos;
			r.swappedP = brdf * eyeCos;
			r.reciprocityError = Relative(r.referenceP / lightCos, r.swappedP / eyeCos);
		}
	}
	~Attempt() {
		if (!active) return;
		const unsigned s = r.side;
		lastSample.valid = true; lastSample.side = s; lastSample.status = r.status;
		status[s][r.status].fetch_add(1ull, std::memory_order_relaxed);
		if (r.status == ACCEPT) {
			const bool valid = r.reportedPdf > 0. && r.cosinePdf > 0. &&
				std::isfinite(r.reportedPdf) && std::isfinite(r.referenceP) &&
				std::isfinite(r.reciprocityError);
			if (!valid) invalidReference[s].fetch_add(1ull, std::memory_order_relaxed);
			else {
				if (Relative(r.reportedPdf, r.cosinePdf) > 2e-4)
					cosinePdfMismatch[s].fetch_add(1ull, std::memory_order_relaxed);
				if (Relative(r.claimedP, r.referenceP) > 2e-4)
					phaseValueMismatch[s].fetch_add(1ull, std::memory_order_relaxed);
				if (r.reciprocityError > 2e-4)
					reciprocityMismatch[s].fetch_add(1ull, std::memory_order_relaxed);
			}
			if (r.rawBrdf > 1.) {
				upperClamp[s].fetch_add(1ull, std::memory_order_relaxed);
				swappedUpperClamp[s].fetch_add(1ull, std::memory_order_relaxed);
			}
			if (r.rawBrdf < 0.) {
				lowerClamp[s].fetch_add(1ull, std::memory_order_relaxed);
				swappedLowerClamp[s].fetch_add(1ull, std::memory_order_relaxed);
			}
			reciprocitySum[s].fetch_add(r.reciprocityError, std::memory_order_relaxed);
			AtomicMax(reciprocityMax[s], r.reciprocityError);
		}
		r.detail = details[s].fetch_add(1ull, std::memory_order_relaxed);
		if (r.detail < 24ull) reporter(r);
	}
	Attempt(const Attempt &) = delete;
	Attempt &operator=(const Attempt &) = delete;
};

inline void RecordQuery(const LastSample &last, const bool evalBlack, const bool eventMismatch,
		const double samplePdf, const double evalPdf, const double queryPdf,
		const double evalReversePdf, const double queryReversePdf,
		const double sampleWeightError) {
	if (!last.valid || last.status != ACCEPT) return;
	const unsigned s = last.side;
	queryCount[s].fetch_add(1ull, std::memory_order_relaxed);
	if (evalBlack) queryEvalBlack[s].fetch_add(1ull, std::memory_order_relaxed);
	if (eventMismatch) queryEventMismatch[s].fetch_add(1ull, std::memory_order_relaxed);
	const double se = Relative(samplePdf, evalPdf), sq = Relative(samplePdf, queryPdf),
		eq = Relative(evalPdf, queryPdf), rq = Relative(evalReversePdf, queryReversePdf);
	sampleEvalPdfSum[s].fetch_add(se, std::memory_order_relaxed); AtomicMax(sampleEvalPdfMax[s], se);
	sampleQueryPdfSum[s].fetch_add(sq, std::memory_order_relaxed); AtomicMax(sampleQueryPdfMax[s], sq);
	evalQueryPdfSum[s].fetch_add(eq, std::memory_order_relaxed); AtomicMax(evalQueryPdfMax[s], eq);
	reverseEvalQueryPdfSum[s].fetch_add(rq, std::memory_order_relaxed); AtomicMax(reverseEvalQueryPdfMax[s], rq);
	sampleWeightSum[s].fetch_add(sampleWeightError, std::memory_order_relaxed);
	AtomicMax(sampleWeightMax[s], sampleWeightError);
}

inline void WriteRecord(FILE *f, const Record &r, const unsigned thread, const unsigned sample) {
	fprintf(f, "phase15lq VELVET_SAMPLE side=%s detail=%llu thread=%u sample=%u materialID=%u status=%s u=(%.17g,%.17g,%.17g) params=(%.17g,%.17g,%.17g,%.17g) fixed=(%.17g,%.17g,%.17g) sampled=(%.17g,%.17g,%.17g) reportedPdf=%.17g cosinePdf=%.17g reverseCosinePdf=%.17g cosv=%.17g polynomial=%.17g rawPhaseThickness=%.17g rawBrdf=%.17g claimedP=%.17g referenceP=%.17g swappedP=%.17g reciprocityRelErr=%.17g diagnosticOnly=0 physicsChanged=RECIPROCAL_BRDF_CLAMP\n",
		r.side ? "LIGHT" : "EYE", r.detail, thread, sample, r.material, StatusName(r.status),
		r.u0, r.u1, r.branchU, r.a1, r.a2, r.a3, r.thickness,
		r.fixed.x, r.fixed.y, r.fixed.z, r.sampled.x, r.sampled.y, r.sampled.z,
		r.reportedPdf, r.cosinePdf, r.reverseCosinePdf, r.cosv, r.polynomial,
		r.rawPhaseThickness, r.rawBrdf, r.claimedP, r.referenceP, r.swappedP, r.reciprocityError);
}

inline void WriteSummary(FILE *f) {
	for (unsigned s = 0u; s < 2u; ++s) {
		const unsigned long long q = queryCount[s].load(std::memory_order_relaxed);
		const unsigned long long a = status[s][ACCEPT].load(std::memory_order_relaxed);
		const double qd = q ? double(q) : 1., ad = a ? double(a) : 1.;
		fprintf(f, "phase15lq VELVET_SUMMARY side=%s attempts=%llu drawOnly=%llu directionRejected=%llu accepted=%llu invalidReference=%llu cosinePdfMismatches=%llu phaseValueMismatches=%llu upperClamp=%llu lowerClamp=%llu swappedUpperClamp=%llu swappedLowerClamp=%llu reciprocityMismatches=%llu meanReciprocityRelErr=%.9g worstReciprocityRelErr=%.9g queryCount=%llu evalBlack=%llu eventMismatches=%llu meanSampleVsEvaluatePdfRelErr=%.9g worstSampleVsEvaluatePdfRelErr=%.9g meanSampleVsPdfQueryRelErr=%.9g worstSampleVsPdfQueryRelErr=%.9g meanEvaluateVsPdfQueryRelErr=%.9g worstEvaluateVsPdfQueryRelErr=%.9g meanReverseEvaluateVsPdfQueryRelErr=%.9g worstReverseEvaluateVsPdfQueryRelErr=%.9g meanSampleVsEvaluateOverPdfRelErr=%.9g worstSampleVsEvaluateOverPdfRelErr=%.9g detailsSelected=%llu writeFailures=%llu toleranceRelative=0.0002 extraSamplerDraws=0 rendererPhysicsChanged=RECIPROCAL_BRDF_CLAMP\n",
			s ? "LIGHT" : "EYE", attempts[s].load(), status[s][DRAW_ONLY].load(),
			status[s][DIRECTION_REJECT].load(), a, invalidReference[s].load(),
			cosinePdfMismatch[s].load(), phaseValueMismatch[s].load(), upperClamp[s].load(),
			lowerClamp[s].load(), swappedUpperClamp[s].load(), swappedLowerClamp[s].load(),
			reciprocityMismatch[s].load(), reciprocitySum[s].load() / ad, reciprocityMax[s].load(),
			q, queryEvalBlack[s].load(), queryEventMismatch[s].load(),
			sampleEvalPdfSum[s].load() / qd, sampleEvalPdfMax[s].load(),
			sampleQueryPdfSum[s].load() / qd, sampleQueryPdfMax[s].load(),
			evalQueryPdfSum[s].load() / qd, evalQueryPdfMax[s].load(),
			reverseEvalQueryPdfSum[s].load() / qd, reverseEvalQueryPdfMax[s].load(),
			sampleWeightSum[s].load() / qd, sampleWeightMax[s].load(),
			details[s].load() < 24ull ? details[s].load() : 24ull, writeFailures[s].load());
	}
}

} }
#endif
