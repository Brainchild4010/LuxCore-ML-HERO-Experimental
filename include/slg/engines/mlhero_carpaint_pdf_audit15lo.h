#ifndef SLG_MLHERO_CARPAINT_PDF_AUDIT15LO_H
#define SLG_MLHERO_CARPAINT_PDF_AUDIT15LO_H

#include <atomic>
#include <cmath>
#include <cstdio>

#include "slg/engines/mlhero_metal_pdf15lb.h"
#include "slg/engines/mlhero_rough_optics15kk.h"

namespace slg { namespace mlhero15lo {

using V = mlhero15kk::V;
constexpr unsigned Budget = 256u;
constexpr unsigned BranchCount = 4u;
constexpr unsigned StatusCount = 4u;
enum Branch : unsigned { BASE = 0u, LAYER1 = 1u, LAYER2 = 2u, LAYER3 = 3u };
enum Status : unsigned { DRAW_ONLY = 0u, DIRECTION_REJECT = 1u, PDF_REJECT = 2u, ACCEPT = 3u };

inline double Relative(const double a, const double b) {
	return std::fabs(a - b) /
		std::fmax(1e-300, std::fmax(std::fabs(a), std::fabs(b)));
}

struct Record {
	unsigned side = 0u, branch = 0u, material = 0u, status = DRAW_ONLY;
	unsigned activeMask = 0u, componentCount = 1u, expectedBranch = 0u;
	unsigned long long detail = 0ull;
	double u0 = 0., u1 = 0., branchU = 0.;
	double rough[3] = {0., 0., 0.};
	V fixed, sampled, h;
	double reportedPdf = 0., legacyReplayPdf = 0.;
	double claimedForwardPdf = 0., generatedForwardPdf = 0.;
	double sampledHalfClaim = 0., sampledHalfReference = 0.;
};

struct LastSample {
	bool valid = false;
	unsigned side = 0u, branch = 0u, status = DRAW_ONLY;
};

using Reporter = void (*)(const Record &);
inline thread_local Reporter reporter = nullptr;
inline thread_local unsigned used[2][BranchCount]{};
inline thread_local LastSample lastSample;

inline std::atomic<unsigned long long> attempts[2][BranchCount]{}, status[2][BranchCount][StatusCount]{},
	selectionMismatch[2][BranchCount]{}, invalidReference[2][BranchCount]{},
	legacyReplayMismatch[2][BranchCount]{}, claimedForwardMismatch[2][BranchCount]{},
	generatedProposalMismatch[2][BranchCount]{}, sampledHalfMismatch[2][BranchCount]{},
	details[2][BranchCount]{}, writeFailures[2][BranchCount]{};

inline std::atomic<unsigned long long> queryCount[2][BranchCount]{}, queryEvalBlack[2][BranchCount]{},
	queryEventMismatch[2][BranchCount]{};
inline std::atomic<double> sampleEvalPdfSum[2][BranchCount]{}, sampleEvalPdfMax[2][BranchCount]{},
	sampleQueryPdfSum[2][BranchCount]{}, sampleQueryPdfMax[2][BranchCount]{},
	evalQueryPdfSum[2][BranchCount]{}, evalQueryPdfMax[2][BranchCount]{},
	sampleWeightSum[2][BranchCount]{}, sampleWeightMax[2][BranchCount]{};

inline void AtomicMax(std::atomic<double> &dst, const double value) {
	double current = dst.load(std::memory_order_relaxed);
	while (value > current && !dst.compare_exchange_weak(current, value,
			std::memory_order_relaxed, std::memory_order_relaxed)) { }
}

inline const char *BranchName(const unsigned branch) {
	static const char *names[BranchCount] = {"BASE", "LAYER1", "LAYER2", "LAYER3"};
	return branch < BranchCount ? names[branch] : "UNKNOWN";
}
inline const char *StatusName(const unsigned value) {
	static const char *names[StatusCount] = {"DRAW_ONLY", "DIRECTION_REJECT", "PDF_REJECT", "ACCEPT"};
	return value < StatusCount ? names[value] : "UNKNOWN";
}

inline unsigned ExpectedBranch(const unsigned activeMask, const unsigned componentCount,
		const double branchU) {
	unsigned slot = unsigned(branchU * componentCount);
	if (slot >= componentCount) slot = componentCount - 1u;
	if (slot == 0u) return BASE;
	for (unsigned layer = 0u; layer < 3u; ++layer)
		if (activeMask & (1u << layer)) {
			if (--slot == 0u) return layer + 1u;
		}
	return BASE;
}

inline void Reset() {
	for (unsigned s = 0u; s < 2u; ++s) for (unsigned b = 0u; b < BranchCount; ++b) {
		attempts[s][b] = 0ull; selectionMismatch[s][b] = 0ull;
		invalidReference[s][b] = 0ull; legacyReplayMismatch[s][b] = 0ull;
		claimedForwardMismatch[s][b] = 0ull; generatedProposalMismatch[s][b] = 0ull;
		sampledHalfMismatch[s][b] = 0ull; details[s][b] = 0ull; writeFailures[s][b] = 0ull;
		queryCount[s][b] = 0ull; queryEvalBlack[s][b] = 0ull; queryEventMismatch[s][b] = 0ull;
		sampleEvalPdfSum[s][b] = 0.; sampleEvalPdfMax[s][b] = 0.;
		sampleQueryPdfSum[s][b] = 0.; sampleQueryPdfMax[s][b] = 0.;
		evalQueryPdfSum[s][b] = 0.; evalQueryPdfMax[s][b] = 0.;
		sampleWeightSum[s][b] = 0.; sampleWeightMax[s][b] = 0.;
		for (unsigned k = 0u; k < StatusCount; ++k) status[s][b][k] = 0ull;
	}
}

struct WorkerScope {
	WorkerScope(const bool enabled, Reporter callback) {
		for (auto &side : used) for (auto &value : side) value = 0u;
		lastSample = LastSample(); reporter = enabled ? callback : nullptr;
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
	Attempt(const bool fromLight, const unsigned material, const unsigned branch,
			const unsigned activeMask, const unsigned componentCount,
			const double rough1, const double rough2, const double rough3,
			const double u0, const double u1, const double branchU, const V fixed) {
		const unsigned s = fromLight ? 1u : 0u;
		if (!reporter || branch >= BranchCount || used[s][branch] >= Budget) return;
		++used[s][branch]; active = true; r.side = s; r.branch = branch; r.material = material;
		r.activeMask = activeMask; r.componentCount = componentCount;
		r.expectedBranch = ExpectedBranch(activeMask, componentCount, branchU);
		r.rough[0] = rough1; r.rough[1] = rough2; r.rough[2] = rough3;
		r.u0 = u0; r.u1 = u1; r.branchU = branchU; r.fixed = mlhero15kk::Unit(fixed);
		attempts[s][branch].fetch_add(1ull, std::memory_order_relaxed);
		if (branch != r.expectedBranch)
			selectionMismatch[s][branch].fetch_add(1ull, std::memory_order_relaxed);
	}
	bool Active() const { return active; }
	void Reject(const Status reason) { if (active) r.status = reason; }
	void Accept(const V sampled, const double reportedPdf, const double sampledHalfClaim,
			const double claimed1, const double claimed2, const double claimed3) {
		if (!active) return;
		r.status = ACCEPT; r.sampled = mlhero15kk::Unit(sampled);
		r.h = mlhero15kk::Unit(r.fixed + r.sampled); r.reportedPdf = reportedPdf;
		r.sampledHalfClaim = sampledHalfClaim;
		const double claimed[3] = {claimed1, claimed2, claimed3};
		const double jacobianDenominator = 4. * std::fabs(mlhero15kk::Dot(r.fixed, r.h));
		double claimedSum = 0., generatedSum = 0.;
		for (unsigned layer = 0u; layer < 3u; ++layer) if (r.activeMask & (1u << layer)) {
			claimedSum += claimed[layer];
			generatedSum += mlhero15lb::HalfPdf(r.rough[layer], 0., r.h.x, r.h.y, r.h.z);
		}
		const double invPi = 0.31830988618379067154;
		const double generatedBase = std::fabs(r.sampled.z) * invPi;
		// The native light-side fixed-cosine term is used only when a glossy
		// branch adds the diffuse component. A diffuse-selected sample retains
		// the cosine density of the generated direction.
		const double legacyBase = (r.side && r.branch != BASE ?
			std::fabs(r.fixed.z) : std::fabs(r.sampled.z)) * invPi;
		if (jacobianDenominator > 0.) {
			r.legacyReplayPdf = (legacyBase + claimedSum / jacobianDenominator) / r.componentCount;
			r.claimedForwardPdf = (generatedBase + claimedSum / jacobianDenominator) / r.componentCount;
			r.generatedForwardPdf = (generatedBase + generatedSum / jacobianDenominator) / r.componentCount;
		}
		if (r.branch != BASE)
			r.sampledHalfReference = mlhero15lb::HalfPdf(r.rough[r.branch - 1u], 0.,
				r.h.x, r.h.y, r.h.z);
	}
	~Attempt() {
		if (!active) return;
		const unsigned s = r.side, b = r.branch;
		lastSample.valid = true; lastSample.side = s; lastSample.branch = b; lastSample.status = r.status;
		status[s][b][r.status].fetch_add(1ull, std::memory_order_relaxed);
		if (r.status == ACCEPT) {
			const bool valid = r.reportedPdf > 0. && r.legacyReplayPdf > 0. &&
				r.claimedForwardPdf > 0. && r.generatedForwardPdf > 0. &&
				std::isfinite(r.reportedPdf) && std::isfinite(r.generatedForwardPdf);
			if (!valid) invalidReference[s][b].fetch_add(1ull, std::memory_order_relaxed);
			else {
				if (Relative(r.reportedPdf, r.legacyReplayPdf) > 2e-4)
					legacyReplayMismatch[s][b].fetch_add(1ull, std::memory_order_relaxed);
				if (Relative(r.reportedPdf, r.claimedForwardPdf) > 2e-4)
					claimedForwardMismatch[s][b].fetch_add(1ull, std::memory_order_relaxed);
				if (Relative(r.reportedPdf, r.generatedForwardPdf) > 2e-4)
					generatedProposalMismatch[s][b].fetch_add(1ull, std::memory_order_relaxed);
				if (b != BASE && Relative(r.sampledHalfClaim, r.sampledHalfReference) > 2e-4)
					sampledHalfMismatch[s][b].fetch_add(1ull, std::memory_order_relaxed);
			}
		}
		r.detail = details[s][b].fetch_add(1ull, std::memory_order_relaxed);
		if (r.detail < 16ull) reporter(r);
	}
	Attempt(const Attempt &) = delete;
	Attempt &operator=(const Attempt &) = delete;
};

inline void RecordQuery(const LastSample &last, const bool evalBlack, const bool eventMismatch,
		const double samplePdf, const double evalPdf, const double queryPdf,
		const double sampleWeightError) {
	if (!last.valid || last.status != ACCEPT || last.branch >= BranchCount) return;
	const unsigned s = last.side, b = last.branch;
	queryCount[s][b].fetch_add(1ull, std::memory_order_relaxed);
	if (evalBlack) queryEvalBlack[s][b].fetch_add(1ull, std::memory_order_relaxed);
	if (eventMismatch) queryEventMismatch[s][b].fetch_add(1ull, std::memory_order_relaxed);
	const double se = Relative(samplePdf, evalPdf), sq = Relative(samplePdf, queryPdf),
		eq = Relative(evalPdf, queryPdf);
	sampleEvalPdfSum[s][b].fetch_add(se, std::memory_order_relaxed); AtomicMax(sampleEvalPdfMax[s][b], se);
	sampleQueryPdfSum[s][b].fetch_add(sq, std::memory_order_relaxed); AtomicMax(sampleQueryPdfMax[s][b], sq);
	evalQueryPdfSum[s][b].fetch_add(eq, std::memory_order_relaxed); AtomicMax(evalQueryPdfMax[s][b], eq);
	sampleWeightSum[s][b].fetch_add(sampleWeightError, std::memory_order_relaxed);
	AtomicMax(sampleWeightMax[s][b], sampleWeightError);
}

inline void WriteRecord(FILE *f, const Record &r, const unsigned thread, const unsigned sample) {
	fprintf(f, "phase15lo CARPAINT_PDF_SAMPLE side=%s branch=%s detail=%llu thread=%u sample=%u materialID=%u status=%s activeMask=%u componentCount=%u expectedBranch=%s selectionMatch=%u u=(%.17g,%.17g,%.17g) roughness=(%.17g,%.17g,%.17g) fixed=(%.17g,%.17g,%.17g) sampled=(%.17g,%.17g,%.17g) h=(%.17g,%.17g,%.17g) reportedPdf=%.17g legacyReplayPdf=%.17g claimedForwardPdf=%.17g generatedForwardPdf=%.17g generatedOverReported=%.17g sampledHalfClaim=%.17g sampledHalfReference=%.17g halfReferenceOverClaim=%.17g diagnosticOnly=1 physicsChanged=0\n",
		r.side ? "LIGHT" : "EYE", BranchName(r.branch), r.detail, thread, sample, r.material,
		StatusName(r.status), r.activeMask, r.componentCount, BranchName(r.expectedBranch),
		r.branch == r.expectedBranch ? 1u : 0u, r.u0, r.u1, r.branchU,
		r.rough[0], r.rough[1], r.rough[2], r.fixed.x, r.fixed.y, r.fixed.z,
		r.sampled.x, r.sampled.y, r.sampled.z, r.h.x, r.h.y, r.h.z,
		r.reportedPdf, r.legacyReplayPdf, r.claimedForwardPdf, r.generatedForwardPdf,
		r.reportedPdf > 0. ? r.generatedForwardPdf / r.reportedPdf : 0.,
		r.sampledHalfClaim, r.sampledHalfReference,
		r.sampledHalfClaim > 0. ? r.sampledHalfReference / r.sampledHalfClaim : 0.);
}

inline void WriteSummary(FILE *f) {
	for (unsigned s = 0u; s < 2u; ++s) for (unsigned b = 0u; b < BranchCount; ++b) {
		const unsigned long long q = queryCount[s][b].load(std::memory_order_relaxed);
		const double d = q ? double(q) : 1.;
		fprintf(f, "phase15lo CARPAINT_PDF_SUMMARY side=%s branch=%s selectedAttempts=%llu drawOnly=%llu directionRejected=%llu pdfRejected=%llu accepted=%llu selectionMismatches=%llu invalidReference=%llu legacyReplayMismatches=%llu claimedForwardMismatches=%llu generatedProposalMismatches=%llu sampledHalfPdfMismatches=%llu queryCount=%llu evalBlack=%llu eventMismatches=%llu meanSampleVsEvaluatePdfRelErr=%.9g worstSampleVsEvaluatePdfRelErr=%.9g meanSampleVsPdfQueryRelErr=%.9g worstSampleVsPdfQueryRelErr=%.9g meanEvaluateVsPdfQueryRelErr=%.9g worstEvaluateVsPdfQueryRelErr=%.9g meanSampleVsEvaluateOverPdfRelErr=%.9g worstSampleVsEvaluateOverPdfRelErr=%.9g detailsSelected=%llu writeFailures=%llu toleranceRelative=0.0002 reference=INDEPENDENT_SCHLICK_INVERSE_CDF_PLUS_COSINE_FOUR_WAY_MIXTURE rejectedSamplesIncluded=1 extraSamplerDraws=0 rendererPhysicsChanged=0\n",
			s ? "LIGHT" : "EYE", BranchName(b), attempts[s][b].load(),
			status[s][b][DRAW_ONLY].load(), status[s][b][DIRECTION_REJECT].load(),
			status[s][b][PDF_REJECT].load(), status[s][b][ACCEPT].load(),
			selectionMismatch[s][b].load(), invalidReference[s][b].load(),
			legacyReplayMismatch[s][b].load(), claimedForwardMismatch[s][b].load(),
			generatedProposalMismatch[s][b].load(), sampledHalfMismatch[s][b].load(), q,
			queryEvalBlack[s][b].load(), queryEventMismatch[s][b].load(),
			sampleEvalPdfSum[s][b].load() / d, sampleEvalPdfMax[s][b].load(),
			sampleQueryPdfSum[s][b].load() / d, sampleQueryPdfMax[s][b].load(),
			evalQueryPdfSum[s][b].load() / d, evalQueryPdfMax[s][b].load(),
			sampleWeightSum[s][b].load() / d, sampleWeightMax[s][b].load(),
			details[s][b].load() < 16ull ? details[s][b].load() : 16ull,
			writeFailures[s][b].load());
	}
}

} }
#endif
