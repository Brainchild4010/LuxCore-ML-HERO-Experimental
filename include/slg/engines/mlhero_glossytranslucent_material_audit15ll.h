#ifndef SLG_MLHERO_GLOSSYTRANSLUCENT_MATERIAL_AUDIT15LL_H
#define SLG_MLHERO_GLOSSYTRANSLUCENT_MATERIAL_AUDIT15LL_H

#include <atomic>
#include <cmath>
#include <cstdio>

#include "slg/engines/mlhero_rough_optics15kk.h"

namespace slg { namespace mlhero15ll {

using V = mlhero15kk::V;

enum Kind : unsigned {
	REFLECTION_NONFINITE = 0,
	TRANSMISSION_NONFINITE = 1,
	FRONT_COATING_BOUNDARY = 2,
	KindCount = 3
};

struct RGB {
	double r = 0., g = 0., b = 0.;
};

struct Record {
	unsigned kind = 0, side = 0, material = 0, rejectReason = 0;
	unsigned nativeAccepted = 0, referenceAccepted = 0, mbounce = 0;
	unsigned long long detail = 0;
	V light, eye, fixed, sampled, rawHalf, half, referenceHalf, referenceSampled;
	double sideTest = 0., rawHalfLength = 0., cosI = 0., cosO = 0.;
	double roughness = 0., anisotropy = 0., index = 0., depth = 0.;
	double u0 = 0., u1 = 0., nativePdf = 0., correctedPdf = 0.;
	double cosFixedHalf = 0., d = 0., g = 0., fresnelCos = 0.;
	double directPdf = 0., reversePdf = 0.;
	RGB ks, alpha, base, absorption, fresnel, coating, result;
};

using Reporter = void (*)(const Record &);
inline thread_local Reporter reporter = nullptr;
inline thread_local unsigned long long localEvaluationCalls[2]{},
	localDegenerateHalf[2]{}, localTinyCosine[2]{}, localBoundaryRejectedChecked{};
inline std::atomic<unsigned long long> evaluationCalls[2]{}, nonfinite[2]{},
	degenerateHalf[2]{}, tinyCosine[2]{}, boundaryRejectedChecked{},
	boundaryReferenceAccepted{}, details[KindCount]{}, writeFailures[KindCount]{};

inline void Reset() {
	for (unsigned i = 0; i < 2; ++i) {
		evaluationCalls[i] = 0;
		nonfinite[i] = 0;
		degenerateHalf[i] = 0;
		tinyCosine[i] = 0;
	}
	boundaryRejectedChecked = 0;
	boundaryReferenceAccepted = 0;
	for (unsigned i = 0; i < KindCount; ++i) {
		details[i] = 0;
		writeFailures[i] = 0;
	}
}

struct WorkerScope {
	WorkerScope(bool enabled, Reporter callback) {
		for (unsigned i = 0; i < 2; ++i) {
			localEvaluationCalls[i] = 0;
			localDegenerateHalf[i] = 0;
			localTinyCosine[i] = 0;
		}
		localBoundaryRejectedChecked = 0;
		reporter = enabled ? callback : nullptr;
	}
	~WorkerScope() {
		if (reporter) {
			for (unsigned i = 0; i < 2; ++i) {
				evaluationCalls[i].fetch_add(localEvaluationCalls[i], std::memory_order_relaxed);
				degenerateHalf[i].fetch_add(localDegenerateHalf[i], std::memory_order_relaxed);
				tinyCosine[i].fetch_add(localTinyCosine[i], std::memory_order_relaxed);
			}
			boundaryRejectedChecked.fetch_add(localBoundaryRejectedChecked,
				std::memory_order_relaxed);
		}
		reporter = nullptr;
	}
	WorkerScope(const WorkerScope &) = delete;
	WorkerScope &operator=(const WorkerScope &) = delete;
};

inline bool Enabled() { return reporter != nullptr; }

inline void ObserveEvaluation(const bool reflection, const bool isNonfinite,
		const bool isDegenerateHalf, const bool hasTinyCosine, Record record) {
	if (!reporter)
		return;
	const unsigned event = reflection ? 0u : 1u;
	++localEvaluationCalls[event];
	if (isDegenerateHalf)
		++localDegenerateHalf[event];
	if (hasTinyCosine)
		++localTinyCosine[event];
	if (!isNonfinite)
		return;
	nonfinite[event].fetch_add(1ull, std::memory_order_relaxed);
	record.kind = reflection ? REFLECTION_NONFINITE : TRANSMISSION_NONFINITE;
	record.detail = details[record.kind].fetch_add(1ull, std::memory_order_relaxed);
	if (record.detail < 32ull)
		reporter(record);
}

inline void ObserveBoundary(Record record) {
	if (!reporter)
		return;
	++localBoundaryRejectedChecked;
	if (!record.referenceAccepted)
		return;
	boundaryReferenceAccepted.fetch_add(1ull, std::memory_order_relaxed);
	record.kind = FRONT_COATING_BOUNDARY;
	record.detail = details[FRONT_COATING_BOUNDARY].fetch_add(1ull,
		std::memory_order_relaxed);
	if (record.detail < 32ull)
		reporter(record);
}

inline const char *KindName(const unsigned kind) {
	const char *names[KindCount] = {
		"REFLECTION_NONFINITE", "TRANSMISSION_NONFINITE", "FRONT_COATING_BOUNDARY"
	};
	return kind < KindCount ? names[kind] : "UNKNOWN";
}

inline void WriteRGB(FILE *f, const char *name, const RGB &v) {
	fprintf(f, " %s=(%.17g,%.17g,%.17g)", name, v.r, v.g, v.b);
}

inline void WriteV(FILE *f, const char *name, const V &v) {
	fprintf(f, " %s=(%.17g,%.17g,%.17g)", name, v.x, v.y, v.z);
}

inline void WriteRecord(FILE *f, const Record &r, const unsigned thread,
		const unsigned sample) {
	fprintf(f, "phase15ll GLOSSYTRANSLUCENT_MATERIAL_DETAIL kind=%s detail=%llu thread=%u sample=%u side=%s materialID=%u rejectReason=%u nativeAccepted=%u referenceAccepted=%u mbounce=%u sideTest=%.17g rawHalfLength=%.17g cosI=%.17g cosO=%.17g roughness=%.17g anisotropy=%.17g index=%.17g depth=%.17g u=(%.17g,%.17g) nativePdf=%.17g correctedPdf=%.17g cosFixedHalf=%.17g D=%.17g G=%.17g fresnelCos=%.17g directPdf=%.17g reversePdf=%.17g diagnosticOnly=1 physicsChanged=0 extraSamplerDraws=0\n",
		KindName(r.kind), r.detail, thread, sample, r.side ? "LIGHT" : "EYE",
		r.material, r.rejectReason, r.nativeAccepted, r.referenceAccepted,
		r.mbounce, r.sideTest, r.rawHalfLength, r.cosI, r.cosO,
		r.roughness, r.anisotropy, r.index, r.depth, r.u0, r.u1,
		r.nativePdf, r.correctedPdf, r.cosFixedHalf, r.d, r.g,
		r.fresnelCos, r.directPdf, r.reversePdf);
	WriteV(f, "light", r.light); WriteV(f, "eye", r.eye);
	WriteV(f, "fixed", r.fixed); WriteV(f, "sampled", r.sampled);
	WriteV(f, "rawHalf", r.rawHalf); WriteV(f, "half", r.half);
	WriteV(f, "referenceHalf", r.referenceHalf);
	WriteV(f, "referenceSampled", r.referenceSampled);
	WriteRGB(f, "ks", r.ks); WriteRGB(f, "alpha", r.alpha);
	WriteRGB(f, "base", r.base); WriteRGB(f, "absorption", r.absorption);
	WriteRGB(f, "fresnel", r.fresnel); WriteRGB(f, "coating", r.coating);
	WriteRGB(f, "result", r.result);
	fprintf(f, "\n");
}

inline void WriteSummary(FILE *f) {
	fprintf(f, "phase15ll GLOSSYTRANSLUCENT_MATERIAL_SUMMARY event=REFLECTION evaluationCalls=%llu nonfinite=%llu degenerateHalf=%llu tinyCosine=%llu detailsSelected=%llu writeFailures=%llu diagnosticOnly=1 physicsChanged=0\n",
		evaluationCalls[0].load(), nonfinite[0].load(), degenerateHalf[0].load(),
		tinyCosine[0].load(), details[REFLECTION_NONFINITE].load() < 32ull ?
		details[REFLECTION_NONFINITE].load() : 32ull,
		writeFailures[REFLECTION_NONFINITE].load());
	fprintf(f, "phase15ll GLOSSYTRANSLUCENT_MATERIAL_SUMMARY event=TRANSMISSION evaluationCalls=%llu nonfinite=%llu degenerateHalf=%llu tinyCosine=%llu detailsSelected=%llu writeFailures=%llu diagnosticOnly=1 physicsChanged=0\n",
		evaluationCalls[1].load(), nonfinite[1].load(), degenerateHalf[1].load(),
		tinyCosine[1].load(), details[TRANSMISSION_NONFINITE].load() < 32ull ?
		details[TRANSMISSION_NONFINITE].load() : 32ull,
		writeFailures[TRANSMISSION_NONFINITE].load());
	fprintf(f, "phase15ll GLOSSYTRANSLUCENT_BOUNDARY_SUMMARY rejectedFrontCoatingChecked=%llu referenceAccepted=%llu detailsSelected=%llu writeFailures=%llu reference=DOUBLE_SCHLICK_INVERSE_CDF_SAME_U01 native=FLOAT_SCHLICK_SAMPLE diagnosticOnly=1 physicsChanged=0 extraSamplerDraws=0\n",
		boundaryRejectedChecked.load(), boundaryReferenceAccepted.load(),
		details[FRONT_COATING_BOUNDARY].load() < 32ull ?
		details[FRONT_COATING_BOUNDARY].load() : 32ull,
		writeFailures[FRONT_COATING_BOUNDARY].load());
}

} }

#endif
