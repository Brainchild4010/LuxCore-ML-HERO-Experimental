/***************************************************************************
 * ML HERO phase 15nl: diagnostic-only Mix material decomposition.        *
 ***************************************************************************/

#ifndef _SLG_MLHERO_MIX_AUDIT15NL_H
#define _SLG_MLHERO_MIX_AUDIT15NL_H

#include <algorithm>
#include <cmath>

namespace slg { namespace mlhero15nl {

struct Closure {
	bool finite;
	bool scalarWrapper;
	double wrapper;
	double maxAbsError;
	double relativeError;
};

// A MixMaterial::Evaluate() result is followed only by the scalar public
// BSDF wrappers (shadow terminator and, on light paths, adjoint correction).
// Fit that scalar without changing or reusing it in transport.
inline Closure AnalyzeClosure(const double combined[3], const double nativeValue[3],
		const double relativeTolerance = 2e-5) {
	Closure result = { true, false, 0., 0., 0. };
	double denominator = 0.;
	double numerator = 0.;
	for (unsigned int i = 0; i < 3; ++i) {
		result.finite = result.finite && std::isfinite(combined[i]) &&
			std::isfinite(nativeValue[i]);
		denominator += combined[i] * combined[i];
		numerator += combined[i] * nativeValue[i];
	}
	if (!result.finite)
		return result;

	if (denominator <= 1e-30) {
		const double nativeMax = std::max(std::fabs(nativeValue[0]),
			std::max(std::fabs(nativeValue[1]), std::fabs(nativeValue[2])));
		result.scalarWrapper = nativeMax <= 1e-12;
		result.maxAbsError = nativeMax;
		result.relativeError = nativeMax;
		return result;
	}

	result.wrapper = numerator / denominator;
	double nativeMax = 0.;
	for (unsigned int i = 0; i < 3; ++i) {
		const double error = std::fabs(combined[i] * result.wrapper - nativeValue[i]);
		result.maxAbsError = std::max(result.maxAbsError, error);
		nativeMax = std::max(nativeMax, std::fabs(nativeValue[i]));
	}
	result.relativeError = result.maxAbsError / std::max(1e-12, nativeMax);
	result.scalarWrapper = std::isfinite(result.wrapper) &&
		(result.relativeError <= relativeTolerance);
	return result;
}

} }

#endif
